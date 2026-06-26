#!/usr/bin/env python3
"""
ADNI MCI→Dementia 转化预测 — 修正版训练管线
=============================================
Phase 2 核心实验脚本。基于 v2.0 目标变量（含删失标记），对三个时间窗口
分别训练 s01-s05 LightGBM 管线。

关键修正 (vs 之前实验):
  1. 终点: Broad Dementia (AD + Parkinson's + Other)
  2. 排除删失样本: 非转化且 f/u < 窗口年数的人不参与训练
  3. 新增 3yr 窗口
  4. 小样本校准: CalibratedClassifierCV(cv=3)

时间窗口:
  - MCI→Dementia 3yr:   600 有效, 196 事件, 32.7% 转化率
  - MCI→Dementia 5yr:   500 有效, 267 事件, 53.4% 转化率
  - MCI→Dementia 10yr:  385 有效, 311 事件, 80.8% 转化率

用法:
  cd UKB_DRP-main
  python src/training/run_adni_mci_dementia.py [--data-dir PATH] [--window 3,5,10]
"""

import os, sys, warnings, time, gc, argparse
import numpy as np
import pandas as pd
from collections import Counter
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, average_precision_score,
    confusion_matrix, brier_score_loss, roc_curve)
from lightgbm import LGBMClassifier
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════
_PROJECT_ROOT = os.environ.get('PROJECT_ROOT',
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_DEFAULT_ADNI_DIR = os.path.join(_PROJECT_ROOT, 'local_data', 'adni')

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str,
                    default=os.environ.get('ADNI_DATA_DIR', _DEFAULT_ADNI_DIR),
                    help='ADNI dataset root directory')
parser.add_argument('--window', type=str, default='3,5,10',
                    help='Time windows to run (comma-separated: 3,5,10)')
parser.add_argument('--model', type=str, default='all',
                    choices=['all', 'bio_img', 'bio_only'],
                    help='Model type: bio_img (Biomarkers+Imaging), bio_only, or all')
args = parser.parse_args()

DATA_DIR = os.path.join(args.data_dir, 'processed')
RESULTS_DIR = os.path.join(_PROJECT_ROOT, 'local_data', 'Results_adni', 'mci_to_dementia')
os.makedirs(RESULTS_DIR, exist_ok=True)

WINDOWS = [int(w.strip()) for w in args.window.split(',')]

N_SPLITS = 5
RANDOM_STATE = 2022
TOP_N_S01 = 50
CLUSTER_THRESHOLD = 0.75
TOP_N_SFS = 15
TOP_N_FINAL = 10
SFS_EARLY_STOP = 0.0005

LGB_PARAMS = dict(
    n_estimators=500, max_depth=15, num_leaves=10,
    subsample=0.7, learning_rate=0.01, colsample_bytree=0.7,
    objective='binary', is_unbalance=True, metric='auc',
    verbosity=-1, seed=2020, n_jobs=4,
    min_data_in_leaf=5,
    min_gain_to_split=0.0,
)

# ══════════════════════════════════════════════════════════════════════
# FEATURE EXCLUSIONS (防止诊断泄漏)
# ══════════════════════════════════════════════════════════════════════

# 认知测试 — 全都排除 (它们本身就是诊断 MCI→痴呆 的依据)
COGNITIVE_PREFIXES = [
    'MMSE_', 'MOCA_', 'ADAS_', 'FAQ_', 'GDS_', 'NPIQ_', 'NB_', 'HACH_',
]
# CDR 子项 — 排除 (CDR 是痴呆诊断的金标准组件)
CDR_COLS = ['CDMEMORY', 'CDORIENT', 'CDJUDGE', 'CDCOMMUN',
            'CDHOME', 'CDCARE', 'CDGLOBAL', 'CDRSB']
# ID/元数据列
ID_COLS = ['PTID', 'RID', 'PHASE', 'VISCODE', 'VISCODE2',
           'APOE_genotype', 'subject_id', 'entry_research_group', 'PTDOBYY',
           # CDR builder 内部列 (object 类型, 需排除)
           '_feat_baseline_date', '_birth_year',
           'cdr_cohort', 'index_date', 'index_viscode',
           'index_CDGLOBAL']
# 目标标签列
TARGET_COLS = [
    'AD_status', 'MCI_status', 'dementia_status', 'DIAGNOSIS',
    'DXNORM', 'DXNODEP', 'DXMCI', 'DXDSEV',
    'DXDDUE', 'DXAD', 'DXPARK', 'DXDEP', 'DXOTHDEM',
    # v1 targets
    'AD_5yrs', 'AD_10yrs', 'AD_status_incident', 'AD_years',
    'Dementia_5yrs', 'Dementia_10yrs', 'Dementia_status_incident', 'Dementia_years',
    # v2 targets
    'AD_3yrs', 'Dementia_3yrs',
    # conversion
    'converted_to_ad', 'converted_to_dementia', 'converted_to_mci',
    'ad_conversion_years', 'dementia_conversion_years',
    'baseline_diagnosis',
    # censoring
    'censored_ad_3yr', 'censored_ad_5yr', 'censored_ad_10yr',
    'censored_dementia_3yr', 'censored_dementia_5yr', 'censored_dementia_10yr',
    # other
    'last_followup_years', 'diag_label',
]
IMAGING_PFX = ('FS_', 'BSI_', 'WMH_', 'AMY_', 'TAU_', 'TAUPVC_')

# ══════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════
def normal_imp(d):
    s = sum(d.values())
    return d if s == 0 else {k: v / s for k, v in d.items()}

def get_full_eval(y_true, y_pred):
    fpr, tpr, thresh = roc_curve(y_true, y_pred)
    best_i = np.argmax(tpr - fpr)
    best_cut = thresh[best_i] if best_i < len(thresh) else 0.5
    y_bin = (y_pred >= best_cut).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_bin).ravel()
    return {
        'auc': roc_auc_score(y_true, y_pred),
        'ap': average_precision_score(y_true, y_pred),
        'brier': brier_score_loss(y_true, y_pred),
        'accuracy': (tp + tn) / max(tp + tn + fp + fn, 1),
        'sensitivity': tp / max(tp + fn, 1),
        'specificity': tn / max(tn + fp, 1),
        'precision': tp / max(tp + fp, 1),
        'n_events': int(y_true.sum()),
        'n_total': len(y_true),
        'best_cutoff': float(best_cut),
    }

def run_s01_s05(X, y, verbose=True):
    """完整 s01-s05 管线: LightGBM排序→Ward聚类→重排序→SFS→校准评估"""
    kf = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=True)

    # ── s01: 5-fold CV Gain排序 ──
    tg_cv = Counter()
    for tr, te in kf.split(X, y):
        gbm = LGBMClassifier(**LGB_PARAMS)
        gbm.fit(X.iloc[tr], y[tr])
        tg = dict(zip(gbm.booster_.feature_name(),
                      gbm.booster_.feature_importance(importance_type='gain')))
        tg_cv += Counter(normal_imp(tg))
    s01_r = sorted(tg_cv.items(), key=lambda x: -x[1])
    top50 = [f for f, _ in s01_r[:min(TOP_N_S01, len(s01_r))]]
    if verbose:
        n_img = sum(1 for f in top50 if f.startswith(IMAGING_PFX))
        print(f"  s01: Top50 ({n_img} imaging) — #1={top50[0]}")

    # ── s02: Ward聚类去冗余 ──
    X_t = X[top50].fillna(X[top50].median()).copy()
    corr = np.array(X_t.corr(method='spearman'))
    corr = np.nan_to_num(corr)
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)
    link = linkage(squareform(1 - np.abs(corr)), method='ward')
    clusters = fcluster(link, t=CLUSTER_THRESHOLD, criterion='distance')
    kept = []; seen = set()
    for f, c in zip(top50, clusters):
        if c not in seen:
            kept.append(f); seen.add(c)
    X_k = X[kept]
    if verbose:
        print(f"  s02: {len(top50)} → {len(kept)} features after clustering")

    # ── s03: 聚类后重排序 ──
    tg_cv3 = Counter()
    for tr, te in kf.split(X_k, y):
        gbm = LGBMClassifier(**LGB_PARAMS)
        gbm.fit(X_k.iloc[tr], y[tr])
        tg = dict(zip(gbm.booster_.feature_name(),
                      gbm.booster_.feature_importance(importance_type='gain')))
        tg_cv3 += Counter(normal_imp(tg))
    s03_r = sorted(tg_cv3.items(), key=lambda x: -x[1])
    s03_f = [f for f, _ in s03_r]
    if verbose:
        n_img = sum(1 for f in s03_f[:5] if f.startswith(IMAGING_PFX))
        print(f"  s03: Top5 ({n_img} imaging) — {s03_f[:3]}")

    # ── s04: Sequential Forward Selection ──
    selected = []; remaining = list(s03_f); prev_auc = 0
    sfs_history = []
    for step in range(min(TOP_N_SFS, len(s03_f))):
        best_f, best_auc = None, -1
        pool = remaining[:min(15, len(remaining))]
        for cand in pool:
            trial = selected + [cand]
            aucs = []
            for tr, te in kf.split(X_k[trial], y):
                g = LGBMClassifier(**LGB_PARAMS)
                g.fit(X_k[trial].iloc[tr], y[tr])
                aucs.append(roc_auc_score(y[te],
                    g.predict_proba(X_k[trial].iloc[te])[:, 1]))
            avg_auc = np.mean(aucs)
            if avg_auc > best_auc:
                best_f, best_auc = cand, avg_auc
        if best_f is None: break
        gain = best_auc - prev_auc if step > 0 else best_auc
        sfs_history.append({
            'step': step + 1, 'feature': best_f,
            'auc': best_auc, 'gain': gain,
            'is_imaging': best_f.startswith(IMAGING_PFX),
        })
        if gain < SFS_EARLY_STOP and len(selected) >= TOP_N_FINAL:
            selected.append(best_f); remaining.remove(best_f); break
        selected.append(best_f)
        remaining.remove(best_f)
        prev_auc = best_auc
        if verbose:
            tag = '[IMG]' if best_f.startswith(IMAGING_PFX) else '[BIO]'
            print(f"  s04 step{step+1}: {tag} {best_f}  AUC={best_auc:.4f} (gain={gain:+.4f})")

    # ── s05: CalibratedClassifierCV + 5-fold CV 评估 ──
    top_f = selected[:TOP_N_FINAL]
    X_ff = X_k[top_f]
    preds, trues, fold_metrics = [], [], []

    for fold_i, (tr, te) in enumerate(kf.split(X_ff, y)):
        X_tr, y_tr = X_ff.iloc[tr], y[tr]
        X_te, y_te = X_ff.iloc[te], y[te]

        gbm = LGBMClassifier(**LGB_PARAMS)
        try:
            calib = CalibratedClassifierCV(gbm, method='isotonic', cv=3)
            calib.fit(X_tr, y_tr)
            y_pr = np.clip(calib.predict_proba(X_te)[:, 1], 0, 1)
        except Exception as e:
            # Fallback: no calibration if calibration fails
            print(f"    [WARN] CalibratedCV failed (fold {fold_i}): {e}")
            print(f"    [WARN] Using raw LightGBM probabilities instead")
            gbm.fit(X_tr, y_tr)
            y_pr = np.clip(gbm.predict_proba(X_te)[:, 1], 0, 1)

        preds.append(y_pr)
        trues.append(y_te)
        fold_metrics.append({
            'fold': fold_i,
            'auc': roc_auc_score(y_te, y_pr),
            'brier': brier_score_loss(y_te, y_pr),
        })

    yt_all = np.concatenate(trues)
    yp_all = np.concatenate(preds)
    aucs_cv = [roc_auc_score(yt, yp) for yt, yp in zip(trues, preds)]

    metrics = get_full_eval(yt_all, yp_all)
    metrics['auc_cv_mean'] = np.mean(aucs_cv)
    metrics['auc_cv_std'] = np.std(aucs_cv)

    if verbose:
        n_img = sum(1 for f in top_f if f.startswith(IMAGING_PFX))
        print(f"  s05: AUC={metrics['auc_cv_mean']:.4f}±{metrics['auc_cv_std']:.4f}, "
              f"Brier={metrics['brier']:.4f}, Sens={metrics['sensitivity']:.3f}, "
              f"Spec={metrics['specificity']:.3f}")
        print(f"  Top10 ({n_img} imaging): {top_f[:5]}...")

    return metrics, top_f, sfs_history, fold_metrics


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
def main():
    print("=" * 72)
    print("  ADNI MCI→Dementia 转化预测 — Phase 2 修正版")
    print("=" * 72)

    # ── 1. LOAD DATA ──
    data_path = os.path.join(DATA_DIR, 'ADNI_baseline_with_time_targets_v2.csv')
    print(f"\n[1] Loading: {data_path}")
    full_df = pd.read_csv(data_path, low_memory=False)
    print(f"    Total subjects: {len(full_df):,}, columns: {len(full_df.columns)}")

    # ── 2. FILTER TO BASELINE MCI ──
    mci_df = full_df[full_df['baseline_diagnosis'] == 2.0].copy()
    print(f"\n[2] Baseline MCI: {len(mci_df)} subjects")

    # Report censoring
    for w in [3, 5, 10]:
        col = f'censored_dementia_{w}yr'
        if col in mci_df.columns:
            print(f"    {col}: {mci_df[col].sum()} censored")

    # Report Dementia events
    for col in ['Dementia_3yrs', 'Dementia_5yrs', 'Dementia_10yrs',
                 'Dementia_status_incident']:
        if col in mci_df.columns:
            print(f"    {col}: {int(mci_df[col].sum())} events")

    # ── 3. BUILD FEATURE SETS ──
    print(f"\n[3] Building feature sets...")
    all_exclude = set(ID_COLS + TARGET_COLS + CDR_COLS)
    all_feat = [c for c in mci_df.columns if c not in all_exclude]
    clean_feat = [c for c in all_feat
                  if not any(c.startswith(p) for p in COGNITIVE_PREFIXES)]
    # Drop non-numeric columns (auto-filters leftover object/string meta columns)
    num_cols = set(mci_df.select_dtypes(include=[np.number]).columns)
    clean_feat = [c for c in clean_feat if c in num_cols]
    clean_clinical = [c for c in clean_feat if not c.startswith(IMAGING_PFX)]
    clean_imaging = [c for c in clean_feat if c.startswith(IMAGING_PFX)]

    print(f"    Clean clinical (Bio only): {len(clean_clinical)}")
    print(f"    Clean imaging:              {len(clean_imaging)}")
    print(f"    Clean total (Bio+Img):      {len(clean_feat)}")

    # Show key modalities available in MCI
    for label, col in [
        ('CSF Aβ42', 'ABETA42'), ('CSF pTau', 'PTAU'),
        ('Plasma pTau217', 'pT217_F'), ('Plasma NfL', 'NfL_Q'),
        ('Plasma GFAP', 'GFAP_Q'), ('MRI FS', 'FS_ST101SV'),
        ('Amyloid PET', 'AMY_CENTILOIDS'), ('Tau PET', 'TAU_META_TEMPORAL_SUVR'),
        ('APOE4', 'APOE4_carrier'),
    ]:
        if col in mci_df.columns:
            print(f"    {label:<18}: {mci_df[col].notna().sum():>5}/{len(mci_df)} "
                  f"({mci_df[col].notna().sum()/len(mci_df)*100:.0f}%)")

    # ── 4. DEFINE TARGETS ──
    WINDOW_TARGETS = []
    for w in WINDOWS:
        target_col = f'Dementia_{w}yrs'
        censored_col = f'censored_dementia_{w}yr'
        if target_col in mci_df.columns:
            WINDOW_TARGETS.append((target_col, censored_col, w,
                                    f'MCI→Dementia {w}yr'))
    # Always add all-time
    if 'Dementia_status_incident' in mci_df.columns:
        WINDOW_TARGETS.append(('Dementia_status_incident', None, None,
                                'MCI→Dementia all-time'))

    print(f"\n[4] Targets to run: {len(WINDOW_TARGETS)}")
    for _, _, w, label in WINDOW_TARGETS:
        print(f"    {label}")

    # ── 5. RUN TRAINING ──
    all_results = []
    all_features = {}
    all_sfs = {}

    for target_col, censored_col, window_yr, label in WINDOW_TARGETS:
        if args.model not in ['all', 'bio_img']:
            continue

        print(f"\n{'─'*72}")
        print(f"  MODEL A: Biomarkers + Imaging — {label}")
        print(f"{'─'*72}")

        # Filter to valid subjects (排除删失)
        df_target = mci_df.copy()
        n_before = len(df_target)
        if censored_col and censored_col in df_target.columns:
            n_censored = int(df_target[censored_col].sum())
            df_target = df_target[df_target[censored_col] == 0]
            print(f"  Excluded {n_censored} censored → {len(df_target)} valid subjects")
        else:
            n_censored = 0
            print(f"  No censoring applied (all-time target)")

        y = df_target[target_col].values.astype(int)
        n_events = int(y.sum())
        n_total = len(y)
        print(f"  n={n_total}, events={n_events} ({n_events/max(n_total,1)*100:.1f}%)")

        if n_events < 5 or (n_total - n_events) < 5:
            print(f"  [SKIP] Too few events or negatives")
            continue

        X = df_target[[c for c in clean_feat if c in df_target.columns]].copy()
        # Drop constant/near-zero-variance columns
        const_c = [c for c in X.columns if X[c].nunique() <= 1]
        X = X.drop(columns=const_c)
        print(f"  Features: {X.shape[1]} (dropped {len(const_c)} constant)")

        metrics, features, sfs_hist, fold_m = run_s01_s05(X, y)
        metrics['target'] = target_col
        metrics['label'] = label
        metrics['model'] = 'Biomarkers + Imaging'
        metrics['window_yr'] = window_yr
        metrics['n_censored'] = n_censored
        metrics['n_features_input'] = X.shape[1]
        all_results.append(metrics)
        all_features[f'bio_img_{target_col}'] = features
        all_sfs[f'bio_img_{target_col}'] = sfs_hist

        # Save features
        pd.DataFrame({'rank': range(1, len(features)+1),
                      'feature': features}).to_csv(
            os.path.join(RESULTS_DIR, f'features_bio_img_{target_col}.csv'), index=False)
        pd.DataFrame(sfs_hist).to_csv(
            os.path.join(RESULTS_DIR, f'sfs_history_{target_col}.csv'), index=False)

        gc.collect()

    # ── 6. MODEL B: Bio only (if requested) ──
    if args.model in ['all', 'bio_only']:
        for target_col, censored_col, window_yr, label in WINDOW_TARGETS:
            print(f"\n{'─'*72}")
            print(f"  MODEL B: Demographics + Biomarkers only — {label}")
            print(f"{'─'*72}")

            df_target = mci_df.copy()
            if censored_col and censored_col in df_target.columns:
                n_censored = int(df_target[censored_col].sum())
                df_target = df_target[df_target[censored_col] == 0]
                print(f"  Excluded {n_censored} censored → {len(df_target)}")
            else:
                n_censored = 0

            y = df_target[target_col].values.astype(int)
            n_events = int(y.sum())

            X = df_target[[c for c in clean_clinical if c in df_target.columns]].copy()
            const_c = [c for c in X.columns if X[c].nunique() <= 1]
            X = X.drop(columns=const_c)

            if X.shape[1] < 5:
                print(f"  [SKIP] Too few features ({X.shape[1]})")
                continue

            print(f"  Features: {X.shape[1]}")
            metrics, features, sfs_hist, fold_m = run_s01_s05(X, y)
            metrics['target'] = target_col
            metrics['label'] = label
            metrics['model'] = 'Demographics + Biomarkers'
            metrics['window_yr'] = window_yr
            metrics['n_censored'] = n_censored
            metrics['n_features_input'] = X.shape[1]
            all_results.append(metrics)
            all_features[f'bio_only_{target_col}'] = features
            all_sfs[f'bio_only_{target_col}'] = sfs_hist

            pd.DataFrame({'rank': range(1, len(features)+1),
                          'feature': features}).to_csv(
                os.path.join(RESULTS_DIR, f'features_bio_only_{target_col}.csv'), index=False)
            pd.DataFrame(sfs_hist).to_csv(
                os.path.join(RESULTS_DIR, f'sfs_history_bio_only_{target_col}.csv'), index=False)
            gc.collect()

    # ══════════════════════════════════════════════════════════════════
    # 7. SUMMARY
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*90}")
    print("  MCI→DEMENTIA PREDICTION — RESULTS SUMMARY")
    print(f"{'='*90}")

    res_df = pd.DataFrame(all_results)
    res_path = os.path.join(RESULTS_DIR, 'mci_to_dementia_results.csv')
    res_df.to_csv(res_path, index=False)
    print(f"\n  Full results saved to: {res_path}")

    print(f"\n  {'Model':<30} {'Target':<25} {'AUC':>8} {'±':>6} {'Brier':>8} "
          f"{'Sens':>6} {'Spec':>6} {'n':>6}")
    print(f"  {'─'*30} {'─'*25} {'─'*8} {'─'*6} {'─'*8} {'─'*6} {'─'*6} {'─'*6}")
    for _, row in res_df.iterrows():
        print(f"  {row['model']:<30} {row['label']:<25} "
              f"{row['auc_cv_mean']:>8.4f} {row['auc_cv_std']:>5.4f} "
              f"{row['brier']:>8.4f} {row['sensitivity']:>6.3f} "
              f"{row['specificity']:>6.3f} {row['n_events']:>6}")

    # ── Imaging delta ──
    if args.model in ['all']:
        print(f"\n  Imaging contribution (ΔAUC):")
        for tcol, _, w, label in WINDOW_TARGETS:
            r_img = res_df[(res_df['target'] == tcol) &
                           (res_df['model'] == 'Biomarkers + Imaging')]
            r_bio = res_df[(res_df['target'] == tcol) &
                           (res_df['model'] == 'Demographics + Biomarkers')]
            if len(r_img) > 0 and len(r_bio) > 0:
                auc_img = r_img['auc_cv_mean'].values[0]
                auc_bio = r_bio['auc_cv_mean'].values[0]
                d = auc_img - auc_bio
                print(f"    {label:<28} +Img {auc_img:.4f}  Bio only {auc_bio:.4f}  "
                      f"Δ={d:+.4f}")

    # ── 8. PLOTS ──
    make_plots(res_df, all_features)

    # ── 9. TOP FEATURES ──
    print(f"\n{'='*72}")
    print("  TOP FEATURES (Biomarkers + Imaging)")
    print(f"{'='*72}")

    for tcol, _, w, label in WINDOW_TARGETS:
        feat_key = f'bio_img_{tcol}'
        if feat_key in all_features:
            features = all_features[feat_key]
            n_img = sum(1 for f in features[:10] if f.startswith(IMAGING_PFX))
            print(f"\n  {label} ({n_img}/10 imaging):")
            for i, f in enumerate(features[:10]):
                cat = '[IMG]' if f.startswith(IMAGING_PFX) else '[BIO]'
                print(f"    {i+1:>2}. {cat} {f}")

    print(f"\n{'='*72}")
    print(f"  ✅ Results saved to: {RESULTS_DIR}")
    print(f"  📁 Results files:")
    for fname in sorted(os.listdir(RESULTS_DIR)):
        fpath = os.path.join(RESULTS_DIR, fname)
        if os.path.isfile(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"     {fname} ({size_kb:.0f} KB)")
    print(f"{'='*72}")


# ══════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════
def make_plots(res_df, all_features):
    if len(res_df) == 0:
        return

    # ── AUC bar chart ──
    fig, ax = plt.subplots(figsize=(12, 6))
    img_rows = res_df[res_df['model'] == 'Biomarkers + Imaging']

    labels = [r['label'] for _, r in img_rows.iterrows()]
    aucs = [r['auc_cv_mean'] for _, r in img_rows.iterrows()]
    stds = [r['auc_cv_std'] for _, r in img_rows.iterrows()]

    n = len(labels)
    x = np.arange(n)
    colors = ['#3B82F6' if '3yr' in l else '#F59E0B' if '5yr' in l
              else '#EF4444' if '10yr' in l else '#8B5CF6' for l in labels]

    bars = ax.bar(x, aucs, 0.5, color=colors, edgecolor='white', linewidth=0.8)
    ax.errorbar(x, aucs, yerr=stds, fmt='none', ecolor='#374151',
                capsize=5, capthick=1.5, linewidth=1.5)

    for i, (auc, std) in enumerate(zip(aucs, stds)):
        ax.text(i, auc + std + 0.015, f'{auc:.3f}±{std:.3f}',
                ha='center', fontsize=11, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('AUC (5-fold CV)', fontsize=12)
    ax.set_ylim(0.5, 1.05)
    ax.set_title('ADNI: MCI→Dementia Conversion Prediction\n'
                 '(Excluding censored — only subjects with adequate follow-up)',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'mci_to_dementia_auc.png'), dpi=150)
    plt.close()

    # ── SFS feature accumulation plot ──
    try:
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        for tcol, cens, w, label in [x for x in [
            ('Dementia_3yrs', None, 3, 'MCI→Dementia 3yr'),
            ('Dementia_5yrs', None, 5, 'MCI→Dementia 5yr'),
            ('Dementia_10yrs', None, 10, 'MCI→Dementia 10yr'),
        ] if x[0] in all_features]:
            sfs_df = pd.DataFrame(all_features.get(tcol + '_sfs', []))
            # Actually get from results dir
            sfs_path = os.path.join(RESULTS_DIR, f'sfs_history_{tcol}.csv')
            if os.path.exists(sfs_path):
                sfs_df = pd.read_csv(sfs_path)
                if len(sfs_df) > 0:
                    if 'step' in sfs_df.columns:
                        steps = sfs_df['step'].values
                        auc_vals = sfs_df['auc'].values
                        ax2.plot(steps, auc_vals, 'o-', linewidth=2, label=label,
                                markersize=6)

        if ax2.get_lines():
            ax2.set_xlabel('SFS Step', fontsize=12)
            ax2.set_ylabel('Cumulative AUC', fontsize=12)
            ax2.set_title('SFS Feature Accumulation — MCI→Dementia',
                         fontsize=14, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(alpha=0.3)
            fig2.tight_layout()
            fig2.savefig(os.path.join(RESULTS_DIR, 'sfs_accumulation.png'), dpi=150)
        plt.close()
    except Exception as e:
        print(f"  [WARN] SFS plot failed: {e}")

    # ── Model comparison (if bio_only exists) ──
    bio_rows = res_df[res_df['model'] == 'Demographics + Biomarkers']
    if len(bio_rows) > 0:
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        img_rows_sorted = img_rows.sort_values('label')
        bio_rows_sorted = bio_rows.sort_values('label')

        labels2 = [r['label'] for _, r in img_rows_sorted.iterrows()]
        n2 = len(labels2)
        x2 = np.arange(n2)
        w = 0.35

        aucs_i = [r['auc_cv_mean'] for _, r in img_rows_sorted.iterrows()]
        aucs_b = [r['auc_cv_mean'] for _, r in bio_rows_sorted.iterrows()]

        ax3.bar(x2 - w/2, aucs_i, w, label='Biomarkers + Imaging',
                color='#3B82F6', edgecolor='white')
        ax3.bar(x2 + w/2, aucs_b, w, label='Demographics + Biomarkers only',
                color='#93C5FD', edgecolor='white')

        for i, (ai, ab) in enumerate(zip(aucs_i, aucs_b)):
            ax3.text(i - w/2, ai + 0.01, f'{ai:.3f}', ha='center', fontsize=9,
                    fontweight='bold')
            ax3.text(i + w/2, ab + 0.01, f'{ab:.3f}', ha='center', fontsize=9)
            d = ai - ab
            ax3.annotate(f'Δ{d:+.3f}', xy=(i, max(ai, ab) + 0.06),
                        ha='center', fontsize=8,
                        color='#10B981' if d >= 0 else '#EF4444', fontweight='bold')

        ax3.set_xticks(x2)
        ax3.set_xticklabels(labels2, fontsize=11)
        ax3.set_ylabel('AUC', fontsize=12)
        ax3.set_ylim(0.5, 1.05)
        ax3.set_title('ADNI: MCI→Dementia — Feature Ablation',
                     fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10, loc='lower right')
        ax3.grid(axis='y', alpha=0.3)
        ax3.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)

        fig3.tight_layout()
        fig3.savefig(os.path.join(RESULTS_DIR, 'model_comparison.png'), dpi=150)
        plt.close()


if __name__ == '__main__':
    t0 = time.time()
    main()
    elapsed = time.time() - t0
    print(f"\n⏱  Total time: {elapsed/60:.1f} min ({elapsed:.0f} sec)")

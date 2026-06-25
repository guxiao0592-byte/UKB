#!/usr/bin/env python3
"""
ADNI CDR 恶化二分类预测 — Phase 2C-2 v2 (Revised Design)
==========================================================
v2 修订:
  1. CN 和 MCI 分队列作为主分析
  2. 合并队列作为次要分析 (含 baseline CDR group)
  3. 3年为主要窗口, 5年为次要, 10年为探索性
  4. 持续性恶化敏感性分析
  5. CDR-SB ≥1 恶化敏感性分析
  6. 修复信息泄漏: 训练折内完成缺失值填补

基于 Phase 2A 的 s01-s05 LightGBM 管线

用法:  python src/training/run_adni_cdr_binary_v2.py
"""

import os, sys, warnings, time, gc, argparse
import numpy as np; import pandas as pd
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

# ═══════════════════════════════ CONFIG ═══════════════════════════════
BASE = os.environ.get('PROJECT_ROOT',
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ADNI_DATA_DIR = os.environ.get('ADNI_DATA_DIR',
    os.path.join(BASE, 'local_data', 'adni'))
DATA_PATH = os.path.join(ADNI_DATA_DIR, 'processed', 'ADNI_baseline_with_time_targets_v2.csv')
RESULTS_DIR = os.path.join(BASE, 'local_data', 'Results_adni', 'cdr_binary')
os.makedirs(RESULTS_DIR, exist_ok=True)

N_SPLITS = 5; RANDOM_STATE = 2022
TOP_S01 = 50; CLUST_TH = 0.75
TOP_SFS = 15; TOP_FINAL = 10; STOP = 0.0005

LGB_PARAMS = dict(
    n_estimators=500, max_depth=15, num_leaves=10,
    subsample=0.7, learning_rate=0.01, colsample_bytree=0.7,
    objective='binary', is_unbalance=True, metric='auc',
    verbosity=-1, seed=2020, n_jobs=4,
    min_data_in_leaf=5, min_gain_to_split=0.0,
)

COG_PFX = ['MMSE_', 'MOCA_', 'ADAS_', 'FAQ_', 'GDS_', 'NPIQ_', 'NB_', 'HACH_']
CDR_COLS = ['CDMEMORY', 'CDORIENT', 'CDJUDGE', 'CDCOMMUN',
            'CDHOME', 'CDCARE', 'CDGLOBAL', 'CDRSB']
ID_COLS = ['PTID', 'RID', 'PHASE', 'VISCODE', 'VISCODE2',
           'APOE_genotype', 'subject_id', 'entry_research_group', 'PTDOBYY']
TARGET_COLS = [
    'AD_status', 'MCI_status', 'dementia_status', 'DIAGNOSIS',
    'DXNORM', 'DXNODEP', 'DXMCI', 'DXDSEV', 'DXDDUE', 'DXAD',
    'DXPARK', 'DXDEP', 'DXOTHDEM',
    'AD_3yrs', 'AD_5yrs', 'AD_10yrs', 'AD_status_incident', 'AD_years',
    'Dementia_3yrs', 'Dementia_5yrs', 'Dementia_10yrs', 'Dementia_status_incident',
    'Dementia_years', 'converted_to_ad', 'converted_to_dementia', 'converted_to_mci',
    'ad_conversion_years', 'dementia_conversion_years', 'baseline_diagnosis',
    'censored_ad_3yr', 'censored_ad_5yr', 'censored_ad_10yr',
    'censored_dementia_3yr', 'censored_dementia_5yr', 'censored_dementia_10yr',
    'last_followup_years', 'diag_label',
    # v1+v2 CDR targets
    'baseline_CDGLOBAL', 'baseline_CDRSB',
    'CDR_worsen_3yr', 'CDR_worsen_5yr', 'CDR_worsen_10yr', 'CDR_worsen_alltime',
    'CDR_worsen_censored_3yr', 'CDR_worsen_censored_5yr', 'CDR_worsen_censored_10yr',
    'CDR_worsen_sustained_3yr', 'CDR_worsen_sustained_5yr',
    'CDR_worsen_sustained_alltime',
    'CDR_worsen_sustained_censored_3yr', 'CDR_worsen_sustained_censored_5yr',
    'CDRSB_worsen_3yr', 'CDRSB_worsen_5yr', 'CDRSB_worsen_alltime',
    'CDRSB_worsen_censored_3yr', 'CDRSB_worsen_censored_5yr',
    'CDR_surv_time', 'CDR_surv_event',
    'CDR_surv_sustained_time', 'CDR_surv_sustained_event',
    'CDRSB_surv_time', 'CDRSB_surv_event',
    'composite_event', 'composite_time',
    'CDR_worsened', 'CDR_worsening_years',
    'CDR_worsened_sustained',
    'CDRSB_worsened', 'CDRSB_worsening_years', 'CDRSB_worsened_sustained',
    'CDR_worsened_CDRSB2',
    'CDR_last_followup_years', 'n_cdr_visits', 'n_post_idx_visits',
    'cdr_cohort', 'index_CDGLOBAL', 'index_date', 'index_viscode',
    '_feat_baseline_date', '_birth_year',
]
IMG_PFX = ('FS_', 'BSI_', 'WMH_', 'AMY_', 'TAU_', 'TAUPVC_')

# ═══════════════════════════════ UTILITIES ═══════════════════════════════
def norm_imp(d):
    s = sum(d.values())
    return d if s == 0 else {k: v / s for k, v in d.items()}

def median_impute_train_apply(X_train, X_test):
    """Impute within fold only."""
    X_tr = X_train.copy(); X_te = X_test.copy()
    num_cols = X_tr.select_dtypes(include=[np.number]).columns
    medians = X_tr[num_cols].median()
    X_tr[num_cols] = X_tr[num_cols].fillna(medians)
    X_te[num_cols] = X_te[num_cols].fillna(medians)
    non_num = [c for c in X_tr.columns if c not in num_cols]
    if non_num:
        X_tr = X_tr.drop(columns=non_num)
        X_te = X_te.drop(columns=non_num)
    return X_tr, X_te

def get_full_eval(y_true, y_pred):
    try:
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
            'n_events': int(y_true.sum()), 'n_total': len(y_true),
            'best_cutoff': float(best_cut),
        }
    except Exception:
        return {
            'auc': np.nan, 'ap': np.nan, 'brier': np.nan,
            'sensitivity': np.nan, 'specificity': np.nan,
            'n_events': int(y_true.sum()), 'n_total': len(y_true),
        }


def run_binary_pipeline(X, y, verbose=True):
    """s01-s05 LightGBM pipeline with within-fold imputation."""
    kf = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=True)

    # s01: LGB Gain ranking (within CV)
    tg_cv = Counter()
    for tr, te in kf.split(X, y):
        X_tr, X_te = median_impute_train_apply(X.iloc[tr], X.iloc[te])
        gbm = LGBMClassifier(**LGB_PARAMS)
        gbm.fit(X_tr, y[tr])
        tg = dict(zip(gbm.booster_.feature_name(),
                      gbm.booster_.feature_importance(importance_type='gain')))
        tg_cv += Counter(norm_imp(tg))
    s01_r = sorted(tg_cv.items(), key=lambda x: -x[1])
    top50 = [f for f, _ in s01_r[:min(TOP_S01, len(s01_r))]]
    if verbose:
        n_img = sum(1 for f in top50 if f.startswith(IMG_PFX))
        print(f"  s01: Top50 ({n_img} img) — #1={top50[0]}")

    # s02: Ward clustering (documented limitation: on full data)
    X_t = X[top50].copy()
    for col in X_t.columns:
        if X_t[col].isna().any():
            X_t[col] = X_t[col].fillna(X_t[col].median())
    corr = np.array(X_t.corr(method='spearman'))
    corr = np.nan_to_num(corr); corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)
    cl = fcluster(linkage(squareform(1 - np.abs(corr)), 'ward'),
                  t=CLUST_TH, criterion='distance')
    kept = []; seen = set()
    for f, c in zip(top50, cl):
        if c not in seen: kept.append(f); seen.add(c)
    X_k = X[kept]
    if verbose: print(f"  s02: {len(top50)} → {len(kept)} features")

    # s03: Re-rank
    tg_cv3 = Counter()
    for tr, te in kf.split(X_k, y):
        X_tr, X_te = median_impute_train_apply(X_k.iloc[tr], X_k.iloc[te])
        gbm = LGBMClassifier(**LGB_PARAMS)
        gbm.fit(X_tr, y[tr])
        tg = dict(zip(gbm.booster_.feature_name(),
                      gbm.booster_.feature_importance(importance_type='gain')))
        tg_cv3 += Counter(norm_imp(tg))
    s03_f = [f for f, _ in sorted(tg_cv3.items(), key=lambda x: -x[1])]
    if verbose: print(f"  s03: Top3={s03_f[:3]}")

    # s04: SFS
    selected = []; remaining = list(s03_f); prev_auc = 0; sfs_hist = []
    for step in range(min(TOP_SFS, len(s03_f))):
        best_f, best_auc = None, -1
        pool = remaining[:min(15, len(remaining))]
        for cand in pool:
            trial = selected + [cand]; aucs = []
            for tr, te in kf.split(X_k[trial], y):
                X_tr, X_te = median_impute_train_apply(
                    X_k[trial].iloc[tr], X_k[trial].iloc[te])
                g = LGBMClassifier(**LGB_PARAMS)
                g.fit(X_tr, y[tr])
                aucs.append(roc_auc_score(y[te],
                    g.predict_proba(X_te)[:, 1]))
            avg_auc = np.mean(aucs)
            if avg_auc > best_auc: best_f, best_auc = cand, avg_auc
        if best_f is None: break
        gain = best_auc - prev_auc if step > 0 else best_auc
        sfs_hist.append({'step': step + 1, 'feature': best_f,
                         'auc': best_auc, 'gain': gain,
                         'is_imaging': best_f.startswith(IMG_PFX)})
        if gain < STOP and len(selected) >= TOP_FINAL:
            selected.append(best_f); remaining.remove(best_f); break
        selected.append(best_f); remaining.remove(best_f); prev_auc = best_auc
        if verbose:
            tag = '[IMG]' if best_f.startswith(IMG_PFX) else '[BIO]'
            print(f"  s04 step{step+1}: {tag} {best_f} AUC={best_auc:.4f} (Δ={gain:+.4f})")

    # s05: Calibrated LGBM (already nested CV — correct)
    top_f = selected[:TOP_FINAL]
    X_ff = X_k[top_f]
    preds, trues = [], []
    for tr, te in kf.split(X_ff, y):
        X_tr, X_te = median_impute_train_apply(X_ff.iloc[tr], X_ff.iloc[te])
        try:
            calib = CalibratedClassifierCV(
                LGBMClassifier(**LGB_PARAMS), method='isotonic', cv=3)
            calib.fit(X_tr, y[tr])
            y_pr = np.clip(calib.predict_proba(X_te)[:, 1], 0, 1)
        except Exception:
            gbm = LGBMClassifier(**LGB_PARAMS)
            gbm.fit(X_tr, y[tr])
            y_pr = np.clip(gbm.predict_proba(X_te)[:, 1], 0, 1)
        preds.append(y_pr); trues.append(y[te])

    yt_all = np.concatenate(trues); yp_all = np.concatenate(preds)
    aucs_cv = [roc_auc_score(yt, yp) for yt, yp in zip(trues, preds)]
    metrics = get_full_eval(yt_all, yp_all)
    metrics['auc_cv_mean'] = np.mean(aucs_cv)
    metrics['auc_cv_std'] = np.std(aucs_cv)
    if verbose:
        print(f"  s05: AUC={metrics['auc_cv_mean']:.4f}±{metrics['auc_cv_std']:.4f}, "
              f"Brier={metrics['brier']:.4f}")
    return metrics, top_f, sfs_hist

# ═══════════════════════════════ MAIN ═══════════════════════════════
print("=" * 72)
print("  ADNI CDR Worsening — Binary Classification v2")
print("  (Separate CN/MCI cohorts, within-fold imputation)")
print("=" * 72)

df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"\n[1] Loaded: {len(df):,} subjects")

# Define cohorts
COHORTS = {
    'CN': {
        'label': 'CN (CDR=0 → ≥0.5)',
        'filter': lambda d: (d['cdr_cohort'] == 'CN') & (d['n_post_idx_visits'] >= 1) & (d['CDR_surv_time'] > 0),
    },
    'MCI': {
        'label': 'MCI (CDR=0.5 → ≥1)',
        'filter': lambda d: (d['cdr_cohort'] == 'MCI') & (d['n_post_idx_visits'] >= 1) & (d['CDR_surv_time'] > 0),
    },
}

# Endpoints
BINARY_ENDPOINTS = {
    'primary_3yr': {'col': 'CDR_worsen_3yr', 'cens_col': 'CDR_worsen_censored_3yr', 'label': 'CDR worsen 3yr', 'window': 3},
    'primary_5yr': {'col': 'CDR_worsen_5yr', 'cens_col': 'CDR_worsen_censored_5yr', 'label': 'CDR worsen 5yr', 'window': 5},
    'primary_10yr': {'col': 'CDR_worsen_10yr', 'cens_col': 'CDR_worsen_censored_10yr', 'label': 'CDR worsen 10yr', 'window': 10},
    'sustained_3yr': {'col': 'CDR_worsen_sustained_3yr', 'cens_col': 'CDR_worsen_sustained_censored_3yr', 'label': 'Sustained 3yr', 'window': 3},
    'sustained_5yr': {'col': 'CDR_worsen_sustained_5yr', 'cens_col': 'CDR_worsen_sustained_censored_5yr', 'label': 'Sustained 5yr', 'window': 5},
    'cdrsb_3yr': {'col': 'CDRSB_worsen_3yr', 'cens_col': 'CDRSB_worsen_censored_3yr', 'label': 'CDRSB worsen 3yr', 'window': 3},
    'cdrsb_5yr': {'col': 'CDRSB_worsen_5yr', 'cens_col': 'CDRSB_worsen_censored_5yr', 'label': 'CDRSB worsen 5yr', 'window': 5},
}

all_results = []

for cohort_key in ['CN', 'MCI']:
    cohort_info = COHORTS[cohort_key]
    df_cohort = df[cohort_info['filter'](df)].copy()
    print(f"\n{'=' * 72}")
    print(f"  {cohort_info['label']} — N={len(df_cohort)}")
    print(f"{'=' * 72}")

    # Build features
    all_ex = set(ID_COLS + TARGET_COLS + CDR_COLS)
    clean_feat = [c for c in df_cohort.columns
                  if c not in all_ex
                  and not any(c.startswith(p) for p in COG_PFX)]
    print(f"  Features available: {len(clean_feat)} "
          f"({sum(1 for f in clean_feat if f.startswith(IMG_PFX))} imaging + "
          f"{sum(1 for f in clean_feat if not f.startswith(IMG_PFX))} bio)")

    for ep_key, ep_info in BINARY_ENDPOINTS.items():
        target_col = ep_info['col']
        cens_col = ep_info['cens_col']

        if target_col not in df_cohort.columns:
            continue

        print(f"\n  ── {ep_info['label']} ({ep_info['window']}yr) ──")

        wp = df_cohort.copy()
        n_censored = int(wp[cens_col].sum()) if cens_col in wp.columns else 0
        wp = wp[wp[cens_col] == 0] if cens_col in wp.columns else wp
        y = wp[target_col].values.astype(int)
        n_events = int(y.sum())

        if n_events < 5 or (len(wp) - n_events) < 5:
            print(f"  [SKIP] Too few events (n={n_events})")
            continue

        print(f"  Valid: {len(wp)} ({n_events} events, {n_events/len(wp)*100:.1f}%, "
              f"{n_censored} censored)")

        X = wp[[c for c in clean_feat if c in wp.columns]].copy()
        # Remove constant columns and non-numeric
        const_c = [c for c in X.columns if X[c].nunique() <= 1]
        X = X.drop(columns=const_c)
        num_c = X.select_dtypes(include=[np.number]).columns
        X = X[num_c]

        try:
            metrics, features, sfs_hist = run_binary_pipeline(X, y)
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue

        metrics['target'] = target_col
        metrics['label'] = ep_info['label']
        metrics['cohort'] = cohort_key
        metrics['population'] = cohort_info['label']
        metrics['window'] = ep_info['window']
        metrics['n_censored'] = n_censored
        all_results.append(metrics)

        # Save per-result files
        tag = f"{cohort_key}_{ep_key}"
        pd.DataFrame({'rank': range(1, len(features) + 1), 'feature': features}).to_csv(
            os.path.join(RESULTS_DIR, f'features_{tag}.csv'), index=False)
        pd.DataFrame(sfs_hist).to_csv(
            os.path.join(RESULTS_DIR, f'sfs_history_{tag}.csv'), index=False)
        gc.collect()

# ═══════════════════════════════ ALL-TIME ═══════════════════════════
for cohort_key in ['CN', 'MCI']:
    cohort_info = COHORTS[cohort_key]
    df_cohort = df[cohort_info['filter'](df)].copy()

    if 'CDR_worsen_alltime' not in df_cohort.columns:
        continue

    all_ex = set(ID_COLS + TARGET_COLS + CDR_COLS)
    clean_feat = [c for c in df_cohort.columns
                  if c not in all_ex
                  and not any(c.startswith(p) for p in COG_PFX)]

    wp = df_cohort.copy()
    y = wp['CDR_worsen_alltime'].values.astype(int)
    n_events = int(y.sum())
    print(f"\n  ── {cohort_key} all-time ──")
    print(f"  n={len(wp)}, events={n_events} ({n_events/len(wp)*100:.1f}%)")

    X = wp[[c for c in clean_feat if c in wp.columns]].copy()
    const_c = [c for c in X.columns if X[c].nunique() <= 1]
    X = X.drop(columns=const_c)
    num_c = X.select_dtypes(include=[np.number]).columns
    X = X[num_c]

    try:
        metrics, features, sfs_hist = run_binary_pipeline(X, y)
        metrics['target'] = 'CDR_worsen_alltime'
        metrics['label'] = f'{cohort_key} all-time'
        metrics['cohort'] = cohort_key
        metrics['window'] = 99
        all_results.append(metrics)
        tag = f"{cohort_key}_alltime"
        pd.DataFrame({'rank': range(1, len(features) + 1), 'feature': features}).to_csv(
            os.path.join(RESULTS_DIR, f'features_{tag}.csv'), index=False)
        pd.DataFrame(sfs_hist).to_csv(
            os.path.join(RESULTS_DIR, f'sfs_history_{tag}.csv'), index=False)
    except Exception as e:
        print(f"  [ERROR] {e}")

# ═══════════════════════════════ POOLED (SECONDARY) ═══════════════════
print(f"\n{'=' * 72}")
print(f"  POOLED CN+MCI (with baseline CDR group) — SECONDARY")
print(f"{'=' * 72}")

df_pooled = df[
    (df['cdr_cohort'].isin(['CN', 'MCI'])) &
    (df['n_post_idx_visits'] >= 1) &
    (df['CDR_surv_time'] > 0)
].copy()

# Create baseline CDR group feature if not exists
if 'baseline_CDR_group' not in df_pooled.columns:
    df_pooled['baseline_CDR_group'] = (df_pooled['cdr_cohort'] == 'MCI').astype(int)

all_ex_p = set(ID_COLS + TARGET_COLS + CDR_COLS)
clean_feat_p = [c for c in df_pooled.columns
                if c not in all_ex_p
                and not any(c.startswith(p) for p in COG_PFX)]
# Add baseline CDR group
if 'baseline_CDR_group' not in clean_feat_p:
    clean_feat_p = clean_feat_p + ['baseline_CDR_group']

for window_yr in [3, 5]:
    target_col = f'CDR_worsen_{window_yr}yr'
    cens_col = f'CDR_worsen_censored_{window_yr}yr'

    if target_col not in df_pooled.columns:
        continue

    print(f"\n  ── Pooled {window_yr}yr ──")
    wp = df_pooled.copy()
    n_censored = int(wp[cens_col].sum())
    wp = wp[wp[cens_col] == 0]
    y = wp[target_col].values.astype(int)
    n_events = int(y.sum())
    print(f"  Valid: {len(wp)} ({n_events} events, {n_events/len(wp)*100:.1f}%, "
          f"{n_censored} censored)")

    X = wp[[c for c in clean_feat_p if c in wp.columns]].copy()
    X = X.loc[:, ~X.columns.duplicated()]
    const_c = [c for c in X.columns if X[c].nunique() <= 1]
    X = X.drop(columns=const_c)
    num_c = X.select_dtypes(include=[np.number]).columns
    X = X[num_c]

    try:
        metrics, features, sfs_hist = run_binary_pipeline(X, y)
        metrics['target'] = target_col
        metrics['label'] = f'Pooled {window_yr}yr'
        metrics['cohort'] = 'POOLED'
        metrics['window'] = window_yr
        metrics['n_censored'] = n_censored
        all_results.append(metrics)
        tag = f"POOLED_{window_yr}yr"
        pd.DataFrame({'rank': range(1, len(features) + 1), 'feature': features}).to_csv(
            os.path.join(RESULTS_DIR, f'features_{tag}.csv'), index=False)
        pd.DataFrame(sfs_hist).to_csv(
            os.path.join(RESULTS_DIR, f'sfs_history_{tag}.csv'), index=False)
    except Exception as e:
        print(f"  [ERROR] {e}")

# ═══════════════════════════════ SUMMARY ═══════════════════════════
if all_results:
    res_df = pd.DataFrame(all_results)
    res_df.to_csv(os.path.join(RESULTS_DIR, 'cdr_binary_results_v2.csv'), index=False)

    print(f"\n{'=' * 100}")
    print(f"  CDR WORSENING BINARY CLASSIFICATION v2 — RESULTS")
    print(f"{'=' * 100}")
    print(f"\n  {'Cohort':<7} {'Window':<16} {'n':>5} {'Events':>6} "
          f"{'AUC':>8} {'±':>6} {'Brier':>8} {'Sens':>6} {'Spec':>6}")
    print(f"  {'─' * 7} {'─' * 16} {'─' * 5} {'─' * 6} {'─' * 8} {'─' * 6} {'─' * 8} {'─' * 6} {'─' * 6}")
    for _, row in res_df.iterrows():
        print(f"  {row['cohort']:<7} {row['label']:<16} {row['n_events']:>5} "
              f"{row['n_total']:>6} "
              f"{row['auc_cv_mean']:>8.4f} {row['auc_cv_std']:>5.4f} "
              f"{row['brier']:>8.4f} {row['sensitivity']:>6.3f} "
              f"{row['specificity']:>6.3f}")

    # ── Plots ──
    # Primary results plot (CN and MCI, primary endpoint only)
    primary_targets = ['CDR_worsen_3yr', 'CDR_worsen_5yr', 'CDR_worsen_10yr', 'CDR_worsen_alltime']
    primary_results = res_df[res_df['target'].isin(primary_targets)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ai, cohort in enumerate(['CN', 'MCI']):
        ax = axes[ai]
        cr = primary_results[primary_results['cohort'] == cohort]
        if len(cr) > 0:
            labels = [r['label'] for _, r in cr.iterrows()]
            aucs = [r['auc_cv_mean'] for _, r in cr.iterrows()]
            stds = [r['auc_cv_std'] for _, r in cr.iterrows()]
            x = np.arange(len(labels))
            colors = ['#3B82F6', '#F59E0B', '#EF4444', '#8B5CF6'][:len(labels)]
            ax.bar(x, aucs, 0.5, color=colors, edgecolor='white')
            ax.errorbar(x, aucs, yerr=stds, fmt='none', ecolor='#374151', capsize=5)
            for i, (a, s) in enumerate(zip(aucs, stds)):
                ax.text(i, a + s + 0.015, f'{a:.3f}', ha='center', fontsize=11, fontweight='bold')
            ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9, rotation=15)
            ax.set_ylabel('AUC (5-fold CV)', fontsize=12)
            ax.set_ylim(0.45, 1.05)
            ax.set_title(f'{cohort} Cohort: CDR Worsening Prediction', fontweight='bold', fontsize=13)
            ax.grid(axis='y', alpha=0.3); ax.axhline(y=0.5, color='gray', ls=':', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'cdr_binary_auc_v2.png'), dpi=150)
    plt.close()

    # Sensitivity analysis plot (MCI, 3yr window: primary vs sustained vs CDRSB)
    mci_3yr_targets = ['CDR_worsen_3yr', 'CDR_worsen_sustained_3yr', 'CDRSB_worsen_3yr']
    mci_sensitivity = res_df[(res_df['cohort'] == 'MCI') & (res_df['target'].isin(mci_3yr_targets))]
    if len(mci_sensitivity) >= 2:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        labels = [r['label'] for _, r in mci_sensitivity.iterrows()]
        aucs = [r['auc_cv_mean'] for _, r in mci_sensitivity.iterrows()]
        stds = [r['auc_cv_std'] for _, r in mci_sensitivity.iterrows()]
        x = np.arange(len(labels))
        ax2.bar(x, aucs, 0.5, color=['#3B82F6', '#F59E0B', '#8B5CF6'][:len(labels)], edgecolor='white')
        ax2.errorbar(x, aucs, yerr=stds, fmt='none', ecolor='#374151', capsize=5)
        for i, (a, s) in enumerate(zip(aucs, stds)):
            ax2.text(i, a + s + 0.01, f'{a:.3f}±{s:.3f}', ha='center', fontsize=11, fontweight='bold')
        ax2.set_xticks(x); ax2.set_xticklabels(labels, fontsize=9, rotation=15, ha='right')
        ax2.set_ylabel('AUC (5-fold CV)', fontsize=12)
        ax2.set_title('MCI Cohort 3yr: Sensitivity to Endpoint Definition', fontweight='bold', fontsize=14)
        ax2.set_ylim(0.45, 1.0); ax2.grid(axis='y', alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(os.path.join(RESULTS_DIR, 'endpoint_sensitivity_binary_v2.png'), dpi=150)
        plt.close()

    print(f"\n  ✅ Results saved to: {RESULTS_DIR}")
    for fname in sorted(os.listdir(RESULTS_DIR)):
        fpath = os.path.join(RESULTS_DIR, fname)
        if os.path.isfile(fpath):
            print(f"     {fname} ({os.path.getsize(fpath)/1024:.0f} KB)")

else:
    print("\n  [WARNING] No results generated!")

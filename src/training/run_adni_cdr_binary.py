#!/usr/bin/env python3
"""
ADNI CDR 恶化二分类预测 — Phase 2C-2
=====================================
基于 Phase 2A 的 s01-s05 LightGBM 管线（统一方法论），
将结局从 DXSUM Dementia 转化替换为 CDR 恶化。

CDR 恶化定义（分层阈值）:
  - 基线 CDR=0:  恶化 = CDGLOBAL ≥ 0.5 (CN → MCI)
  - 基线 CDR=0.5: 恶化 = CDGLOBAL ≥ 1.0 (MCI → 轻度痴呆)

人群: CN + MCI (baseline CDGLOBAL = 0 或 0.5, ≥2 次 CDR 访视)

用法:
  cd UKB_DRP-main
  python src/training/run_adni_cdr_binary.py --window 3,5,10
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

# ═══════════════════════════════ CONFIG ═══════════════════════════════
_BASE = os.environ.get('PROJECT_ROOT',
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_DEFAULT_ADNI_DIR = os.path.join(_BASE, 'local_data', 'adni')

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str,
                    default=os.environ.get('ADNI_DATA_DIR', _DEFAULT_ADNI_DIR),
                    help='ADNI dataset root directory')
parser.add_argument('--window', type=str, default='3,5,10',
                    help='Time windows (comma-separated: 3,5,10)')
parser.add_argument('--population', type=str, default='all',
                    choices=['all', 'cn', 'mci'],
                    help='Target population: all (CN+MCI), cn (CDR=0 only), mci (CDR=0.5 only)')
args = parser.parse_args()

DATA_DIR = os.path.join(args.data_dir, 'processed')
RESULTS_DIR = os.path.join(_BASE, 'local_data', 'Results_adni', 'cdr_binary')
os.makedirs(RESULTS_DIR, exist_ok=True)

WINDOWS = [int(w.strip()) for w in args.window.split(',')]

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

# Exclusions — same as Phase 2A/2B
COG_PFX = ['MMSE_', 'MOCA_', 'ADAS_', 'FAQ_', 'GDS_', 'NPIQ_', 'NB_', 'HACH_']
CDR_COLS = ['CDMEMORY', 'CDORIENT', 'CDJUDGE', 'CDCOMMUN',
            'CDHOME', 'CDCARE', 'CDGLOBAL', 'CDRSB']
ID_COLS = ['PTID', 'RID', 'PHASE', 'VISCODE', 'VISCODE2',
           'APOE_genotype', 'subject_id', 'entry_research_group', 'PTDOBYY']
TARGET_COLS = [
    # DXSUM targets
    'AD_status', 'MCI_status', 'dementia_status', 'DIAGNOSIS',
    'DXNORM', 'DXNODEP', 'DXMCI', 'DXDSEV',
    'DXDDUE', 'DXAD', 'DXPARK', 'DXDEP', 'DXOTHDEM',
    'AD_3yrs', 'AD_5yrs', 'AD_10yrs', 'AD_status_incident', 'AD_years',
    'Dementia_3yrs', 'Dementia_5yrs', 'Dementia_10yrs', 'Dementia_status_incident',
    'Dementia_years', 'converted_to_ad', 'converted_to_dementia', 'converted_to_mci',
    'ad_conversion_years', 'dementia_conversion_years', 'baseline_diagnosis',
    'censored_ad_3yr', 'censored_ad_5yr', 'censored_ad_10yr',
    'censored_dementia_3yr', 'censored_dementia_5yr', 'censored_dementia_10yr',
    'last_followup_years', 'diag_label',
    # CDR targets (these are the NEW labels, must be excluded from features too!)
    'CDR_worsen_3yr', 'CDR_worsen_5yr', 'CDR_worsen_10yr', 'CDR_worsen_alltime',
    'CDR_worsen_censored_3yr', 'CDR_worsen_censored_5yr', 'CDR_worsen_censored_10yr',
    'CDR_surv_time', 'CDR_surv_event',
    'CDRSB_surv_time', 'CDRSB_surv_event',
    'CDR_worsened', 'CDR_worsening_years',
    'CDR_worsened_CDRSB2', 'CDRSB_worsening_years',
    'CDR_last_followup_years', 'n_cdr_visits',
    'baseline_CDGLOBAL', 'baseline_CDRSB',
]
IMG_PFX = ('FS_', 'BSI_', 'WMH_', 'AMY_', 'TAU_', 'TAUPVC_')

# ═══════════════════════════════ UTILITIES ═══════════════════════════════
def norm_imp(d):
    s = sum(d.values())
    return d if s == 0 else {k: v / s for k, v in d.items()}

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
            'n_events': int(y_true.sum()),
            'n_total': len(y_true),
            'best_cutoff': float(best_cut),
        }
    except Exception:
        return {
            'auc': np.nan, 'ap': np.nan, 'brier': np.nan,
            'sensitivity': np.nan, 'specificity': np.nan,
            'n_events': int(y_true.sum()), 'n_total': len(y_true),
        }

def run_s01_s05(X, y, verbose=True):
    """s01-s05 LightGBM pipeline — identical to Phase 2A."""
    kf = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=True)

    # s01: LightGBM Gain
    tg_cv = Counter()
    for tr, te in kf.split(X, y):
        gbm = LGBMClassifier(**LGB_PARAMS)
        gbm.fit(X.iloc[tr], y[tr])
        tg = dict(zip(gbm.booster_.feature_name(),
                      gbm.booster_.feature_importance(importance_type='gain')))
        tg_cv += Counter(norm_imp(tg))
    s01_r = sorted(tg_cv.items(), key=lambda x: -x[1])
    top50 = [f for f, _ in s01_r[:min(TOP_S01, len(s01_r))]]
    if verbose:
        n_img = sum(1 for f in top50 if f.startswith(IMG_PFX))
        print(f"  s01: Top50 ({n_img} img) — #1={top50[0]}")

    # s02: Ward clustering
    X_t = X[top50].fillna(X[top50].median())
    corr = np.array(X_t.corr(method='spearman'))
    corr = np.nan_to_num(corr); corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)
    cl = fcluster(linkage(squareform(1 - np.abs(corr)), 'ward'),
                  t=CLUST_TH, criterion='distance')
    kept = []; seen = set()
    for f, c in zip(top50, cl):
        if c not in seen: kept.append(f); seen.add(c)
    X_k = X[kept]
    if verbose:
        print(f"  s02: {len(top50)} → {len(kept)} features")

    # s03: Re-rank
    tg_cv3 = Counter()
    for tr, te in kf.split(X_k, y):
        gbm = LGBMClassifier(**LGB_PARAMS)
        gbm.fit(X_k.iloc[tr], y[tr])
        tg = dict(zip(gbm.booster_.feature_name(),
                      gbm.booster_.feature_importance(importance_type='gain')))
        tg_cv3 += Counter(norm_imp(tg))
    s03_f = [f for f, _ in sorted(tg_cv3.items(), key=lambda x: -x[1])]
    if verbose:
        print(f"  s03: Top3={s03_f[:3]}")

    # s04: SFS
    selected = []; remaining = list(s03_f); prev_auc = 0; sfs_hist = []
    for step in range(min(TOP_SFS, len(s03_f))):
        best_f, best_auc = None, -1
        pool = remaining[:min(15, len(remaining))]
        for cand in pool:
            trial = selected + [cand]; aucs = []
            for tr, te in kf.split(X_k[trial], y):
                g = LGBMClassifier(**LGB_PARAMS)
                g.fit(X_k[trial].iloc[tr], y[tr])
                aucs.append(roc_auc_score(y[te],
                    g.predict_proba(X_k[trial].iloc[te])[:, 1]))
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

    # s05: Calibrated LGBM
    top_f = selected[:TOP_FINAL]
    X_ff = X_k[top_f]
    preds, trues = [], []
    for tr, te in kf.split(X_ff, y):
        try:
            calib = CalibratedClassifierCV(
                LGBMClassifier(**LGB_PARAMS), method='isotonic', cv=3)
            calib.fit(X_ff.iloc[tr], y[tr])
            y_pr = np.clip(calib.predict_proba(X_ff.iloc[te])[:, 1], 0, 1)
        except Exception:
            gbm = LGBMClassifier(**LGB_PARAMS)
            gbm.fit(X_ff.iloc[tr], y[tr])
            y_pr = np.clip(gbm.predict_proba(X_ff.iloc[te])[:, 1], 0, 1)
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
def main():
    print("=" * 72)
    print("  ADNI CDR Worsening Prediction — Binary Classification")
    print("=" * 72)

    # ── 1. Load data ──
    data_path = os.path.join(DATA_DIR, 'ADNI_baseline_with_time_targets_v2.csv')
    print(f"\n[1] Loading: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"    Total: {len(df):,} subjects, {len(df.columns)} columns")

    # ── 2. Filter target population ──
    if args.population == 'cn':
        target_pop = df[df['baseline_CDGLOBAL'] == 0.0].copy()
        pop_label = 'CN (CDR=0)'
    elif args.population == 'mci':
        target_pop = df[df['baseline_CDGLOBAL'] == 0.5].copy()
        pop_label = 'MCI (CDR=0.5)'
    else:
        target_pop = df[df['baseline_CDGLOBAL'].isin([0.0, 0.5])].copy()
        pop_label = 'CN + MCI'

    # Require ≥2 CDR visits and non-zero follow-up
    target_pop = target_pop[
        (target_pop['n_cdr_visits'] >= 2) &
        (target_pop['CDR_surv_time'] > 0)
    ]
    print(f"\n[2] Target population: {pop_label}")
    print(f"    Subjects: {len(target_pop)}")
    print(f"    CN: {(target_pop['baseline_CDGLOBAL']==0.0).sum()}, "
          f"MCI: {(target_pop['baseline_CDGLOBAL']==0.5).sum()}")

    # ── 3. Build features ──
    all_exclude = set(ID_COLS + TARGET_COLS + CDR_COLS)
    clean_feat = [c for c in target_pop.columns
                  if c not in all_exclude
                  and not any(c.startswith(p) for p in COG_PFX)]
    imaging_feat = [c for c in clean_feat if c.startswith(IMG_PFX)]
    bio_feat = [c for c in clean_feat if not c.startswith(IMG_PFX)]
    print(f"\n[3] Features: {len(clean_feat)} total "
          f"({len(imaging_feat)} imaging + {len(bio_feat)} bio)")

    # ── 4. Run all windows ──
    all_results = []; all_features = {}

    for w in WINDOWS:
        target_col = f'CDR_worsen_{w}yr'
        censored_col = f'CDR_worsen_censored_{w}yr'

        if target_col not in target_pop.columns:
            print(f"\n  [SKIP] {target_col} not found")
            continue

        print(f"\n{'─' * 72}")
        print(f"  CDR Worsening {w}yr — {pop_label}")
        print(f"{'─' * 72}")

        wp = target_pop.copy()
        n_censored = int(wp[censored_col].sum())
        wp = wp[wp[censored_col] == 0]
        y = wp[target_col].values.astype(int)
        n_events = int(y.sum())
        print(f"  Excluded {n_censored} censored → {len(wp)} valid "
              f"({n_events} events, {n_events/max(len(wp),1)*100:.1f}%)")

        if n_events < 5 or (len(wp) - n_events) < 5:
            print(f"  [SKIP] Too few events/negatives")
            continue

        X = wp[[c for c in clean_feat if c in wp.columns]].copy()
        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())
        const_c = [c for c in X.columns if X[c].nunique() <= 1]
        X = X.drop(columns=const_c)

        metrics, features, sfs_hist = run_s01_s05(X, y)
        metrics['target'] = target_col
        metrics['label'] = f'CDR worsen {w}yr'
        metrics['population'] = pop_label
        metrics['n_censored'] = n_censored
        all_results.append(metrics)
        all_features[target_col] = features

        pd.DataFrame({'rank': range(1, len(features) + 1),
                      'feature': features}).to_csv(
            os.path.join(RESULTS_DIR, f'features_{target_col}.csv'), index=False)
        pd.DataFrame(sfs_hist).to_csv(
            os.path.join(RESULTS_DIR, f'sfs_history_{target_col}.csv'), index=False)
        gc.collect()

    # ── Also run all-time if available ──
    if 'CDR_worsen_alltime' in target_pop.columns:
        print(f"\n{'─' * 72}")
        print(f"  CDR Worsening all-time — {pop_label}")
        print(f"{'─' * 72}")
        wp = target_pop.copy()
        y = wp['CDR_worsen_alltime'].values.astype(int)
        n_events = int(y.sum())
        print(f"  n={len(wp)}, events={n_events} ({n_events/len(wp)*100:.1f}%)")
        X = wp[[c for c in clean_feat if c in wp.columns]].copy()
        for col in X.columns:
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())
        const_c = [c for c in X.columns if X[c].nunique() <= 1]
        X = X.drop(columns=const_c)
        metrics, features, sfs_hist = run_s01_s05(X, y)
        metrics['target'] = 'CDR_worsen_alltime'
        metrics['label'] = 'CDR worsen all-time'
        metrics['population'] = pop_label
        all_results.append(metrics)
        all_features['CDR_worsen_alltime'] = features
        pd.DataFrame({'rank': range(1, len(features) + 1),
                      'feature': features}).to_csv(
            os.path.join(RESULTS_DIR, 'features_CDR_worsen_alltime.csv'), index=False)
        pd.DataFrame(sfs_hist).to_csv(
            os.path.join(RESULTS_DIR, 'sfs_history_CDR_worsen_alltime.csv'), index=False)

    if not all_results:
        print("\n[ERROR] No results generated. Check data.")
        return

    # ═══════════════════════════════ SUMMARY ═══════════════════════════════
    res_df = pd.DataFrame(all_results)
    res_df.to_csv(os.path.join(RESULTS_DIR, 'cdr_binary_results.csv'), index=False)

    print(f"\n{'=' * 90}")
    print(f"  CDR WORSENING BINARY CLASSIFICATION — RESULTS")
    print(f"{'=' * 90}")
    print(f"\n  {'Label':<28} {'n':>6} {'AUC':>8} {'±':>6} {'Brier':>8} "
          f"{'Sens':>6} {'Spec':>6}")
    print(f"  {'─' * 28} {'─' * 6} {'─' * 8} {'─' * 6} {'─' * 8} {'─' * 6} {'─' * 6}")
    for _, row in res_df.iterrows():
        print(f"  {row['label']:<28} {row['n_events']:>6} "
              f"{row['auc_cv_mean']:>8.4f} {row['auc_cv_std']:>5.4f} "
              f"{row['brier']:>8.4f} {row['sensitivity']:>6.3f} "
              f"{row['specificity']:>6.3f}")

    # ── Top features ──
    print(f"\n{'=' * 72}")
    print(f"  TOP FEATURES (CDR Binary)")
    print(f"{'=' * 72}")
    for k, feats in all_features.items():
        n_img = sum(1 for f in feats[:10] if f.startswith(IMG_PFX))
        print(f"\n  {k} ({n_img}/10 imaging):")
        for i, f in enumerate(feats[:10]):
            cat = '[IMG]' if f.startswith(IMG_PFX) else '[BIO]'
            print(f"    {i+1:>2}. {cat} {f}")

    # ── Plots ──
    make_plots(res_df, all_features)

    print(f"\n{'=' * 72}")
    print(f"  ✅ CDR Binary results saved to: {RESULTS_DIR}")
    for fname in sorted(os.listdir(RESULTS_DIR)):
        fpath = os.path.join(RESULTS_DIR, fname)
        if os.path.isfile(fpath):
            print(f"     {fname} ({os.path.getsize(fpath) / 1024:.0f} KB)")
    print(f"{'=' * 72}")

def make_plots(res_df, all_features):
    if len(res_df) == 0: return

    # AUC bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    labels = [r['label'] for _, r in res_df.iterrows()]
    aucs = [r['auc_cv_mean'] for _, r in res_df.iterrows()]
    stds = [r['auc_cv_std'] for _, r in res_df.iterrows()]
    x = np.arange(len(labels))
    colors = ['#3B82F6' if '3yr' in l else '#F59E0B' if '5yr' in l
              else '#EF4444' if '10yr' in l else '#8B5CF6' for l in labels]
    ax.bar(x, aucs, 0.5, color=colors, edgecolor='white', linewidth=0.8)
    ax.errorbar(x, aucs, yerr=stds, fmt='none', ecolor='#374151', capsize=5)
    for i, (a, s) in enumerate(zip(aucs, stds)):
        ax.text(i, a + s + 0.015, f'{a:.3f}±{s:.3f}',
                ha='center', fontsize=11, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('AUC (5-fold CV)', fontsize=12); ax.set_ylim(0.45, 1.05)
    ax.set_title(f'CDR Worsening Prediction — {args.population.upper()}',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3); ax.axhline(y=0.5, color='gray', ls=':', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'cdr_binary_auc.png'), dpi=150)
    plt.close()

    # SFS curves
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    for k, sfs_f in all_features.items():
        sfs_path = os.path.join(RESULTS_DIR, f'sfs_history_{k}.csv')
        if os.path.exists(sfs_path):
            sd = pd.read_csv(sfs_path)
            if len(sd) > 0 and 'step' in sd.columns:
                ax2.plot(sd['step'], sd['auc'], 'o-', lw=2, label=k, ms=6)
    if ax2.get_lines():
        ax2.set_xlabel('SFS Step', fontsize=12)
        ax2.set_ylabel('Cumulative AUC', fontsize=12)
        ax2.set_title('SFS Feature Accumulation — CDR Worsening',
                      fontweight='bold', fontsize=14)
        ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(os.path.join(RESULTS_DIR, 'sfs_accumulation_cdr.png'), dpi=150)
    plt.close()

if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f"\n⏱  Total time: {(time.time() - t0) / 60:.1f} min")

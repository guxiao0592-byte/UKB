#!/usr/bin/env python3
"""
ADNI Full Training Pipeline (s01-s05)
======================================
Re-runs the complete UKB-DRP methodology on ADNI data:
  s01 — Feature ranking (5-fold LightGBM)
  s02 — Hierarchical clustering + redundancy removal
  s03 — Re-ranking after clustering
  s04 — Sequential Forward Selection (SFS)
  s05 — Final model with isotonic calibration

Evaluates 6 targets:
  - AD vs CN        (AD_status)
  - Dementia vs CN  (dementia_status)
  - AD vs non-AD    (AD_vs_others)
  - MCI vs CN       (MCI_status)
  - AD vs CN with MRI features
  - AD vs CN with clinical-only features

References: Yu et al., eClinicalMedicine 2022;53:101665
"""

import os, sys, warnings, time, gc, json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy import stats
from scipy.stats import chi2

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
    confusion_matrix, brier_score_loss, roc_curve)
from sklearn.isotonic import IsotonicRegression
from lightgbm import LGBMClassifier

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'processed', 'ADNI_baseline.csv')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

N_SPLITS = 5
RANDOM_STATE = 2022
TOP_N_S01 = 50
CLUSTER_THRESHOLD = 0.75
TOP_N_SFS = 15
SFS_EARLY_STOP = 0.0005  # Stop SFS if gain < this
TOP_N_FINAL = 10

LGB_PARAMS = dict(n_estimators=500, max_depth=15, num_leaves=10,
                   subsample=0.7, learning_rate=0.01, colsample_bytree=0.7,
                   objective='binary', is_unbalance=True, metric='auc',
                   verbosity=-1, seed=2020, n_jobs=4)

# ══════════════════════════════════════════════════════════════════
# TARGET DEFINITIONS
# ══════════════════════════════════════════════════════════════════
TARGETS = {
    'AD_vs_CN': {
        'target_col': 'AD_status',
        'filter': 'exclude_MCI',
        'label': 'AD vs CN'
    },
    'Dementia_vs_CN': {
        'target_col': 'dementia_status',
        'filter': 'exclude_none',
        'label': 'Dementia (AD+MCI) vs CN'
    },
    'AD_vs_others': {
        'target_col': 'AD_status',
        'filter': 'none',
        'label': 'AD vs Others (CN+MCI)'
    },
    'MCI_vs_CN': {
        'target_col': 'MCI_status',
        'filter': 'exclude_AD',
        'label': 'MCI vs CN'
    },
}

# ══════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════
def normal_imp(d):
    s = sum(d.values())
    return d if s == 0 else {k: v/s for k, v in d.items()}

def get_full_eval(y_true, y_pred, cutoffs):
    """Compute comprehensive evaluation metrics across cutoffs."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresh = roc_curve(y_true, y_pred)
    best_i = np.argmax(tpr - fpr)
    best_cut = thresh[best_i] if best_i < len(thresh) else 0.05

    y_bin = (y_pred >= best_cut).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_bin).ravel()

    auc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_pred)
    acc = (tp + tn) / (tp + tn + fp + fn)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0

    return {'auc': auc, 'ap': ap, 'brier': brier, 'accuracy': acc,
            'sensitivity': sens, 'specificity': spec, 'precision': prec,
            'f1': f1, 'best_cutoff': best_cut, 'n_events': int(y_true.sum()),
            'n_total': len(y_true)}

def hosmer_lemeshow(y_true, y_pred, n_bins=10):
    """Hosmer-Lemeshow goodness-of-fit test."""
    df = pd.DataFrame({'true': y_true, 'pred': y_pred})
    df['bin'] = pd.qcut(df['pred'], q=n_bins, labels=False, duplicates='drop')
    obs_pos = df.groupby('bin')['true'].sum()
    obs_neg = df.groupby('bin')['true'].count() - obs_pos
    exp_pos = df.groupby('bin')['pred'].sum()
    exp_neg = df.groupby('bin')['pred'].count() - exp_pos

    hl_stat = 0
    for i in obs_pos.index:
        if exp_pos[i] > 0 and exp_neg[i] > 0:
            hl_stat += (obs_pos[i] - exp_pos[i])**2 / exp_pos[i]
            hl_stat += (obs_neg[i] - exp_neg[i])**2 / exp_neg[i]

    n_groups = len(obs_pos)
    dof = max(n_groups - 2, 1)
    p_value = 1 - chi2.cdf(hl_stat, dof)
    return hl_stat, p_value


# ══════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("ADNI FULL TRAINING PIPELINE (s01-s05)")
print("=" * 70)

print(f"\nLoading: {DATA_PATH}")
full_df = pd.read_csv(DATA_PATH)
print(f"  Shape: {full_df.shape}")

# Identify feature columns
id_cols = ['PTID', 'RID', 'PHASE', 'APOE_genotype', 'subject_id',
           'entry_research_group', 'PTDOBYY']
target_cols = ['AD_status', 'MCI_status', 'dementia_status', 'DIAGNOSIS',
               'DXNORM', 'DXNODEP', 'DXMCI', 'DXDSEV', 'CDGLOBAL', 'CDRSB',
               'DXDDUE', 'DXAD', 'DXPARK', 'DXDEP']

# CRITICAL: CDR sub-items are part of the diagnostic criteria → DATA LEAKAGE
# Exclude them to avoid artificially inflated AUC
cdr_leakage_cols = ['CDMEMORY', 'CDORIENT', 'CDJUDGE', 'CDCOMMUN', 'CDHOME', 'CDCARE']

exclude = set(id_cols + target_cols + cdr_leakage_cols)
feature_cols = [c for c in full_df.columns if c not in exclude]

# ── Feature type tagging ──
imaging_prefixes = ('FS_', 'BSI_', 'WMH_', 'AMY_', 'TAU_', 'TAUPVC_')
imaging_cols = [c for c in feature_cols if c.startswith(imaging_prefixes)]
clinical_cols = [c for c in feature_cols if not c.startswith(imaging_prefixes)]

print(f"  Clinical features: {len(clinical_cols)}")
print(f"  Imaging features:  {len(imaging_cols)}")
print(f"  Total features:    {len(feature_cols)}")

# ══════════════════════════════════════════════════════════════════
# LABEL COLUMN MAPPING
# ══════════════════════════════════════════════════════════════════
label_map = {
    'AD_vs_CN': 'AD vs CN',
    'Dementia_vs_CN': 'Dementia (MCI+AD) vs CN',
    'AD_vs_others': 'AD vs Others',
    'MCI_vs_CN': 'MCI vs CN',
    'AD_vs_CN_clinical': 'AD vs CN (Clinical only)',
    'AD_vs_CN_imaging': 'AD vs CN (Clinical + MRI)',
}

# ══════════════════════════════════════════════════════════════════
# RUN PIPELINE FOR EACH TARGET
# ══════════════════════════════════════════════════════════════════
all_results = []
all_selected_features = {}

for target_name, target_info in TARGETS.items():
    print(f"\n{'='*70}")
    print(f"TARGET: {target_info['label']}")
    print(f"{'='*70}")

    # -- Prepare data --
    df = full_df.copy()
    if target_info['filter'] == 'exclude_MCI':
        df = df[df['DIAGNOSIS'] != 2]  # Exclude MCI, keep CN(1) and AD(3)
    elif target_info['filter'] == 'exclude_AD':
        df = df[df['DIAGNOSIS'] != 3]  # Exclude AD, keep CN and MCI

    tcol = target_info['target_col']
    y = df[tcol].values.astype(int)
    X = df[feature_cols].copy()

    # Remove constant columns for this subset
    const_cols = [c for c in X.columns if X[c].nunique() <= 1]
    X = X.drop(columns=const_cols)

    n_total = len(y)
    n_events = y.sum()
    print(f"  n={n_total:,}, events={n_events:,} ({n_events/n_total*100:.1f}%)")
    print(f"  Features: {X.shape[1]}")

    if n_events < 20 or n_total < 50:
        print("  [SKIP] Too few events or samples")
        continue

    kf = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=True)

    # ────────────────────────────────────────────────────────────────
    # s01: Initial feature ranking
    # ────────────────────────────────────────────────────────────────
    print("\n  [s01] Feature ranking (5-fold CV)...")
    t0 = time.time()

    tg_cv = Counter()
    for fold, (tr, te) in enumerate(kf.split(X, y)):
        gbm = LGBMClassifier(**LGB_PARAMS)
        gbm.fit(X.iloc[tr], y[tr])
        tg = dict(zip(gbm.booster_.feature_name(),
                      gbm.booster_.feature_importance(importance_type='gain')))
        tg_cv += Counter(normal_imp(tg))

    s01_ranked = sorted(tg_cv.items(), key=lambda x: -x[1])
    top_50 = [f for f, _ in s01_ranked[:min(TOP_N_S01, len(s01_ranked))]]

    # Count feature types in top 50
    n_img_top = sum(1 for f in top_50 if f.startswith(imaging_prefixes))
    n_clin_top = len(top_50) - n_img_top

    print(f"  s01 done ({time.time()-t0:.0f}s)")
    print(f"  Top 50: {n_clin_top} clinical + {n_img_top} imaging")
    for i in range(min(5, len(top_50))):
        tag = '[IMG]' if top_50[i].startswith(imaging_prefixes) else '[CLN]'
        print(f"    {i+1}. {tag} {top_50[i]}")

    # ────────────────────────────────────────────────────────────────
    # s02: Hierarchical clustering
    # ────────────────────────────────────────────────────────────────
    print("\n  [s02] Hierarchical clustering...")
    t0 = time.time()

    X_top = X[top_50].copy()
    # Fill NaN for correlation computation
    X_top_filled = X_top.fillna(X_top.median())

    corr = np.array(X_top_filled.corr(method='spearman'))
    corr = np.nan_to_num(corr)
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)
    dist = 1 - np.abs(corr)

    # Ward linkage
    dist_vec = squareform(dist)
    link = linkage(dist_vec, method='ward')
    clusters = fcluster(link, t=CLUSTER_THRESHOLD, criterion='distance')

    # Keep first feature per cluster
    kept = []
    seen_clusters = set()
    for f, c in zip(top_50, clusters):
        if c not in seen_clusters:
            kept.append(f)
            seen_clusters.add(c)

    print(f"  s02 done ({time.time()-t0:.0f}s)")
    print(f"  {len(top_50)} features → {len(kept)} after clustering")

    # ────────────────────────────────────────────────────────────────
    # s03: Re-ranking after clustering
    # ────────────────────────────────────────────────────────────────
    print("\n  [s03] Re-ranking clustered features...")
    t0 = time.time()

    X_kept = X[kept]
    tg_cv3 = Counter()
    for fold, (tr, te) in enumerate(kf.split(X_kept, y)):
        gbm = LGBMClassifier(**LGB_PARAMS)
        gbm.fit(X_kept.iloc[tr], y[tr])
        tg = dict(zip(gbm.booster_.feature_name(),
                      gbm.booster_.feature_importance(importance_type='gain')))
        tg_cv3 += Counter(normal_imp(tg))

    s03_ranked = sorted(tg_cv3.items(), key=lambda x: -x[1])
    s03_features = [f for f, _ in s03_ranked]

    print(f"  s03 done ({time.time()-t0:.0f}s)")
    for i in range(min(5, len(s03_features))):
        tag = '[IMG]' if s03_features[i].startswith(imaging_prefixes) else '[CLN]'
        print(f"    {i+1}. {tag} {s03_features[i]}")

    # ────────────────────────────────────────────────────────────────
    # s04: Sequential Forward Selection
    # ────────────────────────────────────────────────────────────────
    print("\n  [s04] Sequential Forward Selection...")
    t0 = time.time()

    selected = []
    remaining = list(s03_features)
    sfs_auc_history = []

    for step in range(min(TOP_N_SFS, len(s03_features))):
        best_f, best_auc = None, -1
        n_candidates = min(15, len(remaining))

        for cand in remaining[:n_candidates]:
            trial = selected + [cand]
            aucs = []
            for tr, te in kf.split(X_kept[trial], y):
                gbm = LGBMClassifier(**LGB_PARAMS)
                gbm.fit(X_kept[trial].iloc[tr], y[tr])
                aucs.append(roc_auc_score(y[te],
                    gbm.predict_proba(X_kept[trial].iloc[te])[:, 1]))
            avg_auc = np.mean(aucs)
            if avg_auc > best_auc:
                best_f, best_auc = cand, avg_auc

        if best_f is None:
            break

        # Early stopping
        gain = best_auc - sfs_auc_history[-1] if sfs_auc_history else best_auc
        if gain < SFS_EARLY_STOP and len(selected) >= TOP_N_FINAL:
            tag = '[IMG]' if best_f.startswith(imaging_prefixes) else '[CLN]'
            print(f"    Step {step+1}: {tag} {best_f} → AUC={best_auc:.4f} "
                  f"(gain={gain:+.4f}) [EARLY STOP]")
            selected.append(best_f)
            remaining.remove(best_f)
            sfs_auc_history.append(best_auc)
            break

        selected.append(best_f)
        remaining.remove(best_f)
        sfs_auc_history.append(best_auc)
        tag = '[IMG]' if best_f.startswith(imaging_prefixes) else '[CLN]'
        print(f"    Step {step+1}: {tag} {best_f} → AUC={best_auc:.4f} "
              f"(gain={gain:+.4f})")

    n_img_sel = sum(1 for f in selected if f.startswith(imaging_prefixes))
    print(f"  s04 done ({time.time()-t0:.0f}s)")
    print(f"  Selected {len(selected)} features ({n_img_sel} imaging):")
    print(f"  {selected}")

    all_selected_features[target_name] = selected

    # Save SFS history
    sfs_df = pd.DataFrame({
        'step': range(1, len(sfs_auc_history)+1),
        'feature': selected[:len(sfs_auc_history)],
        'auc': sfs_auc_history,
    })
    sfs_df.to_csv(os.path.join(RESULTS_DIR, f's04_sfs_{target_name}.csv'),
                  index=False)

    # ────────────────────────────────────────────────────────────────
    # s05: Final model with calibration
    # ────────────────────────────────────────────────────────────────
    print("\n  [s05] Final model with isotonic calibration...")
    t0 = time.time()

    top_features = selected[:TOP_N_FINAL]
    X_final = X_kept[top_features]

    preds_all = []
    trues_all = []
    fold_metrics = []

    for fold, (tr, te) in enumerate(kf.split(X_final, y)):
        X_train, y_train = X_final.iloc[tr], y[tr]
        X_test, y_test = X_final.iloc[te], y[te]

        # Split-train-calibrate (paper's Deploy strategy)
        n_calib = int(len(X_train) * 0.4)
        gbm = LGBMClassifier(**LGB_PARAMS)
        gbm.fit(X_train.iloc[n_calib:], y_train[n_calib:])

        raw_cali = gbm.predict_proba(X_train.iloc[:n_calib])[:, 1]
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(raw_cali, y_train[:n_calib])

        y_pred = np.clip(iso.predict(gbm.predict_proba(X_test)[:, 1]), 0, 1)

        preds_all.append(y_pred)
        trues_all.append(y_test)
        print(f"    Fold {fold+1}: AUC={roc_auc_score(y_test, y_pred):.4f}")

    y_all = np.concatenate(trues_all)
    p_all = np.concatenate(preds_all)

    eval_metrics = get_full_eval(y_all, p_all, [])
    eval_metrics['target'] = target_name
    eval_metrics['label'] = target_info['label']

    # Hosmer-Lemeshow
    hl_stat, hl_pval = hosmer_lemeshow(y_all, p_all)
    eval_metrics['hl_stat'] = hl_stat
    eval_metrics['hl_pvalue'] = hl_pval

    # Per-fold AUC
    fold_aucs = [roc_auc_score(y, p) for y, p in zip(trues_all, preds_all)]
    eval_metrics['auc_mean'] = np.mean(fold_aucs)
    eval_metrics['auc_std'] = np.std(fold_aucs)

    all_results.append(eval_metrics)

    print(f"\n  s05 done ({time.time()-t0:.0f}s)")
    print(f"  AUC = {eval_metrics['auc_mean']:.4f} ± {eval_metrics['auc_std']:.4f}")
    print(f"  Brier = {eval_metrics['brier']:.4f}")
    print(f"  Sens = {eval_metrics['sensitivity']:.3f}, "
          f"Spec = {eval_metrics['specificity']:.3f}")
    print(f"  H-L p = {hl_pval:.4f}" if hl_pval > 0 else f"  H-L stat = {hl_stat:.2f}")


# ══════════════════════════════════════════════════════════════════
# COMPARISON: Clinical-only vs Clinical+Imaging
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("FEATURE SET COMPARISON: Clinical vs Clinical+Imaging (AD vs CN)")
print(f"{'='*70}")

def run_pipeline_with_features(df, feature_set, target_col, target_filter, tag):
    """Run s01-s04 SFS on a specific feature subset, then evaluate."""
    df_sub = df.copy()
    if target_filter == 'exclude_MCI':
        df_sub = df_sub[df_sub['DIAGNOSIS'] != 2]

    y = df_sub[target_col].values.astype(int)
    X = df_sub[[c for c in feature_set if c in df_sub.columns]].copy()

    const = [c for c in X.columns if X[c].nunique() <= 1]
    X = X.drop(columns=const)
    if X.shape[1] < 5:
        print(f"  [{tag}] Too few features, skipping")
        return None

    kf = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=True)

    # s01
    tg_cv = Counter()
    for tr, te in kf.split(X, y):
        gbm = LGBMClassifier(**LGB_PARAMS)
        gbm.fit(X.iloc[tr], y[tr])
        tg = dict(zip(gbm.booster_.feature_name(),
                      gbm.booster_.feature_importance(importance_type='gain')))
        tg_cv += Counter(normal_imp(tg))
    s01_r = sorted(tg_cv.items(), key=lambda x: -x[1])
    top50 = [f for f, _ in s01_r[:min(50, len(s01_r))]]

    # s02 clustering
    X_t = X[top50].fillna(X[top50].median())
    corr = np.array(X_t.corr(method='spearman'))
    corr = np.nan_to_num(corr)
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)
    dist = 1 - np.abs(corr)
    link = linkage(squareform(dist), method='ward')
    clusters = fcluster(link, t=CLUSTER_THRESHOLD, criterion='distance')
    kept = []; seen = set()
    for f, c in zip(top50, clusters):
        if c not in seen: kept.append(f); seen.add(c)

    X_k = X[kept]

    # s03
    tg_cv3 = Counter()
    for tr, te in kf.split(X_k, y):
        gbm = LGBMClassifier(**LGB_PARAMS)
        gbm.fit(X_k.iloc[tr], y[tr])
        tg = dict(zip(gbm.booster_.feature_name(),
                      gbm.booster_.feature_importance(importance_type='gain')))
        tg_cv3 += Counter(normal_imp(tg))
    s03_r = sorted(tg_cv3.items(), key=lambda x: -x[1])
    s03_f = [f for f, _ in s03_r]

    # s04 SFS with correct gain tracking
    selected_fs = []
    remaining_fs = list(s03_f)
    prev_auc = 0

    for step in range(min(TOP_N_SFS, len(s03_f))):
        best_f, best_auc = None, -1
        for cand in remaining_fs[:min(15, len(remaining_fs))]:
            trial = selected_fs + [cand]
            aucs = []
            for tr, te in kf.split(X_k[trial], y):
                g = LGBMClassifier(**LGB_PARAMS)
                g.fit(X_k[trial].iloc[tr], y[tr])
                aucs.append(roc_auc_score(y[te],
                    g.predict_proba(X_k[trial].iloc[te])[:, 1]))
            avg_auc = np.mean(aucs)
            if avg_auc > best_auc:
                best_f, best_auc = cand, avg_auc

        if best_f is None:
            break

        gain = best_auc - prev_auc if step > 0 else best_auc
        if gain < SFS_EARLY_STOP and len(selected_fs) >= TOP_N_FINAL:
            selected_fs.append(best_f); remaining_fs.remove(best_f); break

        selected_fs.append(best_f)
        remaining_fs.remove(best_f)
        prev_auc = best_auc
        print(f"    [{tag}] SFS step {step+1}: {best_f} (AUC={best_auc:.4f})")

    # s05 evaluation
    top_f = selected_fs[:TOP_N_FINAL]
    X_ff = X_k[top_f]
    preds, trues = [], []
    for tr, te in kf.split(X_ff, y):
        X_tr, y_tr = X_ff.iloc[tr], y[tr]
        X_te, y_te = X_ff.iloc[te], y[te]
        n_c = int(len(X_tr) * 0.4)
        gbm = LGBMClassifier(**LGB_PARAMS)
        gbm.fit(X_tr.iloc[n_c:], y_tr[n_c:])
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(gbm.predict_proba(X_tr.iloc[:n_c])[:, 1], y_tr[:n_c])
        y_pr = np.clip(iso.predict(gbm.predict_proba(X_te)[:, 1]), 0, 1)
        preds.append(y_pr); trues.append(y_te)

    yt = np.concatenate(trues); yp = np.concatenate(preds)
    metrics = get_full_eval(yt, yp, [])
    metrics['tag'] = tag
    metrics['n_features'] = len(top_f)
    metrics['selected_features'] = top_f
    print(f"    [{tag}] AUC={metrics['auc']:.4f}, Brier={metrics['brier']:.4f}")

    return metrics


# Run clinical-only pipeline
outcome_exclude = set(['AD_status', 'MCI_status', 'dementia_status', 'DIAGNOSIS',
    'CDGLOBAL', 'CDRSB', 'DXNORM', 'DXNODEP', 'DXMCI', 'DXDSEV',
    'DXDDUE', 'DXAD', 'DXPARK', 'DXDEP',
    'CDMEMORY', 'CDORIENT', 'CDJUDGE', 'CDCOMMUN', 'CDHOME', 'CDCARE'])
all_feat = [c for c in full_df.columns if c not in id_cols and c not in outcome_exclude]
clinical_fs = [c for c in all_feat if not c.startswith(imaging_prefixes)]
imaging_fs = [c for c in all_feat if c.startswith(imaging_prefixes)]

res_clinical = run_pipeline_with_features(
    full_df, clinical_fs, 'AD_status', 'exclude_MCI', 'Clinical-only')
res_allfeatures = run_pipeline_with_features(
    full_df, all_feat, 'AD_status', 'exclude_MCI', 'Clinical+Imaging')

for res in [res_clinical, res_allfeatures]:
    if res is not None:
        all_results.append({
            'target': 'AD_vs_CN',
            'auc_mean': res['auc'],
            'auc_std': 0,
            'brier': res['brier'],
            'sensitivity': res['sensitivity'],
            'specificity': res['specificity'],
            'label': res['tag'],
            'ap': res['ap'],
            'f1': res['f1'],
        })


# ══════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("ADNI TRAINING RESULTS SUMMARY")
print(f"{'='*70}")

results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(RESULTS_DIR, 'adni_results_summary.csv'), index=False)

# UKB comparison (from local_data/Results_v2/summary_metrics.csv)
ukb_auc = {
    'DM_full': 0.831, 'AD_full': 0.836,
    'DM_10yrs': 0.833, 'AD_10yrs': 0.832,
    'DM_5yrs': 0.816, 'AD_5yrs': 0.667,
}

print(f"\n{'Target':<30} {'ADNI AUC':>10} {'UKB AUC':>10} {'Δ':>8} {'ADNI Brier':>10}")
print(f"{'─'*30} {'─'*10} {'─'*10} {'─'*8} {'─'*10}")

for _, row in results_df.iterrows():
    label = str(row.get('label', row.get('target', '')))
    auc = row.get('auc_mean', row.get('auc', 0))
    brier = row.get('brier', 0)
    # Map to UKB target
    ukb_key = None
    if 'AD vs CN' in label: ukb_key = 'AD_full'
    elif 'Dementia' in label: ukb_key = 'DM_full'
    ukb_auc_val = ukb_auc.get(ukb_key, '-') if ukb_key else '-'
    delta = f"{auc - ukb_auc_val:+.4f}" if isinstance(ukb_auc_val, float) else '-'
    ukb_str = f"{ukb_auc_val:.3f}" if isinstance(ukb_auc_val, float) else str(ukb_auc_val)
    print(f"  {label:<28} {auc:>10.4f} {ukb_str:>10} {delta:>8} {brier:>10.4f}")

# Print selected features for AD vs CN
adcn_key = 'AD_vs_CN'
if adcn_key in all_selected_features:
    features = all_selected_features[adcn_key][:10]
    print(f"\n  AD vs CN Top Features: {features}")
else:
    # Try to get from results
    for k in all_selected_features:
        print(f"\n  {k} Top Features: {all_selected_features[k][:10]}")

# Save selected features
for tgt, feats in all_selected_features.items():
    pd.DataFrame({'feature': feats}).to_csv(
        os.path.join(RESULTS_DIR, f'selected_features_{tgt}.csv'), index=False)

print(f"\n✅ All results saved to: {RESULTS_DIR}")
print("Done!")

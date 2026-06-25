#!/usr/bin/env python3
"""
ADNI MCI→AD Conversion Prediction
==================================
Specialized pipeline focused on the clinically relevant task:
  Predict whether a patient with MCI will convert to AD, and when.

Targets (all from baseline MCI subjects):
  - MCI→AD 5yr: conversion within 5 years
  - MCI→AD 10yr: conversion within 10 years
  - MCI→AD all-time: any conversion during follow-up

Feature sets (no cognitive test leakage):
  - Biomarkers + Imaging (clean)
  - Demographics + Biomarkers only (clinical)
  - Full (incl. cognitive tests, for comparison)

Contrasts with UKB: this is a PROGNOSTIC model for memory clinic patients,
not a POPULATION SCREENING model for the general public.
"""

import os, sys, warnings, time, gc
import numpy as np
import pandas as pd
from collections import Counter
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
    confusion_matrix, brier_score_loss, roc_curve)
from sklearn.isotonic import IsotonicRegression
from lightgbm import LGBMClassifier
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'processed', 'ADNI_baseline_with_time_targets.csv')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'mci_to_ad')
os.makedirs(RESULTS_DIR, exist_ok=True)

N_SPLITS = 5; RANDOM_STATE = 2022
TOP_N_S01 = 50; CLUSTER_THRESHOLD = 0.75
TOP_N_SFS = 15; TOP_N_FINAL = 10
SFS_EARLY_STOP = 0.0005

LGB_PARAMS = dict(n_estimators=500, max_depth=15, num_leaves=10,
                   subsample=0.7, learning_rate=0.01, colsample_bytree=0.7,
                   objective='binary', is_unbalance=True, metric='auc',
                   verbosity=-1, seed=2020, n_jobs=4)

# ── Feature leakage exclusion ──
COGNITIVE_LEAKAGE_PREFIXES = [
    'MMSE_', 'MOCA_', 'ADAS_', 'FAQ_', 'GDS_', 'NPIQ_', 'NB_', 'HACH_',
]
CDR_LEAKAGE_COLS = ['CDMEMORY', 'CDORIENT', 'CDJUDGE', 'CDCOMMUN',
                     'CDHOME', 'CDCARE', 'CDGLOBAL', 'CDRSB']
ID_COLS = ['PTID', 'RID', 'PHASE', 'APOE_genotype', 'subject_id',
           'entry_research_group', 'PTDOBYY']
ALL_TARGET_NAMES = ['AD_status', 'MCI_status', 'dementia_status', 'DIAGNOSIS',
    'DXNORM', 'DXNODEP', 'DXMCI', 'DXDSEV', 'CDGLOBAL', 'CDRSB',
    'DXDDUE', 'DXAD', 'DXPARK', 'DXDEP',
    'AD_5yrs', 'AD_10yrs', 'AD_status_incident', 'AD_years',
    'Dementia_5yrs', 'Dementia_10yrs', 'Dementia_status_incident',
    'Dementia_years', 'converted_to_ad', 'converted_to_dementia',
    'converted_to_mci', 'ad_conversion_years', 'dementia_conversion_years',
    'baseline_diagnosis']

imaging_pfx = ('FS_', 'BSI_', 'WMH_', 'AMY_', 'TAU_', 'TAUPVC_')

# ══════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════
def normal_imp(d):
    s = sum(d.values())
    return d if s == 0 else {k: v/s for k, v in d.items()}

def get_full_eval(y_true, y_pred):
    fpr, tpr, thresh = roc_curve(y_true, y_pred)
    best_i = np.argmax(tpr - fpr)
    best_cut = thresh[best_i] if best_i < len(thresh) else 0.05
    y_bin = (y_pred >= best_cut).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_bin).ravel()
    return {
        'auc': roc_auc_score(y_true, y_pred),
        'ap': average_precision_score(y_true, y_pred),
        'brier': brier_score_loss(y_true, y_pred),
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'n_events': int(y_true.sum()),
        'n_total': len(y_true),
        'best_cutoff': best_cut,
    }

def run_s01_s05(X, y, tag="", verbose=True):
    """Run full s01-s05 pipeline."""
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
    top50 = [f for f, _ in s01_r[:min(TOP_N_S01, len(s01_r))]]
    if verbose:
        n_img = sum(1 for f in top50 if f.startswith(imaging_pfx))
        print(f"  s01: Top50 ({n_img} imaging) — #{1}: {top50[0]}")

    # s02
    X_t = X[top50].fillna(X[top50].median())
    corr = np.array(X_t.corr(method='spearman'))
    corr = np.nan_to_num(corr); corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)
    link = linkage(squareform(1 - np.abs(corr)), method='ward')
    clusters = fcluster(link, t=CLUSTER_THRESHOLD, criterion='distance')
    kept = []; seen = set()
    for f, c in zip(top50, clusters):
        if c not in seen: kept.append(f); seen.add(c)
    X_k = X[kept]
    if verbose: print(f"  s02: {len(top50)} → {len(kept)} after clustering")

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
    if verbose:
        n_img = sum(1 for f in s03_f[:5] if f.startswith(imaging_pfx))
        print(f"  s03: Top5 ({n_img} imaging) — {s03_f[:3]}")

    # s04
    selected = []; remaining = list(s03_f); prev_auc = 0
    for step in range(min(TOP_N_SFS, len(s03_f))):
        best_f, best_auc = None, -1
        for cand in remaining[:min(15, len(remaining))]:
            trial = selected + [cand]
            aucs = []
            for tr, te in kf.split(X_k[trial], y):
                g = LGBMClassifier(**LGB_PARAMS)
                g.fit(X_k[trial].iloc[tr], y[tr])
                aucs.append(roc_auc_score(y[te],
                    g.predict_proba(X_k[trial].iloc[te])[:, 1]))
            avg_auc = np.mean(aucs)
            if avg_auc > best_auc: best_f, best_auc = cand, avg_auc
        if best_f is None: break
        gain = best_auc - prev_auc if step > 0 else best_auc
        if gain < SFS_EARLY_STOP and len(selected) >= TOP_N_FINAL:
            selected.append(best_f); remaining.remove(best_f); break
        selected.append(best_f); remaining.remove(best_f); prev_auc = best_auc
        if verbose:
            tag2 = '[IMG]' if best_f.startswith(imaging_pfx) else '[CLN]'
            print(f"  s04 step{step+1}: {tag2} {best_f} AUC={best_auc:.4f} (gain={gain:+.4f})")

    # s05 — use CalibratedClassifierCV for small sample safety (no data-splitting)
    top_f = selected[:TOP_N_FINAL]
    X_ff = X_k[top_f]
    preds, trues = [], []
    from sklearn.calibration import CalibratedClassifierCV
    for tr, te in kf.split(X_ff, y):
        X_tr, y_tr = X_ff.iloc[tr], y[tr]
        X_te, y_te = X_ff.iloc[te], y[te]
        gbm = LGBMClassifier(**LGB_PARAMS)
        calib = CalibratedClassifierCV(gbm, method='isotonic', cv=3)
        calib.fit(X_tr, y_tr)
        y_pr = np.clip(calib.predict_proba(X_te)[:, 1], 0, 1)
        preds.append(y_pr); trues.append(y_te)

    yt = np.concatenate(trues); yp = np.concatenate(preds)
    metrics = get_full_eval(yt, yp)
    metrics['auc_cv_mean'] = np.mean([roc_auc_score(yt, yp) for yt, yp in zip(trues, preds)])
    metrics['auc_cv_std'] = np.std([roc_auc_score(yt, yp) for yt, yp in zip(trues, preds)])

    if verbose:
        n_img = sum(1 for f in top_f if f.startswith(imaging_pfx))
        print(f"  s05: AUC={metrics['auc_cv_mean']:.4f}±{metrics['auc_cv_std']:.4f}, "
              f"Brier={metrics['brier']:.4f}, Sens={metrics['sensitivity']:.3f}, "
              f"Spec={metrics['specificity']:.3f}")
        print(f"  Top10: {top_f[:5]}... ({n_img} imaging)")

    return metrics, top_f


# ══════════════════════════════════════════════════════════════════
# LOAD DATA & FILTER TO BASELINE MCI
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("ADNI MCI→AD CONVERSION PREDICTION")
print("=" * 70)

print(f"\nLoading: {DATA_PATH}")
full_df = pd.read_csv(DATA_PATH)

# Keep only MCI at baseline
mci_df = full_df[full_df['baseline_diagnosis'] == 2.0].copy()
print(f"  Baseline MCI subjects: {len(mci_df):,}")

# Verify targets
for col in ['AD_5yrs', 'AD_10yrs', 'AD_status_incident']:
    n = mci_df[col].sum()
    print(f"  {col}: {n} events ({n/len(mci_df)*100:.1f}%)")

# Build clean feature set
all_exclude = set(ID_COLS + ALL_TARGET_NAMES + CDR_LEAKAGE_COLS)
all_feat = [c for c in mci_df.columns if c not in all_exclude]
clean_feat = [c for c in all_feat if not any(c.startswith(p) for p in COGNITIVE_LEAKAGE_PREFIXES)]
clean_clinical = [c for c in clean_feat if not c.startswith(imaging_pfx)]
clean_imaging = [c for c in clean_feat if c.startswith(imaging_pfx)]

print(f"\n  Clean clinical features: {len(clean_clinical)}")
print(f"  Clean imaging features:  {len(clean_imaging)}")

# ══════════════════════════════════════════════════════════════════
# TARGET DEFINITIONS: MCI→AD
# ══════════════════════════════════════════════════════════════════
MCI_TARGETS = [
    ('AD_5yrs',             'MCI→AD within 5yr'),
    ('AD_10yrs',            'MCI→AD within 10yr'),
    ('AD_status_incident',  'MCI→AD all-time'),
]

# For comparison: also run on the old cross-sectional targets
# (AD vs CN within the MCI+AD subset, to show why it's not the right task)
XTRA_TARGETS = [
    ('AD_status', 'AD vs Others (among MCI+AD)', 'cross_sectional'),
]

# ══════════════════════════════════════════════════════════════════
# RUN: Model A — Biomarkers + Imaging (clean)
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("MODEL A: Biomarkers + Imaging (No cognitive tests)")
print(f"{'='*70}")

results_clean = []
features_clean = {}

for target_col, label in MCI_TARGETS:
    print(f"\n── {label} ({target_col}) ──")

    # Apply censoring: exclude non-converters with insufficient follow-up
    df_target = mci_df.copy()
    n_before = len(df_target)
    if target_col == 'AD_5yrs':
        n_censored = df_target['censored_5yr'].sum()
        df_target = df_target[df_target['censored_5yr'] == 0]
    elif target_col == 'AD_10yrs':
        n_censored = df_target['censored_10yr'].sum()
        df_target = df_target[df_target['censored_10yr'] == 0]
    else:
        n_censored = 0  # all-time: keep all subjects
    if n_censored > 0:
        print(f"  Censored (insufficient f/u): {n_censored} → retained {len(df_target)}")

    y = df_target[target_col].values.astype(int)
    n_total = len(y); n_events = y.sum()
    print(f"  n={n_total:,}, events={n_events:,} ({n_events/n_total*100:.1f}%)")

    X = df_target[[c for c in clean_feat if c in df_target.columns]].copy()
    const_c = [c for c in X.columns if X[c].nunique() <= 1]
    X = X.drop(columns=const_c)

    metrics, features = run_s01_s05(X, y, tag=target_col)
    metrics['target'] = target_col
    metrics['label'] = label
    metrics['model'] = 'Biomarkers + Imaging'
    metrics['n_censored'] = n_censored
    results_clean.append(metrics)
    features_clean[target_col] = features

    pd.DataFrame({'feature': features}).to_csv(
        os.path.join(RESULTS_DIR, f'features_clean_{target_col}.csv'), index=False)
    gc.collect()

# ══════════════════════════════════════════════════════════════════
# RUN: Model B — Demographics + Biomarkers only
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("MODEL B: Demographics + Biomarkers only")
print(f"{'='*70}")

results_clin = []

for target_col, label in MCI_TARGETS:
    print(f"\n── {label} ({target_col}) ──")

    # Apply censoring
    df_target = mci_df.copy()
    if target_col == 'AD_5yrs':
        n_censored = df_target['censored_5yr'].sum()
        df_target = df_target[df_target['censored_5yr'] == 0]
    elif target_col == 'AD_10yrs':
        n_censored = df_target['censored_10yr'].sum()
        df_target = df_target[df_target['censored_10yr'] == 0]
    else:
        n_censored = 0
    if n_censored > 0:
        print(f"  Censored (insufficient f/u): {n_censored} → retained {len(df_target)}")

    y = df_target[target_col].values.astype(int)

    X = df_target[[c for c in clean_clinical if c in df_target.columns]].copy()
    const_c = [c for c in X.columns if X[c].nunique() <= 1]
    X = X.drop(columns=const_c)

    if X.shape[1] < 5:
        print(f"  [SKIP] Too few features ({X.shape[1]})")
        continue

    print(f"  Features: {X.shape[1]}")
    metrics, features = run_s01_s05(X, y, tag=f"{target_col}_clin")
    metrics['target'] = target_col
    metrics['label'] = label
    metrics['model'] = 'Demographics + Biomarkers'
    metrics['n_censored'] = n_censored
    results_clin.append(metrics)

    pd.DataFrame({'feature': features}).to_csv(
        os.path.join(RESULTS_DIR, f'features_clinical_{target_col}.csv'), index=False)
    gc.collect()

# ══════════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print("MCI→AD CONVERSION PREDICTION — RESULTS")
print(f"{'='*90}")

all_res = results_clean + results_clin
res_df = pd.DataFrame(all_res)
res_df.to_csv(os.path.join(RESULTS_DIR, 'mci_to_ad_results.csv'), index=False)

# Print clean model summary
print(f"\n{'Model':<35} {'Target':<25} {'AUC':>8} {'Brier':>8} {'Sens':>6} {'Spec':>6} {'n':>6}")
print(f"{'─'*35} {'─'*25} {'─'*8} {'─'*8} {'─'*6} {'─'*6} {'─'*6}")
for _, row in res_df.iterrows():
    print(f"  {row['model']:<33} {row['label']:<25} "
          f"{row['auc_cv_mean']:>8.4f} {row['brier']:>8.4f} "
          f"{row['sensitivity']:>6.3f} {row['specificity']:>6.3f} "
          f"{row['n_events']:>6}")

# Delta between Biomarkers+Imaging vs Biomarkers only
print(f"\n{'Target':<30} {'Bio+Img AUC':>12} {'Bio only AUC':>12} {'ΔImg':>8}")
print(f"{'─'*30} {'─'*12} {'─'*12} {'─'*8}")
for tcol, label in MCI_TARGETS:
    r_img = res_df[(res_df['target'] == tcol) & (res_df['model'] == 'Biomarkers + Imaging')]
    r_clin = res_df[(res_df['target'] == tcol) & (res_df['model'] == 'Demographics + Biomarkers')]
    if len(r_img) > 0 and len(r_clin) > 0:
        auc_img = r_img['auc_cv_mean'].values[0]
        auc_clin = r_clin['auc_cv_mean'].values[0]
        print(f"  {label:<28} {auc_img:>12.4f} {auc_clin:>12.4f} {auc_img-auc_clin:>+8.4f}")

# ══════════════════════════════════════════════════════════════════
# BAR CHART: MCI→AD AUC by model
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 6))

labels_short = [r['label'].replace('MCI→AD ', '') for _, r in res_df.iterrows()
                if r['model'] == 'Biomarkers + Imaging']
aucs_img = [r['auc_cv_mean'] for _, r in res_df.iterrows()
            if r['model'] == 'Biomarkers + Imaging']
aucs_clin = [r['auc_cv_mean'] for _, r in res_df.iterrows()
             if r['model'] == 'Demographics + Biomarkers']
# Align lengths
n_targets = len(labels_short)
aucs_clin = aucs_clin[:n_targets]

x = np.arange(n_targets)
w = 0.35

ax.bar(x - w/2, aucs_img, w, label='Biomarkers + Imaging', color='#3B82F6', edgecolor='white')
ax.bar(x + w/2, aucs_clin, w, label='Demographics + Biomarkers only', color='#93C5FD', edgecolor='white')

for i, (auc_i, auc_c) in enumerate(zip(aucs_img, aucs_clin)):
    ax.text(i - w/2, auc_i + 0.01, f'{auc_i:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax.text(i + w/2, auc_c + 0.01, f'{auc_c:.3f}', ha='center', fontsize=9)
    delta = auc_i - auc_c
    ax.annotate(f'Δ{delta:+.3f}', xy=(i, max(auc_i, auc_c) + 0.06),
                ha='center', fontsize=8, color='#EF4444' if delta < 0 else '#10B981',
                fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(labels_short, fontsize=11)
ax.set_ylabel('AUC', fontsize=12)
ax.set_ylim(0.5, 1.05)
ax.set_title('ADNI: MCI→AD Conversion Prediction\n(Prognostic model for memory clinic patients)',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'mci_to_ad_auc.png'), dpi=150)
plt.close()

# ══════════════════════════════════════════════════════════════════
# TOP FEATURES HEATMAP
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("TOP FEATURES: MCI→AD all-time")
print(f"{'='*70}")

if 'AD_status_incident' in features_clean:
    top_f = features_clean['AD_status_incident'][:10]
    for i, f in enumerate(top_f):
        cat = 'IMG' if f.startswith(imaging_pfx) else 'BIO'
        print(f"  {i+1:>2}. [{cat}] {f}")

print(f"\n✅ Results saved to: {RESULTS_DIR}")
print("Done!")

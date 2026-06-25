#!/usr/bin/env python3
"""
ADNI Clean Predictive Model — No Cognitive Leakage
===================================================
Runs s01-s05 pipeline using ONLY features available BEFORE clinical assessment:
  - Demographics (age, sex, education)
  - Genetics (APOE4)
  - CSF/Plasma biomarkers
  - MRI FreeSurfer volumes
  - Amyloid/Tau PET

Excludes all cognitive tests (MMSE, ADAS, MOCA, FAQ, neuropsych battery, GDS, NPI-Q, Hachinski)
that are part of the diagnostic workup in ADNI.

Then compares:
  A) Biomarkers+Imaging model (clean prediction)
  B) Full model (clinical assist, includes cognitive tests)
  C) UK Biobank results (population-based reference)
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
DATA_PATH = os.path.join(BASE_DIR, 'processed', 'ADNI_baseline.csv')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'clean_comparison')
os.makedirs(RESULTS_DIR, exist_ok=True)

N_SPLITS = 5; RANDOM_STATE = 2022; TOP_N_S01 = 50
CLUSTER_THRESHOLD = 0.75; TOP_N_SFS = 15; TOP_N_FINAL = 10
SFS_EARLY_STOP = 0.0005

LGB_PARAMS = dict(n_estimators=500, max_depth=15, num_leaves=10,
                   subsample=0.7, learning_rate=0.01, colsample_bytree=0.7,
                   objective='binary', is_unbalance=True, metric='auc',
                   verbosity=-1, seed=2020, n_jobs=4)

# ══════════════════════════════════════════════════════════════════
# IDENTIFY COGNITIVE LEAKAGE FEATURES
# ══════════════════════════════════════════════════════════════════
# These are used to MAKE the diagnosis — including them is circular
COGNITIVE_LEAKAGE_PREFIXES = [
    'MMSE_', 'MOCA_', 'ADAS_', 'FAQ_', 'GDS_', 'NPIQ_',
    'NB_',       # Neuropsychological Battery
    'HACH_',     # Hachinski Ischemia Score
    'CCI_',      # Charlson Comorbidity Index (medical history — borderline)
]
COGNITIVE_LEAKAGE_COLS = [
    'CDMEMORY', 'CDORIENT', 'CDJUDGE', 'CDCOMMUN', 'CDHOME', 'CDCARE',
    'CDGLOBAL', 'CDRSB',
]
# We'll keep CCI (comorbidity) as it's medical history, not cognitive test

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
    auc = roc_auc_score(y_true, y_pred)
    ap = average_precision_score(y_true, y_pred)
    brier = brier_score_loss(y_true, y_pred)
    acc = (tp + tn) / (tp + tn + fp + fn)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return {'auc': auc, 'ap': ap, 'brier': brier, 'accuracy': acc,
            'sensitivity': sens, 'specificity': spec, 'n_events': int(y_true.sum()),
            'n_total': len(y_true), 'best_cutoff': best_cut}

def run_s01_s05(X, y, tag="", verbose=True):
    """Run s01-s05 pipeline. Returns (metrics, selected_features)."""
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

    # s02 clustering
    X_t = X[top50].fillna(X[top50].median())
    corr = np.array(X_t.corr(method='spearman'))
    corr = np.nan_to_num(corr)
    corr = (corr + corr.T) / 2; np.fill_diagonal(corr, 1)
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

    # s04 SFS
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
        selected.append(best_f); remaining.remove(best_f)
        prev_auc = best_auc

    # s05
    top_f = selected[:TOP_N_FINAL]
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
    metrics = get_full_eval(yt, yp)
    metrics['auc_cv_mean'] = np.mean([roc_auc_score(yt, yp) for yt, yp in zip(trues, preds)])
    metrics['auc_cv_std'] = np.std([roc_auc_score(yt, yp) for yt, yp in zip(trues, preds)])

    if verbose:
        print(f"  [{tag}] AUC={metrics['auc']:.4f}, Brier={metrics['brier']:.4f}, "
              f"Sens={metrics['sensitivity']:.3f}, Spec={metrics['specificity']:.3f}")
        print(f"  [{tag}] Top features: {top_f[:5]}")

    return metrics, top_f


# ══════════════════════════════════════════════════════════════════
# LOAD & PREPARE
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("ADNI CLEAN PREDICTIVE MODEL — No Cognitive Test Leakage")
print("=" * 70)

full_df = pd.read_csv(DATA_PATH)
print(f"Loaded: {full_df.shape}")

id_cols = ['PTID', 'RID', 'PHASE', 'APOE_genotype', 'subject_id',
           'entry_research_group', 'PTDOBYY']
target_cols = ['AD_status', 'MCI_status', 'dementia_status', 'DIAGNOSIS',
               'DXNORM', 'DXNODEP', 'DXMCI', 'DXDSEV', 'CDGLOBAL', 'CDRSB',
               'DXDDUE', 'DXAD', 'DXPARK', 'DXDEP']

all_exclude = set(id_cols + target_cols + COGNITIVE_LEAKAGE_COLS)

# Classify all columns
all_feat = [c for c in full_df.columns if c not in all_exclude]
cog_leak = [c for c in all_feat if any(c.startswith(p) for p in COGNITIVE_LEAKAGE_PREFIXES)]
clean_feat = [c for c in all_feat if not any(c.startswith(p) for p in COGNITIVE_LEAKAGE_PREFIXES)]

imaging_pfx = ('FS_', 'BSI_', 'WMH_', 'AMY_', 'TAU_', 'TAUPVC_')
clean_clinical = [c for c in clean_feat if not c.startswith(imaging_pfx)]
clean_imaging = [c for c in clean_feat if c.startswith(imaging_pfx)]

print(f"\nFeature breakdown:")
print(f"  Cognitive tests (excluded):  {len(cog_leak)}")
print(f"  Clean clinical (demographics, genetics, biomarkers): {len(clean_clinical)}")
print(f"  Clean imaging (MRI, PET):    {len(clean_imaging)}")
print(f"  Total clean features:        {len(clean_feat)}")

# ── Target definitions ──
TARGETS = [
    ('AD_vs_CN', 'AD_status', [(1, 3)], 'AD vs CN'),
    ('AD_vs_MCI+CN', 'AD_status', [(1,), (2,), (3,)], 'AD vs All Others'),
    ('Dementia_vs_CN', 'dementia_status', [(1,), (2,), (3,)], 'Dementia vs CN'),
    ('MCI_vs_CN', 'MCI_status', [(1, 2)], 'MCI vs CN'),
]

# ══════════════════════════════════════════════════════════════════
# RUN MODELS
# ══════════════════════════════════════════════════════════════════
comparison_rows = []

for target_name, tcol, diag_filter, label in TARGETS:
    print(f"\n{'='*70}")
    print(f"TARGET: {label}")
    print(f"{'='*70}")

    # Filter to relevant diagnostic groups
    df = full_df.copy()
    if len(diag_filter) == 1 and isinstance(diag_filter[0], tuple):
        include = diag_filter[0]
        df = df[df['DIAGNOSIS'].isin(include)]
    elif target_name == 'AD_vs_CN':
        df = df[df['DIAGNOSIS'] != 2]
    elif target_name == 'MCI_vs_CN':
        df = df[df['DIAGNOSIS'] != 3]

    y = df[tcol].values.astype(int)
    n_total = len(y); n_events = y.sum()
    print(f"  n={n_total:,}, events={n_events:,} ({n_events/n_total*100:.1f}%)")

    if n_events < 20 or n_total < 50:
        print("  [SKIP] Insufficient data")
        continue

    # ── Model A: Biomarkers + Imaging only (CLEAN) ──
    X_clean = df[[c for c in clean_feat if c in df.columns]].copy()
    const_c = [c for c in X_clean.columns if X_clean[c].nunique() <= 1]
    X_clean = X_clean.drop(columns=const_c)
    print(f"  [Clean] Features: {X_clean.shape[1]} "
          f"({sum(1 for c in X_clean.columns if c.startswith(imaging_pfx))} imaging)")

    clean_metrics, clean_feats = run_s01_s05(X_clean, y, tag="CLEAN")
    clean_metrics['target'] = target_name
    clean_metrics['label'] = label
    clean_metrics['model'] = 'Biomarkers + Imaging (No cognitive tests)'
    clean_metrics['n_features'] = X_clean.shape[1]

    # ── Model B: Clean clinical only (demographics + genetics + biomarkers) ──
    X_clin = df[[c for c in clean_clinical if c in df.columns]].copy()
    const_cl = [c for c in X_clin.columns if X_clin[c].nunique() <= 1]
    X_clin = X_clin.drop(columns=const_cl)
    n_features_clinical = X_clin.shape[1]
    if n_features_clinical >= 5:
        print(f"  [Clinical] Features: {n_features_clinical}")

        clin_metrics, clin_feats = run_s01_s05(X_clin, y, tag="CLINICAL-ONLY")
        clin_metrics['target'] = target_name
        clin_metrics['label'] = label
        clin_metrics['model'] = 'Demographics + Biomarkers only'
        clin_metrics['n_features'] = n_features_clinical
    else:
        clin_metrics = None

    # ── Model C: Full model (WITH cognitive tests, for comparison) ──
    X_full = df[[c for c in all_feat if c in df.columns]].copy()
    const_f = [c for c in X_full.columns if X_full[c].nunique() <= 1]
    X_full = X_full.drop(columns=const_f)
    print(f"  [Full] Features: {X_full.shape[1]}")
    full_metrics, full_feats = run_s01_s05(X_full, y, tag="FULL")

    full_metrics['target'] = target_name
    full_metrics['label'] = label
    full_metrics['model'] = 'Full (incl. cognitive tests)'
    full_metrics['n_features'] = X_full.shape[1]

    comparison_rows.append(clean_metrics)
    if clin_metrics: comparison_rows.append(clin_metrics)
    comparison_rows.append(full_metrics)

    # ── Save feature lists ──
    for feats, fname in [(clean_feats, f'features_clean_{target_name}.csv'),
                           (full_feats, f'features_full_{target_name}.csv')]:
        pd.DataFrame({'feature': feats}).to_csv(
            os.path.join(RESULTS_DIR, fname), index=False)


# ══════════════════════════════════════════════════════════════════
# COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print("COMPREHENSIVE COMPARISON: ADNI vs UKB")
print(f"{'='*90}")

comp_df = pd.DataFrame(comparison_rows)
comp_df.to_csv(os.path.join(RESULTS_DIR, 'clean_comparison_all.csv'), index=False)

# UKB reference values (from local_data/Results_v2/)
ukb_ref = {
    'AD_s01-s05': 0.836,
    'DM_s01-s05': 0.831,
}

print(f"\n{'Model':<45} {'Target':<25} {'AUC':>8} {'Brier':>8} {'Sens':>6} {'Spec':>6}")
print(f"{'─'*45} {'─'*25} {'─'*8} {'─'*8} {'─'*6} {'─'*6}")

for _, row in comp_df.iterrows():
    print(f"  {row['model']:<43} {row['label']:<25} "
          f"{row['auc']:>8.4f} {row['brier']:>8.4f} "
          f"{row['sensitivity']:>6.3f} {row['specificity']:>6.3f}")

# ══════════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════════
print(f"\n[Plot] Generating comparison figures...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Plot 1: AUC comparison by model type for AD vs CN
adcn_rows = comp_df[comp_df['target'] == 'AD_vs_CN']
if len(adcn_rows) > 0:
    ax = axes[0]
    models = [r['model'] for _, r in adcn_rows.iterrows()]
    aucs = [r['auc'] for _, r in adcn_rows.iterrows()]
    colors = ['#3B82F6', '#F59E0B', '#EF4444', '#10B981'][:len(models)]

    bars = ax.barh(models, aucs, color=colors, edgecolor='white')
    ax.axvline(x=ukb_ref['AD_s01-s05'], color='gray', linestyle='--',
               linewidth=2, label=f"UKB AD_full AUC = {ukb_ref['AD_s01-s05']:.3f}")
    ax.axvline(x=0.5, color='black', linestyle=':', alpha=0.3)

    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{auc:.3f}', va='center', fontweight='bold')

    ax.set_xlim(0, 1.05)
    ax.set_xlabel('AUC', fontsize=12)
    ax.set_title('ADNI: AD vs CN — Model Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(axis='x', alpha=0.3)

# Plot 2: AUC across targets — clean model only
clean_rows = comp_df[comp_df['model'].str.contains('Biomarkers', na=False)]
if len(clean_rows) > 0:
    ax = axes[1]
    targets = [r['label'] for _, r in clean_rows.iterrows()]
    aucs = [r['auc'] for _, r in clean_rows.iterrows()]

    bars = ax.bar(range(len(targets)), aucs, color=['#3B82F6', '#8B5CF6', '#EC4899', '#F59E0B'],
                  edgecolor='white')
    ax.axhline(y=ukb_ref['AD_s01-s05'], color='gray', linestyle='--',
               linewidth=2, label=f"UKB AD_full = {ukb_ref['AD_s01-s05']:.3f}")
    ax.axhline(y=ukb_ref['DM_s01-s05'], color='gray', linestyle=':',
               linewidth=2, label=f"UKB DM_full = {ukb_ref['DM_s01-s05']:.3f}")

    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{auc:.3f}', ha='center', fontweight='bold')

    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels([t.replace('(', '\n(') for t in targets], fontsize=9)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title('ADNI: Clean Predictive Model (Biomarkers+Imaging only)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'adni_model_comparison.png'), dpi=150)
plt.close()

# ── Feature importance heatmap ──
# Show top features across models
fig2, ax2 = plt.subplots(figsize=(12, 6))
feature_counts = Counter()
for _, row in comp_df.iterrows():
    if 'selected_features' in row:
        pass  # Not stored; skip for now

ax2.text(0.5, 0.5, 'ADNI External Validation Complete\nSee CSV for details',
         transform=ax2.transAxes, ha='center', va='center', fontsize=18,
         fontweight='bold', color='#3B82F6')
ax2.set_title('ADNI Pipeline Validation Summary', fontsize=16)
fig2.tight_layout()
fig2.savefig(os.path.join(RESULTS_DIR, 'adni_summary_cover.png'), dpi=150)
plt.close()

print(f"\n✅ Clean comparison complete!")
print(f"   Results saved to: {RESULTS_DIR}")
print(f"   Files:")
for f in sorted(os.listdir(RESULTS_DIR)):
    print(f"     - {f}")

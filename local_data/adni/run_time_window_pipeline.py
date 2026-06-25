#!/usr/bin/env python3
"""
ADNI Time-Window Target Training
=================================
Runs s01-s05 pipeline on 5yr/10yr time-window targets,
using clean features (no cognitive test leakage).

Targets:
  AD_5yrs / AD_10yrs / AD_status_incident (all-time incident AD)
  Dementia_5yrs / Dementia_10yrs / Dementia_status_incident

Comparison with UKB:
  UKB AD_full=0.836, AD_10yrs=0.832, AD_5yrs=0.667
  UKB DM_full=0.831, DM_10yrs=0.833, DM_5yrs=0.816
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
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'time_window_targets')
os.makedirs(RESULTS_DIR, exist_ok=True)

N_SPLITS = 5; RANDOM_STATE = 2022; TOP_N_S01 = 50
CLUSTER_THRESHOLD = 0.75; TOP_N_SFS = 15; TOP_N_FINAL = 10
SFS_EARLY_STOP = 0.0005

LGB_PARAMS = dict(n_estimators=500, max_depth=15, num_leaves=10,
                   subsample=0.7, learning_rate=0.01, colsample_bytree=0.7,
                   objective='binary', is_unbalance=True, metric='auc',
                   verbosity=-1, seed=2020, n_jobs=4)

# ── Feature leakage exclusion ──
COGNITIVE_LEAKAGE_PREFIXES = [
    'MMSE_', 'MOCA_', 'ADAS_', 'FAQ_', 'GDS_', 'NPIQ_',
    'NB_', 'HACH_',
]
CDR_LEAKAGE_COLS = ['CDMEMORY', 'CDORIENT', 'CDJUDGE', 'CDCOMMUN',
                     'CDHOME', 'CDCARE', 'CDGLOBAL', 'CDRSB']
ID_COLS = ['PTID', 'RID', 'PHASE', 'APOE_genotype', 'subject_id',
           'entry_research_group', 'PTDOBYY']
TARGET_COLS = ['AD_status', 'MCI_status', 'dementia_status', 'DIAGNOSIS',
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
    """Run s01-s05 pipeline. Returns (metrics, selected_features)."""
    kf = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=True)

    # s01: Full feature ranking
    tg_cv = Counter()
    for tr, te in kf.split(X, y):
        gbm = LGBMClassifier(**LGB_PARAMS)
        gbm.fit(X.iloc[tr], y[tr])
        tg = dict(zip(gbm.booster_.feature_name(),
                      gbm.booster_.feature_importance(importance_type='gain')))
        tg_cv += Counter(normal_imp(tg))
    s01_r = sorted(tg_cv.items(), key=lambda x: -x[1])
    top50 = [f for f, _ in s01_r[:min(TOP_N_S01, len(s01_r))]]

    # s02: Hierarchical clustering
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

    # s03: Re-ranking
    tg_cv3 = Counter()
    for tr, te in kf.split(X_k, y):
        gbm = LGBMClassifier(**LGB_PARAMS)
        gbm.fit(X_k.iloc[tr], y[tr])
        tg = dict(zip(gbm.booster_.feature_name(),
                      gbm.booster_.feature_importance(importance_type='gain')))
        tg_cv3 += Counter(normal_imp(tg))
    s03_r = sorted(tg_cv3.items(), key=lambda x: -x[1])
    s03_f = [f for f, _ in s03_r]

    # s04: SFS
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

    # s05: Final model
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
        print(f"  [{tag}] AUC={metrics['auc']:.4f} ± {metrics['auc_cv_std']:.4f}, "
              f"Brier={metrics['brier']:.4f}, Sens={metrics['sensitivity']:.3f}, "
              f"Spec={metrics['specificity']:.3f}")
        n_img = sum(1 for f in top_f if f.startswith(imaging_pfx))
        print(f"  [{tag}] Top features ({n_img} imaging): {top_f[:5]}...")

    return metrics, top_f


# ══════════════════════════════════════════════════════════════════
# LOAD & PREPARE
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("ADNI TIME-WINDOW TARGET TRAINING (s01-s05)")
print("=" * 70)

print(f"\nLoading: {DATA_PATH}")
full_df = pd.read_csv(DATA_PATH)
print(f"  Shape: {full_df.shape}")

# Build clean feature set
all_exclude = set(ID_COLS + TARGET_COLS + CDR_LEAKAGE_COLS)
all_feat = [c for c in full_df.columns if c not in all_exclude]
clean_feat = [c for c in all_feat if not any(c.startswith(p) for p in COGNITIVE_LEAKAGE_PREFIXES)]
print(f"  Clean features: {len(clean_feat)} "
      f"(clinical: {sum(1 for c in clean_feat if not c.startswith(imaging_pfx))}, "
      f"imaging: {sum(1 for c in clean_feat if c.startswith(imaging_pfx))})")

# ══════════════════════════════════════════════════════════════════
# TARGET DEFINITIONS (time-window)
# ══════════════════════════════════════════════════════════════════
TIME_TARGETS = [
    ('AD_5yrs',  'AD within 5yr',   'ad'),
    ('AD_10yrs', 'AD within 10yr',  'ad'),
    ('AD_status_incident', 'AD all-time (incident)', 'ad'),
    ('Dementia_5yrs',  'Dementia within 5yr',  'dementia'),
    ('Dementia_10yrs', 'Dementia within 10yr', 'dementia'),
    ('Dementia_status_incident', 'Dementia all-time (incident)', 'dementia'),
]

# UKB reference
UKB_REF = {
    'AD_full': 0.836, 'AD_10yrs': 0.832, 'AD_5yrs': 0.667,
    'DM_full': 0.831, 'DM_10yrs': 0.833, 'DM_5yrs': 0.816,
}

# ══════════════════════════════════════════════════════════════════
# RUN ALL TARGETS
# ══════════════════════════════════════════════════════════════════
all_results = []
all_features = {}

for target_col, label, domain in TIME_TARGETS:
    print(f"\n{'='*70}")
    print(f"TARGET: {label} ({target_col})")
    print(f"{'='*70}")

    # Exclude subjects already diagnosed (prevalent), keep at-risk only
    df = full_df.copy()
    # For AD targets: exclude baseline AD
    if domain == 'ad':
        df = df[df['baseline_diagnosis'] != 3.0]
    # For dementia targets: exclude baseline AD and MCI (keep only CN)
    if domain == 'dementia':
        df = df[df['baseline_diagnosis'] == 1.0]

    # ── Censoring: exclude non-converters with insufficient follow-up ──
    n_before = len(df)
    if target_col == 'AD_5yrs':
        n_cens = df['censored_5yr'].sum()
        df = df[df['censored_5yr'] == 0]
    elif target_col == 'AD_10yrs':
        n_cens = df['censored_10yr'].sum()
        df = df[df['censored_10yr'] == 0]
    elif target_col == 'Dementia_5yrs':
        n_cens = df['censored_dementia_5yr'].sum()
        df = df[df['censored_dementia_5yr'] == 0]
    elif target_col == 'Dementia_10yrs':
        n_cens = df['censored_dementia_10yr'].sum()
        df = df[df['censored_dementia_10yr'] == 0]
    else:
        n_cens = 0  # all-time targets: keep all
    if n_cens > 0:
        print(f"  Censored (insufficient f/u): {n_cens} → retained {len(df)}")

    y = df[target_col].values.astype(int)
    n_total = len(y); n_events = y.sum()

    if n_events < 20 or n_total < 50:
        print(f"  [SKIP] n={n_total}, events={n_events} — insufficient")
        continue

    print(f"  n={n_total:,}, events={n_events:,} ({n_events/n_total*100:.1f}%)")

    # Prepare features
    X = df[[c for c in clean_feat if c in df.columns]].copy()
    const_c = [c for c in X.columns if X[c].nunique() <= 1]
    X = X.drop(columns=const_c)
    print(f"  Features: {X.shape[1]} "
          f"({sum(1 for c in X.columns if c.startswith(imaging_pfx))} imaging)")

    # Run pipeline
    metrics, features = run_s01_s05(X, y, tag=target_col)
    metrics['target'] = target_col
    metrics['label'] = label
    all_results.append(metrics)
    all_features[target_col] = features

    # Save features
    pd.DataFrame({'feature': features}).to_csv(
        os.path.join(RESULTS_DIR, f'features_{target_col}.csv'), index=False)

    # UKB comparison
    ukb_key = None
    if target_col == 'AD_5yrs': ukb_key = 'AD_5yrs'
    elif target_col == 'AD_10yrs': ukb_key = 'AD_10yrs'
    elif target_col == 'AD_status_incident': ukb_key = 'AD_full'
    elif target_col == 'Dementia_5yrs': ukb_key = 'DM_5yrs'
    elif target_col == 'Dementia_10yrs': ukb_key = 'DM_10yrs'
    elif target_col == 'Dementia_status_incident': ukb_key = 'DM_full'

    if ukb_key and ukb_key in UKB_REF:
        delta = metrics['auc'] - UKB_REF[ukb_key]
        print(f"  UKB {ukb_key} ref: {UKB_REF[ukb_key]:.3f}, Δ = {delta:+.4f}")

    gc.collect()

# ══════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print("ADNI TIME-WINDOW RESULTS — With UKB Comparison")
print(f"{'='*90}")

res_df = pd.DataFrame(all_results)
res_df.to_csv(os.path.join(RESULTS_DIR, 'time_window_results.csv'), index=False)

print(f"\n{'Target':<32} {'AUC':>8} {'Brier':>8} {'Sens':>6} {'Spec':>6} "
      f"{'UKB':>8} {'Δ':>8}")
print(f"{'─'*32} {'─'*8} {'─'*8} {'─'*6} {'─'*6} {'─'*8} {'─'*8}")

for _, row in res_df.iterrows():
    tcol = row['target']
    ukb_map = {
        'AD_5yrs': ('AD_5yrs', UKB_REF['AD_5yrs']),
        'AD_10yrs': ('AD_10yrs', UKB_REF['AD_10yrs']),
        'AD_status_incident': ('AD_full', UKB_REF['AD_full']),
        'Dementia_5yrs': ('DM_5yrs', UKB_REF['DM_5yrs']),
        'Dementia_10yrs': ('DM_10yrs', UKB_REF['DM_10yrs']),
        'Dementia_status_incident': ('DM_full', UKB_REF['DM_full']),
    }
    ukb_key, ukb_val = ukb_map.get(tcol, ('—', None))
    delta = f"{row['auc'] - ukb_val:+.4f}" if ukb_val is not None else '—'
    ukb_str = f"{ukb_val:.3f}" if ukb_val is not None else '—'
    print(f"  {row['label']:<30} {row['auc']:>8.4f} {row['brier']:>8.4f} "
          f"{row['sensitivity']:>6.3f} {row['specificity']:>6.3f} "
          f"{ukb_str:>8} {delta:>8}")

# ══════════════════════════════════════════════════════════════════
# COMPARISON BAR CHART
# ══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 7))

labels = [r['label'] for _, r in res_df.iterrows()]
adni_aucs = [r['auc'] for _, r in res_df.iterrows()]

# UKB bars
ukb_keys = ['DM_full', 'DM_10yrs', 'DM_5yrs', 'AD_full', 'AD_10yrs', 'AD_5yrs']
ukb_labels = ['Dementia\nall-time', 'Dementia\n10yr', 'Dementia\n5yr',
              'AD\nall-time', 'AD\n10yr', 'AD\n5yr']

x = np.arange(len(labels))
w = 0.35

bars1 = ax.bar(x - w/2, adni_aucs, w, label='ADNI (Biomarkers+Imaging)', color='#3B82F6',
               edgecolor='white')

# UKB reference lines
for i, (tcol, _) in enumerate(TIME_TARGETS):
    ukb_map2 = {
        'AD_5yrs': UKB_REF['AD_5yrs'], 'AD_10yrs': UKB_REF['AD_10yrs'],
        'AD_status_incident': UKB_REF['AD_full'],
        'Dementia_5yrs': UKB_REF['DM_5yrs'], 'Dementia_10yrs': UKB_REF['DM_10yrs'],
        'Dementia_status_incident': UKB_REF['DM_full'],
    }
    ukb_v = ukb_map2.get(tcol)
    if ukb_v:
        ax.scatter(i + w/2, ukb_v, marker='s', s=80, color='#EF4444', zorder=5,
                   edgecolors='white', linewidth=1)

# Dummy scatter for legend
ax.scatter([], [], marker='s', s=80, color='#EF4444', label='UKB reference',
           edgecolors='white', linewidth=1)

for bar, auc in zip(bars1, adni_aucs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
            f'{auc:.3f}', ha='center', fontsize=8.5, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels([l.replace(' ', '\n') for l in labels], fontsize=8.5)
ax.set_ylabel('AUC', fontsize=12)
ax.set_ylim(0, 1.1)
ax.set_title('ADNI Time-Window Targets vs UKB Reference\n'
             '(Clean predictive model: biomarkers + imaging, no cognitive tests)',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'time_window_auc_comparison.png'), dpi=150)
plt.close()

print(f"\nDone! Results saved to: {RESULTS_DIR}")

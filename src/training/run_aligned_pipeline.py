#!/usr/bin/env python3
"""
Re-run pipeline with feature set aligned to the paper's 106 features.
We match 95 of 106 paper features to our data (missing: field 404 and genetic features).
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              confusion_matrix, brier_score_loss)
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from lightgbm import LGBMClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

DPATH = 'local_data/Preprocessed_Data'
PREPROCESSED_CSV = os.path.join(DPATH, 'Preprocessed_Data_full.csv')
RESULTS_DIR = 'local_data/Results_v2/_aligned_features_v2'
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_STATE = 2022; N_SPLITS = 5; TOP_N = 10

# Paper's best params
BEST_PARAMS = dict(n_estimators=500, max_depth=20, num_leaves=10,
                   subsample=0.5, learning_rate=0.02, objective='binary',
                   is_unbalance=True, metric='auc', verbosity=-1, seed=2022, n_jobs=4)

TARGETS = {
    'DM_full': ('dementia_status', 'dementia_years', None),
    'DM_10yrs': ('dementia_status', 'dementia_years', 10),
    'DM_5yrs': ('dementia_status', 'dementia_years', 5),
    'AD_full': ('AD_status', 'AD_years', None),
    'AD_10yrs': ('AD_status', 'AD_years', 10),
    'AD_5yrs': ('AD_status', 'AD_years', 5),
}

TARGET_LABELS = {
    'DM_full': 'All-cause Dementia (full)', 'DM_10yrs': 'All-cause Dementia (10yr)',
    'DM_5yrs': 'All-cause Dementia (5yr)', 'AD_full': "Alzheimer's Disease (full)",
    'AD_10yrs': "Alzheimer's Disease (10yr)", 'AD_5yrs': "Alzheimer's Disease (5yr)",
}

print("=" * 70)
print("PIPELINE: Paper-Aligned Features")
print("=" * 70)

# --- Load data ---
print("\n[1] Loading data...")
mydf = pd.read_csv(PREPROCESSED_CSV)
print(f"  Loaded: {mydf.shape}")

# Feature columns (exclude outcome columns)
outcome_cols = ['dementia_status', 'dementia_years', 'AD_status', 'AD_years',
                'VD_status', 'VD_years', 'stroke_status', 'stroke_years']
feat_cols = [c for c in mydf.columns if c not in outcome_cols]
print(f"  Feature columns: {len(feat_cols)} for {len(mydf):,} participants")
X_all = mydf[feat_cols]

# --- Feature Selection (s01-s04) on DM_full ---
print("\n[2] Feature Selection on DM_full...")
target_col, years_col, max_years = TARGETS['DM_full']
y = mydf[target_col].copy()
if max_years is not None:
    y.loc[(y == 1) & (mydf[years_col] > max_years)] = 0
print(f"  n={len(y):,}, events={y.sum():,} ({y.mean()*100:.2f}%)")

# RM_HES
RM_HES = ['41234-0.0', '41259-0.0', '41149-0.0', '41289-0.0',
          '41218-0.0', '41235-0.0', '41214-0.0']
X = X_all.drop(columns=[c for c in RM_HES if c in X_all.columns], errors='ignore')
print(f"  X shape after HES removal: {X.shape}")

kf = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=True)

# s01: Initial ranking
print("\n  s01: Full feature ranking...")
t0 = time.time()
gains = []
for tr, te in kf.split(X, y):
    gbm = LGBMClassifier(**BEST_PARAMS)
    gbm.fit(X.iloc[tr], y.iloc[tr])
    gains.append(gbm.feature_importances_ / gbm.feature_importances_.sum())
mean_gain = np.mean(gains, axis=0)
s01_ranked = sorted(zip(X.columns, mean_gain), key=lambda x: -x[1])
top_50 = [f for f, _ in s01_ranked[:min(50, len(s01_ranked))]]
s01_df = pd.DataFrame(s01_ranked, columns=['Feature', 'Gain'])
s01_df.to_csv(os.path.join(RESULTS_DIR, 's01_feature_importance.csv'), index=False)
print(f"  s01 done ({time.time()-t0:.0f}s), Top 50 saved")

# s02: Hierarchical clustering
print("  s02: Hierarchical clustering...")
t0 = time.time()
X_top50 = X[top_50]
corr = X_top50.corr(method='spearman')
dist = (1 - np.abs(corr.fillna(0))).values.copy()  # Handle NaN correlations
dist[np.isnan(dist)] = 0
dist_vec = squareform(dist)
dist_vec = np.nan_to_num(dist_vec, nan=0, posinf=1, neginf=1)
link = hierarchy.linkage(dist_vec, method='ward')
clusters = hierarchy.fcluster(link, t=0.75, criterion='distance')
kept = []; seen = set()
for f, c in zip(top_50, clusters):
    if c not in seen:
        kept.append(f); seen.add(c)
s02_df = pd.DataFrame({'feature': kept, 'cluster': [c for f, c in zip(top_50, clusters) if f in kept]})
s02_df.to_csv(os.path.join(RESULTS_DIR, 's02_clustered_features.csv'), index=False)
# Dendrogram
fig, ax = plt.subplots(figsize=(16, 6))
hierarchy.dendrogram(link, labels=top_50, leaf_font_size=8, ax=ax, color_threshold=0.75)
ax.set_title('Ward Hierarchical Clustering (threshold=0.75)')
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 's02_dendrogram.png'), dpi=150)
plt.close()
print(f"  s02 done ({time.time()-t0:.0f}s), {len(kept)} features after clustering")

# s03: Re-rank
print("  s03: Re-ranking...")
t0 = time.time()
X_kept = X[kept]
gains_s03 = []
for tr, te in kf.split(X_kept, y):
    gbm = LGBMClassifier(**BEST_PARAMS)
    gbm.fit(X_kept.iloc[tr], y.iloc[tr])
    gains_s03.append(gbm.feature_importances_ / gbm.feature_importances_.sum())
mean_gain_s03 = np.mean(gains_s03, axis=0)
s03_ranked = sorted(zip(kept, mean_gain_s03), key=lambda x: -x[1])
s03_features = [f for f, _ in s03_ranked]
pd.DataFrame(s03_ranked, columns=['Feature', 'Gain']).to_csv(
    os.path.join(RESULTS_DIR, 's03_final_importance.csv'), index=False)
print(f"  s03 done ({time.time()-t0:.0f}s)")

# s04: SFS
print("  s04: SFS...")
t0 = time.time()
selected = []; remaining = list(s03_features)
sfs_history = []
for step in range(TOP_N):
    best_f, best_auc = None, -1
    for cand in remaining[:min(10, len(remaining))]:
        trial = selected + [cand]
        aucs = []
        for tr, te in kf.split(X_kept[trial], y):
            gbm = LGBMClassifier(**BEST_PARAMS)
            gbm.fit(X_kept[trial].iloc[tr], y.iloc[tr])
            aucs.append(roc_auc_score(y.iloc[te], gbm.predict_proba(X_kept[trial].iloc[te])[:, 1]))
        avg_auc = np.mean(aucs)
        if avg_auc > best_auc:
            best_f, best_auc = cand, avg_auc
    if best_f is None: break
    selected.append(best_f); remaining.remove(best_f)
    sfs_history.append({'step': step+1, 'feature': best_f, 'auc': round(best_auc, 4)})
    print(f"    Step {step+1}: {best_f} (AUC={best_auc:.4f})")
pd.DataFrame(sfs_history).to_csv(os.path.join(RESULTS_DIR, 's04_sfs_history.csv'), index=False)
pd.DataFrame({'feature': selected}).to_csv(os.path.join(RESULTS_DIR, 's04_selected_features.csv'), index=False)
print(f"  s04 done ({time.time()-t0:.0f}s)")
print(f"  Selected: {selected}")

# --- s05: Train & Evaluate for all targets ---
print(f"\n[3] Training & Evaluation (Deploy strategy)...")

def train_evaluate_one_target(target_name, features, mydf, X_kept):
    """Train following paper's Deploy approach: split-train-calibrate."""
    target_col, years_col, max_years = TARGETS[target_name]
    y = mydf[target_col].copy()
    if max_years is not None:
        y.loc[(y == 1) & (mydf[years_col] > max_years)] = 0

    X = X_kept[[f for f in features if f in X_kept.columns]]
    cutoff_list = [np.round(0.001 + i * 0.001, 3) for i in range(100)]

    outer_kf = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=True)
    preds_all, trues_all = [], []
    fold_metrics = []

    for fold, (otr, ote) in enumerate(outer_kf.split(X, y)):
        X_tr, y_tr = X.iloc[otr], y.iloc[otr]
        X_te, y_te = X.iloc[ote], y.iloc[ote]

        n_calib = int(len(X_tr) * 0.4)
        gbm = LGBMClassifier(**BEST_PARAMS)
        gbm.fit(X_tr.iloc[n_calib:], y_tr.iloc[n_calib:])
        raw_cali = gbm.predict_proba(X_tr.iloc[:n_calib])[:, 1]
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(raw_cali, y_tr.iloc[:n_calib])
        y_pred = np.clip(iso.predict(gbm.predict_proba(X_te)[:, 1]), 0, 1)

        preds_all.append(y_pred); trues_all.append(np.array(y_te))
        fold_metrics.append({'fold': fold, 'auc': roc_auc_score(y_te, y_pred)})

    yt_all = np.concatenate(trues_all); yp_all = np.concatenate(preds_all)
    auc_mean = np.mean([m['auc'] for m in fold_metrics])
    auc_std = np.std([m['auc'] for m in fold_metrics])
    brier = brier_score_loss(yt_all, yp_all)
    ap = average_precision_score(yt_all, yp_all)

    # Youden best cutoff
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(yt_all, yp_all)
    best_idx = np.argmax(tpr - fpr)
    best_cut = thresholds[best_idx] if best_idx < len(thresholds) else 0.01
    y_bin = (yp_all >= best_cut).astype(int)
    tn, fp, fn, tp = confusion_matrix(yt_all, y_bin).ravel()
    sens = tp/(tp+fn) if (tp+fn)>0 else 0
    spec = tn/(tn+fp) if (tn+fp)>0 else 0

    print(f"  {target_name}: AUC={auc_mean:.4f}±{auc_std:.4f}, Brier={brier:.4f}, "
          f"Sens={sens:.3f}, Spec={spec:.3f}")

    return {'target': target_name, 'auc_mean': auc_mean, 'auc_std': auc_std,
            'brier': brier, 'ap': ap, 'sensitivity': sens, 'specificity': spec,
            'n_events': int(yt_all.sum()), 'n_total': len(yt_all)}

all_metrics = []
for tgt in TARGETS:
    m = train_evaluate_one_target(tgt, selected, mydf, X_kept)
    all_metrics.append(m)

metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv(os.path.join(RESULTS_DIR, 'aligned_results.csv'), index=False)

# Compare with previous CV results
print(f"\n{'='*70}")
print("COMPARISON: Aligned Features vs Original 1076 Features")
print(f"{'='*70}")
print(f"\n{'Target':<15} {'Aligned AUC':>12} {'Original AUC':>12} {'Δ':>8}")
print(f"{'─'*15} {'─'*12} {'─'*12} {'─'*8}")

orig_auc = {'DM_full': 0.831, 'DM_10yrs': 0.833, 'DM_5yrs': 0.816,
            'AD_full': 0.836, 'AD_10yrs': 0.832, 'AD_5yrs': 0.667}
for _, row in metrics_df.iterrows():
    t = row['target']
    orig = orig_auc[t]
    al_auc = row['auc_mean']
    print(f"  {t:<15} {al_auc:>12.4f} {orig:>12.4f} {al_auc-orig:>+8.4f}")

print(f"\n  Feature set size: {len(feat_cols)} (paper-aligned) vs 1076 (original)")
print(f"  Selected features: {selected}")
print(f"\n  Results saved to: {RESULTS_DIR}")

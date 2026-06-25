#!/usr/bin/env python3
"""
External validation: hold-out 80/20 split.
Train full pipeline (s01-s05) on training set, evaluate on held-out test set.
Compare with CAIDE & ANU-ADRI risk scores on the SAME test set.
"""
import os, json, time, warnings, sys
import numpy as np
import pandas as pd

# Support both direct invocation and import
PROJECT_ROOT = os.environ.get('PROJECT_ROOT',
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DPATH = os.path.join(PROJECT_ROOT, 'local_data', 'Preprocessed_Data')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'Results_v2', '_external_validation')
os.makedirs(RESULTS_DIR, exist_ok=True)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, recall_score, precision_score,
    f1_score, brier_score_loss, confusion_matrix, roc_curve,
    average_precision_score
)
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from scipy import stats
from lightgbm import LGBMClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

N_SPLITS = 5
RANDOM_STATE = 2022
TOP_N = 10
TEST_SIZE = 0.2  # 80/20 split

# Best params from hyperparameter search
BEST_PARAMS = dict(n_estimators=500, max_depth=20, num_leaves=10,
                   subsample=0.5, learning_rate=0.02, objective='binary',
                   is_unbalance=True, metric='auc', verbosity=-1,
                   seed=RANDOM_STATE, n_jobs=4)

TARGETS = {
    'DM_full': ('dementia_status', 'dementia_years', None),
    'DM_10yrs': ('dementia_status', 'dementia_years', 10),
    'DM_5yrs': ('dementia_status', 'dementia_years', 5),
    'AD_full': ('AD_status', 'AD_years', None),
    'AD_10yrs': ('AD_status', 'AD_years', 10),
    'AD_5yrs': ('AD_status', 'AD_years', 5),
}

TARGET_LABELS = {
    'DM_full': 'All-cause Dementia (full)',
    'DM_10yrs': 'All-cause Dementia (10yr)',
    'DM_5yrs': 'All-cause Dementia (5yr)',
    'AD_full': "Alzheimer's Disease (full)",
    'AD_10yrs': "Alzheimer's Disease (10yr)",
    'AD_5yrs': "Alzheimer's Disease (5yr)",
}

# ═══════════════════════════════════════════════════════════════
# 1. LOAD & SPLIT
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("EXTERNAL VALIDATION: 80/20 Hold-Out")
print("=" * 70)

print("\nLoading data...")
X_all = pd.read_csv(os.path.join(DPATH, 'Preprocessed_Data.csv'))
target_df = pd.read_csv(os.path.join(DPATH, 'Dementia_target.csv'))
for col in target_df.columns:
    X_all[col] = target_df[col].values
print(f"  Full dataset: {X_all.shape}")

# Use DM_full for stratification
y_strat = target_df['dementia_status'].values
train_idx, test_idx = train_test_split(
    np.arange(len(X_all)), test_size=TEST_SIZE,
    stratify=y_strat, random_state=RANDOM_STATE
)
X_train_all = X_all.iloc[train_idx].reset_index(drop=True)
X_test_all = X_all.iloc[test_idx].reset_index(drop=True)
print(f"  Train: {len(X_train_all):,}, Test: {len(X_test_all):,}")

# ═══════════════════════════════════════════════════════════════
# 2. COMPUTE CAIDE & ANU-ADRI ON TEST SET
# ═══════════════════════════════════════════════════════════════
def compute_caide(data):
    """CAIDE without APOE on given DataFrame."""
    n = len(data)
    age = data['21022-0.0'].values
    a = np.zeros(n); a[(age >= 47) & (age <= 53)] = 3; a[age > 53] = 4
    e  = np.zeros(n); ey = data['845-0.0'].fillna(data['845-0.0'].median()).values - 5
    e[ey <= 6] = 3; e[(ey > 6) & (ey < 10)] = 2
    sx = data['31-0.0_c1'].values.astype(float)
    s = np.zeros(n); s[data['4080-0.0'].fillna(data['4080-0.0'].median()).values >= 140] = 2
    bmi = data['21001-0.0'].copy()
    bmi.loc[bmi.isnull()] = data['23104-0.0'].loc[bmi.isnull()]
    bmi.loc[bmi.isnull()] = bmi.median()
    b = np.zeros(n); b[bmi.values > 30] = 2
    c = np.zeros(n); c[data['30690-0.0'].fillna(data['30690-0.0'].median()).values > 6.5] = 2
    act = np.ones(n); act[data['22032-0.0_c1'].values > 0.5] = 0
    return a + e + sx + s + b + c + act

def compute_anu_adri(data):
    """Simplified ANU-ADRI on given DataFrame."""
    n = len(data); age = data['21022-0.0'].values; is_f = data['31-0.0_c0'].values.astype(float)
    am = np.zeros(n); af = np.zeros(n)
    for (lo, hi), vm, vf in [((65, 70), 1, 5), ((70, 75), 12, 14), ((75, 80), 18, 21),
                              ((80, 85), 26, 29), ((85, 90), 33, 35)]:
        am[(age >= lo) & (age < hi)] = vm; af[(age >= lo) & (age < hi)] = vf
    am[age >= 90] = 38; af[age >= 90] = 41
    a_pts = am * (1 - is_f) + af * is_f
    ey = data['845-0.0'].fillna(data['845-0.0'].median()).values - 5
    e_pts = np.full(n, 6); e_pts[(ey > 8) & (ey <= 11)] = 3; e_pts[ey > 11] = 0
    bmi = data['21001-0.0'].copy()
    bmi.loc[bmi.isnull()] = data['23104-0.0'].loc[bmi.isnull()]
    bmi.loc[bmi.isnull()] = bmi.median(); bm = bmi.values
    b_pts = np.zeros(n); b_pts[(age >= 60) & (bm >= 25) & (bm < 30)] = 2
    b_pts[(age >= 60) & (bm >= 30)] = 5
    dc = [c for c in data.columns if c.startswith('2443-0.0_c')]
    d_pts = np.zeros(n)
    if dc: d_pts = data[dc].values.max(axis=1).clip(0, 1) * 3
    chol = data['30690-0.0'].fillna(data['30690-0.0'].median()).values
    c_pts = np.zeros(n); c_pts[(age >= 60) & (chol > 6.5)] = 3
    sk = np.zeros(n)
    if '20116-0.0_c1' in data.columns: sk[data['20116-0.0_c1'].values > 0.5] = 4
    if '20116-0.0_c2' in data.columns: sk[data['20116-0.0_c2'].values > 0.5] = 4
    sc = [c for c in data.columns if c.startswith('1031-0.0_c')]
    nc = np.zeros(n)
    for ci, c in enumerate(sc): nc += data[c].fillna(0).values * (ci + 1)
    so = np.zeros(n); so[(nc > 1) & (nc <= 3)] = 1; so[(nc > 3) & (nc <= 5)] = 4; so[nc > 5] = 6
    act = np.zeros(n)
    if '22032-0.0_c1' in data.columns: act[data['22032-0.0_c1'].values > 0.5] = -2
    cog = data['20023-0.0'].fillna(data['20023-0.0'].median()).values
    co = np.full(n, -6); co[(cog >= 500) & (cog < 585)] = -7; co[cog >= 585] = 0
    fv = np.zeros(n)
    for fid in ['1329', '1339']:
        fc = [c for c in data.columns if c.startswith(f'{fid}-0.0_c')]
        fs = np.zeros(n)
        for ci, c in enumerate(fc): fs += data[c].fillna(0).values * (ci + 1)
        fv = np.maximum(fv, fs)
    fi = np.zeros(n); fi[fv >= 4] = -5; fi[fv == 3] = -4; fi[fv == 2] = -3
    return a_pts + e_pts + b_pts + d_pts + c_pts + sk + so + act + co + fi

print("Computing risk scores on test set...")
caide_test = compute_caide(X_test_all)
anu_test = compute_anu_adri(X_test_all)

# ═══════════════════════════════════════════════════════════════
# 3. FEATURE SELECTION (s01-s04 SFS) ON TRAINING SET
# ═══════════════════════════════════════════════════════════════
# Use DM_full target for feature selection (paper's Deploy approach)
target_col, years_col, max_years = TARGETS['DM_full']
y_train_dm = X_train_all[target_col].copy()
if max_years is not None:
    y_train_dm.loc[(y_train_dm == 1) & (X_train_all[years_col] > max_years)] = 0

print(f"\nFeature selection on DM_full training set: n={len(y_train_dm):,}, "
      f"events={y_train_dm.sum():,}")

# Exclude outcome and identifier columns
exclude_prefixes = ('dementia_', 'AD_', 'VD_', 'stroke_', 'eid')
feat_cols = [c for c in X_train_all.columns
             if not any(c.startswith(p) for p in exclude_prefixes)]
X_feat = X_train_all[feat_cols]

# s01: Initial ranking
print("  s01: Full feature ranking...")
t0 = time.time()
kf_s01 = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=True)
gains = []
for tr, te in kf_s01.split(X_feat, y_train_dm):
    gbm = LGBMClassifier(**BEST_PARAMS)
    gbm.fit(X_feat.iloc[tr], y_train_dm.iloc[tr])
    gains.append(gbm.feature_importances_ / gbm.feature_importances_.sum())
mean_gain = np.mean(gains, axis=0)
s01_ranked = sorted(zip(feat_cols, mean_gain), key=lambda x: -x[1])
top_50 = [f for f, _ in s01_ranked[:50]]
print(f"  s01 done ({time.time()-t0:.0f}s), Top 50 features identified")

# s02: Hierarchical clustering (Ward, threshold 0.75)
print("  s02: Hierarchical clustering...")
t0 = time.time()
X_top50 = X_feat[top_50]
corr = X_top50.corr(method='spearman')
dist = 1 - np.abs(corr)
from scipy.cluster.hierarchy import linkage, fcluster
link = linkage(dist.values[np.triu_indices(len(top_50), k=1)], method='ward')
# Approximate: map back to feature distance matrix
# Use complete distance matrix for clustering
from scipy.spatial.distance import squareform
dist_vec = dist.values[np.triu_indices(len(top_50), k=1)]
link = linkage(dist_vec, method='ward')
clusters = fcluster(link, t=0.75, criterion='distance')
# Keep first feature per cluster
kept = []; seen = set()
for f, c in zip(top_50, clusters):
    if c not in seen:
        kept.append(f); seen.add(c)
print(f"  s02 done ({time.time()-t0:.0f}s), {len(kept)} features after clustering")

# s03: Re-rank after clustering
print("  s03: Re-ranking clustered features...")
t0 = time.time()
X_kept = X_feat[kept]
gains_s03 = []
for tr, te in kf_s01.split(X_kept, y_train_dm):
    gbm = LGBMClassifier(**BEST_PARAMS)
    gbm.fit(X_kept.iloc[tr], y_train_dm.iloc[tr])
    gains_s03.append(gbm.feature_importances_ / gbm.feature_importances_.sum())
mean_gain_s03 = np.mean(gains_s03, axis=0)
s03_ranked = sorted(zip(kept, mean_gain_s03), key=lambda x: -x[1])
s03_features = [f for f, _ in s03_ranked]
print(f"  s03 done ({time.time()-t0:.0f}s)")

# s04: SFS - Sequential Forward Selection
print("  s04: Sequential Forward Selection...")
t0 = time.time()
selected = []
remaining = list(s03_features)
sfs_history = []

for step in range(TOP_N):
    best_f, best_auc = None, -1
    for cand in remaining[:min(10, len(remaining))]:  # Search top-10 candidates per step
        trial = selected + [cand]
        aucs = []
        for tr, te in kf_s01.split(X_kept[trial], y_train_dm):
            gbm = LGBMClassifier(**BEST_PARAMS)
            gbm.fit(X_kept[trial].iloc[tr], y_train_dm.iloc[tr])
            pred = gbm.predict_proba(X_kept[trial].iloc[te])[:, 1]
            aucs.append(roc_auc_score(y_train_dm.iloc[te], pred))
        avg_auc = np.mean(aucs)
        if avg_auc > best_auc:
            best_f, best_auc = cand, avg_auc

    if best_f is None:
        break
    selected.append(best_f)
    remaining.remove(best_f)
    sfs_history.append({'step': step + 1, 'feature': best_f, 'auc': best_auc})
    print(f"    Step {step+1}: {best_f} (AUC={best_auc:.4f})")

print(f"  s04 done ({time.time()-t0:.0f}s)")
print(f"  Selected features: {selected}")

# ═══════════════════════════════════════════════════════════════
# 4. TRAIN FINAL MODEL (s05) & EVALUATE ON TEST SET
# ═══════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("s05: Train & External Validation")
print(f"{'='*70}")

results = []

for target_name, (target_col, years_col, max_years) in TARGETS.items():
    print(f"\n── {TARGET_LABELS[target_name]} ──")

    # Build target with proper censoring
    y_train = X_train_all[target_col].copy()
    include_tr = pd.Series(True, index=X_train_all.index)
    if max_years is not None:
        y_train.loc[(y_train == 1) & (X_train_all[years_col] > max_years)] = 0
        insufficient_tr = (
            (y_train == 0) & (X_train_all[years_col] < 0) &
            (X_train_all[years_col].abs() < max_years)
        )
        include_tr = ~insufficient_tr

    y_test = X_test_all[target_col].copy()
    include_te = pd.Series(True, index=X_test_all.index)
    if max_years is not None:
        y_test.loc[(y_test == 1) & (X_test_all[years_col] > max_years)] = 0
        insufficient_te = (
            (y_test == 0) & (X_test_all[years_col] < 0) &
            (X_test_all[years_col].abs() < max_years)
        )
        include_te = ~insufficient_te

    # Features available in test set
    test_features = [f for f in selected if f in X_test_all.columns]
    X_tr = X_train_all[test_features][include_tr]
    X_te = X_test_all[test_features][include_te]
    y_train = y_train[include_tr]
    y_test = y_test[include_te]

    print(f"  Train: n={len(y_train):,}, events={y_train.sum():,}")
    print(f"  Test:  n={len(y_test):,}, events={y_test.sum():,}")

    # Split-train-calibrate (paper's Deploy approach)
    n_calib = int(len(X_tr) * 0.4)
    gbm = LGBMClassifier(**BEST_PARAMS)
    gbm.fit(X_tr.iloc[n_calib:], y_train.iloc[n_calib:])

    raw_cali = gbm.predict_proba(X_tr.iloc[:n_calib])[:, 1]
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(raw_cali, y_train.iloc[:n_calib])

    # Predict on test set
    y_pred_test = iso.predict(gbm.predict_proba(X_te)[:, 1])
    y_pred_test = np.clip(y_pred_test, 0, 1)

    # Evaluate
    auc_val = roc_auc_score(y_test, y_pred_test)
    ap_val = average_precision_score(y_test, y_pred_test)
    brier_val = brier_score_loss(y_test, y_pred_test)

    # Best cutoff by Youden on test set
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_test)
    best_idx = np.argmax(tpr - fpr)
    best_cut = thresholds[best_idx] if best_idx < len(thresholds) else 0.05

    y_bin = (y_pred_test >= best_cut).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_bin).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0

    # Risk scores on test set
    for score_name, score_vals in [('CAIDE', caide_test), ('ANU_ADRI', anu_test)]:
        score_auc = roc_auc_score(y_test, score_vals)
        # Normalize for Brier
        s_min, s_max = score_vals.min(), score_vals.max()
        s_norm = (score_vals - s_min) / (s_max - s_min) if s_max > s_min else score_vals
        score_brier = brier_score_loss(y_test, s_norm)
        # Youden cutoff
        sfpr, stpr, sthresh = roc_curve(y_test, score_vals)
        s_best = sthresh[np.argmax(stpr - sfpr)] if len(sthresh) > 0 else 0
        s_bin = (score_vals >= s_best).astype(int)
        stn, sfp, sfn, stp = confusion_matrix(y_test, s_bin).ravel()
        s_acc = (stp + stn) / (stp + stn + sfp + sfn)
        s_sens = stp / (stp + sfn) if (stp + sfn) > 0 else 0
        s_spec = stn / (stn + sfp) if (stn + sfp) > 0 else 0

        results.append({
            'Target': TARGET_LABELS[target_name],
            'Model': score_name,
            'AUC': round(score_auc, 4),
            'Accuracy': round(s_acc, 4),
            'Sensitivity': round(s_sens, 4),
            'Specificity': round(s_spec, 4),
            'Brier': round(score_brier, 4),
        })

    # Our model result
    results.append({
        'Target': TARGET_LABELS[target_name],
        'Model': 'OUR MODEL',
        'AUC': round(auc_val, 4),
        'Accuracy': round(acc, 4),
        'Sensitivity': round(sens, 4),
        'Specificity': round(spec, 4),
        'Brier': round(brier_val, 4),
        'AP': round(ap_val, 4),
        'F1': round(f1, 4),
        'Precision': round(prec, 4),
        'Best_Cutoff': round(best_cut, 4),
    })

    print(f"  AUC={auc_val:.4f}, AP={ap_val:.4f}, Brier={brier_val:.4f}")
    print(f"  Acc={acc:.3f}, Sens={sens:.3f}, Spec={spec:.3f}, F1={f1:.3f}")

# ═══════════════════════════════════════════════════════════════
# 5. SUMMARY
# ═══════════════════════════════════════════════════════════════
results_df = pd.DataFrame(results)

# Separate tables
model_df = results_df[results_df['Model'] == 'OUR MODEL'].reset_index(drop=True)
caide_df = results_df[results_df['Model'] == 'CAIDE'].reset_index(drop=True)
anu_df = results_df[results_df['Model'] == 'ANU_ADRI'].reset_index(drop=True)

# Merge for comparison
compare_df = model_df[['Target', 'AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'Brier']].copy()
compare_df.columns = ['Target', 'Model_AUC', 'Model_Acc', 'Model_Sens', 'Model_Spec', 'Model_Brier']
compare_df['CAIDE_AUC'] = caide_df['AUC'].values
compare_df['CAIDE_Acc'] = caide_df['Accuracy'].values
compare_df['CAIDE_Sens'] = caide_df['Sensitivity'].values
compare_df['CAIDE_Spec'] = caide_df['Specificity'].values
compare_df['CAIDE_Brier'] = caide_df['Brier'].values
compare_df['ANU_AUC'] = anu_df['AUC'].values
compare_df['ANU_Brier'] = anu_df['Brier'].values

compare_df.to_csv(os.path.join(RESULTS_DIR, 'external_validation_summary.csv'), index=False)
model_df.to_csv(os.path.join(RESULTS_DIR, 'model_metrics.csv'), index=False)

# Print
print(f"\n{'='*80}")
print("EXTERNAL VALIDATION RESULTS")
print(f"{'='*80}")
print(f"\n{'Target':<30} {'Model AUC':>10} {'CAIDE AUC':>10} {'ANU AUC':>10} {'Δ vs CAIDE':>10}")
print(f"{'─'*30} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")
for _, row in compare_df.iterrows():
    delta = float(row['Model_AUC']) - float(row['CAIDE_AUC'])
    print(f"{row['Target']:<30} {float(row['Model_AUC']):>10.4f} "
          f"{float(row['CAIDE_AUC']):>10.4f} {float(row['ANU_AUC']):>10.4f} {delta:>+10.4f}")

print(f"\n{'Target':<30} {'Model Brier':>12} {'CAIDE Brier':>12} {'ANU Brier':>12}")
print(f"{'─'*30} {'─'*12} {'─'*12} {'─'*12}")
for _, row in compare_df.iterrows():
    print(f"{row['Target']:<30} {float(row['Model_Brier']):>12.6f} "
          f"{float(row['CAIDE_Brier']):>12.6f} {float(row['ANU_Brier']):>12.6f}")

# Bar chart
fig, ax = plt.subplots(figsize=(14, 6))
targets_short = [t.replace('All-cause Dementia', 'DM').replace("Alzheimer's Disease", 'AD')
                  for t in compare_df['Target']]
x = np.arange(len(targets_short)); w = 0.25
ax.bar(x - w, compare_df['CAIDE_AUC'].astype(float), w, label='CAIDE', color='#E8A87C')
ax.bar(x, compare_df['ANU_AUC'].astype(float), w, label='ANU-ADRI', color='#95E1D3')
ax.bar(x + w, compare_df['Model_AUC'].astype(float), w, label='OUR MODEL', color='#3B82F6')
ax.set_xticks(x); ax.set_xticklabels(targets_short, fontsize=10)
ax.set_ylabel('AUC', fontsize=12)
ax.set_title('External Validation (80/20 Hold-Out): AUC Comparison',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3); ax.set_ylim(0.4, 1.0)
for i in range(len(targets_short)):
    for j, col in enumerate(['CAIDE_AUC', 'ANU_AUC', 'Model_AUC']):
        v = float(compare_df[col].iloc[i])
        ax.text(x[i] + (j-1)*w, v + 0.01, f'{v:.3f}', ha='center', fontsize=7.5)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'external_validation_auc.png'), dpi=150)
plt.close()

# Selected features
pd.DataFrame({'feature': selected}).to_csv(
    os.path.join(RESULTS_DIR, 'selected_features.csv'), index=False)

print(f"\nResults saved to: {RESULTS_DIR}")
print("Done!")

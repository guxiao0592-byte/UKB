#!/usr/bin/env python3
"""
External Validation: 80/20 Hold-Out — Imaging vs Non-Imaging Models
====================================================================
1. 80/20 split, stratified by DM_full
2. Run s01-s04 SFS on training set (from clinical + imaging features)
3. Train + Isotonic calibrate on training set
4. Evaluate on held-out test set
5. Compare with CAIDE & ANU-ADRI risk scores
6. Compare imaging vs non-imaging models on same test set
"""
import os, sys, time, warnings, gc
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import chi2
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score,
    brier_score_loss, confusion_matrix, roc_curve)
from sklearn.isotonic import IsotonicRegression
from lightgbm import LGBMClassifier
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DPATH = os.path.join(PROJECT_ROOT, 'local_data')
RESULTS_DIR = os.path.join(DPATH, 'Results_imaging', '_external_validation')
os.makedirs(RESULTS_DIR, exist_ok=True)

N_SPLITS = 5; RANDOM_STATE = 2022; TOP_N = 10; TEST_SIZE = 0.2
CLUSTER_THRESHOLD = 0.75; TOP_N_S01 = 50; SFS_MAX = 15

BASE_PARAMS = {'n_estimators': 500, 'max_depth': 15, 'num_leaves': 10,
               'subsample': 0.7, 'learning_rate': 0.01, 'colsample_bytree': 0.7}

TARGETS = {
    'DM_full':  ('dementia_status', 'dementia_years', None),
    'DM_10yrs': ('dementia_status', 'dementia_years', 10),
    'DM_5yrs':  ('dementia_status', 'dementia_years', 5),
    'AD_full':  ('AD_status', 'AD_years', None),
    'AD_10yrs': ('AD_status', 'AD_years', 10),
    'AD_5yrs':  ('AD_status', 'AD_years', 5),
}

TARGET_LABELS = {
    'DM_full': 'DM (full)', 'DM_10yrs': 'DM (10yr)', 'DM_5yrs': 'DM (5yr)',
    'AD_full': 'AD (full)', 'AD_10yrs': 'AD (10yr)', 'AD_5yrs': 'AD (5yr)',
}

# ============================================================================
# 1. LOAD & SPLIT
# ============================================================================
print("=" * 70)
print("EXTERNAL VALIDATION: 80/20 Hold-Out (Imaging vs Non-Imaging)")
print("=" * 70)

print("\nLoading imaging data...")
X_img = pd.read_csv(os.path.join(DPATH, 'Preprocessed_Data', 'Preprocessed_Data_imaging.csv'))
print(f"  Shape: {X_img.shape}")

# Also load non-imaging data for comparison
X_noimg = pd.read_csv(os.path.join(DPATH, 'Preprocessed_Data', 'Preprocessed_Data.csv'))
print(f"  Non-imaging shape: {X_noimg.shape}")

# Load imaging feature list
img_list = pd.read_csv(os.path.join(DPATH, 'Preprocessed_Data', 'imaging_feature_list.csv'))
imaging_cols = set(img_list['feature'].tolist())

# Fix AD_years: preserve negative values for follow-up duration (censoring)
for df in [X_img, X_noimg]:
    df['AD_years'] = df['AD_years'].fillna(df['dementia_years'])

# Exclude baseline stroke
for df in [X_img, X_noimg]:
    sm = (df['stroke_years'] < 0) & (df['stroke_years'].notna())
    df.drop(df[sm].index, inplace=True)

# Align rows between imaging and non-imaging (same participants)
common_idx = X_img.index.intersection(X_noimg.index)
X_img = X_img.loc[common_idx].reset_index(drop=True)
X_noimg = X_noimg.loc[common_idx].reset_index(drop=True)

# 80/20 split stratified by DM_full
y_strat = X_img['dementia_status'].values
train_idx, test_idx = train_test_split(
    np.arange(len(X_img)), test_size=TEST_SIZE,
    stratify=y_strat, random_state=RANDOM_STATE
)
Xtr_img, Xte_img = X_img.iloc[train_idx].reset_index(drop=True), X_img.iloc[test_idx].reset_index(drop=True)
Xtr_nimg, Xte_nimg = X_noimg.iloc[train_idx].reset_index(drop=True), X_noimg.iloc[test_idx].reset_index(drop=True)

print(f"\nTrain: {len(Xtr_img):,}  |  Test: {len(Xte_img):,}")
print(f"Train DM_full events: {Xtr_img['dementia_status'].sum():,}")
print(f"Test  DM_full events: {Xte_img['dementia_status'].sum():,}")

# ============================================================================
# 2. RISK SCORES ON TEST SET
# ============================================================================
def compute_caide(data):
    n = len(data); age = data['21022-0.0'].values
    a = np.zeros(n); a[(age >= 47) & (age <= 53)] = 3; a[age > 53] = 4
    ey = data['845-0.0'].fillna(data['845-0.0'].median()).values - 5
    e = np.zeros(n); e[ey <= 6] = 3; e[(ey > 6) & (ey < 10)] = 2
    s = np.zeros(n); s[data['4080-0.0'].fillna(140).values >= 140] = 2
    bmi = data['21001-0.0'].copy(); bmi.loc[bmi.isnull()] = data['23104-0.0'].loc[bmi.isnull()]
    bmi.loc[bmi.isnull()] = bmi.median(); b = np.zeros(n); b[bmi > 30] = 2
    c = np.zeros(n); c[data['30690-0.0'].fillna(0).values > 6.5] = 2
    act = np.ones(n); act[data['22032-0.0_c1'].values > 0.5] = 0
    sx = data['31-0.0_c1'].values.astype(float)
    return a + e + sx + s + b + c + act

def compute_anu_adri(data):
    n = len(data); age = data['21022-0.0'].values; is_f = data['31-0.0_c0'].values.astype(float)
    am = np.zeros(n); af = np.zeros(n)
    for (lo, hi), vm, vf in [((65, 70), 1, 5), ((70, 75), 12, 14), ((75, 80), 18, 21),
                              ((80, 85), 26, 29), ((85, 90), 33, 35)]:
        am[(age >= lo) & (age < hi)] = vm; af[(age >= lo) & (age < hi)] = vf
    am[age >= 90] = 38; af[age >= 90] = 41
    a_pts = am * (1 - is_f) + af * is_f
    ey = data['845-0.0'].fillna(data['845-0.0'].median()).values - 5
    e_pts = np.full(n, 6); e_pts[(ey > 8) & (ey <= 11)] = 3; e_pts[ey > 11] = 0
    bmi = data['21001-0.0'].copy(); bmi.loc[bmi.isnull()] = data['23104-0.0'].loc[bmi.isnull()]
    bmi.loc[bmi.isnull()] = bmi.median(); bm = bmi.values
    b_pts = np.zeros(n); b_pts[(age >= 60) & (bm >= 25) & (bm < 30)] = 2; b_pts[(age >= 60) & (bm >= 30)] = 5
    dc = [c for c in data.columns if c.startswith('2443-0.0_c')]; d_pts = np.zeros(n)
    if dc: d_pts = data[dc].values.max(axis=1).clip(0, 1) * 3
    chol = data['30690-0.0'].fillna(0).values; c_pts = np.zeros(n); c_pts[(age >= 60) & (chol > 6.5)] = 3
    sk = np.zeros(n)
    if '20116-0.0_c1' in data.columns: sk[data['20116-0.0_c1'].values > 0.5] = 4
    if '20116-0.0_c2' in data.columns: sk[data['20116-0.0_c2'].values > 0.5] = 4
    sc = [c for c in data.columns if c.startswith('1031-0.0_c')]; nc = np.zeros(n)
    for ci, c in enumerate(sc): nc += data[c].fillna(0).values * (ci + 1)
    so = np.zeros(n); so[(nc > 1) & (nc <= 3)] = 1; so[(nc > 3) & (nc <= 5)] = 4; so[nc > 5] = 6
    act = np.zeros(n)
    if '22032-0.0_c1' in data.columns: act[data['22032-0.0_c1'].values > 0.5] = -2
    cog = data['20023-0.0'].fillna(500).values
    co = np.full(n, -6); co[(cog >= 500) & (cog < 585)] = -7; co[cog >= 585] = 0
    fv = np.zeros(n)
    for fid in ['1329', '1339']:
        fc = [c for c in data.columns if c.startswith(f'{fid}-0.0_c')]; fs = np.zeros(n)
        for ci, c in enumerate(fc): fs += data[c].fillna(0).values * (ci + 1)
        fv = np.maximum(fv, fs)
    fi = np.zeros(n); fi[fv >= 4] = -5; fi[fv == 3] = -4; fi[fv == 2] = -3
    return a_pts + e_pts + b_pts + d_pts + c_pts + sk + so + act + co + fi

print("\nComputing risk scores on test set...")
caide_te = compute_caide(Xte_img)
anu_te = compute_anu_adri(Xte_img)

# ============================================================================
# 3. FEATURE SELECTION ON TRAINING SET (IMAGING MODEL)
# ============================================================================
def normal_imp(d): s = sum(d.values()); return d if s == 0 else {k: v/s for k, v in d.items()}

def run_s01_s04(X_all, y, feat_cols, tag=""):
    """Run s01-s04 SFS on given feature columns. Returns list of selected features."""
    X_feat = X_all[feat_cols]
    print(f"\n  [{tag}] Features: {len(feat_cols)}")

    # s01: Initial ranking
    print("  s01: Ranking...")
    t0 = time.time()
    kf = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=True)
    tg_cv = Counter()
    for tr, te in kf.split(X_feat, y):
        gbm = LGBMClassifier(objective='binary', metric='auc', is_unbalance=True, verbosity=-1, seed=2020)
        gbm.set_params(**BASE_PARAMS)
        gbm.fit(X_feat.iloc[tr], y.iloc[tr])
        tg = dict(zip(gbm.booster_.feature_name(), gbm.booster_.feature_importance(importance_type='gain')))
        tg_cv += Counter(normal_imp(tg))
    s01_ranked = sorted(tg_cv.items(), key=lambda x: -x[1])
    top_50 = [f for f, _ in s01_ranked[:TOP_N_S01]]
    n_img_top = sum(1 for f in top_50 if f in imaging_cols)
    print(f"  s01 done ({time.time()-t0:.0f}s). Top 50: {n_img_top} imaging")
    for i, (f, g) in enumerate(s01_ranked[:5]):
        tag2 = '[IMG]' if f in imaging_cols else '[CLN]'
        print(f"    {i+1}. {tag2} {f} (Gain={g:.4f})")

    # s02: Hierarchical clustering
    print("  s02: Clustering...")
    t0 = time.time()
    X_top = X_feat[top_50].fillna(X_feat[top_50].median())
    corr = np.array(X_top.corr(method='spearman')); corr = np.nan_to_num(corr)
    corr = (corr + corr.T) / 2; np.fill_diagonal(corr, 1)
    dist_vec = (1 - np.abs(corr))[np.triu_indices(len(top_50), k=1)]
    link = linkage(dist_vec, method='ward')
    clusters = fcluster(link, t=CLUSTER_THRESHOLD, criterion='distance')
    kept = []; seen = set()
    for f, c in zip(top_50, clusters):
        if c not in seen: kept.append(f); seen.add(c)
    print(f"  s02 done ({time.time()-t0:.0f}s). {len(kept)} features after clustering")

    # s03: Re-rank
    print("  s03: Re-ranking...")
    t0 = time.time()
    X_kept = X_feat[kept]
    tg_cv3 = Counter()
    for tr, te in kf.split(X_kept, y):
        gbm = LGBMClassifier(objective='binary', metric='auc', is_unbalance=True, verbosity=-1, seed=2020)
        gbm.set_params(**BASE_PARAMS)
        gbm.fit(X_kept.iloc[tr], y.iloc[tr])
        tg = dict(zip(gbm.booster_.feature_name(), gbm.booster_.feature_importance(importance_type='gain')))
        tg_cv3 += Counter(normal_imp(tg))
    s03_ranked = sorted(tg_cv3.items(), key=lambda x: -x[1])
    s03_features = [f for f, _ in s03_ranked]
    print(f"  s03 done ({time.time()-t0:.0f}s)")
    for i, (f, g) in enumerate(s03_ranked[:5]):
        tag2 = '[IMG]' if f in imaging_cols else '[CLN]'
        print(f"    {i+1}. {tag2} {f} (Gain={g:.4f})")

    # s04: SFS
    print("  s04: SFS...")
    t0 = time.time()
    selected = []; remaining = list(s03_features)
    sfs_auc = []
    for step in range(SFS_MAX):
        best_f, best_auc = None, -1
        for cand in remaining[:min(15, len(remaining))]:
            trial = selected + [cand]
            aucs = []
            for tr, te in kf.split(X_kept[trial], y):
                g = LGBMClassifier(objective='binary', metric='auc', is_unbalance=True, verbosity=-1, seed=2020)
                g.set_params(**BASE_PARAMS)
                g.fit(X_kept[trial].iloc[tr], y.iloc[tr])
                aucs.append(roc_auc_score(y.iloc[te], g.predict_proba(X_kept[trial].iloc[te])[:, 1]))
            avg_auc = np.mean(aucs)
            if avg_auc > best_auc:
                best_f, best_auc = cand, avg_auc
        if best_f is None: break
        gain = best_auc - sfs_auc[-1] if sfs_auc else best_auc
        if gain < 0.001 and len(selected) >= TOP_N: break
        selected.append(best_f); remaining.remove(best_f); sfs_auc.append(best_auc)
        tag2 = '[IMG]' if best_f in imaging_cols else '[CLN]'
        print(f"    Step {step+1}: {tag2} {best_f} → AUC={best_auc:.4f} (gain={gain:+.4f})")
    n_img_sel = sum(1 for f in selected if f in imaging_cols)
    print(f"  s04 done ({time.time()-t0:.0f}s). {len(selected)} features ({n_img_sel} imaging)")
    return selected

# ---- FEATURE SELECTION FOR IMAGING MODEL ----
target_col, years_col, max_years = TARGETS['DM_full']
y_train_dm = Xtr_img[target_col].copy()
if max_years is not None:
    y_train_dm.loc[(y_train_dm == 1) & (Xtr_img[years_col] > max_years)] = 0

exclude_pfx = ('dementia_', 'AD_', 'VD_', 'stroke_', 'eid', 'has_brain_mri')
img_feat_cols = [c for c in Xtr_img.columns
                 if not any(c.startswith(p) for p in exclude_pfx)]

print(f"\n{'='*70}")
print("FEATURE SELECTION: IMAGING MODEL")
print(f"{'='*70}")
sel_img = run_s01_s04(Xtr_img, y_train_dm, img_feat_cols, tag="IMAGING")

# ---- FEATURE SELECTION FOR NON-IMAGING MODEL ----
nimg_feat_cols = [c for c in Xtr_nimg.columns
                  if not any(c.startswith(p) for p in exclude_pfx)
                  and c not in imaging_cols]

print(f"\n{'='*70}")
print("FEATURE SELECTION: NON-IMAGING MODEL")
print(f"{'='*70}")
sel_nimg = run_s01_s04(Xtr_nimg, y_train_dm, nimg_feat_cols, tag="NON-IMAGING")

# ============================================================================
# 4. TRAIN & EVALUATE ON TEST SET
# ============================================================================
def evaluate_model(X_train_all, X_test_all, selected_features, model_name):
    """Train + calibrate on train, evaluate on test for all 6 targets."""
    results = []
    for tname, (tcol, ycol, my) in TARGETS.items():
        y_tr = X_train_all[tcol].copy()
        include_tr = pd.Series(True, index=X_train_all.index)
        if my is not None:
            y_tr.loc[(y_tr == 1) & (X_train_all[ycol] > my)] = 0
            # Censoring: exclude non-converters with insufficient follow-up
            insufficient_tr = (
                (y_tr == 0) & (X_train_all[ycol] < 0) &
                (X_train_all[ycol].abs() < my)
            )
            include_tr = ~insufficient_tr

        y_te = X_test_all[tcol].copy()
        include_te = pd.Series(True, index=X_test_all.index)
        if my is not None:
            y_te.loc[(y_te == 1) & (X_test_all[ycol] > my)] = 0
            insufficient_te = (
                (y_te == 0) & (X_test_all[ycol] < 0) &
                (X_test_all[ycol].abs() < my)
            )
            include_te = ~insufficient_te

        test_f = [f for f in selected_features if f in X_test_all.columns]
        X_tr = X_train_all[test_f][include_tr]
        X_te = X_test_all[test_f][include_te]
        y_tr = y_tr[include_tr]
        y_te = y_te[include_te]

        n_calib = int(len(X_tr) * 0.4)
        gbm = LGBMClassifier(objective='binary', is_unbalance=True, metric='auc', verbosity=-1, seed=2022)
        gbm.set_params(**BASE_PARAMS)
        gbm.fit(X_tr.iloc[n_calib:], y_tr.iloc[n_calib:])

        raw_cali = gbm.predict_proba(X_tr.iloc[:n_calib])[:, 1]
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(raw_cali, y_tr.iloc[:n_calib])

        y_pred = np.clip(iso.predict(gbm.predict_proba(X_te)[:, 1]), 0, 1)

        auc_v = roc_auc_score(y_te, y_pred)
        brier_v = brier_score_loss(y_te, y_pred)
        ap_v = average_precision_score(y_te, y_pred)

        fpr, tpr, thresholds = roc_curve(y_te, y_pred)
        bi = np.argmax(tpr - fpr)
        best_c = thresholds[bi] if bi < len(thresholds) else 0.05

        y_bin = (y_pred >= best_c).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te, y_bin).ravel()
        acc = (tp + tn) / (tp + tn + fp + fn)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0

        results.append({
            'Target': tname, 'Model': model_name,
            'AUC': round(auc_v, 4), 'Brier': round(brier_v, 4),
            'AP': round(ap_v, 4), 'Accuracy': round(acc, 4),
            'Sensitivity': round(sens, 4), 'Specificity': round(spec, 4),
            'Best_Cutoff': round(best_c, 4),
            'n_events': int(y_te.sum()), 'n_total': len(y_te),
        })
    return results

print(f"\n{'='*70}")
print("EVALUATION ON TEST SET")
print(f"{'='*70}")

# Imaging model evaluation
res_img = evaluate_model(Xtr_img, Xte_img, sel_img, 'OUR MODEL (+MRI)')
# Non-imaging model evaluation
res_nimg = evaluate_model(Xtr_nimg, Xte_nimg, sel_nimg, 'OUR MODEL (no MRI)')

# CAIDE & ANU-ADRI on test set
res_risk = []
for tname, (tcol, ycol, my) in TARGETS.items():
    y_te = Xte_img[tcol].copy()
    if my is not None:
        y_te.loc[(y_te == 1) & (Xte_img[ycol] > my)] = 0

    for sname, svals in [('CAIDE', caide_te), ('ANU-ADRI', anu_te)]:
        auc_s = roc_auc_score(y_te, svals)
        s_min, s_max = svals.min(), svals.max()
        s_norm = (svals - s_min) / (s_max - s_min) if s_max > s_min else svals
        brier_s = brier_score_loss(y_te, s_norm)
        sfpr, stpr, sthresh = roc_curve(y_te, svals)
        s_best = sthresh[np.argmax(stpr - sfpr)]
        s_bin = (svals >= s_best).astype(int)
        stn, sfp, sfn, stp = confusion_matrix(y_te, s_bin).ravel()
        s_acc = (stp + stn) / (stp + stn + sfp + sfn)
        s_sens = stp / (stp + sfn) if (stp + sfn) > 0 else 0
        s_spec = stn / (stn + sfp) if (stn + sfp) > 0 else 0
        res_risk.append({
            'Target': tname, 'Model': sname, 'AUC': round(auc_s, 4),
            'Brier': round(brier_s, 4), 'Accuracy': round(s_acc, 4),
            'Sensitivity': round(s_sens, 4), 'Specificity': round(s_spec, 4),
        })

# ============================================================================
# 5. SUMMARY
# ============================================================================
all_results = res_risk + res_img + res_nimg
results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(RESULTS_DIR, 'external_validation_all.csv'), index=False)

# Comparison table
print(f"\n{'='*90}")
print("EXTERNAL VALIDATION: AUC COMPARISON")
print(f"{'='*90}")
print(f"{'Target':<12} {'CAIDE':<10} {'ANU-ADRI':<10} {'OUR(无MRI)':<12} {'OUR(+MRI)':<12} {'Δ(无MRI)':<10} {'Δ(+MRI)':<10}")
print(f"{'─'*12} {'─'*10} {'─'*10} {'─'*12} {'─'*12} {'─'*10} {'─'*10}")

summary_rows = []
for tname in TARGETS.keys():
    rows = results_df[results_df['Target'] == tname]
    caide_auc = float(rows[rows['Model'] == 'CAIDE']['AUC'].iloc[0])
    anu_auc = float(rows[rows['Model'] == 'ANU-ADRI']['AUC'].iloc[0])
    nimg_auc = float(rows[rows['Model'] == 'OUR MODEL (no MRI)']['AUC'].iloc[0])
    img_auc = float(rows[rows['Model'] == 'OUR MODEL (+MRI)']['AUC'].iloc[0])

    d_nimg = nimg_auc - caide_auc
    d_img = img_auc - caide_auc

    print(f"{tname:<12} {caide_auc:<10.4f} {anu_auc:<10.4f} {nimg_auc:<12.4f} {img_auc:<12.4f} {d_nimg:<+10.4f} {d_img:<+10.4f}")

    summary_rows.append({
        'Target': tname, 'CAIDE_AUC': caide_auc, 'ANU_ADRI_AUC': anu_auc,
        'Ours_noMRI_AUC': nimg_auc, 'Ours_MRI_AUC': img_auc,
        'Delta_noMRI_vs_CAIDE': round(d_nimg, 4),
        'Delta_MRI_vs_CAIDE': round(d_img, 4),
        'MRI_improvement': round(img_auc - nimg_auc, 4),
    })

pd.DataFrame(summary_rows).to_csv(os.path.join(RESULTS_DIR, 'external_validation_summary.csv'), index=False)

# Brier comparison
print(f"\n{'='*90}")
print("BRIER SCORE COMPARISON")
print(f"{'='*90}")
print(f"{'Target':<12} {'CAIDE':<12} {'ANU-ADRI':<12} {'OUR(无MRI)':<14} {'OUR(+MRI)':<14}")
print(f"{'─'*12} {'─'*12} {'─'*12} {'─'*14} {'─'*14}")
for tname in TARGETS.keys():
    rows = results_df[results_df['Target'] == tname]
    cb = float(rows[rows['Model'] == 'CAIDE']['Brier'].iloc[0])
    ab = float(rows[rows['Model'] == 'ANU-ADRI']['Brier'].iloc[0])
    nb = float(rows[rows['Model'] == 'OUR MODEL (no MRI)']['Brier'].iloc[0])
    ib = float(rows[rows['Model'] == 'OUR MODEL (+MRI)']['Brier'].iloc[0])
    print(f"{tname:<12} {cb:<12.6f} {ab:<12.6f} {nb:<14.6f} {ib:<14.6f}")

# ============================================================================
# 6. BAR CHART
# ============================================================================
fig, ax = plt.subplots(figsize=(14, 6))
targets_short = [t.replace('DM_', 'DM_').replace('AD_', 'AD_') for t in TARGETS.keys()]
x = np.arange(len(targets_short)); w = 0.2
for i, tname in enumerate(TARGETS.keys()):
    rows = results_df[results_df['Target'] == tname]
    ax.bar(i - 1.5*w, float(rows[rows['Model'] == 'CAIDE']['AUC'].iloc[0]), w, label='CAIDE' if i == 0 else '', color='#E8A87C')
    ax.bar(i - 0.5*w, float(rows[rows['Model'] == 'ANU-ADRI']['AUC'].iloc[0]), w, label='ANU-ADRI' if i == 0 else '', color='#95E1D3')
    ax.bar(i + 0.5*w, float(rows[rows['Model'] == 'OUR MODEL (no MRI)']['AUC'].iloc[0]), w, label='OUR (no MRI)' if i == 0 else '', color='#6B7280')
    ax.bar(i + 1.5*w, float(rows[rows['Model'] == 'OUR MODEL (+MRI)']['AUC'].iloc[0]), w, label='OUR (+MRI)' if i == 0 else '', color='#3B82F6')

ax.set_xticks(x); ax.set_xticklabels(targets_short, fontsize=10)
ax.set_ylabel('AUC', fontsize=12); ax.set_ylim(0.4, 1.0)
ax.set_title('External Validation (80/20 Hold-Out): AUC Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=9, loc='lower right'); ax.grid(axis='y', alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, 'external_validation_auc.png'), dpi=150); plt.close()

# Feature comparison
feat_df = pd.DataFrame({
    'Rank': range(1, max(len(sel_img), len(sel_nimg)) + 1),
    'Imaging_Model_Features': sel_img + [''] * (max(len(sel_img), len(sel_nimg)) - len(sel_img)),
    'NonImaging_Model_Features': sel_nimg + [''] * (max(len(sel_img), len(sel_nimg)) - len(sel_nimg)),
})
feat_df.to_csv(os.path.join(RESULTS_DIR, 'selected_features_comparison.csv'), index=False)

print(f"\nSelected features (Imaging): {sel_img}")
print(f"Selected features (No-MRI):  {sel_nimg}")
print(f"\nResults saved to: {RESULTS_DIR}")
print("Done!")

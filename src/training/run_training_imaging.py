#!/usr/bin/env python3
"""
UKB-DRP Training with Brain MRI Imaging Features
=================================================
Extends the paper's pipeline (s01-s05) with brain MRI imaging-derived phenotypes.

Key design:
  - Full-cohort mode: All ~425K participants, LightGBM handles NaN in imaging features
  - Imaging-subset mode: Only ~46K participants with imaging data (complete case)

Reference: Yu et al., eClinicalMedicine 2022;53:101665
"""
import os, sys, warnings, time, gc, json
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy import stats
from collections import Counter, defaultdict
from itertools import product
import random

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              confusion_matrix, brier_score_loss)
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from lightgbm import LGBMClassifier

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ============================================================================
# CONFIGURATION
# ============================================================================

DPATH = os.path.join(PROJECT_ROOT, 'local_data') + '/'
PREPROCESSED_CSV = os.path.join(DPATH, 'Preprocessed_Data', 'Preprocessed_Data_imaging.csv')
IMAGING_LIST = os.path.join(DPATH, 'Preprocessed_Data', 'imaging_feature_list.csv')
RESULTS_BASE = os.path.join(DPATH, 'Results_imaging')
DATA_DIR = os.path.join(DPATH, 'Data')
os.makedirs(RESULTS_BASE, exist_ok=True)

RANDOM_STATE = 2022
N_SPLITS = 5

ALL_TARGETS = {
    'DM_full':  ('dementia_status', 'dementia_years', None),
    'DM_10yrs': ('dementia_status', 'dementia_years', 10),
    'DM_5yrs':  ('dementia_status', 'dementia_years', 5),
    'AD_full':  ('AD_status', 'AD_years', None),
    'AD_10yrs': ('AD_status', 'AD_years', 10),
    'AD_5yrs':  ('AD_status', 'AD_years', 5),
}

BASE_PARAMS = {'n_estimators': 500, 'max_depth': 15, 'num_leaves': 10,
               'subsample': 0.7, 'learning_rate': 0.01, 'colsample_bytree': 0.7}

PARAM_GRID = {
    'n_estimators': [200, 300, 500, 800],
    'max_depth': [8, 10, 15, 20],
    'num_leaves': [8, 10, 16, 31, 50],
    'subsample': [0.5, 0.6, 0.7, 0.8],
    'learning_rate': [0.005, 0.01, 0.02, 0.05, 0.1],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8],
}

N_PARAM_COMBOS = 1000
TOP_N_S01 = 50
CLUSTER_THRESHOLD = 0.75
TOP_N_S04 = 10
SFS_MAX_FEATURES = 15
N_PARAM_COMBOS = 1000

# HES fields to exclude
RM_HES = ['41234-0.0', '41259-0.0', '41149-0.0', '41289-0.0',
          '41218-0.0', '41235-0.0', '41214-0.0']


# ============================================================================
# UTILITY FUNCTIONS (same as paper pipeline)
# ============================================================================

def threshold(array, cutoff):
    arr = array.copy()
    arr[arr < cutoff] = 0
    arr[arr >= cutoff] = 1
    return arr

def extend(my_array, nb_points):
    if len(my_array) == nb_points:
        pass
    else:
        nb2impute = nb_points - len(my_array)
        impute_array = np.zeros(nb2impute)
        my_array = np.concatenate((impute_array, my_array), axis=0)
    return np.expand_dims(my_array, -1)

def normal_imp(mydict):
    mysum = sum(mydict.values())
    if mysum == 0:
        return mydict
    return {k: v / mysum for k, v in mydict.items()}

def get_full_eval(y_test, pred_prob, cutoff_list):
    evaluations = []
    for cutoff in cutoff_list:
        pred_binary = threshold(pred_prob, cutoff)
        tn, fp, fn, tp = confusion_matrix(y_test, pred_binary).ravel()
        acc = (tp + tn) / (tp + tn + fp + fn)
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        y_idx = sens + spec - 1
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0
        auc = roc_auc_score(y_test, pred_prob)
        apr = average_precision_score(y_test, pred_prob)
        nnd = 1 / y_idx if y_idx > 0 else np.inf
        evaluations.append(np.round((cutoff, acc, sens, spec, prec, y_idx, f1, auc, apr, nnd), 4))
    evaluations = pd.DataFrame(evaluations)
    evaluations.columns = ['Cutoff', 'Acc', 'Sens', 'Spec', 'Prec', 'Youden', 'F1', 'AUC', 'APR', 'NND']
    return evaluations

def avg_results(results_list):
    col_names = results_list[0].columns.tolist()
    col_names_std = [item + '_std' for item in col_names]
    nb_fold, nb_row, nb_col = len(results_list), results_list[0].shape[0], results_list[0].shape[1]
    results = np.zeros((nb_fold, nb_row, nb_col))
    for i in range(nb_fold):
        results[i] = np.array(results_list[i])
    results_avg = pd.DataFrame(np.round(np.average(results, axis=0), 3), columns=col_names)
    results_std = pd.DataFrame(np.round(np.std(results, axis=0), 3), columns=col_names_std)
    return pd.concat((results_avg, results_std), axis=1)

def select_params_combo(my_dict, nb_items):
    combo_list = [dict(zip(my_dict.keys(), v)) for v in product(*my_dict.values())]
    random.seed(2020)
    return random.sample(combo_list, min(nb_items, len(combo_list)))

def hl_pvalue(obs_prob, pred_prob, percentage=1, bin_obs_nb=None):
    obs_prob = obs_prob / percentage
    pred_prob = pred_prob / percentage
    nominator = (obs_prob - pred_prob) ** 2 * bin_obs_nb if bin_obs_nb is not None else (obs_prob - pred_prob) ** 2
    denominator = pred_prob * (1 - pred_prob)
    denominator[denominator == 0] = 1
    stat = nominator / denominator
    stat[stat == np.inf] = 0.0
    stat[stat == -np.inf] = 0.0
    test_stat = np.nansum(stat)
    pvalue = 1 - stats.chi2.cdf(test_stat, 8)
    return np.round(pvalue, 4)

def hl_test_per_fold(y_true, y_pred, n_bins=10):
    try:
        _, bin_edges = pd.qcut(y_pred, q=n_bins, retbins=True, duplicates='drop')
        bin_labels = pd.cut(y_pred, bins=bin_edges, include_lowest=True, duplicates='drop')
        obs_probs, pred_probs, bin_counts = [], [], []
        categories = getattr(bin_labels, 'cat', bin_labels).categories
        for interval in categories:
            mask = bin_labels == interval
            if mask.sum() > 0:
                obs_probs.append(y_true[mask].mean())
                pred_probs.append(y_pred[mask].mean())
                bin_counts.append(mask.sum())
        if len(obs_probs) < 2:
            return 1.0
    except Exception:
        return 1.0
    return hl_pvalue(np.array(obs_probs), np.array(pred_probs),
                     percentage=1, bin_obs_nb=np.array(bin_counts))


# ============================================================================
# DATA LOADING
# ============================================================================

def load_and_prepare_data(imaging_subset_only=False):
    """Load combined clinical + imaging data."""
    mydf = pd.read_csv(PREPROCESSED_CSV)

    # Fix AD_years
    mydf['AD_years'] = mydf['AD_years'].fillna(mydf['dementia_years'])
    mydf['AD_years'] = mydf['AD_years'].clip(lower=-1)

    # Load imaging feature list
    if os.path.exists(IMAGING_LIST):
        img_list = pd.read_csv(IMAGING_LIST)
        imaging_cols = set(img_list['feature'].tolist())
    else:
        imaging_cols = set()

    # Filter to imaging subset if requested
    if imaging_subset_only and 'has_brain_mri' in mydf.columns:
        mydf = mydf[mydf['has_brain_mri'] == 1].copy()
        print(f"  Imaging subset: {len(mydf):,} participants")
        # Remove fully-missing imaging columns within this subset
        img_cols_in_df = [c for c in imaging_cols if c in mydf.columns]
        fully_missing = [c for c in img_cols_in_df if mydf[c].isnull().mean() > 0.95]
        if fully_missing:
            mydf.drop(columns=fully_missing, inplace=True)
            print(f"  Removed {len(fully_missing)} imaging cols with >95% missing in subset")

    # Outcome/ID columns to remove
    rm_f1 = ['Unnamed: 0', 'eid', 'dementia_status', 'dementia_years',
             'AD_status', 'AD_years', 'VD_status', 'VD_years',
             'stroke_status', 'stroke_years']
    rm_f = [c for c in rm_f1 + RM_HES if c in mydf.columns]

    return mydf, rm_f, imaging_cols


def apply_time_window(mydf, target_col, years_col, max_years=None):
    y = mydf[target_col].copy()
    if max_years is not None:
        y_years = mydf[years_col]
        mask = (y == 1) & (y_years > max_years)
        y.loc[mask] = 0
    return y


def exclude_baseline_stroke(mydf):
    stroke_mask = (mydf['stroke_years'] < 0) & (mydf['stroke_years'].notna())
    print(f"  Excluding {stroke_mask.sum()} participants with baseline stroke")
    return mydf[~stroke_mask].copy()


# ============================================================================
# S01: INITIAL FEATURE IMPORTANCE (ALL FEATURES: clinical + imaging)
# ============================================================================

def run_s01(mydf, X, y, results_dir, imaging_cols):
    print(f"\n{'='*70}")
    print(f"[s01] Initial feature importance ranking ({X.shape[1]} features)")
    n_img = len([c for c in X.columns if c in imaging_cols])
    n_clin = X.shape[1] - n_img
    print(f"  Clinical features: {n_clin}, Imaging features: {n_img}")
    print(f"{'='*70}")
    t0 = time.time()

    mykf = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=True)
    tg_imp_cv = Counter()
    tc_imp_cv = Counter()

    for fold, (train_idx, test_idx) in enumerate(mykf.split(X, y)):
        X_train, y_train = X.iloc[train_idx, :], y.iloc[train_idx]
        my_lgb = LGBMClassifier(objective='binary', metric='auc',
                                 is_unbalance=True, verbosity=-1, seed=2020)
        my_lgb.set_params(**BASE_PARAMS)
        my_lgb.fit(X_train, y_train)

        tg_imp = my_lgb.booster_.feature_importance(importance_type='gain')
        tg_imp = dict(zip(my_lgb.booster_.feature_name(), tg_imp.tolist()))
        tc_imp = my_lgb.booster_.feature_importance(importance_type='split')
        tc_imp = dict(zip(my_lgb.booster_.feature_name(), tc_imp.tolist()))

        tg_imp_cv += Counter(normal_imp(tg_imp))
        tc_imp_cv += Counter(normal_imp(tc_imp))
        print(f"  Fold {fold+1}/{N_SPLITS} complete")

    imp_df = pd.DataFrame({
        'Features': list(tg_imp_cv.keys()),
        'Cover': [tc_imp_cv.get(k, 0) for k in tg_imp_cv.keys()],
        'Gain': list(tg_imp_cv.values()),
    })
    imp_df.sort_values(by='Gain', ascending=False, inplace=True)

    # Mark feature source
    imp_df['IsImaging'] = [f in imaging_cols for f in imp_df['Features']]

    pos_df = mydf.loc[y == 1]
    imp_df['NA_full'] = [round(mydf[ele].isnull().sum() * 100 / len(mydf), 1)
                         for ele in imp_df['Features']]
    imp_df['NA_target'] = [round(pos_df[ele].isnull().sum() * 100 / len(pos_df), 1)
                           for ele in imp_df['Features']]
    imp_df['Path'] = ''
    imp_df['Field'] = imp_df['Features']
    imp_df['ValueType'] = ''
    imp_df['Units'] = ''

    imp_df.to_csv(os.path.join(results_dir, 's01_feature_importance.csv'), index=False)

    top_f = imp_df['Features'][:TOP_N_S01].tolist()
    n_top_img = sum(1 for f in top_f if f in imaging_cols)
    print(f"  Done in {time.time()-t0:.0f}s.")
    print(f"  Top {TOP_N_S01} includes {n_top_img} imaging features")
    print(f"  Top 10:")
    for i, f in enumerate(top_f[:10]):
        tag = "[IMG]" if f in imaging_cols else "[CLN]"
        print(f"    {i+1:2d}. {tag} {f}")
    return imp_df


# ============================================================================
# S02: HIERARCHICAL CLUSTERING
# ============================================================================

def run_s02(mydf, X_all, y, s01_df, results_dir):
    print(f"\n{'='*70}")
    print(f"[s02] Hierarchical clustering")
    print(f"{'='*70}")
    t0 = time.time()

    top_f_list = s01_df['Features'][:TOP_N_S01].tolist()
    X = X_all[top_f_list]

    corr = np.array(X.corr(method='spearman'))
    corr = np.nan_to_num(corr)
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)

    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))

    # Dendrogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
    dendro = hierarchy.dendrogram(dist_linkage, labels=top_f_list, ax=ax2)
    ax2.set_xticklabels(dendro["ivl"], rotation=60, fontsize=7, horizontalalignment='right')
    dendro_idx = np.arange(0, len(dendro["ivl"]))
    ax1.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]], cmap='RdBu_r', vmin=-1, vmax=1)
    ax1.set_xticks(dendro_idx)
    ax1.set_yticks(dendro_idx)
    ax1.set_xticklabels(dendro["ivl"], rotation=60, fontsize=7, horizontalalignment='right')
    ax1.set_yticklabels(dendro["ivl"], fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 's02_dendrogram.png'), dpi=150)
    plt.close()

    cluster_ids = hierarchy.fcluster(dist_linkage, CLUSTER_THRESHOLD, criterion="distance")
    cluster_to_features = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_to_features[cluster_id].append(idx)

    selected_idx = [v[0] for v in cluster_to_features.values()]
    selected_f = [top_f_list[i] for i in selected_idx]

    s02_out = s01_df[s01_df['Features'].isin(selected_f)].copy()
    s02_out.to_csv(os.path.join(results_dir, 's02_clustered_features.csv'), index=False)

    print(f"  Done in {time.time()-t0:.0f}s. {len(selected_f)} features after clustering")
    return s02_out


# ============================================================================
# S03: FINAL FEATURE IMPORTANCE
# ============================================================================

def run_s03(mydf, X_all, y, s02_df, results_dir):
    print(f"\n{'='*70}")
    print(f"[s03] Final feature importance ({len(s02_df)} features)")
    print(f"{'='*70}")
    t0 = time.time()

    my_f_list = s02_df['Features'].tolist()
    X = X_all[my_f_list]

    mykf = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=True)
    tg_imp_cv = Counter()
    tc_imp_cv = Counter()

    for fold, (train_idx, test_idx) in enumerate(mykf.split(X, y)):
        X_train, y_train = X.iloc[train_idx, :], y.iloc[train_idx]
        my_lgb = LGBMClassifier(objective='binary', metric='auc',
                                 is_unbalance=True, verbosity=-1, seed=2020)
        my_lgb.set_params(**BASE_PARAMS)
        my_lgb.fit(X_train, y_train)

        tg_imp = my_lgb.booster_.feature_importance(importance_type='gain')
        tg_imp = dict(zip(my_lgb.booster_.feature_name(), tg_imp.tolist()))
        tc_imp = my_lgb.booster_.feature_importance(importance_type='split')
        tc_imp = dict(zip(my_lgb.booster_.feature_name(), tc_imp.tolist()))

        tg_imp_cv += Counter(normal_imp(tg_imp))
        tc_imp_cv += Counter(normal_imp(tc_imp))
        print(f"  Fold {fold+1}/{N_SPLITS} complete")

    imp_df = pd.DataFrame({
        'Features': list(tg_imp_cv.keys()),
        'Cover': [tc_imp_cv.get(k, 0) / N_SPLITS for k in tg_imp_cv.keys()],
        'Gain': [v / N_SPLITS for k, v in tg_imp_cv.items()],
    })
    imp_df.sort_values(by='Gain', ascending=False, inplace=True)

    pos_df = mydf.loc[y == 1]
    imp_df['NA_full'] = [round(mydf[ele].isnull().sum() * 100 / len(mydf), 1)
                         for ele in imp_df['Features']]
    imp_df['NA_target'] = [round(pos_df[ele].isnull().sum() * 100 / len(pos_df), 1)
                           for ele in imp_df['Features']]
    imp_df['Path'] = ''
    imp_df['Field'] = imp_df['Features']
    imp_df['ValueType'] = ''
    imp_df['Units'] = ''

    imp_df.to_csv(os.path.join(results_dir, 's03_final_importance.csv'), index=False)

    print(f"  Done in {time.time()-t0:.0f}s. Top features:")
    for i, row in imp_df.head(10).iterrows():
        print(f"    {imp_df.index.get_loc(i)+1:2d}. {row['Features']} (Gain={row['Gain']:.4f})")
    return imp_df


# ============================================================================
# S04: SEQUENTIAL FORWARD SELECTION
# ============================================================================

def run_s04_sfs(mydf, X_all, y, s03_df, results_dir):
    print(f"\n{'='*70}")
    print(f"[s04] Sequential Forward Selection (SFS)")
    print(f"{'='*70}")
    t0 = time.time()

    all_features = s03_df['Features'].tolist()
    max_features = min(SFS_MAX_FEATURES, len(all_features))

    mykf = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=True)

    def evaluate_feature_set(feature_set):
        X_sub = X_all[list(feature_set)]
        aucs = []
        for train_idx, test_idx in mykf.split(X_sub, y):
            X_tr, X_te = X_sub.iloc[train_idx, :], X_sub.iloc[test_idx, :]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            lgb = LGBMClassifier(objective='binary', metric='auc',
                                  is_unbalance=True, n_jobs=4,
                                  verbosity=-1, seed=2020)
            lgb.set_params(**BASE_PARAMS)
            lgb.fit(X_tr, y_tr)
            aucs.append(roc_auc_score(y_te, lgb.predict_proba(X_te)[:, 1]))
        return np.mean(aucs), np.std(aucs)

    selected = []
    remaining = list(range(len(all_features)))
    sfs_history = []

    first_idx = 0
    selected.append(first_idx)
    remaining.remove(first_idx)

    current_auc, current_std = evaluate_feature_set([all_features[i] for i in selected])
    sfs_history.append({
        'step': 1, 'feature_added': all_features[first_idx],
        'selected_count': 1,
        'selected_features': [all_features[i] for i in selected],
        'auc_mean': current_auc, 'auc_std': current_std
    })
    print(f"  Step 1: Added {all_features[first_idx]:40s} → AUC = {current_auc:.4f}")

    for step in range(1, max_features):
        best_auc = -1
        best_idx = None
        best_std = 0

        for cand_idx in remaining:
            trial_set = [all_features[i] for i in selected + [cand_idx]]
            auc_mean, auc_std = evaluate_feature_set(trial_set)
            if auc_mean > best_auc:
                best_auc = auc_mean
                best_idx = cand_idx
                best_std = auc_std

        if best_idx is None:
            break

        auc_gain = best_auc - current_auc
        if auc_gain < 0.001 and len(selected) >= TOP_N_S04:
            print(f"  Stopping: AUC gain ({auc_gain:.4f}) < 0.001")
            break

        selected.append(best_idx)
        remaining.remove(best_idx)
        current_auc = best_auc
        current_std = best_std

        sfs_history.append({
            'step': step + 1, 'feature_added': all_features[best_idx],
            'selected_count': len(selected),
            'selected_features': [all_features[i] for i in selected],
            'auc_mean': current_auc, 'auc_std': current_std
        })
        print(f"  Step {step+1}: Added {all_features[best_idx]:40s} → AUC = {current_auc:.4f} (gain={auc_gain:.4f})")

    # Save SFS history
    sfs_save = [{k: v for k, v in d.items() if k != 'selected_features'} for d in sfs_history]
    sfs_df = pd.DataFrame(sfs_save)
    sfs_df['selected_features'] = [', '.join(d['selected_features']) for d in sfs_history]
    sfs_df.to_csv(os.path.join(results_dir, 's04_sfs_history.csv'), index=False)

    selected_features = [all_features[i] for i in selected]
    s04_out = pd.DataFrame({'Features': selected_features})
    s04_out['SelectionOrder'] = range(1, len(selected) + 1)
    s04_out['CumulativeAUC'] = [d['auc_mean'] for d in sfs_history]
    s04_out['AUC_std'] = [d['auc_std'] for d in sfs_history]

    s04_out = s04_out.merge(
        s03_df[['Features', 'Gain', 'Cover', 'NA_full', 'NA_target']],
        on='Features', how='left'
    )

    s04_out['Path'] = ''
    s04_out['Field'] = s04_out['Features']
    s04_out['ValueType'] = ''
    s04_out['Units'] = ''

    s04_out.to_csv(os.path.join(results_dir, 's04_selected_features.csv'), index=False)

    # Cumulative AUC comparison
    _save_cumulative_auc_comparison(X_all, y, selected_features, mykf, results_dir)

    n_img_selected = sum(1 for f in selected_features if f.startswith(('250', '251', '252', '253', '254', '255', '256', '257', '258', '259', '260', '261', '262', '263', '264', '265', '266', '267', '268', '269', '270', '271', '272', '273', '274', '275', '276', '277')))
    print(f"\n  Done in {time.time()-t0:.0f}s.")
    print(f"  Selected {len(selected)} features ({n_img_selected} imaging, {len(selected)-n_img_selected} clinical)")
    return s04_out


def _save_cumulative_auc_comparison(X_all, y, selected_features, mykf, results_dir):
    tmp_f, AUC_cv_lst = [], []
    for f in selected_features:
        tmp_f.append(f)
        X_sub = X_all[tmp_f]
        AUC_cv = []
        for train_idx, test_idx in mykf.split(X_sub, y):
            X_tr, X_te = X_sub.iloc[train_idx, :], X_sub.iloc[test_idx, :]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            lgb = LGBMClassifier(objective='binary', metric='auc',
                                  is_unbalance=True, n_jobs=4,
                                  verbosity=-1, seed=2020)
            lgb.set_params(**BASE_PARAMS)
            lgb.fit(X_tr, y_tr)
            AUC_cv.append(roc_auc_score(y_te, lgb.predict_proba(X_te)[:, 1]))
        tmp_out = [np.mean(AUC_cv), np.std(AUC_cv)] + AUC_cv
        AUC_cv_lst.append(np.round(tmp_out, 4))

    cum_df = pd.DataFrame({
        'Features': tmp_f,
        'AUC_mean': [a[0] for a in AUC_cv_lst],
        'AUC_std': [a[1] for a in AUC_cv_lst],
    })
    for i in range(N_SPLITS):
        cum_df[f'AUC{i}'] = [a[2+i] for a in AUC_cv_lst]
    cum_df.to_csv(os.path.join(results_dir, 's04_cumulative_auc_comparison.csv'), index=False)


# ============================================================================
# S05: FINAL MODEL WITH HYPERPARAMETER TUNING + ISOTONIC CALIBRATION
# ============================================================================

def run_s05(mydf, X_all, y, s04_df, results_dir, target_name):
    print(f"\n{'='*70}")
    print(f"[s05] Final model with hyperparameter tuning & Isotonic calibration")
    print(f"{'='*70}")
    t0 = time.time()

    my_f_list = s04_df['Features'][:TOP_N_S04].tolist()
    X = X_all[my_f_list]

    outer_kf = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=True)
    cutoff_list = [np.round(0.001 + i * 0.001, 3) for i in range(100)]

    print(f"\n  Hyperparameter tuning: sampling {N_PARAM_COMBOS} combos...")
    param_combos = select_params_combo(PARAM_GRID, N_PARAM_COMBOS)
    all_best_params = []

    results_cv = []
    obs_array, pred_array = np.zeros((10, 1)), np.zeros((10, 1))
    y_test_lst, y_pred_prob_lst = [], []
    all_fold_predictions = []

    for fold, (outer_train_idx, outer_test_idx) in enumerate(outer_kf.split(X, y)):
        print(f"\n  --- Outer Fold {fold+1}/{N_SPLITS} ---")

        X_outer_train = X.iloc[outer_train_idx, :]
        y_outer_train = y.iloc[outer_train_idx]
        X_outer_test = X.iloc[outer_test_idx, :]
        y_outer_test = y.iloc[outer_test_idx]

        # Nested CV for hyperparameter selection
        inner_kf = StratifiedKFold(n_splits=3, random_state=RANDOM_STATE + fold, shuffle=True)
        param_scores = []

        for pi, params in enumerate(param_combos):
            inner_aucs = []
            for inner_train_idx, inner_val_idx in inner_kf.split(X_outer_train, y_outer_train):
                X_it, X_iv = X_outer_train.iloc[inner_train_idx, :], X_outer_train.iloc[inner_val_idx, :]
                y_it, y_iv = y_outer_train.iloc[inner_train_idx], y_outer_train.iloc[inner_val_idx]
                lgb = LGBMClassifier(objective='binary', is_unbalance=True,
                                      metric='auc', verbosity=-1, seed=2020)
                lgb.set_params(**params)
                lgb.fit(X_it, y_it)
                inner_aucs.append(roc_auc_score(y_iv, lgb.predict_proba(X_iv)[:, 1]))
            param_scores.append(np.mean(inner_aucs))

        best_idx = np.argmax(param_scores)
        best_params = param_combos[best_idx]
        best_params['n_jobs'] = 4
        all_best_params.append(best_params)
        print(f"  Best params (AUC={param_scores[best_idx]:.4f}): {best_params}")

        n_calib = int(len(X_outer_train) * 0.4)

        my_gbm = LGBMClassifier(objective='binary', is_unbalance=True,
                                 metric='auc', verbosity=-1, seed=2022)
        my_gbm.set_params(**{k: v for k, v in best_params.items() if k != 'n_jobs'})

        X_train_gbm = X_outer_train.iloc[n_calib:]
        y_train_gbm = y_outer_train.iloc[n_calib:]
        X_train_cali = X_outer_train.iloc[:n_calib]
        y_train_cali = y_outer_train.iloc[:n_calib]

        my_gbm.fit(X_train_gbm, y_train_gbm)
        raw_scores_cali = my_gbm.predict_proba(X_train_cali)[:, 1]

        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(raw_scores_cali, y_train_cali)

        raw_scores_test = my_gbm.predict_proba(X_outer_test)[:, 1]
        y_pred_prob = iso_reg.predict(raw_scores_test)
        y_pred_prob = np.clip(y_pred_prob, 0, 1)

        obsf, predf = calibration_curve(y_outer_test, y_pred_prob, n_bins=10, strategy='quantile')
        obs_array = np.concatenate([obs_array, extend(obsf, nb_points=10)], axis=1)
        pred_array = np.concatenate([pred_array, extend(predf, nb_points=10)], axis=1)

        results_cv.append(get_full_eval(y_outer_test, y_pred_prob, cutoff_list))
        y_pred_prob_lst.append(y_pred_prob)
        y_test_lst.append(np.array(y_outer_test))

        hl_p = hl_test_per_fold(y_outer_test.values, y_pred_prob)
        fold_auc = roc_auc_score(y_outer_test, y_pred_prob)
        fold_apr = average_precision_score(y_outer_test, y_pred_prob)
        fold_brier = brier_score_loss(y_outer_test, y_pred_prob)

        print(f"  Fold AUC={fold_auc:.4f}, APR={fold_apr:.4f}, Brier={fold_brier:.4f}, HL-p={hl_p:.4f}")

        all_fold_predictions.append({
            'fold': fold, 'auc': fold_auc, 'apr': fold_apr,
            'brier': fold_brier, 'hl_pvalue': hl_p,
            'best_params': best_params,
        })

    # Save predictions
    pd.DataFrame(y_pred_prob_lst).T.to_csv(os.path.join(results_dir, 'pred_prob_cv_df.csv'), index=False)
    pd.DataFrame(y_test_lst).T.to_csv(os.path.join(results_dir, 'test_cv_df.csv'), index=False)

    final_output = avg_results(results_cv)
    final_output.to_csv(os.path.join(results_dir, 's05_final_model.csv'), index=False)

    pd.DataFrame(all_best_params).to_csv(os.path.join(results_dir, 's05_best_params_per_fold.csv'), index=False)

    obs_mean = np.round(np.mean(obs_array[:, 1:], axis=1), 4)
    pred_mean = np.round(np.mean(pred_array[:, 1:], axis=1), 4)
    pd.DataFrame({'Observed': obs_mean, 'Predicted': pred_mean}).to_csv(
        os.path.join(results_dir, 's05_calibration.csv'), index=False)

    all_y_true = np.concatenate([np.array(yl) for yl in y_test_lst])
    all_y_pred = np.concatenate([np.array(yp) for yp in y_pred_prob_lst])
    overall_hl_p = hl_test_per_fold(all_y_true, all_y_pred)
    overall_brier = brier_score_loss(all_y_true, all_y_pred)

    print(f"\n  {'='*60}")
    print(f"  FINAL RESULTS: {target_name}")
    print(f"  {'='*60}")
    auc_row = final_output.iloc[0]
    print(f"  AUC        = {auc_row['AUC']:.4f} ± {auc_row.get('AUC_std', 0):.4f}")
    print(f"  Brier      = {overall_brier:.4f}")
    print(f"  HL p-value = {overall_hl_p:.4f}")
    print(f"  Features: {my_f_list}")

    best_youden_idx = final_output['Youden'].idxmax()
    best_row = final_output.iloc[best_youden_idx]
    print(f"\n  Best cutoff (Youden): {best_row['Cutoff']:.3f}")
    print(f"    Sens={best_row['Sens']:.4f}, Spec={best_row['Spec']:.4f}, Youden={best_row['Youden']:.4f}")
    print(f"\n  Total time: {time.time()-t0:.0f}s")

    metrics = {
        'target': target_name,
        'auc_mean': float(auc_row['AUC']),
        'auc_std': float(auc_row.get('AUC_std', 0)),
        'brier_score': float(overall_brier),
        'hl_pvalue': float(overall_hl_p),
        'best_cutoff': float(best_row['Cutoff']),
        'sensitivity': float(best_row['Sens']),
        'specificity': float(best_row['Spec']),
        'youden': float(best_row['Youden']),
        'features': my_f_list,
        'fold_results': all_fold_predictions,
    }
    with open(os.path.join(results_dir, 's05_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, default=str)

    return metrics


# ============================================================================
# SHAP & CALIBRATION PLOTS
# ============================================================================

def run_shap(mydf, X_all, y, features, results_dir, target_name):
    print(f"\n{'='*70}")
    print(f"[SHAP] SHAP analysis for {target_name}")
    print(f"{'='*70}")
    try:
        import shap
    except ImportError:
        print("  shap not installed, skipping")
        return

    t0 = time.time()
    my_f = features[:TOP_N_S04]
    X = X_all[my_f]
    X_shap = X.copy()
    X_shap.columns = my_f

    mykf = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=True)
    for train_idx, test_idx in mykf.split(X_shap, y):
        X_train, y_train = X_shap.iloc[train_idx, :], y.iloc[train_idx]
        X_test = X_shap.iloc[test_idx, :]
        break

    my_lgb = LGBMClassifier(objective='binary', metric='auc',
                             is_unbalance=True, verbosity=-1, seed=2020)
    my_lgb.set_params(**BASE_PARAMS)
    my_lgb.fit(X_train, y_train)

    explainer = shap.Explainer(my_lgb)
    shap_values = explainer(X_test)

    plt.figure()
    shap.plots.beeswarm(shap_values[:, :, 1], show=False,
                         order=list(range(min(TOP_N_S04, len(my_f)))))
    plt.gcf().set_size_inches(18, 5.5)
    ax = plt.gca()
    ax.set_ylabel('Selected Predictors', fontsize=20, weight='bold')
    ax.set_xlabel('SHAP Values', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'shap_beeswarm.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Done in {time.time()-t0:.0f}s.")


def plot_calibration(results_dir, target_name):
    calib_path = os.path.join(results_dir, 's05_calibration.csv')
    if not os.path.exists(calib_path):
        return
    calib_df = pd.read_csv(calib_path)
    obs_mean = calib_df['Observed'].values
    pred_mean = calib_df['Predicted'].values

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    ax.plot(pred_mean, obs_mean, 'ro-', label='Model')
    ax.set_xlabel('Predicted Probability', fontsize=14)
    ax.set_ylabel('Observed Probability', fontsize=14)
    ax.set_title(f'{target_name} Calibration', fontsize=16, weight='bold')
    ax.legend(fontsize=12)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'calibration_plot.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(use_deploy_strategy=True, exclude_stroke=True,
                 imaging_subset_only=False):
    print(f"{'='*70}")
    print(f"UKB-DRP Training with Brain MRI Imaging Features")
    print(f"{'='*70}")
    print(f"  Deploy strategy: {use_deploy_strategy}")
    print(f"  Stroke exclusion: {exclude_stroke}")
    print(f"  Imaging subset only: {imaging_subset_only}")
    print(f"  Data: {PREPROCESSED_CSV}")
    print()

    # Load
    print("[Load] Loading combined clinical + imaging data...")
    mydf, rm_f, imaging_cols = load_and_prepare_data(imaging_subset_only)

    if exclude_stroke:
        mydf = exclude_baseline_stroke(mydf)

    print(f"  Sample: {len(mydf):,} participants")
    print(f"  Dementia positive: {mydf['dementia_status'].sum():,}")
    print(f"  AD positive: {mydf['AD_status'].sum():,}")
    if 'has_brain_mri' in mydf.columns:
        print(f"  Has brain MRI: {mydf['has_brain_mri'].sum():,}")

    X_all = mydf.drop(columns=rm_f)
    print(f"  Total features: {X_all.shape[1]}")
    n_img = len([c for c in X_all.columns if c in imaging_cols])
    print(f"  Clinical: {X_all.shape[1] - n_img}, Imaging: {n_img}")

    # Determine targets
    if use_deploy_strategy:
        primary_targets = ['DM_full']
        deploy_targets = ['DM_10yrs', 'DM_5yrs', 'AD_full', 'AD_10yrs', 'AD_5yrs']
    else:
        primary_targets = list(ALL_TARGETS.keys())
        deploy_targets = []

    primary_features = None
    all_metrics = {}

    for target_name in primary_targets:
        target_col, years_col, max_years = ALL_TARGETS[target_name]
        y = apply_time_window(mydf, target_col, years_col, max_years)

        results_dir = os.path.join(RESULTS_BASE, target_name)
        os.makedirs(results_dir, exist_ok=True)

        print(f"\n{'#'*70}")
        print(f"# PRIMARY MODEL: {target_name} (n_pos={y.sum()})")
        print(f"{'#'*70}")

        s01_df = run_s01(mydf, X_all, y, results_dir, imaging_cols)
        s02_df = run_s02(mydf, X_all, y, s01_df, results_dir)
        s03_df = run_s03(mydf, X_all, y, s02_df, results_dir)
        s04_df = run_s04_sfs(mydf, X_all, y, s03_df, results_dir)
        metrics = run_s05(mydf, X_all, y, s04_df, results_dir, target_name)
        all_metrics[target_name] = metrics

        primary_features = s04_df['Features'][:TOP_N_S04].tolist()
        run_shap(mydf, X_all, y, primary_features, results_dir, target_name)
        plot_calibration(results_dir, target_name)
        gc.collect()

    # Deploy to other targets
    if deploy_targets and primary_features:
        for deploy_name in deploy_targets:
            deploy_dir = os.path.join(RESULTS_BASE, deploy_name)
            os.makedirs(deploy_dir, exist_ok=True)
            target_col, years_col, max_years = ALL_TARGETS[deploy_name]
            y_deploy = apply_time_window(mydf, target_col, years_col, max_years)
            metrics = run_s05(mydf, X_all, y_deploy,
                            pd.DataFrame({'Features': primary_features}),
                            deploy_dir, deploy_name)
            all_metrics[deploy_name] = metrics
            run_shap(mydf, X_all, y_deploy, primary_features, deploy_dir, deploy_name)
            gc.collect()

    # Summary
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Model':<15} {'AUC':<12} {'HL-p':<10} {'Sens':<10} {'Spec':<10}")
    print("-"*57)
    for name, m in sorted(all_metrics.items()):
        print(f"{name:<15} {m.get('auc_mean', 0):.4f}±{m.get('auc_std', 0):.4f}   "
              f"{m.get('hl_pvalue', 0):.4f}     "
              f"{m.get('sensitivity', 0):.4f}   {m.get('specificity', 0):.4f}")

    summary = []
    for name, m in sorted(all_metrics.items()):
        summary.append({
            'model': name, 'auc_mean': m.get('auc_mean'), 'auc_std': m.get('auc_std'),
            'hl_pvalue': m.get('hl_pvalue'), 'sensitivity': m.get('sensitivity'),
            'specificity': m.get('specificity'), 'youden': m.get('youden'),
            'best_cutoff': m.get('best_cutoff'),
        })
    pd.DataFrame(summary).to_csv(os.path.join(RESULTS_BASE, 'summary_metrics.csv'), index=False)
    print(f"\nResults saved to: {RESULTS_BASE}")
    return all_metrics


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='UKB-DRP Training with Brain MRI Imaging')
    parser.add_argument('--no-deploy', action='store_true',
                        help='Train each target independently')
    parser.add_argument('--no-stroke-exclusion', action='store_true',
                        help='Do not exclude baseline stroke')
    parser.add_argument('--target', type=str, default=None,
                        help='Train single target only (e.g., DM_full)')
    parser.add_argument('--imaging-subset', action='store_true',
                        help='Train only on participants with brain MRI data')
    parser.add_argument('--n-combos', type=int, default=N_PARAM_COMBOS,
                        help=f'Number of HP combos (default: {N_PARAM_COMBOS})')
    args = parser.parse_args()

    if args.n_combos != N_PARAM_COMBOS:
        N_PARAM_COMBOS = args.n_combos

    if args.target:
        mydf, rm_f, imaging_cols = load_and_prepare_data(args.imaging_subset)
        if not args.no_stroke_exclusion:
            mydf = exclude_baseline_stroke(mydf)
        X_all = mydf.drop(columns=rm_f)

        target_col, years_col, max_years = ALL_TARGETS[args.target]
        y = apply_time_window(mydf, target_col, years_col, max_years)

        mode_tag = '_img_subset' if args.imaging_subset else '_img_full'
        results_dir = os.path.join(RESULTS_BASE, args.target + mode_tag)
        os.makedirs(results_dir, exist_ok=True)

        print(f"\n{'#'*70}")
        print(f"# SINGLE TARGET: {args.target} (n_pos={y.sum()})")
        print(f"{'#'*70}")

        s01_df = run_s01(mydf, X_all, y, results_dir, imaging_cols)
        s02_df = run_s02(mydf, X_all, y, s01_df, results_dir)
        s03_df = run_s03(mydf, X_all, y, s02_df, results_dir)
        s04_df = run_s04_sfs(mydf, X_all, y, s03_df, results_dir)
        metrics = run_s05(mydf, X_all, y, s04_df, results_dir, args.target)

        features = s04_df['Features'][:TOP_N_S04].tolist()
        run_shap(mydf, X_all, y, features, results_dir, args.target)
        plot_calibration(results_dir, args.target)

    else:
        run_pipeline(
            use_deploy_strategy=not args.no_deploy,
            exclude_stroke=not args.no_stroke_exclusion,
            imaging_subset_only=args.imaging_subset,
        )

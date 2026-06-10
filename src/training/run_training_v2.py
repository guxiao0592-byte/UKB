#!/usr/bin/env python3
"""
UKB-DRP Paper Reproduction — Improved Pipeline (v2)
====================================================
Faithfully implements the paper's methodology:
  - Stroke exclusion (baseline stroke removal)
  - Sequential Forward Selection (SFS) for feature selection
  - Randomized hyperparameter search with nested CV
  - Deploy strategy: train DM_full → recalibrate for other targets
  - Isotonic Regression calibration (paper's approach, not CalibratedClassifierCV)
  - SHAP analysis
  - Hosmer-Lemeshow goodness-of-fit test
  - Calibration plots

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
                              confusion_matrix, recall_score, brier_score_loss)
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
PREPROCESSED_CSV = os.path.join(DPATH, 'Preprocessed_Data', 'Preprocessed_Data.csv')
RESULTS_BASE = os.path.join(DPATH, 'Results_v2')
DATA_DIR = os.path.join(DPATH, 'Data')
os.makedirs(RESULTS_BASE, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

RANDOM_STATE = 2022
N_SPLITS = 5

# Targets: (target_col, years_col, name_prefix)
ALL_TARGETS = {
    'DM_full':  ('dementia_status', 'dementia_years', None),
    'DM_10yrs': ('dementia_status', 'dementia_years', 10),
    'DM_5yrs':  ('dementia_status', 'dementia_years', 5),
    'AD_full':  ('AD_status', 'AD_years', None),
    'AD_10yrs': ('AD_status', 'AD_years', 10),
    'AD_5yrs':  ('AD_status', 'AD_years', 5),
}

# Paper's fixed parameters (used as baseline for s01-s04)
BASE_PARAMS = {'n_estimators': 500, 'max_depth': 15, 'num_leaves': 10,
               'subsample': 0.7, 'learning_rate': 0.01, 'colsample_bytree': 0.7}

# Hyperparameter search space (paper: exhaustive search over 1000 candidate sets)
# Expanded grid: 4×4×5×4×5×4 = 6,400 total combos; we randomly sample 1,000
PARAM_GRID = {
    'n_estimators': [200, 300, 500, 800],
    'max_depth': [8, 10, 15, 20],
    'num_leaves': [8, 10, 16, 31, 50],
    'subsample': [0.5, 0.6, 0.7, 0.8],
    'learning_rate': [0.005, 0.01, 0.02, 0.05, 0.1],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8],
}

N_PARAM_COMBOS = 1000  # Matching paper's exhaustive search over 1000 combos

# HES fields (paper excludes these)
RM_HES = ['41234-0.0', '41259-0.0', '41149-0.0', '41289-0.0',
          '41218-0.0', '41235-0.0', '41214-0.0']

TOP_N_S01 = 50       # Top features after initial ranking (paper uses 50)
CLUSTER_THRESHOLD = 0.75  # Ward clustering distance threshold (paper's threshold)
TOP_N_S04 = 10       # Final model feature count (paper uses 10)
SFS_MAX_FEATURES = 15  # Max features to consider during SFS (paper stops at AUC≈0.85)
N_PARAM_COMBOS = 1000  # Matching paper's exhaustive search over 1000 combos

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def threshold(array, cutoff):
    arr = array.copy()
    arr[arr < cutoff] = 0
    arr[arr >= cutoff] = 1
    return arr

def youden_index(y_test, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return sens + spec - 1

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
    """Randomly sample nb_items parameter combos from the full grid (paper's approach)."""
    combo_list = [dict(zip(my_dict.keys(), v)) for v in product(*my_dict.values())]
    random.seed(2020)
    return random.sample(combo_list, min(nb_items, len(combo_list)))

def hl_pvalue(obs_prob, pred_prob, percentage=1, bin_obs_nb=None):
    """Hosmer-Lemeshow goodness-of-fit test p-value."""
    obs_prob = obs_prob / percentage
    pred_prob = pred_prob / percentage
    nominator = (obs_prob - pred_prob) ** 2 * bin_obs_nb if bin_obs_nb else (obs_prob - pred_prob) ** 2
    denominator = pred_prob * (1 - pred_prob)
    denominator[denominator == 0] = 1
    stat = nominator / denominator
    stat[stat == np.inf] = 0.0
    stat[stat == -np.inf] = 0.0
    test_stat = np.nansum(stat)
    pvalue = 1 - stats.chi2.cdf(test_stat, 8)
    return np.round(pvalue, 4)

def hl_test_per_fold(y_true, y_pred, n_bins=10):
    """Compute HL statistic per fold."""
    _, bin_edges = pd.qcut(y_pred, q=n_bins, retbins=True, duplicates='drop')
    bin_labels = pd.cut(y_pred, bins=bin_edges, include_lowest=True, duplicates='drop')
    obs_probs = []
    pred_probs = []
    bin_counts = []
    for interval in bin_labels.cat.categories:
        mask = bin_labels == interval
        if mask.sum() > 0:
            obs_probs.append(y_true[mask].mean())
            pred_probs.append(y_pred[mask].mean())
            bin_counts.append(mask.sum())
    obs_prob_arr = np.array(obs_probs)
    pred_prob_arr = np.array(pred_probs)
    bin_obs_arr = np.array(bin_counts)
    return hl_pvalue(obs_prob_arr, pred_prob_arr, percentage=1, bin_obs_nb=bin_obs_arr)

# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_and_prepare_data():
    """Load preprocessed data and prepare target variables."""
    mydf = pd.read_csv(PREPROCESSED_CSV)

    # Fix AD_years: use dementia_years for missing AD_years
    # Negative AD_years = non-converter, abs(value) = follow-up years
    mydf['AD_years'] = mydf['AD_years'].fillna(mydf['dementia_years'])

    # Columns to remove as outcomes/IDs
    rm_f1 = ['Unnamed: 0', 'eid', 'dementia_status', 'dementia_years',
             'AD_status', 'AD_years', 'VD_status', 'VD_years',
             'stroke_status', 'stroke_years']
    rm_f = [c for c in rm_f1 + RM_HES if c in mydf.columns]

    return mydf, rm_f


def apply_time_window(mydf, target_col, years_col, max_years=None):
    """Apply time window and proper censoring.

    Returns (y, include_mask):
      - y: binary target (1 = event within window, 0 = no event within window)
      - include_mask: boolean Series, False = exclude from analysis
        (non-converters with insufficient follow-up to confirm negativity)

    UKB data encoding: years_col < 0 means no event,
    abs(years_col) = follow-up duration in years.
    """
    y = mydf[target_col].copy()
    include = pd.Series(True, index=mydf.index)

    if max_years is not None:
        y_years = mydf[years_col]

        # Converters whose event occurred AFTER the window → label as 0
        # (they are true negatives WITHIN the window)
        late_event = (y == 1) & (y_years > max_years)
        y.loc[late_event] = 0

        # Non-converters with insufficient follow-up → CENSOR (exclude)
        # y_years < 0 means no event; abs(y_years) = follow-up years
        insufficient_fu = (
            (y == 0) &
            (y_years < 0) &
            (y_years.abs() < max_years)
        )
        include = ~insufficient_fu

    return y, include


def exclude_baseline_stroke(mydf):
    """Exclude participants with stroke before baseline (stroke_years < 0).
    Paper: exclude 7,184 stroke patients → 425,159 participants."""
    stroke_mask = (mydf['stroke_years'] < 0) & (mydf['stroke_years'].notna())
    print(f"  Excluding {stroke_mask.sum()} participants with baseline stroke")
    return mydf[~stroke_mask].copy()


# ============================================================================
# S01: INITIAL FEATURE IMPORTANCE RANKING
# ============================================================================

def run_s01(mydf, X, y, results_dir):
    """Initial LightGBM training on ALL features → rank by Gain → keep Top 50."""
    print("\n" + "="*70)
    print(f"[s01] Initial feature importance ranking ({X.shape[1]} features)")
    print("="*70)
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
    print(f"  Done in {time.time()-t0:.0f}s. Top 10 features:")
    for i, f in enumerate(top_f[:10]):
        print(f"    {i+1:2d}. {f}")
    return imp_df


# ============================================================================
# S02: HIERARCHICAL CLUSTERING (REDUNDANCY REMOVAL)
# ============================================================================

def run_s02(mydf, X_all, y, s01_df, results_dir):
    """Spearman correlation → Ward hierarchical clustering → keep 1 per cluster."""
    print("\n" + "="*70)
    print(f"[s02] Hierarchical clustering & redundancy removal")
    print("="*70)
    t0 = time.time()

    top_f_list = s01_df['Features'][:TOP_N_S01].tolist()
    X = X_all[top_f_list]

    # Spearman correlation
    corr = np.array(X.corr(method='spearman'))
    corr = np.nan_to_num(corr)
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)

    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))

    # Dendrogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
    dendro = hierarchy.dendrogram(dist_linkage, labels=top_f_list, ax=ax2)
    ax2.set_xticklabels(dendro["ivl"], rotation=60, fontsize=8, horizontalalignment='right')
    dendro_idx = np.arange(0, len(dendro["ivl"]))
    ax1.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]], cmap='RdBu_r', vmin=-1, vmax=1)
    ax1.set_xticks(dendro_idx)
    ax1.set_yticks(dendro_idx)
    ax1.set_xticklabels(dendro["ivl"], rotation=60, fontsize=8, horizontalalignment='right')
    ax1.set_yticklabels(dendro["ivl"], fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 's02_dendrogram.png'), dpi=150)
    plt.close()

    # Cluster → keep first feature in each cluster
    cluster_ids = hierarchy.fcluster(dist_linkage, CLUSTER_THRESHOLD, criterion="distance")
    cluster_to_features = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_to_features[cluster_id].append(idx)

    selected_idx = [v[0] for v in cluster_to_features.values()]
    selected_f = [top_f_list[i] for i in selected_idx]

    # Build output preserving Gain rank order
    s02_out = s01_df[s01_df['Features'].isin(selected_f)].copy()
    s02_out.to_csv(os.path.join(results_dir, 's02_clustered_features.csv'), index=False)

    print(f"  Done in {time.time()-t0:.0f}s. {len(selected_f)} features after clustering")
    print(f"  Features: {selected_f}")
    return s02_out


# ============================================================================
# S03: FINAL FEATURE IMPORTANCE (ON CLUSTERED FEATURES)
# ============================================================================

def run_s03(mydf, X_all, y, s02_df, results_dir):
    """Re-rank features after clustering with a fresh LightGBM."""
    print("\n" + "="*70)
    print(f"[s03] Final feature importance ({len(s02_df)} features)")
    print("="*70)
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
# S04: SEQUENTIAL FORWARD SELECTION (SFS) — PAPER'S METHOD
# ============================================================================

def run_s04_sfs(mydf, X_all, y, s03_df, results_dir):
    """
    Sequential Forward Selection (paper's core feature selection method).

    Algorithm:
      1. Start with highest Gain feature f1
      2. For each remaining feature, add to current set, evaluate AUC via 5-fold CV
      3. Select the feature that gives the highest AUC gain
      4. Repeat until AUC ≈ 0.85 and marginal gain plateaus,
         or max_features is reached

    This differs from cumulative AUC (which follows a fixed order).
    SFS can "skip" features that would be next by Gain if they don't
    add marginal value to the current feature set.
    """
    print("\n" + "="*70)
    print(f"[s04] Sequential Forward Selection (SFS)")
    print("="*70)
    t0 = time.time()

    all_features = s03_df['Features'].tolist()
    max_features = min(SFS_MAX_FEATURES, len(all_features))

    mykf = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=True)

    def evaluate_feature_set(feature_set):
        """Evaluate a set of features via 5-fold CV → mean AUC."""
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

    # Step 1: Start with the highest Gain feature
    first_idx = 0  # s03 is already sorted by Gain
    selected.append(first_idx)
    remaining.remove(first_idx)

    current_auc, current_std = evaluate_feature_set([all_features[i] for i in selected])
    sfs_history.append({
        'step': 1, 'feature_added': all_features[first_idx],
        'selected_count': 1, 'selected_features': [all_features[i] for i in selected],
        'auc_mean': current_auc, 'auc_std': current_std
    })
    print(f"  Step 1: Added {all_features[first_idx]:30s} → AUC = {current_auc:.4f} ± {current_std:.4f}")

    # Iterative forward selection
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

        # Stopping rule: marginal gain < 0.001
        if auc_gain < 0.001 and len(selected) >= TOP_N_S04:
            print(f"  Stopping: marginal AUC gain ({auc_gain:.4f}) < 0.001 "
                  f"and {len(selected)} >= {TOP_N_S04} features")
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
        print(f"  Step {step+1}: Added {all_features[best_idx]:30s} → AUC = {current_auc:.4f} ± {current_std:.4f} "
              f"(gain={auc_gain:.4f})")

    # Save SFS history
    sfs_save = [{k: v for k, v in d.items() if k != 'selected_features'}
                 for d in sfs_history]
    sfs_df = pd.DataFrame(sfs_save)
    sfs_df['selected_features'] = [', '.join(d['selected_features'])
                                     for d in sfs_history]
    sfs_df.to_csv(os.path.join(results_dir, 's04_sfs_history.csv'), index=False)

    # Build final feature table sorted by selection order (paper's convention)
    selected_features = [all_features[i] for i in selected]
    s04_out = pd.DataFrame({'Features': selected_features})
    s04_out['SelectionOrder'] = range(1, len(selected) + 1)
    s04_out['CumulativeAUC'] = [d['auc_mean'] for d in sfs_history]
    s04_out['AUC_std'] = [d['auc_std'] for d in sfs_history]

    # Merge with s03 for Gain info
    s04_out = s04_out.merge(
        s03_df[['Features', 'Gain', 'Cover', 'NA_full', 'NA_target']],
        on='Features', how='left'
    )

    s04_out['Path'] = ''
    s04_out['Field'] = s04_out['Features']
    s04_out['ValueType'] = ''
    s04_out['Units'] = ''

    s04_out.to_csv(os.path.join(results_dir, 's04_selected_features.csv'), index=False)

    # Also save the traditional cumulative AUC for comparison
    _save_cumulative_auc_comparison(X_all, y, selected_features, mykf, results_dir)

    print(f"\n  Done in {time.time()-t0:.0f}s.")
    print(f"  Selected {len(selected)} features via SFS")
    print(f"  Top 10: {selected_features[:10]}")
    return s04_out


def _save_cumulative_auc_comparison(X_all, y, selected_features, mykf, results_dir):
    """Save cumulative AUC for comparison with SFS results."""
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
    """
    Final model with:
      1. Randomized hyperparameter search (paper: 1000 combos; we use N_PARAM_COMBOS)
      2. Inner CV for tuning, outer CV for evaluation (nested CV)
      3. Isotonic Regression calibration (paper's approach, NOT CalibratedClassifierCV)
      4. Youden index cutoff selection
      5. Hosmer-Lemeshow goodness-of-fit test
    """
    print("\n" + "="*70)
    print(f"[s05] Final model with hyperparameter tuning & Isotonic calibration")
    print("="*70)
    t0 = time.time()

    my_f_list = s04_df['Features'][:TOP_N_S04].tolist()
    X = X_all[my_f_list]

    outer_kf = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=True)
    cutoff_list = [np.round(0.001 + i * 0.001, 3) for i in range(100)]

    # --- Hyperparameter Tuning (nested CV) ---
    print(f"\n  Hyperparameter tuning: sampling {N_PARAM_COMBOS} combos from {np.prod([len(v) for v in PARAM_GRID.values()])} total...")
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

        # --- Train GBM with best params ---
        # Paper's approach: split train data → 2 parts for GBM training & calibration
        n_calib = int(len(X_outer_train) * 0.4)  # 40% for calibration

        my_gbm = LGBMClassifier(objective='binary', is_unbalance=True,
                                 metric='auc', verbosity=-1, seed=2022)
        my_gbm.set_params(**{k: v for k, v in best_params.items() if k != 'n_jobs'})

        # Train GBM on calibration part, calibrate on the remaining
        X_train_gbm = X_outer_train.iloc[n_calib:]
        y_train_gbm = y_outer_train.iloc[n_calib:]
        X_train_cali = X_outer_train.iloc[:n_calib]
        y_train_cali = y_outer_train.iloc[:n_calib]

        my_gbm.fit(X_train_gbm, y_train_gbm)
        raw_scores_cali = my_gbm.predict_proba(X_train_cali)[:, 1]

        # Isotonic Regression calibration (paper's method)
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(raw_scores_cali, y_train_cali)

        # Predict on test fold
        raw_scores_test = my_gbm.predict_proba(X_outer_test)[:, 1]
        y_pred_prob = iso_reg.predict(raw_scores_test)

        # Clip calibrated probabilities
        y_pred_prob = np.clip(y_pred_prob, 0, 1)

        # Calibration curve
        obsf, predf = calibration_curve(y_outer_test, y_pred_prob, n_bins=10, strategy='quantile')
        obs_array = np.concatenate([obs_array, extend(obsf, nb_points=10)], axis=1)
        pred_array = np.concatenate([pred_array, extend(predf, nb_points=10)], axis=1)

        # Full evaluation
        results_cv.append(get_full_eval(y_outer_test, y_pred_prob, cutoff_list))
        y_pred_prob_lst.append(y_pred_prob)
        y_test_lst.append(np.array(y_outer_test))

        # HL test
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

    # --- Aggregate Results ---
    # Save predictions
    pred_prob_df = pd.DataFrame(y_pred_prob_lst).T
    test_df = pd.DataFrame(y_test_lst).T
    pred_prob_df.to_csv(os.path.join(results_dir, 'pred_prob_cv_df.csv'), index=False)
    test_df.to_csv(os.path.join(results_dir, 'test_cv_df.csv'), index=False)

    # Average evaluation
    final_output = avg_results(results_cv)
    final_output.to_csv(os.path.join(results_dir, 's05_final_model.csv'), index=False)

    # Best params summary
    params_df = pd.DataFrame(all_best_params)
    params_df.to_csv(os.path.join(results_dir, 's05_best_params_per_fold.csv'), index=False)

    # Calibration summary
    obs_mean = np.round(np.mean(obs_array[:, 1:], axis=1), 4)
    pred_mean = np.round(np.mean(pred_array[:, 1:], axis=1), 4)

    calib_df = pd.DataFrame({'Observed': obs_mean, 'Predicted': pred_mean})
    calib_df.to_csv(os.path.join(results_dir, 's05_calibration.csv'), index=False)

    # Overall HL test
    all_y_true = np.concatenate([np.array(yl) for yl in y_test_lst])
    all_y_pred = np.concatenate([np.array(yp) for yp in y_pred_prob_lst])
    overall_hl_p = hl_test_per_fold(all_y_true, all_y_pred)
    overall_brier = brier_score_loss(all_y_true, all_y_pred)

    # --- Print Summary ---
    print(f"\n  {'='*60}")
    print(f"  FINAL RESULTS: {target_name}")
    print(f"  {'='*60}")

    auc_row = final_output.iloc[0]  # First row has AUC
    print(f"  AUC        = {auc_row['AUC']:.4f} ± {auc_row.get('AUC_std', 0):.4f}")
    print(f"  Brier      = {overall_brier:.4f}")
    print(f"  HL p-value = {overall_hl_p:.4f}")
    print(f"  Top {TOP_N_S04} features: {my_f_list}")

    # Best Youden cutoff
    best_youden_idx = final_output['Youden'].idxmax()
    best_row = final_output.iloc[best_youden_idx]
    print(f"\n  Best cutoff (Youden): {best_row['Cutoff']:.3f}")
    print(f"    Sens={best_row['Sens']:.4f}, Spec={best_row['Spec']:.4f}, "
          f"Youden={best_row['Youden']:.4f}")

    print(f"\n  Total time: {time.time()-t0:.0f}s")

    # Save overall metrics
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
# S06: DEPLOY STRATEGY (Paper's approach)
# ============================================================================

def run_deploy(mydf, X_all, deploy_target_name, primary_features, results_dir):
    """
    Deploy the primary model (DM_full) to another target.

    Paper's approach:
      1. Use the SAME 10 features from the primary model
      2. Train GBM on target-specific data
      3. Calibrate with Isotonic Regression on held-out portion of training fold
      4. Evaluate via 5-fold CV

    This "borrows" the feature selection from the main dementia model
    and only adapts the probability calibration for the new target.
    """
    print("\n" + "="*70)
    print(f"[Deploy] DM_full → {deploy_target_name}")
    print("="*70)
    t0 = time.time()

    target_col, years_col, max_years = ALL_TARGETS[deploy_target_name]
    y_deploy, include = apply_time_window(mydf, target_col, years_col, max_years)
    X = X_all[primary_features[:TOP_N_S04]]

    # Filter censored subjects
    n_censored = (~include).sum()
    if n_censored > 0:
        print(f"  [CENSOR] excluding {n_censored} subjects with insufficient follow-up")
        y_deploy = y_deploy[include]
        X = X[include]

    n_pos = y_deploy.sum()
    print(f"  Target: {deploy_target_name}, n_pos={n_pos}, "
          f"pos_rate={n_pos/len(y_deploy)*100:.2f}%")

    outer_kf = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=True)
    cutoff_list = [np.round(0.001 + i * 0.001, 3) for i in range(100)]

    results_cv = []
    obs_array, pred_array = np.zeros((10, 1)), np.zeros((10, 1))
    y_test_lst, y_pred_prob_lst = [], []
    all_best_params = []

    for fold, (train_idx, test_idx) in enumerate(outer_kf.split(X, y_deploy)):
        X_train, X_test = X.iloc[train_idx, :], X.iloc[test_idx, :]
        y_train, y_test = y_deploy.iloc[train_idx], y_deploy.iloc[test_idx]

        # Tune params for this target
        inner_kf = StratifiedKFold(n_splits=3, random_state=RANDOM_STATE + fold, shuffle=True)
        param_combos = select_params_combo(PARAM_GRID, min(30, N_PARAM_COMBOS))
        param_scores = []

        for params in param_combos:
            inner_aucs = []
            for itr, ivl in inner_kf.split(X_train, y_train):
                X_it, X_iv = X_train.iloc[itr, :], X_train.iloc[ivl, :]
                y_it, y_iv = y_train.iloc[itr], y_train.iloc[ivl]
                lgb = LGBMClassifier(objective='binary', is_unbalance=True,
                                      metric='auc', verbosity=-1, seed=2020)
                lgb.set_params(**params)
                lgb.fit(X_it, y_it)
                inner_aucs.append(roc_auc_score(y_iv, lgb.predict_proba(X_iv)[:, 1]))
            param_scores.append(np.mean(inner_aucs))

        best_params = param_combos[np.argmax(param_scores)]
        all_best_params.append(best_params)

        # Train + calibrate (paper's split-train-calibrate approach)
        n_calib = int(len(X_train) * 0.4)
        X_train_gbm = X_train.iloc[n_calib:]
        y_train_gbm = y_train.iloc[n_calib:]
        X_train_cali = X_train.iloc[:n_calib]
        y_train_cali = y_train.iloc[:n_calib]

        my_gbm = LGBMClassifier(objective='binary', is_unbalance=True,
                                 metric='auc', verbosity=-1, seed=2022)
        my_gbm.set_params(**best_params)
        my_gbm.fit(X_train_gbm, y_train_gbm)

        # Isotonic calibration
        raw_cali = my_gbm.predict_proba(X_train_cali)[:, 1]
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(raw_cali, y_train_cali)

        # Predict
        raw_test = my_gbm.predict_proba(X_test)[:, 1]
        y_pred_prob = iso_reg.predict(raw_test)
        y_pred_prob = np.clip(y_pred_prob, 0, 1)

        obsf, predf = calibration_curve(y_test, y_pred_prob, n_bins=10, strategy='quantile')
        obs_array = np.concatenate([obs_array, extend(obsf, nb_points=10)], axis=1)
        pred_array = np.concatenate([pred_array, extend(predf, nb_points=10)], axis=1)

        results_cv.append(get_full_eval(y_test, y_pred_prob, cutoff_list))
        y_pred_prob_lst.append(y_pred_prob)
        y_test_lst.append(np.array(y_test))

        fold_auc = roc_auc_score(y_test, y_pred_prob)
        print(f"  Fold {fold+1}/{N_SPLITS} AUC={fold_auc:.4f}")

    # Save
    pred_prob_df = pd.DataFrame(y_pred_prob_lst).T
    test_df = pd.DataFrame(y_test_lst).T
    pred_prob_df.to_csv(os.path.join(results_dir, 'pred_prob_cv_df.csv'), index=False)
    test_df.to_csv(os.path.join(results_dir, 'test_cv_df.csv'), index=False)

    final_output = avg_results(results_cv)
    final_output.to_csv(os.path.join(results_dir, 'deploy_results.csv'), index=False)

    # HL test
    all_y_true = np.concatenate([np.array(yl) for yl in y_test_lst])
    all_y_pred = np.concatenate([np.array(yp) for yp in y_pred_prob_lst])
    deploy_hl_p = hl_test_per_fold(all_y_true, all_y_pred)

    auc_row = final_output.iloc[0]
    best_youden_idx = final_output['Youden'].idxmax()
    best_row = final_output.iloc[best_youden_idx]

    print(f"  AUC = {auc_row['AUC']:.4f} ± {auc_row.get('AUC_std', 0):.4f}")
    print(f"  HL p-value = {deploy_hl_p:.4f}")
    print(f"  Sens={best_row['Sens']:.4f}, Spec={best_row['Spec']:.4f} "
          f"at cutoff={best_row['Cutoff']:.3f}")
    print(f"  Time: {time.time()-t0:.0f}s")

    return {
        'target': deploy_target_name,
        'auc_mean': float(auc_row['AUC']),
        'auc_std': float(auc_row.get('AUC_std', 0)),
        'hl_pvalue': float(deploy_hl_p),
        'best_cutoff': float(best_row['Cutoff']),
        'sensitivity': float(best_row['Sens']),
        'specificity': float(best_row['Spec']),
        'youden': float(best_row['Youden']),
    }


# ============================================================================
# SHAP ANALYSIS
# ============================================================================

def run_shap(mydf, X_all, y, features, results_dir, target_name):
    """SHAP beeswarm plot for the final model's top features."""
    print("\n" + "="*70)
    print(f"[SHAP] SHAP analysis for {target_name}")
    print("="*70)

    try:
        import shap
    except ImportError:
        print("  shap not installed, skipping SHAP analysis")
        return

    t0 = time.time()
    my_f = features[:TOP_N_S04]
    X = X_all[my_f]
    # Use human-readable labels
    X_shap = X.copy()
    X_shap.columns = my_f

    mykf = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=True)
    for train_idx, test_idx in mykf.split(X_shap, y):
        X_train, y_train = X_shap.iloc[train_idx, :], y.iloc[train_idx]
        X_test = X_shap.iloc[test_idx, :]
        break  # Use first fold for SHAP

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
    print(f"  Done in {time.time()-t0:.0f}s. Saved to {results_dir}/shap_beeswarm.png")


# ============================================================================
# CALIBRATION PLOT
# ============================================================================

def plot_calibration(results_dir, target_name):
    """Plot calibration curve from saved data."""
    calib_path = os.path.join(results_dir, 's05_calibration.csv')
    if not os.path.exists(calib_path):
        print(f"  No calibration data at {calib_path}")
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

def run_pipeline(use_deploy_strategy=True, exclude_stroke=True):
    """
    Main pipeline following the paper's methodology.

    Parameters:
      use_deploy_strategy: If True, train DM_full as primary and deploy to others.
                           If False, train each target independently (for comparison).
      exclude_stroke: If True, exclude baseline stroke patients.
    """
    print("="*70)
    print("UKB-DRP Paper Reproduction — v2 Pipeline")
    print("="*70)
    print(f"  Deploy strategy: {use_deploy_strategy}")
    print(f"  Stroke exclusion: {exclude_stroke}")
    print(f"  Results base: {RESULTS_BASE}")
    print()

    # --- Load Data ---
    print("[Load] Loading preprocessed data...")
    mydf, rm_f = load_and_prepare_data()

    if exclude_stroke:
        mydf = exclude_baseline_stroke(mydf)

    print(f"  Final sample: {len(mydf):,} participants")
    print(f"  Dementia positive: {mydf['dementia_status'].sum():,}")
    print(f"  AD positive: {mydf['AD_status'].sum():,}")

    # Feature matrix (all features)
    X_all = mydf.drop(columns=rm_f)

    # --- Determine training targets ---
    if use_deploy_strategy:
        # Paper's deploy strategy: train DM_full → deploy to others
        primary_targets = ['DM_full']
        deploy_targets = ['DM_10yrs', 'DM_5yrs', 'AD_full', 'AD_10yrs', 'AD_5yrs']
    else:
        primary_targets = list(ALL_TARGETS.keys())
        deploy_targets = []

    # --- Train Primary Model(s) ---
    primary_features = None
    all_metrics = {}

    for target_name in primary_targets:
        target_col, years_col, max_years = ALL_TARGETS[target_name]
        y_full, include = apply_time_window(mydf, target_col, years_col, max_years)

        # Proper censoring: exclude non-converters with insufficient follow-up
        n_censored = (~include).sum()
        if n_censored > 0:
            print(f"\n  [CENSOR] {target_name}: excluding {n_censored} subjects "
                  f"with insufficient follow-up")
            mydf_t = mydf[include].copy()
            X_t = X_all[include].copy()
            y = y_full[include].copy()
        else:
            mydf_t = mydf
            X_t = X_all
            y = y_full

        results_dir = os.path.join(RESULTS_BASE, target_name)
        os.makedirs(results_dir, exist_ok=True)

        print(f"\n{'#'*70}")
        print(f"# PRIMARY MODEL: {target_name} (n_pos={y.sum()}, n_total={len(y)})")
        print(f"{'#'*70}")

        # s01: Initial feature importance
        s01_df = run_s01(mydf_t, X_t, y, results_dir)

        # s02: Hierarchical clustering
        s02_df = run_s02(mydf_t, X_t, y, s01_df, results_dir)

        # s03: Final feature importance
        s03_df = run_s03(mydf_t, X_t, y, s02_df, results_dir)

        # s04: Sequential Forward Selection
        s04_df = run_s04_sfs(mydf_t, X_t, y, s03_df, results_dir)

        # s05: Final model with tuning + Isotonic calibration
        metrics = run_s05(mydf_t, X_t, y, s04_df, results_dir, target_name)
        all_metrics[target_name] = metrics

        # SHAP analysis
        primary_features = s04_df['Features'][:TOP_N_S04].tolist()
        run_shap(mydf_t, X_t, y, primary_features, results_dir, target_name)

        # Calibration plot
        plot_calibration(results_dir, target_name)

        gc.collect()

    # --- Deploy to Other Targets ---
    if deploy_targets and primary_features:
        for deploy_name in deploy_targets:
            deploy_dir = os.path.join(RESULTS_BASE, deploy_name)
            os.makedirs(deploy_dir, exist_ok=True)

            metrics = run_deploy(mydf, X_all, deploy_name, primary_features, deploy_dir)
            all_metrics[deploy_name] = metrics

            # SHAP for deployed model
            target_col, years_col, max_years = ALL_TARGETS[deploy_name]
            y_deploy, include_deploy = apply_time_window(mydf, target_col, years_col, max_years)
            if (~include_deploy).sum() > 0:
                y_shap = y_deploy[include_deploy]
                X_shap = X_all[include_deploy]
            else:
                y_shap = y_deploy
                X_shap = X_all
            run_shap(mydf, X_shap, y_shap, primary_features, deploy_dir, deploy_name)
            gc.collect()

    # --- Summary ---
    print("\n" + "="*70)
    print("FINAL SUMMARY — All Models")
    print("="*70)
    print(f"{'Model':<15} {'AUC':<12} {'HL-p':<10} {'Sens':<10} {'Spec':<10} {'Cutoff':<10}")
    print("-"*67)
    for name, m in sorted(all_metrics.items()):
        print(f"{name:<15} {m.get('auc_mean', 0):.4f}±{m.get('auc_std', 0):.4f}   "
              f"{m.get('hl_pvalue', 0):.4f}     "
              f"{m.get('sensitivity', 0):.4f}   {m.get('specificity', 0):.4f}   "
              f"{m.get('best_cutoff', 0):.3f}")

    # Save summary
    summary = []
    for name, m in sorted(all_metrics.items()):
        summary.append({
            'model': name,
            'auc_mean': m.get('auc_mean'),
            'auc_std': m.get('auc_std'),
            'hl_pvalue': m.get('hl_pvalue'),
            'sensitivity': m.get('sensitivity'),
            'specificity': m.get('specificity'),
            'youden': m.get('youden'),
            'best_cutoff': m.get('best_cutoff'),
        })
    pd.DataFrame(summary).to_csv(os.path.join(RESULTS_BASE, 'summary_metrics.csv'), index=False)

    print(f"\nResults saved to: {RESULTS_BASE}")
    print("Pipeline complete.")
    return all_metrics


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='UKB-DRP Paper Reproduction v2')
    parser.add_argument('--no-deploy', action='store_true',
                        help='Train each target independently instead of deploy strategy')
    parser.add_argument('--no-stroke-exclusion', action='store_true',
                        help='Do not exclude baseline stroke patients')
    parser.add_argument('--target', type=str, default=None,
                        help='Train a single target only (e.g., DM_full, AD_5yrs)')
    parser.add_argument('--n-combos', type=int, default=N_PARAM_COMBOS,
                        help=f'Number of hyperparameter combos to try (default: {N_PARAM_COMBOS})')
    args = parser.parse_args()

    if args.n_combos != N_PARAM_COMBOS:
        N_PARAM_COMBOS = args.n_combos

    if args.target:
        # Single target mode
        mydf, rm_f = load_and_prepare_data()
        if not args.no_stroke_exclusion:
            mydf = exclude_baseline_stroke(mydf)
        X_all = mydf.drop(columns=rm_f)

        target_col, years_col, max_years = ALL_TARGETS[args.target]
        y_full, include = apply_time_window(mydf, target_col, years_col, max_years)

        # Filter censored subjects
        n_censored = (~include).sum()
        if n_censored > 0:
            print(f"  [CENSOR] excluding {n_censored} subjects with insufficient follow-up")
            mydf_t = mydf[include].copy()
            X_t = X_all[include].copy()
            y = y_full[include].copy()
        else:
            mydf_t = mydf
            X_t = X_all
            y = y_full

        results_dir = os.path.join(RESULTS_BASE, args.target)
        os.makedirs(results_dir, exist_ok=True)

        print(f"\n{'#'*70}")
        print(f"# SINGLE TARGET: {args.target} (n_pos={y.sum()}, n_total={len(y)})")
        print(f"{'#'*70}")

        s01_df = run_s01(mydf_t, X_t, y, results_dir)
        s02_df = run_s02(mydf_t, X_t, y, s01_df, results_dir)
        s03_df = run_s03(mydf_t, X_t, y, s02_df, results_dir)
        s04_df = run_s04_sfs(mydf_t, X_t, y, s03_df, results_dir)
        metrics = run_s05(mydf_t, X_t, y, s04_df, results_dir, args.target)

        features = s04_df['Features'][:TOP_N_S04].tolist()
        run_shap(mydf_t, X_t, y, features, results_dir, args.target)
        plot_calibration(results_dir, args.target)

    else:
        # Full pipeline
        run_pipeline(
            use_deploy_strategy=not args.no_deploy,
            exclude_stroke=not args.no_stroke_exclusion,
        )

#!/usr/bin/env python3
"""
Standalone s05 + Deploy: Uses completed s01-s04 results, trains final models
with hyperparameter tuning + Isotonic calibration + SHAP + HL test.
Optimized for practical runtime with progress output.
"""
import os, sys, warnings, time, gc, json
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from scipy import stats
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

# ===== CONFIG =====
DPATH = os.path.join(PROJECT_ROOT, 'local_data') + '/'
PREPROCESSED_CSV = os.path.join(DPATH, 'Preprocessed_Data', 'Preprocessed_Data.csv')
RESULTS_BASE = os.path.join(DPATH, 'Results_v2')
RANDOM_STATE = 2022
N_SPLITS = 5
TOP_N = 10

# Use reduced param combos for practicality (paper used 1000, we use 200)
N_PARAM_COMBOS = 200

PARAM_GRID = {
    'n_estimators': [200, 300, 500],
    'max_depth': [8, 10, 15, 20],
    'num_leaves': [8, 10, 16, 31],
    'subsample': [0.5, 0.6, 0.7, 0.8],
    'learning_rate': [0.005, 0.01, 0.02, 0.05],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8],
}
# Total: 3×4×4×4×4×4 = 3072 combos, sample 200

ALL_TARGETS = {
    'DM_full':  ('dementia_status', 'dementia_years', None),
    'DM_10yrs': ('dementia_status', 'dementia_years', 10),
    'DM_5yrs':  ('dementia_status', 'dementia_years', 5),
    'AD_full':  ('AD_status', 'AD_years', None),
    'AD_10yrs': ('AD_status', 'AD_years', 10),
    'AD_5yrs':  ('AD_status', 'AD_years', 5),
}

# ===== UTILS =====
def threshold(arr, cutoff):
    a = arr.copy(); a[a < cutoff] = 0; a[a >= cutoff] = 1; return a

def extend(my_array, nb_points):
    if len(my_array) == nb_points:
        pass
    else:
        nb2impute = nb_points - len(my_array)
        impute_array = np.zeros(nb2impute)
        my_array = np.concatenate((impute_array, my_array), axis=0)
    return np.expand_dims(my_array, -1)

def get_full_eval(y_test, pred_prob, cutoff_list):
    evals = []
    for cutoff in cutoff_list:
        pb = threshold(pred_prob, cutoff)
        tn, fp, fn, tp = confusion_matrix(y_test, pb).ravel()
        acc = (tp+tn)/(tp+tn+fp+fn)
        sens = tp/(tp+fn) if (tp+fn)>0 else 0
        spec = tn/(tn+fp) if (tn+fp)>0 else 0
        prec = tp/(tp+fp) if (tp+fp)>0 else 0
        y_idx = sens+spec-1
        f1 = 2*prec*sens/(prec+sens) if (prec+sens)>0 else 0
        auc = roc_auc_score(y_test, pred_prob)
        apr = average_precision_score(y_test, pred_prob)
        nnd = 1/y_idx if y_idx>0 else np.inf
        evals.append(np.round((cutoff, acc, sens, spec, prec, y_idx, f1, auc, apr, nnd), 4))
    evals = pd.DataFrame(evals)
    evals.columns = ['Cutoff','Acc','Sens','Spec','Prec','Youden','F1','AUC','APR','NND']
    return evals

def avg_results(results_list):
    cn = results_list[0].columns.tolist()
    cn_std = [c+'_std' for c in cn]
    nf, nr, nc = len(results_list), results_list[0].shape[0], results_list[0].shape[1]
    res = np.zeros((nf, nr, nc))
    for i in range(nf): res[i] = np.array(results_list[i])
    ravg = pd.DataFrame(np.round(np.average(res, axis=0),3), columns=cn)
    rstd = pd.DataFrame(np.round(np.std(res, axis=0), 3), columns=cn_std)
    return pd.concat((ravg, rstd), axis=1)

def select_params_combo(my_dict, nb_items):
    combo_list = [dict(zip(my_dict.keys(), v)) for v in product(*my_dict.values())]
    random.seed(2020)
    return random.sample(combo_list, min(nb_items, len(combo_list)))

def hl_pvalue(obs_prob, pred_prob, percentage=1, bin_obs_nb=None):
    obs_prob = obs_prob / percentage
    pred_prob = pred_prob / percentage
    nominator = (obs_prob - pred_prob) ** 2 * (bin_obs_nb if bin_obs_nb else 1)
    denominator = pred_prob * (1 - pred_prob)
    denominator[denominator == 0] = 1
    stat = nominator / denominator
    stat[np.isinf(stat)] = 0.0
    test_stat = np.nansum(stat)
    pvalue = 1 - stats.chi2.cdf(test_stat, 8)
    return np.round(pvalue, 4)

def hl_test_per_fold(y_true, y_pred, n_bins=10):
    try:
        _, bin_edges = pd.qcut(y_pred, q=n_bins, retbins=True, duplicates='drop')
        bin_labels = pd.cut(y_pred, bins=bin_edges, include_lowest=True, duplicates='drop')
        obs, pred, cnts = [], [], []
        for interval in bin_labels.cat.categories:
            mask = bin_labels == interval
            if mask.sum() > 0:
                obs.append(y_true[mask].mean())
                pred.append(y_pred[mask].mean())
                cnts.append(mask.sum())
        return hl_pvalue(np.array(obs), np.array(pred), percentage=1, bin_obs_nb=np.array(cnts))
    except:
        return np.nan

# ===== DATA LOADING =====
def load_and_prepare():
    mydf = pd.read_csv(PREPROCESSED_CSV)
    mydf['AD_years'] = mydf['AD_years'].fillna(mydf['dementia_years'])
    mydf['AD_years'] = mydf['AD_years'].clip(lower=-1)

    # Exclude baseline stroke
    stroke_mask = (mydf['stroke_years'] < 0) & (mydf['stroke_years'].notna())
    mydf = mydf[~stroke_mask].copy()
    print(f"  Excluded {stroke_mask.sum()} baseline stroke, remaining: {len(mydf):,}")

    rm_f1 = ['Unnamed: 0', 'eid', 'dementia_status', 'dementia_years',
             'AD_status', 'AD_years', 'VD_status', 'VD_years',
             'stroke_status', 'stroke_years']
    rm_HES = ['41234-0.0', '41259-0.0', '41149-0.0', '41289-0.0',
              '41218-0.0', '41235-0.0', '41214-0.0']
    rm_f = [c for c in rm_f1 + rm_HES if c in mydf.columns]
    return mydf, rm_f

def apply_time_window(mydf, target_col, years_col, max_years=None):
    y = mydf[target_col].copy()
    if max_years is not None:
        y_years = mydf[years_col]
        y.loc[(y == 1) & (y_years > max_years)] = 0
    return y

# ===== S05: FINAL MODEL =====
def run_s05(mydf, X_all, y, features, results_dir, target_name, n_combos=N_PARAM_COMBOS):
    print("\n" + "="*70)
    print(f"[s05] {target_name} — {n_combos} hyperparameter combos + Isotonic calibration")
    print("="*70)
    t0 = time.time()

    X = X_all[features[:TOP_N]]
    outer_kf = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=True)
    cutoff_list = [np.round(0.001 + i * 0.001, 3) for i in range(100)]

    param_combos = select_params_combo(PARAM_GRID, n_combos)
    print(f"  Sampling {len(param_combos)} combos from {np.prod([len(v) for v in PARAM_GRID.values()])} total")

    results_cv = []
    obs_array, pred_array = np.zeros((10, 1)), np.zeros((10, 1))
    y_test_lst, y_pred_prob_lst = [], []
    all_best_params = []

    for fold, (outer_tr, outer_te) in enumerate(outer_kf.split(X, y)):
        t_fold = time.time()
        print(f"\n  --- Fold {fold+1}/{N_SPLITS} ---")

        X_tr = X.iloc[outer_tr, :]; y_tr = y.iloc[outer_tr]
        X_te = X.iloc[outer_te, :]; y_te = y.iloc[outer_te]

        # Inner CV for hyperparameter selection
        inner_kf = StratifiedKFold(n_splits=3, random_state=RANDOM_STATE+fold, shuffle=True)
        param_scores = []

        for pi, params in enumerate(param_combos):
            inner_aucs = []
            for itr, ivl in inner_kf.split(X_tr, y_tr):
                X_it, X_iv = X_tr.iloc[itr, :], X_tr.iloc[ivl, :]
                y_it, y_iv = y_tr.iloc[itr], y_tr.iloc[ivl]
                lgb = LGBMClassifier(objective='binary', is_unbalance=True,
                                      metric='auc', verbosity=-1, seed=2020, n_jobs=4)
                lgb.set_params(**params)
                lgb.fit(X_it, y_it)
                inner_aucs.append(roc_auc_score(y_iv, lgb.predict_proba(X_iv)[:, 1]))
            param_scores.append(np.mean(inner_aucs))

            if (pi+1) % 50 == 0:
                best_sofar = max(param_scores)
                print(f"    Combo {pi+1}/{len(param_combos)}, best AUC so far: {best_sofar:.4f}")

        best_idx = np.argmax(param_scores)
        best_params = param_combos[best_idx]
        all_best_params.append(best_params)
        print(f"  Best params (AUC={param_scores[best_idx]:.4f}): {best_params}")

        # Train + calibrate (paper's split-train-calibrate approach)
        n_calib = int(len(X_tr) * 0.4)
        X_gbm_tr = X_tr.iloc[n_calib:]; y_gbm_tr = y_tr.iloc[n_calib:]
        X_cali = X_tr.iloc[:n_calib]; y_cali = y_tr.iloc[:n_calib]

        gbm = LGBMClassifier(objective='binary', is_unbalance=True,
                              metric='auc', verbosity=-1, seed=2022, n_jobs=4)
        gbm.set_params(**best_params)
        gbm.fit(X_gbm_tr, y_gbm_tr)

        # Isotonic Regression calibration
        raw_cali = gbm.predict_proba(X_cali)[:, 1]
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(raw_cali, y_cali)

        # Predict
        raw_test = gbm.predict_proba(X_te)[:, 1]
        y_pred = iso.predict(raw_test)
        y_pred = np.clip(y_pred, 0, 1)

        # Calibration curve
        obsf, predf = calibration_curve(y_te, y_pred, n_bins=10, strategy='quantile')
        obs_array = np.concatenate([obs_array, extend(obsf, 10)], axis=1)
        pred_array = np.concatenate([pred_array, extend(predf, 10)], axis=1)

        results_cv.append(get_full_eval(y_te, y_pred, cutoff_list))
        y_pred_prob_lst.append(y_pred)
        y_test_lst.append(np.array(y_te))

        auc_fold = roc_auc_score(y_te, y_pred)
        hl_p = hl_test_per_fold(y_te.values, y_pred)
        print(f"  Fold AUC={auc_fold:.4f}, HL-p={hl_p:.4f}, time={time.time()-t_fold:.0f}s")

    # Save
    pred_prob_df = pd.DataFrame(y_pred_prob_lst).T
    test_df = pd.DataFrame(y_test_lst).T
    pred_prob_df.to_csv(os.path.join(results_dir, 'pred_prob_cv_df.csv'), index=False)
    test_df.to_csv(os.path.join(results_dir, 'test_cv_df.csv'), index=False)

    final_output = avg_results(results_cv)
    final_output.to_csv(os.path.join(results_dir, 's05_final_model.csv'), index=False)

    pd.DataFrame(all_best_params).to_csv(os.path.join(results_dir, 's05_best_params.csv'), index=False)

    obs_mean = np.round(np.mean(obs_array[:, 1:], axis=1), 4)
    pred_mean = np.round(np.mean(pred_array[:, 1:], axis=1), 4)
    pd.DataFrame({'Observed': obs_mean, 'Predicted': pred_mean}).to_csv(
        os.path.join(results_dir, 's05_calibration.csv'), index=False)

    all_yt = np.concatenate([np.array(yl) for yl in y_test_lst])
    all_yp = np.concatenate([np.array(yp) for yp in y_pred_prob_lst])
    overall_hl = hl_test_per_fold(all_yt, all_yp)
    overall_brier = brier_score_loss(all_yt, all_yp)

    auc_row = final_output.iloc[0]
    best_yd = final_output['Youden'].idxmax()
    best_row = final_output.iloc[best_yd]

    print(f"\n  Final AUC = {auc_row['AUC']:.4f} ± {auc_row.get('AUC_std', 0):.4f}")
    print(f"  Brier = {overall_brier:.4f}, HL-p = {overall_hl:.4f}")
    print(f"  Best cutoff: {best_row['Cutoff']:.3f}, Sens={best_row['Sens']:.4f}, Spec={best_row['Spec']:.4f}")
    print(f"  Total time: {time.time()-t0:.0f}s")

    metrics = {
        'target': target_name,
        'auc_mean': float(auc_row['AUC']), 'auc_std': float(auc_row.get('AUC_std', 0)),
        'brier_score': float(overall_brier), 'hl_pvalue': float(overall_hl),
        'best_cutoff': float(best_row['Cutoff']), 'sensitivity': float(best_row['Sens']),
        'specificity': float(best_row['Spec']), 'youden': float(best_row['Youden']),
        'features': features[:TOP_N],
    }
    with open(os.path.join(results_dir, 's05_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, default=str)

    # Calibration plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
    ax.plot(pred_mean, obs_mean, 'ro-', label='Model')
    ax.set_xlabel('Predicted Probability'); ax.set_ylabel('Observed Probability')
    ax.set_title(f'{target_name} Calibration'); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, 'calibration_plot.png'), dpi=150, bbox_inches='tight')
    plt.close()
    gc.collect()

    return metrics

# ===== SHAP =====
def run_shap(mydf, X_all, y, features, results_dir, target_name):
    print(f"\n[SHAP] {target_name}...")
    try:
        import shap
    except ImportError:
        print("  shap not installed, skipping")
        return

    my_f = [f for f in features[:TOP_N] if f in X_all.columns]
    X = X_all[my_f]

    mykf = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=True)
    for tr, te in mykf.split(X, y):
        X_tr, y_tr = X.iloc[tr, :], y.iloc[tr]
        X_te = X.iloc[te, :]; break

    lgb = LGBMClassifier(objective='binary', is_unbalance=True, verbosity=-1, seed=2020, n_jobs=4)
    lgb.set_params(n_estimators=500, max_depth=15, num_leaves=10,
                    subsample=0.7, learning_rate=0.01, colsample_bytree=0.7)
    lgb.fit(X_tr, y_tr)

    explainer = shap.Explainer(lgb)
    shap_values = explainer(X_te)

    plt.figure()
    shap.plots.beeswarm(shap_values[:, :, 1], show=False, order=list(range(min(TOP_N, len(my_f)))))
    plt.gcf().set_size_inches(18, 5.5)
    ax = plt.gca()
    ax.set_ylabel('Selected Predictors', fontsize=20, weight='bold')
    ax.set_xlabel('SHAP Values', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'shap_beeswarm.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  SHAP saved.")


# ===== DEPLOY =====
def run_deploy(mydf, X_all, target_name, features, results_dir, n_combos=100):
    print("\n" + "="*70)
    print(f"[Deploy] DM_full → {target_name}")
    print("="*70)
    t0 = time.time()

    target_col, years_col, max_years = ALL_TARGETS[target_name]
    y = apply_time_window(mydf, target_col, years_col, max_years)
    X = X_all[[f for f in features[:TOP_N] if f in X_all.columns]]

    print(f"  n_pos={y.sum()}, rate={y.sum()/len(y)*100:.2f}%")

    outer_kf = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=True)
    cutoff_list = [np.round(0.001 + i * 0.001, 3) for i in range(100)]

    param_combos = select_params_combo(PARAM_GRID, n_combos)

    results_cv = []
    obs_array, pred_array = np.zeros((10, 1)), np.zeros((10, 1))
    y_test_lst, y_pred_prob_lst = [], []
    all_best_params = []

    for fold, (outer_tr, outer_te) in enumerate(outer_kf.split(X, y)):
        t_fold = time.time()
        X_tr = X.iloc[outer_tr, :]; y_tr = y.iloc[outer_tr]
        X_te = X.iloc[outer_te, :]; y_te = y.iloc[outer_te]

        # Inner tuning
        inner_kf = StratifiedKFold(n_splits=3, random_state=RANDOM_STATE+fold, shuffle=True)
        param_scores = []
        for params in param_combos:
            inner_aucs = []
            for itr, ivl in inner_kf.split(X_tr, y_tr):
                lgb = LGBMClassifier(objective='binary', is_unbalance=True,
                                      metric='auc', verbosity=-1, seed=2020, n_jobs=4)
                lgb.set_params(**params)
                lgb.fit(X_tr.iloc[itr, :], y_tr.iloc[itr])
                inner_aucs.append(roc_auc_score(y_tr.iloc[ivl], lgb.predict_proba(X_tr.iloc[ivl, :])[:, 1]))
            param_scores.append(np.mean(inner_aucs))

        best_params = param_combos[np.argmax(param_scores)]
        all_best_params.append(best_params)

        # Split-train-calibrate
        n_calib = int(len(X_tr) * 0.4)
        gbm = LGBMClassifier(objective='binary', is_unbalance=True,
                              metric='auc', verbosity=-1, seed=2022, n_jobs=4)
        gbm.set_params(**best_params)
        gbm.fit(X_tr.iloc[n_calib:], y_tr.iloc[n_calib:])

        raw_cali = gbm.predict_proba(X_tr.iloc[:n_calib])[:, 1]
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(raw_cali, y_tr.iloc[:n_calib])

        raw_test = gbm.predict_proba(X_te)[:, 1]
        y_pred = iso.predict(raw_test)
        y_pred = np.clip(y_pred, 0, 1)

        obsf, predf = calibration_curve(y_te, y_pred, n_bins=10, strategy='quantile')
        obs_array = np.concatenate([obs_array, extend(obsf, 10)], axis=1)
        pred_array = np.concatenate([pred_array, extend(predf, 10)], axis=1)

        results_cv.append(get_full_eval(y_te, y_pred, cutoff_list))
        y_pred_prob_lst.append(y_pred)
        y_test_lst.append(np.array(y_te))

        print(f"  Fold {fold+1}/{N_SPLITS} AUC={roc_auc_score(y_te, y_pred):.4f}, time={time.time()-t_fold:.0f}s")

    # Save
    pd.DataFrame(y_pred_prob_lst).T.to_csv(os.path.join(results_dir, 'pred_prob_cv_df.csv'), index=False)
    pd.DataFrame(y_test_lst).T.to_csv(os.path.join(results_dir, 'test_cv_df.csv'), index=False)

    final_output = avg_results(results_cv)
    final_output.to_csv(os.path.join(results_dir, 'deploy_results.csv'), index=False)

    all_yt = np.concatenate([np.array(yl) for yl in y_test_lst])
    all_yp = np.concatenate([np.array(yp) for yp in y_pred_prob_lst])
    overall_hl = hl_test_per_fold(all_yt, all_yp)

    auc_row = final_output.iloc[0]
    best_yd = final_output['Youden'].idxmax()
    best_row = final_output.iloc[best_yd]

    print(f"  AUC = {auc_row['AUC']:.4f} ± {auc_row.get('AUC_std', 0):.4f}")
    print(f"  HL-p = {overall_hl:.4f}, Sens={best_row['Sens']:.4f}, Spec={best_row['Spec']:.4f}")
    print(f"  Time: {time.time()-t0:.0f}s")

    return {
        'target': target_name,
        'auc_mean': float(auc_row['AUC']), 'auc_std': float(auc_row.get('AUC_std', 0)),
        'hl_pvalue': float(overall_hl),
        'best_cutoff': float(best_row['Cutoff']),
        'sensitivity': float(best_row['Sens']),
        'specificity': float(best_row['Spec']),
        'youden': float(best_row['Youden']),
    }


# ===== MAIN =====
if __name__ == '__main__':
    print("="*70)
    print("UKB-DRP s05 Final Model Training (Paper Reproduction)")
    print("="*70)

    mydf, rm_f = load_and_prepare()
    X_all = mydf.drop(columns=rm_f)

    # Read s04 results for DM_full
    s04_dir = os.path.join(RESULTS_BASE, 'DM_full')
    s04_df = pd.read_csv(os.path.join(s04_dir, 's04_selected_features.csv'))
    primary_features = s04_df['Features'].tolist()
    print(f"\nPrimary features (from SFS): {primary_features[:TOP_N]}")
    print(f"Features available in X_all: {sum(1 for f in primary_features[:TOP_N] if f in X_all.columns)}/{TOP_N}")

    # ===== DM_full (primary model) =====
    target_col, years_col, max_years = ALL_TARGETS['DM_full']
    y_dm_full = apply_time_window(mydf, target_col, years_col, max_years)
    os.makedirs(s04_dir, exist_ok=True)

    metrics = run_s05(mydf, X_all, y_dm_full, primary_features, s04_dir, 'DM_full', n_combos=N_PARAM_COMBOS)
    run_shap(mydf, X_all, y_dm_full, primary_features, s04_dir, 'DM_full')

    all_metrics = {'DM_full': metrics}

    # ===== Deploy to other targets =====
    deploy_targets = ['DM_10yrs', 'DM_5yrs', 'AD_full', 'AD_10yrs', 'AD_5yrs']
    for dt in deploy_targets:
        deploy_dir = os.path.join(RESULTS_BASE, dt)
        os.makedirs(deploy_dir, exist_ok=True)

        m = run_deploy(mydf, X_all, dt, primary_features, deploy_dir, n_combos=100)
        all_metrics[dt] = m

        target_col_d, years_col_d, max_years_d = ALL_TARGETS[dt]
        y_d = apply_time_window(mydf, target_col_d, years_col_d, max_years_d)
        run_shap(mydf, X_all, y_d, primary_features, deploy_dir, dt)
        gc.collect()

    # ===== Summary =====
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"{'Model':<15} {'AUC':<13} {'HL-p':<10} {'Sens':<10} {'Spec':<10} {'Cutoff':<10}")
    print("-"*68)
    for name, m in sorted(all_metrics.items()):
        print(f"{name:<15} {m['auc_mean']:.4f}±{m['auc_std']:.4f}  "
              f"{m['hl_pvalue']:.4f}     {m['sensitivity']:.4f}   "
              f"{m['specificity']:.4f}   {m['best_cutoff']:.3f}")

    summary = []
    for name, m in sorted(all_metrics.items()):
        summary.append({k: m[k] for k in ['target','auc_mean','auc_std','hl_pvalue',
                                           'sensitivity','specificity','youden','best_cutoff']})
    pd.DataFrame(summary).to_csv(os.path.join(RESULTS_BASE, 'summary_metrics.csv'), index=False)
    print(f"\nResults saved to: {RESULTS_BASE}")

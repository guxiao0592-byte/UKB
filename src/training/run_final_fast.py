#!/usr/bin/env python3
"""
Fast Final Training: Uses SFS features + Isotonic calibration + 5-fold CV.
Uses tuned-like base params (from Fold 1 best params).
Completes all 6 models in ~30 min.
"""
import os, sys, warnings, time, json
warnings.filterwarnings('ignore')
PROJECT_ROOT = os.environ.get('PROJECT_ROOT',
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
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

DPATH = os.path.join(PROJECT_ROOT, 'local_data') + '/'
PREPROCESSED_CSV = os.path.join(DPATH, 'Preprocessed_Data', 'Preprocessed_Data.csv')
RESULTS_BASE = os.path.join(DPATH, 'Results_v2')
RANDOM_STATE = 2022; N_SPLITS = 5; TOP_N = 10

# Best params found from 200-combo tuning (Fold 1 of DM_full)
BEST_PARAMS = {'n_estimators': 500, 'max_depth': 20, 'num_leaves': 10,
               'subsample': 0.5, 'learning_rate': 0.02, 'colsample_bytree': 0.5}

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
def extend(arr, nb):
    if len(arr) == nb: pass
    else: arr = np.concatenate((np.zeros(nb-len(arr)), arr))
    return np.expand_dims(arr, -1)

def get_full_eval(y_test, pred_prob, cutoff_list):
    evals = []
    for cutoff in cutoff_list:
        pb = threshold(pred_prob, cutoff)
        tn, fp, fn, tp = confusion_matrix(y_test, pb).ravel()
        acc = (tp+tn)/(tp+tn+fp+fn); sens = tp/(tp+fn) if (tp+fn)>0 else 0
        spec = tn/(tn+fp) if (tn+fp)>0 else 0; prec = tp/(tp+fp) if (tp+fp)>0 else 0
        y_idx = sens+spec-1; f1 = 2*prec*sens/(prec+sens) if (prec+sens)>0 else 0
        auc = roc_auc_score(y_test, pred_prob); apr = average_precision_score(y_test, pred_prob)
        nnd = 1/y_idx if y_idx>0 else np.inf
        evals.append(np.round((cutoff, acc, sens, spec, prec, y_idx, f1, auc, apr, nnd), 4))
    return pd.DataFrame(evals, columns=['Cutoff','Acc','Sens','Spec','Prec','Youden','F1','AUC','APR','NND'])

def avg_results(rl):
    cn = rl[0].columns.tolist(); cn_std = [c+'_std' for c in cn]
    nf, nr, nc = len(rl), rl[0].shape[0], rl[0].shape[1]
    res = np.zeros((nf,nr,nc))
    for i in range(nf): res[i] = np.array(rl[i])
    return pd.concat((pd.DataFrame(np.round(np.average(res,axis=0),3),columns=cn),
                       pd.DataFrame(np.round(np.std(res,axis=0),3),columns=cn_std)), axis=1)

def hl_test_per_fold(y_true, y_pred, n_bins=10):
    try:
        _, be = pd.qcut(y_pred, q=n_bins, retbins=True, duplicates='drop')
        bl = pd.cut(y_pred, bins=be, include_lowest=True, duplicates='drop')
        obs, pred, cnts = [], [], []
        for iv in bl.cat.categories:
            m = bl == iv
            if m.sum()>0: obs.append(y_true[m].mean()); pred.append(y_pred[m].mean()); cnts.append(m.sum())
        obs=np.array(obs); pred=np.array(pred); cnts=np.array(cnts)
        nom=(obs-pred)**2*cnts; denom=pred*(1-pred); denom[denom==0]=1
        stat=nom/denom; stat[np.isinf(stat)]=0.0
        return np.round(1-stats.chi2.cdf(np.nansum(stat),8), 4)
    except: return np.nan

def flush_print(s):
    print(s, flush=True)

# ===== TRAIN + CALIBRATE ONE TARGET =====
def train_one_target(mydf, X_all, target_name, features, results_dir, params=BEST_PARAMS):
    target_col, years_col, max_years = ALL_TARGETS[target_name]
    y = mydf[target_col].copy()
    if max_years is not None:
        y.loc[(y == 1) & (mydf[years_col] > max_years)] = 0

    X = X_all[[f for f in features[:TOP_N] if f in X_all.columns]]
    flush_print(f"\n{'='*60}")
    flush_print(f"[{target_name}] n_pos={y.sum()}, rate={y.sum()/len(y)*100:.2f}%")
    flush_print(f"{'='*60}")

    outer_kf = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=True)
    cutoff_list = [np.round(0.001 + i * 0.001, 3) for i in range(100)]

    results_cv = []; obs_arr = np.zeros((10,1)); pred_arr = np.zeros((10,1))
    y_test_lst = []; y_pred_lst = []
    fold_metrics = []

    for fold, (otr, ote) in enumerate(outer_kf.split(X, y)):
        t0 = time.time()
        X_tr, y_tr = X.iloc[otr,:], y.iloc[otr]
        X_te, y_te = X.iloc[ote,:], y.iloc[ote]

        # Split-train-calibrate (paper's approach)
        n_calib = int(len(X_tr) * 0.4)
        gbm = LGBMClassifier(objective='binary', is_unbalance=True,
                              metric='auc', verbosity=-1, seed=2022, n_jobs=4)
        gbm.set_params(**params)
        gbm.fit(X_tr.iloc[n_calib:], y_tr.iloc[n_calib:])

        raw_cali = gbm.predict_proba(X_tr.iloc[:n_calib])[:, 1]
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(raw_cali, y_tr.iloc[:n_calib])

        y_pred = iso.predict(gbm.predict_proba(X_te)[:, 1])
        y_pred = np.clip(y_pred, 0, 1)

        obsf, predf = calibration_curve(y_te, y_pred, n_bins=10, strategy='quantile')
        obs_arr = np.concatenate([obs_arr, extend(obsf, 10)], axis=1)
        pred_arr = np.concatenate([pred_arr, extend(predf, 10)], axis=1)

        results_cv.append(get_full_eval(y_te, y_pred, cutoff_list))
        y_pred_lst.append(y_pred); y_test_lst.append(np.array(y_te))

        auc_f = roc_auc_score(y_te, y_pred)
        hl_p = hl_test_per_fold(y_te.values, y_pred)
        fold_metrics.append({'fold': fold, 'auc': auc_f, 'hl_p': hl_p,
                              'time': time.time()-t0})
        flush_print(f"  Fold {fold+1}/{N_SPLITS}: AUC={auc_f:.4f}, HL-p={hl_p}, "
                     f"time={time.time()-t0:.0f}s")

    # Save fold-level results
    pd.DataFrame(fold_metrics).to_csv(os.path.join(results_dir, 'fold_metrics.csv'), index=False)
    pd.DataFrame(y_pred_lst).T.to_csv(os.path.join(results_dir, 'pred_prob_cv_df.csv'), index=False)
    pd.DataFrame(y_test_lst).T.to_csv(os.path.join(results_dir, 'test_cv_df.csv'), index=False)

    final_output = avg_results(results_cv)
    final_output.to_csv(os.path.join(results_dir, 's05_final_model.csv'), index=False)

    obs_mean = np.round(np.mean(obs_arr[:, 1:], axis=1), 4)
    pred_mean = np.round(np.mean(pred_arr[:, 1:], axis=1), 4)
    pd.DataFrame({'Observed': obs_mean, 'Predicted': pred_mean}).to_csv(
        os.path.join(results_dir, 's05_calibration.csv'), index=False)

    all_yt = np.concatenate([np.array(yl) for yl in y_test_lst])
    all_yp = np.concatenate([np.array(yp) for yp in y_pred_lst])
    overall_hl = hl_test_per_fold(all_yt, all_yp)
    overall_brier = brier_score_loss(all_yt, all_yp)

    auc_row = final_output.iloc[0]
    best_yd = final_output['Youden'].idxmax(); best_row = final_output.iloc[best_yd]

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
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5); ax.plot(pred_mean, obs_mean, 'ro-')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Observed'); ax.set_title(f'{target_name}')
    ax.legend(['Perfect', 'Model']); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(results_dir, 'calibration_plot.png'), dpi=150)
    plt.close()

    flush_print(f"  AUC={auc_row['AUC']:.4f}±{auc_row.get('AUC_std',0):.4f}, "
                f"Brier={overall_brier:.4f}, HL-p={overall_hl:.4f}, "
                f"Sens={best_row['Sens']:.4f}, Spec={best_row['Spec']:.4f}")
    return metrics

# ===== SHAP =====
def run_shap(mydf, X_all, y, features, results_dir, target_name):
    flush_print(f"\n[SHAP] {target_name}...")
    try:
        import shap
    except ImportError:
        flush_print("  shap not installed, skipping"); return

    my_f = [f for f in features[:TOP_N] if f in X_all.columns]
    X = X_all[my_f]
    mykf = StratifiedKFold(n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=True)
    for tr, te in mykf.split(X, y):
        X_tr, y_tr = X.iloc[tr,:], y.iloc[tr]; X_te = X.iloc[te,:]; break

    lgb = LGBMClassifier(objective='binary', is_unbalance=True, verbosity=-1, seed=2020, n_jobs=4)
    lgb.set_params(**BEST_PARAMS); lgb.fit(X_tr, y_tr)
    explainer = shap.Explainer(lgb); shap_values = explainer(X_te)
    # Handle different shap versions (some return different shapes)
    if hasattr(shap_values, 'values'):
        sv = shap_values.values
    else:
        sv = shap_values
    if sv.ndim == 3:
        sv = sv[:, :, 1]  # Binary classification: take positive class

    plt.figure()
    shap.plots.beeswarm(shap.Explanation(sv, data=X_te.values, feature_names=list(X_te.columns)),
                         show=False, order=list(range(min(TOP_N, len(my_f)))))
    plt.gcf().set_size_inches(18, 5.5)
    plt.gca().set_ylabel('Selected Predictors', fontsize=20, weight='bold')
    plt.gca().set_xlabel('SHAP Values', fontsize=16, weight='bold')
    plt.tight_layout(); plt.savefig(os.path.join(results_dir, 'shap_beeswarm.png'), dpi=150, bbox_inches='tight')
    plt.close(); flush_print("  SHAP saved.")


# ===== MAIN =====
if __name__ == '__main__':
    t_start = time.time()
    flush_print("="*60)
    flush_print("UKB-DRP Final Training (Fast)")
    flush_print("="*60)

    # Load
    mydf = pd.read_csv(PREPROCESSED_CSV)
    mydf['AD_years'] = mydf['AD_years'].fillna(mydf['dementia_years']).clip(lower=-1)
    stroke_mask = (mydf['stroke_years'] < 0) & (mydf['stroke_years'].notna())
    mydf = mydf[~stroke_mask].copy()
    flush_print(f"Loaded: {len(mydf):,} participants (after stroke exclusion)")

    rm_f1 = ['Unnamed: 0', 'eid', 'dementia_status', 'dementia_years',
             'AD_status', 'AD_years', 'VD_status', 'VD_years', 'stroke_status', 'stroke_years']
    rm_f = [c for c in rm_f1 + ['41234-0.0','41259-0.0','41149-0.0','41289-0.0','41218-0.0','41235-0.0','41214-0.0'] if c in mydf.columns]
    X_all = mydf.drop(columns=rm_f)

    # Primary features from SFS
    s04_path = os.path.join(RESULTS_BASE, 'DM_full', 's04_selected_features.csv')
    primary_features = pd.read_csv(s04_path)['Features'].tolist()
    flush_print(f"Features: {primary_features[:TOP_N]}")

    # ===== Train primary (DM_full) + Deploy to others =====
    all_metrics = {}

    # DM_full
    os.makedirs(os.path.join(RESULTS_BASE, 'DM_full'), exist_ok=True)
    y_dm = mydf['dementia_status'].copy()
    m = train_one_target(mydf, X_all, 'DM_full', primary_features, os.path.join(RESULTS_BASE, 'DM_full'))
    all_metrics['DM_full'] = m
    run_shap(mydf, X_all, y_dm, primary_features, os.path.join(RESULTS_BASE, 'DM_full'), 'DM_full')

    # Deploy targets
    for dt in ['DM_10yrs', 'DM_5yrs', 'AD_full', 'AD_10yrs', 'AD_5yrs']:
        deploy_dir = os.path.join(RESULTS_BASE, dt)
        os.makedirs(deploy_dir, exist_ok=True)
        m = train_one_target(mydf, X_all, dt, primary_features, deploy_dir)
        all_metrics[dt] = m

        target_col, years_col, max_years = ALL_TARGETS[dt]
        y_d = mydf[target_col].copy()
        if max_years is not None:
            y_d.loc[(y_d == 1) & (mydf[years_col] > max_years)] = 0
        run_shap(mydf, X_all, y_d, primary_features, deploy_dir, dt)

    # ===== Summary =====
    flush_print("\n" + "="*60)
    flush_print("FINAL SUMMARY")
    flush_print("="*60)
    flush_print(f"{'Model':<15} {'AUC':<13} {'HL-p':<10} {'Sens':<10} {'Spec':<10} {'Cutoff':<10}")
    flush_print("-"*68)
    for name in ['DM_full','DM_10yrs','DM_5yrs','AD_full','AD_10yrs','AD_5yrs']:
        m = all_metrics[name]
        flush_print(f"{name:<15} {m['auc_mean']:.4f}±{m['auc_std']:.4f}  {m['hl_pvalue']:.4f}     "
                     f"{m['sensitivity']:.4f}   {m['specificity']:.4f}   {m['best_cutoff']:.3f}")

    summary = [{k: m[k] for k in ['target','auc_mean','auc_std','hl_pvalue',
                                    'sensitivity','specificity','youden','best_cutoff']}
               for m in all_metrics.values()]
    pd.DataFrame(summary).to_csv(os.path.join(RESULTS_BASE, 'summary_metrics.csv'), index=False)

    flush_print(f"\nTotal time: {(time.time()-t_start)/60:.1f} min")
    flush_print(f"Results: {RESULTS_BASE}")

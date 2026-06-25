#!/usr/bin/env python3
"""
Comprehensive Training Runner for UKB-DRP Reproduction
Chains together s01 → s02 → s03 → s04 → s05 for AD_full model.
"""

import os, sys, warnings, gc
warnings.filterwarnings('ignore')

PROJECT_ROOT = os.environ.get('PROJECT_ROOT',
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)  # So Utility/ modules can be found

# ===== CONFIGURATION =====
DPATH = os.path.join(PROJECT_ROOT, 'local_data') + '/'
PREPROCESSED_CSV = os.path.join(DPATH, 'Preprocessed_Data', 'Preprocessed_Data.csv')
RESULTS_DIR = os.path.join(DPATH, 'Results', 'Results_AD_woHES', 'AD_full')
DATA_DIR = os.path.join(DPATH, 'Data')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 70)
print("UKB-DRP Reproduction: AD_full Model Training Pipeline")
print("=" * 70)
print(f"  Data path: {DPATH}")
print(f"  Results:   {RESULTS_DIR}")
print()

# ===== STEP 0: Prepare FieldID_selected.csv =====
print("[Step 0] Creating Data/FieldID_selected.csv for feature-description mapping...")
import pandas as pd
import numpy as np

# Read feature columns from Preprocessed_Data.csv
full_df_sample = pd.read_csv(PREPROCESSED_CSV, nrows=1)
all_columns = full_df_sample.columns.tolist()
target_cols = {'dementia_status', 'dementia_years', 'AD_status', 'AD_years',
               'VD_status', 'VD_years', 'stroke_status', 'stroke_years'}
feature_cols = [c for c in all_columns if c not in target_cols]

# Create a mapping with empty descriptions (the scripts merge for descriptions)
# The key is to have 'Features' column present so inner joins work
dict_f = pd.DataFrame({'Features': feature_cols})
dict_f.to_csv(os.path.join(DATA_DIR, 'FieldID_selected.csv'), index=False)
print(f"  Created FieldID_selected.csv with {len(feature_cols)} feature names")
print()

# ===== STEP 1: s01_init_feature_importance =====
print("[Step 1] Initial feature importance (s01)...")
import time
t0 = time.time()

from sklearn.model_selection import StratifiedKFold
from collections import Counter
from lightgbm import LGBMClassifier
from Utility.Training_Utilities import *

mydf = pd.read_csv(PREPROCESSED_CSV)
mydf['AD_years'] = mydf['AD_years'].fillna(mydf['dementia_years'])
mydf['AD_years'] = mydf['AD_years'].clip(lower=-1)  # Handle negative years

# Remove outcome columns and HES fields
rm_f1 = ['Unnamed: 0', 'eid', 'dementia_status', 'dementia_years',
         'AD_status', 'AD_years', 'VD_status', 'VD_years',
         'stroke_status', 'stroke_years']
rm_HES = ['41234-0.0', '41259-0.0', '41149-0.0', '41289-0.0',
          '41218-0.0', '41235-0.0', '41214-0.0']
rm_f = [c for c in rm_f1 + rm_HES if c in mydf.columns]

X = mydf.drop(columns=rm_f)
y = mydf['AD_status']
mykf = StratifiedKFold(n_splits=5, random_state=2022, shuffle=True)

my_params = {'n_estimators': 500, 'max_depth': 15, 'num_leaves': 10,
             'subsample': 0.7, 'learning_rate': 0.01, 'colsample_bytree': 0.7}

def normal_imp(mydict):
    mysum = sum(mydict.values())
    if mysum == 0:
        return mydict
    for key in mydict:
        mydict[key] = mydict[key] / mysum
    return mydict

tg_imp_cv = Counter()
tc_imp_cv = Counter()

print(f"  Training with {X.shape[1]} features, {X.shape[0]} samples...")
for fold, (train_idx, test_idx) in enumerate(mykf.split(X, y)):
    X_train, y_train = X.iloc[train_idx, :], y.iloc[train_idx]
    my_lgb = LGBMClassifier(objective='binary', metric='auc',
                             is_unbalance=True, verbosity=-1, seed=2020)
    my_lgb.set_params(**my_params)
    my_lgb.fit(X_train, y_train)

    tg_imp = my_lgb.booster_.feature_importance(importance_type='gain')
    tg_imp = dict(zip(my_lgb.booster_.feature_name(), tg_imp.tolist()))
    tc_imp = my_lgb.booster_.feature_importance(importance_type='split')
    tc_imp = dict(zip(my_lgb.booster_.feature_name(), tc_imp.tolist()))

    tg_imp_cv += Counter(normal_imp(tg_imp))
    tc_imp_cv += Counter(normal_imp(tc_imp))
    print(f"  Fold {fold+1}/5 complete")

tg_df = pd.DataFrame({'Features': list(tg_imp_cv.keys()),
                       'TotalGain_cv': list(tg_imp_cv.values())})
tc_df = pd.DataFrame({'Features': list(tc_imp_cv.keys()),
                       'TotalCover_cv': list(tc_imp_cv.values())})
imp_df = pd.merge(left=tc_df, right=tg_df, how='left')
imp_df.sort_values(by='TotalGain_cv', ascending=False, inplace=True)

# Rename columns to match expected format and save
imp_df.rename(columns={'TotalCover_cv': 'Cover', 'TotalGain_cv': 'Gain'}, inplace=True)
# Add placeholder description columns (original scripts had FieldID_selected.csv with descriptions)
imp_df['Path'] = ''
imp_df['Field'] = imp_df['Features']
imp_df['ValueType'] = ''
imp_df['Units'] = ''
top_f = imp_df['Features']
pos_df = mydf.loc[mydf['AD_status'] == 1]
na_full = [round(mydf[ele].isnull().sum() * 100 / len(mydf), 1) for ele in top_f]
na_pos = [round(pos_df[ele].isnull().sum() * 100 / len(pos_df), 1) for ele in top_f]
s01_out = imp_df.copy()
s01_out['NA_full'] = na_full
s01_out['NA_target'] = na_pos
s01_out.to_csv(os.path.join(RESULTS_DIR, 's01_AD_full.csv'), index=False)
print(f"  s01 complete: {len(top_f)} features ranked. Time: {time.time()-t0:.0f}s")
print(f"  Top 10 features: {top_f[:10].tolist()}")
print()

# Clean up
del mydf, X, y, my_lgb
gc.collect()

# ===== STEP 2: s02_feature_clustering =====
print("[Step 2] Hierarchical clustering & redundancy removal (s02)...")
t0 = time.time()

from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Read top 50 features from s01
my_f_list = pd.read_csv(os.path.join(RESULTS_DIR, 's01_AD_full.csv'))['Features'][:50].tolist()
mydf = pd.read_csv(PREPROCESSED_CSV)
mydf['AD_years'] = mydf['AD_years'].fillna(mydf['dementia_years'])
X = mydf[my_f_list]
y = mydf['AD_status']

# Correlation
corr = np.array(X.corr(method='spearman'))
corr = np.nan_to_num(corr)
corr = (corr + corr.T) / 2
np.fill_diagonal(corr, 1)
distance_matrix = 1 - np.abs(corr)
dist_linkage = hierarchy.ward(squareform(distance_matrix))

# Dendrogram plot
label_map = {f: f for f in my_f_list}
my_label = []
for f in my_f_list:
    label_map_val = label_map.get(f, f)
    if isinstance(label_map_val, str) and len(label_map_val) > 40:
        label_map_val = label_map_val[:40]
    my_label.append(str(label_map_val))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
dendro = hierarchy.dendrogram(dist_linkage, labels=my_label, ax=ax2)
ax2.set_xticklabels(dendro["ivl"], rotation=60, fontsize=8, horizontalalignment='right')
dendro_idx = np.arange(0, len(dendro["ivl"]))
ax1.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
ax1.set_xticks(dendro_idx)
ax1.set_yticks(dendro_idx)
ax1.set_xticklabels(dendro["ivl"], rotation=60, fontsize=8, horizontalalignment='right')
ax1.set_yticklabels(dendro["ivl"], fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 's02_AD_full.png'), dpi=150)

# Feature selection by cluster
cluster_ids = hierarchy.fcluster(dist_linkage, 0.75, criterion="distance")
cluster_id_to_feature_ids = defaultdict(list)
for idx, cluster_id in enumerate(cluster_ids):
    cluster_id_to_feature_ids[cluster_id].append(idx)
selected_idx = [v[0] for v in cluster_id_to_feature_ids.values()]
selected_f = X.columns[selected_idx]
s02_f = list(selected_f)
s02_out = pd.DataFrame({'Features': s02_f})
s02_out['Path'] = ''
s02_out['Field'] = s02_out['Features']
s02_out['ValueType'] = ''
s02_out['Units'] = ''
pos_df = mydf.loc[mydf['AD_status'] == 1]
s02_out['NA_full'] = [round(mydf[ele].isnull().sum() * 100 / len(mydf), 1) for ele in s02_f]
s02_out['NA_target'] = [round(pos_df[ele].isnull().sum() * 100 / len(pos_df), 1) for ele in s02_f]
s02_out.to_csv(os.path.join(RESULTS_DIR, 's02_AD_full.csv'), index=False)
print(f"  s02 complete: {len(selected_f)} features after clustering. Time: {time.time()-t0:.0f}s")
print()

del mydf, X, y
gc.collect()

# ===== STEP 3: s03_final_feature_importance =====
print("[Step 3] Final feature importance (s03)...")
t0 = time.time()

my_f_list = pd.read_csv(os.path.join(RESULTS_DIR, 's02_AD_full.csv'))['Features'].tolist()
mydf = pd.read_csv(PREPROCESSED_CSV)
mydf['AD_years'] = mydf['AD_years'].fillna(mydf['dementia_years'])
X = mydf[my_f_list]
y = mydf['AD_status']
mykf = StratifiedKFold(n_splits=5, random_state=2022, shuffle=True)

tg_imp_cv = Counter()
tc_imp_cv = Counter()

for fold, (train_idx, test_idx) in enumerate(mykf.split(X, y)):
    X_train, y_train = X.iloc[train_idx, :], y.iloc[train_idx]
    my_lgb = LGBMClassifier(objective='binary', metric='auc',
                             is_unbalance=True, verbosity=-1, seed=2020)
    my_lgb.set_params(**my_params)
    my_lgb.fit(X_train, y_train)

    tg_imp = my_lgb.booster_.feature_importance(importance_type='gain')
    tg_imp = dict(zip(my_lgb.booster_.feature_name(), tg_imp.tolist()))
    tc_imp = my_lgb.booster_.feature_importance(importance_type='split')
    tc_imp = dict(zip(my_lgb.booster_.feature_name(), tc_imp.tolist()))

    tg_imp_cv += Counter(normal_imp(tg_imp))
    tc_imp_cv += Counter(normal_imp(tc_imp))
    print(f"  Fold {fold+1}/5 complete")

tg_df = pd.DataFrame({'Features': list(tg_imp_cv.keys()),
                       'TotalGain_cv': list(tg_imp_cv.values())})
tc_df = pd.DataFrame({'Features': list(tc_imp_cv.keys()),
                       'TotalCover_cv': list(tc_imp_cv.values())})
imp_df = pd.merge(left=tc_df, right=tg_df, how='left')
imp_df.sort_values(by='TotalGain_cv', ascending=False, inplace=True)
imp_df['TotalGain_cv'] = imp_df['TotalGain_cv'] / 5
imp_df['TotalCover_cv'] = imp_df['TotalCover_cv'] / 5

imp_df.rename(columns={'TotalCover_cv': 'Cover', 'TotalGain_cv': 'Gain'}, inplace=True)
imp_df['Path'] = ''
imp_df['Field'] = imp_df['Features']
imp_df['ValueType'] = ''
imp_df['Units'] = ''
s03_f = imp_df['Features']
pos_df = mydf.loc[mydf['AD_status'] == 1]
na_full = [round(mydf[ele].isnull().sum() * 100 / len(mydf), 1) for ele in s03_f]
na_pos = [round(pos_df[ele].isnull().sum() * 100 / len(pos_df), 1) for ele in s03_f]
s03_out = imp_df.copy()
s03_out['NA_full'] = na_full
s03_out['NA_target'] = na_pos
s03_out.to_csv(os.path.join(RESULTS_DIR, 's03_AD_full.csv'), index=False)
print(f"  s03 complete: {len(s03_f)} features. Time: {time.time()-t0:.0f}s")
print(f"  Top 10 features: {s03_f[:10].tolist()}")
print()

del mydf, X, y, my_lgb
gc.collect()

# ===== STEP 4: s04_cumulative_AUC =====
print("[Step 4] Cumulative AUC analysis (s04)...")
t0 = time.time()

from sklearn.metrics import roc_auc_score

df_f = pd.read_csv(os.path.join(RESULTS_DIR, 's03_AD_full.csv'))
s04_f_list = df_f['Features'].tolist()
mydf = pd.read_csv(PREPROCESSED_CSV)
mydf['AD_years'] = mydf['AD_years'].fillna(mydf['dementia_years'])
X = mydf[s04_f_list]
y = mydf['AD_status']

tmp_f, AUC_cv_lst = [], []

for f in s04_f_list:
    tmp_f.append(f)
    my_X = X[tmp_f]
    AUC_cv = []
    for train_idx, test_idx in mykf.split(my_X, y):
        X_train, X_test = my_X.iloc[train_idx, :], my_X.iloc[test_idx, :]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        my_lgb = LGBMClassifier(objective='binary', metric='auc',
                                 is_unbalance=True, n_jobs=4,
                                 verbosity=-1, seed=2020)
        my_lgb.set_params(**my_params)
        my_lgb.fit(X_train, y_train)
        y_pred_prob = my_lgb.predict_proba(X_test)[:, 1]
        AUC_cv.append(roc_auc_score(y_test, y_pred_prob))
    tmp_out = np.array([np.mean(AUC_cv), np.std(AUC_cv)] + AUC_cv)
    AUC_cv_lst.append(np.round(tmp_out, 3))

# Build proper column-named DataFrame
auc_col_names = ['AUC_mean', 'AUC_std'] + [f'AUC{i}' for i in range(5)]
AUC_data_rows = []
for i, f in enumerate(tmp_f):
    row_data = [f] + AUC_cv_lst[i].tolist()
    AUC_data_rows.append(row_data)

s04_out = pd.DataFrame(AUC_data_rows, columns=['Features'] + auc_col_names)
s04_out['Path'] = ''
s04_out['Field'] = s04_out['Features']
s04_out['ValueType'] = ''
s04_out['Units'] = ''
s04_out = s04_out[['Features', 'AUC_mean', 'Path', 'Field', 'ValueType', 'Units',
                    'AUC0', 'AUC1', 'AUC2', 'AUC3', 'AUC4', 'AUC_std']]
s04_out.to_csv(os.path.join(RESULTS_DIR, 's04_AD_full.csv'), index=False)

# Find optimal feature count
best_idx = s04_out['AUC_mean'].idxmax()
print(f"  s04 complete: {len(tmp_f)} features evaluated. Time: {time.time()-t0:.0f}s")
print(f"  Best AUC at {best_idx+1} features: {s04_out['AUC_mean'].iloc[best_idx]:.4f}")
print(f"  Top 10 features: {s04_out['Features'][:10].tolist()}")
print()

del mydf, X, y
gc.collect()

# ===== STEP 5: s05_final_model =====
print("[Step 5] Final model with calibration (s05)...")
t0 = time.time()

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score
from Utility.Processing_Utilities import *
from Utility.Training_Utilities import *
from Utility.Evaluation_Utilities import *

top_n = 10
my_f_list = pd.read_csv(os.path.join(RESULTS_DIR, 's04_AD_full.csv'))['Features'][:top_n].tolist()
mydf = pd.read_csv(PREPROCESSED_CSV)
mydf['AD_years'] = mydf['AD_years'].fillna(mydf['dementia_years'])
X = mydf[my_f_list]
y = mydf['AD_status']
mykf = StratifiedKFold(n_splits=5, random_state=2022, shuffle=True)

cutoff_list = [np.round(0.001 + i * 0.001, 3) for i in range(100)]
results_cv = []
obs_array, pred_array = np.zeros((10, 1)), np.zeros((10, 1))
y_test_lst, y_pred_prob_lst = [], []

for fold, (train_idx, test_idx) in enumerate(mykf.split(X, y)):
    X_train, X_test = X.iloc[train_idx, :], X.iloc[test_idx, :]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    my_gbm = LGBMClassifier(objective='binary', is_unbalance=True, n_jobs=4,
                             metric='auc', verbose=-1, seed=2022)
    my_gbm.set_params(**my_params)
    calibrate = CalibratedClassifierCV(my_gbm, method='isotonic', cv=5)
    calibrate.fit(X_train, y_train)
    y_pred_prob = calibrate.predict_proba(X_test)[:, 1]

    obsf, predf = calibration_curve(y_test, y_pred_prob, n_bins=10, strategy='quantile')
    obs_array = np.concatenate([obs_array, extend(obsf, nb_points=10)], axis=1)
    pred_array = np.concatenate([pred_array, extend(predf, nb_points=10)], axis=1)

    results_cv.append(get_full_eval(y_test, y_pred_prob, cutoff_list))
    y_pred_prob_lst.append(y_pred_prob)
    y_test_lst.append(y_test)
    print(f"  Fold {fold+1}/5 complete (AUC={roc_auc_score(y_test, y_pred_prob):.4f})")

# Save predictions
pred_prob_df = pd.DataFrame(y_pred_prob_lst).T
test_df = pd.DataFrame(y_test_lst).T
pred_prob_df.to_csv(os.path.join(RESULTS_DIR, 'pred_prob_cv_df.csv'))
test_df.to_csv(os.path.join(RESULTS_DIR, 'test_cv_df.csv'))

# Average results
final_output = avg_results(results_cv)
final_output.to_csv(os.path.join(RESULTS_DIR, 's05_AD_full.csv'))
print(f"\n  s05 complete. Time: {time.time()-t0:.0f}s")
print(f"\n  Top {top_n} features: {my_f_list}")

# ===== FINAL SUMMARY =====
print("\n" + "=" * 70)
print("FINAL RESULTS: AD_full Model")
print("=" * 70)

# Print key metrics from final evaluation
final_output_display = final_output.iloc[:15, :8]
for idx, row in final_output_display.iterrows():
    vals = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in row]
    print(f"  Cutoff={vals[0]}: AUC={vals[1]} Sens={vals[2]} Spec={vals[3]} Prec={vals[4]} F1={vals[5]} APR={vals[6]}")

print(f"\n  Best features: {my_f_list}")
print()

# Calibration data
obs_array = obs_array[:, 1:]
pred_array = pred_array[:, 1:]
obs_mean = np.round(np.mean(obs_array, axis=1), 4)
pred_mean = np.round(np.mean(pred_array, axis=1), 4)
print("  Calibration (deciles):")
print(f"  Observed:   {obs_mean}")
print(f"  Predicted:  {pred_mean}")

print("\n✅ AD_full training pipeline complete!")
print(f"   All results saved to: {RESULTS_DIR}")

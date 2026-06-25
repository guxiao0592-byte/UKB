#!/usr/bin/env python3
"""
MCI→Dementia 生存模型 — RSF优化 + Lasso特征选择 v3
====================================================
1. Lasso Cox: 用完整的 alpha path 拟合, 选最佳 alpha 对应的特征
2. RSF 参数网格搜索 (n_estimators, max_depth, min_samples_leaf)
3. 最优模型 5-fold CV + 与二分类 LGBM 全对比

用法: python src/training/run_adni_survival_v3.py
"""

import os, sys, warnings, time
import numpy as np; import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index as lci
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
t0 = time.time()

# ═══════════════════════════════ CONFIG ═══════════════════════════════
BASE = os.environ.get('PROJECT_ROOT',
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ADNI_DATA_DIR = os.environ.get('ADNI_DATA_DIR',
    os.path.join(BASE, 'local_data', 'adni'))
DATA_PATH = os.path.join(ADNI_DATA_DIR, 'processed', 'ADNI_baseline_with_time_targets_v2.csv')
RESULTS_DIR = os.path.join(BASE, 'local_data', 'Results_adni', 'survival')
os.makedirs(RESULTS_DIR, exist_ok=True)

N_SPLITS=5; SEED=2022; CLUST_TH=0.65; TOP_FINAL=10

COG_PFX = ['MMSE_','MOCA_','ADAS_','FAQ_','GDS_','NPIQ_','NB_','HACH_']
CDR_COLS = ['CDMEMORY','CDORIENT','CDJUDGE','CDCOMMUN','CDHOME','CDCARE','CDGLOBAL','CDRSB']
ID_COLS = ['PTID','RID','PHASE','VISCODE','VISCODE2','APOE_genotype','subject_id',
           'entry_research_group','PTDOBYY']
TARGET_COLS = [
    'AD_status','MCI_status','dementia_status','DIAGNOSIS','DXNORM','DXNODEP',
    'DXMCI','DXDSEV','DXDDUE','DXAD','DXPARK','DXDEP','DXOTHDEM',
    'AD_3yrs','AD_5yrs','AD_10yrs','AD_status_incident','AD_years',
    'Dementia_3yrs','Dementia_5yrs','Dementia_10yrs','Dementia_status_incident',
    'Dementia_years','converted_to_ad','converted_to_dementia','converted_to_mci',
    'ad_conversion_years','dementia_conversion_years','baseline_diagnosis',
    'last_followup_years','diag_label',
    'censored_ad_3yr','censored_ad_5yr','censored_ad_10yr',
    'censored_dementia_3yr','censored_dementia_5yr','censored_dementia_10yr',
]
SURV_COLS = ['surv_time','surv_event']
IMG_PFX = ('FS_','BSI_','WMH_','AMY_','TAU_','TAUPVC_')

def c_index(yt, ye, yp):
    try: return lci(yt, -yp, ye)
    except: return 0.5

def make_surv_y(yt, ye):
    return np.array([(bool(e), t) for e, t in zip(ye, yt)],
                     dtype=[('event', bool), ('time', float)])

# ═══════════════════════════════ 1. DATA ═══════════════════════════════
print("=" * 70)
print("  Phase 2B v3: RSF Grid Search + Lasso Cox Feature Selection")
print("=" * 70)

df = pd.read_csv(DATA_PATH, low_memory=False)
mci = df[df['baseline_diagnosis'] == 2.0].copy()
mci['surv_time'] = np.where(mci['converted_to_dementia'] == 1,
    mci['dementia_conversion_years'], mci['last_followup_years'])
mci['surv_event'] = mci['converted_to_dementia'].astype(int)
mci_v = mci[mci['surv_time'] > 0].copy()
print(f"\n[1] MCI: {len(mci)} → {len(mci_v)} (excl {(mci['surv_time']==0).sum()} time=0)")
print(f"    {mci_v['surv_event'].sum()} events + {(mci_v['surv_event']==0).sum()} censored")

# Build feature matrix
all_ex = set(ID_COLS + TARGET_COLS + CDR_COLS + SURV_COLS)
feat_all = [c for c in mci_v.columns if c not in all_ex
            and not any(c.startswith(p) for p in COG_PFX)]
X_raw = mci_v[feat_all].copy()
for col in X_raw.columns:
    if X_raw[col].isna().any(): X_raw[col] = X_raw[col].fillna(X_raw[col].median())
X_raw = X_raw.drop(columns=[c for c in X_raw.columns if X_raw[c].nunique() <= 1])
X = X_raw.astype(float)
yt, ye = mci_v['surv_time'].values, mci_v['surv_event'].values.astype(int)
print(f"    Features: {X.shape}")

sc = StandardScaler()
Xs = pd.DataFrame(sc.fit_transform(X), columns=X.columns, index=X.index)
sy_full = make_surv_y(yt, ye)
kf = StratifiedKFold(N_SPLITS, random_state=SEED, shuffle=True)

bio_feats = [c for c in X.columns if not any(c.startswith(p) for p in IMG_PFX)]
img_feats = [c for c in X.columns if any(c.startswith(p) for p in IMG_PFX)]
print(f"    Bio: {len(bio_feats)}, Imaging: {len(img_feats)}")

# ═══════════════════════════════ 2. LASSO COX (AUTO PATH) ═════════════
print(f"\n[2] Lasso Cox with auto alpha path...")
coxnet = CoxnetSurvivalAnalysis(
    n_alphas=50, l1_ratio=0.9,
    alpha_min_ratio='auto', max_iter=100000,
    fit_baseline_model=False)
coxnet.fit(Xs.values, sy_full)

# coef_ shape: (n_alphas, n_features)
# Select best alpha: use the one that minimizes deviance or maximizes CV C-index
print(f"    Alpha path: {len(coxnet.alphas_)} alphas, range [{coxnet.alphas_[-1]:.4f}, {coxnet.alphas_[0]:.4f}]")

# Evaluate each alpha with 5-fold CV to find best
best_alpha_idx = 0; best_ci = 0; ci_by_alpha = []
for idx in range(len(coxnet.alphas_)):
    cf = coxnet.coef_[idx]
    n_nz = (np.abs(cf) > 1e-6).sum()
    if n_nz < 5: continue  # need at least 5 features
    cols = [X.columns[i] for i in range(len(cf)) if abs(cf[i]) > 1e-6]
    X_sub = Xs[cols]
    cis = []
    for tr, te in kf.split(X_sub, ye):
        rsf = RandomSurvivalForest(n_estimators=100, max_depth=5, min_samples_leaf=5,
                                    random_state=SEED, n_jobs=4)
        rsf.fit(X_sub.iloc[tr].values, make_surv_y(yt[tr], ye[tr]))
        cis.append(c_index(yt[te], ye[te], rsf.predict(X_sub.iloc[te].values)))
    avg_ci = np.mean(cis)
    ci_by_alpha.append({'idx': idx, 'alpha': coxnet.alphas_[idx], 'n_features': n_nz, 'c_index': avg_ci})
    # print progress every 10
    if idx % 10 == 0 and ci_by_alpha:
        print(f"    alpha[{idx}]={coxnet.alphas_[idx]:.4f}: {n_nz} feats, CI={avg_ci:.4f}")

ci_df = pd.DataFrame(ci_by_alpha)
if len(ci_df) == 0:
    print("ERROR: No alpha produced >=5 non-zero features. Trying with more alphas...")
    sys.exit(1)

best_row = ci_df.loc[ci_df['c_index'].idxmax()]
best_alpha_idx = best_row['idx'].astype(int)
print(f"\n    Best alpha[{best_alpha_idx}] = {best_row['alpha']:.4f}: "
      f"{best_row['n_features']:.0f} features, CI = {best_row['c_index']:.4f}")

# Get features at best alpha
best_coef = coxnet.coef_[best_alpha_idx]
lasso_selected = [str(X.columns[i]) for i in range(len(best_coef)) if abs(best_coef[i]) > 1e-6]
n_lasso = len(lasso_selected)
n_img_lasso = sum(1 for f in lasso_selected if any(f.startswith(p) for p in IMG_PFX))
n_bio_lasso = n_lasso - n_img_lasso
print(f"    Selected: {n_lasso} features ({n_img_lasso} imaging + {n_bio_lasso} bio)")

# Top 10 by |coef|
order = np.argsort(-np.abs(best_coef))
print(f"\n    Top 10:")
for rank in range(min(10, len(order))):
    i = order[rank]
    if abs(best_coef[i]) > 1e-8:
        cat = '[IMG]' if any(str(X.columns[i]).startswith(p) for p in IMG_PFX) else '[BIO]'
        print(f"      {rank+1:>2}. {cat} {str(X.columns[i]):<40} |coef|={abs(best_coef[i]):.6f}")

ci_df.to_csv(os.path.join(RESULTS_DIR, 'lasso_alpha_selection.csv'), index=False)

# ═══════════════════════════════ 3. RSF GRID SEARCH ═════════════════
print(f"\n[3] RSF grid search on {n_lasso} Lasso-selected features...")
Xl = Xs[lasso_selected]

param_grid = {
    'n_estimators': [100, 200, 500, 1000],
    'max_depth': [3, 5, 8, None],
    'min_samples_leaf': [3, 5, 10, 15],
}

best_params = None; best_ci_val = 0; grid = []
total = 64; cnt = 0
for n_est in param_grid['n_estimators']:
    for md in param_grid['max_depth']:
        for msl in param_grid['min_samples_leaf']:
            cis = []
            for tr, te in kf.split(Xl, ye):
                rsf = RandomSurvivalForest(n_estimators=n_est, max_depth=md,
                    min_samples_leaf=msl, random_state=SEED, n_jobs=4)
                rsf.fit(Xl.iloc[tr].values, make_surv_y(yt[tr], ye[tr]))
                cis.append(c_index(yt[te], ye[te], rsf.predict(Xl.iloc[te].values)))
            ci_val = np.mean(cis)
            grid.append({'n_estimators': n_est, 'max_depth': str(md),
                'min_samples_leaf': msl, 'c_index': ci_val, 'std': np.std(cis)})
            if ci_val > best_ci_val:
                best_ci_val = ci_val; best_params = (n_est, md, msl)
            cnt += 1
            if cnt % 16 == 0:
                print(f"    [{cnt}/{total}] best: RSF({best_params[0]},{best_params[1]},{best_params[2]}) CI={best_ci_val:.4f}")

grid_df = pd.DataFrame(grid).sort_values('c_index', ascending=False)
print(f"\n    Best: RSF(n={best_params[0]}, depth={best_params[1]}, leaf={best_params[2]})  CI={best_ci_val:.4f}")
print(f"\n    Top 5:")
for _, r in grid_df.head(5).iterrows():
    print(f"      RSF(n={int(r['n_estimators']):>4}, depth={r['max_depth']:>4}, "
          f"leaf={int(r['min_samples_leaf']):>2})  CI={r['c_index']:.4f}±{r['std']:.4f}")

grid_df.to_csv(os.path.join(RESULTS_DIR, 'rsf_grid_search.csv'), index=False)

# ═══════════════════════════════ 4. FULL 5-FOLD CV ═════════════════
print(f"\n[4] Final model 5-fold CV...")
n_b, d_b, l_b = best_params

fold_cis, tp, tb = [], [], []
for _ in [1, 3, 5]: tp.append([]); tb.append([])

for fi, (tr, te) in enumerate(kf.split(Xl, ye)):
    rsf = RandomSurvivalForest(n_estimators=n_b, max_depth=d_b,
                                min_samples_leaf=l_b, random_state=SEED, n_jobs=4)
    rsf.fit(Xl.iloc[tr].values, make_surv_y(yt[tr], ye[tr]))
    pr = rsf.predict(Xl.iloc[te].values)
    fold_cis.append(c_index(yt[te], ye[te], pr))
    for ti, tev in enumerate([1, 3, 5]):
        yb = ((yt[te] <= tev) & (ye[te] == 1)).astype(int)
        v = ((ye[te] == 1) & (yt[te] <= tev)) | (yt[te] > tev)
        tp[ti].extend(pr[v]); tb[ti].extend(yb[v])

rsf_cidx = np.mean(fold_cis); rsf_std = np.std(fold_cis)
print(f"    C-index = {rsf_cidx:.4f} ± {rsf_std:.4f}")

taucs, tbriers = {}, {}
for ti, tev in enumerate([1, 3, 5]):
    if len(set(tb[ti])) >= 2 and sum(tb[ti]) >= 2:
        taucs[tev] = roc_auc_score(tb[ti], tp[ti])
        pn = (np.array(tp[ti]) - min(tp[ti])) / (max(tp[ti]) - min(tp[ti]) + 1e-8)
        tbriers[tev] = brier_score_loss(tb[ti], pn)
        print(f"    tAUC@{tev}yr = {taucs[tev]:.4f},  Brier = {tbriers[tev]:.4f},  n = {len(tb[ti])}")

# ── Cox PH ──
try:
    cph_df = Xl.copy(); cph_df['time'] = yt; cph_df['event'] = ye
    lv = [c for c in Xl.columns if Xl[c].std() < 0.001]
    if lv: cph_df = cph_df.drop(columns=lv)
    cph = CoxPHFitter(penalizer=0.1); cph.fit(cph_df, 'time', 'event')
    cox_ci_val = cph.concordance_index_
    print(f"\n    Cox PH C-index = {cox_ci_val:.4f}")
except Exception as e:
    print(f"\n    Cox PH failed: {e}"); cox_ci_val = np.nan

# ═══════════════════════════════ 5. ABLATION ═════════════════════
print(f"\n[5] Feature ablation...")
abl = []
for max_k in [3, 5, 10, 15, 20, 30, min(50, n_lasso)]:
    top_k = lasso_selected[:max_k]
    Xk = Xs[top_k]
    cis = []
    for tr, te in kf.split(Xk, ye):
        rsf = RandomSurvivalForest(n_estimators=n_b, max_depth=d_b,
                                    min_samples_leaf=l_b, random_state=SEED, n_jobs=4)
        rsf.fit(Xk.iloc[tr].values, make_surv_y(yt[tr], ye[tr]))
        cis.append(c_index(yt[te], ye[te], rsf.predict(Xk.iloc[te].values)))
    c_avg = np.mean(cis)
    n_img = sum(1 for f in top_k if any(f.startswith(p) for p in IMG_PFX))
    abl.append({'k': max_k, 'c_index': c_avg, 'n_imaging': n_img})
    print(f"    k={max_k:>3}: CI={c_avg:.4f} ({n_img} imaging)")

abl_df = pd.DataFrame(abl)
best_k = int(abl_df.loc[abl_df['c_index'].idxmax(), 'k'])
abl_df.to_csv(os.path.join(RESULTS_DIR, 'feature_ablation.csv'), index=False)

# ═══════════════════════════════ 6. SUMMARY ═══════════════════════════
print(f"\n{'='*70}")
print(f"  FINAL RESULTS")
print(f"{'='*70}")

print(f"\n  {'Model':<24} {'C-index':<10} {'AUC@1yr':<9} {'AUC@3yr':<9} {'AUC@5yr':<9}")
print(f"  {'─'*24} {'─'*10} {'─'*9} {'─'*9} {'─'*9}")
a1 = taucs.get(1, np.nan); a3 = taucs.get(3, np.nan); a5 = taucs.get(5, np.nan)
for name, ci, au1, au3, au5 in [
    ('RSF v3 (Lasso+Grid)',  rsf_cidx, a1, a3, a5),
    ('RSF v1 (SFS, 0.745)',  0.7448, 0.7295, 0.7914, 0.8132),
    ('Cox PH',               cox_ci_val, np.nan, np.nan, np.nan),
    ('Binary LGBM 3yr',      np.nan, np.nan, 0.8336, np.nan),
    ('Binary LGBM 5yr',      np.nan, np.nan, np.nan, 0.8551),
]:
    ci_s = f"{ci:.4f}" if not np.isnan(ci) else "—"
    au1s = f"{au1:.4f}" if not np.isnan(au1) else "—"
    au3s = f"{au3:.4f}" if not np.isnan(au3) else "—"
    au5s = f"{au5:.4f}" if not np.isnan(au5) else "—"
    print(f"  {name:<24} {ci_s:<10} {au1s:<9} {au3s:<9} {au5s:<9}")

dv1 = rsf_cidx - 0.7448
dv3 = a3 - 0.8336 if not np.isnan(a3) else np.nan
dv5 = a5 - 0.8551 if not np.isnan(a5) else np.nan
print(f"\n  Δ vs RSF v1:  {dv1:+.4f}")
if not np.isnan(dv3): print(f"  Δ vs Binary @3yr: {dv3:+.4f}")
if not np.isnan(dv5): print(f"  Δ vs Binary @5yr: {dv5:+.4f}")

print(f"\n  Lasso selected: {n_lasso} features ({n_img_lasso} imaging + {n_bio_lasso} bio)")
print(f"  Best RSF: n={n_b}, depth={d_b}, leaf={l_b}")

pd.DataFrame({
    'model': ['RSF v3','RSF v1','Cox PH','Binary 3yr','Binary 5yr'],
    'c_index': [rsf_cidx, 0.7448, cox_ci_val, np.nan, np.nan],
    'auc_1yr': [a1, 0.7295, np.nan, np.nan, np.nan],
    'auc_3yr': [a3, 0.7914, np.nan, 0.8336, np.nan],
    'auc_5yr': [a5, 0.8132, np.nan, np.nan, 0.8551],
    'brier_3yr': [tbriers.get(3,np.nan), 0.1767, np.nan, np.nan, np.nan],
    'brier_5yr': [tbriers.get(5,np.nan), 0.2770, np.nan, np.nan, np.nan],
}).to_csv(os.path.join(RESULTS_DIR, 'model_comparison.csv'), index=False)

# Save selected features
feat_df = pd.DataFrame({
    'feature': [str(X.columns[i]) for i in order[:50] if abs(best_coef[i]) > 1e-8],
    'coef': [best_coef[i] for i in order[:50] if abs(best_coef[i]) > 1e-8],
    'is_imaging': [any(str(X.columns[i]).startswith(p) for p in IMG_PFX) for i in order[:50] if abs(best_coef[i]) > 1e-8],
})
feat_df.to_csv(os.path.join(RESULTS_DIR, 'lasso_selected_features.csv'), index=False)

# ═══════════════════════════════ 7. PLOTS ═══════════════════════════
print(f"\n[7] Plots...")
rsf_all = RandomSurvivalForest(n_estimators=n_b, max_depth=d_b, min_samples_leaf=l_b,
                                random_state=SEED, n_jobs=4)
rsf_all.fit(Xl.values, sy_full)
all_risk = rsf_all.predict(Xl.values)

# -- KM --
risk_g = pd.qcut(all_risk, 3, labels=['Low','Med','High'])
fig, ax = plt.subplots(figsize=(10, 6))
for lb, c in [('Low', '#10B981'), ('Med', '#F59E0B'), ('High', '#EF4444')]:
    m = risk_g == lb
    kmf = KaplanMeierFitter(); kmf.fit(yt[m], ye[m], label=f'{lb} (n={m.sum()})')
    kmf.plot_survival_function(ax=ax, color=c, linewidth=2)
ax.set_xlabel('Years'); ax.set_ylabel('Dementia-free probability')
ax.set_title(f'MCI→Dementia: KM by RSF Risk (C={rsf_cidx:.3f})', fontweight='bold')
ax.set_xlim(0, 8); ax.set_ylim(0, 1.05); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR, 'km_by_risk.png'), dpi=150); plt.close()

# -- Ablation --
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(abl_df['k'], abl_df['c_index'], 'o-', color='#3B82F6', lw=2, ms=10, mfc='white', mew=2)
ax2.axhline(y=0.7448, color='#EF4444', ls='--', alpha=0.5, label='RSF v1 (SFS)')
ax2.set_xlabel('Number of features'); ax2.set_ylabel('C-index')
ax2.set_title('Feature Ablation', fontweight='bold'); ax2.legend(); ax2.grid(alpha=0.3)
fig2.tight_layout(); fig2.savefig(os.path.join(RESULTS_DIR, 'feature_ablation.png'), dpi=150); plt.close()

# -- Lasso coef plot --
n_show = min(30, len(order))
top_names = [str(X.columns[order[i]]) for i in range(n_show) if abs(best_coef[order[i]]) > 1e-8]
top_vals = [abs(best_coef[order[i]]) for i in range(n_show) if abs(best_coef[order[i]]) > 1e-8]
colors = ['#3B82F6' if any(n.startswith(p) for p in IMG_PFX) else '#F59E0B' for n in top_names]
fig3, ax3 = plt.subplots(figsize=(12, 5))
ax3.barh(range(len(top_vals))[::-1], top_vals[::-1], color=colors[::-1])
ax3.set_yticks(range(len(top_vals))[::-1])
ax3.set_yticklabels(top_names[::-1], fontsize=7); ax3.set_xlabel('|Coefficient|')
ax3.set_title('Lasso Cox: Top Features', fontweight='bold')
fig3.tight_layout(); fig3.savefig(os.path.join(RESULTS_DIR, 'lasso_coefficients.png'), dpi=150); plt.close()

# -- Calibration --
fig4, axes = plt.subplots(1, 3, figsize=(15, 5))
for ai, tev in enumerate([1, 3, 5]):
    ax = axes[ai]; ar, ab = [], []
    for fi, (tr, te) in enumerate(kf.split(Xl, ye)):
        rsf_f = RandomSurvivalForest(n_estimators=n_b, max_depth=d_b, min_samples_leaf=l_b,
                                      random_state=SEED, n_jobs=4)
        rsf_f.fit(Xl.iloc[tr].values, make_surv_y(yt[tr], ye[tr]))
        pr = rsf_f.predict(Xl.iloc[te].values)
        pn = (pr - pr.min()) / (pr.max() - pr.min() + 1e-8)
        yb = ((yt[te] <= tev) & (ye[te] == 1)).astype(int)
        v = ((ye[te] == 1) & (yt[te] <= tev)) | (yt[te] > tev)
        ar.extend(pn[v]); ab.extend(yb[v])
    if len(set(ab)) >= 2:
        qs = np.percentile(ar, np.linspace(0, 100, 11)); bc, bo = [], []
        for i in range(len(qs) - 1):
            mk = (np.array(ar) >= qs[i]) & (np.array(ar) < qs[i + 1])
            if mk.sum() >= 5: bc.append(np.mean(np.array(ar)[mk])); bo.append(np.mean(np.array(ab)[mk]))
        if bc: ax.plot(bc, bo, 'o-', color='#3B82F6', lw=2, ms=8)
        ax.plot([0, 1], [0, 1], '--', color='gray', alpha=0.5)
        ax.set_xlabel('Predicted'); ax.set_ylabel('Observed'); ax.set_title(f'{tev}-yr'); ax.grid(alpha=0.3)
fig4.suptitle('Calibration: Optimized RSF', fontweight='bold')
fig4.tight_layout(); fig4.savefig(os.path.join(RESULTS_DIR, 'calibration.png'), dpi=150); plt.close()

elapsed = time.time() - t0
print(f"\n{'='*70}")
print(f"  ✅ Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")
for f in sorted(os.listdir(RESULTS_DIR)):
    p = os.path.join(RESULTS_DIR, f)
    if os.path.isfile(p): print(f"     {f} ({os.path.getsize(p)/1024:.0f} KB)")
print(f"{'='*70}")

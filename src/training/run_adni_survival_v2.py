#!/usr/bin/env python3
"""
RSF 参数优化 + Lasso Cox 特征选择 — Phase 2B 优化版
=====================================================
1. Lasso Cox (CoxnetSurvivalAnalysis) 特征选择 (替代 SFS)
2. RSF 参数网格搜索 (n_estimators, max_depth, min_samples_leaf)
3. 最优模型 5-fold CV 评估 + 与二分类 LGBM 对比

用法: python src/training/run_adni_survival_v2.py
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
DATA_PATH = '/Users/guxiao/Downloads/MCI-AD/思路一/ADNI数据集/processed/ADNI_baseline_with_time_targets_v2.csv'
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE, 'local_data', 'Results_adni', 'survival')
os.makedirs(RESULTS_DIR, exist_ok=True)

N_SPLITS=5; SEED=2022; CLUST_TH=0.70

# Exclusions
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

# ═══════════════════════════════ HELPERS ═══════════════════════════════
def c_index(yt, ye, yp):
    try: return lci(yt, -yp, ye)
    except: return 0.5

def make_surv_y(yt, ye):
    return np.array([(bool(e), t) for e, t in zip(ye, yt)],
                     dtype=[('event', bool), ('time', float)])

def get_t_auc(yt, ye, pred, t_eval):
    """Time-dependent AUC at specific time point."""
    yb = ((yt<=t_eval)&(ye==1)).astype(int)
    valid = ((ye==1)&(yt<=t_eval))|(yt>t_eval)
    if yb[valid].sum() < 2 or (valid.sum()-yb[valid].sum()) < 2:
        return np.nan
    return roc_auc_score(yb[valid], pred[valid])

def get_t_brier(yt, ye, pred, t_eval):
    yb = ((yt<=t_eval)&(ye==1)).astype(int)
    valid = ((ye==1)&(yt<=t_eval))|(yt>t_eval)
    pn = (pred[valid]-pred[valid].min())/(pred[valid].max()-pred[valid].min()+1e-8)
    return brier_score_loss(yb[valid], pn)

# ═══════════════════════════════ 1. DATA ═══════════════════════════════
print("="*70)
print("  Phase 2B v2: RSF Optimization + Lasso Cox Feature Selection")
print("="*70)

df = pd.read_csv(DATA_PATH, low_memory=False)
mci = df[df['baseline_diagnosis']==2.0].copy()
mci['surv_time'] = np.where(mci['converted_to_dementia']==1,
                            mci['dementia_conversion_years'], mci['last_followup_years'])
mci['surv_event'] = mci['converted_to_dementia'].astype(int)
mci_v = mci[mci['surv_time']>0].copy()
print(f"\n[1] MCI: {len(mci)}→{len(mci_v)} (excl {(mci['surv_time']==0).sum()} time=0)")
print(f"    {mci_v['surv_event'].sum()} events / {(mci_v['surv_event']==0).sum()} censored")

# Feature matrix
all_ex = set(ID_COLS+TARGET_COLS+CDR_COLS+SURV_COLS)
feat_all = [c for c in mci_v.columns if c not in all_ex and not any(c.startswith(p) for p in COG_PFX)]
X_raw = mci_v[feat_all].copy()
for col in X_raw.columns:
    if X_raw[col].isna().any(): X_raw[col] = X_raw[col].fillna(X_raw[col].median())
con = [c for c in X_raw.columns if X_raw[c].nunique()<=1]
X = X_raw.drop(columns=con)
yt, ye = mci_v['surv_time'].values, mci_v['surv_event'].values.astype(int)
print(f"    Features: {X.shape}")

# Scale and prepare survival array
sc = StandardScaler(); Xs = pd.DataFrame(sc.fit_transform(X), columns=X.columns)
sy_full = make_surv_y(yt, ye)
kf = StratifiedKFold(N_SPLITS, random_state=SEED, shuffle=True)

# Key feature categories
bio_feats = [c for c in Xs.columns if not c.startswith(IMG_PFX)]
img_feats = [c for c in Xs.columns if c.startswith(IMG_PFX)]
print(f"    Bio: {len(bio_feats)}, Imaging: {len(img_feats)}")

# ═══════════════════════════════ 2. LASSO COX FEATURE SELECTION ═════════
print(f"\n[2] Lasso Cox (Coxnet) feature selection...")

# Find best alpha via CV on Coxnet
from sklearn.model_selection import KFold
alphas = np.logspace(-2, 1, 20)  # 0.01 to 10
kf_inner = KFold(n_splits=3, shuffle=True, random_state=SEED)

# Evaluate each alpha with 3-fold CV
alpha_cis = {}
for alpha in alphas:
    cis = []
    for tr, te in kf_inner.split(Xs):
        try:
            coxnet = CoxnetSurvivalAnalysis(l1_ratio=0.9, alphas=[alpha],
                                             max_iter=5000, fit_baseline_model=False)
            coxnet.fit(Xs.iloc[tr].values, make_surv_y(yt[tr], ye[tr]))
            pr = coxnet.predict(Xs.iloc[te].values)
            cis.append(c_index(yt[te], ye[te], pr))
        except:
            cis.append(0.5)
    alpha_cis[alpha] = np.mean(cis)

# Select alpha that maximizes C-index
best_alpha = max(alpha_cis, key=alpha_cis.get)
print(f"    Best alpha = {best_alpha:.4f} (C-index = {alpha_cis[best_alpha]:.4f})")

# Fit Lasso Cox with best alpha on ALL data
coxnet_final = CoxnetSurvivalAnalysis(l1_ratio=0.9, alphas=[best_alpha],
                                       max_iter=5000, fit_baseline_model=False)
coxnet_final.fit(Xs.values, sy_full)
coef = coxnet_final.coef_[0]  # 2D array, take first alpha

# Select features with non-zero coefficients
lasso_selected = [X.columns[i] for i in range(len(coef)) if abs(coef[i]) > 1e-6]
n_lasso = len(lasso_selected)
n_img_lasso = sum(1 for f in lasso_selected if f.startswith(IMG_PFX))
n_bio_lasso = n_lasso - n_img_lasso
print(f"    Lasso selected: {n_lasso} features ({n_img_lasso} imaging + {n_bio_lasso} bio)")

if n_lasso < 5:
    # Too few features: take top N by |coef|
    top_idx = np.argsort(-np.abs(coef))[:30]
    lasso_selected = [X.columns[i] for i in top_idx]
    n_lasso = len(lasso_selected); n_img_lasso = sum(1 for f in lasso_selected if f.startswith(IMG_PFX))
    print(f"    → Expanded to {n_lasso} features ({n_img_lasso} imaging)")

print(f"    Top 10 (by |coef|):")
top_idx = np.argsort(-np.abs(coef))[:10]
for rank, i in enumerate(top_idx):
    cat = '[IMG]' if str(X.columns[i]).startswith(IMG_PFX) else '[BIO]'
    print(f"      {rank+1:>2}. {cat} {str(X.columns[i]):<40} coef={coef[i]:+.4f}")

Xl = Xs[lasso_selected]

# ═══════════════════════════════ 3. RSF PARAMETER GRID SEARCH ═════════
print(f"\n[3] RSF parameter optimization...")

param_grid = {
    'n_estimators': [100, 200, 500, 1000],
    'max_depth': [3, 5, 8, None],
    'min_samples_leaf': [3, 5, 10, 15],
}

best_params = None; best_ci = 0; grid_results = []
total_combos = len(param_grid['n_estimators'])*len(param_grid['max_depth'])*len(param_grid['min_samples_leaf'])
done = 0

for n_est in param_grid['n_estimators']:
    for md in param_grid['max_depth']:
        for msl in param_grid['min_samples_leaf']:
            cis = []
            for tr, te in kf.split(Xl, ye):
                rsf = RandomSurvivalForest(
                    n_estimators=n_est, max_depth=md, min_samples_leaf=msl,
                    random_state=SEED, n_jobs=4)
                rsf.fit(Xl.iloc[tr].values, make_surv_y(yt[tr], ye[tr]))
                cis.append(c_index(yt[te], ye[te], rsf.predict(Xl.iloc[te].values)))
            avg_ci = np.mean(cis)
            grid_results.append({
                'n_estimators': n_est, 'max_depth': str(md),
                'min_samples_leaf': msl, 'c_index': avg_ci,
                'c_index_std': np.std(cis),
            })
            done += 1
            if avg_ci > best_ci:
                best_ci = avg_ci; best_params = (n_est, md, msl)
            if done % 8 == 0:
                print(f"    [{done}/{total_combos}] best so far: RSF({best_params[0]},{best_params[1]},{best_params[2]}) C={best_ci:.4f}")

grid_df = pd.DataFrame(grid_results).sort_values('c_index', ascending=False)
print(f"\n    Best: RSF(n={best_params[0]}, depth={best_params[1]}, leaf={best_params[2]})")
print(f"        C-index = {best_ci:.4f}")
print(f"\n    Top 5 configurations:")
for _, r in grid_df.head(5).iterrows():
    print(f"      RSF(n={int(r['n_estimators']):>4}, depth={r['max_depth']:>4}, "
          f"leaf={int(r['min_samples_leaf']):>2})  "
          f"C={r['c_index']:.4f}±{r['c_index_std']:.4f}")

grid_df.to_csv(os.path.join(RESULTS_DIR,'rsf_grid_search.csv'), index=False)

# ═══════════════════════════════ 4. s05: FULL 5-FOLD CV ═══════════════
print(f"\n[4] Final RSF ({best_params[0]}, {best_params[1]}, {best_params[2]}) — 5-fold CV")

n_best, d_best, l_best = best_params
fold_cis, tp, tb = [], [], []
for _ in [1,3,5]: tp.append([]); tb.append([])

for fi, (tr, te) in enumerate(kf.split(Xl, ye)):
    rsf = RandomSurvivalForest(n_estimators=n_best, max_depth=d_best,
                                min_samples_leaf=l_best, random_state=SEED, n_jobs=4)
    rsf.fit(Xl.iloc[tr].values, make_surv_y(yt[tr], ye[tr]))
    pr = rsf.predict(Xl.iloc[te].values)
    fold_cis.append(c_index(yt[te], ye[te], pr))
    for ti, tev in enumerate([1,3,5]):
        yb = ((yt[te]<=tev)&(ye[te]==1)).astype(int)
        v = ((ye[te]==1)&(yt[te]<=tev))|(yt[te]>tev)
        tp[ti].extend(pr[v]); tb[ti].extend(yb[v])

rsf_cidx = np.mean(fold_cis)
print(f"    C-index = {rsf_cidx:.4f}±{np.std(fold_cis):.4f}")

taucs, tbriers = {}, {}
for ti, tev in enumerate([1,3,5]):
    if len(set(tb[ti]))>=2 and sum(tb[ti])>=2:
        taucs[tev] = roc_auc_score(tb[ti], tp[ti])
        pn = (np.array(tp[ti])-min(tp[ti]))/(max(tp[ti])-min(tp[ti])+1e-8)
        tbriers[tev] = brier_score_loss(tb[ti], pn)
        print(f"    tAUC@{tev}yr = {taucs[tev]:.4f},  Brier = {tbriers[tev]:.4f},  n = {len(tb[ti])}")

# ── Cox PH baseline ──
cph_ok = False
try:
    cph_df = Xl.copy(); cph_df['time']=yt; cph_df['event']=ye
    lv = [c for c in Xl.columns if Xl[c].std()<0.001]
    if lv: cph_df = cph_df.drop(columns=lv)
    cph = CoxPHFitter(penalizer=0.1); cph.fit(cph_df, 'time', 'event')
    cox_ci_val = cph.concordance_index_
    print(f"\n    Cox PH C-index = {cox_ci_val:.4f}")
    hr = cph.summary[['coef','exp(coef)','p']].sort_values('p')
    hr.to_csv(os.path.join(RESULTS_DIR,'cox_ph_summary.csv'))
    cph_ok = True
except Exception as e:
    print(f"\n    Cox PH failed: {e}"); cox_ci_val = np.nan

# ═══════════════════════════════ 5. ABLATION: MODEL SIZE ═════════════
print(f"\n[5] Ablation: feature count vs C-index...")
abl_results = []
for max_k in [3, 5, 10, 15, 20, min(30, n_lasso)]:
    top_k = lasso_selected[:max_k]
    Xk = Xs[top_k]
    cis = []
    for tr, te in kf.split(Xk, ye):
        rsf = RandomSurvivalForest(n_estimators=n_best, max_depth=d_best,
                                    min_samples_leaf=l_best, random_state=SEED, n_jobs=4)
        rsf.fit(Xk.iloc[tr].values, make_surv_y(yt[tr], ye[tr]))
        cis.append(c_index(yt[te], ye[te], rsf.predict(Xk.iloc[te].values)))
    ci_val = np.mean(cis)
    n_img = sum(1 for f in top_k if f.startswith(IMG_PFX))
    abl_results.append({'k': max_k, 'c_index': ci_val, 'n_imaging': n_img})
    print(f"    Top {max_k:>2}: C={ci_val:.4f} ({n_img} imaging)")

abl_df = pd.DataFrame(abl_results)
best_k = abl_df.loc[abl_df['c_index'].idxmax(), 'k'].astype(int)
best_k_ci = abl_df.loc[abl_df['c_index'].idxmax(), 'c_index']
print(f"    Optimal: k={best_k}, C-index={best_k_ci:.4f}")

# ═══════════════════════════════ 6. SUMMARY ═══════════════════════════
print(f"\n{'='*70}")
print(f"  FINAL RESULTS — RSF Optimized + Lasso Cox")
print(f"{'='*70}")

print(f"\n  {'Model':<24} {'C-index':<10} {'AUC@1yr':<9} {'AUC@3yr':<9} {'AUC@5yr':<9}")
print(f"  {'─'*24} {'─'*10} {'─'*9} {'─'*9} {'─'*9}")
for name, ci, a1, a3, a5 in [
    ('RSF v2 (optimized)', rsf_cidx, taucs.get(1,np.nan), taucs.get(3,np.nan), taucs.get(5,np.nan)),
    ('RSF v1 (baseline)',  0.7448, 0.7295, 0.7914, 0.8132),
    ('Cox PH',             cox_ci_val, np.nan, np.nan, np.nan),
    ('Binary LGBM',        np.nan, np.nan, 0.8336, 0.8551),
]:
    ci_s = f"{ci:.4f}" if not np.isnan(ci) else "—"
    print(f"  {name:<24} {ci_s:<10} {a1:.4f}    {a3:.4f}    {a5:.4f}"
          if not np.isnan(a1) else
          f"  {name:<24} {ci_s:<10} {'—':<9} {'—':<9} {'—':<9}")

delta3 = taucs.get(3,np.nan) - 0.8336
delta5 = taucs.get(5,np.nan) - 0.8551
print(f"\n  Δ vs Binary @3yr: {delta3:+.4f}  {'(survival better)' if delta3>0 else '(binary better)'}")
print(f"  Δ vs Binary @5yr: {delta5:+.4f}  {'(survival better)' if delta5>0 else '(binary better)'}")
print(f"  Δ vs RSF v1:      {rsf_cidx - 0.7448:+.4f}")

print(f"\n  Lasso Cox selected: {n_lasso} features ({n_img_lasso} imaging + {n_bio_lasso} bio)")
print(f"  Optimal k: {best_k} (C-index = {best_k_ci:.4f})")
print(f"  Best RSF params: n_estimators={n_best}, max_depth={d_best}, min_samples_leaf={l_best}")

# Top 10 features by Lasso |coef|
print(f"\n  Top 10 features (Lasso Cox |coef|):")
for rank, i in enumerate(top_idx[:10]):
    cat = '[IMG]' if str(X.columns[i]).startswith(IMG_PFX) else '[BIO]'
    print(f"    {rank+1:>2}. {cat} {str(X.columns[i]):<40} |coef|={abs(coef[i]):.4f}")

# ═══════════════════════════════ 7. SAVE ═══════════════════════════════
pd.DataFrame({
    'model':['RSF v2 (optimized)','RSF v1 (baseline)','Cox PH','Binary 3yr','Binary 5yr'],
    'c_index':[rsf_cidx,0.7448,cox_ci_val,np.nan,np.nan],
    'auc_1yr':[taucs.get(1,np.nan),0.7295,np.nan,np.nan,np.nan],
    'auc_3yr':[taucs.get(3,np.nan),0.7914,np.nan,0.8336,np.nan],
    'auc_5yr':[taucs.get(5,np.nan),0.8132,np.nan,np.nan,0.8551],
    'brier_3yr':[tbriers.get(3,np.nan),0.1767,np.nan,np.nan,np.nan],
    'brier_5yr':[tbriers.get(5,np.nan),0.2770,np.nan,np.nan,np.nan],
}).to_csv(os.path.join(RESULTS_DIR,'model_comparison.csv'), index=False)

pd.DataFrame({'feature':lasso_selected,'coef':coef[np.argsort(-np.abs(coef))],
    'is_imaging':[f.startswith(IMG_PFX) for f in lasso_selected]}).to_csv(
    os.path.join(RESULTS_DIR,'lasso_selected_features.csv'), index=False)

abl_df.to_csv(os.path.join(RESULTS_DIR,'feature_ablation.csv'), index=False)

# ═══════════════════════════════ 8. PLOTS ═══════════════════════════════
print(f"\n[8] Plots...")

# 8a. KM by risk tertile (best model)
rsf_all = RandomSurvivalForest(n_estimators=n_best, max_depth=d_best,
                                min_samples_leaf=l_best, random_state=SEED, n_jobs=4)
rsf_all.fit(Xl.values, sy_full); all_risk = rsf_all.predict(Xl.values)
risk_g = pd.qcut(all_risk, 3, labels=['Low','Med','High'])
fig,ax=plt.subplots(figsize=(10,6))
for lb,c in [('Low','#10B981'),('Med','#F59E0B'),('High','#EF4444')]:
    m=risk_g==lb; kmf=KaplanMeierFitter()
    kmf.fit(yt[m],ye[m],label=f'{lb} (n={m.sum()})')
    kmf.plot_survival_function(ax=ax,color=c,linewidth=2)
ax.set_xlabel('Years'); ax.set_ylabel('Dementia-free probability')
ax.set_title(f'MCI→Dementia: KM by Optimized RSF Risk Tertile (C={rsf_cidx:.3f})', fontweight='bold')
ax.set_xlim(0,8); ax.set_ylim(0,1.05); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR,'km_by_risk.png'),dpi=150); plt.close()

# 8b. Feature ablation curve
fig2,ax2=plt.subplots(figsize=(10,6))
ax2.plot(abl_df['k'],abl_df['c_index'],'o-',color='#3B82F6',lw=2,ms=10,mfc='white',mew=2)
for _,r in abl_df.iterrows():
    ax2.text(r['k'],r['c_index']-0.003,str(int(r['n_imaging'])),fontsize=8,ha='center')
ax2.axhline(y=0.7448,color='#EF4444',linestyle='--',alpha=0.5,label='RSF v1 (SFS)')
ax2.axvline(x=best_k,color='#10B981',linestyle=':',alpha=0.5,label=f'Optimal k={best_k}')
ax2.set_xlabel('Number of features (ordered by |coef|)')
ax2.set_ylabel('C-index'); ax2.legend()
ax2.set_title('Feature Ablation: Lasso Cox Top-k',fontweight='bold'); ax2.grid(alpha=0.3)
fig2.tight_layout(); fig2.savefig(os.path.join(RESULTS_DIR,'feature_ablation.png'),dpi=150); plt.close()

# 8c. Lasso coefficient plot
fig3,ax3=plt.subplots(figsize=(12,5))
top30 = np.argsort(-np.abs(coef))[:30]
top30_names = [X.columns[i] for i in top30]
top30_vals = [coef[i] for i in top30]
colors = ['#3B82F6' if f.startswith(IMG_PFX) else '#F59E0B' for f in top30_names]
ax3.barh(range(len(top30_names))[::-1], [abs(v) for v in top30_vals[::-1]], color=colors[::-1])
ax3.set_yticks(range(len(top30_names))[::-1])
ax3.set_yticklabels([f"{'[IMG]' if n.startswith(IMG_PFX) else '[BIO]'} {n[:30]}"
                      for n in top30_names[::-1]], fontsize=7)
ax3.set_xlabel('|Coefficient|'); ax3.set_title('Lasso Cox: Top 30 Features',fontweight='bold')
fig3.tight_layout(); fig3.savefig(os.path.join(RESULTS_DIR,'lasso_coefficients.png'),dpi=150); plt.close()

# 8d. Calibration
fig4,axes=plt.subplots(1,3,figsize=(15,5))
for ai,tev in enumerate([1,3,5]):
    ax=axes[ai]; ar,ab=[],[]
    for fi,(tr,te) in enumerate(kf.split(Xl,ye)):
        rsf_f=RandomSurvivalForest(n_estimators=n_best,max_depth=d_best,
                                    min_samples_leaf=l_best,random_state=SEED,n_jobs=4)
        rsf_f.fit(Xl.iloc[tr].values,make_surv_y(yt[tr],ye[tr]))
        pr=rsf_f.predict(Xl.iloc[te].values)
        pn=(pr-pr.min())/(pr.max()-pr.min()+1e-8)
        yb=((yt[te]<=tev)&(ye[te]==1)).astype(int)
        v=((ye[te]==1)&(yt[te]<=tev))|(yt[te]>tev)
        ar.extend(pn[v]); ab.extend(yb[v])
    if len(set(ab))>=2:
        qs=np.percentile(ar,np.linspace(0,100,11)); bc,bo=[],[]
        for i in range(len(qs)-1):
            mk=(np.array(ar)>=qs[i])&(np.array(ar)<qs[i+1])
            if mk.sum()>=5: bc.append(np.mean(np.array(ar)[mk])); bo.append(np.mean(np.array(ab)[mk]))
        if bc: ax.plot(bc,bo,'o-',color='#3B82F6',lw=2,ms=8)
        ax.plot([0,1],[0,1],'--',color='gray',alpha=0.5)
        ax.set_xlabel('Predicted'); ax.set_ylabel('Observed')
        ax.set_title(f'{tev}-yr'); ax.grid(alpha=0.3)
fig4.suptitle('Calibration: Optimized RSF',fontweight='bold')
fig4.tight_layout(); fig4.savefig(os.path.join(RESULTS_DIR,'calibration.png'),dpi=150); plt.close()

# ═══════════════════════════════ DONE ═══════════════════════════════
elapsed = time.time()-t0
print(f"\n{'='*70}")
print(f"  ✅ Done in {elapsed:.1f} sec ({elapsed/60:.1f} min)")
for f in sorted(os.listdir(RESULTS_DIR)):
    p=os.path.join(RESULTS_DIR,f)
    if os.path.isfile(p): print(f"     {f} ({os.path.getsize(p)/1024:.0f} KB)")
print(f"{'='*70}")

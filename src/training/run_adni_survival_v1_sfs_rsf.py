#!/usr/bin/env python3
"""
ADNI MCI→Dementia 生存预测 — Phase 2B
======================================
s01-s05 管线 + 生存标签 (time, event). 用 C-index 取代 AUC 贯穿全流程.

模型引擎:
  - RandomSurvivalForest (scikit-survival) — 主模型, 树模型, 与 LightGBM 哲学一致
  - CoxPHFitter (lifelines) — 可解释性基准, 输出 Hazard Ratio
  - (GB-Cox: LightGBM objective='cox' 在 pip 版本中未编译, 暂不可用)

用法:  python src/training/run_adni_survival.py
"""

import os, sys, warnings, time
import numpy as np; import pandas as pd
from collections import Counter
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index as lifelines_ci
from sksurv.ensemble import RandomSurvivalForest
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
t0 = time.time()

# ═══════════════════════════════ CONFIG ═══════════════════════════════
BASE = os.environ.get('PROJECT_ROOT',
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ADNI_DATA_DIR = os.environ.get('ADNI_DATA_DIR',
    os.path.join(BASE, 'local_data', 'adni'))
DATA_PATH = os.path.join(ADNI_DATA_DIR, 'processed', 'ADNI_baseline_with_time_targets_v2.csv')
RESULTS_DIR = os.path.join(BASE, 'local_data', 'Results_adni', 'survival_v1_backup')
os.makedirs(RESULTS_DIR, exist_ok=True)

N_SPLITS=5; SEED=2022; TOP_S01=50; CLUST_TH=0.45; TOP_SFS=15; TOP_FINAL=10; STOP=0.001

# Exclusions (identical to binary pipeline)
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
    'censored_ad_3yr','censored_ad_5yr','censored_ad_10yr',
    'censored_dementia_3yr','censored_dementia_5yr','censored_dementia_10yr',
    'last_followup_years','diag_label',
]
IMG_PFX = ('FS_','BSI_','WMH_','AMY_','TAU_','TAUPVC_')

# ═══════════════════════════════ HELPERS ═══════════════════════════════
def norm_imp(d):
    s = sum(d.values()); return d if s==0 else {k:v/s for k,v in d.items()}

def c_index(yt, ye, yp):
    try: return lifelines_ci(yt, -yp, ye)
    except: return 0.5

def make_surv_y(yt, ye):
    return np.array([(bool(e), t) for e, t in zip(ye, yt)],
                     dtype=[('event', bool), ('time', float)])

def univariate_ci(X_col, yt, ye):
    """C-index for a single feature (higher risk = higher value)."""
    try:
        if np.isnan(X_col).any() or np.all(X_col == X_col[0]):
            return 0.5
        return lifelines_ci(yt, X_col, ye)
    except:
        return 0.5

# ═══════════════════════════════ 1. SURVIVAL LABELS ═══════════════════
print("="*70)
print("  Phase 2B: MCI→Dementia Survival Prediction")
print("="*70)

df = pd.read_csv(DATA_PATH, low_memory=False)
mci = df[df['baseline_diagnosis']==2.0].copy()
mci['surv_time'] = np.where(mci['converted_to_dementia']==1,
                            mci['dementia_conversion_years'], mci['last_followup_years'])
mci['surv_event'] = mci['converted_to_dementia'].astype(int)
n_exc = (mci['surv_time']==0).sum()
mci_v = mci[mci['surv_time']>0].copy()
print(f"\n[1] MCI: {len(mci)}→{len(mci_v)} (excl {n_exc} time=0)")
print(f"    {mci_v['surv_event'].sum()} events (median {mci_v.loc[mci_v['surv_event']==1,'surv_time'].median():.1f}yr) + "
      f"{(mci_v['surv_event']==0).sum()} censored (median {mci_v.loc[mci_v['surv_event']==0,'surv_time'].median():.1f}yr)")
mci_v[['RID','surv_time','surv_event']].to_csv(os.path.join(RESULTS_DIR,'survival_labels.csv'), index=False)

# ═══════════════════════════════ 2. FEATURES ═══════════════════════════
# !!! CRITICAL: exclude surv_time/surv_event from features
SURV_COLS = ['surv_time','surv_event']
all_ex = set(ID_COLS+TARGET_COLS+CDR_COLS+SURV_COLS)
feat_all = [c for c in mci_v.columns if c not in all_ex and not any(c.startswith(p) for p in COG_PFX)]
X_raw = mci_v[[c for c in feat_all if c in mci_v.columns]].copy()
# Fill NaN then drop constant
for col in X_raw.columns:
    if X_raw[col].isna().any():
        X_raw[col] = X_raw[col].fillna(X_raw[col].median())
con = [c for c in X_raw.columns if X_raw[c].nunique()<=1]
X = X_raw.drop(columns=con)
yt, ye = mci_v['surv_time'].values, mci_v['surv_event'].values.astype(int)
print(f"\n[2] Features: {X.shape} (dropped {len(con)} constant)")

# Scale for Cox PH
sc = StandardScaler(); Xs = pd.DataFrame(sc.fit_transform(X), columns=X.columns, index=X.index)
kf = StratifiedKFold(N_SPLITS, random_state=SEED, shuffle=True)

# ═══════════════════════════════ 3. s01: UNIVARIATE C-INDEX RANKING ═══
print(f"\n[3] s01: Univariate C-index ranking...")
uni_cis = {}
for col in Xs.columns:
    uni_cis[col] = univariate_ci(Xs[col].values, yt, ye)
top50 = sorted(uni_cis, key=lambda k: -uni_cis[k])[:TOP_S01]
n_img = sum(1 for f in top50 if f.startswith(IMG_PFX))
print(f"    Top50 ({n_img} imaging), #1={top50[0]} (CI={uni_cis[top50[0]]:.4f})")

# ═══════════════════════════════ 4. s02: WARD CLUSTERING ═══════════════
Xt = Xs[top50].fillna(0); corr = np.array(Xt.corr('spearman'))
corr = np.nan_to_num(corr); corr = (corr+corr.T)/2; np.fill_diagonal(corr, 1.0)
cl = fcluster(linkage(squareform(1-np.abs(corr)), 'ward'), t=CLUST_TH, criterion='distance')
kept = []; seen = set()
for f, c in zip(top50, cl):
    if c not in seen: kept.append(f); seen.add(c)
Xk = Xs[kept]
print(f"    s02: {len(top50)}→{len(kept)} after clustering")

# ═══════════════════════════════ 5. s03: RE-RANK ═══════════════════════
uni_cis2 = {col: univariate_ci(Xk[col].values, yt, ye) for col in Xk.columns}
s03f = sorted(uni_cis2, key=lambda k: -uni_cis2[k])
print(f"    s03: Top5={s03f[:3]}")

# ═══════════════════════════════ 6. s04: SFS WITH C-INDEX ═══════════
print(f"    s04: SFS (C-index, RSF)...")
selected = []; remaining = list(s03f); prev_ci = 0; sfs_hist = []
# Build full survival array once
sy_full = make_surv_y(yt, ye)

for step in range(min(TOP_SFS, len(s03f))):
    best_f, best_ci = None, -1
    pool = remaining[:min(20, len(remaining))]
    for cand in pool:
        trial = selected + [cand]; cis = []
        for tr, te in kf.split(Xk[trial], ye):
            rsf = RandomSurvivalForest(
                n_estimators=100, max_depth=6, min_samples_leaf=5,
                random_state=SEED, n_jobs=4)
            rsf.fit(Xk[trial].iloc[tr].values, make_surv_y(yt[tr], ye[tr]))
            pr = rsf.predict(Xk[trial].iloc[te].values)
            cis.append(c_index(yt[te], ye[te], pr))
        avg = np.mean(cis)
        if avg > best_ci: best_f, best_ci = cand, avg
    gain = best_ci - prev_ci if step>0 else best_ci
    isi = best_f.startswith(IMG_PFX)
    sfs_hist.append({'step':step+1,'feature':best_f,'c_index':best_ci,'gain':gain,'is_imaging':isi})
    print(f"      Step{step+1}: {'[IMG]' if isi else '[BIO]'} {best_f}  "
          f"C={best_ci:.4f} Δ={gain:+.4f}")
    selected.append(best_f); remaining.remove(best_f); prev_ci = best_ci
    if step>=TOP_FINAL-1 and gain<STOP: break

top_f = selected[:TOP_FINAL]
n_img_f = sum(1 for f in top_f if f.startswith(IMG_PFX))
print(f"    Top10 ({n_img_f} imaging)")
pd.DataFrame(sfs_hist).to_csv(os.path.join(RESULTS_DIR,'sfs_history.csv'), index=False)

# ═══════════════════════════════ 7. s05: RSF + Cox PH ═══════════════
print(f"\n[7] s05: RSF (main) + Cox PH (baseline) 5-fold CV")
Xf = Xk[top_f]
# Re-scale
sc2 = StandardScaler(); Xf_s = pd.DataFrame(sc2.fit_transform(Xf), columns=Xf.columns, index=Xf.index)

# ── RSF (main) ──
rsf_cis, tp, tb = [], [], []
# Buffer for time-dependent metrics
for _ in [1,3,5]: tp.append([]); tb.append([])

for fi, (tr, te) in enumerate(kf.split(Xf_s, ye)):
    rsf = RandomSurvivalForest(n_estimators=200, max_depth=5, min_samples_leaf=5,
                                random_state=SEED, n_jobs=4)
    rsf.fit(Xf_s.iloc[tr].values, make_surv_y(yt[tr], ye[tr]))
    pr = rsf.predict(Xf_s.iloc[te].values)
    rsf_cis.append(c_index(yt[te], ye[te], pr))

    for ti, t_eval in enumerate([1,3,5]):
        yb = ((yt[te]<=t_eval)&(ye[te]==1)).astype(int)
        v = ((ye[te]==1)&(yt[te]<=t_eval))|(yt[te]>t_eval)
        tp[ti].extend(pr[v]); tb[ti].extend(yb[v])

rsf_cidx = np.mean(rsf_cis); rsf_std = np.std(rsf_cis)
print(f"    RSF (200 trees): C-index = {rsf_cidx:.4f}±{rsf_std:.4f}")

taucs, tbriers = {}, {}
for ti, tev in enumerate([1,3,5]):
    if len(set(tb[ti]))>=2 and sum(tb[ti])>=2:
        taucs[tev] = roc_auc_score(tb[ti], tp[ti])
        pn = (np.array(tp[ti])-min(tp[ti]))/(max(tp[ti])-min(tp[ti])+1e-8)
        tbriers[tev] = brier_score_loss(tb[ti], pn)
        print(f"    tAUC@{tev}yr = {taucs[tev]:.4f}, Brier = {tbriers[tev]:.4f}, n = {len(tb[ti])}")

# ── Cox PH (baseline) ──
print(f"\n    Cox PH...")
try:
    cph_df = Xf_s.copy(); cph_df['time']=yt; cph_df['event']=ye
    lv = [c for c in Xf_s.columns if Xf_s[c].std()<0.001]
    if lv: cph_df=cph_df.drop(columns=lv)
    cph = CoxPHFitter(penalizer=0.1); cph.fit(cph_df, 'time', 'event')
    cox_ci_val = cph.concordance_index_
    print(f"    C-index = {cox_ci_val:.4f}")

    # Extract HR for top features
    hr = cph.summary[['coef','exp(coef)','p']].sort_values('p')
    hr.to_csv(os.path.join(RESULTS_DIR,'cox_ph_summary.csv'))
    print(f"    Top significant: {list(hr.index[:5])}")
except Exception as e:
    print(f"    Failed: {e}"); cox_ci_val = np.nan

# ═══════════════════════════════ 8. SUMMARY ═══════════════════════════
print(f"\n{'='*70}")
print(f"  SURVIVAL MODEL RESULTS")
print(f"{'='*70}")

print(f"\n  {'Model':<18} {'C-index':<12} {'AUC@1yr':<10} {'AUC@3yr':<10} {'AUC@5yr':<10}")
print(f"  {'─'*18} {'─'*12} {'─'*10} {'─'*10} {'─'*10}")
for name, ci, a1, a3, a5 in [
    ('RSF (main)',  rsf_cidx, taucs.get(1,np.nan), taucs.get(3,np.nan), taucs.get(5,np.nan)),
    ('Cox PH',      cox_ci_val, np.nan, np.nan, np.nan),
    ('Binary LGBM', np.nan, np.nan, 0.8336, 0.8551),
]:
    ci_s = f"{ci:.4f}" if not np.isnan(ci) else "—"
    print(f"  {name:<18} {ci_s:<12} {a1:.4f}     {a3:.4f}     {a5:.4f}"
          if not np.isnan(a1) else
          f"  {name:<18} {ci_s:<12} {'—':<10} {'—':<10} {'—':<10}")

print(f"\n  RSF vs Binary 3yr:  ΔtAUC = {taucs.get(3,np.nan)-0.8336:+.4f}" if not np.isnan(taucs.get(3,np.nan)) else "")
print(f"  RSF vs Binary 5yr:  ΔtAUC = {taucs.get(5,np.nan)-0.8551:+.4f}" if not np.isnan(taucs.get(5,np.nan)) else "")

print(f"\n  Top 10 survival features:")
for i,f in enumerate(top_f[:10]):
    print(f"    {i+1:>2}. {'[IMG]' if f.startswith(IMG_PFX) else '[BIO]'} {f}")

# Save all results
pd.DataFrame({'model':['RSF','Cox PH','Binary 3yr','Binary 5yr'],
    'c_index':[rsf_cidx, cox_ci_val, np.nan, np.nan],
    'auc_1yr':[taucs.get(1,np.nan),np.nan,np.nan,np.nan],
    'auc_3yr':[taucs.get(3,np.nan),np.nan,0.8336,np.nan],
    'auc_5yr':[taucs.get(5,np.nan),np.nan,np.nan,0.8551],
    'brier_3yr':[tbriers.get(3,np.nan),np.nan,np.nan,np.nan],
    'brier_5yr':[tbriers.get(5,np.nan),np.nan,np.nan,np.nan],
}).to_csv(os.path.join(RESULTS_DIR,'model_comparison.csv'), index=False)
pd.DataFrame({'rank':range(1,len(top_f)+1),'feature':top_f}).to_csv(
    os.path.join(RESULTS_DIR,'selected_features.csv'), index=False)

# ═══════════════════════════════ 9. PLOTS ═══════════════════════════
print(f"\n[9] Plots...")
# Train final RSF on all data
rsf_all = RandomSurvivalForest(n_estimators=300, max_depth=5, min_samples_leaf=5,
                                random_state=SEED, n_jobs=4)
rsf_all.fit(Xf_s.values, sy_full)
all_risk = rsf_all.predict(Xf_s.values)

# 9a. KM by risk tertile
risk_g = pd.qcut(all_risk, 3, labels=['Low','Med','High'])
fig,ax=plt.subplots(figsize=(10,6))
for lb, c in [('Low','#10B981'),('Med','#F59E0B'),('High','#EF4444')]:
    m = risk_g==lb; kmf=KaplanMeierFitter()
    kmf.fit(yt[m], ye[m], label=f'{lb} (n={m.sum()})')
    kmf.plot_survival_function(ax=ax, color=c, linewidth=2)
ax.set_xlabel('Years from MCI baseline'); ax.set_ylabel('Dementia-free probability')
ax.set_title('MCI→Dementia: KM Curves by RSF Risk Tertile', fontweight='bold')
ax.set_xlim(0,8); ax.set_ylim(0,1.05); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS_DIR,'km_by_risk.png'), dpi=150); plt.close()

# 9b. SFS curve
sd = pd.DataFrame(sfs_hist)
fig2,ax2=plt.subplots(figsize=(10,6))
ax2.plot(sd['step'], sd['c_index'], 'o-', color='#3B82F6', lw=2, ms=8, mfc='white', mew=2)
for _,r in sd.iterrows():
    ax2.text(r['step'], r['c_index']+0.005, r['feature'][:18], fontsize=7, ha='center')
ax2.set_xlabel('SFS Step'); ax2.set_ylabel('C-index')
ax2.set_title('SFS Feature Accumulation — RSF C-index', fontweight='bold'); ax2.grid(alpha=0.3)
fig2.tight_layout(); fig2.savefig(os.path.join(RESULTS_DIR,'sfs_accumulation.png'), dpi=150); plt.close()

# 9c. Calibration
fig3,axes=plt.subplots(1,3,figsize=(15,5))
for ai,tev in enumerate([1,3,5]):
    ax=axes[ai]; ar,ab=[],[]
    for fi,(tr,te) in enumerate(kf.split(Xf_s, ye)):
        rsf_f = RandomSurvivalForest(n_estimators=200, max_depth=5, min_samples_leaf=5,
                                      random_state=SEED, n_jobs=4)
        rsf_f.fit(Xf_s.iloc[tr].values, make_surv_y(yt[tr],ye[tr]))
        pr=rsf_f.predict(Xf_s.iloc[te].values)
        pn=(pr-pr.min())/(pr.max()-pr.min()+1e-8)
        yb=((yt[te]<=tev)&(ye[te]==1)).astype(int)
        v=((ye[te]==1)&(yt[te]<=tev))|(yt[te]>tev)
        ar.extend(pn[v]); ab.extend(yb[v])
    if len(set(ab))>=2:
        qs=np.percentile(ar, np.linspace(0,100,11)); bc,bo=[],[]
        for i in range(len(qs)-1):
            mk=(np.array(ar)>=qs[i])&(np.array(ar)<qs[i+1])
            if mk.sum()>=5: bc.append(np.mean(np.array(ar)[mk])); bo.append(np.mean(np.array(ab)[mk]))
        if bc: ax.plot(bc,bo,'o-',color='#3B82F6',lw=2,ms=8)
        ax.plot([0,1],[0,1],'--',color='gray',alpha=0.5)
        ax.set_xlabel('Predicted'); ax.set_ylabel('Observed')
        ax.set_title(f'{tev}-year Calibration'); ax.grid(alpha=0.3)
fig3.suptitle('RSF Calibration: Predicted vs Observed Conversion Risk', fontweight='bold')
fig3.tight_layout(); fig3.savefig(os.path.join(RESULTS_DIR,'calibration.png'), dpi=150); plt.close()

# ═══════════════════════════════ DONE ═══════════════════════════════
elapsed = time.time()-t0
print(f"\n{'='*70}")
print(f"  ✅ Done in {elapsed/60:.1f} min")
for f in sorted(os.listdir(RESULTS_DIR)):
    p=os.path.join(RESULTS_DIR,f)
    if os.path.isfile(p): print(f"     {f} ({os.path.getsize(p)/1024:.0f} KB)")
print(f"{'='*70}")

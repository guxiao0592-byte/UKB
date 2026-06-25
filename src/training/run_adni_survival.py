#!/usr/bin/env python3
"""
ADNI MCI→Dementia 生存预测 — Phase 2B v2 (统一 LightGBM 特征选择)
==================================================================
v2 改进: s01-s04 使用 LightGBM 做特征选择 (与 Phase 2A 方法论统一),
        s05 仍使用 RSF + Cox PH 做生存建模.

v1 → v2 变化:
  s01: Univariate C-index → LightGBM Gain ranking (5-fold CV)
  s02: Ward 聚类阈值 0.45 → 0.75 (与 Phase 2A 一致)
  s03: Univariate C-index → LightGBM Gain re-rank
  s04: RSF C-index SFS  → LightGBM AUC SFS
  s05: RSF + Cox PH      → 不变

模型引擎:
  - RandomSurvivalForest (scikit-survival) — 主模型
  - CoxPHFitter (lifelines) — 可解释性基线

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
from lightgbm import LGBMClassifier
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
RESULTS_DIR = os.path.join(BASE, 'local_data', 'Results_adni', 'survival')
os.makedirs(RESULTS_DIR, exist_ok=True)

N_SPLITS = 5; SEED = 2022
TOP_S01 = 50; CLUST_TH = 0.75   # ★ v2: 0.75 (统一 Phase 2A)
TOP_SFS = 15; TOP_FINAL = 10
SFS_EARLY_STOP = 0.0005

# LightGBM params — exactly matching Phase 2A (run_adni_mci_dementia.py)
LGB_PARAMS = dict(
    n_estimators=500, max_depth=15, num_leaves=10,
    subsample=0.7, learning_rate=0.01, colsample_bytree=0.7,
    objective='binary', is_unbalance=True, metric='auc',
    verbosity=-1, seed=2020, n_jobs=4,
    min_data_in_leaf=5, min_gain_to_split=0.0,
)

# Exclusions — matching Phase 2A
COG_PFX = ['MMSE_', 'MOCA_', 'ADAS_', 'FAQ_', 'GDS_', 'NPIQ_', 'NB_', 'HACH_']
CDR_COLS = ['CDMEMORY', 'CDORIENT', 'CDJUDGE', 'CDCOMMUN',
            'CDHOME', 'CDCARE', 'CDGLOBAL', 'CDRSB']
ID_COLS = ['PTID', 'RID', 'PHASE', 'VISCODE', 'VISCODE2',
           'APOE_genotype', 'subject_id', 'entry_research_group', 'PTDOBYY']
TARGET_COLS = [
    'AD_status', 'MCI_status', 'dementia_status', 'DIAGNOSIS',
    'DXNORM', 'DXNODEP', 'DXMCI', 'DXDSEV',
    'DXDDUE', 'DXAD', 'DXPARK', 'DXDEP', 'DXOTHDEM',
    'AD_3yrs', 'AD_5yrs', 'AD_10yrs', 'AD_status_incident', 'AD_years',
    'Dementia_3yrs', 'Dementia_5yrs', 'Dementia_10yrs', 'Dementia_status_incident',
    'Dementia_years', 'converted_to_ad', 'converted_to_dementia', 'converted_to_mci',
    'ad_conversion_years', 'dementia_conversion_years', 'baseline_diagnosis',
    'censored_ad_3yr', 'censored_ad_5yr', 'censored_ad_10yr',
    'censored_dementia_3yr', 'censored_dementia_5yr', 'censored_dementia_10yr',
    'last_followup_years', 'diag_label',
]
IMG_PFX = ('FS_', 'BSI_', 'WMH_', 'AMY_', 'TAU_', 'TAUPVC_')

# ═══════════════════════════════ HELPERS ═══════════════════════════════
def norm_imp(d):
    """Normalize importance scores to sum to 1."""
    s = sum(d.values())
    return d if s == 0 else {k: v / s for k, v in d.items()}

def c_index(yt, ye, yp):
    """Harrell's C-index. yp is risk score (higher = riskier)."""
    try:
        return lifelines_ci(yt, -yp, ye)
    except Exception:
        return 0.5

def make_surv_y(yt, ye):
    """Build structured array for scikit-survival."""
    return np.array([(bool(e), t) for e, t in zip(ye, yt)],
                     dtype=[('event', bool), ('time', float)])

def get_full_eval(y_true, y_pred):
    """Compute comprehensive binary classification metrics."""
    fpr, tpr, thresh = roc_auc_score(y_true, y_pred), None, None
    auc_val = roc_auc_score(y_true, y_pred)
    return {'auc': auc_val}

# ═══════════════════════════════ 1. SURVIVAL LABELS ═══════════════════
print("=" * 70)
print("  Phase 2B v2: MCI→Dementia Survival (Unified LightGBM Feature Selection)")
print("=" * 70)

df = pd.read_csv(DATA_PATH, low_memory=False)
mci = df[df['baseline_diagnosis'] == 2.0].copy()
mci['surv_time'] = np.where(mci['converted_to_dementia'] == 1,
                            mci['dementia_conversion_years'],
                            mci['last_followup_years'])
mci['surv_event'] = mci['converted_to_dementia'].astype(int)

# Exclude subjects with zero follow-up (only baseline visit)
n_exc = (mci['surv_time'] == 0).sum()
mci_v = mci[mci['surv_time'] > 0].copy()
print(f"\n[1] MCI subjects: {len(mci)} → {len(mci_v)} (excluded {n_exc} with time=0)")
print(f"    Events: {mci_v['surv_event'].sum()} "
      f"(median time {mci_v.loc[mci_v['surv_event']==1, 'surv_time'].median():.1f}yr)")
print(f"    Censored: {(mci_v['surv_event']==0).sum()} "
      f"(median f/u {mci_v.loc[mci_v['surv_event']==0, 'surv_time'].median():.1f}yr)")
mci_v[['RID', 'surv_time', 'surv_event']].to_csv(
    os.path.join(RESULTS_DIR, 'survival_labels.csv'), index=False)

# ═══════════════════════════════ 2. FEATURES ═══════════════════════════
SURV_COLS = ['surv_time', 'surv_event']
all_ex = set(ID_COLS + TARGET_COLS + CDR_COLS + SURV_COLS)
feat_all = [c for c in mci_v.columns
            if c not in all_ex
            and not any(c.startswith(p) for p in COG_PFX)]
X_raw = mci_v[[c for c in feat_all if c in mci_v.columns]].copy()

# Median imputation
for col in X_raw.columns:
    if X_raw[col].isna().any():
        X_raw[col] = X_raw[col].fillna(X_raw[col].median())

# Drop constant columns
const_cols = [c for c in X_raw.columns if X_raw[c].nunique() <= 1]
X = X_raw.drop(columns=const_cols)
yt = mci_v['surv_time'].values
ye = mci_v['surv_event'].values.astype(int)
print(f"\n[2] Features: {X.shape[1]} (dropped {len(const_cols)} constant)")

# ★ Binary label for LightGBM feature selection = eventual conversion status
#    This is used ONLY for s01-s04, not for the final survival evaluation
y_binary = ye  # surv_event == converted_to_dementia (0/1)
kf = StratifiedKFold(N_SPLITS, random_state=SEED, shuffle=True)

# ═══════════════════════════════ 3. s01: LIGHTGBM GAIN RANKING ═══════
# ★ v2 CHANGE: LightGBM gain importance (5-fold CV) instead of univariate C-index
#    This captures multivariate interactions, matching Phase 2A methodology.
print(f"\n[3] s01: LightGBM Gain ranking (5-fold CV, n={LGB_PARAMS['n_estimators']})...")
tg_cv = Counter()
for tr, te in kf.split(X, y_binary):
    gbm = LGBMClassifier(**LGB_PARAMS)
    gbm.fit(X.iloc[tr], y_binary[tr])
    tg = dict(zip(gbm.booster_.feature_name(),
                  gbm.booster_.feature_importance(importance_type='gain')))
    tg_cv += Counter(norm_imp(tg))

s01_r = sorted(tg_cv.items(), key=lambda x: -x[1])
top50 = [f for f, _ in s01_r[:min(TOP_S01, len(s01_r))]]
n_img = sum(1 for f in top50 if f.startswith(IMG_PFX))
n_bio = len(top50) - n_img
print(f"    Top 50: {n_img} imaging + {n_bio} bio/other")
print(f"    #1 = {top50[0]} (gain={s01_r[0][1]:.4f})")

# ═══════════════════════════════ 4. s02: WARD CLUSTERING ═══════════════
# ★ v2 CHANGE: Cluster threshold 0.75 (matching Phase 2A), was 0.45 in v1
Xt = X[top50].fillna(X[top50].median()).copy()
corr = np.array(Xt.corr('spearman'))
corr = np.nan_to_num(corr)
corr = (corr + corr.T) / 2
np.fill_diagonal(corr, 1.0)
link = linkage(squareform(1 - np.abs(corr)), method='ward')
clusters = fcluster(link, t=CLUST_TH, criterion='distance')

kept = []; seen = set()
for f, c in zip(top50, clusters):
    if c not in seen:
        kept.append(f)
        seen.add(c)
Xk = X[kept]
n_img_kept = sum(1 for f in kept if f.startswith(IMG_PFX))
print(f"    s02: {len(top50)} → {len(kept)} features after Ward clustering "
      f"(threshold={CLUST_TH}, {n_img_kept} imaging)")

# ═══════════════════════════════ 5. s03: LIGHTGBM RE-RANK ═════════════
# ★ v2 CHANGE: LightGBM re-rank (matching Phase 2A), not univariate C-index
print(f"    s03: LightGBM re-rank after clustering...")
tg_cv3 = Counter()
for tr, te in kf.split(Xk, y_binary):
    gbm = LGBMClassifier(**LGB_PARAMS)
    gbm.fit(Xk.iloc[tr], y_binary[tr])
    tg = dict(zip(gbm.booster_.feature_name(),
                  gbm.booster_.feature_importance(importance_type='gain')))
    tg_cv3 += Counter(norm_imp(tg))

s03_r = sorted(tg_cv3.items(), key=lambda x: -x[1])
s03f = [f for f, _ in s03_r]
n_img_top5 = sum(1 for f in s03f[:5] if f.startswith(IMG_PFX))
print(f"    s03: Top 5 ({n_img_top5}/5 imaging): {s03f[:3]} ...")

# ═══════════════════════════════ 6. s04: SFS WITH LIGHTGBM AUC ═══════
# ★ v2 CHANGE: LightGBM AUC as SFS criterion (matching Phase 2A),
#    not RSF C-index as in v1.
print(f"    s04: Sequential Forward Selection (LightGBM AUC, 5-fold CV)...")
selected = []; remaining = list(s03f); prev_auc = 0
sfs_hist = []

for step in range(min(TOP_SFS, len(s03f))):
    best_f, best_auc = None, -1
    pool = remaining[:min(15, len(remaining))]
    for cand in pool:
        trial = selected + [cand]
        aucs = []
        for tr, te in kf.split(Xk[trial], y_binary):
            g = LGBMClassifier(**LGB_PARAMS)
            g.fit(Xk[trial].iloc[tr], y_binary[tr])
            aucs.append(roc_auc_score(y_binary[te],
                         g.predict_proba(Xk[trial].iloc[te])[:, 1]))
        avg_auc = np.mean(aucs)
        if avg_auc > best_auc:
            best_f, best_auc = cand, avg_auc

    if best_f is None:
        break

    gain = best_auc - prev_auc if step > 0 else best_auc
    isi = best_f.startswith(IMG_PFX)
    sfs_hist.append({
        'step': step + 1, 'feature': best_f,
        'auc': best_auc, 'gain': gain,
        'is_imaging': isi,
    })
    print(f"      Step {step+1}: {'[IMG]' if isi else '[BIO]'} {best_f}  "
          f"AUC={best_auc:.4f} (Δ={gain:+.4f})")

    if gain < SFS_EARLY_STOP and len(selected) >= TOP_FINAL:
        selected.append(best_f)
        remaining.remove(best_f)
        break

    selected.append(best_f)
    remaining.remove(best_f)
    prev_auc = best_auc

top_f = selected[:TOP_FINAL]
n_img_f = sum(1 for f in top_f if f.startswith(IMG_PFX))
n_bio_f = len(top_f) - n_img_f
print(f"    Top {len(top_f)} selected: {n_img_f} imaging + {n_bio_f} bio/other")
pd.DataFrame(sfs_hist).to_csv(os.path.join(RESULTS_DIR, 'sfs_history.csv'), index=False)

# ═══════════════════════════════ 7. s05: RSF + Cox PH ═══════════════
# ★ Final survival modeling — unchanged from v1
print(f"\n[7] s05: RSF (main) + Cox PH (baseline) — 5-fold CV")
Xf = Xk[top_f]
# Standardize for Cox PH
sc2 = StandardScaler()
Xf_s = pd.DataFrame(sc2.fit_transform(Xf), columns=Xf.columns, index=Xf.index)

# Build full survival array once
sy_full = make_surv_y(yt, ye)

# ── 7a. RSF (main model) ──
rsf_cis = []
tp_buf = {1: [], 3: [], 5: []}
tb_buf = {1: [], 3: [], 5: []}

for fi, (tr, te) in enumerate(kf.split(Xf_s, ye)):
    rsf = RandomSurvivalForest(
        n_estimators=200, max_depth=5, min_samples_leaf=5,
        random_state=SEED, n_jobs=4)
    rsf.fit(Xf_s.iloc[tr].values, make_surv_y(yt[tr], ye[tr]))
    pr = rsf.predict(Xf_s.iloc[te].values)
    rsf_cis.append(c_index(yt[te], ye[te], pr))

    # Time-dependent AUC / Brier at 1, 3, 5 years
    for t_eval in [1, 3, 5]:
        # Binary label at t_eval: event=1 if converted within t_eval
        yb_t = ((yt[te] <= t_eval) & (ye[te] == 1)).astype(int)
        # Valid subjects: either converted within t_eval or followed past t_eval
        valid_t = ((ye[te] == 1) & (yt[te] <= t_eval)) | (yt[te] > t_eval)
        tp_buf[t_eval].extend(pr[valid_t])
        tb_buf[t_eval].extend(yb_t[valid_t])

rsf_cidx = np.mean(rsf_cis)
rsf_std = np.std(rsf_cis)
print(f"    RSF (200 trees, max_depth=5, min_samples_leaf=5):")
print(f"      C-index = {rsf_cidx:.4f} ± {rsf_std:.4f}")

taucs, tbriers = {}, {}
for t_eval in [1, 3, 5]:
    if len(set(tb_buf[t_eval])) >= 2 and sum(tb_buf[t_eval]) >= 2:
        taucs[t_eval] = roc_auc_score(tb_buf[t_eval], tp_buf[t_eval])
        # Normalize risk scores to [0,1] for Brier
        pr_arr = np.array(tp_buf[t_eval])
        pn = (pr_arr - pr_arr.min()) / (pr_arr.max() - pr_arr.min() + 1e-8)
        tbriers[t_eval] = brier_score_loss(tb_buf[t_eval], pn)
        print(f"      tAUC@{t_eval}yr = {taucs[t_eval]:.4f},  "
              f"Brier = {tbriers[t_eval]:.4f},  n = {len(tb_buf[t_eval])}")

# ── 7b. Cox PH (baseline, interpretable) ──
print(f"\n    Cox PH (penalizer=0.1)...")
try:
    cph_df = Xf_s.copy()
    cph_df['time'] = yt
    cph_df['event'] = ye
    # Drop near-zero-variance columns (Cox can't handle them)
    lv = [c for c in Xf_s.columns if Xf_s[c].std() < 0.001]
    if lv:
        cph_df = cph_df.drop(columns=lv)
        print(f"      Dropped {len(lv)} low-variance cols for Cox")

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(cph_df, 'time', 'event')
    cox_ci_val = cph.concordance_index_
    print(f"      C-index = {cox_ci_val:.4f}")

    # Save Hazard Ratios
    hr = cph.summary[['coef', 'exp(coef)', 'p']].sort_values('p')
    hr.to_csv(os.path.join(RESULTS_DIR, 'cox_ph_summary.csv'))
    print(f"      Top significant features: {list(hr.index[:5])}")
except Exception as e:
    print(f"      Failed: {e}")
    cox_ci_val = np.nan

# ═══════════════════════════════ 8. SUMMARY ═══════════════════════════
print(f"\n{'=' * 70}")
print(f"  SURVIVAL MODEL RESULTS — v2 (Unified LightGBM Feature Selection)")
print(f"{'=' * 70}")

# Format summary table
print(f"\n  {'Model':<22} {'C-index':<12} {'tAUC@1yr':<10} "
      f"{'tAUC@3yr':<10} {'tAUC@5yr':<10}")
print(f"  {'─' * 22} {'─' * 12} {'─' * 10} {'─' * 10} {'─' * 10}")

results_table = [
    ('RSF (main)',         rsf_cidx,    taucs.get(1, np.nan),
                           taucs.get(3, np.nan), taucs.get(5, np.nan)),
    ('Cox PH',             cox_ci_val,  np.nan, np.nan, np.nan),
    # Phase 2A binary LGBM references (for comparison, not from this run)
    ('Binary LGBM (Phase2A)', np.nan,   np.nan, 0.8336, 0.8551),
]

for name, ci, a1, a3, a5 in results_table:
    ci_s = f"{ci:.4f}" if not np.isnan(ci) else "—"
    a1_s = f"{a1:.4f}" if not np.isnan(a1) else "—"
    a3_s = f"{a3:.4f}" if not np.isnan(a3) else "—"
    a5_s = f"{a5:.4f}" if not np.isnan(a5) else "—"
    print(f"  {name:<22} {ci_s:<12} {a1_s:<10} {a3_s:<10} {a5_s:<10}")

# Δ vs Phase 2A binary
if not np.isnan(taucs.get(3, np.nan)):
    d3 = taucs[3] - 0.8336
    print(f"\n  RSF vs Binary LGBM @3yr: ΔtAUC = {d3:+.4f}")
if not np.isnan(taucs.get(5, np.nan)):
    d5 = taucs[5] - 0.8551
    print(f"  RSF vs Binary LGBM @5yr: ΔtAUC = {d5:+.4f}")

# Print top features with their s01 ranks
print(f"\n  Top {len(top_f)} survival features (selected by LightGBM SFS):")
for i, f in enumerate(top_f):
    s01_rank = None
    for j, (fn, _) in enumerate(s01_r):
        if fn == f:
            s01_rank = j + 1
            break
    tag = '[IMG]' if f.startswith(IMG_PFX) else '[BIO]'
    rank_s = f"(s01 #{s01_rank})" if s01_rank else ""
    print(f"    {i+1:>2}. {tag} {f:<30} {rank_s}")

# ── Save results ──
pd.DataFrame({
    'model':       ['RSF', 'Cox PH', 'Binary LGBM 3yr', 'Binary LGBM 5yr'],
    'c_index':     [rsf_cidx, cox_ci_val, np.nan, np.nan],
    'auc_1yr':     [taucs.get(1, np.nan), np.nan, np.nan, np.nan],
    'auc_3yr':     [taucs.get(3, np.nan), np.nan, 0.8336, np.nan],
    'auc_5yr':     [taucs.get(5, np.nan), np.nan, np.nan, 0.8551],
    'brier_3yr':   [tbriers.get(3, np.nan), np.nan, np.nan, np.nan],
    'brier_5yr':   [tbriers.get(5, np.nan), np.nan, np.nan, np.nan],
}).to_csv(os.path.join(RESULTS_DIR, 'model_comparison.csv'), index=False)

pd.DataFrame({'rank': range(1, len(top_f) + 1),
              'feature': top_f}).to_csv(
    os.path.join(RESULTS_DIR, 'selected_features.csv'), index=False)

# Save full LGB feature ranking for reference
lgb_rank_df = pd.DataFrame([
    {'s01_rank': i + 1, 'feature': f, 's01_gain': g,
     'in_top10_survival': f in top_f[:10]}
    for i, (f, g) in enumerate(s01_r)
])
lgb_rank_df.to_csv(os.path.join(RESULTS_DIR, 'lgb_feature_ranking.csv'), index=False)

# ═══════════════════════════════ 9. PLOTS ═══════════════════════════════
print(f"\n[9] Generating plots...")

# ── 9a. KM curves by risk tertile ──
rsf_all = RandomSurvivalForest(
    n_estimators=300, max_depth=5, min_samples_leaf=5,
    random_state=SEED, n_jobs=4)
rsf_all.fit(Xf_s.values, sy_full)
all_risk = rsf_all.predict(Xf_s.values)

risk_g = pd.qcut(all_risk, 3, labels=['Low Risk', 'Medium Risk', 'High Risk'])
fig, ax = plt.subplots(figsize=(10, 6))
for lb, color in [('Low Risk', '#10B981'),
                   ('Medium Risk', '#F59E0B'),
                   ('High Risk', '#EF4444')]:
    mask = risk_g == lb
    kmf = KaplanMeierFitter()
    kmf.fit(yt[mask], ye[mask], label=f'{lb} (n={mask.sum()})')
    kmf.plot_survival_function(ax=ax, color=color, linewidth=2.5)

ax.set_xlabel('Years from MCI Baseline', fontsize=13)
ax.set_ylabel('Dementia-free Probability', fontsize=13)
ax.set_title('MCI→Dementia: KM Curves by RSF Risk Tertile\n'
             '(v2: Features from LightGBM SFS)',
             fontweight='bold', fontsize=14)
ax.set_xlim(0, 8)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=11, loc='lower left')
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(RESULTS_DIR, 'km_by_risk.png'), dpi=150)
plt.close()

# ── 9b. SFS accumulation curve (LightGBM AUC) ──
sd = pd.DataFrame(sfs_hist)
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(sd['step'], sd['auc'], 'o-', color='#3B82F6', lw=2.5,
         ms=10, mfc='white', mew=2)
for _, r in sd.iterrows():
    ax2.text(r['step'], r['auc'] + 0.006, r['feature'][:20],
             fontsize=7, ha='center', rotation=0)
ax2.set_xlabel('SFS Step', fontsize=13)
ax2.set_ylabel('AUC (LightGBM Cross-Validation)', fontsize=13)
ax2.set_title('SFS Feature Accumulation — LightGBM AUC\n'
              '(v2: Features then fed to RSF for survival modeling)',
              fontweight='bold', fontsize=14)
ax2.grid(alpha=0.25)
fig2.tight_layout()
fig2.savefig(os.path.join(RESULTS_DIR, 'sfs_accumulation.png'), dpi=150)
plt.close()

# ── 9c. Calibration plots @ 1/3/5yr ──
fig3, axes = plt.subplots(1, 3, figsize=(16, 5))
for ai, tev in enumerate([1, 3, 5]):
    ax = axes[ai]
    ar, ab = [], []
    for tr, te in kf.split(Xf_s, ye):
        rsf_f = RandomSurvivalForest(
            n_estimators=200, max_depth=5, min_samples_leaf=5,
            random_state=SEED, n_jobs=4)
        rsf_f.fit(Xf_s.iloc[tr].values, make_surv_y(yt[tr], ye[tr]))
        pr = rsf_f.predict(Xf_s.iloc[te].values)
        pn = (pr - pr.min()) / (pr.max() - pr.min() + 1e-8)
        yb = ((yt[te] <= tev) & (ye[te] == 1)).astype(int)
        valid = ((ye[te] == 1) & (yt[te] <= tev)) | (yt[te] > tev)
        ar.extend(pn[valid])
        ab.extend(yb[valid])

    if len(set(ab)) >= 2:
        qs = np.percentile(ar, np.linspace(0, 100, 11))
        bc, bo = [], []
        for i in range(len(qs) - 1):
            mk = (np.array(ar) >= qs[i]) & (np.array(ar) < qs[i + 1])
            if mk.sum() >= 5:
                bc.append(np.mean(np.array(ar)[mk]))
                bo.append(np.mean(np.array(ab)[mk]))
        if bc:
            ax.plot(bc, bo, 'o-', color='#3B82F6', lw=2.5, ms=10,
                    mfc='white', mew=2)
        ax.plot([0, 1], [0, 1], '--', color='gray', alpha=0.4)
        ax.set_xlabel('Predicted Probability', fontsize=12)
        ax.set_ylabel('Observed Frequency', fontsize=12)
        ax.set_title(f'{tev}-Year Calibration', fontweight='bold', fontsize=13)
        ax.grid(alpha=0.2)

fig3.suptitle('RSF Calibration: Predicted vs Observed Conversion Risk\n'
              '(v2: Features from LightGBM SFS)',
              fontweight='bold', fontsize=14)
fig3.tight_layout()
fig3.savefig(os.path.join(RESULTS_DIR, 'calibration.png'), dpi=150)
plt.close()

# ── 9d. Feature ablation: C-index when removing features bottom-up ──
print(f"    Feature ablation (RSF C-index vs n_features)...")
ablation_results = []
for i in range(len(top_f), 0, -1):
    sub_f = top_f[:i]
    cis_i = []
    for tr, te in kf.split(Xf_s[sub_f], ye):
        rsf_a = RandomSurvivalForest(
            n_estimators=100, max_depth=5, min_samples_leaf=5,
            random_state=SEED, n_jobs=4)
        rsf_a.fit(Xf_s[sub_f].iloc[tr].values,
                   make_surv_y(yt[tr], ye[tr]))
        pr_a = rsf_a.predict(Xf_s[sub_f].iloc[te].values)
        cis_i.append(c_index(yt[te], ye[te], pr_a))
    ablation_results.append({
        'n_features': i,
        'c_index': np.mean(cis_i),
        'c_index_std': np.std(cis_i),
        'removed': top_f[i - 1] if i < len(top_f) else 'none'
    })

ab_df = pd.DataFrame(ablation_results)
ab_df.to_csv(os.path.join(RESULTS_DIR, 'feature_ablation.csv'), index=False)

fig4, ax4 = plt.subplots(figsize=(10, 6))
ax4.errorbar(ab_df['n_features'], ab_df['c_index'],
             yerr=ab_df['c_index_std'], fmt='o-',
             color='#EF4444', lw=2, ms=8, capsize=4)
ax4.set_xlabel('Number of Features (removing from bottom)', fontsize=13)
ax4.set_ylabel('RSF C-index (5-fold CV)', fontsize=13)
ax4.set_title('Feature Ablation: RSF C-index vs Number of Features',
              fontweight='bold', fontsize=14)
ax4.invert_xaxis()
ax4.grid(alpha=0.25)
fig4.tight_layout()
fig4.savefig(os.path.join(RESULTS_DIR, 'feature_ablation.png'), dpi=150)
plt.close()

# ═══════════════════════════════ DONE ═══════════════════════════════════
elapsed = time.time() - t0
print(f"\n{'=' * 70}")
print(f"  ✅ Phase 2B v2 completed in {elapsed / 60:.1f} min")
print(f"  📁 Results saved to: {RESULTS_DIR}")
for f in sorted(os.listdir(RESULTS_DIR)):
    p = os.path.join(RESULTS_DIR, f)
    if os.path.isfile(p):
        print(f"     {f} ({os.path.getsize(p) / 1024:.0f} KB)")
print(f"{'=' * 70}")

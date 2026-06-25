#!/usr/bin/env python3
"""
CDR Worsening Target Builder — v2 (Revised Design)
====================================================
v2 修订要点 (基于审稿级别审查):
  1. 统一 index_date = 首次满足纳入条件的 CDR 访视
  2. CN (CDR=0) 和 MCI (CDR=0.5) 分队列构建标签
  3. 持续性恶化 (sustained worsening) 敏感性标签
  4. CDR-SB ≥1 恶化次要终点
  5. 复合终点 (CDR + DXSUM) for MCI cohort
  6. index_date 与特征基线日期对齐检查
  7. CONSORT/STROBE 式样本流图统计
  8. 纳入 vs 排除人群比较

用法:
  cd ADNI数据集 && python build_cdr_targets_v2.py
"""

import os, sys, warnings
import numpy as np
import pandas as pd
from datetime import datetime

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'extracted')
OUT_DIR = os.path.join(BASE_DIR, 'processed')
BASELINE_PATH = os.path.join(OUT_DIR, 'ADNI_baseline_with_time_targets_v2.csv')
CONSORT_DIR = '/Users/guxiao/Downloads/MCI-AD/UKB_DRP-main/local_data/Results_adni/cdr_survival'
os.makedirs(CONSORT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════
# 1. LOAD CDR LONGITUDINAL DATA
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("  CDR Worsening Target Builder — v2 (Revised Design)")
print("=" * 70)

print("\n[1] Loading CDR longitudinal data...")
cdr = pd.read_csv(os.path.join(DATA_DIR, 'All_Subjects_CDR_20May2026.csv'),
                  low_memory=False)
cdr['VISDATE'] = pd.to_datetime(cdr['VISDATE'], errors='coerce')
cdr = cdr.dropna(subset=['VISDATE', 'CDGLOBAL'])
cdr = cdr.sort_values(['RID', 'VISDATE'])

print(f"    CDR records: {len(cdr):,}")
print(f"    Unique subjects: {cdr['RID'].nunique():,}")

# ══════════════════════════════════════════════════════════════════
# 2. LOAD BASELINE FEATURES FOR DATE ALIGNMENT
# ══════════════════════════════════════════════════════════════════
print("\n[2] Loading baseline feature table for date alignment...")
base = pd.read_csv(BASELINE_PATH, low_memory=False)
print(f"    Baseline subjects: {len(base):,}")

# Parse birth year and compute approximate baseline feature date
base['_birth_year'] = pd.to_datetime(base['PTDOBYY'], errors='coerce').dt.year
# Approximate feature baseline date = birth_year + entry_age (mid-year)
base['_feat_baseline_date'] = base.apply(
    lambda r: datetime(int(r['_birth_year'] + r['entry_age']), 7, 1)
    if pd.notna(r['_birth_year']) and pd.notna(r['entry_age']) else pd.NaT,
    axis=1
)

# ══════════════════════════════════════════════════════════════════
# 3. TRACK CDR WORSENING FROM LONGITUDINAL DATA (v2)
# ══════════════════════════════════════════════════════════════════
print("\n[3] Tracking CDR worsening with unified index_date...")

# CONSORT counters
consort = {
    'total_cdr_records': len(cdr),
    'total_subjects_with_cdr': cdr['RID'].nunique(),
    'valid_cdr_visits': 0,
    'baseline_cdr_0_or_05': 0,
    'ge2_cdr_visits': 0,
    'in_baseline_features': 0,
    'surv_time_gt_0': 0,
    'cn_cohort': 0,
    'mci_cohort': 0,
}

worsening_records = []

for rid, group in cdr.groupby('RID'):
    # ── Determine index_date = first visit with CDGLOBAL ∈ {0, 0.5} ──
    eligible_visits = group[group['CDGLOBAL'].isin([0.0, 0.5])]
    if len(eligible_visits) == 0:
        continue  # Subject never had CDR=0 or 0.5

    consort['valid_cdr_visits'] += 1

    # index visit = first qualifying visit
    idx_visit = eligible_visits.iloc[0]
    idx_pos = group.index.get_loc(idx_visit.name)
    idx_date = idx_visit['VISDATE']
    idx_cdrg = idx_visit['CDGLOBAL']
    idx_cdrsb = idx_visit['CDRSB'] if pd.notna(idx_visit['CDRSB']) else np.nan

    # Subject must have at least one follow-up visit AFTER index_date
    later_visits = group.iloc[idx_pos + 1:]
    if len(later_visits) == 0:
        continue  # No follow-up CDR visits

    consort['baseline_cdr_0_or_05'] += 1
    cohort = 'CN' if idx_cdrg == 0.0 else 'MCI'

    # ── Compute last follow-up date ──
    last_date = later_visits['VISDATE'].max()
    last_followup_years = (last_date - idx_date).days / 365.25
    n_cdr_visits = len(group)
    n_post_idx_visits = len(later_visits)

    if n_post_idx_visits < 1 or last_followup_years <= 0:
        continue

    consort['ge2_cdr_visits'] += 1

    # ── Search for first worsening (primary: CDGLOBAL increase) ──
    cdrg_worsening_years = np.nan
    cdrg_worsened = 0
    cdrg_worsening_sustained = 0  # confirmed at next visit or never reverted
    cdrsb_worsening_years = np.nan
    cdrsb_worsened_1pt = 0
    cdrsb_worsening_sustained = 0

    for i in range(idx_pos + 1, len(group)):
        visit = group.iloc[i]
        visit_cdrg = visit['CDGLOBAL']
        visit_cdrsb = visit['CDRSB']
        visit_date = visit['VISDATE']
        years = (visit_date - idx_date).days / 365.25

        if years <= 0:
            continue

        # ── Primary: CDGLOBAL worsening ──
        if pd.notna(visit_cdrg) and visit_cdrg > idx_cdrg:
            if pd.isna(cdrg_worsening_years):
                cdrg_worsening_years = years
                cdrg_worsened = 1

                # Check sustained: is the NEXT visit also at worsened level?
                # Or is there no subsequent recovery?
                next_worse = False
                no_recovery = True
                for j in range(i + 1, len(group)):
                    later_cdrg = group.iloc[j]['CDGLOBAL']
                    later_date = group.iloc[j]['VISDATE']
                    if pd.notna(later_cdrg):
                        if later_cdrg > idx_cdrg:
                            next_worse = True
                            break
                        elif later_cdrg <= idx_cdrg and (later_date - visit_date).days > 0:
                            no_recovery = False
                            break

                if next_worse or (no_recovery and next_worse is False):
                    # If no recovery was observed but we don't have confirmation,
                    # be conservative: require either next visit confirms OR
                    # we've checked all later visits and none show recovery
                    # Actually, let's define sustained as: after first worsening,
                    # the next valid CDR visit (if any) is also > baseline
                    pass

                # Simpler definition: sustained = first worsening confirmed at next visit
                # or if no further visits exist, consider as unconfirmed
                if next_worse:
                    cdrg_worsening_sustained = 1

        # ── Secondary: CDRSB ≥1.0 worsening ──
        if pd.notna(visit_cdrsb) and pd.notna(idx_cdrsb):
            cdrsb_change = visit_cdrsb - idx_cdrsb
            if cdrsb_change >= 1.0:
                if pd.isna(cdrsb_worsening_years):
                    cdrsb_worsening_years = years
                    cdrsb_worsened_1pt = 1

                    # Check sustained for CDRSB
                    sustained_cdrsb = False
                    for j in range(i + 1, len(group)):
                        later_cdrsb = group.iloc[j]['CDRSB']
                        if pd.notna(later_cdrsb):
                            if (later_cdrsb - idx_cdrsb) >= 1.0:
                                sustained_cdrsb = True
                                break
                            elif (later_cdrsb - idx_cdrsb) < 1.0:
                                break
                    if sustained_cdrsb:
                        cdrsb_worsening_sustained = 1

    # ── Build record ──
    rec = {
        'RID': int(rid),
        # Cohort assignment
        'cdr_cohort': cohort,
        'index_CDGLOBAL': float(idx_cdrg),
        'index_CDRSB': float(idx_cdrsb) if pd.notna(idx_cdrsb) else np.nan,
        'index_date': idx_date,
        'index_viscode': idx_visit.get('VISCODE', ''),
        # Primary endpoint: CDGLOBAL worsening
        'CDR_worsened': cdrg_worsened,
        'CDR_worsening_years': cdrg_worsening_years,
        'CDR_worsened_sustained': cdrg_worsening_sustained,
        # Secondary endpoint: CDRSB ≥1 worsening
        'CDRSB_worsened': cdrsb_worsened_1pt,
        'CDRSB_worsening_years': cdrsb_worsening_years,
        'CDRSB_worsened_sustained': cdrsb_worsening_sustained,
        # Follow-up info
        'CDR_last_followup_years': last_followup_years,
        'n_cdr_visits': n_cdr_visits,
        'n_post_idx_visits': n_post_idx_visits,
    }
    worsening_records.append(rec)

cdr_targets = pd.DataFrame(worsening_records)
print(f"    Subjects with eligible CDR trajectory: {len(cdr_targets):,}")
print(f"    CN cohort: {(cdr_targets['cdr_cohort']=='CN').sum():,}")
print(f"    MCI cohort: {(cdr_targets['cdr_cohort']=='MCI').sum():,}")
print(f"    CN events: {cdr_targets[cdr_targets['cdr_cohort']=='CN']['CDR_worsened'].sum():.0f}")
print(f"    MCI events: {cdr_targets[cdr_targets['cdr_cohort']=='MCI']['CDR_worsened'].sum():.0f}")

# ══════════════════════════════════════════════════════════════════
# 4. INDEX DATE vs FEATURE BASELINE DATE ALIGNMENT
# ══════════════════════════════════════════════════════════════════
print("\n[4] Checking index_date vs feature baseline date alignment...")

# Merge with baseline feature dates
date_check = cdr_targets[['RID', 'cdr_cohort', 'index_date', 'index_CDGLOBAL',
                           'CDR_worsened', 'CDR_worsening_years']].merge(
    base[['RID', '_feat_baseline_date', '_birth_year', 'entry_age', 'PTGENDER']],
    on='RID', how='left'
)

date_check['date_diff_days'] = (
    date_check['index_date'] - date_check['_feat_baseline_date']
).dt.days

# Flag issues
date_check['feat_after_cdr'] = date_check['date_diff_days'] < -365  # features >1yr after CDR baseline
date_check['feat_before_cdr_gt1yr'] = date_check['date_diff_days'] > 365  # features >1yr before CDR baseline
date_check['feat_after_worsening'] = False
for idx, row in date_check.iterrows():
    if pd.notna(row['CDR_worsening_years']) and row['CDR_worsened'] == 1:
        worsen_date = row['index_date'] + pd.Timedelta(days=int(row['CDR_worsening_years'] * 365.25))
        if row['_feat_baseline_date'] > worsen_date:
            date_check.at[idx, 'feat_after_worsening'] = True

n_feat_missing = date_check['_feat_baseline_date'].isna().sum()
n_feat_after_cdr = date_check['feat_after_cdr'].sum()
n_feat_far_before = date_check['feat_before_cdr_gt1yr'].sum()
n_feat_after_worsen = date_check['feat_after_worsening'].sum()
n_date_ok = len(date_check) - n_feat_missing - n_feat_after_cdr - n_feat_far_before

print(f"    Subjects with feature baseline: {(~date_check['_feat_baseline_date'].isna()).sum():,}")
print(f"    Missing feature date (no baseline in feature table): {n_feat_missing}")
print(f"    Feature date AFTER CDR index_date (potential leakage): {n_feat_after_cdr}")
print(f"    Feature date >1yr BEFORE CDR index_date: {n_feat_far_before}")
print(f"    Feature date AFTER worsening event (definite leakage): {n_feat_after_worsen}")
print(f"    Median date diff: {date_check['date_diff_days'].median():.0f} days")
print(f"    Date diff IQR: [{date_check['date_diff_days'].quantile(0.25):.0f}, "
      f"{date_check['date_diff_days'].quantile(0.75):.0f}] days")

# Also check alignment for the subjects that survive to the final modeling set
model_set = date_check[
    date_check['RID'].isin(cdr_targets['RID']) &
    (date_check['date_diff_days'].notna())
]
within_90d = (model_set['date_diff_days'].abs() <= 90).sum()
within_180d = (model_set['date_diff_days'].abs() <= 180).sum()
print(f"\n    Among eligible subjects with feature dates:")
print(f"      Within ±90 days: {within_90d}/{len(model_set)} ({within_90d/len(model_set)*100:.1f}%)")
print(f"      Within ±180 days: {within_180d}/{len(model_set)} ({within_180d/len(model_set)*100:.1f}%)")

# Save alignment report
date_check[['RID', 'cdr_cohort', 'date_diff_days', 'feat_after_cdr',
             'feat_before_cdr_gt1yr', 'feat_after_worsening']].to_csv(
    os.path.join(CONSORT_DIR, 'index_date_alignment.csv'), index=False)

# ══════════════════════════════════════════════════════════════════
# 5. BUILD TIME-WINDOW BINARY LABELS
# ══════════════════════════════════════════════════════════════════
print(f"\n[5] Building time-window binary targets...")

for window_years in [3, 5, 10]:
    col = f'CDR_worsen_{window_years}yr'
    cdr_targets[col] = 0

    # Event within window
    event_in_window = (
        (cdr_targets['CDR_worsened'] == 1) &
        (cdr_targets['CDR_worsening_years'] <= window_years)
    )
    cdr_targets.loc[event_in_window, col] = 1

    # Censored: no event AND follow-up < window
    censored = (
        (cdr_targets['CDR_worsened'] == 0) &
        (cdr_targets['CDR_last_followup_years'] < window_years)
    )
    cdr_targets.loc[censored, col] = np.nan

    # Censoring flag
    cdr_targets[f'CDR_worsen_censored_{window_years}yr'] = censored.astype(int)

    # Sustained worsening version
    col_sus = f'CDR_worsen_sustained_{window_years}yr'
    cdr_targets[col_sus] = 0
    event_sus = (
        (cdr_targets['CDR_worsened_sustained'] == 1) &
        (cdr_targets['CDR_worsening_years'] <= window_years)
    )
    cdr_targets.loc[event_sus, col_sus] = 1
    cdr_targets.loc[censored, col_sus] = np.nan
    cdr_targets[f'CDR_worsen_sustained_censored_{window_years}yr'] = censored.astype(int)

    # CDRSB version
    col_sb = f'CDRSB_worsen_{window_years}yr'
    cdr_targets[col_sb] = 0
    event_sb = (
        (cdr_targets['CDRSB_worsened'] == 1) &
        (cdr_targets['CDRSB_worsening_years'] <= window_years)
    )
    cdr_targets.loc[event_sb, col_sb] = 1
    cdr_targets.loc[censored, col_sb] = np.nan
    cdr_targets[f'CDRSB_worsen_censored_{window_years}yr'] = censored.astype(int)

# All-time (no censoring needed)
cdr_targets['CDR_worsen_alltime'] = cdr_targets['CDR_worsened']
cdr_targets['CDR_worsen_sustained_alltime'] = cdr_targets['CDR_worsened_sustained']
cdr_targets['CDRSB_worsen_alltime'] = cdr_targets['CDRSB_worsened']

# ══════════════════════════════════════════════════════════════════
# 6. BUILD SURVIVAL LABELS
# ══════════════════════════════════════════════════════════════════
print(f"\n[6] Building survival labels...")

# Primary: CDGLOBAL worsening survival
cdr_targets['CDR_surv_time'] = np.where(
    cdr_targets['CDR_worsened'] == 1,
    cdr_targets['CDR_worsening_years'],
    cdr_targets['CDR_last_followup_years']
)
cdr_targets['CDR_surv_event'] = cdr_targets['CDR_worsened']

# Sustained worsening survival
cdr_targets['CDR_surv_sustained_time'] = np.where(
    cdr_targets['CDR_worsened_sustained'] == 1,
    cdr_targets['CDR_worsening_years'],
    cdr_targets['CDR_last_followup_years']
)
cdr_targets['CDR_surv_sustained_event'] = cdr_targets['CDR_worsened_sustained']

# CDRSB ≥1 worsening survival
cdr_targets['CDRSB_surv_time'] = np.where(
    cdr_targets['CDRSB_worsened'] == 1,
    cdr_targets['CDRSB_worsening_years'],
    cdr_targets['CDR_last_followup_years']
)
cdr_targets['CDRSB_surv_event'] = cdr_targets['CDRSB_worsened']

# ══════════════════════════════════════════════════════════════════
# 7. COMPOSITE ENDPOINT (MCI only: CDR worsening OR DXSUM Dementia)
# ══════════════════════════════════════════════════════════════════
print(f"\n[7] Building composite endpoint (MCI: CDR OR DXSUM Dementia)...")

# Merge DXSUM conversion data for composite endpoint
base_dx = base[['RID', 'converted_to_dementia', 'dementia_conversion_years',
                 'last_followup_years', 'baseline_diagnosis']].copy()

cdr_targets = cdr_targets.merge(base_dx, on='RID', how='left')

# Composite: for MCI cohort, event = CDR worsening OR DXSUM Dementia conversion
# Take the EARLIER of the two events
mci_mask = cdr_targets['cdr_cohort'] == 'MCI'
cdr_targets['composite_event'] = 0
cdr_targets['composite_time'] = cdr_targets['CDR_last_followup_years']

for idx in cdr_targets[mci_mask].index:
    row = cdr_targets.loc[idx]
    events = []

    if row['CDR_worsened'] == 1 and pd.notna(row['CDR_worsening_years']):
        events.append(('CDR', row['CDR_worsening_years']))

    if row.get('converted_to_dementia', 0) == 1 and pd.notna(row.get('dementia_conversion_years', np.nan)):
        events.append(('DX', row['dementia_conversion_years']))

    if events:
        earliest = min(events, key=lambda x: x[1])
        cdr_targets.at[idx, 'composite_event'] = 1
        cdr_targets.at[idx, 'composite_time'] = earliest[1]

# For CN cohort, composite = CDR worsening only (DXSUM isn't designed for CN→MCI)
cn_mask = cdr_targets['cdr_cohort'] == 'CN'
cdr_targets.loc[cn_mask, 'composite_event'] = cdr_targets.loc[cn_mask, 'CDR_worsened']
cdr_targets.loc[cn_mask, 'composite_time'] = np.where(
    cdr_targets.loc[cn_mask, 'CDR_worsened'] == 1,
    cdr_targets.loc[cn_mask, 'CDR_worsening_years'],
    cdr_targets.loc[cn_mask, 'CDR_last_followup_years']
)

print(f"    MCI composite events: {cdr_targets[mci_mask]['composite_event'].sum():.0f}")
print(f"    MCI CDR-only events: {cdr_targets[mci_mask]['CDR_worsened'].sum():.0f}")
print(f"    MCI additional from DXSUM: "
      f"{cdr_targets[mci_mask]['composite_event'].sum() - cdr_targets[mci_mask]['CDR_worsened'].sum():.0f}")

# ══════════════════════════════════════════════════════════════════
# 8. CONSORT FLOW STATISTICS
# ══════════════════════════════════════════════════════════════════
print(f"\n[8] CONSORT Flow Statistics...")

# First, drop v1 CDR columns from base to avoid merge conflicts
v1_cdr_cols = [
    'baseline_CDGLOBAL', 'baseline_CDRSB',
    'CDR_worsen_3yr', 'CDR_worsen_5yr', 'CDR_worsen_10yr', 'CDR_worsen_alltime',
    'CDR_worsen_censored_3yr', 'CDR_worsen_censored_5yr', 'CDR_worsen_censored_10yr',
    'CDR_surv_time', 'CDR_surv_event', 'CDRSB_surv_time', 'CDRSB_surv_event',
    'CDR_worsened', 'CDR_worsening_years', 'CDR_worsened_CDRSB2', 'CDRSB_worsening_years',
    'CDR_last_followup_years', 'n_cdr_visits',
]
base_clean = base.drop(columns=[c for c in v1_cdr_cols if c in base.columns and c != 'RID'])

# The flow from the baseline feature table perspective
total_in_baseline = len(base_clean)

# v2 targets: use cdr_targets which has the correct index_date-based CDGLOBAL
# Now merge into baseline (clean of v1 CDR columns)
base_v2 = base_clean.merge(
    cdr_targets[['RID', 'cdr_cohort', 'index_CDGLOBAL', 'index_date',
                  'CDR_worsened', 'CDR_worsening_years', 'CDR_last_followup_years',
                  'CDR_surv_time', 'CDR_surv_event',
                  'CDRSB_worsened', 'CDRSB_worsening_years',
                  'CDRSB_surv_time', 'CDRSB_surv_event',
                  'CDR_worsened_sustained', 'CDR_worsen_3yr', 'CDR_worsen_5yr',
                  'CDR_worsen_10yr', 'CDR_worsen_alltime',
                  'CDR_worsen_censored_3yr', 'CDR_worsen_censored_5yr',
                  'CDR_worsen_censored_10yr',
                  'CDR_worsen_sustained_3yr', 'CDR_worsen_sustained_5yr',
                  'CDRSB_worsen_3yr', 'CDRSB_worsen_5yr',
                  'composite_event', 'composite_time',
                  'n_cdr_visits', 'n_post_idx_visits',
                  'index_date', 'index_viscode']],
    on='RID', how='left'
)

# CONSORT flow
consort['total_in_baseline'] = total_in_baseline
consort['has_cdr_data'] = base_v2['cdr_cohort'].notna().sum()
consort['baseline_cdr_0_or_05_v2'] = base_v2['cdr_cohort'].notna().sum()
consort['ge2_cdr_visits_v2'] = (base_v2['n_post_idx_visits'] >= 1).sum()
consort['surv_time_gt_0_v2'] = ((base_v2['n_post_idx_visits'] >= 1) &
                                  (base_v2['CDR_surv_time'] > 0)).sum()
consort['cn_cohort'] = (base_v2['cdr_cohort'] == 'CN').sum()
consort['mci_cohort'] = (base_v2['cdr_cohort'] == 'MCI').sum()
consort['final_modeling_set'] = (
    (base_v2['cdr_cohort'].notna()) &
    (base_v2['n_post_idx_visits'] >= 1) &
    (base_v2['CDR_surv_time'] > 0)
).sum()
consort['final_cn'] = (
    (base_v2['cdr_cohort'] == 'CN') &
    (base_v2['n_post_idx_visits'] >= 1) &
    (base_v2['CDR_surv_time'] > 0)
).sum()
consort['final_mci'] = (
    (base_v2['cdr_cohort'] == 'MCI') &
    (base_v2['n_post_idx_visits'] >= 1) &
    (base_v2['CDR_surv_time'] > 0)
).sum()

# Final modeling set stats
final_set = base_v2[
    base_v2['cdr_cohort'].notna() &
    (base_v2['n_post_idx_visits'] >= 1) &
    (base_v2['CDR_surv_time'] > 0)
]
consort['final_cn_events'] = int(final_set[final_set['cdr_cohort']=='CN']['CDR_surv_event'].sum())
consort['final_mci_events'] = int(final_set[final_set['cdr_cohort']=='MCI']['CDR_surv_event'].sum())

print(f"\n  CONSORT Flow Diagram:")
print(f"  {'─' * 55}")
print(f"  {'Step':<40} {'N':>6}  {'Lost':>6}")
print(f"  {'─' * 55}")
steps = [
    ('Total in baseline feature table', consort['total_in_baseline'], 0),
    ('  Has any CDR data', consort['has_cdr_data'],
     consort['total_in_baseline'] - consort['has_cdr_data']),
    ('  Baseline CDR = 0 or 0.5 (v2 index)', consort['baseline_cdr_0_or_05_v2'],
     consort['has_cdr_data'] - consort['baseline_cdr_0_or_05_v2']),
    ('  ≥2 CDR visits (≥1 post-index)', consort['ge2_cdr_visits_v2'],
     consort['baseline_cdr_0_or_05_v2'] - consort['ge2_cdr_visits_v2']),
    ('  surv_time > 0', consort['surv_time_gt_0_v2'],
     consort['ge2_cdr_visits_v2'] - consort['surv_time_gt_0_v2']),
    ('FINAL MODELING SET', consort['final_modeling_set'], '—'),
    ('  ├─ CN (CDR=0→≥0.5)', consort['final_cn'],
     f"events={consort['final_cn_events']}"),
    ('  └─ MCI (CDR=0.5→≥1)', consort['final_mci'],
     f"events={consort['final_mci_events']}"),
]
for label, n, lost in steps:
    lost_s = f"{lost:>6}" if isinstance(lost, int) else f"{lost:>6}"
    print(f"  {label:<40} {n:>6}  {lost_s}")

# ══════════════════════════════════════════════════════════════════
# 9. INCLUDED vs EXCLUDED COMPARISON
# ══════════════════════════════════════════════════════════════════
print(f"\n[9] Comparing included vs excluded subjects...")

final_rids = set(final_set['RID'].values)
excluded = base_v2[~base_v2['RID'].isin(final_rids)].copy()

# Get demographics from baseline
for label, subset in [('INCLUDED', final_set), ('EXCLUDED', excluded)]:
    n = len(subset)
    age_mean = subset['entry_age'].mean() if 'entry_age' in subset.columns else np.nan
    age_median = subset['entry_age'].median() if 'entry_age' in subset.columns else np.nan
    female_pct = (subset['PTGENDER'] == 2.0).mean() * 100 if 'PTGENDER' in subset.columns else np.nan
    edu_mean = subset['PTEDUCAT'].mean() if 'PTEDUCAT' in subset.columns else np.nan

    print(f"\n  {label} (N={n}):")
    print(f"    Age: {age_mean:.1f} ± {subset['entry_age'].std():.1f} (median {age_median:.1f})")
    print(f"    Female: {female_pct:.1f}%")
    print(f"    Education: {edu_mean:.1f} yrs")

# Save CONSORT report
consort_df = pd.DataFrame([
    {'step': k, 'n': v} for k, v in consort.items()
])
consort_df.to_csv(os.path.join(CONSORT_DIR, 'consort_flow.csv'), index=False)

# Save included vs excluded comparison
comp_data = []
for label, subset in [('INCLUDED', final_set), ('EXCLUDED', excluded)]:
    comp_data.append({
        'group': label,
        'n': len(subset),
        'age_mean': subset['entry_age'].mean(),
        'age_std': subset['entry_age'].std(),
        'female_pct': (subset['PTGENDER'] == 2.0).mean() * 100,
        'education_mean': subset['PTEDUCAT'].mean(),
    })
pd.DataFrame(comp_data).to_csv(
    os.path.join(CONSORT_DIR, 'included_vs_excluded.csv'), index=False)

# ══════════════════════════════════════════════════════════════════
# 10. SUMMARY STATISTICS BY COHORT
# ══════════════════════════════════════════════════════════════════
print(f"\n[10] Summary statistics by cohort...")

for cohort_name, cohort_mask in [('CN', final_set['cdr_cohort'] == 'CN'),
                                   ('MCI', final_set['cdr_cohort'] == 'MCI')]:
    sub = final_set[cohort_mask]
    n = len(sub)
    ev = sub['CDR_surv_event'].sum()
    ce = n - int(ev)
    med_t = sub.loc[sub['CDR_surv_event']==1, 'CDR_surv_time'].median() if ev > 0 else np.nan
    med_fu = sub.loc[sub['CDR_surv_event']==0, 'CDR_surv_time'].median() if ce > 0 else np.nan

    print(f"\n  ── {cohort_name} Cohort (n={n}) ──")
    print(f"    Events: {int(ev)} ({ev/n*100:.1f}%)")
    print(f"    Censored: {int(ce)} ({ce/n*100:.1f}%)")
    print(f"    Median time to event: {med_t:.1f}yr")
    print(f"    Median follow-up (censored): {med_fu:.1f}yr")

    # Binary labels
    for w in [3, 5, 10]:
        col = f'CDR_worsen_{w}yr'
        ev_w = sub[col].sum()
        cen_w = sub[col].isna().sum()
        val_w = int((~sub[col].isna()).sum())
        print(f"    {w}yr: events={int(ev_w):>4}, neg={val_w-int(ev_w):>4}, "
              f"censored={int(cen_w):>4}, valid={val_w:>4}")

    # Sustained worsening
    ev_sus = sub['CDR_worsened_sustained'].sum()
    print(f"    Sustained worsening events: {int(ev_sus)}")

    # CDRSB worsening
    ev_sb = sub['CDRSB_worsened'].sum()
    print(f"    CDRSB ≥1 worsening events: {int(ev_sb)}")

# ══════════════════════════════════════════════════════════════════
# 11. COHORT LABEL SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════
print(f"\n[11] Cohort label summary...")

label_summary = []
for cohort_name in ['CN', 'MCI']:
    sub = final_set[final_set['cdr_cohort'] == cohort_name]
    label_summary.append({
        'cohort': cohort_name,
        'n_total': len(sub),
        'n_events_survival': int(sub['CDR_surv_event'].sum()),
        'n_events_sustained': int(sub['CDR_worsened_sustained'].sum()),
        'n_events_CDRSB': int(sub['CDRSB_worsened'].sum()),
        'n_events_composite': int(sub['composite_event'].sum()),
        'median_event_time': sub.loc[sub['CDR_surv_event']==1, 'CDR_surv_time'].median(),
        'valid_3yr': int((~sub['CDR_worsen_3yr'].isna()).sum()),
        'events_3yr': int(sub['CDR_worsen_3yr'].sum()),
        'valid_5yr': int((~sub['CDR_worsen_5yr'].isna()).sum()),
        'events_5yr': int(sub['CDR_worsen_5yr'].sum()),
        'valid_10yr': int((~sub['CDR_worsen_10yr'].isna()).sum()),
        'events_10yr': int(sub['CDR_worsen_10yr'].sum()),
    })

label_summary_df = pd.DataFrame(label_summary)
label_summary_df.to_csv(os.path.join(CONSORT_DIR, 'cohort_label_summary.csv'), index=False)
print(label_summary_df.to_string(index=False))

# ══════════════════════════════════════════════════════════════════
# 12. SAVE
# ══════════════════════════════════════════════════════════════════
print(f"\n[12] Saving...")

# Build the merge columns list
merge_cols = [
    'RID',
    # Cohort
    'cdr_cohort', 'index_CDGLOBAL', 'index_date', 'index_viscode',
    # Primary: CDGLOBAL worsening
    'CDR_worsened', 'CDR_worsening_years', 'CDR_worsened_sustained',
    'CDR_surv_time', 'CDR_surv_event',
    'CDR_surv_sustained_time', 'CDR_surv_sustained_event',
    # Binary windows (primary)
    'CDR_worsen_3yr', 'CDR_worsen_5yr', 'CDR_worsen_10yr', 'CDR_worsen_alltime',
    'CDR_worsen_censored_3yr', 'CDR_worsen_censored_5yr', 'CDR_worsen_censored_10yr',
    # Sustained worsening
    'CDR_worsen_sustained_3yr', 'CDR_worsen_sustained_5yr', 'CDR_worsen_sustained_alltime',
    'CDR_worsen_sustained_censored_3yr', 'CDR_worsen_sustained_censored_5yr',
    # Secondary: CDRSB ≥1 worsening
    'CDRSB_worsened', 'CDRSB_worsening_years', 'CDRSB_worsened_sustained',
    'CDRSB_surv_time', 'CDRSB_surv_event',
    'CDRSB_worsen_3yr', 'CDRSB_worsen_5yr', 'CDRSB_worsen_alltime',
    'CDRSB_worsen_censored_3yr', 'CDRSB_worsen_censored_5yr',
    # Composite endpoint
    'composite_event', 'composite_time',
    # Follow-up info
    'CDR_last_followup_years', 'n_cdr_visits', 'n_post_idx_visits',
]

# Save standalone CDR targets
cdr_targets_out = cdr_targets[[c for c in merge_cols if c in cdr_targets.columns]].copy()
cdr_targets_out.to_csv(os.path.join(OUT_DIR, 'CDR_time_targets_v2.csv'), index=False)
print(f"    CDR targets v2: CDR_time_targets_v2.csv ({len(cdr_targets_out.columns)} cols, {len(cdr_targets_out)} rows)")

# Update baseline with v2 CDR targets
# Drop old v1 CDR columns first
v1_cdr_cols = [
    'baseline_CDGLOBAL', 'baseline_CDRSB',
    'CDR_worsen_3yr', 'CDR_worsen_5yr', 'CDR_worsen_10yr', 'CDR_worsen_alltime',
    'CDR_worsen_censored_3yr', 'CDR_worsen_censored_5yr', 'CDR_worsen_censored_10yr',
    'CDR_surv_time', 'CDR_surv_event', 'CDRSB_surv_time', 'CDRSB_surv_event',
    'CDR_worsened', 'CDR_worsening_years', 'CDR_worsened_CDRSB2', 'CDRSB_worsening_years',
    'CDR_last_followup_years', 'n_cdr_visits',
]
for col in v1_cdr_cols:
    if col in base.columns and col not in ['RID']:
        base = base.drop(columns=[col])

# Merge v2 targets
base_v2_out = base.merge(
    cdr_targets[[c for c in merge_cols if c in cdr_targets.columns]],
    on='RID', how='left'
)

# Fill missing for subjects not in CDR v2
for col in ['CDR_worsened', 'CDR_worsened_sustained', 'CDRSB_worsened',
            'composite_event']:
    if col in base_v2_out.columns:
        base_v2_out[col] = base_v2_out[col].fillna(0).astype(int)

for col in ['CDR_worsen_3yr', 'CDR_worsen_5yr', 'CDR_worsen_10yr', 'CDR_worsen_alltime',
            'CDR_worsen_sustained_3yr', 'CDR_worsen_sustained_5yr', 'CDRSB_worsen_3yr',
            'CDRSB_worsen_5yr', 'CDRSB_worsen_alltime']:
    if col in base_v2_out.columns:
        base_v2_out[col] = base_v2_out[col].fillna(0)

for col in ['CDR_worsen_censored_3yr', 'CDR_worsen_censored_5yr',
            'CDR_worsen_censored_10yr', 'CDR_worsen_sustained_censored_3yr',
            'CDR_worsen_sustained_censored_5yr', 'CDRSB_worsen_censored_3yr',
            'CDRSB_worsen_censored_5yr']:
    if col in base_v2_out.columns:
        base_v2_out[col] = base_v2_out[col].fillna(1)

# Fill survival times
for col in ['CDR_surv_time', 'CDR_surv_sustained_time', 'CDRSB_surv_time', 'composite_time']:
    if col in base_v2_out.columns:
        base_v2_out[col] = base_v2_out[col].fillna(0.0)

for col in ['CDR_surv_event', 'CDR_surv_sustained_event', 'CDRSB_surv_event']:
    if col in base_v2_out.columns:
        base_v2_out[col] = base_v2_out[col].fillna(0).astype(int)

base_v2_out['n_cdr_visits'] = base_v2_out['n_cdr_visits'].fillna(0).astype(int)
base_v2_out['n_post_idx_visits'] = base_v2_out['n_post_idx_visits'].fillna(0).astype(int)

# Save updated baseline
out_path = os.path.join(OUT_DIR, 'ADNI_baseline_with_time_targets_v2.csv')
base_v2_out.to_csv(out_path, index=False)
print(f"    Baseline with v2 CDR targets: {out_path}")
print(f"    Size: {os.path.getsize(out_path) / 1024**2:.1f} MB, {len(base_v2_out.columns)} columns")

# ══════════════════════════════════════════════════════════════════
# 13. CROSS-ENDPOINT OVERLAP (v2)
# ══════════════════════════════════════════════════════════════════
print(f"\n[13] Cross-endpoint overlap (v2, MCI cohort)...")

mci_final = final_set[final_set['cdr_cohort'] == 'MCI']
if 'converted_to_dementia' in mci_final.columns:
    n_mci = len(mci_final)
    dx_only = ((mci_final['converted_to_dementia'] == 1) & (mci_final['CDR_worsened'] == 0)).sum()
    cdr_only = ((mci_final['converted_to_dementia'] == 0) & (mci_final['CDR_worsened'] == 1)).sum()
    both_pos = ((mci_final['converted_to_dementia'] == 1) & (mci_final['CDR_worsened'] == 1)).sum()
    neither = ((mci_final['converted_to_dementia'] == 0) & (mci_final['CDR_worsened'] == 0)).sum()

    print(f"    MCI final cohort: {n_mci}")
    print(f"    Both positive:  {int(both_pos):>5} ({both_pos/n_mci*100:4.1f}%)")
    print(f"    DXSUM only:     {int(dx_only):>5} ({dx_only/n_mci*100:4.1f}%)")
    print(f"    CDR only:       {int(cdr_only):>5} ({cdr_only/n_mci*100:4.1f}%)")
    print(f"    Neither:        {int(neither):>5} ({neither/n_mci*100:4.1f}%)")

    agree = int(both_pos + neither)
    po = agree / n_mci
    pe = ((both_pos+dx_only)*(both_pos+cdr_only) + (cdr_only+neither)*(dx_only+neither)) / n_mci**2
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0
    print(f"    Agreement: {agree}/{n_mci} = {po:.3f}")
    print(f"    Cohen's Kappa: {kappa:.3f}")

print(f"\n{'=' * 70}")
print(f"  ✅ CDR targets v2 built successfully!")
print(f"  📁 CONSORT flow: {os.path.join(CONSORT_DIR, 'consort_flow.csv')}")
print(f"  📁 Cohort summary: {os.path.join(CONSORT_DIR, 'cohort_label_summary.csv')}")
print(f"  📁 Date alignment: {os.path.join(CONSORT_DIR, 'index_date_alignment.csv')}")
print(f"  📁 Targets: {os.path.join(OUT_DIR, 'CDR_time_targets_v2.csv')}")
print(f"  📁 Baseline: {out_path}")
print(f"{'=' * 70}")

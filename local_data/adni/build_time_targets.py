#!/usr/bin/env python3
"""
ADNI Time-Window Target Builder
================================
Adds 5yr/10yr time-window targets to ADNI baseline data by tracking
diagnosis conversion across follow-up visits.

ADNI conversion tracking:
  UKB:  baseline → HES diagnosis date (exact day)
  ADNI: baseline visit → first follow-up visit with converted diagnosis

Targets added:
  - AD_5yrs / AD_10yrs: converted to AD within 5/10 years
  - Dementia_5yrs / Dementia_10yrs: converted to MCI or AD within 5/10 years
  - AD_years_conversion / Dementia_years_conversion: time to first conversion

Censoring (v2.0):
  Non-converters with insufficient follow-up are marked as censored
  rather than being treated as true negatives.
  - censored_5yr: follow-up < 5yr without conversion → exclude from 5yr targets
  - censored_10yr: follow-up < 10yr without conversion → exclude from 10yr targets
"""

import os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'extracted')
OUT_DIR = os.path.join(BASE_DIR, 'processed')
BASELINE_PATH = os.path.join(OUT_DIR, 'ADNI_baseline.csv')

# ══════════════════════════════════════════════════════════════════
# 1. BUILD CONVERSION TARGETS FROM DXSUM
# ══════════════════════════════════════════════════════════════════
print("=" * 60)
print("ADNI Time-Window Target Builder")
print("=" * 60)

print("\n[1] Tracking diagnosis conversion from DXSUM...")

dx = pd.read_csv(os.path.join(DATA_DIR, 'All_Subjects_DXSUM_20May2026.csv'),
                 low_memory=False)
dx['EXAMDATE'] = pd.to_datetime(dx['EXAMDATE'], errors='coerce')
dx = dx.sort_values(['RID', 'EXAMDATE'])

# For each subject: baseline diagnosis + time to first conversion
conversion_records = []

for rid, group in dx.groupby('RID'):
    if group['EXAMDATE'].isna().all():
        continue

    bl = group.iloc[0]
    bl_diag = bl['DIAGNOSIS']
    bl_date = bl['EXAMDATE']

    if pd.isna(bl_date) or pd.isna(bl_diag):
        continue

    # Initialize
    rec = {
        'RID': rid,
        'baseline_diagnosis': int(bl_diag),
        'ad_conversion_years': np.nan,
        'dementia_conversion_years': np.nan,
        'converted_to_ad': 0,
        'converted_to_dementia': 0,
        'converted_to_mci': 0,
        'last_followup_years': 0.0,
    }

    # Track through visits
    has_converted_to_mci = False
    for i in range(1, len(group)):
        visit_diag = group['DIAGNOSIS'].iloc[i]
        visit_date = group['EXAMDATE'].iloc[i]

        if pd.isna(visit_date) or pd.isna(visit_diag):
            continue

        years = (visit_date - bl_date).days / 365.25
        if years < 0:
            continue  # Skip visits before baseline

        rec['last_followup_years'] = max(rec['last_followup_years'], years)

        # First conversion to MCI (from CN)
        if bl_diag == 1.0 and visit_diag == 2.0 and not has_converted_to_mci:
            rec['dementia_conversion_years'] = years
            rec['converted_to_dementia'] = 1
            rec['converted_to_mci'] = 1
            has_converted_to_mci = True

        # First conversion to AD
        if visit_diag == 3.0:
            if np.isnan(rec['ad_conversion_years']):
                rec['ad_conversion_years'] = years
                rec['converted_to_ad'] = 1
            if np.isnan(rec['dementia_conversion_years']):
                rec['dementia_conversion_years'] = years
                rec['converted_to_dementia'] = 1

    conversion_records.append(rec)

conv_df = pd.DataFrame(conversion_records)
print(f"  Subjects tracked: {len(conv_df):,}")
print(f"  CN at baseline: {(conv_df['baseline_diagnosis']==1).sum()}")
print(f"  MCI at baseline: {(conv_df['baseline_diagnosis']==2).sum()}")
print(f"  AD at baseline: {(conv_df['baseline_diagnosis']==3).sum()}")

# Conversion stats
for target in ['converted_to_ad', 'converted_to_dementia', 'converted_to_mci']:
    n = conv_df[target].sum()
    print(f"  {target}: {n}")

# Time window stats
for window, years in [('5yr', 5), ('10yr', 10)]:
    has_ad = (conv_df['converted_to_ad'] == 1) & (conv_df['ad_conversion_years'] <= years)
    has_dem = (conv_df['converted_to_dementia'] == 1) & (conv_df['dementia_conversion_years'] <= years)
    print(f"  {window}: AD={has_ad.sum()}, Dementia={has_dem.sum()}")

# ══════════════════════════════════════════════════════════════════
# 2. MERGE INTO BASELINE DATASET
# ══════════════════════════════════════════════════════════════════
print(f"\n[2] Merging into baseline dataset...")

base = pd.read_csv(BASELINE_PATH)
print(f"  Baseline: {base.shape}")

# Merge conversion data
base = base.merge(conv_df[['RID', 'converted_to_ad', 'converted_to_dementia',
                            'converted_to_mci', 'ad_conversion_years',
                            'dementia_conversion_years', 'baseline_diagnosis']],
                  on='RID', how='left', suffixes=('', '_conv'))

# Fill missing — subjects not in DXSUM or with only 1 visit
for col in ['converted_to_ad', 'converted_to_dementia', 'converted_to_mci']:
    base[col] = base[col].fillna(0).astype(int)

# ── Build time-window targets ──
# For subjects with AD at baseline, set conversion years to 0 or below
# (they are prevalent cases — include as events or exclude)

# Option A: Include prevalent AD as events (years = 0 for time windows)
# Option B: Exclude prevalent AD from time-window analysis
# We'll create both: _incident (only baseline non-AD) and _all (include prevalent)

# For "incident" targets (only baseline CN or MCI):
at_risk_mask = base['baseline_diagnosis'].isin([1.0, 2.0])

# AD targets (incident only)
base['AD_status_incident'] = 0
base.loc[at_risk_mask & (base['converted_to_ad'] == 1), 'AD_status_incident'] = 1

base['AD_5yrs'] = 0
mask_5 = at_risk_mask & (base['converted_to_ad'] == 1) & (base['ad_conversion_years'] <= 5)
base.loc[mask_5, 'AD_5yrs'] = 1

base['AD_10yrs'] = 0
mask_10 = at_risk_mask & (base['converted_to_ad'] == 1) & (base['ad_conversion_years'] <= 10)
base.loc[mask_10, 'AD_10yrs'] = 1

base['AD_years'] = base['ad_conversion_years'].copy()

# Dementia targets (incident only)
base['Dementia_status_incident'] = 0
base.loc[at_risk_mask & (base['converted_to_dementia'] == 1), 'Dementia_status_incident'] = 1

base['Dementia_5yrs'] = 0
mask_d5 = at_risk_mask & (base['converted_to_dementia'] == 1) & (base['dementia_conversion_years'] <= 5)
base.loc[mask_d5, 'Dementia_5yrs'] = 1

base['Dementia_10yrs'] = 0
mask_d10 = at_risk_mask & (base['converted_to_dementia'] == 1) & (base['dementia_conversion_years'] <= 10)
base.loc[mask_d10, 'Dementia_10yrs'] = 1

base['Dementia_years'] = base['dementia_conversion_years'].copy()

# ── Censoring flags ──
# Non-converters whose last follow-up is shorter than the time window
# cannot be reliably labeled as negatives — they are censored.
base['censored_5yr'] = 0
base['censored_10yr'] = 0

# 5yr censoring: non-converter AND follow-up < 5 years
censored_5 = (
    (base['converted_to_ad'] == 0) &
    (base['last_followup_years'] < 5)
)
base.loc[censored_5, 'censored_5yr'] = 1

# 10yr censoring: non-converter AND follow-up < 10 years
censored_10 = (
    (base['converted_to_ad'] == 0) &
    (base['last_followup_years'] < 10)
)
base.loc[censored_10, 'censored_10yr'] = 1

# For dementia targets: same logic using converted_to_dementia
base['censored_dementia_5yr'] = 0
base['censored_dementia_10yr'] = 0

censored_d5 = (
    (base['converted_to_dementia'] == 0) &
    (base['last_followup_years'] < 5)
)
base.loc[censored_d5, 'censored_dementia_5yr'] = 1

censored_d10 = (
    (base['converted_to_dementia'] == 0) &
    (base['last_followup_years'] < 10)
)
base.loc[censored_d10, 'censored_dementia_10yr'] = 1

# ══════════════════════════════════════════════════════════════════
# 3. SUMMARY
# ══════════════════════════════════════════════════════════════════
print(f"\n[3] Time-window target summary:")
print(f"  {'Target':<20} {'n_events':>10} {'n_at_risk':>10} {'prev_%':>8} {'censored':>10}")
print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*8} {'─'*10}")

at_risk = base['baseline_diagnosis'].isin([1.0, 2.0]).sum()

for tcol in ['AD_5yrs', 'AD_10yrs', 'AD_status_incident',
             'Dementia_5yrs', 'Dementia_10yrs', 'Dementia_status_incident']:
    events = base[tcol].sum()
    prev = events / at_risk * 100 if at_risk > 0 else 0
    print(f"  {tcol:<20} {int(events):>10} {at_risk:>10} {prev:>7.1f}%")

print(f"\n[3b] Censoring summary (non-converters with insufficient follow-up):")
print(f"  censored_5yr (f/u < 5yr, no AD):          {base['censored_5yr'].sum():>6}")
print(f"  censored_10yr (f/u < 10yr, no AD):         {base['censored_10yr'].sum():>6}")
print(f"  censored_dementia_5yr (f/u < 5yr, no Dem): {base['censored_dementia_5yr'].sum():>6}")
print(f"  censored_dementia_10yr (f/u < 10yr, no Dem):{base['censored_dementia_10yr'].sum():>6}")

# Breakdown by baseline diagnosis
for diag, label in [(1.0, 'CN'), (2.0, 'MCI')]:
    sub = base[base['baseline_diagnosis'] == diag]
    n = len(sub)
    print(f"\n  {label} (n={n}):")
    print(f"    Median follow-up: {sub['last_followup_years'].median():.1f} yr")
    print(f"    f/u < 5yr:  {(sub['last_followup_years'] < 5).sum()} ({(sub['last_followup_years'] < 5).sum()/n*100:.1f}%)")
    print(f"    f/u < 10yr: {(sub['last_followup_years'] < 10).sum()} ({(sub['last_followup_years'] < 10).sum()/n*100:.1f}%)")
    print(f"    censored_5yr:  {sub['censored_5yr'].sum()} ({(sub['censored_5yr'].sum()/n*100):.1f}%)")
    print(f"    censored_10yr: {sub['censored_10yr'].sum()} ({(sub['censored_10yr'].sum()/n*100):.1f}%)")

# ══════════════════════════════════════════════════════════════════
# 4. SAVE
# ══════════════════════════════════════════════════════════════════
out_path = os.path.join(OUT_DIR, 'ADNI_baseline_with_time_targets.csv')
base.to_csv(out_path, index=False)
print(f"\n  Saved: {out_path}")
print(f"  Size: {os.path.getsize(out_path) / 1024**2:.1f} MB")
print(f"  Total columns: {len(base.columns)}")

# Also save standalone targets
target_cols = ['RID', 'AD_5yrs', 'AD_10yrs', 'AD_status_incident', 'AD_years',
               'Dementia_5yrs', 'Dementia_10yrs', 'Dementia_status_incident',
               'Dementia_years', 'converted_to_ad', 'converted_to_dementia',
               'converted_to_mci', 'ad_conversion_years', 'dementia_conversion_years',
               'last_followup_years',
               'censored_5yr', 'censored_10yr',
               'censored_dementia_5yr', 'censored_dementia_10yr']
base[target_cols].to_csv(os.path.join(OUT_DIR, 'ADNI_time_targets.csv'), index=False)

print("\nDone!")

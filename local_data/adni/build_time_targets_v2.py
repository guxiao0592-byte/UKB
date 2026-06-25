#!/usr/bin/env python3
"""
ADNI Time-Window Target Builder v2.0
=====================================
修正版：正确计算随访时长、删失标记、新增3年窗口。

v2.0 修正:
  1. last_followup_years 正确合并并保存
  2. 新增 3yr 窗口目标 (AD_3yrs, Dementia_3yrs)
  3. 删失标记完整保存到两个输出文件
  4. 在输出中报告每个窗口的有效样本数

用法:
  cd ADNI数据集 && python build_time_targets_v2.py
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
print("ADNI Time-Window Target Builder v2.0")
print("=" * 60)

print("\n[1] Tracking diagnosis conversion from DXSUM...")

dx = pd.read_csv(os.path.join(DATA_DIR, 'All_Subjects_DXSUM_20May2026.csv'),
                 low_memory=False)
dx['EXAMDATE'] = pd.to_datetime(dx['EXAMDATE'], errors='coerce')
dx = dx.sort_values(['RID', 'EXAMDATE'])

conversion_records = []
for rid, group in dx.groupby('RID'):
    if group['EXAMDATE'].isna().all():
        continue
    bl = group.iloc[0]
    bl_diag = bl['DIAGNOSIS']
    bl_date = bl['EXAMDATE']
    if pd.isna(bl_date) or pd.isna(bl_diag):
        continue

    rec = {
        'RID': int(rid),
        'baseline_diagnosis': int(bl_diag),
        'ad_conversion_years': np.nan,
        'dementia_conversion_years': np.nan,
        'converted_to_ad': 0,
        'converted_to_dementia': 0,
        'converted_to_mci': 0,
        'last_followup_years': 0.0,
    }

    has_converted_to_mci = (bl_diag == 2.0)
    last_date = bl_date

    for i in range(1, len(group)):
        visit_diag = group['DIAGNOSIS'].iloc[i]
        visit_date = group['EXAMDATE'].iloc[i]
        if pd.isna(visit_date) or pd.isna(visit_diag):
            continue

        years = (visit_date - bl_date).days / 365.25
        if years < 0:
            continue

        last_date = max(last_date, visit_date)
        rec['last_followup_years'] = max(rec['last_followup_years'], years)

        # First MCI conversion (from CN)
        if bl_diag == 1.0 and visit_diag == 2.0 and not has_converted_to_mci:
            rec['dementia_conversion_years'] = years
            rec['converted_to_dementia'] = 1
            rec['converted_to_mci'] = 1
            has_converted_to_mci = True

        # First AD conversion (Narrow endpoint)
        if visit_diag == 3.0:
            if np.isnan(rec['ad_conversion_years']):
                rec['ad_conversion_years'] = years
                rec['converted_to_ad'] = 1

        # First Dementia conversion (Broad endpoint: AD or Parkinson's or Other)
        # ★ FIX v2.1: also check DXPARK and DXOTHDEM, not just DIAGNOSIS==3
        is_dementia = (
            visit_diag == 3.0 or
            group['DXPARK'].iloc[i] == 1.0 or
            group['DXOTHDEM'].iloc[i] == 1.0
        )
        if is_dementia:
            if np.isnan(rec['dementia_conversion_years']):
                rec['dementia_conversion_years'] = years
                rec['converted_to_dementia'] = 1

    conversion_records.append(rec)

conv_df = pd.DataFrame(conversion_records)
print(f"  Subjects tracked: {len(conv_df):,}")
print(f"  CN at baseline:  {(conv_df['baseline_diagnosis']==1).sum():>5}")
print(f"  MCI at baseline: {(conv_df['baseline_diagnosis']==2).sum():>5}")
print(f"  AD at baseline:  {(conv_df['baseline_diagnosis']==3).sum():>5}")
print(f"  Median follow-up: {conv_df['last_followup_years'].median():.1f} yr")

# Conversion stats
for label, col in [('converted_to_ad', 'converted_to_ad'),
                    ('converted_to_dementia', 'converted_to_dementia'),
                    ('converted_to_mci', 'converted_to_mci')]:
    print(f"  {label}: {conv_df[col].sum():>5}")

# ══════════════════════════════════════════════════════════════════
# 2. MERGE INTO BASELINE
# ══════════════════════════════════════════════════════════════════
print(f"\n[2] Merging into baseline dataset...")

base = pd.read_csv(BASELINE_PATH)
print(f"  Baseline: {base.shape}")

# ★ FIX: include last_followup_years in the merge
merge_cols = ['RID', 'converted_to_ad', 'converted_to_dementia',
              'converted_to_mci', 'ad_conversion_years',
              'dementia_conversion_years', 'baseline_diagnosis',
              'last_followup_years']  # <-- v2.0: was missing in v1.0
base = base.merge(conv_df[merge_cols], on='RID', how='left',
                  suffixes=('', '_conv'))

# Fill missing for subjects not in DXSUM
for col in ['converted_to_ad', 'converted_to_dementia', 'converted_to_mci']:
    base[col] = base[col].fillna(0).astype(int)

base['last_followup_years'] = base['last_followup_years'].fillna(0.0)

# ══════════════════════════════════════════════════════════════════
# 3. BUILD TIME-WINDOW TARGETS (3yr + 5yr + 10yr)
# ══════════════════════════════════════════════════════════════════
print(f"\n[3] Building time-window targets (3yr, 5yr, 10yr)...")

at_risk_mask = base['baseline_diagnosis'].isin([1.0, 2.0])

# ── AD targets ──
# 3yr
base['AD_3yrs'] = 0
base.loc[at_risk_mask & (base['converted_to_ad'] == 1) &
         (base['ad_conversion_years'] <= 3), 'AD_3yrs'] = 1

# 5yr
base['AD_5yrs'] = 0
base.loc[at_risk_mask & (base['converted_to_ad'] == 1) &
         (base['ad_conversion_years'] <= 5), 'AD_5yrs'] = 1

# 10yr
base['AD_10yrs'] = 0
base.loc[at_risk_mask & (base['converted_to_ad'] == 1) &
         (base['ad_conversion_years'] <= 10), 'AD_10yrs'] = 1

# All-time
base['AD_status_incident'] = 0
base.loc[at_risk_mask & (base['converted_to_ad'] == 1), 'AD_status_incident'] = 1

base['AD_years'] = base['ad_conversion_years'].copy()

# ── Dementia targets ──
base['Dementia_3yrs'] = 0
base.loc[at_risk_mask & (base['converted_to_dementia'] == 1) &
         (base['dementia_conversion_years'] <= 3), 'Dementia_3yrs'] = 1

base['Dementia_5yrs'] = 0
base.loc[at_risk_mask & (base['converted_to_dementia'] == 1) &
         (base['dementia_conversion_years'] <= 5), 'Dementia_5yrs'] = 1

base['Dementia_10yrs'] = 0
base.loc[at_risk_mask & (base['converted_to_dementia'] == 1) &
         (base['dementia_conversion_years'] <= 10), 'Dementia_10yrs'] = 1

base['Dementia_status_incident'] = 0
base.loc[at_risk_mask & (base['converted_to_dementia'] == 1),
         'Dementia_status_incident'] = 1

base['Dementia_years'] = base['dementia_conversion_years'].copy()

# ══════════════════════════════════════════════════════════════════
# 4. CENSORING FLAGS
# ══════════════════════════════════════════════════════════════════
print(f"\n[4] Computing censoring flags...")

# For each window: non-converters with f/u < window are censored
for window_years in [3, 5, 10]:
    for prefix, conv_col in [('AD', 'converted_to_ad'),
                              ('Dementia', 'converted_to_dementia')]:
        col_name = f'censored_{prefix.lower()}_{window_years}yr'
        base[col_name] = 0
        censored_mask = (
            (base[conv_col] == 0) &
            (base['last_followup_years'] < window_years)
        )
        base.loc[censored_mask, col_name] = 1

# ══════════════════════════════════════════════════════════════════
# 5. SUMMARY
# ══════════════════════════════════════════════════════════════════
print(f"\n[5] Time-window target summary (at-risk = CN+MCI):")
at_risk = base[at_risk_mask]
n_ar = len(at_risk)

for target, window_years in [('AD_3yrs', 3), ('AD_5yrs', 5), ('AD_10yrs', 10),
                              ('AD_status_incident', None),
                              ('Dementia_3yrs', 3), ('Dementia_5yrs', 5),
                              ('Dementia_10yrs', 10), ('Dementia_status_incident', None)]:
    events = int(at_risk[target].sum())
    if window_years:
        censored_col = f'censored_{target.split("_")[0].lower()}_{window_years}yr'
        n_censored = int(at_risk[censored_col].sum())
    else:
        n_censored = 0
    valid = n_ar - n_censored
    prev = events / valid * 100 if valid > 0 else 0
    print(f"  {target:<24} events={events:>4}  censored={n_censored:>5}  "
          f"valid={valid:>5}  prev={prev:>5.1f}%")

# Breakdown by baseline diagnosis
for diag, label in [(1.0, 'CN'), (2.0, 'MCI')]:
    sub = at_risk[at_risk['baseline_diagnosis'] == diag]
    n = len(sub)
    print(f"\n  [{label}] n={n}, median f/u={sub['last_followup_years'].median():.1f}yr")
    for window_years in [3, 5, 10]:
        f'{int(sub[target].sum())}'
        ev = int(((sub['converted_to_ad'] == 1) &
                  (sub['ad_conversion_years'] <= window_years)).sum())
        cen = int(((sub['converted_to_ad'] == 0) &
                   (sub['last_followup_years'] < window_years)).sum())
        val = n - cen
        print(f"    AD {window_years}yr: events={ev:>3}, censored={cen:>4}, "
              f"valid={val:>4} ({val/n*100:.0f}%)")

# ══════════════════════════════════════════════════════════════════
# 6. SAVE
# ══════════════════════════════════════════════════════════════════
out_path = os.path.join(OUT_DIR, 'ADNI_baseline_with_time_targets_v2.csv')
base.to_csv(out_path, index=False)
print(f"\n[6] Saved: {out_path}")
print(f"  Size: {os.path.getsize(out_path) / 1024**2:.1f} MB")
print(f"  Total columns: {len(base.columns)}")

# Standalone targets
target_cols = [
    'RID',
    # AD targets
    'AD_3yrs', 'AD_5yrs', 'AD_10yrs', 'AD_status_incident', 'AD_years',
    # Dementia targets
    'Dementia_3yrs', 'Dementia_5yrs', 'Dementia_10yrs',
    'Dementia_status_incident', 'Dementia_years',
    # Conversion
    'converted_to_ad', 'converted_to_dementia', 'converted_to_mci',
    'ad_conversion_years', 'dementia_conversion_years',
    # Follow-up & censoring (v2.0)
    'last_followup_years',
    'censored_ad_3yr', 'censored_ad_5yr', 'censored_ad_10yr',
    'censored_dementia_3yr', 'censored_dementia_5yr', 'censored_dementia_10yr',
]
base[target_cols].to_csv(os.path.join(OUT_DIR, 'ADNI_time_targets_v2.csv'), index=False)
print(f"  Targets saved: ADNI_time_targets_v2.csv ({len(target_cols)} columns)")

print("\n✅ Done! v2.0 targets ready for censored-aware training.")

#!/usr/bin/env python3
"""
CDR Worsening Target Builder
=============================
从 All_Subjects_CDR_20May2026.csv 构建 CDR 恶化标签:
  1. CDGLOBAL 恶化二分类标签 (3yr/5yr/10yr 窗口 + 删失标记)
  2. CDGLOBAL 恶化生存标签 (time, event)
  3. CDRSB≥2 恶化标签 (次要终点)

与 build_time_targets_v2.py 并行 — 分别从 CDR 和 DXSUM 构建标签，
全部 merge 到同一个 baseline 特征矩阵上。

用法:
  cd ADNI数据集 && python build_cdr_targets.py
"""

import os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'extracted')
OUT_DIR = os.path.join(BASE_DIR, 'processed')
BASELINE_PATH = os.path.join(OUT_DIR, 'ADNI_baseline_with_time_targets_v2.csv')

# ══════════════════════════════════════════════════════════════════
# 1. TRACK CDR WORSENING FROM LONGITUDINAL CDR DATA
# ══════════════════════════════════════════════════════════════════
print("=" * 60)
print("CDR Worsening Target Builder")
print("=" * 60)

print("\n[1] Tracking CDR worsening from longitudinal CDR data...")

cdr = pd.read_csv(os.path.join(DATA_DIR, 'All_Subjects_CDR_20May2026.csv'),
                  low_memory=False)
cdr['VISDATE'] = pd.to_datetime(cdr['VISDATE'], errors='coerce')
cdr = cdr.sort_values(['RID', 'VISDATE'])

print(f"  CDR records: {len(cdr):,}")
print(f"  Unique subjects: {cdr['RID'].nunique():,}")

worsening_records = []

for rid, group in cdr.groupby('RID'):
    if group['VISDATE'].isna().all():
        continue

    bl = group.iloc[0]
    bl_date = bl['VISDATE']
    bl_cdrg = bl['CDGLOBAL']
    bl_cdrsb = bl['CDRSB']

    if pd.isna(bl_date) or pd.isna(bl_cdrg):
        continue

    rec = {
        'RID': int(rid),
        'baseline_CDGLOBAL': float(bl_cdrg),
        'baseline_CDRSB': float(bl_cdrsb) if pd.notna(bl_cdrsb) else np.nan,
        'CDR_worsening_years': np.nan,        # time to first CDGLOBAL worsening
        'CDR_worsened': 0,                     # 1 = CDGLOBAL increased from baseline
        'CDR_worsened_CDRSB2': 0,              # 1 = CDRSB increased ≥2.0
        'CDRSB_worsening_years': np.nan,       # time to CDRSB≥2 worsening
        'CDR_last_followup_years': 0.0,
        'n_cdr_visits': len(group),
    }

    last_date = bl_date

    for i in range(1, len(group)):
        visit_cdrg = group['CDGLOBAL'].iloc[i]
        visit_cdrsb = group['CDRSB'].iloc[i]
        visit_date = group['VISDATE'].iloc[i]

        if pd.isna(visit_date):
            continue

        years = (visit_date - bl_date).days / 365.25
        if years < 0:
            continue

        last_date = max(last_date, visit_date)
        rec['CDR_last_followup_years'] = max(rec['CDR_last_followup_years'], years)

        # ── CDGLOBAL worsening: ordinal increase from baseline ──
        if pd.notna(visit_cdrg) and visit_cdrg > bl_cdrg:
            if pd.isna(rec['CDR_worsening_years']):
                rec['CDR_worsening_years'] = years
                rec['CDR_worsened'] = 1

        # ── CDRSB worsening: ≥2.0 point increase ──
        if pd.notna(visit_cdrsb) and pd.notna(bl_cdrsb):
            if (visit_cdrsb - bl_cdrsb) >= 2.0:
                if pd.isna(rec['CDRSB_worsening_years']):
                    rec['CDRSB_worsening_years'] = years
                    rec['CDR_worsened_CDRSB2'] = 1

    worsening_records.append(rec)

cdr_targets = pd.DataFrame(worsening_records)
print(f"  Subjects tracked: {len(cdr_targets):,}")
print(f"  Baseline CDGLOBAL distribution:")
for val in [0.0, 0.5, 1.0, 2.0, 3.0]:
    n = (cdr_targets['baseline_CDGLOBAL'] == val).sum()
    if n > 0:
        print(f"    CDR={val}: {n:>5}")

# ── Worsening statistics ──
for bl_val, label in [(0.0, 'CN (CDR=0)'), (0.5, 'MCI (CDR=0.5)')]:
    sub = cdr_targets[cdr_targets['baseline_CDGLOBAL'] == bl_val]
    n_worsen = sub['CDR_worsened'].sum()
    n_cdrsb2 = sub['CDR_worsened_CDRSB2'].sum()
    n_total = len(sub)
    n_ge2 = (sub['n_cdr_visits'] >= 2).sum()
    print(f"\n  {label}: total={n_total}, ≥2 visits={n_ge2}")
    print(f"    CDGLOBAL worsened: {n_worsen} ({n_worsen/max(n_total,1)*100:.1f}%)")
    print(f"    CDRSB ≥2 increase: {n_cdrsb2} ({n_cdrsb2/max(n_total,1)*100:.1f}%)")
    if n_worsen > 0:
        med_t = sub.loc[sub['CDR_worsened']==1, 'CDR_worsening_years'].median()
        print(f"    Median time to CDGLOBAL worsening: {med_t:.1f}yr")

# ══════════════════════════════════════════════════════════════════
# 2. BUILD TIME-WINDOW BINARY LABELS
# ══════════════════════════════════════════════════════════════════
print(f"\n[2] Building time-window binary targets (3yr, 5yr, 10yr)...")

# Target population: baseline CDGLOBAL ∈ {0, 0.5}
at_risk = cdr_targets['baseline_CDGLOBAL'].isin([0.0, 0.5])

for window_years in [3, 5, 10]:
    col_name = f'CDR_worsen_{window_years}yr'

    # Worsened within window → 1
    cdr_targets[col_name] = 0
    cdr_targets.loc[
        at_risk &
        (cdr_targets['CDR_worsened'] == 1) &
        (cdr_targets['CDR_worsening_years'] <= window_years),
        col_name
    ] = 1

    # Did NOT worsen AND follow-up < window → censored (mark as NaN/99)
    # Did NOT worsen AND follow-up ≥ window → true negative (keep 0)
    censored_mask = (
        at_risk &
        (cdr_targets['CDR_worsened'] == 0) &
        (cdr_targets['CDR_last_followup_years'] < window_years)
    )
    cdr_targets.loc[censored_mask, col_name] = np.nan

    # Censoring flag
    cdr_targets[f'CDR_worsen_censored_{window_years}yr'] = censored_mask.astype(int)

# All-time: no censoring needed
cdr_targets['CDR_worsen_alltime'] = 0
cdr_targets.loc[at_risk & (cdr_targets['CDR_worsened'] == 1), 'CDR_worsen_alltime'] = 1

# ══════════════════════════════════════════════════════════════════
# 3. BUILD SURVIVAL LABELS
# ══════════════════════════════════════════════════════════════════
print(f"\n[3] Building survival labels...")

# CDGLOBAL worsening survival
cdr_targets['CDR_surv_time'] = np.where(
    cdr_targets['CDR_worsened'] == 1,
    cdr_targets['CDR_worsening_years'],
    cdr_targets['CDR_last_followup_years']
)
cdr_targets['CDR_surv_event'] = cdr_targets['CDR_worsened']

# CDRSB≥2 survival (secondary)
cdr_targets['CDRSB_surv_time'] = np.where(
    cdr_targets['CDR_worsened_CDRSB2'] == 1,
    cdr_targets['CDRSB_worsening_years'],
    cdr_targets['CDR_last_followup_years']
)
cdr_targets['CDRSB_surv_event'] = cdr_targets['CDR_worsened_CDRSB2']

# ══════════════════════════════════════════════════════════════════
# 4. SUMMARY STATISTICS
# ══════════════════════════════════════════════════════════════════
print(f"\n[4] Time-window target summary (at-risk = CDR 0 or 0.5, ≥2 visits):")

ar_multi = cdr_targets[
    at_risk &
    (cdr_targets['n_cdr_visits'] >= 2) &
    (cdr_targets['CDR_surv_time'] > 0)
]

for bl_val, label in [(0.0, 'CN (CDR=0)'), (0.5, 'MCI (CDR=0.5)')]:
    sub = ar_multi[ar_multi['baseline_CDGLOBAL'] == bl_val]
    print(f"\n  ── {label} (n={len(sub)}) ──")
    for w in [3, 5, 10]:
        col = f'CDR_worsen_{w}yr'
        ev = sub[col].sum()
        cen = sub[col].isna().sum()
        val = int((~sub[col].isna()).sum())
        neg = val - int(ev)
        print(f"    {w}yr: events={int(ev):>4}, neg={neg:>4}, censored={cen:>4}, "
              f"valid={val:>4} ({val/max(len(sub),1)*100:.0f}%)")

# ── Overall ──
print(f"\n  ── ALL at-risk (CN+MCI, n={len(ar_multi)}) ──")
for w in [3, 5, 10]:
    col = f'CDR_worsen_{w}yr'
    ev = ar_multi[col].sum()
    cen = ar_multi[col].isna().sum()
    val = int((~ar_multi[col].isna()).sum())
    print(f"    {w}yr: events={int(ev):>4}, censored={cen:>4}, valid={val:>4} "
          f"({val/max(len(ar_multi),1)*100:.0f}%)")

# Survival stats
for bl_val, label in [(0.0, 'CN'), (0.5, 'MCI')]:
    sub = ar_multi[ar_multi['baseline_CDGLOBAL'] == bl_val]
    ev = sub['CDR_surv_event'].sum()
    ce = (sub['CDR_surv_event'] == 0).sum()
    print(f"\n  Survival {label}: {len(sub)} subjects, {int(ev)} events, "
          f"{int(ce)} censored")

# ══════════════════════════════════════════════════════════════════
# 5. MERGE INTO BASELINE
# ══════════════════════════════════════════════════════════════════
print(f"\n[5] Merging CDR targets into baseline dataset...")

base = pd.read_csv(BASELINE_PATH)
print(f"  Baseline: {base.shape}")

# Select columns to merge (avoid duplicating existing columns)
merge_cols = [
    'RID',
    'baseline_CDGLOBAL', 'baseline_CDRSB',
    # Binary labels
    'CDR_worsen_3yr', 'CDR_worsen_5yr', 'CDR_worsen_10yr', 'CDR_worsen_alltime',
    # Censoring flags
    'CDR_worsen_censored_3yr', 'CDR_worsen_censored_5yr', 'CDR_worsen_censored_10yr',
    # Survival labels
    'CDR_surv_time', 'CDR_surv_event',
    'CDRSB_surv_time', 'CDRSB_surv_event',
    # Metadata
    'CDR_worsened', 'CDR_worsening_years',
    'CDR_worsened_CDRSB2', 'CDRSB_worsening_years',
    'CDR_last_followup_years', 'n_cdr_visits',
]

base = base.merge(cdr_targets[merge_cols], on='RID', how='left', suffixes=('', '_cdr'))

# Fill missing for subjects not in CDR table
for col in [
    'CDR_worsened', 'CDR_worsened_CDRSB2',
    'CDR_worsen_3yr', 'CDR_worsen_5yr', 'CDR_worsen_10yr', 'CDR_worsen_alltime',
]:
    if col in base.columns:
        base[col] = base[col].fillna(0)
for col in ['CDR_worsen_censored_3yr', 'CDR_worsen_censored_5yr',
             'CDR_worsen_censored_10yr']:
    if col in base.columns:
        base[col] = base[col].fillna(1)  # no CDR data = censored

# Fill survival time for missing subjects
base['CDR_surv_time'] = base['CDR_surv_time'].fillna(0.0)
base['CDRSB_surv_time'] = base['CDRSB_surv_time'].fillna(0.0)
base['CDR_surv_event'] = base['CDR_surv_event'].fillna(0).astype(int)
base['CDRSB_surv_event'] = base['CDRSB_surv_event'].fillna(0).astype(int)
base['n_cdr_visits'] = base['n_cdr_visits'].fillna(0).astype(int)

print(f"  After merge: {base.shape}")
print(f"  Subjects with CDR data: {(base['n_cdr_visits'] > 0).sum():,}")
print(f"  Subjects with ≥2 CDR visits: {(base['n_cdr_visits'] >= 2).sum():,}")

# ══════════════════════════════════════════════════════════════════
# 6. SAVE
# ══════════════════════════════════════════════════════════════════
# Save full baseline with CDR targets
out_path = os.path.join(OUT_DIR, 'ADNI_baseline_with_time_targets_v2.csv')
base.to_csv(out_path, index=False)
print(f"\n[6] Saved (overwrite): {out_path}")
print(f"  Size: {os.path.getsize(out_path) / 1024**2:.1f} MB")
print(f"  Total columns: {len(base.columns)}")

# Standalone CDR targets
cdr_target_cols = [
    'RID', 'baseline_CDGLOBAL', 'baseline_CDRSB',
    'CDR_worsen_3yr', 'CDR_worsen_5yr', 'CDR_worsen_10yr', 'CDR_worsen_alltime',
    'CDR_worsen_censored_3yr', 'CDR_worsen_censored_5yr', 'CDR_worsen_censored_10yr',
    'CDR_surv_time', 'CDR_surv_event',
    'CDRSB_surv_time', 'CDRSB_surv_event',
    'CDR_worsened', 'CDR_worsening_years',
    'CDR_worsened_CDRSB2', 'CDRSB_worsening_years',
    'CDR_last_followup_years', 'n_cdr_visits',
]
cdr_targets[cdr_target_cols].to_csv(
    os.path.join(OUT_DIR, 'CDR_time_targets.csv'), index=False)
print(f"  CDR targets saved: CDR_time_targets.csv ({len(cdr_target_cols)} columns)")

# Cross-endpoint overlap summary
print(f"\n{'=' * 60}")
print(f"  CROSS-ENDPOINT OVERLAP (MCI subjects, ≥2 CDR visits)")
print(f"{'=' * 60}")

mci_cdr = base[
    (base['baseline_diagnosis'] == 2.0) &
    (base['n_cdr_visits'] >= 2) &
    (base['CDR_surv_time'] > 0)
]
n_mci = len(mci_cdr)
dx_only = ((mci_cdr['converted_to_dementia'] == 1) & (mci_cdr['CDR_worsened'] == 0)).sum()
cdr_only = ((mci_cdr['converted_to_dementia'] == 0) & (mci_cdr['CDR_worsened'] == 1)).sum()
both = ((mci_cdr['converted_to_dementia'] == 1) & (mci_cdr['CDR_worsened'] == 1)).sum()
neither = ((mci_cdr['converted_to_dementia'] == 0) & (mci_cdr['CDR_worsened'] == 0)).sum()

print(f"  MCI with CDR data: {n_mci}")
print(f"  Both endpoints positive:     {both:>5} ({both/n_mci*100:4.1f}%)")
print(f"  DXSUM only (Dementia, no CDR worsen): {dx_only:>5} ({dx_only/n_mci*100:4.1f}%)")
print(f"  CDR only (worsen, no Dementia dx):    {cdr_only:>5} ({cdr_only/n_mci*100:4.1f}%)")
print(f"  Neither:                     {neither:>5} ({neither/n_mci*100:4.1f}%)")

# Agreement
agree = both + neither
kappa_denom = n_mci
po = agree / kappa_denom
pe = ((both+dx_only)*(both+cdr_only) + (cdr_only+neither)*(dx_only+neither)) / kappa_denom**2
kappa = (po - pe) / (1 - pe) if pe < 1 else 0
print(f"  Agreement: {agree}/{n_mci} = {po:.3f}")
print(f"  Cohen's Kappa: {kappa:.3f}")

print(f"\n✅ Done! CDR targets merged into baseline dataset.")

#!/usr/bin/env python3
"""
ADNI Data Preprocessing
=======================
Builds a clean baseline subject-level table from the 66 ADNI CSV exports.
Joins demographics, cognitive assessments, CSF/plasma biomarkers,
genetics (APOE4), and imaging features (FreeSurfer, amyloid/tau PET).

Output: ADNI_baseline.csv — one row per subject at their first visit.
"""

import os, sys, warnings, gc
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── Paths ────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'extracted')
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'processed')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Helpers ──────────────────────────────────────────────────────
def load_csv(name):
    path = os.path.join(DATA_DIR, f'All_Subjects_{name}_20May2026.csv')
    if os.path.exists(path):
        return pd.read_csv(path, low_memory=False)
    print(f"  [WARN] Missing: {path}")
    return None

def to_baseline(df, viscode_col='VISCODE'):
    """Keep only baseline visit rows. Baseline codes vary by ADNI phase."""
    baseline_codes = ['bl', 'sc', 'init']
    # Also catch '4_bl', '4_sc', '4_init' for ADNI4
    baseline_codes += ['4_bl', '4_sc', '4_init']
    if viscode_col in df.columns:
        return df[df[viscode_col].isin(baseline_codes)].copy()
    return df

def dedup_subjects(df, id_col='RID'):
    """Keep first row per subject after sorting by visit date if available."""
    if 'EXAMDATE' in df.columns or 'VISDATE' in df.columns:
        date_col = 'EXAMDATE' if 'EXAMDATE' in df.columns else 'VISDATE'
        df = df.sort_values(date_col)
    return df.drop_duplicates(subset=id_col, keep='first')


# ══════════════════════════════════════════════════════════════════
# 1. LOAD KEY TABLES
# ══════════════════════════════════════════════════════════════════
print("=" * 60)
print("ADNI Preprocessing: Building Baseline Dataset")
print("=" * 60)

print("\n[1] Loading tables...")

# Core tables
ptdemog = load_csv('PTDEMOG')        # Demographics
dxsum = load_csv('DXSUM')            # Diagnosis summary
cdr = load_csv('CDR')                # Clinical Dementia Rating
study_entry = load_csv('Study_Entry') # Study entry info
blchange = load_csv('BLCHANGE')       # Baseline change

# Cognitive
mmse = load_csv('MMSE')
moca = load_csv('MOCA')
adas = load_csv('ADAS')
neurobat = load_csv('NEUROBAT')
faq = load_csv('FAQ')
gdscale = load_csv('GDSCALE')
npiq = load_csv('NPIQ')
modhach = load_csv('MODHACH')
cci = load_csv('CCI')

# Genetics
apoe = load_csv('APOERES')

# CSF biomarkers (Roche Elecsys)
csf_bm = load_csv('UPENNBIOMK_ROCHE_ELECSYS')

# Plasma biomarkers
plasma_fuji = load_csv('UPENN_PLASMA_FUJIREBIO_QUANTERIX')
plasma_c2n = load_csv('C2N_PRECIVITYAD2_PLASMA')

# Imaging — structural MRI
ucsffsx7 = load_csv('UCSFFSX7')      # FreeSurfer v7
foxlabbsi = load_csv('FOXLABBSI')    # BSI volumes
ucd_wmh = load_csv('UCD_WMH')        # White matter hyperintensities

# Imaging — PET
amy_pet = load_csv('UCBERKELEY_AMY_6MM')     # Amyloid PET
tau_pet = load_csv('UCBERKELEY_TAU_6MM')     # Tau PET
taupvc_pet = load_csv('UCBERKELEY_TAUPVC_6MM') # Tau PVC PET

for name, df in [('PTDEMOG', ptdemog), ('DXSUM', dxsum), ('CDR', cdr),
                  ('MMSE', mmse), ('APOE', apoe), ('CSF_BM', csf_bm),
                  ('UCSFFSX7', ucsffsx7), ('AMY_PET', amy_pet)]:
    if df is not None:
        print(f"  {name}: {len(df):,} rows, {df['RID'].nunique():,} subjects")
    else:
        print(f"  {name}: NOT FOUND")


# ══════════════════════════════════════════════════════════════════
# 2. BUILD BASELINE DEMOGRAPHICS + DIAGNOSIS
# ══════════════════════════════════════════════════════════════════
print("\n[2] Building baseline demographics & diagnosis...")

# Study Entry — already one row per subject
study = study_entry[['subject_id', 'entry_age', 'entry_research_group']].copy()
study = study.rename(columns={'subject_id': 'PTID'})

# Demographics — baseline only
demo_bl = to_baseline(ptdemog)
demo_bl = dedup_subjects(demo_bl)

# Extract key demographics
demo = demo_bl[['PTID', 'RID', 'PHASE', 'PTGENDER', 'PTEDUCAT', 'PTMARRY',
                 'PTDOBYY', 'PTHAND', 'PTWORKHS']].copy()
demo['sex'] = demo['PTGENDER'].map({1: 1, 2: 0}).fillna(-4)  # 1=Male, 0=Female
demo['education_years'] = pd.to_numeric(demo['PTEDUCAT'], errors='coerce')

# Diagnosis — baseline only
dx_bl = to_baseline(dxsum)
dx_bl = dedup_subjects(dx_bl)
dx = dx_bl[['RID', 'DIAGNOSIS', 'DXNORM', 'DXNODEP', 'DXMCI', 'DXDSEV', 'DXDDUE', 'DXAD', 'DXPARK', 'DXDEP']].copy()
# DIAGNOSIS: 1=CN, 2=MCI, 3=AD
dx['AD_status'] = (dx['DIAGNOSIS'] == 3).astype(int)
dx['MCI_status'] = (dx['DIAGNOSIS'] == 2).astype(int)
dx['dementia_status'] = (dx['DIAGNOSIS'].isin([2, 3])).astype(int)

# CDR — baseline only
cdr_bl = to_baseline(cdr)
cdr_bl = dedup_subjects(cdr_bl)
cdr_k = cdr_bl[['RID', 'CDGLOBAL', 'CDRSB', 'CDMEMORY', 'CDORIENT',
                 'CDJUDGE', 'CDCOMMUN', 'CDHOME', 'CDCARE']].copy()

# Merge core
base = demo.merge(dx, on='RID', how='inner')
base = base.merge(cdr_k, on='RID', how='left')
base = base.merge(study, on='PTID', how='left')

# Compute age
base['entry_age'] = pd.to_numeric(base['entry_age'], errors='coerce')
base['age'] = base['entry_age'].copy()

print(f"  Baseline subjects: {len(base):,}")
print(f"  CN: {(base['DIAGNOSIS']==1).sum()}, "
      f"MCI: {(base['DIAGNOSIS']==2).sum()}, "
      f"AD: {(base['DIAGNOSIS']==3).sum()}")


# ══════════════════════════════════════════════════════════════════
# 3. COGNITIVE ASSESSMENTS
# ══════════════════════════════════════════════════════════════════
print("\n[3] Adding cognitive assessments...")

def add_cognitive_tests(base_df, tbl, name, columns, tbl_id='RID'):
    """Join cognitive test scores from a table onto the base dataframe."""
    if tbl is None:
        return base_df
    bl = to_baseline(tbl)
    bl = dedup_subjects(bl, id_col=tbl_id)
    # Keep only specified columns
    keep_cols = [tbl_id] + [c for c in columns if c in bl.columns]
    subset = bl[keep_cols].copy()
    # Remove duplicate columns that exist in base
    existing = [c for c in subset.columns if c in base_df.columns and c != tbl_id]
    subset = subset.drop(columns=existing, errors='ignore')
    # Add prefix
    rename = {c: f'{name}_{c}' for c in subset.columns if c != tbl_id}
    subset = subset.rename(columns=rename)
    return base_df.merge(subset, on=tbl_id, how='left')

# MMSE — key total score
base = add_cognitive_tests(base, mmse, 'MMSE', ['MMSCORE'], 'RID')

# MOCA — total score
base = add_cognitive_tests(base, moca, 'MOCA', ['MOCA'], 'RID')

# ADAS — total scores
base = add_cognitive_tests(base, adas, 'ADAS', ['TOTSCORE', 'TOTAL13'], 'RID')

# FAQ — functional
base = add_cognitive_tests(base, faq, 'FAQ', ['FAQTOTAL'], 'RID')

# GDSCALE — depression
base = add_cognitive_tests(base, gdscale, 'GDS', ['GDTOTAL'], 'RID')

# NPI-Q — neuropsychiatric
base = add_cognitive_tests(base, npiq, 'NPIQ', ['NPIA', 'NPIB', 'NPIC', 'NPID',
    'NPIE', 'NPIF', 'NPIG', 'NPIH', 'NPII', 'NPIJ', 'NPIK', 'NPIL'], 'RID')

# MODHACH — Hachinski ischemia score
base = add_cognitive_tests(base, modhach, 'HACH', ['HMSCORE'], 'RID')

# NEUROBAT — neuropsychological battery
neuro_cols = ['LIMMTOTAL', 'LDELTOTAL', 'AVTOT1', 'AVTOT2', 'AVTOT3',
              'AVTOT4', 'AVTOT5', 'AVERR1', 'AVERR2', 'AVERR3', 'AVERR4', 'AVERR5',
              'LMSTORY', 'CLOCKSCOR', 'COPYSCOR', 'DIGITSCOR', 'TRAILSCOR',
              'ANIMALS', 'VEGETABLES', 'VFTSCORE', 'BNTTOTAL', 'WAIS']
base = add_cognitive_tests(base, neurobat, 'NB', neuro_cols, 'RID')

# CCI — Charlson Comorbidity Index
base = add_cognitive_tests(base, cci, 'CCI',
    ['CCI1', 'CCI2', 'CCI3', 'CCI4', 'CCI5', 'CCI6', 'CCI7', 'CCI8',
     'CCI9', 'CCI10', 'CCI11', 'CCI12', 'CCI13', 'CCI14', 'CCI15',
     'CCI16', 'CCI17', 'CCI18', 'CCI19'], 'RID')

print(f"  After cognitive: {len(base):,} subjects, {len(base.columns)} columns")


# ══════════════════════════════════════════════════════════════════
# 4. APOE GENOTYPE
# ══════════════════════════════════════════════════════════════════
print("\n[4] Adding APOE genotype...")

if apoe is not None:
    apoe_bl = dedup_subjects(apoe)
    apoe_bl['APOE4_count'] = apoe_bl['GENOTYPE'].apply(
        lambda g: str(g).count('4'))
    apoe_bl['APOE4_carrier'] = (apoe_bl['APOE4_count'] >= 1).astype(int)
    apoe_bl['APOE_genotype'] = apoe_bl['GENOTYPE'].astype(str)
    base = base.merge(apoe_bl[['RID', 'APOE_genotype', 'APOE4_count', 'APOE4_carrier']],
                      on='RID', how='left')

print(f"  APOE4 carriers: {base['APOE4_carrier'].sum():,}")
print(f"  APOE4 non-carriers: {(base['APOE4_carrier']==0).sum():,}")


# ══════════════════════════════════════════════════════════════════
# 5. CSF BIOMARKERS (Roche Elecsys)
# ══════════════════════════════════════════════════════════════════
print("\n[5] Adding CSF biomarkers...")

if csf_bm is not None:
    csf_bl = to_baseline(csf_bm, viscode_col='VISCODE2')
    csf_bl = dedup_subjects(csf_bl)
    csf_cols = ['RID', 'ABETA40', 'ABETA42', 'TAU', 'PTAU']
    csf_vals = csf_bl[[c for c in csf_cols if c in csf_bl.columns]].copy()
    # Compute ratios
    for col in ['ABETA40', 'ABETA42', 'TAU', 'PTAU']:
        if col in csf_vals.columns:
            csf_vals[col] = pd.to_numeric(csf_vals[col], errors='coerce')
    if 'ABETA42' in csf_vals.columns and 'ABETA40' in csf_vals.columns:
        csf_vals['AB42_40_ratio'] = csf_vals['ABETA42'] / csf_vals['ABETA40']
    if 'PTAU' in csf_vals.columns and 'ABETA42' in csf_vals.columns:
        csf_vals['PTAU_AB42_ratio'] = csf_vals['PTAU'] / csf_vals['ABETA42']
    if 'TAU' in csf_vals.columns and 'ABETA42' in csf_vals.columns:
        csf_vals['TAU_AB42_ratio'] = csf_vals['TAU'] / csf_vals['ABETA42']
    base = base.merge(csf_vals, on='RID', how='left')

csf_cols = [c for c in base.columns if c in ['ABETA40', 'ABETA42', 'TAU', 'PTAU',
    'AB42_40_ratio', 'PTAU_AB42_ratio', 'TAU_AB42_ratio']]
print(f"  Subjects with CSF: {base['ABETA42'].notna().sum():,}")


# ══════════════════════════════════════════════════════════════════
# 6. PLASMA BIOMARKERS
# ══════════════════════════════════════════════════════════════════
print("\n[6] Adding plasma biomarkers...")

# Fujirebio / Quanterix
if plasma_fuji is not None:
    pf_bl = to_baseline(plasma_fuji)
    pf_bl = dedup_subjects(pf_bl)
    pf_cols = ['RID', 'pT217_F', 'AB42_F', 'AB40_F', 'AB42_AB40_F',
               'pT217_AB42_F', 'NfL_Q', 'GFAP_Q', 'NfL_F', 'GFAP_F']
    pf = pf_bl[[c for c in pf_cols if c in pf_bl.columns]].copy()
    for c in pf.columns:
        if c != 'RID':
            pf[c] = pd.to_numeric(pf[c], errors='coerce')
    base = base.merge(pf, on='RID', how='left', suffixes=('', '_pf'))

# C2N PrecivityAD2
if plasma_c2n is not None:
    pc_bl = to_baseline(plasma_c2n)
    pc_bl = dedup_subjects(pc_bl)
    pc_cols = ['RID', 'pT217_C2N', 'npT217_C2N', 'AB42_C2N', 'AB40_C2N',
               'AB42_AB40_C2N', 'pT217_npT217_C2N', 'APS2_C2N', 'APOE_C2N']
    pc = pc_bl[[c for c in pc_cols if c in pc_bl.columns]].copy()
    for c in pc.columns:
        if c != 'RID':
            pc[c] = pd.to_numeric(pc[c], errors='coerce')
    base = base.merge(pc, on='RID', how='left', suffixes=('', '_c2n'))

plasma_cols = [c for c in base.columns if any(p in c for p in
    ['pT217', 'AB42', 'AB40', 'NfL', 'GFAP', 'APS2'])]
print(f"  Plasma biomarker columns: {len(plasma_cols)}")
print(f"  Subjects with pT217: {base[[c for c in base.columns if 'pT217' in c][0]].notna().sum() if any('pT217' in c for c in base.columns) else 0:,}")


# ══════════════════════════════════════════════════════════════════
# 7. IMAGING — FreeSurfer v7 Structural MRI
# ══════════════════════════════════════════════════════════════════
print("\n[7] Adding FreeSurfer structural MRI features...")

if ucsffsx7 is not None:
    # Keep baseline scans only
    fs_bl = to_baseline(ucsffsx7)
    fs_bl = dedup_subjects(fs_bl)

    # Drop metadata columns
    meta_cols = ['PHASE', 'PTID', 'VISCODE', 'VISCODE2', 'IMAGEUID',
                 'FIELD_STRENGTH', 'EXAMDATE', 'RUNDATE', 'STATUS', 'FSVER',
                 'OVERALLQC', 'TEMPQC', 'FRONTQC', 'PARQC', 'INSULAQC',
                 'OCCQC', 'BGQC', 'CWMQC', 'VENTQC', 'HIPPOQC', 'update_stamp']
    fs_feat = fs_bl.drop(columns=[c for c in meta_cols if c in fs_bl.columns],
                         errors='ignore')

    # Add FS_ prefix to avoid name collisions
    fs_feat = fs_feat.rename(columns={c: f'FS_{c}' for c in fs_feat.columns if c != 'RID'})

    base = base.merge(fs_feat, on='RID', how='left')

fs_count = sum(1 for c in base.columns if c.startswith('FS_'))
print(f"  FreeSurfer features: {fs_count}")
print(f"  Subjects with FS data: {base['FS_ST101SV'].notna().sum() if 'FS_ST101SV' in base.columns else 0:,}")


# ══════════════════════════════════════════════════════════════════
# 8. IMAGING — BSI & WMH
# ══════════════════════════════════════════════════════════════════
print("\n[8] Adding BSI brain volumes & WMH...")

if foxlabbsi is not None:
    bsi_bl = to_baseline(foxlabbsi)
    bsi_bl = dedup_subjects(bsi_bl)
    bsi_cols = ['RID', 'BRAINVOL', 'VENTVOL', 'HIPPOVOL_R', 'HIPPOVOL_L',
                'DBCBBSI', 'VBSI', 'HBSI_R', 'HBSI_L']
    bsi = bsi_bl[[c for c in bsi_cols if c in bsi_bl.columns]].copy()
    bsi = bsi.rename(columns={c: f'BSI_{c}' for c in bsi.columns if c != 'RID'})
    base = base.merge(bsi, on='RID', how='left')

if ucd_wmh is not None:
    wmh_bl = to_baseline(ucd_wmh)
    wmh_bl = dedup_subjects(wmh_bl)
    wmh_cols = ['RID', 'CEREBRUM_TCV', 'CEREBRUM_TCB', 'CEREBRUM_TCC',
                'CEREBRUM_GRAY', 'CEREBRUM_WHITE', 'LEFT_HIPPO', 'RIGHT_HIPPO',
                'LEFT_LAT_VENT', 'RIGHT_LAT_VENT', 'WMH_VOLUME',
                'WMH_DEEP', 'WMH_PERI']
    wmh = wmh_bl[[c for c in wmh_cols if c in wmh_bl.columns]].copy()
    wmh = wmh.rename(columns={c: f'WMH_{c}' for c in wmh.columns if c != 'RID'})
    base = base.merge(wmh, on='RID', how='left')

print(f"  BSI+WMH features added")


# ══════════════════════════════════════════════════════════════════
# 9. IMAGING — Amyloid & Tau PET
# ══════════════════════════════════════════════════════════════════
print("\n[9] Adding PET features...")

if amy_pet is not None:
    amy_bl = to_baseline(amy_pet)
    amy_bl = dedup_subjects(amy_bl)
    amy_cols = ['RID', 'AMYLOID_STATUS', 'CENTILOIDS', 'SUMMARY_SUVR',
                'COMPOSITE_REF_SUVR', 'WHOLECEREBELLUM_SUVR']
    amy = amy_bl[[c for c in amy_cols if c in amy_bl.columns]].copy()
    for c in amy.columns:
        if c != 'RID':
            amy[c] = pd.to_numeric(amy[c], errors='coerce')
    amy = amy.rename(columns={c: f'AMY_{c}' for c in amy.columns if c != 'RID'})
    base = base.merge(amy, on='RID', how='left')

if tau_pet is not None:
    tau_bl = to_baseline(tau_pet)
    tau_bl = dedup_subjects(tau_bl)
    tau_cols = ['RID', 'META_TEMPORAL_SUVR', 'CTX_ENTORHINAL_SUVR',
                'INFERIORCEREBELLUM_SUVR']
    tau = tau_bl[[c for c in tau_cols if c in tau_bl.columns]].copy()
    tau = tau.rename(columns={c: f'TAU_{c}' for c in tau.columns if c != 'RID'})
    for c in tau.columns:
        if c != 'RID':
            tau[c] = pd.to_numeric(tau[c], errors='coerce')
    base = base.merge(tau, on='RID', how='left')

print(f"  Subjects with Amyloid PET: {base['AMY_CENTILOIDS'].notna().sum() if 'AMY_CENTILOIDS' in base.columns else 0:,}")
print(f"  Subjects with Tau PET: {base['TAU_META_TEMPORAL_SUVR'].notna().sum() if 'TAU_META_TEMPORAL_SUVR' in base.columns else 0:,}")


# ══════════════════════════════════════════════════════════════════
# 10. CLEAN & SAVE
# ══════════════════════════════════════════════════════════════════
print("\n[10] Cleaning and saving...")

# Drop rows without diagnosis
base = base[base['DIAGNOSIS'].notna()].copy()
print(f"  After removing missing diagnosis: {len(base):,}")

# Drop mostly-empty columns (>60% missing)
missing_pct = base.isnull().mean()
high_miss = missing_pct[missing_pct > 0.60].index
base = base.drop(columns=high_miss)
print(f"  Dropped {len(high_miss)} columns with >60% missing")

# Drop columns with zero variance
nunique = base.nunique()
constant_cols = nunique[nunique <= 1].index
base = base.drop(columns=constant_cols)
print(f"  Dropped {len(constant_cols)} constant columns")

# Separate feature and ID columns
id_cols = ['PTID', 'RID', 'PHASE', 'APOE_genotype', 'subject_id',
           'entry_research_group', 'PTDOBYY', 'VISCODE', 'VISCODE2']
target_cols = ['AD_status', 'MCI_status', 'dementia_status', 'DIAGNOSIS',
               'DXNORM', 'DXNODEP', 'DXMCI', 'DXDSEV', 'CDGLOBAL', 'CDRSB']

# Keep ID cols for reference but don't use as features
id_df = base[[c for c in id_cols if c in base.columns]].copy()
target_df = base[[c for c in target_cols if c in base.columns]].copy()

# Build feature-only dataframe
exclude = set(id_cols + target_cols)
feature_cols = [c for c in base.columns if c not in exclude]
feature_df = base[feature_cols].copy()

# Convert all features to numeric
for col in feature_df.columns:
    feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')

# Remove near-zero variance features
stds = feature_df.std()
low_var = stds[stds < 0.001].index
feature_df = feature_df.drop(columns=low_var)
print(f"  Dropped {len(low_var)} near-zero variance features")

# Fill remaining NaN with median (for training compatibility)
feature_df = feature_df.fillna(feature_df.median())

# Detect any remaining non-numeric columns
non_num = [c for c in feature_df.columns if feature_df[c].dtype == 'object']
if non_num:
    print(f"  Dropping {len(non_num)} non-numeric columns: {non_num}")
    feature_df = feature_df.drop(columns=non_num)

print(f"\n  Final feature matrix: {feature_df.shape}")
print(f"  Targets: {list(target_df.columns)}")
print(f"  Total subjects: {len(feature_df):,}")
print(f"  AD: {target_df['AD_status'].sum():,}, "
      f"MCI: {target_df['MCI_status'].sum():,}, "
      f"Dementia: {target_df['dementia_status'].sum():,}")

# Combine & save
final = pd.concat([id_df.reset_index(drop=True),
                   feature_df.reset_index(drop=True),
                   target_df.reset_index(drop=True)], axis=1)

out_path = os.path.join(OUT_DIR, 'ADNI_baseline.csv')
final.to_csv(out_path, index=False)
print(f"\n  Saved: {out_path}")
print(f"  Size: {os.path.getsize(out_path) / 1024**2:.1f} MB")

# Also save feature-only + target for pipeline
feature_df.to_csv(os.path.join(OUT_DIR, 'ADNI_features.csv'), index=False)
target_df.to_csv(os.path.join(OUT_DIR, 'ADNI_targets.csv'), index=False)

# Feature list
pd.DataFrame({'feature': list(feature_df.columns)}).to_csv(
    os.path.join(OUT_DIR, 'ADNI_feature_list.csv'), index=False)

print("\n✅ ADNI preprocessing complete!")

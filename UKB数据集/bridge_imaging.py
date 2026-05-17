#!/usr/bin/env python3
"""
Bridge Script: Extract Brain MRI Imaging-Derived Phenotypes (IDPs)
==================================================================
Extracts all available brain MRI IDP features from the .npz feature files
and merges them with the existing clinical features from Preprocessed_Data.csv.

UK Biobank brain MRI categories covered:
  - Category 110: T1 structural brain MRI (regional volumes, cortical measures)
  - Category 134: dMRI TBSS skeleton (FA/MD in white matter tracts)
  - Category 135-139: Various derived diffusion MRI measures
  - Category 190-199: Additional brain MRI derived phenotypes
  - Category 100: Brain MRI QC and protocol fields

Output:
  - Preprocessed_Data_imaging.csv : clinical + imaging features
  - imaging_feature_list.csv      : list of imaging feature columns
  - imaging_only.csv              : imaging features only
"""
import os, sys, gc, time
import numpy as np
import pandas as pd
import configparser

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'UKB数据集'))

UKB_DIR = os.path.join(PROJECT_ROOT, 'UKB数据集')
FEATURE_DIR = os.path.join(UKB_DIR, 'features')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'Preprocessed_Data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONFIG_DATA = os.path.join(UKB_DIR, 'config_data.ini')
DATA_LIST_CSV = os.path.join(UKB_DIR, 'data_list.csv')
CLINICAL_CSV = os.path.join(OUTPUT_DIR, 'Preprocessed_Data.csv')
N_OVERALL = 502241

data_config = configparser.ConfigParser()
data_config.optionxform = lambda option: option
data_config.read(CONFIG_DATA)
data_list = pd.read_csv(DATA_LIST_CSV, low_memory=False)

# ============================================================================
# STEP 1: Identify all available brain MRI imaging fields
# ============================================================================

# Brain MRI category IDs for actual imaging-derived phenotypes (IDPs)
# Excluding categories that contain cognitive/behavioral tests done during imaging visits
BRAIN_MRI_CATEGORIES = {
    '110',   # T1 structural brain MRI (regional volumes)
    '134',   # dMRI TBSS skeleton (FA/MD)
    '135',   # dMRI tract measures
    '136',   # dMRI tract measures (continued)
    '137',   # dMRI tract measures (continued)
    '138',   # dMRI NODDI
    '139',   # Brain MRI IDPs
    '190',   # Brain MRI additional
    '191',   # Brain MRI additional
    '192',   # Brain MRI additional
    '193',   # Brain MRI additional
    '194',   # Brain MRI additional
    '195',   # Brain MRI additional
    '196',   # Brain MRI additional
    '197',   # Brain MRI additional
}

# Blacklist: categories that look like imaging but contain non-imaging data
EXCLUDE_CATEGORIES = {'100', '121', '122'}

# Path keywords to exclude (non-imaging data collected during imaging visits)
EXCLUDE_PATH_KEYWORDS = [
    'online follow-up', 'mental health', 'cognitive function online',
    'trail making', 'symbol digit',
]

# Also search by path pattern for robustness
def is_brain_mri_field(row):
    """Determine if a field is a brain MRI derived phenotype."""
    path_str = str(row.get('Path', '')).lower()
    category_str = str(row.get('Category', ''))

    # Skip raw DICOM/NIFTI bulk data
    vt = str(row.get('ValueType', ''))
    if vt in ('Text', 'Compound', 'Bulk'):
        return False

    # Skip excluded categories
    if category_str in EXCLUDE_CATEGORIES:
        return False

    # Skip excluded path keywords
    for kw in EXCLUDE_PATH_KEYWORDS:
        if kw in path_str:
            return False

    # Check category
    if category_str in BRAIN_MRI_CATEGORIES:
        return True

    # Check path for brain MRI keywords (only for categories not already excluded)
    brain_keywords = ['brain mri', 't1 brain', 'diffusion brain', 'dmri',
                      'tbss', 'freesurfer', 'brain segmentation',
                      'cortical thickness', 'subcortical volume',
                      'white matter hyperintensit', 'brain atrophy',
                      'hippocamp', 'amygdala', 'brain volume']
    for kw in brain_keywords:
        if kw in path_str:
            return True

    return False


MAX_COLS_PER_FIELD = 100  # skip fields with excessively many columns (not IDPs)


# Collect all brain MRI fields that have .npz files
imaging_fields = []
imaging_field_info = {}

for _, row in data_list.iterrows():
    fid = str(int(row['FieldID'])) if not np.isnan(row.get('FieldID', float('nan'))) else None
    if not fid:
        continue

    feat_file = os.path.join(FEATURE_DIR, f'{fid}.npz')
    if not os.path.exists(feat_file):
        continue
    if fid not in data_config:
        continue

    if is_brain_mri_field(row):
        imaging_fields.append(fid)
        imaging_field_info[fid] = {
            'Field': row.get('Field', ''),
            'ValueType': data_config[fid].get('ValueType', 'N/A'),
            'Category': str(row.get('Category', '')),
            'Path': str(row.get('Path', '')),
        }

imaging_fields = sorted(set(imaging_fields))
print(f"Found {len(imaging_fields)} brain MRI imaging fields with .npz files")

# Count by ValueType
vt_counts = {}
for fid in imaging_fields:
    vt = imaging_field_info[fid]['ValueType']
    vt_counts[vt] = vt_counts.get(vt, 0) + 1
print(f"  ValueType distribution: {vt_counts}")

# ============================================================================
# STEP 2: Extract baseline imaging features
# ============================================================================

def extract_imaging_feature(field_id, value_type, feature):
    """Extract baseline imaging values. Imaging fields use instance 0 (first scan)."""
    n_cols = feature.shape[1]
    result = {}

    if feature.shape[1] > MAX_COLS_PER_FIELD:
        return {}  # skip fields with too many columns (not real IDPs)

    if value_type in ('Continuous', 'Integer', 'Date'):
        if n_cols >= 2:
            values = feature[:, 0].copy().astype(np.float64)
            missing_mask = feature[:, 1] == 1
            values[missing_mask] = np.nan
            result[f'{field_id}-0.0'] = values
        elif n_cols == 1:
            result[f'{field_id}-0.0'] = feature[:, 0].astype(np.float64)

    elif value_type == 'Categorical single':
        # For categorical, find v_count (number of categories)
        # Categorical single with 1 instance has v_count columns
        # With 2 instances it has 2*v_count columns
        v_count = n_cols
        for vc in range(2, 50):
            if n_cols % vc == 0 and n_cols // vc <= 2:
                v_count = vc
                break
        for c in range(v_count):
            if c < n_cols:
                result[f'{field_id}-0.0_c{c}'] = feature[:, c].astype(np.float64)

    elif value_type == 'Categorical multiple':
        vc_guess = n_cols // 2
        for vc in range(2, 30):
            if n_cols % (vc * 2) == 0:
                vc_guess = vc
                break
        for c in range(vc_guess):
            pos_idx = c * 2
            neg_idx = c * 2 + 1
            if pos_idx < n_cols:
                result[f'{field_id}-0.0_{c}_pos'] = feature[:, pos_idx].astype(np.float64)
            if neg_idx < n_cols:
                result[f'{field_id}-0.0_{c}_neg'] = feature[:, neg_idx].astype(np.float64)

    return result


print(f"\n{'='*60}")
print(f"Extracting imaging features...")
print(f"{'='*60}")

CHUNK_SIZE = 200
all_chunks = []
processed = 0
skipped = 0

for i in range(0, len(imaging_fields), CHUNK_SIZE):
    chunk_ids = imaging_fields[i:i + CHUNK_SIZE]
    chunk_data = {}

    for field_id in chunk_ids:
        if field_id not in data_config:
            skipped += 1
            continue
        cfg = data_config[field_id]
        feat_file = os.path.join(FEATURE_DIR, f'{field_id}.npz')
        try:
            vt = cfg.get('ValueType', 'N/A')
            feature = np.load(feat_file, allow_pickle=True)['feature']
            extracted = extract_imaging_feature(field_id, vt, feature)
            chunk_data.update(extracted)
            processed += 1
        except Exception:
            skipped += 1

    if chunk_data:
        df_chunk = pd.DataFrame(chunk_data)
        all_chunks.append(df_chunk)
        gc.collect()

    chunk_num = i // CHUNK_SIZE + 1
    total_chunks = (len(imaging_fields) + CHUNK_SIZE - 1) // CHUNK_SIZE
    total_cols = sum(c.shape[1] for c in all_chunks)
    print(f"  Chunk {chunk_num}/{total_chunks}: {len(chunk_data)} cols, "
          f"running total: {total_cols} cols, "
          f"mem: {sum(c.memory_usage(deep=True).sum() for c in all_chunks)/1024**2:.0f} MB")

print(f"\n  Processed: {processed}, Skipped: {skipped}")

# ============================================================================
# STEP 3: Assemble imaging feature matrix
# ============================================================================

print(f"\n{'='*60}")
print(f"Assembling imaging feature matrix...")
print(f"{'='*60}")

t0 = time.time()
X_img = pd.concat(all_chunks, axis=1)
del all_chunks
gc.collect()
print(f"  Shape: {X_img.shape} ({time.time()-t0:.1f}s)")

# Remove constant columns
img_cols_before = X_img.shape[1]
constant_cols = [c for c in X_img.columns if X_img[c].nunique() <= 1]
if constant_cols:
    X_img.drop(columns=constant_cols, inplace=True)
    print(f"  Removed {len(constant_cols)} constant columns")

# Missing rate summary
missing_rates = X_img.isnull().mean()
n_has_imaging = (missing_rates < 1.0).sum()
n_all_missing = (missing_rates == 1.0).sum()
print(f"  Columns with any data: {n_has_imaging}")
print(f"  Columns with 100% missing: {n_all_missing}")

# Remove 100% missing columns
all_missing_cols = missing_rates[missing_rates == 1.0].index
if len(all_missing_cols) > 0:
    X_img.drop(columns=all_missing_cols, inplace=True)
    print(f"  Removed {len(all_missing_cols)} columns with 100%% missing")

# Add has_imaging flag based on core T1 structural field (25000: brain volume scaling)
# This field is only available for participants who actually had a brain MRI scan
if '25000-0.0' in X_img.columns:
    has_imaging = (~X_img['25000-0.0'].isnull()).astype(np.int32)
else:
    has_imaging = (~X_img.isnull().all(axis=1)).astype(np.int32)
X_img['has_brain_mri'] = has_imaging

print(f"  Final imaging features: {X_img.shape[1]} columns")
print(f"  Participants with imaging: {has_imaging.sum():,} "
      f"({has_imaging.sum()/N_OVERALL*100:.1f}%)")

# ============================================================================
# STEP 4: Merge with clinical features
# ============================================================================

print(f"\n{'='*60}")
print(f"Merging with clinical features...")
print(f"{'='*60}")

t0 = time.time()

if os.path.exists(CLINICAL_CSV):
    print(f"  Loading clinical data from {CLINICAL_CSV}...")
    df_clinical = pd.read_csv(CLINICAL_CSV)
    clinical_cols = [c for c in df_clinical.columns
                     if c not in X_img.columns]
    print(f"  Clinical: {len(clinical_cols)} columns, {len(df_clinical):,} rows")
else:
    print(f"  WARNING: {CLINICAL_CSV} not found!")
    print(f"  Run bridge_to_training_v3.py first to generate clinical features.")
    print(f"  Continuing with imaging features only...")
    df_clinical = None

if df_clinical is not None:
    # Align rows
    assert len(df_clinical) == len(X_img), \
        f"Row mismatch: clinical {len(df_clinical)} vs imaging {len(X_img)}"
    df_combined = pd.concat([df_clinical[clinical_cols], X_img], axis=1)
else:
    df_combined = X_img.copy()

# Save imaging feature list
img_feature_cols = list(X_img.columns)
pd.DataFrame({'feature': img_feature_cols}).to_csv(
    os.path.join(OUTPUT_DIR, 'imaging_feature_list.csv'), index=False)

# Save imaging-only matrix
print(f"  Saving imaging-only features...")
X_img.to_csv(os.path.join(OUTPUT_DIR, 'imaging_only.csv'), index=False)

# ============================================================================
# STEP 5: Save combined data
# ============================================================================

print(f"\n{'='*60}")
print(f"Saving combined data...")
print(f"{'='*60}")

combined_path = os.path.join(OUTPUT_DIR, 'Preprocessed_Data_imaging.csv')
t0 = time.time()
df_combined.to_csv(combined_path, index=False)
file_size_mb = os.path.getsize(combined_path) / (1024 * 1024)

print(f"  Combined: {df_combined.shape[1]} columns × {df_combined.shape[0]:,} rows")
print(f"  File size: {file_size_mb:.1f} MB")
print(f"  Save time: {time.time()-t0:.0f}s")

# Count unique field IDs
img_fid_set = set(c.split('-')[0] for c in img_feature_cols if c.startswith(('1', '2')))
print(f"\n  Unique imaging field IDs: {len(img_fid_set)}")
print(f"  Imaging feature columns: {len(img_feature_cols)}")

print(f"\nDone! Output written to {OUTPUT_DIR}/")
print(f"  - Preprocessed_Data_imaging.csv : clinical + imaging features")
print(f"  - imaging_feature_list.csv      : list of imaging feature names")
print(f"  - imaging_only.csv              : imaging features only")

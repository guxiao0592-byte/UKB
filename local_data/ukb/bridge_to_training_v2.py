#!/usr/bin/env python3
"""
Expanded Bridge Script: Transform ALL available UKB features to Preprocessed_Data.csv
Includes Type 1 (primary) and Type 2 (derived) features, matching the paper's ~366 approach.
"""

import os, sys, gc, time
import numpy as np
import pandas as pd
import configparser
from collections import Counter

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'UKB数据集'))

UKB_DIR = os.path.join(PROJECT_ROOT, 'UKB数据集')
FEATURE_DIR = os.path.join(UKB_DIR, 'features')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'Preprocessed_Data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONFIG_DATA = os.path.join(UKB_DIR, 'config_data.ini')
DATA_LIST_CSV = os.path.join(UKB_DIR, 'data_list.csv')
N_OVERALL = 502241
SECONDS_PER_YEAR = 31536000.0

data_config = configparser.ConfigParser()
data_config.optionxform = lambda option: option
data_config.read(CONFIG_DATA)

data_list = pd.read_csv(DATA_LIST_CSV, low_memory=False)

# ===== FIELD SELECTION =====
# Fields to ALWAYS exclude
EXCLUDE_FIELD_IDS = {
    '42018', '42019', '42020', '42021', '42022', '42023',  # Dementia/AD/VD dates
    '42024', '42025', '42006', '42007',                      # Stroke dates
    '53', '40007', '40008',                                   # Visit date, age at death
    '42016', '42017',                                         # Date of first X diagnosis
}

# From bridge_to_training.py - fields that the training pipeline references
CORE_FIELDS = [
    '20023', '400', '398', '399', '403', '404',
    '3062', '3063', '3064', '46', '47', '23104', '21001',
    '23107', '23108', '23110', '23111', '23112', '23115', '23116', '4080',
    '21022', '31', '21000', '22006', '137', '3526', '845', '2443', '2188',
    '1558', '1031', '1329', '1339', '20116', '22032', '189',
    '30710', '30650', '30620', '30690', '30040',
    '20002', '20110', '20008', '20010',
    '131287', '131351', '34', '52',
]

# Get Type 1 & 2 fields from data_list (matching the paper's feature sources)
type1_mask = data_list['Type'] == 1.0  # Primary data
type2_mask = data_list['Type'] == 2.0  # Derived data
type3_mask = data_list['Type'] == 3.0  # Other (includes some useful data)
primary_fields = data_list[type1_mask | type2_mask | type3_mask]

# Get field IDs that have .npz feature files
primary_field_ids = set()
for _, row in primary_fields.iterrows():
    fid = str(int(row['FieldID'])) if not np.isnan(row['FieldID']) else None
    if fid and fid not in EXCLUDE_FIELD_IDS:
        feat_file = os.path.join(FEATURE_DIR, f'{fid}.npz')
        if os.path.exists(feat_file) and fid in data_config:
            primary_field_ids.add(fid)

# Merge with core fields
all_field_ids = primary_field_ids | set(CORE_FIELDS)
all_field_ids -= EXCLUDE_FIELD_IDS

# Remove HES-related fields (Type 4 in data_list)
type4_mask = data_list['Type'] == 4.0
hes_field_ids = set()
for _, row in data_list[type4_mask].iterrows():
    fid = str(int(row['FieldID'])) if not np.isnan(row['FieldID']) else None
    if fid:
        hes_field_ids.add(fid)

# Also remove fields that look like HES (41xxx, 42xxx starting fields)
for fid in list(all_field_ids):
    if fid.startswith('41') or fid.startswith('42'):
        if fid not in CORE_FIELDS:
            all_field_ids.discard(fid)

print(f"Selected {len(all_field_ids)} field IDs for feature extraction")
print(f"  Type 1+2+3 fields with data: {len(primary_field_ids)}")
print(f"  Core training fields added: {len(CORE_FIELDS - primary_field_ids)}")
print(f"  HES/outcome fields excluded: {len(all_field_ids & (hes_field_ids | EXCLUDE_FIELD_IDS))}")

# ===== HELPER: Extract column names and values =====

def get_column_names(field_id, value_type, instances, array_n, feature):
    M = instances * array_n
    n_cols = feature.shape[1]
    col_names = []

    if value_type in ('Continuous', 'Integer'):
        for m in range(M):
            col_idx = m * 2
            if col_idx < n_cols:
                arr = m % array_n
                col_names.append(f'{field_id}-0.{arr}')

    elif value_type == 'Date':
        for m in range(M):
            col_idx = m * 2
            if col_idx < n_cols:
                col_names.append(f'{field_id}-0.{m}')

    elif value_type == 'Categorical single':
        v_count = n_cols // M if M > 0 else n_cols
        for m in range(M):
            arr = m % array_n
            for c in range(v_count):
                col_names.append(f'{field_id}-0.{arr}_c{c}')

    elif value_type == 'Categorical multiple':
        v_count = n_cols // (2 * max(1, instances))
        for m in range(min(M, instances)):
            for c in range(v_count):
                col_names.append(f'{field_id}-{m}.{c}_pos')
                col_names.append(f'{field_id}-{m}.{c}_neg')

    else:
        for i in range(min(n_cols, M)):
            col_names.append(f'{field_id}-0.{i}')

    return col_names


def extract_values(field_id, value_type, instances, array_n, feature):
    M = instances * array_n
    n_cols = feature.shape[1]
    result = {}

    if value_type in ('Continuous', 'Integer', 'Date'):
        for m in range(M):
            col_idx = m * 2
            if col_idx < n_cols:
                arr = m % array_n
                values = feature[:, col_idx].copy().astype(np.float64)
                if col_idx + 1 < n_cols:
                    missing_mask = feature[:, col_idx + 1] == 1
                    values[missing_mask] = np.nan
                result[f'{field_id}-0.{arr}'] = values

    elif value_type == 'Categorical single':
        v_count = n_cols // M if M > 0 else n_cols
        for m in range(min(M, n_cols // max(v_count, 1) if v_count > 0 else n_cols)):
            arr = m % array_n
            for c in range(v_count):
                idx = m * v_count + c
                if idx < n_cols:
                    result[f'{field_id}-0.{arr}_c{c}'] = feature[:, idx].astype(np.float64)

    elif value_type == 'Categorical multiple':
        v_count = n_cols // (2 * max(1, instances))
        for m in range(min(M, instances)):
            for c in range(v_count):
                pos_idx = c * 2 + m * v_count * 2
                neg_idx = c * 2 + 1 + m * v_count * 2
                if pos_idx < n_cols:
                    result[f'{field_id}-{m}.{c}_pos'] = feature[:, pos_idx].astype(np.float64)
                if neg_idx < n_cols:
                    result[f'{field_id}-{m}.{c}_neg'] = feature[:, neg_idx].astype(np.float64)

    return result


# ===== LOAD TARGET VARIABLES =====
print("\n" + "=" * 60)
print("Step 1: Deriving target variables from date fields")
print("=" * 60)

def load_date_feature(field_id):
    fpath = os.path.join(FEATURE_DIR, f'{field_id}.npz')
    if not os.path.exists(fpath):
        return None
    data = np.load(fpath, allow_pickle=True)['feature']
    values = data[:, 0].copy()
    if data.shape[1] >= 2:
        missing = data[:, 1] == 1
        values[missing] = np.nan
    return values

date_visit   = load_date_feature('53')
date_dem     = load_date_feature('42018')
date_ad      = load_date_feature('42020')
date_vd      = load_date_feature('42022')
date_stroke  = load_date_feature('42006')

dementia_status = np.where(~np.isnan(date_dem), 1, 0).astype(np.int32)
dementia_years = ((date_dem - date_visit) / SECONDS_PER_YEAR).astype(np.float32)
ad_status = np.where(~np.isnan(date_ad), 1, 0).astype(np.int32)
ad_years = ((date_ad - date_visit) / SECONDS_PER_YEAR).astype(np.float32)
vd_status = np.where(~np.isnan(date_vd), 1, 0).astype(np.int32)
vd_years = ((date_vd - date_visit) / SECONDS_PER_YEAR).astype(np.float32)
stroke_status = np.where(~np.isnan(date_stroke), 1, 0).astype(np.int32)
stroke_years = ((date_stroke - date_visit) / SECONDS_PER_YEAR).astype(np.float32)

target_dict = {
    'dementia_status': dementia_status,
    'dementia_years': dementia_years,
    'AD_status': ad_status,
    'AD_years': ad_years,
    'VD_status': vd_status,
    'VD_years': vd_years,
    'stroke_status': stroke_status,
    'stroke_years': stroke_years,
}
target_df = pd.DataFrame(target_dict)
target_df.to_csv(os.path.join(OUTPUT_DIR, 'Dementia_target.csv'), index=False)

print(f"  Dementia: {dementia_status.sum()}, AD: {ad_status.sum()}, "
      f"Stroke: {stroke_status.sum()} / {N_OVERALL}")


# ===== BUILD FEATURE MATRIX =====
print("\n" + "=" * 60)
print(f"Step 2: Building feature matrix from {len(all_field_ids)} fields")
print("=" * 60)

all_field_data = {}
processed = 0
skipped_not_found = 0
skipped_no_config = 0
skipped_errors = 0
total_columns = 0

for field_id in sorted(all_field_ids):
    if field_id not in data_config:
        skipped_no_config += 1
        continue

    feat_file = os.path.join(FEATURE_DIR, f'{field_id}.npz')
    if not os.path.exists(feat_file):
        skipped_not_found += 1
        continue

    try:
        cfg = data_config[field_id]
        value_type = cfg.get('ValueType', 'N/A')
        instances = int(cfg.get('Instances', '1'))
        array_n = int(cfg.get('Array', '1'))

        feature = np.load(feat_file, allow_pickle=True)['feature']
        extracted = extract_values(field_id, value_type, instances, array_n, feature)
        all_field_data.update(extracted)
        total_columns += len(extracted)
        processed += 1

        if processed % 100 == 0:
            print(f"  Processed {processed} fields, {total_columns} feature columns, "
                  f"mem usage: {sum(v.nbytes for v in all_field_data.values())/1024/1024:.0f} MB")

    except Exception as e:
        skipped_errors += 1
        if skipped_errors <= 10:
            print(f"  ERROR field {field_id}: {e}")

print(f"\nField processing summary:")
print(f"  Successfully processed: {processed}")
print(f"  Skipped (not found):   {skipped_not_found}")
print(f"  Skipped (no config):   {skipped_no_config}")
print(f"  Skipped (errors):      {skipped_errors}")
print(f"  Total feature columns: {total_columns}")


# ===== BUILD DATAFRAME & POST-PROCESS =====
print("\n" + "=" * 60)
print("Step 3: Creating DataFrame")
print("=" * 60)

t0 = time.time()
X_df = pd.DataFrame(all_field_data)
print(f"  DataFrame created in {time.time()-t0:.1f}s")
print(f"  Shape: {X_df.shape}")

# Free memory
del all_field_data
gc.collect()

# ===== FILTER FEATURES WITH >40% MISSING =====
print("\n" + "=" * 60)
print("Step 4: Filtering features (>40% missing)")
print("=" * 60)

missing_rates = X_df.isnull().mean()
high_missing = missing_rates[missing_rates > 0.4]
print(f"  Features with >40% missing: {len(high_missing)} / {len(missing_rates)}")

X_df = X_df.drop(columns=high_missing.index)
print(f"  After filtering: {X_df.shape}")

# Add target variables
print("\nAdding target variables...")
result_df = pd.concat([X_df, target_df], axis=1)
print(f"  Full DataFrame shape: {result_df.shape}")

# ===== SAVE =====
print("\n" + "=" * 60)
print("Step 5: Saving")
print("=" * 60)

csv_path = os.path.join(OUTPUT_DIR, 'Preprocessed_Data.csv')
t0 = time.time()
result_df.to_csv(csv_path, index=False)
print(f"  Saved in {time.time()-t0:.1f}s")
file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
print(f"  File size: {file_size_mb:.1f} MB")

# Save feature list
feat_cols_list = [c for c in X_df.columns]
feat_df = pd.DataFrame({'Features': feat_cols_list})
feat_df.to_csv(os.path.join(OUTPUT_DIR, 'raw_features.csv'), index=False)

# Save FieldID_selected.csv for training compatibility
field_ids_set = set()
for col in feat_cols_list:
    fid = col.split('-')[0]
    field_ids_set.add(fid)

field_df = pd.DataFrame({'Features': feat_cols_list})
field_df['Field'] = field_df['Features']
field_df['Path'] = ''
field_df['ValueType'] = ''
field_df['Units'] = ''
field_df.to_csv(os.path.join(PROJECT_ROOT, 'local_data', 'Data', 'FieldID_selected.csv'), index=False)

print(f"\n  Unique field IDs: {len(field_ids_set)}")
print(f"  Feature columns: {len(feat_cols_list)}")
print(f"\n=== Expanded bridge script complete ===")
print(f"Output directory: {OUTPUT_DIR}/")
for f in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, f)
    size_mb = os.path.getsize(fpath) / (1024 * 1024)
    print(f"  {f}: {size_mb:.1f} MB")

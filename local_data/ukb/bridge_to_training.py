#!/usr/bin/env python3
"""
Bridge Script: Transform UKB数据集/features/ .npz to Preprocessed_Data.csv
Focuses on field IDs referenced by the training pipeline (AD_Training, DataGeneration).
"""

import os, sys, gc, time
import numpy as np
import pandas as pd
import configparser

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'UKB数据集'))

UKB_DIR = os.path.join(PROJECT_ROOT, 'UKB数据集')
FEATURE_DIR = os.path.join(UKB_DIR, 'features')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'Preprocessed_Data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONFIG_DATA = os.path.join(UKB_DIR, 'config_data.ini')
N_OVERALL = 502241
SECONDS_PER_YEAR = 31536000.0

data_config = configparser.ConfigParser()
data_config.optionxform = lambda option: option
data_config.read(CONFIG_DATA)

# ===== FIELDS REFERENCED IN TRAINING SCRIPTS (73 field IDs + important extras) =====
REFERENCED_FIELDS = [
    # Cognitive tests
    '20023',  # Mean time to correctly identify matches
    '400',    # Time to complete round (pairs matching)
    '398',    # Number of correct matches in round
    '399',    # Number of incorrect matches in round
    '403',    # Number of times snap-button pressed
    '404',    # Duration to first press of snap-button
    # Physical measures
    '3062',   # Forced vital capacity (FVC)
    '3063',   # Forced expiratory volume in 1-second (FEV1)
    '3064',   # Peak expiratory flow (PEF)
    '46',     # Hand grip strength (left)
    '47',     # Hand grip strength (right)
    '23104',  # BMI
    '21001',  # Body mass index (BMI)
    '23107',  # Impedance of leg (right)
    '23108',  # Impedance of leg (left)
    '23110',  # Impedance of arm (left)
    '23111',  # Leg fat percentage (right)
    '23112',  # Leg fat mass (right)
    '23115',  # Leg fat percentage (left)
    '23116',  # Leg fat mass (left)
    '4080',   # Systolic blood pressure, automated reading
    # Demographics / Lifestyle
    '21022',  # Age at recruitment
    '31',     # Sex
    '21000',  # Ethnic background
    '22006',  # Genetic ethnic grouping
    '137',    # Number of treatments/medications taken
    '3526',   # Mother's age at death
    '845',    # Age completed full time education
    '2443',   # Diabetes diagnosed by doctor
    '2188',   # Long-standing illness, disability or infirmity
    '1558',   # Alcohol intake frequency
    '1031',   # Frequency of friend/family visits
    '1329',   # Oily fish intake
    '1339',   # Non-oily fish intake
    '20116',  # Smoking status
    '22032',  # IPAQ activity group
    '189',    # Townsend deprivation index at recruitment
    # Blood biomarkers
    '30710',  # C-reactive protein
    '30650',  # Aspartate aminotransferase
    '30620',  # Alanine aminotransferase
    '30690',  # Cholesterol
    '30040',  # Mean corpuscular volume
    # Self-reported illnesses
    '20002',  # Non-cancer illness code, self-reported
    '20110',  # Illnesses of mother
    '20008',  # Interpolated Year when non-cancer illness first diagnosed
    '20010',  # Interpolated Year when operation took place
    # Hospital / HES fields (excluded from training but referenced)
    '41234',  # Records in HES inpatient diagnoses dataset
    '41235',  # Spells in hospital
    '41259',  # Records in HES inpatient main dataset
    '41289',  # Records in HES inpatient psychiatric dataset
    '41149',  # Records in HES inpatient operations dataset
    '41214',  # Carer support indicators
    '41218',  # History of psychiatric care on admission
    # Other fields from S06 add_manual_features.py
    '131287', # Source of report of I10 (hypertension)
    '131351', # Source of report of I48 (atrial fibrillation)
    # Date of birth (for age calculation if needed, but use '53' specially)
    '34',     # Year of birth
    '52',     # Month of birth
]

# Fields to exclude from features (outcome fields handled separately)
EXCLUDE_FIELDS = {'42018', '42019', '42020', '42021', '42022', '42023',
                  '42024', '42025', '42006', '42007', '53', '40007', '40008'}

# HES fields (excluded in training scripts)
HES_FIELDS = ['41234-0.0', '41259-0.0', '41149-0.0', '41289-0.0',
              '41218-0.0', '41235-0.0', '41214-0.0']

# ===== Helper: extract column name from feature shape =====
def get_column_names(field_id, value_type, instances, array_n, feature):
    """Generate column names for a feature field, handling type-specific encodings."""
    M = instances * array_n
    n_cols = feature.shape[1]
    col_names = []

    if value_type in ('Continuous', 'Integer'):
        for m in range(M):
            col_idx = m * 2
            if col_idx < n_cols:
                arr = m % array_n
                col_names.append(f'{field_id}-0.{arr}')
            else:
                break

    elif value_type == 'Date':
        for m in range(M):
            col_idx = m * 2
            if col_idx < n_cols:
                col_names.append(f'{field_id}-0.{m}')
            else:
                break

    elif value_type == 'Categorical single':
        v_count = n_cols // M if M > 0 else n_cols
        for m in range(M):
            arr = m % array_n
            for c in range(v_count):
                col_names.append(f'{field_id}-0.{arr}_c{c}')

    elif value_type == 'Categorical multiple':
        # Structure: v_count * 2 * instances columns
        # Find v_count by dividing n_cols
        v_count = n_cols // (2 * max(1, instances))
        for m in range(min(M, instances)):
            for c in range(v_count):
                col_names.append(f'{field_id}-{m}.{c}_pos')
                col_names.append(f'{field_id}-{m}.{c}_neg')

    else:
        # Unknown type, just name by index
        for i in range(min(n_cols, M)):
            col_names.append(f'{field_id}-0.{i}')

    return col_names


def extract_values(field_id, value_type, instances, array_n, feature):
    """Extract feature values as flat dict of {col_name: array}."""
    M = instances * array_n
    n_cols = feature.shape[1]
    result = {}
    n_samples = feature.shape[0]

    if value_type in ('Continuous', 'Integer', 'Date'):
        # Value column every other position, missing flag follows
        for m in range(M):
            col_idx = m * 2
            if col_idx < n_cols:
                arr = m % array_n
                values = feature[:, col_idx].copy()
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
                    result[f'{field_id}-0.{arr}_c{c}'] = feature[:, idx]

    elif value_type == 'Categorical multiple':
        v_count = n_cols // (2 * max(1, instances))
        for m in range(min(M, instances)):
            for c in range(v_count):
                pos_idx = c * 2 + m * v_count * 2
                neg_idx = c * 2 + 1 + m * v_count * 2
                if pos_idx < n_cols:
                    result[f'{field_id}-{m}.{c}_pos'] = feature[:, pos_idx]
                if neg_idx < n_cols:
                    result[f'{field_id}-{m}.{c}_neg'] = feature[:, neg_idx]

    return result


# ===== LOAD TARGET VARIABLES FROM DATE FIELDS =====
print("=" * 60)
print("Step 1: Deriving target variables from date fields")
print("=" * 60)

def load_date_feature(field_id):
    """Load a date field's timestamp column, with NaN for missing."""
    fpath = os.path.join(FEATURE_DIR, f'{field_id}.npz')
    if not os.path.exists(fpath):
        return None
    data = np.load(fpath, allow_pickle=True)['feature']
    values = data[:, 0].copy()
    if data.shape[1] >= 2:
        missing = data[:, 1] == 1
        values[missing] = np.nan
    return values

# Load baseline date and event dates
date_visit   = load_date_feature('53')    # Date of attending assessment centre
date_dem     = load_date_feature('42018') # Date of all cause dementia report
date_ad      = load_date_feature('42020') # Date of Alzheimer's disease report
date_vd      = load_date_feature('42022') # Date of vascular dementia report
date_stroke  = load_date_feature('42006') # Date of stroke

# Dementia outcome
dementia_status = np.where(~np.isnan(date_dem), 1, 0).astype(np.int32)
dementia_years = ((date_dem - date_visit) / SECONDS_PER_YEAR).astype(np.float32)
dementia_years[dementia_years < 0] = dementia_years[dementia_years < 0].copy()  # keep negative values as-is

# AD outcome
ad_status = np.where(~np.isnan(date_ad), 1, 0).astype(np.int32)
ad_years = ((date_ad - date_visit) / SECONDS_PER_YEAR).astype(np.float32)

# VD outcome
vd_status = np.where(~np.isnan(date_vd), 1, 0).astype(np.int32)
vd_years = ((date_vd - date_visit) / SECONDS_PER_YEAR).astype(np.float32)

# Stroke outcome
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

print(f"  Dementia cases: {dementia_status.sum()} / {N_OVERALL}")
print(f"  AD cases:       {ad_status.sum()} / {N_OVERALL}")
print(f"  Stroke cases:   {stroke_status.sum()} / {N_OVERALL}")
print(f"  Saved: Dementia_target.csv")


# ===== BUILD FEATURE MATRIX =====
print("\n" + "=" * 60)
print("Step 2: Building feature matrix from .npz files")
print("=" * 60)

all_field_data = {}
processed = 0
skipped_not_found = 0
skipped_no_config = 0

for field_id in sorted(set(REFERENCED_FIELDS)):
    if field_id in EXCLUDE_FIELDS:
        continue

    # Check config
    if field_id not in data_config:
        print(f"  WARNING: {field_id} not in config_data.ini, skipping")
        skipped_no_config += 1
        continue

    # Check feature file
    feat_file = os.path.join(FEATURE_DIR, f'{field_id}.npz')
    if not os.path.exists(feat_file):
        print(f"  WARNING: {field_id} feature file not found, skipping")
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
        processed += 1

        if processed % 10 == 0:
            print(f"  Processed {processed} fields... (current columns: {len(all_field_data)})")

    except Exception as e:
        print(f"  ERROR processing field {field_id}: {e}")

print(f"\nField processing summary:")
print(f"  Successfully processed: {processed}")
print(f"  Skipped (not found):   {skipped_not_found}")
print(f"  Skipped (no config):   {skipped_no_config}")
print(f"  Total feature columns: {len(all_field_data)}")

# ===== BUILD DATAFRAME =====
print("\n" + "=" * 60)
print("Step 3: Creating DataFrame and saving")
print("=" * 60)

print("Building DataFrame from columns dict...")
import time as time_module
t0 = time_module.time()
X_df = pd.DataFrame(all_field_data)
print(f"  DataFrame created in {time_module.time()-t0:.1f}s")
print(f"  Feature matrix shape: {X_df.shape}")

# Add target variables
print("Adding target variables...")
result_df = pd.concat([X_df, target_df], axis=1)
print(f"  Full DataFrame shape: {result_df.shape}")

# Save
csv_path = os.path.join(OUTPUT_DIR, 'Preprocessed_Data.csv')
print(f"\nSaving {csv_path}...")
t0 = time_module.time()
result_df.to_csv(csv_path, index=False)
print(f"  Saved in {time_module.time()-t0:.1f}s")
print(f"  File size: {os.path.getsize(csv_path)/1024/1024:.1f} MB")

# Also save feature list
feat_list = [c for c in X_df.columns]
feat_df = pd.DataFrame({'Features': feat_list})
feat_df.to_csv(os.path.join(OUTPUT_DIR, 'raw_features.csv'), index=False)

# Memory info
import psutil
process = psutil.Process(os.getpid())
mem_gb = process.memory_info().rss / 1024 / 1024 / 1024
print(f"\nMemory usage: {mem_gb:.1f} GB")

print("\n=== Bridge script complete ===")
print(f"Output: {OUTPUT_DIR}/")
for f in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, f)
    print(f"  {f}: {os.path.getsize(fpath)/1024/1024:.1f} MB")

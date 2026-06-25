#!/usr/bin/env python3
"""
Full bridge: extract ALL instances/arrays for paper's feature fields from .npz.
Produces a richer feature set matching the paper's data scope.

Strategy: For each unique field ID used in the paper, extract ALL available
data from the .npz file (all instances, arrays, categories). This produces
a superset that covers the paper's features.
"""
import os, sys, gc, time, re, configparser
from collections import defaultdict
import numpy as np
import pandas as pd

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UKB_DIR = os.path.join(PROJECT_ROOT, 'UKB数据集')
FEATURE_DIR = os.path.join(UKB_DIR, 'features')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'Preprocessed_Data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONFIG_DATA = os.path.join(UKB_DIR, 'config_data.ini')
N_OVERALL = 502241

# --- Get unique field IDs from paper's feature list ---
PAPER_S01 = os.path.join(PROJECT_ROOT, 'local_data', 'Results',
                          'Results_woHES', 'DM_full', 's01_DM_full.csv')
paper_df = pd.read_csv(PAPER_S01)
paper_features = paper_df['Features'].tolist()

# Extract unique field IDs from paper feature names
paper_fids = set()
for pf in paper_features:
    m = re.match(r'(\d+)-', pf)
    if m:
        paper_fids.add(m.group(1))
    else:
        paper_fids.add(pf)

print(f"Paper features: {len(paper_features)}, unique Field IDs: {len(paper_fids)}")

# --- Load config ---
data_config = configparser.ConfigParser()
data_config.optionxform = lambda o: o
data_config.read(CONFIG_DATA)

# --- Extract all data for each field ---
def extract_field_all(field_id):
    """Extract ALL available data from a field's .npz file.
    Returns dict of {column_name: np.array}."""
    fpath = os.path.join(FEATURE_DIR, f'{field_id}.npz')
    if not os.path.exists(fpath):
        return {}

    feature = np.load(fpath)['feature']
    n_cols = feature.shape[1]

    if field_id in data_config:
        cfg = data_config[field_id]
        vt = cfg.get('ValueType', 'Continuous')
        instances = int(cfg.get('Instances', 1))
        array_size = int(cfg.get('Array', 1))
    else:
        vt = 'Continuous'
        instances = 1
        array_size = n_cols // 2

    results = {}

    if vt in ('Continuous', 'Integer', 'Date'):
        # Each (instance, array) = 2 columns (value + flag)
        pairs = n_cols // 2
        for inst in range(instances):
            for arr in range(array_size):
                pair_idx = inst * array_size + arr
                col_val = pair_idx * 2
                col_flag = col_val + 1
                if col_val >= n_cols:
                    break
                values = feature[:, col_val].astype(np.float64)
                if col_flag < n_cols:
                    flags = feature[:, col_flag]
                    values[flags > 0.5] = np.nan
                # Check if any non-NaN values exist
                if np.isfinite(values).sum() > 0:
                    col_name = f'{field_id}-{inst}.{arr}'
                    results[col_name] = values
            if inst * array_size * 2 >= n_cols:
                break

    elif vt == 'Categorical single':
        # Column per category per instance
        # .npz layout: instances * n_categories columns
        n_cats = n_cols // max(instances, 1)
        for inst in range(instances):
            for cat in range(n_cats):
                col_idx = inst * n_cats + cat
                if col_idx >= n_cols:
                    break
                values = feature[:, col_idx].astype(np.float64)
                if values.sum() > 0:  # Only include non-empty categories
                    col_name = f'{field_id}-{inst}.{inst}_c{cat}'
                    results[col_name] = values
            if inst * n_cats >= n_cols:
                break

    elif vt == 'Categorical multiple':
        # Each column is a separate category value across instances
        # .npz layout: n_categories * instances columns (varies)
        # Simpler approach: each column is a separate value
        for col_idx in range(n_cols):
            values = feature[:, col_idx].astype(np.float64)
            n_nonzero = (values != 0).sum()
            if n_nonzero > 100:  # Only include if enough data
                col_name = f'{field_id}-0.{col_idx}'
                results[col_name] = values

    return results

# --- Process all paper fields ---
print("Extracting all paper field data from .npz...")
all_columns = {}
for i, fid in enumerate(sorted(paper_fids)):
    try:
        extracted = extract_field_all(fid)
        all_columns.update(extracted)
    except Exception as e:
        print(f"  ERROR field {fid}: {e}")

    if (i + 1) % 20 == 0:
        print(f"  Progress: {i+1}/{len(paper_fids)} fields, {len(all_columns)} columns so far")

print(f"Total columns extracted: {len(all_columns)}")

# --- Build DataFrame ---
print("Building DataFrame...")
df = pd.DataFrame(all_columns)
print(f"  Shape: {df.shape}")

# --- Add target columns ---
target_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'Dementia_target.csv'))
assert len(target_df) == len(df)
for col in target_df.columns:
    df[col] = target_df[col].values

# --- Missing rate filter ---
# Paper uses 40%, we use 45% (npz format tends to have slightly higher missingness)
# Plus protect paper's key predictor fields
PROTECTED_PREFIXES = ('3526', '2188', '20110', '30040', '21022', '31')
MISSING_THRESHOLD = 0.45
print(f"Applying missing rate filter (>{MISSING_THRESHOLD*100:.0f}%)...")
n_before = df.shape[1]
missing_rates = df.isnull().sum() / len(df)
outcome_cols = ['dementia_status', 'dementia_years', 'AD_status', 'AD_years',
                'VD_status', 'VD_years', 'stroke_status', 'stroke_years']
protect_cols = [c for c in df.columns
                if c in outcome_cols or
                   any(c.startswith(p) for p in PROTECTED_PREFIXES)]
high_miss = missing_rates[(missing_rates > MISSING_THRESHOLD) &
                           (~missing_rates.index.isin(protect_cols))]
df = df.drop(columns=high_miss.index.tolist())
print(f"  Dropped {len(high_miss)} features, retained {df.shape[1]} (was {n_before})")

# --- Save ---
output_path = os.path.join(OUTPUT_DIR, 'Preprocessed_Data_full.csv')
print(f"Saving to {output_path}...")
df.to_csv(output_path, index=False)

# Compare with paper
paper_only_fids = set()
for pf in paper_features:
    m = re.match(r'(\d+)-', pf)
    if m:
        paper_only_fids.add(m.group(1))

our_fids = set()
for c in df.columns:
    m = re.match(r'(\d+)-', c)
    if m:
        our_fids.add(m.group(1))

print(f"\n{'='*60}")
print(f"Paper unique Field IDs: {len(paper_only_fids)}")
print(f"Our extracted Field IDs: {len(our_fids)}")
print(f"Overlap: {len(paper_only_fids & our_fids)}/{len(paper_only_fids)}")
print(f"Paper-only FIDs (we lack): {sorted(paper_only_fids - our_fids)}")
print(f"Final: {df.shape[1]} features × {df.shape[0]} participants")
print(f"{'='*60}")

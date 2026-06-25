#!/usr/bin/env python3
"""
Efficient Bridge Script v3: Baseline-only features from Type 1 + blood biomarkers
Limits to instance 0 (baseline assessment only, matching paper's approach).
Target: ~400-700 feature columns (comparable to paper's 366).
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
DATA_DIR = os.path.join(PROJECT_ROOT, 'local_data', 'Data')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

CONFIG_DATA = os.path.join(UKB_DIR, 'config_data.ini')
DATA_LIST_CSV = os.path.join(UKB_DIR, 'data_list.csv')
N_OVERALL = 502241
SECONDS_PER_YEAR = 31536000.0

data_config = configparser.ConfigParser()
data_config.optionxform = lambda option: option
data_config.read(CONFIG_DATA)
data_list = pd.read_csv(DATA_LIST_CSV, low_memory=False)

EXCLUDE_FIDS = {
    '42018', '42019', '42020', '42021', '42022', '42023',
    '42024', '42025', '42006', '42007', '53', '40007', '40008',
    '42016', '42017',
}

# Important fields to always include (paper's key predictors + demographics + cognition)
ALWAYS_INCLUDE = {
    '21022', '31', '21000', '22006', '137', '3526', '845', '2443', '2188',
    '1558', '1031', '1329', '1339', '20116', '22032', '189', '34', '52',
    '20023', '400', '398', '399', '403', '404',
    '3062', '3063', '3064', '46', '47', '23104', '21001',
    '23107', '23108', '23110', '23111', '23112', '23115', '23116', '4080',
    '30710', '30650', '30620', '30690', '30040',
    '20002', '20110', '20008', '20010',
    '131287', '131351',
    # Additional cognitive tests
    '4282', '4286', '6350', '6348', '6770', '6771', '20132', '20191',
    # Additional physical measures
    '21002', '48', '49', '50', '93', '94', '95', '23099', '23100',
    '23101', '23102', '23105', '23106', '23113', '23114', '23117', '23118',
    # Additional blood biomarkers
    '30700', '30720', '30730', '30740', '30750', '30760', '30770', '30780',
    '30790', '30800', '30810', '30830', '30840', '30850', '30860', '30870',
    '30880', '30890', '30600', '30610', '30630', '30640', '30660', '30670',
    # Additional health/lifestyle
    '21003', '6160', '6150', '6145', '6141', '6138', '20117', '20118',
    '20119', '20160', '864', '874', '884', '894', '904',
    # Family history
    '20107', '20111',
}

# Fields to protect from >40% missing filter (paper's key predictors)
PROTECTED_PREFIXES = ('3526', '2188', '20110', '30040')

# ===== SELECT FIELDS =====
# Type 1 (primary) - all clinically relevant fields from baseline
# Type 2 (derived) - blood biomarkers only
selected_fields = set(ALWAYS_INCLUDE)

for _, row in data_list.iterrows():
    fid = str(int(row['FieldID'])) if not np.isnan(row.get('FieldID')) else None
    if not fid or fid in EXCLUDE_FIDS:
        continue
    if not os.path.exists(os.path.join(FEATURE_DIR, f'{fid}.npz')):
        continue
    if fid not in data_config:
        continue

    ftype = row.get('Type', 0)
    path_str = str(row.get('Path', '')).lower()
    vt = data_config[fid].get('ValueType', 'N/A')

    # Skip HES-like fields
    if fid.startswith('41') or fid.startswith('42'):
        continue

    # Type 1: All fields with useful value types
    if ftype == 1.0 and vt in ('Continuous', 'Integer', 'Categorical single', 'Categorical multiple'):
        # Exclude local environment / noise / greenspace (not clinically relevant for dementia)
        if any(kw in path_str for kw in ['local environment', 'greenspace', 'coastal', 'noise', 'air pollution', 'home location']):
            continue
        selected_fields.add(fid)

    # Type 2: Blood biomarkers only
    elif ftype == 2.0 and vt in ('Continuous', 'Integer'):
        if any(kw in path_str for kw in ['blood', 'assay', 'biochem', 'count', 'lipid', 'glucose', 'protein', 'metabol', 'hormone', 'igf', 'c-reactive', 'vitamin', 'cystatin', 'creatinine', 'albumin', 'urate', 'phosphate', 'calcium', 'bilirubin', 'ast', 'alt', 'ggt']):
            selected_fields.add(fid)

all_field_ids = sorted(selected_fields)
print(f"Selected {len(all_field_ids)} fields:")
print(f"  Always include: {len(ALWAYS_INCLUDE)}")
print(f"  Additional: {len(all_field_ids) - len(ALWAYS_INCLUDE)}")
print(f"  Total: {len(all_field_ids)}")


# ===== EXTRACTION (BASELINE ONLY: instance 0) =====
def extract_baseline(field_id, value_type, feature):
    """Extract baseline (instance 0, array 0) values only."""
    n_cols = feature.shape[1]
    result = {}

    if value_type in ('Continuous', 'Integer', 'Date'):
        if n_cols >= 2:
            values = feature[:, 0].copy().astype(np.float64)
            missing_mask = feature[:, 1] == 1
            values[missing_mask] = np.nan
            result[f'{field_id}-0.0'] = values

    elif value_type == 'Categorical single':
        # Count categories from all instances (n_cols / instances = v_count)
        instances = n_cols // max(1, n_cols // max(1, (n_cols // 2)))
        # Simpler: first row analysis - but we can't easily peek at first row
        # Instead, find v_count from a sample
        sample_vals = feature[0, :]
        non_zero = np.count_nonzero(sample_vals)
        if non_zero == 0:
            non_zero = 1
        v_count = n_cols
        # Heuristic: each instance has v_count columns, and n_cols = instances * v_count
        # For baseline only, take first v_count columns
        # Find v_count by looking for where the pattern ends
        for vc in [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30]:
            if n_cols % vc == 0:
                v_count = vc
                break
        for c in range(v_count):
            if c < n_cols:
                result[f'{field_id}-0.0_c{c}'] = feature[:, c].astype(np.float64)

    elif value_type == 'Categorical multiple':
        # Same heuristic - just take first 2*v_count columns for instance 0
        vc_guess = n_cols // 2
        for vc in [2, 3, 4, 5, 6, 8, 10, 12, 15]:
            if n_cols % (vc * 2) == 0:
                vc_guess = vc
                break
        for c in range(vc_guess):
            pos_idx = c * 2
            neg_idx = c * 2 + 1
            if pos_idx < n_cols:
                result[f'{field_id}-0.{c}_pos'] = feature[:, pos_idx].astype(np.float64)
            if neg_idx < n_cols:
                result[f'{field_id}-0.{c}_neg'] = feature[:, neg_idx].astype(np.float64)

    return result


# ===== TARGET VARIABLES =====
print("\n" + "="*60)
print("Step 1: Target variables")
print("="*60)

def load_date_feature(field_id):
    fpath = os.path.join(FEATURE_DIR, f'{field_id}.npz')
    if not os.path.exists(fpath):
        return None
    data = np.load(fpath, allow_pickle=True)['feature']
    values = data[:, 0].copy()
    if data.shape[1] >= 2:
        values[data[:, 1] == 1] = np.nan
    return values

dates = {
    'visit': load_date_feature('53'),
    'dementia': load_date_feature('42018'),
    'ad': load_date_feature('42020'),
    'vd': load_date_feature('42022'),
    'stroke': load_date_feature('42006'),
}

target_df = pd.DataFrame({
    'dementia_status': np.where(~np.isnan(dates['dementia']), 1, 0).astype(np.int32),
    'dementia_years': ((dates['dementia'] - dates['visit']) / SECONDS_PER_YEAR).astype(np.float32),
    'AD_status': np.where(~np.isnan(dates['ad']), 1, 0).astype(np.int32),
    'AD_years': ((dates['ad'] - dates['visit']) / SECONDS_PER_YEAR).astype(np.float32),
    'VD_status': np.where(~np.isnan(dates['vd']), 1, 0).astype(np.int32),
    'VD_years': ((dates['vd'] - dates['visit']) / SECONDS_PER_YEAR).astype(np.float32),
    'stroke_status': np.where(~np.isnan(dates['stroke']), 1, 0).astype(np.int32),
    'stroke_years': ((dates['stroke'] - dates['visit']) / SECONDS_PER_YEAR).astype(np.float32),
})
target_df.to_csv(os.path.join(OUTPUT_DIR, 'Dementia_target.csv'), index=False)
print(f"  Dementia: {target_df['dementia_status'].sum()}, AD: {target_df['AD_status'].sum()}, Stroke: {target_df['stroke_status'].sum()}")

# ===== BUILD FEATURE MATRIX =====
print("\n" + "="*60)
print(f"Step 2: Processing {len(all_field_ids)} fields (baseline only)")
print("="*60)

# Process in chunks
CHUNK_SIZE = 200
all_chunks = []
processed = 0
skipped = 0
total_cols = 0

for i in range(0, len(all_field_ids), CHUNK_SIZE):
    chunk_ids = all_field_ids[i:i + CHUNK_SIZE]
    chunk_data = {}

    for field_id in chunk_ids:
        if field_id not in data_config:
            skipped += 1
            continue
        feat_file = os.path.join(FEATURE_DIR, f'{field_id}.npz')
        if not os.path.exists(feat_file):
            skipped += 1
            continue
        try:
            cfg = data_config[field_id]
            vt = cfg.get('ValueType', 'N/A')
            feature = np.load(feat_file, allow_pickle=True)['feature']
            extracted = extract_baseline(field_id, vt, feature)
            chunk_data.update(extracted)
            processed += 1
        except Exception as e:
            skipped += 1

    if chunk_data:
        df_chunk = pd.DataFrame(chunk_data)
        total_cols += df_chunk.shape[1]
        all_chunks.append(df_chunk)

    # Intermediate filtering
    if len(all_chunks) >= 4:
        df_temp = pd.concat(all_chunks, axis=1)
        high_miss = df_temp.columns[(df_temp.isnull().mean() > 0.45) &
                                      ~df_temp.columns.str.startswith(tuple(PROTECTED_PREFIXES))]
        for c in all_chunks:
            c.drop(columns=[col for col in high_miss if col in c.columns], inplace=True, errors='ignore')
        gc.collect()

    print(f"  Chunk {i//CHUNK_SIZE + 1}: {len(chunk_data)} cols from {len(chunk_ids)} fields, "
          f"total: {total_cols} cols, mem: {sum(c.memory_usage(deep=True).sum() for c in all_chunks)/1024**2:.0f} MB")

# ===== FINAL ASSEMBLY =====
print("\n" + "="*60)
print("Step 3: Final assembly & filtering")
print("="*60)

t0 = time.time()
X_df = pd.concat(all_chunks, axis=1)
del all_chunks
gc.collect()
print(f"  Shape: {X_df.shape} ({time.time()-t0:.1f}s)")

# Remove >40% missing
missing_rates = X_df.isnull().mean()
high_missing = missing_rates[(missing_rates > 0.45) &
                              ~missing_rates.index.str.startswith(PROTECTED_PREFIXES)]
print(f"  Removing {len(high_missing)} cols with >45% missing (excluding protected)")
X_df.drop(columns=high_missing.index, inplace=True)

# Remove constant columns
constant_cols = [c for c in X_df.columns if X_df[c].nunique() <= 1]
if constant_cols:
    print(f"  Removing {len(constant_cols)} constant columns")
    X_df.drop(columns=constant_cols, inplace=True)

# Remove duplicate columns
t0 = time.time()
X_df_t = X_df.T.drop_duplicates().T
dups = X_df.shape[1] - X_df_t.shape[1]
if dups > 0:
    print(f"  Removing {dups} duplicate columns")
    X_df = X_df_t
print(f"  After filtering: {X_df.shape}")

# Add targets
result_df = pd.concat([X_df, target_df], axis=1)
print(f"  Final: {result_df.shape}")

# ===== SAVE =====
print("\n" + "="*60)
print("Step 4: Saving")
print("="*60)

csv_path = os.path.join(OUTPUT_DIR, 'Preprocessed_Data.csv')
t0 = time.time()
result_df.to_csv(csv_path, index=False)
file_size_mb = os.path.getsize(csv_path) / (1024 * 1024)
print(f"  Saved: {file_size_mb:.1f} MB in {time.time()-t0:.0f}s")

# Feature list
feat_cols = [c for c in X_df.columns]
pd.DataFrame({'Features': feat_cols}).to_csv(os.path.join(OUTPUT_DIR, 'raw_features.csv'), index=False)

# FieldID_selected.csv for training compatibility
fid_df = pd.DataFrame({'Features': feat_cols})
fid_df['Path'] = ''
fid_df['Field'] = fid_df['Features']
fid_df['ValueType'] = ''
fid_df['Units'] = ''
fid_df.to_csv(os.path.join(DATA_DIR, 'FieldID_selected.csv'), index=False)

unique_fids = len(set(c.split('-')[0] for c in feat_cols))
print(f"\n  Unique field IDs: {unique_fids}")
print(f"  Feature columns: {len(feat_cols)}")
print(f"\nDone!")

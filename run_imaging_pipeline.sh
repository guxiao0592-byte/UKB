#!/bin/bash
# ============================================================================
# UKB-DRP: Brain MRI Imaging Pipeline
# ============================================================================
# Usage:
#   chmod +x run_imaging_pipeline.sh
#   bash run_imaging_pipeline.sh                        # full cohort, DM_full only
#   bash run_imaging_pipeline.sh --subset               # imaging subset only
#   bash run_imaging_pipeline.sh --all-targets          # all 6 targets, full cohort
#   bash run_imaging_pipeline.sh --subset --all-targets # all 6 targets, subset
#   nohup bash run_imaging_pipeline.sh --all-targets > logs/pipeline.log 2>&1 &
# ============================================================================
set -euo pipefail

# Auto-activate conda if available
if command -v conda &> /dev/null && [ -n "${CONDA_PREFIX:-}" ]; then
    true  # already in a conda env
elif command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
fi

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

MODE="full"
TARGETS="single"
N_COMBOS=1000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --subset)    MODE="subset"; shift ;;
        --all-targets) TARGETS="all"; shift ;;
        --fast)      N_COMBOS=200; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================"
echo "UKB-DRP Imaging Pipeline"
echo "  Mode:      $MODE (subset=imaging participants only)"
echo "  Targets:   $TARGETS (all=6 targets with deploy)"
echo "  N combos:  $N_COMBOS"
echo "  Started:   $(date)"
echo "  Host:      $(hostname)"
echo "============================================"

# ---- Step 1: Clinical bridge (if needed) ----
if [ ! -f "local_data/Preprocessed_Data/Preprocessed_Data.csv" ]; then
    echo ""
    echo "[$(date)] Step 1: Running clinical bridge..."
    python3 UKB数据集/bridge_to_training_v3.py
else
    echo "[$(date)] Step 1: Clinical data exists, skip."
    ls -lh local_data/Preprocessed_Data/Preprocessed_Data.csv
fi

# ---- Step 2: Imaging bridge ----
if [ ! -f "local_data/Preprocessed_Data/Preprocessed_Data_imaging.csv" ]; then
    echo ""
    echo "[$(date)] Step 2: Extracting brain MRI features + merging..."
    python3 UKB数据集/bridge_imaging.py
else
    echo "[$(date)] Step 2: Imaging data exists, skip."
    ls -lh local_data/Preprocessed_Data/Preprocessed_Data_imaging.csv
fi

# ---- Step 3: Training ----
echo ""
echo "[$(date)] Step 3: Training..."

IMG_FLAG=""
if [ "$MODE" = "subset" ]; then
    IMG_FLAG="--imaging-subset"
    echo "  -> Imaging subset mode (~46K participants)"
else
    echo "  -> Full cohort mode (~425K participants, LightGBM handles NaN)"
fi

if [ "$TARGETS" = "all" ]; then
    echo "  -> All 6 targets (deploy strategy)"
    python3 src/training/run_training_imaging.py \
        $IMG_FLAG \
        --n-combos $N_COMBOS
else
    echo "  -> DM_full only"
    python3 src/training/run_training_imaging.py \
        --target DM_full \
        $IMG_FLAG \
        --n-combos $N_COMBOS
fi

echo ""
echo "============================================"
echo "[$(date)] Done! Results: local_data/Results_imaging/"
echo "============================================"

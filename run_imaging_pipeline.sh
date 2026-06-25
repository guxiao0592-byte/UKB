#!/bin/bash
# ============================================================================
# UKB-DRP — MRI 影像管线
# 用法: bash run_imaging_pipeline.sh [--all-targets] [--subset] [--fast]
# 新位置: src/scripts/server/02_ukb_imaging.sh
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
bash "$SCRIPT_DIR/src/scripts/server/02_ukb_imaging.sh" "$@"

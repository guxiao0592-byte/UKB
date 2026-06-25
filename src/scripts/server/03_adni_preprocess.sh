#!/bin/bash
# ============================================================================
# 03_adni_preprocess.sh — ADNI 数据预处理
# ============================================================================
# 从 66 个 ADNI CSV 构建基线数据集, 并生成时间目标和 CDR 目标。
#
# 数据依赖:
#   - local_data/adni/extracted/All_Subjects_*.csv  (66 个原始 CSV)
#   - local_data/adni/preprocess_adni.py
#   - local_data/adni/build_time_targets_v2.py
#   - local_data/adni/build_cdr_targets_v2.py
#
# 输出:
#   - local_data/adni/processed/ADNI_baseline.csv
#   - local_data/adni/processed/ADNI_baseline_with_time_targets_v2.csv
#   - local_data/adni/processed/CDR_time_targets_v2.csv
#
# 用法:
#   bash src/scripts/server/03_adni_preprocess.sh
#   bash src/scripts/server/03_adni_preprocess.sh --skip-baseline
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/env_setup.sh"
cd "$PROJECT_ROOT"

SKIP_BASELINE=false
for arg in "$@"; do
    case "$arg" in
        --skip-baseline) SKIP_BASELINE=true ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/03_adni_preprocess_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo ""
echo "################################################################"
echo "# ADNI 数据预处理"
echo "# ADNI 数据目录: $ADNI_DATA_DIR"
echo "# 开始: $(date)"
echo "################################################################"

# 验证 ADNI 目录结构
EXTRACTED_DIR="$ADNI_DATA_DIR/extracted"
PROCESSED_DIR="$ADNI_DATA_DIR/processed"
mkdir -p "$PROCESSED_DIR"

if [ ! -d "$EXTRACTED_DIR" ]; then
    echo "  ERROR: ADNI extracted/ 不存在: $EXTRACTED_DIR"
    echo "  请先运行 migrate_data.sh 迁移 ADNI 数据到 local_data/adni/"
    echo "  或将 66 个 All_Subjects_*_20May2026.csv 放入此目录"
    exit 1
fi

N_CSV=$(ls "$EXTRACTED_DIR"/All_Subjects_*.csv 2>/dev/null | wc -l | tr -d ' ')
echo "  找到 $N_CSV 个 ADNI CSV 文件"
if [ "$N_CSV" -lt 30 ]; then
    echo "  [WARN] CSV 数较少, 预期 ~66 个"
fi

# ══════════════════════════════════════════════════════════════
# Step 1: 基线数据构建
# ══════════════════════════════════════════════════════════════
PREPROCESS_SCRIPT="$ADNI_DATA_DIR/preprocess_adni.py"
BASELINE_CSV="$PROCESSED_DIR/ADNI_baseline.csv"

if $SKIP_BASELINE && [ -f "$BASELINE_CSV" ]; then
    echo ""
    echo "[Step 1] SKIP — ADNI_baseline.csv 已存在"
else
    echo ""
    echo "[Step 1] 构建基线数据集..."

    if [ ! -f "$PREPROCESS_SCRIPT" ]; then
        echo "  ERROR: preprocess_adni.py 不存在: $PREPROCESS_SCRIPT"
        echo "  请确保已运行 migrate_data.sh"
        exit 1
    fi

    echo "  运行: $PREPROCESS_SCRIPT"
    # preprocess_adni.py 使用 os.path.dirname(__file__) 定位 extracted/processed/
    # 放在 local_data/adni/ 后会自动找到同级的 extracted/ 和 processed/
    (cd "$ADNI_DATA_DIR" && python3 preprocess_adni.py)

    echo "  ✅ 基线数据集构建完成"
    ls -lh "$BASELINE_CSV" 2>/dev/null || echo "  [WARN] 输出文件未生成"
fi

# ══════════════════════════════════════════════════════════════
# Step 2: 时间目标构建
# ══════════════════════════════════════════════════════════════
TIME_TARGETS_SCRIPT="$ADNI_DATA_DIR/build_time_targets_v2.py"
TIME_TARGETS_CSV="$PROCESSED_DIR/ADNI_baseline_with_time_targets_v2.csv"

echo ""
echo "[Step 2] 构建时间目标 (DXSUM 纵向追踪)..."

if [ ! -f "$TIME_TARGETS_SCRIPT" ]; then
    echo "  ERROR: build_time_targets_v2.py 不存在: $TIME_TARGETS_SCRIPT"
    exit 1
fi

echo "  运行: $TIME_TARGETS_SCRIPT"
(cd "$ADNI_DATA_DIR" && python3 build_time_targets_v2.py)
echo "  ✅ 时间目标构建完成"
ls -lh "$TIME_TARGETS_CSV" 2>/dev/null || echo "  [WARN] 输出文件未生成"

# ══════════════════════════════════════════════════════════════
# Step 3: CDR 目标构建
# ══════════════════════════════════════════════════════════════
CDR_TARGETS_SCRIPT="$ADNI_DATA_DIR/build_cdr_targets_v2.py"
CDR_TARGETS_CSV="$PROCESSED_DIR/CDR_time_targets_v2.csv"

echo ""
echo "[Step 3] 构建 CDR 目标 (CDR 量表恶化追踪)..."

if [ ! -f "$CDR_TARGETS_SCRIPT" ]; then
    echo "  [WARN] build_cdr_targets_v2.py 不存在, 跳过 CDR 目标"
    echo "  Phase 2C 将无法运行"
else
    echo "  运行: $CDR_TARGETS_SCRIPT"
    (cd "$ADNI_DATA_DIR" && python3 build_cdr_targets_v2.py)
    echo "  ✅ CDR 目标构建完成"
    ls -lh "$CDR_TARGETS_CSV" 2>/dev/null || echo "  [WARN] 输出文件未生成"
fi

echo ""
echo "################################################################"
echo "# ✅ ADNI 预处理完成: $(date)"
echo "#"
echo "# 输出文件:"
ls -lh "$PROCESSED_DIR"/*.csv 2>/dev/null | awk '{print "#   " $NF " (" $5 ")"}'
echo "#"
echo "# 日志: $LOG_FILE"
echo "################################################################"

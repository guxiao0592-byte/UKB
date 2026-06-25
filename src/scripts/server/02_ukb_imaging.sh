#!/bin/bash
# ============================================================================
# 02_ukb_imaging.sh — UKB + 脑 MRI 影像实验
# ============================================================================
# Deploy 策略: DM_full 选特征 → 其余 5 目标独立训练+校准
#
# 数据依赖:
#   - local_data/ukb/numpy_data/*.npz
#   - local_data/ukb/bridge_imaging.py
#   - local_data/Preprocessed_Data/Preprocessed_Data.csv
#
# 输出: local_data/Preprocessed_Data/Preprocessed_Data_imaging.csv
#       local_data/Results_imaging/
#
# 用法:
#   bash src/scripts/server/02_ukb_imaging.sh
#   bash src/scripts/server/02_ukb_imaging.sh --all-targets
#   bash src/scripts/server/02_ukb_imaging.sh --subset
#   bash src/scripts/server/02_ukb_imaging.sh --fast
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/env_setup.sh"
cd "$PROJECT_ROOT"

MODE="full"
TARGETS="single"
N_COMBOS=1000
SKIP_BRIDGE=false

for arg in "$@"; do
    case "$arg" in
        --subset)    MODE="subset" ;;
        --all-targets) TARGETS="all" ;;
        --fast)      N_COMBOS=200 ;;
        --skip-bridge) SKIP_BRIDGE=true ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/02_ukb_imaging_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo ""
echo "################################################################"
echo "# 实验 2: UKB + 脑 MRI 影像"
echo "# 模式: $MODE  目标: $TARGETS  参数组合: $N_COMBOS"
echo "# 开始: $(date)"
echo "################################################################"

# ══════════════════════════════════════════════════════════════
# Step 1: 影像数据桥接
# ══════════════════════════════════════════════════════════════
if $SKIP_BRIDGE; then
    echo ""
    echo "[Step 1] SKIP — 影像桥接已跳过"
else
    echo ""
    echo "[Step 1] 影像数据桥接..."

    BRIDGE_SCRIPT="$UKB_DATA_DIR/bridge_imaging.py"
    CLINICAL_CSV="$RESULTS_DIR/Preprocessed_Data/Preprocessed_Data.csv"
    IMAGING_CSV="$RESULTS_DIR/Preprocessed_Data/Preprocessed_Data_imaging.csv"

    if [ ! -f "$CLINICAL_CSV" ]; then
        echo "  ERROR: 临床数据未找到: $CLINICAL_CSV"
        echo "  请先运行 01_ukb_baseline.sh"
        exit 1
    fi

    if [ ! -f "$BRIDGE_SCRIPT" ]; then
        echo "  ERROR: 桥接脚本不存在: $BRIDGE_SCRIPT"
        exit 1
    fi

    if [ -f "$IMAGING_CSV" ]; then
        echo "  $IMAGING_CSV 已存在, 跳过"
    else
        echo "  运行影像桥接..."
        export PROJECT_ROOT
        python3 "$BRIDGE_SCRIPT"
    fi
    echo "  ✅ 影像桥接完成"
    ls -lh "$IMAGING_CSV" 2>/dev/null || echo "  [WARN] 影像数据未生成"
fi

# ══════════════════════════════════════════════════════════════
# Step 2: UKB + MRI 训练
# ══════════════════════════════════════════════════════════════
echo ""
echo "[Step 2] UKB + MRI 训练..."

IMG_FLAG=""
if [ "$MODE" = "subset" ]; then
    IMG_FLAG="--imaging-subset"
    echo "  → 影像子集模式 (~46K 参与者)"
else
    echo "  → 全队列模式 (~425K, LightGBM 原生处理 NaN)"
fi

if [ "$TARGETS" = "all" ]; then
    echo "  → 全部 6 目标 (Deploy 策略)"
    python3 src/training/run_training_imaging.py $IMG_FLAG --n-combos $N_COMBOS
else
    echo "  → DM_full only"
    python3 src/training/run_training_imaging.py --target DM_full $IMG_FLAG --n-combos $N_COMBOS
fi
echo "  ✅ 完成 → $RESULTS_DIR/Results_imaging/"

# ══════════════════════════════════════════════════════════════
# Step 3: 外部验证 (影像 vs 非影像)
# ══════════════════════════════════════════════════════════════
echo ""
echo "[Step 3] 外部验证 (影像 vs 非影像 80/20 对比)..."
python3 src/evaluation/external_validation_imaging.py
echo "  ✅ 完成 → $RESULTS_DIR/Results_imaging/_external_validation/"

echo ""
echo "################################################################"
echo "# ✅ 实验 2 完成: $(date)"
echo "# 日志: $LOG_FILE"
echo "################################################################"

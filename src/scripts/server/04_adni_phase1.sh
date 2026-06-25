#!/bin/bash
# ============================================================================
# 04_adni_phase1.sh — ADNI Phase 1 外部验证实验
# ============================================================================
# 四个子实验:
#   3a: 横断面诊断预测 (AD vs CN, Dementia vs CN, MCI vs CN, AD vs All)
#   3b: 时间窗口预测 (CN→AD, CN→Dementia)
#   3c: MCI→AD 转化预测 (核心任务)
#   3d: 特征消融 (Bio only vs Bio+Imaging)
#
# 数据依赖:
#   - local_data/adni/processed/ADNI_baseline_with_time_targets_v2.csv
#   - local_data/adni/run_adni_pipeline.py 等脚本
#
# 输出: local_data/adni/results/  或  $RESULTS_DIR/Results_adni/
#
# 用法:
#   bash src/scripts/server/04_adni_phase1.sh
#   bash src/scripts/server/04_adni_phase1.sh --skip-cross-sectional
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/env_setup.sh"
cd "$PROJECT_ROOT"

SKIP_CROSS=false; SKIP_TIME=false; SKIP_MCI=false
for arg in "$@"; do
    case "$arg" in
        --skip-cross-sectional) SKIP_CROSS=true ;;
        --skip-time-window) SKIP_TIME=true ;;
        --skip-mci) SKIP_MCI=true ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/04_adni_phase1_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo ""
echo "################################################################"
echo "# 实验 3: ADNI Phase 1 外部验证"
echo "# ADNI 脚本目录: $ADNI_DATA_DIR"
echo "# 开始: $(date)"
echo "################################################################"

# 验证数据
DATA_CSV="$ADNI_DATA_DIR/processed/ADNI_baseline_with_time_targets_v2.csv"
if [ ! -f "$DATA_CSV" ]; then
    DATA_CSV="$ADNI_DATA_DIR/processed/ADNI_baseline_with_time_targets.csv"
fi
if [ ! -f "$DATA_CSV" ]; then
    echo "  ERROR: 数据文件未找到"
    echo "  请先运行 03_adni_preprocess.sh"
    exit 1
fi
echo "  数据文件: $DATA_CSV"

PHASE1_RESULTS="$ADNI_DATA_DIR/results"
mkdir -p "$PHASE1_RESULTS"

# ══════════════════════════════════════════════════════════════
# 实验 3a: 横断面诊断预测
# ══════════════════════════════════════════════════════════════
if $SKIP_CROSS; then
    echo ""
    echo "[3a] SKIP — 横断面诊断预测"
else
    echo ""
    echo "[3a] 横断面诊断预测 (AD vs CN, Dementia vs CN, MCI vs CN)..."

    (cd "$ADNI_DATA_DIR"
        if [ -f "run_adni_pipeline.py" ]; then
            python3 run_adni_pipeline.py
        fi
        if [ -f "run_clean_comparison.py" ]; then
            python3 run_clean_comparison.py
        fi
    )
    echo "  ✅ 完成"
fi

# ══════════════════════════════════════════════════════════════
# 实验 3b: 时间窗口预测
# ══════════════════════════════════════════════════════════════
if $SKIP_TIME; then
    echo ""
    echo "[3b] SKIP — 时间窗口预测"
else
    echo ""
    echo "[3b] 时间窗口预测 (CN→AD, CN→Dementia)..."

    (cd "$ADNI_DATA_DIR"
        if [ -f "run_time_window_pipeline.py" ]; then
            python3 run_time_window_pipeline.py
        fi
    )
    echo "  ✅ 完成"
fi

# ══════════════════════════════════════════════════════════════
# 实验 3c: MCI→AD 转化预测 (正确目标)
# ══════════════════════════════════════════════════════════════
if $SKIP_MCI; then
    echo ""
    echo "[3c] SKIP — MCI→AD 转化预测"
else
    echo ""
    echo "[3c] MCI→AD 转化预测 (核心任务)..."

    (cd "$ADNI_DATA_DIR"
        if [ -f "run_mci_to_ad_pipeline.py" ]; then
            python3 run_mci_to_ad_pipeline.py
        fi
    )
    echo "  ✅ 完成"
fi

echo ""
echo "################################################################"
echo "# ✅ ADNI Phase 1 完成: $(date)"
echo "# 结果: $PHASE1_RESULTS/"
echo "# 日志: $LOG_FILE"
echo "################################################################"

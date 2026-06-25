#!/bin/bash
# ============================================================================
# 05_adni_phase2.sh — ADNI Phase 2 多终点预测实验
# ============================================================================
# 三阶段实验:
#   2A: DXSUM 二分类 — MCI→Dementia 固定窗口 (3yr/5yr/10yr/all-time)
#   2B: DXSUM 生存分析 — MCI→Dementia 连续时间 (RSF + Cox PH)
#   2C: CDR 量表扩展 — CN+MCI 双队列 (二分类 + 生存)
#
# 数据依赖:
#   - local_data/adni/processed/ADNI_baseline_with_time_targets_v2.csv
#   - local_data/adni/processed/CDR_time_targets_v2.csv (Phase 2C 需要)
#
# 输出: local_data/Results_adni/
#
# 用法:
#   bash src/scripts/server/05_adni_phase2.sh                # 全部运行
#   bash src/scripts/server/05_adni_phase2.sh --phase 2A     # 仅 Phase 2A
#   bash src/scripts/server/05_adni_phase2.sh --skip-survival # 跳过生存分析
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/env_setup.sh"
cd "$PROJECT_ROOT"

PHASE="all"
SKIP_SURVIVAL=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --phase) PHASE="$2"; shift 2 ;;
        --skip-survival) SKIP_SURVIVAL=true; shift ;;
        *) shift ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/05_adni_phase2_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo ""
echo "################################################################"
echo "# ADNI Phase 2: 多终点预测"
echo "# Phase: $PHASE"
echo "# 开始: $(date)"
echo "################################################################"

# 验证数据文件
DATA_CSV="$ADNI_DATA_DIR/processed/ADNI_baseline_with_time_targets_v2.csv"
if [ ! -f "$DATA_CSV" ]; then
    echo "  ERROR: 数据文件不存在: $DATA_CSV"
    echo "  请先运行 03_adni_preprocess.sh"
    exit 1
fi
echo "  数据文件: $DATA_CSV ($(wc -c < "$DATA_CSV" | tr -d ' ') bytes)"

# 导出环境变量供 Python 脚本使用
export ADNI_DATA_DIR
export PROJECT_ROOT

# ══════════════════════════════════════════════════════════════
# Phase 2A: DXSUM 二分类
# ══════════════════════════════════════════════════════════════
if [ "$PHASE" = "all" ] || [ "$PHASE" = "2A" ] || [ "$PHASE" = "2a" ]; then
    echo ""
    echo "────────────────────────────────────────────────────"
    echo " Phase 2A: DXSUM 二分类 (MCI→Dementia)"
    echo "────────────────────────────────────────────────────"

    echo ""
    echo "[2A-1] 完整模型 (Bio+Imaging, 358 特征)..."
    python3 src/training/run_adni_mci_dementia.py \
        --window 3,5,10 --model bio_img

    echo ""
    echo "[2A-2] 消融模型 (Bio only, ~20 特征)..."
    python3 src/training/run_adni_mci_dementia.py \
        --window 3,5,10 --model bio_only

    echo ""
    echo "  ✅ Phase 2A → $RESULTS_DIR/Results_adni/mci_to_dementia/"
fi

# ══════════════════════════════════════════════════════════════
# Phase 2B: DXSUM 生存分析
# ══════════════════════════════════════════════════════════════
if [ "$PHASE" = "all" ] || [ "$PHASE" = "2B" ] || [ "$PHASE" = "2b" ]; then
    if $SKIP_SURVIVAL; then
        echo ""
        echo "[2B] SKIP — 跳过生存分析"
    else
        echo ""
        echo "────────────────────────────────────────────────────"
        echo " Phase 2B: DXSUM 生存分析 (MCI→Dementia)"
        echo "────────────────────────────────────────────────────"

        echo ""
        echo "[2B-1] 主实验 (RSF + Cox PH, LightGBM 特征选择)..."
        python3 src/training/run_adni_survival.py

        echo ""
        echo "[2B-2] 优化实验 v3 (Lasso Cox + RSF 参数搜索)..."
        python3 src/training/run_adni_survival_v3.py

        echo ""
        echo "  ✅ Phase 2B → $RESULTS_DIR/Results_adni/survival/"
    fi
fi

# ══════════════════════════════════════════════════════════════
# Phase 2C: CDR 量表扩展
# ══════════════════════════════════════════════════════════════
if [ "$PHASE" = "all" ] || [ "$PHASE" = "2C" ] || [ "$PHASE" = "2c" ]; then
    echo ""
    echo "────────────────────────────────────────────────────"
    echo " Phase 2C: CDR 量表扩展 (CN+MCI 双队列)"
    echo "────────────────────────────────────────────────────"

    # 检查 CDR 目标数据
    CDR_CSV="$ADNI_DATA_DIR/processed/CDR_time_targets_v2.csv"
    if [ ! -f "$CDR_CSV" ]; then
        echo "  [WARN] CDR 目标文件未找到: $CDR_CSV"
        echo "  请确保已运行 03_adni_preprocess.sh 的 Step 3"
    fi

    echo ""
    echo "[2C-1] CDR 二分类 (分队列 — CN & MCI)..."
    python3 src/training/run_adni_cdr_binary_v2.py

    if ! $SKIP_SURVIVAL; then
        echo ""
        echo "[2C-2] CDR 生存分析 (分队列 + SFS 方法对比)..."
        python3 src/training/run_adni_cdr_survival_v2.py
    fi

    echo ""
    echo "  ✅ Phase 2C → $RESULTS_DIR/Results_adni/cdr_binary/ + cdr_survival/"
fi

echo ""
echo "################################################################"
echo "# ✅ ADNI Phase 2 完成: $(date)"
echo "# 日志: $LOG_FILE"
echo "################################################################"

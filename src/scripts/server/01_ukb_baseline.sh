#!/bin/bash
# ============================================================================
# 01_ukb_baseline.sh — UKB 基线实验 (不含 MRI)
# ============================================================================
# 对 6 个目标 (DM/AD × full/10yr/5yr) 运行 s01-s05 管线。
#
# 数据依赖:
#   - local_data/ukb/numpy_data/*.npz        → UKB 原始 .npz 文件
#   - local_data/ukb/bridge_to_training_v3.py → 数据桥接脚本
#
# 输出: local_data/Preprocessed_Data/Preprocessed_Data.csv
#       local_data/Results_v2/
#
# 用法:
#   bash src/scripts/server/01_ukb_baseline.sh
#   bash src/scripts/server/01_ukb_baseline.sh --skip-bridge
#   bash src/scripts/server/01_ukb_baseline.sh --fast
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/env_setup.sh"
cd "$PROJECT_ROOT"

SKIP_BRIDGE=false
FAST_MODE=false
RUN_ALIGNED=true
RUN_EXTERNAL=true

for arg in "$@"; do
    case "$arg" in
        --skip-bridge) SKIP_BRIDGE=true ;;
        --fast) FAST_MODE=true ;;
        --no-aligned) RUN_ALIGNED=false ;;
        --no-external) RUN_EXTERNAL=false ;;
    esac
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/01_ukb_baseline_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo ""
echo "################################################################"
echo "# 实验 1: UKB 基线 (6 目标: DM/AD × full/10yr/5yr)"
echo "# 开始: $(date)"
echo "################################################################"

# ══════════════════════════════════════════════════════════════
# Step 1: 数据桥接 (从 .npz → Preprocessed_Data.csv)
# ══════════════════════════════════════════════════════════════
if $SKIP_BRIDGE; then
    echo ""
    echo "[Step 1] SKIP — 数据桥接已跳过"
else
    echo ""
    echo "[Step 1] 数据桥接..."

    BRIDGE_SCRIPT="$UKB_DATA_DIR/bridge_to_training_v3.py"
    if [ ! -f "$BRIDGE_SCRIPT" ]; then
        echo "  ERROR: 桥接脚本不存在: $BRIDGE_SCRIPT"
        echo "  请先运行 migrate_data.sh 迁移 UKB 数据到 local_data/ukb/"
        exit 1
    fi

    echo "  桥接脚本: $BRIDGE_SCRIPT"
    echo "  输出目录: $RESULTS_DIR/Preprocessed_Data/"

    # 导出 PROJECT_ROOT 让 bridge 脚本找到正确路径
    export PROJECT_ROOT
    python3 "$BRIDGE_SCRIPT"

    echo "  ✅ 数据桥接完成"
    ls -lh "$RESULTS_DIR"/Preprocessed_Data/Preprocessed_Data.csv 2>/dev/null || \
        echo "  [WARN] Preprocessed_Data.csv 未生成 — 检查 UKB numpy_data/ 是否完整"
fi

# 验证数据文件
DATA_CSV="$RESULTS_DIR/Preprocessed_Data/Preprocessed_Data.csv"
if [ ! -f "$DATA_CSV" ]; then
    echo ""
    echo "  ERROR: 数据文件不存在: $DATA_CSV"
    echo "  请先确保 UKB .npz 数据在 local_data/ukb/numpy_data/ 中"
    exit 1
fi
echo "  数据文件: $DATA_CSV ($(wc -c < "$DATA_CSV" | tr -d ' ') bytes)"

# ══════════════════════════════════════════════════════════════
# Step 2: 论文对齐特征实验 (138 特征)
# ══════════════════════════════════════════════════════════════
if $RUN_ALIGNED; then
    echo ""
    echo "[Step 2] 论文对齐特征实验 (138 特征匹配)..."

    FULL_CSV="$RESULTS_DIR/Preprocessed_Data/Preprocessed_Data_full.csv"
    if [ ! -f "$FULL_CSV" ]; then
        echo "  [WARN] Preprocessed_Data_full.csv 未找到, 跳过对齐实验"
        echo "  如需要, 运行 bridge_full.py 生成"
    else
        python3 src/training/run_aligned_pipeline.py
        echo "  ✅ 对齐特征实验完成"
    fi
fi

# ══════════════════════════════════════════════════════════════
# Step 3: 全特征复现 (6 目标)
# ══════════════════════════════════════════════════════════════
echo ""
echo "[Step 3] 全特征复现 (6 目标)..."

if $FAST_MODE; then
    echo "  快速模式: run_s05_final.py"
    python3 src/training/run_s05_final.py
else
    echo "  完整模式: run_training_v2.py (超参数搜索)"
    python3 src/training/run_training_v2.py
fi
echo "  ✅ 完成 → $RESULTS_DIR/Results_v2/"

# ══════════════════════════════════════════════════════════════
# Step 4: 外部验证 (80/20 + CAIDE/ANU-ADRI)
# ══════════════════════════════════════════════════════════════
if $RUN_EXTERNAL; then
    echo ""
    echo "[Step 4] 外部验证 (80/20 + CAIDE/ANU-ADRI)..."
    python3 src/evaluation/external_validation.py
    echo "  ✅ 完成 → $RESULTS_DIR/Results_v2/_external_validation/"

    echo ""
    echo "[Step 5] 扩展评估 (决策曲线, 校准, 风险分层)..."
    python3 src/evaluation/evaluate_extended.py
    echo "  ✅ 完成 → $RESULTS_DIR/Results_v2/_extended_eval/"
fi

echo ""
echo "################################################################"
echo "# ✅ 实验 1 完成: $(date)"
echo "# 日志: $LOG_FILE"
echo "################################################################"

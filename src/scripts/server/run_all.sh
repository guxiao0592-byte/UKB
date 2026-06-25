#!/bin/bash
# ============================================================================
# run_all.sh — 全实验主控脚本
# ============================================================================
# 按依赖顺序运行全部实验:
#   UKB 基线 → UKB+MRI → ADNI 预处理 → ADNI Phase 1 → ADNI Phase 2
#
# 所有数据都在 local_data/ 内部:
#   local_data/ukb/numpy_data/   ← UKB 源 .npz
#   local_data/adni/extracted/   ← ADNI 源 CSV
#
# 用法:
#   bash src/scripts/server/run_all.sh                  # 全量
#   bash src/scripts/server/run_all.sh --skip-ukb       # 仅 ADNI
#   bash src/scripts/server/run_all.sh --fast           # 快速模式
#   bash src/scripts/server/run_all.sh --only adni-phase2  # 仅某阶段
#
#   后台运行:
#   nohup bash src/scripts/server/run_all.sh > logs/run_all.log 2>&1 &
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/env_setup.sh"
cd "$PROJECT_ROOT"

# 解析参数
SKIP_UKB=false; SKIP_ADNI=false; FAST_MODE=false; ONLY=""
SKIP_BRIDGE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-ukb) SKIP_UKB=true; shift ;;
        --skip-adni) SKIP_ADNI=true; shift ;;
        --fast) FAST_MODE=true; shift ;;
        --only) ONLY="$2"; shift 2 ;;
        --skip-bridge) SKIP_BRIDGE="--skip-bridge"; shift ;;
        *) shift ;;
    esac
done

FAST_FLAG=""
if $FAST_MODE; then FAST_FLAG="--fast"; fi

START_TIME=$(date)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="$LOG_DIR/run_all_${TIMESTAMP}.log"
exec > >(tee -a "$MAIN_LOG") 2>&1

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  UKB-DRP 全实验运行                                        ║"
echo "║  开始:  $START_TIME                            ║"
echo "║  主机:  $(hostname)                                        ║"
echo "║  目录:  $PROJECT_ROOT"
echo "╚══════════════════════════════════════════════════════════════╝"

# ══════════════════════════════════════════════════════════════
# 阶段 0: 环境检查
# ══════════════════════════════════════════════════════════════
echo ""
echo "━━━ 阶段 0: 环境检查 ━━━"

python3 -c "
import numpy, pandas, scipy, sklearn, lightgbm, matplotlib
print(f'  numpy={numpy.__version__}, pandas={pandas.__version__}')
print(f'  scipy={scipy.__version__}, sklearn={sklearn.__version__}')
print(f'  lightgbm={lightgbm.__version__}')
" || { echo "ERROR: 核心依赖缺失"; exit 1; }

python3 -c "import sksurv, lifelines" 2>/dev/null && \
    echo "  sksurv + lifelines: OK" || \
    echo "  [WARN] sksurv/lifelines 未安装 — Phase 2B/2C 生存分析将跳过"

check_dir "$UKB_DATA_DIR/numpy_data" && echo "  UKB 源: ✅" || echo "  [WARN] UKB numpy_data 未找到于 $UKB_DATA_DIR"
check_dir "$ADNI_DATA_DIR/extracted" && echo "  ADNI 源: ✅" || echo "  [WARN] ADNI extracted 未找到于 $ADNI_DATA_DIR"

# ══════════════════════════════════════════════════════════════
# 阶段 1: UKB 基线
# ══════════════════════════════════════════════════════════════
run_stage() {
    local STAGE="$1"
    local ONLY_MATCH="$2"
    local SKIP="$3"
    local SCRIPT="$4"
    shift 4

    if [ -n "$ONLY" ] && [ "$ONLY" != "$ONLY_MATCH" ]; then
        echo ""
        echo "━━━ 阶段 $STAGE — SKIP (--only=$ONLY) ━━━"
        return
    fi
    if $SKIP; then
        echo ""
        echo "━━━ 阶段 $STAGE — SKIP ━━━"
        return
    fi

    echo ""
    echo "━━━ 阶段 $STAGE ━━━"
    bash "$SCRIPT_DIR/$SCRIPT" "$@"
}

run_stage 1 "ukb-baseline" $SKIP_UKB "01_ukb_baseline.sh" $FAST_FLAG $SKIP_BRIDGE
run_stage 2 "ukb-imaging" $SKIP_UKB "02_ukb_imaging.sh" --all-targets $FAST_FLAG $SKIP_BRIDGE
run_stage 3 "adni-preprocess" $SKIP_ADNI "03_adni_preprocess.sh"
run_stage 4 "adni-phase1" $SKIP_ADNI "04_adni_phase1.sh"
run_stage 5 "adni-phase2" $SKIP_ADNI "05_adni_phase2.sh"

# ══════════════════════════════════════════════════════════════
# 完成
# ══════════════════════════════════════════════════════════════
END_TIME=$(date)
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✅ 全实验完成                                             ║"
echo "║  开始: $START_TIME                            ║"
echo "║  结束: $END_TIME                            ║"
echo "║  日志: $MAIN_LOG                                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "结果目录:"
echo "  UKB 基线:       $RESULTS_DIR/Results_v2/"
echo "  UKB + MRI:      $RESULTS_DIR/Results_imaging/"
echo "  ADNI Phase 2A:  $RESULTS_DIR/Results_adni/mci_to_dementia/"
echo "  ADNI Phase 2B:  $RESULTS_DIR/Results_adni/survival/"
echo "  ADNI Phase 2C:  $RESULTS_DIR/Results_adni/cdr_binary/ + cdr_survival/"

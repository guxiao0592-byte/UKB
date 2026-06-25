#!/bin/bash
# ============================================================================
# env_setup.sh — 通用环境变量加载 (被其他脚本 source)
# ============================================================================
# 用法: source src/scripts/server/env_setup.sh
# ============================================================================

# 项目根目录 (UKB_DRP-main)
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"

# UKB 源数据目录 (numpy_data/ + bridge_*.py)
export UKB_DATA_DIR="${UKB_DATA_DIR:-$PROJECT_ROOT/local_data/ukb}"

# ADNI 源数据目录 (extracted/ + processed/ + *.py)
export ADNI_DATA_DIR="${ADNI_DATA_DIR:-$PROJECT_ROOT/local_data/adni}"

# 结果目录
export RESULTS_DIR="$PROJECT_ROOT/local_data"

# 日志目录
export LOG_DIR="${LOG_DIR:-$PROJECT_ROOT/logs}"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "UKB-DRP 实验环境"
echo "  PROJECT_ROOT: $PROJECT_ROOT"
echo "  UKB_DATA_DIR: $UKB_DATA_DIR"
echo "  ADNI_DATA_DIR: $ADNI_DATA_DIR"
echo "  RESULTS_DIR:  $RESULTS_DIR"
echo "  LOG_DIR:      $LOG_DIR"
echo "  HOST:         $(hostname)"
echo "  DATE:         $(date)"
echo "============================================"

# 检查关键文件/目录是否存在
check_dir() {
    if [ ! -d "$1" ]; then
        echo "[WARN] 目录不存在: $1"
        return 1
    fi
    return 0
}

check_file() {
    if [ ! -f "$1" ]; then
        echo "[WARN] 文件不存在: $1"
        return 1
    fi
    return 0
}

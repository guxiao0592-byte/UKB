#!/bin/bash
# ============================================================================
# 00_install_deps.sh — 服务器环境安装
# ============================================================================
# 在服务器上首次运行时执行此脚本, 安装所有 Python 依赖
#
# 用法:
#   bash src/scripts/server/00_install_deps.sh
#   bash src/scripts/server/00_install_deps.sh --with-survival  # 含生存分析依赖
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "============================================"
echo "UKB-DRP 服务器环境安装"
echo "  项目目录: $PROJECT_ROOT"
echo "  主机: $(hostname)"
echo "  时间: $(date)"
echo "============================================"

# ---- 检测 Python ----
PYTHON=""
for py in python3 python; do
    if command -v $py &>/dev/null; then
        $py -c "import sys; assert sys.version_info >= (3, 9)" 2>/dev/null || continue
        PYTHON=$py
        break
    fi
done
if [ -z "$PYTHON" ]; then
    echo "ERROR: 未找到 Python ≥ 3.9"
    exit 1
fi
echo "  Python: $($PYTHON --version)"

# ---- 核心依赖 ----
echo ""
echo "[1/3] 安装核心机器学习依赖..."
$PYTHON -m pip install --quiet \
    numpy pandas scipy scikit-learn \
    lightgbm matplotlib

# ---- 生存分析依赖 (可选) ----
WITH_SURVIVAL=false
for arg in "$@"; do
    case "$arg" in
        --with-survival|-s) WITH_SURVIVAL=true ;;
    esac
done

if $WITH_SURVIVAL; then
    echo ""
    echo "[2/3] 安装生存分析依赖..."
    $PYTHON -m pip install --quiet \
        scikit-survival lifelines
else
    echo ""
    echo "[2/3] 跳过生存分析依赖 (Phase 2B/2C 需要; 加 --with-survival 安装)"
fi

# ---- 特有依赖 (ADNI Phase 2) ----
echo ""
echo "[3/3] 安装其他依赖..."
$PYTHON -m pip install --quiet \
    joblib

# ---- 验证 ----
echo ""
echo "============================================"
echo "验证安装..."
echo "============================================"
$PYTHON -c "
import numpy;     print(f'  numpy:          {numpy.__version__}')
import pandas;    print(f'  pandas:         {pandas.__version__}')
import scipy;     print(f'  scipy:          {scipy.__version__}')
import sklearn;   print(f'  scikit-learn:   {sklearn.__version__}')
import lightgbm;  print(f'  lightgbm:       {lightgbm.__version__}')
import matplotlib;print(f'  matplotlib:     {matplotlib.__version__}')
"
if $WITH_SURVIVAL; then
    $PYTHON -c "
import sksurv;     print(f'  scikit-survival:{sksurv.__version__}')
import lifelines;  print(f'  lifelines:      {lifelines.__version__}')
" || echo "  [WARN] 生存分析包导入失败, 请检查"
fi

echo ""
echo "✅ 环境安装完成"
echo ""
echo "接下来:"
echo "  1. 确保 UKB 数据目录存在:   export UKB_DATA_DIR=/path/to/UKB数据集"
echo "  2. 确保 ADNI 数据目录存在:   export ADNI_DATA_DIR=/path/to/ADNI数据集"
echo "  3. 运行实验:  bash src/scripts/server/run_all.sh"

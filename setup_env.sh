#!/bin/bash
# ============================================================================
# UKB-DRP — 环境安装
# 用法: bash setup_env.sh [--with-survival]
# 新位置: src/scripts/server/00_install_deps.sh
# ============================================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
bash "$SCRIPT_DIR/src/scripts/server/00_install_deps.sh" "$@"

#!/bin/bash
# ============================================================================
# migrate_data.sh — 源数据迁入 local_data/
# ============================================================================
# 将 UKB 和 ADNI 的源数据从旧位置复制/链接到 local_data/ 内部。
#
# 目标结构:
#   local_data/
#   ├── ukb/numpy_data/    ← UKB .npz 文件
#   ├── ukb/bridge_*.py    ← UKB 数据桥接脚本
#   ├── adni/extracted/    ← ADNI 66 个原始 CSV
#   ├── adni/processed/    ← ADNI 预处理输出 (或由预处理生成)
#   └── adni/*.py          ← ADNI 预处理/训练脚本
#
# 用法:
#   # 自动检测并使用旧位置
#   bash src/scripts/server/migrate_data.sh
#
#   # 明确指定旧位置
#   bash src/scripts/server/migrate_data.sh \
#       --ukb-src /path/to/UKB数据集 \
#       --adni-src /path/to/ADNI数据集
#
#   # 仅链接 (不复制)
#   bash src/scripts/server/migrate_data.sh --link
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/env_setup.sh"
cd "$PROJECT_ROOT"

USE_LINK=false
UKB_SRC=""
ADNI_SRC=""

for arg in "$@"; do
    case "$arg" in
        --link) USE_LINK=true ;;
        --ukb-src) UKB_SRC="$2"; shift ;;
        --adni-src) ADNI_SRC="$2"; shift ;;
    esac
done

# ---- 自动检测旧位置 ----
if [ -z "$UKB_SRC" ]; then
    for candidate in \
        "$(dirname "$PROJECT_ROOT")/UKB数据集" \
        "$(dirname "$PROJECT_ROOT")/思路一/UKB数据集" \
        "/Users/guxiao/Downloads/MCI-AD/思路一/UKB数据集"; do
        if [ -d "$candidate/numpy_data" ]; then
            UKB_SRC="$candidate"
            break
        fi
    done
fi

if [ -z "$ADNI_SRC" ]; then
    for candidate in \
        "$(dirname "$PROJECT_ROOT")/ADNI数据集" \
        "$(dirname "$PROJECT_ROOT")/思路一/ADNI数据集" \
        "/Users/guxiao/Downloads/MCI-AD/思路一/ADNI数据集"; do
        if [ -d "$candidate/extracted" ]; then
            ADNI_SRC="$candidate"
            break
        fi
    done
fi

COPY_CMD="cp -r"
if $USE_LINK; then
    COPY_CMD="ln -sf"
    echo "⚠️  使用符号链接模式 (--link)"
fi

echo "============================================"
echo "源数据迁移 → local_data/"
echo "  PROJECT_ROOT: $PROJECT_ROOT"
echo "  UKB 源:  $UKB_SRC"
echo "  ADNI 源: $ADNI_SRC"
echo "  方式:    $([ "$USE_LINK" = true ] && echo '符号链接' || echo '复制')"
echo "============================================"

# ══════════════════════════════════════════════════════════════
# 1. UKB 数据
# ══════════════════════════════════════════════════════════════
echo ""
echo "[1/4] 迁移 UKB 数据..."

UKB_DST="$PROJECT_ROOT/local_data/ukb"
mkdir -p "$UKB_DST"

if [ -z "$UKB_SRC" ] || [ ! -d "$UKB_SRC" ]; then
    echo "  [SKIP] UKB 源目录未找到, 请手动放置"
else
    # numpy_data
    if [ -d "$UKB_SRC/numpy_data" ] && [ ! -d "$UKB_DST/numpy_data" ]; then
        echo "  迁移 numpy_data/ ($(du -sh "$UKB_SRC/numpy_data" | cut -f1))..."
        if $USE_LINK; then
            ln -sf "$(cd "$UKB_SRC" && pwd)/numpy_data" "$UKB_DST/numpy_data"
        else
            cp -r "$UKB_SRC/numpy_data" "$UKB_DST/"
        fi
        echo "  ✅ numpy_data/"
    fi

    # features
    if [ -d "$UKB_SRC/features" ] && [ ! -d "$UKB_DST/features" ]; then
        echo "  迁移 features/"
        if $USE_LINK; then
            ln -sf "$(cd "$UKB_SRC" && pwd)/features" "$UKB_DST/features"
        else
            cp -r "$UKB_SRC/features" "$UKB_DST/"
        fi
        echo "  ✅ features/"
    fi

    # bridge scripts
    for py in bridge_to_training.py bridge_to_training_v2.py \
        bridge_to_training_v3.py bridge_imaging.py bridge_full.py; do
        if [ -f "$UKB_SRC/$py" ] && [ ! -f "$UKB_DST/$py" ]; then
            $USE_LINK && ln -sf "$(cd "$UKB_SRC" && pwd)/$py" "$UKB_DST/$py" || \
                cp "$UKB_SRC/$py" "$UKB_DST/"
        fi
    done
    echo "  ✅ bridge 脚本"

    # config files
    for f in config_coding.ini config_data.ini config_vdt.py data_list.csv \
        data_prep.py data_utils.py info_parser.py main.py; do
        if [ -f "$UKB_SRC/$f" ] && [ ! -f "$UKB_DST/$f" ]; then
            $USE_LINK && ln -sf "$(cd "$UKB_SRC" && pwd)/$f" "$UKB_DST/$f" || \
                cp "$UKB_SRC/$f" "$UKB_DST/"
        fi
    done

    echo "  UKB 目标: $(du -sh "$UKB_DST" | cut -f1)"
fi

# ══════════════════════════════════════════════════════════════
# 2. ADNI 数据
# ══════════════════════════════════════════════════════════════
echo ""
echo "[2/4] 迁移 ADNI 数据..."

ADNI_DST="$PROJECT_ROOT/local_data/adni"
mkdir -p "$ADNI_DST/extracted" "$ADNI_DST/processed"

if [ -z "$ADNI_SRC" ] || [ ! -d "$ADNI_SRC" ]; then
    echo "  [SKIP] ADNI 源目录未找到, 请手动放置"
else
    # extracted CSV files
    if [ -d "$ADNI_SRC/extracted" ]; then
        N_CSV=$(ls "$ADNI_SRC/extracted"/*.csv 2>/dev/null | wc -l | tr -d ' ')
        if [ "$N_CSV" -gt 0 ]; then
            echo "  迁移 extracted/ ($N_CSV CSV, $(du -sh "$ADNI_SRC/extracted" | cut -f1))..."
            if $USE_LINK; then
                ln -sf "$(cd "$ADNI_SRC" && pwd)/extracted" "$ADNI_DST/extracted"
            else
                cp "$ADNI_SRC/extracted"/*.csv "$ADNI_DST/extracted/"
            fi
            echo "  ✅ extracted/"
        fi
    fi

    # Python scripts
    for py in preprocess_adni.py build_time_targets.py build_time_targets_v2.py \
        build_cdr_targets.py build_cdr_targets_v2.py \
        run_adni_pipeline.py run_mci_to_ad_pipeline.py \
        run_time_window_pipeline.py run_clean_comparison.py; do
        if [ -f "$ADNI_SRC/$py" ] && [ ! -f "$ADNI_DST/$py" ]; then
            $USE_LINK && ln -sf "$(cd "$ADNI_SRC" && pwd)/$py" "$ADNI_DST/$py" || \
                cp "$ADNI_SRC/$py" "$ADNI_DST/"
        fi
    done
    echo "  ✅ ADNI 脚本"

    # processed data (if exists)
    if [ -d "$ADNI_SRC/processed" ] && [ "$(ls -A "$ADNI_SRC/processed" 2>/dev/null)" ]; then
        echo "  迁移 processed/ ($(du -sh "$ADNI_SRC/processed" | cut -f1))..."
        cp "$ADNI_SRC/processed"/*.csv "$ADNI_DST/processed/" 2>/dev/null || true
        echo "  ✅ processed/"
    fi

    echo "  ADNI 目标: $(du -sh "$ADNI_DST" | cut -f1)"
fi

# ══════════════════════════════════════════════════════════════
# 3. UKB bridge 脚本路径修复
# ══════════════════════════════════════════════════════════════
echo ""
echo "[3/4] 修复 bridge 脚本路径..."

# bridge_to_training_v3.py 使用了 os.path.dirname(os.path.dirname(__file__))
# 来计算 PROJECT_ROOT。放在 local_data/ukb/ 后它会计算为 local_data/。
# 需要让它优先使用 PROJECT_ROOT 环境变量。
fix_bridge_script() {
    local SCRIPT="$1"
    if [ ! -f "$SCRIPT" ]; then return; fi
    local OLD_LINE='PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))'
    local NEW_LINE='PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))'
    if grep -q "$OLD_LINE" "$SCRIPT" 2>/dev/null; then
        # macOS sed needs ''
        sed -i '' "s|$OLD_LINE|$NEW_LINE|" "$SCRIPT" 2>/dev/null || \
        sed -i "s|$OLD_LINE|$NEW_LINE|" "$SCRIPT"
        echo "  已修复: $SCRIPT"
    else
        echo "  [SKIP] 无需修复或已修复: $SCRIPT"
    fi
}

for py in "$UKB_DST"/bridge_*.py; do
    fix_bridge_script "$py"
done

# ══════════════════════════════════════════════════════════════
# 4. 验证
# ══════════════════════════════════════════════════════════════
echo ""
echo "[4/4] 验证..."

verify_dir() {
    if [ -d "$1" ]; then
        echo "  ✅ $1 ($(ls "$1" 2>/dev/null | wc -l | tr -d ' ') 项)"
    else
        echo "  ❌ $1 — 不存在"
    fi
}
verify_file() {
    if [ -f "$1" ]; then
        echo "  ✅ $1"
    else
        echo "  ❌ $1 — 不存在"
    fi
}

echo "UKB:"
verify_dir "$UKB_DST/numpy_data"
verify_file "$UKB_DST/bridge_to_training_v3.py"
verify_file "$UKB_DST/bridge_imaging.py"

echo "ADNI:"
verify_dir "$ADNI_DST/extracted"
verify_file "$ADNI_DST/preprocess_adni.py"
verify_file "$ADNI_DST/build_time_targets_v2.py"

echo ""
echo "============================================"
echo "迁移完成!"
echo ""
echo "新目录结构:"
echo "  local_data/ukb/   ← UKB 源数据 ($(du -sh "$UKB_DST" 2>/dev/null | cut -f1))"
echo "  local_data/adni/  ← ADNI 源数据 ($(du -sh "$ADNI_DST" 2>/dev/null | cut -f1))"
echo ""
echo "接下来运行实验: bash src/scripts/server/run_all.sh"
echo "============================================"

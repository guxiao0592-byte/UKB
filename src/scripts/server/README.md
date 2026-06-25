# UKB-DRP 服务器运行指南

## 目录结构

所有数据都在 `local_data/` 内部，项目完全自包含：

```
UKB_DRP-main/
├── src/scripts/server/       ← 实验脚本 (本目录)
│   ├── env_setup.sh          ← 环境变量加载
│   ├── 00_install_deps.sh    ← 依赖安装
│   ├── migrate_data.sh       ← 源数据迁入
│   ├── 01_ukb_baseline.sh    ← UKB 基线
│   ├── 02_ukb_imaging.sh     ← UKB + MRI
│   ├── 03_adni_preprocess.sh ← ADNI 预处理
│   ├── 04_adni_phase1.sh     ← ADNI Phase 1
│   ├── 05_adni_phase2.sh     ← ADNI Phase 2
│   └── run_all.sh            ← 一键全流程
│
├── local_data/
│   ├── ukb/                  ← UKB 源数据 (需迁移)
│   │   ├── numpy_data/       ← 3000 个 .npz 文件 (416MB)
│   │   ├── features/         ← UKB 特征配置
│   │   ├── bridge_to_training_v3.py
│   │   └── bridge_imaging.py
│   │
│   ├── adni/                 ← ADNI 源数据 (需迁移)
│   │   ├── extracted/        ← 66 个 All_Subjects_*_20May2026.csv (224MB)
│   │   ├── processed/        ← 预处理输出 (自动生成)
│   │   ├── preprocess_adni.py
│   │   ├── build_time_targets_v2.py
│   │   └── build_cdr_targets_v2.py
│   │
│   ├── Preprocessed_Data/    ← UKB 桥接输出 CSV (7.7GB, 桥接后生成)
│   ├── Data/                 ← 特征映射文件
│   ├── Results_v2/           ← UKB 实验结果
│   ├── Results_imaging/      ← UKB+MRI 结果
│   └── Results_adni/         ← ADNI Phase 2 结果
```

## 迁移数据

将本机的 UKB 和 ADNI 源数据迁入 `local_data/`:

```bash
cd UKB_DRP-main

# 自动检测并迁移 (从上级目录的 UKB数据集/ 和 ADNI数据集/)
bash src/scripts/server/migrate_data.sh

# 或明确指定源位置
bash src/scripts/server/migrate_data.sh \
    --ukb-src /path/to/UKB数据集 \
    --adni-src /path/to/ADNI数据集

# 或使用符号链接 (节省磁盘空间)
bash src/scripts/server/migrate_data.sh --link
```

## 安装依赖

```bash
cd UKB_DRP-main
bash src/scripts/server/00_install_deps.sh --with-survival
```

## 运行实验

```bash
cd UKB_DRP-main

# 全部实验
nohup bash src/scripts/server/run_all.sh > logs/run_all.log 2>&1 &

# 快速模式
bash src/scripts/server/run_all.sh --fast

# 仅 UKB
bash src/scripts/server/run_all.sh --skip-adni

# 仅 ADNI Phase 2
bash src/scripts/server/run_all.sh --only adni-phase2

# 单步运行
bash src/scripts/server/01_ukb_baseline.sh --fast
bash src/scripts/server/05_adni_phase2.sh --phase 2A
```

## 实验清单

| 脚本 | 实验内容 | 输出 | 预计耗时 |
|------|---------|------|---------|
| 01 | UKB 基线 (6目标) | Results_v2/ | 2-6h |
| 02 | UKB + MRI | Results_imaging/ | 4-12h |
| 03 | ADNI 预处理 | adni/processed/ | 5min |
| 04 | ADNI Phase 1 | adni/results/ | 1-2h |
| 05 --phase 2A | DXSUM 二分类 | Results_adni/mci_to_dementia/ | 1h |
| 05 --phase 2B | DXSUM 生存分析 | Results_adni/survival/ | 2h |
| 05 --phase 2C | CDR 扩展 | Results_adni/cdr_binary/ + cdr_survival/ | 3h |

## 注意事项

### 内存需求
- UKB 全队列 (~425K 人): 至少 32GB RAM
- UKB + MRI: 至少 64GB RAM
- ADNI (~1000 人): 8GB RAM 足够

### UKB bridge 脚本路径修复
`migrate_data.sh` 会自动修复 bridge 脚本中的 `PROJECT_ROOT` 计算。
bridge 脚本原本通过 `os.path.dirname(os.path.dirname(__file__))` 推算项目根目录，
迁移到 `local_data/ukb/` 后会计算错误。修复后优先使用 `$PROJECT_ROOT` 环境变量。

### 生存分析依赖
```bash
pip install scikit-survival lifelines
```

### Git 忽略
`.gitignore` 已配置为忽略所有大文件 (numpy_data/, extracted/, Preprocessed_Data/ 等)。
只追踪脚本代码和轻量结果 CSV。

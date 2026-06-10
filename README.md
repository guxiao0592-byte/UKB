# UKB-DRP 痴呆风险预测模型 — 复现实验 + ADNI 外部验证

> 论文: Yu et al., "Development of a novel dementia risk prediction model..." (eClinicalMedicine, 2022)

## 项目结构

```
UKB_DRP-main/
├── src/                              # 复现代码
│   ├── training/                     # 训练管线 (s01-s05)
│   └── evaluation/                   # 评估分析 (DCA/ECE/PR/外部验证)
│
├── ADNI数据集/                        # ★ ADNI 外部验证
│   ├── extracted/                    # 66 个原始 CSV 表 (ADNI IDA 导出)
│   ├── processed/                    # 预处理数据 + 时间目标
│   ├── results/                      # 训练结果 (MCI→AD / 消融 / 时间窗口)
│   ├── preprocess_adni.py            # 66表 → 基线合并
│   ├── build_time_targets.py         # 纵向诊断追踪 → 时间窗口
│   ├── run_mci_to_ad_pipeline.py     # MCI→AD 转化预测 (核心)
│   └── run_clean_comparison.py       # 三模型消融对比
│
├── UKB数据集/                         # UKB 数据预处理
├── docs/                             # 文档
│   ├── ADNI验证实验报告.md            # ★ ADNI 完整实验报告
│   ├── ADNI队列差异与模型效果分析报告.md # ★ 队列差异分析
│   ├── compare.md                    # 论文 vs 复现 16步对照
│   ├── three_way_comparison.md       # 论文文字 vs 代码 vs 复现
│   ├── 实验日志.md                   # 完整实验历程
│   └── PPT/                          # PPT 汇报文件
│
├── local_data/                       # 本地数据和结果
└── [论文原始开源代码]
```

## 实验体系总览

```
┌──────────┬───────────────┬───────────────────┬──────────────┬───────────────────────┐
│  实验    │ 数据          │ 预测任务          │ 临床场景     │ AUC                   │
├──────────┼───────────────┼───────────────────┼──────────────┼───────────────────────┤
│ 论文原实验│ UK Biobank    │ 人群 → 痴呆/AD    │ 社区筛查     │ DM 0.848 / AD 0.862   │
│ UKB 复现 │ UKB (本地)    │ 人群 → 痴呆/AD    │ 社区筛查     │ DM 0.831 / AD 0.836   │
│ ADNI 验证│ ADNI          │ MCI → AD 转化     │ 记忆诊所预后 │ 0.860 (MCI→AD 5yr)    │
└──────────┴───────────────┴───────────────────┴──────────────┴───────────────────────┘
```

> **核心结论：UKB（筛查）和 ADNI（预后）互补，不具可比性。** 详见 `docs/ADNI验证实验报告.md`

## 快速开始

```bash
# === UKB 复现实验 ===
python UKB数据集/bridge_full.py                      # 数据预处理
python src/training/run_aligned_pipeline.py           # s01-s05 训练
python src/evaluation/external_validation.py          # 80/20 外部验证

# === ADNI 外部验证 ===
python ADNI数据集/preprocess_adni.py                  # 66表 → 基线数据
python ADNI数据集/build_time_targets.py               # 随访追踪 → 时间窗口
python ADNI数据集/run_mci_to_ad_pipeline.py           # MCI→AD 转化预测
```

## 主要发现

**UKB 复现**:
1. DM 系列基本可复现（AUC 0.83，偏差 < 0.02）
2. AD_5yrs Deploy 策略失效：缺 ApoEε4 时独立训练可达 AUC 0.85
3. 论文 s04 代码是累积AUC，非 SFS（与文字描述不一致）
4. APOE4 被论文 s01 的 inner join 意外排除
5. 138 论文对齐特征 vs 1076 特征，AUC 仅差 0.001
6. 模型校准极好（ECE < 0.001），NPV > 99.5%

**ADNI 外部验证**:
7. UKB（筛查）和 ADNI（预后）不具可比性，应互补使用
8. ADNI 正确目标是 MCI→AD 转化 (AUC 0.86)，而非人群筛查
9. Amyloid PET + pTau217 + NfL 是 MCI→AD 的核心预测组合
10. 小样本下校准方法选择至关重要：CalibratedClassifierCV 避免数据浪费

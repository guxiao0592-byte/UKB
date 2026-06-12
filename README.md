# UKB-DRP: 痴呆风险预测模型 — 从人群筛查到临床预后

[![Phase 1](https://img.shields.io/badge/Phase%201-UKB%20复现-blue)](https://github.com/guxiao0592-byte/UKB/tree/main/docs/UKB)
[![Phase 2](https://img.shields.io/badge/Phase%202-ADNI%20验证-green)](https://github.com/guxiao0592-byte/UKB/tree/main/docs/ADNI)
[![Model](https://img.shields.io/badge/Model-LightGBM%20%2B%20RSF-orange)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey)]()

复现 Yu et al. (eClinicalMedicine, 2024) 的痴呆风险预测模型，并扩展至脑 MRI 影像特征和 ADNI 独立临床队列的外部验证，最终构建 MCI→Dementia 的生存预测模型。

---

## 实验体系

```
┌─────────────────────────────────────────────────────────────────┐
│                    两阶段实验覆盖临床全路径                        │
├──────────────────┬──────────────────┬───────────────────────────┤
│   Phase 1: UKB    │   Phase 2: ADNI   │                           │
│   人群筛查         │   临床预后         │                           │
├──────────────────┼──────────────────┼───────────────────────────┤
│ 数据    425K 人   │ 数据    2.7K 人   │                           │
│ 目标    健康→痴呆  │ 目标    MCI→痴呆   │                           │
│ 特征    问卷+血检  │ 特征    PET+CSF    │                           │
│         +脑MRI    │         +血浆+MRI  │                           │
│ 模型    LGBM      │ 模型    LGBM+RSF  │                           │
│ AUC     0.84      │ AUC     0.86      │                           │
│ C-index —         │ C-index 0.75      │                           │
├──────────────────┴──────────────────┴───────────────────────────┤
│  同一个 s01-s05 管线 (特征排序→聚类→选择→校准)                    │
│  同一套评估体系 (AUC + C-index + Brier + 校准曲线)                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 主要结果

### UK Biobank — 人群筛查 (Phase 1)

| 目标 | 仅临床 AUC | +MRI AUC | MRI 提升 | 论文 AUC |
|------|:---------:|:--------:|:--------:|:--------:|
| DM_full | 0.831 | 0.837 | +0.006 | ~0.848 |
| DM_5yrs | 0.816 | 0.842 | +0.026 | ~0.847 |
| AD_full | 0.836 | 0.845 | +0.009 | ~0.862 |
| AD_5yrs | 0.667 | **0.851** | **+0.184** | ~0.890 |

> AD_5yrs 的 +0.184 提升来自海马下托体积——Braak II 期最早的 tau 沉积区域，恰好覆盖 AD 的 5 年预测窗口。

### ADNI — MCI→Dementia 预后 (Phase 2)

| 模型 | 指标 | 3yr | 5yr | 10yr |
|------|------|:---:|:---:|:----:|
| **二分类 LGBM** (主) | AUC | 0.834 | 0.855 | – |
| **生存 RSF** (主) | C-index | – | 0.745 | – |
| 生存 RSF | tAUC | 0.733 | 0.791 | 0.813 |

> C-index=0.745 是在正确排除认知测试 (防止诊断泄漏) 后的合理性能，与文献基线 (Jahani 2025 RSF=0.878 含认知测试; Aghajanian 2025 纯影像=0.70) 一致。

---

## 项目结构

```
UKB_DRP-main/
├── src/
│   ├── training/                     # 训练管线
│   │   ├── run_training_v2.py        #   UKB 复现 s01-s05
│   │   ├── run_training_imaging.py   #   UKB + MRI 影像
│   │   ├── run_adni_mci_dementia.py  #   ADNI 二分类
│   │   ├── run_adni_survival.py      #   ADNI 生存模型 (RSF SFS)
│   │   └── run_adni_survival_v3.py   #   ADNI 生存模型 (Lasso+Grid)
│   ├── evaluation/                   # 评估分析
│   │   ├── evaluate_extended.py      #   DCA/ECE/PR/风险分层
│   │   ├── compare_risk_scores.py    #   vs CAIDE & ANU-ADRI
│   │   └── external_validation.py    #   80/20 Hold-out
│   ├── visualization/                # 图表生成
│   └── utils/
│       └── task_history.py           #   版本化任务历史
│
├── docs/                             # 完整文档
│   ├── UKB/                          #   Phase 1 文档
│   │   ├── compare.md                #     论文 vs 复现 16步对照
│   │   └── three_way_comparison.md   #     论文文字 vs 代码 vs 复现
│   ├── ADNI/
│   │   ├── phase1/                   #   ADNI 初始验证文档
│   │   └── phase2/                   #   Phase 2 完整文档
│   │       ├── ADNI数据条目目录.md    #      66 CSV 完整目录
│   │       ├── MCI-Dementia实验报告.md #    二分类完整报告
│   │       └── Phase2B-生存模型实验报告.md # 生存模型完整报告
│   ├── 实验日志.md                    #   完整时间线
│   └── 全部实验总览.md                #   结构化总览
│
├── local_data/Results_adni/          # ADNI 实验结果
│   ├── mci_to_dementia/              #   二分类结果
│   └── survival/                     #   生存模型结果
│
└── 论文原始代码/                      # 论文开源代码
    ├── AD_Training/                  #   s01-s05
    ├── DataGeneration/               #   S01-S06
    └── Deploy_Models/               #   Deploy
```

---

## 方法论

### s01-s05 管线

所有实验共享同一管线。Phase 1 (分类) 和 Phase 2 (生存) 的唯一差异是评估指标和模型引擎：

```
┌───────┬──────────────────────────────┬─────────────────────────────────┐
│ 步骤   │ 方法                          │ Phase 1 (分类) / Phase 2 (生存)  │
├───────┼──────────────────────────────┼─────────────────────────────────┤
│ s01    │ LightGBM / 单变量 C-index 排序 │ Gain 排序 / C-index 排序          │
│ s02    │ Ward 层次聚类 (Spearman ρ)    │ 相同 (阈值 0.45-0.75)             │
│ s03    │ 聚类后重排序                  │ 相同                             │
│ s04    │ 贪心 SFS 特征选择             │ AUC 增益 / C-index 增益           │
│ s05    │ 校准 + 5-fold CV 评估         │ Isotonic / RSF + Cox PH          │
└───────┴──────────────────────────────┴─────────────────────────────────┘
```

### 关键方法论决策

- **s04 从累积 AUC 修正为 SFS**: 论文代码是累积 AUC，与正文描述不一致。本复现采用正文描述的 SFS。
- **认知测试排除**: MMSE, ADAS, FAQ, CDR 等全部排除以避免诊断泄漏。这是文献 C-index=0.88 与本实验 C-index=0.75 差异的主因。
- **全队列 NaN 处理**: UKB+MRI 实验中，无 MRI 者保留 NaN，LightGBM 原生处理。证明仅用影像子集训练会因健康志愿者偏差导致过拟合。
- **小样本校准**: ADNI 中用 CalibratedClassifierCV 替代 split-train-calibrate，避免 1,120→538 训练样本的数据浪费。
- **删失处理**: ADNI 生存模型正确排除 `time=0` 的零随访受试者 (176人)，621 个删失者保留为 `(t, 0)`。

---

## 快速开始

**先打开 conda 环境，确保 lightgbm, scikit-survival, lifelines 可用.**

```bash
git clone https://github.com/guxiao0592-byte/UKB.git
cd UKB-DRP-main

# === Phase 1: UKB 复现 ===
python src/training/run_training_v2.py         # s01-s05 (需 UKB .npz 数据)
python src/evaluation/external_validation.py   # 80/20 外部验证

# === Phase 2: ADNI 验证 ===
# 二分类 (约 12 分钟)
python src/training/run_adni_mci_dementia.py --window 3,5

# 生存模型 (约 2 分钟)
python src/training/run_adni_survival.py
```

---

## 关键发现

1. **论文可复现** — UKB DM 系列 AUC 偏差 < 0.02，AD_5yrs 偏低归因于缺 APOE4
2. **脑 MRI 显著提升 AD 预测** — 尤其 AD_5yrs: 0.667→0.851 (+0.184)
3. **ADNI 的正确目标是 MCI→Dementia** — CN→AD 在 ADNI 中不可行 (3yr 仅 1 事件)
4. **生存模型优于二分类** — 保留删失样本 (多 57% 数据)，自动保证时间一致性
5. **C-index=0.75 是排除认知测试后的合理上界** — Lasso + Grid Search 均未能突破
6. **Amyloid PET 是最稳定的跨窗口预测因子** — 在二分类 4/4 窗口 Top 3 中稳定出现
7. **UKB (筛查) 和 ADNI (预后) 互补** — 覆盖临床路径不同阶段，不应直接比较 AUC

---

## 依赖

```
python >= 3.9
lightgbm >= 4.0
scikit-learn >= 1.3
scikit-survival >= 0.22
lifelines >= 0.27
scipy, pandas, numpy, matplotlib
```

---

## 引用

- Yu J, et al. "Development and validation of machine learning models for predicting the risk of dementia..." *eClinicalMedicine*, 2024.
- Jahani S, et al. "Assessing the accuracy of survival machine learning and traditional statistical models for Alzheimer's disease prediction over time." *BMC Medical Research Methodology*, 2025.

## 许可证

MIT

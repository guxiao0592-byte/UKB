# UKB-DRP 痴呆风险预测模型 — 复现实验

> 论文: Yu et al., "Development of a novel dementia risk prediction model..." (eClinicalMedicine, 2022)

## 项目结构

```
UKB_DRP-main/
├── src/                          # 复现代码
│   ├── training/                 # 训练管线
│   │   ├── run_training_v2.py        # 完整 s01-s05 + Deploy + SHAP
│   │   ├── run_final_fast.py         # 快速版 s05 + Deploy
│   │   ├── run_s05_final.py          # 带超参数搜索版
│   │   └── run_aligned_pipeline.py   # 论文特征对齐 + 重跑
│   └── evaluation/               # 评估分析
│       ├── evaluate_extended.py      # DCA/ECE/PR/风险分层/KS
│       ├── compare_risk_scores.py    # vs CAIDE & ANU-ADRI + DeLong
│       └── external_validation.py    # 80/20 Hold-out 验证
│
├── UKB数据集/                    # 数据预处理
│   ├── bridge_to_training_v3.py  # baseline提取 (.npz → CSV)
│   ├── bridge_full.py            # 完整多实例提取
│   └── data_list.csv             # UKB字段目录 (7968字段)
│
├── [论文原始开源代码]             # DataGeneration/, AD_Training/, Plots/,
│   RiskScoreEvaluation/, Utility/, Web-ShinyApp-Develop/, Deploy_Models/
│
├── docs/                         # 文档
│   ├── compare.md                # 论文 vs 复现 16步对照
│   ├── three_way_comparison.md   # 论文文字 vs 代码 vs 复现 三方对比
│   └── 实验日志.md                # 完整实验历程
│
├── local_data/                   # 数据和结果
│   └── Results_v2/               # 6目标 + 扩展评估 + 外部验证 + 风险评分
│
└── README.md
```

## 快速开始

```bash
# 1. 数据预处理
python UKB数据集/bridge_full.py      # 提取论文对齐的138特征
# 或
python UKB数据集/bridge_to_training_v3.py  # 提取1076特征

# 2. 特征选择 + 训练
python src/training/run_aligned_pipeline.py   # 对齐版
# 或
python src/training/run_training_v2.py        # 完整版

# 3. 扩展评估
python src/evaluation/evaluate_extended.py    # DCA/ECE/PR/风险分层
python src/evaluation/compare_risk_scores.py  # vs 传统风险评分
python src/evaluation/external_validation.py  # 独立测试集验证
```

## 主要发现

1. **DM 系列基本可复现**（AUC 0.83，偏差 < 0.02）
2. **AD_5yrs Deploy 策略失效**（缺 ApoEε4 时独立训练可达 AUC 0.85）
3. **论文 s04 代码是累积AUC，非 SFS**（与文字描述不一致）
4. **APOE4 被论文 s01 的 inner join 意外排除**
5. **138 论文对齐特征 vs 1076 特征，AUC 仅差 0.001**
6. **模型校准极好**（ECE < 0.001），NPV > 99.5%，适合 rule-out 筛查

## 下一步

- 纳入脑 MRI 影像数据（.npz 中有 902 个脑 MRI IDP 字段未使用）
- 结合影像 + 临床特征进行特征筛选

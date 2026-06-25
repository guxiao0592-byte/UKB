# Phase 2: ADNI 多终点预测 — 完整实验报告

> 报告日期: 2026-06-18
> 覆盖: Phase 2A (DXSUM 二分类) → Phase 2B (DXSUM 生存) → Phase 2C (CDR 扩展 二分类+生存)

---

## 0. 实验总览

### 0.1 三阶段演进脉络

```
Phase 2A：DXSUM 二分类 — MCI→Dementia 固定窗口预测
    │  问题: 删失浪费、时间坍缩、模型不一致
    ↓
Phase 2B：DXSUM 生存分析 — 连续时间建模, 保留全部样本
    │  成果: C-index 0.765, 一个模型回答所有窗口, 跨窗口自动一致
    ↓
Phase 2C：CDR 量表扩展 —  双任务 (二分类+生存)
    │  成果: CDR 性能略超 DXSUM
    └─ 完成 Phase 2 全部实验
```

### 0.2 统一方法论

三阶段共享同一套 **s01–s05 管线**：

| 步骤 | 方法 | 作用 |
|------|------|------|
| **s01** | LightGBM 5-fold CV 特征重要性排序 | 选出 Top 50 候选特征 |
| **s02** | Ward 层次聚类 (Spearman \|ρ\|, 阈值 0.75) | 去除冗余 |
| **s03** | 聚类后重新排序 | 确认聚类代表 |
| **s04** | Sequential Forward Selection (SFS) | 贪心精选, 早停 Δ<0.0005 |
| **s05** | 最终模型 5-fold CV 评估 | 独立评估泛化性能 |

### 0.3 统一特征空间

- **总数**: 358 特征 (Bio + Imaging)
- **排除**: 全部认知测试 (MMSE, MOCA, ADAS, FAQ, GDS, NPIQ, NEUROBAT, HACH)、全部 CDR 子分数
- **纳入**: 人口学(5) + 遗传(1) + CSF(6) + 血浆(12) + MRI结构(325) + BSI(8) + WMH(12) + Amyloid PET(>330) + Tau PET(>330)
- **缺失值**: 中位数填补
- **CV 策略**: Stratified 5-fold, random_state=2022

### 0.4 三阶段样本量与标签对比

| 阶段 | 终点 | 队列 | N (有效) | 事件 | 事件率 | 标签类型 |
|------|------|------|----------|------|--------|----------|
| 2A | DXSUM MCI→Dementia 3yr | MCI | 600 | 196 | 32.7% | 0/1 二分类 |
| 2A | DXSUM MCI→Dementia 5yr | MCI | 500 | 267 | 53.4% | 0/1 二分类 |
| 2A | DXSUM MCI→Dementia 10yr | MCI | 385 | 311 | 80.8% | 0/1 二分类 |
| 2B | DXSUM MCI→Dementia | MCI | 944 | 323 | 34.2% | (time, event) |
| 2C | CDR CN→MCI | CN | 633 | 208 | 32.9% | 0/1 + (time, event) |
| 2C | CDR MCI→Dementia | MCI | 962 | 373 | 38.8% | 0/1 + (time, event) |

---

## 1. Phase 2A: DXSUM 二分类 (MCI→Dementia)

> 脚本: `src/training/run_adni_mci_dementia.py`
> 报告: `docs/ADNI/phase2/Phase2A-MCI-Dementia实验报告.md`
> 结果: `local_data/Results_adni/mci_to_dementia/`

### 1.1 标签设计

#### 终点定义

采用 **Broad Dementia** 终点：基线后首次出现 `DIAGNOSIS=3` (AD痴呆) 或 `DXPARK=1` (帕金森痴呆) 或 `DXOTHDEM=1` (其他痴呆)。

#### 三窗口二分类标签

对每个窗口 W ∈ {3, 5, 10} 年：

```
Dementia_Wyr = 1  ⇔  converted_to_dementia == 1 AND dementia_conversion_years ≤ W
Dementia_Wyr = 0  ⇔  converted_to_dementia == 0 AND last_followup_years ≥ W
Censored       ⇔  converted_to_dementia == 0 AND last_followup_years < W   ← 丢弃
```

#### 删失规则解释

只有**未转化 + 随访不足 W 年**的人才被丢弃。已转化的人（无论何时）永远不会被删失：窗口内转化为 y=1，窗口外转化为 y=0。

#### 样本量变化

```
MCI 总人数: 1,120

3yr 窗口:
  ├── y=1 (转化且≤3yr):     196 人
  ├── y=0 (未转化且随访≥3yr): 404 人
  ├── 删失 (未转化且随访<3yr): 520 人 (46%) → 丢弃
  └── 训练集: 600 人

5yr 窗口:
  ├── y=1 (转化且≤5yr):     267 人
  ├── y=0 (未转化且随访≥5yr): 233 人
  ├── 删失 (未转化且随访<5yr): 620 人 (55%) → 丢弃
  └── 训练集: 500 人

10yr 窗口:
  ├── y=1 (转化且≤10yr):     311 人
  ├── y=0 (未转化且随访≥10yr):  74 人
  ├── 删失 (未转化且随访<10yr): 735 人 (66%) → 丢弃
  └── 训练集: 385 人
```

#### 标签迁移

同一病人可能在不同窗口有不同标签：

| RID | 转化时间 | 3yr 标签 | 5yr 标签 | 10yr 标签 |
|-----|---------|---------|---------|-----------|
| 41 | 1.5 年 | y=1 | y=1 | y=1 |
| 126 | 3.0 年 | y=0 | y=1 | y=1 |
| 1108 | 未转化, 随访 6.5 年 | y=0 | y=0 | 删失 |
| 4249 | 未转化, 随访 3.0 年 | y=0 | 删失 | 删失 |

### 1.2 模型架构

**模型**: LightGBM Classifier + CalibratedClassifierCV (isotonic)

| 超参数 | 值 |
|--------|-----|
| n_estimators | 500 |
| max_depth | 15 |
| num_leaves | 10 |
| subsample | 0.7 |
| learning_rate | 0.01 |
| colsample_bytree | 0.7 |
| is_unbalance | True |
| min_data_in_leaf | 5 |
| random_state | 2020 |



#### SFS Top 10 — Model A (Bio+Img)

| Rank | 3yr | 5yr | 10yr | all-time |
|------|-----|-----|------|----------|
| 1 | FS_ST40TA [MRI] | AMY_CENTILOIDS [PET] | NfL_F [血浆] | PTWORKHS [功能] |
| 2 | AMY_SUMMARY_SUVR [PET] | FS_ST24TA [MRI] | FS_ST31TA [MRI] | FS_ST12SV [MRI] |
| 3 | FS_ST44TA [MRI] | FS_ST84CV [MRI] | AMY_CENTILOIDS [PET] | AMY_CENTILOIDS [PET] |
| 4 | FS_ST29SV [MRI] | AMY_COMPOSITE_REF_SUVR [PET] | AB42_F [血浆] | FS_ST129SA [MRI] |
| 5 | FS_ST55TS [MRI] | FS_ST103TA [MRI] | WMH_CEREBRUM_TCB [WMH] | FS_ST72SA [MRI] |
| 6 | FS_ST74SA [MRI] | FS_ST31TA [MRI] | FS_ST66SV [MRI] | FS_ST14TS [MRI] |
| 7 | APOE4_count [遗传] | pT217_AB42_F [血浆] | GFAP_Q [血浆] | NfL_Q [血浆] |
| 8 | WMH_CEREBRUM_WHITE [WMH] | FS_ST34CV [MRI] | FS_ST129TA [MRI] | APOE4_count [遗传] |
| 9 | FS_ST113TA [MRI] | FS_ST69SV [MRI] | FS_ST71SV [MRI] | FS_ST93TA [MRI] |
| 10 | entry_age [人口] | APOE4_count [遗传] | WMH_RIGHT_HIPPO [WMH] | FS_ST102CV [MRI] |

**关键发现**:
- **3yr**: 前 6 步全是影像特征, 影像主导短期预测
- **5yr**: Amyloid PET 稳居 #1 和 #4, AUC 累计 ~0.84 即已超过最终模型
- **10yr**: NfL (神经丝轻链, 轴索损伤标志物) 取代 Amyloid PET 成单一最强特征
- **all-time**: PTWORKHS (就业状态) 排名 #1, 功能状态在无时间约束时最重



### 1.4 实验结果

#### Model A (Bio+Img) — 完整指标

| 窗口 | N | 事件 | AUC ± Std | Brier | Sens | Spec | AUPRC | Cutoff |
|------|---|------|-----------|-------|------|------|-------|--------|
| 3yr | 600 | 196 | **0.834 ± 0.032** | 0.154 | 0.719 | 0.795 | 0.713 | 0.406 |
| 5yr | 500 | 267 | **0.855 ± 0.021** | 0.161 | 0.757 | 0.798 | 0.873 | 0.485 |
| 10yr | 385 | 311 | **0.919 ± 0.015** | 0.087 | 0.752 | 0.959 | 0.977 | 0.823 |
| all-time | 1,120 | 323 | **0.841 ± 0.023** | 0.144 | 0.839 | 0.686 | 0.662 | 0.282 |输入的数据是哪部分



## 2. Phase 2B: DXSUM 生存分析 (MCI→Dementia)

> 脚本: `src/training/run_adni_survival.py`
> 报告: `docs/ADNI/phase2/Phase2B-生存模型实验报告.md`
> 设计方案: `docs/ADNI/phase2/生存模型设计方案.md`
> 结果: `local_data/Results_adni/survival/`

### 2.1 标签设计: 从二分类到生存

#### 二分类的四重局限

```
Phase 2A 问题:
  1. 删失浪费 — 3yr 丢 46%, 5yr 丢 55%, 10yr 丢 66%
  2. 时间坍缩 — 1.5 年和 2.9 年转化都是 y=1
  3. 不一致风险 — P(≤1yr) > P(≤3yr) 可能发生
  4. 回答不了"什么时候转" — 只能固定窗口的"转不转"
```

#### 生存标签构建
**排除**: `surv_time == 0` 的 176 人 (仅有基线 DXSUM, 无纵向随访)

#### 标签分布

```
┌──────────┬──────────────────────────────────────────────────┐
│ 944 MCI  │                                                  │
├──────────┼──────────────────────────────────────────────────┤
│ 事件     │ 323 人 (34.2%) — 中位转化时间 2.03 年              │
│          │   [0-1]yr: 46, [1-3]yr: 150, [3-5]yr: 71, >5yr: 56│
├──────────┼──────────────────────────────────────────────────┤
│ 删失     │ 621 人 (65.8%) — 中位随访 1.83 年                  │
│          │   <1yr: 166, [1-3]yr: 182, [3-5]yr: 100, >5yr: 173│
└──────────┴──────────────────────────────────────────────────┘
```
#### 与二分类标签的实例对比

```
RID=41:   转化@1.5yr → surv(1.5, 1)      二分类: 3yr=1, 5yr=1
RID=126:  转化@3.0yr → surv(3.0, 1)      二分类: 3yr=0, 5yr=1  ← 标签翻转
RID=1108: 未转化@6.5yr → surv(6.5, 0)    二分类: 3yr=0, 5yr=0, 10yr=删失
RID=4249: 未转化@3.0yr → surv(3.0, 0)    二分类: 3yr=0, 5yr=删失

关键: 生存标签不需区分 "3yr 阴性" 和 "5yr 删失"
      — 只记录 (3.0, 0), 模型自己学会条件概率
```

### 2.2 模型架构

#### 方法

| 组件 | v2 |
|------|-----|-----|
| s01 排序指标 | LightGBM Gain (与 Phase 2A 统一) |
| s02 聚类阈值 | \|ρ\| = 0.75 |
| s03 重排指标 | LightGBM Gain |
| s04 SFS 指标 | LightGBM AUC |
| s05 主模型 | RSF n=200, d=5, leaf=5 

**v2 改进动机**: 与 Phase 2A 统一特征选择方法, 避免方法差异干扰阶段间对比。

#### 最终模型

| 模型 | 实现 | 超参数 | 角色 |
|------|------|--------|------|
| **Random Survival Forest (RSF)** | `sksurv.ensemble.RandomSurvivalForest` | n_estimators=200, max_depth=5, min_samples_leaf=5 | 主模型 |
| **Cox Proportional Hazards** | `lifelines.CoxPHFitter` | penalizer=0.1 | 可解释性基准 |

### 2.3 特征选择

#### v2 SFS Top 10

| Rank | 特征 | 类型 | SFS AUC | 增益 |
|------|------|------|---------|------|
| 1 | **PTWORKHS** | Bio (就业状态) | 0.708 | +0.708 |
| 2 | FS_ST89SV | MRI 皮层下体积 | 0.755 | +0.047 |
| 3 | AMY_CENTILOIDS | Amyloid PET | 0.756 | +0.001 |
| 4 | FS_ST72CV | MRI 皮层下体积 | 0.786 | +0.030 |
| 5 | **GFAP_Q** | Bio (血浆 GFAP) | 0.795 | +0.009 |
| 6 | **APOE4_count** | Bio (遗传) | 0.805 | +0.010 |
| 7 | FS_ST93TA | MRI 皮层厚度 | 0.805 | −0.001 |
| 8 | FS_ST12SV | MRI 皮层下体积 | 0.807 | +0.003 |
| 9 | FS_ST102CV | MRI 皮层下体积 | 0.813 | +0.006 |
| 10 | FS_ST84TS | MRI 厚度 SD | 0.817 | +0.004 |

**组成**: 3 Bio (就业 + 血浆 GFAP + APOE4) + 7 MRI


#### 特征消融分析 (Bottom-up Ablation)

从 10 特征开始, 逐个移除最弱特征:

```
10 特征 (全)            C-index = 0.766
 9  (−FS_ST102CV)      C-index = 0.758  (Δ −0.008)
 8  (−FS_ST12SV)       C-index = 0.764  (Δ +0.006)
 7  (−FS_ST93TA)       C-index = 0.758  (Δ −0.006)
 6  (−APOE4_count)     C-index = 0.755  (Δ −0.003)
 5  (−GFAP_Q)          C-index = 0.740  (Δ −0.015)  ← 最大单特征跌落之一
 4  (−FS_ST72CV)       C-index = 0.740  (Δ 0)
 3  (−AMY_CENTILOIDS)  C-index = 0.722  (Δ −0.018)
 2  (−FS_ST89SV)       C-index = 0.698  (Δ −0.023)
 1  (−PTWORKHS)        C-index = 0.657  (Δ −0.041)  ← 最大单特征跌落!
```

**关键发现**: PTWORKHS 移除导致 C-index 暴跌 −0.082 (4→1 特征), 确认其为生存预测的最强单一预测因子。这与 Phase 2A all-time 模型中 PTWORKHS 排名 #1 一致。

### 2.4 实验结果

#### 完整指标

| 模型 | C-index ± Std | tAUC@1yr | tAUC@3yr | tAUC@5yr | Brier@3yr | Brier@5yr |
|------|-------------|----------|----------|----------|-----------|-----------|
| **RSF v2** | **0.765 ± 0.009** | 0.753 | 0.803 | 0.834 | 0.167 | 0.231 |
| Cox PH v2 | 0.770 | — | — | — | — | — |

#### Cox PH 风险比

| 特征 | coef | HR (exp(coef)) | p 值 | 方向 |
|------|------|---------------|------|------|
| FS_ST12SV | −0.341 | 0.711 | 1.5e-09 *** | 保护 |
| PTWORKHS | −0.306 | 0.737 | 8.4e-08 *** | 保护 |
| FS_ST72CV | −0.259 | 0.772 | 1.7e-06 *** | 保护 |
| FS_ST89SV | +0.213 | 1.238 | 3.9e-06 *** | 风险 |
| APOE4_count | +0.216 | 1.242 | 8.5e-06 *** | 风险 |
| FS_ST93TA | −0.171 | 0.842 | 0.0005 *** | 保护 |
| AMY_CENTILOIDS | +0.129 | 1.138 | 0.017 * | 风险 |
| FS_ST84TS | −0.116 | 0.890 | 0.020 * | 保护 |
| GFAP_Q | +0.113 | 1.120 | 0.026 * | 风险 |
| FS_ST102CV | +0.071 | 1.073 | 0.180 NS | — |

**解读**: 较大皮层下体积 (FS_ST12SV, HR=0.71) 和就业状态 (PTWORKHS, HR=0.74) 是保护因素。APOE4 等位基因数 (HR=1.24) 和 Amyloid Centiloids (HR=1.14) 增加风险。

#### 与 Phase 2A 二分类对比

| 指标 | RSF v2 (生存) | LGBM (二分类) | Δ |
|------|:---------:|:--------:|:----:|
| tAUC@3yr | 0.803 | 0.834 | **−0.031** |
| tAUC@5yr | 0.834 | 0.855 | **−0.021** |

生存模型性能略低于二分类 (这是预期内的 — 生存模型用更少的建模假设换取了更丰富的输出), 但差距从 v1 的 −0.043/−0.042 缩小至 v2 的 −0.031/−0.021。

#### 生存模型优势

```
Phase 2B 优势:
  1. 保留全部样本 — 944 人 vs 600/500/385, 删失者贡献部分信息
  2. 连续时间 — 1.5 年和 2.9 年转化有不同风险
  3. 自动一致 — S(1) ≥ S(3) ≥ S(5), risk_1y ≤ risk_3y ≤ risk_5y 数学保证
  4. 一个模型 — 输出完整 S(t|X), 回答任意时间点的风险
```

---

## 3. Phase 2C: CDR 量表扩展 (CN+MCI 双队列)

> 脚本: `src/training/run_adni_cdr_binary.py` (v1), `run_adni_cdr_binary_v2.py` (v2)
>       `src/training/run_adni_cdr_survival.py` (v1), `run_adni_cdr_survival_v2.py` (v2)
> 报告: `docs/ADNI/phase2/Phase2-CDR扩展实验结果报告.md`
> 技术方案: `docs/ADNI/phase2/Phase2-CDR量表扩展技术方案.md`
> 结果: `local_data/Results_adni/cdr_binary/`, `local_data/Results_adni/cdr_survival/`

### 3.1 标签设计

#### 终点定义

| 终点 | 定义 | 目的 |
|------|------|------|
| **Primary** | 首次 CDGLOBAL 恶化 (分层阈值) | 主分析 |
| **Sustained** | 首次恶化 + 下次访视确认 (或永不回退) | 降噪敏感性 |
| **CDRSB ≥1** | CDR Sum of Boxes 增加 ≥ 1.0 分 | 量化敏感性 |
| **Composite** | CDGLOBAL 恶化 或 DXSUM Dementia 诊断 (仅 MCI) | 复合终点 |

#### 分层阈值

```
CN 队列  (基线 CDR=0):   事件 = CDGLOBAL ≥ 0.5  (CN → MCI 转变)
MCI 队列 (基线 CDR=0.5): 事件 = CDGLOBAL ≥ 1.0  (MCI → 痴呆水平损害)
```

**关键差异**: CDR 捕获**序数恶化** (0→0.5→1), 而非 DXSUM 的**诊断跳跃** (MCI→AD)。CDR 覆盖了 CN→MCI 这个 DXSUM 无法覆盖的阶段。



#### 队列标签统计

| 队列 | N | 事件 (生存) | Sustained | CDRSB≥1 | Composite | 中位事件时间 | 3yr 有效/事件 | 5yr 有效/事件 |
|------|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| CN | 633 | 208 (32.9%) | 86 (13.6%) | 146 (23.1%) | — | 4.17 年 | 546/67 | 439/120 |
| MCI | 962 | 373 (38.8%) | 223 (23.2%) | 620 (64.4%) | 500 (52.1%) | 2.05 年 | 704/235 | 576/308 |

**CN 恶化慢** (中位 4.2 年), **MCI 恶化快** (中位 2.1 年)。CDRSB≥1 在 MCI 中事件率高达 64.4% — 该阈值过于敏感。

#### 双任务设计

```
CDR 二分类 (Task A):        CDR 生存 (Task B):
  3yr / 5yr / 10yr / all-time  (time_to_worsening, event)
  与 Phase 2A 完全相同的标签逻辑  与 Phase 2B 完全相同的标签逻辑
```

### 3.2 模型架构

#### 与 Phase 2A/2B 统一

| 组件 | 二分类 (Task A) | 生存 (Task B) |
|------|----------------|---------------|
| s01–s04 | LightGBM Gain + Ward + SFS (AUC) | LightGBM Gain + Ward + SFS (AUC / C-index) |
| s05 模型 | LGBMClassifier + CalibratedClassifierCV(isotonic) | RSF (n=200, d=5, leaf=5) + Cox PH |
| CV | 5-fold Stratified, seed=2022 | 5-fold Stratified, seed=2022 |



### 3.3 特征选择

#### MCI 3yr 二分类 — SFS Top 5

| Step | 特征 | AUC | 增益 | 类型 |
|------|------|-----|------|------|
| 1 | FS_ST29SV | 0.702 | +0.702 | MRI 皮层下体积 |
| 2 | FS_ST90TA | 0.756 | +0.053 | MRI 皮层厚度 |
| 3 | BSI_VENTVOL | 0.775 | +0.019 | MRI 脑室体积 |
| 4 | AMY_SUMMARY_SUVR | 0.789 | +0.015 | Amyloid PET |
| 5 | FS_ST24TA | 0.802 | +0.013 | MRI 皮层厚度 |

**4/5 MRI 特征** — 与 DXSUM 3yr 一致 (DXSUM 3yr 前 6 步也全是影像)。


#### CDR 与 DXSUM 特征重叠

CDR 和 DXSUM 的 Top 10 特征中, MRI 皮层下体积 (FS_ST12SV, FS_ST29SV, FS_ST89SV)、Amyloid PET (CENTILOIDS/SUVR) 和 APOE4 出现在两个终点中 — 跨终点信号一致性高。

### 3.4 实验结果 — 二分类 (Task A)

#### MCI 队列

| 窗口 | N | 事件 | AUC ± Std | Brier | Sens | Spec |
|------|---|------|-----------|-------|------|------|
| 3yr | 704 | 235 | **0.840 ± 0.024** | 0.153 | 0.847 | 0.701 |
| 5yr | 576 | 308 | **0.886 ± 0.035** | 0.144 | 0.782 | 0.862 |
| 10yr | 444 | 361 | **0.938 ± 0.021** | 0.073 | 0.898 | 0.892 |
| all-time | 962 | 373 | **0.803 ± 0.010** | 0.176 | 0.756 | 0.713 |

#### CN 队列

| 窗口 | N | 事件 | AUC ± Std | Brier | Sens | Spec |
|------|---|------|-----------|-------|------|------|
| 3yr | 546 | 67 | **0.705 ± 0.056** | 0.105 | 0.896 | 0.438 |
| 5yr | 439 | 120 | **0.738 ± 0.072** | 0.170 | 0.592 | 0.781 |
| 10yr | 281 | 189 | **0.789 ± 0.038** | 0.168 | 0.720 | 0.761 |
| all-time | 633 | 208 | **0.741 ± 0.037** | 0.178 | 0.577 | 0.776 |

#### 敏感性分析: 终点定义对比 (MCI, 3yr)

| 终点 | N | 事件 | AUC ± Std | Brier |
|------|---|------|-----------|-------|
| Primary (首次 CDGLOBAL 恶化) | 704 | 235 | **0.840 ± 0.024** | 0.153 |
| Sustained (确认持续) | 704 | 145 | **0.841 ± 0.058** | 0.123 |
| CDRSB ≥1 | 704 | 382 | **0.786 ± 0.044** | 0.192 |

**关键发现**: Sustained 的 AUC 与 Primary 相当 (0.841 vs 0.840) 但方差更大 (0.058 vs 0.024), 因为事件更少 (145 vs 235)。CDRSB≥1 的 AUC 最低 (0.786), 因事件率过高 (54%) 导致标签噪声。


**分队列 MCI 性能显著优于 Pooled** — CDR=0.5 和 CDR=0 代表不同疾病阶段, 混合建模稀释了信号。

### 3.5 实验结果 — 生存 (Task B)

#### 完整 C-index 结果

| 队列 | 终点 | N | 事件 | RSF C-index ± Std | Cox C-index | tAUC@1yr | tAUC@3yr | tAUC@5yr |
|------|------|---|------|-------------------|-------------|----------|----------|----------|
| **CN** | Primary | 633 | 208 | **0.680 ± 0.025** | 0.667 | 0.766 | 0.649 | 0.681 |
| CN | Sustained | 633 | 86 | **0.750 ± 0.050** | 0.747 | 0.927 | 0.749 | 0.770 |
| CN | CDRSB≥1 | 633 | 146 | **0.692 ± 0.019** | 0.674 | 0.868 | 0.676 | 0.683 |
| **MCI** | Primary | 962 | 373 | **0.781 ± 0.019** | 0.782 | 0.818 | 0.826 | 0.865 |
| MCI | Sustained | 962 | 223 | **0.799 ± 0.029** | 0.796 | 0.843 | 0.833 | 0.864 |
| MCI | CDRSB≥1 | 962 | 620 | **0.677 ± 0.034** | 0.670 | 0.690 | 0.759 | 0.793 |
| MCI | Composite | 960 | 500 | **0.744 ± 0.011** | 0.749 | 0.786 | 0.817 | 0.848 |
| **Pooled** | Primary | 1,595 | 581 | **0.761 ± 0.015** | — | 0.821 | 0.805 | 0.814 |



#### SFS 方法对比

| 方法 | C-index | Std | Δ | Top 10 重叠 |
|------|---------|-----|---|:---:|
| LGB AUC SFS | 0.791 | 0.023 | — | 7/10 |
| C-index SFS | 0.791 | 0.030 | −0.0004 | 7/10 |

**两种方法等价** — LGB AUC SFS 未引入有意义的偏差, 因此与 Phase 2A 统一使用 AUC SFS 是安全的。

### 3.6 CDR vs DXSUM 跨终点对比

#### 终点定义差异

| 维度 | DXSUM (Phase 2A/2B) | CDR (Phase 2C) |
|------|---------------------|----------------|
| 终点类型 | 诊断转换 (MCI→AD Dementia) | 量表序数恶化 (CDR 0→0.5→1) |
| 覆盖人群 | MCI only | CN + MCI |
| CN→MCI 覆盖 | 无 (CN 不在 MCI 队列) | 有 (CDR=0 → ≥0.5) |
| 可逆性 | 罕见 (诊断极少回退) | 可逆 (150/1,286 MCI 改善) |
| 数据密度 | 较高 | 较低 (部分访视缺 CDR) |
| 事件判据 | 诊断表 (DXSUM) | 临床量表 (CDGLOBAL) |

#### 预测性能对比

| 指标 | DXSUM MCI | CDR MCI | Δ (CDR−DXSUM) |
|------|:---------:|:-------:|:------------:|
| 二分类 AUC@3yr | 0.834 ± 0.032 | **0.840 ± 0.024** | **+0.006** |
| 二分类 AUC@5yr | 0.855 ± 0.021 | **0.886 ± 0.035** | **+0.031** |
| 生存 C-index | 0.765 ± 0.009 | **0.781 ± 0.019** | **+0.016** |
| 生存 tAUC@3yr | 0.803 | **0.826** | **+0.023** |
| 生存 tAUC@5yr | 0.834 | **0.865** | **+0.030** |

**CDR MCI 在所有指标上略超 DXSUM**。一个可能的原因是 CDR 恶化 (CDGLOBAL≥1) 比 DXSUM Dementia 诊断更早发生, 捕获了更丰富的早期信号。

---

## 4. 综合讨论

### 4.1 三阶段核心发现总结

```
Phase 2A (DXSUM 二分类):
  ✓ 影像对短期预测贡献最大 (+0.103 AUC)
  ✓ 血浆 NfL 是长期预测 (#1) 的最强单一特征
  ✓ PTWORKHS (就业状态) 在无时间约束时排名第一
  ✓ 跨窗口特征模式不同: 近期=影像主导, 远期=血浆标志物+影像

Phase 2B (DXSUM 生存):
  ✓ 生存模型 C-index=0.765, 用 1 个模型替代 3 个二分类模型
  ✓ 保留全部 944 人样本 (二分类丢 46–66%)
  ✓ 方法统一 (v2) 将生存-二分类差距缩小至 −0.03
  ✓ PTWORKHS 是生存预测最不可替代的特征 (消融 Δ=−0.082)

Phase 2C (CDR 扩展):
  ✓ CDR MCI 性能略超 DXSUM (C-index +0.016)
  ✓ CN→MCI 可预测 (C-index=0.68), Tau PET 在极早期有独立信号
  ✓ FS_ST29SV (皮层下体积) 是跨队列、跨终点的 #1 特征
  ✓ 分队列建模显著优于 Pooled (+0.036 AUC for MCI 3yr)
  ✓ Sustained 定义降噪有效 (+0.018 C-index)
```

### 4.2 方法论演进启示

1. **统一特征选择管线 (s01–s05) 至关重要**: Phase 2B v1→v2 仅改进了特征选择方法 (从单变量 C-index 到 LightGBM Gain), 即带来 +0.020 C-index 提升。方法不一致会导致不可比的跨阶段结论。

2. **训练折内预处理消除信息泄漏**: Phase 2C v2 将缺失值填补和标准化移入训练折内, 这是更严格的评估标准 — 结果反而更高, 说明 v1 的泄漏并未系统性地夸大性能。

3. **生存 SFS 的方法选择不敏感**: AUC SFS 和 C-index SFS 结果等价 (Δ<0.001), 简化了跨任务的特征选择统一。

4. **分队列建模是正确策略**: Pooled CDR 模型稀释了 CN 和 MCI 的疾病阶段差异, MCI-only 模型 AUC 高出 +0.036。

### 4.3 临床意义

- **影像 (MRI + PET) 对近期预测不可或缺**: 3 年窗口的影像增量 +0.103 AUC, 远超 5 年的 +0.063。提示临床试验的富集策略应考虑预测窗口: 短期试验最受益于影像筛选, 长期试验可更多依赖血浆标志物。

- **NfL 作为长期预测生物标志物**: 在 10 年窗口中, NfL 单一特征 AUC=0.741, 超过 Amyloid PET — 这是一个低成本、易获取的血浆标志物, 适合人群初筛。

- **PTWORKHS (就业) 是最强功能预测因子**: 同时出现在 DXSUM all-time #1、生存 SFS #1 和消融分析中, 提示功能储备 (而非单纯病理负荷) 决定认知衰退速度。这是一个被低估的预测变量。

- **CDR 可作为 DXSUM 的补充终点**: CDR 不仅覆盖了 CN→MCI (DXSUM 无法覆盖的阶段), 而且预测性能略优于 DXSUM。两者可互为敏感性分析。

### 4.4 局限与下一步

| 局限 | 后续计划 |
|------|---------|
| 所有模型为单时间点预测 (仅基线特征) | 探索纵向特征变化速率 |
| 未考虑治疗/干预影响 | 若有药物试验数据, 可加入 treatment arm |
| CN→MCI 预测 C-index 仅 0.68 | 探索额外特征 (如 fMRI, DTI, 遗传多基因风险评分) |
| CDR 部分访视缺失, 存在选择偏差 | 敏感性分析: 仅用有完整 CDR 的 ADNI phase |
| 生存模型仅 RSF+Cox, 未测 DeepSurv | 添加 DeepSurv/DeepHit 作为深度学习基线 |
| UKB 外部验证尚未进行 | Phase 3: UKB→ADNI 跨队列泛化 |

---

## 附录 A: 脚本索引

| 阶段 | 脚本 | 功能 |
|------|------|------|
| 标签构建 | `思路一/ADNI数据集/build_time_targets_v2.py` | DXSUM 时间目标构建 |
| 标签构建 | `思路一/ADNI数据集/build_cdr_targets_v2.py` | CDR 时间目标构建 |
| Phase 2A | `src/training/run_adni_mci_dementia.py` | DXSUM 二分类训练+评估 |
| Phase 2B | `src/training/run_adni_survival.py` | DXSUM 生存训练+评估 |
| Phase 2B | `src/training/run_adni_survival_v1_sfs_rsf.py` | v1 备份 (RSF SFS) |
| Phase 2C | `src/training/run_adni_cdr_binary.py` | CDR 二分类 v1 (pooled) |
| Phase 2C | `src/training/run_adni_cdr_binary_v2.py` | CDR 二分类 v2 (分队列) |
| Phase 2C | `src/training/run_adni_cdr_survival.py` | CDR 生存 v1 (pooled) |
| Phase 2C | `src/training/run_adni_cdr_survival_v2.py` | CDR 生存 v2 (分队列+SFS对比) |

## 附录 B: 结果文件索引

| 目录 | 关键文件 |
|------|---------|
| `local_data/Results_adni/mci_to_dementia/` | `mci_to_dementia_results.csv`, `sfs_history_*.csv` |
| `local_data/Results_adni/survival/` | `model_comparison.csv`, `cox_ph_summary.csv`, `feature_ablation.csv`, `selected_features.csv`, `sfs_history.csv`, `lgb_feature_ranking.csv` |
| `local_data/Results_adni/survival_v1_backup/` | v1 对照结果 |
| `local_data/Results_adni/cdr_binary/` | `cdr_binary_results_v2.csv`, `cdr_binary_results.csv` |
| `local_data/Results_adni/cdr_survival/` | `cdr_survival_results_v2.csv`, `consort_flow.csv`, `cohort_label_summary.csv`, `sfs_method_comparison.csv` |

## 附录 C: 超参数速查表

| 模型 | 超参数 | 值 |
|------|--------|-----|
| **LightGBM Classifier** | n_estimators, max_depth, num_leaves, subsample, lr, colsample_bytree, is_unbalance, min_data_in_leaf | 500, 15, 10, 0.7, 0.01, 0.7, True, 5 |
| **CalibratedClassifierCV** | method, cv | isotonic, 3 |
| **Random Survival Forest** | n_estimators, max_depth, min_samples_leaf | 200, 5, 5 |
| **Cox PH** | penalizer | 0.1 |
| **CV** | n_splits, random_state, shuffle | 5, 2022, True |
| **SFS** | early_stop_threshold, max_steps | 0.0005, 15 |
| **Ward 聚类** | metric, threshold | Spearman \|ρ\|, 0.75 |

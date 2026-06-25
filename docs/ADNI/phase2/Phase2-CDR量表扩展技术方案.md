# Phase 2 CDR 量表扩展 — 技术方案报告

> 日期: 2026-06-15 (v2 修订)
> 状态: ✅ v2 实验已完成，结果见 `Phase2-CDR扩展实验结果报告.md`
> 基于: Phase 2A (MCI→Dementia 二分类) + Phase 2B v2 (MCI→Dementia 生存模型, 统一 LightGBM)
> 扩展方向: 以 CDR 量表替代诊断标签作为结局变量
> 约束: 不改模型架构，只改任务定义 + 数据处理
>
> **v2 修订要点:**
> - 统一 index_date: 首次满足纳入条件的 CDR 访视日期
> - CN (CDR=0) 和 MCI (CDR=0.5) 分队列作为主分析
> - CONSORT/STROBE 样本流图, index_date 与特征基线日期对齐检查
> - 持续性恶化 + CDR-SB 恶化 + 复合终点 敏感性分析
> - 训练折内缺失值填补和标准化（修复信息泄漏）
> - 生存感知特征选择 vs LGB-AUC 特征选择 敏感性对比

---

## 1. 动机与背景

### 1.1 为什么需要 CDR 作为结局

Phase 2A/2B 使用 DXSUM 诊断表（DIAGNOSIS=3 → AD Dementia）作为终点，存在以下局限：

| DXSUM 诊断终点 | CDR 评分终点 |
|:---|:---|
| 离散类别：MCI vs AD Dementia，二元跳变 | 有序等级：0 → 0.5 → 1 → 2 → 3，渐进恶化 |
| 依赖临床医生的定性判断 | 半定量评分，6 个认知域 + 全局评分 |
| MCI→AD 转化漏掉 MCI 内部的严重度变化 | CDR-SB (Sum of Boxes) 可检测连续恶化 |
| 不覆盖从 CN→MCI 的早期转变 | CDR=0→0.5 捕获临床前→前驱期转变 |
| ADNI 中仅 323/1120 MCI 转化 (29%) | 477/2161 CDR=0.5 恶化 (22%)，298/1746 CDR=0 恶化 (17%) |

**核心优势**:
- CDR 是 AD 临床试验的标准主要终点（FDA/EMA 认可）
- CDR-SB 的连续变化可用于疾病修饰疗法的疗效评估
- 从"是否会痴呆"变成"认知衰退有多快/多严重"
- 允许将 CN 人群纳入预测范围（CDR=0 → 0.5 = incident MCI）

### 1.2 扩展的定位

```
Phase 2A (DXSUM)          Phase 2B (DXSUM)          本扩展 (CDR)
──────────────────        ──────────────────         ──────────────────
MCI→Dementia 二分类       MCI→Dementia 生存          CDR 恶化 二分类+生存
标签: 0/1 (转化/未转化)   标签: (time, event)        标签: (time, CDR worsen)
样本: 仅 MCI, 排除删失    样本: 仅 MCI, 保留删失      样本: CN+MCI, 保留删失
主要指标: AUC            主要指标: C-index           主要指标: AUC + C-index
```

三个实验可互相验证：如果在排除认知测试的前提下，同样的特征体系能同时预测诊断转化 **和** CDR 恶化，则信号是稳健的。

---

## 2. 数据基础

### 2.1 CDR 纵向数据概况

```
来源: All_Subjects_CDR_20May2026.csv
总行数: 14,766 (全部访视)
总受试者: 4,320
中位访视次数: 3 次/人 (范围 1-12)
与 DXSUM 重叠: 3,693 受试者 (85.5%)
```

**可用字段**:

| 字段 | 类型 | 范围 | 说明 |
|:---|:---|:---|:---|
| `CDGLOBAL` | 有序分类 | 0, 0.5, 1, 2, 3 | 全局 CDR 评分 |
| `CDRSB` | 连续 | 0–18 | CDR Sum of Boxes |
| `CDMEMORY` | 有序分类 | 0, 0.5, 1, 2, 3 | 记忆域 |
| `CDORIENT` | 有序分类 | 0, 0.5, 1, 2, 3 | 定向域 |
| `CDJUDGE` | 有序分类 | 0, 0.5, 1, 2, 3 | 判断力域 |
| `CDCOMMUN` | 有序分类 | 0, 0.5, 1, 2 | 社交域 |
| `CDHOME` | 有序分类 | 0, 0.5, 1, 2, 3 | 家务域 |
| `CDCARE` | 有序分类 | 0, 1, 2, 3 | 自理域 |
| `VISDATE` | 日期 | — | 访视日期，用于计算时间 |

### 2.2 CDGLOBAL 基线分布（≥2 次访视的 2,348 人）

```
Baseline CDGLOBAL   人数    恶化人数    恶化率    改善人数
──────────────────────────────────────────────────────
0 (正常)             854      221       25.9%       9
0.5 (MCI)          1,286      446       34.7%     150
1 (轻度痴呆)          206       84       40.8%      15
2 (中度痴呆)            2        0        0.0%       0
3 (重度痴呆)           —        —          —        —
──────────────────────────────────────────────────────
合计                2,348      751       32.0%     174
```

**关键观察**:
- CDR=0.5 (MCI) 受试者中 34.7% 出现 CDGLOBAL 恶化（≥1），与 DXSUM 的 29% 痴呆转化率接近但更高——CDR 捕捉到了 DXSUM 未标记的临床恶化
- CDR=0 受试者中 25.9% 恶化为 ≥0.5，代表从正常认知到 MCI 的转变——这是 DXSUM 终点无法覆盖的人群
- 150 名 CDR=0.5 受试者 CDGLOBAL 出现改善（→0），这反映了 MCI 诊断的不稳定性或临床波动——需要在标签设计中处理

### 2.3 恶化时间分布

```
                        基线 CDR=0.5 (MCI)         基线 CDR=0 (CN)
                        ──────────────────          ───────────────
事件数                   477                        298
中位恶化时间              2.1 年                     4.0 年
均值恶化时间              2.8 年                     4.5 年
≤1 年内恶化              86 (18.0%)                 36 (12.1%)
≤3 年内恶化             296 (62.1%)                122 (40.9%)
≤5 年内恶化             398 (83.4%)                185 (62.1%)
```

**关键观察**:
- CDR=0.5 受试者的恶化速度快于 CDR=0（中位 2.1yr vs 4.0yr）——符合 AD 连续谱的生物学预期
- 3 年窗口可捕获 62% 的 MCI 恶化和 41% 的 CN 恶化——是合理的"行动窗口"
- 5 年窗口可捕获 >80% 的 MCI 恶化——适合长期预后评估

### 2.4 CDRSB 变化分布（≥2 次访视，2,348 人）

```
CDRSB 增加阈值    人数      占比
────────────────────────────────
≥0.5            1,280     54.5%
≥1.0            1,034     44.0%
≥2.0              745     31.7%
≥3.0              548     23.3%
```

**CDRSB 提供比 CDGLOBAL 更细粒度的恶化检测。** 例如，CDRSB 从 1.0 升到 2.5（CDGLOBAL 仍为 0.5）的受试者被 CDGLOBAL 阈值法漏掉，但 CDRSB 阈值法能捕获。

---

## 3. 任务设计总览

### 3.1 两个任务

```
┌──────────────────────────────────────────────────────────────────────┐
│                    CDR 扩展 — 任务体系                                 │
├──────────────────┬───────────────────────┬────────────────────────────┤
│                  │ Task A: CDR 二分类     │ Task B: CDR 生存           │
├──────────────────┼───────────────────────┼────────────────────────────┤
│ 对标             │ Phase 2A               │ Phase 2B                   │
│ 标签类型          │ 二值 0/1               │ (time, event)              │
│ 结局定义          │ 固定窗口内CDR恶化超阈值  │ 首次 CDR 恶化的时间         │
│ 模型             │ LGBM + Isotonic 校准    │ RSF / Cox PH               │
│ 评估指标          │ AUC, Brier, Sens/Spec  │ C-index, tAUC, Brier      │
│ 管线             │ s01-s05 (同 Phase 2A)   │ s01-s05 (同 Phase 2B)     │
│ 特征体系          │ 358 特征 (排除认知+CDR)  │ 358 特征 (排除认知+CDR)    │
└──────────────────┴───────────────────────┴────────────────────────────┘
```

### 3.2 目标人群

```
Phase 2A/2B: 仅 MCI (baseline_diagnosis=2)
    ↓ 扩展
本方案: CN (CDGLOBAL=0) + MCI (CDGLOBAL=0.5)
    
排除: 基线已是痴呆 (CDGLOBAL≥1)，因天花板效应——CDR=1→2→3 的恶化与 CDR=0→0.5 或 0.5→1 的临床意义不同
```

**人群对比**:

| | Phase 2A/2B | 本扩展 (CDR) |
|:---|:---|:---|
| 基线人群 | MCI only (1,120 人) | CN + MCI (CDR 0 + 0.5) |
| 预期有效人数 | 944 (生存) | ~3,000 (CN+MCI, CDR纵向) |
| 排除 | baseline_diagnosis=3 (AD) | CDGLOBAL≥1 (已痴呆) |
| 新增覆盖 | — | CN→MCI 转变 (早期预警) |

---

## 4. Task A: CDR 恶化二分类

### 4.1 标签构建逻辑

**核心思路**: 与 Phase 2A 相同——在固定时间窗口内，CDR 恶化超过阈值 → y=1，未恶化且随访足够 → y=0，随访不足且未恶化 → 删失（排除）。

```
算法: build_cdr_worsening_labels(cdr_df, window_years, worsen_threshold)

对每个受试者:
  baseline_CDR = 首次访视的 CDGLOBAL
  
  if baseline_CDR ∉ {0, 0.5}:       # 排除非目标基线
      skip
  
  遍历随访访视 (按日期排序):
    years = (访视日期 - 基线日期) / 365.25
    if years < 0: continue
    
    更新 last_followup_years = max(last_followup_years, years)
    
    if 当前 CDGLOBAL > 基线 CDGLOBAL:     # ★ 首次恶化
      cdr_worsening_years = years
      break
  
  对每个时间窗口 W ∈ {3, 5, 10}:
    if cdr_worsening_years ≤ W:
      CDR_worsen_Wyr = 1   ← 事件
    elif last_followup_years ≥ W:
      CDR_worsen_Wyr = 0   ← 阴性 (确定观察了整个窗口)
    else:
      CDR_worsen_Wyr = NaN ← 删失 (随访不足，标签不确定)
```

### 4.2 恶化阈值的分层定义

不同基线 CDR 的恶化阈值需要不同定义，因为临床意义不同：

```
方案 A1 (推荐): 分层阈值
─────────────────────────────────────────────
基线 CDGLOBAL = 0:  恶化 = CDGLOBAL ≥ 0.5   (CN → MCI 或更差)
基线 CDGLOBAL = 0.5: 恶化 = CDGLOBAL ≥ 1     (MCI → 轻度痴呆或更差)

理由: 
- CDR=0→0.5 代表 incident MCI，是临床前AD的第一个可检测阶段
- CDR=0.5→1 代表 incident dementia，与 Phase 2A 的 DXSUM 终点可对比
- 分层后两组的事件率相近 (~25% vs ~35%)，避免类别不平衡的极端偏移
```

```
方案 A2 (备选): 统一阈值 (≥0.5 增量)
─────────────────────────────────────────────
基线 CDGLOBAL = 0:  恶化 = CDGLOBAL ≥ 0.5    
基线 CDGLOBAL = 0.5: 恶化 = CDGLOBAL ≥ 1.0   

等价于 A1，因为 ordinal scale 的最小增量就是 0.5
```

```
方案 A3 (备选): CDRSB 连续变化
─────────────────────────────────────────────
使用 CDRSB (Sum of Boxes) 替代 CDGLOBAL
阈值: CDRSB 增加 ≥ 1.0 或 ≥ 2.0

优势: 更细粒度，可检测 CDGLOBAL 未变的亚临床恶化
劣势: 阈值选择较主观，需要敏感性分析
```

**推荐 A1 作为主方案**，A3 作为敏感性分析。

### 4.3 预计样本量

#### 窗口 3yr

```
                    基线 CDR=0.5 (MCI)         基线 CDR=0 (CN)
                    ──────────────────          ──────────────
总人数 (≥2次访视)     1,286                       854
  事件 (≤3yr恶化)      296 (23.0%)                122 (14.3%)
  阴性 (≥3yr未恶化)    387 (30.1%)                430 (50.4%)
  删失 (<3yr未恶化)    603 (46.9%) ★              302 (35.4%)
                    ──────────────────          ──────────────
有效训练              683                         552
合计有效: 1,235 人 (vs Phase 2A 3yr 的 600 人)
```

#### 窗口 5yr

```
                    基线 CDR=0.5 (MCI)         基线 CDR=0 (CN)
                    ──────────────────          ──────────────
  事件 (≤5yr恶化)      398 (30.9%)                185 (21.7%)
  阴性 (≥5yr未恶化)    151 (11.7%)                250 (29.3%)
  删失 (<5yr未恶化)    737 (57.3%)                419 (49.1%)
                    ──────────────────          ──────────────
有效训练              549                         435
合计有效: 984 人 (vs Phase 2A 5yr 的 500 人)
```

**删失率高达 47-57%（CDR=0.5），主要原因是 ADNI 中大量受试者的 CDR 随访次数不足。** 这是 CDR 终点的固有限制——与 DXSUM（每次访视都记录诊断）不同，CDR 并非每次访视都完整评估。

### 4.4 二分类标签设计总结

```
输出列:
  CDR_worsen_3yr:  0/1/NaN  (1 = 3年内恶化)
  CDR_worsen_5yr:  0/1/NaN
  CDR_worsen_10yr: 0/1/NaN
  CDR_worsen_censored_3yr:  0/1
  CDR_worsen_censored_5yr:  0/1
  CDR_worsen_censored_10yr: 0/1
  CDR_worsening_years: float (首次恶化时间, NaN=未恶化)
  CDR_last_followup_years: float
```

### 4.5 与 Phase 2A 的对比验证

为了与 Phase 2A 结果可比，可以：
1. **对同一批 MCI 受试者**，同时用 DXSUM 标签 (Dementia_3yr) 和 CDR 标签 (CDR_worsen_3yr) 训练
2. **一致性分析**: 两者的标签重叠率、kappa 系数
3. **AUC 差异分析**: CDR 标签的 AUC 可能略低于 DXSUM 标签，因为 CDR 恶化包含了 DXSUM 未标记的"亚临床恶化"——这些受试者在 CDR 标签中是"阳性"但在 DXSUM 标签中是"阴性"

---

## 5. Task B: CDR 恶化时间预测（生存模型）

### 5.1 标签构建逻辑

**核心思路**: 与 Phase 2B 完全相同——将 `(time, event)` 标签从 DXSUM 诊断转化替换为 CDR 首次恶化。

```
surv_time = CDR_worsening_years    if CDR_worsened == 1
          = CDR_last_followup_years if CDR_worsened == 0

surv_event = 1 if CDR_worsened else 0
```

### 5.2 关键差异 vs Phase 2B

| 维度 | Phase 2B (DXSUM) | Task B (CDR) |
|:---|:---|:---|
| 事件定义 | MCI→AD/Parkinson's/Other Dementia | CDR 恶化 (分层阈值) |
| 事件数 (MCI) | 323/944 (34.2%) | ~477/1,286 (37.1%, ≥2次访视) |
| 人群 | 仅 MCI | CN + MCI |
| 删失规则 | 排除 time=0 (仅基线) | 排除 time=0 + 排除仅1次访视 |
| 中位事件时间 | 2.0 年 (MCI→Dementia) | 2.1 年 (CDR=0.5 恶化) / 4.0 年 (CDR=0 恶化) |
| 零随访排除数 | 176/1,120 (15.7%) | 需从 4,320 人中筛选 ≥2 次 CDR 访视者 |
| 预计有效人数 | 944 | ~2,200 (CN+MCI, ≥2次CDR访视) |

### 5.3 零随访/单次访视排除

与 Phase 2B 的 `time=0` 排除类似，CDR 生存分析需要排除以下病态标签：

```
排除条件:
  1. 仅 1 次 CDR 记录 → 无法计算纵向变化
  2. 基线 CDGLOBAL ≥ 1 (已是痴呆，天花板效应)
  3. CDR_last_followup_years = 0 (仅基线，无纵向信息)
  
排除后预计保留:
  - CDR=0.5: ~1,286 人 (≥2 次访视)
  - CDR=0:   ~854 人 (≥2 次访视)
  - 合计:    ~2,140 人
```

### 5.4 多终点设计（可选扩展）

CDR 生存模型可支持多个竞争终点：

```
终点 1 (CDR-worsen-0.5): CDGLOBAL 增加 ≥0.5 (主要终点)
  → 对基线 CDR=0: 恶化到 ≥0.5
  → 对基线 CDR=0.5: 恶化到 ≥1.0
  
终点 2 (CDR-worsen-1.0, CDRSB≥2): CDRSB 增加 ≥2.0 (次要终点)
  → 更显著的临床恶化，可能对应 FDA 临床试验终点

终点 3 (CDR-worsen-1.5): CDGLOBAL 增加 ≥1.0
  → 对基线 CDR=0: 恶化到 ≥1.0 (CN→痴呆)
  → 跨阶段恶化，事件更少但临床意义更明确
```

**推荐**: 主分析用终点 1，次要分析用终点 2（CDRSB）。

### 5.5 标签示例

```
RID    基线CDR   轨迹                           surv_time  surv_event
─────────────────────────────────────────────────────────────────────
41     0.5      0.5→0.5→1.0→1.0                 2.0        1 (恶化到1)
126    0.5      0.5→0.5→0.5→1.0                 3.5        1 (恶化到1)
1108   0.5      0.5→0.5→0.5→0.5 (6.5年)        6.5        0 (删失，未恶化)
4249   0        0→0→0→0.5                       4.0        1 (恶化到0.5)
5001   0        0→0→0 (2.0年, 失访)             2.0        0 (删失)
─────────────────────────────────────────────────────────────────────
```

与 Phase 2B 标签体系的对比：

```
RID=41:   DXSUM: (1.5, 1) MCI→AD       CDR: (2.0, 1) CDR恶化
          → CDR 恶化晚于诊断转化0.5年——CDR评分滞后于临床诊断

RID=4249: DXSUM: (3.0, 0) MCI未转化     CDR: (4.0, 1) CN→MCI
          → 不同终点！DXSUM 看不到此人变化，但 CDR 捕获了从正常到MCI的转变
```

---

## 6. 实现方案

### 6.1 新增脚本

```
src/training/
├── run_adni_cdr_binary.py       # Task A: CDR 二分类 (对应 run_adni_mci_dementia.py)
├── run_adni_cdr_survival.py     # Task B: CDR 生存   (对应 run_adni_survival.py)

思路一/ADNI数据集/
├── build_cdr_targets.py          # ★ 新增: 从 CDR 表构建 CDR 恶化标签
│                                  #   输入: All_Subjects_CDR_20May2026.csv
│                                  #   输出: CDR_time_targets.csv
│                                  #   (含 worsen_3yr/5yr/10yr, worsening_years,
│                                  #    last_followup_years, censoring flags)
```

### 6.2 `build_cdr_targets.py` 伪代码

```python
#!/usr/bin/env python3
"""
CDR Worsening Target Builder
=============================
从 All_Subjects_CDR_20May2026.csv 构建：
  1. CDR 恶化二分类标签 (3yr/5yr/10yr 窗口)
  2. CDR 恶化生存标签 (time, event)
  
与 build_time_targets_v2.py 并行：两者分别从 CDR 和 DXSUM 构建标签，
后续 merge 到同一个 baseline 特征矩阵上。
"""

import pandas as pd, numpy as np

# ── 1. Load CDR longitudinal data ──
cdr = pd.read_csv('All_Subjects_CDR_20May2026.csv')
cdr['VISDATE'] = pd.to_datetime(cdr['VISDATE'])
cdr = cdr.sort_values(['RID', 'VISDATE'])

# ── 2. Track CDR worsening for each subject ──
records = []
for rid, group in cdr.groupby('RID'):
    bl = group.iloc[0]
    bl_date = bl['VISDATE']
    bl_cdrg = bl['CDGLOBAL']
    bl_cdrsb = bl['CDRSB']
    
    rec = {
        'RID': rid,
        'baseline_CDGLOBAL': bl_cdrg,
        'baseline_CDRSB': bl_cdrsb,
        'CDR_worsening_years': np.nan,
        'CDR_worsened': 0,
        'CDR_worsened_CDRSB': 0,     # ≥2.0 increase
        'CDR_last_followup_years': 0.0,
        'n_cdr_visits': len(group),
    }
    
    # Skip subjects not in target population
    if bl_cdrg not in [0.0, 0.5]:
        rec['CDR_worsened'] = -1  # excluded
        records.append(rec)
        continue
    
    last_date = bl_date
    for i in range(1, len(group)):
        visit_cdrg = group['CDGLOBAL'].iloc[i]
        visit_cdrsb = group['CDRSB'].iloc[i]
        visit_date = group['VISDATE'].iloc[i]
        
        years = (visit_date - bl_date).days / 365.25
        if years < 0: continue
        
        last_date = max(last_date, visit_date)
        rec['CDR_last_followup_years'] = max(
            rec['CDR_last_followup_years'], years)
        
        # CDGLOBAL worsening (ordinal increase)
        if pd.notna(visit_cdrg) and visit_cdrg > bl_cdrg:
            if pd.isna(rec['CDR_worsening_years']):
                rec['CDR_worsening_years'] = years
                rec['CDR_worsened'] = 1
        
        # CDRSB worsening (≥2.0 increase)
        if pd.notna(visit_cdrsb) and pd.notna(bl_cdrsb):
            if (visit_cdrsb - bl_cdrsb) >= 2.0:
                if rec['CDR_worsened_CDRSB'] == 0:
                    rec['CDR_worsened_CDRSB'] = 1
    
    records.append(rec)

cdr_targets = pd.DataFrame(records)

# ── 3. Build time-window binary labels ──
for window in [3, 5, 10]:
    # CDGLOBAL-based
    cdr_targets[f'CDR_worsen_{window}yr'] = np.where(
        (cdr_targets['CDR_worsened'] == 1) & 
        (cdr_targets['CDR_worsening_years'] <= window), 1,
        np.where(
            (cdr_targets['CDR_worsened'] == 0) & 
            (cdr_targets['CDR_last_followup_years'] >= window), 0,
            np.nan  # censored
        )
    )
    
    cdr_targets[f'CDR_worsen_censored_{window}yr'] = (
        (cdr_targets['CDR_worsened'] == 0) & 
        (cdr_targets['CDR_last_followup_years'] < window)
    ).astype(int)

# ── 4. Survival labels ──
cdr_targets['CDR_surv_time'] = np.where(
    cdr_targets['CDR_worsened'] == 1,
    cdr_targets['CDR_worsening_years'],
    cdr_targets['CDR_last_followup_years']
)
cdr_targets['CDR_surv_event'] = cdr_targets['CDR_worsened']

# ── 5. Merge with existing baseline and save ──
# ... merge to ADNI_baseline_with_time_targets_v2.csv via RID
```

### 6.3 训练脚本的修改点

**相对于 `run_adni_mci_dementia.py` (Task A)**:

```diff
- # 终点定义: convert_to_dementia + dementia_conversion_years
+ # 终点定义: CDR_worsened + CDR_worsening_years

- TARGET_COLS 中新增:
+ 'CDR_worsen_3yr', 'CDR_worsen_5yr', 'CDR_worsen_10yr',
+ 'CDR_worsened', 'CDR_worsening_years', 'CDR_last_followup_years',
+ 'CDR_worsen_censored_3yr', 'CDR_worsen_censored_5yr', 'CDR_worsen_censored_10yr'

- # 人群筛选:
- mci = df[df['baseline_diagnosis']==2.0]
+ # 方案: CN + MCI (baseline CDGLOBAL ∈ {0, 0.5})
+ at_risk = df[df['baseline_CDGLOBAL'].isin([0.0, 0.5])]

- # CDR_COLS 仍然需要排除 (因为是结局变量，不能作为特征)
+ # CDR_COLS 排除不变 — CDR 是结局，不能泄漏到特征中
```

**相对于 `run_adni_survival.py` (Task B)**:

```diff
- surv_time  = dementia_conversion_years OR last_followup_years
- surv_event = converted_to_dementia
+ surv_time  = CDR_worsening_years OR CDR_last_followup_years
+ surv_event = CDR_worsened

- # 排除 time=0:
- mci_v = mci[mci['surv_time']>0]
+ # 排除 time=0 + n_cdr_visits < 2:
+ valid = at_risk[(at_risk['CDR_surv_time']>0) & (at_risk['n_cdr_visits']>=2)]
```

### 6.4 不改动的部分

以下完全保持不变：

```
✅ s01: LightGBM Gain 排序 / Univariate C-index 排序 → Top 50
✅ s02: Ward 层次聚类 (Spearman |ρ|, 阈值 0.45–0.75)
✅ s03: 聚类后重新排序
✅ s04: SFS 贪心搜索 (AUC 增益 / C-index 增益, 早停 0.0005)
✅ s05: CalibratedClassifierCV (Task A) / RSF+Cox (Task B)
✅ LGB 超参: n=500, depth=15, leaves=10, subsample=0.7
✅ RSF 超参: n=200, max_depth=5, min_samples_leaf=5
✅ CV 策略: StratifiedKFold, 5-fold, seed=2022
✅ 特征排除: COGNITIVE_PREFIXES + CDR_COLS + ID_COLS + TARGET_COLS
✅ 特征工程: 无新增 (不引入多项式、交互项、embedding)
✅ 缺失值处理: median imputation
```

---

## 7. 评估指标

### 7.1 Task A (二分类)

与 Phase 2A 完全相同的指标：

| 指标 | 说明 |
|:---|:---|
| AUC ± std (5-fold CV) | 区分能力 |
| Brier Score | 概率校准度 |
| Sensitivity @ Youden | 最佳阈值的敏感性 |
| Specificity @ Youden | 最佳阈值的特异性 |
| SFS 累积 AUC 曲线 | 特征增益可视化 |

### 7.2 Task B (生存)

与 Phase 2B 完全相同的指标：

| 指标 | 说明 |
|:---|:---|
| Harrell's C-index | 全局排序能力 |
| time-dependent AUC @ 1/3/5yr | 各时间点区分力 |
| Brier Score @ 1/3/5yr | 概率校准 |
| KM 分层曲线 (三分位) | 风险分层可视化 |

### 7.3 新增对比分析

| 对比 | 目的 |
|:---|:---|
| CDR 二分类 AUC vs DXSUM 二分类 AUC (同 MCI 子集) | 验证 CDR 标签的预测难度 |
| CDR 生存 C-index vs DXSUM 生存 C-index (同 MCI 子集) | 验证两种终点的预测信号一致性 |
| CN 人群 vs MCI 人群的 AUC/C-index | 评估 CDR 终点的新增人群覆盖价值 |
| CDGLOBAL 恶化 vs CDRSB≥2 恶化 | 敏感性分析：阈值选择的影响 |
| CDR 标签选中的特征 vs DXSUM 标签选中的特征 | 跨终点的特征稳定性分析 |

---

## 8. 预期结果与假设

### 8.1 关于性能的假设

**H1**: CDR 二分类 AUC 可能略低于 DXSUM 二分类 AUC（预计 Δ ≈ -0.02 到 -0.05）

理由：CDR 恶化是一个比临床诊断转化更"模糊"的结局——CDR 评分存在评分者间变异、受试者当日状态波动、以及改善（reversion）现象。DXSUM 的 AD 诊断是一个更"硬"的临床决策。

**H2**: CDR 生存 C-index 预计在 0.70–0.75 区间

理由：Phase 2B DXSUM 生存 C-index=0.745。CDR 恶化的"信号"可能略弱（评分主观性），但更多的事件数（477 vs 323）可能提供更多训练信号。两者力量大致抵消。

**H3**: CN 人群 (CDR=0) 的预测难度 > MCI 人群 (CDR=0.5)

理由：CN→MCI 的转变比 MCI→痴呆的转变更早期、更缓慢（中位恶化时间 4.0yr vs 2.1yr），病理信号更弱。预计 CN 人群 AUC 比 MCI 人群低 0.05–0.10。

**H4**: 跨终点的特征选择应显现一致性

MRI 结构特征（皮层厚度、皮层下体积）应在两个终点（DXSUM 和 CDR）中都是重要的——它们反映神经退行性变，与认知衰退的生物学过程而非诊断标签更相关。

### 8.2 主要的实验风险

1. **CDR 数据稀疏性**: 大量受试者 CDR 访视不足 → 高删失率 → 有效样本可能远小于预期
2. **CDR=0 天花板效应**: CN 人群中恶化率仅 26%，且 62% 的恶化发生在 5 年之后 → 3 年窗口的事件率可能过低（~14%）
3. **评分者间变异**: CDR 是半结构化访谈，不同评分者/中心可能有系统偏差
4. **改善（reversion）的污染**: 150 名 CDR=0.5 受试者出现改善（→0），这些人如果被标记为"未恶化"会引入标签噪音

### 8.3 风险缓解

| 风险 | 缓解策略 |
|:---|:---|
| 高删失率 | 报告各窗口有效样本量，仅当 valid≥200 时报告 AUC；放宽窗口至 all-time |
| CDR=0 事件率低 | 优先 5yr 窗口（事件率 21.7% vs 3yr 的 14.3%）；考虑合并 CN+MCI 分析 |
| Reversion 污染 | 敏感性分析：排除 baseline→改善→再恶化 的受试者；或使用"持续恶化"终点 |
| 评分者变异 | 使用 CDGLOBAL（标准培训评分者，ICC>0.9）而非单个域分数 |

---

## 9. 输出文件结构

```
local_data/Results_adni/cdr/
├── cdr_binary/
│   ├── CDR_time_targets.csv              # CDR 恶化标签 (全部受试者)
│   ├── cdr_binary_results.csv            # 各窗口 AUC + Brier + Sens/Spec
│   ├── features_CDR_worsen_3yrs.csv      # SFS 选中特征
│   ├── features_CDR_worsen_5yrs.csv
│   ├── features_CDR_worsen_10yrs.csv
│   ├── sfs_history_CDR_worsen_*.csv      # SFS 逐步增益
│   ├── cdr_binary_auc.png                # AUC 柱状图
│   ├── cdr_vs_dxsum_comparison.png       # CDR vs DXSUM AUC 对比
│   └── sfs_accumulation_cdr.png          # 特征累积曲线
│
├── cdr_survival/
│   ├── cdr_survival_labels.csv           # (time, event) 标签
│   ├── selected_features_cdr.csv         # Top 10 特征
│   ├── sfs_history_cdr.csv               # SFS 逐步 C-index
│   ├── model_comparison_cdr.csv          # RSF vs Cox PH
│   ├── km_by_risk_cdr.png                # KM 三分位曲线
│   ├── calibration_cdr.png               # 校准曲线
│   └── sfs_accumulation_cdr.png          # SFS 累积 C-index
│
└── cdr_cross_endpoint_analysis.csv       # 跨终点一致性分析
```

---

## 10. 实施步骤

### Phase 2C-1: CDR 标签构建 (1 天)

```
1. [新增] build_cdr_targets.py
   - 从 All_Subjects_CDR_20May2026.csv 提取纵向 CDR 轨迹
   - 构建 CDGLOBAL 恶化标签 (分层阈值)
   - 构建 CDRSB≥2 恶化标签 (替代阈值)
   - 计算 last_followup_years (从 CDR 表)
   - 输出: CDR_time_targets.csv
   
2. [合并] 将 CDR 标签 merge 到 ADNI_baseline_with_time_targets_v2.csv
   - 新增 15-20 列 CDR 相关标签
```

### Phase 2C-2: Task A 二分类 (1–2 天)

```
3. [新增] run_adni_cdr_binary.py
   - 复制 run_adni_mci_dementia.py 骨架
   - 替换标签列为 CDR_worsen_3yr/5yr/10yr
   - 替换目标人群为 CN+MCI (baseline CDGLOBAL ∈ {0, 0.5})
   - 其余 s01-s05 保持不变
   - 新增: 与 DXSUM 标签的对比分析
```

### Phase 2C-3: Task B 生存模型 (1–2 天)

```
4. [新增] run_adni_cdr_survival.py
   - 复制 run_adni_survival.py 骨架
   - 替换 surv_time/surv_event 为 CDR 版本
   - 替换目标人群 + 排除规则 (time=0 + n_visits<2)
   - 其余 s01-s05 + RSF/Cox 保持不变
   - 新增: 多终点对比 (CDGLOBAL vs CDRSB)
```

### Phase 2C-4: 跨终点分析报告 (1 天)

```
5. [分析] CDR vs DXSUM 终点的系统性对比
   - 同受试者的两种标签一致性
   - 两种终点下模型性能的差异
   - 特征选择的跨终点稳定性
   - 论文图表: 四象限对比图 (二分类 CDR vs DXSUM, 生存 CDR vs DXSUM)
```

---

## 11. 论文定位建议

在论文中，CDR 扩展可作为 **"次要终点分析"** 或 **"敏感性分析"** 章节：

```
论文结构建议:
─────────────
§3.1 主要终点: MCI→Dementia (DXSUM 诊断)
    - 二分类: AUC 0.834–0.855 (Phase 2A)
    - 生存: C-index 0.745 (Phase 2B)

§3.2 次要终点: CDR 恶化 (CDR 量表)
    - 二分类: 预计 AUC ~0.78–0.83
    - 生存: 预计 C-index ~0.72–0.75
    - 验证主要终点的稳健性

§3.3 终点一致性分析
    - 两种终点的标签重叠矩阵
    - 特征选择的跨终点稳定性 (ICC of feature ranks)
    - 两种终点均选中的"高置信度"特征
```

---

## 12. v2 设计修正（基于审稿级审查）

### 12.1 index_date 统一 (修正问题二)

**v1 问题**: CDR 基线取首次 CDR 记录, 特征基线来自合并表——两者日期可能不同。

**v2 修正**:
- `index_date` = 首次满足 CDGLOBAL ∈ {0, 0.5} 的 CDR 访视日期
- 特征测量必须在 index_date 之前 (或允许时间窗: ±90 天血液, ±180 天 MRI/PET)
- 任何 index_date 之后的特征视为潜在时间泄漏

**实际对齐情况**: 中位差 168 天 (IQR 60–282). 50.9% 在 ±180 天内. 1 人特征日期在 index 后, 0 人在事件后.

### 12.2 CN/MCI 分队列设计 (修正问题三)

**v1 问题**: CN+MCI 合并建模, 基线风险不同但模型不知.

**v2 修正**:
- **Primary**: CN 队列 (CDR=0→≥0.5) 和 MCI 队列 (CDR=0.5→≥1) 分别建模
- **Secondary**: Pooled CN+MCI, 必须加入 `baseline_CDR_group` 作为协变量
- 检验 predictor × baseline CDR group 交互

### 12.3 持续性恶化 (修正问题六)

**v1 问题**: 首次恶化即为事件, 不验证是否持续.

**v2 新增标签**:
```
持续性恶化 =
  首次达到恶化阈值后,
  下一次有效访视仍处于恶化水平,
  或后续所有访视均未恢复至基线水平
```

### 12.4 生存感知特征选择 (修正问题四)

**v1 问题**: s01-s04 用 LGB AUC (binary), surv_event 当 0/1, 未正确处理删失.

**v2 修正**: 以 LGB AUC SFS 为主方案, 同时进行 C-index SFS 敏感性分析. v2 实验显示两者差距极小 (ΔC-index = −0.0004).

### 12.5 训练折内预处理 (修正问题五)

**v1 问题**: 中位数值填补 + Ward 聚类 + StandardScaler 在全局数据上完成.

**v2 修正**: 缺失值填补和标准化在每个 CV 训练折内完成 (fit on train, apply to test). Ward 聚类因需要全局相关结构而保留 (作为 known limitation 记录).

### 12.6 CONSORT 流图 (修正问题一)

**v2 新增**: 完整的样本流图 (2,749 → 1,595) + 纳入者 vs 排除者比较.

---

## 13. v2 脚本清单

```
src/training/
├── run_adni_cdr_binary_v2.py       # Task A v2: CDR 二分类 (分队列)
├── run_adni_cdr_survival_v2.py     # Task B v2: CDR 生存 (分队列)

思路一/ADNI数据集/
├── build_cdr_targets_v2.py          # v2 CDR 标签构建
│                                    # - 统一 index_date
│                                    # - 分队列 (CN/MCI)
│                                    # - 持续性恶化
│                                    # - CDR-SB 恶化
│                                    # - 复合终点
│                                    # - CONSORT 统计
│                                    # - 日期对齐检查
```

---

## 附录 A: 与 Phase 2A/2B 的完整对比矩阵

```
┌──────────────────────┬─────────────────────┬─────────────────────┬──────────────────────┐
│                      │ Phase 2A (DXSUM)    │ Phase 2B (DXSUM)    │ CDR 扩展 (Task A+B)   │
├──────────────────────┼─────────────────────┼─────────────────────┼──────────────────────┤
│ 终点                 │ Dementia转化 (诊断)  │ Dementia转化 (诊断)  │ CDR 恶化 (评分)        │
│ 终点类型              │ 二元 (0/1)          │ (time, event)       │ 二元 + (time, event)  │
│ 基线人群              │ MCI (DX=2)          │ MCI (DX=2)          │ CN (CDR=0) + MCI(0.5) │
│ 窗口                  │ 3yr/5yr/10yr        │ 连续时间             │ 3yr/5yr/10yr + 连续   │
│ 事件率 @3yr           │ 32.7%               │ —                   │ ~23% (预计)           │
│ 有效样本              │ 600 (3yr)           │ 944                  │ ~1,235 (3yr, 预计)    │
│ 删失策略              │ 排除不足随访         │ 保留 (censored)      │ 排除不足随访 +         │
│                       │                     │                     │   保留 (censored)      │
│ 模型                  │ LGBM + Isotonic     │ RSF + CoxPH         │ LGBM/RSF (相同架构)   │
│ 管线                  │ s01-s05             │ s01-s05             │ s01-s05 (不变)        │
│ 主要指标              │ AUC                 │ C-index             │ AUC + C-index         │
│ 特征体系              │ 358 (排除认知+CDR)   │ 358 (排除认知+CDR)   │ 358 (排除认知+CDR)    │
│ 认知测试              │ 排除                │ 排除                │ 排除                  │
│ CDR 列                │ 排除 (标签泄漏)      │ 排除 (标签泄漏)      │ 排除 (结局变量)        │
└──────────────────────┴─────────────────────┴─────────────────────┴──────────────────────┘
```

## 附录 B: CDR 量表速查

```
CDR 全局评分 (CDGLOBAL):
  0   = 认知正常 (No dementia)
  0.5 = 可疑痴呆 / 轻度认知障碍 (MCI)
  1   = 轻度痴呆 (Mild dementia)
  2   = 中度痴呆 (Moderate dementia)  
  3   = 重度痴呆 (Severe dementia)

CDR Sum of Boxes (CDRSB):
  范围: 0–18
  计算: CDMEMORY + CDORIENT + CDJUDGE + CDCOMMUN + CDHOME + CDCARE
  (每个域 0–3, 6个域求和)
  
  临床解读:
    0–2.5:   正常/极轻度
    3.0–4.0: 轻度认知障碍
    4.5–9.0: 轻度痴呆
    9.5–15.5: 中度痴呆
    16.0–18.0: 重度痴呆

六域评分标准:
  0   = 无损害 (None)
  0.5 = 可疑损害 (Questionable)  — 仅 CDMEMORY, CDORIENT, CDJUDGE, CDCOMMUN, CDHOME
  1   = 轻度损害 (Mild)
  2   = 中度损害 (Moderate)
  3   = 重度损害 (Severe)
  
  注意: CDCARE (个人自理) 无 0.5 级，直接从 0 跳到 1
```

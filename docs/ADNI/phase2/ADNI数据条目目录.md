# ADNI 数据条目完整目录

> 数据导出日期: 2026-05-20
> 来源: ADNI IDA 门户 (Alzheimer's Disease Neuroimaging Initiative)
> 总表数: 66 个 CSV
> 总受试者: ~4,868 (去重后约 2,749 有完整基线)

---

## 1. 人口学与入组 (8 个表)

### 1.1 PTDEMOG — 人口学主表
| 项目 | 值 |
|------|-----|
| 文件 | `All_Subjects_PTDEMOG_20May2026.csv` |
| 规模 | 6,149 rows, 4,868 subjects, 84 columns, 2.2 MB |
| 访视 | 多访视 (bl/sc/init 为基线) |
| **用** | ✅ 用于预处理 — 提取年龄、性别、教育、婚姻、就业 |

**关键列**: `PTID`, `RID`, `PTGENDER`(1=Male,2=Female), `PTEDUCAT`(教育年限), `PTMARRY`(婚姻), `PTWORKHS`(就业状态), `PTDOBYY`(出生年), `PTHAND`(利手)

---

### 1.2 Study_Entry — 入组信息
| 项目 | 值 |
|------|-----|
| 文件 | `All_Subjects_Study_Entry_20May2026.csv` |
| 规模 | 4,856 rows, 5 columns, 0.2 MB |
| 访视 | 每人一行 |
| **用** | ✅ 用于预处理 — 提取入组年龄和研究组别 |

**关键列**: `subject_id`, `entry_age`, `entry_research_group`

---

### 1.3 BLCHANGE — 基线变化评估
| 项目 | 值 |
|------|-----|
| 文件 | `All_Subjects_BLCHANGE_20May2026.csv` |
| 规模 | 16,113 rows, 3,720 subjects, 29 columns, 2.7 MB |
| 访视 | 多访视 |
| **用** | ❌ 未使用 — 辅助性临床判断数据 |

**关键列**: `BCPREDX`(基线前诊断), `BCADAS`, `BCMMSE`

---

### 1.4-1.8 家族史与社会经济

| 表名 | 规模 | 内容 | 预处理 |
|------|------|------|--------|
| `FAMHXPAR` | 2,313 rows, 24 cols | 父母痴呆/AD 病史 | ❌ |
| `FAMHXSIB` | 6,761 rows, 22 cols | 兄弟姐妹痴呆/AD 病史 | ❌ |
| `FHQ` | 2,952 rows, 18 cols | 家族史问卷 | ❌ |
| `RECFHQ` | 7,407 rows, 15 cols | 修订版家族史问卷 | ❌ |
| `RURALITY` | 1,868 rows, 19 cols | 城乡分类 (RUCA code) | ❌ |

---

## 2. 诊断与临床评估 (5 个表)

### 2.1 DXSUM — 诊断汇总 ★核心表
| 项目 | 值 |
|------|-----|
| 文件 | `All_Subjects_DXSUM_20May2026.csv` |
| 规模 | 16,118 rows, 3,711 subjects, 41 columns, 3.3 MB |
| 访视 | 多访视 (bl/sc/init 基线) |
| **用** | ✅ 用于构建目标变量 (DIAGNOSIS, DXPARK, DXOTHDEM) |

**关键列**: `DIAGNOSIS`(1=CN,2=MCI,3=AD), `DXAD`, `DXPARK`, `DXOTHDEM`, `DXNORM`, `DXMCI`, `EXAMDATE`

---

### 2.2 CDR — 临床痴呆评定量表 ★
| 项目 | 值 |
|------|-----|
| 文件 | `All_Subjects_CDR_20May2026.csv` |
| 规模 | 14,766 rows, 4,320 subjects, 25 columns, 2.5 MB |
| 访视 | 多访视 |
| **用** | ⚠️ 用于基线特征但建模时排除 (诊断泄漏风险) |

**关键列**: `CDGLOBAL`(0-3), `CDRSB`(Sum of Boxes, 0-18), `CDMEMORY`, `CDORIENT`, `CDJUDGE`, `CDCOMMUN`, `CDHOME`, `CDCARE`

---

### 2.3 My_Table — 纵向追踪汇总
| 项目 | 值 |
|------|-----|
| 文件 | `All_Subjects_My_Table_20May2026.csv` |
| 规模 | 24,840 rows, 59 columns, 4.9 MB |
| 内容 | 受试者级别的纵向追踪记录，含 AMAS 和 ADI |
| **用** | ❌ 未使用 — DXSUM 已覆盖诊断追踪 |

---

### 2.4-2.5 其他诊断相关

| 表名 | 规模 | 内容 | 预处理 |
|------|------|------|--------|
| `MRIFind` | 7,322 rows, 41 cols | MRI 影像学发现 (放射科读片) | ❌ |
| `ADSXLIST` | 4,884 rows, 39 cols | AD 症状列表 (恶心/头晕等副作用) | ❌ |

---

## 3. 认知测试 (8 个表)

### 3.1 MMSE — 简易精神状态检查 ★
| 项目 | 值 |
|------|-----|
| 文件 | `All_Subjects_MMSE_20May2026.csv` |
| 规模 | 14,729 rows, 4,668 subjects, 58 columns, 4.0 MB |
| **用** | ⚠️ 用于基线特征但建模时排除 |

**关键列**: `MMSCORE`(0-30 总分), 以及各子项 (定向力、回忆、注意等)

### 3.2 MOCA — 蒙特利尔认知评估
| 项目 | 值 |
|------|-----|
| 文件 | `All_Subjects_MOCA_20May2026.csv` |
| 规模 | 9,116 rows, 2,445 subjects, 58 columns, 2.6 MB |
| **用** | ⚠️ 排除 (认知测试) |

### 3.3 ADAS — 阿尔茨海默病评估量表
| 项目 | 值 |
|------|-----|
| 文件 | `All_Subjects_ADAS_20May2026.csv` |
| 规模 | 13,021 rows, 2,971 subjects, 16 columns, 1.7 MB |
| **用** | ⚠️ 排除 (认知测试) |

**关键列**: `TOTSCORE`(ADAS-Cog 11题版), `TOTAL13`(13题版)

### 3.4 NEUROBAT — 神经心理成套测验
| 项目 | 值 |
|------|-----|
| 文件 | `All_Subjects_NEUROBAT_20May2026.csv` |
| 规模 | 17,697 rows, 4,706 subjects, 83 columns, 6.4 MB |
| **用** | ⚠️ 排除 (认知测试) |

**关键列**: 逻辑记忆 (`LIMMTOTAL`, `LDELTOTAL`), 词语学习 (`AVTOT1`-`AVTOT5`), 画钟 (`CLOCKSCOR`), 连线 (`TRAILSCOR`), 命名 (`BNTTOTAL`)

### 3.5-3.8 其他认知

| 表名 | 规模 | 内容 | 预处理 |
|------|------|------|--------|
| `MODHACH` | 3,462 rows, 21 cols | Hachinski 缺血量表 (鉴别血管性痴呆) | ⚠️ |
| `AMNART` | 1,045 rows, 67 cols | 美国成人阅读测试 (病前智力估计) | ❌ |
| `AMAS` | 1,016 rows, 58 cols | AD 评估量表 | ❌ |
| `CBBCOMP` | 2,857 rows, 13 cols | 计算机化认知电池综合 | ❌ |

---

## 4. 功能与行为评估 (17 个表)

### 4.1 FAQ — 功能活动问卷 ★
| 项目 | 值 |
|------|-----|
| 文件 | `All_Subjects_FAQ_20May2026.csv` |
| 规模 | 13,439 rows, 2,966 subjects, 27 columns, 2.1 MB |
| **用** | ⚠️ 排除 (功能评估，MCI/AD诊断的一部份) |

**关键列**: `FAQTOTAL`(0-30), 各子项 (理财、购物、做饭等)

### 4.2 GDSCALE — 老年抑郁量表
| 项目 | 值 |
|------|-----|
| 文件 | `All_Subjects_GDSCALE_20May2026.csv` |
| 规模 | 13,843 rows, 4,509 subjects, 32 columns, 2.5 MB |
| **用** | ⚠️ 排除 (情绪评估) |

### 4.3-4.4 NPI / NPIQ — 神经精神量表
| 表名 | 规模 | 内容 |
|------|------|------|
| `NPI` | 8,395 rows, 168 cols, 5.0 MB | 完整版 (12项+频率×严重度) |
| `NPIQ` | 7,574 rows, 41 cols, 1.6 MB | 简明版 |

### 4.5-4.8 ECOG — 日常认知评估

| 表名 | 规模 | 内容 |
|------|------|------|
| `ECOGPT` | 8,093 rows, 62 cols, 2.5 MB | 受试者自评, 完整版 |
| `ECOGSP` | 8,107 rows, 59 cols, 2.4 MB | 知情者评估, 完整版 |
| `ECOG12PT` | 1,383 rows, 31 cols, 0.3 MB | 受试者自评, 12题版 |
| `ECOG12SP` | 1,380 rows, 28 cols, 0.2 MB | 知情者评估, 12题版 |

### 4.9-4.17 其他行为/心理评估

| 表名 | 规模 | 内容 |
|------|------|------|
| `FCI` | 2,626 rows, 69 cols, 0.9 MB | 功能认知指数 (财务+检查) |
| `PSS` | 1,404 rows, 26 cols, 0.2 MB | 感知压力量表 |
| `RYFF` | 1,063 rows, 28 cols, 0.2 MB | 心理幸福感量表 |
| `STAIAD` | 420 rows, 21 cols, 0.1 MB | 状态-特质焦虑量表 |
| `IES` | 787 rows, 29 cols, 0.1 MB | 生活事件影响量表 |
| `CSSRSAD` | 413 rows, 18 cols, 0.1 MB | 自杀意念量表 |
| `PEDQCV` | 1,402 rows, 33 cols, 0.3 MB | 日常认知问卷 |
| `WATC` | 958 rows, 58 cols, 0.2 MB | 词汇获取测试 |
| `BHR` | 1,104 rows, 13 cols, 0.1 MB | 脑健康登记 |

---

## 5. 遗传 (2 个表)

### 5.1 APOERES — APOE 基因型 ★
| 项目 | 值 |
|------|-----|
| 文件 | `All_Subjects_APOERES_20May2026.csv` |
| 规模 | 3,208 rows, 3,208 subjects, 16 columns, 0.3 MB |
| 访视 | 每人一行 (一次性) |
| **用** | ✅ 用于提取 APOE4_count 和 APOE4_carrier |

**关键列**: `GENOTYPE`(如 "33","34","44")

### 5.2 GENETIC — 遗传样本信息
| 项目 | 值 |
|------|-----|
| 文件 | `All_Subjects_GENETIC_20May2026.csv` |
| 规模 | 10,277 rows, 3,285 subjects, 57 columns, 2.8 MB |
| 内容 | 血液采集、DNA/RNA提取、样本处理 |
| **用** | ❌ — APOERES 已覆盖基因型数据 |

---

## 6. CSF 生物标记 (3 个表)

### 6.1 UPENNBIOMK_ROCHE_ELECSYS — CSF 核心标记 ★
| 项目 | 值 |
|------|-----|
| 文件 | `All_Subjects_UPENNBIOMK_ROCHE_ELECSYS_20May2026.csv` |
| 规模 | 3,174 rows, 1,660 subjects, 13 columns, 0.4 MB |
| **用** | ✅ 用于提取 CSF Aβ42, Aβ40, T-tau, pTau181 |

**关键列**: `ABETA40`, `ABETA42`, `TAU`(总 tau), `PTAU`(p-Tau181)

### 6.2-6.3 其他

| 表名 | 规模 | 内容 | 预处理 |
|------|------|------|--------|
| `BIOMARK` | 14,179 rows, 65 cols, 4.6 MB | 生物样本采集详情 (血/尿/CSF) | ❌ |
| `CSFMETH` | 1,165 rows, 19 cols, 0.1 MB | 腰椎穿刺详情 (针号、体位) | ❌ |

---

## 7. 血浆生物标记 (5 个表)

### 7.1 UPENN_PLASMA_FUJIREBIO_QUANTERIX — 血浆核心标记 ★
| 项目 | 值 |
|------|-----|
| 文件 | `All_Subjects_UPENN_PLASMA_FUJIREBIO_QUANTERIX_20May2026.csv` |
| 规模 | 2,298 rows, 1,593 subjects, 19 columns, 0.4 MB |
| 平台 | Fujirebio (pTau217, Aβ42/40) + Quanterix (NfL, GFAP) |
| **用** | ✅ 用于提取 pT217_F, NfL_Q, GFAP_Q |

**关键列**: `pT217_F`, `AB42_F`, `AB40_F`, `AB42_AB40_F`, `NfL_Q`(神经丝轻链), `GFAP_Q`(胶质纤维酸性蛋白)

### 7.2 C2N_PRECIVITYAD2_PLASMA — C2N PrecivityAD2
| 项目 | 值 |
|------|-----|
| 文件 | `All_Subjects_C2N_PRECIVITYAD2_PLASMA_20May2026.csv` |
| 规模 | 1,069 rows, 926 subjects, 18 columns, 0.2 MB |
| **用** | ✅ 用于提取 pT217_C2N, Aβ42/40_C2N, APS2 |

**关键列**: `pT217_C2N`, `npT217_C2N`, `AB42_C2N`, `AB40_C2N`, `APS2_C2N`(Amyloid Probability Score 2)

### 7.3-7.5 其他血浆

| 表名 | 规模 | 内容 | 预处理 |
|------|------|------|--------|
| `LILLY_PTAU217_MSD600` | 299 rows, 18 cols | Lilly MSD pTau217 检测 | ❌ |
| `JANSSEN_PLASMA_P217_TAU` | 130 rows, 9 cols | Janssen pTau217 检测 | ❌ |
| `FNIHBC_BLOOD_BIOMARKER_TRAJECTORIES` | 24,154 rows, 20 cols | FNIH 纵向血浆标记轨迹 | ❌ |

---

## 8. 血液检测 (5 个表)

| 表名 | 规模 | 内容 | 预处理 |
|------|------|------|--------|
| `LABDATA` | 2,463 rows, 132 cols, 1.8 MB | 标准血液生化 (CBC, 肝功能, 肾功能, 电解质) | ❌ |
| `LABTESTS` | 5,556 rows, 21 cols, 0.8 MB | 血液/尿液样本采集记录 | ❌ |
| `LOCLAB` | 8,579 rows, 22 cols, 1.2 MB | 本地实验室检测 (CSF 蛋白/葡萄糖) | ❌ |
| `URMC_LABDATA` | 137,664 rows, 28 cols, 29.3 MB | 统一实验室检测数据 (大规模) | ❌ |
| `CCI` | 1,228 rows, 33 cols, 0.2 MB | Charlson 共病指数 | ⚠️ |

---

## 9. MRI 结构影像 (3 个表)

### 9.1 UCSFFSX7 — FreeSurfer v7 ★
| 项目 | 值 |
|------|-----|
| 文件 | `All_Subjects_UCSFFSX7_20May2026.csv` |
| 规模 | 12,166 rows, 3,174 subjects, **347 columns**, 30.9 MB |
| **用** | ✅ 核心影像特征 — 提取 ~330 个脑区体积/厚度/面积 |

**特征类型** (以 `FS_` 为前缀):
- `STxxSV`: 皮层下核团体积 (海马、杏仁核、丘脑等)
- `STxxTA`: 皮层厚度 (按 Desikan-Killiany 分区)
- `STxxSA`: 皮层表面积
- 质量控制: `OVERALLQC`, `TEMPQC`, `FRONTQC`, `PARQC`, `HIPPOQC`

### 9.2 FOXLABBSI — 脑结构体积
| 项目 | 值 |
|------|-----|
| 文件 | `All_Subjects_FOXLABBSI_20May2026.csv` |
| 规模 | 15,960 rows, 3,189 subjects, 28 columns, 3.2 MB |
| **用** | ✅ 提取 BSI 全脑/脑室/海马体积 |

**关键列**: `BRAINVOL`, `VENTVOL`, `HIPPOVOL_R`, `HIPPOVOL_L`, `DBCBBSI`

### 9.3 UCD_WMH — 白质高信号
| 项目 | 值 |
|------|-----|
| 文件 | `All_Subjects_UCD_WMH_20May2026.csv` |
| 规模 | 7,746 rows, 2,422 subjects, 27 columns, 1.9 MB |
| **用** | ✅ 提取白质病变体积 |

**关键列**: `WMH_VOLUME`, `WMH_DEEP`(深部), `WMH_PERI`(脑室旁), `CEREBRUM_TCV`(总颅内容积)

---

## 10. MRI DTI (2 个表)

| 表名 | 规模 | 内容 | 预处理 |
|------|------|------|--------|
| `DTIROI_MEAN` | 2,636 rows, 309 cols, 9.4 MB | DTI ROI 均值 (FA, MD, AxD, RD per ROI) | ❌ |
| `DTIROI_ROBUSTMEAN` | 2,572 rows, 309 cols, 9.2 MB | DTI ROI 稳健均值 | ❌ |

---

## 11. MRI 其他 (3 个表)

| 表名 | 规模 | 内容 | 预处理 |
|------|------|------|--------|
| `Key_MRI` | 91,572 rows, 23 cols, 27.0 MB | MRI 扫描元数据 (序列/厂商/参数) | ❌ |
| `MRIQSM` | 28,823 rows, 38 cols, 11.5 MB | 定量磁敏感图 (QSM) 脑区值 | ❌ |
| `MRINFQ` | 947 rows, 46 cols, 0.9 MB | MRI 静息态 fMRI 网络质量 | ❌ |

---

## 12. Amyloid PET (1 个表)

### 12.1 UCBERKELEY_AMY_6MM ★
| 项目 | 值 |
|------|-----|
| 文件 | `All_Subjects_UCBERKELEY_AMY_6MM_20May2026.csv` |
| 规模 | 4,727 rows, 2,240 subjects, **344 columns**, 11.4 MB |
| 处理 | UC Berkeley 6mm 空间分辨率 |
| **用** | ✅ 核心 PET 特征 |

**关键列**: `AMYLOID_STATUS`(阳性/阴性), `CENTILOIDS`(标准量化), `SUMMARY_SUVR`(全脑 SUVR), `COMPOSITE_REF_SUVR`(复合参考区 SUVR), 以及 ~330 个脑区的 SUVR 值

---

## 13. Tau PET (2 个表)

### 13.1 UCBERKELEY_TAU_6MM ★
| 项目 | 值 |
|------|-----|
| 文件 | `All_Subjects_UCBERKELEY_TAU_6MM_20May2026.csv` |
| 规模 | 2,489 rows, 1,448 subjects, **339 columns**, 6.0 MB |
| 处理 | 标准 6mm 空间分辨率 |
| **用** | ✅ 用于提取 Tau PET 特征 |

**关键列**: `META_TEMPORAL_SUVR`(颞叶 meta-ROI), `CTX_ENTORHINAL_SUVR`(内嗅皮层), `INFERIORCEREBELLUM_SUVR`(参考区)

### 13.2 UCBERKELEY_TAUPVC_6MM
| 项目 | 值 |
|------|-----|
| 文件 | `All_Subjects_UCBERKELEY_TAUPVC_6MM_20May2026.csv` |
| 规模 | 2,398 rows, 1,415 subjects, 335 columns, 5.8 MB |
| 处理 | 部份容积校正版 |
| **用** | ❌ 未使用 — 使用标准 Tau 6mm |

---

## 14. 其他 (2 个表)

| 表名 | 规模 | 内容 |
|------|------|------|
| `ADI` | 1,866 rows, 17 cols, 0.3 MB | ADI 知情同意/入组状态 |
| `Key_PET` | 10,089 rows, 8 cols, 1.1 MB | PET 扫描元数据 (示踪剂类型、日期) |

---

## 附录 A: 预处理流程中使用到的表

```
66 个原始 CSV
    │
    ├── ✅ 已用于预处理 (14 个):
    │   PTDEMOG         → 人口学 (年龄/性别/教育/就业)
    │   Study_Entry      → 入组年龄
    │   DXSUM           → 诊断追踪 (目标变量★)
    │   CDR              → 临床痴呆评分
    │   MMSE             → 认知筛查
    │   ADAS             → 认知评估
    │   FAQ              → 功能评估
    │   GDSCALE          → 抑郁量表
    │   NPIQ             → 神经精神
    │   MODHACH          → Hachinski 缺血
    │   NEUROBAT         → 神经心理
    │   CCI              → 共病指数
    │   APOERES          → APOE 基因型
    │   UPENNBIOMK_ROCHE_ELECSYS → CSF Aβ/pTau
    │   UPENN_PLASMA_FUJIREBIO_QUANTERIX → 血浆 pTau217/NfL/GFAP
    │   C2N_PRECIVITYAD2_PLASMA → 血浆 PrecivityAD2
    │   UCSFFSX7         → FreeSurfer v7 (347列)
    │   FOXLABBSI        → BSI 脑体积
    │   UCD_WMH          → 白质病变
    │   UCBERKELEY_AMY_6MM → Amyloid PET
    │   UCBERKELEY_TAU_6MM → Tau PET
    │
    ├── ⚠️ 部分使用/建模时排除 (认知测试类):
    │   MMSE, MOCA, ADAS, FAQ, GDSCALE, NPIQ, NEUROBAT, MODHACH, CCI
    │   (用于基线特征分析，但在预测建模中排除以避免诊断泄漏)
    │
    └── ❌ 未使用 (48 个):
        详见各分类中的标注
```

---

## 附录 B: 最终预处理产物

| 输出文件 | 规模 | 说明 |
|----------|------|------|
| `ADNI_baseline.csv` | 2,749人 × 388列 | 基线特征矩阵 (所有受试者) |
| `ADNI_features.csv` | 2,749人 × ~358列 | 纯特征矩阵 |
| `ADNI_targets.csv` | 2,749人 | 诊断目标变量 |
| `ADNI_baseline_with_time_targets_v2.csv` | 2,749人 × 411列 | 含 v2.0 时间窗口目标 + 删失标记 |
| `ADNI_time_targets_v2.csv` | 2,749人 × 23列 | 3yr/5yr/10yr 目标 + 随访 + 删失 |

---

## 附录 C: 数据文件大小

```
extracted/                                     ~235 MB (压缩)
  ├── UCSFFSX7                                        30.9 MB  (最大)
  ├── URMC_LABDATA                                     29.3 MB
  ├── Key_MRI                                          27.0 MB
  ├── MRIQSM                                           11.5 MB
  ├── UCBERKELEY_AMY_6MM                               11.4 MB
  ├── DTIROI_MEAN                                       9.4 MB
  ├── DTIROI_ROBUSTMEAN                                 9.2 MB
  ├── NEUROBAT                                          6.4 MB
  ├── UCBERKELEY_TAU_6MM                                6.0 MB
  ├── UCBERKELEY_TAUPVC_6MM                             5.8 MB
  ├── FNIHBC_BLOOD_BIOMARKER_TRAJECTORIES               5.4 MB
  ├── NPI                                               5.0 MB
  ├── My_Table                                          4.9 MB
  ├── BIOMARK                                           4.6 MB
  ├── MMSE                                              4.0 MB
  ├── DXSUM                                             3.3 MB
  ├── FOXLABBSI                                         3.2 MB
  ├── MRIFind                                           3.1 MB
  ├── GENETIC                                           2.8 MB
  ├── BLCHANGE                                          2.7 MB
  ├── MOCA                                              2.6 MB
  ├── GDSCALE                                           2.5 MB
  ├── CDR                                               2.5 MB
  ├── ECOGPT                                            2.5 MB
  ├── ECOGSP                                            2.4 MB
  ├── PTDEMOG                                           2.2 MB
  ├── FAQ                                               2.1 MB
  ├── UCD_WMH                                           1.9 MB
  ├── LABDATA                                           1.8 MB
  ├── ADAS                                              1.7 MB
  ├── NPIQ                                              1.6 MB
  ├── LOCLAB                                            1.2 MB
  ├── Key_PET                                           1.1 MB
  ├── ADSXLIST                                          1.0 MB
  ├── FAMHXSIB                                          1.0 MB
  ├── FCI                                               0.9 MB
  ├── MRINFQ                                            0.9 MB
  ├── LABTESTS                                          0.8 MB
  ├── RECFHQ                                            0.8 MB
  └── (其余 30 个 < 0.5 MB each)
```

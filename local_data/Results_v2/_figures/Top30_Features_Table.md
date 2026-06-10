# Top 30 Features — s01 Gain Ranking

## Non-MRI Experiment (1076 clinical features)

| Rank | Feature ID | Gain | Description |
|------|-----------|------|-------------|
| 1 | `34-0.0` | 1.4940 | Year of birth |
| 2 | `21022-0.0` | 1.2112 | Age at recruitment |
| 3 | `21003-0.0` | 0.4195 | Age at assessment centre |
| 4 | `400-0.0` | 0.2416 | Reaction time (mean, round) |
| 5 | `2188-0.0_c1` | 0.1799 | Diabetes (Dr diagnosed, 2) |
| 6 | `137-0.0` | 0.1191 | Number of medications |
| 7 | `6142-0.0_pos` | 0.0858 | Employment: employed (3) |
| 8 | `6142-0.3_pos` | 0.0807 | Employment: retired |
| 9 | `20110-0.4_pos` | 0.0666 | Mother: dementia/Alzheimer's |
| 10 | `6146-0.1_pos` | 0.0621 | Physical activity: walking |
| 11 | `20023-0.0` | 0.0609 | Reaction time: mean correct |
| 12 | `30710-0.0` | 0.0543 | C-reactive protein (CRP) |
| 13 | `2188-0.0_c0` | 0.0369 | Diabetes (Dr diagnosed, 1) |
| 14 | `2296-0.0_c0` | 0.0322 | Hearing difficulty |
| 15 | `3526-0.0` | 0.0307 | Mother's age at death |
| 16 | `135-0.0` | 0.0302 | Non-cancer illnesses count (2) |
| 17 | `1835-0.0_c0` | 0.0245 | Number in household (2) |
| 18 | `2492-0.0_c1` | 0.0239 | Other Rx medications (2) |
| 19 | `1080-0.0` | 0.0229 | Time to identify matches (2) |
| 20 | `30650-0.0` | 0.0227 | Aspartate aminotransferase |
| 21 | `20107-0.4_pos` | 0.0211 | Father: dementia/Alzheimer's |
| 22 | `1060-0.0` | 0.0209 | Reaction time (SD) |
| 23 | `1588-0.0` | 0.0207 | Average weekly beer/cider |
| 24 | `2178-0.0_c3` | 0.0203 | High BP diagnosed (3) |
| 25 | `1200-0.0_c0` | 0.0191 | Sleep duration (4, h) |
| 26 | `20127-0.0` | 0.0182 | Neuroticism score |
| 27 | `20110-0.1_neg` | 0.0170 | Mother: no major illness |
| 28 | `30040-0.0` | 0.0167 | Mean platelet volume |
| 29 | `3064-0.0` | 0.0137 | Haemoglobin conc (4) |
| 30 | `1070-0.0` | 0.0134 | Reaction time (mean) |

## +MRI Experiment (3250 clinical + MRI features)

| Rank | Feature ID | Gain | Imaging? | Description |
|------|-----------|------|----------|-------------|
| 1 | `34-0.0` | 1.9526 |  | Year of birth |
| 2 | `21022-0.0` | 0.8292 |  | Age at recruitment |
| 3 | `400-0.0` | 0.2112 |  | Reaction time (mean, round) |
| 4 | `12651-0.0` | 0.1843 | MRI | ★ MRI: L hippocampal subiculum vol |
| 5 | `2188-0.0_c1` | 0.1831 |  | Diabetes (Dr diagnosed, 2) |
| 6 | `21003-0.0` | 0.1644 |  | Age at assessment centre |
| 7 | `137-0.0` | 0.1050 |  | Number of medications |
| 8 | `6142-0.0_pos` | 0.0928 |  | Employment: employed (3) |
| 9 | `6142-0.3_pos` | 0.0719 |  | Employment: retired |
| 10 | `20110-0.4_pos` | 0.0607 |  | Mother: dementia/Alzheimer's |
| 11 | `6146-0.1_pos` | 0.0548 |  | Physical activity: walking |
| 12 | `20023-0.0` | 0.0507 |  | Reaction time: mean correct |
| 13 | `30710-0.0` | 0.0486 |  | C-reactive protein (CRP) |
| 14 | `2296-0.0_c0` | 0.0294 |  | Hearing difficulty |
| 15 | `135-0.0` | 0.0290 |  | Non-cancer illnesses count (2) |
| 16 | `26555-0.0` | 0.0254 | MRI | MRI: ★ Subiculum volume (R) |
| 17 | `3526-0.0` | 0.0240 |  | Mother's age at death |
| 18 | `2492-0.0_c1` | 0.0223 |  | Other Rx medications (2) |
| 19 | `20107-0.4_pos` | 0.0187 |  | Father: dementia/Alzheimer's |
| 20 | `2188-0.0_c0` | 0.0180 |  | Diabetes (Dr diagnosed, 1) |
| 21 | `2178-0.0_c3` | 0.0178 |  | High BP diagnosed (3) |
| 22 | `30650-0.0` | 0.0176 |  | Aspartate aminotransferase |
| 23 | `1200-0.0_c0` | 0.0175 |  | Sleep duration (4, h) |
| 24 | `1588-0.0` | 0.0169 |  | Average weekly beer/cider |
| 25 | `1835-0.0_c0` | 0.0165 |  | Number in household (2) |
| 26 | `1060-0.0` | 0.0159 |  | Reaction time (SD) |
| 27 | `20110-0.1_neg` | 0.0151 |  | Mother: no major illness |
| 28 | `20127-0.0` | 0.0139 |  | Neuroticism score |
| 29 | `26643-0.0` | 0.0133 | MRI | ★ MRI: Left subiculum volume |
| 30 | `30040-0.0` | 0.0133 |  | Mean platelet volume |

## SFS Selected Features Comparison

### Non-MRI SFS Selection Order (s04)

| Step | Feature ID | Cumulative AUC | Description |
|------|-----------|---------------|-------------|
| 1 | `34-0.0` | 0.7984 | Year of birth |
| 2 | `400-0.0` | 0.8098 | Reaction time (mean, round) |
| 3 | `137-0.0` | 0.8178 | Number of medications |
| 4 | `2188-0.0_c1` | 0.8234 | Diabetes (Dr diagnosed, 2) |
| 5 | `30710-0.0` | 0.8252 | C-reactive protein (CRP) |
| 6 | `20110-0.4_pos` | 0.8271 | Mother: dementia/Alzheimer's |
| 7 | `1090-0.0` | 0.8285 | Trail making: time to complete |
| 8 | `6142-0.3_pos` | 0.8298 | Employment: retired |
| 9 | `20023-0.0` | 0.8310 | Reaction time: mean correct |
| 10 | `3526-0.0` | 0.8320 | Mother's age at death |

### +MRI SFS Selection Order (s04)

| Step | Feature ID | Cumulative AUC | Type | Description |
|------|-----------|---------------|------|-------------|
| 1 | `34-0.0` | 0.7984 | Clinical | Year of birth |
| 2 | `400-0.0` | 0.8098 | Clinical | Reaction time (mean, round) |
| 3 | `137-0.0` | 0.8178 | Clinical | Number of medications |
| 4 | `12651-0.0` | 0.8268 | MRI | ★ MRI: L hippocampal subiculum vol |
| 5 | `2188-0.0_c1` | 0.8297 | Clinical | Diabetes (Dr diagnosed, 2) |
| 6 | `30710-0.0` | 0.8321 | Clinical | C-reactive protein (CRP) |
| 7 | `20110-0.4_pos` | 0.8337 | Clinical | Mother: dementia/Alzheimer's |
| 8 | `6142-0.3_pos` | 0.8348 | Clinical | Employment: retired |
| 9 | `26643-0.0` | 0.8361 | MRI | ★ MRI: Left subiculum volume |
| 10 | `1200-0.0_c0` | 0.8373 | Clinical | Sleep duration (4, h) |

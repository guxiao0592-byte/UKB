#!/usr/bin/env python3
"""
Generate SHAP beeswarm plots with human-readable feature names.
Replaces feature codes (e.g., "34-0.0") with descriptive labels.

Covers:
  - Non-MRI experiment (1076 features): All 6 targets (DM_full Deploy strategy)
  - +MRI experiment (3250 features): DM_full features deployed to all 6 targets
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
import os, sys, warnings, json
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
BASE        = '/Users/guxiao/Downloads/MCI-AD/UKB_DRP-main'
DATA_CLIN   = os.path.join(BASE, 'local_data/Preprocessed_Data/Preprocessed_Data.csv')
DATA_IMG    = os.path.join(BASE, 'local_data/Preprocessed_Data/Preprocessed_Data_imaging.csv')
OUT_BASE    = os.path.join(BASE, 'local_data/Results_v2/_figures')
os.makedirs(OUT_BASE, exist_ok=True)

# ── Feature name mapping ───────────────────────────────────────────────────
# (Same mapping as before, extended)
FEATURE_NAMES = {
    '34-0.0':              'Year of birth (Age)',
    '21022-0.0':           'Age at recruitment',
    '21003-0.0':           'Age at assessment',
    '400-0.0':             'Reaction time (mean)',
    '2188-0.0_c1':         'Diabetes (Dr diagnosed)',
    '2188-0.0_c0':         'Diabetes (Dr diagnosed, alt)',
    '137-0.0':             'Number of medications',
    '6142-0.0_pos':        'Employment: employed',
    '6142-0.3_pos':        'Employment: retired',
    '20110-0.4_pos':       "Mother: dementia/Alzheimer's",
    '20110-0.1_neg':       'Mother: no major illness',
    '20110-0.6_pos':       "Mother: Parkinson's disease",
    '20107-0.4_pos':       "Father: dementia/Alzheimer's",
    '20111-0.4_pos':       "Father: dementia/Alzheimer's",
    '6146-0.1_pos':        'Physical activity: walking',
    '20023-0.0':           'Reaction time: correct matches',
    '30710-0.0':           'C-reactive protein (CRP)',
    '1090-0.0':            'Trail making: time to complete',
    '3526-0.0':            "Mother's age at death",
    '2296-0.0_c0':         'Hearing difficulty',
    '135-0.0':             'Non-cancer illness count',
    '1835-0.0_c0':         'Number in household',
    '2492-0.0_c1':         'Taking Rx medications',
    '1080-0.0':            'Cognition: match time',
    '1060-0.0':            'Cognition: match SD',
    '30650-0.0':           'Aspartate aminotransferase',
    '1588-0.0':            'Beer/cider intake (weekly)',
    '2178-0.0_c3':         'High BP (Dr diagnosed)',
    '1200-0.0_c0':         'Sleep duration (hours)',
    '20127-0.0':           'Neuroticism score',
    '30040-0.0':           'Mean platelet volume',
    '3064-0.0':            'Haemoglobin concentration',
    '1070-0.0':            'Reaction time (mean, alt)',
    '4642-0.0_c1':         'Wears glasses/contacts',
    '23112-0.0':           'Leg impedance (whole body)',
    '30510-0.0':           'Cystatin C',
    '3637-0.0_c6':         'Vascular/heart problem',
    '30670-0.0':           'Glucose',
    '30180-0.0':           'Eosinophil count',
    '23111-0.0':           'Leg impedance (right)',
    '23115-0.0':           'Leg impedance (left)',
    '20111-0.4_pos':       "Father: dementia/Alzheimer's",
    '20111-0.1_neg':       'Father: no major illness',
    '2443-0.0_c0':         'Diabetes (alt)',
    '2473-0.0_c1':         'Other physical activity',
    '738-0.0_c0':          'Household income',
    '1618-0.0_c0':         'Alcohol drinker status',
    '1488-0.0':            'Days/week moderate activity',
    '2784-0.0_c0':         'High cholesterol',
    '2694-0.0_c0':         'High cholesterol (alt)',
    '46-0.0':              'Hand grip strength (left)',
    '47-0.0':              'Hand grip strength (right)',
    '30050-0.0':           'Mean corpuscular volume',
    '6138-0.0_c6':         'Education: college degree',
    '30270-0.0':           'Red blood cell count',
    '30620-0.0':           'Total bilirubin',
    '23107-0.0':           'Body fat percentage',
    '3062-0.0':            'Haemoglobin (alt)',
    '30200-0.0':           'White blood cell count',
    '2139-0.0':            'Computer use duration',
    '398-0.0':             'Reaction time (ms)',
    '30830-0.0':           'SHBG',
    '3063-0.0':            'Haemoglobin (alt2)',
    '2986-0.0_c2':         'Alcohol intake frequency',
    '20015-0.0':           'Days/week walked 10+ min',
    '26413-0.0':           'Grey matter volume (total)',
    '1807-0.0':            'Days/week walked (alt)',
    '2060-0.0_c0':         'Risk taking tendency',
    '2149-0.0':            'Falls in last year',
    '30000-0.0':           'Mean corpuscular volume (alt)',
    '30010-0.0':           'Mean corpuscular haemoglobin',
    '30020-0.0':           'Haemoglobin (alt3)',
    '30030-0.0':           'Haematocrit',
    '30070-0.0':           'Mean reticulocyte volume',
    '30080-0.0':           'Platelet count (alt)',
    '30090-0.0':           'Mean platelet volume (alt)',
    '30100-0.0':           'Eosinophil percentage',
    '30110-0.0':           'Basophil count',
    '30120-0.0':           'Lymphocyte count',
    '30130-0.0':           'Monocyte count',
    '30140-0.0':           'Neutrophil count',
    '30190-0.0':           'Reticulocyte count',
    '30210-0.0':           'RBC distribution width',
    '30240-0.0':           'Reticulocyte percentage',
    '30280-0.0':           'Platelet count',
    '30290-0.0':           'Platelet crit',
    '30300-0.0':           'Mean platelet volume',
    '30520-0.0':           'Lipoprotein A',
    '30530-0.0':           'Rheumatoid factor',
    '30630-0.0':           'Total bilirubin (alt)',
    '30640-0.0':           'GGT',
    '30660-0.0':           'Direct bilirubin',
    '30720-0.0':           'Cystatin C (alt)',
    '30730-0.0':           'IGF-1',
    '30740-0.0':           'Lipoprotein A (alt)',
    '30750-0.0':           'HbA1c',
    '30760-0.0':           'HDL cholesterol',
    '30770-0.0':           'Vitamin D',
    '30780-0.0':           'LDL direct',
    '30790-0.0':           'Total cholesterol',
    '30810-0.0':           'Phosphate',
    '30840-0.0':           'Oestradiol',
    '30850-0.0':           'Testosterone',
    '30860-0.0':           'Vitamin D (alt)',
    '30870-0.0':           'Rheumatoid factor (alt)',
    '30880-0.0':           'Vitamin D (alt2)',
    '4079-0.0':            'Diastolic BP',
    '4080-0.0':            'Systolic BP',
    '4501-0.0_c2':         'Puzzles/games per week',
    '4526-0.0_c1':         'Happiness with health',
    '4570-0.0_c4':         'Puzzles/games (alt)',
    '6164-0.1_pos':        'Walking for pleasure',
    '6164-0.3_pos':        'Leisure: sports club',
    '6150-0.1_pos':        'Physical activity: light DIY',
    '6144-0.0_pos':        'Physical activity: strenuous sport',
    '6145-0.1_pos':        'Physical activity: other',
    '6155-0.1_pos':        'Leisure: other group',
    '6177-0.1_pos':        'Leisure: adult education',
    '6147-0.2_pos':        'Walking: frequency',
    '6147-0.3_pos':        'Walking: frequency (alt)',
    '6148-0.0_pos':        'Physical activity: none',
    '6162-0.0_pos':        'Leisure: religious group',
    '6160-0.0_pos':        'Leisure: none',
    '6160-0.3_pos':        'Leisure: sports club (alt)',
    '1319-0.0':            'Tobacco smoke outside home',
    '1279-0.0':            'Risk taking (alt)',
    '1269-0.0':            'Tobacco smoke at home',
    '1289-0.0':            'Days/week moderate (alt)',
    '1299-0.0':            'Days/week vigorous (alt)',
    '1438-0.0':            'Bread intake',
    '1458-0.0':            'Cereal intake',
    '1568-0.0':            'Red wine intake (weekly)',
    '1578-0.0':            'White wine intake (weekly)',
    '1598-0.0':            'Fortified wine (weekly)',
    '1608-0.0':            'Spirits intake (weekly)',
    '1628-0.0_c1':         'Alcohol vs 10yr ago',
    '1873-0.0':            'Number of full pregnancies',
    '1883-0.0':            'Age at first live birth',
    '1920-0.0_c1':         'Nervous feelings',
    '2000-0.0_c0':         'Depression/low mood',
    '2016-0.0_c0':         'Smoking status',
    '2050-0.0_c0':         'Depressed mood frequency',
    '2050-0.0_c4':         'Depressed mood (alt)',
    '2070-0.0_c0':         'Tenseness/restlessness',
    '2080-0.0_c3':         'Alcohol intake frequency',
    '2100-0.0_c0':         'Unenthusiasm/disinterest',
    '22033-0.0':           'Leg fat-free mass (L)',
    '22034-0.0':           'Trunk fat percentage',
    '22038-0.0':           'Trunk fat-free mass',
    '22039-0.0':           'Trunk predicted mass',
    '22040-0.0':           'Trunk fat mass',
    '2257-0.0_c1':         'Full pregnancies (alt)',
    '2267-0.0_c0':         'Children fathered',
    '2277-0.0':            'Age first live birth (alt)',
    '23100-0.0':           'Whole body fat mass',
    '23108-0.0':           'Body impedance',
    '23116-0.0':           'Arm impedance (R)',
    '23119-0.0':           'Arm predicted mass (L)',
    '23123-0.0':           'Arm predicted mass (R)',
    '23127-0.0':           'Trunk fat-free mass (alt)',
    '2624-0.0_c6':         'Other exercises frequency',
    '2654-0.0_c0':         'Vascular/heart problem (alt)',
    '2664-0.0_c1':         'Alcohol frequency (alt)',
    '2674-0.0_c1':         'Alcohol frequency (alt2)',
    '2724-0.0_c1':         'Stair climbing frequency',
    '2814-0.0_c1':         'Yearly fruit intake',
    '2834-0.0_c2':         'Smoking: current vs past',
    '2956-0.0_c0':         'Full pregnancies (alt2)',
    '3393-0.0_c2':         'Population density',
    '3591-0.0_c2':         'Age at last live birth',
    '3637-0.0_c3':         'Vascular/heart (alt)',
    '3720-0.0_c2':         'Age first depression',
    '4041-0.0_c0':         'Time to complete round',
    '4548-0.0_c1':         'Employment: employed (alt)',
    '4581-0.0_c1':         'General happiness',
    '4717-0.0_c1':         'Insulin medication',
    '4728-0.0_c1':         'Cholesterol lowering meds',
    '5452-0.0_c0':         'Cognitive test score',
    '5463-0.0_c0':         'Cognitive test (alt)',
    '5959-0.0_c0':         'Cognitive test (alt2)',
    '6138-0.0_c0':         'Education: none',
    '6138-0.0_c2':         'Education: secondary',
    '6138-0.0_c3':         'Education: vocational',
    '6138-0.0_c4':         'Education: A-levels',
    '6138-0.0_c5':         'Education: other',
    '6142-0.0_neg':        'Employment: unemployed',
    '6142-0.1_pos':        'Employment: self-employed',
    '6142-0.2_pos':        'Employment: student',
    '6146-0.0_pos':        'Physical activity: none (alt)',
    '6146-0.2_pos':        'Physical activity: cycling',
    '6147-0.0_pos':        'Walking: none',
    '6147-0.1_pos':        'Walking: once/week',
    '6149-0.0_pos':        'Physical activity: swimming',
    '6149-0.1_pos':        'Physical activity: tennis',
    '6151-0.1_pos':        'Leisure: pub/social',
    '6152-0.0_pos':        'Leisure: none (alt)',
    '6152-0.1_pos':        'Leisure: visiting friends',
    '6153-0.0_pos':        'Leisure: none (alt2)',
    '6154-0.0_pos':        'Leisure: gardening',
    '6154-0.1_pos':        'Leisure: DIY',
    '6156-0.0_pos':        'Leisure: computer games',
    '6159-0.0_pos':        'Leisure: internet',
    '6164-0.2_pos':        'Walking: moderate pace',
    '6179-0.0_pos':        'Leisure: none (alt3)',
    '971-0.0_c1':          'Transport: car/motor',
    '971-0.0_c2':          'Transport: walk',
    '971-0.0_c6':          'Transport: public',
    '924-0.0_c0':          'Usual walking pace',
    '894-0.0':             'Duration of moderate activity',
    '884-0.0':             'Days/week moderate activity',
    '874-0.0':             'Duration of walks',
    '864-0.0':             'Days/week vigorous activity',
    '845-0.0':             'Days/week moderate (alt)',
    '904-0.0':             'Days/week vigorous (alt)',
    '943-0.0_c2':          'Stair climbing frequency',
    '981-0.0_c0':          'Duration walking',
    '991-0.0_c6':          'Leisure: social activity',
    '1011-0.0_c6':         'Wears glasses/contacts',
    '102-0.0':             'Pulse rate',
    '1100-0.0_c1':         'Walking pace (alt)',
    '1110-0.0_c0':         'Smoking: never',
    '1130-0.0_c0':         'Sleep duration (alt)',
    '1150-0.0_c0':         'Sleep duration (alt2)',
    '1160-0.0':            'Sleep duration (alt3)',
    '1180-0.0_c0':         'Number in household (alt)',
    '2060-0.0_c1':         'Risk taking (alt2)',
    '1920-0.0_c0':         'Nervous feelings (alt)',
    '129-0.0':             'Non-cancer illness count (alt)',
    '130-0.0':             'BMI comparative',
    '136-0.0':             'Self-reported: diabetes',
    '399-0.0':             'Reaction time (alt)',
    '403-0.0':             'Forced expiratory volume',
    '1528-0.0':            'Days/week vigorous activity',
    '1548-0.0_c0':         'Alcohol frequency (alt)',
    '1558-0.0_c5':         'Alcohol: red wine',
    '1697-0.0_c1':         'Age started smoking',
    '1707-0.0_c1':         'Cigarettes per day',
    '1717-0.0_c5':         'Smoking: pack-years',
    '1727-0.0_c2':         'Smoking: age stopped',
    '1737-0.0':            'Smoking: current vs never',
    '1747-0.0_c5':         'Alcohol: intake frequency',
    '1757-0.0_c0':         'Facial aging',
    '1797-0.0_c0':         'Depressed mood (alt)',
    '1960-0.0_c0':         'Anxious feelings',
    '1970-0.0_c0':         'Worrier/anxious',
    '1980-0.0_c0':         'Mood swings',
    '1990-0.0_c0':         'Miserableness',
    '20001-0.0_pos':       'Self-reported: cancer',
    '20001-0.1_pos':       'Self-reported: heart disease',
    '20008-0.0':           'Year of first diagnosis',
    '20009-0.0':           'Age at recruitment (alt)',
    '20011-0.0':           'Full pregnancies count',
    '20022-0.0':           'Age at assessment (alt)',
    '2010-0.0_c1':         'Tiredness/lethargy',
    '20115-0.0_c0':        'Country of birth: UK',
    '20116-0.0_c0':        'Smoking status (alt)',
    '20117-0.0_c1':        'Alcohol: never',
    '20118-0.0_c0':        'Country of birth: non-UK',
    '20119-0.0_c1':        'Languages spoken',
    '20122-0.0_c0':        'Workplace: mainly indoor',
    '20123-0.0_c0':        'Workplace: mixed',
    '20124-0.0_c0':        'Workplace: mainly outdoor',
    '20125-0.0_c0':        'Workplace: always outdoor',
    '20126-0.0_c4':        'Workplace: shift work',
    '20126-0.0_c6':        'Workplace: night shifts',
    '2020-0.0_c1':         'Loneliness',
    '2030-0.0_c1':         'Irritability',
    '2040-0.0_c1':         'Sensitivity/hurt feelings',
    '2110-0.0_c0':         'Nervous feelings (alt)',
    '2129-0.0_c1':         'Mobile phone use',
    '2159-0.0_c2':         'Computer use (alt)',
    '2178-0.0_c0':         'High BP (alt)',
    '2178-0.0_c1':         'High BP (alt2)',
    '22032-0.0_c1':        'Leg fat mass (R)',
    '22035-0.0_c0':        'Arm fat mass (L)',
    '22036-0.0_c0':        'Arm fat-free mass (L)',
    '22037-0.0':           'Leg fat percentage (R)',
    '2207-0.0_c1':         'Age started oral contraceptive',
    '2217-0.0':            'Age at menopause',
    '2227-0.0_c0':         'Number of live births',
    '2237-0.0_c0':         'Number of stillbirths',
    '2247-0.0_c0':         'Ever had hysterectomy',
    '23050-0.0_c0':        'Body composition: weight',
    '2306-0.0_c1':         'Body composition: fat mass',
    '23099-0.0':           'Body fat percentage (alt)',
    '23101-0.0':           'Trunk fat mass (alt)',
    '23102-0.0':           'Trunk fat-free mass (alt)',
    '23104-0.0':           'Arm fat mass (alt)',
    '23105-0.0':           'Arm fat-free mass',
    '23106-0.0':           'Body fat mass (alt)',
    '23109-0.0':           'Body water mass',
    '23110-0.0':           'Body water mass (alt)',
    '23113-0.0':           'Leg fat mass (whole)',
    '23114-0.0':           'Leg fat-free mass (whole)',
    '23117-0.0':           'Arm fat mass (whole)',
    '23118-0.0':           'Arm fat-free mass (whole)',
    '23120-0.0':           'Whole body predicted mass',
    '23121-0.0':           'Trunk predicted mass (alt)',
    '23122-0.0':           'Leg predicted mass (whole)',
    '23124-0.0':           'Arm impedance (L)',
    '23125-0.0':           'Arm impedance (whole)',
    '23126-0.0':           'Leg impedance (L, alt)',
    '23128-0.0':           'Leg predicted mass (R)',
    '23129-0.0':           'Leg predicted mass (L)',
    '23130-0.0':           'Arm predicted mass (R, alt)',
    '2316-0.0_c0':         'Body composition: impedance',
    '2335-0.0_c0':         'Body composition: BMI',
    '2375-0.0_c0':         'Body composition: waist',
    '2385-0.0_c0':         'Body composition: hip',
    '2395-0.0_c0':         'Body composition: fat % (alt)',
    '2415-0.0_c0':         'Number of operations',
    '2443-0.0_c1':         'Diabetes (alt2)',
    '2453-0.0_c0':         'Cancer (Dr diagnosed)',
    '2463-0.0_c0':         'Fracture (Dr diagnosed)',
    '2473-0.0_c0':         'Other physical activity (alt)',
    '2492-0.0_c0':         'Taking Rx medications (alt)',
    '25094-0.0':           'MRI: Ventral DC vol (L)',
    '25242-0.0':           'MRI: Choroid plexus (L)',
    '25263-0.0':           'MRI: Corpus callosum mid-post',
    '25285-0.0':           'MRI: Corpus callosum central',
    '25286-0.0':           'MRI: Corpus callosum mid-ant',
    '25287-0.0':           'MRI: Corpus callosum anterior',
    '25291-0.0':           'MRI: Optic chiasm vol',
    '25292-0.0':           'MRI: Optic chiasm vol (R)',
    '25293-0.0':           'MRI: Optic chiasm vol (L)',
    '25296-0.0':           'MRI: Pons volume',
    '25305-0.0':           'MRI: Hippocampus volume (R)',
    '25309-0.0':           'MRI: Amygdala volume (L)',
    '25312-0.0':           'MRI: Amygdala volume (R)',
    '25332-0.0':           'MRI: Putamen volume (R)',
    '25333-0.0':           'MRI: Putamen volume (L)',
    '25334-0.0':           'MRI: Pallidum volume (L)',
    '25339-0.0':           'MRI: Caudate volume (L)',
    '25340-0.0':           'MRI: Accumbens volume (L)',
    '25342-0.0':           'MRI: Accumbens volume (R)',
    '25349-0.0':           'MRI: Lateral ventricle (L)',
    '25377-0.0':           'MRI: Inf lat ventricle (L)',
    '25379-0.0':           'MRI: Inf lat ventricle (R)',
    '25394-0.0':           'MRI: Cerebellum WM vol (L)',
    '25407-0.0':           'MRI: Cerebellum cortex (L)',
    '25416-0.0':           'MRI: Cerebellum cortex (R)',
    '25427-0.0':           'MRI: Pallidum volume (L, alt)',
    '25430-0.0':           'MRI: Pallidum volume (R, alt)',
    '25433-0.0':           'MRI: Thalamus proper (L)',
    '25443-0.0':           'MRI: Ventral DC vol (L, alt)',
    '25463-0.0':           'MRI: Ventral DC vol (R)',
    '25478-0.0':           'MRI: Choroid plexus (R)',
    '25482-0.0':           'MRI: Accumbens volume (R, alt)',
    '25483-0.0':           'MRI: Choroid plexus (L)',
    '25485-0.0':           'MRI: Accumbens volume (L, alt)',
    '25489-0.0':           'MRI: Hippocampus volume (L)',
    '25494-0.0':           'MRI: Amygdala volume (L, alt)',
    '25521-0.0':           'MRI: Caudate volume (R, alt)',
    '25522-0.0':           'MRI: Caudate volume (L, alt)',
    '25534-0.0':           'MRI: Caudate volume (L, alt2)',
    '25547-0.0':           'MRI: Lateral ventricle (R)',
    '25557-0.0':           'MRI: Lateral ventricle (R, alt)',
    '25565-0.0':           'MRI: Lateral ventricle (L, alt)',
    '25576-0.0':           'MRI: Inf lat ventricle (R, alt)',
    '25597-0.0':           'MRI: Cerebellum WM vol (L, alt)',
    '25598-0.0':           'MRI: Cerebellum WM vol (R, alt)',
    '25602-0.0':           'MRI: Cerebellum cortex vol',
    '25603-0.0':           'MRI: Cerebellum cortex (R, alt)',
    '25612-0.0':           'MRI: Cerebellum WM vol (R)',
    '25615-0.0':           'MRI: Cerebellum WM vol (L)',
    '25629-0.0':           'MRI: Ventral diencephalon vol',
    '25630-0.0':           'MRI: Ventral DC vol (R, alt)',
    '25639-0.0':           'MRI: Choroid plexus vol (R)',
    '25678-0.0':           'MRI: WM hypointensities vol',
    '25698-0.0':           'MRI: WM hypointensities (L)',
    '25706-0.0':           'MRI: WM hypointensities (R)',
    '25707-0.0':           'MRI: Non-WM hypointensities (L)',
    '25711-0.0':           'MRI: Non-WM hypointensities (L, alt)',
    '25716-0.0':           'MRI: Non-WM hypointensities (R, alt)',
    '25723-0.0':           'MRI: Subcortical grey (L)',
    '25724-0.0':           'MRI: Subcortical grey (R)',
    '25734-0.0':           'MRI: Total cortical GM vol',
    '25735-0.0':           'MRI: Total cortical WM vol',
    '25783-0.0':           'MRI: Mean thickness (L, alt)',
    '25792-0.0':           'MRI: Mean thickness (R, alt)',
    '25804-0.0':           'MRI: Mean thickness (L)',
    '25805-0.0':           'MRI: Mean thickness (L, alt2)',
    '25809-0.0':           'MRI: Mean thickness (R)',
    '25813-0.0':           'MRI: Superior frontal thick (R)',
    '25825-0.0':           'MRI: Superior frontal thick',
    '25826-0.0':           'MRI: Sup frontal thick (R, alt)',
    '25855-0.0':           'MRI: Rostral mid frontal thick',
    '25862-0.0':           'MRI: Rostral mid frontal (R)',
    '25863-0.0':           'MRI: Caudal mid frontal thick',
    '25871-0.0':           'MRI: Pars opercularis (L)',
    '25886-0.0':           'MRI: Pars triangularis',
    '25887-0.0':           'MRI: Pars orbitalis (L)',
    '25888-0.0':           'MRI: Lateral orbitofrontal',
    '25889-0.0':           'MRI: Medial orbitofrontal',
    '25890-0.0':           'MRI: Medial orbitofrontal (R)',
    '25900-0.0':           'MRI: Precentral (R)',
    '25908-0.0':           'MRI: Paracentral (L)',
    '25910-0.0':           'MRI: Paracentral (R)',
    '25919-0.0':           'MRI: Postcentral (L)',
    '25926-0.0':           'MRI: Supramarginal (R)',
    '25927-0.0':           'MRI: Superior parietal',
    '25928-0.0':           'MRI: Sup parietal (R)',
    '25929-0.0':           'MRI: Inferior parietal',
    '25930-0.0':           'MRI: Inf parietal (R)',
    # MRI hippocampal subfields and other key regions
    '12651-0.0':           'MRI: L Hippocampal Subiculum Vol',
    '26555-0.0':           'MRI: R Subiculum Volume',
    '26643-0.0':           'MRI: L Subiculum Volume',
    '26644-0.0':           'MRI: R Subiculum Volume (alt)',
    '26645-0.0':           'MRI: L Presubiculum Volume',
    '26647-0.0':           'MRI: R Presubiculum Volume',
    '26604-0.0':           'MRI: L Whole Hippocampus Vol',
    '26606-0.0':           'MRI: R Whole Hippocampus Vol',
    '26611-0.0':           'MRI: L Hippocampal Tail Vol',
    '26612-0.0':           'MRI: R Hippocampal Tail Vol',
    '26614-0.0':           'MRI: L Subiculum (alt)',
    '26620-0.0':           'MRI: L Presubiculum (alt)',
    '26621-0.0':           'MRI: R Presubiculum (alt)',
    '26622-0.0':           'MRI: L Parasubiculum',
    '26623-0.0':           'MRI: R Parasubiculum',
    '26624-0.0':           'MRI: L HATA',
    '26626-0.0':           'MRI: R HATA',
    '26627-0.0':           'MRI: L Fimbria',
    '26631-0.0':           'MRI: L Hippocampal Fissure',
    '26632-0.0':           'MRI: R Hippocampal Fissure',
    '26633-0.0':           'MRI: L HP-Amygdala Trans Area',
    '26634-0.0':           'MRI: R HP-Amygdala Trans Area',
    '26635-0.0':           'MRI: L Whole Hippocampus (alt)',
    '26637-0.0':           'MRI: R Whole Hippocampus (alt)',
    '26639-0.0':           'MRI: LGN Thalamus (L)',
    '26640-0.0':           'MRI: LGN Thalamus (R)',
    '26641-0.0':           'MRI: MGN Thalamus (L)',
    '26642-0.0':           'MRI: MGN Thalamus (R)',
    '26649-0.0':           'MRI: L Entorhinal Cortex',
    '26660-0.0':           'MRI: L Molecular Layer HP',
    '26661-0.0':           'MRI: R Molecular Layer HP',
    '26662-0.0':           'MRI: L GC-ML-DG',
    '26663-0.0':           'MRI: R GC-ML-DG',
    '26562-0.0':           'MRI: L CA1 Volume',
    '26563-0.0':           'MRI: R CA1 Volume',
    '26568-0.0':           'MRI: L CA3 Volume',
    '26570-0.0':           'MRI: R CA3 Volume',
    '26577-0.0':           'MRI: L CA4 Volume',
    '26586-0.0':           'MRI: L Molecular Layer HP (alt)',
    '26600-0.0':           'MRI: L GC-ML-DG (alt)',
    '26602-0.0':           'MRI: R GC-ML-DG (alt)',
    # DTI / white matter tracts
    '26504-0.0':           'MRI: FA Tract (L)',
    '26511-0.0':           'MRI: MD Tract (L)',
    '26512-0.0':           'MRI: MD Tract (R)',
    '26528-0.0':           'MRI: ICVF Tract (L)',
    '26538-0.0':           'MRI: Vol Thalamic Rad (L)',
    '26539-0.0':           'MRI: Vol Thalamic Rad (R)',
    '26546-0.0':           'MRI: FA Post Thalamic Rad',
    '26551-0.0':           'MRI: MD Cingulum Cing (L)',
    '26692-0.0':           'MRI: WM FA Tract (L)',
    '26738-0.0':           'MRI: Mean OD Tract (L)',
    '26761-0.0':           'MRI: Vol CST (L)',
    '26764-0.0':           'MRI: Vol CST (R)',
    '26779-0.0':           'MRI: Vol SLF (L)',
    '26784-0.0':           'MRI: Vol SLF (R)',
    '26786-0.0':           'MRI: Vol ILF (L)',
    '26796-0.0':           'MRI: Vol ILF (R)',
    '26805-0.0':           'MRI: Vol IFOF (L)',
    '26817-0.0':           'MRI: Vol IFOF (R)',
    '26837-0.0':           'MRI: Vol UF (L)',
    '26863-0.0':           'MRI: Vol UF (R)',
    '26880-0.0':           'MRI: Vol Forceps Minor',
    '26885-0.0':           'MRI: Vol Forceps Major',
    '26904-0.0':           'MRI: Vol ATR (L)',
    '26948-0.0':           'MRI: Vol STR (L)',
    '26960-0.0':           'MRI: Vol STR (R)',
    '26968-0.0':           'MRI: Vol SFO (L)',
    '27003-0.0':           'MRI: Vol SFO (R)',
    '27077-0.0':           'MRI: Vol CC Body',
    '27095-0.0':           'MRI: Vol CC Splenium',
    '27143-0.0':           'MRI: Vol FX/ST (L)',
    '27154-0.0':           'MRI: Vol FX/ST (R)',
    '27181-0.0':           'MRI: ICVF CST (L)',
    '27196-0.0':           'MRI: OD SLF (L)',
    '27198-0.0':           'MRI: OD SLF (R)',
    '27199-0.0':           'MRI: OD ILF (L)',
    '27201-0.0':           'MRI: OD ILF (R)',
    '27211-0.0':           'MRI: ISOVF ATR (L)',
    '27216-0.0':           'MRI: ISOVF STR (L)',
    '27222-0.0':           'MRI: ISOVF SFO (L)',
    '27249-0.0':           'MRI: FA CC Body',
    '27270-0.0':           'MRI: FA FX/ST (L)',
    '27279-0.0':           'MRI: FA FX/ST (R)',
    '27289-0.0':           'MRI: MD CC Genu',
    '27294-0.0':           'MRI: MD CC Body',
    '27303-0.0':           'MRI: MD CC Splenium',
    '27308-0.0':           'MRI: MD ATR (L)',
    '27339-0.0':           'MRI: MD STR (L)',
    '27342-0.0':           'MRI: MD STR (R)',
    '27373-0.0':           'MRI: MD SFO (L)',
    '27400-0.0':           'MRI: MD FX/ST (L)',
    '27417-0.0':           'MRI: OD CST (L)',
    '27427-0.0':           'MRI: OD CST (R)',
    '27457-0.0':           'MRI: OD ATR (L)',
    '27463-0.0':           'MRI: OD ATR (R)',
    '27470-0.0':           'MRI: OD STR (L)',
    '27473-0.0':           'MRI: OD STR (R)',
    '27476-0.0':           'MRI: OD SFO (L)',
    '27490-0.0':           'MRI: OD SFO (R)',
    '27513-0.0':           'MRI: OD FX/ST (L)',
    '27514-0.0':           'MRI: OD FX/ST (R)',
    '27515-0.0':           'MRI: ISOVF CST (L)',
    '27518-0.0':           'MRI: ISOVF CST (R)',
    '27525-0.0':           'MRI: ISOVF SLF (L)',
    '27526-0.0':           'MRI: ISOVF SLF (R)',
    '27528-0.0':           'MRI: ISOVF ILF (L)',
    '27549-0.0':           'MRI: ISOVF IFOF (L)',
    '27602-0.0':           'MRI: ISOVF UF (L)',
    '27611-0.0':           'MRI: ISOVF UF (R)',
    '27643-0.0':           'MRI: ISOVF FX/ST (L)',
    '27658-0.0':           'MRI: ICVF ATR (L)',
    '27664-0.0':           'MRI: ICVF ATR (R)',
    '27674-0.0':           'MRI: ICVF STR (L)',
    '27679-0.0':           'MRI: ICVF STR (R)',
    '27680-0.0':           'MRI: ICVF SFO (L)',
    '27695-0.0':           'MRI: ICVF SFO (R)',
    '27697-0.0':           'MRI: ICVF FX/ST (L)',
    '27759-0.0':           'MRI: ICVF CC Body',
    '27761-0.0':           'MRI: ICVF CC Splenium',
    '27417-0.0':           'MRI: OD CST (L)',
    '27457-0.0':           'MRI: OD ATR (L)',
    '27463-0.0':           'MRI: OD ATR (R)',
    '27470-0.0':           'MRI: OD STR (L)',
}

# ── Target definitions ─────────────────────────────────────────────────────
ALL_TARGETS = ['DM_full', 'DM_5yrs', 'DM_10yrs', 'AD_full', 'AD_5yrs', 'AD_10yrs']
# Deploy strategy: all targets use DM_full's selected features
NON_MRI_FEATURES = ['34-0.0', '400-0.0', '137-0.0', '2188-0.0_c1', '30710-0.0',
                    '20110-0.4_pos', '1090-0.0', '6142-0.3_pos', '20023-0.0', '3526-0.0']
MRI_FEATURES = ['34-0.0', '400-0.0', '137-0.0', '12651-0.0', '2188-0.0_c1',
                '30710-0.0', '20110-0.4_pos', '6142-0.3_pos', '26643-0.0', '1200-0.0_c0']

TARGET_CONFIG = {
    'DM_full':   {'status_col': 'dementia_status', 'years_col': 'dementia_years', 'window': None},
    'DM_5yrs':   {'status_col': 'dementia_status', 'years_col': 'dementia_years', 'window': 5},
    'DM_10yrs':  {'status_col': 'dementia_status', 'years_col': 'dementia_years', 'window': 10},
    'AD_full':   {'status_col': 'AD_status',       'years_col': 'AD_years',       'window': None},
    'AD_5yrs':   {'status_col': 'AD_status',       'years_col': 'AD_years',       'window': 5},
    'AD_10yrs':  {'status_col': 'AD_status',       'years_col': 'AD_years',       'window': 10},
}

LGB_PARAMS = {
    'n_estimators': 500, 'max_depth': 15, 'num_leaves': 10,
    'subsample': 0.7, 'learning_rate': 0.01, 'colsample_bytree': 0.7,
    'objective': 'binary', 'metric': 'auc', 'is_unbalance': True,
    'verbosity': -1, 'random_state': 2022, 'n_jobs': -1,
}


def get_feature_name(fid):
    return FEATURE_NAMES.get(fid, fid)


def build_target(df, target_name):
    """Build binary target vector with time window right-censoring."""
    cfg = TARGET_CONFIG[target_name]
    status = df[cfg['status_col']].values
    years = df[cfg['years_col']].values
    y = status.copy().astype(float)
    if cfg['window'] is not None:
        # Right-censor: events beyond the window become controls (y=0)
        y[(years > cfg['window']) & (status == 1)] = 0
        y[(years > cfg['window']) & (years != cfg['window'])] = 0
        # More precisely: keep only events within the window
        mask = (status == 1) & (years > cfg['window'])
        y[mask] = 0
    return y


def generate_shap_for_experiment(data_path, selected_features, out_subdir, exp_label):
    """Generate SHAP beeswarm plots for all 6 targets."""
    out_dir = os.path.join(OUT_BASE, out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    print(f'\n{"="*60}')
    print(f'Generating SHAP for: {exp_label}')
    print(f'Data: {data_path}')
    print(f'Features: {selected_features}')
    print(f'Output: {out_dir}')
    print(f'{"="*60}')

    # Load data
    print('Loading data...')
    df = pd.read_csv(data_path)

    # Verify all features exist
    missing = [f for f in selected_features if f not in df.columns]
    if missing:
        print(f'WARNING: Missing features: {missing}')
        selected_features = [f for f in selected_features if f in df.columns]
    print(f'Using {len(selected_features)} features')

    # Exclude baseline stroke
    stroke_mask = (df['stroke_status'] == 1) & (df['stroke_years'] < 0)
    df_clean = df[~stroke_mask].copy()
    print(f'Samples after stroke exclusion: {len(df_clean)}')

    X = df_clean[selected_features].copy()
    # Handle any remaining NaN
    X = X.fillna(X.median())

    # Build feature label mapping (keep IDs for training, rename only for SHAP plotting)
    feature_labels = [get_feature_name(f) for f in selected_features]
    id_to_label = dict(zip(selected_features, feature_labels))

    for target_name in ALL_TARGETS:
        print(f'\n--- {target_name} ---')
        y = build_target(df_clean, target_name)
        n_pos = int(y.sum())
        print(f'  Positive cases: {n_pos} ({n_pos/len(y)*100:.3f}%)')

        if n_pos < 10:
            print(f'  SKIPPING: too few positive cases ({n_pos})')
            continue

        # Train on a stratified subset for SHAP efficiency (use up to 100K samples)
        n_samples = min(len(X), 100000)
        if len(X) > n_samples:
            # Stratified subsample
            from sklearn.model_selection import train_test_split
            X_sub, _, y_sub, _ = train_test_split(
                X, y, train_size=n_samples, stratify=y, random_state=2022
            )
        else:
            X_sub, y_sub = X, y

        print(f'  Training on {len(X_sub)} samples (pos={int(y_sub.sum())})...')

        # Train LightGBM
        model = LGBMClassifier(**LGB_PARAMS)
        model.fit(X_sub, y_sub)

        # SHAP
        print(f'  Computing SHAP values...')
        # Use a subset for SHAP explanation (SHAP is O(n²) for TreeExplainer)
        n_explain = min(len(X_sub), 5000)
        if n_explain < len(X_sub):
            X_explain = X_sub.sample(n=n_explain, random_state=2022)
        else:
            X_explain = X_sub

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_explain)

        # If shap returns a list (for binary classification), take class 1
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Rename X_explain columns to human-readable names for plotting
        X_plot = X_explain.rename(columns=id_to_label)

        # Plot beeswarm
        fig, ax = plt.subplots(figsize=(16, 5.5))
        shap.summary_plot(
            shap_values, X_plot,
            plot_type='dot',
            show=False,
            max_display=10,
            color_bar=True,
            alpha=0.6,
        )
        ax = plt.gca()
        ax.set_ylabel('Selected Predictors', fontsize=16, weight='bold')
        ax.set_xlabel('SHAP Value (impact on model output)', fontsize=13, weight='bold')
        ax.set_title(f'SHAP Beeswarm — {exp_label}\n{target_name} (n={n_pos} cases, AUC≈{model.score(X_sub, y_sub):.3f})',
                     fontsize=13, weight='bold')
        ax.tick_params(axis='y', labelsize=12)
        plt.tight_layout()

        outpath = os.path.join(out_dir, f'shap_{target_name}.png')
        fig.savefig(outpath, dpi=200)
        plt.close(fig)
        print(f'  Saved: {outpath}')

    print(f'\nDone with {exp_label}!')


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
if __name__ == '__main__':
    import time
    t0 = time.time()

    # Experiment 1: Non-MRI
    generate_shap_for_experiment(
        data_path=DATA_CLIN,
        selected_features=NON_MRI_FEATURES,
        out_subdir='shap_non_mri',
        exp_label='Non-MRI (1076 Clinical Features)'
    )

    # Experiment 2: +MRI
    generate_shap_for_experiment(
        data_path=DATA_IMG,
        selected_features=MRI_FEATURES,
        out_subdir='shap_mri',
        exp_label='+MRI (3250 Clinical + MRI Features)'
    )

    print(f'\n{"="*60}')
    print(f'Total time: {(time.time()-t0)/60:.1f} minutes')
    print(f'All SHAP plots saved to:')
    print(f'  {OUT_BASE}/shap_non_mri/')
    print(f'  {OUT_BASE}/shap_mri/')

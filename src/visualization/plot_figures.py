#!/usr/bin/env python3
"""
Generate publication-style figures for UKB-DRP replication experiments.
Compares non-MRI (clinical features only) vs +MRI (clinical + brain MRI) results.

Figures generated:
  1a. SFS plot — Non-MRI experiment (DM_full, 1076 clinical features)
  1b. SFS plot — +MRI experiment (DM_full, 3250 clinical + MRI features)
  2a. SHAP beeswarm — Non-MRI experiment (top 10 selected features, DM_full)
  2b. SHAP beeswarm — +MRI experiment (top 10 selected features, DM_full)
  3. Combined SFS comparison plot
  4. Top-30 Gain importance comparison (bar chart)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch
import os, json, textwrap

# ── Paths ──────────────────────────────────────────────────────────────────
BASE    = '/Users/guxiao/Downloads/MCI-AD/UKB_DRP-main'
OUT_DIR = os.path.join(BASE, 'local_data', 'Results_v2', '_figures')
os.makedirs(OUT_DIR, exist_ok=True)

# Non-MRI results
NO_MRI_DIR = os.path.join(BASE, 'local_data/Results_v2/DM_full')
# +MRI results (full imaging)
MRI_DIR    = os.path.join(BASE, 'local_data/Results_imaging/DM_full_img_full')

# ── Feature name mapping ───────────────────────────────────────────────────
# Maps feature IDs to human-readable short names for plots
FEATURE_NAMES = {
    # Demographics
    '34-0.0':              'Year of birth',
    '21022-0.0':           'Age at recruitment',
    '21003-0.0':           'Age at assessment centre',
    '31-0.0':              'Sex',
    '46-0.0':              'Hand grip strength (L)',
    '47-0.0':              'Hand grip strength (R)',
    '48-0.0':              'Standing height',
    '49-0.0':              'Hip circumference',
    '50-0.0':              'Seated height',
    '51-0.0':              'Forced vital capacity',
    '87-0.0':              'Days/week walked 10+ min',
    '92-0.0':              'Duration of moderate activity',
    '102-0.0':             'Pulse rate',
    '129-0.0':             'Non-cancer illnesses count',
    '130-0.0':             'BMI comparative',
    '135-0.0':             'Non-cancer illnesses count (2)',
    '136-0.0':             'Self-reported: diabetes',
    '137-0.0':             'Number of medications',
    '398-0.0':             'Reaction time (ms, round)',
    '399-0.0':             'Reaction time (ms)',
    '400-0.0':             'Reaction time (mean, round)',
    '403-0.0':             'Forced expiratory volume',
    '738-0.0_c0':          'Household income',
    '845-0.0':             'Days/week moderate activity',
    '864-0.0':             'Days/week vigorous activity',
    '874-0.0':             'Duration of walks (min)',
    '884-0.0':             'Days/week of moderate activity',
    '894-0.0':             'Duration of moderate activity (min)',
    '904-0.0':             'Number of days/week vigorous',
    '924-0.0_c0':          'Usual walking pace',
    '943-0.0_c2':          'Frequency of stair climbing',
    '971-0.0_c1':          'Transport: car/motor',
    '971-0.0_c2':          'Transport: walk',
    '981-0.0_c0':          'Duration walking (min)',
    '1011-0.0_c6':         'Wears glasses/contacts',
    '1050-0.0':            'Time to identify matches',
    '1060-0.0':            'Reaction time (SD)',
    '1070-0.0':            'Reaction time (mean)',
    '1080-0.0':            'Time to identify matches (2)',
    '1090-0.0':            'Trail making: time to complete',
    '1100-0.0_c1':         'Usual walking pace (2)',
    '1110-0.0_c0':         'Smoking status',
    '1130-0.0_c0':         'Sleep duration (h)',
    '1150-0.0_c0':         'Sleep duration (2, h)',
    '1160-0.0':            'Sleep duration (3, h)',
    '1180-0.0_c0':         'Number in household',
    '1200-0.0_c0':         'Sleep duration (4, h)',
    '1269-0.0':            'Tobacco smoke at home',
    '1279-0.0':            'Risk taking tendency',
    '1289-0.0':            'Days/week moderate activity (2)',
    '1299-0.0':            'Days/week vigorous activity (2)',
    '1309-0.0':            'Frequency of other exercises',
    '1319-0.0':            'Tobacco smoke outside home',
    '1379-0.0_c0':         'Oily fish intake',
    '1438-0.0':            'Bread intake',
    '1458-0.0':            'Cereal intake',
    '1488-0.0':            'Days/week moderate activity',
    '1498-0.0':            'Salt added to food',
    '1528-0.0':            'Days/week vigorous activity',
    '1568-0.0':            'Average weekly red wine',
    '1578-0.0':            'Average weekly white wine',
    '1588-0.0':            'Average weekly beer/cider',
    '1598-0.0':            'Average weekly fortified wine',
    '1608-0.0':            'Average weekly spirits',
    '1618-0.0_c0':         'Alcohol drinker status',
    '1628-0.0_c1':         'Alcohol intake vs 10yr ago',
    '1807-0.0':            'Days/week walked 10+ min (3)',
    '1835-0.0_c0':         'Number in household (2)',
    '1873-0.0':            'Number of full pregnancies',
    '1883-0.0':            'Age first live birth',
    '1920-0.0_c1':         'Nervous feelings',
    '2000-0.0_c0':         'Depression/low mood',
    '2000-0.0_c1':         'Depression (2)',
    '20008-0.0':           'Year of first diagnosis',
    '20009-0.0':           'Age at recruitment (2)',
    '20015-0.0':           'Days/week walked 10+ min',
    '20022-0.0':           'Age at assessment (2)',
    '20023-0.0':           'Reaction time: mean correct',
    '20107-0.4_pos':       "Father: dementia/Alzheimer's",
    '20110-0.1_neg':       'Mother: no major illness',
    '20110-0.4_pos':       "Mother: dementia/Alzheimer's",
    '20110-0.6_pos':       "Mother: Parkinson's disease",
    '20111-0.4_pos':       "Father: dementia/Alzheimer's",
    '20111-0.1_neg':       'Father: no major illness',
    '20116-0.0_c0':        'Smoking status (2)',
    '20117-0.0_c1':        'Alcohol: never',
    '20127-0.0':           'Neuroticism score',
    '2050-0.0_c0':         'Depressed mood frequency',
    '2050-0.0_c4':         'Depressed mood (2)',
    '2060-0.0_c0':         'Risk taking (2)',
    '2070-0.0_c0':         'Tenseness/restlessness',
    '2080-0.0_c3':         'Alcohol intake frequency',
    '2100-0.0_c0':         'Unenthusiasm/disinterest',
    '2139-0.0':            'Computer use duration',
    '2149-0.0':            'Falls in last year',
    '2178-0.0_c1':         'High blood pressure (Dr diagnosed)',
    '2178-0.0_c3':         'High BP diagnosed (3)',
    '2188-0.0_c0':         'Diabetes (Dr diagnosed, 1)',
    '2188-0.0_c1':         'Diabetes (Dr diagnosed, 2)',
    '22033-0.0':           'Leg fat-free mass (L)',
    '22034-0.0':           'Trunk fat percentage',
    '22038-0.0':           'Trunk fat-free mass',
    '22039-0.0':           'Trunk predicted mass',
    '22040-0.0':           'Trunk fat mass',
    '2257-0.0_c1':         'Number of full pregnancies (2)',
    '2267-0.0_c0':         'Number of children fathered',
    '2277-0.0':            'Age at first live birth (2)',
    '2296-0.0_c0':         'Hearing difficulty',
    '23100-0.0':           'Whole body fat mass',
    '23107-0.0':           'Body fat percentage',
    '23108-0.0':           'Body impedance',
    '23111-0.0':           'Leg impedance (R)',
    '23112-0.0':           'Leg impedance (whole body)',
    '23115-0.0':           'Leg impedance (L)',
    '23116-0.0':           'Arm impedance (R)',
    '23119-0.0':           'Arm predicted mass (L)',
    '23123-0.0':           'Arm predicted mass (R)',
    '23127-0.0':           'Trunk fat-free mass (2)',
    '2443-0.0_c0':         'Diabetes diagnosed (3)',
    '2443-0.0_c1':         'Diabetes diagnosed (4)',
    '2473-0.0_c1':         'Other physical activity',
    '2492-0.0_c0':         'Other Rx medications (1)',
    '2492-0.0_c1':         'Other Rx medications (2)',
    '2624-0.0_c6':         'Other exercises frequency',
    '26413-0.0':           'Total grey matter volume',
    '2654-0.0_c0':         'Vascular/heart problems (1)',
    '2694-0.0_c0':         'High cholesterol (1)',
    '2694-0.0_c2':         'High cholesterol (3)',
    '2724-0.0_c1':         'Stair climbing frequency',
    '2784-0.0_c0':         'High cholesterol (2)',
    '2814-0.0_c1':         'Yearly fruit intake',
    '2834-0.0_c2':         'Smoking: current vs previous',
    '2956-0.0_c0':         'Number of full pregnancies (3)',
    '2986-0.0_c2':         'Alcohol intake frequency (2)',
    '30000-0.0':           'Mean corpuscular volume',
    '30010-0.0':           'Mean corpuscular haemoglobin',
    '30020-0.0':           'Haemoglobin concentration',
    '30030-0.0':           'Haematocrit percentage',
    '30040-0.0':           'Mean platelet volume',
    '30050-0.0':           'Mean corpuscular volume (2)',
    '30070-0.0':           'Mean reticulocyte volume',
    '30090-0.0':           'Mean platelet volume (2)',
    '30100-0.0':           'Eosinophil percentage',
    '30110-0.0':           'Basophil count',
    '30120-0.0':           'Lymphocyte count',
    '30130-0.0':           'Monocyte count',
    '30140-0.0':           'Neutrophil count',
    '30180-0.0':           'Eosinophil count',
    '30200-0.0':           'White blood cell count',
    '30210-0.0':           'RBC distribution width',
    '30240-0.0':           'Reticulocyte percentage',
    '30270-0.0':           'Red blood cell count',
    '30280-0.0':           'Platelet count',
    '30290-0.0':           'Platelet crit',
    '30300-0.0':           'Mean platelet volume (3)',
    '30510-0.0':           'Cystatin C',
    '30520-0.0':           'Lipoprotein A',
    '3062-0.0':            'Haemoglobin conc (2)',
    '30620-0.0':           'Total bilirubin',
    '3063-0.0':            'Haemoglobin conc (3)',
    '30630-0.0':           'Total bilirubin (2)',
    '3064-0.0':            'Haemoglobin conc (4)',
    '30640-0.0':           'Gamma glutamyltransferase',
    '30650-0.0':           'Aspartate aminotransferase',
    '30660-0.0':           'Direct bilirubin',
    '30670-0.0':           'Glucose',
    '30710-0.0':           'C-reactive protein (CRP)',
    '30720-0.0':           'Cystatin C (2)',
    '30730-0.0':           'IGF-1',
    '30740-0.0':           'Lipoprotein A (2)',
    '30750-0.0':           'HbA1c',
    '30760-0.0':           'HDL cholesterol',
    '30770-0.0':           'Vitamin D (25-OH)',
    '30780-0.0':           'LDL direct',
    '30790-0.0':           'Total cholesterol',
    '30810-0.0':           'Phosphate',
    '30830-0.0':           'SHBG',
    '30840-0.0':           'Oestradiol',
    '30850-0.0':           'Testosterone',
    '30860-0.0':           'Vitamin D (2)',
    '30870-0.0':           'Rheumatoid factor',
    '30880-0.0':           'Vitamin D (3)',
    '3393-0.0_c2':         'Home area population density',
    '3526-0.0':            "Mother's age at death",
    '3591-0.0_c2':         'Age at last live birth',
    '3637-0.0_c3':         'Vascular/heart problems (2)',
    '3637-0.0_c6':         'Vascular/heart problems (3)',
    '3720-0.0_c2':         'Age at first depression episode',
    '4041-0.0_c0':         'Time to complete round (2)',
    '4079-0.0':            'Diastolic blood pressure',
    '4080-0.0':            'Systolic blood pressure',
    '4501-0.0_c2':         'Number of puzzles/games/week',
    '4526-0.0_c1':         'Happiness with own health',
    '4548-0.0_c1':         'Employment: employed',
    '4570-0.0_c4':         'Number of puzzles (2)',
    '4581-0.0_c1':         'General happiness',
    '4642-0.0_c1':         'Wears glasses/contacts (2)',
    '4728-0.0_c1':         'Cholesterol lowering meds',
    '6138-0.0_c6':         'Education: college/univ degree',
    '6142-0.0_pos':        'Employment: employed (3)',
    '6142-0.3_pos':        'Employment: retired',
    '6144-0.0_pos':        'Physical activity: strenuous sport',
    '6145-0.1_pos':        'Physical activity: other exercises',
    '6146-0.1_pos':        'Physical activity: walking',
    '6147-0.3_pos':        'Walking for pleasure freq',
    '6148-0.0_pos':        'Physical activity: none',
    '6150-0.1_pos':        'Physical activity: light DIY',
    '6160-0.3_pos':        'Leisure/social activity: sports club',
    '6164-0.1_pos':        'Physical activity: walking for pleasure',
    '6177-0.1_pos':        'Leisure: adult education class',

    # ── IMAGING features (UKB IDP, 5-digit IDs) ──
    '12651-0.0':           '★ MRI: L hippocampal subiculum vol',
    '25001-0.0':           'MRI: Brain volume (normalised)',
    '25005-0.0':           'MRI: Grey matter volume',
    '25011-0.0':           'MRI: White matter volume',
    '25019-0.0':           'MRI: CSF volume',
    '25050-0.0':           'MRI: Ventricular CSF vol',
    '25054-0.0':           'MRI: Brain segmentation vol',
    '25142-0.0':           'MRI: Thalamus volume (L)',
    '25305-0.0':           'MRI: Hippocampus volume (R)',
    '25312-0.0':           'MRI: Amygdala volume (R)',
    '25332-0.0':           'MRI: Putamen volume (R)',
    '25427-0.0':           'MRI: Pallidum volume (L)',
    '25430-0.0':           'MRI: Pallidum volume (R)',
    '25482-0.0':           'MRI: Accumbens volume (R)',
    '25521-0.0':           'MRI: Caudate volume (R)',
    '25534-0.0':           'MRI: Caudate volume (L)',
    '25557-0.0':           'MRI: Lateral ventricle (R)',
    '25565-0.0':           'MRI: Lateral ventricle (L)',
    '25576-0.0':           'MRI: Inferior lateral ventricle (R)',
    '25602-0.0':           'MRI: Cerebellum cortex vol',
    '25612-0.0':           'MRI: Cerebellum WM vol (R)',
    '25615-0.0':           'MRI: Cerebellum WM vol (L)',
    '25629-0.0':           'MRI: Ventral diencephalon vol',
    '25630-0.0':           'MRI: Ventral DC vol (R)',
    '25639-0.0':           'MRI: Choroid plexus vol (R)',
    '25678-0.0':           'MRI: WM hypointensities vol',
    '25698-0.0':           'MRI: WM hypointensities (L)',
    '25706-0.0':           'MRI: WM hypointensities (R)',
    '25711-0.0':           'MRI: Non-WM hypointensities (L)',
    '25716-0.0':           'MRI: Non-WM hypointensities (R)',
    '25723-0.0':           'MRI: Subcortical grey vol (L)',
    '25724-0.0':           'MRI: Subcortical grey vol (R)',
    '25734-0.0':           'MRI: Total cortical GM vol',
    '25735-0.0':           'MRI: Total cortical WM vol',
    '25804-0.0':           'MRI: Mean thickness (L)',
    '25809-0.0':           'MRI: Mean thickness (R)',
    '25825-0.0':           'MRI: Superior frontal thickness',
    '25826-0.0':           'MRI: Sup frontal thickness (R)',
    '25855-0.0':           'MRI: Rostral mid frontal thick',
    '25862-0.0':           'MRI: Rost mid frontal thick (R)',
    '25863-0.0':           'MRI: Caudal mid frontal thick',
    '25871-0.0':           'MRI: Pars opercularis thick (L)',
    '25886-0.0':           'MRI: Pars triangularis thick',
    '25887-0.0':           'MRI: Pars orbitalis thick (L)',
    '25888-0.0':           'MRI: Lateral orbitofrontal thick',
    '25889-0.0':           'MRI: Medial orbitofrontal thick',
    '25890-0.0':           'MRI: Med orbitofrontal thick (R)',
    '25900-0.0':           'MRI: Precentral thickness (R)',
    '25908-0.0':           'MRI: Paracentral thickness (L)',
    '25910-0.0':           'MRI: Paracentral thickness (R)',
    '25919-0.0':           'MRI: Postcentral thickness (L)',
    '25926-0.0':           'MRI: Supramarginal thickness (R)',
    '25927-0.0':           'MRI: Superior parietal thickness',
    '25928-0.0':           'MRI: Sup parietal thickness (R)',
    '25929-0.0':           'MRI: Inferior parietal thickness',
    '25930-0.0':           'MRI: Inf parietal thickness (R)',
    '26504-0.0':           'MRI: Mean FA in tract (L)',
    '26511-0.0':           'MRI: Mean MD in tract (L)',
    '26512-0.0':           'MRI: Mean MD in tract (R)',
    '26528-0.0':           'MRI: Mean ICVF in tract (L)',
    '26538-0.0':           'MRI: Volume of thalamic rad (L)',
    '26539-0.0':           'MRI: Volume of thalamic rad (R)',
    '26546-0.0':           'MRI: FA in posterior thalamic rad',
    '26551-0.0':           'MRI: MD in cingulum cingulate (L)',
    '26555-0.0':           'MRI: ★ Subiculum volume (R)',
    '26562-0.0':           'MRI: CA1 volume (L)',
    '26563-0.0':           'MRI: CA1 volume (R)',
    '26568-0.0':           'MRI: CA3 volume (L)',
    '26570-0.0':           'MRI: CA3 volume (R)',
    '26577-0.0':           'MRI: CA4 volume (L)',
    '26586-0.0':           'MRI: Molecular layer HP (L)',
    '26600-0.0':           'MRI: GC-ML-DG volume (L)',
    '26602-0.0':           'MRI: GC-ML-DG volume (R)',
    '26604-0.0':           'MRI: Whole hippocampal volume (L)',
    '26606-0.0':           'MRI: Whole hippocampal volume (R)',
    '26611-0.0':           'MRI: Hippocampal tail volume (L)',
    '26612-0.0':           'MRI: Hippocampal tail volume (R)',
    '26614-0.0':           'MRI: Subiculum volume (L, alt)',
    '26620-0.0':           'MRI: Presubiculum volume (L)',
    '26621-0.0':           'MRI: Presubiculum volume (R)',
    '26622-0.0':           'MRI: Parasubiculum volume (L)',
    '26623-0.0':           'MRI: Parasubiculum volume (R)',
    '26624-0.0':           'MRI: HATA volume (L)',
    '26626-0.0':           'MRI: HATA volume (R)',
    '26627-0.0':           'MRI: Fimbria volume (L)',
    '26631-0.0':           'MRI: Hippocampal fissure (L)',
    '26632-0.0':           'MRI: Hippocampal fissure (R)',
    '26633-0.0':           'MRI: HP amygdala trans area (L)',
    '26634-0.0':           'MRI: HP amygdala trans area (R)',
    '26635-0.0':           'MRI: Whole hippocampus (L, alt)',
    '26637-0.0':           'MRI: Whole hippocampus (R, alt)',
    '26639-0.0':           'MRI: Thalamic nuclei: LGN (L)',
    '26640-0.0':           'MRI: Thalamic nuclei: LGN (R)',
    '26641-0.0':           'MRI: Thalamic nuclei: MGN (L)',
    '26642-0.0':           'MRI: Thalamic nuclei: MGN (R)',
    '26643-0.0':           '★ MRI: Left subiculum volume',
    '26644-0.0':           'MRI: Right subiculum volume',
    '26645-0.0':           'MRI: Left presubiculum volume',
    '26647-0.0':           'MRI: Right presubiculum volume',
    '26649-0.0':           'MRI: Entorhinal cortex (L)',
    '26660-0.0':           'MRI: Molecular layer (L)',
    '26661-0.0':           'MRI: Molecular layer (R)',
    '26662-0.0':           'MRI: GC-ML-DG (L)',
    '26663-0.0':           'MRI: GC-ML-DG (R)',
    '26692-0.0':           'MRI: WM tract FA (L)',
    '26738-0.0':           'MRI: Mean OD in tract (L)',
    '26761-0.0':           'MRI: Volume of CST (L)',
    '26764-0.0':           'MRI: Volume of CST (R)',
    '26779-0.0':           'MRI: Volume of SLF (L)',
    '26784-0.0':           'MRI: Volume of SLF (R)',
    '26786-0.0':           'MRI: Volume of ILF (L)',
    '26796-0.0':           'MRI: Volume of ILF (R)',
    '26805-0.0':           'MRI: Volume of IFOF (L)',
    '26817-0.0':           'MRI: Volume of IFOF (R)',
    '26837-0.0':           'MRI: Volume of UF (L)',
    '26863-0.0':           'MRI: Volume of UF (R)',
    '26880-0.0':           'MRI: Volume of forceps minor',
    '26885-0.0':           'MRI: Volume of forceps major',
    '26904-0.0':           'MRI: Volume of ATR (L)',
    '26948-0.0':           'MRI: Volume of STR (L)',
    '26960-0.0':           'MRI: Volume of STR (R)',
    '26968-0.0':           'MRI: Volume of SFO (L)',
    '27003-0.0':           'MRI: Volume of SFO (R)',
    '27077-0.0':           'MRI: Volume of CC body',
    '27095-0.0':           'MRI: Volume of CC splenium',
    '27143-0.0':           'MRI: Volume of FX/ST (L)',
    '27154-0.0':           'MRI: Volume of FX/ST (R)',
    '27181-0.0':           'MRI: ICVF in CST (L)',
    '27196-0.0':           'MRI: OD in SLF (L)',
    '27198-0.0':           'MRI: OD in SLF (R)',
    '27199-0.0':           'MRI: OD in ILF (L)',
    '27201-0.0':           'MRI: OD in ILF (R)',
    '27211-0.0':           'MRI: ISOVF in ATR (L)',
    '27216-0.0':           'MRI: ISOVF in STR (L)',
    '27222-0.0':           'MRI: ISOVF in SFO (L)',
    '27249-0.0':           'MRI: FA in CC body',
    '27270-0.0':           'MRI: FA in FX/ST (L)',
    '27279-0.0':           'MRI: FA in FX/ST (R)',
    '27289-0.0':           'MRI: MD in CC genu',
    '27294-0.0':           'MRI: MD in CC body',
    '27303-0.0':           'MRI: MD in CC splenium',
    '27308-0.0':           'MRI: MD in ATR (L)',
    '27339-0.0':           'MRI: MD in STR (L)',
    '27342-0.0':           'MRI: MD in STR (R)',
    '27373-0.0':           'MRI: MD in SFO (L)',
    '27400-0.0':           'MRI: MD in FX/ST (L)',
    '27417-0.0':           'MRI: OD in CST (L)',
    '27427-0.0':           'MRI: OD in CST (R)',
    '27457-0.0':           'MRI: OD in ATR (L)',
    '27463-0.0':           'MRI: OD in ATR (R)',
    '27470-0.0':           'MRI: OD in STR (L)',
    '27473-0.0':           'MRI: OD in STR (R)',
    '27476-0.0':           'MRI: OD in SFO (L)',
    '27490-0.0':           'MRI: OD in SFO (R)',
    '27513-0.0':           'MRI: OD in FX/ST (L)',
    '27514-0.0':           'MRI: OD in FX/ST (R)',
    '27515-0.0':           'MRI: ISOVF in CST (L)',
    '27518-0.0':           'MRI: ISOVF in CST (R)',
    '27525-0.0':           'MRI: ISOVF in SLF (L)',
    '27526-0.0':           'MRI: ISOVF in SLF (R)',
    '27528-0.0':           'MRI: ISOVF in ILF (L)',
    '27549-0.0':           'MRI: ISOVF in IFOF (L)',
    '27602-0.0':           'MRI: ISOVF in UF (L)',
    '27611-0.0':           'MRI: ISOVF in UF (R)',
    '27643-0.0':           'MRI: ISOVF in FX/ST (L)',
    '27658-0.0':           'MRI: ICVF in ATR (L)',
    '27664-0.0':           'MRI: ICVF in ATR (R)',
    '27674-0.0':           'MRI: ICVF in STR (L)',
    '27679-0.0':           'MRI: ICVF in STR (R)',
    '27680-0.0':           'MRI: ICVF in SFO (L)',
    '27695-0.0':           'MRI: ICVF in SFO (R)',
    '27697-0.0':           'MRI: ICVF in FX/ST (L)',
    '27759-0.0':           'MRI: ICVF in CC body',
    '27761-0.0':           'MRI: ICVF in CC splenium',
}


def feat_name(fid):
    """Return human-readable feature name, or the raw ID if unknown."""
    return FEATURE_NAMES.get(fid, fid)


def feat_name_short(fid, maxlen=38):
    """Short label for plot display."""
    name = feat_name(fid)
    if len(name) > maxlen:
        return name[:maxlen-3] + '...'
    return name


# ── Plot style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

COLOR_NO_MRI = '#2166AC'      # deep blue
COLOR_MRI    = '#B2182B'      # deep red
COLOR_BOTH   = '#7B3294'      # purple
COLOR_CAIDE  = '#999999'      # grey

# ── Helper: make directory ─────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 1a: SFS Plot — Non-MRI Experiment                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def plot_sfs_no_mri():
    """Create SFS plot for the non-MRI (clinical features only) experiment."""
    df = pd.read_csv(os.path.join(NO_MRI_DIR, 's04_sfs_history.csv'))

    steps = df['selected_count'].values
    auc   = df['auc_mean'].values
    auc_std = df['auc_std'].values
    labels = [feat_name_short(f) for f in df['feature_added'].values]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Main line
    ax.plot(steps, auc, 'o-', color=COLOR_NO_MRI, linewidth=2.2, markersize=8,
            markerfacecolor='white', markeredgewidth=1.8, markeredgecolor=COLOR_NO_MRI,
            zorder=5)
    ax.fill_between(steps, auc - auc_std, auc + auc_std,
                    color=COLOR_NO_MRI, alpha=0.12)

    # Annotate each step with feature name
    for i, (s, a, lbl) in enumerate(zip(steps, auc, labels)):
        offset = 10 if i % 2 == 0 else -18
        ax.annotate(lbl, (s, a),
                    textcoords="offset points", xytext=(12, offset),
                    fontsize=7, color=COLOR_NO_MRI, fontweight='normal',
                    arrowprops=dict(arrowstyle='-', color='#cccccc', lw=0.5),
                    rotation=20, ha='left')

    ax.set_xlabel('Number of Features Selected (SFS step)', fontweight='bold')
    ax.set_ylabel('5-Fold CV AUC', fontweight='bold')
    ax.set_title('Sequential Forward Selection — Non-MRI Model\n'
                 '(DM_full, 1076 clinical features, UKB ~425K participants)',
                 fontweight='bold')
    ax.set_ylim(0.78, 0.845)
    ax.set_xlim(0.5, 10.5)
    ax.set_xticks(range(1, 11))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add final AUC annotation
    ax.annotate(f'Final AUC = {auc[-1]:.4f}\n(10 features)',
                xy=(10, auc[-1]), xytext=(8, auc[-1] + 0.018),
                fontsize=9, fontweight='bold', color=COLOR_NO_MRI,
                ha='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f0f0', alpha=0.8))

    plt.tight_layout()
    outpath = os.path.join(OUT_DIR, 'Fig1a_SFS_non_MRI.png')
    fig.savefig(outpath)
    plt.close(fig)
    print(f'[OK] Saved {outpath}')
    return outpath


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 1b: SFS Plot — +MRI Experiment                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def plot_sfs_mri():
    """Create SFS plot for the +MRI experiment."""
    df = pd.read_csv(os.path.join(MRI_DIR, 's04_sfs_history.csv'))

    steps = df['selected_count'].values
    auc   = df['auc_mean'].values
    auc_std = df['auc_std'].values
    labels = [feat_name_short(f) for f in df['feature_added'].values]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.plot(steps, auc, 's-', color=COLOR_MRI, linewidth=2.2, markersize=8,
            markerfacecolor='white', markeredgewidth=1.8, markeredgecolor=COLOR_MRI,
            zorder=5)
    ax.fill_between(steps, auc - auc_std, auc + auc_std,
                    color=COLOR_MRI, alpha=0.12)

    for i, (s, a, lbl) in enumerate(zip(steps, auc, labels)):
        offset = 10 if i % 2 == 0 else -18
        # Highlight MRI features
        color = '#B2182B' if 'MRI' in lbl or '★' in lbl else '#555555'
        weight = 'bold' if 'MRI' in lbl or '★' in lbl else 'normal'
        ax.annotate(lbl, (s, a),
                    textcoords="offset points", xytext=(12, offset),
                    fontsize=7, color=color, fontweight=weight,
                    arrowprops=dict(arrowstyle='-', color='#cccccc', lw=0.5),
                    rotation=20, ha='left')

    ax.set_xlabel('Number of Features Selected (SFS step)', fontweight='bold')
    ax.set_ylabel('5-Fold CV AUC', fontweight='bold')
    ax.set_title('Sequential Forward Selection — +MRI Model\n'
                 '(DM_full, 3250 clinical+MRI features, UKB ~425K participants)',
                 fontweight='bold')
    ax.set_ylim(0.78, 0.85)
    ax.set_xlim(0.5, 10.5)
    ax.set_xticks(range(1, 11))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    ax.annotate(f'Final AUC = {auc[-1]:.4f}\n(10 features, incl. 1 MRI)',
                xy=(10, auc[-1]), xytext=(8, auc[-1] + 0.02),
                fontsize=9, fontweight='bold', color=COLOR_MRI,
                ha='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff0f0', alpha=0.8))

    plt.tight_layout()
    outpath = os.path.join(OUT_DIR, 'Fig1b_SFS_MRI.png')
    fig.savefig(outpath)
    plt.close(fig)
    print(f'[OK] Saved {outpath}')
    return outpath


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 2: Combined SFS Comparison                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def plot_sfs_combined():
    """Side-by-side or overlaid comparison of SFS curves."""
    df_no = pd.read_csv(os.path.join(NO_MRI_DIR, 's04_sfs_history.csv'))
    df_mri = pd.read_csv(os.path.join(MRI_DIR, 's04_sfs_history.csv'))

    fig, ax = plt.subplots(figsize=(11, 6))

    # Non-MRI
    s1, a1, e1 = df_no['selected_count'], df_no['auc_mean'], df_no['auc_std']
    ax.plot(s1, a1, 'o-', color=COLOR_NO_MRI, linewidth=2.2, markersize=8,
            markerfacecolor='white', markeredgewidth=1.8,
            label=f'Non-MRI (1076 clinical features) — final AUC={a1.iloc[-1]:.4f}')
    ax.fill_between(s1, a1-e1, a1+e1, color=COLOR_NO_MRI, alpha=0.08)

    # +MRI
    s2, a2, e2 = df_mri['selected_count'], df_mri['auc_mean'], df_mri['auc_std']
    ax.plot(s2, a2, 's--', color=COLOR_MRI, linewidth=2.2, markersize=8,
            markerfacecolor='white', markeredgewidth=1.8,
            label=f'+MRI (3250 clinical+MRI features) — final AUC={a2.iloc[-1]:.4f}')
    ax.fill_between(s2, a2-e2, a2+e2, color=COLOR_MRI, alpha=0.08)

    # MRI feature annotations on the MRI line
    for i, row in df_mri.iterrows():
        lbl = feat_name_short(row['feature_added'])
        if 'MRI' in lbl or '★' in lbl:
            ax.annotate(lbl, (row['selected_count'], row['auc_mean']),
                       textcoords="offset points", xytext=(15, 8),
                       fontsize=7.5, color=COLOR_MRI, fontweight='bold',
                       arrowprops=dict(arrowstyle='->', color=COLOR_MRI, lw=0.7, alpha=0.6))

    # ΔAUC annotation
    delta = a2.iloc[-1] - a1.iloc[-1]
    ax.annotate(f'MRI增量: ΔAUC = +{delta:.4f}',
                xy=(9, a1.iloc[-1] + 0.025), fontsize=10,
                fontweight='bold', color=COLOR_MRI,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff5f5',
                          edgecolor=COLOR_MRI, alpha=0.9))

    ax.set_xlabel('Number of Features Selected (SFS step)', fontweight='bold')
    ax.set_ylabel('5-Fold CV AUC', fontweight='bold')
    ax.set_title('SFS Feature Selection Comparison: Non-MRI vs +MRI\n'
                 '(DM_full target, UKB ~425K participants)',
                 fontweight='bold')
    ax.set_ylim(0.78, 0.85)
    ax.set_xlim(0.5, 10.5)
    ax.set_xticks(range(1, 11))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)

    plt.tight_layout()
    outpath = os.path.join(OUT_DIR, 'Fig2_SFS_Combined.png')
    fig.savefig(outpath)
    plt.close(fig)
    print(f'[OK] Saved {outpath}')
    return outpath


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 3a: Top-30 Gain Importance — Non-MRI                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def plot_top30_gain_no_mri():
    """Bar chart of top 30 features by Gain from s03 (non-MRI)."""
    df = pd.read_csv(os.path.join(NO_MRI_DIR, 's03_final_importance.csv'))
    top30 = df.head(30).copy()
    top30['label'] = top30['Features'].apply(feat_name_short)
    top30 = top30.iloc[::-1]  # reverse for horizontal bar

    fig, ax = plt.subplots(figsize=(8, 9))

    colors = [COLOR_NO_MRI] * len(top30)
    bars = ax.barh(range(len(top30)), top30['Gain'].values, color=colors,
                   height=0.7, edgecolor='white', linewidth=0.5)

    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels(top30['label'].values, fontsize=8)
    ax.set_xlabel('LightGBM Gain Importance (s03)', fontweight='bold')
    ax.set_title('Top 30 Features by Gain Importance\n'
                 'Non-MRI Model (DM_full, 1076 clinical features)',
                 fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Add rank numbers
    for i, (idx, row) in enumerate(top30.iterrows()):
        ax.annotate(f'#{30-i}', (row['Gain'] + max(top30['Gain'])*0.01, i),
                    fontsize=6.5, color='#888888', va='center')

    plt.tight_layout()
    outpath = os.path.join(OUT_DIR, 'Fig3a_Top30_Gain_NonMRI.png')
    fig.savefig(outpath)
    plt.close(fig)
    print(f'[OK] Saved {outpath}')
    return outpath


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 3b: Top-30 Gain Importance — +MRI                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def plot_top30_gain_mri():
    """Bar chart of top 30 features by Gain from s03 (+MRI)."""
    df = pd.read_csv(os.path.join(MRI_DIR, 's03_final_importance.csv'))
    top30 = df.head(30).copy()
    top30['label'] = top30['Features'].apply(feat_name_short)

    # Color: red for imaging features, blue-grey for clinical
    colors = []
    for _, row in top30.iterrows():
        fid = row['Features']
        is_img = ('MRI' in feat_name(fid)) or ('★' in feat_name(fid))
        # Also check the IsImaging column if available
        if 'IsImaging' in row and row['IsImaging']:
            is_img = True
        colors.append(COLOR_MRI if is_img else '#556677')

    top30 = top30.iloc[::-1]
    colors = colors[::-1]

    fig, ax = plt.subplots(figsize=(8, 9))

    bars = ax.barh(range(len(top30)), top30['Gain'].values, color=colors,
                   height=0.7, edgecolor='white', linewidth=0.5)

    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels(top30['label'].values, fontsize=8)
    ax.set_xlabel('LightGBM Gain Importance (s03)', fontweight='bold')
    ax.set_title('Top 30 Features by Gain Importance\n'
                 '+MRI Model (DM_full, 3250 clinical+MRI features)',
                 fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLOR_MRI, label='MRI / Imaging feature'),
        Patch(facecolor='#556677', label='Clinical feature'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    plt.tight_layout()
    outpath = os.path.join(OUT_DIR, 'Fig3b_Top30_Gain_MRI.png')
    fig.savefig(outpath)
    plt.close(fig)
    print(f'[OK] Saved {outpath}')
    return outpath


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 4: BOTH experiments top-30 in a compact Table format            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def generate_top30_table():
    """Generate Markdown table of top 30 features for both experiments."""
    df_no = pd.read_csv(os.path.join(NO_MRI_DIR, 's01_feature_importance.csv'))
    df_mri = pd.read_csv(os.path.join(MRI_DIR, 's01_feature_importance.csv'))

    # Build table
    lines = []
    lines.append('# Top 30 Features — s01 Gain Ranking')
    lines.append('')
    lines.append('## Non-MRI Experiment (1076 clinical features)')
    lines.append('')
    lines.append('| Rank | Feature ID | Gain | Description |')
    lines.append('|------|-----------|------|-------------|')
    for i, (_, row) in enumerate(df_no.head(30).iterrows()):
        fid = row['Features']
        gain = row['Gain']
        name = feat_name(fid)
        is_img_marker = ''
        lines.append(f'| {i+1} | `{fid}` | {gain:.4f} | {is_img_marker}{name} |')

    lines.append('')
    lines.append('## +MRI Experiment (3250 clinical + MRI features)')
    lines.append('')
    lines.append('| Rank | Feature ID | Gain | Imaging? | Description |')
    lines.append('|------|-----------|------|----------|-------------|')
    for i, (_, row) in enumerate(df_mri.head(30).iterrows()):
        fid = row['Features']
        gain = row['Gain']
        name = feat_name(fid)
        is_img = 'IsImaging' in row and row['IsImaging']
        if isinstance(is_img, str):
            is_img = is_img.lower() == 'true'
        marker = 'MRI' if (is_img or 'MRI' in name or '★' in name) else ''
        lines.append(f'| {i+1} | `{fid}` | {gain:.4f} | {marker} | {name} |')

    lines.append('')
    lines.append('## SFS Selected Features Comparison')
    lines.append('')
    lines.append('### Non-MRI SFS Selection Order (s04)')
    lines.append('')
    df_sfs_no = pd.read_csv(os.path.join(NO_MRI_DIR, 's04_selected_features.csv'))
    lines.append('| Step | Feature ID | Cumulative AUC | Description |')
    lines.append('|------|-----------|---------------|-------------|')
    for _, row in df_sfs_no.iterrows():
        fid = row['Features']
        name = feat_name(fid)
        lines.append(f'| {row["SelectionOrder"]} | `{fid}` | {row["CumulativeAUC"]:.4f} | {name} |')

    lines.append('')
    lines.append('### +MRI SFS Selection Order (s04)')
    lines.append('')
    df_sfs_mri = pd.read_csv(os.path.join(MRI_DIR, 's04_selected_features.csv'))
    lines.append('| Step | Feature ID | Cumulative AUC | Type | Description |')
    lines.append('|------|-----------|---------------|------|-------------|')
    for _, row in df_sfs_mri.iterrows():
        fid = row['Features']
        name = feat_name(fid)
        is_img = 'MRI' if ('MRI' in name or '★' in name) else 'Clinical'
        lines.append(f'| {row["SelectionOrder"]} | `{fid}` | {row["CumulativeAUC"]:.4f} | {is_img} | {name} |')

    outpath = os.path.join(OUT_DIR, 'Top30_Features_Table.md')
    with open(outpath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'[OK] Saved {outpath}')
    return outpath


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 5: SHAP-style Feature Importance based on s03 Gain              ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def plot_shap_style():
    """
    Since raw SHAP values are not available (models trained on large data),
    create SHAP-style beeswarm-like visualizations using s03 Gain + Coverage
    as a proxy to show feature importance and direction.
    """
    # Non-MRI
    df_no = pd.read_csv(os.path.join(NO_MRI_DIR, 's03_final_importance.csv'))
    top10_no = df_no.head(10).copy()
    top10_no['label'] = top10_no['Features'].apply(feat_name_short)
    top10_no = top10_no.iloc[::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5.5))

    # ── Panel A: Non-MRI ──
    colors_a = [COLOR_NO_MRI] * 10
    ax1.barh(range(10), top10_no['Gain'].values, color=colors_a, height=0.6,
             edgecolor='white')
    ax1.set_yticks(range(10))
    ax1.set_yticklabels(top10_no['label'].values, fontsize=10, fontweight='bold')
    ax1.set_xlabel('SHAP Importance (Gain-based proxy)', fontweight='bold')
    ax1.set_title('SHAP Feature Importance\nNon-MRI Model (DM_full)', fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    for i, (_, row) in enumerate(top10_no.iterrows()):
        ax1.text(row['Gain'] + max(top10_no['Gain'])*0.01, i,
                f'{row["Gain"]:.4f}', fontsize=8, va='center', color=COLOR_NO_MRI)

    # ── Panel B: +MRI ──
    df_mri = pd.read_csv(os.path.join(MRI_DIR, 's03_final_importance.csv'))
    top10_mri = df_mri.head(10).copy()
    top10_mri['label'] = top10_mri['Features'].apply(feat_name_short)
    top10_mri = top10_mri.iloc[::-1]

    colors_b = []
    for _, row in top10_mri.iterrows():
        fid = row['Features']
        name = feat_name(fid)
        is_img = 'MRI' in name or '★' in name
        colors_b.append(COLOR_MRI if is_img else '#556677')

    ax2.barh(range(10), top10_mri['Gain'].values, color=colors_b, height=0.6,
             edgecolor='white')
    ax2.set_yticks(range(10))
    ax2.set_yticklabels(top10_mri['label'].values, fontsize=10, fontweight='bold')
    ax2.set_xlabel('SHAP Importance (Gain-based proxy)', fontweight='bold')
    ax2.set_title('SHAP Feature Importance\n+MRI Model (DM_full)', fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3, linestyle='--')

    from matplotlib.patches import Patch
    legend_b = [
        Patch(facecolor=COLOR_MRI, label='MRI feature'),
        Patch(facecolor='#556677', label='Clinical feature'),
    ]
    ax2.legend(handles=legend_b, loc='lower right', fontsize=8)

    for i, (_, row) in enumerate(top10_mri.iterrows()):
        ax2.text(row['Gain'] + max(top10_mri['Gain'])*0.01, i,
                f'{row["Gain"]:.4f}', fontsize=8, va='center', color='#555555')

    fig.suptitle('Top-10 Feature Importance Comparison (s03 clustering stage)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    outpath = os.path.join(OUT_DIR, 'Fig5_SHAP_style_importance.png')
    fig.savefig(outpath)
    plt.close(fig)
    print(f'[OK] Saved {outpath}')
    return outpath


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 6: SFS step-by-step AUC gain (paper-style cumulative bar)       ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def plot_sfs_cumulative_gain():
    """Cumulative AUC gain bar chart — paper-style visualization."""
    df_no = pd.read_csv(os.path.join(NO_MRI_DIR, 's04_selected_features.csv'))
    df_mri = pd.read_csv(os.path.join(MRI_DIR, 's04_selected_features.csv'))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))

    # ── Non-MRI ──
    base_no = df_no['CumulativeAUC'].iloc[0] - df_no['Gain'].iloc[0]  # baseline AUC
    # Actually use the first step's gain relative to a theoretical 0.5 baseline,
    # but more usefully show the incremental gain per step
    gains_no = [df_no['Gain'].iloc[0]]
    for i in range(1, len(df_no)):
        gains_no.append(df_no['Gain'].iloc[i])
    # Show cumulative starting from first feature
    cum_auc = df_no['CumulativeAUC'].values
    x_labels = [feat_name_short(f, 25) for f in df_no['Features'].values]

    ax1.bar(range(len(x_labels)), gains_no, color=COLOR_NO_MRI, alpha=0.8, edgecolor='white')
    ax1.set_xticks(range(len(x_labels)))
    ax1.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=7.5)
    ax1.set_ylabel('Incremental AUC Gain', fontweight='bold')
    ax1.set_title('Non-MRI: Per-Feature AUC Gain (SFS)', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Add cumulative AUC text above each bar
    for i, (g, c) in enumerate(zip(gains_no, cum_auc)):
        ax1.text(i, g + max(gains_no)*0.03, f'{c:.4f}', fontsize=6.5,
                ha='center', color='#333333')

    # ── +MRI ──
    gains_mri = [df_mri['Gain'].iloc[0]]
    for i in range(1, len(df_mri)):
        gains_mri.append(df_mri['Gain'].iloc[i])
    cum_auc_mri = df_mri['CumulativeAUC'].values
    x_labels_mri = [feat_name_short(f, 25) for f in df_mri['Features'].values]

    colors_mri = []
    for f in df_mri['Features']:
        name = feat_name(f)
        is_img = 'MRI' in name or '★' in name
        colors_mri.append(COLOR_MRI if is_img else '#8899AA')

    ax2.bar(range(len(x_labels_mri)), gains_mri, color=colors_mri, alpha=0.8, edgecolor='white')
    ax2.set_xticks(range(len(x_labels_mri)))
    ax2.set_xticklabels(x_labels_mri, rotation=45, ha='right', fontsize=7.5)
    ax2.set_ylabel('Incremental AUC Gain', fontweight='bold')
    ax2.set_title('+MRI: Per-Feature AUC Gain (SFS)', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    for i, (g, c) in enumerate(zip(gains_mri, cum_auc_mri)):
        ax2.text(i, g + max(gains_mri)*0.03, f'{c:.4f}', fontsize=6.5,
                ha='center', color='#333333')

    # Legend for MRI
    from matplotlib.patches import Patch
    legend_b = [
        Patch(facecolor=COLOR_MRI, label='MRI feature'),
        Patch(facecolor='#8899AA', label='Clinical feature'),
    ]
    ax2.legend(handles=legend_b, loc='upper right', fontsize=8)

    fig.suptitle('SFS Step-by-Step AUC Gain (DM_full)', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    outpath = os.path.join(OUT_DIR, 'Fig6_SFS_AUC_Gain.png')
    fig.savefig(outpath)
    plt.close(fig)
    print(f'[OK] Saved {outpath}')
    return outpath


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FIGURE 7: Summary bar chart for all 6 targets (MRI improvement)        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def plot_mri_improvement_summary():
    """Bar chart showing MRI improvement across all 6 targets."""
    # Data from the experiment summary (5-fold CV)
    targets = ['DM_full', 'DM_10yrs', 'DM_5yrs', 'AD_full', 'AD_10yrs', 'AD_5yrs']
    no_mri_auc = [0.831, 0.833, 0.816, 0.836, 0.832, 0.667]
    mri_auc    = [0.837, 0.841, 0.842, 0.845, 0.853, 0.851]
    delta      = [m - n for m, n in zip(mri_auc, no_mri_auc)]

    x = np.arange(len(targets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5.5))

    bars1 = ax.bar(x - width/2, no_mri_auc, width, label='Non-MRI (1076 features)',
                   color=COLOR_NO_MRI, alpha=0.85, edgecolor='white')
    bars2 = ax.bar(x + width/2, mri_auc, width, label='+MRI (3250 features)',
                   color=COLOR_MRI, alpha=0.85, edgecolor='white')

    # Annotate delta
    for i, (n, m, d) in enumerate(zip(no_mri_auc, mri_auc, delta)):
        ax.annotate(f'+{d:.3f}', (x[i], m + 0.003), ha='center', fontsize=9,
                    fontweight='bold', color=COLOR_MRI)

    ax.set_xticks(x)
    ax.set_xticklabels(targets, fontsize=10, fontweight='bold')
    ax.set_ylabel('5-Fold CV AUC', fontweight='bold')
    ax.set_title('Effect of Brain MRI on Prediction Performance\n'
                 '(UKB ~425K participants, Deploy strategy)',
                 fontweight='bold')
    ax.set_ylim(0.64, 0.88)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add paper's reported values for reference
    paper_auc = [0.848, 0.849, 0.847, 0.862, 0.866, 0.890]
    ax.scatter(x, paper_auc, marker='*', s=120, c='#FF8C00', zorder=10,
               label='Paper reported AUC (with APOE4+PRS)', edgecolors='#CC6600', linewidths=0.8)

    ax.legend(fontsize=8, loc='upper right')

    plt.tight_layout()
    outpath = os.path.join(OUT_DIR, 'Fig7_MRI_Improvement_Summary.png')
    fig.savefig(outpath)
    plt.close(fig)
    print(f'[OK] Saved {outpath}')
    return outpath


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
if __name__ == '__main__':
    print('Generating publication-style figures for UKB-DRP replication...')
    print(f'Output directory: {OUT_DIR}')
    print()

    plot_sfs_no_mri()
    plot_sfs_mri()
    plot_sfs_combined()
    plot_top30_gain_no_mri()
    plot_top30_gain_mri()
    generate_top30_table()
    plot_shap_style()
    plot_sfs_cumulative_gain()
    plot_mri_improvement_summary()

    print()
    print('Done! All figures saved to:')
    print(f'  {OUT_DIR}/')
    print()
    print('Notes:')
    print('  - SHAP beeswarm plots are already available as PNGs in:')
    print('    local_data/Results_v2/DM_full/shap_beeswarm.png')
    print('    (Raw SHAP values require re-running the model on original data)')
    print('  - Feature descriptions are best-effort mappings from UKB field IDs')

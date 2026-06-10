#!/usr/bin/env python3
"""
Three-way SFS comparison plot:
  - Paper-aligned (138 features, SFS)  → blue
  - Replication (1076 features, SFS)   → green
  - +MRI (3250 features, SFS)          → red
"""
import pandas as pd, numpy as np, matplotlib, os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

BASE = '/Users/guxiao/Downloads/MCI-AD/UKB_DRP-main'
OUT_DIR = os.path.join(BASE, 'local_data/Results_v2/_figures')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Feature names ──
FEAT_NAMES = {
    '34-0.0': 'Year of birth (Age)',
    '400-0.0': 'Reaction time (mean)',
    '137-0.0': 'Number of medications',
    '20023-0.0': 'Reaction time (matches)',
    '23112-0.0': 'Leg impedance (whole body)',
    '30040-0.0': 'Mean platelet volume',
    '3526-0.0': "Mother's age at death",
    '20110-0.1_neg': 'Mother: no major illness',
    '30650-0.0': 'Aspartate aminotransferase',
    '30710-0.0': 'C-reactive protein (CRP)',
    '2188-0.0_c1': 'Diabetes (Dr diagnosed)',
    '20110-0.4_pos': "Mother: dementia/Alzheimer's",
    '1090-0.0': 'Trail making test',
    '6142-0.3_pos': 'Employment: retired',
    '12651-0.0': 'MRI: L Hippocampal Subiculum Vol',
    '26643-0.0': 'MRI: L Subiculum Volume',
    '1200-0.0_c0': 'Sleep duration (hours)',
}
def fl(fid): return FEAT_NAMES.get(fid, fid)

# ── Load data ──
df_paper = pd.read_csv(os.path.join(BASE, 'local_data/Results_v2/_aligned_features/s04_sfs_history.csv'))
df_repro = pd.read_csv(os.path.join(BASE, 'local_data/Results_v2/DM_full/s04_sfs_history.csv'))
df_mri   = pd.read_csv(os.path.join(BASE, 'local_data/Results_imaging/DM_full_img_full/s04_sfs_history.csv'))

# Add step numbers if not present
for df in [df_paper, df_repro, df_mri]:
    if 'selected_count' not in df.columns:
        df['selected_count'] = range(1, len(df)+1)
    if 'auc_mean' not in df.columns and 'auc' in df.columns:
        df['auc_mean'] = df['auc']
    if 'auc_std' not in df.columns:
        df['auc_std'] = 0.003  # default

# ── Style ──
plt.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':12, 'axes.titlesize':16, 'axes.labelsize':13,
    'xtick.labelsize':10, 'ytick.labelsize':10, 'figure.dpi':150, 'savefig.dpi':300,
})

COLOR_PAPER = '#2166AC'   # blue
COLOR_REPRO = '#4DAF4A'   # green
COLOR_MRI   = '#D64545'   # red

fig, ax = plt.subplots(figsize=(16, 7.5))

# ═══ Plot three curves ═══
datasets = [
    (df_paper, COLOR_PAPER, 'Paper-aligned (138 features, SFS)', 'o', 'Paper-aligned'),
    (df_repro, COLOR_REPRO, 'Replication (1076 clinical features, SFS)', 's', 'Replication'),
    (df_mri,   COLOR_MRI,   '+MRI (3250 clinical + MRI features, SFS)', 'D', '+MRI'),
]

all_auc = []
for df, color, label, marker, short_label in datasets:
    steps = df['selected_count'].values
    auc   = df['auc_mean'].values
    auc_std = df['auc_std'].values
    fids  = df['feature_added'].values if 'feature_added' in df.columns else ['']
    all_auc.extend(auc)

    # Confidence band
    ax.fill_between(steps, auc - auc_std, auc + auc_std, alpha=0.08, color=color)

    # Line
    ax.plot(steps, auc, '-', color=color, linewidth=2.0, alpha=0.7, zorder=2)

    # Markers
    ax.scatter(steps, auc, c=color, s=90, marker=marker, zorder=5,
               edgecolors='white', linewidths=0.8, label=label)

    # Annotate feature names
    if short_label != 'Paper-aligned':
        # Annotate replication and MRI features
        for s, a, fid in zip(steps, auc, fids):
            name = fl(fid)
            short = name if len(name) <= 24 else name[:22] + '..'
            is_img = 'MRI' in name
            fc = COLOR_MRI if is_img else color
            fw = 'bold' if is_img else 'normal'
            va = 'bottom'; off = 0.003
            if s == 1: off = -0.004; va = 'top'
            elif s % 2 == 0: off = -0.004; va = 'top'
            ax.annotate(short, (s, a + off), fontsize=6.8, color=fc,
                        fontweight=fw, ha='center', va=va, rotation=20)
    else:
        # Annotate paper features (key ones only)
        for s, a, fid in zip(steps, auc, fids):
            name = fl(fid)
            if s in [1, 3, 5, 8, 10]:
                short = name if len(name) <= 24 else name[:22] + '..'
                ax.annotate(short, (s, a - 0.005), fontsize=6.5, color=COLOR_PAPER,
                            ha='center', va='top', rotation=15)

# ═══ Final AUC annotations ═══
final_offsets = [
    (df_paper, COLOR_PAPER, 'Paper-aligned\n(SFS)', 9.5),
    (df_repro, COLOR_REPRO, 'Replication\n(1076 feat.)', 8.0),
    (df_mri,   COLOR_MRI,   '+MRI\n(3250 feat.)', 7.0),
]
for df, color, label, xpos in final_offsets:
    auc_final = df['auc_mean'].values[-1]
    ax.annotate(f'{label}\nAUC={auc_final:.4f}',
                xy=(10, auc_final), xytext=(xpos, auc_final + 0.008),
                fontsize=8.5, fontweight='bold', color=color, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor=color, alpha=0.85))

# ═══ Axis settings ═══
ax.set_xlabel('Number of Features Selected (SFS Step)', fontweight='bold')
ax.set_ylabel('Cumulative AUC (5-fold CV)', fontweight='bold')
ax.set_title('Three-Way SFS Comparison: Paper-aligned vs Replication vs +MRI\n(DM_full target, UKB ~425K participants)',
             fontweight='bold', pad=12)
ax.set_xticks(range(1, 11))
ax.set_xlim(0.4, 10.6)
ax.set_ylim(0.792, 0.848)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
ax.grid(axis='y', alpha=0.2, linestyle='--')

# Legend
ax.legend(loc='lower right', fontsize=10, framealpha=0.9,
          title='Experiment', title_fontsize=11)

# ═══ Bottom annotations table ═══
table_text = (
    "Note: Paper-aligned uses 138 features matching the paper's feature space; "
    "Replication uses 1,076 clinical features from UKB .npz data; "
    "+MRI adds 2,176 brain MRI IDPs (total 3,250). "
    "All curves use the same True SFS methodology (not the paper's cumulative-AUC code). "
    "Paper-reported DM_full AUC ≈ 0.848 (with APOE4 + PRS data not available in our dataset)."
)
fig.text(0.5, -0.01, table_text, ha='center', fontsize=8, color='#888888',
         transform=ax.transAxes, wrap=True)

plt.tight_layout()
outpath = os.path.join(OUT_DIR, 'SFS_ThreeWay_Comparison.png')
fig.savefig(outpath)
plt.close(fig)
print(f'[OK] {outpath}')

# ═══ Also generate single-panel versions for PPT ═══
for df, color, title, marker, fname in datasets:
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    steps = df['selected_count'].values
    auc   = df['auc_mean'].values
    auc_std = df['auc_std'].values
    fids  = df['feature_added'].values if 'feature_added' in df.columns else ['']
    is_img = ['MRI' in fl(f) for f in fids]

    ax2.fill_between(steps, auc - auc_std, auc + auc_std, alpha=0.12, color=color)
    ax2.plot(steps, auc, '-', color='#444444', linewidth=1.5, alpha=0.6, zorder=2)

    point_colors = [COLOR_MRI if img else color for img in is_img]
    for s, a, pc, img in zip(steps, auc, point_colors, is_img):
        mkr = 'D' if img else marker
        ax2.scatter(s, a, c=pc, s=90, marker=mkr, zorder=5, edgecolors='white', linewidths=0.8)

    for s, a, fid, img in zip(steps, auc, fids, is_img):
        name = fl(fid)
        short = name[:28] + '..' if len(name) > 30 else name
        fc = COLOR_MRI if img else '#444444'
        va = 'bottom'; off = 0.003
        if s == 1: off = -0.005; va = 'top'
        elif s % 3 == 0: off = -0.005; va = 'top'
        ax2.annotate(short, (s, a + off), fontsize=8, color=fc,
                     fontweight='bold' if img else 'normal', ha='center', va=va)

    for s, a in zip(steps, auc):
        ax2.annotate(f'{a:.4f}', (s, a - 0.005), fontsize=7, color='#666666', ha='center', va='top')

    ax2.set_xlabel('SFS Step', fontweight='bold')
    ax2.set_ylabel('Cumulative AUC (5-fold CV)', fontweight='bold')
    ax2.set_title(f'{title}\n(DM_full, UKB ~425K)', fontweight='bold')
    ax2.set_xticks(range(1, 11))
    ax2.set_xlim(0.3, 10.7)
    ax2.set_ylim(0.792, 0.848)
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    ax2.grid(axis='y', alpha=0.2, linestyle='--')

    out_single = os.path.join(OUT_DIR, f'SFS_{fname.replace(" ","_")}.png')
    fig2.savefig(out_single)
    plt.close(fig2)
    print(f'[OK] {out_single}')

print('\nDone!')

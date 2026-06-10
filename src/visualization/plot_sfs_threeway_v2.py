#!/usr/bin/env python3
"""
Corrected three-way SFS comparison:
  - Replication SFS curve (1076 features)
  - +MRI SFS curve (3250 features)
  - Paper's reported FINAL AUC as a horizontal reference line (no step-by-step data available)
"""
import pandas as pd, numpy as np, matplotlib, os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BASE = '/Users/guxiao/Downloads/MCI-AD/UKB_DRP-main'
OUT_DIR = os.path.join(BASE, 'local_data/Results_v2/_figures')
os.makedirs(OUT_DIR, exist_ok=True)

FEAT_NAMES = {
    '34-0.0':'Year of birth (Age)','400-0.0':'Reaction time (mean)',
    '137-0.0':'Number of medications','2188-0.0_c1':'Diabetes (Dr diagnosed)',
    '30710-0.0':'C-reactive protein (CRP)','20110-0.4_pos':"Mother: dementia/Alzheimer's",
    '1090-0.0':'Trail making test','6142-0.3_pos':'Employment: retired',
    '20023-0.0':'Reaction time (matches)','3526-0.0':"Mother's age at death",
    '12651-0.0':'MRI: L Hippocampal Subiculum Vol','26643-0.0':'MRI: L Subiculum Volume',
    '1200-0.0_c0':'Sleep duration (hours)',
}
def fl(fid): return FEAT_NAMES.get(fid, fid)

df_repro = pd.read_csv(os.path.join(BASE, 'local_data/Results_v2/DM_full/s04_sfs_history.csv'))
df_mri   = pd.read_csv(os.path.join(BASE, 'local_data/Results_imaging/DM_full_img_full/s04_sfs_history.csv'))

plt.rcParams.update({
    'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':12, 'axes.titlesize':15, 'axes.labelsize':13,
    'xtick.labelsize':10, 'ytick.labelsize':10, 'figure.dpi':150, 'savefig.dpi':300,
})

COLOR_REPRO = '#2166AC'   # blue
COLOR_MRI   = '#B2182B'   # red
COLOR_PAPER = '#E69F00'   # orange/gold for paper reference

fig, ax = plt.subplots(figsize=(16, 7.5))

# ═══ Plot Replication SFS curve ═══
for df, color, marker, label, short in [
    (df_repro, COLOR_REPRO, 's', 'Replication (1076 clinical features)', 'Replication'),
    (df_mri,   COLOR_MRI,   'D', '+MRI (3250 clinical + MRI features)', '+MRI'),
]:
    steps = df['selected_count'].values
    auc   = df['auc_mean'].values
    auc_std = df['auc_std'].values
    fids  = df['feature_added'].values
    is_img = ['MRI' in fl(f) for f in fids]

    ax.fill_between(steps, auc - auc_std, auc + auc_std, alpha=0.08, color=color)
    ax.plot(steps, auc, '-', color=color, linewidth=2.2, alpha=0.5, zorder=2)

    # Points - color MRI features red even on replication curve (none for repro)
    for s, a in zip(steps, auc):
        ax.scatter(s, a, c=color, s=85, marker=marker, zorder=5,
                   edgecolors='white', linewidths=0.8)

    # Feature annotations
    for s, a, fid, img in zip(steps, auc, fids, is_img):
        name = fl(fid)
        short_n = name if len(name) <= 22 else name[:20]+'..'
        fc = COLOR_MRI if img else color
        fw = 'bold' if img else 'normal'
        va = 'bottom'; off = 0.0032
        if s == 1: off = -0.0045; va = 'top'
        elif s in [3,5,7,9]: off = -0.0045; va = 'top'
        ax.annotate(short_n, (s, a + off), fontsize=7, color=fc,
                    fontweight=fw, ha='center', va=va, rotation=18)

    # AUC labels
    for s, a in zip(steps, auc):
        ax.annotate(f'{a:.4f}', (s, a - 0.0055), fontsize=6.5, color='#666666',
                    ha='center', va='top')

    # Final AUC annotation
    ax.annotate(f'{short}\nAUC = {auc[-1]:.4f}',
                xy=(10, auc[-1]), xytext=(7.5, auc[-1] + 0.012),
                fontsize=9.5, fontweight='bold', color=color, ha='center',
                bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor=color, alpha=0.9))

# ═══ Paper's reported AUC: horizontal reference line + band ═══
# Paper reported DM_full AUC = 0.848, but NO step-by-step data available.
# Public code produces ~0.835 cumulative AUC maximum.
# The gap (0.848-0.835≈0.013) is attributed to APOE4 + PRS + manual features.
PAPER_REPORTED = 0.848
PAPER_CODE_MAX = 0.835  # approximate from docs

# Reference line at paper's reported AUC
ax.axhline(y=PAPER_REPORTED, color=COLOR_PAPER, linewidth=2.0, linestyle='--', zorder=1, alpha=0.8)
ax.fill_between([0.5, 10.5], PAPER_CODE_MAX, PAPER_REPORTED,
                color=COLOR_PAPER, alpha=0.06)

# Annotations
ax.annotate(f'Paper reported DM_full AUC ≈ {PAPER_REPORTED}\n(with APOE4 + PRS + manual features)',
            xy=(10, PAPER_REPORTED), xytext=(6.0, PAPER_REPORTED + 0.003),
            fontsize=9, fontweight='bold', color=COLOR_PAPER, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF8E1', edgecolor=COLOR_PAPER, alpha=0.9))

ax.annotate(f'Paper public code\ncumulative AUC ≈ {PAPER_CODE_MAX}',
            xy=(3, PAPER_CODE_MAX), fontsize=8, color='#B8860B', ha='center')

# Gap annotation
ax.annotate(f'Gap ≈ {PAPER_REPORTED - PAPER_CODE_MAX:.3f}\n(APOE4+PRS\n+manual features)',
            xy=(8.5, (PAPER_REPORTED + PAPER_CODE_MAX)/2),
            fontsize=8, color='#B8860B', ha='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#FFF8E1', alpha=0.7))

# ═══ Axis settings ═══
ax.set_xlabel('Number of Features Selected (SFS Step)', fontweight='bold')
ax.set_ylabel('Cumulative AUC (5-fold CV)', fontweight='bold')
ax.set_title('SFS Feature Selection: Replication vs +MRI\n(Paper reported AUC shown as reference — no step-by-step data available)',
             fontweight='bold', pad=12)
ax.set_xticks(range(1, 11))
ax.set_xlim(0.3, 10.7)
ax.set_ylim(0.792, 0.858)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
ax.grid(axis='y', alpha=0.2, linestyle='--')

# ═══ Legend ═══
legend_els = [
    Line2D([0],[0], marker='s', color=COLOR_REPRO, markersize=9, markerfacecolor=COLOR_REPRO,
           markeredgewidth=0.8, markeredgecolor='white', linewidth=2.2,
           label='Replication (1076 clinical features)'),
    Line2D([0],[0], marker='D', color=COLOR_MRI, markersize=9, markerfacecolor=COLOR_MRI,
           markeredgewidth=0.8, markeredgecolor='white', linewidth=2.2,
           label='+MRI (3250 clinical + MRI features)'),
    Line2D([0],[0], color=COLOR_PAPER, linewidth=2.0, linestyle='--',
           label=f'Paper reported AUC ≈ {PAPER_REPORTED}\n(no step-by-step data published)'),
]
ax.legend(handles=legend_els, loc='lower right', fontsize=9.5, framealpha=0.9)

# ═══ Bottom note ═══
fig.text(0.5, 0.01,
         "Note: Paper (Yu et al. 2024) reported DM_full AUC ≈ 0.848 with APOE4, PRS, and manual features — "
         "but step-by-step SFS curves were never published. "
         "Our replication achieves AUC 0.832 (no APOE4/PRS); +MRI achieves AUC 0.837.",
         ha='center', fontsize=8, color='#888888')

plt.tight_layout()
outpath = os.path.join(OUT_DIR, 'SFS_ThreeWay_v2.png')
fig.savefig(outpath)
plt.close(fig)
print(f'[OK] {outpath}')

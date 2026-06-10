#!/usr/bin/env python3
"""
Recreate SFS plot in the requested style:
  - Bottom: step-by-step bar chart showing AUC gain per feature added
  - Top: cumulative AUC curve with error bands
  - Color-coded: clinical (blue) vs imaging (red) features
  - Features annotated with readable names

Generates two versions: Non-MRI only, and +MRI comparison.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
import os, sys

# ── Paths ──
BASE = '/Users/guxiao/Downloads/MCI-AD/UKB_DRP-main'
NO_MRI_DIR = os.path.join(BASE, 'local_data/Results_v2/DM_full')
MRI_DIR    = os.path.join(BASE, 'local_data/Results_imaging/DM_full_img_full')
OUT_DIR    = os.path.join(BASE, 'local_data/Results_v2/_figures')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Feature name mapping ──
FEAT_NAMES = {
    '34-0.0':           'Year of birth (Age)',
    '400-0.0':          'Reaction time (mean)',
    '137-0.0':          'Number of medications',
    '2188-0.0_c1':      'Diabetes (Dr diagnosed)',
    '30710-0.0':        'C-reactive protein',
    '20110-0.4_pos':    'Mother: dementia/Alzheimer',
    '1090-0.0':         'Trail making test',
    '6142-0.3_pos':     'Employment: retired',
    '20023-0.0':        'Reaction time (match)',
    '3526-0.0':         "Mother's age at death",
    '12651-0.0':        'MRI: L Hippocampal Subiculum',
    '26643-0.0':        'MRI: L Subiculum Volume',
    '1200-0.0_c0':      'Sleep duration (hours)',
}

def feat_label(fid):
    return FEAT_NAMES.get(fid, fid)

# ── Load data ──
df_no = pd.read_csv(os.path.join(NO_MRI_DIR, 's04_sfs_history.csv'))
df_mri = pd.read_csv(os.path.join(MRI_DIR, 's04_sfs_history.csv'))

# ── Plot style ──
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'font.size': 11,
    'axes.titlesize': 15,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

COLOR_CLINICAL = '#3A7CA5'   # blue-grey
COLOR_IMAGING  = '#D64545'   # red
BAND_ALPHA = 0.15

# ═══════════════════════════════════════════════════════════════
# FIGURE 1: Non-MRI experiment SFS
# ═══════════════════════════════════════════════════════════════
def plot_sfs_styled(df, outpath, title_suffix, highlight_mri=False):
    """Create styled SFS plot: bar chart (gain) + line (cumulative AUC)."""

    steps = df['selected_count'].values
    auc_mean = df['auc_mean'].values
    auc_std = df['auc_std'].values
    features = df['feature_added'].values
    labels = [feat_label(f) for f in features]

    # Compute per-step gain
    gains = [auc_mean[0] - 0.5]  # first gain from baseline
    for i in range(1, len(auc_mean)):
        gains.append(auc_mean[i] - auc_mean[i-1])
    gains = np.array(gains)

    # Determine which features are imaging
    is_imaging = ['MRI' in feat_label(f) or 'Imaging' in feat_label(f) for f in features]

    # ── Create figure with two stacked panels ──
    fig = plt.figure(figsize=(12, 10))

    # --- Top panel: Cumulative AUC curve ---
    ax1 = fig.add_axes([0.12, 0.48, 0.82, 0.44])

    # Color each point
    point_colors = [COLOR_IMAGING if img else COLOR_CLINICAL for img in is_imaging]

    # Plot confidence band
    ax1.fill_between(steps, auc_mean - auc_std, auc_mean + auc_std,
                     alpha=BAND_ALPHA, color='#888888')

    # Plot connecting line
    ax1.plot(steps, auc_mean, '-', color='#666666', linewidth=1.5, zorder=3)

    # Plot points
    for i, (s, a, c) in enumerate(zip(steps, auc_mean, point_colors)):
        marker = 'D' if is_imaging[i] else 'o'
        sz = 90 if is_imaging[i] else 70
        ax1.scatter(s, a, c=c, s=sz, zorder=5, edgecolors='white', linewidths=0.8)

    # Annotate each point with feature name
    for i, (s, a, lbl, img) in enumerate(zip(steps, auc_mean, labels, is_imaging)):
        va = 'bottom'
        offset = 0.003
        if i == 0:
            offset = -0.006
            va = 'top'
        elif i == len(steps) - 1:
            offset = 0.003
            va = 'bottom'
        elif i % 3 == 0:
            offset = -0.005
            va = 'top'

        color = COLOR_IMAGING if img else '#444444'
        weight = 'bold' if img else 'normal'
        ax1.annotate(lbl, (s, a + offset),
                    fontsize=8.5, color=color, fontweight=weight,
                    ha='center', va=va,
                    arrowprops=dict(arrowstyle='-', color='#cccccc', lw=0.3) if i > 0 else None)

    ax1.set_ylabel('Cumulative AUC (5-fold CV)', fontweight='bold')
    ax1.set_title(f'Sequential Forward Selection — {title_suffix}\n(DM_full target, UKB ~425K participants)',
                  fontweight='bold')
    ax1.set_xlim(0.5, 10.5)
    ax1.set_xticks(range(1, 11))
    ax1.set_xticklabels([])
    ax1.set_ylim(0.794, 0.844)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    ax1.grid(axis='y', alpha=0.3, linestyle='--')

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_CLINICAL,
                   markersize=10, label='Clinical feature'),
    ]
    if any(is_imaging):
        legend_elements.append(
            plt.Line2D([0], [0], marker='D', color='w', markerfacecolor=COLOR_IMAGING,
                       markersize=10, label='MRI / Imaging feature')
        )
    legend_elements.append(
        plt.Rectangle((0,0), 1, 1, fc='#888888', alpha=BAND_ALPHA, label='±1 SD')
    )
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9)

    # Final AUC annotation
    final_label = f'Final AUC = {auc_mean[-1]:.4f}'
    ax1.annotate(final_label, xy=(steps[-1], auc_mean[-1]),
                xytext=(steps[-1]-1.2, auc_mean[-1] + 0.015),
                fontsize=10, fontweight='bold', color=COLOR_CLINICAL,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f0f0', alpha=0.9))

    # --- Bottom panel: Per-step gain bar chart ---
    ax2 = fig.add_axes([0.12, 0.10, 0.82, 0.30])

    bar_colors = [COLOR_IMAGING if img else COLOR_CLINICAL for img in is_imaging]
    bars = ax2.bar(steps, gains, color=bar_colors, width=0.55, edgecolor='white', linewidth=0.5, zorder=3)

    # Add cumulative AUC text above bars
    for i, (s, g, a) in enumerate(zip(steps, gains, auc_mean)):
        ax2.text(s, g + max(gains)*0.04, f'{a:.4f}', fontsize=7.5,
                ha='center', color='#444444', fontweight='bold')

    # Highlight imaging feature bars with an edge
    for i, (bar, img) in enumerate(zip(bars, is_imaging)):
        if img:
            bar.set_edgecolor(COLOR_IMAGING)
            bar.set_linewidth(1.5)

    ax2.set_xlabel('Number of Features Selected (SFS step)', fontweight='bold')
    ax2.set_ylabel('Incremental AUC\nGain', fontweight='bold')
    ax2.set_xlim(0.5, 10.5)
    ax2.set_xticks(range(1, 11))
    ax2.set_xticklabels([feat_label(f)[:25] for f in features], rotation=45, ha='right', fontsize=8)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Add a horizontal line at gain=0
    ax2.axhline(y=0, color='#cccccc', linewidth=0.5, zorder=1)

    # ── Save ──
    fig.savefig(outpath)
    plt.close(fig)
    print(f'[OK] {outpath}')
    return outpath

# ═══════════════════════════════════════════════════════════════
# FIGURE 2: Combined Non-MRI vs +MRI SFS comparison
# ═══════════════════════════════════════════════════════════════
def plot_sfs_comparison(df_no, df_mri, outpath):
    """Combined figure showing both non-MRI and +MRI SFS in the requested style."""

    fig = plt.figure(figsize=(18, 10))

    for panel, (df, title, is_right) in enumerate([
        (df_no, 'Non-MRI (1076 Clinical Features)', False),
        (df_mri, '+MRI (3250 Clinical + MRI Features)', True)
    ]):
        steps = df['selected_count'].values
        auc_mean = df['auc_mean'].values
        auc_std = df['auc_std'].values
        features = df['feature_added'].values
        labels = [feat_label(f) for f in features]

        gains = [auc_mean[0] - 0.5]
        for i in range(1, len(auc_mean)):
            gains.append(auc_mean[i] - auc_mean[i-1])
        gains = np.array(gains)

        is_imaging = ['MRI' in feat_label(f) or 'Imaging' in feat_label(f) for f in features]

        # Top subplot
        ax_top = fig.add_axes([0.06 + panel*0.47, 0.48, 0.42, 0.44])

        point_colors = [COLOR_IMAGING if img else COLOR_CLINICAL for img in is_imaging]
        ax_top.fill_between(steps, auc_mean - auc_std, auc_mean + auc_std,
                           alpha=BAND_ALPHA, color='#888888')
        ax_top.plot(steps, auc_mean, '-', color='#666666', linewidth=1.5, zorder=3)

        for i, (s, a, c) in enumerate(zip(steps, auc_mean, point_colors)):
            marker = 'D' if is_imaging[i] else 'o'
            sz = 90 if is_imaging[i] else 70
            ax_top.scatter(s, a, c=c, s=sz, zorder=5, edgecolors='white', linewidths=0.8)

        for i, (s, a, lbl, img) in enumerate(zip(steps, auc_mean, labels, is_imaging)):
            va = 'bottom'; offset = 0.003
            if i == 0: offset = -0.006; va = 'top'
            elif i % 3 == 0: offset = -0.005; va = 'top'
            color = COLOR_IMAGING if img else '#444444'
            weight = 'bold' if img else 'normal'
            ax_top.annotate(lbl, (s, a + offset),
                          fontsize=8, color=color, fontweight=weight,
                          ha='center', va=va,
                          arrowprops=dict(arrowstyle='-', color='#cccccc', lw=0.3) if i > 0 else None)

        ax_top.set_ylabel('Cumulative AUC (5-fold CV)', fontweight='bold')
        ax_top.set_title(title, fontweight='bold', fontsize=14)
        ax_top.set_xlim(0.5, 10.5)
        ax_top.set_xticks(range(1, 11))
        ax_top.set_xticklabels([])
        ax_top.set_ylim(0.794, 0.844)
        ax_top.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
        ax_top.grid(axis='y', alpha=0.3, linestyle='--')

        final_label = f'Final AUC = {auc_mean[-1]:.4f}'
        ax_top.annotate(final_label, xy=(steps[-1], auc_mean[-1]),
                       xytext=(steps[-1]-1.2, auc_mean[-1] + 0.015),
                       fontsize=10, fontweight='bold', color=COLOR_CLINICAL,
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f0f0', alpha=0.9))

        # Bottom subplot
        ax_bot = fig.add_axes([0.06 + panel*0.47, 0.10, 0.42, 0.30])

        bar_colors = [COLOR_IMAGING if img else COLOR_CLINICAL for img in is_imaging]
        bars = ax_bot.bar(steps, gains, color=bar_colors, width=0.55,
                         edgecolor='white', linewidth=0.5, zorder=3)

        for i, (s, g, a) in enumerate(zip(steps, gains, auc_mean)):
            ax_bot.text(s, g + max(gains)*0.04, f'{a:.4f}', fontsize=7.5,
                       ha='center', color='#444444', fontweight='bold')

        for bar, img in zip(bars, is_imaging):
            if img:
                bar.set_edgecolor(COLOR_IMAGING)
                bar.set_linewidth(1.5)

        ax_bot.set_xlabel('SFS Step (features added)', fontweight='bold')
        if not is_right:
            ax_bot.set_ylabel('AUC Gain per Step', fontweight='bold')
        ax_bot.set_xlim(0.5, 10.5)
        ax_bot.set_xticks(range(1, 11))
        # Short labels for bottom
        short_labels = []
        for f in features:
            name = feat_label(f)
            if len(name) > 22:
                name = name[:20] + '..'
            short_labels.append(name)
        ax_bot.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=7.5)
        ax_bot.grid(axis='y', alpha=0.3, linestyle='--')
        ax_bot.axhline(y=0, color='#cccccc', linewidth=0.5, zorder=1)

    # MRI improvement annotation
    delta = df_mri['auc_mean'].values[-1] - df_no['auc_mean'].values[-1]
    fig.text(0.5, 0.02, f'MRI improvement: ΔAUC = +{delta:.4f}',
             ha='center', fontsize=12, fontweight='bold',
             color=COLOR_IMAGING,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff5f5',
                      edgecolor=COLOR_IMAGING, alpha=0.9))

    fig.savefig(outpath)
    plt.close(fig)
    print(f'[OK] {outpath}')
    return outpath


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    # Single experiment: Non-MRI
    plot_sfs_styled(df_no,
                    os.path.join(OUT_DIR, 'SFS_NonMRI_Styled.png'),
                    'Non-MRI Model')

    # Single experiment: +MRI
    plot_sfs_styled(df_mri,
                    os.path.join(OUT_DIR, 'SFS_MRI_Styled.png'),
                    '+MRI Model')

    # Combined comparison
    plot_sfs_comparison(df_no, df_mri,
                        os.path.join(OUT_DIR, 'SFS_Combined_Styled.png'))

    print('\nDone! Files in:', OUT_DIR)

#!/usr/bin/env python3
"""
Combined SFS figure: bars = Predictor Importance (Gain), line = Cumulative AUC.
Dual Y-axes, single panel. Both Non-MRI and +MRI versions + side-by-side comparison.
"""
import pandas as pd, numpy as np, matplotlib, os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

BASE = '/Users/guxiao/Downloads/MCI-AD/UKB_DRP-main'
NO_MRI_DIR = os.path.join(BASE, 'local_data/Results_v2/DM_full')
MRI_DIR    = os.path.join(BASE, 'local_data/Results_imaging/DM_full_img_full')
OUT_DIR    = os.path.join(BASE, 'local_data/Results_v2/_figures')
os.makedirs(OUT_DIR, exist_ok=True)

FEAT_NAMES = {
    '34-0.0': 'Year of birth (Age)',
    '400-0.0': 'Reaction time (mean)',
    '137-0.0': 'Number of medications',
    '2188-0.0_c1': 'Diabetes (Dr diagnosed)',
    '30710-0.0': 'C-reactive protein (CRP)',
    '20110-0.4_pos': "Mother: dementia/Alzheimer's",
    '1090-0.0': 'Trail making test',
    '6142-0.3_pos': 'Employment: retired',
    '20023-0.0': 'Reaction time (matches)',
    '3526-0.0': "Mother's age at death",
    '12651-0.0': 'MRI: L Hippocampal Subiculum Vol',
    '26643-0.0': 'MRI: L Subiculum Volume',
    '1200-0.0_c0': 'Sleep duration (hours)',
}
def fl(fid): return FEAT_NAMES.get(fid, fid)

df_no = pd.read_csv(os.path.join(NO_MRI_DIR, 's04_selected_features.csv'))
df_mri = pd.read_csv(os.path.join(MRI_DIR, 's04_selected_features.csv'))

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['Arial','DejaVu Sans'],
    'font.size': 11, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'figure.dpi': 150, 'savefig.dpi': 300,
})

CLIN = '#3A7CA5'; IMG = '#D64545'

def make_combined(df, outpath, title):
    """Single panel: bars=Gain importance, line=Cumulative AUC."""
    steps  = np.arange(1, 11)
    gain   = df['Gain'].values
    auc    = df['CumulativeAUC'].values
    fids   = df['Features'].values
    labels = [fl(f) for f in fids]
    is_img = ['MRI' in fl(f) for f in fids]

    fig, ax1 = plt.subplots(figsize=(14.5, 7))

    # ═══ BARS: Predictor Importance (Gain) ═══
    bar_colors = [IMG if img else CLIN for img in is_img]
    bars = ax1.bar(steps, gain, color=bar_colors, width=0.55, edgecolor='white', linewidth=0.6, zorder=2, alpha=0.88)
    for bar, img in zip(bars, is_img):
        if img: bar.set_edgecolor(IMG); bar.set_linewidth(1.8)

    ax1.set_ylabel('Predictor Importance (LightGBM Gain)', fontweight='bold', color='#444444')
    ax1.set_ylim(0, max(gain) * 1.18)
    ax1.tick_params(axis='y', labelcolor='#666666')

    # ═══ LINE: Cumulative AUC ═══
    ax2 = ax1.twinx()
    ax2.plot(steps, auc, 'o-', color='#222222', linewidth=2.5, markersize=10,
             markerfacecolor='white', markeredgewidth=2.0, zorder=5)
    ax2.set_ylabel('Cumulative AUC (5-fold CV)', fontweight='bold', color='#222222')
    y_bot = max(0.79, np.floor(auc.min()*100)/100 - 0.003)
    y_top = min(0.85, np.ceil(auc.max()*100)/100 + 0.005)
    ax2.set_ylim(y_bot, y_top)
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    ax2.tick_params(axis='y', labelcolor='#222222')

    # ═══ Cumulative AUC labels near each point ═══
    for s, a in zip(steps, auc):
        ax2.annotate(f'{a:.4f}', (s, a + 0.0018), fontsize=7.8, color='#333333', ha='center',
                     fontweight='bold')

    # ═══ Feature name labels on bars ═══
    for i, (s, g, lbl, img) in enumerate(zip(steps, gain, labels, is_img)):
        short = lbl if len(lbl) <= 28 else lbl[:26] + '..'
        color = IMG if img else '#444444'
        weight = 'bold' if img else 'normal'
        # Put label above bar, rotated
        ax1.annotate(short, (s, g + max(gain)*0.02), fontsize=7.8, color=color,
                     fontweight=weight, ha='center', va='bottom', rotation=25)

    # ═══ Axis labels ═══
    ax1.set_xlabel('SFS Step (Sequential Forward Selection order)', fontweight='bold')
    ax1.set_xticks(steps)
    ax1.set_xlim(0.2, 10.8)

    # ═══ Legend ═══
    legend_els = [
        Patch(facecolor=CLIN, alpha=0.88, label='Clinical feature (Importance)'),
        Line2D([0],[0], marker='o', color='#222222', markerfacecolor='white',
               markeredgewidth=2.0, markersize=9, label='Cumulative AUC'),
    ]
    if any(is_img):
        legend_els.insert(1, Patch(facecolor=IMG, alpha=0.88, label='MRI feature (Importance)'))
    # Right-axis indicator
    ax1.legend(handles=legend_els, loc='upper left', fontsize=9, framealpha=0.9)

    ax1.set_title(f'Sequential Forward Selection — {title}\n(DM_full, UKB ~425K participants)', fontweight='bold', pad=12)
    ax1.grid(axis='y', alpha=0.15, linestyle='--')
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f'[OK] {outpath}')


# ═══ Side-by-side comparison ═══
def make_side_by_side(df1, df2, title1, title2, outpath):
    fig, (ax1a, ax1b) = plt.subplots(1, 2, figsize=(24, 7.5))

    for panel_idx, (df, title, ax1) in enumerate([
        (df1, title1, ax1a), (df2, title2, ax1b)
    ]):
        steps  = np.arange(1, 11)
        gain   = df['Gain'].values
        auc    = df['CumulativeAUC'].values
        fids   = df['Features'].values
        labels = [fl(f) for f in fids]
        is_img = ['MRI' in fl(f) for f in fids]

        # Bars
        bar_colors = [IMG if img else CLIN for img in is_img]
        bars = ax1.bar(steps, gain, color=bar_colors, width=0.55, edgecolor='white', linewidth=0.6, zorder=2, alpha=0.88)
        for bar, img in zip(bars, is_img):
            if img: bar.set_edgecolor(IMG); bar.set_linewidth(1.8)

        if panel_idx == 0:
            ax1.set_ylabel('Predictor Importance\n(LightGBM Gain)', fontweight='bold', color='#444444')
        ax1.set_ylim(0, 0.72)
        ax1.tick_params(axis='y', labelcolor='#666666')

        # Line
        ax2 = ax1.twinx()
        ax2.plot(steps, auc, 'o-', color='#222222', linewidth=2.5, markersize=10,
                 markerfacecolor='white', markeredgewidth=2.0, zorder=5)
        if panel_idx == 1:
            ax2.set_ylabel('Cumulative AUC', fontweight='bold', color='#222222')
        ax2.set_ylim(0.793, 0.844)
        ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
        ax2.tick_params(axis='y', labelcolor='#222222')

        # AUC labels
        for s, a in zip(steps, auc):
            ax2.annotate(f'{a:.4f}', (s, a + 0.0015), fontsize=7.5, color='#333333', ha='center', fontweight='bold')

        # Feature labels
        for s, g, lbl, img in zip(steps, gain, labels, is_img):
            short = lbl if len(lbl) <= 26 else lbl[:24] + '..'
            color = IMG if img else '#444444'
            ax1.annotate(short, (s, g + 0.012), fontsize=7.5, color=color,
                         fontweight='bold' if img else 'normal', ha='center', va='bottom', rotation=25)

        ax1.set_xlabel('SFS Step', fontweight='bold')
        ax1.set_xticks(steps)
        ax1.set_xlim(0.2, 10.8)
        ax1.set_title(title, fontweight='bold', fontsize=13)
        ax1.grid(axis='y', alpha=0.15, linestyle='--')

    # Shared legend below
    legend_els = [
        Patch(facecolor=CLIN, alpha=0.88, label='Clinical (Importance)'),
        Patch(facecolor=IMG, alpha=0.88, label='MRI (Importance)'),
        Line2D([0],[0], marker='o', color='#222222', markerfacecolor='white',
               markeredgewidth=2.0, markersize=8, label='Cumulative AUC'),
    ]
    fig.legend(handles=legend_els, loc='lower center', ncol=3, fontsize=10, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.06))

    fig.suptitle('Sequential Forward Selection: Predictor Importance & Cumulative AUC\n(DM_full target, UKB ~425K participants)',
                 fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f'[OK] {outpath}')


# ═══ Generate all ═══
make_combined(df_no, os.path.join(OUT_DIR, 'SFS_NoMRI_Importance.png'),
              'Non-MRI Model (1076 clinical features)')
make_combined(df_mri, os.path.join(OUT_DIR, 'SFS_MRI_Importance.png'),
              '+MRI Model (3250 clinical + MRI features)')
make_side_by_side(df_no, df_mri,
                  'Non-MRI (1076 clinical features)', '+MRI (3250 clinical + MRI features)',
                  os.path.join(OUT_DIR, 'SFS_SideBySide_Importance.png'))

print('\nDone!')

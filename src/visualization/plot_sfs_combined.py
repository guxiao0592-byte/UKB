#!/usr/bin/env python3
"""
Combined SFS plot: bars (per-step gain) + line (cumulative AUC) in ONE panel.
Dual Y-axes: left = cumulative AUC (line), right = per-step gain (bars).
"""
import pandas as pd, numpy as np, matplotlib, os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

BASE = '/Users/guxiao/Downloads/MCI-AD/UKB_DRP-main'
NO_MRI_DIR = os.path.join(BASE, 'local_data/Results_v2/DM_full')
MRI_DIR    = os.path.join(BASE, 'local_data/Results_imaging/DM_full_img_full')
OUT_DIR    = os.path.join(BASE, 'local_data/Results_v2/_figures')
os.makedirs(OUT_DIR, exist_ok=True)

FEAT_NAMES = {
    '34-0.0':'Year of birth (Age)','400-0.0':'Reaction time (mean)',
    '137-0.0':'Number of medications','2188-0.0_c1':'Diabetes (Dr diagnosed)',
    '30710-0.0':'C-reactive protein','20110-0.4_pos':"Mother: dementia/Alzheimer's",
    '1090-0.0':'Trail making test','6142-0.3_pos':'Employment: retired',
    '20023-0.0':'Reaction time (match)','3526-0.0':"Mother's age at death",
    '12651-0.0':'MRI: L Hippocampal Subiculum','26643-0.0':'MRI: L Subiculum Volume',
    '1200-0.0_c0':'Sleep duration (hours)',
}
def fl(fid): return FEAT_NAMES.get(fid, fid)

df_no = pd.read_csv(os.path.join(NO_MRI_DIR, 's04_sfs_history.csv'))
df_mri = pd.read_csv(os.path.join(MRI_DIR, 's04_sfs_history.csv'))

plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
    'font.size':11,'axes.titlesize':14,'axes.labelsize':11,'figure.dpi':150,'savefig.dpi':300})

CLIN = '#3A7CA5'; IMG = '#D64545'; BAND = 0.12

def plot_combined(df, outpath, title):
    steps = df['selected_count'].values
    auc = df['auc_mean'].values
    auc_std = df['auc_std'].values
    fids = df['feature_added'].values
    labels = [fl(f) for f in fids]
    is_img = ['MRI' in fl(f) for f in fids]

    gains = [auc[0] - 0.5]
    for i in range(1, len(auc)): gains.append(auc[i] - auc[i-1])
    gains = np.array(gains)

    fig, ax1 = plt.subplots(figsize=(14, 6.5))

    # ── Bars: per-step gain (left axis, but lower values) ──
    bar_colors = [IMG if img else CLIN for img in is_img]
    bars = ax1.bar(steps, gains, color=bar_colors, width=0.5, edgecolor='white', linewidth=0.6, zorder=2, alpha=0.85)
    for bar, img in zip(bars, is_img):
        if img: bar.set_edgecolor(IMG); bar.set_linewidth(1.5)
    ax1.set_ylabel('Incremental AUC Gain per Step', fontweight='bold', color='#555555')
    ax1.tick_params(axis='y', labelcolor='#888888')
    ax1.set_ylim(0, max(gains)*1.15)

    # ── Line: cumulative AUC (right axis) ──
    ax2 = ax1.twinx()
    ax2.fill_between(steps, auc - auc_std, auc + auc_std, alpha=BAND, color='#888888', zorder=1)
    ax2.plot(steps, auc, '-', color='#444444', linewidth=2.0, zorder=4)
    # Scatter points
    for i, (s, a, img) in enumerate(zip(steps, auc, is_img)):
        marker = 'D' if img else 'o'
        sz = 110 if img else 80
        ax2.scatter(s, a, c=IMG if img else CLIN, s=sz, zorder=5, edgecolors='white', linewidths=0.8)

    ax2.set_ylabel('Cumulative AUC (5-fold CV)', fontweight='bold', color='#333333')
    ax2.set_ylim(0.793, 0.845)
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))

    # ── Annotations ──
    for i, (s, a, lbl, img) in enumerate(zip(steps, auc, labels, is_img)):
        va = 'bottom'; off = 0.004
        if s <= 2: off = -0.006; va = 'top'
        elif s >= 9: off = 0.004; va = 'bottom'
        elif s % 3 == 0: off = -0.005; va = 'top'
        color = IMG if img else '#444444'
        weight = 'bold' if img else 'normal'
        ax2.annotate(lbl, (s, a + off), fontsize=8.5, color=color,
                     fontweight=weight, ha='center', va=va)

    # Cumulative AUC labels near points
    for i, (s, a) in enumerate(zip(steps, auc)):
        ax2.annotate(f'{a:.4f}', (s, a - 0.007), fontsize=6.5, color='#666666', ha='center', va='top')

    ax1.set_xlabel('SFS Step (Number of Features Selected)', fontweight='bold')
    ax1.set_xticks(steps)
    ax1.set_xticklabels([str(s) for s in steps])
    ax1.set_xlim(0.3, 10.7)
    ax1.set_title(f'Sequential Forward Selection — {title}\n(DM_full target, UKB ~425K participants)', fontweight='bold')

    # ── Legend ──
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_els = [
        Patch(facecolor=CLIN, alpha=0.85, label='Clinical feature (gain)'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor=CLIN, markersize=9, label='Clinical feature (AUC)'),
    ]
    if any(is_img):
        legend_els += [
            Patch(facecolor=IMG, alpha=0.85, label='MRI feature (gain)'),
            Line2D([0],[0], marker='D', color='w', markerfacecolor=IMG, markersize=9, label='MRI feature (AUC)'),
        ]
    legend_els.append(Patch(facecolor='#888888', alpha=BAND, label='±1 SD'))
    ax1.legend(handles=legend_els, loc='upper left', fontsize=8.5, framealpha=0.9, ncol=2)

    ax1.grid(axis='y', alpha=0.2, linestyle='--')
    plt.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)
    print(f'[OK] {outpath}')

# ═══ Generate ═══
plot_combined(df_no, os.path.join(OUT_DIR, 'SFS_Combined_NoMRI.png'), 'Non-MRI Model (1076 clinical features)')
plot_combined(df_mri, os.path.join(OUT_DIR, 'SFS_Combined_MRI.png'), '+MRI Model (3250 clinical + MRI features)')

# ═══ Side-by-side comparison ═══
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(22, 7))

for ax_idx, (df, title, ax) in enumerate([
    (df_no, 'Non-MRI (1076 features)', None),
    (df_mri, '+MRI (3250 features)', None),
]):
    if ax_idx == 0: ax = ax_l
    else: ax = ax_r

    steps = df['selected_count'].values
    auc = df['auc_mean'].values
    auc_std = df['auc_std'].values
    fids = df['feature_added'].values
    labels = [fl(f) for f in fids]
    is_img = ['MRI' in fl(f) for f in fids]

    gains = [auc[0] - 0.5]
    for i in range(1, len(auc)): gains.append(auc[i] - auc[i-1])
    gains = np.array(gains)

    # Bars
    bar_colors = [IMG if img else CLIN for img in is_img]
    bars = ax.bar(steps, gains, color=bar_colors, width=0.5, edgecolor='white', linewidth=0.6, zorder=2, alpha=0.85)
    for bar, img in zip(bars, is_img):
        if img: bar.set_edgecolor(IMG); bar.set_linewidth(1.5)

    # Line on twin axis
    ax2_ = ax.twinx()
    ax2_.fill_between(steps, auc - auc_std, auc + auc_std, alpha=BAND, color='#888888', zorder=1)
    ax2_.plot(steps, auc, '-', color='#444444', linewidth=2.0, zorder=4)

    for i, (s, a, img) in enumerate(zip(steps, auc, is_img)):
        marker = 'D' if img else 'o'
        ax2_.scatter(s, a, c=IMG if img else CLIN, s=100 if img else 75, zorder=5, edgecolors='white', linewidths=0.8)

    # Short labels
    for i, (s, a, lbl, img) in enumerate(zip(steps, auc, labels, is_img)):
        short = lbl[:22] + '..' if len(lbl) > 24 else lbl
        va = 'bottom'; off = 0.004
        if s <= 2: off = -0.006; va = 'top'
        color = IMG if img else '#444444'
        ax2_.annotate(short, (s, a + off), fontsize=7.5, color=color,
                      fontweight='bold' if img else 'normal', ha='center', va=va)

    for i, (s, a) in enumerate(zip(steps, auc)):
        ax2_.annotate(f'{a:.4f}', (s, a - 0.007), fontsize=6, color='#666666', ha='center', va='top')

    ax.set_xlabel('SFS Step', fontweight='bold')
    if ax_idx == 0:
        ax.set_ylabel('AUC Gain per Step', fontweight='bold', color='#555555')
    ax2_.set_ylim(0.793, 0.845)
    ax2_.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    ax2_.set_ylabel('Cumulative AUC', fontweight='bold', color='#333333')
    ax.set_xticks(steps)
    ax.set_xlim(0.3, 10.7)
    ax.set_title(title, fontweight='bold', fontsize=13)
    ax.grid(axis='y', alpha=0.2, linestyle='--')

# Legend
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
fig.legend(handles=[
    Patch(facecolor=CLIN, alpha=0.85, label='Clinical gain'),
    Patch(facecolor=IMG, alpha=0.85, label='MRI gain'),
    Line2D([0],[0], marker='o', color=CLIN, markersize=8, label='Clinical AUC'),
    Line2D([0],[0], marker='D', color=IMG, markersize=8, label='MRI AUC'),
], loc='lower center', ncol=4, fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, -0.12))

fig.suptitle('Sequential Forward Selection — Non-MRI vs +MRI\n(DM_full target, UKB ~425K participants)',
             fontweight='bold', fontsize=15, y=1.02)
plt.tight_layout()
outpath = os.path.join(OUT_DIR, 'SFS_Combined_TwoPanel.png')
fig.savefig(outpath)
plt.close(fig)
print(f'[OK] {outpath}')
print('Done!')

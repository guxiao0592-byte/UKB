#!/usr/bin/env python3
"""
Extended evaluation: clinical interpretability metrics beyond AUC.
- Decision Curve Analysis (net benefit)
- Risk stratification (decile-based)
- Precision-Recall curves
- ECE (Expected Calibration Error)
- PPV/NPV at key cutoffs
- Risk distribution by case/control
"""
import os, json, warnings
import numpy as np
import pandas as pd

PROJECT_ROOT = os.environ.get('PROJECT_ROOT',
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RESULTS_ROOT = os.path.join(PROJECT_ROOT, 'local_data', 'Results_v2')
from scipy import stats
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    brier_score_loss, confusion_matrix, log_loss
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

warnings.filterwarnings('ignore')

TARGETS = ['DM_full', 'DM_10yrs', 'DM_5yrs', 'AD_full', 'AD_10yrs', 'AD_5yrs']
TARGET_LABELS = {
    'DM_full': 'All-cause Dementia (full)', 'DM_10yrs': 'All-cause Dementia (10yr)',
    'DM_5yrs': 'All-cause Dementia (5yr)', 'AD_full': "Alzheimer's Disease (full)",
    'AD_10yrs': "Alzheimer's Disease (10yr)", 'AD_5yrs': "Alzheimer's Disease (5yr)",
}
PAPER_AUC = {
    'DM_full': 0.848, 'DM_10yrs': 0.849, 'DM_5yrs': 0.847,
    'AD_full': 0.862, 'AD_10yrs': 0.866, 'AD_5yrs': 0.890,
}
EXTENDED_DIR = os.path.join(RESULTS_ROOT, '_extended_eval')
os.makedirs(EXTENDED_DIR, exist_ok=True)

# ── Load CV predictions ────────────────────────────────────────
def load_cv_predictions(target):
    """Load 5-fold CV predictions, flatten into single arrays."""
    pred_df = pd.read_csv(os.path.join(RESULTS_ROOT, target, 'pred_prob_cv_df.csv'))
    test_df = pd.read_csv(os.path.join(RESULTS_ROOT, target, 'test_cv_df.csv'))
    y_true_all, y_pred_all = [], []
    for col in pred_df.columns:
        mask = pred_df[col].notna() & test_df[col].notna()
        y_true_all.append(test_df[col][mask].values)
        y_pred_all.append(pred_df[col][mask].values)
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    return y_true, y_pred

# ── 1. Decision Curve Analysis ─────────────────────────────────
def net_benefit(y_true, y_pred, pt):
    """Net benefit at threshold probability pt."""
    y_pred_bin = (y_pred >= pt).astype(int)
    tp = np.sum((y_pred_bin == 1) & (y_true == 1))
    fp = np.sum((y_pred_bin == 1) & (y_true == 0))
    n = len(y_true)
    nb = tp / n - fp / n * (pt / (1 - pt))
    return nb

def decision_curve(y_true, y_pred, n_points=100):
    """Compute decision curve: net benefit vs threshold probability."""
    thresholds = np.linspace(0.001, 0.30, n_points)
    nb_model = [net_benefit(y_true, y_pred, t) for t in thresholds]
    nb_all = [net_benefit(y_true, np.ones_like(y_pred), t) for t in thresholds]
    nb_none = np.zeros(len(thresholds))
    return thresholds, nb_model, nb_all, nb_none

def plot_dca_all(targets_data, save_path):
    """Multi-panel DCA plot for all 6 targets."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for ax, target in zip(axes.flat, TARGETS):
        y_true, y_pred = targets_data[target]
        thresholds, nb_model, nb_all, nb_none = decision_curve(y_true, y_pred)
        ax.plot(thresholds, nb_model, 'b-', lw=2, label='Model')
        ax.plot(thresholds, nb_all, 'k--', lw=1.5, label='Treat All')
        ax.plot(thresholds, nb_none, 'k:', lw=1, label='Treat None')
        ax.set_xlabel('Threshold Probability', fontsize=11)
        ax.set_ylabel('Net Benefit', fontsize=11)
        ax.set_title(TARGET_LABELS[target], fontsize=13, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_xlim(0, 0.3)
        n_pos = y_true.sum(); rate = n_pos / len(y_true) * 100
        ax.text(0.98, 0.95, f'n={len(y_true):,}\nEvents={n_pos:,} ({rate:.2f}%)',
                transform=ax.transAxes, ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    fig.suptitle('Decision Curve Analysis — Net Benefit of 5-Year Dementia Risk Models',
                 fontsize=16, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

# ── 2. Risk Stratification ─────────────────────────────────────
def risk_stratification_table(y_true, y_pred, n_groups=10):
    """Compute observed event rate per predicted risk decile."""
    df = pd.DataFrame({'true': y_true, 'pred': y_pred})
    df['decile'] = pd.qcut(df['pred'], q=n_groups, labels=False, duplicates='drop')
    n_groups_actual = df['decile'].nunique()
    table = []
    for d in range(n_groups_actual):
        sub = df[df['decile'] == d]
        n = len(sub); events = sub['true'].sum()
        table.append({
            'decile': d + 1,
            'n': n, 'events': int(events),
            'obs_rate': round(events / n * 100, 2) if n > 0 else 0,
            'pred_mean': round(sub['pred'].mean(), 4),
            'pred_min': round(sub['pred'].min(), 4),
            'pred_max': round(sub['pred'].max(), 4),
        })
    return pd.DataFrame(table)

def plot_risk_stratification(targets_data, save_path):
    """Observed vs expected risk by decile for all targets."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for ax, target in zip(axes.flat, TARGETS):
        y_true, y_pred = targets_data[target]
        tbl = risk_stratification_table(y_true, y_pred)
        x = range(len(tbl))
        ax.bar(x, tbl['obs_rate'], color='steelblue', alpha=0.7, label='Observed rate (%)')
        ax.plot(x, tbl['pred_mean'] * 100, 'ro-', lw=2, markersize=6, label='Mean predicted (%)')
        ax.set_xticks(x); ax.set_xticklabels(tbl['decile'], fontsize=8)
        ax.set_xlabel('Risk Decile (1=lowest, 10=highest)', fontsize=10)
        ax.set_ylabel('Event Rate (%)', fontsize=10)
        ax.set_title(TARGET_LABELS[target], fontsize=12, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
        ece = expected_calibration_error(y_true, y_pred, n_bins=10)
        ax.text(0.98, 0.92, f'ECE={ece:.4f}', transform=ax.transAxes,
                ha='right', fontsize=9, bbox=dict(boxstyle='round', facecolor='lightyellow'))
    fig.suptitle('Risk Stratification: Observed vs Predicted Event Rate by Decile',
                 fontsize=16, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

# ── 3. Expected Calibration Error ──────────────────────────────
def expected_calibration_error(y_true, y_pred, n_bins=10):
    """ECE: weighted average of |observed - predicted| per bin."""
    bins = np.percentile(y_pred, np.linspace(0, 100, n_bins + 1))
    bins[0], bins[-1] = 0, 1.001
    ece = 0
    for i in range(n_bins):
        mask = (y_pred >= bins[i]) & (y_pred < bins[i + 1])
        if mask.sum() == 0:
            continue
        obs_rate = y_true[mask].mean()
        pred_mean = y_pred[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(obs_rate - pred_mean)
    return ece

# ── 4. Precision-Recall ────────────────────────────────────────
def plot_pr_curves(targets_data, save_path):
    """Precision-Recall curves for all targets (better for imbalanced data)."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for ax, target in zip(axes.flat, TARGETS):
        y_true, y_pred = targets_data[target]
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)
        baseline = y_true.sum() / len(y_true)
        ax.plot(recall, precision, 'b-', lw=2, label=f'Model (AP={ap:.3f})')
        ax.axhline(baseline, color='k', linestyle='--', lw=1, label=f'Baseline ({baseline:.4f})')
        ax.set_xlabel('Recall', fontsize=11); ax.set_ylabel('Precision', fontsize=11)
        ax.set_title(f"{TARGET_LABELS[target]}", fontsize=12, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.suptitle('Precision-Recall Curves',
                 fontsize=16, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

# ── 5. Risk Distribution ───────────────────────────────────────
def plot_risk_distribution(targets_data, save_path):
    """Histogram of predicted risk, split by case/control."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for ax, target in zip(axes.flat, TARGETS):
        y_true, y_pred = targets_data[target]
        cases = y_pred[y_true == 1]; controls = y_pred[y_true == 0]
        ax.hist(controls, bins=60, alpha=0.6, color='steelblue', density=True, label='Controls')
        ax.hist(cases, bins=60, alpha=0.6, color='coral', density=True, label='Cases')
        ax.set_xlabel('Predicted Risk', fontsize=10); ax.set_ylabel('Density', fontsize=10)
        ax.set_title(TARGET_LABELS[target], fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        # KS statistic
        ks_stat, ks_p = stats.ks_2samp(cases, controls)
        ax.text(0.98, 0.92, f'KS={ks_stat:.3f}\np={ks_p:.2e}',
                transform=ax.transAxes, ha='right', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='lightcyan'))
    fig.suptitle('Risk Score Distribution: Cases vs Controls',
                 fontsize=16, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

# ── 6. ROC curves with CI ──────────────────────────────────────
def plot_roc_summary(targets_data, save_path):
    """Single-panel ROC overlay + AUC comparison table."""
    fig, ax = plt.subplots(figsize=(10, 9))
    colors = plt.cm.tab10.colors
    summary = []
    for i, target in enumerate(TARGETS):
        y_true, y_pred = targets_data[target]
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc_val = roc_auc_score(y_true, y_pred)
        # Bootstrap CI
        n_boot = 500; aucs = []
        rng = np.random.RandomState(42)
        n = len(y_true)
        for _ in range(n_boot):
            idx = rng.choice(n, n, replace=True)
            try:
                aucs.append(roc_auc_score(y_true[idx], y_pred[idx]))
            except ValueError:
                pass
        ci_low, ci_high = np.percentile(aucs, [2.5, 97.5])
        summary.append({
            'Target': TARGET_LABELS[target],
            'AUC': f'{auc_val:.3f}',
            '95% CI': f'({ci_low:.3f}, {ci_high:.3f})',
            'Paper AUC': f'{PAPER_AUC[target]:.3f}',
            'Δ': f'{auc_val - PAPER_AUC[target]:.3f}',
        })
        ax.plot(fpr, tpr, color=colors[i], lw=2,
                label=f"{target}: AUC={auc_val:.3f} ({ci_low:.3f}-{ci_high:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves with 95% Bootstrap CI', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right'); ax.grid(alpha=0.3)

    # Summary table inset
    tbl = pd.DataFrame(summary)
    table_ax = fig.add_axes([0.15, 0.08, 0.7, 0.3])
    table_ax.axis('off')
    table_ax.table(cellText=tbl.values, colLabels=tbl.columns, cellLoc='center',
                   loc='center', colColours=['#2a2a4e']*len(tbl.columns))
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return summary

# ── 7. Clinical utility at key thresholds ──────────────────────
def clinical_utility_summary(targets_data):
    """Compute PPV, NPV, LR+, LR- at clinically relevant cutoffs."""
    results = []
    for target in TARGETS:
        y_true, y_pred = targets_data[target]
        auc_val = roc_auc_score(y_true, y_pred)
        ap_val = average_precision_score(y_true, y_pred)
        brier = brier_score_loss(y_true, y_pred)
        ece = expected_calibration_error(y_true, y_pred)

        # Find best cutoff by Youden
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        youden = tpr - fpr
        best_idx = np.argmax(youden)
        best_cutoff = thresholds[best_idx] if best_idx < len(thresholds) else 0.05

        # At best cutoff
        y_bin = (y_pred >= best_cutoff).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_bin).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        lr_pos = sens / (1 - spec) if spec < 1 else float('inf')
        lr_neg = (1 - sens) / spec if spec > 0 else float('inf')
        prev = y_true.sum() / len(y_true)

        # Low risk (bottom 50%) vs High risk (top 20%)
        p50 = np.percentile(y_pred, 50)
        p80 = np.percentile(y_pred, 80)
        low_mask = y_pred <= p50
        high_mask = y_pred >= p80
        low_rate = y_true[low_mask].mean() if low_mask.sum() > 0 else 0
        high_rate = y_true[high_mask].mean() if high_mask.sum() > 0 else 0

        results.append({
            'Target': TARGET_LABELS[target],
            'AUC': round(auc_val, 3),
            'AP': round(ap_val, 3),
            'Brier': round(brier, 4),
            'ECE': round(ece, 4),
            'Best_Cutoff': round(best_cutoff, 4),
            'Sens': round(sens, 3),
            'Spec': round(spec, 3),
            'PPV': round(ppv, 3),
            'NPV': round(npv, 3),
            'LR+': round(lr_pos, 1) if np.isfinite(lr_pos) else 'inf',
            'LR-': round(lr_neg, 3),
            'Prevalence': f'{prev*100:.2f}%',
            'LowRisk_Rate': f'{low_rate*100:.2f}%',
            'HighRisk_Rate': f'{high_rate*100:.2f}%',
            'Risk_Gradient': round(high_rate / max(low_rate, 1e-6), 1),
        })
    return pd.DataFrame(results)

# ── 8. Comprehensive per-target report ─────────────────────────
def per_target_report(target, y_true, y_pred, out_dir):
    """Generate detailed report for a single target."""
    os.makedirs(out_dir, exist_ok=True)

    # Risk stratification
    risk_tbl = risk_stratification_table(y_true, y_pred)
    risk_tbl.to_csv(os.path.join(out_dir, 'risk_stratification.csv'), index=False)

    # Full ROC data
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'threshold': thresholds}).to_csv(
        os.path.join(out_dir, 'roc_data.csv'), index=False)

    # Full PR data
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pd.DataFrame({'precision': precision, 'recall': recall}).to_csv(
        os.path.join(out_dir, 'pr_data.csv'), index=False)

    # DCA data
    dca_thresh, nb_model, nb_all, nb_none = decision_curve(y_true, y_pred)
    pd.DataFrame({
        'threshold': dca_thresh, 'nb_model': nb_model,
        'nb_treat_all': nb_all, 'nb_treat_none': nb_none
    }).to_csv(os.path.join(out_dir, 'dca_data.csv'), index=False)

    return risk_tbl

# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 70)
    print("Extended Evaluation: Clinical Interpretability Metrics")
    print("=" * 70)

    # Load all data
    targets_data = {}
    for target in TARGETS:
        y_true, y_pred = load_cv_predictions(target)
        targets_data[target] = (y_true, y_pred)
        print(f"  {target}: n={len(y_true):,}, events={y_true.sum():,} "
              f"({y_true.mean()*100:.3f}%)")

    # 1. Decision Curve Analysis
    print("\n[1/7] Decision Curve Analysis...")
    plot_dca_all(targets_data, os.path.join(EXTENDED_DIR, 'dca_all_targets.png'))
    for target in TARGETS:
        d = os.path.join(EXTENDED_DIR, target)
        os.makedirs(d, exist_ok=True)
        decision_curve(targets_data[target][0], targets_data[target][1])

    # 2. Risk Stratification
    print("[2/7] Risk Stratification Tables...")
    plot_risk_stratification(targets_data, os.path.join(EXTENDED_DIR, 'risk_stratification.png'))

    # 3. ECE
    print("[3/7] Expected Calibration Error...")
    for target in TARGETS:
        ece = expected_calibration_error(*targets_data[target])
        print(f"  {target}: ECE = {ece:.4f}")

    # 4. Precision-Recall
    print("[4/7] Precision-Recall Curves...")
    plot_pr_curves(targets_data, os.path.join(EXTENDED_DIR, 'pr_curves.png'))

    # 5. Risk Distribution
    print("[5/7] Risk Score Distributions...")
    plot_risk_distribution(targets_data, os.path.join(EXTENDED_DIR, 'risk_distribution.png'))

    # 6. ROC Summary
    print("[6/7] ROC Summary with Bootstrap CI...")
    roc_summary = plot_roc_summary(targets_data, os.path.join(EXTENDED_DIR, 'roc_summary.png'))

    # 7. Clinical Utility
    print("[7/7] Clinical Utility Summary...")
    clin_df = clinical_utility_summary(targets_data)
    clin_df.to_csv(os.path.join(EXTENDED_DIR, 'clinical_utility_summary.csv'), index=False)

    # Per-target detailed reports
    print("\nPer-target reports...")
    for target in TARGETS:
        y_true, y_pred = targets_data[target]
        per_target_report(target, y_true, y_pred, os.path.join(EXTENDED_DIR, target))

    # Print clinical utility table
    print("\n" + "=" * 90)
    print("CLINICAL UTILITY SUMMARY")
    print("=" * 90)
    print(clin_df[['Target', 'AUC', 'AP', 'Brier', 'ECE', 'PPV', 'NPV',
                    'LowRisk_Rate', 'HighRisk_Rate', 'Risk_Gradient']].to_string(index=False))

    print(f"\nAll results saved to: {EXTENDED_DIR}")

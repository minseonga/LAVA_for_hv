"""
Hard Veto Visualization Suite
================================
FRG = faithful_minus_global_attn  (C feature)
GMI = guidance_mismatch_score      (E feature)

Plots:
  1. FRG × GMI scatter decision map (D1/D2 + veto boundary)
  2. FRG × GMI utility heatmap (net_gain per grid cell)
  3. D1 vs D2 violin+box – FRG
  4. D1 vs D2 violin+box – GMI
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.colors import TwoSlopeNorm
import seaborn as sns
from pathlib import Path

# ─── paths ───────────────────────────────────────────────────────────────────
CASES_CSV = '/home/kms/LLaVA_calibration/experiments/pope_full_9000/vista_method_9000/taxonomy/per_case_compare.csv'
FEAT_CSV  = '/home/kms/LLaVA_calibration/experiments/pope_feature_screen_v1_full9000/features_unified_table.csv'
OUT_DIR   = Path('/home/kms/LLaVA_calibration/experiments/pope_full_9000/vista_method_9000/viz')
OUT_DIR.mkdir(parents=True, exist_ok=True)

FRG_COL = 'faithful_minus_global_attn'   # C
GMI_COL = 'guidance_mismatch_score'       # E
TAU_C   = -0.012011878027745549
TAU_E   =  0.04606491539150529

# ─── load & merge ─────────────────────────────────────────────────────────────
print("[info] Loading data …")
df_c = pd.read_csv(CASES_CSV)
df_f = pd.read_csv(FEAT_CSV, usecols=['id', FRG_COL, GMI_COL])
df   = pd.merge(df_c, df_f, on='id')
print(f"[info] Merged {len(df)} rows")

df_d1 = df[df['case_type'] == 'vga_improvement'].copy()   # D1: keep
df_d2 = df[df['case_type'] == 'vga_regression'].copy()    # D2: veto targets
df_other = df[~df['case_type'].isin(['vga_improvement', 'vga_regression'])].copy()
print(f"[info] D1={len(df_d1)}  D2={len(df_d2)}  other={len(df_other)}")

# clip to [1%, 99%] for display
def clip_pct(series, lo=0.01, hi=0.99):
    return series.clip(series.quantile(lo), series.quantile(hi))

c_all = clip_pct(df[FRG_COL])
e_all = clip_pct(df[GMI_COL])
XLIM = (c_all.min() - 0.02, c_all.max() + 0.02)
YLIM = (e_all.min() - 0.005, e_all.max() + 0.005)

COLORS = {'D1': '#2A6DD9', 'D2': '#D93025', 'other': '#AAAAAA'}
ALPHA_SC = 0.35
DOT_SZ   = 10

# ═══════════════════════════════════════════════════════════════════════════════
# Plot 1 – FRG × GMI decision map
# ═══════════════════════════════════════════════════════════════════════════════
print("[plot 1] Decision map …")
fig, ax = plt.subplots(figsize=(9, 7))

ax.scatter(clip_pct(df_other[FRG_COL]), clip_pct(df_other[GMI_COL]),
           c=COLORS['other'], alpha=0.15, s=DOT_SZ*0.6, label='No-change cases')
ax.scatter(clip_pct(df_d1[FRG_COL]), clip_pct(df_d1[GMI_COL]),
           c=COLORS['D1'], alpha=ALPHA_SC, s=DOT_SZ, label='D1 – Improvement (keep)')
ax.scatter(clip_pct(df_d2[FRG_COL]), clip_pct(df_d2[GMI_COL]),
           c=COLORS['D2'], alpha=ALPHA_SC+0.1, s=DOT_SZ, marker='x', label='D2 – Regression (veto target)')

# veto boundary
ax.axvline(TAU_C, color='#555', lw=1.8, ls='--', label=f'τ_C = {TAU_C:.4f}')
ax.axhline(TAU_E, color='#888', lw=1.8, ls=':',  label=f'τ_E = {TAU_E:.4f}')

# shade veto region
xmin_n = (TAU_C - XLIM[0]) / (XLIM[1] - XLIM[0])
ymin_n = (TAU_E - YLIM[0]) / (YLIM[1] - YLIM[0])
ax.axvspan(TAU_C, XLIM[1], alpha=0.08, color='#D93025', zorder=0)
ax.axhspan(TAU_E, YLIM[1], xmin=0, xmax=xmin_n, alpha=0.08, color='#D93025', zorder=0)

# region labels
ax.text(XLIM[1] - 0.01, YLIM[1] - 0.003, 'VETO REGION',
        ha='right', va='top', fontsize=9, color='#D93025', alpha=0.7, style='italic')
ax.text(TAU_C - 0.005, YLIM[0] + 0.003, 'SAFE REGION',
        ha='right', va='bottom', fontsize=9, color='#2A6DD9', alpha=0.7, style='italic')

ax.set_xlim(XLIM); ax.set_ylim(YLIM)
ax.set_xlabel('FRG – Faithful Minus Global Attn (C)', fontsize=12)
ax.set_ylabel('GMI – Guidance Mismatch Score (E)', fontsize=12)
ax.set_title('Hard Veto Decision Map: FRG × GMI Feature Space (VISTA / POPE)', fontsize=13, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(OUT_DIR / 'plot1_decision_map.png', dpi=200)
plt.close(fig)
print(f"  → saved plot1_decision_map.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Plot 2 – FRG × GMI utility heatmap  (grid: avg label = D2 fraction - D1 fraction)
# ═══════════════════════════════════════════════════════════════════════════════
print("[plot 2] Utility heatmap …")

BINS = 30
frg_bins = np.linspace(XLIM[0], XLIM[1], BINS + 1)
gmi_bins = np.linspace(YLIM[0], YLIM[1], BINS + 1)

def grid_stat(d1_vals_c, d1_vals_e, d2_vals_c, d2_vals_e, frg_bins, gmi_bins):
    """For each cell, compute net_veto_utility = D2_count – D1_count."""
    H_d2, _, _ = np.histogram2d(
        d2_vals_c.clip(XLIM[0], XLIM[1]),
        d2_vals_e.clip(YLIM[0], YLIM[1]),
        bins=[frg_bins, gmi_bins])
    H_d1, _, _ = np.histogram2d(
        d1_vals_c.clip(XLIM[0], XLIM[1]),
        d1_vals_e.clip(YLIM[0], YLIM[1]),
        bins=[frg_bins, gmi_bins])
    return H_d2 - H_d1   # positive = veto is beneficial

util = grid_stat(
    df_d1[FRG_COL], df_d1[GMI_COL],
    df_d2[FRG_COL], df_d2[GMI_COL],
    frg_bins, gmi_bins
)

fig, ax = plt.subplots(figsize=(9, 7))
norm = TwoSlopeNorm(vmin=util.min(), vcenter=0, vmax=util.max())
im = ax.pcolormesh(frg_bins, gmi_bins, util.T,
                   cmap='RdBu_r', norm=norm, shading='auto')
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('Net Veto Utility  (D2 count – D1 count per cell)', fontsize=10)

ax.axvline(TAU_C, color='black', lw=2, ls='--', label=f'τ_C = {TAU_C:.4f}')
ax.axhline(TAU_E, color='gray',  lw=2, ls=':',  label=f'τ_E = {TAU_E:.4f}')
ax.set_xlim(XLIM); ax.set_ylim(YLIM)
ax.set_xlabel('FRG – Faithful Minus Global Attn (C)', fontsize=12)
ax.set_ylabel('GMI – Guidance Mismatch Score (E)', fontsize=12)
ax.set_title('Veto Utility Heatmap: Red = veto helps, Blue = veto hurts (VISTA / POPE)', fontsize=12, fontweight='bold')
ax.legend(loc='upper left', fontsize=9)
ax.grid(False)
fig.tight_layout()
fig.savefig(OUT_DIR / 'plot2_utility_heatmap.png', dpi=200)
plt.close(fig)
print(f"  → saved plot2_utility_heatmap.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Plot 3 & 4 – D1 vs D2 violin+box for FRG and GMI (side by side)
# ═══════════════════════════════════════════════════════════════════════════════
print("[plot 3+4] Violin/box plots …")

violin_data = pd.concat([
    df_d1[[FRG_COL, GMI_COL]].assign(Type='D1\n(Improvement)'),
    df_d2[[FRG_COL, GMI_COL]].assign(Type='D2\n(Regression)'),
])

palette = {'D1\n(Improvement)': COLORS['D1'], 'D2\n(Regression)': COLORS['D2']}

for col, tau, label, fname in [
    (FRG_COL, TAU_C, 'FRG – Faithful Minus Global Attn (C)', 'plot3_violin_frg.png'),
    (GMI_COL, TAU_E, 'GMI – Guidance Mismatch Score (E)',    'plot4_violin_gmi.png'),
]:
    fig, ax = plt.subplots(figsize=(6, 7))

    sns.violinplot(data=violin_data, x='Type', y=col, palette=palette,
                   inner=None, ax=ax, saturation=0.85, linewidth=1.2)
    sns.boxplot(data=violin_data, x='Type', y=col,
                width=0.15, palette=palette, showfliers=False,
                ax=ax, linewidth=1.5, medianprops=dict(color='white', linewidth=2.5),
                boxprops=dict(alpha=0.8))

    ax.axhline(tau, color='black', ls='--', lw=2, label=f'Veto threshold τ = {tau:.4f}')

    # annotate p-value – Mann-Whitney U
    from scipy.stats import mannwhitneyu
    u, p = mannwhitneyu(df_d1[col].dropna(), df_d2[col].dropna(), alternative='two-sided')
    p_txt = f'p = {p:.2e}' if p >= 1e-300 else 'p < 10⁻³⁰⁰'
    ylim = ax.get_ylim()
    ax.annotate(p_txt,
                xy=(0.5, 0.96), xycoords='axes fraction',
                ha='center', va='top', fontsize=11,
                bbox=dict(boxstyle='round,pad=0.3', fc='#FFFBE6', ec='gray', alpha=0.85))

    n1 = len(df_d1[col].dropna())
    n2 = len(df_d2[col].dropna())
    ax.set_xlabel('')
    ax.set_xticklabels([f'D1 – Improvement\n(n={n1})', f'D2 – Regression\n(n={n2})'], fontsize=11)
    ax.set_ylabel(label, fontsize=11)
    ax.set_title(f'D1 vs D2 Distribution: {label.split("–")[0].strip()}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / fname, dpi=200)
    plt.close(fig)
    print(f"  → saved {fname}")

print(f"\n[done] All plots in: {OUT_DIR}")

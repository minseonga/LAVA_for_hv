import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load Data
df_cases = pd.read_csv('/home/kms/LLaVA_calibration/experiments/pope_full_9000/vista_method_9000/taxonomy/per_case_compare.csv')
df_feat = pd.read_csv('/home/kms/LLaVA_calibration/experiments/pope_feature_screen_v1_full9000/features_unified_table.csv')

# Join
df = pd.merge(df_cases, df_feat, on='id', suffixes=('', '_feat'))

# Map case_type to simpler names
# vga_improvement = D1 (Baseline Wrong, VISTA Correct) # We want to keep these
# vga_regression = D2 (Baseline Correct, VISTA Wrong)  # We want to veto these

c_col = 'faithful_minus_global_attn'
e_col = 'guidance_mismatch_score'

df_d1 = df[df['case_type'] == 'vga_improvement']
df_d2 = df[df['case_type'] == 'vga_regression']

# Thresholds from our previous VISTA run logs
tau_c = -0.012011878027745549
tau_e = 0.04606491539150529

plt.figure(figsize=(10, 8))

# Scatter plot
plt.scatter(df_d1[c_col], df_d1[e_col], color='royalblue', alpha=0.3, s=15, label='D1 (Improvement: Keep)')
plt.scatter(df_d2[c_col], df_d2[e_col], color='crimson', alpha=0.4, s=15, marker='x', label='D2 (Regression: Veto)')

# Density (KDE) plot overlay to show clusters better
sns.kdeplot(x=df_d1[c_col], y=df_d1[e_col], cmap='Blues', alpha=0.5, levels=5, linewidths=1.5)
sns.kdeplot(x=df_d2[c_col], y=df_d2[e_col], cmap='Reds', alpha=0.5, levels=5, linewidths=1.5)

# Plot Threshold Lines
plt.axvline(x=tau_c, color='darkgreen', linestyle='--', linewidth=2, label=f'tau_c (C >= {tau_c:.4f})')
plt.axhline(y=tau_e, color='darkmagenta', linestyle='--', linewidth=2, label=f'tau_e (E >= {tau_e:.4f})')

# Shade Veto Region (C >= tau_c OR E >= tau_e)
# Set axis limits dynamically based on percentiles to ignore crazy outliers
c_p1, c_p99 = df[c_col].quantile(0.01), df[c_col].quantile(0.99)
e_p1, e_p99 = df[e_col].quantile(0.01), df[e_col].quantile(0.99)
padding_c = (c_p99 - c_p1) * 0.1
padding_e = (e_p99 - e_p1) * 0.1
xlim = (c_p1 - padding_c, c_p99 + padding_c)
ylim = (e_p1 - padding_e, e_p99 + padding_e)

# Shade Veto region manually
plt.axvspan(tau_c, xlim[1], color='gray', alpha=0.15, label='Veto Region')
plt.axhspan(tau_e, ylim[1], xmin=(tau_c - xlim[0])/(xlim[1] - xlim[0]), xmax=1.0, color='gray', alpha=0.15)
plt.axhspan(tau_e, ylim[1], xmin=0, xmax=(tau_c - xlim[0])/(xlim[1] - xlim[0]), color='gray', alpha=0.15)

plt.xlim(xlim)
plt.ylim(ylim)

plt.title('Feature Distribution: D1 (Improvement) vs D2 (Regression) [VISTA]', fontsize=14, fontweight='bold')
plt.xlabel('Faithful Minus Global Attn (C Score)', fontsize=12)
plt.ylabel('Guidance Mismatch Score (E Score)', fontsize=12)

# Fix duplicate labels
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper left')

output_path = '/home/kms/LLaVA_calibration/experiments/pope_full_9000/vista_method_9000/feature_veto_visualization.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to {output_path}")

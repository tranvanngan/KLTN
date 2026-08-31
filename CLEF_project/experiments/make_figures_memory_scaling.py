import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

# Sync font settings with the rest of the paper figures.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from plot_utils import set_time_new_roman_font

set_time_new_roman_font()

# Resolve paths relative to the project root.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TABLE_PATH = os.path.join(BASE_DIR, "results", "tables", "table_dataset_scaling_benchmark_mlp_limeca_ig.csv")
FIG_DIR = os.path.join(BASE_DIR, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv(TABLE_PATH)
lime_df = df[df['method'] == 'LIME-CA']
ig_df = df[df['method'] == 'IntegratedGradients']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# 1) Runtime scaling
ax1.plot(lime_df['target_n'], lime_df['runtime_wallclock_s'], marker='o', linewidth=2,
         label='LIME-CA (Ours)', color='#2ca02c')
ax1.plot(ig_df['target_n'], ig_df['runtime_wallclock_s'], marker='s', linewidth=2,
         label='Integrated Gradients', color='#d62728')
ax1.set_title("Execution Latency Scaling", fontsize=12, fontweight='bold')
ax1.set_xlabel("Dataset Size (Samples)", fontsize=10)
ax1.set_ylabel("Wall-clock Time (Seconds)", fontsize=10)
ax1.set_xticks(df['target_n'].unique())
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# 2) Memory scaling (log-scale)
ax2.plot(lime_df['target_n'], lime_df['memory_peak_mb'], marker='o', linewidth=2,
         label='LIME-CA (Ours)', color='#2ca02c')
ax2.plot(ig_df['target_n'], ig_df['memory_peak_mb'], marker='s', linewidth=2,
         label='Integrated Gradients', color='#d62728')
ax2.set_title("Peak Memory Consumption Profile", fontsize=12, fontweight='bold')
ax2.set_xlabel("Dataset Size (Samples)", fontsize=10)
ax2.set_ylabel("Peak RAM (MB) - Log Scale", fontsize=10)
ax2.set_yscale('log')
ax2.set_xticks(df['target_n'].unique())
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

plt.tight_layout()
out_pdf = os.path.join(FIG_DIR, "scalability_benchmark_curves.pdf")
out_png = os.path.join(FIG_DIR, "scalability_benchmark_curves.png")

plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
plt.savefig(out_png, dpi=300, bbox_inches="tight")

print("Wrote", out_pdf)
print("Wrote", out_png)
plt.close(fig)



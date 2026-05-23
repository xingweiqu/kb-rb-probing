"""
Draw fig_cot_margin.png: paired (no-CoT, CoT) margin_v on the three
wrong-distractor cells (K-D / R-D / C-D), across the four Qwen3 models.

Reads cot_diff.csv (produced by the inline extraction in PROGRESS_REPORT
or scripts/extract_cot_diff.py).
"""
from pathlib import Path
import csv
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

ROOT = Path(__file__).parent.parent
CSV = ROOT / 'data' / 'main4_metrics.csv'
OUT = ROOT / 'output'
OUT.mkdir(exist_ok=True)

MODELS = ['Qwen3-1.7B-Base', 'Qwen3-4B-Base', 'Qwen3-8B-Base', 'Qwen3-8B']
LABELS = ['1.7B-Base', '4B-Base', '8B-Base', '8B-Inst']
CELLS  = ['K-D', 'R-D', 'C-D']
TITLES = {'K-D': 'K-D (Wrong-Claim)',
          'R-D': 'R-D (Wrong-Step)',
          'C-D': 'C-D (Wrong-Bridge)'}

# Read CSV into {(model, cell): row}
data = {}
for row in csv.DictReader(open(CSV)):
    data[(row['model'], row['cell'])] = row

def m(model, cell, key):
    v = data.get((model, cell), {}).get(key)
    return float(v) if v not in (None, '', 'None') else float('nan')

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), dpi=200,
                         gridspec_kw={'wspace': 0.30})

C_NO  = '#2E75B6'   # solid blue
C_COT = '#C0392B'   # red for CoT

bar_w = 0.36
x_pos = np.arange(len(MODELS))

for ax, cell in zip(axes, CELLS):
    no_cot  = [m(mod, cell, 'margin_v_no_cot')   for mod in MODELS]
    cot     = [m(mod, cell, 'margin_v_with_cot') for mod in MODELS]
    b1 = ax.bar(x_pos - bar_w/2, no_cot, width=bar_w,
                color=C_NO, edgecolor='#222', linewidth=0.4,
                label='no-CoT', zorder=3)
    b2 = ax.bar(x_pos + bar_w/2, cot,    width=bar_w,
                color=C_COT, edgecolor='#222', linewidth=0.4,
                label='CoT', zorder=3)
    ax.axhline(0, color='#444', linewidth=0.9, zorder=2)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(LABELS, fontsize=12, fontweight='bold')
    ax.set_title(TITLES[cell], fontsize=14, fontweight='bold', pad=8)
    ax.set_ylabel(r'margin$_v$  ($\log p_v$(gold) $-$ $\log p_v$(wrong))',
                  fontsize=10.5)
    ax.grid(axis='y', alpha=0.25, zorder=0)
    ax.tick_params(axis='y', labelsize=10.5)
    # value labels
    for bars, vals in [(b1, no_cot), (b2, cot)]:
        for bar, v in zip(bars, vals):
            if np.isnan(v): continue
            offset = 0.18 if v >= 0 else -0.45
            ax.annotate(f'{v:+.1f}',
                        xy=(bar.get_x()+bar.get_width()/2, v + offset),
                        ha='center', fontsize=9, color='#222')

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=11,
           frameon=False, bbox_to_anchor=(0.5, 1.04))

fig.savefig(OUT / 'fig7_cot_margin.png',
            dpi=240, bbox_inches='tight', facecolor='white')
plt.close(fig)
print('  ↳ fig7_cot_margin.png written')

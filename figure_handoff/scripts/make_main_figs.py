"""
Render Fig 3 (family vs capacity, grouped bars on 4 canonical Qwen3 models)
and Fig 5 (wrong-distractor sensitivity vs margin, categorical x-axis on
the same 4 models), both from figures/data/main4_metrics.csv.

Body-text canonical model set:
  Qwen3-1.7B-Base, Qwen3-4B-Base, Qwen3-8B-Base, Qwen3-8B (instruct).

Note: family-score inline annotation is intentionally omitted from Fig 3
because the instruct model uses a chat template; its mean log p_orig is
not comparable to the base models'. Per-family score table goes to the
appendix instead.
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
BAR_COLOURS = ['#9DC3E6', '#5B9BD5', '#2E75B6', '#1F3864']

CAP_ORDER = {'Knowledge': ['K-P', 'K-B', 'K-D'],
             'Reasoning': ['R-P', 'R-B', 'R-D'],
             'Hybrid':    ['C-P', 'C-B', 'C-D']}

data = {}
for row in csv.DictReader(open(CSV)):
    data[(row['model'], row['cell'])] = row

def mval(model, cell, key):
    v = data.get((model, cell), {}).get(key)
    return float(v) if v not in (None, '', 'None') else float('nan')


def make_family_vs_capacity():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), dpi=200,
                             gridspec_kw={'wspace': 0.30, 'top': 0.80})
    for ax, (family, capnames) in zip(axes, CAP_ORDER.items()):
        n_caps = len(capnames); n_mods = len(MODELS)
        x_pos = np.arange(n_caps); bar_w = 0.18
        offsets = (np.arange(n_mods) - (n_mods - 1)/2) * bar_w
        for i, mod in enumerate(MODELS):
            heights = [mval(mod, c, 'delta_abs_no_cot') for c in capnames]
            ax.bar(x_pos + offsets[i], heights, width=bar_w,
                   color=BAR_COLOURS[i], edgecolor='#333',
                   linewidth=0.4, label=LABELS[i], zorder=3)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(capnames, fontsize=13, fontweight='bold')
        ax.set_title(family, fontsize=14, fontweight='bold', pad=10)
        ax.set_ylabel(r'$|\overline{\Delta\log p}|$  per token', fontsize=11.5)
        ymax = max(mval(m, c, 'delta_abs_no_cot') for m in MODELS for c in capnames)
        ax.set_ylim(0, ymax * 1.20)
        ax.grid(axis='y', alpha=0.25, zorder=0)
        ax.tick_params(axis='y', labelsize=10.5)

    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=11,
                     frameon=False, bbox_to_anchor=(0.5, 1.02),
                     title='Qwen3 main-text model set',
                     title_fontsize=11)
    fig.savefig(OUT / 'fig4_family_vs_capacity.png',
                dpi=240, bbox_inches='tight', facecolor='white',
                bbox_extra_artists=[leg])
    plt.close(fig)
    print('  ↳ fig4_family_vs_capacity.png written')


def make_wrong_distractor():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), dpi=200,
                             gridspec_kw={'wspace': 0.70, 'top': 0.80,
                                          'bottom': 0.13})
    blue = '#1F77B4'; red = '#C0392B'
    CELLS = [('K-D', 'K-D (Wrong-Claim)'),
             ('R-D', 'R-D (Wrong-Step)'),
             ('C-D', 'C-D (Wrong-Bridge)')]
    x_pos = np.arange(len(MODELS))
    for ax_left, (cell, title) in zip(axes, CELLS):
        ax_right = ax_left.twinx()
        ax_right.spines['top'].set_visible(False)
        ax_right.spines['right'].set_color(red)
        ax_left.spines['left'].set_color(blue)

        gd  = [mval(m, cell, 'gold_drop_no_cot') for m in MODELS]
        mgn = [mval(m, cell, 'margin_v_no_cot')  for m in MODELS]

        ax_left.plot(x_pos, gd, marker='o', markersize=10, linewidth=2.6,
                     color=blue, label='gold_drop')
        ax_right.plot(x_pos, mgn, marker='s', markersize=10, linewidth=2.6,
                      color=red, linestyle='--', label='margin_v')
        ax_right.axhline(0, color='#888', linewidth=0.8, linestyle=':')

        ax_left.set_xticks(x_pos)
        ax_left.set_xticklabels(LABELS, fontsize=11, fontweight='bold')
        ax_left.set_title(title, fontsize=14, fontweight='bold', pad=12)
        ax_left.set_ylabel('gold_drop  (sensitivity)', fontsize=11, color=blue)
        ax_right.set_ylabel(r'margin$_v$  (decision robustness)',
                            fontsize=11, color=red)
        ax_left.tick_params(axis='y', labelsize=10.5, colors=blue)
        ax_right.tick_params(axis='y', labelsize=10.5, colors=red)
        ax_left.grid(axis='y', alpha=0.25)

        # extra y-axis headroom so annotations don't clip against the frame
        gd_min, gd_max = min(gd), max(gd)
        gd_span = max(gd_max - gd_min, 1e-3)
        ax_left.set_ylim(gd_min - 0.28 * gd_span, gd_max + 0.28 * gd_span)
        mg_min, mg_max = min(mgn), max(mgn)
        mg_span = max(mg_max - mg_min, 1e-3)
        ax_right.set_ylim(mg_min - 0.30 * mg_span, mg_max + 0.32 * mg_span)

        # Annotate only the two endpoints (smallest and largest model).
        # On dense panels like C-D, labelling every x crowds adjacent
        # markers; the endpoints carry the visual story (base scale + the
        # instruct cliff) and the y-axis already reads the intermediate
        # values.
        endpoints = [0, len(x_pos) - 1]
        for i in endpoints:
            ax_left.annotate(f'{gd[i]:+.2f}', xy=(x_pos[i], gd[i]),
                             xytext=(-8, -16), textcoords='offset points',
                             color=blue, fontsize=10,
                             ha='right', va='top', fontweight='bold')
            ax_right.annotate(f'{mgn[i]:+.2f}', xy=(x_pos[i], mgn[i]),
                              xytext=(8, 12), textcoords='offset points',
                              color=red, fontsize=10,
                              ha='left', va='bottom', fontweight='bold')

    leg = fig.legend(handles=[
        plt.Line2D([], [], color=blue, marker='o', linewidth=2.5,
                   label='gold_drop = log p_v(gold) - log p_o(gold)  (sensitivity)'),
        plt.Line2D([], [], color=red, marker='s', linewidth=2.5, linestyle='--',
                   label='margin_v = log p_v(gold) - log p_v(wrong)  (robustness)'),
    ], loc='lower center', ncol=2, fontsize=11, frameon=False,
       bbox_to_anchor=(0.5, 1.02))

    fig.savefig(OUT / 'fig6_wrong_claim_margin.png',
                dpi=240, bbox_inches='tight', facecolor='white',
                bbox_extra_artists=[leg])
    plt.close(fig)
    print('  ↳ fig6_wrong_claim_margin.png written')


if __name__ == '__main__':
    print('Rendering main-text figures with canonical 4-Qwen3 set ...')
    make_family_vs_capacity()
    make_wrong_distractor()
    print('Done.')

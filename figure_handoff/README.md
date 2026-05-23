# Figure handoff: Fig 4, Fig 6, Fig 7

Self-contained package to regenerate three figures from the paper:

| Paper figure | Output PNG | What it shows |
|---|---|---|
| **Fig 4** | `output/fig4_family_vs_capacity.png` | Per-family grouped bars of $\lvert\overline{\Delta\!\log p}\rvert$ per token across the canonical four Qwen3 models, three panels (Knowledge / Reasoning / Hybrid), three capacity bars per panel. |
| **Fig 6** | `output/fig6_wrong_claim_margin.png` | Wrong-distractor cells (K-D / R-D / C-D), dual-axis line plot per panel: `gold_drop` (sensitivity, blue) and `margin_v` (decision robustness, red) across the same four models. |
| **Fig 7** | `output/fig7_cot_margin.png` | Paired `(no-CoT, CoT)` `margin_v` bars on the same three wrong-distractor cells across the same four models. |

Reference outputs that match the current PDF are checked in under `current_output/`.

## Layout

```
figure_handoff/
├── README.md              ← this file
├── requirements.txt       ← Python deps
├── scripts/
│   ├── make_main_figs.py  ← writes Fig 4 + Fig 6
│   └── make_cot_fig.py    ← writes Fig 7
├── data/
│   └── main4_metrics.csv  ← single input file, see schema below
├── current_output/        ← the three PNGs that are in the paper right now
└── output/                ← regenerated PNGs land here (created on first run)
```

## How to run

```bash
cd figure_handoff
pip install -r requirements.txt   # matplotlib + numpy
python3 scripts/make_main_figs.py  # writes output/fig4_*.png and output/fig6_*.png
python3 scripts/make_cot_fig.py    # writes output/fig7_*.png
```

No GPU, no model downloads. Pure plotting from one 16-row CSV.

## Data: `data/main4_metrics.csv`

One row per (model, cell), 9 cells × 4 models = 36 rows. Fields used by the scripts:

| Field | Meaning |
|---|---|
| `model` | one of `Qwen3-1.7B-Base`, `Qwen3-4B-Base`, `Qwen3-8B-Base`, `Qwen3-8B` |
| `cell` | one of `K-P`, `K-B`, `K-D`, `R-P`, `R-B`, `R-D`, `C-P`, `C-B`, `C-D` |
| `delta_abs_no_cot` | $\lvert\overline{\Delta\!\log p}\rvert$ per token, no-CoT. **Fig 4 uses this.** |
| `gold_drop_no_cot` | $\log p_v(\text{gold}) - \log p_o(\text{gold})$, no-CoT. Fig 6 left axis. K-D / R-D / C-D only. |
| `margin_v_no_cot` | $\log p_v(\text{gold}) - \log p_v(\text{wrong})$, no-CoT. Fig 6 right axis, Fig 7 left bars. K-D / R-D / C-D only. |
| `margin_v_with_cot` | Same, with the zero-shot CoT prefix `"Let's think step by step.\n"` prepended to both original and variant. Fig 7 right bars. K-D / R-D / C-D only. |

Cells outside the wrong-distractor trio (K-P, K-B, R-P, R-B, C-P, C-B) leave the margin/gold-drop columns empty — Fig 6 and Fig 7 ignore those rows.

## What's already worth fixing (suggestions to the friend)

These are the rough edges I noticed but didn't have time to fix; treat as hints, not requirements.

- **Fig 4** drops the inline "family-score" annotation that earlier versions had. The legend says "Qwen3 main-text model set" — could be sharper ("Qwen3 canonical four"). Bars are blue-shaded by scale; the 8B-Inst (darkest) is sometimes hard to tell apart from 8B-Base. Consider a hue shift for the instruct model.
- **Fig 6** uses categorical x-axis (four model labels) with two y-axes per panel. Endpoint-only value labels — middle two points read off the axis. Margin\_v on C-D is much more negative than on K-D/R-D, so each panel has its own y-range; the visual scale doesn't compare across panels. If the friend wants a shared y-range across the three panels, the C-D drop dwarfs the others.
- **Fig 7** value labels (`±0.5` etc.) hover above/below bars; on the 8B-Inst panel for C-D, the bar is so tall (-12-ish) the label might collide with the panel title. Worth giving each axis a per-panel ymin/ymax.
- All three figures use Helvetica; falls back to DejaVu Sans if Helvetica isn't installed (no error, just different glyphs).
- DPI is 240 — fine for PDF, but if exported to PDF directly (`.pdf` instead of `.png`) the axes would stay vector. Could replace `savefig('*.png')` with `savefig('*.pdf')` if needed.

## Branch

`figure-handoff` off `main` at the commit that produced this folder.

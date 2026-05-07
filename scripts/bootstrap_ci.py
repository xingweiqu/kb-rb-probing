"""Bootstrap confidence intervals and paired-bootstrap tests for the
intervention signature matrix.

Reads `runs/<model>/model_outputs.jsonl` (must be present locally; pull from
server if needed). Resamples by `family_id` to preserve original-variant
pairing. Emits two CSVs:

    behavior_matrix_with_ci.csv
        one row per (model, family, capability, cot_state); columns include
        n, mean Δlogp/token, |Δ| mean, lower 2.5%, upper 97.5%, signed
        z-score within (capability, cot).

    headline_trend_tests.csv
        one row per pre-registered headline trend; reports the bootstrap
        difference of means and a one-sided p-value.

Usage:
    python -m scripts.bootstrap_ci \
        --runs runs/Qwen3-8B runs/Qwen3-8B-Base runs/Qwen3.5-9B runs/Qwen3.5-9B-Base \
        --output_matrix reports/arr_revision/behavior_matrix_with_ci.csv \
        --output_tests  reports/arr_revision/headline_trend_tests.csv \
        --bootstrap 10000

If `model_outputs.jsonl` is missing for a run dir, that model is skipped
with a warning.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


VARIANT_TO_CAPABILITY = {
    "hint": "KP", "wrongclaim": "KD", "paraphrase": "KB",
    "scaffold": "RP", "rule_removal": "RB", "wrong_intermediate": "RD",
    "explicit_fact": "CP", "retrieval_blocked": "CB",
    "wrong_bridge": "CD",
    "both_blocked": "CB-control",
}
TASK_FAMILY_OF_VARIANT = {
    "hint": "KB", "wrongclaim": "KB", "paraphrase": "KB",
    "scaffold": "RB", "rule_removal": "RB", "wrong_intermediate": "RB",
    "explicit_fact": "Hybrid", "retrieval_blocked": "Hybrid",
    "wrong_bridge": "Hybrid", "both_blocked": "Hybrid",
}


def _load_outputs(model_outputs_path: Path) -> dict:
    """Return {(family_id, mode, cot_state): {variant: gold_logprob_mean, ...}}."""
    out = defaultdict(dict)
    for line in open(model_outputs_path, encoding="utf-8"):
        rec = json.loads(line)
        fid = rec.get("family_id")
        mode = rec.get("mode", "natural")
        cot = rec.get("cot_state", "no_cot")
        variant = rec.get("variant")
        lp = rec.get("gold_logprob_mean")
        if fid is None or variant is None or lp is None:
            continue
        out[(fid, mode, cot)][variant] = float(lp)
    return out


def _per_family_deltas(outputs: dict, mode: str = "natural"
                       ) -> dict[tuple[str, str], dict[str, float]]:
    """Pair each variant against its original. Returns
    {(capability, cot_state): {family_id: delta}}."""
    out: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for (fid, m, cot), variants in outputs.items():
        if m != mode:
            continue
        if "original" not in variants:
            continue
        orig = variants["original"]
        for variant, lp in variants.items():
            if variant == "original":
                continue
            cap = VARIANT_TO_CAPABILITY.get(variant)
            if cap is None:
                continue
            out[(cap, cot)][fid] = lp - orig
    return out


def _bootstrap(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Resample with replacement; return n_boot bootstrap means."""
    n = len(values)
    if n == 0:
        return np.array([])
    idx = rng.integers(0, n, size=(n_boot, n))
    return values[idx].mean(axis=1)


def _ci(boot_means: np.ndarray) -> tuple[float, float]:
    if len(boot_means) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def matrix_with_ci(model_dirs: list[Path], output_csv: Path,
                   n_boot: int, rng_seed: int) -> dict:
    """Per-cell mean ± 95% CI from bootstrap resampling by family_id."""
    rng = np.random.default_rng(rng_seed)
    rows: list[dict] = []
    by_model_cap_cot: dict[tuple[str, str, str], np.ndarray] = {}

    for run_dir in model_dirs:
        mo = run_dir / "model_outputs.jsonl"
        if not mo.exists():
            logger.warning("missing %s — skipping", mo)
            continue
        model = run_dir.name
        outputs = _load_outputs(mo)
        deltas = _per_family_deltas(outputs)

        for (cap, cot), fid_to_delta in deltas.items():
            values = np.array(list(fid_to_delta.values()), dtype=np.float64)
            family = next((fam for v, c in VARIANT_TO_CAPABILITY.items()
                           if c == cap for fam, vv in TASK_FAMILY_OF_VARIANT.items()
                           if vv == fam and v == fam), None)
            # Resolve task family by reverse-lookup
            family = next((TASK_FAMILY_OF_VARIANT[v]
                           for v, c in VARIANT_TO_CAPABILITY.items() if c == cap), "?")
            boot = _bootstrap(values, n_boot, rng)
            lo, hi = _ci(boot)
            by_model_cap_cot[(model, cap, cot)] = values
            rows.append({
                "model": model,
                "task_family": family,
                "capability": cap,
                "cot_state": cot,
                "n": int(len(values)),
                "mean_delta": float(values.mean()) if len(values) else float("nan"),
                "abs_mean_delta": float(np.abs(values).mean()) if len(values) else float("nan"),
                "ci_low_2.5": lo,
                "ci_high_97.5": hi,
            })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                                ["model", "task_family", "capability", "cot_state", "n",
                                 "mean_delta", "abs_mean_delta", "ci_low_2.5", "ci_high_97.5"])
        writer.writeheader(); writer.writerows(rows)
    logger.info("wrote %d rows -> %s", len(rows), output_csv)
    return by_model_cap_cot


def paired_bootstrap_test(values_a: np.ndarray, values_b: np.ndarray,
                          n_boot: int, rng: np.random.Generator
                          ) -> tuple[float, float, float]:
    """Two-sample bootstrap test on means. Returns (delta, p, ci_diff)."""
    if len(values_a) == 0 or len(values_b) == 0:
        return float("nan"), float("nan"), float("nan")
    a_boot = _bootstrap(values_a, n_boot, rng)
    b_boot = _bootstrap(values_b, n_boot, rng)
    diff = b_boot - a_boot
    # two-sided p
    obs = float(values_b.mean() - values_a.mean())
    if obs >= 0:
        p_one = float((diff <= 0).mean())
    else:
        p_one = float((diff >= 0).mean())
    p_two = 2 * min(p_one, 1 - p_one)
    ci_low = float(np.percentile(diff, 2.5))
    ci_high = float(np.percentile(diff, 97.5))
    return obs, p_two, (ci_low, ci_high)


def headline_tests(by_model_cap_cot: dict, output_csv: Path,
                   n_boot: int, rng_seed: int) -> None:
    rng = np.random.default_rng(rng_seed + 1)
    pre_registered = [
        # name, capability, cot, model_a, model_b, hypothesis
        ("KD growth: 8B-Base -> 8B-instruct", "KD", "no_cot", "Qwen3-8B-Base", "Qwen3-8B",
         "Δlogprob magnitude on Wrong-Claim variant grows from base to instruct at 8B"),
        ("CP decay: 0.6B-Base -> 8B-Base", "CP", "no_cot", "Qwen3-0.6B-Base", "Qwen3-8B-Base",
         "Bridge-Fact Gain magnitude shrinks with scale on base models"),
        ("RP instruct jump: 8B-Base -> 8B-instruct", "RP", "no_cot", "Qwen3-8B-Base", "Qwen3-8B",
         "Scaffold Gain magnitude grows from base to instruct at 8B"),
        ("CB recovery: 1.7B-Base -> 8B-instruct", "CB", "no_cot", "Qwen3-1.7B-Base", "Qwen3-8B",
         "Retrieval-Block Drop weakens with scale and instruct"),
    ]
    rows = []
    for name, cap, cot, a, b, hyp in pre_registered:
        va = by_model_cap_cot.get((a, cap, cot))
        vb = by_model_cap_cot.get((b, cap, cot))
        if va is None or vb is None:
            rows.append({"name": name, "capability": cap, "cot": cot,
                         "a": a, "b": b, "delta": "MISSING", "p_two": "MISSING",
                         "ci_low": "", "ci_high": "", "hypothesis": hyp})
            continue
        # use absolute values for magnitude comparisons
        va_abs = np.abs(va); vb_abs = np.abs(vb)
        delta, p, ci = paired_bootstrap_test(va_abs, vb_abs, n_boot, rng)
        rows.append({"name": name, "capability": cap, "cot": cot,
                     "a": a, "b": b,
                     "delta": f"{delta:+.4f}", "p_two": f"{p:.4f}",
                     "ci_low": f"{ci[0]:+.4f}", "ci_high": f"{ci[1]:+.4f}",
                     "hypothesis": hyp})

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader(); writer.writerows(rows)
    logger.info("wrote %d rows -> %s", len(rows), output_csv)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True,
                   help="run directories containing model_outputs.jsonl each")
    p.add_argument("--output_matrix",
                   default="reports/arr_revision/behavior_matrix_with_ci.csv")
    p.add_argument("--output_tests",
                   default="reports/arr_revision/headline_trend_tests.csv")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    by_model = matrix_with_ci([Path(r) for r in args.runs],
                              Path(args.output_matrix),
                              args.bootstrap, args.seed)
    headline_tests(by_model, Path(args.output_tests),
                   args.bootstrap, args.seed)


if __name__ == "__main__":
    main()

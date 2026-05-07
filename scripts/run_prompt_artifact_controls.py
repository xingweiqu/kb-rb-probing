"""Run prompt-artefact and split controls for the task-family three-way label.

For each control, train a logistic-regression classifier with `GroupKFold`
on `family_id` to prevent backbone leakage, then report balanced accuracy
mean ± std over folds.

Controls implemented:
  length      : 5 numeric features (prompt-token len, gold-token len, digit
                count, capitalised-word count, punctuation count)
  tfidf       : TF-IDF (1, 2)-grams of the prompt only (no metadata)
  random_label: shuffle labels within each fold's train set; no other
                feature change
  random_feat : shuffle nothing but replace the input with N(0, I) of
                matching width; included as a sanity baseline that the
                pipeline learns nothing without signal

Hidden-state probe accuracies are NOT recomputed here; the script reads them
from each `runs/<model>/summary.json` so the comparison row in the output
table reflects whatever probe was last run server-side.

Usage:
    python -m scripts.run_prompt_artifact_controls \
        --dataset runs/full_25/output/dataset.jsonl \
        --summaries runs/Qwen3-8B/summary.json runs/Qwen3-8B-Base/summary.json \
        --output_csv reports/arr_revision/prompt_artifact_controls.csv \
        --output_md  reports/arr_revision/prompt_artifact_controls.md
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


FAMILY_TO_LABEL = {"KB": 0, "RB": 1, "Hybrid": 2}
LABEL_NAMES = ["KB", "RB", "Hybrid"]


def _load_originals(path: Path) -> list[dict]:
    """Return one (family_id, family, prompt, gold) row per natural original."""
    rows = []
    for line in open(path, encoding="utf-8"):
        it = json.loads(line)
        if it.get("variant") != "original" or it.get("mode", "natural") != "natural":
            continue
        rows.append({
            "family_id": it["family_id"],
            "family": it["task_family"],
            "prompt": it.get("question", ""),
            "gold": it.get("gold_answer", ""),
        })
    return rows


def _length_features(rows: list[dict]) -> np.ndarray:
    feats = []
    for r in rows:
        prompt = r["prompt"] or ""
        gold = r["gold"] or ""
        feats.append([
            len(prompt.split()),
            len(gold.split()),
            sum(1 for c in prompt if c.isdigit()),
            len(re.findall(r"\b[A-Z][a-zA-Z]+", prompt)),
            sum(1 for c in prompt if c in ".,;:!?\"'()-"),
        ])
    return np.array(feats, dtype=np.float32)


def _kfold_run(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
               n_splits: int, rng: int) -> tuple[float, float]:
    n_splits = min(n_splits, len(np.unique(groups)))
    if n_splits < 2 or len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    gkf = GroupKFold(n_splits=n_splits)
    baccs = []
    for tr, te in gkf.split(X, y, groups=groups):
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=rng)
        clf.fit(X[tr], y[tr])
        baccs.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
    return float(np.mean(baccs)), float(np.std(baccs))


def _kfold_run_tfidf(prompts: list[str], y: np.ndarray, groups: np.ndarray,
                     n_splits: int, rng: int) -> tuple[float, float]:
    n_splits = min(n_splits, len(np.unique(groups)))
    if n_splits < 2 or len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    gkf = GroupKFold(n_splits=n_splits)
    baccs = []
    prompts_arr = np.asarray(prompts)
    for tr, te in gkf.split(prompts_arr, y, groups=groups):
        # fit vectorizer ONLY on train fold to avoid test leakage
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=5000)
        Xtr = vec.fit_transform(prompts_arr[tr])
        Xte = vec.transform(prompts_arr[te])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=rng)
        clf.fit(Xtr, y[tr])
        baccs.append(balanced_accuracy_score(y[te], clf.predict(Xte)))
    return float(np.mean(baccs)), float(np.std(baccs))


def _hidden_state_probe_baccs(summary_paths: list[Path]) -> list[dict]:
    """Pull the no_cot/last task_family bacc from each summary.json."""
    out = []
    for p in summary_paths:
        s = json.load(open(p, encoding="utf-8"))
        rec = next((r for r in s.get("task_family", {}).get("linear", [])
                    if r.get("cot_state") == "no_cot" and r.get("pool") == "last"), None)
        if rec is None:
            continue
        out.append({
            "model": Path(s.get("run_dir", p.parent.name)).name or p.parent.name,
            "bacc": float(rec.get("best_balanced_accuracy") or float("nan")),
            "best_layer": rec.get("best_layer"),
        })
    return out


def run(dataset_path: Path, summary_paths: list[Path],
        output_csv: Path, output_md: Path,
        n_splits: int = 5, rng: int = 42) -> None:
    rows = _load_originals(dataset_path)
    if not rows:
        logger.error("no natural-original rows found in %s", dataset_path); return
    logger.info("%d natural-original rows", len(rows))

    y = np.array([FAMILY_TO_LABEL[r["family"]] for r in rows])
    groups = np.array([r["family_id"] for r in rows])

    length_X = _length_features(rows)
    length_mean, length_std = _kfold_run(length_X, y, groups, n_splits, rng)

    tfidf_mean, tfidf_std = _kfold_run_tfidf([r["prompt"] for r in rows],
                                              y, groups, n_splits, rng)

    rng_np = np.random.default_rng(rng)
    y_shuffled = y.copy(); rng_np.shuffle(y_shuffled)
    rl_mean, rl_std = _kfold_run(length_X, y_shuffled, groups, n_splits, rng)

    rand_X = rng_np.standard_normal((len(rows), 64)).astype(np.float32)
    rf_mean, rf_std = _kfold_run(rand_X, y, groups, n_splits, rng)

    summary_baccs = _hidden_state_probe_baccs(summary_paths)

    out_rows = [
        {"control": "length_features", "mean_bacc": f"{length_mean:.3f}", "std_bacc": f"{length_std:.3f}", "model_or_note": "5 numeric features, GroupKFold by family_id"},
        {"control": "tfidf_prompt", "mean_bacc": f"{tfidf_mean:.3f}", "std_bacc": f"{tfidf_std:.3f}", "model_or_note": "(1,2)-gram, max_features=5000, fit per-fold"},
        {"control": "random_label", "mean_bacc": f"{rl_mean:.3f}", "std_bacc": f"{rl_std:.3f}", "model_or_note": "labels shuffled, length features, expect ~0.33"},
        {"control": "random_feature", "mean_bacc": f"{rf_mean:.3f}", "std_bacc": f"{rf_std:.3f}", "model_or_note": "N(0,I) features dim=64, expect ~0.33"},
    ]
    for h in summary_baccs:
        out_rows.append({
            "control": "hidden_state_probe",
            "mean_bacc": f"{h['bacc']:.3f}",
            "std_bacc": "",
            "model_or_note": f"{h['model']}, layer {h['best_layer']}",
        })

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["control", "mean_bacc", "std_bacc", "model_or_note"])
        writer.writeheader(); writer.writerows(out_rows)

    md = ["# Prompt-artefact and split controls", "",
          "All baselines use `GroupKFold` over `family_id` so backbones never",
          "appear across train/test splits. Chance level on the three-way",
          "task-family label is 1/3 ≈ 0.333.", "",
          "| Control | Mean balanced acc | Std | Note |",
          "|---|---|---|---|"]
    for r in out_rows:
        md.append(f"| {r['control']} | {r['mean_bacc']} | {r['std_bacc']} | {r['model_or_note']} |")
    md.append("")
    md.append(f"Dataset: `{dataset_path}` ({len(rows)} natural originals).")
    md.append(f"Folds: {n_splits} GroupKFold by `family_id`.")
    md.append("")
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    logger.info("wrote %s and %s", output_csv, output_md)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="runs/full_25/output/dataset.jsonl")
    p.add_argument("--summaries", nargs="*",
                   default=[
                       "runs/Qwen3-8B/summary.json",
                       "runs/Qwen3-8B-Base/summary.json",
                       "runs/Qwen3-4B-Base/summary.json",
                       "runs/Qwen3-1.7B-Base/summary.json",
                   ])
    p.add_argument("--output_csv", default="reports/arr_revision/prompt_artifact_controls.csv")
    p.add_argument("--output_md", default="reports/arr_revision/prompt_artifact_controls.md")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--rng", type=int, default=42)
    args = p.parse_args()
    run(Path(args.dataset),
        [Path(s) for s in args.summaries if Path(s).exists()],
        Path(args.output_csv), Path(args.output_md),
        args.n_splits, args.rng)


if __name__ == "__main__":
    main()

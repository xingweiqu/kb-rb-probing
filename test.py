from __future__ import annotations
import argparse
import ast
import asyncio
from copy import deepcopy
import hashlib
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import json
import os
import random
import re
import shutil
import sys
import time
import urllib.request
from collections import Counter, defaultdict, deque
from contextlib import redirect_stderr, redirect_stdout

try:
    from tqdm import tqdm as _tqdm
    def _make_tqdm(iterable, **kw):
        return _tqdm(iterable, **kw)
except ImportError:
    def _make_tqdm(iterable, **kw):  # type: ignore[misc]
        return iterable
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


REPO_BASE = "/mnt/hdfs/yuhao/repo_urls_jsonl_tmp/repo_result"
OUTPUT_JSONL = "/tmp/python_samples.jsonl"
OUTPUT_PARQUET = "/tmp/python_samples.parquet"
OUTPUT_QA_JSON = "/tmp/python_samples.qa.json"
OUTPUT_CONTENT_SPLIT_JSONL = ""
OUTPUT_RAW_SAMPLES_JSONL = ""
OUTPUT_JUDGE_JSON = "/tmp/python_samples.judge.json"
OUTPUT_AUDIT_JSONL = "/tmp/python_samples.audit.jsonl"
DEFAULT_TOKENIZER_DIR = "/opt/tiger/coding-agent-synth-data/bbpe155k-v6.4.3-ml.pret_v5.7_20251015"
TARGET = 30
VIRTUAL_ROOT = "/tmp/virtual_project_pipeline"
MASK_BEGIN = "<<MASK_B>>"
MASK_END = "<<MASK_E>>"
DEFAULT_COMMIT_JUDGE_API_BASE = "http://127.0.0.1:9003/v1"
DEFAULT_COMMIT_JUDGE_MODEL = "Qwen3-8B"
DEFAULT_HARD_NEGATIVE_API_BASE = DEFAULT_COMMIT_JUDGE_API_BASE
DEFAULT_HARD_NEGATIVE_MODEL = DEFAULT_COMMIT_JUDGE_MODEL
DEFAULT_ROOT_GROUNDING_JUDGE_API_BASE = DEFAULT_COMMIT_JUDGE_API_BASE
DEFAULT_ROOT_GROUNDING_JUDGE_MODEL = DEFAULT_COMMIT_JUDGE_MODEL
DEFAULT_SAMPLE_QUALITY_JUDGE_API_BASE = DEFAULT_ROOT_GROUNDING_JUDGE_API_BASE
DEFAULT_SAMPLE_QUALITY_JUDGE_MODEL = DEFAULT_ROOT_GROUNDING_JUDGE_MODEL
ROOT_GROUNDING_MIN_CONFIDENCE = 0.68
NEXT_ACTION_MIN_CONFIDENCE = 0.62
MAX_CONTEXT_CHARS = 3000
MAX_CHANGED_PY_FILES = 10
MIN_COMMIT_MESSAGE_LEN = 18
MAX_FILE_CANDIDATES = 7
MAX_SYMBOL_CANDIDATES = 7
AUDIT_RECORD_LIMIT = 200
BASELINE_ACTIONS = ["global_search", "read_full_file", "read_full_repo"]
TRUNCATION_MARKERS = ("[...", "truncated", "omitted", "ELLIPSIZATION")
DEFAULT_DATASET_MODE = "grounded_3tasks"
PATCH_TYPE_TAXONOMY = [
    "add_argument_propagation",
    "change_condition_logic",
    "signature_change_propagation",
    "state_update_fix",
    "config_or_schema_propagation",
    "test_only_adjustment",
    "rename_only",
    "cleanup_only",
]
RISK_SURFACE_TAXONOMY = [
    "service_to_repo_propagation",
    "caller_callee_signature_consistency",
    "state_transition_consistency",
    "config_consumer_consistency",
    "regression_test_updates",
    "low_risk_local_refactor",
]
NEGATIVE_TYPE_TAXONOMY = [
    "nearby_but_not_best",
    "wrong_layer",
    "premature_broad_action",
    "downstream_too_early",
    "upstream_low_information",
    "plausible_but_irrelevant",
    "false_change_propagation",
    "overread_action",
]
MEANINGLESS_REASON_TAXONOMY = [
    "merge_commit",
    "revert_only",
    "formatting_only",
    "comment_only",
    "lockfile_only",
    "generated_artifact_only",
    "docs_only",
    "ambiguous_low_signal",
    "bulk_mechanical_noise",
]
INTENT_LABEL_TAXONOMY = [
    "api_signature_propagation",
    "root_fix_plus_test_followup",
    "config_schema_propagation",
    "rename_propagation",
    "local_bugfix_followup",
    "same_feature_increment",
    "unrelated_or_distinct",
]

RNG = random.Random(42)

PIPELINE_VERSION = "action_schema_v2_fix2_strict_quality"

ROOT_FILE_MARGIN = 8.0
ROOT_FILE_AUX_MARGIN = 12.0
MIN_READABLE_SPAN_LINES = 6
PREFERRED_SPAN_MIN = 8
PREFERRED_SPAN_MAX = 40
MAX_ALLOWED_SPAN_LINES = 120
NON_NOISY_RELATED_MIN = 2
STRONG_HEURISTIC_FALLBACK_MARGIN = 10.0


# ── data classes ──────────────────────────────────────────────────────────────


@dataclass
class FileChangeEvidence:
    file_path: str
    before_source: str
    after_source: str
    patch_text: str
    patch_present: bool
    snapshot_present: bool
    before_after_present: bool
    changed_symbols: list[dict[str, Any]]
    symbol_pool: list[dict[str, Any]]
    patch_changed_lines: int
    patch_added_lines: int
    patch_removed_lines: int
    changed_line_spans: list[list[int]]
    primary_changed_span: list[int] | None
    score: int

    @property
    def is_consistent(self) -> bool:
        return self.patch_present and self.snapshot_present and self.before_after_present


@dataclass
class CommitJudgeProfile:
    commit_id: str
    repo: str
    commit_message: str
    changed_files: list[str]
    changed_py_files: list[str]
    patch_text: str
    patch_changed_lines: int
    evidence_list: list[FileChangeEvidence]
    changed_symbols: list[str]
    top_directories: list[str]
    dep_map: dict[str, list[str]]
    reverse_dep_map: dict[str, list[str]]
    relevant_files: list[dict[str, Any]]


@dataclass
class CommitJudgeDecision:
    commit_id: str
    intent_label: str
    should_merge: bool
    merge_group_id: str
    merge_confidence: float
    is_meaningless: bool
    meaningless_reason: str
    short_judge_rationale: str


@dataclass
class QATracker:
    total_commits_seen: int = 0
    eligible_commits: int = 0
    emitted_samples: int = 0
    skipped: Counter = field(default_factory=Counter)
    task_counts: Counter = field(default_factory=Counter)
    candidate_counts: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    changed_files_per_commit: list[int] = field(default_factory=list)
    changed_symbols_per_gold_file: list[int] = field(default_factory=list)
    audit_rows: list[dict[str, Any]] = field(default_factory=list)
    llm_root_grounding_keep_count: int = 0
    llm_root_grounding_drop_count: int = 0
    llm_root_grounding_fallback_count: int = 0
    llm_root_grounding_confidence_sum: float = 0.0
    llm_root_grounding_confidence_n: int = 0
    per_task_root_grounding_drop_reasons: Counter = field(default_factory=Counter)
    prompt_length_chars_sum: int = 0
    target_length_chars_sum: int = 0
    serialized_length_n: int = 0
    per_task_snippet_counts: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    per_task_read_history_steps: dict[str, list[int]] = field(default_factory=lambda: defaultdict(list))
    task_instruction_fallback_count: int = 0
    next_action_selection_fallback_count: int = 0
    # Per-task accepted-fallback counters
    accepted_patch_grounding_fallback_count: int = 0
    accepted_ast_dependency_trace_fallback_count: int = 0
    accepted_reading_summary_fallback_count: int = 0
    # Rejection detail counters
    rejected_generic_next_question_count: int = 0
    rejected_aux_root_file_count: int = 0
    rejected_symbol_span_mismatch_count: int = 0
    # New strict-quality rejection counters
    rejected_reading_summary_generic_nq: int = 0
    rejected_reading_summary_placeholder_region: int = 0
    rejected_reading_summary_ungrounded_supporting: int = 0
    rejected_generic_action_reason: int = 0
    rejected_fallback_margin_too_small: int = 0
    rejected_fallback_broad_action: int = 0
    rejected_root_grounding_noisy_related: int = 0
    rejected_root_grounding_placeholder_span: int = 0

    def skip(self, reason: str) -> None:
        self.skipped[reason] += 1

    def observe_commit(self, changed_file_count: int, changed_symbol_count: int) -> None:
        self.eligible_commits += 1
        self.changed_files_per_commit.append(changed_file_count)
        self.changed_symbols_per_gold_file.append(changed_symbol_count)

    def observe_sample(self, sample: dict[str, Any]) -> None:
        task_type = sample.get("task_type", "unknown")
        self.task_counts[task_type] += 1
        self.emitted_samples += 1
        # average_candidates_per_task is a task-specific "decision context size" metric:
        #   patch_grounding / ast_dependency_trace → number of available_actions offered
        #   reading_summary → number of read_history steps (no action pool for this task)
        if task_type in {"patch_grounding", "ast_dependency_trace"}:
            n = len(sample.get("input", {}).get("available_actions", []))
        elif task_type == "reading_summary":
            n = len(sample.get("input", {}).get("read_history", []))
        else:
            n = len(sample.get("candidates", []))
        self.candidate_counts[task_type].append(n)

    def observe_root_grounding_decision(
        self,
        decision: str,
        confidence: float | None,
        used_fallback: bool,
        drop_reason: str,
        task_type: str,
    ) -> None:
        if decision == "keep":
            self.llm_root_grounding_keep_count += 1
            if confidence is not None:
                self.llm_root_grounding_confidence_sum += float(confidence)
                self.llm_root_grounding_confidence_n += 1
        else:
            self.llm_root_grounding_drop_count += 1
            if drop_reason:
                self.per_task_root_grounding_drop_reasons[f"{task_type}:{drop_reason}"] += 1
        if used_fallback:
            self.llm_root_grounding_fallback_count += 1

    def observe_serialized_record(
        self,
        task_type: str,
        prompt_len: int,
        target_len: int,
        snippet_count: int,
        read_history_steps: int,
        used_instruction_fallback: bool,
        used_action_fallback: bool,
    ) -> None:
        self.prompt_length_chars_sum += int(prompt_len)
        self.target_length_chars_sum += int(target_len)
        self.serialized_length_n += 1
        self.per_task_snippet_counts[task_type].append(int(snippet_count))
        self.per_task_read_history_steps[task_type].append(int(read_history_steps))
        if used_instruction_fallback:
            self.task_instruction_fallback_count += 1
        if used_action_fallback:
            self.next_action_selection_fallback_count += 1

    def record_audit(
        self,
        sample: dict[str, Any] | None,
        validation_status: str,
        skip_reason: str = "",
        prompt_text: str = "",
        target_text: str = "",
        leakage_flags: list[str] | None = None,
        task_purity_status: str = "unknown",
        rejection_reason: str = "",
        bucket: str = "",
    ) -> None:
        if len(self.audit_rows) >= AUDIT_RECORD_LIMIT:
            return
        sample = sample or {}
        metadata = sample.get("metadata", {}) if isinstance(sample, dict) else {}
        candidates = sample.get("candidates", []) if isinstance(sample, dict) else []
        candidate_meta = []
        for candidate in candidates[:12]:
            candidate_meta.append({
                "action_type": candidate.get("action_type", ""),
                "file_path": candidate.get("file_path", ""),
                "symbol": candidate.get("symbol", ""),
                "candidate_source": candidate.get("candidate_source", ""),
                "negative_type": candidate.get("negative_type", ""),
            })
        self.audit_rows.append({
            "repo": str(metadata.get("repo", "")),
            "commit_id": str(metadata.get("commit_id", "")),
            "task_type": sample.get("task_type", "") if isinstance(sample, dict) else "",
            "gold_file": str(metadata.get("gold_file", "")),
            "gold_symbol": str(metadata.get("gold_symbol", "")),
            "gold_line_span": metadata.get("gold_line_span", []),
            "related_files": metadata.get("related_files", []),
            "prompt_text": prompt_text,
            "target_text": target_text,
            "raw_sample_json": json.dumps(sample, ensure_ascii=False) if sample else "",
            "validation_status": validation_status,
            "skip_reason": skip_reason,
            "leakage_flags": leakage_flags or [],
            "task_purity_status": task_purity_status,
            "rejection_reason": rejection_reason,
            "candidate_metadata": candidate_meta,
            "bucket": bucket or "",
        })

    def to_report(self) -> dict[str, Any]:
        average_candidates = {
            task_type: round(sum(counts) / len(counts), 2) if counts else 0.0
            for task_type, counts in sorted(self.candidate_counts.items())
        }
        # Ensure key audit skip reasons are always present and expose
        # aggregate counts for message-overlap related rejections.
        skipped = dict(sorted(self.skipped.items()))
        overstrict_overlap = sum(
            count for reason, count in self.skipped.items()
            if reason.startswith("weak_message_support_")
        )
        if overstrict_overlap and "rejected_due_to_overstrict_message_overlap" not in skipped:
            skipped["rejected_due_to_overstrict_message_overlap"] = overstrict_overlap
        for key in (
            "rejected_due_to_patch_leakage",
            "rejected_due_to_added_symbol_visibility_mismatch",
            "rejected_due_to_overstrict_message_overlap",
            "rejected_due_to_unwired_merge_logic",
        ):
            skipped.setdefault(key, 0)
        return {
            "total_commits_seen": self.total_commits_seen,
            "eligible_commits": self.eligible_commits,
            "total_samples": self.emitted_samples,
            "per_task_counts": dict(sorted(self.task_counts.items())),
            "skipped_counts_by_reason": skipped,
            "average_candidates_per_task": average_candidates,
            "average_changed_files_per_commit": round(
                sum(self.changed_files_per_commit) / len(self.changed_files_per_commit), 2
            ) if self.changed_files_per_commit else 0.0,
            "average_changed_symbols_per_selected_gold_file": round(
                sum(self.changed_symbols_per_gold_file) / len(self.changed_symbols_per_gold_file), 2
            ) if self.changed_symbols_per_gold_file else 0.0,
            "audit_row_count": len(self.audit_rows),
            "llm_root_grounding_keep_count": self.llm_root_grounding_keep_count,
            "llm_root_grounding_drop_count": self.llm_root_grounding_drop_count,
            "llm_root_grounding_fallback_count": self.llm_root_grounding_fallback_count,
            "average_root_grounding_confidence": round(
                self.llm_root_grounding_confidence_sum / max(1, self.llm_root_grounding_confidence_n),
                3,
            ),
            "per_task_root_grounding_drop_reasons": dict(self.per_task_root_grounding_drop_reasons),
            "average_prompt_length_chars": round(self.prompt_length_chars_sum / max(1, self.serialized_length_n), 2),
            "average_target_length_chars": round(self.target_length_chars_sum / max(1, self.serialized_length_n), 2),
            "per_task_average_snippet_count": {
                k: round(sum(v) / len(v), 2) if v else 0.0
                for k, v in sorted(self.per_task_snippet_counts.items())
            },
            "per_task_average_read_history_steps": {
                k: round(sum(v) / len(v), 2) if v else 0.0
                for k, v in sorted(self.per_task_read_history_steps.items())
            },
            "task_instruction_fallback_count": self.task_instruction_fallback_count,
            "next_action_selection_fallback_count": self.next_action_selection_fallback_count,
            "accepted_patch_grounding_fallback_count": self.accepted_patch_grounding_fallback_count,
            "accepted_ast_dependency_trace_fallback_count": self.accepted_ast_dependency_trace_fallback_count,
            "accepted_reading_summary_fallback_count": self.accepted_reading_summary_fallback_count,
            "rejected_generic_next_question_count": self.rejected_generic_next_question_count,
            "rejected_aux_root_file_count": self.rejected_aux_root_file_count,
            "rejected_symbol_span_mismatch_count": self.rejected_symbol_span_mismatch_count,
            "rejected_reading_summary_generic_nq": self.rejected_reading_summary_generic_nq,
            "rejected_reading_summary_placeholder_region": self.rejected_reading_summary_placeholder_region,
            "rejected_reading_summary_ungrounded_supporting": self.rejected_reading_summary_ungrounded_supporting,
            "rejected_generic_action_reason": self.rejected_generic_action_reason,
            "rejected_fallback_margin_too_small": self.rejected_fallback_margin_too_small,
            "rejected_fallback_broad_action": self.rejected_fallback_broad_action,
            "rejected_root_grounding_noisy_related": self.rejected_root_grounding_noisy_related,
            "rejected_root_grounding_placeholder_span": self.rejected_root_grounding_placeholder_span,
        }


# ── parsing / loading helpers ─────────────────────────────────────────────────


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def dedupe_preserve(items: list[Any]) -> list[Any]:
    seen = set()
    out = []
    for item in items:
        key = stable_json(item) if isinstance(item, (dict, list)) else str(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def action_identity(action: dict[str, Any]) -> str:
    payload = {
        "action": action.get("action"),
        "file_path": action.get("file_path"),
        "symbol": action.get("symbol"),
        "span": action.get("span"),
    }
    return stable_json(payload)


def dedupe_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for item in actions:
        key = action_identity(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def strip_action_meta(action: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "action",
        "file_path",
        "symbol",
        "span",
    }
    return {k: v for k, v in action.items() if k in allowed}


def is_public_action_payload(action: Any) -> bool:
    if not isinstance(action, dict) or not action:
        return False
    allowed = {"action", "file_path", "symbol", "span"}
    if any(k not in allowed for k in action.keys()):
        return False
    if not isinstance(action.get("action"), str) or not action.get("action"):
        return False
    span = action.get("span")
    if span is not None:
        if not (isinstance(span, list) and len(span) == 2 and all(isinstance(x, int) and x > 0 for x in span)):
            return False
    return True


def is_aux_path(file_path: str) -> bool:
    lower = str(file_path or "").lower()
    return any(
        tok in lower
        for tok in (
            "/tests/", "tests/", "/examples/", "examples/", "/demo/", "demo/",
            "notebook", "migration", "script", "cli", "wrapper"
        )
    )


def has_aux_tags(candidate: dict[str, Any]) -> bool:
    tags = set(candidate.get("reason_tags", []) or [])
    return bool(tags & {"aux_module", "aux_path", "tiny_span"})


def root_file_margin_ok(root_file_candidates: list[dict[str, Any]]) -> bool:
    if not root_file_candidates:
        return False
    if len(root_file_candidates) == 1:
        return True
    top1 = float(root_file_candidates[0].get("heuristic_score", 0.0) or 0.0)
    top2 = float(root_file_candidates[1].get("heuristic_score", 0.0) or 0.0)
    margin = ROOT_FILE_AUX_MARGIN if has_aux_tags(root_file_candidates[0]) else ROOT_FILE_MARGIN
    return (top1 - top2) >= margin


def strong_root_grounding_heuristic_ok(root_file_candidates: list[dict[str, Any]]) -> bool:
    if not root_file_candidates:
        return False
    if len(root_file_candidates) == 1:
        return True
    top1 = float(root_file_candidates[0].get("heuristic_score", 0.0) or 0.0)
    top2 = float(root_file_candidates[1].get("heuristic_score", 0.0) or 0.0)
    margin = STRONG_HEURISTIC_FALLBACK_MARGIN
    if has_aux_tags(root_file_candidates[0]):
        margin += 4.0
    return (top1 - top2) >= margin


def strong_action_heuristic_ok(
    action_candidates: list[dict[str, Any]],
    task_type: str = "",
) -> bool:
    """Return True only when the top action is clearly dominant and task-compatible."""
    if not action_candidates:
        return False
    if len(action_candidates) == 1:
        return True
    top1 = float(action_candidates[0].get("candidate_score", 0.0) or 0.0)
    top2 = float(action_candidates[1].get("candidate_score", 0.0) or 0.0)
    # Raised threshold: top action must be clearly ahead.
    if (top1 - top2) < 10.0:
        return False
    # Task-compatibility check: fallback must not pick a weak action type.
    top_action = str(action_candidates[0].get("action", "") or "")
    if task_type == "patch_grounding":
        # open_file alone is too broad — require open_symbol or read_region.
        if top_action == "open_file":
            # Only allow if no open_symbol/read_region candidate exists.
            better_exists = any(
                str(c.get("action", "")) in ("open_symbol", "read_region")
                for c in action_candidates[1:]
            )
            if better_exists:
                return False
    elif task_type == "ast_dependency_trace":
        # stop_and_summarize as fallback is almost always wrong.
        if top_action == "stop_and_summarize":
            return False
    return True


def first_visible_symbol_from_outline(outline: list[dict[str, Any]]) -> str:
    for item in outline or []:
        qualname = str(item.get("qualname", "") or "")
        if qualname:
            return qualname
    return ""


def _span_from_outline(qualname: str, outline: list[dict[str, Any]]) -> list[int] | None:
    """Return [lineno, end_lineno] for a symbol from an AST outline, or None if not found."""
    for item in outline or []:
        if str(item.get("qualname", "") or "") == qualname:
            lo = int(item.get("lineno", 0) or 0)
            hi = int(item.get("end_lineno", 0) or 0)
            if lo > 0 and hi > 0:
                return [lo, hi]
    return None


def resolve_symbol_span(
    qualname: str,
    *,
    source: str | None = None,
    file_path: str | None = None,
    symbol_candidates: list[dict[str, Any]] | None = None,
    outline: list[dict[str, Any]] | None = None,
) -> list[int] | None:
    """Resolve a symbol span from grounded metadata, then source text, then outline."""
    qualname = str(qualname or "").strip()
    file_path = str(file_path or "").strip()
    if not qualname:
        return None

    def _coerce_span(item: dict[str, Any]) -> list[int] | None:
        raw_span = item.get("span")
        if isinstance(raw_span, list) and len(raw_span) == 2:
            lo = int(raw_span[0] or 0)
            hi = int(raw_span[1] or 0)
            if lo > 0 and hi >= lo:
                return [lo, hi]
        lo = int(item.get("lineno", 0) or 0)
        hi = int(item.get("end_lineno", 0) or 0)
        if lo > 0 and hi >= lo:
            return [lo, hi]
        return None

    matched_without_path: list[dict[str, Any]] = []
    for item in symbol_candidates or []:
        if not isinstance(item, dict):
            continue
        candidate_qualname = str(item.get("qualname", item.get("symbol", "")) or "")
        if candidate_qualname != qualname:
            continue
        candidate_file_path = str(item.get("file_path", "") or "").strip()
        if file_path and candidate_file_path == file_path:
            span = _coerce_span(item)
            if span is not None:
                return span
        elif not candidate_file_path:
            matched_without_path.append(item)

    for item in matched_without_path:
        span = _coerce_span(item)
        if span is not None:
            return span

    if source and file_path:
        for item in build_symbol_index(source, file_path):
            if str(item.get("qualname", "") or "") == qualname:
                span = _coerce_span(item)
                if span is not None:
                    return span

    return _span_from_outline(qualname, outline or [])


def is_valid_span(span: Any) -> bool:
    return (
        isinstance(span, list)
        and len(span) == 2
        and all(isinstance(x, int) and x > 0 for x in span)
        and int(span[1]) >= int(span[0])
    )


def spans_nearly_equal(span_a: Any, span_b: Any, tol: int = 2) -> bool:
    if not is_valid_span(span_a) or not is_valid_span(span_b):
        return False
    return (
        abs(int(span_a[0]) - int(span_b[0])) <= int(tol)
        and abs(int(span_a[1]) - int(span_b[1])) <= int(tol)
    )


def outline_span_for_symbol(outline: list[dict[str, Any]], qualname: str) -> list[int] | None:
    target = str(qualname or "")
    for item in outline or []:
        if str(item.get("qualname", "") or "") != target:
            continue
        lo = int(item.get("lineno", 0) or 0)
        hi = int(item.get("end_lineno", 0) or 0)
        if lo > 0 and hi >= lo:
            return [lo, hi]
    return None


def span_length(span: list[int] | None) -> int:
    if not is_valid_span(span):
        return 0
    return int(span[1]) - int(span[0]) + 1


def clip_span_to_max_width(span: list[int] | None, max_width: int) -> list[int] | None:
    if not is_valid_span(span):
        return None
    start, end = int(span[0]), int(span[1])
    width = end - start + 1
    if width <= max_width:
        return [start, end]
    center = (start + end) // 2
    half = max_width // 2
    clipped_start = max(1, center - half)
    clipped_end = clipped_start + max_width - 1
    return [clipped_start, clipped_end]


def spans_overlap(span_a: list[int] | None, span_b: list[int] | None) -> bool:
    if not is_valid_span(span_a) or not is_valid_span(span_b):
        return False
    return max(0, min(int(span_a[1]), int(span_b[1])) - max(int(span_a[0]), int(span_b[0])) + 1) > 0


def snippet_region(snippet: str) -> list[int] | None:
    nums: list[int] = []
    for line in str(snippet or "").splitlines():
        m = re.match(r"^\s*(\d+):", line)
        if m:
            nums.append(int(m.group(1)))
    if not nums:
        return None
    return [min(nums), max(nums)]


def observed_region_from_step(step: dict[str, Any]) -> list[int] | None:
    if not isinstance(step, dict):
        return None
    action = step.get("action", {})
    if isinstance(action, dict):
        span = action.get("span")
        if is_valid_span(span):
            return [int(span[0]), int(span[1])]
    observation = step.get("observation", {})
    if isinstance(observation, dict):
        return snippet_region(str(observation.get("snippet", "") or ""))
    return None


def observed_regions_for_file(history: list[dict[str, Any]], file_path: str) -> list[list[int]]:
    target_file = str(file_path or "")
    regions: list[list[int]] = []
    for step in history:
        if not isinstance(step, dict):
            continue
        obs = step.get("observation", {})
        if not isinstance(obs, dict):
            continue
        if str(obs.get("file_path", "") or "") != target_file:
            continue
        region = observed_region_from_step(step)
        if is_valid_span(region):
            regions.append([int(region[0]), int(region[1])])
    return regions


def next_question_mentions_observed_evidence(text: str, history: list[dict[str, Any]]) -> bool:
    q = str(text or "")
    if not q.strip():
        return False
    observed_files: set[str] = set()
    observed_symbols: set[str] = set()
    for step in history:
        if not isinstance(step, dict):
            continue
        obs = step.get("observation", {})
        if isinstance(obs, dict):
            fp = str(obs.get("file_path", "") or "")
            if fp:
                observed_files.add(fp)
        act = step.get("action", {})
        if isinstance(act, dict):
            symbol = str(act.get("symbol", "") or "")
            if symbol:
                observed_symbols.add(symbol)
                observed_symbols.add(symbol.split(".")[-1])
    return any(token and token in q for token in list(observed_files) + list(observed_symbols))


def collect_observed_symbols(history: list[dict[str, Any]], file_path: str = "") -> list[str]:
    target_file = str(file_path or "")
    symbols: list[str] = []
    for step in history:
        if not isinstance(step, dict):
            continue
        action = step.get("action", {})
        if not isinstance(action, dict):
            continue
        symbol = str(action.get("symbol", "") or "")
        if not symbol:
            continue
        step_file = str(action.get("file_path", "") or step.get("observation", {}).get("file_path", "") or "")
        if target_file and step_file != target_file:
            continue
        symbols.append(symbol)
    return dedupe_preserve(symbols)


def _strip_line_number_prefix(line: str) -> str:
    return re.sub(r"^\s*\d+:\s?", "", str(line or ""))


def snippet_has_real_logic(snippet: str) -> bool:
    cleaned: list[str] = []
    signal = 0
    for raw in str(snippet or "").splitlines():
        line = _strip_line_number_prefix(raw).strip()
        if not line:
            continue
        if line.startswith("#"):
            continue
        cleaned.append(line)
        if line == "pass":
            continue
        if line.startswith(("import ", "from ")):
            continue
        if line.startswith((
            "def ", "class ", "return ", "raise ", "if ", "elif ", "for ", "while ", "with ", "try:", "except ", "@",
        )):
            signal += 1
            continue
        if re.search(r"\breturn\b", line):
            signal += 1
            continue
        if re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*=", line):
            signal += 1
            continue
        if re.search(r"[A-Za-z_][A-Za-z0-9_]*\(", line) and not line.startswith(("import ", "from ")):
            signal += 1
            continue
    if len(cleaned) < 4:
        return False
    if signal <= 0:
        return False
    non_trivial = [x for x in cleaned if x not in {"pass"} and not x.startswith(("import ", "from "))]
    return len(non_trivial) >= 2


def is_trivial_root_file(file_path: str, history: list[dict[str, Any]]) -> bool:
    basename = os.path.basename(str(file_path or ""))
    if basename == "__init__.py":
        return True
    snippets: list[str] = []
    for step in history or []:
        if not isinstance(step, dict):
            continue
        obs = step.get("observation", {})
        if not isinstance(obs, dict):
            continue
        if str(obs.get("file_path", "") or "") != str(file_path or ""):
            continue
        snippet = str(obs.get("snippet", "") or "")
        if snippet.strip():
            snippets.append(snippet)
    if not snippets:
        return True
    return not snippet_has_real_logic("\n".join(snippets))


def patch_grounding_symbol_is_semantically_supported(sample: dict[str, Any]) -> bool:
    input_data = sample.get("input", {}) if isinstance(sample, dict) else {}
    obs = input_data.get("current_observation", {}) if isinstance(input_data, dict) else {}
    target = sample.get("target", {}) if isinstance(sample, dict) else {}
    selected_action = target.get("selected_action", {}) if isinstance(target, dict) else {}
    if not isinstance(selected_action, dict):
        return False
    if str(selected_action.get("action", "") or "") != "open_symbol":
        return True

    symbol = str(selected_action.get("symbol", "") or "")
    if not symbol:
        return False
    leaf = symbol.split(".")[-1].lower()
    goal_text = str(input_data.get("goal_text", "") or "").lower()
    current_snippet = str(obs.get("current_snippet", "") or "")
    current_snippet_lower = current_snippet.lower()
    related_snippets = obs.get("related_snippets", []) if isinstance(obs, dict) else []
    file_ast_outline = obs.get("file_ast_outline", []) if isinstance(obs, dict) else []

    if leaf and leaf in goal_text:
        return True
    if leaf and leaf in current_snippet_lower:
        return True
    for item in related_snippets if isinstance(related_snippets, list) else []:
        if not isinstance(item, dict):
            continue
        snippet_text = str(item.get("snippet", "") or "").lower()
        if leaf and leaf in snippet_text:
            return True

    current_region = snippet_region(current_snippet)
    if is_valid_span(current_region):
        outline_span = outline_span_for_symbol(file_ast_outline if isinstance(file_ast_outline, list) else [], symbol)
        if is_valid_span(outline_span) and spans_overlap(current_region, outline_span):
            return True
        selected_span = selected_action.get("span")
        if is_valid_span(selected_span) and spans_overlap(current_region, selected_span):
            return True
    return False


def choose_symbol_resolution_source(
    qualname: str,
    evidence: FileChangeEvidence | None,
    default_source: str = "",
) -> str:
    """Prefer post-change source for added / after-only symbols, else use stable fallback order."""
    if evidence is None:
        return default_source

    before_source = str(evidence.before_source or "")
    after_source = str(evidence.after_source or "")
    file_path = str(evidence.file_path or "")
    qualname = str(qualname or "").strip()

    if qualname:
        for sym in evidence.changed_symbols or []:
            if str(sym.get("qualname", "") or "") == qualname and str(sym.get("change_type", "") or "").lower() == "added":
                if after_source:
                    return after_source
                break
        if after_source and resolve_symbol_span(qualname, source=after_source, file_path=file_path) is not None:
            before_span = resolve_symbol_span(qualname, source=before_source, file_path=file_path) if before_source else None
            if before_span is None:
                return after_source

    return before_source or after_source or default_source


def filter_symbol_candidates_for_file(
    symbol_candidates: list[dict[str, Any]] | None,
    file_path: str,
) -> list[dict[str, Any]]:
    """Prefer file-local symbol candidates while retaining file-less grounded metadata."""
    target_file = str(file_path or "").strip()
    local: list[dict[str, Any]] = []
    fileless: list[dict[str, Any]] = []
    seen = set()

    for item in symbol_candidates or []:
        if not isinstance(item, dict):
            continue
        candidate_file = str(item.get("file_path", "") or "").strip()
        if candidate_file and target_file and candidate_file != target_file:
            continue
        key = stable_json(item)
        if key in seen:
            continue
        seen.add(key)
        if candidate_file:
            local.append(item)
        else:
            fileless.append(item)
    return local + fileless


# ── reading_summary quality helpers ───────────────────────────────────────────

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "and",
    "or", "but", "if", "this", "that", "it", "its", "which", "what",
    "how", "where", "when", "who", "there", "their", "they", "we", "i",
    "you", "he", "she", "not", "no", "any", "all", "more", "also",
})

_GENERIC_NQ_PATTERNS = (
    "does the observed code align",
    "align with the commit intent",
    "which related file",
    "best confirms the commit intent",
    "remaining evidence",
    "primary callsite/wiring",
    "how do the",
    "what changed",
    "next?",
    "need more info",
    "tbd",
    "unknown",
    "n/a",
    "todo",
    "...",
    "what remaining evidence",
    "which file shows",
    "for the change:",
    "confirm the change",
    "direct call or config dependency",
    "how does `",
    "which callsite or wiring",
)


def is_generic_next_question(text: str) -> bool:
    """Return True if the next_question is too generic / templatic to be useful."""
    if not text or not text.strip():
        return True
    t = text.strip().lower()
    # Exact or prefix match against known generic patterns.
    for pat in _GENERIC_NQ_PATTERNS:
        if t == pat or t.startswith(pat):
            return True
    if re.match(r"^how does `[^`]+` connect to `[^`]+`", t):
        return True
    if t.startswith("which callsite or wiring") and "confirm the change" in t:
        return True
    # Too few lexical tokens after stopword removal.
    all_tokens = re.findall(r"[a-z_][a-z0-9_]*", t)
    tokens = [w for w in all_tokens if w not in _STOPWORDS]
    if len(tokens) < 6:
        return True
    if all_tokens and len(tokens) / max(1, len(all_tokens)) < 0.4 and len(tokens) < 10:
        return True
    return False


def next_question_has_specific_anchor(text: str, history: list[dict[str, Any]], likely_root_file: str) -> bool:
    q = str(text or "")
    if not q.strip():
        return False
    observed_files = dedupe_preserve([
        str(step.get("observation", {}).get("file_path", "") or "")
        for step in history
        if isinstance(step, dict) and isinstance(step.get("observation", {}), dict)
    ])
    observed_files = [p for p in observed_files if p]
    observed_symbols = collect_observed_symbols(history, likely_root_file) or collect_observed_symbols(history)
    symbol_leaves = [sym.split(".")[-1] for sym in observed_symbols if sym]
    file_anchors = observed_files + ([likely_root_file] if likely_root_file else [])
    has_anchor = any(anchor and anchor in q for anchor in file_anchors + observed_symbols + symbol_leaves)
    technical_terms = (
        "callsite", "called", "configured", "config", "argument", "propagation", "state update",
        "branch", "return value", "wrapper", "adapter", "helper", "consumed",
    )
    has_technical_uncertainty = any(term in q.lower() for term in technical_terms)
    has_line_anchor = bool(re.search(r"lines?\s+\d+\s*-\s*\d+", q.lower()))
    return has_anchor and has_technical_uncertainty and (has_line_anchor or any(anchor and anchor in q for anchor in file_anchors))


_SUMMARY_SNIPPET_IGNORE = frozenset({
    "def", "class", "return", "self", "true", "false", "none", "logger",
    "info", "debug", "error", "warning", "import", "from", "with", "open",
    "json", "data", "value", "result", "args", "kwargs", "line", "lines",
})


def snippet_identifiers(text: str, limit: int = 64) -> set[str]:
    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(text or ""))
    out: list[str] = []
    for tok in toks:
        low = tok.lower()
        if len(low) < 4:
            continue
        if low in _STOPWORDS or low in _SUMMARY_SNIPPET_IGNORE:
            continue
        out.append(low)
        if len(out) >= limit:
            break
    return set(out)


def next_question_grounded_in_supporting_context(
    text: str,
    history: list[dict[str, Any]],
    supporting_files: list[str],
) -> bool:
    q = str(text or "").strip()
    if not q:
        return False
    q_tokens = {tok.lower() for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", q)}
    if not q_tokens:
        return False
    support_set = {str(x) for x in supporting_files if str(x)}
    for step in history:
        if not isinstance(step, dict):
            continue
        obs = step.get("observation", {})
        fp = str(obs.get("file_path", "") or "")
        if fp not in support_set:
            continue
        ids = snippet_identifiers(str(obs.get("snippet", "") or ""))
        act = step.get("action", {})
        if isinstance(act, dict):
            sym = str(act.get("symbol", "") or "")
            if sym:
                ids.add(sym.lower())
                ids.add(sym.split(".")[-1].lower())
        if ids & q_tokens:
            return True
    return False


def build_specific_next_question(
    goal_text: str,
    read_history: list[dict[str, Any]],
    likely_root_file: str,
    root_focus_region: list[int] | None,
    supporting_files: list[str],
) -> str:
    del goal_text
    root_file = str(likely_root_file or "")
    if not root_file:
        return ""
    support_file = next((str(p) for p in supporting_files if str(p or "") and str(p) != root_file), "")
    root_symbols = collect_observed_symbols(read_history, root_file) or collect_observed_symbols(read_history)
    root_symbol_leaf = root_symbols[-1].split(".")[-1] if root_symbols else ""
    region = root_focus_region if is_valid_span(root_focus_region) else None
    if region is None:
        regions = observed_regions_for_file(read_history, root_file)
        region = regions[-1] if regions else None
    if not is_valid_span(region):
        return ""
    line_text = f"lines {int(region[0])}-{int(region[1])}"

    if support_file and root_symbol_leaf:
        return (
            f"Where in `{support_file}` is `{root_symbol_leaf}` called, configured, wrapped, or supplied with arguments "
            f"before execution reaches `{root_file}` {line_text}?"
        )
    if support_file:
        return (
            f"What value, config, helper logic, or adapter layer from `{support_file}` is consumed by `{root_file}` {line_text}?"
        )
    if root_symbol_leaf:
        return (
            f"What branch condition, argument propagation, state update, or return value inside `{root_file}` {line_text} "
            f"around `{root_symbol_leaf}` should be verified next?"
        )
    return (
        f"What branch condition, argument propagation, state update, return value, or wrapper logic inside `{root_file}` {line_text} "
        f"should be verified next?"
    )


def is_placeholder_focus_region(
    region: list[int],
    history: list[dict[str, Any]],
) -> bool:
    """Return True if the focus region looks like a fallback placeholder.

    Rules:
    - [1, 10] is rejected unless the read_history snippet genuinely starts near line 1.
    - Region must be grounded in at least one observed snippet line range.
    - Span length < 3 is rejected unless explicitly supported.
    """
    if not (isinstance(region, list) and len(region) == 2):
        return True
    start, end = int(region[0]), int(region[1])
    if end < start or (end - start + 1) < 3:
        return True

    # Collect all line numbers seen in observed snippets.
    observed_line_nums: list[int] = []
    for step in history:
        if not isinstance(step, dict):
            continue
        snip = str(step.get("observation", {}).get("snippet", "") or "")
        for line in snip.splitlines():
            m = re.match(r"^\s*(\d+):", line)
            if m:
                observed_line_nums.append(int(m.group(1)))

    if not observed_line_nums:
        # No line numbers in history — can't verify; reject [1,10] default.
        if start == 1 and end == 10:
            return True
        return False

    obs_min = min(observed_line_nums)
    obs_max = max(observed_line_nums)

    # [1, 10] is only OK if observed snippets genuinely start near line 1.
    if start == 1 and end == 10 and obs_min > 15:
        return True

    # Region must overlap with observed line range.
    overlap = max(0, min(end, obs_max) - max(start, obs_min) + 1)
    if overlap == 0:
        return True

    return False


_PURE_TEMPLATE_REASON_PATTERNS = (
    "n/a", "none", "...", "tbd", "unknown", "todo",
)

# Patterns that are only generic when they are the *entire* reason (no extra content).
_TEMPLATE_ONLY_PATTERNS = (
    "best candidate",
    "highest score",
    "top candidate",
    "top-ranked",
    "top ranked",
    "obvious choice",
    "clear choice",
    "only option",
    "default",
)

# Concrete technical terms that indicate a grounded reason.
_CONCRETE_REASON_TERMS = frozenset({
    "callsite", "wiring", "dependency", "propagat", "entry_point", "root_cause",
    "changed", "patch", "import", "call", "function", "method", "class",
    "symbol", "region", "span", "line", "argument", "parameter", "field",
    "attribute", "return", "config", "consumer", "because", "since", "as it",
    "directly", "specifically", "unlike", "whereas", "instead",
})


def is_generic_action_reason(text: str) -> bool:
    """Return True if the action selection reason is too generic to be trusted.

    A reason is generic if:
    - It is empty or a pure placeholder.
    - It is entirely a template phrase with no additional content.
    - It contains no concrete technical term AND has fewer than 10 non-stopword tokens.

    A reason is NOT generic if it mentions a concrete file/symbol/technical term,
    even if it also contains template phrases like "most likely" or "best match".
    """
    if not text or not text.strip():
        return True
    t = text.strip().lower()
    # Pure placeholders — always generic regardless of length.
    for pat in _PURE_TEMPLATE_REASON_PATTERNS:
        if t == pat or t.startswith(pat + " ") or t.startswith(pat + "."):
            return True
    # Template-only phrases — generic only when the entire reason is the phrase.
    for pat in _TEMPLATE_ONLY_PATTERNS:
        if t == pat:
            return True
    # If the reason contains a concrete technical term, it is grounded.
    if any(term in t for term in _CONCRETE_REASON_TERMS):
        return False
    # Looks for a file path pattern (e.g. "foo/bar.py" or "foo.py").
    if re.search(r"[a-z_][a-z0-9_/]*\.[a-z]{2,4}", t):
        return False
    # Looks for a qualified symbol (e.g. "MyClass.method" or "module.func").
    if re.search(r"[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*", t):
        return False
    # Last resort: require at least 10 non-stopword tokens.
    tokens = [w for w in re.findall(r"[a-z_][a-z0-9_]*", t) if w not in _STOPWORDS]
    return len(tokens) < 10


def deterministic_working_summary(
    goal_text: str,
    read_history: list[dict[str, Any]],
) -> dict[str, Any] | None:
    observed_files: list[str] = []
    observed_symbols: list[str] = []
    root_focus_region: list[int] | None = None
    likely_root_file = ""
    fallback_regions_by_file: dict[str, list[int]] = {}

    for step in read_history:
        if not isinstance(step, dict):
            continue
        obs = step.get("observation", {})
        if not isinstance(obs, dict):
            continue
        fp = str(obs.get("file_path", "") or "")
        if fp:
            observed_files.append(fp)
            region = observed_region_from_step(step)
            if is_valid_span(region):
                fallback_regions_by_file[fp] = [int(region[0]), int(region[1])]
        act = step.get("action", {})
        if isinstance(act, dict):
            if act.get("symbol"):
                observed_symbols.append(str(act["symbol"]))
            action_type = str(act.get("action", "") or "")
            if fp and action_type in {"open_symbol", "read_region"}:
                likely_root_file = fp
                region = observed_region_from_step(step)
                if is_valid_span(region):
                    root_focus_region = [int(region[0]), int(region[1])]

    observed_files = dedupe_preserve([p for p in observed_files if p])
    if not likely_root_file and observed_files:
        likely_root_file = observed_files[0]
    if root_focus_region is None and likely_root_file:
        root_focus_region = fallback_regions_by_file.get(likely_root_file)

    if len(read_history) < 2 or not observed_files or not likely_root_file:
        return None
    if root_focus_region is None:
        return None

    supporting_files = [p for p in observed_files if p != likely_root_file][:2]
    if not supporting_files:
        return None

    next_q = build_specific_next_question(
        goal_text=goal_text,
        read_history=read_history,
        likely_root_file=likely_root_file,
        root_focus_region=root_focus_region,
        supporting_files=supporting_files,
    )
    if (
        not next_q
        or is_generic_next_question(next_q)
        or not next_question_mentions_observed_evidence(next_q, read_history)
        or not next_question_has_specific_anchor(next_q, read_history, likely_root_file)
    ):
        return None

    return {
        "likely_root_file": likely_root_file,
        "likely_focus_region": root_focus_region,
        "supporting_files": supporting_files,
        "next_question": next_q,
    }


def normalize_text(text: Any, limit: int = MAX_CONTEXT_CHARS) -> str:
    return str(text or "").strip()[:limit]


def patch_summary(patch: Any, limit: int = 1400) -> str:
    return normalize_text(patch, limit=limit)


def trim_lines(text: Any, max_lines: int = 40, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    lines = str(text or "").splitlines()
    kept: list[str] = []
    total_chars = 0
    for line in lines:
        projected = total_chars + len(line) + 1
        if len(kept) >= max_lines or projected > max_chars:
            break
        kept.append(line)
        total_chars = projected
    return "\n".join(kept).strip()


def compact_bullets(items: list[str], limit: int = 8) -> list[str]:
    return [str(item).strip() for item in items if str(item).strip()][:limit]


def parent_directories(file_path: str) -> list[str]:
    parts = [part for part in str(file_path).split("/") if part]
    prefixes = []
    for idx in range(1, len(parts)):
        prefixes.append("/".join(parts[:idx]))
    return prefixes


def compact_repo_tree(repo_tree_structure: Any, focus_paths: list[str], max_lines: int = 60, max_chars: int = 1800) -> str:
    lines = [line.rstrip() for line in str(repo_tree_structure or "").splitlines() if line.strip()]
    if not lines:
        return ""
    tokens = set()
    for path in focus_paths:
        if not path:
            continue
        tokens.add(path)
        tokens.update(parent_directories(path))
        tokens.add(top_level_directory(path))
    selected = []
    for line in lines:
        normalized = line.strip().lstrip("├└│─ ")
        if not tokens or any(token and (normalized == token or normalized.startswith(f"{token}/") or token in normalized) for token in tokens):
            selected.append(normalized)
    if not selected:
        selected = lines[:max_lines]
    return trim_lines("\n".join(dedupe_preserve(selected)), max_lines=max_lines, max_chars=max_chars)


def compact_patch_hunk(patch_text: str, max_hunks: int = 2, max_lines: int = 40, max_chars: int = 1400) -> str:
    lines = str(patch_text or "").splitlines()
    if not lines:
        return ""
    hunks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if line.startswith("@@"):
            if current:
                hunks.append(current)
            current = [line]
            continue
        if current:
            current.append(line)
    if current:
        hunks.append(current)
    if not hunks:
        return trim_lines(patch_text, max_lines=max_lines, max_chars=max_chars)
    flattened = []
    for hunk in hunks[:max_hunks]:
        flattened.extend(hunk)
    return trim_lines("\n".join(flattened), max_lines=max_lines, max_chars=max_chars)


def symbol_context_snippet(source: str, symbol: dict[str, Any], padding: int = 3, max_lines: int = 24) -> str:
    lines = source.splitlines()
    if not lines:
        return ""
    start = max(0, int(symbol.get("lineno", 0) or 1) - 1 - padding)
    end = min(len(lines), int(symbol.get("end_lineno", 0) or len(lines)) + padding)
    selected = []
    for idx in range(start, end):
        selected.append(f"{idx + 1}: {lines[idx]}")
        if len(selected) >= max_lines:
            break
    return "\n".join(selected).strip()


def span_context_snippet(source: str, lineno: int, end_lineno: int, padding: int = 3, max_lines: int = 40) -> str:
    """Return a compact, line-numbered snippet around a changed span."""
    lines = source.splitlines()
    if not lines:
        return ""
    start = max(0, (lineno or 1) - 1 - padding)
    end = min(len(lines), (end_lineno or lineno or 1) + padding)
    selected: list[str] = []
    for idx in range(start, end):
        selected.append(f"{idx + 1}: {lines[idx]}")
        if len(selected) >= max_lines:
            break
    return "\n".join(selected).strip()


def numbered_file_snippet(source: str, max_lines: int = 60) -> str:
    """Return a simple head snippet with line numbers for an entire file."""
    lines = source.splitlines()
    if not lines:
        return ""
    out: list[str] = []
    for idx, line in enumerate(lines[:max_lines]):
        out.append(f"{idx + 1}: {line}")
    return "\n".join(out).strip()


def truncate_text_for_audit(text: str, max_chars: int = 600) -> str:
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20] + "\n...[truncated]"


def public_root_file_candidates(candidates: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in candidates or []:
        if not isinstance(item, dict):
            continue
        file_path = str(item.get("file_path", "") or "").strip()
        if not file_path or file_path in seen:
            continue
        seen.add(file_path)
        out.append({"file_path": file_path})
    return out[:5]


def public_root_symbol_candidates(candidates: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in candidates or []:
        if not isinstance(item, dict):
            continue
        file_path = str(item.get("file_path", "") or "").strip()
        symbol = str(item.get("symbol", "") or "").strip()
        kind = str(item.get("kind", "") or "").strip()
        span = item.get("span", [])
        if not file_path and not symbol:
            continue
        key = stable_json({
            "file_path": file_path,
            "symbol": symbol,
            "kind": kind,
            "span": span,
        })
        if key in seen:
            continue
        seen.add(key)
        record: dict[str, Any] = {
            "file_path": file_path,
            "symbol": symbol,
            "kind": kind,
        }
        if is_valid_span(span):
            record["span"] = [int(span[0]), int(span[1])]
        out.append(record)
    return out[:6]


def public_root_span_candidates(candidates: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in candidates or []:
        if not isinstance(item, dict):
            continue
        file_path = str(item.get("file_path", "") or "").strip()
        span = item.get("span", [])
        source = str(item.get("source", "") or "").strip()
        if not file_path or not is_valid_span(span):
            continue
        key = stable_json({
            "file_path": file_path,
            "span": span,
            "source": source,
        })
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "file_path": file_path,
            "span": [int(span[0]), int(span[1])],
            "source": source,
        })
    return out[:5]


def public_related_snippet_candidates(candidates: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in candidates or []:
        if not isinstance(item, dict):
            continue
        file_path = str(item.get("file_path", "") or "").strip()
        provenance = str(item.get("provenance", "") or "").strip()
        if not file_path or file_path in seen:
            continue
        seen.add(file_path)
        out.append({
            "file_path": file_path,
            "provenance": provenance,
        })
    return out[:8]


def is_noisy_snippet(snippet: str) -> tuple[bool, list[str]]:
    """Lightweight noise detector for snippet candidates.

    Flags large constants / ABI blobs / generated-looking regions.
    """
    snippet = str(snippet or "")
    flags: list[str] = []
    if not snippet.strip():
        return True, ["empty"]

    # ABI / contract blobs
    if "ABI" in snippet or " abi" in snippet.lower() or '"abi"' in snippet:
        flags.append("abi_blob")

    # Hex literal density
    if snippet.count("0x") >= 6:
        flags.append("many_hex_literals")

    # Large array / matrix literals: many brackets or commas on same lines
    bracket_count = snippet.count("[") + snippet.count("]")
    if bracket_count >= 40:
        flags.append("large_array_like")

    # Large numeric array: lines that are mostly numbers/commas/brackets
    numeric_lines = 0
    for line in snippet.splitlines():
        stripped = line.strip()
        if len(stripped) > 20:
            non_numeric = re.sub(r"[\d\s,.\-+eE\[\]()]+", "", stripped)
            if len(non_numeric) <= max(2, len(stripped) * 0.08):
                numeric_lines += 1
    if numeric_lines >= 4:
        flags.append("large_numeric_array")

    # Very long lines (minified / generated code)
    long_lines = [line for line in snippet.splitlines() if len(line) > 150]
    if len(long_lines) >= 2:
        flags.append("many_long_lines")

    # Import-only files with no logic
    if snippet.count("import ") >= 8 and "def " not in snippet and "class " not in snippet:
        flags.append("import_only")

    # Repeated identical short lines (e.g. generated enum tables)
    lines = [l.strip() for l in snippet.splitlines() if l.strip()]
    if len(lines) >= 10:
        counter = Counter(lines)
        most_common_count = counter.most_common(1)[0][1]
        if most_common_count >= max(5, len(lines) * 0.4):
            flags.append("repetitive_lines")

    # Migration / schema files: mostly string literals or SQL-like content
    if len(lines) >= 6:
        string_literal_lines = sum(
            1 for l in lines
            if l.startswith(("'", '"', "b'", 'b"', "migrations.", "operations."))
        )
        if string_literal_lines >= max(5, len(lines) * 0.6):
            flags.append("migration_or_schema_blob")

    # Mostly comment lines — no executable logic
    if len(lines) >= 8:
        comment_lines = sum(1 for l in lines if l.startswith("#"))
        if comment_lines >= max(6, len(lines) * 0.7):
            flags.append("comment_only_block")

    # Dense string-constant blocks (e.g. translation files, fixture data)
    if snippet.count('": "') >= 8 or snippet.count("': '") >= 8:
        flags.append("string_constant_block")

    noisy = bool(flags)
    return noisy, flags


def choose_non_answer_snippet_for_file(source: str) -> str:
    """Pick a readable, non-answer-biased snippet from a file.

    Preference:
    1. non-noisy def/class region
    2. non-noisy mid-file window
    3. non-noisy head window
    4. fallback head window
    """
    source = str(source or "")
    if not source.strip():
        return ""

    lines = source.splitlines()

    # 1) try def/class regions
    for idx, line in enumerate(lines[:800]):
        if line.lstrip().startswith(("def ", "class ")):
            snippet = span_context_snippet(source, idx + 1, idx + 1, padding=8, max_lines=22)
            noisy, _ = is_noisy_snippet(snippet)
            if not noisy:
                return snippet

    # 2) scan windows through the file
    window = 20
    stride = 15
    for start in range(0, min(len(lines), 900), stride):
        end = min(len(lines), start + window)
        snippet = "\n".join(f"{i + 1}: {lines[i]}" for i in range(start, end))
        noisy, _ = is_noisy_snippet(snippet)
        if not noisy:
            return snippet

    # 3) try head window if unavoidable
    snippet = numbered_file_snippet(source, max_lines=20)
    return snippet


def action_open_file(file_path: str) -> dict[str, Any]:
    return {"action": "open_file", "file_path": str(file_path)}


def action_open_symbol(file_path: str, symbol: str, span: list[int] | None = None) -> dict[str, Any]:
    a: dict[str, Any] = {"action": "open_symbol", "file_path": str(file_path), "symbol": str(symbol)}
    if span and isinstance(span, list) and len(span) == 2 and int(span[0]) > 0 and int(span[1]) > 0:
        a["span"] = [int(span[0]), int(span[1])]
    return a


def action_read_region(file_path: str, span: list[int]) -> dict[str, Any]:
    return {"action": "read_region", "file_path": str(file_path), "span": [int(span[0]), int(span[1])]}


def action_follow_dependency(file_path: str) -> dict[str, Any]:
    return {"action": "follow_dependency", "file_path": str(file_path)}


def action_stop_and_summarize() -> dict[str, Any]:
    return {"action": "stop_and_summarize"}


def build_teacher_read_trace(
    row: dict[str, Any],
    grounding: dict[str, Any],
    repo_snapshot: dict[str, str],
    dep_map: dict[str, list[str]],
    reverse_dep_map: dict[str, list[str]],
) -> list[dict[str, Any]]:
    """Reconstruct a tiny teacher read trace from commit + repo evidence.

    The trace is intentionally short (2–3 steps) and is used to construct
    action-style tasks and a working-memory summary.

    Step 0: open root file with a non-answer-biased snippet.
    Step 1: narrow to root symbol/span — always uses the actual root_span_preview
            so the trace is grounded in the real changed region.
    Step 2 (optional): one related file for cross-file context.
    """
    root_file = str(grounding.get("root_file", "") or "")
    root_symbol = str(grounding.get("root_symbol", "") or "")
    root_span = grounding.get("root_line_span", [])
    root_span_preview = str(grounding.get("root_span_preview", "") or "")
    related = grounding.get("related_snippets", []) if isinstance(grounding.get("related_snippets", []), list) else []
    root_source = repo_snapshot.get(root_file, "")
    grounded_symbol_candidates = filter_symbol_candidates_for_file(grounding.get("root_symbol_candidates", []), root_file)
    root_outline = build_file_outline(root_source, root_file) if root_source else []

    steps: list[dict[str, Any]] = []

    # Step 0: open root file with a non-answer snippet.
    steps.append({
        "action": action_open_file(root_file),
        "observation": {
            "file_path": root_file,
            "snippet": choose_non_answer_snippet_for_file(repo_snapshot.get(root_file, "")),
        },
    })

    # Step 1: narrow to the actual root region — only emit open_symbol when the
    # observation snippet can be grounded to the symbol itself.
    symbol_span = resolve_symbol_span(
        root_symbol,
        source=root_source,
        file_path=root_file,
        symbol_candidates=grounded_symbol_candidates,
        outline=root_outline,
    ) if root_symbol else None
    symbol_preview = ""
    if root_source and is_valid_span(symbol_span):
        symbol_preview = extract_span_preview(root_source, int(symbol_span[0]), int(symbol_span[1]), max_lines=8)

    if root_symbol and symbol_preview:
        steps.append({
            "action": action_open_symbol(root_file, root_symbol, span=symbol_span),
            "observation": {
                "file_path": root_file,
                "snippet": symbol_preview,
            },
        })
    elif isinstance(root_span, list) and len(root_span) == 2 and int(root_span[0]) > 0 and root_span_preview:
        steps.append({
            "action": action_read_region(root_file, root_span),
            "observation": {
                "file_path": root_file,
                "snippet": root_span_preview,
            },
        })

    # Step 2: one related file for cross-file context.
    for item in related[:2]:
        fp = str(item.get("file_path", "") or "")
        snip = str(item.get("snippet", "") or "")
        if fp and snip and fp != root_file:
            steps.append({
                "action": action_open_file(fp),
                "observation": {
                    "file_path": fp,
                    "snippet": snip,
                },
            })
            break
    return steps[:4]


def format_key_value_block(title: str, value: str) -> str:
    value = str(value or "").strip()
    if not value:
        return ""
    return f"{title}:\n{value}"


def format_list_block(title: str, items: list[str]) -> str:
    entries = [f"- {item}" for item in items if str(item).strip()]
    if not entries:
        return ""
    return f"{title}:\n" + "\n".join(entries)


def is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    try:
        missing = pd.isna(value)
    except Exception:
        return False
    return bool(missing) if isinstance(missing, (bool, int)) else False


def is_python_file(path: str) -> bool:
    return str(path).endswith(".py")


def parse_changed_files(file_changed_content: Any) -> list[str]:
    files = []
    for line in str(file_changed_content or "").splitlines():
        raw = line.strip()
        if not raw:
            continue
        path = raw.split("\t")[-1].strip()
        if path:
            files.append(path)
    return dedupe_preserve(files)


def top_level_directory(file_path: str) -> str:
    parts = [part for part in str(file_path).split("/") if part]
    if not parts:
        return ""
    return parts[0] if len(parts) > 1 else "."


def lexical_tokens(text: str) -> set[str]:
    stopwords = {
        "fix", "update", "add", "remove", "cleanup", "clean", "style", "lint", "tests", "test",
        "api", "bug", "issue", "support", "use", "for", "from", "with", "the", "and", "into",
    }
    tokens = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]+", str(text or "").lower()))
    return {token for token in tokens if len(token) >= 3 and token not in stopwords}


_GENERIC_GOAL_PATTERNS = (
    "added few minor changes",
    "few minor changes",
    "minor changes",
    "misc fixes",
    "misc fix",
    "small updates",
    "small update",
    "cleanup",
    "clean up",
    "refactor",
    "refactoring",
)

_GENERIC_GOAL_TOKENS = frozenset({
    "add", "added", "few", "minor", "change", "changes", "misc", "fix", "fixes",
    "small", "update", "updates", "cleanup", "clean", "refactor", "refactoring",
    "tweak", "tweaks", "code", "stuff", "various", "some",
})


def is_low_information_goal_text(
    commit_message: str,
    changed_py_files: list[str],
    evidence_list: list[FileChangeEvidence],
) -> tuple[bool, str]:
    message = normalize_text(commit_message, 300).strip().lower()
    if not message:
        return True, "too_few_specific_tokens"

    if any(message == pat or message.startswith(pat + " ") or pat in message for pat in _GENERIC_GOAL_PATTERNS):
        return True, "generic_goal_phrase"

    raw_tokens = set(re.findall(r"[a-z][a-z0-9_]*", message))
    specific_tokens = {
        tok for tok in raw_tokens
        if len(tok) >= 3 and tok not in _GENERIC_GOAL_TOKENS and tok not in _STOPWORDS
    }
    if len(specific_tokens) < 2:
        return True, "too_few_specific_tokens"

    anchor_tokens: set[str] = set()
    for path in changed_py_files:
        stem = Path(str(path or "")).stem
        anchor_tokens.update(lexical_tokens(stem.replace("_", " ")))
    for evidence in evidence_list or []:
        if not isinstance(evidence, FileChangeEvidence):
            continue
        for sym in evidence.changed_symbols or []:
            leaf = str(sym.get("symbol", "") or sym.get("qualname", "") or "").split(".")[-1]
            anchor_tokens.update(lexical_tokens(leaf.replace("_", " ")))

    if not (specific_tokens & anchor_tokens):
        return True, "goal_text_not_grounded"
    return False, ""


def message_to_path_overlap(commit_message: str, path_or_symbol: str) -> int:
    return len(lexical_tokens(commit_message) & lexical_tokens(path_or_symbol.replace("/", " ").replace(".", " ")))


def message_supports_candidate_choice(commit_message: str, gold_name: str, distractor_names: list[str], margin: int = 1) -> bool:
    """Conservative filter for hopelessly underspecified commit messages.

    Historically this helper enforced lexical overlap between the commit
    message and the gold/distractor names. That proved far too strict and
    dropped many structurally valid tasks just because filenames or symbols
    were not mentioned verbatim in the message.

    The new behavior keeps only a light sanity check on the message itself
    (to rule out extremely low-signal commits) and does *not* require any
    direct lexical overlap with candidate names.
    """
    message = normalize_text(commit_message, 400).lower()
    tokens = lexical_tokens(message)
    # Treat very low-signal messages (after earlier length checks) as
    # unsupported; everything else is considered usable supervision.
    return len(tokens) >= 3


def parse_relevant_files(relevant_file_content: Any) -> list[dict[str, Any]]:
    if is_missing_value(relevant_file_content):
        return []
    try:
        data = json.loads(relevant_file_content) if isinstance(relevant_file_content, str) else relevant_file_content
    except Exception:
        return []
    if not isinstance(data, list):
        return []

    results: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        file_path = str(item.get("file_path", "")).strip()
        if not file_path:
            continue
        try:
            distance = float(item.get("distance", 9.9))
        except Exception:
            distance = 9.9
        results.append({
            "file_path": file_path,
            "distance": distance,
            "content": normalize_text(item.get("content", ""), 800),
        })
    return results


def extract_before_after(row: dict[str, Any]) -> dict[str, dict[str, str]]:
    raw = row.get("before_after_files")
    if is_missing_value(raw):
        return {}
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}

    result: dict[str, dict[str, str]] = {}
    for file_path, versions in data.items():
        if not isinstance(versions, dict):
            continue
        path = str(file_path).strip()
        if not path:
            continue
        result[path] = {
            "before": str(versions.get("code_before", "") or ""),
            "after": str(versions.get("code_after", "") or ""),
        }
    return result


def patch_is_truncated(patch_text: str) -> bool:
    lower = patch_text.lower()
    return any(marker.lower() in lower for marker in TRUNCATION_MARKERS)


def extract_changed_line_spans_from_patch(patch_text: str) -> list[list[int]]:
    """Extract after-side changed line spans from a unified diff.

    - Uses @@ -a,b +c,d @@ headers.
    - Returns [start_line, end_line] spans in the *after* file.
    - Ignores hunks with no after-side lines (pure deletions).
    - Merges nearby spans to keep the signal compact.
    """
    if not patch_text:
        return []

    hunk_header_re = re.compile(r"^@@ -(?P<a_start>\d+)(?:,(?P<a_len>\d+))? \+(?P<b_start>\d+)(?:,(?P<b_len>\d+))? @@")
    changed_lines: list[int] = []

    lines = str(patch_text).splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        match = hunk_header_re.match(line)
        if not match:
            idx += 1
            continue

        after_start = int(match.group("b_start") or 0)
        after_len = int(match.group("b_len") or 0)
        # If there is no after-side range, this hunk is delete-only; ignore it.
        if after_len <= 0:
            idx += 1
            continue

        after_line = after_start
        idx += 1
        while idx < len(lines) and not lines[idx].startswith("@@ "):
            body = lines[idx]
            if body.startswith("+++") or body.startswith("---") or body.startswith("diff --git "):
                idx += 1
                continue
            if body.startswith("+"):
                # Added/modified line in after-file.
                changed_lines.append(after_line)
                after_line += 1
            elif body.startswith("-"):
                # Deletion: consumes before-side only.
                pass
            else:
                # Context line: advances both sides.
                after_line += 1
            idx += 1

    if not changed_lines:
        return []

    changed_lines = sorted(set(changed_lines))
    spans: list[list[int]] = []
    start = changed_lines[0]
    prev = changed_lines[0]
    # Merge lines that are close together into spans.
    MAX_GAP = 2
    for line_no in changed_lines[1:]:
        if line_no <= prev + MAX_GAP:
            prev = line_no
            continue
        spans.append([start, prev])
        start = prev = line_no
    spans.append([start, prev])
    return spans


def parse_patch_sections(patch: Any) -> dict[str, dict[str, Any]]:
    sections: dict[str, dict[str, Any]] = {}
    current_path: str | None = None
    current_lines: list[str] = []
    added = 0
    removed = 0

    def finalize() -> None:
        nonlocal current_path, current_lines, added, removed
        if not current_path:
            current_lines = []
            added = 0
            removed = 0
            return
        text = "\n".join(current_lines).strip()
        sections[current_path] = {
            "text": text,
            "added_lines": added,
            "removed_lines": removed,
            "changed_lines": added + removed,
            "is_truncated": patch_is_truncated(text),
        }
        current_path = None
        current_lines = []
        added = 0
        removed = 0

    for line in str(patch or "").splitlines():
        if line.startswith("diff --git "):
            finalize()
            match = re.match(r"diff --git a/(.*?) b/(.*)", line)
            current_path = match.group(2).strip() if match else None
            current_lines = [line]
            continue
        if current_path is None:
            continue
        current_lines.append(line)
        if line.startswith("rename to "):
            current_path = line.split("rename to ", 1)[1].strip()
        elif line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    finalize()
    return sections


def build_repo_snapshot(initial_df: pd.DataFrame | None, before_after: dict[str, dict[str, str]]) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    if initial_df is not None:
        for _, row in initial_df.iterrows():
            file_path = str(row.get("file_path", "")).strip()
            content = row.get("content", "")
            if not file_path or not is_python_file(file_path):
                continue
            if content in (None, ""):
                continue
            snapshot[file_path] = str(content)

    for file_path, versions in before_after.items():
        if not is_python_file(file_path):
            continue
        before = versions.get("before", "")
        after = versions.get("after", "")
        if before:
            snapshot[file_path] = before
        elif file_path in snapshot:
            continue
        elif after:
            snapshot[file_path] = after
    return snapshot


def clear_directory(dir_path: str | Path) -> None:
    dir_path = Path(dir_path)
    if not dir_path.exists():
        return
    if not dir_path.is_dir():
        raise ValueError(f"{dir_path} is not a directory")
    for item in dir_path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def build_file_tree_from_path_map(path2node: dict[str, dict[str, Any]], overwrite: bool = False) -> dict[str, Any]:
    root: dict[str, Any] = {}
    for raw_path, node in path2node.items():
        path = str(raw_path or "").strip()
        if not path:
            continue
        if path.startswith("./"):
            path = path[2:]
        parts = [part for part in path.split("/") if part and part != "."]
        if not parts:
            continue
        current = root
        for name in parts[:-1]:
            existing = current.get(name)
            if not isinstance(existing, dict):
                existing = {}
                current[name] = existing
            current = existing
        leaf = parts[-1]
        if leaf not in current or overwrite:
            current[leaf] = node
    return root


def extract_top_level_api(source: str, filename: str = "<unknown>") -> dict[str, Any]:
    tree = ast.parse(source, filename=filename)
    lines = source.splitlines(keepends=True)

    def get_src(node: ast.AST) -> str:
        try:
            segment = ast.get_source_segment(source, node)
            if segment is not None:
                return segment
        except Exception:
            pass
        if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
            return "".join(lines[node.lineno - 1: node.end_lineno])
        return ""

    def unparse(node: ast.AST | None) -> str:
        return ast.unparse(node) if node else ""

    def format_args(arguments: ast.arguments) -> str:
        parts: list[str] = []
        posonly = getattr(arguments, "posonlyargs", [])
        for arg in posonly:
            parts.append(arg.arg + (f": {unparse(arg.annotation)}" if arg.annotation else ""))
        if posonly:
            parts.append("/")
        for arg in arguments.args:
            parts.append(arg.arg + (f": {unparse(arg.annotation)}" if arg.annotation else ""))
        if arguments.vararg:
            parts.append("*" + arguments.vararg.arg)
        elif arguments.kwonlyargs:
            parts.append("*")
        for arg, default in zip(arguments.kwonlyargs, arguments.kw_defaults or []):
            text = arg.arg + (f": {unparse(arg.annotation)}" if arg.annotation else "")
            if default:
                text += f"={unparse(default)}"
            parts.append(text)
        if arguments.kwarg:
            parts.append("**" + arguments.kwarg.arg)
        if arguments.defaults:
            for index, default in enumerate(arguments.defaults):
                target_index = len(parts) - len(arguments.defaults) + index
                parts[target_index] += f"={unparse(default)}"
        return "(" + ", ".join(parts) + ")"

    def signature(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        sig = format_args(fn.args)
        if fn.returns:
            sig += f" -> {unparse(fn.returns)}"
        return sig

    api = {"functions": {}, "classes": {}}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            api["functions"][node.name] = {
                "name": node.name,
                "qualname": node.name,
                "signature": signature(node),
                "lineno": node.lineno,
                "end_lineno": node.end_lineno,
                "source": get_src(node),
            }
        elif isinstance(node, ast.ClassDef):
            class_entry = {
                "name": node.name,
                "qualname": node.name,
                "bases": [unparse(base) for base in node.bases],
                "lineno": node.lineno,
                "end_lineno": node.end_lineno,
                "source": get_src(node),
                "methods": {},
            }
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    class_entry["methods"][child.name] = {
                        "name": child.name,
                        "qualname": f"{node.name}.{child.name}",
                        "signature": signature(child),
                        "lineno": child.lineno,
                        "end_lineno": child.end_lineno,
                        "source": get_src(child),
                    }
            api["classes"][node.name] = class_entry
    return api


# ── repo snapshot + symbol extraction ─────────────────────────────────────────


def build_symbol_index(source: str, file_path: str) -> list[dict[str, Any]]:
    if not source.strip():
        return []
    try:
        parsed = extract_top_level_api(source, filename=file_path)
    except Exception:
        return []

    symbols: list[dict[str, Any]] = []
    for fn in parsed.get("functions", {}).values():
        symbols.append({
            "file_path": file_path,
            "symbol": fn.get("name", ""),
            "qualname": fn.get("qualname", fn.get("name", "")),
            "kind": "function",
            "lineno": int(fn.get("lineno", 0) or 0),
            "end_lineno": int(fn.get("end_lineno", 0) or 0),
            "signature": fn.get("signature", ""),
            "source": fn.get("source", ""),
        })

    for cls in parsed.get("classes", {}).values():
        symbols.append({
            "file_path": file_path,
            "symbol": cls.get("name", ""),
            "qualname": cls.get("qualname", cls.get("name", "")),
            "kind": "class",
            "lineno": int(cls.get("lineno", 0) or 0),
            "end_lineno": int(cls.get("end_lineno", 0) or 0),
            "signature": f"class {cls.get('name', '')}",
            "source": cls.get("source", ""),
        })
        for method in cls.get("methods", {}).values():
            symbols.append({
                "file_path": file_path,
                "symbol": method.get("name", ""),
                "qualname": method.get("qualname", method.get("name", "")),
                "kind": "method",
                "lineno": int(method.get("lineno", 0) or 0),
                "end_lineno": int(method.get("end_lineno", 0) or 0),
                "signature": method.get("signature", ""),
                "source": method.get("source", ""),
            })
    return [sym for sym in symbols if sym.get("qualname")]


def symbol_priority(symbol: dict[str, Any]) -> tuple[int, int, int, str]:
    change_rank = {"modified": 0, "added": 1, "removed": 2}.get(symbol.get("change_type", "modified"), 9)
    kind_rank = {"method": 0, "function": 1, "class": 2}.get(symbol.get("kind", ""), 9)
    span = max(1, int(symbol.get("end_lineno", 0) or 0) - int(symbol.get("lineno", 0) or 0) + 1)
    return (change_rank, kind_rank, span, str(symbol.get("qualname", "")))


def diff_symbol_spans(before: str, after: str, file_path: str) -> list[dict[str, Any]]:
    before_index = {item["qualname"]: item for item in build_symbol_index(before, file_path)}
    after_index = {item["qualname"]: item for item in build_symbol_index(after, file_path)}
    changed: list[dict[str, Any]] = []

    for qualname in sorted(set(before_index) | set(after_index)):
        before_item = before_index.get(qualname)
        after_item = after_index.get(qualname)
        if before_item is None and after_item is not None:
            item = dict(after_item)
            item["change_type"] = "added"
            changed.append(item)
            continue
        if after_item is None and before_item is not None:
            item = dict(before_item)
            item["change_type"] = "removed"
            changed.append(item)
            continue
        if not before_item or not after_item:
            continue
        if before_item.get("kind") != after_item.get("kind") or before_item.get("source") != after_item.get("source"):
            item = dict(after_item)
            item["change_type"] = "modified"
            changed.append(item)

    changed.sort(key=symbol_priority)
    return [symbol for symbol in changed if symbol_is_meaningful(symbol)]


def choose_primary_changed_span(
    changed_line_spans: list[list[int]],
    changed_symbols: list[dict[str, Any]],
) -> list[int] | None:
    """Choose a primary changed span, preferring overlap with key symbols.

    Heuristic:
    - If there are changed symbols, prefer spans that overlap the highest
      priority symbol (using the same ordering as `symbol_priority`).
    - Otherwise, or on tie, fall back to the longest span; if still tied,
      choose the earliest.
    """
    if not changed_line_spans:
        return None

    spans = sorted(changed_line_spans, key=lambda s: (s[0], s[1]))
    if not changed_symbols:
        spans.sort(key=lambda s: (-(s[1] - s[0] + 1), s[0], s[1]))
        return spans[0]

    # Highest-priority symbol according to existing ranking.
    ranked_symbols = [sym for sym in changed_symbols if symbol_is_meaningful(sym)]
    if not ranked_symbols:
        spans.sort(key=lambda s: (-(s[1] - s[0] + 1), s[0], s[1]))
        return spans[0]
    ranked_symbols.sort(key=symbol_priority)
    top_symbol = ranked_symbols[0]
    sym_start = int(top_symbol.get("lineno", 0) or 0)
    sym_end = int(top_symbol.get("end_lineno", sym_start) or sym_start)

    def span_score(span: list[int]) -> tuple[int, int, int]:
        s_start, s_end = span
        overlap = max(0, min(s_end, sym_end) - max(s_start, sym_start) + 1)
        length = s_end - s_start + 1
        return (1 if overlap > 0 else 0, overlap, length)

    best_span = None
    best_key = (-1, -1, -1)
    for span in spans:
        key = span_score(span)
        if key > best_key:
            best_key = key
            best_span = span

    if best_span is not None:
        return best_span

    spans.sort(key=lambda s: (-(s[1] - s[0] + 1), s[0], s[1]))
    return spans[0]


def symbol_is_meaningful(symbol: dict[str, Any]) -> bool:
    qualname = str(symbol.get("qualname", "")).strip()
    lineno = int(symbol.get("lineno", 0) or 0)
    end_lineno = int(symbol.get("end_lineno", 0) or 0)
    source = str(symbol.get("source", "") or "")
    return bool(qualname and lineno > 0 and end_lineno >= lineno and source.strip())


def build_file_outline(source: str, file_path: str, limit: int = 20) -> list[dict[str, Any]]:
    outline = []
    for sym in build_symbol_index(source, file_path)[:limit]:
        outline.append({
            "qualname": sym["qualname"],
            "kind": sym["kind"],
            "lineno": sym["lineno"],
            "end_lineno": sym["end_lineno"],
        })
    return outline


def outline_source_for_symbol_task(evidence: FileChangeEvidence, gold_symbol: dict[str, Any] | None) -> str:
    """Choose a non-leaky but symbol-consistent source for AST outlines.

    For newly *added* symbols the outline must be built from the post-change
    source so that the symbol is visible in the prompt-side AST. For other
    cases we prefer the pre-change source when available, falling back to
    the post-change snapshot only if necessary.
    """
    if gold_symbol is not None and str(gold_symbol.get("change_type", "")).lower() == "added":
        if str(evidence.after_source or "").strip():
            return evidence.after_source
    if str(evidence.before_source or "").strip():
        return evidence.before_source
    return evidence.after_source


def choose_clear_gold_symbol(changed_symbols: list[dict[str, Any]], patch_text: str) -> dict[str, Any] | None:
    ranked = [sym for sym in changed_symbols if symbol_is_meaningful(sym)]
    if not ranked:
        return None

    patch_lower = patch_text.lower()
    with_patch_support = []
    for sym in ranked:
        leaf = str(sym.get("symbol", "")).lower()
        if leaf and leaf in patch_lower:
            with_patch_support.append(sym)
    if with_patch_support:
        ranked = with_patch_support

    ranked = sorted(ranked, key=symbol_priority)
    if len(ranked) >= 2:
        first_key = symbol_priority(ranked[0])[:3]
        second_key = symbol_priority(ranked[1])[:3]
        if first_key == second_key:
            return None
    return ranked[0]


def extract_span_preview(source: str, lineno: int, end_lineno: int, max_lines: int = 6) -> str:
    lines = source.splitlines()
    if not lines:
        return ""
    start = max(0, lineno - 1)
    end = min(len(lines), end_lineno)
    if end - start > max_lines:
        end = start + max_lines
    return "\n".join(f"{idx + 1}: {lines[idx]}" for idx in range(start, end))


# ── dependency analysis ────────────────────────────────────────────────────────


def module_name_for_file(file_path: str) -> str:
    module = file_path[:-3] if file_path.endswith(".py") else file_path
    if module.endswith("/__init__"):
        module = module[: -len("/__init__")]
    return module.replace("/", ".")


def module_name_candidates(module_name: str) -> list[str]:
    if not module_name:
        return []
    path = module_name.replace(".", "/")
    return [f"{path}.py", f"{path}/__init__.py"]


def resolve_relative_module(file_path: str, module: str | None, level: int) -> str:
    if level == 0:
        return module or ""
    current_parts = module_name_for_file(file_path).split(".")
    if file_path.endswith("/__init__.py"):
        package_parts = current_parts
    else:
        package_parts = current_parts[:-1]
    if level > 0:
        package_parts = package_parts[: max(0, len(package_parts) - (level - 1))]
    suffix = module.split(".") if module else []
    return ".".join(part for part in package_parts + suffix if part)


def build_static_import_dep_map(file_data: dict[str, str]) -> dict[str, list[str]]:
    dep_map: dict[str, list[str]] = {file_path: [] for file_path in file_data}
    module_to_file = {
        module_name_for_file(file_path): file_path
        for file_path in file_data
    }

    for file_path, source in file_data.items():
        try:
            tree = ast.parse(source, filename=file_path)
        except Exception:
            continue
        deps: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    base = alias.name or ""
                    for candidate in module_name_candidates(base):
                        if candidate in file_data and candidate != file_path and candidate not in deps:
                            deps.append(candidate)
                            break
                        resolved = module_to_file.get(base)
                        if resolved and resolved != file_path and resolved not in deps:
                            deps.append(resolved)
            elif isinstance(node, ast.ImportFrom):
                resolved_module = resolve_relative_module(file_path, node.module, node.level)
                for candidate in module_name_candidates(resolved_module):
                    if candidate in file_data and candidate != file_path and candidate not in deps:
                        deps.append(candidate)
                for alias in node.names:
                    symbol_module = ".".join(part for part in [resolved_module, alias.name] if part)
                    for candidate in module_name_candidates(symbol_module):
                        if candidate in file_data and candidate != file_path and candidate not in deps:
                            deps.append(candidate)
        dep_map[file_path] = deps
    return dep_map


def build_dep_map_from_snapshot(file_data: dict[str, str]) -> dict[str, list[str]]:
    try:
        from dep_extractor.analysis_final import DynamicImportAnalyzer  # local import to keep demo/fallback lightweight
    except Exception:
        return build_static_import_dep_map(file_data)

    os.makedirs(VIRTUAL_ROOT, exist_ok=True)
    try:
        analyzer = DynamicImportAnalyzer(project_root=VIRTUAL_ROOT, file_data=file_data)
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            raw = analyzer.run()
        result: dict[str, list[str]] = {}
        for key, deps in raw.items():
            rel_deps = []
            for dep in deps:
                try:
                    rel = str(Path(dep).relative_to(VIRTUAL_ROOT))
                except Exception:
                    continue
                if rel in file_data and rel != key and rel not in rel_deps:
                    rel_deps.append(rel)
            result[key] = rel_deps
        return result or build_static_import_dep_map(file_data)
    finally:
        clear_directory(VIRTUAL_ROOT)


def build_reverse_dep_map(dep_map: dict[str, list[str]]) -> dict[str, list[str]]:
    reverse: dict[str, list[str]] = defaultdict(list)
    for source, deps in dep_map.items():
        for dep in deps:
            if source not in reverse[dep]:
                reverse[dep].append(source)
    return dict(reverse)


def find_dependency_chain(entry: str, targets: set[str], dep_map: dict[str, list[str]]) -> list[str]:
    if entry in targets:
        return [entry]
    queue = deque([[entry]])
    seen = {entry}
    while queue:
        path = queue.popleft()
        node = path[-1]
        for nxt in dep_map.get(node, []):
            if nxt in seen:
                continue
            new_path = path + [nxt]
            if nxt in targets:
                return new_path
            seen.add(nxt)
            queue.append(new_path)
    return []


def choose_dependency_chain_to_gold(
    gold_file: str,
    candidate_entries: list[str],
    dep_map: dict[str, list[str]],
) -> list[str]:
    chains = []
    for entry in dedupe_preserve(candidate_entries):
        if entry == gold_file:
            continue
        chain = find_dependency_chain(entry, {gold_file}, dep_map)
        if len(chain) >= 2:
            chains.append(chain)
    if not chains:
        return []
    chains.sort(key=lambda chain: (-len(chain), chain[0], chain[-1]))
    return chains[0]


def dependency_neighbors(file_path: str, dep_map: dict[str, list[str]], reverse_dep_map: dict[str, list[str]], limit: int = 4) -> list[str]:
    return dedupe_preserve(dep_map.get(file_path, []) + reverse_dep_map.get(file_path, []))[:limit]


def upstream_dependency_candidates(root_file: str, reverse_dep_map: dict[str, list[str]], max_depth: int = 3) -> list[str]:
    discovered: list[str] = []
    queue = deque([(root_file, 0)])
    seen = {root_file}
    while queue:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for parent in reverse_dep_map.get(node, []):
            if parent in seen:
                continue
            seen.add(parent)
            discovered.append(parent)
            queue.append((parent, depth + 1))
    return discovered


# ── ranking / selection helpers ───────────────────────────────────────────────


def same_directory_candidates(file_path: str, pool: list[str], limit: int = 3) -> list[str]:
    parent = os.path.dirname(file_path)
    return [path for path in pool if path != file_path and os.path.dirname(path) == parent][:limit]


def annotate_candidate(
    candidate: dict[str, Any],
    candidate_source: str,
    negative_type: str = "",
    hard_negative_score: float | None = None,
    provenance: str = "",
) -> dict[str, Any]:
    out = dict(candidate)
    out["candidate_source"] = candidate_source
    if negative_type:
        out["negative_type"] = negative_type
    if hard_negative_score is not None:
        out["hard_negative_score"] = round(float(hard_negative_score), 3)
    if provenance:
        out["candidate_provenance"] = provenance
    return out


def baseline_action_candidates() -> list[dict[str, str]]:
    mapping = {
        "global_search": "premature_broad_action",
        "read_full_file": "overread_action",
        "read_full_repo": "overread_action",
    }
    return [
        annotate_candidate({"action_type": name}, "baseline", negative_type=mapping.get(name, "premature_broad_action"), provenance="baseline")
        for name in BASELINE_ACTIONS
    ]


def candidate_identity(candidate: dict[str, Any]) -> str:
    stable_fields = {
        "action_type": candidate.get("action_type"),
        "file_path": candidate.get("file_path"),
        "symbol": candidate.get("symbol"),
        "kind": candidate.get("kind"),
        "span": candidate.get("span"),
    }
    return stable_json(stable_fields)


def dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    output = []
    for item in candidates:
        key = candidate_identity(item)
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
    return output


def candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "action_type": candidate.get("action_type", ""),
        "file_path": candidate.get("file_path", ""),
        "symbol": candidate.get("symbol", ""),
        "kind": candidate.get("kind", ""),
        "span": candidate.get("span", []),
        "candidate_source": candidate.get("candidate_source", "heuristic"),
        "negative_type": candidate.get("negative_type", ""),
        "candidate_provenance": candidate.get("candidate_provenance", ""),
    }


def build_llm_endpoint_cycle() -> list[str]:
    configured = os.environ.get("HARD_NEGATIVE_API_BASES", "").strip()
    if configured:
        return [item.strip().rstrip("/") for item in configured.split(",") if item.strip()]
    single = os.environ.get("HARD_NEGATIVE_API_BASE", DEFAULT_HARD_NEGATIVE_API_BASE).strip().rstrip("/")
    if single and single != DEFAULT_HARD_NEGATIVE_API_BASE:
        return [single]
    return [f"http://127.0.0.1:{port}/v1" for port in range(9003, 9011)]


async def async_post_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any] | None:
    def _request() -> dict[str, Any] | None:
        request = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))

    try:
        return await asyncio.to_thread(_request)
    except Exception:
        return None


async def select_llm_hard_negatives_async(jobs: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    api_bases = build_llm_endpoint_cycle()
    model_name = os.environ.get("HARD_NEGATIVE_MODEL", DEFAULT_HARD_NEGATIVE_MODEL).strip()
    if not api_bases or not model_name or not jobs:
        return [[] for _ in jobs]

    api_key = os.environ.get("HARD_NEGATIVE_API_KEY", os.environ.get("COMMIT_JUDGE_API_KEY", "")).strip()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async def run_one(job: dict[str, Any], idx: int) -> list[dict[str, Any]]:
        candidate_pool = job.get("candidate_pool", [])
        if len(candidate_pool) < 3:
            return []
        prompt = {
            "task": "choose grounded hard negatives for coding-agent pretraining",
            "instruction": (
                "Select 1-3 strongest hard negatives only from the provided grounded candidate pool. "
                "Do not invent files, symbols, or actions. Prefer plausible but wrong near-misses."
            ),
            "task_type": job.get("task_type", "unknown"),
            "gold_action": candidate_summary(job.get("gold_action", {})),
            "input_context": job.get("input_context", {}),
            "negative_type_taxonomy": NEGATIVE_TYPE_TAXONOMY,
            "candidate_pool": [
                {"candidate_id": f"cand_{i}", **candidate_summary(candidate)}
                for i, candidate in enumerate(candidate_pool)
            ],
            "response_schema": {
                "selected": [
                    {
                        "candidate_id": "cand_0",
                        "negative_type": "nearby_but_not_best",
                        "hard_negative_score": 0.87,
                    }
                ]
            },
        }
        payload = {
            "model": model_name,
            "temperature": 0.0,
            "max_tokens": 256,
            "messages": [
                {"role": "system", "content": "Return strict JSON only. Never invent candidates."},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
        }
        endpoint = api_bases[idx % len(api_bases)]
        response = await async_post_json(f"{endpoint}/chat/completions", payload, headers)
        if not response or completion_is_truncated(response):
            return []
        try:
            content = completion_content(response)
            parsed = extract_json_object(content)
        except Exception:
            return []
        if not isinstance(parsed, dict):
            return []
        selected = []
        valid_ids = {f"cand_{i}": candidate for i, candidate in enumerate(candidate_pool)}
        for item in parsed.get("selected", [])[:3]:
            if not isinstance(item, dict):
                continue
            candidate_id = str(item.get("candidate_id", ""))
            negative_type = str(item.get("negative_type", "")).strip()
            if candidate_id not in valid_ids or negative_type not in NEGATIVE_TYPE_TAXONOMY:
                continue
            score = float(item.get("hard_negative_score", 0.0) or 0.0)
            selected.append({
                "candidate_id": candidate_id,
                "negative_type": negative_type,
                "hard_negative_score": max(0.0, min(score, 1.0)),
            })
        return selected

    return await asyncio.gather(*[run_one(job, idx) for idx, job in enumerate(jobs)])


def enrich_distractors_with_llm_hard_negatives(jobs: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    if not jobs:
        return []
    try:
        selections = asyncio.run(select_llm_hard_negatives_async(jobs))
    except Exception:
        selections = [[] for _ in jobs]

    outputs: list[list[dict[str, Any]]] = []
    for job, selected in zip(jobs, selections):
        candidate_pool = dedupe_candidates(job.get("candidate_pool", []))
        selected_map = {item["candidate_id"]: item for item in selected}
        enriched = []
        for idx, candidate in enumerate(candidate_pool):
            item = dict(candidate)
            choice = selected_map.get(f"cand_{idx}")
            if choice:
                item["candidate_source"] = "llm_hard_negative"
                item["negative_type"] = choice["negative_type"]
                item["hard_negative_score"] = round(choice["hard_negative_score"], 3)
            enriched.append(item)
        enriched.sort(
            key=lambda item: (
                0 if item.get("candidate_source") == "llm_hard_negative" else 1,
                -float(item.get("hard_negative_score", 0.0) or 0.0),
                item.get("candidate_provenance", ""),
                candidate_identity(item),
            )
        )
        outputs.append(enriched)
    return outputs


def validate_minimum_file_distractors(distractors: list[dict[str, Any]], gold_file: str) -> bool:
    plausible = [
        item for item in distractors
        if item.get("action_type") == "open_file"
        and item.get("file_path") != gold_file
        and item.get("candidate_source") != "baseline"
    ]
    return len(plausible) >= 2


def build_action_candidates(
    gold_action: dict[str, Any],
    distractors: list[dict[str, Any]],
    max_candidates: int,
) -> tuple[list[dict[str, Any]], int]:
    unique_distractors = [item for item in dedupe_candidates(distractors) if candidate_identity(item) != candidate_identity(gold_action)]
    if len(unique_distractors) > max_candidates - 1:
        unique_distractors = unique_distractors[: max_candidates - 1]
    candidates = unique_distractors + [gold_action]
    RNG.shuffle(candidates)
    gold_key = candidate_identity(gold_action)
    for idx, item in enumerate(candidates):
        if candidate_identity(item) == gold_key:
            return candidates, idx
    candidates.append(gold_action)
    return candidates, len(candidates) - 1


def symbol_to_action(action_type: str, file_path: str, symbol: dict[str, Any]) -> dict[str, Any]:
    return {
        "action_type": action_type,
        "file_path": file_path,
        "symbol": symbol.get("qualname", ""),
        "kind": symbol.get("kind", ""),
        "span": [int(symbol.get("lineno", 0) or 0), int(symbol.get("end_lineno", 0) or 0)],
    }


def region_action(file_path: str, symbol: dict[str, Any], source: str) -> dict[str, Any]:
    return {
        "action_type": "read_region",
        "file_path": file_path,
        "symbol": symbol.get("qualname", ""),
        "kind": symbol.get("kind", ""),
        "span": [int(symbol.get("lineno", 0) or 0), int(symbol.get("end_lineno", 0) or 0)],
        "preview": extract_span_preview(source, int(symbol.get("lineno", 0) or 0), int(symbol.get("end_lineno", 0) or 0)),
    }


def span_action(file_path: str, span: list[int], source: str) -> dict[str, Any]:
    start = int(span[0]) if span else 0
    end = int(span[1]) if len(span) > 1 else start
    return {
        "action_type": "read_region",
        "file_path": file_path,
        "symbol": "",
        "kind": "span",
        "span": [start, end],
        "preview": extract_span_preview(source, start, end),
    }


def nearby_symbol_distractors(gold_symbol: dict[str, Any], symbol_pool: list[dict[str, Any]], limit: int = 4) -> list[dict[str, Any]]:
    gold_span = int(gold_symbol.get("lineno", 0) or 0)
    gold_kind = gold_symbol.get("kind", "")
    ranked = []
    for sym in symbol_pool:
        if sym.get("qualname") == gold_symbol.get("qualname"):
            continue
        same_kind = 0 if sym.get("kind") == gold_kind else 1
        line_distance = abs(int(sym.get("lineno", 0) or 0) - gold_span)
        ranked.append(((same_kind, line_distance, str(sym.get("qualname", ""))), sym))
    ranked.sort(key=lambda item: item[0])
    return [sym for _, sym in ranked[:limit]]


def format_candidate(candidate: dict[str, Any], idx: int) -> str:
    action_type = candidate.get("action_type", "unknown")
    if action_type in {"open_file", "jump_to_file"}:
        suffix = []
        if candidate.get("candidate_source"):
            suffix.append(str(candidate.get("candidate_source")))
        if candidate.get("negative_type"):
            suffix.append(str(candidate.get("negative_type")))
        meta = f" ({', '.join(suffix)})" if suffix else ""
        return f"{idx}. {action_type} -> {candidate.get('file_path', '')}{meta}"
    if action_type in {"open_symbol", "jump_to_symbol", "read_region"}:
        suffix = []
        if candidate.get("candidate_source"):
            suffix.append(str(candidate.get("candidate_source")))
        if candidate.get("negative_type"):
            suffix.append(str(candidate.get("negative_type")))
        meta = f" ({', '.join(suffix)})" if suffix else ""
        return (
            f"{idx}. {action_type} -> {candidate.get('file_path', '')}"
            f" :: {candidate.get('symbol', '')} [{candidate.get('kind', '')}]"
            f" @ {candidate.get('span', [])}{meta}"
        )
    return f"{idx}. {action_type}"


# ── sample validation / QA ────────────────────────────────────────────────────


def commit_skip_reason(
    row: dict[str, Any],
    changed_py_files: list[str],
    before_after: dict[str, dict[str, str]],
    patch_sections: dict[str, dict[str, Any]],
) -> str | None:
    message = normalize_text(row.get("commit_message", ""), 400)
    message_lower = message.lower()
    if len(message) < MIN_COMMIT_MESSAGE_LEN:
        return "short_commit_message"
    if any(token in message_lower for token in ("merge pull request", "merge branch", "merge remote-tracking", "merge commit")):
        return "merge_like_commit"
    # Low-value message-only changes.
    low_value_tokens = (
        "wip", "tmp", "typo", "spelling", "whitespace", "tabs", "tab", "format", "formatting",
        "lint", "ruff", "black", "isort", "prettier", "style", "cleanup", "clean up",
        "reorder imports", "import reorder", "imports reorder",
    )
    if any(tok in message_lower for tok in low_value_tokens):
        return "low_value_commit_message"
    if not changed_py_files:
        return "no_python_changes"
    if not before_after:
        return "missing_before_after"
    if len(changed_py_files) > MAX_CHANGED_PY_FILES:
        return "diffuse_python_change"

    # Drop tests/examples-only churn.
    low_value_path_tokens = ("/tests/", "tests/", "/test_", "/examples/", "examples/", "/demo/", "demo/", "/notebooks/", "notebooks/")
    if all(any(tok in path.lower() for tok in low_value_path_tokens) for path in changed_py_files):
        return "tests_or_examples_only"

    # Drop import-only reorder when it dominates the patch.
    import_like = 0
    total_change = 0
    for path in changed_py_files:
        patch_text = str(patch_sections.get(path, {}).get("text", "") or "")
        for line in patch_text.splitlines():
            if not line.startswith(("+", "-")) or line.startswith(("+++", "---")):
                continue
            content = line[1:].strip()
            if not content:
                continue
            total_change += 1
            if content.startswith(("import ", "from ")):
                import_like += 1
    if total_change >= 6 and import_like / max(1, total_change) >= 0.85:
        return "import_reorder_only"
    covered = sum(1 for file_path in changed_py_files if file_path in before_after)
    if covered / max(1, len(changed_py_files)) < 0.6:
        return "weak_before_after_coverage"
    truncated_sections = [item for item in patch_sections.values() if item.get("is_truncated")]
    if truncated_sections and len(truncated_sections) / max(1, len(patch_sections)) >= 0.5:
        return "patch_truncated"
    return None


def score_file_change(
    file_path: str,
    changed_symbols: list[dict[str, Any]],
    patch_stats: dict[str, Any],
    before_after_present: bool,
    snapshot_present: bool,
) -> int:
    score = 0
    if before_after_present:
        score += 25
    if snapshot_present:
        score += 20
    if patch_stats.get("text"):
        score += 15
    score += min(int(patch_stats.get("changed_lines", 0) or 0), 30)
    score += min(len(changed_symbols), 5) * 15
    if any(sym.get("change_type") == "modified" for sym in changed_symbols):
        score += 20
    if any(sym.get("kind") in {"method", "function"} for sym in changed_symbols):
        score += 10
    if "/tests/" in file_path or file_path.startswith("tests/"):
        score -= 10
    return score


def collect_file_evidence(
    changed_py_files: list[str],
    before_after: dict[str, dict[str, str]],
    repo_snapshot: dict[str, str],
    patch_sections: dict[str, dict[str, Any]],
) -> list[FileChangeEvidence]:
    evidence: list[FileChangeEvidence] = []
    for file_path in changed_py_files:
        versions = before_after.get(file_path, {})
        before_source = versions.get("before", "") or repo_snapshot.get(file_path, "")
        after_source = versions.get("after", "")
        patch_stats = patch_sections.get(file_path, {})
        changed_symbols = diff_symbol_spans(before_source, after_source, file_path) if before_source and after_source else []
        symbol_pool = build_symbol_index(before_source or after_source, file_path)
        changed_line_spans = extract_changed_line_spans_from_patch(str(patch_stats.get("text", "") or ""))
        primary_changed_span = choose_primary_changed_span(changed_line_spans, changed_symbols) if changed_line_spans else None
        evidence.append(FileChangeEvidence(
            file_path=file_path,
            before_source=before_source,
            after_source=after_source,
            patch_text=str(patch_stats.get("text", "") or ""),
            patch_present=bool(patch_stats),
            snapshot_present=file_path in repo_snapshot and bool(repo_snapshot.get(file_path, "")),
            before_after_present=file_path in before_after,
            changed_symbols=changed_symbols,
            symbol_pool=symbol_pool,
            patch_changed_lines=int(patch_stats.get("changed_lines", 0) or 0),
            patch_added_lines=int(patch_stats.get("added_lines", 0) or 0),
            patch_removed_lines=int(patch_stats.get("removed_lines", 0) or 0),
            changed_line_spans=changed_line_spans,
            primary_changed_span=primary_changed_span,
            score=score_file_change(file_path, changed_symbols, patch_stats, file_path in before_after, file_path in repo_snapshot),
        ))
    return evidence


def choose_gold_file_evidence(evidence_list: list[FileChangeEvidence]) -> FileChangeEvidence | None:
    strong = [
        item
        for item in evidence_list
        if item.is_consistent and item.changed_symbols and item.primary_changed_span
    ]
    if not strong:
        return None
    strong.sort(key=lambda item: (-item.score, -len(item.changed_symbols), item.file_path))
    if len(strong) >= 2 and strong[0].score == strong[1].score and len(strong[0].changed_symbols) == len(strong[1].changed_symbols):
        return None
    return strong[0]


def build_changed_symbol_lookup(evidence: FileChangeEvidence) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    before_index = {item["qualname"]: item for item in build_symbol_index(evidence.before_source, evidence.file_path)}
    after_index = {item["qualname"]: item for item in build_symbol_index(evidence.after_source, evidence.file_path)}
    return before_index, after_index


def detect_signature_changes(evidence: FileChangeEvidence) -> list[str]:
    before_index, after_index = build_changed_symbol_lookup(evidence)
    changed = []
    for qualname in sorted(set(before_index) & set(after_index)):
        if before_index[qualname].get("signature", "") != after_index[qualname].get("signature", ""):
            changed.append(qualname)
    return changed


def is_test_or_config_file(file_path: str) -> tuple[bool, bool]:
    lower = file_path.lower()
    is_test = "/tests/" in lower or lower.startswith("tests/") or lower.endswith("_test.py") or lower.startswith("test_")
    is_config = any(token in lower for token in ("config", "settings", "schema", "migration", "yaml", "yml", "toml", "json", ".ini", ".cfg"))
    return is_test, is_config


def is_low_semantic_commit(row: dict[str, Any], patch_text: str, changed_files: list[str]) -> tuple[bool, str]:
    message = normalize_text(row.get("commit_message", ""), 300).lower()
    lowered_patch = str(patch_text or "")

    # Message-level low-value signals.
    low_signal_tokens = (
        "wip", "tmp", "typo", "spelling", "whitespace", "tabs", "tab",
        "style", "format", "formatting", "lint", "ruff", "black", "isort", "cleanup", "clean up",
        "reorder imports", "import reorder",
    )
    if any(token in message for token in low_signal_tokens):
        return True, "low_semantic_message"

    # Docs-only or generated-only churn.
    if changed_files and all(path.endswith((".md", ".rst", ".txt")) for path in changed_files):
        return True, "docs_only"
    if changed_files and all(any(tok in path.lower() for tok in ("lock", "poetry.lock", "package-lock", "pnpm-lock", "yarn.lock")) for path in changed_files):
        return True, "lockfile_only"

    # Patch-level signals.
    import_like = 0
    comment_like = 0
    trivial_like = 0
    total = 0
    for line in lowered_patch.splitlines():
        if not line.startswith(("+", "-")) or line.startswith(("+++", "---")):
            continue
        content = line[1:].rstrip("\n")
        stripped = content.strip()
        if not stripped:
            trivial_like += 1
            continue
        total += 1
        if stripped.startswith(("import ", "from ")):
            import_like += 1
        if stripped.startswith("#"):
            comment_like += 1
        if re.fullmatch(r"[\"'`\[\]{}(),.;:_\-+*/\\\s]+", stripped):
            trivial_like += 1

    if total >= 6 and import_like / max(1, total) >= 0.85:
        return True, "import_reorder_only"
    if total > 0 and (comment_like + trivial_like) / max(1, total) >= 0.9:
        return True, "comment_or_trivial_only"
    return False, ""


def choose_first_read_entry_file(
    change_center_file: str,
    changed_py_files: list[str],
    relevant_files: list[dict[str, Any]],
    dep_map: dict[str, list[str]],
    reverse_dep_map: dict[str, list[str]],
) -> str:
    upstream_changed = [path for path in changed_py_files if path in reverse_dep_map.get(change_center_file, [])]
    if upstream_changed:
        return sorted(upstream_changed)[0]
    relevant_paths = [item["file_path"] for item in sorted(relevant_files, key=lambda item: item["distance"]) if is_python_file(item["file_path"])]
    for path in relevant_paths:
        if path in reverse_dep_map.get(change_center_file, []):
            return path
    return change_center_file


def rank_file_evidence(
    evidence_list: list[FileChangeEvidence],
    dep_map: dict[str, list[str]],
    reverse_dep_map: dict[str, list[str]],
) -> list[FileChangeEvidence]:
    return sorted(
        [item for item in evidence_list if item.is_consistent and item.changed_symbols],
        key=lambda item: (
            -(item.score + len(dep_map.get(item.file_path, [])) * 5 + len(reverse_dep_map.get(item.file_path, [])) * 4),
            -len(item.changed_symbols),
            item.file_path,
        ),
    )


def choose_intent_target_files(
    evidence_list: list[FileChangeEvidence],
    dep_map: dict[str, list[str]],
    reverse_dep_map: dict[str, list[str]],
) -> list[FileChangeEvidence]:
    ranked = rank_file_evidence(evidence_list, dep_map, reverse_dep_map)
    if not ranked:
        return []
    selected = [ranked[0]]
    if len(ranked) >= 2:
        first = ranked[0]
        second = ranked[1]
        related = (
            second.file_path in dep_map.get(first.file_path, [])
            or second.file_path in reverse_dep_map.get(first.file_path, [])
            or os.path.dirname(second.file_path) == os.path.dirname(first.file_path)
        )
        if related and second.score >= max(40, int(first.score * 0.7)):
            selected.append(second)
    return selected


def changed_callsite_lines(patch_text: str) -> int:
    count = 0
    for line in patch_text.splitlines():
        if not line.startswith(("+", "-")) or line.startswith(("+++", "---")):
            continue
        content = line[1:].strip()
        if "(" in content and ")" in content and "def " not in content:
            count += 1
    return count


def classify_patch_type(
    row: dict[str, Any],
    selected_targets: list[FileChangeEvidence],
    changed_py_files: list[str],
) -> tuple[str | None, list[str]]:
    message = normalize_text(row.get("commit_message", ""), 300).lower()
    patch_text = "\n".join(item.patch_text for item in selected_targets if item.patch_text)
    signature_change_count = sum(len(detect_signature_changes(item)) for item in selected_targets)
    condition_lines = sum(
        1
        for line in patch_text.splitlines()
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---")) and re.search(r"\b(if|elif|while|and|or|not)\b", line)
    )
    callsite_lines = changed_callsite_lines(patch_text)
    total_changed_lines = sum(item.patch_changed_lines for item in selected_targets)
    test_flags = [is_test_or_config_file(path)[0] for path in changed_py_files]
    config_flags = [is_test_or_config_file(path)[1] for path in changed_py_files]
    scores: Counter[str] = Counter()

    if test_flags and all(test_flags):
        scores["test_only_adjustment"] += 5
    if signature_change_count:
        scores["signature_change_propagation"] += 5
    if callsite_lines >= 2 and len(changed_py_files) >= 2:
        scores["add_argument_propagation"] += 4
    if condition_lines:
        scores["change_condition_logic"] += 4
    if any(token in patch_text.lower() for token in ("state", "status", "append(", "update(", ".add(", ".remove(", "transition")):
        scores["state_update_fix"] += 4
    if any(config_flags) or any(token in message for token in ("config", "schema", "setting", "migration", "env")):
        scores["config_or_schema_propagation"] += 4
    if "rename" in message and total_changed_lines <= 30:
        scores["rename_only"] += 5
    if any(token in message for token in ("cleanup", "clean up", "refactor", "format", "lint")) and total_changed_lines <= 40:
        scores["cleanup_only"] += 4

    ranked = [item for item in PATCH_TYPE_TAXONOMY if scores[item] > 0]
    if not ranked:
        return None, []
    ranked.sort(key=lambda item: (-scores[item], PATCH_TYPE_TAXONOMY.index(item)))
    if len(ranked) >= 2 and scores[ranked[0]] == scores[ranked[1]]:
        return None, []
    candidates = ranked[:4]
    for item in PATCH_TYPE_TAXONOMY:
        if item not in candidates:
            candidates.append(item)
        if len(candidates) >= 4:
            break
    return ranked[0], candidates


def classify_risk_surface(
    patch_type: str,
    selected_targets: list[FileChangeEvidence],
    changed_py_files: list[str],
    dep_map: dict[str, list[str]],
    reverse_dep_map: dict[str, list[str]],
) -> tuple[str | None, list[str]]:
    scores: Counter[str] = Counter()
    paths = [item.file_path.lower() for item in selected_targets]
    signature_change_count = sum(len(detect_signature_changes(item)) for item in selected_targets)
    has_tests = any(is_test_or_config_file(path)[0] for path in changed_py_files)
    has_config = any(is_test_or_config_file(path)[1] for path in changed_py_files)
    multi_file = len(changed_py_files) >= 2

    if multi_file and any("service" in path for path in paths) and any(token in path for path in paths for token in ("repo", "repository", "dao", "store", "database")):
        scores["service_to_repo_propagation"] += 5
    if signature_change_count and multi_file:
        scores["caller_callee_signature_consistency"] += 5
    if patch_type == "state_update_fix":
        scores["state_transition_consistency"] += 5
    if has_config and multi_file:
        scores["config_consumer_consistency"] += 5
    if has_tests and any(not is_test_or_config_file(path)[0] for path in changed_py_files):
        scores["regression_test_updates"] += 5
    if patch_type in {"rename_only", "cleanup_only"} or (len(selected_targets) == 1 and selected_targets[0].patch_changed_lines <= 25):
        scores["low_risk_local_refactor"] += 4

    for item in selected_targets:
        if dep_map.get(item.file_path) or reverse_dep_map.get(item.file_path):
            if signature_change_count:
                scores["caller_callee_signature_consistency"] += 1
            if has_config:
                scores["config_consumer_consistency"] += 1

    ranked = [item for item in RISK_SURFACE_TAXONOMY if scores[item] > 0]
    if not ranked:
        return None, []
    ranked.sort(key=lambda item: (-scores[item], RISK_SURFACE_TAXONOMY.index(item)))
    if len(ranked) >= 2 and scores[ranked[0]] == scores[ranked[1]]:
        return None, []
    candidates = ranked[:4]
    for item in RISK_SURFACE_TAXONOMY:
        if item not in candidates:
            candidates.append(item)
        if len(candidates) >= 4:
            break
    return ranked[0], candidates


def build_intent_target_file_candidates(
    selected_targets: list[FileChangeEvidence],
    evidence_list: list[FileChangeEvidence],
    repo_snapshot: dict[str, str],
    relevant_files: list[dict[str, Any]],
    dep_map: dict[str, list[str]],
    reverse_dep_map: dict[str, list[str]],
) -> list[dict[str, Any]]:
    anchor_paths = [item.file_path for item in selected_targets]
    ranked_paths = [item.file_path for item in evidence_list]
    relevant_paths = [item["file_path"] for item in sorted(relevant_files, key=lambda item: item["distance"]) if is_python_file(item["file_path"])]
    pool: list[str] = []
    pool.extend(anchor_paths)
    pool.extend(ranked_paths[:4])
    for path in anchor_paths:
        pool.extend(same_directory_candidates(path, list(repo_snapshot.keys()), limit=2))
        pool.extend(dependency_neighbors(path, dep_map, reverse_dep_map, limit=3))
    pool.extend(relevant_paths[:4])
    candidates = [{"file_path": path} for path in dedupe_preserve(pool) if is_python_file(path)]
    return candidates[:6]


def build_intent_root_symbol_candidates(
    primary_target: FileChangeEvidence,
    gold_symbol: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates = [symbol_to_action("open_symbol", primary_target.file_path, gold_symbol)]
    for symbol in nearby_symbol_distractors(gold_symbol, primary_target.symbol_pool, limit=4):
        candidates.append(symbol_to_action("open_symbol", primary_target.file_path, symbol))
    changed_neighbors = [
        symbol_to_action("open_symbol", primary_target.file_path, symbol)
        for symbol in primary_target.changed_symbols
        if symbol.get("qualname") != gold_symbol.get("qualname")
    ]
    candidates.extend(changed_neighbors[:2])
    return dedupe_preserve(candidates)[:6]


# ── commit intent judge ───────────────────────────────────────────────────────


def normalize_patch_line(line: str) -> str:
    return re.sub(r"\s+", "", line.strip())


def changed_patch_body_lines(patch_text: str) -> list[str]:
    return [
        line[1:]
        for line in patch_text.splitlines()
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    ]


def detect_meaningless_reason(
    row: dict[str, Any],
    changed_files: list[str],
    changed_py_files: list[str],
    evidence_list: list[FileChangeEvidence],
    patch_text: str,
) -> tuple[bool, str, str]:
    message = normalize_text(row.get("commit_message", ""), 300).lower()
    body_lines = changed_patch_body_lines(patch_text)
    stripped_lines = [line.strip() for line in body_lines if line.strip()]
    total_changed_lines = len(stripped_lines)

    if any(token in message for token in ("merge pull request", "merge branch", "merge remote-tracking", "merge commit")):
        return True, "merge_commit", "Merge-style commit with low standalone training value."
    if message.startswith("revert") or "this reverts commit" in patch_text.lower():
        return True, "revert_only", "Revert-only commit provides weak standalone reasoning value."
    if changed_files and all(path.endswith(("poetry.lock", "pdm.lock", "requirements.lock", "package-lock.json", "uv.lock")) for path in changed_files):
        return True, "lockfile_only", "Lockfile churn is not useful for navigation-oriented supervision."
    if changed_files and all(path.endswith((".md", ".rst", ".txt")) for path in changed_files):
        return True, "docs_only", "Docs-only commit is not a strong code-navigation signal."
    if changed_files and all(any(token in path.lower() for token in ("generated", "autogen", "dist/", "build/", ".min.")) for path in changed_files):
        return True, "generated_artifact_only", "Generated artifact update is not useful supervision."
    if len(changed_files) >= 12 and not changed_py_files:
        return True, "bulk_mechanical_noise", "Large mechanical change without stable Python reasoning signal."
    if total_changed_lines and all(normalize_patch_line(line).startswith("#") or normalize_patch_line(line).startswith('"""') or normalize_patch_line(line).startswith("'''") for line in stripped_lines):
        return True, "comment_only", "Comment-only edit does not provide stable reasoning content."
    if total_changed_lines and all(normalize_patch_line(line) == "" for line in stripped_lines):
        return True, "formatting_only", "Whitespace-only edit is low-value for training."

    changed_symbol_count = sum(len(item.changed_symbols) for item in evidence_list)
    if total_changed_lines <= 2 and changed_symbol_count == 0:
        return True, "ambiguous_low_signal", "Commit is too small to infer a stable intent."
    if len(changed_files) >= 20 and changed_symbol_count <= 1:
        return True, "bulk_mechanical_noise", "Broad mechanical change with weak semantic anchors."
    return False, "", ""


def build_commit_judge_profile(row: dict[str, Any], initial_df: pd.DataFrame | None) -> CommitJudgeProfile | None:
    changed_files = parse_changed_files(row.get("file_changed_content", ""))
    changed_py_files = [path for path in changed_files if is_python_file(path)]
    before_after = extract_before_after(row)
    repo_snapshot = build_repo_snapshot(initial_df, before_after)
    patch_sections = parse_patch_sections(row.get("patch", ""))
    evidence_list = collect_file_evidence(changed_py_files, before_after, repo_snapshot, patch_sections)
    relevant_files = parse_relevant_files(row.get("relevant_file_content"))
    try:
        dep_map = build_dep_map_from_snapshot(repo_snapshot) if len(repo_snapshot) >= 2 else {}
    except Exception:
        dep_map = {}
    reverse_dep_map = build_reverse_dep_map(dep_map)
    changed_symbols = dedupe_preserve([
        symbol.get("qualname", "")
        for item in evidence_list
        for symbol in item.changed_symbols
        if symbol.get("qualname")
    ])
    top_directories = dedupe_preserve([top_level_directory(path) for path in changed_files if path])
    patch_text = "\n".join(section.get("text", "") for section in patch_sections.values() if section.get("text"))
    patch_changed_lines = sum(int(section.get("changed_lines", 0) or 0) for section in patch_sections.values())

    return CommitJudgeProfile(
        commit_id=str(row.get("commit_id", "")),
        repo=str(row.get("repo_name", "")),
        commit_message=normalize_text(row.get("commit_message", ""), 400),
        changed_files=changed_files,
        changed_py_files=changed_py_files,
        patch_text=patch_text,
        patch_changed_lines=patch_changed_lines,
        evidence_list=evidence_list,
        changed_symbols=changed_symbols,
        top_directories=top_directories,
        dep_map=dep_map,
        reverse_dep_map=reverse_dep_map,
        relevant_files=relevant_files,
    )


def file_jaccard(a: list[str], b: list[str]) -> float:
    left = set(a)
    right = set(b)
    if not left and not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def symbol_jaccard(a: list[str], b: list[str]) -> float:
    left = set(a)
    right = set(b)
    if not left and not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def dependency_cross_signal(left: CommitJudgeProfile, right: CommitJudgeProfile) -> bool:
    left_files = set(left.changed_py_files)
    right_files = set(right.changed_py_files)
    for file_path in left.changed_py_files:
        neighbors = set(left.dep_map.get(file_path, []) + left.reverse_dep_map.get(file_path, []))
        if neighbors & right_files:
            return True
    for file_path in right.changed_py_files:
        neighbors = set(right.dep_map.get(file_path, []) + right.reverse_dep_map.get(file_path, []))
        if neighbors & left_files:
            return True
    return False


def has_followup_message(message: str) -> bool:
    lower = message.lower()
    return any(token in lower for token in ("follow-up", "follow up", "sync", "adjust", "cleanup", "fixup", "propagate", "align", "update tests"))


def infer_pair_intent_label(left: CommitJudgeProfile, right: CommitJudgeProfile) -> str:
    combined_message = f"{left.commit_message} {right.commit_message}".lower()
    left_tests = any(is_test_or_config_file(path)[0] for path in left.changed_files)
    right_tests = any(is_test_or_config_file(path)[0] for path in right.changed_files)
    left_config = any(is_test_or_config_file(path)[1] for path in left.changed_files)
    right_config = any(is_test_or_config_file(path)[1] for path in right.changed_files)
    left_signature = any(detect_signature_changes(item) for item in left.evidence_list)
    right_signature = any(detect_signature_changes(item) for item in right.evidence_list)

    if "rename" in combined_message:
        return "rename_propagation"
    if left_signature or right_signature:
        return "api_signature_propagation"
    if (left_tests or right_tests) and not (left_tests and right_tests):
        return "root_fix_plus_test_followup"
    if left_config or right_config:
        return "config_schema_propagation"
    if has_followup_message(right.commit_message) or has_followup_message(left.commit_message):
        return "local_bugfix_followup"
    if set(left.top_directories) & set(right.top_directories):
        return "same_feature_increment"
    return "unrelated_or_distinct"


def heuristic_pair_merge_score(left: CommitJudgeProfile, right: CommitJudgeProfile) -> tuple[float, dict[str, Any]]:
    file_overlap = file_jaccard(left.changed_files, right.changed_files)
    symbol_overlap = symbol_jaccard(left.changed_symbols, right.changed_symbols)
    directory_overlap = file_jaccard(left.top_directories, right.top_directories)
    dependency_overlap = 1.0 if dependency_cross_signal(left, right) else 0.0
    relevant_overlap = file_jaccard(
        [item["file_path"] for item in left.relevant_files],
        [item["file_path"] for item in right.relevant_files],
    )
    message_followup = 1.0 if has_followup_message(left.commit_message) or has_followup_message(right.commit_message) else 0.0
    score = (
        0.34 * file_overlap
        + 0.26 * symbol_overlap
        + 0.16 * directory_overlap
        + 0.14 * dependency_overlap
        + 0.06 * relevant_overlap
        + 0.04 * message_followup
    )
    return score, {
        "file_overlap": round(file_overlap, 3),
        "symbol_overlap": round(symbol_overlap, 3),
        "directory_overlap": round(directory_overlap, 3),
        "dependency_overlap": round(dependency_overlap, 3),
        "relevant_overlap": round(relevant_overlap, 3),
        "message_followup": round(message_followup, 3),
    }


def make_merge_group_id(repo: str, member_commit_ids: list[str]) -> str:
    seed = "|".join([repo] + sorted(member_commit_ids))
    return f"merge::{hashlib.md5(seed.encode('utf-8')).hexdigest()[:12]}"


def extract_json_object(text: str) -> dict[str, Any] | None:
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def completion_finish_reason(payload: dict[str, Any]) -> str:
    try:
        reason = payload.get("choices", [{}])[0].get("finish_reason", "")
    except Exception:
        return ""
    return str(reason or "").strip().lower()


def completion_content(payload: dict[str, Any]) -> str:
    try:
        return str(payload.get("choices", [{}])[0].get("message", {}).get("content", "") or "")
    except Exception:
        return ""


def completion_is_truncated(payload: dict[str, Any]) -> bool:
    reason = completion_finish_reason(payload)
    if reason in {"length", "max_tokens"}:
        return True
    content = completion_content(payload).strip()
    if not content:
        return True
    return False


def run_llm_commit_judge(
    left: CommitJudgeProfile,
    right: CommitJudgeProfile,
    heuristic_intent: str,
    heuristic_score: float,
    evidence: dict[str, Any],
) -> dict[str, Any] | None:
    api_base = os.environ.get("COMMIT_JUDGE_API_BASE", DEFAULT_COMMIT_JUDGE_API_BASE).strip().rstrip("/")
    model_name = os.environ.get("COMMIT_JUDGE_MODEL", DEFAULT_COMMIT_JUDGE_MODEL).strip()
    if not api_base or not model_name:
        return None

    prompt = {
        "task": "judge commit mergeability for pretraining data construction",
        "instruction": (
            "Given two nearby commits from the same repo, decide whether they share the same underlying intent, "
            "should merge into one training unit, or should remain separate. Be conservative."
        ),
        "heuristic_intent": heuristic_intent,
        "heuristic_score": round(heuristic_score, 3),
        "evidence": evidence,
        "commit_a": {
            "commit_id": left.commit_id,
            "commit_message": left.commit_message,
            "changed_files": left.changed_files[:8],
            "changed_symbols": left.changed_symbols[:8],
        },
        "commit_b": {
            "commit_id": right.commit_id,
            "commit_message": right.commit_message,
            "changed_files": right.changed_files[:8],
            "changed_symbols": right.changed_symbols[:8],
        },
        "response_schema": {
            "intent_label": "one taxonomy label or unrelated_or_distinct",
            "should_merge": True,
            "merge_confidence": 0.0,
            "short_judge_rationale": "brief evidence-based rationale",
        },
    }
    request_payload = {
        "model": model_name,
        "temperature": 0.0,
        "max_tokens": 320,
        "messages": [
            {"role": "system", "content": "Return strict JSON only. Prefer not merging when uncertain."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
    }
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("COMMIT_JUDGE_API_KEY", "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(
        url=f"{api_base}/chat/completions",
        data=json.dumps(request_payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if completion_is_truncated(payload):
            return None
        content = completion_content(payload)
    except Exception:
        return None

    parsed = extract_json_object(content)
    if not isinstance(parsed, dict):
        return None
    return {
        "intent_label": parsed.get("intent_label", heuristic_intent),
        "should_merge": bool(parsed.get("should_merge", False)),
        "merge_confidence": float(parsed.get("merge_confidence", 0.0) or 0.0),
        "short_judge_rationale": normalize_text(parsed.get("short_judge_rationale", ""), 240),
    }


# Thread-safe round-robin counter for root grounding endpoint cycling.
_rg_endpoint_counter: int = 0
_rg_endpoint_lock = __import__("threading").Lock()


def _build_root_grounding_endpoint_cycle() -> list[str]:
    """Return the list of root grounding endpoints to cycle across."""
    configured = os.environ.get("ROOT_GROUNDING_JUDGE_API_BASES", "").strip()
    if configured:
        return [item.strip().rstrip("/") for item in configured.split(",") if item.strip()]
    single = os.environ.get("ROOT_GROUNDING_JUDGE_API_BASE", DEFAULT_ROOT_GROUNDING_JUDGE_API_BASE).strip().rstrip("/")
    # If the env var is explicitly set to a non-default value, honour it as-is.
    if single and single != DEFAULT_ROOT_GROUNDING_JUDGE_API_BASE:
        return [single]
    # Default: round-robin across all 8 vLLM servers.
    return [f"http://127.0.0.1:{port}/v1" for port in range(9003, 9011)]


_RG_ENDPOINTS: list[str] = []  # lazily populated


def _next_rg_endpoint() -> str:
    global _rg_endpoint_counter, _RG_ENDPOINTS
    if not _RG_ENDPOINTS:
        _RG_ENDPOINTS = _build_root_grounding_endpoint_cycle()
    with _rg_endpoint_lock:
        idx = _rg_endpoint_counter
        _rg_endpoint_counter += 1
    return _RG_ENDPOINTS[idx % len(_RG_ENDPOINTS)]


def run_llm_root_grounding_judge(prompt: dict[str, Any]) -> dict[str, Any] | None:
    """Call an LLM judge to select a root grounding decision from candidates."""
    if os.environ.get("ROOT_GROUNDING_JUDGE_DISABLE", "").strip().lower() in {"1", "true", "yes"}:
        return None
    api_base = _next_rg_endpoint()
    model_name = os.environ.get("ROOT_GROUNDING_JUDGE_MODEL", DEFAULT_ROOT_GROUNDING_JUDGE_MODEL).strip()
    if not api_base or not model_name:
        return None
    api_key = os.environ.get("ROOT_GROUNDING_JUDGE_API_KEY", os.environ.get("COMMIT_JUDGE_API_KEY", "")).strip()

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request_payload = {
        "model": model_name,
        "temperature": 0.0,
        "max_tokens": 384,
        "messages": [
            {"role": "system", "content": "Return strict JSON only. Never invent candidates."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
    }

    try:
        request = urllib.request.Request(
            url=f"{api_base}/chat/completions",
            data=json.dumps(request_payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=18) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if completion_is_truncated(payload):
            return None
        content = completion_content(payload)
        parsed = extract_json_object(content)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def run_llm_instruction_selector(prompt: dict[str, Any]) -> dict[str, Any] | None:
    """Select one instruction candidate (id) from a fixed pool."""
    if os.environ.get("TASK_INSTRUCTION_SELECTOR_DISABLE", "").strip().lower() in {"1", "true", "yes"}:
        return None
    # Reuse root grounding judge endpoint/model by default.
    return run_llm_root_grounding_judge(prompt)


def run_llm_sample_quality_judge(prompt: dict[str, Any]) -> dict[str, Any] | None:
    """Final keep/drop/repair judge over a fully built training sample."""
    if os.environ.get("SAMPLE_QUALITY_JUDGE_DISABLE", "").strip().lower() in {"1", "true", "yes"}:
        return None
    api_base = os.environ.get("SAMPLE_QUALITY_JUDGE_API_BASE", DEFAULT_SAMPLE_QUALITY_JUDGE_API_BASE).strip().rstrip("/")
    model_name = os.environ.get("SAMPLE_QUALITY_JUDGE_MODEL", DEFAULT_SAMPLE_QUALITY_JUDGE_MODEL).strip()
    if not api_base or not model_name:
        return None
    api_key = os.environ.get(
        "SAMPLE_QUALITY_JUDGE_API_KEY",
        os.environ.get("ROOT_GROUNDING_JUDGE_API_KEY", os.environ.get("COMMIT_JUDGE_API_KEY", "")),
    ).strip()

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request_payload = {
        "model": model_name,
        "temperature": 0.0,
        "max_tokens": 512,
        "messages": [
            {"role": "system", "content": "Return strict JSON only."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
    }

    try:
        request = urllib.request.Request(
            url=f"{api_base}/chat/completions",
            data=json.dumps(request_payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=18) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if completion_is_truncated(payload):
            return None
        content = completion_content(payload)
        parsed = extract_json_object(content)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def build_gold_grounded_repair(sample: dict[str, Any]) -> dict[str, Any] | None:
    """Repair a sample using gold metadata + observed outline/snippets.

    This is a conservative fallback used when the LLM judge requests repair but
    doesn't return a valid repaired sample.
    """
    if not isinstance(sample, dict):
        return None
    task_type = str(sample.get("task_type", "") or "")
    if task_type not in {"patch_grounding", "ast_dependency_trace"}:
        return None
    meta = sample.get("metadata", {}) if isinstance(sample.get("metadata", {}), dict) else {}
    gold_file = str(meta.get("gold_file", "") or "")
    gold_symbol = str(meta.get("gold_symbol", "") or "")
    gold_span = meta.get("gold_line_span", [])
    if not gold_file or not gold_symbol:
        return None

    repaired = deepcopy(sample)
    target = repaired.get("target", {})
    if not isinstance(target, dict):
        return None
    selected_action = target.get("selected_action")
    if not isinstance(selected_action, dict) or not selected_action:
        return None

    action = str(selected_action.get("action", "") or "")
    input_data = repaired.get("input", {})
    obs = input_data.get("current_observation", {}) if isinstance(input_data, dict) else {}
    if not isinstance(obs, dict):
        obs = {}

    outline: list[dict[str, Any]] = []
    if task_type == "patch_grounding":
        outline = obs.get("file_ast_outline", []) if isinstance(obs.get("file_ast_outline", []), list) else []
    elif task_type == "ast_dependency_trace":
        outline = obs.get("entry_file_ast_outline", []) if isinstance(obs.get("entry_file_ast_outline", []), list) else []

    if action == "open_symbol":
        selected_action = dict(selected_action)
        selected_action["file_path"] = gold_file
        selected_action["symbol"] = gold_symbol
        outline_span = outline_span_for_symbol(outline, gold_symbol)
        if is_valid_span(outline_span):
            selected_action["span"] = [int(outline_span[0]), int(outline_span[1])]
        elif is_valid_span(gold_span):
            selected_action["span"] = [int(gold_span[0]), int(gold_span[1])]
        else:
            return None
        target = dict(target)
        target["selected_action"] = selected_action
        repaired["target"] = target
        return repaired

    # If the sample stopped too early at open_file, upgrade it to open_symbol
    # using the gold symbol/span. This is used as a conservative repair path.
    if action == "open_file":
        selected_action = dict(selected_action)
        selected_action["action"] = "open_symbol"
        selected_action["file_path"] = gold_file
        selected_action["symbol"] = gold_symbol
        outline_span = outline_span_for_symbol(outline, gold_symbol)
        if is_valid_span(outline_span):
            selected_action["span"] = [int(outline_span[0]), int(outline_span[1])]
        elif is_valid_span(gold_span):
            selected_action["span"] = [int(gold_span[0]), int(gold_span[1])]
        else:
            return None
        target = dict(target)
        target["selected_action"] = selected_action
        repaired["target"] = target
        return repaired

    if action == "read_region":
        if not is_valid_span(gold_span):
            return None
        start = int(gold_span[0])
        end = min(int(gold_span[1]), start + 47)
        selected_action = dict(selected_action)
        selected_action["span"] = [start, end]
        target = dict(target)
        target["selected_action"] = selected_action
        repaired["target"] = target
        return repaired

    return None


def judge_sample_with_llm(
    sample: dict[str, Any],
    serialized: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """LLM-based keep/drop/repair judge with gold-grounded repair fallback."""
    meta: dict[str, Any] = {
        "sample_quality_decision": "",
        "sample_quality_confidence": None,
        "sample_quality_reason": "",
        "sample_quality_used_fallback_repair": False,
        "sample_quality_repair_applied": False,
    }

    prompt_text = str(serialized.get("prompt_text", "") or "")
    target_text = str(serialized.get("target_text", "") or "")
    task_type = str(sample.get("task_type", "") or "")
    selected_action = (sample.get("target", {}) or {}).get("selected_action", {})
    if not isinstance(selected_action, dict):
        selected_action = {}

    prompt = {
        "task": "sample_quality_judge",
        "constraints": (
            "Return strict JSON only. Decision must be one of: keep, drop, repair. "
            "If decision=repair, either provide repaired_target (dict) to replace sample.target, "
            "or repaired_sample (full sample dict). Do not invent repo/files/symbols." 
        ),
        "task_schemas": {
            "patch_grounding": {
                "target": {"selected_action": "public_action_payload"},
                "notes": "Target is intentionally whitelisted to selected_action only; do not require gold_* fields.",
            },
            "ast_dependency_trace": {
                "target": {"selected_action": "public_action_payload"},
                "notes": "Target is intentionally whitelisted to selected_action only. Do NOT require extra trace fields beyond what is present in prompt_text.",
            },
            "reading_summary": {
                "target": {"working_summary": "normalized_working_summary"},
                "notes": "Target is intentionally whitelisted to working_summary only; do not require gold_* fields.",
            },
        },
        "hard_drop_rules": [
            "patch_grounding: selected_action.action must not be open_file",
            "open_symbol: span must be a valid [start,end] with start>0 and end>=start",
        ],
        "task_type": task_type,
        "selected_action": selected_action,
        "prompt_text": normalize_text(prompt_text, 2200),
        "target_text": normalize_text(target_text, 1000),
        "metadata": {
            "gold_file": str(sample.get("metadata", {}).get("gold_file", "") or ""),
            "gold_symbol": str(sample.get("metadata", {}).get("gold_symbol", "") or ""),
            "gold_line_span": sample.get("metadata", {}).get("gold_line_span", []),
        },
        "response_schema": {
            "decision": "keep",
            "confidence": 0.8,
            "reason": "short string",
            "repaired_target": None,
            "repaired_sample": None,
        },
    }

    parsed = run_llm_sample_quality_judge(prompt)
    if not isinstance(parsed, dict):
        meta["sample_quality_decision"] = "keep"
        meta["sample_quality_reason"] = "judge_unavailable"
        return sample, meta

    decision = str(parsed.get("decision", parsed.get("action", "")) or "").strip().lower()
    confidence = parsed.get("confidence")
    reason = str(parsed.get("reason", "") or "")
    meta["sample_quality_decision"] = decision
    meta["sample_quality_reason"] = reason
    if isinstance(confidence, (int, float)):
        meta["sample_quality_confidence"] = float(confidence)

    if decision == "keep":
        return sample, meta
    if decision == "drop":
        # The LLM judge can sometimes apply incompatible expectations about
        # downstream schema (e.g., expecting trace fields for ast_dependency_trace
        # even though target is intentionally whitelisted). Prefer conservative
        # gold-grounded repair, and only drop if repair fails.
        fallback = build_gold_grounded_repair(sample)
        if isinstance(fallback, dict) and validate_sample(fallback):
            meta["sample_quality_used_fallback_repair"] = True
            meta["sample_quality_repair_applied"] = True
            # treat as kept after repair
            meta["sample_quality_decision"] = "keep"
            meta["sample_quality_reason"] = reason or "drop_overridden_by_fallback_repair"
            return fallback, meta
        # If no repair is possible, keep samples that are structurally valid
        # for tasks where the judge commonly applies an incompatible schema.
        if task_type in {"ast_dependency_trace", "reading_summary"} and validate_sample(sample):
            meta["sample_quality_decision"] = "keep"
            meta["sample_quality_reason"] = reason or f"drop_overridden_for_{task_type}"
            return sample, meta
        return None, meta
    if decision != "repair":
        # Unknown decision: default to keep.
        meta["sample_quality_decision"] = "keep"
        meta["sample_quality_reason"] = reason or "unknown_decision_default_keep"
        return sample, meta

    # Attempt LLM-provided repair first.
    repaired_sample: dict[str, Any] | None = None
    if isinstance(parsed.get("repaired_sample"), dict):
        repaired_sample = parsed.get("repaired_sample")
    elif isinstance(parsed.get("repaired_target"), dict):
        repaired_sample = deepcopy(sample)
        repaired_sample["target"] = parsed.get("repaired_target")

    if isinstance(repaired_sample, dict) and validate_sample(repaired_sample):
        meta["sample_quality_repair_applied"] = True
        return repaired_sample, meta

    # Fallback: gold-grounded repair.
    fallback = build_gold_grounded_repair(sample)
    if isinstance(fallback, dict) and validate_sample(fallback):
        meta["sample_quality_used_fallback_repair"] = True
        meta["sample_quality_repair_applied"] = True
        return fallback, meta

    # Repair failed.
    meta["sample_quality_reason"] = reason or "repair_failed"
    return None, meta


async def run_llm_root_grounding_judge_async(prompt: dict[str, Any]) -> dict[str, Any] | None:
    """Async twin of run_llm_root_grounding_judge — reuses async_post_json."""
    if os.environ.get("ROOT_GROUNDING_JUDGE_DISABLE", "").strip().lower() in {"1", "true", "yes"}:
        return None
    api_base = _next_rg_endpoint()
    model_name = os.environ.get("ROOT_GROUNDING_JUDGE_MODEL", DEFAULT_ROOT_GROUNDING_JUDGE_MODEL).strip()
    if not api_base or not model_name:
        return None
    api_key = os.environ.get("ROOT_GROUNDING_JUDGE_API_KEY", os.environ.get("COMMIT_JUDGE_API_KEY", "")).strip()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model_name,
        "temperature": 0.0,
        "max_tokens": 384,
        "messages": [
            {"role": "system", "content": "Return strict JSON only. Never invent candidates."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
    }
    response = await async_post_json(f"{api_base}/chat/completions", payload, headers)
    if not response or completion_is_truncated(response):
        return None
    try:
        content = completion_content(response)
        parsed = extract_json_object(content)
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


async def _run_grounding_subtasks_async(
    subtask_prompts: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any] | None]:
    """Fire all sub-task prompts in parallel and return results keyed by name."""
    keys = list(subtask_prompts.keys())
    results = await asyncio.gather(
        *[run_llm_root_grounding_judge_async(subtask_prompts[k]) for k in keys],
    )
    return dict(zip(keys, results))


def run_grounding_subtasks_batch(
    subtask_prompts: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any] | None]:
    """Synchronous entry point: run all sub-task LLM calls in parallel."""
    if not subtask_prompts:
        return {}
    try:
        return asyncio.run(_run_grounding_subtasks_async(subtask_prompts))
    except Exception:
        return {k: None for k in subtask_prompts}


def generate_working_summary_with_llm(
    goal_text: str,
    read_history: list[dict[str, Any]],
    allowed_files: list[str],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    meta = {
        "working_summary_source": "",
        "used_working_summary_fallback": False,
        "working_summary_drop_reason": "",
        "next_question_repaired": False,
    }

    fallback = deterministic_working_summary(goal_text, read_history)
    observed_files = dedupe_preserve([
        str(step.get("observation", {}).get("file_path", "") or "")
        for step in read_history
        if isinstance(step, dict) and isinstance(step.get("observation", {}), dict)
    ])
    observed_files = [p for p in observed_files if p]
    allowed_file_set = set([p for p in allowed_files if p]) if allowed_files else set(observed_files)
    supportable_files = [p for p in observed_files if p in allowed_file_set]

    prompt = {
        "task": "generate_working_summary",
        "constraints": (
            "Return strict JSON only. Do not invent files outside allowed_files. "
            "Compress only the observed read_history. likely_root_file and supporting_files must come from observed files."
        ),
        "goal_text": normalize_text(goal_text, 300),
        "allowed_files": supportable_files or observed_files,
        "read_history": read_history,
        "response_schema": {
            "likely_root_file": (fallback or {}).get("likely_root_file", observed_files[0] if observed_files else ""),
            "likely_focus_region": (fallback or {}).get("likely_focus_region", [20, 28]),
            "supporting_files": (fallback or {}).get("supporting_files", supportable_files[:1]),
            "next_question": (fallback or {}).get(
                "next_question",
                "Which observed callsite or dependency should be verified next?",
            ),
        },
    }

    if os.environ.get("WORKING_SUMMARY_DISABLE", "").strip().lower() in {"1", "true", "yes"}:
        parsed = None
    else:
        parsed = run_llm_root_grounding_judge(prompt)

    if not isinstance(parsed, dict):
        if fallback is None:
            meta["working_summary_drop_reason"] = "llm_failed_and_no_grounded_fallback"
            return None, meta
        meta["working_summary_source"] = "fallback"
        meta["used_working_summary_fallback"] = True
        return fallback, meta

    ws = parsed
    if not isinstance(ws.get("likely_root_file"), str) or not ws.get("likely_root_file"):
        meta["working_summary_drop_reason"] = "invalid_likely_root_file"
        return None, meta
    if ws.get("likely_root_file") not in set(observed_files):
        meta["working_summary_drop_reason"] = "invented_file"
        return None, meta

    region = ws.get("likely_focus_region")
    if not (isinstance(region, list) and len(region) == 2 and all(isinstance(x, int) and x > 0 for x in region)):
        meta["working_summary_drop_reason"] = "invalid_focus_region"
        return None, meta
    # Reject placeholder focus region.
    if is_placeholder_focus_region(region, read_history):
        meta["working_summary_drop_reason"] = "placeholder_focus_region"
        return None, meta

    supporting = ws.get("supporting_files", [])
    if not (isinstance(supporting, list) and all(isinstance(x, str) and x for x in supporting)):
        meta["working_summary_drop_reason"] = "invalid_supporting_files"
        return None, meta
    if any(x not in set(observed_files) for x in supporting):
        meta["working_summary_drop_reason"] = "invented_supporting_file"
        return None, meta
    # Repair empty supporting_files deterministically from observed files (excluding root).
    if not supporting:
        repair = [f for f in observed_files if f != ws.get("likely_root_file")][:2]
        if not repair:
            meta["working_summary_drop_reason"] = "empty_supporting_files_no_repair"
            return None, meta
        ws = dict(ws)
        ws["supporting_files"] = repair

    if not isinstance(ws.get("next_question"), str) or not ws.get("next_question").strip():
        meta["working_summary_drop_reason"] = "missing_next_question"
        return None, meta

    next_question = str(ws.get("next_question", "") or "")
    needs_repair = (
        is_generic_next_question(next_question)
        or not next_question_mentions_observed_evidence(next_question, read_history)
        or not next_question_grounded_in_supporting_context(next_question, read_history, ws.get("supporting_files", []))
        or not next_question_has_specific_anchor(next_question, read_history, str(ws.get("likely_root_file", "") or ""))
    )
    if needs_repair:
        repaired_question = build_specific_next_question(
            goal_text=goal_text,
            read_history=read_history,
            likely_root_file=str(ws.get("likely_root_file", "") or ""),
            root_focus_region=ws.get("likely_focus_region"),
            supporting_files=ws.get("supporting_files", []),
        )
        if repaired_question:
            ws = dict(ws)
            ws["next_question"] = repaired_question
            meta["next_question_repaired"] = True
            next_question = repaired_question
        if (
            not repaired_question
            or is_generic_next_question(next_question)
            or not next_question_mentions_observed_evidence(next_question, read_history)
            or not next_question_grounded_in_supporting_context(next_question, read_history, ws.get("supporting_files", []))
            or not next_question_has_specific_anchor(next_question, read_history, str(ws.get("likely_root_file", "") or ""))
        ):
            meta["working_summary_drop_reason"] = "non_specific_next_question"
            return None, meta

    meta["working_summary_source"] = "llm_repaired_with_fallback" if meta["next_question_repaired"] else "llm"
    return {
        "likely_root_file": ws.get("likely_root_file"),
        "likely_focus_region": ws.get("likely_focus_region"),
        "supporting_files": ws.get("supporting_files"),
        "next_question": ws.get("next_question"),
    }, meta


def build_task_instruction_candidates(task_type: str, subtask: str) -> list[str]:
    task_type = str(task_type or "")
    subtask = str(subtask or "")
    if task_type == "patch_grounding" and subtask == "choose_first_file":
        return [
            "Choose the first file to inspect for this change.",
            "Find the most likely file to open first for this change.",
            "Locate the primary file to inspect for this change.",
        ]
    if task_type == "patch_grounding" and subtask == "narrow_patch_region":
        return [
            "Narrow down the patch landing region for this change.",
            "Identify the most relevant region to inspect in the current file.",
            "Find the local code region most likely touched by this change.",
        ]
    if task_type == "ast_dependency_trace" and subtask == "follow_dependency":
        return [
            "Follow the most likely dependency path for this change.",
            "Choose the next file or symbol to trace for this change.",
            "Continue tracing the implementation path for this change.",
        ]
    if task_type == "ast_dependency_trace" and subtask == "stop_and_summarize":
        return [
            "Decide whether tracing should stop here and be summarized.",
            "Check if the likely implementation site has already been identified.",
            "Determine whether to stop tracing and summarize the current state.",
        ]
    if task_type == "reading_summary" and subtask == "compress_working_state":
        return [
            "Summarize the current code-reading state for this change.",
            "Compress the current reading progress into a working summary.",
            "Write a compact working memory summary for the current investigation.",
        ]
    return ["Choose the next best action for this change."]


def _build_action_selection_prompt(
    task_type: str,
    subtask: str,
    goal_text: str,
    action_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "task": "select_next_action",
        "constraints": "Choose exactly one candidate action by id. Never invent new files, symbols, spans, or actions.",
        "task_type": task_type,
        "subtask": subtask,
        "goal_text": normalize_text(goal_text, 400),
        "action_candidates": [
            {"id": i, **strip_action_meta(c)}
            for i, c in enumerate(action_candidates)
        ],
        "response_schema": {
            "decision": "keep|drop",
            "selected_action_id": 0,
            "confidence": 0.75,
            "issues": [],
            "reason": "...",
        },
    }


def _build_instruction_selection_prompt(
    task_type: str,
    subtask: str,
    goal_text: str,
    instruction_candidates: list[str],
) -> dict[str, Any]:
    return {
        "task": "select_task_instruction",
        "constraints": "Pick exactly one candidate by id. Do not invent new instructions.",
        "task_type": task_type,
        "subtask": subtask,
        "goal_text": normalize_text(goal_text, 300),
        "instruction_candidates": [
            {"id": i, "text": s}
            for i, s in enumerate(instruction_candidates)
        ],
        "response_schema": {"selected_instruction_id": 0, "confidence": 0.8},
    }


def _parse_action_selection_result(
    parsed: dict[str, Any] | None,
    action_candidates: list[dict[str, Any]],
    task_type: str = "",
) -> tuple[int | None, dict[str, Any] | None, dict[str, Any]]:
    """Parse a pre-fetched LLM response for action selection."""
    meta: dict[str, Any] = {
        "selected_action_source": "",
        "used_action_selection_fallback": False,
        "next_action_confidence": None,
        "next_action_drop_reason": "",
        "next_action_reason": "",
        "next_action_issues": [],
    }
    if parsed is None:
        if not strong_action_heuristic_ok(action_candidates, task_type=task_type):
            meta["next_action_drop_reason"] = "llm_call_failed_no_strong_fallback"
            return None, None, meta
        meta["selected_action_source"] = "fallback"
        meta["used_action_selection_fallback"] = True
        return 0, action_candidates[0], meta

    decision = str(parsed.get("decision", "")).strip().lower()
    confidence = float(parsed.get("confidence", 0.0) or 0.0)
    meta["next_action_confidence"] = confidence
    reason_text = str(parsed.get("reason", "") or "")
    meta["next_action_reason"] = reason_text
    meta["next_action_issues"] = parsed.get("issues", []) if isinstance(parsed.get("issues", []), list) else []

    if decision != "keep":
        meta["next_action_drop_reason"] = "llm_drop"
        return None, None, meta
    if confidence < float(os.environ.get("NEXT_ACTION_MIN_CONFIDENCE", str(NEXT_ACTION_MIN_CONFIDENCE)) or NEXT_ACTION_MIN_CONFIDENCE):
        meta["next_action_drop_reason"] = "llm_low_confidence"
        return None, None, meta
    # Reject generic / templatic reasons — they indicate the LLM is not grounded.
    if is_generic_action_reason(reason_text):
        meta["next_action_drop_reason"] = "generic_action_reason"
        return None, None, meta
    try:
        best_id = int(parsed.get("selected_action_id"))
    except Exception:
        meta["next_action_drop_reason"] = "llm_invalid_id"
        return None, None, meta
    if not (0 <= best_id < len(action_candidates)):
        meta["next_action_drop_reason"] = "llm_invalid_id"
        return None, None, meta

    meta["selected_action_source"] = "llm"
    return best_id, action_candidates[best_id], meta


def _parse_instruction_selection_result(
    parsed: dict[str, Any] | None,
    instruction_candidates: list[str],
) -> tuple[str, bool, str]:
    """Parse a pre-fetched LLM response for instruction selection."""
    if not isinstance(parsed, dict):
        return instruction_candidates[0] if instruction_candidates else "", True, "fallback"
    try:
        best_id = int(parsed.get("selected_instruction_id", 0))
    except Exception:
        return instruction_candidates[0] if instruction_candidates else "", True, "fallback"
    if not (0 <= best_id < len(instruction_candidates)):
        return instruction_candidates[0] if instruction_candidates else "", True, "fallback"
    try:
        conf = float(parsed.get("confidence", 0.0) or 0.0)
    except Exception:
        conf = 0.0
    if conf < 0.55:
        return instruction_candidates[0] if instruction_candidates else "", True, "fallback"
    return instruction_candidates[best_id], False, "llm"


def select_task_instruction_with_llm(
    task_type: str,
    subtask: str,
    goal_text: str,
    instruction_candidates: list[str],
) -> tuple[str, bool, str]:
    """Select one task instruction from candidates.

    Returns: (selected_instruction, used_fallback, source)
    where source in {"llm", "fallback"}.
    """
    if not instruction_candidates:
        return "", True, "fallback"
    prompt = _build_instruction_selection_prompt(task_type, subtask, goal_text, instruction_candidates)
    parsed = run_llm_instruction_selector(prompt)
    return _parse_instruction_selection_result(parsed, instruction_candidates)


def select_root_grounding_with_llm(
    row: dict[str, Any],
    root_file_candidates: list[dict[str, Any]],
    root_symbol_candidates: list[dict[str, Any]],
    root_span_candidates: list[dict[str, Any]],
    related_snippet_candidates: list[dict[str, Any]],
    dep_context: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    meta: dict[str, Any] = {
        "llm_root_grounding_decision": "",
        "llm_root_grounding_confidence": None,
        "llm_root_grounding_reason": "",
        "llm_root_grounding_issues": [],
        "used_root_grounding_fallback": False,
        "root_grounding_drop_reason": "",
    }

    if not root_file_candidates or not root_span_candidates or not related_snippet_candidates:
        meta["llm_root_grounding_decision"] = "drop"
        meta["root_grounding_drop_reason"] = "no_candidates"
        return None, meta

    # Drop if top-1 and top-2 file candidates are too close — ambiguous grounding.
    if len(root_file_candidates) >= 2:
        top1_score = float(root_file_candidates[0].get("heuristic_score", 0.0) or 0.0)
        top2_score = float(root_file_candidates[1].get("heuristic_score", 0.0) or 0.0)
        if (top1_score - top2_score) < 3.0:
            meta["root_grounding_drop_reason"] = "ambiguous_file_candidates_gap_too_small"
            return None, meta

    prompt = {
        "task": "root_grounding_selection",
        "rubric": (
            "Pick the single best root grounding from the provided candidates only. "
            "Never invent files, symbols, spans, or snippets."
        ),
        "commit_message": normalize_text(row.get("commit_message", ""), 500),
        "dependency_context": dep_context,
        "root_file_candidates": [
            {"id": i, "file_path": c.get("file_path", ""), "heuristic_score": c.get("heuristic_score", 0.0), "reason_tags": c.get("reason_tags", [])}
            for i, c in enumerate(root_file_candidates)
        ],
        "root_symbol_candidates": [
            {"id": i, "file_id": c.get("file_id", -1), "symbol": c.get("symbol", ""), "kind": c.get("kind", ""), "span": c.get("span", []), "heuristic_score": c.get("heuristic_score", 0.0), "reason_tags": c.get("reason_tags", [])}
            for i, c in enumerate(root_symbol_candidates)
        ],
        "root_span_candidates": [
            {"id": i, "file_id": c.get("file_id", -1), "span": c.get("span", []), "source": c.get("source", ""), "heuristic_score": c.get("heuristic_score", 0.0)}
            for i, c in enumerate(root_span_candidates)
        ],
        "related_snippet_candidates": [
            {"id": i, "file_path": c.get("file_path", ""), "provenance": c.get("provenance", ""), "noise_flags": c.get("noise_flags", []), "heuristic_score": c.get("heuristic_score", 0.0)}
            for i, c in enumerate(related_snippet_candidates)
        ],
        "response_schema": {
            "decision": "keep|drop",
            "best_root_file_id": 0,
            "best_root_symbol_id": 0,
            "best_root_span_id": 0,
            "keep_related_snippet_ids": [0, 1],
            "confidence": 0.8,
            "issues": [],
            "reason": "...",
        },
    }

    parsed = run_llm_root_grounding_judge(prompt)

    # fallback only if heuristic is very strong
    if parsed is None:
        if not strong_root_grounding_heuristic_ok(root_file_candidates):
            meta["root_grounding_drop_reason"] = "llm_call_failed_no_strong_fallback"
            return None, meta
        meta["used_root_grounding_fallback"] = True
        meta["llm_root_grounding_decision"] = "keep"
        meta["llm_root_grounding_reason"] = "fallback_to_top_heuristic_candidates"
        selection = {
            "best_root_file_id": 0,
            "best_root_symbol_id": 0 if root_symbol_candidates else -1,
            "best_root_span_id": 0,
            "keep_related_snippet_ids": list(range(min(2, len(related_snippet_candidates)))),
            "confidence": None,
            "issues": ["used_fallback"],
        }
        return selection, meta

    decision = str(parsed.get("decision", "")).strip().lower()
    confidence = float(parsed.get("confidence", 0.0) or 0.0)
    meta["llm_root_grounding_decision"] = decision
    meta["llm_root_grounding_confidence"] = confidence
    meta["llm_root_grounding_reason"] = str(parsed.get("reason", "") or "")
    meta["llm_root_grounding_issues"] = parsed.get("issues", []) if isinstance(parsed.get("issues", []), list) else []

    if decision != "keep":
        meta["root_grounding_drop_reason"] = "llm_drop"
        return None, meta
    if confidence < float(os.environ.get("ROOT_GROUNDING_MIN_CONFIDENCE", str(ROOT_GROUNDING_MIN_CONFIDENCE)) or ROOT_GROUNDING_MIN_CONFIDENCE):
        meta["root_grounding_drop_reason"] = "llm_low_confidence"
        return None, meta

    try:
        best_file_id = int(parsed.get("best_root_file_id"))
        best_symbol_id = int(parsed.get("best_root_symbol_id", 0))
        best_span_id = int(parsed.get("best_root_span_id"))
    except Exception:
        meta["root_grounding_drop_reason"] = "llm_invalid_ids"
        return None, meta

    if not (0 <= best_file_id < len(root_file_candidates)):
        meta["root_grounding_drop_reason"] = "llm_invalid_file_id"
        return None, meta
    if root_symbol_candidates and not (0 <= best_symbol_id < len(root_symbol_candidates)):
        meta["root_grounding_drop_reason"] = "llm_invalid_symbol_id"
        return None, meta
    if not (0 <= best_span_id < len(root_span_candidates)):
        meta["root_grounding_drop_reason"] = "llm_invalid_span_id"
        return None, meta

    # Post-selection consistency checks.
    chosen_file_tags = root_file_candidates[best_file_id].get("reason_tags", [])
    chosen_file_score = float(root_file_candidates[best_file_id].get("heuristic_score", 0.0) or 0.0)

    # Drop if LLM chose an aux-like file and the gap to top-2 is still small.
    if has_aux_tags(root_file_candidates[best_file_id]) and len(root_file_candidates) >= 2:
        runner_up_score = float(root_file_candidates[1 if best_file_id == 0 else 0].get("heuristic_score", 0.0) or 0.0)
        if (chosen_file_score - runner_up_score) < 6.0:
            meta["root_grounding_drop_reason"] = "aux_file_chosen_with_small_gap"
            return None, meta

    # Symbol must belong to the chosen file.
    if root_symbol_candidates and 0 <= best_symbol_id < len(root_symbol_candidates):
        sym_file_id = int(root_symbol_candidates[best_symbol_id].get("file_id", -1))
        if sym_file_id != best_file_id:
            meta["root_grounding_drop_reason"] = "symbol_file_mismatch"
            return None, meta
        # Symbol span must overlap chosen span (if both are non-trivial).
        sym_span = root_symbol_candidates[best_symbol_id].get("span", [0, 0])
        chosen_span = root_span_candidates[best_span_id].get("span", [0, 0]) if 0 <= best_span_id < len(root_span_candidates) else [0, 0]
        if (sym_span[0] > 0 and sym_span[1] > 0 and chosen_span[0] > 0 and chosen_span[1] > 0):
            overlap = max(0, min(int(sym_span[1]), int(chosen_span[1])) - max(int(sym_span[0]), int(chosen_span[0])) + 1)
            if overlap == 0:
                meta["root_grounding_drop_reason"] = "symbol_span_no_overlap"
                return None, meta

    # Span must belong to the chosen file.
    if 0 <= best_span_id < len(root_span_candidates):
        span_file_id = int(root_span_candidates[best_span_id].get("file_id", -1))
        if span_file_id != best_file_id:
            meta["root_grounding_drop_reason"] = "span_file_mismatch"
            return None, meta
        # Reject span if preview is empty/near-empty or span is too large.
        chosen_span_obj = root_span_candidates[best_span_id]
        span_preview = str(chosen_span_obj.get("preview", "") or "")
        if len(span_preview.strip()) < 30:
            meta["root_grounding_drop_reason"] = "span_preview_empty"
            return None, meta
        chosen_span_range = chosen_span_obj.get("span", [0, 0])
        if (isinstance(chosen_span_range, list) and len(chosen_span_range) == 2
                and chosen_span_range[0] > 0 and chosen_span_range[1] > 0):
            span_len = int(chosen_span_range[1]) - int(chosen_span_range[0]) + 1
            if span_len > MAX_ALLOWED_SPAN_LINES:
                meta["root_grounding_drop_reason"] = "span_too_large"
                return None, meta

    # Require at least 1 clean related snippet; if we are below the preferred
    # clean-snippet count, trim to the clean subset instead of hard-dropping.
    keep_ids_raw = parsed.get("keep_related_snippet_ids", [])
    if not isinstance(keep_ids_raw, list):
        keep_ids_raw = []
    keep_related_ids = []
    for item in keep_ids_raw:
        try:
            idx = int(item)
        except Exception:
            continue
        if 0 <= idx < len(related_snippet_candidates):
            keep_related_ids.append(idx)
    keep_related_ids = dedupe_preserve(keep_related_ids)[:3]
    if not keep_related_ids:
        meta["root_grounding_drop_reason"] = "llm_selected_no_related"
        return None, meta

    # Require at least 1 non-noisy related snippet; prefer 2 but don't hard-drop
    # when only 1 is available (small repos often have only 1 clean related file).
    selected_snippets = [related_snippet_candidates[i] for i in keep_related_ids]
    non_noisy = [s for s in selected_snippets if not s.get("noise_flags")]
    if not non_noisy:
        # All selected snippets are noisy — no clean context at all.
        meta["root_grounding_drop_reason"] = "all_related_snippets_noisy"
        return None, meta
    # If we have fewer than NON_NOISY_RELATED_MIN, trim keep_ids to only non-noisy ones.
    if len(non_noisy) < NON_NOISY_RELATED_MIN:
        non_noisy_ids = [keep_related_ids[i] for i, s in enumerate(selected_snippets) if not s.get("noise_flags")]
        keep_related_ids = non_noisy_ids

    return {
        "best_root_file_id": best_file_id,
        "best_root_symbol_id": best_symbol_id,
        "best_root_span_id": best_span_id,
        "keep_related_snippet_ids": keep_related_ids,
        "confidence": confidence,
        "issues": meta["llm_root_grounding_issues"],
    }, meta


def run_llm_next_action_selector(prompt: dict[str, Any]) -> dict[str, Any] | None:
    """Select a next action from a fixed action candidate pool."""
    if os.environ.get("NEXT_ACTION_SELECTOR_DISABLE", "").strip().lower() in {"1", "true", "yes"}:
        return None
    return run_llm_root_grounding_judge(prompt)


def select_next_action_with_llm(
    task_type: str,
    subtask: str,
    goal_text: str,
    action_candidates: list[dict[str, Any]],
) -> tuple[int | None, dict[str, Any] | None, dict[str, Any]]:
    if not action_candidates:
        return None, None, {"selected_action_source": "", "used_action_selection_fallback": False,
                            "next_action_confidence": None, "next_action_drop_reason": "no_action_candidates",
                            "next_action_reason": "", "next_action_issues": []}
    prompt = _build_action_selection_prompt(task_type, subtask, goal_text, action_candidates)
    parsed = run_llm_next_action_selector(prompt)
    return _parse_action_selection_result(parsed, action_candidates)


def build_available_actions_for_patch_grounding(
    root_file: str,
    root_symbol: str,
    root_line_span: list[int],
    file_ast_outline: list[dict[str, Any]],
    related_snippets: list[dict[str, Any]],
    source: str | None = None,
    symbol_candidates: list[dict[str, Any]] | None = None,
    issues: list[str] | None = None,
) -> list[dict[str, Any]]:
    issues = issues or []
    actions: list[dict[str, Any]] = []
    grounded_symbol_candidates = filter_symbol_candidates_for_file(symbol_candidates, root_file)
    root_sym_span = None

    if root_symbol and "root_symbol_uncertain" not in set(issues):
        root_sym_span = resolve_symbol_span(
            root_symbol,
            source=source,
            file_path=root_file,
            symbol_candidates=grounded_symbol_candidates,
            outline=file_ast_outline,
        )
        actions.append({
            **action_open_symbol(root_file, root_symbol, span=root_sym_span),
            "candidate_score": 30.0,
            "reason_tags": ["best_symbol"],
        })

    if isinstance(root_line_span, list) and len(root_line_span) == 2 and all(isinstance(x, int) and x > 0 for x in root_line_span):
        read_region_span = clip_span_to_max_width(root_line_span, 48)
        if root_sym_span and is_valid_span(root_sym_span) and span_length(root_line_span) > 48:
            read_region_span = None
        if is_valid_span(read_region_span):
            width_penalty = max(0.0, (span_length(read_region_span) - 24) / 8.0)
            actions.append({
                **action_read_region(root_file, read_region_span),
                "candidate_score": 24.0 - width_penalty,
                "reason_tags": ["best_region"],
            })

    target_center = None
    if is_valid_span(root_sym_span):
        target_center = (int(root_sym_span[0]) + int(root_sym_span[1])) / 2.0
    elif is_valid_span(root_line_span):
        target_center = (int(root_line_span[0]) + int(root_line_span[1])) / 2.0

    nearby_candidates: list[tuple[float, str, list[int] | None]] = []
    seen_nearby = {root_symbol} if root_symbol else set()

    def add_nearby_candidate(qualname: str, nearby_span: list[int] | None) -> None:
        qn = str(qualname or "")
        if not qn or qn in seen_nearby:
            return
        if not is_valid_span(nearby_span):
            return
        if is_valid_span(root_line_span) and spans_overlap(nearby_span, root_line_span):
            return
        seen_nearby.add(qn)
        if target_center is not None:
            center = (int(nearby_span[0]) + int(nearby_span[1])) / 2.0
            distance = abs(center - target_center)
        else:
            distance = float(int(nearby_span[0]))
        nearby_candidates.append((distance, qn, nearby_span))

    for item in grounded_symbol_candidates:
        qn = str(item.get("qualname", item.get("symbol", "")) or "")
        nearby_span = resolve_symbol_span(
            qn,
            source=source,
            file_path=root_file,
            symbol_candidates=grounded_symbol_candidates,
            outline=file_ast_outline,
        )
        add_nearby_candidate(qn, nearby_span)
    for item in file_ast_outline[:12]:
        qn = str(item.get("qualname", "") or "")
        nearby_span = _span_from_outline(qn, file_ast_outline)
        add_nearby_candidate(qn, nearby_span)

    nearby_added = 0
    for _, qn, nearby_span in sorted(nearby_candidates, key=lambda item: (item[0], item[1])):
        actions.append({
            **action_open_symbol(root_file, qn, span=nearby_span),
            "candidate_score": 18.0 - nearby_added,
            "reason_tags": ["nearby_symbol"],
        })
        nearby_added += 1
        if nearby_added >= 2:
            break

    actions.append({
        **action_open_file(root_file),
        "candidate_score": 14.0,
        "reason_tags": ["stay_in_root_file"],
    })

    for item in related_snippets[:1]:
        fp = str(item.get("file_path", "") or "")
        if fp and fp != root_file:
            actions.append({
                **action_open_file(fp),
                "candidate_score": 8.0,
                "reason_tags": ["related_file"],
            })

    actions = dedupe_actions(actions)
    actions.sort(key=lambda x: (-float(x.get("candidate_score", 0.0)), action_identity(x)))
    return actions[:6]


def build_available_actions_for_trace(
    next_file: str | None,
    root_file: str,
    next_file_source: str | None = None,
    next_file_outline: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []

    if next_file:
        actions.append({
            **action_follow_dependency(next_file),
            "candidate_score": 30.0,
            "reason_tags": ["best_follow_dependency"],
        })
        actions.append({
            **action_open_file(next_file),
            "candidate_score": 24.0,
            "reason_tags": ["inspect_next_file"],
        })

        visible_symbol = ""
        if next_file_source:
            symbol_index = build_symbol_index(next_file_source, next_file)
            if symbol_index:
                visible_symbol = str(symbol_index[0].get("qualname", "") or "")
        if not visible_symbol:
            visible_symbol = first_visible_symbol_from_outline(next_file_outline or [])
        if visible_symbol:
            visible_sym_span = resolve_symbol_span(
                visible_symbol,
                source=next_file_source,
                file_path=next_file,
                outline=next_file_outline or [],
            )
            actions.append({
                **action_open_symbol(next_file, visible_symbol, span=visible_sym_span),
                "candidate_score": 18.0,
                "reason_tags": ["visible_symbol"],
            })

    if root_file and root_file != next_file:
        actions.append({
            **action_open_file(root_file),
            "candidate_score": 12.0,
            "reason_tags": ["jump_to_root_file"],
        })

    actions.append({
        **action_stop_and_summarize(),
        "candidate_score": 10.0,
        "reason_tags": ["stop_if_enough_signal"],
    })

    actions = dedupe_actions(actions)
    actions.sort(key=lambda x: (-float(x.get("candidate_score", 0.0)), action_identity(x)))
    return actions[:6]


def judge_commit_pair(left: CommitJudgeProfile, right: CommitJudgeProfile) -> dict[str, Any] | None:
    heuristic_score, evidence = heuristic_pair_merge_score(left, right)
    if heuristic_score < 0.35:
        return None
    heuristic_intent = infer_pair_intent_label(left, right)
    llm_verdict = run_llm_commit_judge(left, right, heuristic_intent, heuristic_score, evidence)
    if llm_verdict is not None:
        verdict = dict(llm_verdict)
        verdict["judge_source"] = "heuristic_plus_llm"
        verdict["heuristic_score"] = round(heuristic_score, 3)
        verdict["evidence"] = evidence
        return verdict

    should_merge = heuristic_score >= 0.58 and heuristic_intent != "unrelated_or_distinct"
    return {
        "intent_label": heuristic_intent,
        "should_merge": should_merge,
        "merge_confidence": round(min(0.95, heuristic_score), 3),
        "short_judge_rationale": (
            f"Heuristic score {heuristic_score:.2f} with file/symbol/dependency overlap suggests `{heuristic_intent}`."
        ),
        "judge_source": "heuristic_only",
        "heuristic_score": round(heuristic_score, 3),
        "evidence": evidence,
    }


def build_pairwise_merge_map(profiles: list[CommitJudgeProfile]) -> dict[tuple[str, str], dict[str, Any]]:
    pair_map: dict[tuple[str, str], dict[str, Any]] = {}
    for idx, left in enumerate(profiles):
        for jdx in range(idx + 1, min(len(profiles), idx + 4)):
            right = profiles[jdx]
            verdict = judge_commit_pair(left, right)
            if verdict is None:
                continue
            pair_map[(left.commit_id, right.commit_id)] = verdict
    return pair_map


def union_find_groups(pairs: list[tuple[str, str]]) -> dict[str, list[str]]:
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(left: str, right: str) -> None:
        parent[find(left)] = find(right)

    for left, right in pairs:
        union(left, right)

    groups: dict[str, list[str]] = defaultdict(list)
    for node in parent:
        groups[find(node)].append(node)
    return {root: sorted(members) for root, members in groups.items()}


def judge_repo_commit_mergeability(
    commit_df: pd.DataFrame,
    initial_df: pd.DataFrame | None,
) -> dict[str, Any]:
    profiles: list[CommitJudgeProfile] = []
    commit_rows: dict[str, dict[str, Any]] = {}
    meaningless_records: dict[str, dict[str, Any]] = {}

    for _, row in commit_df.iterrows():
        row_dict = row.to_dict()
        profile = build_commit_judge_profile(row_dict, initial_df)
        if profile is None or not profile.commit_id:
            continue
        commit_rows[profile.commit_id] = row_dict
        meaningless, reason, rationale = detect_meaningless_reason(
            row_dict,
            profile.changed_files,
            profile.changed_py_files,
            profile.evidence_list,
            profile.patch_text,
        )
        if meaningless:
            meaningless_records[profile.commit_id] = {
                "commit_id": profile.commit_id,
                "intent_label": "unrelated_or_distinct",
                "should_merge": False,
                "merge_group_id": profile.commit_id,
                "merge_confidence": 0.0,
                "is_meaningless": True,
                "meaningless_reason": reason,
                "short_judge_rationale": rationale,
            }
            continue
        profiles.append(profile)

    pair_map = build_pairwise_merge_map(profiles)
    merged_pairs = [pair for pair, verdict in pair_map.items() if verdict.get("should_merge")]
    groups = union_find_groups(merged_pairs)
    commit_to_group: dict[str, str] = {}
    group_records: list[dict[str, Any]] = []

    for members in groups.values():
        group_id = make_merge_group_id(profiles[0].repo if profiles else "repo", members)
        confidences = []
        intent_votes: Counter[str] = Counter()
        for left, right in zip(members, members[1:]):
            verdict = pair_map.get((left, right)) or pair_map.get((right, left))
            if verdict:
                confidences.append(float(verdict.get("merge_confidence", 0.0) or 0.0))
                intent_votes[str(verdict.get("intent_label", "unrelated_or_distinct"))] += 1
        shared_intent = intent_votes.most_common(1)[0][0] if intent_votes else "same_feature_increment"
        group_records.append({
            "merge_group_id": group_id,
            "repo": profiles[0].repo if profiles else "",
            "member_commit_ids": members,
            "shared_intent": shared_intent,
            "group_confidence": round(sum(confidences) / len(confidences), 3) if confidences else 0.6,
        })
        for member in members:
            commit_to_group[member] = group_id

    decisions: list[dict[str, Any]] = []
    keep_separate_commit_ids: list[str] = []
    drop_commit_ids: list[str] = []
    for profile in profiles:
        group_id = commit_to_group.get(profile.commit_id, profile.commit_id)
        member_pairs = [
            verdict
            for pair, verdict in pair_map.items()
            if profile.commit_id in pair and verdict.get("should_merge")
        ]
        should_merge = bool(member_pairs)
        intent_label = member_pairs[0].get("intent_label", "unrelated_or_distinct") if member_pairs else "unrelated_or_distinct"
        merge_confidence = round(max(float(item.get("merge_confidence", 0.0) or 0.0) for item in member_pairs), 3) if member_pairs else 0.0
        rationale = member_pairs[0].get("short_judge_rationale", "No strong neighboring intent match; keep separate.") if member_pairs else "No strong neighboring intent match; keep separate."
        decisions.append({
            "commit_id": profile.commit_id,
            "intent_label": intent_label,
            "should_merge": should_merge,
            "merge_group_id": group_id,
            "merge_confidence": merge_confidence,
            "is_meaningless": False,
            "meaningless_reason": "",
            "short_judge_rationale": rationale,
        })
        if should_merge:
            continue
        keep_separate_commit_ids.append(profile.commit_id)

    for record in meaningless_records.values():
        decisions.append(record)
        drop_commit_ids.append(record["commit_id"])

    decisions.sort(key=lambda item: str(item.get("commit_id", "")))
    merge_commit_ids = sorted({member for group in group_records for member in group.get("member_commit_ids", [])})
    return {
        "repo": str(commit_df.iloc[0].get("repo_name", "")) if not commit_df.empty else "",
        "commit_decisions": decisions,
        "merge_groups": group_records,
        "merge_commit_ids": merge_commit_ids,
        "drop_commit_ids": sorted(drop_commit_ids),
        "keep_separate_commit_ids": sorted(keep_separate_commit_ids),
        "merge_candidate_pairs": [
            {
                "left_commit_id": left,
                "right_commit_id": right,
                **verdict,
            }
            for (left, right), verdict in pair_map.items()
        ],
    }


def index_in_candidates(candidates: list[dict[str, Any]], target: dict[str, Any]) -> int:
    target_key = stable_json(target)
    for idx, item in enumerate(candidates):
        if stable_json(item) == target_key:
            return idx
    return -1


def infer_intent_reason(
    target_files: list[FileChangeEvidence],
    patch_type: str,
    risk_surface: str,
    root_symbol: dict[str, Any],
) -> str:
    file_names = [item.file_path for item in target_files]
    return (
        f"Likely edit starts in `{file_names[0]}`, centers on `{root_symbol.get('qualname', '')}`, "
        f"matches `{patch_type}`, and primarily risks `{risk_surface}`."
    )


def build_task_intent_to_edit_sketch(
    row: dict[str, Any],
    evidence_list: list[FileChangeEvidence],
    changed_py_files: list[str],
    repo_snapshot: dict[str, str],
    relevant_files: list[dict[str, Any]],
    dep_map: dict[str, list[str]],
    reverse_dep_map: dict[str, list[str]],
    qa: QATracker,
) -> list[dict[str, Any]]:
    selected_targets = choose_intent_target_files(evidence_list, dep_map, reverse_dep_map)
    if not selected_targets:
        qa.skip("ambiguous_intent_target_files")
        return []

    primary_target = selected_targets[0]
    gold_symbol = choose_clear_gold_symbol(primary_target.changed_symbols, primary_target.patch_text)
    if gold_symbol is None:
        qa.skip("ambiguous_intent_root_symbol")
        return []

    patch_type, patch_type_names = classify_patch_type(row, selected_targets, changed_py_files)
    if not patch_type or len(patch_type_names) < 3:
        qa.skip("ambiguous_intent_patch_type")
        return []

    risk_surface, risk_surface_names = classify_risk_surface(
        patch_type,
        selected_targets,
        changed_py_files,
        dep_map,
        reverse_dep_map,
    )
    if not risk_surface or len(risk_surface_names) < 3:
        qa.skip("ambiguous_intent_risk_surface")
        return []

    target_file_candidates = build_intent_target_file_candidates(
        selected_targets,
        evidence_list,
        repo_snapshot,
        relevant_files,
        dep_map,
        reverse_dep_map,
    )
    if len(target_file_candidates) < 3:
        qa.skip("weak_intent_target_file_candidates")
        return []

    root_symbol_candidates = build_intent_root_symbol_candidates(primary_target, gold_symbol)
    if len(root_symbol_candidates) < 3:
        qa.skip("weak_intent_root_symbol_candidates")
        return []

    best_target_file_ids = []
    for item in selected_targets:
        idx = index_in_candidates(target_file_candidates, {"file_path": item.file_path})
        if idx >= 0:
            best_target_file_ids.append(idx)
    best_target_file_ids = dedupe_preserve(best_target_file_ids)[:2]
    best_root_symbol_id = index_in_candidates(
        root_symbol_candidates,
        symbol_to_action("open_symbol", primary_target.file_path, gold_symbol),
    )
    best_patch_type_id = patch_type_names.index(patch_type) if patch_type in patch_type_names else -1
    best_risk_surface_id = risk_surface_names.index(risk_surface) if risk_surface in risk_surface_names else -1

    if not best_target_file_ids or best_root_symbol_id < 0 or best_patch_type_id < 0 or best_risk_surface_id < 0:
        qa.skip("incomplete_intent_label_mapping")
        return []

    sample = {
        "task_type": "intent_to_edit_sketch",
        "input": {
            "policy_goal": "infer_likely_edit_shape_from_early_intent",
            "commit_message": normalize_text(row.get("commit_message", ""), 500),
            "repo_tree_structure": normalize_text(row.get("repo_tree_structure", ""), 2200),
            "relevant_files_hint": [
                {"file_path": item["file_path"], "distance": item["distance"]}
                for item in sorted(relevant_files, key=lambda item: item["distance"])[:5]
            ],
            "target_file_candidates": target_file_candidates,
            "patch_type_candidates": [{"patch_type": item} for item in patch_type_names],
            "root_symbol_candidates": root_symbol_candidates,
            "risk_surface_candidates": [{"risk_surface": item} for item in risk_surface_names],
        },
        "intent_sketch": {
            "best_target_file_ids": best_target_file_ids,
            "best_patch_type_id": best_patch_type_id,
            "best_root_symbol_id": best_root_symbol_id,
            "best_risk_surface_id": best_risk_surface_id,
            "reason": infer_intent_reason(selected_targets, patch_type, risk_surface, gold_symbol),
        },
        "rationale": (
            f"Early intent points to `{primary_target.file_path}` and `{gold_symbol['qualname']}`; "
            f"the diff shape is most consistent with `{patch_type}` and risk surface `{risk_surface}`."
        ),
        "metadata": {
            "repo": str(row.get("repo_name", "")),
            "commit_id": str(row.get("commit_id", "")),
            "gold_file": primary_target.file_path,
            "gold_symbol": gold_symbol["qualname"],
            "patch_type": patch_type,
            "risk_surface": risk_surface,
        },
    }
    if validate_sample(sample):
        return [sample]
    qa.skip("weak_intent_to_edit_sketch_sample")
    return []


def validate_intent_sketch_sample(sample: dict[str, Any]) -> bool:
    intent = sample.get("intent_sketch", {})
    input_data = sample.get("input", {})
    target_candidates = input_data.get("target_file_candidates", [])
    patch_candidates = input_data.get("patch_type_candidates", [])
    root_candidates = input_data.get("root_symbol_candidates", [])
    risk_candidates = input_data.get("risk_surface_candidates", [])
    if len(target_candidates) < 3 or len(patch_candidates) < 3 or len(root_candidates) < 3 or len(risk_candidates) < 3:
        return False
    file_ids = intent.get("best_target_file_ids", [])
    if not file_ids or any(not 0 <= idx < len(target_candidates) for idx in file_ids):
        return False
    if not 0 <= int(intent.get("best_patch_type_id", -1)) < len(patch_candidates):
        return False
    if not 0 <= int(intent.get("best_root_symbol_id", -1)) < len(root_candidates):
        return False
    if not 0 <= int(intent.get("best_risk_surface_id", -1)) < len(risk_candidates):
        return False
    return True


def detect_leakage_flags(sample: dict[str, Any], serialized_record: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    prompt_text = serialized_record.get("prompt_text", "")
    target_text = serialized_record.get("target_text", "")

    # Patch hunks should never appear in prompts for the grounded tasks.
    # Grounded snippets are allowed, but raw diff markers are not.
    if "diff --git" in prompt_text or "@@ -" in prompt_text:
        flags.append("raw_patch_hunk_present")
    if "+++ b/" in prompt_text or "--- a/" in prompt_text:
        flags.append("raw_patch_file_header_present")

    leakage_terms = (
        '"candidate_score"',
        '"reason_tags"',
        "best_symbol",
        "best_region",
        "best_follow_dependency",
    )
    if any(term in prompt_text for term in leakage_terms):
        flags.append("ranking_metadata_in_prompt")
    if any(term in target_text for term in leakage_terms):
        flags.append("ranking_metadata_in_target")

    return flags


def validate_task_purity(sample: dict[str, Any], serialized_record: dict[str, Any]) -> tuple[bool, list[str], str]:
    flags = detect_leakage_flags(sample, serialized_record)
    task_type = sample.get("task_type", "")
    input_data = sample.get("input", {})
    target = sample.get("target", {})
    prompt_text = serialized_record.get("prompt_text", "")

    if flags:
        return False, flags, "patch_leakage"

    # direct-answer supervision
    available_actions = input_data.get("available_actions", [])
    if task_type in {"patch_grounding", "ast_dependency_trace"}:
        if available_actions and (not isinstance(available_actions, list) or any(not is_public_action_payload(action) for action in available_actions)):
            return False, flags, "non_public_available_action_schema"

        selected_action = target.get("selected_action")
        if not isinstance(selected_action, dict) or not selected_action:
            return False, flags, "invalid_selected_action"
        if not is_public_action_payload(selected_action):
            return False, flags, "non_public_selected_action_schema"

        # Fail hard if old schema fields appear — schema regression guard.
        if "selected_action_id" in target or "best_action_id" in target:
            raise RuntimeError(
                f"[{PIPELINE_VERSION}] Old schema field detected in target for {task_type}: "
                f"keys={list(target.keys())}. This is a pipeline bug — fix the builder."
            )

        if selected_action.get("action") == "open_symbol":
            symbol = str(selected_action.get("symbol", "") or "")
            if symbol and symbol not in prompt_text:
                return False, flags, "unverifiable_root_symbol"

        if selected_action.get("action") == "read_region":
            span = selected_action.get("span")
            if isinstance(span, list) and len(span) == 2:
                s, e = int(span[0]), int(span[1])
                if f"{s}:" in prompt_text and f"{e}:" in prompt_text and "Available Actions" not in prompt_text:
                    return False, flags, "prompt_leaks_target_span"

    return True, flags, ""


def validate_ranking_sample(sample: dict[str, Any]) -> bool:
    candidates = sample.get("candidates", [])
    best_action_id = sample.get("best_action_id", -1)
    if len(candidates) < 4:
        return False
    if not 0 <= best_action_id < len(candidates):
        return False
    if len(dedupe_preserve(candidates)) != len(candidates):
        return False
    return True


def validate_patch_grounding_sample(sample: dict[str, Any]) -> bool:
    input_data = sample.get("input", {})
    target = sample.get("target", {})

    if input_data.get("goal_source") != "commit_message":
        return False
    if not input_data.get("goal_text"):
        return False
    if input_data.get("subtask") not in {"choose_first_file", "narrow_patch_region"}:
        return False
    if not input_data.get("task_instruction"):
        return False

    obs = input_data.get("current_observation", {})
    if not isinstance(obs, dict):
        return False
    if not obs.get("opened_file"):
        return False
    if not obs.get("current_snippet"):
        return False

    related = obs.get("related_snippets", [])
    outline = obs.get("file_ast_outline", [])
    if (not isinstance(related, list) or not related) and (not isinstance(outline, list) or not outline):
        return False

    available_actions = input_data.get("available_actions", [])
    if available_actions and (not isinstance(available_actions, list) or any(not is_public_action_payload(action) for action in available_actions)):
        return False

    selected_action = target.get("selected_action")
    if not isinstance(selected_action, dict) or not selected_action:
        return False
    if not is_public_action_payload(selected_action):
        return False

    # Hard filter: patch_grounding should never stop at open_file.
    if selected_action.get("action") == "open_file":
        return False

    # If action is open_symbol, require a grounded span and strict alignment with gold + outline.
    if selected_action.get("action") == "open_symbol":
        gold_span = sample.get("metadata", {}).get("gold_line_span", [])
        gold_symbol = str(sample.get("metadata", {}).get("gold_symbol", "") or "")
        selected_symbol = str(selected_action.get("symbol", "") or "")
        sym_span = selected_action.get("span", [])
        # Reject if span is missing — span is now always attached when grounded.
        if not (isinstance(sym_span, list) and len(sym_span) == 2
                and int(sym_span[0]) > 0 and int(sym_span[1]) > 0):
            return False
        if gold_symbol and selected_symbol != gold_symbol:
            return False
        if is_valid_span(gold_span) and not spans_nearly_equal(sym_span, gold_span, tol=2):
            return False
        outline_span = outline_span_for_symbol(outline, selected_symbol)
        if outline_span is not None and not spans_nearly_equal(sym_span, outline_span, tol=2):
            return False
        if not patch_grounding_symbol_is_semantically_supported(sample):
            return False

    if selected_action.get("action") == "read_region":
        region = selected_action.get("span", [])
        gold_span = sample.get("metadata", {}).get("gold_line_span", [])
        if not is_valid_span(region):
            return False
        if span_length(region) > 48:
            return False
        if is_valid_span(gold_span) and not spans_overlap(region, gold_span):
            return False

    return True


def validate_ast_dependency_trace_sample(sample: dict[str, Any]) -> bool:
    input_data = sample.get("input", {})
    target = sample.get("target", {})

    if input_data.get("goal_source") != "commit_message":
        return False
    if not input_data.get("goal_text"):
        return False
    if input_data.get("subtask") != "follow_dependency":
        return False
    if not input_data.get("task_instruction"):
        return False

    trace_state = input_data.get("trace_state", {})
    if not isinstance(trace_state, dict):
        return False

    obs = input_data.get("current_observation", {})
    if not isinstance(obs, dict):
        return False
    if not obs.get("entry_file"):
        return False
    if not obs.get("entry_snippet"):
        return False

    outline = obs.get("entry_file_ast_outline", [])
    related = obs.get("related_snippets", [])
    if not isinstance(outline, list) or not outline:
        return False
    if not isinstance(related, list) or not related:
        return False

    available_actions = input_data.get("available_actions", [])
    if available_actions and (not isinstance(available_actions, list) or any(not is_public_action_payload(action) for action in available_actions)):
        return False

    selected_action = target.get("selected_action")
    if not isinstance(selected_action, dict) or not selected_action:
        return False
    if not is_public_action_payload(selected_action):
        return False

    # Strict action schema guard.
    if str(selected_action.get("action", "") or "") not in {"follow_dependency", "open_symbol"}:
        return False

    # If action is open_symbol, require a grounded span.
    # Only enforce strict alignment after the trace reaches gold_file.
    if selected_action.get("action") == "open_symbol":
        sym_span = selected_action.get("span", [])
        if not (isinstance(sym_span, list) and len(sym_span) == 2
                and int(sym_span[0]) > 0 and int(sym_span[1]) > 0):
            return False
        if str(selected_action.get("file_path", "") or "") == str(sample.get("metadata", {}).get("gold_file", "") or ""):
            gold_span = sample.get("metadata", {}).get("gold_line_span", [])
            gold_symbol = str(sample.get("metadata", {}).get("gold_symbol", "") or "")
            selected_symbol = str(selected_action.get("symbol", "") or "")
            if gold_symbol and selected_symbol != gold_symbol:
                return False
            if is_valid_span(gold_span) and not spans_nearly_equal(sym_span, gold_span, tol=2):
                return False
            outline_span = outline_span_for_symbol(outline, selected_symbol)
            if outline_span is not None and not spans_nearly_equal(sym_span, outline_span, tol=2):
                return False

    return True


def validate_reading_summary_sample(sample: dict[str, Any]) -> bool:
    input_data = sample.get("input", {})
    target = sample.get("target", {})
    if input_data.get("goal_source") != "commit_message":
        return False
    if not input_data.get("goal_text"):
        return False
    if input_data.get("subtask") != "compress_working_state":
        return False
    history = input_data.get("read_history", [])
    if not isinstance(history, list) or len(history) < 2:
        return False
    if not isinstance(target, dict) or "working_summary" not in target:
        return False
    ws = target.get("working_summary", {})
    if not isinstance(ws, dict):
        return False
    if not ws.get("likely_root_file"):
        return False
    region = ws.get("likely_focus_region")
    if not (isinstance(region, list) and len(region) == 2 and all(isinstance(x, int) and x > 0 for x in region)):
        return False
    # Reject placeholder focus region.
    if is_placeholder_focus_region(region, history):
        return False
    supporting = ws.get("supporting_files", [])
    if not isinstance(supporting, list) or not supporting:
        return False
    if not ws.get("next_question"):
        return False
    # Reject generic / templatic next_question.
    if is_generic_next_question(str(ws["next_question"])):
        return False
    if not next_question_mentions_observed_evidence(str(ws["next_question"]), history):
        return False
    if not next_question_has_specific_anchor(
        str(ws["next_question"]),
        history,
        str(ws.get("likely_root_file", "") or ""),
    ):
        return False
    if not next_question_grounded_in_supporting_context(
        str(ws["next_question"]),
        history,
        supporting,
    ):
        return False
    # Ensure likely_root_file is grounded in observed history.
    observed_files = set()
    for step in history:
        if isinstance(step, dict):
            fp = step.get("observation", {}).get("file_path", "")
            if fp:
                observed_files.add(fp)
    if ws.get("likely_root_file") not in observed_files and observed_files:
        return False
    # Hard filter: reject trivial / non-logic root files.
    if is_trivial_root_file(str(ws.get("likely_root_file", "") or ""), history):
        return False
    if not all(f in observed_files for f in supporting):
        return False
    root_regions = observed_regions_for_file(history, str(ws.get("likely_root_file", "") or ""))
    if not root_regions:
        return False
    if not any(spans_overlap(region, observed_region) for observed_region in root_regions):
        return False

    # Hard filter: reject templated next_question that isn't grounded in supporting snippets.
    # Example template: "Where in `X.py` is `Foo` called, configured, wrapped, or supplied with arguments ...?"
    next_q = str(ws.get("next_question", "") or "")
    t = next_q.lower()
    looks_like_template = (
        t.startswith("where in ")
        and " is " in t
        and (
            " called" in t
            or " wrapped" in t
            or "supplied with arguments" in t
        )
    )
    if looks_like_template:
        root_symbol = str(sample.get("metadata", {}).get("gold_symbol", "") or "")
        symbol_leaf = root_symbol.split(".")[-1] if root_symbol else ""
        if symbol_leaf:
            root_file = str(ws.get("likely_root_file", "") or "")
            supporting_files = ws.get("supporting_files", [])
            support_set = set(str(p) for p in supporting_files if str(p or ""))
            supporting_snips: list[str] = []
            for step in history:
                if not isinstance(step, dict):
                    continue
                obs = step.get("observation", {})
                if not isinstance(obs, dict):
                    continue
                fp = str(obs.get("file_path", "") or "")
                if not fp or fp == root_file:
                    continue
                if support_set and fp not in support_set:
                    continue
                snip = str(obs.get("snippet", "") or "")
                if snip:
                    supporting_snips.append(snip)
            supporting_snippet_text = "\n".join(supporting_snips)
            if not supporting_snippet_text or symbol_leaf not in supporting_snippet_text:
                return False

    return True


def validate_summary_sample(sample: dict[str, Any]) -> bool:
    focus_files = sample.get("focus_files", [])
    focus_symbols = sample.get("focus_symbols", [])
    compressed_summary = sample.get("compressed_summary", {})
    return bool(len(focus_files) >= 2 and focus_symbols and compressed_summary.get("focus_tree"))


def validate_sample(sample: dict[str, Any]) -> bool:
    task_type = sample.get("task_type")
    if task_type == "patch_grounding":
        return validate_patch_grounding_sample(sample)
    if task_type == "ast_dependency_trace":
        return validate_ast_dependency_trace_sample(sample)
    if task_type == "reading_summary":
        return validate_reading_summary_sample(sample)
    # Legacy proxy tasks are not considered valid for output.
    return False


# ── task builders ──────────────────────────────────────────────────────────────


def build_precise_file_candidates(
    gold_file: str,
    changed_py_files: list[str],
    repo_snapshot: dict[str, str],
    relevant_files: list[dict[str, Any]],
    dep_map: dict[str, list[str]],
    reverse_dep_map: dict[str, list[str]],
) -> list[dict[str, Any]]:
    relevant_paths = [item["file_path"] for item in sorted(relevant_files, key=lambda item: item["distance"]) if is_python_file(item["file_path"])]
    distractors: list[dict[str, Any]] = []
    for file_path in changed_py_files:
        if file_path != gold_file:
            distractors.append(annotate_candidate(
                {"action_type": "open_file", "file_path": file_path},
                "heuristic",
                negative_type="false_change_propagation",
                provenance="other_changed_file",
            ))
    for file_path in same_directory_candidates(gold_file, list(repo_snapshot.keys()), limit=3):
        distractors.append(annotate_candidate(
            {"action_type": "open_file", "file_path": file_path},
            "heuristic",
            negative_type="nearby_but_not_best",
            provenance="same_directory_neighbor",
        ))
    for file_path in relevant_paths[:4]:
        if file_path != gold_file:
            distractors.append(annotate_candidate(
                {"action_type": "open_file", "file_path": file_path},
                "heuristic",
                negative_type="plausible_but_irrelevant",
                provenance="relevant_file_hint",
            ))
    for file_path in dependency_neighbors(gold_file, dep_map, reverse_dep_map, limit=4):
        if file_path != gold_file:
            distractors.append(annotate_candidate(
                {"action_type": "open_file", "file_path": file_path},
                "heuristic",
                negative_type="wrong_layer",
                provenance="dependency_neighbor",
            ))
    distractors = dedupe_candidates(distractors)
    distractors.extend(baseline_action_candidates())
    return distractors


def build_task_precise_localization(
    row: dict[str, Any],
    gold_evidence: FileChangeEvidence,
    changed_py_files: list[str],
    repo_snapshot: dict[str, str],
    relevant_files: list[dict[str, Any]],
    dep_map: dict[str, list[str]],
    reverse_dep_map: dict[str, list[str]],
    qa: QATracker,
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    change_center_file = gold_evidence.file_path
    primary_span = gold_evidence.primary_changed_span or []
    if not primary_span:
        qa.skip("missing_primary_changed_span")
        return []

    root_source = gold_evidence.after_source or gold_evidence.before_source
    root_file_snippet = span_context_snippet(root_source, primary_span[0], primary_span[1]) if root_source else ""
    related_snippets = build_related_file_snippets(
        change_center_file,
        repo_snapshot,
        relevant_files,
        dep_map,
        reverse_dep_map,
        changed_py_files,
        limit=3,
    )
    if not related_snippets:
        qa.skip("missing_related_file_snippets")
    # We still allow the task to proceed even if related snippets are empty,
    # but record the skip reason for QA.

    first_read_entry_file = choose_first_read_entry_file(change_center_file, changed_py_files, relevant_files, dep_map, reverse_dep_map)
    file_choice_names = [path for path in changed_py_files if path != first_read_entry_file][:4]
    if not message_supports_candidate_choice(normalize_text(row.get("commit_message", ""), 500), first_read_entry_file, file_choice_names):
        qa.skip("weak_message_support_for_first_read_entry")
        return []
    file_distractors = build_precise_file_candidates(first_read_entry_file, changed_py_files, repo_snapshot, relevant_files, dep_map, reverse_dep_map)
    if not validate_minimum_file_distractors(file_distractors, first_read_entry_file):
        qa.skip("insufficient_precise_file_distractors")
        return []
    gold_file_action = annotate_candidate({"action_type": "open_file", "file_path": first_read_entry_file}, "real_positive")
    file_job = {
        "task_type": "precise_change_localization_file",
        "gold_action": gold_file_action,
        "candidate_pool": file_distractors,
        "input_context": {
            "commit_message": normalize_text(row.get("commit_message", ""), 300),
            "repo_tree_structure": compact_repo_tree(row.get("repo_tree_structure", ""), [first_read_entry_file, change_center_file]),
            "policy_goal": "choose_where_to_read_first",
        },
    }
    gold_symbol = choose_clear_gold_symbol(gold_evidence.changed_symbols, gold_evidence.patch_text)
    if gold_symbol is None or len(gold_evidence.symbol_pool) < 3:
        enriched_file_distractors = enrich_distractors_with_llm_hard_negatives([file_job])[0]
        file_candidates, best_file_id = build_action_candidates(gold_file_action, enriched_file_distractors, MAX_FILE_CANDIDATES)
        sample = {
            "task_type": "precise_change_localization",
            "input": {
                "granularity": "file",
                "policy_goal": "choose_where_to_read_first",
                "commit_message": normalize_text(row.get("commit_message", ""), 500),
                "repo_tree_structure": compact_repo_tree(row.get("repo_tree_structure", ""), [first_read_entry_file, change_center_file]),
                "root_file_snippet": root_file_snippet,
                "related_file_snippets": related_snippets,
            },
            "candidates": file_candidates,
            "best_action_id": best_file_id,
            "rationale": (
                f"`{first_read_entry_file}` is the best first read for reaching the actual change center `{change_center_file}`."
            ),
            "metadata": {
                "repo": str(row.get("repo_name", "")),
                "commit_id": str(row.get("commit_id", "")),
                "gold_file": first_read_entry_file,
                "gold_line_span": primary_span,
                "change_center_file": change_center_file,
                "first_read_entry_file": first_read_entry_file,
                "gold_changed_symbol_count": len(gold_evidence.changed_symbols),
            },
        }
        if validate_sample(sample):
            samples.append(sample)
        else:
            qa.skip("weak_file_localization_sample")
        if gold_symbol is None:
            qa.skip("ambiguous_symbol_localization")
        else:
            qa.skip("weak_ast_for_symbol_localization")
        return samples

    distractor_symbols = nearby_symbol_distractors(gold_symbol, gold_evidence.symbol_pool, limit=4)
    symbol_distractors = [
        annotate_candidate(
            symbol_to_action("open_symbol", change_center_file, symbol),
            "heuristic",
            negative_type="nearby_but_not_best",
            provenance="same_file_neighbor_symbol",
        )
        for symbol in distractor_symbols
    ]
    if not symbol_distractors:
        qa.skip("missing_same_file_symbol_distractors")
        return samples
    symbol_choice_names = [symbol.get("qualname", "") for symbol in distractor_symbols]
    if not message_supports_candidate_choice(normalize_text(row.get("commit_message", ""), 500), gold_symbol.get("qualname", ""), symbol_choice_names):
        qa.skip("weak_message_support_for_symbol_choice")
        return samples
    symbol_distractors.extend(baseline_action_candidates())
    gold_symbol_action = annotate_candidate(symbol_to_action("open_symbol", change_center_file, gold_symbol), "real_positive")
    symbol_job = {
        "task_type": "precise_change_localization_symbol",
        "gold_action": gold_symbol_action,
        "candidate_pool": symbol_distractors,
        "input_context": {
            "commit_message": normalize_text(row.get("commit_message", ""), 300),
            "file_path": change_center_file,
            "policy_goal": "choose_whether_to_jump_to_symbol",
        },
    }
    enriched_file_distractors, enriched_symbol_distractors = enrich_distractors_with_llm_hard_negatives([file_job, symbol_job])
    file_candidates, best_file_id = build_action_candidates(gold_file_action, enriched_file_distractors, MAX_FILE_CANDIDATES)
    sample = {
        "task_type": "precise_change_localization",
        "input": {
            "granularity": "file",
            "policy_goal": "choose_where_to_read_first",
            "commit_message": normalize_text(row.get("commit_message", ""), 500),
            "repo_tree_structure": compact_repo_tree(row.get("repo_tree_structure", ""), [first_read_entry_file, change_center_file]),
            "root_file_snippet": root_file_snippet,
            "related_file_snippets": related_snippets,
        },
        "candidates": file_candidates,
        "best_action_id": best_file_id,
        "rationale": (
            f"`{first_read_entry_file}` is the best first read for reaching the actual change center `{change_center_file}`."
        ),
        "metadata": {
            "repo": str(row.get("repo_name", "")),
            "commit_id": str(row.get("commit_id", "")),
            "gold_file": first_read_entry_file,
            "gold_line_span": primary_span,
            "change_center_file": change_center_file,
            "first_read_entry_file": first_read_entry_file,
            "gold_changed_symbol_count": len(gold_evidence.changed_symbols),
        },
    }
    if validate_sample(sample):
        samples.append(sample)
    else:
        qa.skip("weak_file_localization_sample")
    symbol_distractors = enriched_symbol_distractors
    symbol_candidates, best_symbol_id = build_action_candidates(gold_symbol_action, symbol_distractors, MAX_SYMBOL_CANDIDATES)
    symbol_sample = {
        "task_type": "precise_change_localization",
        "input": {
            "granularity": "symbol",
            "policy_goal": "choose_whether_to_jump_to_symbol",
            "commit_message": normalize_text(row.get("commit_message", ""), 500),
            "file_path": change_center_file,
            "file_ast_outline": build_file_outline(
                outline_source_for_symbol_task(gold_evidence, gold_symbol),
                change_center_file,
            ),
            "root_file_snippet": root_file_snippet,
            "related_file_snippets": related_snippets,
        },
        "candidates": symbol_candidates,
        "best_action_id": best_symbol_id,
        "rationale": (
            f"`{gold_symbol['qualname']}` is the clearest changed symbol in `{change_center_file}`, so jumping to that symbol "
            "is more precise than reading the entire file."
        ),
        "metadata": {
            "repo": str(row.get("repo_name", "")),
            "commit_id": str(row.get("commit_id", "")),
            "gold_file": change_center_file,
            "gold_symbol": gold_symbol["qualname"],
            "gold_line_span": [int(gold_symbol.get("lineno", 0) or 0), int(gold_symbol.get("end_lineno", 0) or 0)],
        },
    }
    if validate_sample(symbol_sample):
        samples.append(symbol_sample)
    else:
        qa.skip("weak_symbol_localization_sample")
    return samples


def build_task_patch_grounding(
    row: dict[str, Any],
    gold_evidence: FileChangeEvidence,
    changed_py_files: list[str],
    repo_snapshot: dict[str, str],
    relevant_files: list[dict[str, Any]],
    dep_map: dict[str, list[str]],
    reverse_dep_map: dict[str, list[str]],
    qa: QATracker,
    grounding: dict[str, Any] | None = None,
    preresolved_llm: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if not grounding:
        qa.skip("missing_root_grounding")
        return []

    root_file = str(grounding.get("root_file", "") or "")
    root_symbol = str(grounding.get("root_symbol", "") or "")
    root_line_span = grounding.get("root_line_span", [])
    related = grounding.get("related_snippets", [])
    issues = grounding.get("llm_root_grounding_issues", []) if isinstance(grounding.get("llm_root_grounding_issues", []), list) else []

    if not root_file or not (isinstance(root_line_span, list) and len(root_line_span) == 2):
        qa.skip("invalid_root_grounding")
        return []
    if not isinstance(related, list) or not related:
        qa.skip("missing_related_file_snippets")
        return []

    root_source = repo_snapshot.get(root_file, "")
    current_snippet = choose_non_answer_snippet_for_file(root_source)
    file_ast_outline = build_file_outline(root_source, root_file)
    if not file_ast_outline:
        qa.skip("weak_ast_for_patch_grounding")
        return []

    grounded_symbol_candidates: list[dict[str, Any]] = []
    for group in (
        gold_evidence.changed_symbols,
        grounding.get("root_symbol_candidates", []),
        gold_evidence.symbol_pool,
    ):
        if isinstance(group, list):
            grounded_symbol_candidates.extend(item for item in group if isinstance(item, dict))
    grounded_symbol_candidates = filter_symbol_candidates_for_file(grounded_symbol_candidates, root_file)
    symbol_resolution_source = choose_symbol_resolution_source(root_symbol, gold_evidence, default_source=root_source)

    action_candidates = build_available_actions_for_patch_grounding(
        root_file=root_file,
        root_symbol=root_symbol,
        root_line_span=[int(root_line_span[0]), int(root_line_span[1])],
        file_ast_outline=file_ast_outline,
        related_snippets=related,
        source=symbol_resolution_source,
        symbol_candidates=grounded_symbol_candidates,
        issues=issues,
    )

    # Use pre-resolved LLM result if available (batched path), else call sequentially.
    if preresolved_llm is not None and "pg_action" in preresolved_llm:
        selected_action_id, selected_action, action_meta = _parse_action_selection_result(
            preresolved_llm["pg_action"], action_candidates, task_type="patch_grounding"
        )
    else:
        selected_action_id, selected_action, action_meta = select_next_action_with_llm(
            task_type="patch_grounding",
            subtask="narrow_patch_region",
            goal_text=normalize_text(row.get("commit_message", ""), 500),
            action_candidates=action_candidates,
        )
    if selected_action is None or selected_action_id is None:
        qa.skip(f"next_action_drop_{action_meta.get('next_action_drop_reason', 'drop')}")
        return []

    # Reject weak fallback: fallback picked open_file but a better action type exists.
    if action_meta.get("selected_action_source") == "fallback":
        top_action_type = str(selected_action.get("action", "") or "")
        if top_action_type == "open_file":
            better_exists = any(
                str(c.get("action", "")) in ("open_symbol", "read_region")
                for c in action_candidates[1:]
            )
            if better_exists:
                qa.skip("patch_grounding_fallback_open_file_with_better_candidate")
                return []

    instruction_candidates = build_task_instruction_candidates("patch_grounding", "narrow_patch_region")
    if preresolved_llm is not None and "pg_instruction" in preresolved_llm:
        selected_instruction, used_inst_fallback, _ = _parse_instruction_selection_result(
            preresolved_llm["pg_instruction"], instruction_candidates
        )
    else:
        selected_instruction, used_inst_fallback, _ = select_task_instruction_with_llm(
            task_type="patch_grounding",
            subtask="narrow_patch_region",
            goal_text=normalize_text(row.get("commit_message", ""), 500),
            instruction_candidates=instruction_candidates,
        )

    sample = {
        "task_type": "patch_grounding",
        "input": {
            "goal_source": "commit_message",
            "goal_text": normalize_text(row.get("commit_message", ""), 500),
            "subtask": "narrow_patch_region",
            "task_instruction": selected_instruction,
            "changed_files_hint": changed_py_files[:10],
            "current_observation": {
                "opened_file": root_file,
                "file_ast_outline": file_ast_outline,
                "current_snippet": truncate_text_for_audit(current_snippet, 900),
                "related_snippets": related[:3],
            },
            "available_tools": ["open_file", "open_symbol", "read_region"],
        },
        "target": {
            "selected_action": strip_action_meta(action_candidates[selected_action_id]),
        },
        "metadata": {
            "repo": str(row.get("repo_name", "")),
            "commit_id": str(row.get("commit_id", "")),
            "gold_file": root_file,
            "gold_symbol": root_symbol,
            "gold_line_span": [int(root_line_span[0]), int(root_line_span[1])],
            "related_files": [item.get("file_path", "") for item in related if item.get("file_path")][:3],
            "used_task_instruction_fallback": used_inst_fallback,
            "used_action_selection_fallback": bool(action_meta.get("used_action_selection_fallback", False)),
            "selected_action_source": action_meta.get("selected_action_source", ""),
            "next_action_confidence": action_meta.get("next_action_confidence", None),
            "next_action_reason": action_meta.get("next_action_reason", ""),
            "next_action_issues": action_meta.get("next_action_issues", []),
            "pipeline_version": PIPELINE_VERSION,
        },
    }
    if validate_sample(sample):
        return [sample]
    qa.skip("invalid_patch_grounding")
    return []


def build_task_reading_summary(
    row: dict[str, Any],
    gold_evidence: FileChangeEvidence,
    changed_py_files: list[str],
    repo_snapshot: dict[str, str],
    relevant_files: list[dict[str, Any]],
    dep_map: dict[str, list[str]],
    reverse_dep_map: dict[str, list[str]],
    qa: QATracker,
    grounding: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Produce a short reading summary from grounded localization info."""
    if not grounding:
        qa.skip("missing_root_grounding")
        return []
    root_file = str(grounding.get("root_file", "") or "")
    root_symbol = str(grounding.get("root_symbol", "") or "")
    root_line_span = grounding.get("root_line_span", [])
    root_span_preview = str(grounding.get("root_span_preview", "") or "")
    related = grounding.get("related_snippets", [])
    related_files = [item.get("file_path", "") for item in related if item.get("file_path")][:3] if isinstance(related, list) else []
    if not root_file or not (isinstance(root_line_span, list) and len(root_line_span) == 2):
        qa.skip("invalid_root_grounding")
        return []
    if not root_span_preview:
        qa.skip("missing_root_span_preview")
        return []
    if not related_files:
        qa.skip("missing_related_file_snippets")
        return []

    # Build a small teacher trace and compress it into working memory.
    trace = build_teacher_read_trace(row, grounding, repo_snapshot, dep_map, reverse_dep_map)
    read_history = []
    for step in trace[:3]:
        read_history.append({
            "action": step.get("action", {}),
            "observation": step.get("observation", {}),
        })
    if len(read_history) < 2:
        qa.skip("insufficient_teacher_read_trace")
        return []

    goal_text = normalize_text(row.get("commit_message", ""), 500)
    observed_files = dedupe_preserve([
        str(step.get("observation", {}).get("file_path", "") or "")
        for step in read_history
        if isinstance(step.get("observation", {}), dict)
    ])
    observed_files = [p for p in observed_files if p]

    working_summary, ws_meta = generate_working_summary_with_llm(
        goal_text=goal_text,
        read_history=read_history,
        allowed_files=observed_files,
    )
    if working_summary is None:
        qa.skip(f"working_summary_drop_{ws_meta.get('working_summary_drop_reason', 'invalid')}")
        return []
    working_summary_source = ws_meta.get("working_summary_source", "") or "fallback"
    used_working_summary_fallback = bool(ws_meta.get("used_working_summary_fallback", False))

    instruction_candidates = build_task_instruction_candidates("reading_summary", "compress_working_state")
    selected_instruction, used_inst_fallback, inst_source = select_task_instruction_with_llm(
        task_type="reading_summary",
        subtask="compress_working_state",
        goal_text=goal_text,
        instruction_candidates=instruction_candidates,
    )

    teacher_trace_steps = [{"action": step.get("action", {})} for step in read_history]

    sample = {
        "task_type": "reading_summary",
        "input": {
            "goal_source": "commit_message",
            "goal_text": goal_text,
            "subtask": "compress_working_state",
            "task_instruction": selected_instruction,
            "read_history": read_history,
            "current_hypothesis": f"The patch likely lands in `{root_file}`.",
        },
        "target": {
            "working_summary": working_summary,
        },
        "metadata": {
            "repo": str(row.get("repo_name", "")),
            "commit_id": str(row.get("commit_id", "")),
            "gold_file": root_file,
            "gold_symbol": root_symbol,
            "gold_line_span": [int(root_line_span[0]), int(root_line_span[1])],
            "related_files": related_files,
            "root_file_candidates": public_root_file_candidates(grounding.get("root_file_candidates", [])),
            "root_symbol_candidates": public_root_symbol_candidates(grounding.get("root_symbol_candidates", [])),
            "root_span_candidates": public_root_span_candidates(grounding.get("root_span_candidates", [])),
            "related_snippet_candidates": public_related_snippet_candidates(grounding.get("related_snippet_candidates", [])),
            "llm_root_grounding_decision": grounding.get("llm_root_grounding_decision", ""),
            "llm_root_grounding_confidence": grounding.get("llm_root_grounding_confidence", None),
            "llm_root_grounding_reason": grounding.get("llm_root_grounding_reason", ""),
            "llm_root_grounding_issues": grounding.get("llm_root_grounding_issues", []),
            "used_root_grounding_fallback": grounding.get("used_root_grounding_fallback", False),
            "root_grounding_drop_reason": grounding.get("root_grounding_drop_reason", ""),
            "goal_source": "commit_message",
            "goal_text": goal_text,
            "subtask": "compress_working_state",
            "working_summary_source": working_summary_source,
            "used_working_summary_fallback": used_working_summary_fallback,
            "task_instruction_candidates": instruction_candidates,
            "selected_task_instruction": selected_instruction,
            "used_task_instruction_fallback": used_inst_fallback,
            "teacher_trace_steps": teacher_trace_steps,
            "working_summary_drop_reason": ws_meta.get("working_summary_drop_reason", ""),
            "next_question_repaired": bool(ws_meta.get("next_question_repaired", False)),
            "pipeline_version": PIPELINE_VERSION,
        },
    }
    if validate_sample(sample):
        return [sample]
    qa.skip("invalid_reading_summary")
    return []


def build_task_targeted_span_selection(
    row: dict[str, Any],
    gold_evidence: FileChangeEvidence,
    qa: QATracker,
) -> list[dict[str, Any]]:
    gold_symbol = choose_clear_gold_symbol(gold_evidence.changed_symbols, gold_evidence.patch_text)
    if gold_symbol is None:
        qa.skip("ambiguous_targeted_span_selection")
        return []
    if len(gold_evidence.symbol_pool) < 3:
        qa.skip("weak_ast_for_targeted_span_selection")
        return []

    distractor_symbols = nearby_symbol_distractors(gold_symbol, gold_evidence.symbol_pool, limit=4)
    if not distractor_symbols:
        qa.skip("missing_same_file_span_distractors")
        return []
    distractors = [
        annotate_candidate(
            region_action(gold_evidence.file_path, symbol, gold_evidence.before_source or gold_evidence.after_source),
            "heuristic",
            negative_type="nearby_but_not_best",
            provenance="neighboring_region",
        )
        for symbol in distractor_symbols
    ]
    distractors.append(annotate_candidate({"action_type": "read_full_file"}, "baseline", negative_type="overread_action", provenance="baseline"))
    gold_action = annotate_candidate(
        region_action(gold_evidence.file_path, gold_symbol, gold_evidence.before_source or gold_evidence.after_source),
        "real_positive",
    )
    distractors = enrich_distractors_with_llm_hard_negatives([{
        "task_type": "targeted_span_selection",
        "gold_action": gold_action,
        "candidate_pool": distractors,
        "input_context": {
            "commit_message": normalize_text(row.get("commit_message", ""), 300),
            "file_path": gold_evidence.file_path,
            "policy_goal": "choose_which_in_file_region_to_read_first",
        },
    }])[0]
    candidates, best_id = build_action_candidates(gold_action, distractors, MAX_SYMBOL_CANDIDATES)

    sample = {
        "task_type": "targeted_span_selection",
        "input": {
            "policy_goal": "choose_which_in_file_region_to_read_first",
            "commit_message": normalize_text(row.get("commit_message", ""), 500),
            "patch_summary": compact_patch_hunk(gold_evidence.patch_text, max_hunks=1, max_lines=20, max_chars=800),
            "file_path": gold_evidence.file_path,
            "file_ast_outline": build_file_outline(
                outline_source_for_symbol_task(gold_evidence, gold_symbol),
                gold_evidence.file_path,
            ),
            "file_content_before": symbol_context_snippet(gold_evidence.before_source, gold_symbol),
        },
        "candidates": candidates,
        "best_action_id": best_id,
        "rationale": (
            f"The first region to inspect is `{gold_symbol['qualname']}` because its span is directly changed and "
            "gives the shortest useful read path inside the selected file."
        ),
        "metadata": {
            "repo": str(row.get("repo_name", "")),
            "commit_id": str(row.get("commit_id", "")),
            "gold_file": gold_evidence.file_path,
            "gold_symbol": gold_symbol["qualname"],
        },
    }
    if validate_sample(sample):
        return [sample]
    qa.skip("weak_targeted_span_selection_sample")
    return []


def build_task_precise_line_localization(
    row: dict[str, Any],
    gold_evidence: FileChangeEvidence,
    qa: QATracker,
) -> list[dict[str, Any]]:
    """Stronger line-span localization over changed regions.

    Candidates are same-file `read_region` actions over changed spans plus a
    read_full_file baseline. The gold action targets the primary changed span.
    """
    if not gold_evidence.primary_changed_span:
        qa.skip("missing_primary_changed_span")
        return []
    if not gold_evidence.changed_line_spans or len(gold_evidence.changed_line_spans) < 2:
        qa.skip("missing_line_level_distractors")
        return []

    primary_span = gold_evidence.primary_changed_span
    source = gold_evidence.after_source or gold_evidence.before_source
    if not source.strip():
        qa.skip("missing_source_for_line_localization")
        return []

    # Build span candidates from changed spans in the same file.
    span_candidates = []
    for span in gold_evidence.changed_line_spans:
        span_candidates.append(span_action(gold_evidence.file_path, span, source))
    # Require at least one distractor distinct from the primary span.
    unique_spans = [c for c in span_candidates if c["span"] != primary_span]
    if len(unique_spans) < 1:
        qa.skip("missing_line_level_distractors")
        return []

    distractors = [annotate_candidate(c, "heuristic", negative_type="nearby_but_not_best", provenance="changed_line_span") for c in span_candidates if c["span"] != primary_span]
    distractors.append(annotate_candidate({"action_type": "read_full_file", "file_path": gold_evidence.file_path}, "baseline", negative_type="overread_action", provenance="baseline"))
    gold_action = annotate_candidate(span_action(gold_evidence.file_path, primary_span, source), "real_positive")
    candidates, best_id = build_action_candidates(gold_action, distractors, MAX_SYMBOL_CANDIDATES)

    input_payload = {
        "granularity": "line_span",
        "policy_goal": "choose_real_changed_span",
        "commit_message": normalize_text(row.get("commit_message", ""), 500),
        "file_path": gold_evidence.file_path,
        "file_ast_outline": build_file_outline(
            outline_source_for_symbol_task(gold_evidence, choose_clear_gold_symbol(gold_evidence.changed_symbols, gold_evidence.patch_text) or {}),
            gold_evidence.file_path,
        ),
        "root_file_snippet": span_context_snippet(source, primary_span[0], primary_span[1]),
    }
    sample = {
        "task_type": "precise_line_localization",
        "input": input_payload,
        "candidates": candidates,
        "best_action_id": best_id,
        "rationale": (
            f"The span {primary_span} most directly captures the edited region in `{gold_evidence.file_path}`."
        ),
        "metadata": {
            "repo": str(row.get("repo_name", "")),
            "commit_id": str(row.get("commit_id", "")),
            "gold_file": gold_evidence.file_path,
            "gold_line_span": primary_span,
        },
    }
    if validate_sample(sample):
        return [sample]
    qa.skip("weak_line_localization_sample")
    return []


def build_task_ast_dependency_trace(
    row: dict[str, Any],
    gold_evidence: FileChangeEvidence,
    changed_py_files: list[str],
    repo_snapshot: dict[str, str],
    relevant_files: list[dict[str, Any]],
    dep_map: dict[str, list[str]],
    reverse_dep_map: dict[str, list[str]],
    qa: QATracker,
    grounding: dict[str, Any] | None = None,
    preresolved_llm: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    low_semantic, reason = is_low_semantic_commit(row, gold_evidence.patch_text, changed_py_files)
    if low_semantic:
        qa.skip(f"skip_dependency_trace_{reason}")
        return []
    if not dep_map:
        qa.skip("missing_dependency_map")
        return []
    if not grounding:
        qa.skip("missing_root_grounding")
        return []

    root_file = str(grounding.get("root_file", "") or "")
    root_symbol = str(grounding.get("root_symbol", "") or "")
    root_line_span = grounding.get("root_line_span", [])
    if not root_file or not (isinstance(root_line_span, list) and len(root_line_span) == 2):
        qa.skip("invalid_root_grounding")
        return []

    relevant_paths = [item["file_path"] for item in sorted(relevant_files, key=lambda item: item["distance"]) if is_python_file(item["file_path"])]
    upstream_paths = upstream_dependency_candidates(root_file, reverse_dep_map, max_depth=3)
    candidate_entries = [f for f in changed_py_files if f != root_file] + relevant_paths[:6] + upstream_paths[:6]
    chain = choose_dependency_chain_to_gold(root_file, candidate_entries, dep_map)
    if len(chain) < 2:
        qa.skip("degenerate_dependency_trace")
        return []

    entry_file = chain[0]
    next_file = chain[1] if len(chain) >= 2 else None
    entry_source = repo_snapshot.get(entry_file, "")
    if not entry_source.strip():
        qa.skip("missing_entry_source")
        return []

    entry_outline = build_file_outline(entry_source, entry_file)
    if not entry_outline:
        qa.skip("low_information_dependency_entry_file")
        return []

    related = grounding.get("related_snippets", [])
    if not isinstance(related, list) or not related:
        qa.skip("missing_related_file_snippets")
        return []

    next_source = repo_snapshot.get(next_file, "") if next_file else ""
    next_outline = build_file_outline(next_source, next_file) if next_file and next_source else []

    action_candidates = build_available_actions_for_trace(
        next_file=next_file,
        root_file=root_file,
        next_file_source=next_source,
        next_file_outline=next_outline,
    )

    # Use pre-resolved LLM result if available (batched path), else call sequentially.
    if preresolved_llm is not None and "dt_action" in preresolved_llm:
        selected_action_id, selected_action, action_meta = _parse_action_selection_result(
            preresolved_llm["dt_action"], action_candidates, task_type="ast_dependency_trace"
        )
    else:
        selected_action_id, selected_action, action_meta = select_next_action_with_llm(
            task_type="ast_dependency_trace",
            subtask="follow_dependency",
            goal_text=normalize_text(row.get("commit_message", ""), 500),
            action_candidates=action_candidates,
        )
    if selected_action is None or selected_action_id is None:
        qa.skip(f"next_action_drop_{action_meta.get('next_action_drop_reason', 'drop')}")
        return []

    # Reject weak fallback: stop_and_summarize while follow_dependency/open_file exists.
    if action_meta.get("selected_action_source") == "fallback":
        top_action_type = str(selected_action.get("action", "") or "")
        if top_action_type == "stop_and_summarize":
            qa.skip("ast_dependency_trace_fallback_stop_with_better_candidate")
            return []

    dependency_context = {
        "entry_imports": dep_map.get(entry_file, [])[:8],
        "root_importers": reverse_dep_map.get(root_file, [])[:8],
    }
    entry_snippet = choose_non_answer_snippet_for_file(entry_source)

    instruction_candidates = build_task_instruction_candidates("ast_dependency_trace", "follow_dependency")
    if preresolved_llm is not None and "dt_instruction" in preresolved_llm:
        selected_instruction, used_inst_fallback, _ = _parse_instruction_selection_result(
            preresolved_llm["dt_instruction"], instruction_candidates
        )
    else:
        selected_instruction, used_inst_fallback, _ = select_task_instruction_with_llm(
            task_type="ast_dependency_trace",
            subtask="follow_dependency",
            goal_text=normalize_text(row.get("commit_message", ""), 500),
            instruction_candidates=instruction_candidates,
        )

    sample = {
        "task_type": "ast_dependency_trace",
        "input": {
            "goal_source": "commit_message",
            "goal_text": normalize_text(row.get("commit_message", ""), 500),
            "subtask": "follow_dependency",
            "task_instruction": selected_instruction,
            "trace_state": {
                "files_checked": [entry_file],
                "current_hypothesis": f"The patch likely lands in `{root_file}`.",
            },
            "current_observation": {
                "entry_file": entry_file,
                "entry_file_ast_outline": entry_outline,
                "entry_snippet": truncate_text_for_audit(entry_snippet, 900),
                "related_snippets": related[:2],
                "dependency_context": dependency_context,
            },
            "available_tools": ["open_file", "open_symbol", "read_region", "follow_dependency", "stop_and_summarize"],
        },
        "target": {
            "selected_action": strip_action_meta(action_candidates[selected_action_id]),
        },
        "metadata": {
            "repo": str(row.get("repo_name", "")),
            "commit_id": str(row.get("commit_id", "")),
            "entry_file": entry_file,
            "gold_file": root_file,
            "gold_symbol": root_symbol,
            "gold_line_span": [int(root_line_span[0]), int(root_line_span[1])],
            "related_files": [item.get("file_path", "") for item in related if item.get("file_path")][:3],
            "used_task_instruction_fallback": used_inst_fallback,
            "used_action_selection_fallback": bool(action_meta.get("used_action_selection_fallback", False)),
            "selected_action_source": action_meta.get("selected_action_source", ""),
            "next_action_confidence": action_meta.get("next_action_confidence", None),
            "next_action_reason": action_meta.get("next_action_reason", ""),
            "next_action_issues": action_meta.get("next_action_issues", []),
            "pipeline_version": PIPELINE_VERSION,
        },
    }
    if validate_sample(sample):
        return [sample]
    qa.skip("invalid_ast_dependency_trace")
    return []


def build_focus_tree(
    repo_snapshot: dict[str, str],
    dep_map: dict[str, list[str]],
    focus_files: list[str],
    changed_files: set[str],
) -> dict[str, Any]:
    path2node = {}
    for file_path in focus_files:
        path2node[file_path] = {
            "type": "file",
            "changed": file_path in changed_files,
            "dependencies": dep_map.get(file_path, [])[:5],
            "symbols": [sym["qualname"] for sym in build_symbol_index(repo_snapshot.get(file_path, ""), file_path)[:6]],
        }
    return build_file_tree_from_path_map(path2node)


def build_summary_text(entry_file: str, terminal_file: str, terminal_symbol: str, focus_files: list[str]) -> str:
    read_order = " -> ".join(focus_files[:5])
    return (
        f"Read `{entry_file}` first, follow `{read_order}`, and focus on `{terminal_file}` / `{terminal_symbol}` "
        "before expanding to broader file reads."
    )


def build_related_file_snippets(
    gold_file: str,
    repo_snapshot: dict[str, str],
    relevant_files: list[dict[str, Any]],
    dep_map: dict[str, list[str]],
    reverse_dep_map: dict[str, list[str]],
    changed_py_files: list[str],
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Construct compact snippets for a few related Python files.

    Sources of related files:
    - Other changed Python files in the commit.
    - Dependency neighbors of the gold file.
    - Relevant-file hints (semantic neighbors).
    """
    candidates: list[tuple[str, str]] = []  # (file_path, provenance)

    for path in changed_py_files:
        if path != gold_file:
            candidates.append((path, "other_changed_file"))

    for path in dependency_neighbors(gold_file, dep_map, reverse_dep_map, limit=6):
        if path != gold_file:
            candidates.append((path, "dependency_neighbor"))

    for item in sorted(relevant_files, key=lambda it: it.get("distance", 9.9))[:6]:
        path = item.get("file_path", "")
        if path and path != gold_file:
            candidates.append((path, "relevant_hint"))

    snippets: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for path, provenance in candidates:
        if path in seen_paths:
            continue
        seen_paths.add(path)
        source = repo_snapshot.get(path, "")
        if not source.strip():
            continue
        snippet = numbered_file_snippet(source, max_lines=40)
        if not snippet:
            continue
        snippets.append({
            "file_path": path,
            "snippet": snippet,
            "provenance": provenance,
        })
        if len(snippets) >= limit:
            break
    return snippets


def build_root_file_candidates(
    row: dict[str, Any],
    evidence_list: list[FileChangeEvidence],
    changed_py_files: list[str],
    relevant_files: list[dict[str, Any]],
    dep_map: dict[str, list[str]],
    reverse_dep_map: dict[str, list[str]],
    limit: int = 5,
) -> list[dict[str, Any]]:
    message = normalize_text(row.get("commit_message", ""), 400)
    relevant_paths = [item.get("file_path", "") for item in sorted(relevant_files, key=lambda it: it.get("distance", 9.9))]
    evidence_by_path = {e.file_path: e for e in evidence_list}

    # Extended aux token list — any of these in the path strongly suggests non-root.
    _AUX_PATH_TOKENS = (
        "utils", "util", "helper", "helpers", "adapter", "wrapper", "constants",
        "abi", "example", "examples", "demo", "script", "scripts", "cli",
        "notebook", "notebooks", "migration", "migrations", "fixture", "fixtures",
        "mock", "mocks", "stub", "stubs",
    )

    candidates: list[dict[str, Any]] = []
    for path in changed_py_files:
        ev = evidence_by_path.get(path)
        if not ev:
            continue

        lower = path.lower()
        tags: list[str] = []
        score = float(ev.score)

        # Aux path penalties — stronger than before.
        if is_aux_path(path):
            score -= 22.0
            tags.append("aux_path")
        if any(tok in lower for tok in _AUX_PATH_TOKENS):
            score -= 14.0
            tags.append("aux_module")

        # Changed symbols are a strong positive signal.
        if ev.changed_symbols:
            score += min(20.0, float(len(ev.changed_symbols)) * 2.5)
            tags.append("changed_symbol")

        # Span quality.
        if ev.primary_changed_span:
            score += 10.0
            tags.append("has_span")
            span_len = int(ev.primary_changed_span[1]) - int(ev.primary_changed_span[0]) + 1
            if span_len <= 2:
                # Tiny span — very weak signal, penalise more aggressively.
                score -= 14.0
                tags.append("tiny_span")
            elif span_len <= 5:
                score -= 4.0
                tags.append("small_span")
            elif 8 <= span_len <= 40:
                score += 6.0
                tags.append("readable_span")

        # Dependency topology.
        indeg = len(reverse_dep_map.get(path, []))
        outdeg = len(dep_map.get(path, []))
        if indeg >= 3:
            score += min(15.0, indeg * 2.5)
            tags.append("dependency_sink")
        if outdeg >= 6:
            score += 4.0
            tags.append("dependency_hub")

        if path in relevant_paths[:8]:
            score += 5.0
            tags.append("relevant_hint")
        if message_to_path_overlap(message, path) > 0:
            score += 7.0
            tags.append("message_overlap")

        candidates.append({
            "file_path": path,
            "heuristic_score": round(score, 3),
            "reason_tags": sorted(set(tags)),
        })

    candidates.sort(key=lambda c: (-float(c.get("heuristic_score", 0.0)), str(c.get("file_path", ""))))
    return dedupe_preserve(candidates)[: max(2, min(int(limit), 5))]


def build_root_symbol_candidates(
    row: dict[str, Any],
    file_evidence: FileChangeEvidence,
    limit: int = 6,
) -> list[dict[str, Any]]:
    """Generate 2–6 symbol candidates for a given root file."""
    message = normalize_text(row.get("commit_message", ""), 400)
    candidates: list[dict[str, Any]] = []

    # no_symbol is a low-priority fallback — score kept very low so it only
    # survives when no real symbol candidate exists.
    candidates.append({
        "symbol": "",
        "kind": "none",
        "span": [0, 0],
        "heuristic_score": -5.0,
        "reason_tags": ["no_symbol"],
    })

    primary = file_evidence.primary_changed_span or [0, 0]
    p0, p1 = int(primary[0] or 0), int(primary[1] or 0)

    _AUX_SYM_TOKENS = ("helper", "util", "wrapper", "adapter", "abi", "stub", "mock", "base", "mixin", "constant", "config")

    pool = file_evidence.symbol_pool or []
    changed = {str(s.get("qualname", "")) for s in file_evidence.changed_symbols or []}
    for sym in pool[:80]:
        qn = str(sym.get("qualname", ""))
        if not qn:
            continue
        span = [int(sym.get("lineno", 0) or 0), int(sym.get("end_lineno", 0) or 0)]
        if span[0] <= 0 or span[1] <= 0:
            continue
        tags: list[str] = []
        score = 0.0

        if qn in changed:
            score += 10.0
            tags.append("changed_symbol")

        # Overlap with primary changed span.
        overlap = max(0, min(span[1], p1) - max(span[0], p0) + 1)
        if overlap > 0:
            score += 8.0
            tags.append("span_overlap")

        # Prefer functions/methods/classes — they are readable anchors.
        kind = str(sym.get("kind", "") or "").lower()
        if kind in ("function", "method", "async_function", "async_method"):
            score += 4.0
            tags.append("callable")
        elif kind == "class":
            score += 2.0
            tags.append("class")

        # Readable span length bonus.
        sym_len = span[1] - span[0] + 1
        if PREFERRED_SPAN_MIN <= sym_len <= PREFERRED_SPAN_MAX:
            score += 3.0
            tags.append("readable_span")

        leaf = qn.split(".")[-1]
        if leaf and leaf.lower() in message.lower():
            score += 4.0
            tags.append("message_support")

        # Penalise aux/helper symbols.
        leaf_lower = leaf.lower()
        if any(tok in leaf_lower for tok in _AUX_SYM_TOKENS):
            score -= 8.0
            tags.append("aux_symbol")

        candidates.append({
            "symbol": qn,
            "kind": str(sym.get("kind", "")),
            "span": span,
            "heuristic_score": round(score, 3),
            "reason_tags": sorted(set(tags)),
        })

    candidates.sort(key=lambda c: (-float(c.get("heuristic_score", 0.0)), str(c.get("symbol", ""))))
    # Keep 2–6, no_symbol only if it would naturally rank in the window.
    out = []
    seen = set()
    for item in candidates:
        key = (item.get("symbol", ""), tuple(item.get("span", [])))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= max(2, min(int(limit), 6)):
            break
    return out


def build_root_span_candidates(
    file_evidence: FileChangeEvidence,
    root_source: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Generate compact, readable span candidates.

    Goal: first useful reading region, not maximal coverage.
    """
    spans: list[tuple[list[int], str]] = []

    def expand(span: list[int], pad: int) -> list[int]:
        s, e = int(span[0]), int(span[1])
        return [max(1, s - pad), e + pad]

    # changed span anchor
    if file_evidence.primary_changed_span:
        spans.append((expand(list(file_evidence.primary_changed_span), 6), "expanded_context"))

    # merged changed spans if compact enough
    raw_spans = [list(s) for s in (file_evidence.changed_line_spans or []) if len(s) == 2]
    raw_spans = dedupe_preserve(raw_spans)
    if raw_spans:
        min_s = min(int(s[0]) for s in raw_spans)
        max_e = max(int(s[1]) for s in raw_spans)
        spans.append((expand([min_s, max_e], 4), "merged_changed_spans"))

    # symbol-based region
    top = choose_clear_gold_symbol(file_evidence.changed_symbols, file_evidence.patch_text)
    if top:
        sym_span = [safe_int(top.get("lineno"), 0), safe_int(top.get("end_lineno"), 0)]
        if sym_span[0] > 0 and sym_span[1] >= sym_span[0]:
            spans.append((sym_span, "symbol_region"))
            spans.append((expand(sym_span, 8), "symbol_region_expanded"))

    out = []
    seen = set()

    for span, source in spans:
        start, end = int(span[0]), int(span[1])
        if start <= 0 or end < start:
            continue
        length = end - start + 1
        if length > MAX_ALLOWED_SPAN_LINES:
            continue

        key = (start, end)
        if key in seen:
            continue
        seen.add(key)

        score = 0.0
        # Symbol-aligned spans are the most trustworthy anchors.
        if source == "symbol_region":
            score += 12.0
        elif source == "symbol_region_expanded":
            score += 9.0
        elif source == "expanded_context":
            score += 5.0
        elif source == "merged_changed_spans":
            score += 3.0

        # Penalise broad merged spans more aggressively.
        if source == "merged_changed_spans" and length > PREFERRED_SPAN_MAX:
            score -= 6.0

        if PREFERRED_SPAN_MIN <= length <= PREFERRED_SPAN_MAX:
            score += 8.0
        elif MIN_READABLE_SPAN_LINES <= length < PREFERRED_SPAN_MIN:
            score += 4.0
        elif PREFERRED_SPAN_MAX < length <= 80:
            score += 1.0
        else:
            score -= 8.0

        preview = extract_span_preview(root_source, start, end, max_lines=10)
        noisy, flags = is_noisy_snippet(preview)
        if noisy:
            score -= 4.0

        out.append({
            "span": [start, end],
            "source": source,
            "preview": preview,
            "heuristic_score": round(score, 3),
            "noise_flags": flags if noisy else [],
        })

    out.sort(key=lambda c: (-float(c.get("heuristic_score", 0.0)), c["span"][0], c["span"][1]))
    return out[: max(2, min(int(limit), 5))]


def build_related_snippet_candidates(
    root_file_guess: str,
    root_symbol_candidates: list[dict[str, Any]],
    evidence_list: list[FileChangeEvidence],
    repo_snapshot: dict[str, str],
    relevant_files: list[dict[str, Any]],
    dep_map: dict[str, list[str]],
    reverse_dep_map: dict[str, list[str]],
    changed_py_files: list[str],
    limit: int = 8,
) -> list[dict[str, Any]]:
    evidence_by_path = {e.file_path: e for e in evidence_list}
    relevant_paths = [
        item.get("file_path", "")
        for item in sorted(relevant_files, key=lambda it: it.get("distance", 9.9))
        if item.get("file_path")
    ]
    root_leaf = ""
    for c in root_symbol_candidates:
        sym = str(c.get("symbol", "") or "")
        if sym:
            root_leaf = sym.split(".")[-1]
            break

    recall: list[tuple[str, str]] = []
    for p in changed_py_files:
        if p and p != root_file_guess:
            recall.append((p, "other_changed_file"))
    for p in dependency_neighbors(root_file_guess, dep_map, reverse_dep_map, limit=8):
        if p and p != root_file_guess:
            recall.append((p, "dependency_neighbor"))
    for p in reverse_dep_map.get(root_file_guess, [])[:8]:
        if p and p != root_file_guess:
            recall.append((p, "caller"))
    for p in relevant_paths[:10]:
        if p and p != root_file_guess:
            recall.append((p, "relevant_hint"))

    scored: list[dict[str, Any]] = []
    seen = set()

    for path, provenance in recall:
        if path in seen:
            continue
        seen.add(path)

        source = repo_snapshot.get(path, "")
        if not str(source or "").strip():
            continue

        snippet = ""
        score = 0.0
        noise_flags: list[str] = []

        ev = evidence_by_path.get(path)

        # 1) own changed region
        if ev and ev.primary_changed_span and (ev.after_source or ev.before_source):
            s = ev.after_source or ev.before_source
            start = max(1, int(ev.primary_changed_span[0]) - 6)
            end = int(ev.primary_changed_span[1]) + 6
            cand = span_context_snippet(s, start, end, padding=0, max_lines=18)
            noisy, flags = is_noisy_snippet(cand)
            if not noisy:
                snippet = cand
                score += 12.0
            else:
                noise_flags.extend(flags)

        # 2) root symbol callsite/use region
        if not snippet and root_leaf:
            lines = source.splitlines()
            for i, line in enumerate(lines[:900]):
                if root_leaf in line:
                    cand = span_context_snippet(source, i + 1, i + 1, padding=5, max_lines=18)
                    noisy, flags = is_noisy_snippet(cand)
                    if not noisy:
                        snippet = cand
                        score += 10.0
                        break
                    noise_flags.extend(flags)

        # 3) first clean def/class region
        if not snippet:
            lines = source.splitlines()
            for i, line in enumerate(lines[:700]):
                if line.lstrip().startswith(("def ", "class ")):
                    cand = span_context_snippet(source, i + 1, i + 1, padding=6, max_lines=20)
                    noisy, flags = is_noisy_snippet(cand)
                    if not noisy:
                        snippet = cand
                        score += 7.0
                        break
                    noise_flags.extend(flags)

        # 4) fallback file head
        if not snippet:
            cand = numbered_file_snippet(source, max_lines=24)
            noisy, flags = is_noisy_snippet(cand)
            snippet = cand
            if noisy:
                noise_flags.extend(flags)
                score -= 8.0
            else:
                score += 2.0

        scored.append({
            "file_path": path,
            "snippet": truncate_text_for_audit(snippet, max_chars=900),
            "provenance": provenance,
            "heuristic_score": round(score, 3),
            "noise_flags": dedupe_preserve(noise_flags),
        })

    scored.sort(key=lambda c: (-float(c.get("heuristic_score", 0.0)), str(c.get("file_path", ""))))

    non_noisy = [c for c in scored if not c.get("noise_flags")]
    if len(non_noisy) >= NON_NOISY_RELATED_MIN:
        return non_noisy[: max(3, min(int(limit), 8))]
    return scored[: max(3, min(int(limit), 8))]


def build_task_summary_coordinate_compression(
    row: dict[str, Any],
    gold_evidence: FileChangeEvidence,
    changed_py_files: list[str],
    repo_snapshot: dict[str, str],
    dep_map: dict[str, list[str]],
    reverse_dep_map: dict[str, list[str]],
    relevant_files: list[dict[str, Any]],
    qa: QATracker,
) -> list[dict[str, Any]]:
    low_semantic, reason = is_low_semantic_commit(row, gold_evidence.patch_text, changed_py_files)
    if low_semantic:
        qa.skip(f"skip_summary_{reason}")
        return []
    gold_symbol = choose_clear_gold_symbol(gold_evidence.changed_symbols, gold_evidence.patch_text)
    if gold_symbol is None:
        qa.skip("ambiguous_summary_terminal_symbol")
        return []

    relevant_paths = [item["file_path"] for item in sorted(relevant_files, key=lambda item: item["distance"]) if is_python_file(item["file_path"])]
    upstream_paths = upstream_dependency_candidates(gold_evidence.file_path, reverse_dep_map, max_depth=3)
    chain = choose_dependency_chain_to_gold(
        gold_evidence.file_path,
        [file_path for file_path in changed_py_files if file_path != gold_evidence.file_path] + relevant_paths[:4] + upstream_paths[:4],
        dep_map,
    )
    if len(chain) < 2:
        qa.skip("no_focus_chain_for_summary")
        return []

    focus_files = dedupe_preserve(
        chain
        + dependency_neighbors(gold_evidence.file_path, dep_map, reverse_dep_map, limit=2)
        + relevant_paths[:2]
        + [file_path for file_path in changed_py_files if file_path != gold_evidence.file_path][:2]
    )[:5]
    if len(focus_files) < 2:
        qa.skip("generic_focus_plan")
        return []

    focus_symbols = [{
        "file_path": gold_evidence.file_path,
        "symbol": gold_symbol["qualname"],
        "kind": gold_symbol["kind"],
        "span": [gold_symbol["lineno"], gold_symbol["end_lineno"]],
    }]
    compressed_summary = {
        "entry_file": chain[0],
        "terminal_file": gold_evidence.file_path,
        "terminal_symbol": gold_symbol["qualname"],
        "read_order": focus_files,
        "focus_tree": build_focus_tree(repo_snapshot, dep_map, focus_files, set(changed_py_files)),
    }
    sample = {
        "task_type": "summary_coordinate_compression",
        "input": {
            "policy_goal": "compress_reading_plan_into_focus_map",
            "commit_message": normalize_text(row.get("commit_message", ""), 500),
            "repo_tree_structure": compact_repo_tree(row.get("repo_tree_structure", ""), focus_files),
            "changed_files_hint": changed_py_files[:6],
            "relevant_files_hint": [
                {"file_path": item["file_path"], "distance": item["distance"]}
                for item in sorted(relevant_files, key=lambda item: item["distance"])[:5]
            ],
        },
        "focus_files": focus_files,
        "focus_symbols": focus_symbols,
        "dependency_chain": chain,
        "compressed_summary": compressed_summary,
        "rationale": (
            f"The focus map keeps the read order compact while preserving the changed endpoint `{gold_evidence.file_path}` "
            f"and symbol `{gold_symbol['qualname']}`."
        ),
        "metadata": {
            "repo": str(row.get("repo_name", "")),
            "commit_id": str(row.get("commit_id", "")),
        },
    }
    if validate_sample(sample):
        return [sample]
    qa.skip("weak_summary_sample")
    return []


# ── pretrain serialization ────────────────────────────────────────────────────


def build_prompt_payload(sample: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "task_type": sample.get("task_type", "unknown"),
        "input": sample.get("input", {}),
    }
    if sample.get("candidates"):
        payload["candidates"] = sample.get("candidates", [])
    return payload


def normalize_working_summary_payload(ws: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(ws, dict):
        return {}
    out: dict[str, Any] = {}
    likely_root_file = str(ws.get("likely_root_file", "") or "")
    if likely_root_file:
        out["likely_root_file"] = likely_root_file
    region = ws.get("likely_focus_region")
    if is_valid_span(region):
        out["likely_focus_region"] = [int(region[0]), int(region[1])]
    supporting_files = ws.get("supporting_files", [])
    if isinstance(supporting_files, list):
        cleaned_supporting = [str(x) for x in supporting_files if str(x)]
        if cleaned_supporting:
            out["supporting_files"] = cleaned_supporting
    next_question = str(ws.get("next_question", "") or "").strip()
    if next_question:
        out["next_question"] = next_question
    return out


def build_target_payload(sample: dict[str, Any]) -> dict[str, Any]:
    task_type = sample.get("task_type")
    target = sample.get("target", {}) if isinstance(sample.get("target", {}), dict) else {}
    if task_type in {"patch_grounding", "ast_dependency_trace"}:
        selected_action = target.get("selected_action", {})
        if not is_public_action_payload(selected_action):
            return {}
        # hard whitelist: never allow gold_* or any extra fields into target
        return {"selected_action": strip_action_meta(selected_action)}
    if task_type == "reading_summary":
        ws = normalize_working_summary_payload(target.get("working_summary", {}))
        if not ws:
            return {}
        # hard whitelist: only working_summary survives
        return {"working_summary": ws}
    return {}


def sample_dedupe_key(sample: dict[str, Any]) -> tuple[str, str, str, str]:
    metadata = sample.get("metadata", {}) if isinstance(sample, dict) else {}
    repo = str(metadata.get("repo", "") or "")
    commit_id = str(metadata.get("commit_id", "") or "")
    task_type = str(sample.get("task_type", "") or "")
    normalized_target = stable_json(build_target_payload(sample))
    return (repo, commit_id, task_type, normalized_target)


def dedupe_samples(
    samples: list[dict[str, Any]],
    qa: QATracker | None = None,
    skip_reason: str = "duplicate_sample_same_commit_task_target",
) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str]] = set()
    out: list[dict[str, Any]] = []
    for sample in samples:
        key = sample_dedupe_key(sample)
        if key in seen:
            if qa is not None:
                qa.skip(skip_reason)
            continue
        seen.add(key)
        out.append(sample)
    return out


def format_prompt_text(sample: dict[str, Any]) -> str:
    input_data = sample.get("input", {})
    header = [
        f"Task Type: {sample.get('task_type', 'unknown')}",
        f"Goal Source: {input_data.get('goal_source', '')}",
        f"Subtask: {input_data.get('subtask', '')}",
    ]
    if input_data.get("goal_text"):
        header.append("Goal Text: " + str(input_data.get("goal_text", "")))
    if input_data.get("task_instruction"):
        header.append("Task Instruction: " + str(input_data.get("task_instruction", "")))

    blocks = ["\n".join([line for line in header if line.strip()])]

    if input_data.get("changed_files_hint"):
        blocks.append(format_list_block("Changed Python Files", compact_bullets(input_data.get("changed_files_hint", []), limit=10)))

    if input_data.get("current_hypothesis"):
        blocks.append(format_key_value_block("Current Hypothesis", input_data.get("current_hypothesis", "")))

    obs = input_data.get("current_observation", {})
    if isinstance(obs, dict) and obs:
        if obs.get("opened_file"):
            blocks.append(format_key_value_block("Opened File", obs.get("opened_file", "")))
        if obs.get("entry_file"):
            blocks.append(format_key_value_block("Entry File", obs.get("entry_file", "")))
        if obs.get("current_snippet"):
            blocks.append(format_key_value_block("Current Snippet", truncate_text_for_audit(obs.get("current_snippet", ""), 900)))
        if obs.get("entry_snippet"):
            blocks.append(format_key_value_block("Entry Snippet", truncate_text_for_audit(obs.get("entry_snippet", ""), 900)))
        if obs.get("file_ast_outline"):
            outline_lines = [
                f"{item.get('qualname', '')} [{item.get('kind', '')}] @ {item.get('lineno', '')}-{item.get('end_lineno', '')}"
                for item in obs.get("file_ast_outline", [])
            ]
            blocks.append(format_list_block("AST Outline", outline_lines[:25]))
        if obs.get("entry_file_ast_outline"):
            outline_lines = [
                f"{item.get('qualname', '')} [{item.get('kind', '')}] @ {item.get('lineno', '')}-{item.get('end_lineno', '')}"
                for item in obs.get("entry_file_ast_outline", [])
            ]
            blocks.append(format_list_block("Entry AST Outline", outline_lines[:25]))
        if isinstance(obs.get("dependency_context"), dict) and obs.get("dependency_context"):
            ctx = obs.get("dependency_context")
            lines = []
            if ctx.get("entry_imports"):
                lines.append("entry_imports: " + ", ".join(str(x) for x in ctx.get("entry_imports", [])[:8]))
            if ctx.get("root_importers"):
                lines.append("root_importers: " + ", ".join(str(x) for x in ctx.get("root_importers", [])[:8]))
            if lines:
                blocks.append(format_key_value_block("Dependency Context", "\n".join(lines)))
        if obs.get("related_snippets"):
            related = [
                f"{item.get('file_path', '')} [{item.get('provenance', '')}]:\n{truncate_text_for_audit(item.get('snippet', ''), 600)}"
                for item in obs.get("related_snippets", [])
            ]
            blocks.append(format_key_value_block("Related Snippets", "\n\n".join(related)))

    if isinstance(input_data.get("trace_state"), dict) and input_data.get("trace_state"):
        blocks.append(format_key_value_block("Trace State", json.dumps(input_data.get("trace_state", {}), ensure_ascii=False)))

    if isinstance(input_data.get("read_history"), list) and input_data.get("read_history"):
        rendered = []
        for idx, step in enumerate(input_data.get("read_history", [])[:4]):
            act = json.dumps(step.get("action", {}), ensure_ascii=False)
            obs_snip = truncate_text_for_audit(str(step.get("observation", {}).get("snippet", "") or ""), 700)
            rendered.append(f"Step {idx}: {act}\n{obs_snip}")
        blocks.append(format_key_value_block("Read History", "\n\n".join(rendered)))

    if input_data.get("available_tools"):
        blocks.append(format_list_block("Available Tools", [str(x) for x in input_data.get("available_tools", [])]))

    available_actions = input_data.get("available_actions", [])
    if isinstance(available_actions, list) and available_actions:
        lines = []
        for idx, action in enumerate(available_actions):
            rendered = json.dumps(strip_action_meta(action), ensure_ascii=False, sort_keys=True)
            lines.append(f"{idx}: {rendered}")
        blocks.append(format_key_value_block("Available Actions", "\n".join(lines)))

    return "\n\n".join(block for block in blocks if block).strip()


def extract_answer_from_text(text: str) -> str:
    marker = f"{MASK_END}\n"
    if marker not in text:
        return ""
    return text.split(marker, 1)[1].strip()


def validate_serialized_record(sample: dict[str, Any], serialized_record: dict[str, Any]) -> tuple[bool, str]:
    if sample.get("task_type") not in {"patch_grounding", "ast_dependency_trace", "reading_summary"}:
        return False, "unsupported_task_type"

    rebuilt_target = build_target_payload(sample)
    expected_target_text = json.dumps(rebuilt_target, ensure_ascii=False, sort_keys=True, indent=2)

    stored_target_text = str(serialized_record.get("target_text", "") or "")
    if stored_target_text.strip() != expected_target_text.strip():
        return False, "target_text_payload_mismatch"

    task_type = sample.get("task_type")
    if task_type in {"patch_grounding", "ast_dependency_trace"}:
        selected_action = rebuilt_target.get("selected_action")
        if not isinstance(selected_action, dict) or not selected_action:
            return False, "missing_or_invalid_selected_action"
        if not is_public_action_payload(selected_action):
            return False, "non_public_selected_action_schema"
        if "selected_action_id" in rebuilt_target or "best_action_id" in rebuilt_target:
            raise RuntimeError(
                f"[{PIPELINE_VERSION}] Old schema field in serialized target for {task_type}: "
                f"keys={list(rebuilt_target.keys())}. Pipeline bug — fix the builder."
            )

    if task_type == "reading_summary":
        ws = rebuilt_target.get("working_summary")
        if not isinstance(ws, dict) or not ws.get("likely_root_file"):
            return False, "missing_working_summary"

    if extract_answer_from_text(serialized_record.get("text", "")) != stored_target_text.strip():
        return False, "text_answer_mismatch"

    raw_sample_json = serialized_record.get("raw_sample_json", "")
    if raw_sample_json:
        try:
            raw_sample = json.loads(raw_sample_json)
        except Exception:
            return False, "raw_sample_json_decode_error"
        if raw_sample != sample:
            return False, "raw_sample_json_mismatch"

    return True, ""


def should_keep_sample_for_mode(sample: dict[str, Any], dataset_mode: str) -> bool:
    # The training set is intentionally restricted to the 3 grounded tasks.
    # `dataset_mode` is kept for backward compatibility, but no longer
    # re-enables legacy proxy tasks.
    return sample.get("task_type") in {"patch_grounding", "ast_dependency_trace", "reading_summary"}


def serialize_sample_for_pretrain(sample: dict[str, Any]) -> dict[str, Any]:
    target_payload = build_target_payload(sample)
    prompt_text = format_prompt_text(sample)
    target_text = json.dumps(target_payload, ensure_ascii=False, sort_keys=True, indent=2)
    # Only tokens after MASK_END are intended to contribute to loss;
    # everything inside MASK_BEGIN / MASK_END is conditioning context only.
    # No outer <sample> wrapper.
    text = f"{MASK_BEGIN}\n{prompt_text}\n{MASK_END}\n{target_text}"
    metadata = sample.get("metadata", {})
    task_type = sample.get("task_type", "unknown")
    repo = str(metadata.get("repo", ""))
    commit_id = str(metadata.get("commit_id", ""))
    source = str(
        metadata.get("gold_file", "")
        or metadata.get("entry_file", "")
        or f"{repo}/{task_type}"
    )

    record = {
        "task_type": task_type,
        "repo": repo,
        "commit_id": commit_id,
        "source": source,
        "prompt_text": prompt_text,
        "target_text": target_text,
        "text": text,
        "mask_begin_token": MASK_BEGIN,
        "mask_end_token": MASK_END,
    }

    # 只在显式 debug 时才保留 raw_sample_json，避免 main jsonl 看起来像“泄漏”
    if os.environ.get("KEEP_DEBUG_RAW_SAMPLE_JSON", "").strip().lower() in {"1", "true", "yes"}:
        record["raw_sample_json"] = json.dumps(sample, ensure_ascii=False)

    return record


def load_tokenizer_for_counting(tokenizer_dir: str):
    tokenizer_dir = str(tokenizer_dir or "").strip()
    if not tokenizer_dir:
        raise ValueError("tokenizer_dir is required for content_split export")
    try:
        from transformers import PreTrainedTokenizerFast  # type: ignore
        return PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
    except Exception:
        from tokenizers import Tokenizer  # type: ignore
        tokenizer_json = os.path.join(tokenizer_dir, "tokenizer.json")
        if not os.path.exists(tokenizer_json):
            raise
        return Tokenizer.from_file(tokenizer_json)


def count_tokens_with_tokenizer(text: str, tokenizer: Any) -> int:
    if hasattr(tokenizer, "encode"):
        try:
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if hasattr(encoded, "ids"):
                return len(encoded.ids)
            if isinstance(encoded, list):
                return len(encoded)
            if hasattr(encoded, "input_ids"):
                return len(encoded.input_ids)
        except TypeError:
            encoded = tokenizer.encode(text)
            if hasattr(encoded, "ids"):
                return len(encoded.ids)
            if isinstance(encoded, list):
                return len(encoded)
    raise TypeError("Unsupported tokenizer interface for token counting")


FINAL_TASK_TYPES = {"patch_grounding", "ast_dependency_trace", "reading_summary"}
TASK_QUOTAS = {t: 5 for t in FINAL_TASK_TYPES}

ALLOWED_ACTIONS: dict[str, set[str]] = {
    "patch_grounding": {"open_file", "open_symbol", "read_region"},
    "ast_dependency_trace": {"follow_dependency", "open_file", "open_symbol", "read_region"},
}

READING_SUMMARY_REQUIRED_KEYS = {"likely_focus_region", "likely_root_file", "next_question", "supporting_files"}


def build_content_split(prompt_text: str, target_text: str) -> str:
    return f"{MASK_BEGIN}\n{prompt_text}\n{MASK_END}\n{target_text}"


def make_meta(sample: dict[str, Any], accepted_index: int) -> dict[str, Any]:
    metadata = sample.get("metadata", {}) if isinstance(sample, dict) else {}
    task_type = str(sample.get("task_type", "unknown") or "unknown")
    repo = str(metadata.get("repo", "") or "unknown_repo")
    commit_id = str(metadata.get("commit_id", "") or "unknown_commit")
    source = str(
        metadata.get("gold_file", "")
        or metadata.get("entry_file", "")
        or f"{repo}/{task_type}"
    )
    docid = f"{repo}::{commit_id}::{task_type}::{accepted_index:06d}"
    chunk_id = f"{task_type}-{accepted_index:06d}"
    return {"docid": docid, "chunk_id": chunk_id, "source": source}


def validate_candidate(sample: dict[str, Any], serialized: dict[str, Any]) -> tuple[bool, str]:
    task_type = str(sample.get("task_type", "") or "")
    if task_type not in FINAL_TASK_TYPES:
        return False, f"unknown_task_type:{task_type}"
    target_payload = {}
    try:
        target_payload = json.loads(serialized.get("target_text", "") or "{}")
    except Exception:
        return False, "target_text_not_valid_json"
    selected_action = target_payload.get("selected_action", {}) if isinstance(target_payload, dict) else {}
    if task_type in ALLOWED_ACTIONS:
        action_name = str(selected_action.get("action", "") or "")
        if action_name not in ALLOWED_ACTIONS[task_type]:
            return False, f"disallowed_action:{action_name}"
    if task_type == "reading_summary":
        ws = target_payload.get("working_summary", {}) if isinstance(target_payload, dict) else {}
        if not isinstance(ws, dict):
            return False, "missing_working_summary"
        missing = READING_SUMMARY_REQUIRED_KEYS - ws.keys()
        if missing:
            return False, f"missing_working_summary_keys:{','.join(sorted(missing))}"
    return True, ""


def count_tokens(text: str, tokenizer: Any) -> int:
    return count_tokens_with_tokenizer(text, tokenizer)


def finalize_record(sample: dict[str, Any], serialized: dict[str, Any], accepted_index: int, tokenizer: Any) -> dict[str, Any]:
    prompt_text = serialized.get("prompt_text", "") or ""
    target_text = serialized.get("target_text", "") or ""
    content_split = build_content_split(prompt_text, target_text)
    return {
        "meta": make_meta(sample, accepted_index),
        "content_split": content_split,
        "token_count": count_tokens(content_split, tokenizer),
    }


def write_record(record: dict[str, Any], handle: Any) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def summarize_written_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    task_counts = Counter(sample.get("task_type", "unknown") for sample in samples)
    candidate_counts: dict[str, list[int]] = defaultdict(list)
    for sample in samples:
        if sample.get("task_type") == "intent_to_edit_sketch":
            input_data = sample.get("input", {})
            intent_candidate_count = sum(
                len(input_data.get(key, []))
                for key in (
                    "target_file_candidates",
                    "patch_type_candidates",
                    "root_symbol_candidates",
                    "risk_surface_candidates",
                )
            )
            candidate_counts[sample.get("task_type", "unknown")].append(intent_candidate_count)
        else:
            candidate_counts[sample.get("task_type", "unknown")].append(len(sample.get("candidates", [])))
    average_candidates = {
        task_type: round(sum(values) / len(values), 2) if values else 0.0
        for task_type, values in sorted(candidate_counts.items())
    }
    return {
        "total_samples": len(samples),
        "per_task_counts": dict(sorted(task_counts.items())),
        "average_candidates_per_task": average_candidates,
    }


# ── commit/project processing ─────────────────────────────────────────────────


def collect_commit_samples(
    row: dict[str, Any],
    initial_df: pd.DataFrame | None,
    qa: QATracker,
    dataset_mode: str = DEFAULT_DATASET_MODE,
) -> list[dict[str, Any]]:
    qa.total_commits_seen += 1
    changed_files = parse_changed_files(row.get("file_changed_content", ""))
    changed_py_files = [path for path in parse_changed_files(row.get("file_changed_content", "")) if is_python_file(path)]
    before_after = extract_before_after(row)
    patch_sections = parse_patch_sections(row.get("patch", ""))
    skip_reason = commit_skip_reason(row, changed_py_files, before_after, patch_sections)
    if skip_reason:
        qa.skip(skip_reason)
        return []

    repo_snapshot = build_repo_snapshot(initial_df, before_after)
    if len(repo_snapshot) < 2:
        qa.skip("repo_snapshot_too_small")
        return []

    low_semantic, low_reason = is_low_semantic_commit(row, row.get("patch", ""), changed_files)
    if low_semantic:
        qa.skip(f"drop_low_semantic_{low_reason}")
        return []

    relevant_files = parse_relevant_files(row.get("relevant_file_content"))
    try:
        dep_map = build_dep_map_from_snapshot(repo_snapshot)
    except Exception:
        dep_map = {}
        qa.skip("dep_map_build_error")
    reverse_dep_map = build_reverse_dep_map(dep_map)

    evidence_list = collect_file_evidence(changed_py_files, before_after, repo_snapshot, patch_sections)
    if not evidence_list:
        qa.skip("no_clear_gold_file")
        return []
    low_info_goal, goal_reason = is_low_information_goal_text(
        str(row.get("commit_message", "") or ""),
        changed_py_files,
        evidence_list,
    )
    if low_info_goal:
        qa.skip(f"low_information_goal_text_{goal_reason}")
        return []

    # Candidate generation + LLM judge selection for root grounding.
    root_file_candidates = build_root_file_candidates(
        row=row,
        evidence_list=evidence_list,
        changed_py_files=changed_py_files,
        relevant_files=relevant_files,
        dep_map=dep_map,
        reverse_dep_map=reverse_dep_map,
        limit=5,
    )
    if not root_file_candidates:
        qa.skip("root_grounding_drop_no_root_file_candidates")
        qa.observe_root_grounding_decision("drop", None, False, "no_root_file_candidates", "root_grounding")
        return []

    if not root_file_margin_ok(root_file_candidates):
        qa.skip("root_grounding_drop_ambiguous_root_file")
        qa.observe_root_grounding_decision(
            "drop",
            None,
            False,
            "ambiguous_root_file",
            "root_grounding",
        )
        return []

    evidence_by_path = {e.file_path: e for e in evidence_list}
    # Build symbol/span candidates across all file candidates, tagged with file_id.
    root_symbol_candidates: list[dict[str, Any]] = []
    root_span_candidates: list[dict[str, Any]] = []
    for file_id, fc in enumerate(root_file_candidates):
        fp = str(fc.get("file_path", ""))
        ev = evidence_by_path.get(fp)
        if not ev:
            continue
        syms = build_root_symbol_candidates(row=row, file_evidence=ev, limit=6)
        for s in syms:
            root_symbol_candidates.append({"file_id": file_id, "file_path": fp, **s})
        source = ev.after_source or ev.before_source
        spans = build_root_span_candidates(ev, source, limit=5) if source else []
        for sp in spans:
            root_span_candidates.append({"file_id": file_id, "file_path": fp, **sp})

    # Use the top heuristic file as a guess for recalling related snippet candidates.
    root_file_guess = str(root_file_candidates[0].get("file_path", ""))
    related_snippet_candidates = build_related_snippet_candidates(
        root_file_guess=root_file_guess,
        root_symbol_candidates=root_symbol_candidates,
        evidence_list=evidence_list,
        repo_snapshot=repo_snapshot,
        relevant_files=relevant_files,
        dep_map=dep_map,
        reverse_dep_map=reverse_dep_map,
        changed_py_files=changed_py_files,
        limit=8,
    )

    dep_context = {
        "root_file_guess": root_file_guess,
        "root_file_guess_importers": reverse_dep_map.get(root_file_guess, [])[:8],
        "root_file_guess_imports": dep_map.get(root_file_guess, [])[:8],
    }

    selection, llm_meta = select_root_grounding_with_llm(
        row=row,
        root_file_candidates=root_file_candidates,
        root_symbol_candidates=root_symbol_candidates,
        root_span_candidates=root_span_candidates,
        related_snippet_candidates=related_snippet_candidates,
        dep_context=dep_context,
    )

    if selection is None:
        drop_reason = str(llm_meta.get("root_grounding_drop_reason", "root_grounding_drop") or "root_grounding_drop")
        qa.skip(f"root_grounding_drop_{drop_reason}")
        qa.observe_root_grounding_decision(
            "drop",
            llm_meta.get("llm_root_grounding_confidence"),
            bool(llm_meta.get("used_root_grounding_fallback")),
            drop_reason,
            "root_grounding",
        )
        return []

    # Resolve the chosen grounding.
    chosen_file_id = int(selection.get("best_root_file_id", 0) or 0)
    chosen_file = str(root_file_candidates[chosen_file_id].get("file_path", ""))
    chosen_evidence = evidence_by_path.get(chosen_file)
    if not chosen_evidence:
        qa.skip("root_grounding_drop_missing_chosen_evidence")
        qa.observe_root_grounding_decision("drop", llm_meta.get("llm_root_grounding_confidence"), bool(llm_meta.get("used_root_grounding_fallback")), "missing_chosen_evidence", "root_grounding")
        return []

    chosen_span_id = int(selection.get("best_root_span_id", 0) or 0)
    chosen_span = root_span_candidates[chosen_span_id]
    if int(chosen_span.get("file_id", -1)) != chosen_file_id:
        qa.skip("root_grounding_drop_span_file_mismatch")
        qa.observe_root_grounding_decision("drop", llm_meta.get("llm_root_grounding_confidence"), bool(llm_meta.get("used_root_grounding_fallback")), "span_file_mismatch", "root_grounding")
        return []

    chosen_symbol = ""
    if root_symbol_candidates:
        chosen_symbol_id = int(selection.get("best_root_symbol_id", 0) or 0)
        sym = root_symbol_candidates[chosen_symbol_id]
        if int(sym.get("file_id", -1)) != chosen_file_id:
            qa.skip("root_grounding_drop_symbol_file_mismatch")
            qa.observe_root_grounding_decision("drop", llm_meta.get("llm_root_grounding_confidence"), bool(llm_meta.get("used_root_grounding_fallback")), "symbol_file_mismatch", "root_grounding")
            return []
        chosen_symbol = str(sym.get("symbol", "") or "")

    keep_related_ids = selection.get("keep_related_snippet_ids", [])
    selected_related = []
    for idx in keep_related_ids:
        try:
            selected_related.append(related_snippet_candidates[int(idx)])
        except Exception:
            continue
    selected_related = selected_related[:3]
    if not selected_related:
        qa.skip("root_grounding_drop_no_related")
        qa.observe_root_grounding_decision("drop", llm_meta.get("llm_root_grounding_confidence"), bool(llm_meta.get("used_root_grounding_fallback")), "no_related", "root_grounding")
        return []

    grounding = {
        "root_file": chosen_file,
        "root_symbol": chosen_symbol,
        "root_line_span": [int(chosen_span["span"][0]), int(chosen_span["span"][1])],
        "root_span_preview": chosen_span.get("preview", ""),
        "related_snippets": selected_related,
        "root_file_candidates": root_file_candidates,
        "root_symbol_candidates": root_symbol_candidates,
        "root_span_candidates": root_span_candidates,
        "related_snippet_candidates": related_snippet_candidates,
        **llm_meta,
    }

    qa.observe_commit(len(changed_py_files), len(chosen_evidence.changed_symbols))
    qa.observe_root_grounding_decision(
        "keep",
        llm_meta.get("llm_root_grounding_confidence"),
        bool(llm_meta.get("used_root_grounding_fallback")),
        "",
        "root_grounding",
    )

    # Only emit the 3 grounded tasks.
    # Pre-build all 4 sub-task prompts and fire them in a single parallel batch,
    # replacing 4 sequential LLM calls with 1 async gather.
    goal_text = normalize_text(row.get("commit_message", ""), 500)
    subtask_prompts: dict[str, dict[str, Any]] = {}

    # --- patch_grounding prompts ---
    _pg_root_source = repo_snapshot.get(chosen_file, "")
    _pg_outline = build_file_outline(_pg_root_source, chosen_file)
    _pg_related = grounding.get("related_snippets", [])
    _pg_issues = grounding.get("llm_root_grounding_issues", []) if isinstance(grounding.get("llm_root_grounding_issues", []), list) else []
    _pg_symbol_candidates: list[dict[str, Any]] = []
    for group in (
        chosen_evidence.changed_symbols,
        grounding.get("root_symbol_candidates", []),
        chosen_evidence.symbol_pool,
    ):
        if isinstance(group, list):
            _pg_symbol_candidates.extend(item for item in group if isinstance(item, dict))
    _pg_symbol_candidates = filter_symbol_candidates_for_file(_pg_symbol_candidates, chosen_file)
    _pg_symbol_source = choose_symbol_resolution_source(chosen_symbol, chosen_evidence, default_source=_pg_root_source)
    if _pg_outline:
        _pg_action_cands = build_available_actions_for_patch_grounding(
            root_file=chosen_file,
            root_symbol=chosen_symbol,
            root_line_span=[int(grounding["root_line_span"][0]), int(grounding["root_line_span"][1])],
            file_ast_outline=_pg_outline,
            related_snippets=_pg_related,
            source=_pg_symbol_source,
            symbol_candidates=_pg_symbol_candidates,
            issues=_pg_issues,
        )
        subtask_prompts["pg_action"] = _build_action_selection_prompt(
            "patch_grounding", "narrow_patch_region", goal_text, _pg_action_cands
        )
    _pg_inst_cands = build_task_instruction_candidates("patch_grounding", "narrow_patch_region")
    subtask_prompts["pg_instruction"] = _build_instruction_selection_prompt(
        "patch_grounding", "narrow_patch_region", goal_text, _pg_inst_cands
    )

    # --- ast_dependency_trace prompts (only if dep_map exists) ---
    if dep_map:
        _dt_relevant_paths = [item["file_path"] for item in sorted(relevant_files, key=lambda item: item["distance"]) if is_python_file(item["file_path"])]
        _dt_upstream = upstream_dependency_candidates(chosen_file, reverse_dep_map, max_depth=3)
        _dt_entries = [f for f in changed_py_files if f != chosen_file] + _dt_relevant_paths[:6] + _dt_upstream[:6]
        _dt_chain = choose_dependency_chain_to_gold(chosen_file, _dt_entries, dep_map)
        if len(_dt_chain) >= 2:
            _dt_next_file = _dt_chain[1]
            _dt_next_src = repo_snapshot.get(_dt_next_file, "")
            _dt_next_outline = build_file_outline(_dt_next_src, _dt_next_file) if _dt_next_src else []
            _dt_action_cands = build_available_actions_for_trace(
                next_file=_dt_next_file,
                root_file=chosen_file,
                next_file_source=_dt_next_src,
                next_file_outline=_dt_next_outline,
            )
            subtask_prompts["dt_action"] = _build_action_selection_prompt(
                "ast_dependency_trace", "follow_dependency", goal_text, _dt_action_cands
            )
    _dt_inst_cands = build_task_instruction_candidates("ast_dependency_trace", "follow_dependency")
    subtask_prompts["dt_instruction"] = _build_instruction_selection_prompt(
        "ast_dependency_trace", "follow_dependency", goal_text, _dt_inst_cands
    )

    # Fire all sub-task LLM calls in parallel.
    preresolved_llm = run_grounding_subtasks_batch(subtask_prompts)

    new_samples: list[dict[str, Any]] = []
    new_samples.extend(build_task_patch_grounding(
        row=row,
        gold_evidence=chosen_evidence,
        changed_py_files=changed_py_files,
        repo_snapshot=repo_snapshot,
        relevant_files=relevant_files,
        dep_map=dep_map,
        reverse_dep_map=reverse_dep_map,
        qa=qa,
        grounding=grounding,
        preresolved_llm=preresolved_llm,
    ))
    new_samples.extend(build_task_ast_dependency_trace(
        row=row,
        gold_evidence=chosen_evidence,
        changed_py_files=changed_py_files,
        repo_snapshot=repo_snapshot,
        relevant_files=relevant_files,
        dep_map=dep_map,
        reverse_dep_map=reverse_dep_map,
        qa=qa,
        grounding=grounding,
        preresolved_llm=preresolved_llm,
    ))
    new_samples.extend(build_task_reading_summary(
        row=row,
        gold_evidence=chosen_evidence,
        changed_py_files=changed_py_files,
        repo_snapshot=repo_snapshot,
        relevant_files=relevant_files,
        dep_map=dep_map,
        reverse_dep_map=reverse_dep_map,
        qa=qa,
        grounding=grounding,
    ))

    valid_samples = []
    for sample in new_samples:
        if not should_keep_sample_for_mode(sample, dataset_mode):
            qa.skip(f"filtered_by_dataset_mode_{dataset_mode}")
            continue
        if validate_sample(sample):
            valid_samples.append(sample)
            qa.observe_sample(sample)
        else:
            task_type = sample.get("task_type", "sample")
            qa.skip(f"invalid_{task_type}")
            # Increment fine-grained rejection counters from metadata.
            meta = sample.get("metadata", {})
            drop = str(meta.get("next_action_drop_reason", "") or "")
            ws_drop = str(meta.get("working_summary_drop_reason", "") or "")
            rg_drop = str(meta.get("root_grounding_drop_reason", "") or "")
            if drop == "generic_action_reason":
                qa.rejected_generic_action_reason += 1
            if drop == "llm_call_failed_no_strong_fallback":
                qa.rejected_fallback_margin_too_small += 1
            if ws_drop in ("generic_next_question", "placeholder_next_question"):
                qa.rejected_reading_summary_generic_nq += 1
            if ws_drop == "placeholder_focus_region":
                qa.rejected_reading_summary_placeholder_region += 1
            if ws_drop in ("invented_supporting_file", "empty_supporting_files_no_repair"):
                qa.rejected_reading_summary_ungrounded_supporting += 1
            if rg_drop == "all_related_snippets_noisy":
                qa.rejected_root_grounding_noisy_related += 1
            if rg_drop in ("span_preview_empty", "span_too_large"):
                qa.rejected_root_grounding_placeholder_span += 1

    if not valid_samples:
        qa.skip("no_tasks_emitted")
    return valid_samples


def process_project(
    project_dir: str,
    quota: int,
    qa: QATracker,
    return_judge_report: bool = False,
    dataset_mode: str = DEFAULT_DATASET_MODE,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    commit_path = os.path.join(project_dir, "commit_data.parquet")
    initial_path = os.path.join(project_dir, "initial_code.parquet")
    if not os.path.exists(commit_path):
        qa.skip("missing_commit_parquet")
        empty_report = {
            "repo": "",
            "commit_decisions": [],
            "merge_groups": [],
            "drop_commit_ids": [],
            "keep_separate_commit_ids": [],
            "merge_candidate_pairs": [],
        }
        if return_judge_report:
            return [], empty_report
        return []

    commit_df = pd.read_parquet(commit_path)
    commit_df = commit_df[commit_df["main_language"].astype(str).str.lower().str.contains("python", na=False)]
    commit_df = commit_df[commit_df["patch"].astype(str).str.contains(r"\.py", na=False)]
    initial_df = pd.read_parquet(initial_path) if os.path.exists(initial_path) else None
    project_name = os.path.basename(project_dir)
    # Limit per-project scan to `quota` commits to avoid spending excessive
    # time in very large repositories when strict validation drops most samples.
    commit_iter = _make_tqdm(
        commit_df.head(quota).iterrows(),
        total=min(len(commit_df), quota),
        desc=f"  {project_name}",
        unit="commit",
        leave=False,
    )
    for _, row in commit_iter:
        if len(samples) >= quota:
            break
        row_dict = row.to_dict()
        samples.extend(collect_commit_samples(row_dict, initial_df, qa, dataset_mode=dataset_mode))
    if return_judge_report:
        # Merge logic is intentionally not wired into training-unit
        # construction; return a minimal stub report for compatibility.
        empty_report = {
            "repo": str(commit_df.iloc[0].get("repo_name", "")) if not commit_df.empty else "",
            "commit_decisions": [],
            "merge_groups": [],
            "drop_commit_ids": [],
            "keep_separate_commit_ids": [],
            "merge_candidate_pairs": [],
        }
        return samples[:quota], empty_report
    return samples[:quota]


def write_outputs(
    samples: list[dict[str, Any]],
    output_jsonl: str,
    output_parquet: str,
    output_qa_json: str,
    qa: QATracker,
    output_raw_samples_jsonl: str = "",
    judge_reports: list[dict[str, Any]] | None = None,
    output_judge_json: str = "",
    output_audit_jsonl: str = "",
    output_content_split_jsonl: str = "",
    tokenizer_dir: str = DEFAULT_TOKENIZER_DIR,
    per_task_target: int = 0,
    final_task_types: list[str] | None = None,
) -> None:
    samples = dedupe_samples(samples, qa=None)
    final_task_set = set(final_task_types or FINAL_TASK_TYPES)
    quota = per_task_target if per_task_target > 0 else 5
    accepted_counts: Counter = Counter()
    accepted_total = 0
    sample_quality_rows: list[dict[str, Any]] = []
    all_judge_reports_local: list[dict[str, Any]] = []

    tokenizer = load_tokenizer_for_counting(tokenizer_dir)

    with open(output_jsonl, "w", encoding="utf-8") as out_handle:
        for sample in samples:
            task_type = str(sample.get("task_type", "") or "")
            if task_type not in final_task_set:
                continue
            if accepted_counts[task_type] >= quota:
                qa.skip("dropped_due_to_task_quota")
                continue

            serialized = serialize_sample_for_pretrain(sample)

            judged_sample, judge_meta = judge_sample_with_llm(sample, serialized)
            metadata = sample.get("metadata", {}) if isinstance(sample, dict) else {}
            sample_quality_rows.append({
                "repo": str(metadata.get("repo", "")),
                "commit_id": str(metadata.get("commit_id", "")),
                "task_type": task_type,
                "decision": str(judge_meta.get("sample_quality_decision", "") or ""),
                "confidence": judge_meta.get("sample_quality_confidence"),
                "reason": str(judge_meta.get("sample_quality_reason", "") or ""),
                "repair_applied": bool(judge_meta.get("sample_quality_repair_applied", False)),
                "used_fallback_repair": bool(judge_meta.get("sample_quality_used_fallback_repair", False)),
            })
            if judged_sample is None:
                drop_reason = str(judge_meta.get("sample_quality_reason", "") or "sample_quality_drop")
                qa.skip(f"sample_quality_{drop_reason[:80]}")
                qa.record_audit(
                    sample,
                    validation_status="rejected",
                    skip_reason="sample_quality_drop",
                    prompt_text=serialized.get("prompt_text", ""),
                    target_text=serialized.get("target_text", ""),
                    leakage_flags=[],
                    task_purity_status="not_checked",
                    rejection_reason=drop_reason,
                    bucket="rejected",
                )
                continue
            if judged_sample is not sample:
                sample = judged_sample
                serialized = serialize_sample_for_pretrain(sample)

            is_valid, reason = validate_serialized_record(sample, serialized)
            if not is_valid:
                qa.skip(f"serialized_record_{reason}")
                qa.record_audit(
                    sample,
                    validation_status="rejected",
                    skip_reason=reason,
                    prompt_text=serialized.get("prompt_text", ""),
                    target_text=serialized.get("target_text", ""),
                    leakage_flags=[],
                    task_purity_status="not_checked",
                    rejection_reason=reason,
                    bucket="rejected",
                )
                continue

            purity_ok, leakage_flags, purity_reason = validate_task_purity(sample, serialized)
            if not purity_ok:
                if purity_reason == "patch_leakage":
                    skip_reason = "rejected_due_to_patch_leakage"
                elif purity_reason == "added_symbol_visibility_mismatch":
                    skip_reason = "rejected_due_to_added_symbol_visibility_mismatch"
                elif purity_reason == "overstrict_message_overlap":
                    skip_reason = "rejected_due_to_overstrict_message_overlap"
                elif purity_reason == "unwired_merge_logic":
                    skip_reason = "rejected_due_to_unwired_merge_logic"
                else:
                    skip_reason = f"task_purity_{purity_reason or 'unknown'}"
                qa.skip(skip_reason)
                qa.record_audit(
                    sample,
                    validation_status="rejected",
                    skip_reason=skip_reason,
                    prompt_text=serialized.get("prompt_text", ""),
                    target_text=serialized.get("target_text", ""),
                    leakage_flags=leakage_flags,
                    task_purity_status="rejected",
                    rejection_reason=purity_reason,
                    bucket="rejected",
                )
                continue

            ok, reason = validate_candidate(sample, serialized)
            if not ok:
                qa.skip(f"candidate_validation_{reason}")
                continue

            record = finalize_record(sample, serialized, accepted_total, tokenizer)
            write_record(record, out_handle)
            accepted_counts[task_type] += 1
            accepted_total += 1

            bucket = "main_message_localization_bucket"
            metadata_accepted = sample.get("metadata", {})
            if metadata_accepted.get("used_action_selection_fallback"):
                if task_type == "patch_grounding":
                    qa.accepted_patch_grounding_fallback_count += 1
                elif task_type == "ast_dependency_trace":
                    qa.accepted_ast_dependency_trace_fallback_count += 1
            if task_type == "reading_summary" and metadata_accepted.get("used_working_summary_fallback"):
                qa.accepted_reading_summary_fallback_count += 1
            if "aux_path" in metadata_accepted.get("reason_tags", []) or "aux_module" in metadata_accepted.get("reason_tags", []):
                qa.rejected_aux_root_file_count += 1

            qa.record_audit(
                sample,
                validation_status="accepted",
                prompt_text=serialized.get("prompt_text", ""),
                target_text=serialized.get("target_text", ""),
                leakage_flags=leakage_flags,
                task_purity_status="clean",
                bucket=bucket,
            )

            input_data = sample.get("input", {})
            prompt_len = len(serialized.get("prompt_text", "") or "")
            target_len = len(serialized.get("target_text", "") or "")
            snippet_count = 0
            read_history_steps = 0
            if task_type in {"patch_grounding", "ast_dependency_trace"}:
                obs = input_data.get("current_observation", {})
                if isinstance(obs, dict):
                    if obs.get("current_snippet"):
                        snippet_count += 1
                    if obs.get("entry_snippet"):
                        snippet_count += 1
                    rs = obs.get("related_snippets", [])
                    if isinstance(rs, list):
                        snippet_count += len([x for x in rs if isinstance(x, dict) and x.get("snippet")])
            elif task_type == "reading_summary":
                rh = input_data.get("read_history", [])
                if isinstance(rh, list):
                    read_history_steps = len(rh)
                    for step in rh:
                        if isinstance(step, dict) and step.get("observation", {}).get("snippet"):
                            snippet_count += 1
            qa.observe_serialized_record(
                task_type=task_type,
                prompt_len=prompt_len,
                target_len=target_len,
                snippet_count=snippet_count,
                read_history_steps=read_history_steps,
                used_instruction_fallback=bool(metadata_accepted.get("used_task_instruction_fallback", False)),
                used_action_fallback=bool(metadata_accepted.get("used_action_selection_fallback", False)),
            )

            if all(accepted_counts[t] >= quota for t in final_task_set):
                break

    # ── sanity pass ───────────────────────────────────────────────────────────
    forbidden_keys = {"sample", "gold_file", "gold_line_span", "gold_symbol"}
    required_top_keys = {"meta", "content_split", "token_count"}
    sanity_errors: list[str] = []
    sanity_counts: Counter = Counter()
    with open(output_jsonl, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                sanity_errors.append(f"line {lineno}: invalid JSON: {exc}")
                continue
            top_keys = set(obj.keys())
            if top_keys != required_top_keys:
                sanity_errors.append(f"line {lineno}: unexpected top-level keys {top_keys}")
            bad = forbidden_keys & set(json.dumps(obj))
            # check forbidden keys anywhere in the object
            obj_str = json.dumps(obj)
            for fk in forbidden_keys:
                if f'"{fk}"' in obj_str:
                    sanity_errors.append(f"line {lineno}: forbidden key '{fk}' found")
            task_in_chunk = str(obj.get("meta", {}).get("chunk_id", "")).split("-")[0]
            if task_in_chunk:
                sanity_counts[task_in_chunk] += 1
    for t in final_task_set:
        if sanity_counts.get(t, 0) != quota:
            sanity_errors.append(f"quota mismatch for {t}: expected {quota}, got {sanity_counts.get(t, 0)}")
    if sanity_errors:
        print(f"[SANITY] {len(sanity_errors)} issue(s):")
        for err in sanity_errors[:20]:
            print(f"  {err}")
    else:
        print(f"[SANITY] OK — {accepted_total} records, quotas {dict(accepted_counts)}")

    # ── parquet ───────────────────────────────────────────────────────────────
    with open(output_jsonl, encoding="utf-8") as f:
        final_records = [json.loads(line) for line in f if line.strip()]
    pd.DataFrame(final_records).to_parquet(output_parquet, index=False)

    report = qa.to_report()
    report["pipeline_version"] = PIPELINE_VERSION
    report["source_script"] = str(Path(__file__).name)
    report["total_samples"] = accepted_total
    report["per_task_counts"] = dict(accepted_counts)
    report["main_jsonl_record_type"] = "content_split_record"
    report["main_jsonl_path"] = output_jsonl
    report["parquet_path"] = output_parquet
    if judge_reports is not None:
        report["judge_repo_count"] = len(judge_reports)
    if output_audit_jsonl:
        report["audit_jsonl_path"] = output_audit_jsonl
    with open(output_qa_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    if output_judge_json:
        judge_payload = {
            "pipeline_version": PIPELINE_VERSION,
            "repo_judge_reports": judge_reports or [],
            "sample_quality": sample_quality_rows,
        }
        with open(output_judge_json, "w", encoding="utf-8") as handle:
            json.dump(judge_payload, handle, ensure_ascii=False, indent=2)
    if output_audit_jsonl:
        with open(output_audit_jsonl, "w", encoding="utf-8") as handle:
            for row in qa.audit_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nDone. {accepted_total} content_split records -> {output_jsonl}")
    print(f"Parquet export -> {output_parquet}")
    print(f"QA report -> {output_qa_json}")
    print(f"  per_task_counts: {json.dumps(dict(accepted_counts), ensure_ascii=False)}")
    print(f"  skipped_counts_by_reason: {json.dumps(report['skipped_counts_by_reason'], ensure_ascii=False)}")


# ── cli ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Static-only synthetic data pipeline for coding-agent pretraining.")
    parser.add_argument("--repo-base", default=REPO_BASE)
    parser.add_argument("--output-jsonl", default=OUTPUT_JSONL)
    parser.add_argument("--output-parquet", default=OUTPUT_PARQUET)
    parser.add_argument("--output-qa-json", default=OUTPUT_QA_JSON)
    parser.add_argument("--output-judge-json", default=OUTPUT_JUDGE_JSON)
    parser.add_argument("--output-audit-jsonl", default=OUTPUT_AUDIT_JSONL)
    parser.add_argument(
        "--output-content-split-jsonl",
        default=OUTPUT_CONTENT_SPLIT_JSONL,
        help="Optional export path for packed pretrain records with meta/content_split/token_count.",
    )
    parser.add_argument(
        "--tokenizer-dir",
        default=DEFAULT_TOKENIZER_DIR,
        help="Tokenizer directory used to compute token_count for content_split export.",
    )
    parser.add_argument(
        "--dataset-mode",
        default=DEFAULT_DATASET_MODE,
        choices=["grounded_3tasks"],
        help="Dataset shaping mode. The pipeline emits only 3 grounded tasks: patch_grounding, ast_dependency_trace, reading_summary.",
    )
    parser.add_argument(
        "--output-raw-samples-jsonl",
        default=OUTPUT_RAW_SAMPLES_JSONL,
        help="Optional debug path for raw internal task samples. Main JSONL always writes pretraining records.",
    )
    parser.add_argument(
        "--per-task-target",
        type=int,
        default=5,
        help="Emit up to N samples for each of the 3 tasks (default: 5).",
    )
    parser.add_argument(
        "--buffer-multiplier",
        type=int,
        default=2,
        help="When --per-task-target is set, scan until N*buffer_multiplier candidates per task are collected (default: 2).",
    )
    parser.add_argument("--target", type=int, default=TARGET)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    qa = QATracker()

    final_task_types = ["patch_grounding", "ast_dependency_trace", "reading_summary"]
    per_task_target = int(args.per_task_target or 5)
    quota = per_task_target

    projects = sorted(os.listdir(args.repo_base)) if os.path.isdir(args.repo_base) else []
    if os.environ.get("SHUFFLE_PROJECTS", "").strip().lower() in {"1", "true", "yes"}:
        try:
            seed = int(os.environ.get("PROJECT_SHUFFLE_SEED", "0") or 0)
        except Exception:
            seed = 0
        rng = random.Random(seed)
        rng.shuffle(projects)

    all_samples: list[dict[str, Any]] = []
    all_judge_reports: list[dict[str, Any]] = []
    counts: Counter = Counter()
    per_project = max(30, quota * len(final_task_types) * 2)

    project_bar = _make_tqdm(projects, desc="projects", unit="proj", dynamic_ncols=True)
    for project in project_bar:
        if all(counts.get(t, 0) >= quota for t in final_task_types):
            break
        project_dir = os.path.join(args.repo_base, project)
        if not os.path.isdir(project_dir):
            continue
        try:
            new_samples, judge_report = process_project(
                project_dir,
                per_project,
                qa,
                return_judge_report=True,
                dataset_mode=args.dataset_mode,
            )
            if judge_report.get("commit_decisions") or judge_report.get("merge_groups") or judge_report.get("drop_commit_ids"):
                all_judge_reports.append(judge_report)
            if new_samples:
                all_samples.extend(new_samples)
                counts.update(sample.get("task_type", "") for sample in new_samples)
            pg = counts.get("patch_grounding", 0)
            dt = counts.get("ast_dependency_trace", 0)
            rs = counts.get("reading_summary", 0)
            top_drops = sorted(qa.skipped.items(), key=lambda x: -x[1])[:5]
            drops_str = " | ".join(f"{k}={v}" for k, v in top_drops)
            msg = f"[{project}] +{len(new_samples) if new_samples else 0}  pg={pg} dt={dt} rs={rs}  drops: {drops_str}"
            try:
                _tqdm.write(msg)
            except Exception:
                print(msg)
            if hasattr(project_bar, "set_postfix"):
                project_bar.set_postfix(pg=pg, dt=dt, rs=rs)
        except Exception as exc:
            qa.skip("project_processing_error")
            msg = f"[{project}] ERROR: {exc}"
            try:
                _tqdm.write(msg)
            except Exception:
                print(msg)

    all_samples = dedupe_samples(all_samples, qa=qa)

    write_outputs(
        all_samples,
        args.output_jsonl,
        args.output_parquet,
        args.output_qa_json,
        qa,
        output_raw_samples_jsonl=args.output_raw_samples_jsonl,
        judge_reports=all_judge_reports,
        output_judge_json=args.output_judge_json,
        output_audit_jsonl=args.output_audit_jsonl,
        output_content_split_jsonl=args.output_content_split_jsonl,
        tokenizer_dir=args.tokenizer_dir,
        per_task_target=per_task_target,
        final_task_types=final_task_types,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize GAIA runner JSONL outputs.

Usage:
    .venv/bin/python scripts/analyze_runs.py runs/gaia_20260604_*.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))
    return sorted(dict.fromkeys(paths))


def load_run(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            row["_run_path"] = str(path)
            rows.append(row)
    return rows


def norm_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def question_category(row: dict[str, Any]) -> str:
    q = norm_text(row.get("question")).lower()
    attachment = (row.get("attachment_path") or "").lower()
    trace_tools = {item.get("tool", "") for item in row.get("trace") or []}
    error = norm_text(row.get("error")).lower()

    if error:
        if "429" in error or "rate" in error or "capacity" in error or "timeout" in error:
            return "api/retry"
        return "runtime-error"
    if attachment.endswith((".png", ".jpg", ".jpeg", ".webp")) or "image" in q or "photo" in q:
        return "image/ocr"
    if attachment.endswith(".pptx") or "powerpoint" in q or "slides" in q:
        return "presentation"
    if attachment.endswith((".xlsx", ".csv", ".tsv")) or "spreadsheet" in q or "excel" in q:
        return "table/spreadsheet"
    if attachment.endswith((".mp3", ".wav", ".m4a", ".aac")) or "voice memo" in q or "audio" in q:
        return "audio"
    if "wikipedia" in q or "revision" in q or "edit" in q:
        return "wikipedia/history"
    if "youtube" in q or "video" in q:
        return "video/web"
    if "pdf" in q or "doi" in q or "paper" in q or "article" in q or "citation" in q:
        return "paper/pdf/web"
    if "how many" in q or "calculate" in q or "difference" in q or "optimal strategy" in q:
        return "math/logic"
    if "search_web" in trace_tools or "fetch_url" in trace_tools:
        return "web-lookup"
    return "other"


def answer_issue(row: dict[str, Any]) -> str:
    expected = norm_text(str(row.get("expected", "")))
    prediction = norm_text(str(row.get("prediction", "")))
    error = norm_text(row.get("error"))
    if error:
        return "runtime"
    if not prediction:
        return "empty"
    if expected and prediction:
        if expected.lower() in prediction.lower() and expected.lower() != prediction.lower():
            return "over-answer"
        if prediction.lower() in expected.lower() and expected.lower() != prediction.lower():
            return "under-specified"
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", expected) and re.fullmatch(r"[-+]?\d+(?:\.\d+)?", prediction):
            return "numeric-wrong"
        if expected.lower() == prediction.lower() and expected != prediction:
            return "format/case"
    return "wrong-fact"


def trace_signature(row: dict[str, Any]) -> str:
    tools = [item.get("tool", "?") for item in row.get("trace") or []]
    if not tools:
        return "no-tools"
    collapsed: list[str] = []
    for tool in tools:
        if not collapsed or collapsed[-1] != tool:
            collapsed.append(tool)
    return " > ".join(collapsed[:8])


def print_run_scores(runs: dict[Path, list[dict[str, Any]]]) -> None:
    print("Run scores")
    print("----------")
    for path, rows in runs.items():
        total = len(rows)
        correct = sum(1 for row in rows if row.get("correct"))
        pct = (correct / total * 100) if total else 0
        print(f"{path}: {correct}/{total} ({pct:.1f}%)")
    if runs:
        totals = [len(rows) for rows in runs.values()]
        scores = [sum(1 for row in rows if row.get("correct")) for rows in runs.values()]
        total_questions = sum(totals)
        total_correct = sum(scores)
        macro = sum(score / total for score, total in zip(scores, totals) if total) / len(scores)
        print()
        print(f"Micro average: {total_correct}/{total_questions} ({total_correct / total_questions * 100:.1f}%)")
        print(f"Macro average: {macro * 100:.1f}%")


def print_failures(all_rows: list[dict[str, Any]], top: int) -> None:
    failures = [row for row in all_rows if not row.get("correct")]
    print()
    print("Failure breakdown")
    print("-----------------")
    print(f"Failures: {len(failures)}")

    by_category = Counter(question_category(row) for row in failures)
    by_issue = Counter(answer_issue(row) for row in failures)
    by_trace = Counter(trace_signature(row) for row in failures)
    repeated = Counter(row.get("index") for row in failures)

    print("\nBy category:")
    for category, count in by_category.most_common():
        print(f"  {category}: {count}")

    print("\nBy answer issue:")
    for issue, count in by_issue.most_common():
        print(f"  {issue}: {count}")

    print("\nMost common failing tool traces:")
    for signature, count in by_trace.most_common(top):
        print(f"  {count}x  {signature}")

    print("\nRepeated failing indexes:")
    for index, count in repeated.most_common():
        if count > 1:
            print(f"  index {index}: {count} failures")


def print_failure_table(all_rows: list[dict[str, Any]], top: int) -> None:
    failures = [row for row in all_rows if not row.get("correct")]
    failures = sorted(
        failures,
        key=lambda row: (question_category(row), int(row.get("index", -1)), row.get("_run_path", "")),
    )
    print()
    print(f"Failure details (first {min(top, len(failures))})")
    print("--------------------------------")
    for row in failures[:top]:
        q = norm_text(row.get("question"))
        print(f"[{question_category(row)}] index {row.get('index')} level {row.get('level')} ({Path(row['_run_path']).name})")
        print(f"  expected : {norm_text(str(row.get('expected')))}")
        print(f"  predicted: {norm_text(str(row.get('prediction')))}")
        if row.get("error"):
            print(f"  error    : {norm_text(row.get('error'))}")
        print(f"  issue    : {answer_issue(row)}")
        print(f"  tools    : {trace_signature(row)}")
        print(f"  question : {q[:240]}")


def print_priorities(all_rows: list[dict[str, Any]]) -> None:
    failures = [row for row in all_rows if not row.get("correct")]
    category_counts = Counter(question_category(row) for row in failures)
    issue_counts = Counter(answer_issue(row) for row in failures)

    print()
    print("Suggested priorities")
    print("--------------------")
    if not failures:
        print("No failures in selected runs.")
        return

    priorities = []
    for category, count in category_counts.most_common():
        if category == "image/ocr":
            priorities.append((count, "Build a real image/OCR path for diagrams, worksheets, labels, and visual counting."))
        elif category == "wikipedia/history":
            priorities.append((count, "Generalize Wikipedia history/revision queries with timestamp and diff helpers."))
        elif category == "paper/pdf/web":
            priorities.append((count, "Improve scholarly PDF/article extraction and quote verification."))
        elif category == "audio":
            priorities.append((count, "Post-process transcripts into requested entities while preserving modifiers."))
        elif category == "math/logic":
            priorities.append((count, "Route game/probability/counting problems to deterministic solvers."))
        elif category in {"web-lookup", "video/web"}:
            priorities.append((count, "Use source-specific parsers for web/video instead of relying on snippets."))
        elif category == "runtime-error":
            priorities.append((count, "Fix tool-call runtime errors and normalize tool arguments."))

    for count, message in sorted(priorities, reverse=True):
        print(f"- {message} ({count} recent failures)")

    if issue_counts.get("over-answer"):
        print(f"- Add stricter final answer extraction for over-answers ({issue_counts['over-answer']} recent failures)")
    if issue_counts.get("format/case") or issue_counts.get("under-specified"):
        total = issue_counts.get("format/case", 0) + issue_counts.get("under-specified", 0)
        print(f"- Add final formatting guards for exact-answer prompts ({total} recent failures)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="JSONL files or glob patterns")
    parser.add_argument("--top", type=int, default=40, help="number of failure details/tool traces to show")
    args = parser.parse_args()

    paths = expand_paths(args.paths)
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise SystemExit("Missing files: " + ", ".join(str(path) for path in missing))

    runs = {path: load_run(path) for path in paths}
    all_rows = [row for rows in runs.values() for row in rows]
    print_run_scores(runs)
    print_failures(all_rows, args.top)
    print_failure_table(all_rows, args.top)
    print_priorities(all_rows)


if __name__ == "__main__":
    main()

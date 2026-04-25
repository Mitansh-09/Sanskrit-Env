"""
Render a side-by-side comparison of two evaluate.py JSON outputs.

Useful workflow:
    python training/evaluate.py --output runs/eval_baseline.json
    # ... train ...
    python training/evaluate.py --adapter runs/qwen25-1p5b-grpo --output runs/eval_post.json
    python training/compare_evals.py runs/eval_baseline.json runs/eval_post.json

Outputs:
    - A pretty ASCII table on stdout with per-task and overall deltas.
    - An optional Markdown table when `--markdown <path>` is supplied,
      ready to drop into the project README.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


METRIC_COLUMNS: List[str] = [
    "score_mean",
    "score_std",
    "success_rate",
    "full_credit_rate",
]


def load(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def _delta(before: float, after: float) -> str:
    delta = after - before
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.3f}"


def _rel(before: float, after: float) -> str:
    if abs(before) < 1e-3:
        return "n/a"
    rel = (after - before) / before * 100
    sign = "+" if rel >= 0 else ""
    return f"{sign}{rel:.1f}%"


def build_rows(before: Dict[str, Any], after: Dict[str, Any]) -> List[List[str]]:
    rows: List[List[str]] = []
    b_tasks = before["summary"]["tasks"]
    a_tasks = after["summary"]["tasks"]

    for task in sorted(set(b_tasks.keys()) | set(a_tasks.keys())):
        b = b_tasks.get(task, {})
        a = a_tasks.get(task, {})
        bm = float(b.get("score_mean", 0.0))
        am = float(a.get("score_mean", 0.0))
        bs = float(b.get("score_std", 0.0))
        as_ = float(a.get("score_std", 0.0))
        bsr = float(b.get("success_rate", 0.0))
        asr = float(a.get("success_rate", 0.0))
        bfc = float(b.get("full_credit_rate", 0.0))
        afc = float(a.get("full_credit_rate", 0.0))
        rows.append(
            [
                task,
                _fmt(bm),
                _fmt(am),
                _delta(bm, am),
                _rel(bm, am),
                _fmt(bs),
                _fmt(as_),
                _fmt(bsr),
                _fmt(asr),
                _fmt(bfc),
                _fmt(afc),
            ]
        )

    bm = float(before["summary"].get("overall_mean", 0.0))
    am = float(after["summary"].get("overall_mean", 0.0))
    bs = float(before["summary"].get("overall_std", 0.0))
    as_ = float(after["summary"].get("overall_std", 0.0))
    bsr = float(before["summary"].get("overall_success_rate", 0.0))
    asr = float(after["summary"].get("overall_success_rate", 0.0))
    rows.append(
        [
            "OVERALL",
            _fmt(bm),
            _fmt(am),
            _delta(bm, am),
            _rel(bm, am),
            _fmt(bs),
            _fmt(as_),
            _fmt(bsr),
            _fmt(asr),
            "-",
            "-",
        ]
    )
    return rows


HEADER = [
    "Task",
    "Score (before)",
    "Score (after)",
    "Abs delta",
    "Rel %",
    "Std (before)",
    "Std (after)",
    "Success (before)",
    "Success (after)",
    "Full credit (before)",
    "Full credit (after)",
]


def render_text(rows: List[List[str]]) -> str:
    widths = [max(len(HEADER[i]), max(len(r[i]) for r in rows)) for i in range(len(HEADER))]

    def line(parts: List[str]) -> str:
        return " | ".join(parts[i].ljust(widths[i]) for i in range(len(parts)))

    sep = "-+-".join("-" * w for w in widths)
    out: List[str] = [line(HEADER), sep]
    for row in rows:
        out.append(line(row))
    return "\n".join(out)


def render_markdown(rows: List[List[str]]) -> str:
    out: List[str] = []
    out.append("| " + " | ".join(HEADER) + " |")
    out.append("|" + "|".join(["---"] * len(HEADER)) + "|")
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two evaluate.py JSON outputs.")
    parser.add_argument("before", help="Path to the baseline JSON (pre-training).")
    parser.add_argument("after", help="Path to the post-training JSON.")
    parser.add_argument("--markdown", default=None, help="If set, also write a Markdown table to this path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    before = load(args.before)
    after = load(args.after)
    rows = build_rows(before, after)
    print(f"BEFORE: {before.get('label', args.before)}  ({before.get('episodes_per_task', '?')} ep/task)")
    print(f"AFTER : {after.get('label', args.after)}    ({after.get('episodes_per_task', '?')} ep/task)")
    print()
    print(render_text(rows))

    if args.markdown:
        path = Path(args.markdown)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(render_markdown(rows) + "\n")
        print(f"\n[info] markdown table written to {path}")


if __name__ == "__main__":
    main()

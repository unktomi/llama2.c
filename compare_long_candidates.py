#!/usr/bin/env python3
"""Generate and measure length-matched continuations for long prompts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "test" / "stories260K.bin"
TOKENIZER = ROOT / "test" / "tok512.bin"
GENERATOR = ROOT / "candidate_dump"
PROBE = ROOT / "company_probe"


@dataclass(frozen=True)
class Mode:
    name: str
    temperature: float
    top_p: float


MODES = (
    Mode("greedy_fixed", 0.0, 0.0),
    Mode("sample_fixed", 1.0, 0.9),
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "work_traces" / "long_context_32",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "work_traces" / "long_candidate_comparison",
    )
    parser.add_argument("--cases", type=int, default=24)
    parser.add_argument("--minimum-completion-tokens", type=int, default=64)
    parser.add_argument("--window", type=int, default=32)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def repeated_ngram_coverage(tokens: list[int], width: int) -> float:
    if len(tokens) < width:
        return 0.0
    starts: dict[tuple[int, ...], list[int]] = {}
    for index in range(len(tokens) - width + 1):
        ngram = tuple(tokens[index : index + width])
        starts.setdefault(ngram, []).append(index)
    covered: set[int] = set()
    for occurrences in starts.values():
        if len(occurrences) < 2:
            continue
        for start in occurrences:
            covered.update(range(start, start + width))
    return len(covered) / len(tokens)


def longest_nonoverlapping_repeat(tokens: list[int]) -> int:
    count = len(tokens)
    following = [0] * (count + 1)
    maximum = 0
    for left in range(count - 1, -1, -1):
        current = [0] * (count + 1)
        for right in range(count - 1, left, -1):
            if tokens[left] == tokens[right]:
                current[right] = 1 + following[right + 1]
                maximum = max(maximum, min(current[right], right - left))
        following = current
    return maximum


def mean(records: list[dict[str, Any]], field: str) -> float | None:
    values = [record[field] for record in records if record.get(field) is not None]
    return statistics.fmean(values) if values else None


def measure_trace(
    trace_path: Path,
    text_path: Path,
    case: int,
    source_index: int,
    mode: str,
    window: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = load_jsonl(trace_path)
    meta = next(record for record in records if record["kind"] == "meta")
    tokens = sorted(
        (record for record in records if record["kind"] == "token"),
        key=lambda record: record["completion_index"],
    )
    layer4 = {
        record["completion_index"]: record
        for record in records
        if record["kind"] == "layer" and record["layer"] == 4
    }
    terminal = next(record for record in records if record["kind"] == "terminal")
    affine = [
        record
        for record in records
        if record["kind"] == "affine_prefix" and record["layer"] == 4
    ][-1]
    token_ids = [record["token_id"] for record in tokens]
    layer_records = [layer4[index] for index in range(len(tokens))]
    row: dict[str, Any] = {
        "case": case,
        "source_index": source_index,
        "mode": mode,
        "prompt_tokens": meta["prompt_tokens"],
        "completion_tokens": len(tokens),
        "mean_log_probability": statistics.fmean(
            record["log_probability"] for record in tokens
        ),
        "delimiter_log_probability": terminal["delimiter_log_probability"],
        "delimiter_rank": terminal["delimiter_rank"],
        "repeat_2gram_coverage": repeated_ngram_coverage(token_ids, 2),
        "repeat_3gram_coverage": repeated_ngram_coverage(token_ids, 3),
        "repeat_4gram_coverage": repeated_ngram_coverage(token_ids, 4),
        "longest_nonoverlapping_repeat": longest_nonoverlapping_repeat(token_ids),
        "layer4_prior_similarity": mean(layer_records, "prior_state_similarity"),
        "layer4_same_token_similarity": mean(
            layer_records, "same_token_state_similarity"
        ),
        "layer4_residual_inertia": mean(layer_records, "residual_inertia"),
        "layer4_prompt_mass": mean(layer_records, "attention_prompt_mass"),
        "layer4_affine_dimension": affine["effective_dimension"],
        "text_path": str(text_path),
        "trace_path": str(trace_path),
    }
    windows: list[dict[str, Any]] = []
    for first in range(0, len(tokens) - window + 1, window):
        last = first + window
        token_span = tokens[first:last]
        ids = token_ids[first:last]
        layer_span = layer_records[first:last]
        windows.append(
            {
                "case": case,
                "source_index": source_index,
                "mode": mode,
                "first_token": first,
                "last_token": last,
                "normalized_position": (first + window / 2) / len(tokens),
                "absolute_position": token_span[0]["token_position"],
                "mean_log_probability": mean(token_span, "log_probability"),
                "mean_entropy": mean(token_span, "entropy"),
                "repeat_4gram_coverage": repeated_ngram_coverage(ids, 4),
                "layer4_prior_similarity": mean(
                    layer_span, "prior_state_similarity"
                ),
                "layer4_same_token_similarity": mean(
                    layer_span, "same_token_state_similarity"
                ),
                "layer4_residual_inertia": mean(
                    layer_span, "residual_inertia"
                ),
                "layer4_prompt_mass": mean(
                    layer_span, "attention_prompt_mass"
                ),
                "text": "".join(record["piece"] for record in token_span),
            }
        )
    return row, windows


def run_generated(
    case_dir: Path,
    output_dir: Path,
    source_index: int,
    token_count: int,
    mode: Mode,
) -> tuple[Path, Path]:
    mode_dir = output_dir / mode.name
    mode_dir.mkdir(parents=True, exist_ok=True)
    text_path = mode_dir / "completion.txt"
    generation_trace = mode_dir / "generation.jsonl"
    seed = source_index + 1
    result = subprocess.run(
        [
            str(GENERATOR),
            str(MODEL),
            str(TOKENIZER),
            str(case_dir / "prompt.txt"),
            str(seed),
            str(token_count),
            str(token_count),
            str(mode.temperature),
            str(mode.top_p),
            str(text_path),
            str(generation_trace),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    (mode_dir / "generation_summary.txt").write_text(
        result.stdout, encoding="utf-8"
    )
    company_trace = mode_dir / "company.jsonl"
    score = subprocess.run(
        [
            str(PROBE),
            str(MODEL),
            str(TOKENIZER),
            "--files",
            str(case_dir / "prompt.txt"),
            str(text_path),
            "--trace",
            str(company_trace),
            "--checkpoint-every",
            "16",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    (mode_dir / "company_summary.txt").write_text(score.stdout, encoding="utf-8")

    generated_ids = [
        record["token_id"]
        for record in load_jsonl(generation_trace)
        if record["kind"] == "token"
    ]
    rescored_ids = [
        record["token_id"]
        for record in load_jsonl(company_trace)
        if record["kind"] == "token"
    ]
    if generated_ids != rescored_ids:
        raise RuntimeError(f"decoded candidate did not round-trip in {mode_dir}")
    return text_path, company_trace


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    first = 0
    while first < len(order):
        last = first + 1
        while last < len(order) and values[order[last]] == values[order[first]]:
            last += 1
        tied_rank = (first + last - 1) / 2 + 1
        for index in range(first, last):
            result[order[index]] = tied_rank
        first = last
    return result


def pearson(left: list[float], right: list[float]) -> float:
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean) for a, b in zip(left, right)
    )
    left_square = sum((value - left_mean) ** 2 for value in left)
    right_square = sum((value - right_mean) ** 2 for value in right)
    if left_square == 0.0 or right_square == 0.0:
        return math.nan
    return numerator / math.sqrt(left_square * right_square)


def spearman(left: list[float], right: list[float]) -> float:
    return pearson(rank(left), rank(right))


def bootstrap_mean_interval(values: list[float]) -> tuple[float, float]:
    generator = random.Random(20260828)
    estimates = []
    for _ in range(5000):
        estimates.append(
            statistics.fmean(generator.choice(values) for _ in values)
        )
    estimates.sort()
    return estimates[125], estimates[4874]


def print_report(rows: list[dict[str, Any]], windows: list[dict[str, Any]]) -> None:
    fields = (
        "mean_log_probability",
        "delimiter_log_probability",
        "repeat_4gram_coverage",
        "longest_nonoverlapping_repeat",
        "layer4_prior_similarity",
        "layer4_same_token_similarity",
        "layer4_residual_inertia",
        "layer4_affine_dimension",
    )
    modes = ("source", "greedy_fixed", "sample_fixed")
    print("group_means:")
    for mode in modes:
        selected = [row for row in rows if row["mode"] == mode]
        print(f"  mode={mode} count={len(selected)}")
        for field in fields:
            print(
                f"    {field}="
                f"{statistics.fmean(float(row[field]) for row in selected):.9g}"
            )

    source = {row["case"]: row for row in rows if row["mode"] == "source"}
    print("paired_differences_variant_minus_source:")
    for mode in modes[1:]:
        selected = {row["case"]: row for row in rows if row["mode"] == mode}
        print(f"  mode={mode}")
        for field in fields:
            differences = [
                float(selected[case][field]) - float(source[case][field])
                for case in source
            ]
            low, high = bootstrap_mean_interval(differences)
            print(
                f"    {field} mean={statistics.fmean(differences):.9g} "
                f"ci95=[{low:.9g},{high:.9g}] "
                f"positive={sum(value > 0 for value in differences)}/"
                f"{len(differences)}"
            )

    print("fixed_32_token_window_spearman_vs_repeat_4gram_coverage:")
    for mode in modes:
        selected = [row for row in windows if row["mode"] == mode]
        repetition = [float(row["repeat_4gram_coverage"]) for row in selected]
        print(f"  mode={mode} windows={len(selected)}")
        for field in (
            "mean_log_probability",
            "mean_entropy",
            "layer4_prior_similarity",
            "layer4_same_token_similarity",
            "layer4_residual_inertia",
            "layer4_prompt_mass",
        ):
            values = [float(row[field]) for row in selected]
            print(f"    {field} rho={spearman(repetition, values):.9g}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = arguments()
    if args.cases <= 0 or args.minimum_completion_tokens <= 0 or args.window <= 0:
        raise SystemExit("case count, minimum completion, and window must be positive")
    required = (MODEL, TOKENIZER, GENERATOR, PROBE, args.input / "manifest.csv")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing required paths: " + ", ".join(missing))
    manifests = [
        row
        for row in csv.DictReader((args.input / "manifest.csv").open())
        if int(row["completion_tokens"]) >= args.minimum_completion_tokens
    ][: args.cases]
    if len(manifests) < args.cases:
        raise SystemExit(
            f"only {len(manifests)} cases meet the completion-token minimum"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    for case, manifest in enumerate(manifests):
        source_dir = Path(manifest["trace"]).parent
        source_index = int(manifest["source_index"])
        token_count = int(manifest["completion_tokens"])
        output_dir = args.output / f"case_{case:03d}_source_{source_index}"
        output_dir.mkdir(parents=True, exist_ok=True)
        source_row, source_windows = measure_trace(
            source_dir / "trace.jsonl",
            source_dir / "completion.txt",
            case,
            source_index,
            "source",
            args.window,
        )
        rows.append(source_row)
        windows.extend(source_windows)
        for mode in MODES:
            text_path, trace_path = run_generated(
                source_dir,
                output_dir,
                source_index,
                token_count,
                mode,
            )
            row, mode_windows = measure_trace(
                trace_path,
                text_path,
                case,
                source_index,
                mode.name,
                args.window,
            )
            if row["completion_tokens"] != source_row["completion_tokens"]:
                raise RuntimeError("length-matched candidate changed token count")
            rows.append(row)
            windows.extend(mode_windows)
        print(
            f"case={case} source={source_index} tokens={token_count}",
            flush=True,
        )
    write_csv(args.output / "candidates.csv", rows)
    write_csv(args.output / "windows.csv", windows)
    print_report(rows, windows)


if __name__ == "__main__":
    main()

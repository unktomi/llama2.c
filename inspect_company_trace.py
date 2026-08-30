#!/usr/bin/env python3
"""Inspect arbitrary token intervals in a company_probe JSONL trace."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Iterable


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument(
        "--span",
        help="one half-open completion-token interval, for example 64:128",
    )
    parser.add_argument("--layer", type=int, default=4)
    parser.add_argument(
        "--non-top-one",
        action="store_true",
        help="list every selected token whose local rank is greater than one",
    )
    return parser.parse_args()


def mean(records: Iterable[dict[str, Any]], field: str) -> float | None:
    values = [record[field] for record in records if record.get(field) is not None]
    return statistics.fmean(values) if values else None


def parse_span(text: str, token_count: int) -> tuple[int, int]:
    try:
        first_text, last_text = text.split(":", 1)
        first = int(first_text)
        last = int(last_text)
    except (ValueError, TypeError) as error:
        raise SystemExit("--span must have the form START:END") from error
    if first < 0 or last <= first or last > token_count:
        raise SystemExit(f"span must lie within 0:{token_count}")
    return first, last


def quote_text(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def main() -> None:
    args = arguments()
    if args.window <= 0:
        raise SystemExit("--window must be positive")
    records = [json.loads(line) for line in args.trace.read_text().splitlines()]
    meta = next(record for record in records if record["kind"] == "meta")
    tokens = sorted(
        (record for record in records if record["kind"] == "token"),
        key=lambda record: record["completion_index"],
    )
    layers: dict[tuple[int, int], dict[str, Any]] = {
        (record["completion_index"], record["layer"]): record
        for record in records
        if record["kind"] == "layer"
    }
    terminal = next(record for record in records if record["kind"] == "terminal")
    if len(tokens) != meta["completion_tokens"]:
        raise SystemExit("trace has an incomplete token sequence")
    if [record["completion_index"] for record in tokens] != list(range(len(tokens))):
        raise SystemExit("trace token indices are not contiguous")

    if args.span:
        spans = [parse_span(args.span, len(tokens))]
    else:
        spans = [
            (first, min(first + args.window, len(tokens)))
            for first in range(0, len(tokens), args.window)
        ]

    print(
        f"prompt_tokens={meta['prompt_tokens']} "
        f"completion_tokens={meta['completion_tokens']} "
        f"total_tokens={meta['total_tokens']} layer={args.layer}"
    )
    for first, last in spans:
        token_span = tokens[first:last]
        layer_span = [
            layers[(index, args.layer)]
            for index in range(first, last)
            if (index, args.layer) in layers
        ]
        text = "".join(record["piece"] for record in token_span)
        log_probability = sum(record["log_probability"] for record in token_span)
        non_top_one = sum(record["local_rank"] > 1 for record in token_span)
        print(
            f"span={first}:{last} tokens={last - first} "
            f"logp={log_probability:.6f} "
            f"mean_logp={log_probability / (last - first):.6f} "
            f"non_top1={non_top_one} "
            f"max_rank={max(record['local_rank'] for record in token_span)} "
            f"entropy={format_optional(mean(token_span, 'entropy'))} "
            f"surprisal_minus_entropy="
            f"{format_optional(mean(token_span, 'surprisal_minus_entropy'))}"
        )
        print(f"  text={quote_text(text)}")
        if layer_span:
            print(
                f"  layer={args.layer} "
                f"attention_entropy_fraction="
                f"{format_optional(mean(layer_span, 'attention_entropy_fraction'))} "
                f"prompt_mass="
                f"{format_optional(mean(layer_span, 'attention_prompt_mass'))} "
                f"attention_update_ratio="
                f"{format_optional(mean(layer_span, 'attention_update_ratio'))} "
                f"ffn_update_ratio="
                f"{format_optional(mean(layer_span, 'ffn_update_ratio'))} "
                f"residual_inertia="
                f"{format_optional(mean(layer_span, 'residual_inertia'))} "
                f"prior_similarity="
                f"{format_optional(mean(layer_span, 'prior_state_similarity'))} "
                f"same_token_similarity="
                f"{format_optional(mean(layer_span, 'same_token_state_similarity'))}"
            )

    if args.non_top_one:
        print("non_top1_tokens:")
        selected_indices = {
            index
            for first, last in spans
            for index in range(first, last)
        }
        for record in tokens:
            if record["completion_index"] not in selected_indices:
                continue
            if record["local_rank"] == 1:
                continue
            print(
                f"  index={record['completion_index']} "
                f"rank={record['local_rank']} "
                f"logp={record['log_probability']:.6f} "
                f"top_logp={record['top_log_probability']:.6f} "
                f"piece={quote_text(record['piece'])} "
                f"top_piece={quote_text(record['top_piece'])}"
            )
    print(
        f"terminal delimiter_logp={terminal['delimiter_log_probability']:.6f} "
        f"delimiter_rank={terminal['delimiter_rank']}"
    )


if __name__ == "__main__":
    main()

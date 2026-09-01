#!/usr/bin/env python3
"""Browse decoded candidates and contrast scores in a recursive-company trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gather_recursive_company import DEFAULT_OUTPUT


DEFAULT_TRACE = DEFAULT_OUTPUT / "recursive-company.jsonl"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--root", required=True, help="exact root key or unique substring")
    parser.add_argument(
        "--path",
        default="",
        help="comma-separated constructor token IDs; empty means the root demand",
    )
    parser.add_argument("--top", type=int, default=16)
    return parser.parse_args()


def parse_path(text: str) -> tuple[int, ...]:
    if not text.strip():
        return ()
    try:
        return tuple(int(value) for value in text.split(","))
    except ValueError as error:
        raise SystemExit("--path must contain comma-separated token IDs") from error


def main() -> None:
    args = arguments()
    if args.top <= 0:
        raise SystemExit("--top must be positive")
    if not args.trace.is_file():
        raise SystemExit(f"missing trace: {args.trace}")
    requested_path = parse_path(args.path)
    roots: set[str] = set()
    matches: list[dict[str, Any]] = []
    meta: dict[str, Any] | None = None
    with args.trace.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            kind = row.get("kind")
            if kind == "recursive_company_meta":
                meta = row
                continue
            if kind not in {
                "recursive_company_demand",
                "recursive_company_terminal",
            }:
                continue
            root = str(row["root"])
            roots.add(root)
            if args.root not in root:
                continue
            if tuple(int(token) for token in row["path_tokens"]) != requested_path:
                continue
            matches.append(row)
    if meta is None:
        raise SystemExit("trace has no recursive_company_meta record")
    matching_roots = sorted(root for root in roots if args.root in root)
    if not matching_roots:
        raise SystemExit(f"no root contains {args.root!r}")
    if len(matching_roots) != 1:
        print("root substring is ambiguous; matching roots:")
        for root in matching_roots:
            print(root)
        raise SystemExit(2)
    exact_root = matching_roots[0]
    matches = [row for row in matches if row["root"] == exact_root]
    if len(matches) != 1:
        raise SystemExit(
            f"root {exact_root!r} has no unique observation at path {requested_path}"
        )
    row = matches[0]
    field = (
        "candidates"
        if row["kind"] == "recursive_company_demand"
        else "terminal_candidates"
    )
    candidates = sorted(
        row[field],
        key=lambda candidate: (-float(candidate["contrast"]), int(candidate["token"])),
    )
    print(f"root: {exact_root}")
    print(f"path: {list(requested_path)}")
    print(f"text: {row['text']!r}")
    print(f"observation: {row['kind']}")
    print("rank\ttoken\tcontrast\tpiece")
    for rank, candidate in enumerate(candidates[: args.top], start=1):
        print(
            f"{rank}\t{candidate['token']}\t"
            f"{float(candidate['contrast']):.9g}\t{candidate['piece']!r}"
        )


if __name__ == "__main__":
    main()

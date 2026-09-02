#!/usr/bin/env python3
"""Browse constructor candidates, local ballots, and potential defects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from analyze_firth_potential import DEFAULT_CELLS


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=Path, default=DEFAULT_CELLS)
    parser.add_argument("--root", default="", help="exact root or unique substring")
    parser.add_argument("--outer", help="comma-separated outer token pair")
    parser.add_argument("--inner", help="comma-separated inner token pair")
    parser.add_argument(
        "--sort",
        choices=("absolute", "smallest", "signed", "fraction"),
        default="absolute",
    )
    parser.add_argument("--top", type=int, default=8)
    return parser.parse_args()


def token_pair(value: str | None, option: str) -> tuple[int, int] | None:
    if value is None:
        return None
    try:
        pair = tuple(int(token) for token in value.split(","))
    except ValueError as error:
        raise SystemExit(f"{option} must contain two comma-separated token IDs") from error
    if len(pair) != 2:
        raise SystemExit(f"{option} must contain two comma-separated token IDs")
    return pair  # type: ignore[return-value]


def transition_tokens(record: dict[str, Any], field: str) -> tuple[int, int]:
    transition = record[field]
    return int(transition["from"]["token"]), int(transition["to"]["token"])


def main() -> None:
    args = arguments()
    if args.top <= 0:
        raise SystemExit("--top must be positive")
    if not args.cells.is_file():
        raise SystemExit(f"missing detailed potential trace: {args.cells}")
    outer = token_pair(args.outer, "--outer")
    inner = token_pair(args.inner, "--inner")
    roots: set[str] = set()
    matches: list[dict[str, Any]] = []
    with args.cells.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("kind") != "firth_potential_cell":
                raise SystemExit("unexpected detailed potential row")
            root = str(row["root"])
            roots.add(root)
            if args.root and args.root not in root:
                continue
            if outer is not None and transition_tokens(row, "outer_transition") != outer:
                continue
            if inner is not None and transition_tokens(row, "inner_transition") != inner:
                continue
            matches.append(row)
    if args.root:
        matching_roots = sorted(root for root in roots if args.root in root)
        if not matching_roots:
            raise SystemExit(f"no root contains {args.root!r}")
        if len(matching_roots) != 1:
            print("root substring is ambiguous; matching roots:")
            for root in matching_roots:
                print(root)
            raise SystemExit(2)
        matches = [row for row in matches if row["root"] == matching_roots[0]]
    if not matches:
        raise SystemExit("no potential cells match the requested filters")
    if args.sort == "absolute":
        key = lambda row: -abs(float(row["preference_circulation"]))
    elif args.sort == "smallest":
        key = lambda row: abs(float(row["preference_circulation"]))
    elif args.sort == "signed":
        key = lambda row: -float(row["preference_circulation"])
    else:
        key = lambda row: -float(row["hodge_decomposition"]["circulation_fraction"])
    matches.sort(key=lambda row: (key(row), str(row["root"])))

    print(f"matching cells: {len(matches)}")
    for rank, row in enumerate(matches[: args.top], start=1):
        outer_transition = row["outer_transition"]
        inner_transition = row["inner_transition"]
        ballots = row["local_ballots"]
        one_form = row["preference_one_form"]
        paths = row["path_preference_gains"]
        print()
        print(f"[{rank}] root: {row['root']}")
        print(
            "outer: "
            f"{outer_transition['from']['token']} {outer_transition['from']['piece']!r} -> "
            f"{outer_transition['to']['token']} {outer_transition['to']['piece']!r}"
        )
        print(
            "inner: "
            f"{inner_transition['from']['token']} {inner_transition['from']['piece']!r} -> "
            f"{inner_transition['to']['token']} {inner_transition['to']['piece']!r}"
        )
        outer_scores = ballots["outer"]["at_inner_from"]
        inner0 = ballots["inner_at_outer_from"]
        inner1 = ballots["inner_at_outer_to"]
        print(
            "outer local contrasts: "
            f"{outer_scores[0]['piece']!r}={float(outer_scores[0]['contrast']):.9g}, "
            f"{outer_scores[1]['piece']!r}={float(outer_scores[1]['contrast']):.9g}"
        )
        print(
            f"inner local contrasts after {outer_transition['from']['piece']!r}: "
            f"{inner0[0]['piece']!r}={float(inner0[0]['contrast']):.9g}, "
            f"{inner0[1]['piece']!r}={float(inner0[1]['contrast']):.9g}"
        )
        print(
            f"inner local contrasts after {outer_transition['to']['piece']!r}: "
            f"{inner1[0]['piece']!r}={float(inner1[0]['contrast']):.9g}, "
            f"{inner1[1]['piece']!r}={float(inner1[1]['contrast']):.9g}"
        )
        print(
            "preference edges: "
            f"D|E0={float(one_form['outer_at_inner_from']):.9g}, "
            f"D|E1={float(one_form['outer_at_inner_to']):.9g}, "
            f"E|D0={float(one_form['inner_at_outer_from']):.9g}, "
            f"E|D1={float(one_form['inner_at_outer_to']):.9g}"
        )
        print(
            "path gains: "
            f"D-then-E={float(paths['outer_then_inner']):.9g}, "
            f"E-then-D={float(paths['inner_then_outer']):.9g}"
        )
        print(
            f"circulation={float(row['preference_circulation']):.9g}, "
            "fraction="
            f"{float(row['hodge_decomposition']['circulation_fraction']):.9g}"
        )
        backward = row["required_backward_continuation"]
        correction = backward["minimum_l2_outer_edge_correction"]
        print(
            "required backward outer correction: "
            f"E0={float(correction[0]):.9g}, E1={float(correction[1]):.9g}"
        )


if __name__ == "__main__":
    main()

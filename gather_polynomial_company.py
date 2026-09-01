#!/usr/bin/env python3
"""Execute the full fixed verb/hole polynomial as one causal company.

The four pre-verb number corners are recovered from the already validated
number-cube traces as token sequences, not re-tokenized text.  One C process
then shares their identical prefixes, attaches every fixed verb injection, and
observes the outer and shape-indexed hole codata in one learned-filler pass.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from analyze_number_demand_cubes import (
    DEFAULT_CONSTRUCTOR_FAMILY,
    DEFAULT_HOLE_CONSTRUCTOR_FAMILY,
)
from gather_grammar_relations import (
    DEFAULT_MANIFEST,
    DEFAULT_MODEL,
    DEFAULT_TOKENIZER,
    read_manifest,
)
from gather_number_demand_cubes import (
    DEFAULT_OUTPUT as DEFAULT_NUMBER_TRACES,
    NumberDemandCase,
    expand_cases,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_PROGRAM = ROOT / "cps_polynomial_company"
DEFAULT_OUTPUT = ROOT / "work_traces" / "polynomial_company"
CORNER_NAMES = ("x", "ax", "cx", "acx")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def read_context_prefixes(
    case: NumberDemandCase,
    directory: Path,
) -> tuple[tuple[int, ...], ...]:
    path = directory / case.trace_name
    require(path.is_file(), f"missing number cube: {path}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_kind[str(row.get("kind"))].append(row)
    contexts = {
        str(row["corner"]): tuple(int(item["id"]) for item in row["tokens"])
        for row in by_kind["grammatical_cube_context"]
    }
    expected_corners = {"x", "ax", "bx", "abx", "cx", "acx", "bcx", "abcx"}
    require(set(contexts) == expected_corners, f"{path}: cube corners differ")

    positions: list[int] = []
    for left, right in (("x", "bx"), ("ax", "abx"), ("cx", "bcx"), ("acx", "abcx")):
        require(len(contexts[left]) == len(contexts[right]), f"{path}: verb fibers differ")
        differences = [
            index
            for index, (first, second) in enumerate(
                zip(contexts[left], contexts[right])
            )
            if first != second
        ]
        require(len(differences) == 1, f"{path}: verb is not one aligned constructor")
        positions.append(differences[0])
    require(len(set(positions)) == 1, f"{path}: verb position differs by corner")
    verb_position = positions[0]
    require(verb_position > 0, f"{path}: verb has no causal prefix")
    return tuple(contexts[name][:verb_position] for name in CORNER_NAMES)


def trace_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return False
    if not rows:
        return False
    meta = rows[0]
    check = rows[-1]
    return (
        meta.get("kind") == "polynomial_company_meta"
        and meta.get("schema_version") == 1
        and meta.get("contexts") == 44
        and meta.get("outer_width") == 12
        and meta.get("hole_width") == 8
        and meta.get("maximum_calls_per_filler") == 1
        and check.get("kind") == "polynomial_company_check"
        and check.get("contexts") == 44
        and check.get("outer_observations") == 176
        and check.get("shape_indexed_hole_observations") == 2112
        and check.get("maximum_calls_per_filler") == 1
    )


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--number-traces", type=Path, default=DEFAULT_NUMBER_TRACES)
    parser.add_argument("--program", type=Path, default=DEFAULT_PROGRAM)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--outer-family",
        type=Path,
        default=DEFAULT_CONSTRUCTOR_FAMILY,
    )
    parser.add_argument(
        "--hole-family",
        type=Path,
        default=DEFAULT_HOLE_CONSTRUCTOR_FAMILY,
    )
    parser.add_argument("--reference-token", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    required = (
        args.manifest,
        args.number_traces,
        args.program,
        args.model,
        args.tokenizer,
        args.outer_family,
        args.hole_family,
    )
    missing = [str(path) for path in required if not path.exists()]
    require(not missing, "missing required paths: " + ", ".join(missing))
    cases = expand_cases(read_manifest(args.manifest))
    args.output.mkdir(parents=True, exist_ok=True)
    context_path = args.output / "contexts.tsv"
    trace_path = args.output / "polynomial-company.jsonl"
    summary_path = args.output / "gather-summary.json"

    context_lines = [
        "# case\tx\tA\tC\tAC; values are comma-separated token IDs"
    ]
    for case in cases:
        prefixes = read_context_prefixes(case, args.number_traces)
        context_lines.append(
            "\t".join(
                [
                case.key,
                    *(
                        ",".join(str(token) for token in prefix)
                        for prefix in prefixes
                    ),
                ]
            )
        )
    context_path.write_text("\n".join(context_lines) + "\n", encoding="utf-8")

    status = "skipped"
    process_output = "existing complete polynomial company"
    if args.force or not trace_complete(trace_path):
        command = [
            str(args.program.resolve()),
            str(args.model.resolve()),
            str(args.tokenizer.resolve()),
            str(context_path.resolve()),
            str(args.outer_family.resolve()),
            str(args.hole_family.resolve()),
            "--trace",
            str(trace_path.resolve()),
            "--reference-token",
            str(args.reference_token),
        ]
        result = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        process_output = (result.stdout + result.stderr).strip()
        status = (
            "completed"
            if result.returncode == 0 and trace_complete(trace_path)
            else "failed"
        )
        if status == "failed":
            print(process_output)
            raise SystemExit("polynomial company execution failed")

    summary = {
        "schema_version": 1,
        "artifact": "structured_polynomial_company_G_of_H",
        "contexts": len(cases),
        "context_table": str(context_path.resolve()),
        "trace": str(trace_path.resolve()),
        "status": status,
        "process_output": process_output,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(process_output)


if __name__ == "__main__":
    main()

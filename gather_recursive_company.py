#!/usr/bin/env python3
"""Run the recursive finite constructor company on retained Stories contexts.

The source context table contains the four A/C corners used by the preceding
polynomial experiment. They are flattened into independently named roots.
The C evaluator then consumes every supplied family recursively and retains a
terminal contrast family for every complete branch.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from analyze_number_demand_cubes import (
    DEFAULT_CONSTRUCTOR_FAMILY,
    DEFAULT_HOLE_CONSTRUCTOR_FAMILY,
)
from gather_grammar_relations import DEFAULT_MODEL, DEFAULT_TOKENIZER
from gather_polynomial_company import DEFAULT_OUTPUT as DEFAULT_POLYNOMIAL_DIRECTORY


ROOT = Path(__file__).resolve().parent
DEFAULT_PROGRAM = ROOT / "cps_recursive_company_metal"
DEFAULT_CONTEXTS = DEFAULT_POLYNOMIAL_DIRECTORY / "contexts.tsv"
DEFAULT_TERMINAL_FAMILY = ROOT / "grammar_observer_tokens.tsv"
DEFAULT_OUTPUT = ROOT / "work_traces" / "recursive_company"
CORNER_NAMES = ("x", "A", "C", "AC")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def flatten_roots(source: Path, destination: Path) -> int:
    lines = ["# root key\tcomma-separated token IDs"]
    count = 0
    for raw in source.read_text(encoding="utf-8").splitlines():
        if not raw or raw.startswith("#"):
            continue
        fields = raw.split("\t")
        require(len(fields) == 5, f"{source}: expected case plus four corners")
        case = fields[0]
        for corner, tokens in zip(CORNER_NAMES, fields[1:]):
            lines.append(f"{case}::{corner}\t{tokens}")
            count += 1
    require(count > 0, "context table contains no roots")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return count


def first_and_last_json(path: Path) -> tuple[dict[str, object], dict[str, object]]:
    first: dict[str, object] | None = None
    last: dict[str, object] | None = None
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if first is None:
                first = row
            last = row
    require(first is not None and last is not None, f"empty trace: {path}")
    return first, last


def trace_complete(path: Path, roots: int, widths: tuple[int, ...]) -> bool:
    if not path.is_file():
        return False
    try:
        meta, check = first_and_last_json(path)
    except (OSError, json.JSONDecodeError, SystemExit):
        return False
    leaves = roots
    demands = 0
    frontier = roots
    for width in widths:
        demands += frontier
        frontier *= width
    leaves = frontier
    return (
        meta.get("kind") == "recursive_company_meta"
        and meta.get("schema_version") == 1
        and meta.get("roots") == roots
        and tuple(meta.get("family_widths", [])) == widths
        and meta.get("demand_nodes") == demands
        and meta.get("complete_branches") == leaves
        and meta.get("maximum_calls_per_filler") == 1
        and meta.get("observation_semantics") == "continuation_composed_codata"
        and meta.get("codata_constructed_before_observation") is True
        and meta.get("root_observer_runs") == 1
        and meta.get("observations_composed") is True
        and meta.get("completion_selected") is False
        and check.get("kind") == "recursive_company_check"
        and check.get("demand_nodes") == demands
        and check.get("complete_branches") == leaves
        and check.get("maximum_calls_per_filler") == 1
        and check.get("root_observer_runs") == 1
        and check.get("composed_observations") == leaves
        and check.get("composition_steps") == leaves * len(widths)
    )


def family_width(path: Path) -> int:
    return sum(
        1
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program", type=Path, default=DEFAULT_PROGRAM)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--contexts", type=Path, default=DEFAULT_CONTEXTS)
    parser.add_argument(
        "--family",
        type=Path,
        action="append",
        default=None,
        help="recursive constructor family; repeat in depth order",
    )
    parser.add_argument(
        "--terminal-family",
        type=Path,
        default=DEFAULT_TERMINAL_FAMILY,
    )
    parser.add_argument("--reference-token", type=int, default=1)
    parser.add_argument("--metal-library", type=Path, default=ROOT / "metal_kernels.metallib")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    families = args.family or [
        DEFAULT_CONSTRUCTOR_FAMILY,
        DEFAULT_HOLE_CONSTRUCTOR_FAMILY,
    ]
    required = [
        args.program,
        args.model,
        args.tokenizer,
        args.contexts,
        args.terminal_family,
        *families,
    ]
    if not args.cpu:
        required.append(args.metal_library)
    missing = [str(path) for path in required if not path.exists()]
    require(not missing, "missing required paths: " + ", ".join(missing))

    args.output.mkdir(parents=True, exist_ok=True)
    roots_path = args.output / "roots.tsv"
    trace_path = args.output / "recursive-company.jsonl"
    summary_path = args.output / "gather-summary.json"
    root_count = flatten_roots(args.contexts, roots_path)
    widths = tuple(family_width(path) for path in families)

    status = "skipped"
    process_output = "existing complete recursive company"
    if args.force or not trace_complete(trace_path, root_count, widths):
        command = [
            str(args.program.resolve()),
            str(args.model.resolve()),
            str(args.tokenizer.resolve()),
            str(roots_path.resolve()),
        ]
        for family in families:
            command.extend(("--family", str(family.resolve())))
        command.extend(
            (
                "--terminal-family",
                str(args.terminal_family.resolve()),
                "--trace",
                str(trace_path.resolve()),
                "--reference-token",
                str(args.reference_token),
            )
        )
        if args.cpu:
            command.append("--cpu")
        else:
            command.extend(("--metal", "--metal-library", str(args.metal_library.resolve())))
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
            if result.returncode == 0
            and trace_complete(trace_path, root_count, widths)
            else "failed"
        )
        if status == "failed":
            print(process_output)
            raise SystemExit("recursive company execution failed")

    summary = {
        "schema_version": 1,
        "artifact": "recursive_dependent_polynomial_company",
        "roots": root_count,
        "family_widths": list(widths),
        "roots_table": str(roots_path.resolve()),
        "trace": str(trace_path.resolve()),
        "status": status,
        "process_output": process_output,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(process_output)


if __name__ == "__main__":
    main()

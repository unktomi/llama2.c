#!/usr/bin/env python3
"""Collect uniform three-action grammatical cubes from Stories15M.

This scheduler appends one fixed primitive extension to all four corners of
each A/B diagram and invokes the C evaluator on the resulting eight corners.
It does not select observer coordinates from model outputs, assign a scalar
score, or use the controller/attractor role while constructing a cube.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gather_grammar_behaviors import (
    DEFAULT_ACTIONS,
    ActionWord,
    expand_words,
    read_actions,
)
from gather_grammar_relations import (
    DEFAULT_MANIFEST,
    DEFAULT_MODEL,
    DEFAULT_TOKENIZER,
    ActionCase,
    expand_cases,
    read_manifest,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_PROGRAM = ROOT / "cps_grammar_cube"
DEFAULT_OBSERVERS = ROOT / "grammar_observer_tokens.tsv"
DEFAULT_OUTPUT = ROOT / "work_traces" / "grammar_cubes"


@dataclass(frozen=True)
class CubeCase:
    diagram: ActionCase
    extension: ActionWord

    @property
    def stem(self) -> str:
        diagram = self.diagram.trace_name.removesuffix(".jsonl")
        return f"{diagram}--c-{self.extension.name}"

    @property
    def terms(self) -> tuple[str, ...]:
        base = self.diagram.terms[:4]
        extended = tuple(term + self.extension.text for term in base)
        return (*base, *extended)


@dataclass(frozen=True)
class RunResult:
    case: CubeCase
    trace: Path
    jet: Path
    status: str
    seconds: float
    summary: str


def trace_complete(trace: Path, jet: Path) -> bool:
    if not trace.is_file() or not jet.is_file():
        return False
    meta: dict[str, Any] | None = None
    check: dict[str, Any] | None = None
    terminal_count = 0
    local_rows = 0
    maximum_binary_end = 0
    try:
        with trace.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                kind = row.get("kind")
                if meta is None:
                    meta = row
                if kind == "grammatical_cube_local_jet":
                    local_rows += 1
                    maximum_binary_end = max(
                        maximum_binary_end,
                        int(row["binary_byte_offset"])
                        + 4 * int(row["binary_float32_count"]),
                    )
                elif kind == "grammatical_cube_terminal":
                    terminal_count += 1
                elif kind == "grammatical_cube_check":
                    check = row
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return False
    return (
        meta is not None
        and meta.get("kind") == "grammatical_cube_meta"
        and meta.get("schema_version") == 1
        and check is not None
        and int(check.get("typed_boundaries", -1)) == 80
        and local_rows == 160
        and terminal_count == 1
        and maximum_binary_end == jet.stat().st_size
    )


def run_case(
    case: CubeCase,
    program: Path,
    model: Path,
    tokenizer: Path,
    observers: Path,
    reference_token: int,
    output: Path,
    force: bool,
) -> RunResult:
    trace = output / f"{case.stem}.jsonl"
    jet = output / f"{case.stem}.f32"
    if trace_complete(trace, jet) and not force:
        return RunResult(case, trace, jet, "skipped", 0.0, "existing cube")
    command = [
        str(program),
        str(model),
        str(tokenizer),
        *case.terms,
        str(observers),
        "--reference-token",
        str(reference_token),
        "--jet-bin",
        str(jet),
        "--trace",
        str(trace),
    ]
    started = time.monotonic()
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    seconds = time.monotonic() - started
    combined = (result.stdout + result.stderr).strip()
    summary = combined.splitlines()[-1] if combined else "no process output"
    status = (
        "completed"
        if result.returncode == 0 and trace_complete(trace, jet)
        else "failed"
    )
    return RunResult(case, trace, jet, status, seconds, summary)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--actions", type=Path, default=DEFAULT_ACTIONS)
    parser.add_argument("--program", type=Path, default=DEFAULT_PROGRAM)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--observers", type=Path, default=DEFAULT_OBSERVERS)
    parser.add_argument("--reference-token", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.jobs <= 0:
        raise SystemExit("--jobs must be positive")
    required = (
        args.manifest,
        args.actions,
        args.program,
        args.model,
        args.tokenizer,
        args.observers,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing required paths: " + ", ".join(missing))
    diagrams = expand_cases(read_manifest(args.manifest), "all")
    extensions = [
        word
        for word in expand_words(read_actions(args.actions))
        if len(word.factors) == 1
    ]
    if len(diagrams) != 88 or len(extensions) != 9:
        raise SystemExit("expected 88 diagrams and nine primitive extensions")
    cases = [CubeCase(diagram, extension) for diagram in diagrams for extension in extensions]
    args.output.mkdir(parents=True, exist_ok=True)
    print(
        f"collecting {len(cases)} cubes = {len(diagrams)} diagrams x "
        f"{len(extensions)} uniform extensions",
        flush=True,
    )

    results: list[RunResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                run_case,
                case,
                args.program,
                args.model,
                args.tokenizer,
                args.observers,
                args.reference_token,
                args.output,
                args.force,
            ): case
            for case in cases
        }
        for completed, future in enumerate(
            concurrent.futures.as_completed(futures),
            start=1,
        ):
            result = future.result()
            results.append(result)
            if result.status == "failed" or completed % 10 == 0 or completed == len(cases):
                print(
                    f"[{completed:03d}/{len(cases):03d}] {result.status:9s} "
                    f"{result.case.stem} seconds={result.seconds:.3f} "
                    f"{result.summary}",
                    flush=True,
                )

    results.sort(key=lambda item: item.case.stem)
    status_counts = Counter(result.status for result in results)
    summary = {
        "schema_version": 1,
        "manifest": str(args.manifest.resolve()),
        "actions": str(args.actions.resolve()),
        "model": str(args.model.resolve()),
        "tokenizer": str(args.tokenizer.resolve()),
        "observers": str(args.observers.resolve()),
        "reference_token": args.reference_token,
        "diagram_count": len(diagrams),
        "extension_count": len(extensions),
        "cube_count": len(cases),
        "status_counts": dict(status_counts),
        "extensions": [
            {
                "name": extension.name,
                "family": extension.family,
                "text": extension.text,
            }
            for extension in extensions
        ],
        "cases": [
            {
                "stem": result.case.stem,
                "phase": result.case.diagram.phase,
                "template": result.case.diagram.template,
                "family": result.case.diagram.family,
                "role": result.case.diagram.role,
                "extension": result.case.extension.name,
                "extension_family": result.case.extension.family,
                "trace": result.trace.name,
                "jet": result.jet.name,
                "status": result.status,
                "seconds": result.seconds,
            }
            for result in results
        ],
    }
    (args.output / "gather-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    failures = [result for result in results if result.status == "failed"]
    if failures:
        raise SystemExit(f"{len(failures)} grammatical cubes failed")


if __name__ == "__main__":
    main()

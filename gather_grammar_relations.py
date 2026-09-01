#!/usr/bin/env python3
"""Run the C grammatical-action evaluator over matched relation diagrams.

This file only expands the explicit sentence manifest and schedules the C
executable.  It does not calculate a grammatical score or alter any action
diagram to make tokenization succeed.  The C evaluator must accept each
diagram as an exactly typed constructor square.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "grammar_relation_cases.json"
DEFAULT_PROGRAM = ROOT / "cps_grammar_actions"
DEFAULT_MODEL = ROOT.parent / "llama2.c" / "test" / "stories15M.bin"
DEFAULT_TOKENIZER = ROOT / "tokenizer.bin"
DEFAULT_OUTPUT = ROOT / "work_traces" / "grammar_relations"


@dataclass(frozen=True)
class ActionCase:
    phase: str
    template: str
    trace_prefix: str
    family: str
    role: str
    terms: tuple[str, str, str, str, str]

    @property
    def trace_name(self) -> str:
        return f"{self.trace_prefix}-{self.family}-{self.role}.jsonl"


@dataclass(frozen=True)
class RunResult:
    case: ActionCase
    trace: Path
    status: str
    seconds: float
    summary: str


def read_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise SystemExit("grammar relation manifest must have schema version 1")
    return manifest


def render_terms(
    template: dict[str, Any],
    family: dict[str, Any],
    role: str,
) -> tuple[str, str, str, str, str]:
    sentence = template[role]

    def render(target: str, verb: str) -> str:
        return sentence.format(
            context=family["context"],
            target=target,
            verb=verb,
        )

    x = render(family["target"], family["verb_singular"])
    ax = render(family["target_plural"], family["verb_singular"])
    bx = render(family["target"], family["verb_plural"])
    abx = render(family["target_plural"], family["verb_plural"])
    return x, ax, bx, abx, abx


def trace_complete(path: Path, pullback_depth: int) -> bool:
    if not path.is_file():
        return False
    meta: dict[str, Any] | None = None
    last_row: dict[str, Any] | None = None
    try:
        with path.open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    row = json.loads(line)
                    if meta is None:
                        meta = row
                    last_row = row
    except (OSError, json.JSONDecodeError):
        return False
    return (
        meta is not None
        and meta.get("kind") == "grammatical_action_meta"
        and meta.get("schema_version") == 5
        and meta.get("block_pullback_depth") == pullback_depth
        and last_row is not None
        and last_row.get("kind") == "grammatical_action_check"
    )


def expand_cases(manifest: dict[str, Any], phase: str) -> list[ActionCase]:
    phases = (
        ("exploration", "exploration_templates", "exploration_families"),
        ("confirmation", "confirmation_templates", "confirmation_families"),
    )
    cases: list[ActionCase] = []
    for phase_name, template_key, family_key in phases:
        if phase not in ("all", phase_name):
            continue
        for template in manifest[template_key]:
            for family in manifest[family_key]:
                for role in ("controller", "attractor"):
                    cases.append(
                        ActionCase(
                            phase=phase_name,
                            template=template["name"],
                            trace_prefix=template["trace_prefix"],
                            family=family["name"],
                            role=role,
                            terms=render_terms(template, family, role),
                        )
                    )
    names = [case.trace_name for case in cases]
    if len(names) != len(set(names)):
        raise SystemExit("manifest expands to duplicate trace names")
    return cases


def run_case(
    case: ActionCase,
    program: Path,
    model: Path,
    tokenizer: Path,
    output: Path,
    pullback_depth: int,
    force: bool,
) -> RunResult:
    trace = output / case.trace_name
    if trace_complete(trace, pullback_depth) and not force:
        return RunResult(case, trace, "skipped", 0.0, "existing trace")
    command = [
        str(program),
        str(model),
        str(tokenizer),
        *case.terms,
        "--root",
        "last",
        "--pullback-depth",
        str(pullback_depth),
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
        if result.returncode == 0 and trace_complete(trace, pullback_depth)
        else "failed"
    )
    return RunResult(case, trace, status, seconds, summary)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--program", type=Path, default=DEFAULT_PROGRAM)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--phase",
        choices=("all", "exploration", "confirmation"),
        default="all",
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--pullback-depth", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.jobs <= 0:
        raise SystemExit("--jobs must be positive")
    if args.pullback_depth <= 0:
        raise SystemExit("--pullback-depth must be positive")
    required = (args.manifest, args.program, args.model, args.tokenizer)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing required paths: " + ", ".join(missing))
    manifest = read_manifest(args.manifest)
    cases = expand_cases(manifest, args.phase)
    args.output.mkdir(parents=True, exist_ok=True)

    results: list[RunResult] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.jobs
    ) as executor:
        futures = {
            executor.submit(
                run_case,
                case,
                args.program,
                args.model,
                args.tokenizer,
                args.output,
                args.pullback_depth,
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
            print(
                f"[{completed:03d}/{len(cases):03d}] {result.status:9s} "
                f"{result.case.trace_name} seconds={result.seconds:.3f} "
                f"{result.summary}",
                flush=True,
            )

    results.sort(key=lambda item: item.case.trace_name)
    summary = {
        "schema_version": 1,
        "manifest": str(args.manifest.resolve()),
        "model": str(args.model.resolve()),
        "tokenizer": str(args.tokenizer.resolve()),
        "phase": args.phase,
        "pullback_depth": args.pullback_depth,
        "cases": [
            {
                "trace": result.case.trace_name,
                "phase": result.case.phase,
                "template": result.case.template,
                "family": result.case.family,
                "role": result.case.role,
                "status": result.status,
                "seconds": result.seconds,
                "summary": result.summary,
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
        raise SystemExit(f"{len(failures)} grammatical action cases failed")


if __name__ == "__main__":
    main()

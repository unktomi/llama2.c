#!/usr/bin/env python3
"""Collect aligned controller-number x attractor-number x verb cubes.

The manifest supplies real subject--verb and PP-attractor constructions. For
each attractor sentence, A pluralizes the attractor, C pluralizes the
controller, and B changes the singular verb constructor to its plural form.
The C evaluator must accept the resulting eight strings as one exactly aligned
constructor cube. No padding, scalar score, or role-dependent model operation
is introduced.
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

from gather_grammar_cubes import (
    DEFAULT_OBSERVERS,
    DEFAULT_PROGRAM,
    trace_complete as cube_trace_complete,
)
from gather_grammar_relations import (
    DEFAULT_MANIFEST,
    DEFAULT_MODEL,
    DEFAULT_TOKENIZER,
    read_manifest,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "work_traces" / "number_demand_cubes"


@dataclass(frozen=True)
class NumberDemandCase:
    phase: str
    template: str
    trace_prefix: str
    family: str
    controller: str
    controller_plural: str
    attractor: str
    attractor_plural: str
    verb_singular: str
    verb_plural: str
    terms: tuple[str, str, str, str, str, str, str, str]

    @property
    def trace_name(self) -> str:
        return f"number-{self.trace_prefix}-{self.family}.jsonl"

    @property
    def key(self) -> str:
        return f"{self.phase}/{self.template}/{self.family}"


@dataclass(frozen=True)
class RunResult:
    case: NumberDemandCase
    trace: Path
    status: str
    seconds: float
    summary: str


def expand_cases(manifest: dict[str, Any]) -> list[NumberDemandCase]:
    all_families = [
        *manifest["exploration_families"],
        *manifest["confirmation_families"],
    ]
    plural = {str(family["target"]): str(family["target_plural"]) for family in all_families}
    phases = (
        ("exploration", "exploration_templates", "exploration_families"),
        ("confirmation", "confirmation_templates", "confirmation_families"),
    )
    cases: list[NumberDemandCase] = []
    for phase, template_key, family_key in phases:
        for template in manifest[template_key]:
            sentence = str(template["attractor"])
            for family in manifest[family_key]:
                controller = str(family["context"])
                if controller not in plural:
                    raise SystemExit(f"no independent plural constructor for {controller!r}")
                controller_plural = plural[controller]
                attractor = str(family["target"])
                attractor_plural = str(family["target_plural"])
                verb_singular = str(family["verb_singular"])
                verb_plural = str(family["verb_plural"])

                def render(controller_value: str, attractor_value: str, verb: str) -> str:
                    return sentence.format(
                        context=controller_value,
                        target=attractor_value,
                        verb=verb,
                    )

                terms = (
                    render(controller, attractor, verb_singular),
                    render(controller, attractor_plural, verb_singular),
                    render(controller, attractor, verb_plural),
                    render(controller, attractor_plural, verb_plural),
                    render(controller_plural, attractor, verb_singular),
                    render(controller_plural, attractor_plural, verb_singular),
                    render(controller_plural, attractor, verb_plural),
                    render(controller_plural, attractor_plural, verb_plural),
                )
                cases.append(
                    NumberDemandCase(
                        phase=phase,
                        template=str(template["name"]),
                        trace_prefix=str(template["trace_prefix"]),
                        family=str(family["name"]),
                        controller=controller,
                        controller_plural=controller_plural,
                        attractor=attractor,
                        attractor_plural=attractor_plural,
                        verb_singular=verb_singular,
                        verb_plural=verb_plural,
                        terms=terms,
                    )
                )
    names = [case.trace_name for case in cases]
    if len(cases) != 44 or len(names) != len(set(names)):
        raise SystemExit("number-demand manifest did not expand to 44 unique cubes")
    return cases


def trace_complete(path: Path) -> bool:
    if not cube_trace_complete(path, None, True):
        return False
    try:
        with path.open(encoding="utf-8") as source:
            meta = json.loads(next(source))
    except (OSError, StopIteration, json.JSONDecodeError):
        return False
    return (
        meta.get("c_action_kind") == "aligned_constructor"
        and meta.get("edge_c_difference_typed") is True
        and meta.get("base_positions") == meta.get("extended_positions")
        and meta.get("extension_token_count") == 0
    )


def run_case(
    case: NumberDemandCase,
    program: Path,
    model: Path,
    tokenizer: Path,
    observers: Path,
    reference_token: int,
    output: Path,
    force: bool,
) -> RunResult:
    trace = output / case.trace_name
    if trace_complete(trace) and not force:
        return RunResult(case, trace, "skipped", 0.0, "existing aligned cube")
    command = [
        str(program),
        str(model),
        str(tokenizer),
        *case.terms,
        str(observers),
        "--reference-token",
        str(reference_token),
        "--terminal-only",
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
        if result.returncode == 0 and trace_complete(trace)
        else "failed"
    )
    return RunResult(case, trace, status, seconds, summary)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
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
        args.program,
        args.model,
        args.tokenizer,
        args.observers,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing required paths: " + ", ".join(missing))
    cases = expand_cases(read_manifest(args.manifest))
    args.output.mkdir(parents=True, exist_ok=True)

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
            print(
                f"[{completed:02d}/{len(cases):02d}] {result.status:9s} "
                f"{result.case.trace_name} seconds={result.seconds:.3f} "
                f"{result.summary}",
                flush=True,
            )

    results.sort(key=lambda result: result.case.trace_name)
    summary = {
        "schema_version": 1,
        "artifact": "aligned_controller_attractor_verb_number_cubes",
        "manifest": str(args.manifest.resolve()),
        "model": str(args.model.resolve()),
        "tokenizer": str(args.tokenizer.resolve()),
        "observers": str(args.observers.resolve()),
        "actions": {
            "A": "pluralize attractor noun",
            "B": "select plural rather than singular verb constructor",
            "C": "pluralize grammatical controller noun",
        },
        "cases": [
            {
                "key": result.case.key,
                "trace": result.case.trace_name,
                "controller": result.case.controller,
                "controller_plural": result.case.controller_plural,
                "attractor": result.case.attractor,
                "attractor_plural": result.case.attractor_plural,
                "verb_singular": result.case.verb_singular,
                "verb_plural": result.case.verb_plural,
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
        for result in failures:
            print(f"failed {result.case.key}: {result.summary}")
        raise SystemExit(f"{len(failures)} aligned number-demand cubes failed")


if __name__ == "__main__":
    main()

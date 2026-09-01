#!/usr/bin/env python3
"""Consume every fixed verb injection in every aligned number context.

The evaluator's A and B axes are attractor and controller number.  Its C axis
chooses one of a fixed pair of verb injections.  Six pairs therefore cover the
declared twelve-verb coproduct in each of the 44 contexts.  The retained edge
immediately after the verb is the hole observer for that consumed injection.
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

from analyze_grammar_cubes import read_observers
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
from analyze_number_demand_cubes import DEFAULT_CONSTRUCTOR_FAMILY


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "work_traces" / "post_injection_holes"


@dataclass(frozen=True)
class HoleCase:
    phase: str
    template: str
    trace_prefix: str
    family: str
    pair_index: int
    controller: str
    controller_plural: str
    attractor: str
    attractor_plural: str
    first_token: int
    first_verb: str
    second_token: int
    second_verb: str
    terms: tuple[str, str, str, str, str, str, str, str]

    @property
    def key(self) -> str:
        return f"{self.phase}/{self.template}/{self.family}"

    @property
    def trace_name(self) -> str:
        return (
            f"holes-{self.trace_prefix}-{self.family}-"
            f"v{self.pair_index:02d}-{self.first_token}-{self.second_token}.jsonl"
        )


@dataclass(frozen=True)
class RunResult:
    case: HoleCase
    trace: Path
    status: str
    seconds: float
    summary: str


def expand_hole_cases(
    manifest: dict[str, Any],
    constructor_ids: tuple[int, ...],
    constructor_labels: dict[int, str],
) -> list[HoleCase]:
    if len(constructor_ids) < 2 or len(constructor_ids) % 2 != 0:
        raise SystemExit("fixed verb constructor family must contain an even number of injections")
    verb_pairs = [
        (
            constructor_ids[index],
            constructor_labels[constructor_ids[index]].strip(),
            constructor_ids[index + 1],
            constructor_labels[constructor_ids[index + 1]].strip(),
        )
        for index in range(0, len(constructor_ids), 2)
    ]
    families = [
        *manifest["exploration_families"],
        *manifest["confirmation_families"],
    ]
    plural = {
        str(family["target"]): str(family["target_plural"])
        for family in families
    }
    phases = (
        ("exploration", "exploration_templates", "exploration_families"),
        ("confirmation", "confirmation_templates", "confirmation_families"),
    )
    cases: list[HoleCase] = []
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

                def render(controller_value: str, attractor_value: str, verb: str) -> str:
                    return sentence.format(
                        context=controller_value,
                        target=attractor_value,
                        verb=verb,
                    )

                for pair_index, (first_token, first_verb, second_token, second_verb) in enumerate(verb_pairs):
                    terms = (
                        render(controller, attractor, first_verb),
                        render(controller, attractor_plural, first_verb),
                        render(controller_plural, attractor, first_verb),
                        render(controller_plural, attractor_plural, first_verb),
                        render(controller, attractor, second_verb),
                        render(controller, attractor_plural, second_verb),
                        render(controller_plural, attractor, second_verb),
                        render(controller_plural, attractor_plural, second_verb),
                    )
                    cases.append(
                        HoleCase(
                            phase=phase,
                            template=str(template["name"]),
                            trace_prefix=str(template["trace_prefix"]),
                            family=str(family["name"]),
                            pair_index=pair_index,
                            controller=controller,
                            controller_plural=controller_plural,
                            attractor=attractor,
                            attractor_plural=attractor_plural,
                            first_token=first_token,
                            first_verb=first_verb,
                            second_token=second_token,
                            second_verb=second_verb,
                            terms=terms,
                        )
                    )
    expected = 44 * len(verb_pairs)
    names = [case.trace_name for case in cases]
    if len(cases) != expected or len(names) != len(set(names)):
        raise SystemExit("post-injection manifest did not expand to unique full coverage")
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
    case: HoleCase,
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
    parser.add_argument(
        "--constructor-family",
        type=Path,
        default=DEFAULT_CONSTRUCTOR_FAMILY,
    )
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
        args.constructor_family,
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing required paths: " + ", ".join(missing))
    constructor_ids, constructor_labels = read_observers(args.constructor_family)
    cases = expand_hole_cases(
        read_manifest(args.manifest),
        constructor_ids,
        constructor_labels,
    )
    args.output.mkdir(parents=True, exist_ok=True)

    results: list[RunResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                run_case,
                case,
                args.program.resolve(),
                args.model.resolve(),
                args.tokenizer.resolve(),
                args.observers.resolve(),
                args.reference_token,
                args.output.resolve(),
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

    results.sort(key=lambda result: result.case.trace_name)
    summary = {
        "schema_version": 1,
        "artifact": "full_fixed_verb_post_injection_hole_cubes",
        "model": str(args.model.resolve()),
        "tokenizer": str(args.tokenizer.resolve()),
        "observers": str(args.observers.resolve()),
        "constructor_family": str(args.constructor_family.resolve()),
        "actions": {
            "A": "pluralize attractor noun",
            "B": "pluralize grammatical controller noun",
            "C": "choose the second fixed verb injection in the pair",
        },
        "contexts": 44,
        "fixed_injections_per_context": len(constructor_ids),
        "traces": len(results),
        "cases": [
            {
                "key": result.case.key,
                "trace": result.case.trace_name,
                "pair_index": result.case.pair_index,
                "injections": [
                    {"token": result.case.first_token, "text": result.case.first_verb},
                    {"token": result.case.second_token, "text": result.case.second_verb},
                ],
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
            print(f"failed {result.case.key} pair={result.case.pair_index}: {result.summary}")
        raise SystemExit(f"{len(failures)} post-injection cubes failed")


if __name__ == "__main__":
    main()

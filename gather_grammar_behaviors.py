#!/usr/bin/env python3
"""Collect exact future-company root behavior for grammatical diagrams.

Every action word is appended identically to all five terms of an existing
constructor square.  This scheduler neither labels the resulting behavior nor
scores a completion; it only persists the C evaluator's complete root vectors.
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

from gather_grammar_relations import (
    DEFAULT_MANIFEST,
    DEFAULT_MODEL,
    DEFAULT_PROGRAM,
    DEFAULT_TOKENIZER,
    ActionCase,
    expand_cases,
    read_manifest,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_ACTIONS = ROOT / "grammar_future_actions.json"
DEFAULT_OUTPUT = ROOT / "work_traces" / "grammar_behaviors"


@dataclass(frozen=True)
class ActionWord:
    name: str
    text: str
    family: str
    factors: tuple[str, ...]
    repeats: str | None = None


@dataclass(frozen=True)
class BehaviorCase:
    diagram: ActionCase
    word: ActionWord

    @property
    def trace_name(self) -> str:
        stem = self.diagram.trace_name.removesuffix(".jsonl")
        return f"{stem}--w-{self.word.name}.jsonl"

    @property
    def terms(self) -> tuple[str, str, str, str, str]:
        return tuple(term + self.word.text for term in self.diagram.terms)  # type: ignore[return-value]


@dataclass(frozen=True)
class RunResult:
    case: BehaviorCase
    trace: Path
    status: str
    seconds: float
    summary: str


def read_actions(path: Path) -> dict[str, Any]:
    actions = json.loads(path.read_text(encoding="utf-8"))
    if actions.get("schema_version") != 1:
        raise SystemExit("future-company actions must have schema version 1")
    if actions.get("composition_depth") != 2:
        raise SystemExit("the current future-company scheduler requires depth 2")
    return actions


def expand_words(actions: dict[str, Any]) -> list[ActionWord]:
    identity = actions["identity"]
    words = [
        ActionWord(identity["name"], identity["text"], "identity", ()),
    ]
    for control in actions.get("repeat_controls", []):
        words.append(
            ActionWord(
                control["name"],
                control["text"],
                "repeat_control",
                (),
                control["repeats"],
            )
        )
    primitives: list[ActionWord] = []
    composable: list[ActionWord] = []
    for primitive in actions["primitives"]:
        word = ActionWord(
            primitive["name"],
            primitive["text"],
            primitive["family"],
            (primitive["name"],),
        )
        primitives.append(word)
        if primitive.get("compose") is True:
            composable.append(word)
    words.extend(primitives)
    for left in composable:
        for right in composable:
            words.append(
                ActionWord(
                    f"{left.name}__{right.name}",
                    left.text + right.text,
                    "composed_depth_2",
                    (left.name, right.name),
                )
            )
    names = [word.name for word in words]
    if len(names) != len(set(names)):
        raise SystemExit("future-company action names are not unique")
    return words


def trace_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    first: dict[str, Any] | None = None
    last: dict[str, Any] | None = None
    root_count = 0
    try:
        with path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                if first is None:
                    first = row
                if row.get("kind") == "grammatical_behavior_root":
                    root_count += 1
                last = row
    except (OSError, json.JSONDecodeError):
        return False
    return (
        first is not None
        and first.get("kind") == "grammatical_behavior_meta"
        and first.get("schema_version") == 1
        and root_count == 1
        and last is not None
        and last.get("kind") == "grammatical_behavior_check"
    )


def run_case(
    case: BehaviorCase,
    program: Path,
    model: Path,
    tokenizer: Path,
    output: Path,
    force: bool,
) -> RunResult:
    trace = output / case.trace_name
    if trace_complete(trace) and not force:
        return RunResult(case, trace, "skipped", 0.0, "existing trace")
    command = [
        str(program),
        str(model),
        str(tokenizer),
        *case.terms,
        "--root",
        "last",
        "--behavior-only",
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
    parser.add_argument("--actions", type=Path, default=DEFAULT_ACTIONS)
    parser.add_argument("--program", type=Path, default=DEFAULT_PROGRAM)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
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
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing required paths: " + ", ".join(missing))
    diagrams = expand_cases(read_manifest(args.manifest), "all")
    words = expand_words(read_actions(args.actions))
    cases = [BehaviorCase(diagram, word) for diagram in diagrams for word in words]
    args.output.mkdir(parents=True, exist_ok=True)
    print(
        f"collecting {len(cases)} traces = {len(diagrams)} diagrams x "
        f"{len(words)} action words",
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
            if result.status == "failed" or completed % 25 == 0 or completed == len(cases):
                print(
                    f"[{completed:04d}/{len(cases):04d}] {result.status:9s} "
                    f"{result.case.trace_name} seconds={result.seconds:.3f} "
                    f"{result.summary}",
                    flush=True,
                )

    results.sort(key=lambda item: item.case.trace_name)
    summary = {
        "schema_version": 1,
        "manifest": str(args.manifest.resolve()),
        "actions": str(args.actions.resolve()),
        "model": str(args.model.resolve()),
        "tokenizer": str(args.tokenizer.resolve()),
        "diagram_count": len(diagrams),
        "action_word_count": len(words),
        "trace_count": len(cases),
        "words": [
            {
                "name": word.name,
                "family": word.family,
                "factors": list(word.factors),
                "repeats": word.repeats,
                "text": word.text,
            }
            for word in words
        ],
        "cases": [
            {
                "trace": result.case.trace_name,
                "phase": result.case.diagram.phase,
                "template": result.case.diagram.template,
                "family": result.case.diagram.family,
                "role": result.case.diagram.role,
                "word": result.case.word.name,
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
        raise SystemExit(f"{len(failures)} future-company traces failed")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Archived regression for exhaustive-prefix-company-don't-do-this.c.

This is intentionally not named test_*.py after quarantine: automatic test
discovery must not execute or lend legitimacy to the rejected implementation.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "test" / "stories260K.bin"
TOKENIZER = ROOT / "test" / "tok512.bin"
STRENGTH = ROOT / "run_escardo_strength"
REFERENCE = ROOT / "run_atkey_term"
SOURCE = ROOT / "exhaustive-prefix-company-don't-do-this.c"
PROMPT = "Lily was"


def run(arguments: list[str]) -> str:
    completed = subprocess.run(
        arguments,
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout


def field(pattern: str, output: str) -> str:
    match = re.search(pattern, output, re.MULTILINE)
    if match is None:
        raise AssertionError(f"missing output field: {pattern}")
    return match.group(1)


def function_body(source: str, declaration: str) -> str:
    start = source.index(declaration)
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        character = source[index]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
    raise AssertionError(f"unterminated function: {declaration}")


def real_model_backward_induction() -> None:
    output = run(
        [
            str(STRENGTH),
            str(MODEL),
            str(TOKENIZER),
            PROMPT,
            "--verify",
        ]
    )
    reference = run(
        [
            str(REFERENCE),
            str(MODEL),
            "-z", str(TOKENIZER),
            "-n", "2",
            "-k", "512",
            "-i", PROMPT,
        ]
    )

    selected = ast.literal_eval(field(r"^selected_tokens=(\[[^\n]+\])$", output))
    reference_selected = ast.literal_eval(
        field(r"^selected_tokens=(\[[^\n]+\])$", reference)
    )
    assert selected == [261, 376]
    assert selected == reference_selected
    assert field(r"^completion: (.*)$", output) == " a little"
    assert float(field(r"^selected_reward=([^\n]+)$", output)) == float(
        field(r"^selected_reward=([^\n]+)$", reference)
    )

    assert "terminalization=tau_once" in output
    assert "demanded_prefix_tree_nodes=515" in output
    assert "output_contexts=513" in output
    assert "reached_pairs=262144" in output
    assert "learned_filler_count=48 one_shot_crossings=48" in output
    assert "llama2_oracle_logits=262656" in output
    assert "bit_mismatches=0" in output
    assert "tolerance_failures=0" in output
    assert "max_abs_error=0" in output

    reads = int(field(r"coefficient_reads=(\d+)", output))
    uses = int(field(r"logical_uses=(\d+)", output))
    assert reads == 292800
    assert uses == 133883776
    assert uses > 400 * reads


def source_schedule_is_not_eager_ar() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    run_program = function_body(source, "static ProgramResult run_program")
    company = function_body(
        source,
        "static LogitCompany evaluate_prefix_tree_company",
    )
    oracle = function_body(
        source,
        "static Verification verify_against_llama2_forward",
    )
    main = function_body(source, "int main")

    assert "forward(" not in run_program
    assert "forward(" not in company
    assert "forward(" in oracle
    assert source.count("ProgramResult result = run_program(&program)") == 1
    assert main.index("ledger_total_reads(&ledger) != 0") < main.index(
        "run_program(&program)"
    )
    for required in (
        "RanConstConstSelectToken",
        "RanConstConstSelectPair",
        "RanConstConstContPair",
        "dependent_product_strength",
        "selection_to_continuation",
        "remembered_suffixes",
        "resident_row",
        "family_context_member",
    ):
        assert required in source


def main() -> None:
    missing = [
        path
        for path in (MODEL, TOKENIZER, STRENGTH, REFERENCE, SOURCE)
        if not path.exists()
    ]
    if missing:
        raise SystemExit(
            "missing strength system-test artifacts: "
            + ", ".join(map(str, missing))
        )
    real_model_backward_induction()
    source_schedule_is_not_eager_ar()
    print("C prefix-tree Escardo strength system test: ALL OK")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Archived regression for exhaustive-logit-grid-don't-do-this.c.

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
PROGRAM = ROOT / "run_escardo_logit_strength"
SOURCE = ROOT / "exhaustive-logit-grid-don't-do-this.c"
PROMPT = "Lily was"


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


def real_model_complete_company() -> None:
    completed = subprocess.run(
        [str(PROGRAM), str(MODEL), str(TOKENIZER), PROMPT, "--verify"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    output = completed.stdout

    selected = ast.literal_eval(field(r"^selected_tokens=(\[[^\n]+\])$", output))
    assert selected == [261, 376, 298]
    assert field(r"^completion: (.*)$", output) == " a little g"
    assert "selection_carrier=ModelLogit(row,token,value,log_probability)" in output
    assert "selection_term=dependent_product_3(Select_Reward_Logit)" in output
    assert "terminalization=tau_once" in output

    assert "company_nodes=262659 output_logit_rows=262657" in output
    assert "retained_layer_scales=6 retained_rows_per_scale=262659" in output
    assert "row_selection_applications=262657" in output
    assert "candidate_observations=134480384" in output
    assert "terminal_observations=134480385" in output
    assert "learned_filler_count=48 one_shot_crossings=48" in output
    assert "coefficient_reads=292800" in output
    assert "logical_uses=68316489600" in output

    assert "llama2_oracle_logits=134480384" in output
    assert "bit_mismatches=0" in output
    assert "tolerance_failures=0" in output
    assert "max_abs_error=0" in output


def sampled_backward_induction_reaches_the_exact_small_support() -> None:
    completed = subprocess.run(
        [
            str(PROGRAM),
            str(MODEL),
            str(TOKENIZER),
            PROMPT,
            "-k", "4",
            "--sample-ms", "1",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    output = completed.stdout

    assert "selection_mode=sampled_reachability_backward_induction" in output
    assert "selection_backup=max_complete_observer_reward" in output
    assert (
        "coverage_diagnostic="
        "logsumexp_of_distinct_reached_path_probabilities"
    ) in output
    assert "sampling_stop_reason=support_exhausted" in output
    assert "sampled_unique_roots=4" in output
    assert "sampled_unique_pairs=16" in output
    assert "sampled_unique_triples=64" in output
    assert "pair_paths=4" in output
    assert "root_paths=16" in output
    selected = ast.literal_eval(field(r"^selected_tokens=(\[[^\n]+\])$", output))
    assert selected == [261, 376, 298]
    assert field(r"^completion: (.*)$", output) == " a little g"


def source_has_one_run_and_a_separate_oracle() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    evaluator = function_body(
        source,
        "static Depth3Evaluation evaluate_depth3_company",
    )
    run = function_body(source, "static Depth3LogitResult run_depth3_logit_term")
    oracle = function_body(
        source,
        "static Verification verify_depth3_against_llama2_forward",
    )
    main = function_body(source, "int main")

    assert "forward(" not in evaluator
    assert "forward(" not in run
    assert "forward(" in oracle
    assert source.count("run_depth3_logit_term(&term)") == 1
    assert main.index("ledger_total_reads(&ledger) != 0") < main.index(
        "run_depth3_logit_term(&term)"
    )
    for required in (
        "RanConstConstSelectLogit",
        "RanConstConstSelectLogitPair",
        "RanConstConstSelectLogitTriple",
        "RanConstConstContLogitTriple",
        "dependent_logit_pair_strength",
        "dependent_logit_triple_strength",
        "logit_selection_to_continuation",
        "evaluate_depth3_company",
    ):
        assert required in source


def main() -> None:
    missing = [
        path
        for path in (MODEL, TOKENIZER, PROGRAM, SOURCE)
        if not path.exists()
    ]
    if missing:
        raise SystemExit(
            "missing logit-strength system-test artifacts: "
            + ", ".join(map(str, missing))
        )
    real_model_complete_company()
    sampled_backward_induction_reaches_the_exact_small_support()
    source_has_one_run_and_a_separate_oracle()
    print("C three-logit Escardo strength system test: ALL OK")


if __name__ == "__main__":
    main()

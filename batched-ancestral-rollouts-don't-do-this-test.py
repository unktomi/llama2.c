#!/usr/bin/env python3
"""Archived regression for batched-ancestral-rollouts-don't-do-this.c.

This is intentionally not named test_*.py after quarantine: automatic test
discovery must not execute or lend legitimacy to the rejected implementation.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROGRAM = ROOT / "run_sampled_game"
MODEL = ROOT / "test" / "stories260K.bin"
TOKENIZER = ROOT / "test" / "tok512.bin"


def field(pattern: str, output: str) -> str:
    match = re.search(pattern, output, re.MULTILINE)
    if match is None:
        raise AssertionError(f"missing output field: {pattern}")
    return match.group(1)


def deterministic_stock_parity() -> None:
    completed = subprocess.run(
        [
            str(PROGRAM),
            str(MODEL),
            str(TOKENIZER),
            "Lily was",
            "-n", "3",
            "-k", "1",
            "--sample-rollouts", "1",
            "--batch", "16",
            "--verify",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    output = completed.stdout
    assert field(r"^completion: (.*)$", output) == " a little g"
    selected = ast.literal_eval(field(r"^selected_tokens=(\[[^\n]+\])$", output))
    assert selected == [261, 376, 298]
    assert "learned_filler_crossings_before_run=0" in output
    assert "selection_term=sampled_prefix_product_of_logit_selections" in output
    assert "selection_backup=max_complete_observer_reward" in output
    assert "sampling_budget=rollouts count=1" in output
    assert "attempted_rollouts=1 unique_completions=1" in output
    assert "prefix_term_nodes=4 sampled_path_edges=3" in output
    assert "crossings_per_filler_min=3 crossings_per_filler_max=3" in output
    assert "llama2_selected_edge_log_probabilities=3 failures=0" in output
    assert "max_abs_error=0" in output

    selected_reward = float(field(
        r"^selected_token_count=3 selected_leaf_log_reward=([^ ]+)",
        output,
    ))
    root_reward = float(field(r"root_best_reached_reward=([^ ]+)", output))
    assert selected_reward == root_reward


def sampled_product_retains_a_complete_nonlocal_witness() -> None:
    completed = subprocess.run(
        [
            str(PROGRAM),
            str(MODEL),
            str(TOKENIZER),
            "Lily was",
            "-n", "8",
            "-k", "4",
            "--sample-rollouts", "16",
            "--batch", "16",
            "--verify",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    output = completed.stdout
    selected = ast.literal_eval(field(r"^selected_tokens=(\[[^\n]+\])$", output))
    assert selected == [261, 376, 298, 315, 421, 395, 317, 263]
    assert "attempted_rollouts=16 unique_completions=15" in output
    assert "root_completions=15" in output
    assert "sampling_bonus=none" in output
    assert "coverage_diagnostic=" in output
    assert "llama2_selected_edge_log_probabilities=8 failures=0" in output


if __name__ == "__main__":
    deterministic_stock_parity()
    sampled_product_retains_a_complete_nonlocal_witness()
    print("C batched sampled prefix-product system test: ALL OK")

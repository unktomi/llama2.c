#!/usr/bin/env python3
"""DO NOT RUN: checks for the rejected token/prefix-field evaluator.

These checks exercised a recursion whose carrier was a token and whose model
observer ran through ``sample_field`` at each forced prefix.  Passing them did
not establish composed logit selections or one-shot learned fillers.
"""

from __future__ import annotations

import ast
import re
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "test" / "stories260K.bin"
TOKENIZER = ROOT / "test" / "tok512.bin"
TERM = ROOT / "run_atkey_term"
STOCK = ROOT / "run"
SOURCE = ROOT / "atkey_term.c"
BRIDGE = ROOT / "atkey_term_c.c"
QUARANTINED = (
    ROOT / "categorical-rollout-shortcut-don't-do-this.c",
    ROOT / "batched-ancestral-rollouts-don't-do-this.c",
    ROOT / "entropy-distance-reward-don't-do-this.c",
    ROOT / "exhaustive-prefix-company-don't-do-this.c",
    ROOT / "exhaustive-logit-grid-don't-do-this.c",
)
SHORT_PROMPT = "Once upon a time"
LONG_PROMPT = (
    "Once upon a time there was a little girl named Lily who loved to explore "
    "the forest near her home."
)
NONLOCAL_PROMPT = "Lily was"


def run(arguments: list[str]) -> str:
    completed = subprocess.run(
        arguments,
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout


def term(*arguments: str) -> str:
    return run([str(TERM), str(MODEL), "-z", str(TOKENIZER), *arguments])


def field(pattern: str, output: str) -> str:
    match = re.search(pattern, output, re.MULTILINE | re.DOTALL)
    if match is None:
        raise AssertionError(f"missing output field: {pattern}")
    return match.group(1)


def tokens(output: str) -> list[int]:
    return ast.literal_eval(field(r"^selected_tokens=(\[[^\n]+\])$", output))


def completion(output: str) -> str:
    return field(r"^completion: (.*?)\nselected_tokens=", output)


def literal_backward_induction() -> None:
    output = term("-n", "4", "-k", "4", "-i", SHORT_PROMPT, "-d", "1")
    assert tokens(output) == [432, 383, 286, 261]
    assert float(field(r"^selected_reward=([^\n]+)$", output)) == (
        -0.11686533503512936
    )
    assert "learned_kernel_calls_before_run=0" in output
    assert len(re.findall(r"^  depth=", output, re.MULTILINE)) == 4
    assert len(re.findall(r"^      candidate_token=", output, re.MULTILINE)) == 16
    assert "score_status=pruned" not in output


def stock_greedy_numerics() -> None:
    output = term("-n", "48", "-k", "1", "-i", LONG_PROMPT)
    prompt_count = int(field(r"^prompt_token_count=(\d+)$", output))
    stock_steps = prompt_count - 1 + 48
    stock = run(
        [
            str(STOCK),
            str(MODEL),
            "-z", str(TOKENIZER),
            "-t", "0.0",
            "-n", str(stock_steps),
            "-i", LONG_PROMPT,
        ]
    )
    assert stock.rstrip("\n") == LONG_PROMPT + completion(output)
    assert len(tokens(output)) == 48


def stock_delimiter_semantics() -> None:
    output = term("-n", "218", "-k", "1", "-i", LONG_PROMPT)
    selected = tokens(output)
    assert selected[-1] == 1
    assert "termination_token=1" in output
    assert "<s>" not in completion(output)
    assert completion(output).endswith("Then, they went to the park to play.")
    prompt_count = int(field(r"^prompt_token_count=(\d+)$", output))
    stock = run(
        [
            str(STOCK),
            str(MODEL),
            "-z", str(TOKENIZER),
            "-t", "0.0",
            "-n", str(prompt_count - 1 + 218),
            "-i", LONG_PROMPT,
        ]
    )
    assert stock.rstrip("\n") == LONG_PROMPT + completion(output)


def nonlocal_selection_changes_completion() -> None:
    selected = term("-n", "8", "-k", "2", "-i", NONLOCAL_PROMPT, "-d", "1")
    greedy = term("-n", "8", "-k", "1", "-i", NONLOCAL_PROMPT)
    assert tokens(selected) == [261, 376, 298, 315, 421, 263, 415, 414]
    assert completion(selected) == " a little girl who"
    assert completion(greedy) == " a little girl named Lily w"
    assert float(field(r"^selected_reward=([^\n]+)$", selected)) == (
        -3.705980395794089
    )
    assert float(field(r"^selected_reward=([^\n]+)$", greedy)) == (
        -4.36163296662825
    )
    assert re.search(
        r"^  depth=5 selected_token=263 selected_local_rank=1 ",
        selected,
        re.MULTILINE,
    )
    stock = run(
        [
            str(STOCK),
            str(MODEL),
            "-z", str(TOKENIZER),
            "-t", "0.0",
            "-n", "10",
            "-i", NONLOCAL_PROMPT,
        ]
    )
    assert stock.rstrip("\n") == NONLOCAL_PROMPT + completion(greedy)


def terminal_length_bias_regression() -> None:
    with tempfile.TemporaryDirectory() as directory:
        audit_path = Path(directory) / "candidates.audit"
        output = term(
            "-n", "8", "-k", "4",
            "-i", "Lily was playaud",
            "-a", str(audit_path),
        )
        audit = audit_path.read_text(encoding="utf-8")
    assert tokens(output) == [295, 263, 415, 414, 401, 396, 267, 337]
    assert completion(output) == "ar who loved to play"
    assert "termination_token=none" in output
    assert float(field(r"^selected_reward=([^\n]+)$", output)) == (
        -3.6869164731891466
    )
    assert re.search(
        r'^result frame=0 rank=2 '
        r'pruned_upper_bound=-16\.806639718647006$',
        audit,
        re.MULTILINE,
    )


def boolean_satisfying_strength_fast_forwards() -> None:
    output = term("-n", "48", "-k", "4", "-i", NONLOCAL_PROMPT, "-s", "-17")
    assert tokens(output) == [
        261, 376, 298, 315, 421, 395, 317, 263,
        415, 414, 401, 396, 267, 337, 335, 311,
        267, 422, 419, 426, 385, 328, 432, 358,
        263, 377, 267, 265, 282, 295, 433, 335,
        311, 357, 343, 269, 279, 380, 418, 422,
        426, 342, 263, 377, 267, 265, 282, 295,
    ]
    assert completion(output) == (
        " a little girl named Lily who loved to play with her toys. One day, "
        "she went to the park with her mommy and daddy. They went to the par"
    )
    assert float(field(r"^selected_reward=([^\n]+)$", output)) == (
        -16.857306938428646
    )
    assert (
        "selection_goal=first_satisfying_completion reward_threshold=-17"
        in output
    )
    nodes = int(field(r"^selection_function_nodes=(\d+)", output))
    assert nodes < 1_000


def sampled_recursive_support_is_nested_as_budget_grows() -> None:
    rewards = []
    completions = []
    for budget in (8, 32, 192, 512, 4096, 16384):
        sampled = term(
            "-n", "8", "-k", "4", "-i", NONLOCAL_PROMPT,
            "--sample-demands", str(budget),
        )
        rewards.append(float(field(r"^selected_reward=([^\n]+)$", sampled)))
        completions.append(completion(sampled))
        assert int(field(r"^sampled_candidate_demands=(\d+)", sampled)) <= budget
        assert "forced_after_budget=0" in sampled
        assert (
            "selection_goal=maximum_over_recursively_demanded_support"
            in sampled
        )
        assert (
            "sampling_policy=local_top_k_without_replacement_recursive_select"
            in sampled
        )
        assert "sampled_rollouts=" not in sampled
        assert "learned_kernel_calls_before_run=0" in sampled
    assert rewards == sorted(rewards)
    assert completions[0] == " a little girl named J"
    assert completions[-1] == " a little girl who"


def sampled_support_exhaustion_matches_exact_product() -> None:
    exact = term("-n", "8", "-k", "4", "-i", NONLOCAL_PROMPT)
    sampled = term(
        "-n", "8", "-k", "4", "-i", NONLOCAL_PROMPT,
        "--sample-demands", "87380",
    )
    assert tokens(sampled) == tokens(exact)
    assert completion(sampled) == completion(exact)
    assert field(r"^selected_reward=([^\n]+)$", sampled) == field(
        r"^selected_reward=([^\n]+)$",
        exact,
    )
    assert "sampling_stop_reason=support_complete" in sampled
    assert "sampled_candidate_demands=84 forced_after_budget=0" in sampled


def wall_clock_budget_finishes_recursive_demand() -> None:
    sampled = term(
        "-n", "48", "-k", "4", "-i", NONLOCAL_PROMPT,
        "--sample-ms", "20",
    )
    assert len(tokens(sampled)) == 48
    assert (
        "selection_goal=maximum_over_recursively_demanded_support"
        in sampled
    )
    assert "sampling_budget_ms=20" in sampled
    assert "sampling_stop_reason=deadline" in sampled
    assert int(field(r"^sampled_candidate_demands=(\d+)", sampled)) >= 48


def sampled_strength_log_records_recursive_bellman_order() -> None:
    with tempfile.TemporaryDirectory() as directory:
        strength_path = Path(directory) / "strength.log"
        output = term(
            "-n", "3", "-k", "4", "-i", NONLOCAL_PROMPT,
            "--sample-demands", "3",
            "--strength-log", str(strength_path),
        )
        raw = strength_path.read_bytes()
    assert raw.endswith(b"\n")
    log = raw.decode("utf-8")
    assert log.startswith(
        'strength_log version=1 prompt="Lily was" horizon=3 top_k=4 '
    )
    events = [
        line for line in log.splitlines()
        if line.startswith("strength_event ")
    ]
    identifiers = [int(re.search(r" id=(\d+) ", line).group(1)) for line in events]
    assert identifiers == list(range(len(events)))
    assert "kind=tau_begin mode=sampled_recursive" in events[0]
    assert "kind=run_end" in events[-1]
    assert "sample_rollout" not in log
    assert "sample_edge" not in log
    assert "sample_backup" not in log

    def event_index(fragment: str) -> int:
        return next(index for index, line in enumerate(events) if fragment in line)

    root_sample = event_index("kind=continuation_sample frame=0 ordinal=0")
    sampled_rank = re.search(r" local_rank=(\d+) ", events[root_sample])
    assert sampled_rank is not None
    root_demand = event_index(
        f"kind=bellman_demand frame=0 rank={sampled_rank.group(1)}"
    )
    force_begin = event_index("kind=suffix_force_begin owner_frame=0")
    child_enter = event_index("kind=select_enter frame=1 parent_frame=0")
    child_choose = event_index("kind=select_choose frame=1")
    force_end = event_index("kind=suffix_force_end owner_frame=0")
    root_return = event_index(
        f"kind=bellman_return frame=0 rank={sampled_rank.group(1)}"
    )
    root_choose = event_index("kind=select_choose frame=0")
    tau_return = event_index("kind=tau_selection_return")
    assert (
        root_sample < root_demand < force_begin < child_enter < child_choose <
        force_end < root_return < root_choose < tau_return
    )
    island_index, island = next(
        (index, line) for index, line in enumerate(events)
        if "kind=token_role role=island" in line
    )
    occurrence = re.search(
        r"token=(\d+) occurrence_prefix=(\d+)",
        island,
    )
    assert occurrence is not None
    token, prefix = occurrence.groups()
    bridge_index, bridge = next(
        (index, line) for index, line in enumerate(events)
        if "kind=token_role role=bridge" in line and
        f"token={token} occurrence_prefix={prefix}" in line
    )
    assert island_index < bridge_index
    for line in (island, bridge):
        assert 'piece="' in line
        assert 'context_before="' in line
        assert 'context_after="' in line
    assert "strength_log=" in output


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


def composed_c_term_structure() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    bridge = BRIDGE.read_text(encoding="utf-8")
    for forbidden in (
        "SearchNode",
        "ModelState",
        "modelStep",
        "monotoneNormalForm",
        "monotoneDepthFirst",
        "selectLearned",
        "selectModelFillers",
        "PrependObserverEnvironment",
        "observe_with_prepend",
        "sample_complete_rollout",
        "backpropagate_sampled_rewards",
        "run_sampled_strength",
        "categorical_top_k_path_demand",
    ):
        assert forbidden not in source
    assert re.search(r"\bforward\s*\(", source) is None
    assert re.search(r"\bforward\s*\(", bridge) is None
    for required in (
        "struct Field",
        "int dependency;",
        "struct Completion",
        "struct SuffixThunk",
        "Outcome *(*apply)",
        "S -> (A -> B) -> T",
        "family_context",
        "layer_company",
        "network_company",
        "model_fields_term",
        "history_product_select",
        "product_suffix",
        "sample_time_allows_new_demand",
        "sample_next_local_rank",
        "outcome->prefix",
        "observe_model_fields",
        "J_R -> K_R",
        "run_pcont",
    ):
        assert required in source

    # The final model observer may pair a completion with the composed term;
    # it must not construct or execute the model.
    observer = function_body(source, "static Outcome *observe_model_fields")
    for forbidden in (
        "family_context",
        "layer_company",
        "model_fields_term",
        "sample_field",
        "atkey_matmul_apply",
    ):
        assert forbidden not in observer

    # Constructing the causal field term also cannot sample a field or invoke
    # a numerical leaf.
    construction = function_body(source, "static ModelTerm model_fields_term")
    assert "sample_field" not in construction
    assert re.search(r"\batkey_[a-z_]+_apply\s*\(", construction) is None

    main = function_body(source, "int main")
    build_position = main.index("build_fillers")
    model_position = main.index("model_fields_term")
    compose_position = main.index("compose_program")
    zero_position = main.index("calls_before_run")
    run_position = main.index("Outcome *result = run_pcont")
    assert build_position < model_position < compose_position < zero_position < run_position
    assert source.count("Outcome *result = run_pcont(&program)") == 1
    assert source.count("static Outcome *run_pcont(AtkeyProgram *program)") == 1
    run = function_body(source, "static Outcome *run_pcont")
    assert "history_product_select" in run
    assert "run_sampled_strength" not in run


def quarantined_hacks_are_inert() -> None:
    for source_path in QUARANTINED:
        source = source_path.read_text(encoding="utf-8")
        assert source.startswith("/*\n * DO NOT USE")
        assert "I should never" in source[:800]

    for removed_name in (
        "escardo_strength.c",
        "escardo_logit_strength.c",
        "sampled_game.c",
        "zip_score_probe.c",
    ):
        assert not (ROOT / removed_name).exists()

    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    for source_path in QUARANTINED:
        assert source_path.name not in makefile

    discovered_tests = {path.name for path in ROOT.glob("test_*.py")}
    assert "test_atkey_term.py" in discovered_tests
    assert discovered_tests <= {"test_all.py", "test_atkey_term.py"}


def main() -> None:
    missing = [path for path in (MODEL, TOKENIZER, TERM, STOCK) if not path.exists()]
    if missing:
        raise SystemExit("missing system-test artifacts: " + ", ".join(map(str, missing)))
    literal_backward_induction()
    stock_greedy_numerics()
    stock_delimiter_semantics()
    nonlocal_selection_changes_completion()
    terminal_length_bias_regression()
    boolean_satisfying_strength_fast_forwards()
    sampled_recursive_support_is_nested_as_budget_grows()
    sampled_support_exhaustion_matches_exact_product()
    wall_clock_budget_finishes_recursive_demand()
    sampled_strength_log_records_recursive_bellman_order()
    composed_c_term_structure()
    quarantined_hacks_are_inert()
    print("C Escardo/Atkey system tests: ALL OK")


if __name__ == "__main__":
    try:
        main()
    except (AssertionError, subprocess.CalledProcessError) as error:
        print(f"Literal Escardo/Atkey system test failed: {error}")
        raise

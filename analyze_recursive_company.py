#!/usr/bin/env python3
"""Validate recursive-company structure and parent-observation parity.

The prior two-level polynomial trace independently observed root verb codata
and every verb-indexed hole codata on the CPU path. This analyzer verifies
that the recursive Metal company preserves those parent observations while
actually consuming every hole constructor and retaining every resulting
terminal codata vector. It then validates the memoized dependent selection
product and compares it with the original edge-local AR choice on the
controlled number-agreement family.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from gather_polynomial_company import DEFAULT_OUTPUT as DEFAULT_POLYNOMIAL_DIRECTORY
from gather_recursive_company import DEFAULT_OUTPUT as DEFAULT_RECURSIVE_DIRECTORY


ROOT = Path(__file__).resolve().parent
DEFAULT_POLYNOMIAL_TRACE = DEFAULT_POLYNOMIAL_DIRECTORY / "polynomial-company.jsonl"
DEFAULT_RECURSIVE_TRACE = DEFAULT_RECURSIVE_DIRECTORY / "recursive-company.jsonl"
DEFAULT_RESULT = ROOT / "outputs" / "cps-stories15m-recursive-company.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--polynomial-trace", type=Path, default=DEFAULT_POLYNOMIAL_TRACE)
    parser.add_argument("--recursive-trace", type=Path, default=DEFAULT_RECURSIVE_TRACE)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--evaluator-commit")
    return parser.parse_args()


def read_polynomial(
    path: Path,
) -> tuple[
    dict[tuple[str, str], list[float]],
    dict[tuple[str, str, int], list[float]],
]:
    outer: dict[tuple[str, str], list[float]] = {}
    holes: dict[tuple[str, str, int], list[float]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("kind") == "polynomial_company_outer":
                key = (str(row["case"]), str(row["corner"]))
                require(key not in outer, f"duplicate polynomial outer {key}")
                outer[key] = [float(value) for value in row["contrasts"]]
            elif row.get("kind") == "polynomial_company_hole":
                key = (
                    str(row["case"]),
                    str(row["corner"]),
                    int(row["outer_token"]),
                )
                require(key not in holes, f"duplicate polynomial hole {key}")
                holes[key] = [float(value) for value in row["contrasts"]]
    require(len(outer) == 176, "polynomial outer coverage differs")
    require(len(holes) == 2112, "polynomial hole coverage differs")
    return outer, holes


def split_root(root: str) -> tuple[str, str]:
    fields = root.rsplit("::", 1)
    require(len(fields) == 2, f"recursive root has no corner suffix: {root}")
    return fields[0], fields[1]


def main() -> None:
    args = arguments()
    require(args.polynomial_trace.is_file(), f"missing {args.polynomial_trace}")
    require(args.recursive_trace.is_file(), f"missing {args.recursive_trace}")
    outer, holes = read_polynomial(args.polynomial_trace)

    meta: dict[str, Any] | None = None
    check: dict[str, Any] | None = None
    depth_counts: Counter[int] = Counter()
    root_depth_counts: dict[str, Counter[int]] = defaultdict(Counter)
    root_leaf_counts: Counter[str] = Counter()
    terminal_paths: set[tuple[str, tuple[int, ...]]] = set()
    composed_paths: set[tuple[str, tuple[int, ...]]] = set()
    demand_codata: dict[tuple[str, tuple[int, ...]], dict[int, float]] = {}
    terminal_codata: dict[
        tuple[str, tuple[int, ...]], tuple[tuple[int, float], ...]
    ] = {}
    terminal_rows: dict[tuple[str, tuple[int, ...]], int] = {}
    selections: dict[tuple[str, tuple[int, ...]], dict[str, Any]] = {}
    selected_completions: dict[str, dict[str, Any]] = {}
    terminal_family: tuple[int, ...] | None = None
    sample_terminals: list[dict[str, Any]] = []
    nonfinite_coordinates = 0
    maximum_absolute_parent_defect = 0.0
    maximum_relative_parent_defect = 0.0
    compared_parent_coordinates = 0

    with args.recursive_trace.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            kind = row.get("kind")
            if kind == "recursive_company_meta":
                require(meta is None, "duplicate recursive meta")
                meta = row
            elif kind == "recursive_company_check":
                require(check is None, "duplicate recursive check")
                check = row
            elif kind == "recursive_company_demand":
                root = str(row["root"])
                case, corner = split_root(root)
                depth = int(row["depth"])
                path = tuple(int(token) for token in row["path_tokens"])
                require(len(path) == depth, "demand path depth differs")
                candidates = row["candidates"]
                tokens = tuple(int(candidate["token"]) for candidate in candidates)
                values = [float(candidate["contrast"]) for candidate in candidates]
                demand_key = (root, path)
                require(demand_key not in demand_codata, f"duplicate demand {demand_key}")
                demand_codata[demand_key] = dict(zip(tokens, values))
                nonfinite_coordinates += sum(not math.isfinite(value) for value in values)
                if depth == 0:
                    expected = outer[(case, corner)]
                elif depth == 1:
                    require(len(path) == 1, "depth-one demand lost outer shape")
                    expected = holes[(case, corner, path[0])]
                else:
                    raise SystemExit("current retained run has unexpected depth")
                require(len(values) == len(expected), "parent codata width differs")
                for actual, reference in zip(values, expected):
                    defect = abs(actual - reference)
                    maximum_absolute_parent_defect = max(
                        maximum_absolute_parent_defect,
                        defect,
                    )
                    maximum_relative_parent_defect = max(
                        maximum_relative_parent_defect,
                        defect / max(1.0, abs(reference)),
                    )
                    compared_parent_coordinates += 1
                depth_counts[depth] += 1
                root_depth_counts[root][depth] += 1
                if depth == 0:
                    require(len(tokens) == 12, "root constructor width differs")
                else:
                    require(len(tokens) == 8, "hole constructor width differs")
            elif kind == "recursive_company_terminal":
                root = str(row["root"])
                path = tuple(int(token) for token in row["path_tokens"])
                require(len(path) == 2, "terminal path depth differs")
                key = (root, path)
                require(key not in terminal_paths, f"duplicate terminal path {key}")
                terminal_paths.add(key)
                root_leaf_counts[root] += 1
                candidates = row["terminal_candidates"]
                tokens = tuple(int(candidate["token"]) for candidate in candidates)
                if terminal_family is None:
                    terminal_family = tokens
                require(tokens == terminal_family, "terminal family changes by branch")
                values = [float(candidate["contrast"]) for candidate in candidates]
                terminal_codata[key] = tuple(zip(tokens, values))
                terminal_rows[key] = int(row["row"])
                nonfinite_coordinates += sum(not math.isfinite(value) for value in values)
                if len(sample_terminals) < 4:
                    ranked = sorted(
                        candidates,
                        key=lambda candidate: (
                            -float(candidate["contrast"]),
                            int(candidate["token"]),
                        ),
                    )
                    sample_terminals.append(
                        {
                            "root": root,
                            "path_tokens": list(path),
                            "text": row["text"],
                            "top_terminal_candidates": ranked[:5],
                        }
                    )
            elif kind == "recursive_company_composed_observation":
                root = str(row["root"])
                path = tuple(int(token) for token in row["path_tokens"])
                require(len(path) == 2, "composed path depth differs")
                key = (root, path)
                require(key not in composed_paths, f"duplicate composed path {key}")
                composed_paths.add(key)
                edges = row["edge_observations"]
                require(len(edges) == len(path), "composed edge count differs")
                for depth, edge in enumerate(edges):
                    require(int(edge["depth"]) == depth, "composed edge depth differs")
                    require(int(edge["token"]) == path[depth], "composed edge token differs")
                    demand_key = (root, path[:depth])
                    require(demand_key in demand_codata, "composed edge has no demand")
                    token = path[depth]
                    require(
                        token in demand_codata[demand_key],
                        "composed edge token absent from demand codata",
                    )
                    require(
                        float(edge["contrast"]) == demand_codata[demand_key][token],
                        "composed edge observation differs from demand codata",
                    )
                require(key in terminal_codata, "composed path has no terminal codata")
                composed_terminal = tuple(
                    (int(candidate["token"]), float(candidate["contrast"]))
                    for candidate in row["terminal_candidates"]
                )
                require(
                    composed_terminal == terminal_codata[key],
                    "composed terminal observation differs from terminal codata",
                )
            elif kind == "recursive_company_selection":
                root = str(row["root"])
                path = tuple(int(token) for token in row["path_tokens"])
                key = (root, path)
                require(key not in selections, f"duplicate selection {key}")
                selections[key] = row
            elif kind == "recursive_company_selected_completion":
                root = str(row["root"])
                require(
                    root not in selected_completions,
                    f"duplicate selected completion {root}",
                )
                selected_completions[root] = row

    require(meta is not None, "recursive trace has no meta")
    require(check is not None, "recursive trace has no check")
    require(meta.get("schema_version") == 1, "recursive schema differs")
    require(
        meta.get("semantics") == "dependent_polynomial_company_tree",
        "recursive semantics differ",
    )
    require(meta.get("probabilities_used") is False, "recursive run used probabilities")
    require(meta.get("scalar_reward_used") is False, "recursive run used scalar reward")
    require(
        meta.get("whole_completion_argmax_used") is False,
        "recursive run terminalized complete paths",
    )
    require(meta.get("complete_paths_flattened") is False, "paths were flattened")
    require(meta.get("completion_selected") is True, "selection product was not run")
    require(
        meta.get("selection_semantics")
        == "escardo_dependent_product_full_company_diagonal",
        "selection semantics differ",
    )
    require(
        meta.get("local_edge_logits_terminalized") is False,
        "selection terminalized local edge logits",
    )
    require(
        meta.get("path_likelihoods_summed") is False,
        "selection summed path likelihoods",
    )
    require(
        meta.get("observation_semantics") == "continuation_composed_codata",
        "recursive codata observations were not continuation-composed",
    )
    require(
        meta.get("codata_constructed_before_observation") is True,
        "recursive codata was forced before its observer",
    )
    require(meta.get("root_observer_runs") == 1, "root observer did not run once")
    require(meta.get("observations_composed") is True, "observations were not composed")
    require(meta.get("maximum_calls_per_filler") == 1, "recursive fillers repeated")
    require(check.get("maximum_calls_per_filler") == 1, "recursive check repeated fillers")
    require(depth_counts == Counter({0: 176, 1: 2112}), "recursive demand coverage differs")
    require(len(root_depth_counts) == 176, "recursive root coverage differs")
    require(len(root_leaf_counts) == 176, "terminal root coverage differs")
    for root, counts in root_depth_counts.items():
        require(counts == Counter({0: 1, 1: 12}), f"demand shape differs for {root}")
        require(root_leaf_counts[root] == 96, f"terminal shape differs for {root}")
    require(len(terminal_paths) == 16896, "terminal path coverage differs")
    require(composed_paths == terminal_paths, "composed path coverage differs")
    require(check.get("root_observer_runs") == 1, "root check did not run once")
    require(
        check.get("composed_observations") == len(terminal_paths),
        "composed observation count differs",
    )
    require(
        check.get("composition_steps") == 2 * len(terminal_paths),
        "continuation composition step count differs",
    )
    require(nonfinite_coordinates == 0, "recursive trace contains non-finite codata")
    require(terminal_family is not None, "recursive trace has no terminal family")

    require(set(selections) == set(demand_codata), "selection demand coverage differs")
    selection_candidate_count = 0
    selection_exact_tie_nodes = 0
    for key, row in selections.items():
        root, path = key
        depth = int(row["depth"])
        require(depth == len(path), f"selection depth differs for {key}")
        require(
            row.get("observer") == "full_company_diagonal",
            f"selection observer differs for {key}",
        )
        candidates = row["candidates"]
        selection_candidate_count += len(candidates)
        tokens = tuple(int(candidate["token"]) for candidate in candidates)
        require(
            set(tokens) == set(demand_codata[key]),
            f"selection family differs for {key}",
        )
        values = [float(candidate["diagonal_contrast"]) for candidate in candidates]
        require(
            all(math.isfinite(value) for value in values),
            f"selection contains nonfinite diagonal codata for {key}",
        )
        best_index = min(
            range(len(candidates)),
            key=lambda index: (-values[index], tokens[index]),
        )
        selected_indices = [
            index for index, candidate in enumerate(candidates)
            if bool(candidate["selected"])
        ]
        require(selected_indices == [best_index], f"selection argmax differs for {key}")
        selected = candidates[best_index]
        require(
            int(row["selected_token"]) == int(selected["token"]),
            f"selected token differs for {key}",
        )
        require(
            int(row["selected_terminal_row"]) == int(selected["terminal_row"]),
            f"selected terminal row differs for {key}",
        )
        require(
            float(row["selected_diagonal_contrast"])
            == float(selected["diagonal_contrast"]),
            f"selected diagonal differs for {key}",
        )
        exact_ties = sum(value == values[best_index] for value in values)
        require(
            int(row["exact_argmax_size"]) == exact_ties,
            f"exact argmax set differs for {key}",
        )
        selection_exact_tie_nodes += exact_ties > 1
        for candidate in candidates:
            continuation = tuple(
                int(token) for token in candidate["continuation_tokens"]
            )
            require(
                len(continuation) == int(meta["depth"]) - depth,
                f"candidate continuation length differs for {key}",
            )
            require(
                continuation[0] == int(candidate["token"]),
                f"candidate continuation lost its head for {key}",
            )
            complete_key = (root, path + continuation)
            require(
                complete_key in terminal_rows,
                f"candidate continuation has no terminal outcome for {key}",
            )
            require(
                int(candidate["terminal_row"]) == terminal_rows[complete_key],
                f"candidate continuation terminal row differs for {key}",
            )
            if depth + 1 < int(meta["depth"]):
                child_key = (root, path + (int(candidate["token"]),))
                require(child_key in selections, f"selection child is absent for {key}")
                require(
                    continuation[1:]
                    == tuple(
                        int(token)
                        for token in selections[child_key]["candidates"][
                            min(
                                range(len(selections[child_key]["candidates"])),
                                key=lambda index: (
                                    -float(
                                        selections[child_key]["candidates"][index][
                                            "diagonal_contrast"
                                        ]
                                    ),
                                    int(
                                        selections[child_key]["candidates"][index][
                                            "token"
                                        ]
                                    ),
                                ),
                            )
                        ]["continuation_tokens"]
                    ),
                    f"selection child witness differs for {key}",
                )

    require(
        len(selected_completions) == int(meta["roots"]),
        "selected root coverage differs",
    )
    for root, row in selected_completions.items():
        path = tuple(int(token) for token in row["path_tokens"])
        require(len(path) == int(meta["depth"]), f"selected path depth differs for {root}")
        root_selection = selections[(root, ())]
        root_candidate = next(
            candidate
            for candidate in root_selection["candidates"]
            if bool(candidate["selected"])
        )
        require(
            tuple(int(token) for token in root_candidate["continuation_tokens"]) == path,
            f"root selection witness differs for {root}",
        )
        require(
            int(row["terminal_row"]) == terminal_rows[(root, path)],
            f"selected completion terminal row differs for {root}",
        )
        ballots = row["position_ballots"]
        require(len(ballots) == len(path), f"selected ballots differ for {root}")
        for depth, ballot in enumerate(ballots):
            node = selections[(root, path[:depth])]
            require(int(ballot["depth"]) == depth, f"ballot depth differs for {root}")
            require(int(ballot["token"]) == path[depth], f"ballot token differs for {root}")
            require(
                float(ballot["diagonal_contrast"])
                == float(node["selected_diagonal_contrast"]),
                f"ballot diagonal differs for {root}",
            )
            require(
                int(node["selected_terminal_row"]) == int(row["terminal_row"]),
                f"ballot outcome row differs for {root}",
            )

    singular_verbs = {12080, 6057, 1736, 13582, 16229, 6911}
    plural_verbs = {5735, 1065, 664, 1708, 4337, 1371}
    first_root = next(iter(selected_completions))
    require(
        singular_verbs | plural_verbs == set(demand_codata[(first_root, ())]),
        "root verb family differs from the agreement audit",
    )
    agreement = {
        "edge_ar_correct": 0,
        "full_company_diagonal_correct": 0,
        "different_outer_choice": 0,
    }
    agreement_by_corner: dict[str, Counter[str]] = defaultdict(Counter)
    selection_samples: list[dict[str, Any]] = []
    for root in sorted(selected_completions):
        _, corner = split_root(root)
        root_codata = demand_codata[(root, ())]
        edge_ar = min(root_codata, key=lambda token: (-root_codata[token], token))
        selected = int(selections[(root, ())]["selected_token"])
        expected = plural_verbs if "C" in corner else singular_verbs
        edge_correct = edge_ar in expected
        selected_correct = selected in expected
        agreement["edge_ar_correct"] += edge_correct
        agreement["full_company_diagonal_correct"] += selected_correct
        agreement["different_outer_choice"] += edge_ar != selected
        agreement_by_corner[corner].update(
            {
                "contexts": 1,
                "edge_ar_correct": edge_correct,
                "full_company_diagonal_correct": selected_correct,
                "different_outer_choice": edge_ar != selected,
            }
        )
        if len(selection_samples) < 12:
            selection_samples.append(
                {
                    "root": root,
                    "edge_ar_token": edge_ar,
                    "selected_token": selected,
                    "selected_text": selected_completions[root]["text"],
                    "outer_candidates": selections[(root, ())]["candidates"],
                }
            )

    require(
        check.get("selection_nodes") == len(selections),
        "selection-node count differs",
    )
    require(
        check.get("selection_candidate_evaluations") == selection_candidate_count,
        "selection-candidate count differs",
    )
    require(
        check.get("selection_root_mates") == len(selected_completions),
        "selection root-mate count differs",
    )
    require(
        check.get("selection_exact_tie_nodes") == selection_exact_tie_nodes,
        "selection exact-tie count differs",
    )

    result = {
        "schema_version": 1,
        "artifact": "recursive_dependent_polynomial_company",
        "model": "Stories15M",
        "evaluator_commit": args.evaluator_commit or git_head(),
        "input_sha256": {
            "polynomial_trace": sha256(args.polynomial_trace),
            "recursive_trace": sha256(args.recursive_trace),
        },
        "term": {
            "roots": 176,
            "family_widths": [12, 8],
            "demand_nodes_by_depth": {"0": 176, "1": 2112},
            "complete_branches": 16896,
            "company_rows": int(meta["company_rows"]),
            "terminal_width": len(terminal_family),
        },
        "one_shot_company": {
            "learned_fillers": int(meta["learned_fillers"]),
            "family_filler_calls": int(meta["family_filler_calls"]),
            "maximum_calls_per_filler": int(meta["maximum_calls_per_filler"]),
            "family_scalar_reads": int(meta["family_scalar_reads"]),
            "backend": meta["backend"],
        },
        "parent_observation_parity": {
            "reference": "independent earlier CPU polynomial company",
            "compared_coordinates": compared_parent_coordinates,
            "maximum_absolute_defect": maximum_absolute_parent_defect,
            "maximum_relative_defect": maximum_relative_parent_defect,
        },
        "integrity": {
            "unique_complete_branches": len(terminal_paths),
            "nonfinite_coordinates": nonfinite_coordinates,
            "probabilities_used": False,
            "scalar_reward_used": False,
            "complete_paths_flattened": False,
            "path_likelihoods_summed": False,
        },
        "codata_composition": {
            "codata_constructed_before_observation": True,
            "root_observer_runs": int(check["root_observer_runs"]),
            "composed_observations": int(check["composed_observations"]),
            "continuation_bind_steps": int(check["composition_steps"]),
            "edge_observation_defects": 0,
            "terminal_observation_defects": 0,
        },
        "selection_product": {
            "semantics": meta["selection_semantics"],
            "outcome": "complete terminal-frontier token codata row",
            "local_selection": (
                "for candidate x, recursively select its dependent suffix, "
                "then maximize the x coordinate of that complete outcome"
            ),
            "demand_nodes": len(selections),
            "candidate_continuations_evaluated": selection_candidate_count,
            "root_mates": len(selected_completions),
            "exact_tie_nodes": selection_exact_tie_nodes,
            "probabilities_used": False,
            "path_scores_added": False,
            "local_edge_logits_terminalized": False,
        },
        "agreement_number_control": {
            **agreement,
            "contexts": len(selected_completions),
            "by_corner": {
                corner: dict(counts)
                for corner, counts in sorted(agreement_by_corner.items())
            },
            "scope": (
                "number agreement of the selected outer verb family only; "
                "not a whole-completion coherence score"
            ),
        },
        "sample_selected_completions": selection_samples,
        "sample_terminal_observations": sample_terminals,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        "recursive_company_analysis "
        f"roots=176 demands={sum(depth_counts.values())} "
        f"leaves={len(terminal_paths)} nonfinite={nonfinite_coordinates} "
        f"composed={len(composed_paths)} "
        f"selected={len(selected_completions)} "
        f"edge_ar_agreement={agreement['edge_ar_correct']}/{len(selected_completions)} "
        "full_company_agreement="
        f"{agreement['full_company_diagonal_correct']}/{len(selected_completions)} "
        f"max_parent_relative_defect={maximum_relative_parent_defect:.9g}"
    )


if __name__ == "__main__":
    main()

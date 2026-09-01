#!/usr/bin/env python3
"""Recover every shape-indexed post-verb hole demand interface.

Each trace varies attractor number A, controller number C, and a pair of fixed
verb injections.  At the edge immediately after each verb, this analyzer
retains a fixed next-constructor codata family and independently recovers its
contrast-codata, complete-ordering, and selected-injection supports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from analyze_grammar_cubes import read_observers
from analyze_number_demand_cubes import (
    CORNER_NAMES,
    DEFAULT_CONSTRUCTOR_FAMILY,
    DEFAULT_HOLE_CONSTRUCTOR_FAMILY,
    DEFAULT_OBSERVERS,
    EDGE_SUBSETS,
    multiway_observation,
    read_case_trace,
    reconstruct_square,
    require,
)
from gather_grammar_relations import DEFAULT_MANIFEST, read_manifest
from gather_number_demand_cubes import (
    DEFAULT_OUTPUT as DEFAULT_NUMBER_TRACES,
    expand_cases as expand_number_cases,
)
from gather_post_injection_holes import (
    DEFAULT_OUTPUT as DEFAULT_TRACES,
    HoleCase,
    expand_hole_cases,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULT = (
    ROOT / "outputs" / "cps-stories15m-post-injection-hole-polynomial.json"
)


def git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument(
        "--number-traces",
        type=Path,
        default=DEFAULT_NUMBER_TRACES,
    )
    parser.add_argument("--observers", type=Path, default=DEFAULT_OBSERVERS)
    parser.add_argument(
        "--constructor-family",
        type=Path,
        default=DEFAULT_CONSTRUCTOR_FAMILY,
    )
    parser.add_argument(
        "--hole-constructor-family",
        type=Path,
        default=DEFAULT_HOLE_CONSTRUCTOR_FAMILY,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--model-label", default="Stories15M")
    parser.add_argument("--evaluator-commit")
    return parser.parse_args()


def read_hole_trace(
    case: HoleCase,
    directory: Path,
    observer_ids: tuple[int, ...],
    observer_labels: dict[int, str],
    hole_constructor_ids: tuple[int, ...],
    hole_constructor_labels: dict[int, str],
    hole_constructor_indices: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = directory / case.trace_name
    require(path.is_file(), f"missing trace: {path}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    kinds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        kinds[str(row.get("kind"))].append(row)

    def one(kind: str) -> dict[str, Any]:
        values = kinds[kind]
        require(len(values) == 1, f"{path}: expected one {kind}")
        return values[0]

    meta = one("grammatical_cube_meta")
    check = one("grammatical_cube_check")
    require(meta.get("schema_version") == 2, f"{path}: wrong schema")
    require(meta.get("c_action_kind") == "aligned_constructor", f"{path}: C is not aligned")
    require(meta.get("edge_c_difference_typed") is True, f"{path}: C edge is untyped")
    require(meta.get("extension_token_count") == 0, f"{path}: C unexpectedly extends")
    require(meta.get("base_positions") == meta.get("extended_positions"), f"{path}: widths differ")
    require(tuple(meta["observer_tokens"]) == observer_ids, f"{path}: observer coordinates differ")
    require(meta.get("edge_observations_folded") is False, f"{path}: edge observations folded")
    require(meta.get("scalar_completion_reward_used") is False, f"{path}: scalar reward used")
    require(float(check["maximum_typed_chain_output_l2_defect"]) == 0.0, f"{path}: chain defect")

    contexts = {str(row["corner"]): row for row in kinds["grammatical_cube_context"]}
    require(set(contexts) == {"x", "ax", "bx", "abx", "cx", "acx", "bcx", "abcx"}, f"{path}: corners differ")
    x_tokens = [int(item["id"]) for item in contexts["x"]["tokens"]]
    cx_tokens = [int(item["id"]) for item in contexts["cx"]["tokens"]]
    require(len(x_tokens) == len(cx_tokens), f"{path}: injection fibers have different widths")
    differences = [
        position
        for position, (left, right) in enumerate(zip(x_tokens, cx_tokens))
        if left != right
    ]
    require(len(differences) == 1, f"{path}: fixed verb injection is not one aligned constructor")
    verb_position = differences[0]
    require(x_tokens[verb_position] == case.first_token, f"{path}: first verb token differs")
    require(cx_tokens[verb_position] == case.second_token, f"{path}: second verb token differs")
    post_position = verb_position + 1
    require(post_position < len(x_tokens), f"{path}: no typed edge after verb")

    edge_rows: dict[tuple[str, int], dict[str, Any]] = {}
    for row in kinds["grammatical_cube_edge_zip"]:
        key = (str(row["fiber"]), int(row["token_position"]))
        require(key not in edge_rows, f"{path}: duplicate edge {key}")
        edge_rows[key] = row

    entries: list[dict[str, Any]] = []
    for fiber, injection_token, injection_text in (
        ("without_C", case.first_token, case.first_verb),
        ("with_C", case.second_token, case.second_verb),
    ):
        row = edge_rows.get((fiber, post_position))
        require(row is not None, f"{path}: missing post-injection edge in {fiber}")
        require(tuple(row["mobius_subsets"]) == EDGE_SUBSETS, f"{path}: subset order differs")
        next_tokens = tuple(int(token) for token in row["corner_token_ids"])
        require(len(set(next_tokens)) == 1, f"{path}: next constructor differs within number cube")
        coefficients = np.stack(
            [np.asarray(row["coefficients"][name], dtype=np.float64) for name in EDGE_SUBSETS]
        )
        require(coefficients.shape == (4, len(observer_ids)), f"{path}: edge codata shape differs")
        require(np.all(np.isfinite(coefficients)), f"{path}: non-finite edge codata")
        corners = reconstruct_square(coefficients)[:, hole_constructor_indices]
        observation = multiway_observation(corners)
        observation["argmax_constructors"] = {
            corner: [
                {
                    "token": hole_constructor_ids[index],
                    "text": hole_constructor_labels[hole_constructor_ids[index]],
                }
                for index in observation["argmax_indices"][corner]
            ]
            for corner in CORNER_NAMES
        }
        entries.append(
            {
                "case": case.key,
                "phase": case.phase,
                "template": case.template,
                "family": case.family,
                "trace": case.trace_name,
                "verb_token_position": verb_position,
                "hole_token_position": post_position,
                "next_token": {
                    "token": next_tokens[0],
                    "text": observer_labels[next_tokens[0]],
                },
                "injection": {
                    "token": injection_token,
                    "text": injection_text,
                },
                "observation": observation,
                "_corners": corners,
            }
        )
    return entries, check


def support_name(features: list[str]) -> str:
    return "{" + ",".join(features) + "}"


def support_counts(
    entries: list[dict[str, Any]],
    observer: str,
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for entry in entries:
        support = entry["observation"]["supports"][observer]
        counts[support_name(support)] += 1
    return dict(sorted(counts.items()))


def main() -> None:
    args = arguments()
    observer_ids, observer_labels = read_observers(args.observers)
    observer_index = {token: index for index, token in enumerate(observer_ids)}
    constructor_ids, constructor_labels = read_observers(args.constructor_family)
    hole_constructor_ids, hole_constructor_labels = read_observers(
        args.hole_constructor_family
    )
    require(set(hole_constructor_ids) <= set(observer_ids), "hole family is not retained")
    hole_constructor_indices = [
        observer_index[token] for token in hole_constructor_ids
    ]
    manifest = read_manifest(args.manifest)
    cases = expand_hole_cases(manifest, constructor_ids, constructor_labels)

    entries: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for case in cases:
        case_entries, check = read_hole_trace(
            case,
            args.traces,
            observer_ids,
            observer_labels,
            hole_constructor_ids,
            hole_constructor_labels,
            hole_constructor_indices,
        )
        entries.extend(case_entries)
        checks.append(check)

    context_keys = sorted({str(entry["case"]) for entry in entries})
    require(len(context_keys) == 44, "full hole family does not contain 44 contexts")
    by_context: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        by_context[str(entry["case"])].append(entry)
    for key, members in by_context.items():
        tokens = [int(member["injection"]["token"]) for member in members]
        require(
            len(tokens) == len(constructor_ids) and set(tokens) == set(constructor_ids),
            f"{key}: did not consume every fixed injection exactly once",
        )

    entries_by_key_token = {
        (str(entry["case"]), int(entry["injection"]["token"])): entry
        for entry in entries
    }
    require(len(entries_by_key_token) == len(entries), "duplicate shape-indexed hole entry")

    maximum_reordered_cube_defect = 0.0
    maximum_reordered_cube_location: dict[str, Any] | None = None
    duplicate_checks = 0
    duplicate_support_mismatches = {
        "contrast_codata": 0,
        "constructor_ordering": 0,
        "selected_injection": 0,
    }
    duplicate_ordering_mismatches = 0
    duplicate_argmax_mismatches = 0
    for number_case in expand_number_cases(manifest):
        original = read_case_trace(number_case, args.number_traces, observer_ids)
        for injection_token, lower, upper in (
            (int(original["singular"]), 0, 1),
            (int(original["plural"]), 2, 3),
        ):
            expected = np.stack(
                (
                    original["post_base_raw"][lower],
                    original["post_base_raw"][upper],
                    original["post_c_raw"][lower],
                    original["post_c_raw"][upper],
                )
            )[:, hole_constructor_indices]
            entry = entries_by_key_token[(number_case.key, injection_token)]
            actual = entry["_corners"]
            absolute_defect = np.abs(actual - expected)
            flat_index = int(np.argmax(absolute_defect))
            corner_index, observer_index_in_family = np.unravel_index(
                flat_index,
                absolute_defect.shape,
            )
            case_defect = float(absolute_defect[corner_index, observer_index_in_family])
            if case_defect > maximum_reordered_cube_defect:
                maximum_reordered_cube_defect = case_defect
                maximum_reordered_cube_location = {
                    "case": number_case.key,
                    "injection_token": injection_token,
                    "injection_text": constructor_labels[injection_token],
                    "corner": CORNER_NAMES[corner_index],
                    "observer_token": hole_constructor_ids[observer_index_in_family],
                    "observer_text": hole_constructor_labels[
                        hole_constructor_ids[observer_index_in_family]
                    ],
                    "expected": float(expected[corner_index, observer_index_in_family]),
                    "actual": float(actual[corner_index, observer_index_in_family]),
                }
            expected_observation = multiway_observation(expected)
            actual_observation = entry["observation"]
            for observer in duplicate_support_mismatches:
                if (
                    expected_observation["supports"][observer]
                    != actual_observation["supports"][observer]
                ):
                    duplicate_support_mismatches[observer] += 1
            if (
                expected_observation["weak_pairwise_order"]
                != actual_observation["weak_pairwise_order"]
            ):
                duplicate_ordering_mismatches += 1
            if (
                expected_observation["argmax_indices"]
                != actual_observation["argmax_indices"]
            ):
                duplicate_argmax_mismatches += 1
            duplicate_checks += 1

    require(
        all(count == 0 for count in duplicate_support_mismatches.values()),
        "reordered duplicate demanded supports differ",
    )
    require(
        duplicate_ordering_mismatches == 0,
        "reordered duplicate constructor orderings differ",
    )
    require(
        duplicate_argmax_mismatches == 0,
        "reordered duplicate selected injections differ",
    )

    observers = (
        "contrast_codata",
        "constructor_ordering",
        "selected_injection",
    )
    entries_by_token: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        entries_by_token[int(entry["injection"]["token"])].append(entry)
    summands: list[dict[str, Any]] = []
    for token in constructor_ids:
        members = entries_by_token[token]
        require(len(members) == 44, f"injection {token}: expected 44 hole observations")
        global_exponents = {
            observer: [
                feature
                for feature in ("A", "C")
                if any(
                    feature in member["observation"]["supports"][observer]
                    for member in members
                )
            ]
            for observer in observers
        }
        summands.append(
            {
                "injection_token": token,
                "injection_text": constructor_labels[token],
                "observations": len(members),
                "global_sampled_exponents": global_exponents,
                "support_counts": {
                    observer: support_counts(members, observer)
                    for observer in observers
                },
            }
        )

    serializable_entries = []
    for entry in entries:
        serializable = dict(entry)
        del serializable["_corners"]
        serializable_entries.append(serializable)

    result = {
        "schema_version": 1,
        "artifact": "full_shape_indexed_post_injection_hole_polynomial",
        "semantics": {
            "container": "G_star(X)=coproduct_d product_{p in P_star(d)} X_sort(p)",
            "shapes": "the fixed twelve-verb constructor coproduct",
            "positions": "observer-indexed A/C feature support at the edge after d is consumed",
            "observer_hierarchy": [
                "contrast_codata",
                "constructor_ordering",
                "selected_injection",
            ],
            "feature_actions": {
                "A": "pluralize attractor number",
                "C": "pluralize controller number",
            },
            "probabilities_used": False,
            "sequence_observations_folded": False,
            "completion_reward_used": False,
        },
        "provenance": {
            "model": args.model_label,
            "evaluator_commit": args.evaluator_commit or "unspecified",
            "analyzer_commit": git_head(),
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "observers_sha256": hashlib.sha256(args.observers.read_bytes()).hexdigest(),
            "constructor_family_sha256": hashlib.sha256(
                args.constructor_family.read_bytes()
            ).hexdigest(),
            "hole_constructor_family_sha256": hashlib.sha256(
                args.hole_constructor_family.read_bytes()
            ).hexdigest(),
            "trace_count": len(cases),
        },
        "validation": {
            "contexts": len(context_keys),
            "fixed_injections_per_context": len(constructor_ids),
            "shape_indexed_hole_observations": len(entries),
            "reordered_duplicate_checks": duplicate_checks,
            "maximum_reordered_cube_absolute_defect": maximum_reordered_cube_defect,
            "maximum_reordered_cube_location": maximum_reordered_cube_location,
            "reordered_duplicate_support_mismatches": duplicate_support_mismatches,
            "reordered_duplicate_ordering_mismatches": duplicate_ordering_mismatches,
            "reordered_duplicate_argmax_mismatches": duplicate_argmax_mismatches,
            "maximum_typed_chain_output_l2_defect": max(
                float(check["maximum_typed_chain_output_l2_defect"])
                for check in checks
            ),
            "maximum_edge_mobius_inverse_absolute_defect": max(
                float(check["maximum_edge_mobius_inverse_absolute_defect"])
                for check in checks
            ),
            "maximum_stock_logit_contrast_relative_defect": max(
                float(check["maximum_stock_logit_contrast_relative_defect"])
                for check in checks
            ),
        },
        "hole_constructor_family": [
            {"token": token, "text": hole_constructor_labels[token]}
            for token in hole_constructor_ids
        ],
        "observer_indexed_support": {
            observer: support_counts(entries, observer)
            for observer in observers
        },
        "polynomial_summands": summands,
        "entries": serializable_entries,
        "scope": {
            "establishes": "P_codata(d), P_ordering(d), and P_choice(d) for every one of 12 fixed verb injections in every one of 44 contexts",
            "does_not_yet_establish": "direct equality of a composed two-constructor ordering or argmax with polynomial substitution",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"post-injection holes: contexts={len(context_keys)} "
        f"injections={len(constructor_ids)} observations={len(entries)} "
        f"duplicate_defect={maximum_reordered_cube_defect:.9g}"
    )
    for observer, counts in result["observer_indexed_support"].items():
        print(f"  {observer:22s} {counts}")
    for summand in summands:
        exponents = summand["global_sampled_exponents"]
        print(
            f"  d={summand['injection_token']:5d} "
            f"text={summand['injection_text']!r} "
            f"codata={support_name(exponents['contrast_codata'])} "
            f"order={support_name(exponents['constructor_ordering'])} "
            f"choice={support_name(exponents['selected_injection'])}"
        )


if __name__ == "__main__":
    main()

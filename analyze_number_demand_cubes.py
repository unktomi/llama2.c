#!/usr/bin/env python3
"""Analyze the complete two-feature number demand at the pre-verb edge.

For each aligned cube, A changes attractor number, C changes controller number,
and the singular/plural B alternatives are retained as constructor-coordinate
observations before B is consumed. The output retains complete codata
coefficients and the gauge-invariant constructor contrast L=q_plural-q_singular.
No threshold, probability, sequence fold, or completion reward is introduced.
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
from gather_grammar_relations import DEFAULT_MANIFEST, read_manifest
from gather_number_demand_cubes import (
    DEFAULT_OUTPUT as DEFAULT_TRACES,
    NumberDemandCase,
    expand_cases,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OBSERVERS = ROOT / "grammar_observer_tokens.tsv"
DEFAULT_CONSTRUCTOR_FAMILY = ROOT / "verb_constructor_family.tsv"
DEFAULT_RESULT = ROOT / "outputs" / "cps-stories15m-number-demand-analysis.json"
EDGE_SUBSETS = ("carrier", "A", "B", "AB")
CORNER_NAMES = ("x", "A", "C", "AC")


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


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--observers", type=Path, default=DEFAULT_OBSERVERS)
    parser.add_argument(
        "--constructor-family",
        type=Path,
        default=DEFAULT_CONSTRUCTOR_FAMILY,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--model-label", default="Stories15M")
    parser.add_argument("--evaluator-commit")
    return parser.parse_args()


def weak_pairwise_order(values: np.ndarray) -> tuple[int, ...]:
    result: list[int] = []
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            difference = float(values[left] - values[right])
            result.append(1 if difference > 0.0 else -1 if difference < 0.0 else 0)
    return tuple(result)


def argmax_set(values: np.ndarray) -> tuple[int, ...]:
    maximum = float(np.max(values))
    return tuple(index for index, value in enumerate(values) if float(value) == maximum)


def essential_feature_support(signatures: dict[str, Any]) -> list[str]:
    fibers = {
        "A": (("x", "A"), ("C", "AC")),
        "C": (("x", "C"), ("A", "AC")),
    }
    return [
        feature
        for feature, pairs in fibers.items()
        if any(signatures[left] != signatures[right] for left, right in pairs)
    ]


def multiway_observation(corners: np.ndarray) -> dict[str, Any]:
    require(corners.ndim == 2 and corners.shape[0] == 4, "multiway corner shape differs")
    gauge_free = corners[:, 1:] - corners[:, :1]
    codata_signatures = {
        corner: tuple(float(value) for value in gauge_free[index])
        for index, corner in enumerate(CORNER_NAMES)
    }
    order_signatures = {
        corner: weak_pairwise_order(corners[index])
        for index, corner in enumerate(CORNER_NAMES)
    }
    choice_signatures = {
        corner: argmax_set(corners[index])
        for index, corner in enumerate(CORNER_NAMES)
    }
    supports = {
        "contrast_codata": essential_feature_support(codata_signatures),
        "constructor_ordering": essential_feature_support(order_signatures),
        "selected_injection": essential_feature_support(choice_signatures),
    }
    require(
        set(supports["selected_injection"])
        <= set(supports["constructor_ordering"])
        <= set(supports["contrast_codata"]),
        "observer-indexed support inclusion failed",
    )
    return {
        "gauge_free_contrasts": {
            corner: list(codata_signatures[corner]) for corner in CORNER_NAMES
        },
        "weak_pairwise_order": {
            corner: list(order_signatures[corner]) for corner in CORNER_NAMES
        },
        "argmax_indices": {
            corner: list(choice_signatures[corner]) for corner in CORNER_NAMES
        },
        "supports": supports,
    }


def strict_inclusion_oracle() -> dict[str, Any]:
    corners = np.asarray(
        (
            (3.0, 0.0, -1.0),
            (3.0, 0.0, 1.0),
            (0.0, 3.0, -1.0),
            (0.0, 3.0, 1.0),
        ),
        dtype=np.float64,
    )
    observation = multiway_observation(corners)
    expected = {
        "contrast_codata": ["A", "C"],
        "constructor_ordering": ["A", "C"],
        "selected_injection": ["C"],
    }
    require(observation["supports"] == expected, "strict-inclusion oracle differs")
    return {
        "constructors": ["s", "p", "n"],
        "law": {
            "q_s": "3(1-C)",
            "q_p": "3C",
            "q_n": "-1+2A",
        },
        "corners": dict(zip(CORNER_NAMES, corners.tolist())),
        "expected_supports": expected,
        "observation": observation,
    }


def reconstruct_square(coefficients: np.ndarray) -> np.ndarray:
    carrier, a, b, ab = coefficients
    return np.stack(
        (
            carrier,
            carrier + a,
            carrier + b,
            carrier + a + b + ab,
        )
    )


def read_case_trace(
    case: NumberDemandCase,
    directory: Path,
    observer_ids: tuple[int, ...],
) -> dict[str, Any]:
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
    require(
        meta.get("semantics") == "carrier_conditioned_action_jet_with_mealy_edge_zip",
        f"{path}: wrong semantics",
    )
    require(meta.get("c_action_kind") == "aligned_constructor", f"{path}: C is not aligned")
    require(meta.get("edge_c_difference_typed") is True, f"{path}: C edge difference is untyped")
    require(meta.get("extension_token_count") == 0, f"{path}: C unexpectedly adds a suffix")
    require(meta.get("base_positions") == meta.get("extended_positions"), f"{path}: fiber widths differ")
    require(tuple(meta["observer_tokens"]) == observer_ids, f"{path}: observer coordinates differ")
    require(meta.get("edge_observations_folded") is False, f"{path}: edge observations were folded")
    require(meta.get("scalar_completion_reward_used") is False, f"{path}: scalar reward used")
    require(float(check["maximum_typed_chain_output_l2_defect"]) == 0.0, f"{path}: chain defect")

    edge_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in kinds["grammatical_cube_edge_zip"]:
        edge_rows[str(row["fiber"])].append(row)
    require(set(edge_rows) == {"without_C", "with_C"}, f"{path}: edge fibers differ")

    selected: dict[str, dict[str, Any]] = {}
    maximum_b_prefix_leak = 0.0
    for fiber, candidates in edge_rows.items():
        verb = [
            row
            for row in candidates
            if row["corner_token_ids"][0] == row["corner_token_ids"][1]
            and row["corner_token_ids"][2] == row["corner_token_ids"][3]
            and row["corner_token_ids"][0] != row["corner_token_ids"][2]
        ]
        require(len(verb) == 1, f"{path}: expected one B constructor edge in {fiber}")
        row = verb[0]
        require(tuple(row["mobius_subsets"]) == EDGE_SUBSETS, f"{path}: edge subset order differs")
        coefficients = np.stack(
            [np.asarray(row["coefficients"][name], dtype=np.float64) for name in EDGE_SUBSETS]
        )
        require(coefficients.shape == (4, len(observer_ids)), f"{path}: edge shape differs")
        require(np.all(np.isfinite(coefficients)), f"{path}: non-finite edge codata")
        maximum_b_prefix_leak = max(
            maximum_b_prefix_leak,
            float(np.max(np.abs(coefficients[2:]))),
        )
        selected[fiber] = {
            "row": row,
            "coefficients": coefficients,
            "raw": reconstruct_square(coefficients),
        }

    base = selected["without_C"]
    with_c = selected["with_C"]
    require(
        base["row"]["token_position"] == with_c["row"]["token_position"],
        f"{path}: verb positions differ across C",
    )
    require(
        base["row"]["corner_token_ids"] == with_c["row"]["corner_token_ids"],
        f"{path}: verb injections differ across C",
    )
    singular = int(base["row"]["corner_token_ids"][0])
    plural = int(base["row"]["corner_token_ids"][2])
    return {
        "path": path,
        "check": check,
        "token_position": int(base["row"]["token_position"]),
        "singular": singular,
        "plural": plural,
        "base_raw": base["raw"],
        "c_raw": with_c["raw"],
        "maximum_b_prefix_leak": maximum_b_prefix_leak,
    }


def coefficient_summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    require(len(array) > 0 and np.all(np.isfinite(array)), "empty coefficient family")
    return {
        "count": len(values),
        "exact_zero": int(np.sum(array == 0.0)),
        "negative": int(np.sum(array < 0.0)),
        "positive": int(np.sum(array > 0.0)),
        "minimum": float(array.min()),
        "mean": float(array.mean()),
        "maximum": float(array.max()),
        "minimum_absolute": float(np.min(np.abs(array))),
        "maximum_absolute": float(np.max(np.abs(array))),
    }


def margin_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    margins = [float(row["expected_minus_alternative_margin"]) for row in rows]
    summary = coefficient_summary(margins)
    summary["matches_manifest_expectation"] = sum(
        bool(row["matches_manifest_expectation"]) for row in rows
    )
    summary["manifest_match_rate"] = float(
        np.mean([bool(row["matches_manifest_expectation"]) for row in rows])
    )
    return summary


def main() -> None:
    args = arguments()
    observer_ids, observer_labels = read_observers(args.observers)
    observer_index = {token: index for index, token in enumerate(observer_ids)}
    constructor_ids, constructor_labels = read_observers(args.constructor_family)
    require(len(constructor_ids) >= 3, "multiway constructor family must have at least three injections")
    require(
        set(constructor_ids) <= set(observer_ids),
        "multiway constructor family is not contained in the retained observer",
    )
    constructor_indices = [observer_index[token] for token in constructor_ids]
    cases = expand_cases(read_manifest(args.manifest))
    records: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    maximum_b_prefix_leak = 0.0
    maximum_c_mobius_inverse_defect = 0.0
    check_rows: list[dict[str, Any]] = []

    for case in cases:
        trace = read_case_trace(case, args.traces, observer_ids)
        singular = int(trace["singular"])
        plural = int(trace["plural"])
        require(singular in observer_index and plural in observer_index, f"{case.key}: verb observer absent")
        singular_index = observer_index[singular]
        plural_index = observer_index[plural]
        base_raw = trace["base_raw"]
        c_raw = trace["c_raw"]

        q_x = base_raw[0]
        q_a = base_raw[1]
        q_c = c_raw[0]
        q_ac = c_raw[1]
        codata = np.stack(
            (
                q_x,
                q_a - q_x,
                q_c - q_x,
                q_ac - q_c - q_a + q_x,
            )
        )
        reconstructed = np.stack(
            (
                codata[0],
                codata[0] + codata[1],
                codata[0] + codata[2],
                codata[0] + codata[1] + codata[2] + codata[3],
            )
        )
        maximum_c_mobius_inverse_defect = max(
            maximum_c_mobius_inverse_defect,
            float(np.max(np.abs(reconstructed - np.stack((q_x, q_a, q_c, q_ac))))),
        )

        def contrast(q: np.ndarray) -> float:
            return float(q[plural_index] - q[singular_index])

        l_x, l_a, l_c, l_ac = (contrast(q) for q in (q_x, q_a, q_c, q_ac))
        demand = {
            "carrier_L": l_x,
            "D_attractor_L": l_a - l_x,
            "D_controller_L": l_c - l_x,
            "D_controller_D_attractor_L": l_ac - l_c - l_a + l_x,
        }
        require(all(value != 0.0 for value in (l_x, l_a, l_c, l_ac)), f"{case.key}: tied constructor ordering")

        choice_changes = {
            "A": {
                "controller_singular": (l_x > 0.0) != (l_a > 0.0),
                "controller_plural": (l_c > 0.0) != (l_ac > 0.0),
            },
            "C": {
                "attractor_singular": (l_x > 0.0) != (l_c > 0.0),
                "attractor_plural": (l_a > 0.0) != (l_ac > 0.0),
            },
        }
        choice_support = [
            feature
            for feature in ("A", "C")
            if any(choice_changes[feature].values())
        ]
        contrast_codata_support = []
        if demand["D_attractor_L"] != 0.0 or demand["D_controller_D_attractor_L"] != 0.0:
            contrast_codata_support.append("A")
        if demand["D_controller_L"] != 0.0 or demand["D_controller_D_attractor_L"] != 0.0:
            contrast_codata_support.append("C")
        multiway = multiway_observation(
            np.stack((q_x, q_a, q_c, q_ac))[:, constructor_indices]
        )
        multiway["argmax_constructors"] = {
            corner: [
                {
                    "token": constructor_ids[index],
                    "text": constructor_labels[constructor_ids[index]],
                }
                for index in multiway["argmax_indices"][corner]
            ]
            for corner in CORNER_NAMES
        }

        branch_values = {
            "x": (l_x, singular, plural),
            "A": (l_a, singular, plural),
            "C": (l_c, plural, singular),
            "AC": (l_ac, plural, singular),
        }
        branch_choices: dict[str, str] = {}
        case_decisions: list[dict[str, Any]] = []
        for branch, (value, expected, alternative) in branch_values.items():
            expected_is_plural = expected == plural
            margin = value if expected_is_plural else -value
            actual = plural if value > 0.0 else singular if value < 0.0 else None
            branch_choices[branch] = (
                observer_labels[actual] if actual is not None else "tie"
            )
            row = {
                "case": case.key,
                "phase": case.phase,
                "branch": branch,
                "controller_number": "plural" if "C" in branch else "singular",
                "attractor_number": "plural" if "A" in branch else "singular",
                "expected_token": expected,
                "expected_constructor": observer_labels[expected],
                "alternative_token": alternative,
                "alternative_constructor": observer_labels[alternative],
                "actual_constructor": branch_choices[branch],
                "expected_minus_alternative_margin": margin,
                "matches_manifest_expectation": margin > 0.0,
            }
            decisions.append(row)
            case_decisions.append(row)

        a_preserves_at_controller_singular = (l_x > 0.0) == (l_a > 0.0) and l_x != 0.0 and l_a != 0.0
        a_preserves_at_controller_plural = (l_c > 0.0) == (l_ac > 0.0) and l_c != 0.0 and l_ac != 0.0
        maximum_b_prefix_leak = max(maximum_b_prefix_leak, float(trace["maximum_b_prefix_leak"]))
        check_rows.append(trace["check"])
        records.append(
            {
                "case": case.key,
                "phase": case.phase,
                "trace": case.trace_name,
                "lexemes": {
                    "controller": [case.controller, case.controller_plural],
                    "attractor": [case.attractor, case.attractor_plural],
                    "verb": [case.verb_singular, case.verb_plural],
                },
                "preverb_token_position": trace["token_position"],
                "constructor_coordinates": {
                    "singular": {"token": singular, "text": observer_labels[singular]},
                    "plural": {"token": plural, "text": observer_labels[plural]},
                },
                "constructor_contrasts": {
                    "x": l_x,
                    "A": l_a,
                    "C": l_c,
                    "AC": l_ac,
                },
                "number_demand_spectrum": demand,
                "observer_indexed_demand_support": {
                    "contrast_codata": contrast_codata_support,
                    "constructor_ordering": choice_support,
                    "selected_injection": choice_support,
                },
                "fixed_multiway_constructor_observation": multiway,
                "choice_changes_along_feature_fibers": choice_changes,
                "attractor_projection_decision_preserving": {
                    "controller_singular": a_preserves_at_controller_singular,
                    "controller_plural": a_preserves_at_controller_plural,
                },
                "branch_choices": branch_choices,
                "complete_codata_mobius": {
                    name: codata[index].tolist()
                    for index, name in enumerate(("carrier", "A", "C", "AC"))
                },
                "decisions": case_decisions,
            }
        )

    by_branch: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in decisions:
        by_branch[str(row["branch"])].append(row)
        by_phase[str(row["phase"])].append(row)
    spectra = {
        name: [float(record["number_demand_spectrum"][name]) for record in records]
        for name in (
            "D_controller_L",
            "D_attractor_L",
            "D_controller_D_attractor_L",
        )
    }
    spectra_by_phase = {
        phase: {
            name: [
                float(record["number_demand_spectrum"][name])
                for record in records
                if record["phase"] == phase
            ]
            for name in spectra
        }
        for phase in ("exploration", "confirmation")
    }
    decision_preservation = {
        controller_number: sum(
            bool(record["attractor_projection_decision_preserving"][controller_number])
            for record in records
        )
        for controller_number in ("controller_singular", "controller_plural")
    }

    def support_name(features: list[str]) -> str:
        return "{" + ",".join(features) + "}"

    def support_counts(
        rows: list[dict[str, Any]],
        observer: str,
    ) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for record in rows:
            features = record["observer_indexed_demand_support"][observer]
            counts[support_name(features)] += 1
        return dict(sorted(counts.items()))

    observer_indexed_support = {
        observer: {
            "all_cases": support_counts(records, observer),
            "by_phase": {
                phase: support_counts(
                    [record for record in records if record["phase"] == phase],
                    observer,
                )
                for phase in ("exploration", "confirmation")
            },
        }
        for observer in (
            "contrast_codata",
            "constructor_ordering",
            "selected_injection",
        )
    }

    def multiway_support_counts(
        rows: list[dict[str, Any]],
        observer: str,
    ) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for record in rows:
            features = record["fixed_multiway_constructor_observation"]["supports"][observer]
            counts[support_name(features)] += 1
        return dict(sorted(counts.items()))

    multiway_support = {
        observer: {
            "all_cases": multiway_support_counts(records, observer),
            "by_phase": {
                phase: multiway_support_counts(
                    [record for record in records if record["phase"] == phase],
                    observer,
                )
                for phase in ("exploration", "confirmation")
            },
        }
        for observer in (
            "contrast_codata",
            "constructor_ordering",
            "selected_injection",
        )
    }
    multiway_ties = sum(
        value == 0
        for record in records
        for values in record["fixed_multiway_constructor_observation"]["weak_pairwise_order"].values()
        for value in values
    )
    multiway_strict_inclusions = sum(
        record["fixed_multiway_constructor_observation"]["supports"]["constructor_ordering"]
        != record["fixed_multiway_constructor_observation"]["supports"]["selected_injection"]
        for record in records
    )
    decision_preservation_by_phase: dict[str, dict[str, int]] = {}
    for phase in ("exploration", "confirmation"):
        summary = {
            "cases": sum(record["phase"] == phase for record in records),
        }
        for controller_number in ("controller_singular", "controller_plural"):
            summary[controller_number] = sum(
                bool(record["attractor_projection_decision_preserving"][controller_number])
                for record in records
                if record["phase"] == phase
            )
        summary["both_controller_numbers"] = sum(
            all(record["attractor_projection_decision_preserving"].values())
            for record in records
            if record["phase"] == phase
        )
        decision_preservation_by_phase[phase] = summary
    result = {
        "schema_version": 1,
        "artifact": "preverb_number_projection_injection_demand_cube",
        "semantics": {
            "constructor_contrast": "L=(iota_plural^*-iota_singular^*)q",
            "common_logit_gauge_removed": True,
            "feature_actions": {
                "A": "pluralize attractor number",
                "C": "pluralize grammatical controller number",
            },
            "spectrum": ["D_C L", "D_A L", "D_C D_A L"],
            "exact_factorization": "requires absent-feature Mobius coefficients to be exactly zero",
            "decision_preserving_factorization": "requires dropping a feature never to change the selected constructor",
            "threshold_used": False,
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
            "trace_count": len(records),
        },
        "validation": {
            "aligned_eight_corner_cubes": len(records),
            "complete_codata_width": len(observer_ids),
            "maximum_typed_chain_output_l2_defect": max(
                float(row["maximum_typed_chain_output_l2_defect"]) for row in check_rows
            ),
            "maximum_edge_mobius_inverse_absolute_defect": max(
                float(row["maximum_edge_mobius_inverse_absolute_defect"]) for row in check_rows
            ),
            "maximum_cross_fiber_mobius_inverse_absolute_defect": maximum_c_mobius_inverse_defect,
            "maximum_unconsumed_B_prefix_leak": maximum_b_prefix_leak,
            "maximum_stock_hidden_relative_defect": max(
                float(row["maximum_stock_hidden_relative_defect"]) for row in check_rows
            ),
            "maximum_stock_logit_contrast_l2_defect": max(
                float(row["maximum_stock_logit_contrast_l2_defect"]) for row in check_rows
            ),
            "maximum_stock_logit_contrast_relative_defect": max(
                float(row["maximum_stock_logit_contrast_relative_defect"]) for row in check_rows
            ),
        },
        "manifest_comparison": {
            "overall": margin_summary(decisions),
            "by_branch": {branch: margin_summary(rows) for branch, rows in sorted(by_branch.items())},
            "by_phase": {phase: margin_summary(rows) for phase, rows in sorted(by_phase.items())},
            "decisions": decisions,
        },
        "demand_spectrum": {
            name: coefficient_summary(values) for name, values in spectra.items()
        },
        "demand_spectrum_by_phase": {
            phase: {
                name: coefficient_summary(values)
                for name, values in phase_spectra.items()
            }
            for phase, phase_spectra in spectra_by_phase.items()
        },
        "observer_indexed_demand_lattices": {
            "constructor_family": ["singular_verb", "plural_verb"],
            "gauge_free_observer": "q_plural-q_singular",
            "contrast_codata": {
                "criterion": "feature f is droppable only if every Mobius coefficient whose subset contains f is exactly zero",
                **observer_indexed_support["contrast_codata"],
            },
            "constructor_ordering": {
                "criterion": "feature f is droppable only if the strict constructor order is constant on every f-fiber",
                "strict_ties": 0,
                "binary_ordering_equals_selected_injection": True,
                **observer_indexed_support["constructor_ordering"],
            },
            "selected_injection": {
                "criterion": "feature f is droppable only if argmax is constant on every f-fiber",
                **observer_indexed_support["selected_injection"],
            },
            "inclusion": "P_contrast_codata contains P_ordering contains P_choice",
        },
        "fixed_multiway_injection_family": {
            "candidate_policy": "one fixed coproduct is used at every feature corner and is never selected from local top logits",
            "constructors": [
                {"token": token, "text": constructor_labels[token]}
                for token in constructor_ids
            ],
            "gauge_free_contrast_basis": [
                {
                    "left": constructor_ids[index],
                    "right": constructor_ids[0],
                }
                for index in range(1, len(constructor_ids))
            ],
            "weak_pairwise_order_basis": [
                {"left": constructor_ids[left], "right": constructor_ids[right]}
                for left in range(len(constructor_ids))
                for right in range(left + 1, len(constructor_ids))
            ],
            "observer_indexed_support": multiway_support,
            "pairwise_ties_retained": multiway_ties,
            "cases_with_strict_ordering_choice_support_inclusion": multiway_strict_inclusions,
            "strict_inclusion_oracle": strict_inclusion_oracle(),
        },
        "attractor_projection_decision_preservation": {
            "cases": len(records),
            **decision_preservation,
            "by_phase": decision_preservation_by_phase,
            "both_controller_numbers": sum(
                all(record["attractor_projection_decision_preserving"].values())
                for record in records
            ),
        },
        "cases": records,
        "scope": {
            "establishes": "the complete sampled controller/attractor number Mobius support and per-context contrast-codata, ordering, and choice demand sets for the singular/plural verb family",
            "does_not_yet_establish": "the remaining feature lattice or polynomial substitution closure",
            "interpretation": "choice-relevant A is malformed attractor demand; nonzero D_C D_A makes that dependence nonseparable but is not intrinsically erroneous",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"number demand: cubes={len(records)} observer={len(observer_ids)} "
        f"manifest={result['manifest_comparison']['overall']['matches_manifest_expectation']}/{len(decisions)}"
    )
    for name, summary in result["demand_spectrum"].items():
        print(
            f"  {name:34s} min={summary['minimum']:.6f} "
            f"mean={summary['mean']:.6f} max={summary['maximum']:.6f} "
            f"exact_zero={summary['exact_zero']}"
        )
    preservation = result["attractor_projection_decision_preservation"]
    print(
        "  attractor decision preservation "
        f"controller_sg={preservation['controller_singular']}/{len(records)} "
        f"controller_pl={preservation['controller_plural']}/{len(records)} "
        f"both={preservation['both_controller_numbers']}/{len(records)}"
    )
    multiway_result = result["fixed_multiway_injection_family"]
    print(
        f"  fixed multiway constructors={len(constructor_ids)} "
        f"ties={multiway_result['pairwise_ties_retained']} "
        "strict_order_choice="
        f"{multiway_result['cases_with_strict_ordering_choice_support_inclusion']}/{len(records)}"
    )
    for observer, summary in multiway_result["observer_indexed_support"].items():
        print(f"    {observer:22s} {summary['all_cases']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare one-pass G(H) with independently measured component interfaces.

The direct side is one shared LlamaCompanyShape.  The substitution side is
reconstructed from the original pre-verb number cubes and the independently
collected post-injection hole cubes.  Codata remains structured as one outer
family plus one hole family per outer shape; no scalar score or flat ordering
over constructor pairs is introduced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any, Hashable

import numpy as np

from analyze_grammar_cubes import read_observers
from analyze_number_demand_cubes import (
    CORNER_NAMES,
    DEFAULT_CONSTRUCTOR_FAMILY,
    DEFAULT_HOLE_CONSTRUCTOR_FAMILY,
    DEFAULT_OBSERVERS,
    essential_feature_support,
    multiway_observation,
    read_case_trace,
    require,
)
from analyze_post_injection_holes import read_hole_trace, support_name
from gather_grammar_relations import DEFAULT_MANIFEST, read_manifest
from gather_number_demand_cubes import (
    DEFAULT_OUTPUT as DEFAULT_NUMBER_TRACES,
    expand_cases as expand_number_cases,
)
from gather_polynomial_company import DEFAULT_OUTPUT as DEFAULT_DIRECT_DIRECTORY
from gather_post_injection_holes import (
    DEFAULT_OUTPUT as DEFAULT_HOLE_TRACES,
    expand_hole_cases,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_DIRECT_TRACE = DEFAULT_DIRECT_DIRECTORY / "polynomial-company.jsonl"
DEFAULT_RESULT = ROOT / "outputs" / "cps-stories15m-polynomial-substitution.json"
OBSERVERS = (
    "contrast_codata",
    "constructor_ordering",
    "selected_injection",
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
    parser.add_argument("--direct-trace", type=Path, default=DEFAULT_DIRECT_TRACE)
    parser.add_argument("--number-traces", type=Path, default=DEFAULT_NUMBER_TRACES)
    parser.add_argument("--hole-traces", type=Path, default=DEFAULT_HOLE_TRACES)
    parser.add_argument("--observers", type=Path, default=DEFAULT_OBSERVERS)
    parser.add_argument(
        "--outer-family",
        type=Path,
        default=DEFAULT_CONSTRUCTOR_FAMILY,
    )
    parser.add_argument(
        "--hole-family",
        type=Path,
        default=DEFAULT_HOLE_CONSTRUCTOR_FAMILY,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--model-label", default="Stories15M")
    parser.add_argument("--evaluator-commit")
    return parser.parse_args()


def one(rows: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    matches = [row for row in rows if row.get("kind") == kind]
    require(len(matches) == 1, f"direct trace must contain one {kind}")
    return matches[0]


def read_direct_trace(
    path: Path,
    outer_ids: tuple[int, ...],
    hole_ids: tuple[int, ...],
) -> tuple[
    dict[tuple[str, str], np.ndarray],
    dict[tuple[str, str, int], np.ndarray],
    dict[str, Any],
]:
    require(path.is_file(), f"missing direct polynomial trace: {path}")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    meta = one(rows, "polynomial_company_meta")
    check = one(rows, "polynomial_company_check")
    require(meta.get("schema_version") == 1, "direct trace schema differs")
    require(
        meta.get("semantics") == "structured_polynomial_company_G_of_H",
        "direct trace semantics differ",
    )
    require(tuple(meta["outer_tokens"]) == outer_ids, "direct outer family differs")
    require(tuple(meta["hole_tokens"]) == hole_ids, "direct hole family differs")
    require(meta.get("probabilities_used") is False, "direct trace used probabilities")
    require(meta.get("scalar_reward_used") is False, "direct trace used scalar reward")
    require(meta.get("pair_scores_flattened") is False, "direct trace flattened pairs")
    require(meta.get("maximum_calls_per_filler") == 1, "direct fillers were repeated")
    require(check.get("maximum_calls_per_filler") == 1, "direct check repeated fillers")

    outer: dict[tuple[str, str], np.ndarray] = {}
    holes: dict[tuple[str, str, int], np.ndarray] = {}
    for row in rows:
        if row.get("kind") == "polynomial_company_outer":
            key = (str(row["case"]), str(row["corner"]))
            require(key not in outer, f"duplicate direct outer observation: {key}")
            values = np.asarray(row["contrasts"], dtype=np.float64)
            require(values.shape == (len(outer_ids),), "direct outer width differs")
            require(np.all(np.isfinite(values)), "direct outer codata is non-finite")
            outer[key] = values
        elif row.get("kind") == "polynomial_company_hole":
            key = (
                str(row["case"]),
                str(row["corner"]),
                int(row["outer_token"]),
            )
            require(key not in holes, f"duplicate direct hole observation: {key}")
            values = np.asarray(row["contrasts"], dtype=np.float64)
            require(values.shape == (len(hole_ids),), "direct hole width differs")
            require(np.all(np.isfinite(values)), "direct hole codata is non-finite")
            holes[key] = values
    require(len(outer) == 44 * 4, "direct outer observation count differs")
    require(len(holes) == 44 * 4 * len(outer_ids), "direct hole count differs")
    return outer, holes, meta


def expected_components(
    manifest: dict[str, Any],
    number_traces: Path,
    hole_traces: Path,
    observer_ids: tuple[int, ...],
    observer_labels: dict[int, str],
    outer_ids: tuple[int, ...],
    outer_labels: dict[int, str],
    hole_ids: tuple[int, ...],
    hole_labels: dict[int, str],
) -> tuple[
    dict[tuple[str, str], np.ndarray],
    dict[tuple[str, str, int], np.ndarray],
    dict[str, str],
    dict[str, float],
]:
    observer_index = {token: index for index, token in enumerate(observer_ids)}
    outer_indices = [observer_index[token] for token in outer_ids]
    hole_indices = [observer_index[token] for token in hole_ids]
    expected_outer: dict[tuple[str, str], np.ndarray] = {}
    phase_by_case: dict[str, str] = {}
    maximum_stock_logit_contrast_relative_defect = 0.0
    for case in expand_number_cases(manifest):
        trace = read_case_trace(case, number_traces, observer_ids)
        maximum_stock_logit_contrast_relative_defect = max(
            maximum_stock_logit_contrast_relative_defect,
            float(trace["check"]["maximum_stock_logit_contrast_relative_defect"]),
        )
        corners = np.stack(
            (
                trace["base_raw"][0],
                trace["base_raw"][1],
                trace["c_raw"][0],
                trace["c_raw"][1],
            )
        )[:, outer_indices]
        phase_by_case[case.key] = case.phase
        for index, corner in enumerate(CORNER_NAMES):
            expected_outer[(case.key, corner)] = corners[index]

    expected_holes: dict[tuple[str, str, int], np.ndarray] = {}
    for case in expand_hole_cases(manifest, outer_ids, outer_labels):
        entries, check = read_hole_trace(
            case,
            hole_traces,
            observer_ids,
            observer_labels,
            hole_ids,
            hole_labels,
            hole_indices,
        )
        maximum_stock_logit_contrast_relative_defect = max(
            maximum_stock_logit_contrast_relative_defect,
            float(check["maximum_stock_logit_contrast_relative_defect"]),
        )
        for entry in entries:
            token = int(entry["injection"]["token"])
            corners = entry["_corners"]
            for index, corner in enumerate(CORNER_NAMES):
                key = (case.key, corner, token)
                require(key not in expected_holes, f"duplicate expected hole: {key}")
                expected_holes[key] = corners[index]
    require(len(expected_outer) == 44 * 4, "expected outer count differs")
    require(
        len(expected_holes) == 44 * 4 * len(outer_ids),
        "expected hole count differs",
    )
    return (
        expected_outer,
        expected_holes,
        phase_by_case,
        {
            "maximum_stock_logit_contrast_relative_defect":
                maximum_stock_logit_contrast_relative_defect,
        },
    )


def corner_family(
    values: dict[tuple[str, str], np.ndarray],
    case: str,
) -> np.ndarray:
    return np.stack([values[(case, corner)] for corner in CORNER_NAMES])


def hole_corner_family(
    values: dict[tuple[str, str, int], np.ndarray],
    case: str,
    token: int,
) -> np.ndarray:
    return np.stack([values[(case, corner, token)] for corner in CORNER_NAMES])


def freeze(value: Any) -> Hashable:
    if isinstance(value, dict):
        return tuple((key, freeze(item)) for key, item in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    return value


def structured_observation(
    outer_corners: np.ndarray,
    hole_corners: dict[int, np.ndarray],
    outer_ids: tuple[int, ...],
    hole_ids: tuple[int, ...],
) -> dict[str, Any]:
    outer = multiway_observation(outer_corners)
    holes = {
        token: multiway_observation(hole_corners[token])
        for token in outer_ids
    }
    codata_signatures: dict[str, Hashable] = {}
    ordering_signatures: dict[str, Hashable] = {}
    choice_signatures: dict[str, Hashable] = {}
    decoded_choices: dict[str, list[dict[str, Any]]] = {}
    for corner_index, corner in enumerate(CORNER_NAMES):
        codata_parts: list[float] = []
        codata_parts.extend(
            float(value)
            for value in outer_corners[corner_index, 1:]
            - outer_corners[corner_index, :1]
        )
        for token in outer_ids:
            values = hole_corners[token][corner_index]
            codata_parts.extend(float(value) for value in values[1:] - values[:1])
        codata_signatures[corner] = tuple(codata_parts)

        ordering_signatures[corner] = (
            freeze(outer["weak_pairwise_order"][corner]),
            tuple(
                (
                    token,
                    freeze(holes[token]["weak_pairwise_order"][corner]),
                )
                for token in outer_ids
            ),
        )

        selected_pairs: list[tuple[int, tuple[int, ...]]] = []
        decoded: list[dict[str, Any]] = []
        for outer_index in outer["argmax_indices"][corner]:
            outer_token = outer_ids[outer_index]
            inner_tokens = tuple(
                hole_ids[inner_index]
                for inner_index in holes[outer_token]["argmax_indices"][corner]
            )
            selected_pairs.append((outer_token, inner_tokens))
            decoded.append(
                {
                    "outer_token": outer_token,
                    "hole_argmax_tokens": list(inner_tokens),
                }
            )
        choice_signatures[corner] = tuple(selected_pairs)
        decoded_choices[corner] = decoded

    signatures = {
        "contrast_codata": codata_signatures,
        "constructor_ordering": ordering_signatures,
        "selected_injection": choice_signatures,
    }
    supports = {
        observer: essential_feature_support(signatures[observer])
        for observer in OBSERVERS
    }
    reachable_outer_tokens = {
        outer_ids[index]
        for corner in CORNER_NAMES
        for index in outer["argmax_indices"][corner]
    }
    substituted_support: dict[str, list[str]] = {}
    for observer in OBSERVERS:
        predicted = set(outer["supports"][observer])
        retained_tokens = (
            reachable_outer_tokens
            if observer == "selected_injection"
            else set(outer_ids)
        )
        for token in retained_tokens:
            predicted.update(holes[token]["supports"][observer])
        substituted_support[observer] = [
            feature for feature in ("A", "C") if feature in predicted
        ]
        require(
            set(supports[observer]) <= predicted,
            f"structured {observer} exceeds polynomial substitution support",
        )
    return {
        "outer": outer,
        "holes": holes,
        "signatures": signatures,
        "supports": supports,
        "substituted_support_upper_bound": substituted_support,
        "selected_composite_injections": decoded_choices,
    }


def maximum_defect(
    direct: dict[Any, np.ndarray],
    expected: dict[Any, np.ndarray],
) -> tuple[float, Any, int, float, float]:
    require(set(direct) == set(expected), "direct and expected coordinate keys differ")
    maximum = -1.0
    location: Any = None
    coordinate = -1
    direct_value = 0.0
    expected_value = 0.0
    for key in direct:
        defect = np.abs(direct[key] - expected[key])
        index = int(np.argmax(defect))
        value = float(defect[index])
        if value > maximum:
            maximum = value
            location = key
            coordinate = index
            direct_value = float(direct[key][index])
            expected_value = float(expected[key][index])
    return maximum, location, coordinate, direct_value, expected_value


def relative_l2_defect(
    direct: dict[Any, np.ndarray],
    expected: dict[Any, np.ndarray],
) -> float:
    require(set(direct) == set(expected), "relative-defect keys differ")
    defect_square = 0.0
    reference_square = 0.0
    for key in direct:
        difference = direct[key] - expected[key]
        defect_square += float(np.dot(difference, difference))
        reference_square += float(np.dot(expected[key], expected[key]))
    return 0.0 if reference_square == 0.0 else (
        defect_square / reference_square
    ) ** 0.5


def main() -> None:
    args = arguments()
    observer_ids, observer_labels = read_observers(args.observers)
    outer_ids, outer_labels = read_observers(args.outer_family)
    hole_ids, hole_labels = read_observers(args.hole_family)
    require(set(outer_ids) <= set(observer_ids), "outer family is not retained")
    require(set(hole_ids) <= set(observer_ids), "hole family is not retained")
    direct_outer, direct_holes, direct_meta = read_direct_trace(
        args.direct_trace,
        outer_ids,
        hole_ids,
    )
    manifest = read_manifest(args.manifest)
    (
        expected_outer,
        expected_holes,
        phase_by_case,
        independent_validation,
    ) = expected_components(
        manifest,
        args.number_traces,
        args.hole_traces,
        observer_ids,
        observer_labels,
        outer_ids,
        outer_labels,
        hole_ids,
        hole_labels,
    )

    outer_defect = maximum_defect(direct_outer, expected_outer)
    hole_defect = maximum_defect(direct_holes, expected_holes)
    outer_relative_defect = relative_l2_defect(direct_outer, expected_outer)
    hole_relative_defect = relative_l2_defect(direct_holes, expected_holes)
    cases = sorted(phase_by_case)
    behavioral_mismatches = {
        "outer_support": 0,
        "outer_ordering": 0,
        "outer_argmax": 0,
        "hole_support": 0,
        "hole_ordering": 0,
        "hole_argmax": 0,
        "structured_support": 0,
        "structured_ordering": 0,
        "structured_argmax": 0,
    }
    mismatch_by_phase = {
        phase: {name: 0 for name in behavioral_mismatches}
        for phase in sorted(set(phase_by_case.values()))
    }
    records: list[dict[str, Any]] = []
    support_counts: dict[str, dict[str, int]] = {
        observer: defaultdict(int) for observer in OBSERVERS
    }
    bound_counts: dict[str, dict[str, int]] = {
        observer: defaultdict(int) for observer in OBSERVERS
    }
    strict_containments = {observer: 0 for observer in OBSERVERS}

    def mismatch(name: str, phase: str) -> None:
        behavioral_mismatches[name] += 1
        mismatch_by_phase[phase][name] += 1

    for case in cases:
        phase = phase_by_case[case]
        direct_outer_corners = corner_family(direct_outer, case)
        expected_outer_corners = corner_family(expected_outer, case)
        direct_outer_observation = multiway_observation(direct_outer_corners)
        expected_outer_observation = multiway_observation(expected_outer_corners)
        if direct_outer_observation["supports"] != expected_outer_observation["supports"]:
            mismatch("outer_support", phase)
        if (
            direct_outer_observation["weak_pairwise_order"]
            != expected_outer_observation["weak_pairwise_order"]
        ):
            mismatch("outer_ordering", phase)
        if (
            direct_outer_observation["argmax_indices"]
            != expected_outer_observation["argmax_indices"]
        ):
            mismatch("outer_argmax", phase)

        direct_hole_corners = {
            token: hole_corner_family(direct_holes, case, token)
            for token in outer_ids
        }
        expected_hole_corners = {
            token: hole_corner_family(expected_holes, case, token)
            for token in outer_ids
        }
        for token in outer_ids:
            direct_hole_observation = multiway_observation(
                direct_hole_corners[token]
            )
            expected_hole_observation = multiway_observation(
                expected_hole_corners[token]
            )
            if direct_hole_observation["supports"] != expected_hole_observation["supports"]:
                mismatch("hole_support", phase)
            if (
                direct_hole_observation["weak_pairwise_order"]
                != expected_hole_observation["weak_pairwise_order"]
            ):
                mismatch("hole_ordering", phase)
            if (
                direct_hole_observation["argmax_indices"]
                != expected_hole_observation["argmax_indices"]
            ):
                mismatch("hole_argmax", phase)

        direct_structured = structured_observation(
            direct_outer_corners,
            direct_hole_corners,
            outer_ids,
            hole_ids,
        )
        expected_structured = structured_observation(
            expected_outer_corners,
            expected_hole_corners,
            outer_ids,
            hole_ids,
        )
        if direct_structured["supports"] != expected_structured["supports"]:
            mismatch("structured_support", phase)
        if (
            direct_structured["signatures"]["constructor_ordering"]
            != expected_structured["signatures"]["constructor_ordering"]
        ):
            mismatch("structured_ordering", phase)
        if (
            direct_structured["signatures"]["selected_injection"]
            != expected_structured["signatures"]["selected_injection"]
        ):
            mismatch("structured_argmax", phase)

        for observer in OBSERVERS:
            support = direct_structured["supports"][observer]
            bound = direct_structured["substituted_support_upper_bound"][observer]
            support_counts[observer][support_name(support)] += 1
            bound_counts[observer][support_name(bound)] += 1
            if support != bound:
                strict_containments[observer] += 1
        records.append(
            {
                "case": case,
                "phase": phase,
                "direct_support": direct_structured["supports"],
                "substituted_support_upper_bound": direct_structured[
                    "substituted_support_upper_bound"
                ],
                "selected_composite_injections": direct_structured[
                    "selected_composite_injections"
                ],
            }
        )

    require(
        all(count == 0 for count in behavioral_mismatches.values()),
        "one-pass company and independently substituted behavior differ",
    )
    result = {
        "schema_version": 1,
        "artifact": "direct_polynomial_company_substitution",
        "semantics": {
            "container": "G(H) retains q_D and the family d -> q_E(d)",
            "codata_observer": "outer contrasts followed by every shape-indexed hole contrast family",
            "ordering_observer": "outer weak order plus each outer shape's independent hole weak order",
            "choice_observer": "outer argmax shapes paired with their own hole argmax sets",
            "flat_pair_order_invented": False,
            "probabilities_used": False,
            "scalar_completion_reward_used": False,
        },
        "provenance": {
            "model": args.model_label,
            "evaluator_commit": args.evaluator_commit or "unspecified",
            "analyzer_commit": git_head(),
            "direct_trace_sha256": hashlib.sha256(
                args.direct_trace.read_bytes()
            ).hexdigest(),
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "outer_family_sha256": hashlib.sha256(
                args.outer_family.read_bytes()
            ).hexdigest(),
            "hole_family_sha256": hashlib.sha256(
                args.hole_family.read_bytes()
            ).hexdigest(),
        },
        "one_pass_company": {
            "contexts": direct_meta["contexts"],
            "rows": direct_meta["company_rows"],
            "learned_fillers": direct_meta["learned_fillers"],
            "family_filler_calls": direct_meta["family_filler_calls"],
            "maximum_calls_per_filler": direct_meta["maximum_calls_per_filler"],
            "family_scalar_reads": direct_meta["family_scalar_reads"],
            "outer_observations": len(direct_outer),
            "shape_indexed_hole_observations": len(direct_holes),
        },
        "validation": {
            "behavioral_mismatches": behavioral_mismatches,
            "behavioral_mismatches_by_phase": mismatch_by_phase,
            "maximum_outer_codata_absolute_defect": outer_defect[0],
            "outer_codata_relative_l2_defect": outer_relative_defect,
            "maximum_outer_codata_location": {
                "key": list(outer_defect[1]),
                "coordinate": outer_defect[2],
                "direct": outer_defect[3],
                "independent": outer_defect[4],
            },
            "maximum_hole_codata_absolute_defect": hole_defect[0],
            "hole_codata_relative_l2_defect": hole_relative_defect,
            "maximum_hole_codata_location": {
                "key": list(hole_defect[1]),
                "coordinate": hole_defect[2],
                "direct": hole_defect[3],
                "independent": hole_defect[4],
            },
            **independent_validation,
        },
        "polynomial_substitution": {
            observer: {
                "direct_support_counts": dict(sorted(support_counts[observer].items())),
                "substituted_support_upper_bound_counts": dict(
                    sorted(bound_counts[observer].items())
                ),
                "strict_containments": strict_containments[observer],
                "unexpected_support_cases": 0,
            }
            for observer in OBSERVERS
        },
        "cases": records,
        "scope": {
            "establishes": "one-pass structured G(H) equals independent component execution on codata demand, hierarchical weak ordering, and selected composite injections",
            "does_not_claim": "a canonical flat score or total order over constructor pairs, or closure beyond the measured A/C feature and fixed D/E constructor families",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        "polynomial substitution: "
        f"contexts={len(cases)} rows={direct_meta['company_rows']} "
        f"fillers={direct_meta['learned_fillers']} "
        f"max_calls={direct_meta['maximum_calls_per_filler']}"
    )
    print(
        f"  raw defects outer={outer_defect[0]:.9g} "
        f"holes={hole_defect[0]:.9g} "
        f"relative=({outer_relative_defect:.9g},{hole_relative_defect:.9g})"
    )
    print(f"  behavioral mismatches {behavioral_mismatches}")
    for observer in OBSERVERS:
        summary = result["polynomial_substitution"][observer]
        print(
            f"  {observer:22s} direct={summary['direct_support_counts']} "
            f"bound={summary['substituted_support_upper_bound_counts']} "
            f"strict={summary['strict_containments']}"
        )


if __name__ == "__main__":
    main()

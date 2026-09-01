#!/usr/bin/env python3
"""Refine a grammatical observational quotient from future behavior.

The retained observation for state ``x`` and future action word ``w`` is the
complete final-root vector

    V_x(w) = (U_a - I)(U_b - I)(k after w)(x).

Primitive differences ``V_x(c) - V_x(epsilon)`` are third-order Moebius
observations.  Depth-two conditional differences retain the next order.  The
first analysis is exact partition refinement over those vectors.  SVD and
controller/attractor labels are used only afterward as diagnostics; no norm,
residual, rank, or label is an inference reward or an input to refinement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from gather_grammar_behaviors import (
    DEFAULT_ACTIONS,
    DEFAULT_OUTPUT as DEFAULT_TRACES,
    ActionWord,
    expand_words,
    read_actions,
)
from gather_grammar_relations import (
    DEFAULT_MANIFEST,
    ActionCase,
    expand_cases,
    read_manifest,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_ANALYSIS = ROOT / "outputs" / "cps-grammar-relations-analysis.json"
DEFAULT_RESULT = ROOT / "outputs" / "cps-grammar-behaviors-analysis.json"
ROLES = ("x", "ax", "bx", "abx", "bax")


@dataclass(frozen=True)
class StateSpec:
    diagram: ActionCase
    fold: int | None

    @property
    def key(self) -> str:
        return "/".join(
            (
                self.diagram.phase,
                self.diagram.template,
                self.diagram.family,
                self.diagram.role,
            )
        )

    def trace_name(self, word: ActionWord) -> str:
        stem = self.diagram.trace_name.removesuffix(".jsonl")
        return f"{stem}--w-{word.name}.jsonl"


@dataclass
class BehaviorTrace:
    state: StateSpec
    word: ActionWord
    positions: int
    dim: int
    layers: int
    corners: dict[str, np.ndarray]
    mixed: np.ndarray
    commutator: np.ndarray
    maximum_reference_relative_defect: float
    mixed_reconstruction_maximum_absolute_defect: float
    commutator_reconstruction_maximum_absolute_defect: float


@dataclass(frozen=True)
class RowBasis:
    rows: np.ndarray
    gram_pseudoinverse: np.ndarray
    rank: int
    tolerance: float


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--actions", type=Path, default=DEFAULT_ACTIONS)
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--base-analysis", type=Path, default=DEFAULT_BASE_ANALYSIS)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--model-label", default="Stories15M")
    parser.add_argument(
        "--evaluator-commit",
        help="Git commit of the C evaluator that produced the behavior traces",
    )
    return parser.parse_args()


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


def as_float32_vector(value: Any, field: str, path: Path) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    require(vector.ndim == 1, f"{path}: {field} is not a vector")
    require(np.all(np.isfinite(vector)), f"{path}: {field} is not finite")
    return vector


def scaled_float32_defect(calculated: np.ndarray, retained: np.ndarray) -> float:
    difference = np.abs(calculated.astype(np.float64) - retained.astype(np.float64))
    return float(np.max(difference, initial=0.0))


def float32_rounding_bound(*vectors: np.ndarray) -> float:
    scale = max(1.0, *(float(np.max(np.abs(vector), initial=0.0)) for vector in vectors))
    return 8.0 * float(np.finfo(np.float32).eps) * scale


def read_trace(
    state: StateSpec,
    word: ActionWord,
    trace_directory: Path,
) -> BehaviorTrace:
    path = trace_directory / state.trace_name(word)
    require(path.is_file(), f"missing behavior trace: {path}")
    retained: dict[str, list[dict[str, Any]]] = defaultdict(list)
    allowed = {
        "grammatical_behavior_meta",
        "grammatical_action_context",
        "grammatical_behavior_root",
        "grammatical_behavior_reference_check",
        "grammatical_behavior_check",
    }
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f"{path}:{line_number}: {error}") from error
            kind = row.get("kind")
            require(kind in allowed, f"{path}:{line_number}: unexpected row {kind}")
            retained[kind].append(row)

    def one(kind: str) -> dict[str, Any]:
        rows = retained[kind]
        require(len(rows) == 1, f"{path}: expected one {kind}, found {len(rows)}")
        return rows[0]

    meta = one("grammatical_behavior_meta")
    root = one("grammatical_behavior_root")
    check = one("grammatical_behavior_check")
    contexts = retained["grammatical_action_context"]
    references = retained["grammatical_behavior_reference_check"]
    require(meta.get("schema_version") == 1, f"{path}: wrong trace schema")
    require(
        meta.get("semantics") == "future_company_root_behavior",
        f"{path}: wrong behavior semantics",
    )
    require(meta.get("root_scope") == "last", f"{path}: root is not last-token state")
    require(meta.get("factorized_token_square") is True, f"{path}: square is not typed")
    require(meta.get("constructors_commute_on_x") is True, f"{path}: actions do not commute")
    require(meta.get("norms_are_diagnostics_not_scores") is True, f"{path}: unsafe score semantics")
    require(len(contexts) == len(ROLES), f"{path}: expected five contexts")
    require(
        {row.get("role") for row in contexts} == set(ROLES),
        f"{path}: context roles differ",
    )
    require(len(references) == len(ROLES), f"{path}: expected five parity rows")
    require(
        {row.get("role") for row in references} == set(ROLES),
        f"{path}: parity roles differ",
    )

    dim = int(meta["dim"])
    require(int(root["root_width"]) == dim, f"{path}: root width differs from hidden width")
    corner_object = root.get("corners")
    require(isinstance(corner_object, dict), f"{path}: missing root corners")
    require(set(corner_object) == set(ROLES), f"{path}: wrong root corner set")
    corners = {
        role: as_float32_vector(corner_object[role], f"corners.{role}", path)
        for role in ROLES
    }
    require(all(vector.size == dim for vector in corners.values()), f"{path}: corner width differs")
    mixed = as_float32_vector(root["mixed"], "mixed", path)
    commutator = as_float32_vector(root["commutator"], "commutator", path)
    require(mixed.size == dim and commutator.size == dim, f"{path}: derived width differs")

    calculated_mixed = (
        (corners["abx"] - corners["ax"]) - corners["bx"]
    ) + corners["x"]
    calculated_commutator = corners["abx"] - corners["bax"]
    mixed_defect = scaled_float32_defect(calculated_mixed, mixed)
    commutator_defect = scaled_float32_defect(calculated_commutator, commutator)
    bound = float32_rounding_bound(*corners.values(), mixed, commutator)
    require(mixed_defect <= bound, f"{path}: mixed vector does not reconstruct")
    require(commutator_defect <= bound, f"{path}: commutator does not reconstruct")
    require(
        np.array_equal(corners["abx"], corners["bax"]),
        f"{path}: commuting constructor roots differ",
    )
    maximum_reference = float(check["maximum_llama2c_hidden_relative_defect"])
    require(np.isfinite(maximum_reference), f"{path}: nonfinite stock parity")

    return BehaviorTrace(
        state=state,
        word=word,
        positions=int(meta["positions"]),
        dim=dim,
        layers=int(meta["layers"]),
        corners=corners,
        mixed=mixed,
        commutator=commutator,
        maximum_reference_relative_defect=maximum_reference,
        mixed_reconstruction_maximum_absolute_defect=mixed_defect,
        commutator_reconstruction_maximum_absolute_defect=commutator_defect,
    )


def expand_states(manifest: dict[str, Any]) -> list[StateSpec]:
    folds = {
        family["name"]: int(family["fold"])
        for family in manifest["exploration_families"]
    }
    states = [
        StateSpec(
            diagram=diagram,
            fold=folds.get(diagram.family),
        )
        for diagram in expand_cases(manifest, "all")
    ]
    require(len(states) == 88, "expected 88 grammatical states")
    require(len({state.key for state in states}) == len(states), "duplicate state keys")
    return states


def matched_pairs(states: list[StateSpec], phase: str | None = None) -> list[tuple[StateSpec, StateSpec]]:
    grouped: dict[tuple[str, str, str], dict[str, StateSpec]] = defaultdict(dict)
    for state in states:
        if phase is None or state.diagram.phase == phase:
            key = (state.diagram.phase, state.diagram.template, state.diagram.family)
            grouped[key][state.diagram.role] = state
    pairs: list[tuple[StateSpec, StateSpec]] = []
    for key in sorted(grouped):
        roles = grouped[key]
        require(set(roles) == {"controller", "attractor"}, f"unpaired state {key}")
        pairs.append((roles["controller"], roles["attractor"]))
    return pairs


def vector_key(vector: np.ndarray) -> bytes:
    return np.ascontiguousarray(vector).tobytes()


def refine_once(
    partition: list[list[StateSpec]],
    observation: Callable[[StateSpec], np.ndarray],
) -> tuple[list[list[StateSpec]], list[dict[str, Any]]]:
    refined: list[list[StateSpec]] = []
    split_records: list[dict[str, Any]] = []
    for block in partition:
        groups: dict[bytes, list[StateSpec]] = defaultdict(list)
        for state in block:
            groups[vector_key(observation(state))].append(state)
        ordered = sorted(groups.values(), key=lambda group: tuple(item.key for item in group))
        refined.extend(ordered)
        if len(ordered) > 1:
            split_records.append(
                {
                    "prior_members": [state.key for state in block],
                    "result_classes": [
                        [state.key for state in group] for group in ordered
                    ],
                }
            )
    return refined, split_records


def refinement_summary(partition: list[list[StateSpec]]) -> dict[str, Any]:
    sizes = sorted((len(block) for block in partition), reverse=True)
    return {
        "class_count": len(partition),
        "non_singleton_class_count": sum(size > 1 for size in sizes),
        "largest_class_size": max(sizes, default=0),
        "class_sizes": sizes,
        "classes": [[state.key for state in block] for block in partition],
    }


def counterexample_guided_refinement(
    initial_partition: list[list[StateSpec]],
    candidates: list[tuple[str, str, Callable[[StateSpec], np.ndarray]]],
) -> dict[str, Any]:
    partition = [sorted(block, key=lambda state: state.key) for block in initial_partition]
    remaining = list(candidates)
    steps: list[dict[str, Any]] = []
    while remaining and any(len(block) > 1 for block in partition):
        evaluated: list[
            tuple[int, int, str, str, Callable[[StateSpec], np.ndarray], list[list[StateSpec]], list[dict[str, Any]]]
        ] = []
        for index, (name, order, observation) in enumerate(remaining):
            refined, split_records = refine_once(partition, observation)
            gain = len(refined) - len(partition)
            evaluated.append(
                (gain, -index, name, order, observation, refined, split_records)
            )
        selected = max(evaluated, key=lambda item: (item[0], item[1]))
        gain, _, name, order, observation, refined, split_records = selected
        candidate_gains = [
            {
                "name": item[2],
                "order": item[3],
                "class_gain": item[0],
            }
            for item in evaluated
        ]
        if gain == 0:
            steps.append(
                {
                    "status": "stable_under_retained_candidates",
                    "candidate_gains": candidate_gains,
                }
            )
            break
        steps.append(
            {
                "status": "refined",
                "selected": name,
                "observation_order": order,
                "classes_before": len(partition),
                "classes_after": len(refined),
                "class_gain": gain,
                "candidate_gains": candidate_gains,
                "splits": split_records,
            }
        )
        partition = refined
        remaining = [item for item in remaining if item[0] != name]
    return {
        "initial": refinement_summary(initial_partition),
        "steps": steps,
        "final": refinement_summary(partition),
        "selected_words": [step["selected"] for step in steps if "selected" in step],
    }


def numerical_row_rank(matrix: np.ndarray) -> dict[str, Any]:
    require(matrix.ndim == 2 and matrix.shape[0] > 0, "rank matrix is empty")
    gram = matrix @ matrix.T
    eigenvalues = np.linalg.eigvalsh(gram)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    singular_values = np.sqrt(eigenvalues)[::-1]
    require(singular_values[0] > 0.0, "rank matrix is zero")
    tolerance = float(max(matrix.shape) * np.finfo(np.float64).eps * singular_values[0])
    rank = int(np.count_nonzero(singular_values > tolerance))
    energy = singular_values * singular_values
    cumulative = np.cumsum(energy) / np.sum(energy)

    def energy_rank(target: float) -> int:
        return int(np.searchsorted(cumulative, target, side="left") + 1)

    return {
        "shape": list(matrix.shape),
        "rank": rank,
        "row_rank_saturated": rank == matrix.shape[0],
        "rank_tolerance": tolerance,
        "largest_singular_value": float(singular_values[0]),
        "smallest_retained_to_largest": float(
            singular_values[rank - 1] / singular_values[0]
        ),
        "energy_rank_90_percent": energy_rank(0.90),
        "energy_rank_99_percent": energy_rank(0.99),
        "energy_rank_99_9_percent": energy_rank(0.999),
    }


def fit_row_basis(vectors: list[np.ndarray]) -> RowBasis:
    require(bool(vectors), "cannot fit an empty row basis")
    rows = np.stack(vectors).astype(np.float64, copy=False)
    gram = rows @ rows.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    maximum_singular = float(np.sqrt(max(float(eigenvalues[-1]), 0.0)))
    require(maximum_singular > 0.0, "training row basis is zero")
    tolerance = float(max(rows.shape) * np.finfo(np.float64).eps * maximum_singular)
    retained = eigenvalues > tolerance * tolerance
    rank = int(np.count_nonzero(retained))
    require(rank > 0, "training row basis has zero numerical rank")
    vectors_retained = eigenvectors[:, retained]
    gram_pseudoinverse = (
        (vectors_retained / eigenvalues[retained]) @ vectors_retained.T
    )
    return RowBasis(rows, gram_pseudoinverse, rank, tolerance)


def relative_row_residual(vector: np.ndarray, basis: RowBasis) -> float:
    vector64 = vector.astype(np.float64, copy=False)
    norm_squared = float(vector64 @ vector64)
    require(norm_squared > 0.0, "zero behavior vector has no relative residual")
    inner = basis.rows @ vector64
    projection_squared = float(inner @ basis.gram_pseudoinverse @ inner)
    projection_squared = min(max(projection_squared, 0.0), norm_squared)
    return float(np.sqrt(max(norm_squared - projection_squared, 0.0) / norm_squared))


def evaluate_pairs(
    pairs: list[tuple[StateSpec, StateSpec]],
    controller_basis: RowBasis,
    attractor_basis: RowBasis,
    vector_of: Callable[[StateSpec], np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for controller, attractor in pairs:
        cv = vector_of(controller)
        av = vector_of(attractor)
        c_in_c = relative_row_residual(cv, controller_basis)
        a_in_c = relative_row_residual(av, controller_basis)
        a_in_a = relative_row_residual(av, attractor_basis)
        c_in_a = relative_row_residual(cv, attractor_basis)
        rows.append(
            {
                "template": controller.diagram.template,
                "family": controller.diagram.family,
                "fold": controller.fold,
                "controller_in_controller_basis": c_in_c,
                "attractor_in_controller_basis": a_in_c,
                "controller_basis_win": c_in_c < a_in_c,
                "attractor_in_attractor_basis": a_in_a,
                "controller_in_attractor_basis": c_in_a,
                "attractor_basis_win": a_in_a < c_in_a,
                "controller_nearest_basis_win": c_in_c < c_in_a,
                "attractor_nearest_basis_win": a_in_a < a_in_c,
            }
        )
    return rows


def summarize_pair_evaluations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "matched_relation_basis": {
            "wins": sum(row["controller_basis_win"] for row in rows),
            "total": len(rows),
        },
        "symmetric_nearest_basis": {
            "wins": sum(
                row["controller_nearest_basis_win"]
                + row["attractor_nearest_basis_win"]
                for row in rows
            ),
            "total": 2 * len(rows),
        },
    }


def cross_validate_behavior(
    states: list[StateSpec],
    vector_of: Callable[[StateSpec], np.ndarray],
) -> dict[str, Any]:
    exploration = [state for state in states if state.diagram.phase == "exploration"]
    templates = sorted({state.diagram.template for state in exploration})
    folds = sorted({state.fold for state in exploration if state.fold is not None})
    pairs = matched_pairs(states, "exploration")
    evaluations: list[dict[str, Any]] = []
    ranks: list[dict[str, Any]] = []
    for held_template in templates:
        for held_fold in folds:
            training = [
                state
                for state in exploration
                if state.diagram.template != held_template and state.fold != held_fold
            ]
            testing = [
                pair
                for pair in pairs
                if pair[0].diagram.template == held_template
                and pair[0].fold == held_fold
            ]
            require(len(training) == 32, "unexpected behavior training split")
            require(len(testing) == 4, "unexpected behavior held-out split")
            controller_basis = fit_row_basis(
                [vector_of(state) for state in training if state.diagram.role == "controller"]
            )
            attractor_basis = fit_row_basis(
                [vector_of(state) for state in training if state.diagram.role == "attractor"]
            )
            ranks.append(
                {
                    "held_template": held_template,
                    "held_fold": held_fold,
                    "controller": controller_basis.rank,
                    "attractor": attractor_basis.rank,
                }
            )
            evaluations.extend(
                evaluate_pairs(testing, controller_basis, attractor_basis, vector_of)
            )
    require(len(evaluations) == 36, "exploration behavior was not held out exactly once")
    result = summarize_pair_evaluations(evaluations)
    result["basis_rank_range"] = {
        "controller": [
            min(item["controller"] for item in ranks),
            max(item["controller"] for item in ranks),
        ],
        "attractor": [
            min(item["attractor"] for item in ranks),
            max(item["attractor"] for item in ranks),
        ],
    }
    return result


def confirm_behavior(
    states: list[StateSpec],
    vector_of: Callable[[StateSpec], np.ndarray],
) -> dict[str, Any]:
    exploration = [state for state in states if state.diagram.phase == "exploration"]
    controller_basis = fit_row_basis(
        [vector_of(state) for state in exploration if state.diagram.role == "controller"]
    )
    attractor_basis = fit_row_basis(
        [vector_of(state) for state in exploration if state.diagram.role == "attractor"]
    )
    evaluations = evaluate_pairs(
        matched_pairs(states, "confirmation"),
        controller_basis,
        attractor_basis,
        vector_of,
    )
    result = summarize_pair_evaluations(evaluations)
    result["basis_rank"] = {
        "controller": controller_basis.rank,
        "attractor": attractor_basis.rank,
    }
    result["cases"] = evaluations
    return result


def main() -> None:
    args = arguments()
    manifest = read_manifest(args.manifest)
    actions = read_actions(args.actions)
    states = expand_states(manifest)
    words = expand_words(actions)
    word_by_name = {word.name: word for word in words}
    require("epsilon" in word_by_name and "epsilon_repeat" in word_by_name, "missing identity controls")
    primitives = [word for word in words if len(word.factors) == 1]
    composites = [word for word in words if len(word.factors) == 2]
    require(len(primitives) == 9 and len(composites) == 16, "unexpected future-action family")

    base = json.loads(args.base_analysis.read_text(encoding="utf-8"))
    base_validation = base.get("validation", {})
    require(base_validation.get("matched_layer0_qk_pairs_exact") == 44, "base QK pairs are not exact")
    require(base_validation.get("matched_layer0_qk_pairs_total") == 44, "base QK pair count differs")
    require(base_validation.get("matched_layer0_qk_maximum_absolute_difference") == 0.0, "base QK bridge differs")

    traces: dict[tuple[StateSpec, str], BehaviorTrace] = {}
    for state in states:
        for word in words:
            traces[(state, word.name)] = read_trace(state, word, args.traces)
    types = {(trace.dim, trace.layers) for trace in traces.values()}
    require(len(types) == 1, "behavior hidden types differ")

    epsilon = word_by_name["epsilon"]
    repeat = word_by_name["epsilon_repeat"]
    repeat_exact_states = 0
    repeat_maximum = 0.0
    for state in states:
        left = traces[(state, epsilon.name)]
        right = traces[(state, repeat.name)]
        arrays = [(left.mixed, right.mixed), (left.commutator, right.commutator)]
        arrays.extend((left.corners[role], right.corners[role]) for role in ROLES)
        if all(np.array_equal(a, b) for a, b in arrays):
            repeat_exact_states += 1
        repeat_maximum = max(
            repeat_maximum,
            *(float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)), initial=0.0)) for a, b in arrays),
        )
    require(repeat_exact_states == len(states), "duplicate identity execution was not bit-identical")

    absolute: dict[tuple[StateSpec, str], np.ndarray] = {}
    third: dict[tuple[StateSpec, str], np.ndarray] = {}
    fourth: dict[tuple[StateSpec, str], np.ndarray] = {}
    third_expansion_maximum = 0.0
    for state in states:
        base_trace = traces[(state, epsilon.name)]
        for word in words:
            if word.repeats is None:
                absolute[(state, word.name)] = traces[(state, word.name)].mixed
        for word in primitives:
            current = traces[(state, word.name)]
            value = np.subtract(current.mixed, base_trace.mixed, dtype=np.float32)
            third[(state, word.name)] = value
            expanded = (
                current.corners["abx"].astype(np.float64)
                - current.corners["ax"].astype(np.float64)
                - current.corners["bx"].astype(np.float64)
                + current.corners["x"].astype(np.float64)
                - base_trace.corners["abx"].astype(np.float64)
                + base_trace.corners["ax"].astype(np.float64)
                + base_trace.corners["bx"].astype(np.float64)
                - base_trace.corners["x"].astype(np.float64)
            )
            third_expansion_maximum = max(
                third_expansion_maximum,
                float(np.max(np.abs(expanded - value.astype(np.float64)), initial=0.0)),
            )
        for word in composites:
            left_name, right_name = word.factors
            value = (
                traces[(state, word.name)].mixed
                - traces[(state, left_name)].mixed
                - traces[(state, right_name)].mixed
                + base_trace.mixed
            ).astype(np.float32)
            fourth[(state, word.name)] = value

    absolute_words = [word for word in words if word.repeats is None]
    future_words = [word for word in absolute_words if word.name != epsilon.name]

    def concatenate(table: dict[tuple[StateSpec, str], np.ndarray], selected: list[ActionWord], state: StateSpec) -> np.ndarray:
        return np.concatenate([table[(state, word.name)] for word in selected]).astype(np.float64)

    absolute_zip = {state: concatenate(absolute, absolute_words, state) for state in states}
    future_increment_zip = {
        state: np.concatenate(
            [
                np.subtract(
                    traces[(state, word.name)].mixed,
                    traces[(state, epsilon.name)].mixed,
                    dtype=np.float32,
                )
                for word in future_words
            ]
        ).astype(np.float64)
        for state in states
    }
    third_zip = {state: concatenate(third, primitives, state) for state in states}
    fourth_zip = {state: concatenate(fourth, composites, state) for state in states}

    pair_partition = [[controller, attractor] for controller, attractor in matched_pairs(states)]
    absolute_candidates = [
        (
            word.name,
            "second_order_absolute",
            lambda state, name=word.name: absolute[(state, name)],
        )
        for word in absolute_words
    ]
    moebius_candidates = [
        (
            word.name,
            "third_order",
            lambda state, name=word.name: third[(state, name)],
        )
        for word in primitives
    ] + [
        (
            word.name,
            "fourth_order_conditional",
            lambda state, name=word.name: fourth[(state, name)],
        )
        for word in composites
    ]
    absolute_refinement = counterexample_guided_refinement(
        pair_partition,
        absolute_candidates,
    )
    moebius_refinement = counterexample_guided_refinement(
        pair_partition,
        moebius_candidates,
    )
    global_moebius_refinement = counterexample_guided_refinement(
        [states],
        moebius_candidates,
    )

    primitive_partition = [states]
    for word in primitives:
        primitive_partition, _ = refine_once(
            primitive_partition,
            lambda state, name=word.name: third[(state, name)],
        )
    nontrivial_premise_pairs = 0
    congruence_violations = 0
    for block in primitive_partition:
        if len(block) < 2:
            continue
        nontrivial_premise_pairs += len(block) * (len(block) - 1) // 2
        for word in composites:
            groups = {vector_key(fourth[(state, word.name)]) for state in block}
            if len(groups) > 1:
                congruence_violations += 1

    matrices = {
        "absolute_future_behavior": np.stack([absolute_zip[state] for state in states]),
        "future_increment_behavior": np.stack([future_increment_zip[state] for state in states]),
        "third_order_primitive": np.stack([third_zip[state] for state in states]),
        "fourth_order_conditional": np.stack([fourth_zip[state] for state in states]),
    }
    ranks = {name: numerical_row_rank(matrix) for name, matrix in matrices.items()}

    representations = {
        "absolute_future_behavior": absolute_zip,
        "future_increment_behavior": future_increment_zip,
        "third_order_primitive": third_zip,
        "fourth_order_conditional": fourth_zip,
    }
    generalization: dict[str, Any] = {}
    for name, vectors in representations.items():
        vector_of = lambda state, table=vectors: table[state]
        generalization[name] = {
            "exploration_cross_validation": cross_validate_behavior(states, vector_of),
            "confirmation": confirm_behavior(states, vector_of),
        }

    per_primitive: dict[str, Any] = {}
    for word in primitives:
        vector_of = lambda state, name=word.name: third[(state, name)].astype(np.float64)
        per_primitive[word.name] = {
            "family": word.family,
            "exploration_cross_validation": cross_validate_behavior(states, vector_of),
            "confirmation": confirm_behavior(states, vector_of),
        }

    matched_behavior: dict[str, Any] = {}
    all_pairs = matched_pairs(states)
    for word in primitives:
        absolute_exact = 0
        third_exact = 0
        absolute_distances: list[float] = []
        third_distances: list[float] = []
        for controller, attractor in all_pairs:
            av = absolute[(controller, word.name)]
            bv = absolute[(attractor, word.name)]
            tv = third[(controller, word.name)]
            uv = third[(attractor, word.name)]
            absolute_exact += np.array_equal(av, bv)
            third_exact += np.array_equal(tv, uv)
            absolute_distances.append(float(np.linalg.norm(av.astype(np.float64) - bv.astype(np.float64))))
            third_distances.append(float(np.linalg.norm(tv.astype(np.float64) - uv.astype(np.float64))))
        matched_behavior[word.name] = {
            "absolute_equal_pairs": absolute_exact,
            "third_order_equal_pairs": third_exact,
            "pair_count": len(all_pairs),
            "absolute_pair_distance": {
                "minimum": min(absolute_distances),
                "median": float(np.median(absolute_distances)),
                "maximum": max(absolute_distances),
            },
            "third_order_pair_distance": {
                "minimum": min(third_distances),
                "median": float(np.median(third_distances)),
                "maximum": max(third_distances),
            },
        }

    order_records: dict[str, Any] = {}
    composable_names = [
        primitive["name"]
        for primitive in actions["primitives"]
        if primitive.get("compose") is True
    ]
    for left in composable_names:
        for right in composable_names:
            if left >= right:
                continue
            forward = f"{left}__{right}"
            backward = f"{right}__{left}"
            distances = [
                float(
                    np.linalg.norm(
                        absolute[(state, forward)].astype(np.float64)
                        - absolute[(state, backward)].astype(np.float64)
                    )
                )
                for state in states
            ]
            order_records[f"{left}|{right}"] = {
                "unequal_states": sum(distance != 0.0 for distance in distances),
                "state_count": len(states),
                "distance_minimum": min(distances),
                "distance_median": float(np.median(distances)),
                "distance_maximum": max(distances),
            }

    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "counterexample_guided_grammatical_observational_quotient",
        "semantics": {
            "absolute_behavior": "V_x(w)=(U_a-I)(U_b-I)(k after w)(x)",
            "third_order": "V_x(c)-V_x(epsilon)=(U_c-I)(U_a-I)(U_b-I)k(x)",
            "fourth_order": "V_x(cd)-V_x(c)-V_x(d)+V_x(epsilon)",
            "refinement": "exact equality of complete retained vectors; duplicate execution establishes a zero numerical floor",
            "labels": "controller/attractor labels are excluded from behavior construction and refinement and used only for held-out diagnostics",
            "rank": "calculated only after partition refinement and never used as an inference reward",
            "not_an_inference_reward": True,
        },
        "provenance": {
            "model": args.model_label,
            "evaluator_commit": args.evaluator_commit or "unspecified",
            "analyzer_commit": git_head(),
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "actions_sha256": hashlib.sha256(args.actions.read_bytes()).hexdigest(),
            "base_analysis_sha256": hashlib.sha256(args.base_analysis.read_bytes()).hexdigest(),
            "trace_count": len(traces),
            "raw_vectors_embedded": False,
        },
        "validation": {
            "states": len(states),
            "matched_local_qk_pair_blocks": len(pair_partition),
            "future_action_words": len(words),
            "primitive_actions": len(primitives),
            "depth_two_actions": len(composites),
            "hidden_width": next(iter(types))[0],
            "layers": next(iter(types))[1],
            "duplicate_identity_bit_exact_states": repeat_exact_states,
            "duplicate_identity_states": len(states),
            "duplicate_identity_maximum_absolute_difference": repeat_maximum,
            "maximum_stock_forward_hidden_relative_defect": max(
                trace.maximum_reference_relative_defect for trace in traces.values()
            ),
            "maximum_mixed_reconstruction_absolute_defect": max(
                trace.mixed_reconstruction_maximum_absolute_defect for trace in traces.values()
            ),
            "maximum_commutator_reconstruction_absolute_defect": max(
                trace.commutator_reconstruction_maximum_absolute_defect for trace in traces.values()
            ),
            "maximum_third_order_eight_corner_expansion_absolute_defect": third_expansion_maximum,
        },
        "partition_refinement": {
            "starting_partition": "44 matched controller/attractor pair blocks whose within-pair layer-0 directed-QK tensors are bit-identical in the independently checked base artifact; equality between different pair blocks is not assumed",
            "absolute_behavior": absolute_refinement,
            "third_then_fourth_order_behavior": moebius_refinement,
            "global_third_then_fourth_order_behavior": global_moebius_refinement,
            "retained_congruence_check": {
                "premise": "exact equality under all nine primitive third-order observations",
                "consequent": "exact equality under all sixteen retained depth-two conditional observations",
                "premise_classes": refinement_summary(primitive_partition),
                "nontrivial_premise_pairs": nontrivial_premise_pairs,
                "violating_class_action_pairs": congruence_violations,
                "vacuous": nontrivial_premise_pairs == 0,
            },
        },
        "behavior_matrix_rank_after_refinement": ranks,
        "held_out_role_diagnostics": {
            "protocol": "formation and counterexample selection are label-free; these span comparisons are calculated afterward",
            "representations": generalization,
            "per_primitive_third_order": per_primitive,
        },
        "matched_pair_behavior": matched_behavior,
        "future_action_order": order_records,
        "scope": {
            "establishes": "finite exact future-behavior observations and counterexample-guided refinement on the retained Stories15M contexts",
            "does_not_establish": "an exhaustive continuation quotient, a completion observer, or an inference speedup",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(
        "duplicate identity: "
        f"{repeat_exact_states}/{len(states)} bit-exact; maximum={repeat_maximum:.8g}"
    )
    print(
        "matched-QK refinement: "
        f"absolute {absolute_refinement['initial']['class_count']}->"
        f"{absolute_refinement['final']['class_count']} via "
        f"{absolute_refinement['selected_words']}; "
        f"Moebius {moebius_refinement['initial']['class_count']}->"
        f"{moebius_refinement['final']['class_count']} via "
        f"{moebius_refinement['selected_words']}"
    )
    print("behavior matrix ranks after refinement:")
    for name, rank in ranks.items():
        print(
            f"  {name:30s} rank={rank['rank']:2d}/{rank['shape'][0]} "
            f"r99={rank['energy_rank_99_percent']:2d}"
        )
    print("held-out role diagnostics:")
    for name, analysis in generalization.items():
        explore = analysis["exploration_cross_validation"]
        confirm = analysis["confirmation"]
        print(
            f"  {name:30s} "
            f"explore={explore['matched_relation_basis']['wins']:2d}/36 "
            f"nearest={explore['symmetric_nearest_basis']['wins']:2d}/72 "
            f"confirm={confirm['matched_relation_basis']['wins']:d}/8 "
            f"nearest={confirm['symmetric_nearest_basis']['wins']:2d}/16"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Analyze matched grammatical-action continuations without scalarizing them.

The input rows come from cps_grammar_actions.c.  This program validates the
typed constructor squares and compares complete torsor-difference vectors with
uncentered, held-out continuation subspaces.  A projection residual is only a
diagnostic of membership in a sampled relation span; it is not an inference
reward or a score assigned to a completion.
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


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "grammar_relation_cases.json"
DEFAULT_TRACES = ROOT / "work_traces" / "grammar_relations"
REPRESENTATIONS = (
    "layer0_directed_qk",
    "root_pullback_mixed",
    "qk_transition_zip",
    "typed_transition_zip",
)


@dataclass(frozen=True)
class CaseSpec:
    phase: str
    template: str
    trace_prefix: str
    family: str
    fold: int | None
    role: str

    @property
    def trace_name(self) -> str:
        return f"{self.trace_prefix}-{self.family}-{self.role}.jsonl"


@dataclass
class TraceCase:
    spec: CaseSpec
    positions: int
    dim: int
    layers: int
    pullback_depth: int
    action_positions: tuple[int, int]
    context_tokens: dict[str, list[int]]
    vectors: dict[str, np.ndarray]
    qk_layers: list[np.ndarray]
    block_pullbacks: dict[str, list[np.ndarray]]
    maximum_typed_chain_output_l2_defect: float
    maximum_block_pullback_composition_l2_defect: float
    maximum_llama2c_hidden_relative_defect: float
    telescoping_maximum_absolute_defect: float
    maximum_qk_score_reconstruction_relative_defect: float
    maximum_qk_cross_removed_root_relative: float
    maximum_qk_transition_identity_relative_defect: float


@dataclass(frozen=True)
class Basis:
    rows: np.ndarray
    rank: int
    tolerance: float


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-label", default="Stories15M")
    parser.add_argument(
        "--evaluator-commit",
        help="Git commit of the C evaluator that produced the raw traces",
    )
    return parser.parse_args()


def read_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise SystemExit("grammar relation manifest must have schema version 1")
    return manifest


def expand_specs(manifest: dict[str, Any]) -> list[CaseSpec]:
    phases = (
        ("exploration", "exploration_templates", "exploration_families"),
        ("confirmation", "confirmation_templates", "confirmation_families"),
    )
    specs: list[CaseSpec] = []
    for phase, template_key, family_key in phases:
        for template in manifest[template_key]:
            for family in manifest[family_key]:
                for role in ("controller", "attractor"):
                    specs.append(
                        CaseSpec(
                            phase=phase,
                            template=template["name"],
                            trace_prefix=template["trace_prefix"],
                            family=family["name"],
                            fold=family.get("fold"),
                            role=role,
                        )
                    )
    names = [spec.trace_name for spec in specs]
    require(len(names) == len(set(names)), "manifest has duplicate trace names")
    return specs


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def one(rows: list[dict[str, Any]], kind: str, path: Path) -> dict[str, Any]:
    require(len(rows) == 1, f"{path}: expected one {kind}, found {len(rows)}")
    return rows[0]


def as_vector(row: dict[str, Any], field: str, path: Path) -> np.ndarray:
    require(field in row, f"{path}: missing vector {field}")
    vector = np.asarray(row[field], dtype=np.float64)
    require(vector.ndim == 1, f"{path}: {field} is not a vector")
    require(np.all(np.isfinite(vector)), f"{path}: {field} is not finite")
    return vector


def changed_positions(left: list[int], right: list[int]) -> tuple[int, ...]:
    require(len(left) == len(right), "constructor variants have unequal widths")
    return tuple(index for index, pair in enumerate(zip(left, right)) if pair[0] != pair[1])


def read_trace(spec: CaseSpec, trace_directory: Path) -> TraceCase:
    path = trace_directory / spec.trace_name
    require(path.is_file(), f"missing trace: {path}")
    retained: dict[str, list[dict[str, Any]]] = defaultdict(list)
    boundary_indices: list[int] = []
    root_boundary: dict[str, Any] | None = None
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f"{path}:{line_number}: {error}") from error
            kind = row.get("kind")
            require(isinstance(kind, str), f"{path}:{line_number}: missing kind")
            if kind == "grammatical_action_boundary":
                boundary_indices.append(row["boundary_index"])
                if row["boundary"] == "final_rms":
                    require(root_boundary is None, f"{path}: duplicate final_rms")
                    root_boundary = row
            elif kind in {
                "grammatical_action_meta",
                "grammatical_action_context",
                "grammatical_action_transition",
                "grammatical_qk_causal",
                "grammatical_block_pullback",
                "grammatical_action_reference_check",
                "grammatical_action_telescoping_check",
                "grammatical_action_check",
            }:
                retained[kind].append(row)
            else:
                raise SystemExit(f"{path}:{line_number}: unknown trace row {kind}")

    meta = one(retained["grammatical_action_meta"], "meta", path)
    check = one(retained["grammatical_action_check"], "check", path)
    telescoping = one(
        retained["grammatical_action_telescoping_check"], "telescoping check", path
    )
    require(meta.get("schema_version") == 5, f"{path}: trace schema is not 5")
    require(meta.get("semantics") == "exact_constructor_pullbacks", f"{path}: wrong semantics")
    require(meta.get("root_observer") == "post_final_rms_hidden", f"{path}: wrong root")
    require(meta.get("root_scope") == "last", f"{path}: root scope is not last")
    require(meta.get("factorized_token_square") is True, f"{path}: square is not factorized")
    require(meta.get("constructors_commute_on_x") is True, f"{path}: actions do not commute")
    require(meta.get("qk_directed_cross_terms_retained") is True, f"{path}: QK directions absent")
    require(meta.get("block_pullback_pairs_retained") is True, f"{path}: block pullbacks absent")
    require(meta.get("stock_forward_hidden_parity") is True, f"{path}: stock parity absent")
    require(meta.get("norms_are_diagnostics_not_scores") is True, f"{path}: unsafe norm semantics")

    positions = int(meta["positions"])
    dim = int(meta["dim"])
    layers = int(meta["layers"])
    pullback_depth = int(meta.get("block_pullback_depth", 0))
    require(pullback_depth > 0, f"{path}: invalid block pullback depth")
    require(check["boundaries"] == len(boundary_indices), f"{path}: boundary count mismatch")
    require(boundary_indices == list(range(len(boundary_indices))), f"{path}: boundary order mismatch")
    require(root_boundary is not None, f"{path}: final_rms boundary absent")

    contexts = retained["grammatical_action_context"]
    require(len(contexts) == 5, f"{path}: expected five constructor contexts")
    context_tokens: dict[str, list[int]] = {}
    for context in contexts:
        role = context["role"]
        require(role not in context_tokens, f"{path}: duplicate constructor role {role}")
        tokens = [int(token["id"]) for token in context["tokens"]]
        require(len(tokens) == positions, f"{path}: {role} has wrong token width")
        context_tokens[role] = tokens
    require(set(context_tokens) == {"x", "ax", "bx", "abx", "bax"}, f"{path}: bad roles")
    require(context_tokens["abx"] == context_tokens["bax"], f"{path}: actions do not commute as tokens")
    a_changed = changed_positions(context_tokens["x"], context_tokens["ax"])
    b_changed = changed_positions(context_tokens["x"], context_tokens["bx"])
    require(len(a_changed) == 1 and len(b_changed) == 1, f"{path}: actions are not one-token edits")
    require(a_changed != b_changed, f"{path}: actions occupy the same slot")
    require(
        changed_positions(context_tokens["ax"], context_tokens["abx"]) == b_changed,
        f"{path}: b is not stable after a",
    )
    require(
        changed_positions(context_tokens["bx"], context_tokens["abx"]) == a_changed,
        f"{path}: a is not stable after b",
    )
    action_positions = (a_changed[0], b_changed[0])

    transitions = sorted(
        retained["grammatical_action_transition"], key=lambda row: row["to_boundary_index"]
    )
    require(len(transitions) == len(boundary_indices) - 1, f"{path}: transition count mismatch")
    for index, transition in enumerate(transitions, start=1):
        require(
            transition["from_boundary_index"] == index - 1
            and transition["to_boundary_index"] == index,
            f"{path}: non-adjacent transition at {index}",
        )
        require(
            transition.get("delta_tau_is_vector_difference_not_norm_difference") is True,
            f"{path}: transition {index} has unsafe semantics",
        )

    qk_rows = sorted(retained["grammatical_qk_causal"], key=lambda row: row["layer"])
    require([row["layer"] for row in qk_rows] == list(range(layers)), f"{path}: QK layer order mismatch")
    qk_layers = [as_vector(row, "exact_delta_tau", path) for row in qk_rows]
    require(all(vector.size == dim for vector in qk_layers), f"{path}: QK root width mismatch")
    layer0 = np.concatenate(
        (
            as_vector(qk_rows[0], "directed_a_query_b_key", path),
            as_vector(qk_rows[0], "directed_b_query_a_key", path),
        )
    )
    root_mixed = as_vector(root_boundary, "pullback_mixed", path)
    require(root_mixed.size == dim, f"{path}: root mixed width mismatch")
    transition_vectors = [as_vector(row, "delta_tau", path) for row in transitions]
    require(all(vector.size == dim for vector in transition_vectors), f"{path}: transition width mismatch")

    block_order = {"attention": 0, "ffn": 1}
    block_rows = retained["grammatical_block_pullback"]
    require(len(block_rows) == 2 * layers, f"{path}: block pullback count mismatch")
    require(
        all(row.get("block") in block_order for row in block_rows),
        f"{path}: unknown block pullback",
    )
    block_rows.sort(key=lambda row: (row["layer"], block_order[row["block"]]))
    expected_blocks = [
        (layer, block)
        for layer in range(layers)
        for block in ("attention", "ffn")
    ]
    require(
        [(row["layer"], row["block"]) for row in block_rows] == expected_blocks,
        f"{path}: block pullback order mismatch",
    )
    block_pullbacks: dict[str, list[np.ndarray]] = {}
    for row in block_rows:
        block = row["block"]
        expected_map = "attention_residual" if block == "attention" else "swiglu_residual"
        require(row.get("map") == expected_map, f"{path}: wrong {block} map")
        require(row.get("operator") == "U_F(k)=k_after_F", f"{path}: wrong pullback operator")
        require(row.get("action") == "(U_a-I)(U_b-I)", f"{path}: wrong pullback action")
        require(row.get("root_width") == dim, f"{path}: block pullback width mismatch")
        require(
            row.get("recorded_pullback_depth") == pullback_depth,
            f"{path}: recorded block pullback depth mismatch",
        )
        raw_generations = row.get("mixed_generations")
        require(isinstance(raw_generations, list), f"{path}: pullback generations absent")
        require(
            len(raw_generations) == pullback_depth + 1,
            f"{path}: pullback generation count mismatch",
        )
        generations = [
            np.asarray(vector, dtype=np.float64) for vector in raw_generations
        ]
        require(
            all(vector.ndim == 1 and vector.size == dim for vector in generations),
            f"{path}: block pullback generation width mismatch",
        )
        require(
            all(np.all(np.isfinite(vector)) for vector in generations),
            f"{path}: block pullback generation is not finite",
        )
        generation_l2 = np.asarray(row.get("mixed_generation_l2"), dtype=np.float64)
        require(
            generation_l2.ndim == 1 and generation_l2.size == pullback_depth + 1,
            f"{path}: block pullback generation norms mismatch",
        )
        require(
            np.allclose(
                generation_l2,
                [np.linalg.norm(vector) for vector in generations],
                rtol=2e-6,
                atol=2e-7,
            ),
            f"{path}: block pullback generation norms are inconsistent",
        )
        composed = as_vector(row, "composed_pullback_mixed", path)
        defect = as_vector(row, "composition_defect", path)
        require(
            all(vector.size == dim for vector in (composed, defect)),
            f"{path}: block pullback vector width mismatch",
        )
        require(
            np.allclose(composed - generations[1], defect, rtol=2e-6, atol=2e-7),
            f"{path}: block pullback defect vector is inconsistent",
        )
        block_pullbacks[f"layer_{row['layer']}_{block}"] = generations

    references = retained["grammatical_action_reference_check"]
    require(len(references) == 5, f"{path}: expected five stock-forward checks")
    qk_reconstruction = max(float(row["score_reconstruction_relative_defect"]) for row in qk_rows)
    qk_cross_removed = max(float(row["cross_removed_root_relative"]) for row in qk_rows)
    qk_transition_identity = max(float(row["transition_identity_relative_defect"]) for row in qk_rows)
    vectors = {
        "layer0_directed_qk": layer0,
        "root_pullback_mixed": root_mixed,
        "qk_transition_zip": np.concatenate(qk_layers),
        "typed_transition_zip": np.concatenate(transition_vectors),
    }
    return TraceCase(
        spec=spec,
        positions=positions,
        dim=dim,
        layers=layers,
        pullback_depth=pullback_depth,
        action_positions=action_positions,
        context_tokens=context_tokens,
        vectors=vectors,
        qk_layers=qk_layers,
        block_pullbacks=block_pullbacks,
        maximum_typed_chain_output_l2_defect=float(
            check["maximum_typed_chain_output_l2_defect"]
        ),
        maximum_block_pullback_composition_l2_defect=float(
            check["maximum_block_pullback_composition_l2_defect"]
        ),
        maximum_llama2c_hidden_relative_defect=float(
            check["maximum_llama2c_hidden_relative_defect"]
        ),
        telescoping_maximum_absolute_defect=float(
            telescoping["maximum_absolute_defect"]
        ),
        maximum_qk_score_reconstruction_relative_defect=qk_reconstruction,
        maximum_qk_cross_removed_root_relative=qk_cross_removed,
        maximum_qk_transition_identity_relative_defect=qk_transition_identity,
    )


def paired_cases(cases: list[TraceCase], phase: str) -> list[tuple[TraceCase, TraceCase]]:
    grouped: dict[tuple[str, str], dict[str, TraceCase]] = defaultdict(dict)
    for case in cases:
        if case.spec.phase == phase:
            grouped[(case.spec.template, case.spec.family)][case.spec.role] = case
    pairs: list[tuple[TraceCase, TraceCase]] = []
    for key in sorted(grouped):
        roles = grouped[key]
        require(set(roles) == {"controller", "attractor"}, f"unpaired case: {key}")
        controller = roles["controller"]
        attractor = roles["attractor"]
        require(controller.action_positions == attractor.action_positions, f"{key}: positions differ")
        require(
            (controller.positions, controller.dim, controller.layers)
            == (attractor.positions, attractor.dim, attractor.layers),
            f"{key}: trace types differ",
        )
        for role in ("x", "ax", "bx", "abx", "bax"):
            for position in controller.action_positions:
                require(
                    controller.context_tokens[role][position]
                    == attractor.context_tokens[role][position],
                    f"{key}: action token differs at position {position}",
                )
        pairs.append((controller, attractor))
    return pairs


def fit_basis(vectors: list[np.ndarray]) -> Basis:
    require(bool(vectors), "cannot fit an empty basis")
    width = vectors[0].size
    require(all(vector.size == width for vector in vectors), "basis vector widths differ")
    matrix = np.stack(vectors)
    _, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] == 0.0:
        return Basis(np.empty((0, width)), 0, 0.0)
    tolerance = float(max(matrix.shape) * np.finfo(np.float64).eps * singular_values[0])
    rank = int(np.count_nonzero(singular_values > tolerance))
    return Basis(right[:rank], rank, tolerance)


def relative_residual(vector: np.ndarray, basis: Basis) -> float:
    norm = float(np.linalg.norm(vector))
    require(norm > 0.0, "zero relation vector has no relative residual")
    if basis.rank == 0:
        return 1.0
    projection = (vector @ basis.rows.T) @ basis.rows
    return float(np.linalg.norm(vector - projection) / norm)


def evaluate_pairs(
    pairs: list[tuple[TraceCase, TraceCase]],
    controller_basis: Basis,
    attractor_basis: Basis,
    vector_of: Callable[[TraceCase], np.ndarray],
) -> list[dict[str, Any]]:
    evaluations: list[dict[str, Any]] = []
    for controller, attractor in pairs:
        controller_vector = vector_of(controller)
        attractor_vector = vector_of(attractor)
        c_in_c = relative_residual(controller_vector, controller_basis)
        a_in_c = relative_residual(attractor_vector, controller_basis)
        a_in_a = relative_residual(attractor_vector, attractor_basis)
        c_in_a = relative_residual(controller_vector, attractor_basis)
        evaluations.append(
            {
                "template": controller.spec.template,
                "family": controller.spec.family,
                "fold": controller.spec.fold,
                "controller_in_controller_basis": c_in_c,
                "attractor_in_controller_basis": a_in_c,
                "controller_basis_margin": a_in_c - c_in_c,
                "controller_basis_win": c_in_c < a_in_c,
                "attractor_in_attractor_basis": a_in_a,
                "controller_in_attractor_basis": c_in_a,
                "attractor_basis_margin": c_in_a - a_in_a,
                "attractor_basis_win": a_in_a < c_in_a,
                "controller_nearest_basis_win": c_in_c < c_in_a,
                "attractor_nearest_basis_win": a_in_a < a_in_c,
            }
        )
    return evaluations


def summarize_evaluations(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    controller_wins = sum(row["controller_basis_win"] for row in evaluations)
    nearest_wins = sum(
        row["controller_nearest_basis_win"] + row["attractor_nearest_basis_win"]
        for row in evaluations
    )
    return {
        "matched_relation_basis": {"wins": controller_wins, "total": len(evaluations)},
        "symmetric_nearest_basis": {
            "wins": nearest_wins,
            "total": 2 * len(evaluations),
        },
    }


def cross_validate(
    cases: list[TraceCase],
    pairs: list[tuple[TraceCase, TraceCase]],
    vector_of: Callable[[TraceCase], np.ndarray],
) -> dict[str, Any]:
    templates = sorted({case.spec.template for case in cases if case.spec.phase == "exploration"})
    fold_set = {case.spec.fold for case in cases if case.spec.phase == "exploration"}
    require(None not in fold_set, "exploration family lacks a fold")
    folds = sorted(fold for fold in fold_set if fold is not None)
    evaluations: list[dict[str, Any]] = []
    ranks: list[dict[str, Any]] = []
    for held_template in templates:
        for held_fold in folds:
            train = [
                case
                for case in cases
                if case.spec.phase == "exploration"
                and case.spec.template != held_template
                and case.spec.fold != held_fold
            ]
            test_pairs = [
                pair
                for pair in pairs
                if pair[0].spec.template == held_template
                and pair[0].spec.fold == held_fold
            ]
            require(len(train) == 32, "unexpected exploration training count")
            require(len(test_pairs) == 4, "unexpected exploration held-out count")
            controller_basis = fit_basis(
                [vector_of(case) for case in train if case.spec.role == "controller"]
            )
            attractor_basis = fit_basis(
                [vector_of(case) for case in train if case.spec.role == "attractor"]
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
                evaluate_pairs(test_pairs, controller_basis, attractor_basis, vector_of)
            )
    require(len(evaluations) == 36, "exploration cases were not partitioned exactly once")
    summary = summarize_evaluations(evaluations)
    summary["basis_rank_range"] = {
        "controller": [min(row["controller"] for row in ranks), max(row["controller"] for row in ranks)],
        "attractor": [min(row["attractor"] for row in ranks), max(row["attractor"] for row in ranks)],
    }
    per_template: dict[str, Any] = {}
    for template in templates:
        per_template[template] = summarize_evaluations(
            [row for row in evaluations if row["template"] == template]
        )
    summary["per_template"] = per_template
    return summary


def confirmation(
    cases: list[TraceCase],
    pairs: list[tuple[TraceCase, TraceCase]],
    vector_of: Callable[[TraceCase], np.ndarray],
) -> dict[str, Any]:
    exploration = [case for case in cases if case.spec.phase == "exploration"]
    controller_basis = fit_basis(
        [vector_of(case) for case in exploration if case.spec.role == "controller"]
    )
    attractor_basis = fit_basis(
        [vector_of(case) for case in exploration if case.spec.role == "attractor"]
    )
    evaluations = evaluate_pairs(pairs, controller_basis, attractor_basis, vector_of)
    result = summarize_evaluations(evaluations)
    result["basis_rank"] = {
        "controller": controller_basis.rank,
        "attractor": attractor_basis.rank,
    }
    result["cases"] = evaluations
    return result


def closure_fit(
    training: list[TraceCase],
    validation: list[TraceCase],
    block_keys: list[str],
    dictionary_depth: int,
    observation_basis: Basis | None = None,
) -> dict[str, Any]:
    require(bool(training) and bool(validation), "closure split is empty")
    require(bool(block_keys), "closure representation has no blocks")
    require(dictionary_depth > 0, "closure dictionary depth must be positive")
    require(
        all(
            dictionary_depth <= case.pullback_depth
            for case in training + validation
        ),
        "closure dictionary exceeds recorded pullback depth",
    )

    def matrix(cases: list[TraceCase], generation_offset: int) -> np.ndarray:
        rows = [
            np.concatenate([
                case.block_pullbacks[key][generation]
                if observation_basis is None
                else case.block_pullbacks[key][generation] @ observation_basis.rows.T
                for key in block_keys
                for generation in range(
                    generation_offset,
                    generation_offset + dictionary_depth,
                )
            ])
            for case in cases
        ]
        return np.stack(rows)

    fit_input = matrix(training, 0)
    fit_pulled = matrix(training, 1)
    validation_input = matrix(validation, 0)
    validation_pulled = matrix(validation, 1)
    left, singular_values, right = np.linalg.svd(fit_input, full_matrices=False)
    require(singular_values.size > 0 and singular_values[0] > 0.0, "closure input has zero rank")
    tolerance = float(
        max(fit_input.shape) * np.finfo(np.float32).eps * singular_values[0]
    )
    rank = int(np.count_nonzero(singular_values > tolerance))
    require(rank > 0, "closure input has no retained functions")
    left = left[:, :rank]
    right = right[:rank]
    singular_values = singular_values[:rank]

    pulled_norm = float(np.linalg.norm(fit_pulled))
    require(pulled_norm > 0.0, "closure pullback table is zero")
    represented = left @ (left.T @ fit_pulled)
    representation_relative = float(
        np.linalg.norm(fit_pulled - represented) / pulled_norm
    )
    descended = (fit_pulled @ right.T) @ right
    descent_relative = float(np.linalg.norm(fit_pulled - descended) / pulled_norm)

    coefficient_map = (left.T @ fit_pulled) / singular_values[:, np.newaxis]
    validation_prediction = (validation_input @ right.T) @ coefficient_map
    validation_norm = float(np.linalg.norm(validation_pulled))
    require(validation_norm > 0.0, "closure validation pullback table is zero")
    validation_prediction_relative = float(
        np.linalg.norm(validation_prediction - validation_pulled) / validation_norm
    )
    validation_identity_relative = float(
        np.linalg.norm(validation_input - validation_pulled) / validation_norm
    )
    return {
        "training_rows": fit_input.shape[0],
        "validation_rows": validation_input.shape[0],
        "function_columns": fit_input.shape[1],
        "dictionary_depth": dictionary_depth,
        "sampled_rank": rank,
        "sampled_row_rank_fraction": rank / fit_input.shape[0],
        "sampled_row_rank_saturated": rank == fit_input.shape[0],
        "observation_basis_rank": (
            observation_basis.rank
            if observation_basis is not None
            else None
        ),
        "rank_tolerance": tolerance,
        "largest_singular_value": float(singular_values[0]),
        "smallest_retained_singular_value": float(singular_values[-1]),
        "retained_condition_number": float(
            singular_values[0] / singular_values[-1]
        ),
        "fit_representation_relative": representation_relative,
        "fit_descent_relative": descent_relative,
        "validation_prediction_relative": validation_prediction_relative,
        "validation_identity_relative": validation_identity_relative,
    }


def closure_scope(cases: list[TraceCase], phase: str, role: str) -> list[TraceCase]:
    selected = [case for case in cases if case.spec.phase == phase]
    if role != "combined":
        require(role in {"controller", "attractor"}, f"unknown closure scope {role}")
        selected = [case for case in selected if case.spec.role == role]
    return selected


def summarize_closure_splits(splits: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "fit_representation_relative",
        "fit_descent_relative",
        "validation_prediction_relative",
        "validation_identity_relative",
    )
    summary: dict[str, Any] = {
        "split_count": len(splits),
        "sampled_rank_range": [
            min(split["sampled_rank"] for split in splits),
            max(split["sampled_rank"] for split in splits),
        ],
        "row_rank_saturated_splits": sum(
            split["sampled_row_rank_saturated"] for split in splits
        ),
    }
    for field in fields:
        values = [split[field] for split in splits]
        summary[field] = {
            "minimum": min(values),
            "mean": float(np.mean(values)),
            "maximum": max(values),
        }
    summary["splits"] = splits
    return summary


def cross_validate_closure(
    cases: list[TraceCase],
    role: str,
    block_keys: list[str],
    dictionary_depth: int,
    relation_projected: bool,
) -> dict[str, Any]:
    templates = sorted({
        case.spec.template for case in cases if case.spec.phase == "exploration"
    })
    fold_set = {
        case.spec.fold for case in cases if case.spec.phase == "exploration"
    }
    require(None not in fold_set, "exploration family lacks a closure fold")
    folds = sorted(fold for fold in fold_set if fold is not None)
    splits: list[dict[str, Any]] = []
    for held_template in templates:
        for held_fold in folds:
            training = [
                case
                for case in closure_scope(cases, "exploration", role)
                if case.spec.template != held_template
                and case.spec.fold != held_fold
            ]
            validation = [
                case
                for case in closure_scope(cases, "exploration", role)
                if case.spec.template == held_template
                and case.spec.fold == held_fold
            ]
            observation_basis = None
            if relation_projected:
                observation_basis = fit_basis([
                    case.vectors["root_pullback_mixed"] for case in training
                ])
            metrics = closure_fit(
                training,
                validation,
                block_keys,
                dictionary_depth,
                observation_basis,
            )
            metrics["held_template"] = held_template
            metrics["held_fold"] = held_fold
            splits.append(metrics)
    require(len(splits) == 9, "closure cross-validation did not make nine splits")
    return summarize_closure_splits(splits)


def confirmation_closure(
    cases: list[TraceCase],
    role: str,
    block_keys: list[str],
    dictionary_depth: int,
    relation_projected: bool,
) -> dict[str, Any]:
    training = closure_scope(cases, "exploration", role)
    validation = closure_scope(cases, "confirmation", role)
    observation_basis = None
    if relation_projected:
        observation_basis = fit_basis([
            case.vectors["root_pullback_mixed"] for case in training
        ])
    return closure_fit(
        training,
        validation,
        block_keys,
        dictionary_depth,
        observation_basis,
    )


def closure_representations(layers: int) -> dict[str, list[str]]:
    attention = [f"layer_{layer}_attention" for layer in range(layers)]
    ffn = [f"layer_{layer}_ffn" for layer in range(layers)]
    representations: dict[str, list[str]] = {
        "scale_indexed_blocks": [
            key
            for layer in range(layers)
            for key in (f"layer_{layer}_attention", f"layer_{layer}_ffn")
        ],
        "attention_scales": attention,
        "ffn_scales": ffn,
    }
    for layer in range(layers):
        representations[f"layer_{layer}_blocks"] = [
            f"layer_{layer}_attention",
            f"layer_{layer}_ffn",
        ]
    for layer in range(layers):
        representations[f"layer_{layer}_attention"] = [f"layer_{layer}_attention"]
        representations[f"layer_{layer}_ffn"] = [f"layer_{layer}_ffn"]
    return representations


def git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def main() -> None:
    args = arguments()
    manifest = read_manifest(args.manifest)
    specs = expand_specs(manifest)
    cases = [read_trace(spec, args.traces) for spec in specs]
    exploration_pairs = paired_cases(cases, "exploration")
    confirmation_pairs = paired_cases(cases, "confirmation")
    require(len(exploration_pairs) == 36, "expected 36 exploration pairs")
    require(len(confirmation_pairs) == 8, "expected eight confirmation pairs")

    trace_types = {(case.positions, case.dim, case.layers) for case in cases}
    require(len(trace_types) == 1, "trace dimensions differ")
    pullback_depths = {case.pullback_depth for case in cases}
    require(len(pullback_depths) == 1, "trace pullback depths differ")
    recorded_pullback_depth = next(iter(pullback_depths))
    action_positions = {case.action_positions for case in cases}
    require(len(action_positions) == 1, "action positions are not matched globally")
    widths = {
        name: {case.vectors[name].size for case in cases} for name in REPRESENTATIONS
    }
    require(all(len(values) == 1 for values in widths.values()), "representation widths differ")
    expected_block_keys = set(closure_representations(cases[0].layers)["scale_indexed_blocks"])
    require(
        all(set(case.block_pullbacks) == expected_block_keys for case in cases),
        "block pullback keys differ",
    )

    local_pair_maximum = max(
        float(np.max(np.abs(controller.vectors["layer0_directed_qk"] - attractor.vectors["layer0_directed_qk"])))
        for controller, attractor in exploration_pairs + confirmation_pairs
    )
    local_pairs_exact = sum(
        np.array_equal(
            controller.vectors["layer0_directed_qk"],
            attractor.vectors["layer0_directed_qk"],
        )
        for controller, attractor in exploration_pairs + confirmation_pairs
    )

    descriptions = {
        "layer0_directed_qk": "both directed layer-0 QK cross tensors, concatenated",
        "root_pullback_mixed": "final post-RMS mixed pullback vector",
        "qk_transition_zip": "six layer-indexed QK delta-tau vectors, concatenated",
        "typed_transition_zip": "all ordered typed-boundary delta-tau vectors, concatenated",
    }
    exploration_results: dict[str, Any] = {}
    confirmation_results: dict[str, Any] = {}
    for name in REPRESENTATIONS:
        vector_of = lambda case, field=name: case.vectors[field]
        exploration_results[name] = cross_validate(cases, exploration_pairs, vector_of)
        confirmation_results[name] = confirmation(cases, confirmation_pairs, vector_of)

    qk_by_layer: dict[str, Any] = {}
    layers = cases[0].layers
    for layer in range(layers):
        vector_of = lambda case, index=layer: case.qk_layers[index]
        qk_by_layer[str(layer)] = cross_validate(cases, exploration_pairs, vector_of)

    closure_fields = closure_representations(layers)
    closure_cross_validation: dict[str, Any] = {}
    closure_confirmation: dict[str, Any] = {}
    for mode, relation_projected in (
        ("ambient_coordinates", False),
        ("training_relation_projection", True),
    ):
        closure_cross_validation[mode] = {}
        closure_confirmation[mode] = {}
        for dictionary_depth in range(1, recorded_pullback_depth + 1):
            depth_key = f"depth_{dictionary_depth}"
            closure_cross_validation[mode][depth_key] = {}
            for name in ("scale_indexed_blocks", "attention_scales", "ffn_scales"):
                closure_cross_validation[mode][depth_key][name] = {
                    role: cross_validate_closure(
                        cases,
                        role,
                        closure_fields[name],
                        dictionary_depth,
                        relation_projected,
                    )
                    for role in ("controller", "attractor", "combined")
                }
            closure_confirmation[mode][depth_key] = {
                name: {
                    role: confirmation_closure(
                        cases,
                        role,
                        block_keys,
                        dictionary_depth,
                        relation_projected,
                    )
                    for role in ("controller", "attractor", "combined")
                }
                for name, block_keys in closure_fields.items()
            }
    pulled_scale_maximum = 0.0
    for case in cases:
        pulled = [
            case.block_pullbacks[key][1]
            for key in closure_fields["scale_indexed_blocks"]
        ]
        reference = pulled[0]
        pulled_scale_maximum = max(
            pulled_scale_maximum,
            *(float(np.max(np.abs(vector - reference))) for vector in pulled[1:]),
        )

    manifest_digest = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    result: dict[str, Any] = {
        "schema_version": 2,
        "artifact": "matched_grammatical_continuation_geometry",
        "semantics": {
            "vectors": "torsor differences retained at learned continuation boundaries",
            "basis": "uncentered SVD row span; no hidden-state origin is selected",
            "residual": "relative distance to a sampled span; diagnostic only",
            "decision": "strict held-out relative ordering; ties are not wins",
            "not_an_inference_reward": True,
        },
        "provenance": {
            "model": args.model_label,
            "trace_schema_version": 5,
            "evaluator_commit": args.evaluator_commit or git_head(),
            "manifest_sha256": manifest_digest,
            "trace_count": len(cases),
            "raw_vectors_embedded": False,
        },
        "validation": {
            "positions": cases[0].positions,
            "hidden_width": cases[0].dim,
            "layers": cases[0].layers,
            "recorded_pullback_depth": recorded_pullback_depth,
            "action_positions": list(next(iter(action_positions))),
            "maximum_typed_chain_output_l2_defect": max(
                case.maximum_typed_chain_output_l2_defect for case in cases
            ),
            "maximum_block_pullback_composition_l2_defect": max(
                case.maximum_block_pullback_composition_l2_defect for case in cases
            ),
            "maximum_pulled_mixed_difference_across_scales": pulled_scale_maximum,
            "maximum_llama2c_hidden_relative_defect": max(
                case.maximum_llama2c_hidden_relative_defect for case in cases
            ),
            "maximum_telescoping_absolute_defect": max(
                case.telescoping_maximum_absolute_defect for case in cases
            ),
            "maximum_qk_score_reconstruction_relative_defect": max(
                case.maximum_qk_score_reconstruction_relative_defect for case in cases
            ),
            "maximum_qk_cross_removed_root_relative": max(
                case.maximum_qk_cross_removed_root_relative for case in cases
            ),
            "maximum_qk_transition_identity_relative_defect": max(
                case.maximum_qk_transition_identity_relative_defect for case in cases
            ),
            "matched_layer0_qk_pairs_exact": local_pairs_exact,
            "matched_layer0_qk_pairs_total": len(exploration_pairs) + len(confirmation_pairs),
            "matched_layer0_qk_maximum_absolute_difference": local_pair_maximum,
        },
        "representations": {
            name: {"width": next(iter(widths[name])), "description": descriptions[name]}
            for name in REPRESENTATIONS
        },
        "exploration": {
            "protocol": "hold out one whole template and one disjoint lexical fold",
            "training_vectors_per_role_per_split": 16,
            "held_out_pairs": 36,
            "results": exploration_results,
            "qk_layer_results_post_hoc": qk_by_layer,
        },
        "confirmation": {
            "protocol": "fit all three exploration templates, then test unseen by-template and new lexemes",
            "training_vectors_per_role": 36,
            "held_out_pairs": 8,
            "results": confirmation_results,
        },
        "pullback_closure": {
            "semantics": {
                "input": "X[i]=(U_a-I)(U_b-I)k evaluated before each residual block",
                "pulled": "Y[i]=(U_a-I)(U_b-I)(k after F) for the real residual map F",
                "recursive_dictionary": "depth d uses [k,U_F k,...,U_F^(d-1)k] and tests its shift through U_F",
                "product": "attention and FFN pullbacks remain separate scale-indexed factors",
                "representation_defect": "distance of Y from the sampled function-value span of X",
                "descent_defect": "failure of kernel(X) to remain in kernel(Y)",
                "validation_prediction": "relative error of X_validation (X_fit pseudoinverse Y_fit)",
                "training_relation_projection": "optional root relation basis fitted only on each training split",
                "scope_of_action": "F is one repeated endomorphic residual block, not a token-extension action",
                "not_an_inference_reward": True,
            },
            "representations": {
                name: {
                    "blocks": block_keys,
                    "ambient_function_columns_per_generation": len(block_keys) * cases[0].dim,
                }
                for name, block_keys in closure_fields.items()
            },
            "exploration_cross_validation": closure_cross_validation,
            "confirmation": closure_confirmation,
        },
        "scope": [
            "finite evidence for continuation geometry, not a recovered global grammar",
            "residual-block closure does not establish or refute closure under grammatical token actions",
            "no residual or norm is used as a completion score",
            "no inference speedup follows from this diagnostic",
        ],
    }

    print("representation                  explore relation  explore nearest  confirm relation  confirm nearest")
    for name in REPRESENTATIONS:
        explore = exploration_results[name]
        confirm = confirmation_results[name]
        print(
            f"{name:31s} "
            f"{explore['matched_relation_basis']['wins']:2d}/{explore['matched_relation_basis']['total']:<2d}            "
            f"{explore['symmetric_nearest_basis']['wins']:2d}/{explore['symmetric_nearest_basis']['total']:<2d}              "
            f"{confirm['matched_relation_basis']['wins']:2d}/{confirm['matched_relation_basis']['total']:<2d}              "
            f"{confirm['symmetric_nearest_basis']['wins']:2d}/{confirm['symmetric_nearest_basis']['total']:<2d}"
        )
    print(
        "matched layer-0 QK tensors: "
        f"{local_pairs_exact}/{len(exploration_pairs) + len(confirmation_pairs)} exact, "
        f"max_abs={local_pair_maximum:.9g}"
    )
    print("pullback closure confirmation:")
    for mode in ("ambient_coordinates", "training_relation_projection"):
        print(f"  {mode}:")
        for dictionary_depth in range(1, recorded_pullback_depth + 1):
            depth_key = f"depth_{dictionary_depth}"
            for role in ("controller", "attractor", "combined"):
                metrics = closure_confirmation[mode][depth_key]["scale_indexed_blocks"][role]
                print(
                    f"    depth={dictionary_depth} scale_indexed_blocks "
                    f"{role:10s} rank={metrics['sampled_rank']:2d} "
                    f"descent={metrics['fit_descent_relative']:.8g} "
                    f"heldout={metrics['validation_prediction_relative']:.8g} "
                    f"identity={metrics['validation_identity_relative']:.8g}"
                )
        depth_key = f"depth_{recorded_pullback_depth}"
        for name in ("attention_scales", "ffn_scales"):
            for role in ("controller", "attractor", "combined"):
                metrics = closure_confirmation[mode][depth_key][name][role]
                print(
                    f"    depth={recorded_pullback_depth} {name:22s} "
                    f"{role:10s} rank={metrics['sampled_rank']:2d} "
                    f"descent={metrics['fit_descent_relative']:.8g} "
                    f"heldout={metrics['validation_prediction_relative']:.8g} "
                    f"identity={metrics['validation_identity_relative']:.8g}"
                )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

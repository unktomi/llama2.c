#!/usr/bin/env python3
"""Analyze edge demand, terminal behavior, and typed local action jets.

The primary edge object is a Mealy zip of pre-constructor token-indexed codata
and constructor injections. Independent feature actions probe which product
projections each constructor contrast demands. Edge observations remain
separate by position and never become a scalar completion score.

The terminal object is a context x extension x Moebius-subset x token-contrast
tensor.  Its primary Hankel slice is Delta_A Delta_B q(Cx).  Both contexts and
whole extension families are held out as a missing matrix block.

The local diagnostic keeps typed boundaries separate and asks whether the next
AB coefficient is predictable from four equally capacity-controlled feature
families: AB; carrier+AB; A+B+AB; and the complete carrier/A/B/AB jet.  Norms,
residuals, ranks, and labels remain diagnostics and never become completion
scores.
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

from analyze_grammar_behaviors import (
    StateSpec,
    expand_states,
    matched_pairs,
    numerical_row_rank,
)
from gather_grammar_behaviors import (
    DEFAULT_ACTIONS,
    ActionWord,
    expand_words,
    read_actions,
)
from gather_grammar_cubes import DEFAULT_OBSERVERS, DEFAULT_OUTPUT as DEFAULT_TRACES
from gather_grammar_relations import DEFAULT_MANIFEST, read_manifest


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULT = ROOT / "outputs" / "cps-grammar-cubes-analysis.json"
DEFAULT_COMPOSITE_TRACES = ROOT / "work_traces" / "grammar_cube_composites"
SUBSETS = ("carrier", "A", "B", "AB", "C", "AC", "BC", "ABC")
EDGE_SUBSETS = ("carrier", "A", "B", "AB")
FEATURES = {
    "AB": (3,),
    "carrier_AB": (0, 3),
    "A_B_AB": (1, 2, 3),
    "carrier_A_B_AB": (0, 1, 2, 3),
}
RANKS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)


@dataclass(frozen=True)
class LocalReference:
    fiber: str
    boundary_index: int
    layer: int
    phase: str
    boundary: str
    width: int
    offset: int
    count: int


@dataclass(frozen=True)
class EdgeObservation:
    fiber: str
    token_position: int
    predecessor_position: int
    corner_tokens: tuple[int, int, int, int]
    coefficients: np.ndarray


@dataclass
class CubeTrace:
    state: StateSpec
    extension: ActionWord
    path: Path
    jet_path: Path | None
    observer_tokens: tuple[int, ...]
    reference_token: int
    coefficients: dict[str, np.ndarray]
    edges: dict[str, list[EdgeObservation]]
    local: dict[str, list[LocalReference]]
    maximum_chain_defect: float
    maximum_local_inverse_defect: float
    maximum_edge_inverse_defect: float
    terminal_inverse_defect: float
    typed_boundaries: int
    maximum_hidden_relative_defect: float
    maximum_logit_relative_defect: float

    @property
    def stem(self) -> str:
        diagram = self.state.diagram.trace_name.removesuffix(".jsonl")
        return f"{diagram}--c-{self.extension.name}"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--actions", type=Path, default=DEFAULT_ACTIONS)
    parser.add_argument("--observers", type=Path, default=DEFAULT_OBSERVERS)
    parser.add_argument("--traces", type=Path, default=DEFAULT_TRACES)
    parser.add_argument(
        "--composite-traces",
        type=Path,
        default=DEFAULT_COMPOSITE_TRACES,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--model-label", default="Stories15M")
    parser.add_argument("--evaluator-commit")
    parser.add_argument(
        "--primitive-terminal-only",
        action="store_true",
        help="primitive traces retain edge and terminal observations but no local sidecars",
    )
    parser.add_argument(
        "--edge-company-only",
        action="store_true",
        help="analyze constructor/codata edge evaluations without requiring composed traces",
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


def read_observers(path: Path) -> tuple[tuple[int, ...], dict[int, str]]:
    ids: list[int] = []
    labels: dict[int, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            token_field, separator, label = stripped.partition("\t")
            token = int(token_field.split(maxsplit=1)[0])
        except ValueError as error:
            raise SystemExit(f"{path}:{line_number}: invalid token ID") from error
        ids.append(token)
        labels[token] = label if separator else str(token)
    require(bool(ids) and len(ids) == len(set(ids)), "observer IDs are empty or duplicated")
    return tuple(ids), labels


def trace_stem(state: StateSpec, extension: ActionWord) -> str:
    diagram = state.diagram.trace_name.removesuffix(".jsonl")
    return f"{diagram}--c-{extension.name}"


def read_cube_trace(
    state: StateSpec,
    extension: ActionWord,
    directory: Path,
    require_local_jets: bool = True,
) -> CubeTrace:
    stem = trace_stem(state, extension)
    path = directory / f"{stem}.jsonl"
    jet_path = directory / f"{stem}.f32"
    require(
        path.is_file() and (not require_local_jets or jet_path.is_file()),
        f"missing cube artifacts: {stem}",
    )
    retained: dict[str, list[dict[str, Any]]] = defaultdict(list)
    allowed = {
        "grammatical_cube_meta",
        "grammatical_cube_context",
        "grammatical_cube_local_jet",
        "grammatical_cube_edge_zip",
        "grammatical_cube_terminal",
        "grammatical_cube_check",
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

    meta = one("grammatical_cube_meta")
    terminal = one("grammatical_cube_terminal")
    check = one("grammatical_cube_check")
    require(meta.get("schema_version") == 2, f"{path}: wrong cube schema")
    require(
        meta.get("semantics")
        == "carrier_conditioned_action_jet_with_mealy_edge_zip",
        f"{path}: wrong cube semantics",
    )
    require(meta.get("edge_observation_zip_retained") is True, f"{path}: edge zip absent")
    require(meta.get("edge_observations_folded") is False, f"{path}: edge zip was folded")
    require(meta.get("terminal_probabilities_used") is False, f"{path}: probabilities entered observer")
    require(meta.get("scalar_completion_reward_used") is False, f"{path}: scalar reward entered observer")
    require(meta.get("local_c_difference_defined") is False, f"{path}: ill-typed local C difference")
    require(meta.get("norms_are_diagnostics_not_scores") is True, f"{path}: unsafe norm semantics")
    observer_tokens = tuple(int(value) for value in meta["observer_tokens"])
    width = len(observer_tokens)
    require(width > 0 and int(terminal["observer_width"]) == width, f"{path}: observer width differs")
    require(tuple(terminal["mobius_subsets"]) == SUBSETS, f"{path}: terminal subset order differs")
    coefficients = {
        name: np.asarray(terminal["coefficients"][name], dtype=np.float64)
        for name in SUBSETS
    }
    require(
        all(vector.shape == (width,) and np.all(np.isfinite(vector)) for vector in coefficients.values()),
        f"{path}: invalid terminal coefficients",
    )
    require(len(retained["grammatical_cube_context"]) == 8, f"{path}: cube lacks corners")

    edges: dict[str, list[EdgeObservation]] = defaultdict(list)
    for row in retained["grammatical_cube_edge_zip"]:
        require(int(row["observer_width"]) == width, f"{path}: edge width differs")
        require(tuple(row["mobius_subsets"]) == EDGE_SUBSETS, f"{path}: edge subset order differs")
        tokens = tuple(int(token) for token in row["corner_token_ids"])
        require(len(tokens) == 4, f"{path}: edge corner token count differs")
        values = np.stack(
            [np.asarray(row["coefficients"][name], dtype=np.float64) for name in EDGE_SUBSETS]
        )
        require(values.shape == (4, width) and np.all(np.isfinite(values)), f"{path}: invalid edge coefficients")
        edge = EdgeObservation(
            fiber=str(row["fiber"]),
            token_position=int(row["token_position"]),
            predecessor_position=int(row["predecessor_position"]),
            corner_tokens=tokens,  # type: ignore[arg-type]
            coefficients=values,
        )
        require(edge.predecessor_position == edge.token_position - 1, f"{path}: edge predecessor differs")
        edges[edge.fiber].append(edge)
    require(set(edges) == {"without_C", "with_C"}, f"{path}: edge fibers differ")
    expected_positions = {
        "without_C": int(meta["base_positions"]),
        "with_C": int(meta["extended_positions"]),
    }
    for fiber, rows in edges.items():
        rows.sort(key=lambda row: row.token_position)
        require(
            [row.token_position for row in rows]
            == list(range(1, expected_positions[fiber])),
            f"{path}: edge positions differ for {fiber}",
        )

    local: dict[str, list[LocalReference]] = defaultdict(list)
    maximum_end = 0
    for row in retained["grammatical_cube_local_jet"]:
        reference = LocalReference(
            fiber=str(row["fiber"]),
            boundary_index=int(row["boundary_index"]),
            layer=int(row["layer"]),
            phase=str(row["phase"]),
            boundary=str(row["boundary"]),
            width=int(row["local_width"]),
            offset=int(row["binary_byte_offset"]),
            count=int(row["binary_float32_count"]),
        )
        require(reference.count == 4 * reference.width, f"{path}: local shape differs")
        local[reference.fiber].append(reference)
        maximum_end = max(maximum_end, reference.offset + 4 * reference.count)
    typed_boundaries = int(check["typed_boundaries"])
    require(typed_boundaries > 0, f"{path}: check boundary count is not positive")
    if require_local_jets:
        require(
            meta.get("local_jets_retained") in (None, True),
            f"{path}: local jets were not retained",
        )
        require(set(local) == {"without_C", "with_C"}, f"{path}: local fibers differ")
        for fiber, rows in local.items():
            rows.sort(key=lambda row: row.boundary_index)
            require(
                len(rows) == typed_boundaries,
                f"{path}: {fiber} boundary count differs",
            )
            require(
                [row.boundary_index for row in rows]
                == list(range(typed_boundaries)),
                f"{path}: unordered boundaries",
            )
        for left, right in zip(local["without_C"], local["with_C"]):
            require(
                (left.boundary_index, left.layer, left.phase, left.boundary)
                == (right.boundary_index, right.layer, right.phase, right.boundary),
                f"{path}: C fiber boundaries do not align",
            )
        require(maximum_end == jet_path.stat().st_size, f"{path}: jet sidecar size differs")
    else:
        require(meta.get("local_jets_retained") is False, f"{path}: expected terminal-only cube")
        require(not local and maximum_end == 0, f"{path}: terminal-only cube has local jets")
    require(float(check["maximum_typed_chain_output_l2_defect"]) == 0.0, f"{path}: typed chain defect")
    for field in (
        "maximum_local_mobius_inverse_absolute_defect",
        "maximum_edge_mobius_inverse_absolute_defect",
        "terminal_mobius_inverse_absolute_defect",
        "maximum_stock_hidden_relative_defect",
        "maximum_stock_logit_contrast_relative_defect",
    ):
        require(np.isfinite(float(check[field])), f"{path}: nonfinite {field}")
    return CubeTrace(
        state=state,
        extension=extension,
        path=path,
        jet_path=jet_path if require_local_jets else None,
        observer_tokens=observer_tokens,
        reference_token=int(meta["observer_reference_token"]),
        coefficients=coefficients,
        edges=dict(edges),
        local=dict(local),
        maximum_chain_defect=float(check["maximum_typed_chain_output_l2_defect"]),
        maximum_local_inverse_defect=float(check["maximum_local_mobius_inverse_absolute_defect"]),
        maximum_edge_inverse_defect=float(check["maximum_edge_mobius_inverse_absolute_defect"]),
        terminal_inverse_defect=float(check["terminal_mobius_inverse_absolute_defect"]),
        typed_boundaries=typed_boundaries,
        maximum_hidden_relative_defect=float(check["maximum_stock_hidden_relative_defect"]),
        maximum_logit_relative_defect=float(check["maximum_stock_logit_contrast_relative_defect"]),
    )


def load_local(trace: CubeTrace, fiber: str, boundary: int) -> np.ndarray:
    require(trace.jet_path is not None, f"{trace.path}: local jet was not retained")
    reference = trace.local[fiber][boundary]
    mapped = np.memmap(
        trace.jet_path,
        dtype=np.float32,
        mode="r",
        offset=reference.offset,
        shape=(4, reference.width),
    )
    return np.asarray(mapped, dtype=np.float64)


def row_splits(states: list[StateSpec], confirmation: bool) -> list[dict[str, Any]]:
    index = {state: position for position, state in enumerate(states)}
    exploration = [state for state in states if state.diagram.phase == "exploration"]
    if confirmation:
        held = [state for state in states if state.diagram.phase == "confirmation"]
        return [
            {
                "name": "frozen_confirmation",
                "train": np.asarray([index[state] for state in exploration], dtype=np.int64),
                "held": np.asarray([index[state] for state in held], dtype=np.int64),
                "held_states": held,
            }
        ]
    templates = sorted({state.diagram.template for state in exploration})
    folds = sorted({state.fold for state in exploration if state.fold is not None})
    splits: list[dict[str, Any]] = []
    for template in templates:
        for fold in folds:
            train = [
                state
                for state in exploration
                if state.diagram.template != template and state.fold != fold
            ]
            held = [
                state
                for state in exploration
                if state.diagram.template == template and state.fold == fold
            ]
            require(len(train) == 32 and len(held) == 8, "unexpected exploration row split")
            splits.append(
                {
                    "name": f"{template}/fold_{fold}",
                    "held_template": template,
                    "held_fold": fold,
                    "train": np.asarray([index[state] for state in train], dtype=np.int64),
                    "held": np.asarray([index[state] for state in held], dtype=np.int64),
                    "held_states": held,
                }
            )
    return splits


def paired_differences(values: np.ndarray, held_states: list[StateSpec]) -> np.ndarray:
    position = {state: index for index, state in enumerate(held_states)}
    pairs = matched_pairs(held_states)
    require(2 * len(pairs) == len(held_states), "held rows are not complete pairs")
    return np.stack(
        [values[position[controller]] - values[position[attractor]] for controller, attractor in pairs]
    )


def relative_frobenius(actual: np.ndarray, predicted: np.ndarray) -> float:
    norm = float(np.linalg.norm(actual))
    error = float(np.linalg.norm(predicted - actual))
    if norm == 0.0:
        require(error == 0.0, "zero target has a nonzero prediction")
        return 0.0
    return error / norm


def distinction_diagnostics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float | bool | None]:
    actual_norm = float(np.linalg.norm(actual))
    predicted_norm = float(np.linalg.norm(predicted))
    if actual_norm == 0.0:
        return {
            "future_distinction_relative_error": (
                0.0 if predicted_norm == 0.0 else None
            ),
            "future_distinction_norm_ratio": None,
            "future_distinction_cosine": None,
            "future_distinction_target_is_zero": True,
            "future_distinction_prediction_absolute_norm": predicted_norm,
        }
    cosine = 0.0
    if predicted_norm > 0.0:
        cosine = float(np.sum(predicted * actual)) / (predicted_norm * actual_norm)
    return {
        "future_distinction_relative_error": relative_frobenius(actual, predicted),
        "future_distinction_norm_ratio": predicted_norm / actual_norm,
        "future_distinction_cosine": cosine,
        "future_distinction_target_is_zero": False,
        "future_distinction_prediction_absolute_norm": predicted_norm,
    }


def rank_values(maximum: int) -> list[int]:
    if maximum == 0:
        return [0]
    return sorted({rank for rank in (*RANKS, maximum) if rank <= maximum})


def block_prediction_curve(
    matrix: np.ndarray,
    train_rows: np.ndarray,
    held_rows: np.ndarray,
    held_states: list[StateSpec],
    train_columns: np.ndarray,
    held_columns: np.ndarray,
) -> dict[str, Any]:
    a = matrix[np.ix_(train_rows, train_columns)]
    b = matrix[np.ix_(train_rows, held_columns)]
    c = matrix[np.ix_(held_rows, train_columns)]
    d = matrix[np.ix_(held_rows, held_columns)]
    u, singular, vh = np.linalg.svd(a, full_matrices=False)
    tolerance = float(max(a.shape) * np.finfo(np.float64).eps * singular[0])
    sampled_rank = int(np.count_nonzero(singular > tolerance))
    mean_prediction = np.repeat(np.mean(b, axis=0, keepdims=True), len(held_rows), axis=0)
    distances = np.sum((c[:, None, :] - a[None, :, :]) ** 2, axis=2)
    nearest = np.argmin(distances, axis=1)
    nearest_prediction = b[nearest]
    true_distinction = paired_differences(d, held_states)
    curve: dict[str, Any] = {}
    for rank in rank_values(sampled_rank):
        row_coordinates = u[:, :rank] * singular[:rank]
        held_loadings = np.linalg.lstsq(row_coordinates, b, rcond=None)[0]
        held_row_coordinates = c @ vh[:rank].T
        predicted = held_row_coordinates @ held_loadings
        predicted_distinction = paired_differences(predicted, held_states)
        curve[str(rank)] = {
            "rank": rank,
            "held_block_relative_error": relative_frobenius(d, predicted),
            **distinction_diagnostics(true_distinction, predicted_distinction),
        }
    return {
        "training_shape": list(a.shape),
        "held_shape": list(d.shape),
        "training_rank": sampled_rank,
        "training_rank_is_zero": sampled_rank == 0,
        "training_rank_saturated": sampled_rank == min(a.shape),
        "rank_tolerance": tolerance,
        "mean_baseline_relative_error": relative_frobenius(d, mean_prediction),
        "nearest_source_baseline_relative_error": relative_frobenius(d, nearest_prediction),
        "nearest_source_distance": {
            "minimum": float(np.sqrt(np.min(distances, axis=1)).min()),
            "median": float(np.median(np.sqrt(np.min(distances, axis=1)))),
            "maximum": float(np.sqrt(np.min(distances, axis=1)).max()),
        },
        "curve": curve,
    }


def action_column_indices(
    extensions: list[ActionWord],
    block_width: int,
    held_family: str,
) -> tuple[np.ndarray, np.ndarray]:
    train: list[int] = []
    held: list[int] = []
    for action_index, extension in enumerate(extensions):
        destination = held if extension.family == held_family else train
        destination.extend(
            range(action_index * block_width, (action_index + 1) * block_width)
        )
    require(bool(train) and bool(held), f"extension family {held_family} cannot be held out")
    return np.asarray(train, dtype=np.int64), np.asarray(held, dtype=np.int64)


def summarize_curves(results: list[dict[str, Any]]) -> dict[str, Any]:
    rank_keys = sorted(
        {key for result in results for key in result["curve"]},
        key=int,
    )
    curve: dict[str, Any] = {}
    for key in rank_keys:
        available = [result["curve"][key] for result in results if key in result["curve"]]
        curve[key] = {"splits": len(available)}
        for field in (
            "held_block_relative_error",
            "future_distinction_relative_error",
            "future_distinction_norm_ratio",
            "future_distinction_cosine",
        ):
            values = [row[field] for row in available if row[field] is not None]
            curve[key][field] = (
                {
                    "defined_splits": len(values),
                    "minimum": min(values),
                    "mean": float(np.mean(values)),
                    "maximum": max(values),
                }
                if values
                else {
                    "defined_splits": 0,
                    "minimum": None,
                    "mean": None,
                    "maximum": None,
                }
            )
    return {
        "split_count": len(results),
        "training_rank_range": [
            min(result["training_rank"] for result in results),
            max(result["training_rank"] for result in results),
        ],
        "row_rank_saturated_splits": sum(result["training_rank_saturated"] for result in results),
        "mean_baseline_relative_error": float(
            np.mean([result["mean_baseline_relative_error"] for result in results])
        ),
        "nearest_source_baseline_relative_error": float(
            np.mean([result["nearest_source_baseline_relative_error"] for result in results])
        ),
        "curve": curve,
    }


def double_holdout_analysis(
    matrix: np.ndarray,
    states: list[StateSpec],
    extensions: list[ActionWord],
    block_width: int,
) -> dict[str, Any]:
    extension_families = sorted({extension.family for extension in extensions})
    exploration_results: list[dict[str, Any]] = []
    confirmation_results: list[dict[str, Any]] = []
    confirmation_by_family: dict[str, Any] = {}
    for family in extension_families:
        train_columns, held_columns = action_column_indices(
            extensions,
            block_width,
            family,
        )
        for split in row_splits(states, confirmation=False):
            result = block_prediction_curve(
                matrix,
                split["train"],
                split["held"],
                split["held_states"],
                train_columns,
                held_columns,
            )
            result["row_split"] = split["name"]
            result["held_extension_family"] = family
            exploration_results.append(result)
        split = row_splits(states, confirmation=True)[0]
        result = block_prediction_curve(
            matrix,
            split["train"],
            split["held"],
            split["held_states"],
            train_columns,
            held_columns,
        )
        result["row_split"] = split["name"]
        result["held_extension_family"] = family
        confirmation_results.append(result)
        confirmation_by_family[family] = result
    return {
        "semantics": "fit train-row x train-extension block; learn held-extension columns on train rows; infer held rows from train extensions; predict the unseen row x extension block",
        "extension_families": extension_families,
        "exploration": summarize_curves(exploration_results),
        "confirmation": summarize_curves(confirmation_results),
        "confirmation_by_extension_family": confirmation_by_family,
    }


def regression_curve(
    features: np.ndarray,
    targets: np.ndarray,
    train: np.ndarray,
    held: np.ndarray,
    held_states: list[StateSpec],
) -> dict[str, Any]:
    x_train = features[train]
    x_held = features[held]
    y_train = targets[train]
    y_held = targets[held]
    x_origin = np.mean(x_train, axis=0, keepdims=True)
    y_origin = np.mean(y_train, axis=0, keepdims=True)
    x_train_centered = x_train - x_origin
    x_held_centered = x_held - x_origin
    y_train_centered = y_train - y_origin
    u, singular, vh = np.linalg.svd(x_train_centered, full_matrices=False)
    tolerance = float(
        max(x_train_centered.shape) * np.finfo(np.float64).eps * singular[0]
    )
    sampled_rank = int(np.count_nonzero(singular > tolerance))
    distances = np.sum((x_held[:, None, :] - x_train[None, :, :]) ** 2, axis=2)
    nearest = np.argmin(distances, axis=1)
    nearest_prediction = y_train[nearest]
    true_distinction = paired_differences(y_held, held_states)
    curve: dict[str, Any] = {}
    for rank in rank_values(sampled_rank):
        if rank == 0:
            predicted = np.repeat(y_origin, len(y_held), axis=0)
        else:
            coordinates = (x_held_centered @ vh[:rank].T) / singular[:rank]
            predicted = y_origin + coordinates @ u[:, :rank].T @ y_train_centered
        predicted_distinction = paired_differences(predicted, held_states)
        curve[str(rank)] = {
            "rank": rank,
            "held_target_relative_error": relative_frobenius(y_held, predicted),
            **distinction_diagnostics(true_distinction, predicted_distinction),
        }
    return {
        "feature_width": features.shape[1],
        "target_width": targets.shape[1],
        "fit": "training-split affine map in centered coordinates",
        "training_rank": sampled_rank,
        "training_rank_is_zero": sampled_rank == 0,
        "training_rank_saturated": sampled_rank == len(train),
        "nearest_source_relative_error": relative_frobenius(y_held, nearest_prediction),
        "nearest_source_distance": {
            "minimum": float(np.sqrt(np.min(distances, axis=1)).min()),
            "median": float(np.median(np.sqrt(np.min(distances, axis=1)))),
            "maximum": float(np.sqrt(np.min(distances, axis=1)).max()),
        },
        "curve": curve,
    }


def summarize_regressions(results: list[dict[str, Any]]) -> dict[str, Any]:
    rank_keys = sorted({key for result in results for key in result["curve"]}, key=int)
    curve: dict[str, Any] = {}
    for key in rank_keys:
        available = [result["curve"][key] for result in results if key in result["curve"]]
        curve[key] = {"splits": len(available)}
        for field in (
            "held_target_relative_error",
            "future_distinction_relative_error",
            "future_distinction_norm_ratio",
            "future_distinction_cosine",
        ):
            values = [row[field] for row in available if row[field] is not None]
            curve[key][field] = (
                {
                    "defined_splits": len(values),
                    "minimum": min(values),
                    "mean": float(np.mean(values)),
                    "maximum": max(values),
                }
                if values
                else {
                    "defined_splits": 0,
                    "minimum": None,
                    "mean": None,
                    "maximum": None,
                }
            )
    return {
        "split_count": len(results),
        "training_rank_range": [
            min(result["training_rank"] for result in results),
            max(result["training_rank"] for result in results),
        ],
        "row_rank_saturated_splits": sum(result["training_rank_saturated"] for result in results),
        "nearest_source_relative_error_mean": float(
            np.mean([result["nearest_source_relative_error"] for result in results])
        ),
        "curve": curve,
    }


def terminal_action_jet_ablation(
    tensor: np.ndarray,
    states: list[StateSpec],
    extensions: list[ActionWord],
) -> dict[str, Any]:
    base = tensor[:, 0, :4, :]
    maximum_base_difference = float(np.max(np.abs(tensor[:, :, :4, :] - base[:, None, :, :])))
    require(maximum_base_difference == 0.0, "base terminal jet differs across extensions")
    results: dict[str, Any] = {}
    for feature_name, masks in FEATURES.items():
        features = np.concatenate([base[:, mask, :] for mask in masks], axis=1)
        exploration_rows: list[dict[str, Any]] = []
        confirmation_rows: list[dict[str, Any]] = []
        per_extension: dict[str, Any] = {}
        for action_index, extension in enumerate(extensions):
            targets = tensor[:, action_index, 3, :] + tensor[:, action_index, 7, :]
            for split in row_splits(states, confirmation=False):
                result = regression_curve(
                    features,
                    targets,
                    split["train"],
                    split["held"],
                    split["held_states"],
                )
                result["row_split"] = split["name"]
                result["extension"] = extension.name
                exploration_rows.append(result)
            split = row_splits(states, confirmation=True)[0]
            result = regression_curve(
                features,
                targets,
                split["train"],
                split["held"],
                split["held_states"],
            )
            result["row_split"] = split["name"]
            result["extension"] = extension.name
            confirmation_rows.append(result)
            per_extension[extension.name] = result
        results[feature_name] = {
            "feature_masks": list(masks),
            "exploration": summarize_regressions(exploration_rows),
            "confirmation": summarize_regressions(confirmation_rows),
            "confirmation_by_extension": per_extension,
        }
    return {
        "base_jet_maximum_difference_across_extensions": maximum_base_difference,
        "feature_sets": results,
    }


def normalized_condition_coordinates(
    conditions: np.ndarray,
    train: np.ndarray,
) -> tuple[np.ndarray, dict[str, list[float]]]:
    require(
        conditions.ndim == 3 and conditions.shape[1] == 3,
        "atlas conditions must be carrier/A/B blocks",
    )
    blocks: list[np.ndarray] = []
    origins: list[float] = []
    scales: list[float] = []
    for block in range(conditions.shape[1]):
        origin = np.mean(conditions[train, block], axis=0, keepdims=True)
        centered = conditions[:, block] - origin
        scale = float(np.sqrt(np.mean(centered[train] * centered[train])))
        if scale == 0.0:
            scale = 1.0
        blocks.append(centered / scale)
        origins.append(float(np.linalg.norm(origin)))
        scales.append(scale)
    return np.concatenate(blocks, axis=1), {
        "training_origin_norms": origins,
        "training_rms_scales": scales,
    }


def deterministic_kmeans(
    values: np.ndarray,
    count: int,
    maximum_iterations: int = 100,
) -> tuple[np.ndarray, np.ndarray, int]:
    require(values.ndim == 2 and 0 < count <= len(values), "invalid atlas chart count")
    mean = np.mean(values, axis=0)
    first = int(np.argmin(np.sum((values - mean) ** 2, axis=1)))
    chosen = [first]
    nearest = np.sum((values - values[first]) ** 2, axis=1)
    while len(chosen) < count:
        available = np.asarray(
            [index for index in range(len(values)) if index not in chosen],
            dtype=np.int64,
        )
        candidate = int(available[np.argmax(nearest[available])])
        chosen.append(candidate)
        nearest = np.minimum(
            nearest,
            np.sum((values - values[candidate]) ** 2, axis=1),
        )
    centers = values[np.asarray(chosen)].copy()
    labels = np.full(len(values), -1, dtype=np.int64)
    for iteration in range(1, maximum_iterations + 1):
        distances = np.sum((values[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        next_labels = np.argmin(distances, axis=1)
        if np.array_equal(next_labels, labels):
            return labels, centers, iteration - 1
        labels = next_labels
        for chart in range(count):
            members = values[labels == chart]
            if len(members) == 0:
                assigned_distance = distances[np.arange(len(values)), labels]
                replacement = int(np.argmax(assigned_distance))
                labels[replacement] = chart
                members = values[[replacement]]
            centers[chart] = np.mean(members, axis=0)
    return labels, centers, maximum_iterations


def affine_rank_prediction(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    held_features: np.ndarray,
    rank_cap: int,
) -> tuple[np.ndarray, int]:
    x_origin = np.mean(train_features, axis=0, keepdims=True)
    y_origin = np.mean(train_targets, axis=0, keepdims=True)
    x_centered = train_features - x_origin
    y_centered = train_targets - y_origin
    if len(train_features) == 1:
        return np.repeat(y_origin, len(held_features), axis=0), 0
    u, singular, vh = np.linalg.svd(x_centered, full_matrices=False)
    tolerance = float(
        max(x_centered.shape) * np.finfo(np.float64).eps * singular[0]
    )
    available = int(np.count_nonzero(singular > tolerance))
    rank = min(rank_cap, available)
    if rank == 0:
        return np.repeat(y_origin, len(held_features), axis=0), 0
    coordinates = ((held_features - x_origin) @ vh[:rank].T) / singular[:rank]
    predicted = y_origin + coordinates @ u[:, :rank].T @ y_centered
    return predicted, rank


def finite_atlas_prediction(
    features: np.ndarray,
    conditions: np.ndarray,
    targets: np.ndarray,
    train: np.ndarray,
    held: np.ndarray,
    chart_count: int,
    rank_cap: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    condition_coordinates, normalization = normalized_condition_coordinates(
        conditions,
        train,
    )
    train_labels, centers, iterations = deterministic_kmeans(
        condition_coordinates[train],
        chart_count,
    )
    held_distances = np.sum(
        (
            condition_coordinates[held, None, :]
            - centers[None, :, :]
        )
        ** 2,
        axis=2,
    )
    held_labels = np.argmin(held_distances, axis=1)
    predicted = np.empty_like(targets[held])
    effective_ranks: list[int] = []
    train_sizes: list[int] = []
    held_sizes: list[int] = []
    for chart in range(chart_count):
        chart_train_local = np.flatnonzero(train_labels == chart)
        chart_held_local = np.flatnonzero(held_labels == chart)
        train_sizes.append(len(chart_train_local))
        held_sizes.append(len(chart_held_local))
        if len(chart_held_local) == 0:
            effective_ranks.append(0)
            continue
        chart_prediction, effective_rank = affine_rank_prediction(
            features[train[chart_train_local]],
            targets[train[chart_train_local]],
            features[held[chart_held_local]],
            rank_cap,
        )
        predicted[chart_held_local] = chart_prediction
        effective_ranks.append(effective_rank)
    return predicted, {
        "chart_count": chart_count,
        "rank_cap_per_chart": rank_cap,
        "effective_rank_sum": sum(effective_ranks),
        "effective_ranks": effective_ranks,
        "training_chart_sizes": train_sizes,
        "held_chart_sizes": held_sizes,
        "lloyd_iterations": iterations,
        "condition_normalization": normalization,
    }


def finite_atlas_curve(
    features: np.ndarray,
    conditions: np.ndarray,
    targets: np.ndarray,
    split: dict[str, Any],
) -> dict[str, Any]:
    true_targets = targets[split["held"]]
    true_distinction = paired_differences(true_targets, split["held_states"])
    curve: dict[str, Any] = {}
    for chart_count in (1, 2, 4, 8):
        for rank_cap in (1, 2, 4, 8):
            predicted, metadata = finite_atlas_prediction(
                features,
                conditions,
                targets,
                split["train"],
                split["held"],
                chart_count,
                rank_cap,
            )
            predicted_distinction = paired_differences(
                predicted,
                split["held_states"],
            )
            key = f"charts_{chart_count}_rank_{rank_cap}"
            curve[key] = {
                **metadata,
                "held_target_relative_error": relative_frobenius(
                    true_targets,
                    predicted,
                ),
                **distinction_diagnostics(
                    true_distinction,
                    predicted_distinction,
                ),
            }
    return curve


def summarize_atlas_curves(results: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted(results[0])
    curve: dict[str, Any] = {}
    for key in keys:
        rows = [result[key] for result in results]
        curve[key] = {
            "splits": len(rows),
            "effective_rank_sum_mean": float(
                np.mean([row["effective_rank_sum"] for row in rows])
            ),
        }
        for field in (
            "held_target_relative_error",
            "future_distinction_relative_error",
            "future_distinction_norm_ratio",
            "future_distinction_cosine",
        ):
            values = [row[field] for row in rows if row[field] is not None]
            curve[key][field] = {
                "defined_splits": len(values),
                "minimum": min(values) if values else None,
                "mean": float(np.mean(values)) if values else None,
                "maximum": max(values) if values else None,
            }
    return {"split_count": len(results), "curve": curve}


def terminal_finite_atlas_analysis(
    tensor: np.ndarray,
    states: list[StateSpec],
    extensions: list[ActionWord],
) -> dict[str, Any]:
    source = tensor[:, 0, :4, :]
    conditions = source[:, :3, :]
    feature_sets = {
        "AB_only": source[:, 3, :],
        "complete_action_jet": source.reshape(len(states), -1),
    }
    result: dict[str, Any] = {}
    for feature_name, features in feature_sets.items():
        exploration: list[dict[str, Any]] = []
        confirmation: list[dict[str, Any]] = []
        confirmation_by_extension: dict[str, Any] = {}
        for action_index, extension in enumerate(extensions):
            targets = tensor[:, action_index, 3, :] + tensor[:, action_index, 7, :]
            for split in row_splits(states, confirmation=False):
                exploration.append(
                    finite_atlas_curve(features, conditions, targets, split)
                )
            frozen = row_splits(states, confirmation=True)[0]
            frozen_curve = finite_atlas_curve(
                features,
                conditions,
                targets,
                frozen,
            )
            confirmation.append(frozen_curve)
            confirmation_by_extension[extension.name] = frozen_curve
        result[feature_name] = {
            "exploration": summarize_atlas_curves(exploration),
            "confirmation": summarize_atlas_curves(confirmation),
            "confirmation_by_extension": confirmation_by_extension,
        }
    return {
        "semantics": "deterministic charts from centered carrier/A/B geometry; rank-controlled affine transport within each chart",
        "chart_selection_uses_role_labels": False,
        "curve_selected": False,
        "feature_sets": result,
    }


def local_transition_ablation(
    traces: dict[tuple[StateSpec, str], CubeTrace],
    states: list[StateSpec],
    reference_extension: ActionWord,
) -> dict[str, Any]:
    selected = [traces[(state, reference_extension.name)] for state in states]
    boundaries = selected[0].local["without_C"]
    require(
        all(
            [
                (row.layer, row.phase, row.boundary, row.width)
                for row in trace.local["without_C"]
            ]
            == [
                (row.layer, row.phase, row.boundary, row.width)
                for row in boundaries
            ]
            for trace in selected
        ),
        "base local boundary types differ",
    )
    exploration_splits = row_splits(states, confirmation=False)
    confirmation_split = row_splits(states, confirmation=True)[0]
    transitions: list[dict[str, Any]] = []
    feature_wins = {name: 0 for name in FEATURES}
    for boundary_index in range(len(boundaries) - 1):
        source = np.stack(
            [load_local(trace, "without_C", boundary_index) for trace in selected]
        )
        target = np.stack(
            [load_local(trace, "without_C", boundary_index + 1)[3] for trace in selected]
        )
        feature_results: dict[str, Any] = {}
        full_rank_errors: dict[str, float] = {}
        for feature_name, masks in FEATURES.items():
            features = np.concatenate([source[:, mask, :] for mask in masks], axis=1)
            exploration_rows = [
                regression_curve(
                    features,
                    target,
                    split["train"],
                    split["held"],
                    split["held_states"],
                )
                for split in exploration_splits
            ]
            confirmation = regression_curve(
                features,
                target,
                confirmation_split["train"],
                confirmation_split["held"],
                confirmation_split["held_states"],
            )
            full_rank = str(confirmation["training_rank"])
            full_rank_errors[feature_name] = confirmation["curve"][full_rank][
                "held_target_relative_error"
            ]
            feature_results[feature_name] = {
                "exploration": summarize_regressions(exploration_rows),
                "confirmation": confirmation,
            }
        best = min(full_rank_errors, key=full_rank_errors.get)
        feature_wins[best] += 1
        left = boundaries[boundary_index]
        right = boundaries[boundary_index + 1]
        transitions.append(
            {
                "from_boundary_index": boundary_index,
                "to_boundary_index": boundary_index + 1,
                "from": {
                    "layer": left.layer,
                    "phase": left.phase,
                    "boundary": left.boundary,
                    "width": left.width,
                },
                "to": {
                    "layer": right.layer,
                    "phase": right.phase,
                    "boundary": right.boundary,
                    "width": right.width,
                },
                "full_rank_confirmation_error": full_rank_errors,
                "full_rank_best_feature": best,
                "features": feature_results,
            }
        )
    aggregate: dict[str, Any] = {}
    for feature_name in FEATURES:
        values = [row["full_rank_confirmation_error"][feature_name] for row in transitions]
        aggregate[feature_name] = {
            "best_transition_count": feature_wins[feature_name],
            "error_minimum": min(values),
            "error_median": float(np.median(values)),
            "error_mean": float(np.mean(values)),
            "error_maximum": max(values),
        }
    return {
        "fiber": "without_C",
        "reference_extension_file": reference_extension.name,
        "semantics": "predict the next typed boundary's complete AB coefficient from equally rank-controlled current-boundary jet feature sets",
        "transition_count": len(transitions),
        "aggregate": aggregate,
        "transitions": transitions,
    }


def tensor_mode_ranks(tensor: np.ndarray) -> dict[str, Any]:
    names = ("context", "extension", "mobius_subset", "observer")
    result: dict[str, Any] = {}
    for mode, name in enumerate(names):
        matrix = np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)
        result[name] = numerical_row_rank(matrix)
    return result


def pareto_ranks(summary: dict[str, Any]) -> list[int]:
    points: list[tuple[int, float, float]] = []
    for key, row in summary["curve"].items():
        points.append(
            (
                int(key),
                row["held_block_relative_error"]["mean"],
                row["future_distinction_relative_error"]["mean"],
            )
        )
    nondominated: list[int] = []
    for rank, transition, distinction in points:
        dominated = any(
            other_transition <= transition
            and other_distinction <= distinction
            and (other_transition < transition or other_distinction < distinction)
            for other_rank, other_transition, other_distinction in points
            if other_rank != rank
        )
        if not dominated:
            nondominated.append(rank)
    return sorted(nondominated)


def exact_signature_classes(matrix: np.ndarray) -> dict[str, Any]:
    classes: dict[bytes, list[int]] = defaultdict(list)
    for index, row in enumerate(matrix):
        classes[np.ascontiguousarray(row).tobytes()].append(index)
    sizes = sorted((len(group) for group in classes.values()), reverse=True)
    return {
        "class_count": len(classes),
        "non_singleton_class_count": sum(size > 1 for size in sizes),
        "largest_class_size": max(sizes),
        "class_sizes": sizes,
    }


class UnionFind:
    def __init__(self, count: int) -> None:
        self.parent = list(range(count))
        self.size = [1] * count

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]

    def labels(self) -> np.ndarray:
        roots = [self.find(index) for index in range(len(self.parent))]
        identifiers: dict[int, int] = {}
        return np.asarray(
            [identifiers.setdefault(root, len(identifiers)) for root in roots],
            dtype=np.int64,
        )


def relative_distance_matrix(vectors: np.ndarray) -> np.ndarray:
    require(vectors.ndim == 2 and len(vectors) > 0, "distance vectors are empty")
    squares = np.sum(vectors * vectors, axis=1)
    distances_squared = np.maximum(
        squares[:, None] + squares[None, :] - 2.0 * (vectors @ vectors.T),
        0.0,
    )
    scale = np.maximum(np.sqrt(squares)[:, None], np.sqrt(squares)[None, :])
    scale = np.maximum(scale, np.finfo(np.float64).tiny)
    result = np.sqrt(distances_squared) / scale
    np.fill_diagonal(result, 0.0)
    false_zero_candidates = np.argwhere(np.triu(result == 0.0, k=1))
    for left, right in false_zero_candidates:
        if not np.array_equal(vectors[left], vectors[right]):
            direct = float(np.linalg.norm(vectors[left] - vectors[right]))
            direct /= max(
                float(np.linalg.norm(vectors[left])),
                float(np.linalg.norm(vectors[right])),
                np.finfo(np.float64).tiny,
            )
            require(direct > 0.0, "distinct observation vectors have zero distance")
            result[left, right] = direct
            result[right, left] = direct
    return result


def threshold_component_snapshots(
    distances: np.ndarray,
    thresholds: list[float],
) -> dict[float, np.ndarray]:
    require(
        distances.ndim == 2 and distances.shape[0] == distances.shape[1],
        "component distance matrix is not square",
    )
    count = distances.shape[0]
    upper = np.triu_indices(count, k=1)
    edge_distances = distances[upper]
    order = np.argsort(edge_distances, kind="stable")
    left = upper[0][order]
    right = upper[1][order]
    edge_distances = edge_distances[order]
    union = UnionFind(count)
    cursor = 0
    snapshots: dict[float, np.ndarray] = {}
    for threshold in sorted(thresholds):
        while cursor < len(edge_distances) and edge_distances[cursor] <= threshold:
            union.union(int(left[cursor]), int(right[cursor]))
            cursor += 1
        snapshots[threshold] = union.labels()
    return snapshots


def compress_keys(keys: list[tuple[Any, ...]]) -> tuple[np.ndarray, dict[tuple[Any, ...], int]]:
    identifiers: dict[tuple[Any, ...], int] = {}
    labels = np.asarray(
        [identifiers.setdefault(key, len(identifiers)) for key in keys],
        dtype=np.int64,
    )
    return labels, identifiers


def behavioral_tree(
    tensor: np.ndarray,
    states: list[StateSpec],
    primitives: list[ActionWord],
    composites: list[ActionWord],
    primitive_traces: dict[tuple[StateSpec, str], CubeTrace],
    composite_traces: dict[tuple[StateSpec, str], CubeTrace],
) -> tuple[dict[tuple[int, tuple[str, ...]], np.ndarray], list[str]]:
    composable = sorted(
        {
            factor
            for word in composites
            for factor in word.factors
        }
    )
    require(len(composable) == 4, "expected four composable primitive actions")
    primitive_names = {word.name for word in primitives}
    require(set(composable) <= primitive_names, "composite factor lacks primitive cube")
    require(
        {word.factors for word in composites}
        == {(left, right) for left in composable for right in composable},
        "composed action grid is incomplete",
    )
    primitive_index = {word.name: index for index, word in enumerate(primitives)}
    tree: dict[tuple[int, tuple[str, ...]], np.ndarray] = {}
    for state_index, state in enumerate(states):
        base = tensor[state_index, 0, 3]
        tree[(state_index, ())] = base.copy()
        for name in composable:
            trace = primitive_traces[(state, name)]
            from_tensor = tensor[state_index, primitive_index[name], 3] + tensor[
                state_index,
                primitive_index[name],
                7,
            ]
            from_trace = trace.coefficients["AB"] + trace.coefficients["ABC"]
            require(
                np.array_equal(from_tensor, from_trace),
                f"{trace.path}: primitive behavior tensor differs",
            )
            tree[(state_index, (name,))] = from_trace.copy()
        for word in composites:
            trace = composite_traces[(state, word.name)]
            tree[(state_index, word.factors)] = (
                trace.coefficients["AB"] + trace.coefficients["ABC"]
            )
    return tree, composable


def tree_nodes(
    context_indices: list[int],
    alphabet: list[str],
) -> list[tuple[int, tuple[str, ...]]]:
    prefixes: list[tuple[str, ...]] = [()]
    prefixes.extend((action,) for action in alphabet)
    prefixes.extend((left, right) for left in alphabet for right in alphabet)
    return [(context, prefix) for context in context_indices for prefix in prefixes]


def refined_tree_classes(
    context_indices: list[int],
    alphabet: list[str],
    observation_labels: dict[tuple[int, tuple[str, ...]], int],
) -> dict[str, Any]:
    depth_two_nodes = [
        (context, (left, right))
        for context in context_indices
        for left in alphabet
        for right in alphabet
    ]
    depth_two_keys = [(observation_labels[node],) for node in depth_two_nodes]
    depth_two_labels, depth_two_key_map = compress_keys(depth_two_keys)
    depth_two = dict(zip(depth_two_nodes, depth_two_labels.tolist()))

    depth_one_nodes = [
        (context, (left,))
        for context in context_indices
        for left in alphabet
    ]
    depth_one_keys = [
        (
            observation_labels[node],
            *(depth_two[(node[0], (node[1][0], right))] for right in alphabet),
        )
        for node in depth_one_nodes
    ]
    depth_one_labels, depth_one_key_map = compress_keys(depth_one_keys)
    depth_one = dict(zip(depth_one_nodes, depth_one_labels.tolist()))

    root_nodes = [(context, ()) for context in context_indices]
    root_keys = [
        (
            observation_labels[node],
            *(depth_one[(node[0], (action,))] for action in alphabet),
        )
        for node in root_nodes
    ]
    root_labels, root_key_map = compress_keys(root_keys)
    root = dict(zip(root_nodes, root_labels.tolist()))

    def summary(labels: np.ndarray) -> dict[str, Any]:
        sizes = sorted(np.bincount(labels).tolist(), reverse=True)
        return {
            "nodes": len(labels),
            "classes": len(sizes),
            "compression_ratio": len(labels) / len(sizes),
            "non_singleton_classes": sum(size > 1 for size in sizes),
            "largest_class": max(sizes),
        }

    return {
        "depth_two": depth_two,
        "depth_one": depth_one,
        "root": root,
        "depth_two_keys": depth_two_key_map,
        "depth_one_keys": depth_one_key_map,
        "root_keys": root_key_map,
        "summary": {
            "depth_two": summary(depth_two_labels),
            "depth_one": summary(depth_one_labels),
            "root": summary(root_labels),
        },
    }


def assign_held_observation_labels(
    train_nodes: list[tuple[int, tuple[str, ...]]],
    held_nodes: list[tuple[int, tuple[str, ...]]],
    tree: dict[tuple[int, tuple[str, ...]], np.ndarray],
    train_component_labels: np.ndarray,
    threshold: float,
) -> tuple[dict[tuple[int, tuple[str, ...]], int], dict[str, Any]]:
    train_vectors = np.stack([tree[node] for node in train_nodes])
    held_vectors = np.stack([tree[node] for node in held_nodes])
    train_norms = np.linalg.norm(train_vectors, axis=1)
    held_norms = np.linalg.norm(held_vectors, axis=1)
    squares = np.maximum(
        held_norms[:, None] ** 2
        + train_norms[None, :] ** 2
        - 2.0 * (held_vectors @ train_vectors.T),
        0.0,
    )
    scales = np.maximum(held_norms[:, None], train_norms[None, :])
    scales = np.maximum(scales, np.finfo(np.float64).tiny)
    distances = np.sqrt(squares) / scales
    nearest = np.argmin(distances, axis=1)
    nearest_distance = distances[np.arange(len(held_nodes)), nearest]
    for index, (source, distance) in enumerate(zip(nearest, nearest_distance)):
        if distance == 0.0 and not np.array_equal(
            held_vectors[index],
            train_vectors[source],
        ):
            direct = float(
                np.linalg.norm(held_vectors[index] - train_vectors[source])
            )
            direct /= max(
                float(held_norms[index]),
                float(train_norms[source]),
                np.finfo(np.float64).tiny,
            )
            nearest_distance[index] = direct
    next_novel = int(train_component_labels.max()) + 1
    labels: dict[tuple[int, tuple[str, ...]], int] = {}
    reused = 0
    for node, source, distance in zip(held_nodes, nearest, nearest_distance):
        if distance <= threshold:
            labels[node] = int(train_component_labels[source])
            reused += 1
        else:
            labels[node] = next_novel
            next_novel += 1
    return labels, {
        "nodes": len(held_nodes),
        "reused_observation_components": reused,
        "reuse_fraction": reused / len(held_nodes),
        "nearest_relative_distance_minimum": float(nearest_distance.min()),
        "nearest_relative_distance_median": float(np.median(nearest_distance)),
        "nearest_relative_distance_maximum": float(nearest_distance.max()),
    }


def held_refined_class_reuse(
    held_indices: list[int],
    alphabet: list[str],
    held_observation_labels: dict[tuple[int, tuple[str, ...]], int],
    training: dict[str, Any],
) -> dict[str, Any]:
    novel_identifier = -1

    def classify(
        keys: list[tuple[Any, ...]],
        training_keys: dict[tuple[Any, ...], int],
    ) -> tuple[list[int], int]:
        nonlocal novel_identifier
        result: list[int] = []
        reused = 0
        novel_keys: dict[tuple[Any, ...], int] = {}
        for key in keys:
            if key in training_keys:
                result.append(training_keys[key])
                reused += 1
            else:
                if key not in novel_keys:
                    novel_keys[key] = novel_identifier
                    novel_identifier -= 1
                result.append(novel_keys[key])
        return result, reused

    depth_two_nodes = [
        (context, (left, right))
        for context in held_indices
        for left in alphabet
        for right in alphabet
    ]
    depth_two_keys = [(held_observation_labels[node],) for node in depth_two_nodes]
    depth_two_values, depth_two_reused = classify(
        depth_two_keys,
        training["depth_two_keys"],
    )
    depth_two = dict(zip(depth_two_nodes, depth_two_values))

    depth_one_nodes = [
        (context, (left,))
        for context in held_indices
        for left in alphabet
    ]
    depth_one_keys = [
        (
            held_observation_labels[node],
            *(depth_two[(node[0], (node[1][0], right))] for right in alphabet),
        )
        for node in depth_one_nodes
    ]
    depth_one_values, depth_one_reused = classify(
        depth_one_keys,
        training["depth_one_keys"],
    )
    depth_one = dict(zip(depth_one_nodes, depth_one_values))

    root_nodes = [(context, ()) for context in held_indices]
    root_keys = [
        (
            held_observation_labels[node],
            *(depth_one[(node[0], (action,))] for action in alphabet),
        )
        for node in root_nodes
    ]
    _, root_reused = classify(root_keys, training["root_keys"])
    return {
        "depth_two_reused": depth_two_reused,
        "depth_two_total": len(depth_two_nodes),
        "depth_one_reused": depth_one_reused,
        "depth_one_total": len(depth_one_nodes),
        "root_reused": root_reused,
        "root_total": len(root_nodes),
    }


def held_action_counterexamples(
    tree: dict[tuple[int, tuple[str, ...]], np.ndarray],
    exploration_indices: list[int],
    alphabet: list[str],
    held_action: str,
    threshold: float,
    root_labels: dict[tuple[int, tuple[str, ...]], int],
) -> dict[str, Any]:
    train_actions = [action for action in alphabet if action != held_action]
    grouped: dict[int, list[int]] = defaultdict(list)
    for context in exploration_indices:
        grouped[root_labels[(context, ())]].append(context)
    split_classes = 0
    contexts_in_splits = 0
    non_singleton = 0
    added_classes = 0
    for contexts in grouped.values():
        if len(contexts) <= 1:
            continue
        non_singleton += 1
        signatures = np.stack(
            [
                np.concatenate(
                    [
                        tree[(context, (held_action,))],
                        *(
                            tree[(context, (held_action, action))]
                            for action in train_actions
                        ),
                    ]
                )
                for context in contexts
            ]
        )
        labels = threshold_component_snapshots(
            relative_distance_matrix(signatures),
            [threshold],
        )[threshold]
        class_count = len(set(labels.tolist()))
        if class_count > 1:
            split_classes += 1
            contexts_in_splits += len(contexts)
            added_classes += class_count - 1
    return {
        "held_action": held_action,
        "non_singleton_root_classes": non_singleton,
        "classes_split_by_held_action": split_classes,
        "contexts_in_split_classes": contexts_in_splits,
        "additional_classes_after_counterexample": added_classes,
    }


def predictive_partition_refinement(
    tree: dict[tuple[int, tuple[str, ...]], np.ndarray],
    states: list[StateSpec],
    alphabet: list[str],
) -> dict[str, Any]:
    exploration_indices = [
        index for index, state in enumerate(states) if state.diagram.phase == "exploration"
    ]
    confirmation_indices = [
        index for index, state in enumerate(states) if state.diagram.phase == "confirmation"
    ]
    train_nodes = tree_nodes(exploration_indices, alphabet)
    held_nodes = tree_nodes(confirmation_indices, alphabet)
    train_vectors = np.stack([tree[node] for node in train_nodes])
    distances = relative_distance_matrix(train_vectors)
    masked = distances.copy()
    np.fill_diagonal(masked, np.inf)
    nearest = np.min(masked, axis=1)
    quantiles = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0)
    thresholds = sorted(
        {
            0.0,
            *(float(np.quantile(nearest, quantile)) for quantile in quantiles),
        }
    )
    snapshots = threshold_component_snapshots(distances, thresholds)
    curve: list[dict[str, Any]] = []
    for threshold in thresholds:
        component_labels = snapshots[threshold]
        train_observations = dict(zip(train_nodes, component_labels.tolist()))
        training = refined_tree_classes(
            exploration_indices,
            alphabet,
            train_observations,
        )
        held_observations, held_observation_summary = assign_held_observation_labels(
            train_nodes,
            held_nodes,
            tree,
            component_labels,
            threshold,
        )
        held_reuse = held_refined_class_reuse(
            confirmation_indices,
            alphabet,
            held_observations,
            training,
        )
        held_actions: list[dict[str, Any]] = []
        for held_action in alphabet:
            reduced_alphabet = [
                action for action in alphabet if action != held_action
            ]
            reduced_nodes = tree_nodes(exploration_indices, reduced_alphabet)
            reduced_vectors = np.stack([tree[node] for node in reduced_nodes])
            reduced_labels = threshold_component_snapshots(
                relative_distance_matrix(reduced_vectors),
                [threshold],
            )[threshold]
            reduced_observations = dict(zip(reduced_nodes, reduced_labels.tolist()))
            reduced_training = refined_tree_classes(
                exploration_indices,
                reduced_alphabet,
                reduced_observations,
            )
            held_actions.append(
                held_action_counterexamples(
                    tree,
                    exploration_indices,
                    alphabet,
                    held_action,
                    threshold,
                    reduced_training["root"],
                )
            )
        curve.append(
            {
                "relative_observation_threshold": threshold,
                "training_observation_components": len(set(component_labels.tolist())),
                "training_refined_classes": training["summary"],
                "confirmation_observation_assignment": held_observation_summary,
                "confirmation_refined_class_reuse": held_reuse,
                "held_action_counterexamples": held_actions,
            }
        )
    return {
        "semantics": "threshold current token-contrast observations, then recursively split by depth-two successor classes",
        "distance": "symmetric relative Euclidean distance on complete token-contrast vectors",
        "threshold_selection": "none; thresholds are the exploration nearest-neighbor quantiles",
        "alphabet": alphabet,
        "horizon": 2,
        "exploration_contexts": len(exploration_indices),
        "confirmation_contexts": len(confirmation_indices),
        "curve": curve,
    }


def reconstruct_edge_corners(edge: EdgeObservation) -> np.ndarray:
    carrier, a, b, ab = edge.coefficients
    return np.stack(
        (
            carrier,
            carrier + a,
            carrier + b,
            carrier + a + b + ab,
        )
    )


def margin_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    margins = np.asarray([float(row["margin"]) for row in rows], dtype=np.float64)
    require(len(margins) > 0 and np.all(np.isfinite(margins)), "empty company margins")
    return {
        "decisions": len(rows),
        "matches_manifest_expectation": sum(
            bool(row["matches_manifest_expectation"]) for row in rows
        ),
        "manifest_match_rate": float(np.mean(margins > 0.0)),
        "minimum_margin": float(margins.min()),
        "mean_margin": float(margins.mean()),
        "maximum_margin": float(margins.max()),
    }


def branch_margin_summaries(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["role"]), str(row["branch"]))].append(row)
    return {
        f"{role}/{branch}": margin_summary(group)
        for (role, branch), group in sorted(groups.items())
    }


def edge_company_evaluation_analysis(
    traces: dict[tuple[StateSpec, str], CubeTrace],
    states: list[StateSpec],
    extensions: list[ActionWord],
    observer_ids: tuple[int, ...],
    observer_labels: dict[int, str],
) -> dict[str, Any]:
    observer_index = {token: index for index, token in enumerate(observer_ids)}
    reference_extension = extensions[0].name
    maximum_base_edge_difference = 0.0
    for state in states:
        reference = traces[(state, reference_extension)].edges["without_C"]
        for extension in extensions[1:]:
            candidate = traces[(state, extension.name)].edges["without_C"]
            require(len(candidate) == len(reference), "base edge zip length differs")
            for left, right in zip(reference, candidate):
                require(
                    left.token_position == right.token_position
                    and left.corner_tokens == right.corner_tokens,
                    "base edge constructors differ across future actions",
                )
                maximum_base_edge_difference = max(
                    maximum_base_edge_difference,
                    float(np.max(np.abs(left.coefficients - right.coefficients))),
                )
    require(maximum_base_edge_difference == 0.0, "base edge codata differs across C")

    main_rows: list[dict[str, Any]] = []
    future_rows: list[dict[str, Any]] = []
    interactions: dict[str, list[float]] = {"controller": [], "attractor": []}
    future_interactions: dict[str, list[float]] = {"controller": [], "attractor": []}
    maximum_counterfactual_prefix_leak = 0.0

    def append_margin(
        destination: list[dict[str, Any]],
        state: StateSpec,
        branch: str,
        raw: np.ndarray,
        corner: int,
        expected: int,
        alternative: int,
    ) -> None:
        require(expected in observer_index and alternative in observer_index, "choice token absent from observer")
        margin = float(
            raw[corner, observer_index[expected]]
            - raw[corner, observer_index[alternative]]
        )
        destination.append(
            {
                "state": state.key,
                "role": state.diagram.role,
                "branch": branch,
                "expected_token": expected,
                "expected_constructor": observer_labels[expected],
                "alternative_token": alternative,
                "alternative_constructor": observer_labels[alternative],
                "margin": margin,
                "matches_manifest_expectation": margin > 0.0,
            }
        )

    for state in states:
        base_edges = traces[(state, reference_extension)].edges["without_C"]
        verb_edges = [
            edge
            for edge in base_edges
            if edge.corner_tokens[0] == edge.corner_tokens[1]
            and edge.corner_tokens[2] == edge.corner_tokens[3]
            and edge.corner_tokens[0] != edge.corner_tokens[2]
        ]
        require(len(verb_edges) == 1, f"{state.key}: expected one B constructor edge")
        verb_edge = verb_edges[0]
        singular, plural = verb_edge.corner_tokens[0], verb_edge.corner_tokens[2]
        require(singular in observer_index and plural in observer_index, f"{state.key}: verb observer absent")
        maximum_counterfactual_prefix_leak = max(
            maximum_counterfactual_prefix_leak,
            float(np.max(np.abs(verb_edge.coefficients[2:]))),
        )
        raw = reconstruct_edge_corners(verb_edge)
        role = state.diagram.role
        if role == "controller":
            append_margin(main_rows, state, "x", raw, 0, singular, plural)
            append_margin(main_rows, state, "AB", raw, 3, plural, singular)
        else:
            append_margin(main_rows, state, "x", raw, 0, singular, plural)
            append_margin(main_rows, state, "A", raw, 1, singular, plural)
        interactions[role].append(
            float(
                raw[1, observer_index[plural]]
                - raw[1, observer_index[singular]]
                - raw[0, observer_index[plural]]
                + raw[0, observer_index[singular]]
            )
        )

        plural_trace = traces[(state, "plural_pronoun")]
        singular_trace = traces[(state, "singular_pronoun")]
        base_positions = len(plural_trace.edges["without_C"]) + 1
        plural_edges = [
            edge
            for edge in plural_trace.edges["with_C"]
            if edge.token_position >= base_positions
        ]
        singular_edges = [
            edge
            for edge in singular_trace.edges["with_C"]
            if edge.token_position >= base_positions
        ]
        require(plural_edges and singular_edges, f"{state.key}: pronoun action has no edges")
        plural_edge = plural_edges[0]
        singular_edge = singular_edges[0]
        require(
            plural_edge.token_position == singular_edge.token_position
            and np.array_equal(plural_edge.coefficients, singular_edge.coefficients),
            f"{state.key}: pre-pronoun codata depends on unconsumed action",
        )
        plural_token = plural_edge.corner_tokens[0]
        singular_token = singular_edge.corner_tokens[0]
        require(
            len(set(plural_edge.corner_tokens)) == 1
            and len(set(singular_edge.corner_tokens)) == 1
            and plural_token != singular_token,
            f"{state.key}: pronoun constructor test differs",
        )
        raw = reconstruct_edge_corners(plural_edge)
        if role == "controller":
            append_margin(future_rows, state, "x", raw, 0, singular_token, plural_token)
            append_margin(future_rows, state, "AB", raw, 3, plural_token, singular_token)
        else:
            append_margin(future_rows, state, "x", raw, 0, singular_token, plural_token)
            append_margin(future_rows, state, "A", raw, 1, singular_token, plural_token)
        edited_corner = 3 if role == "controller" else 1
        future_interactions[role].append(
            float(
                raw[edited_corner, observer_index[plural_token]]
                - raw[edited_corner, observer_index[singular_token]]
                - raw[0, observer_index[plural_token]]
                + raw[0, observer_index[singular_token]]
            )
        )

    def interaction_summary(values: list[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "minimum": float(array.min()),
            "mean": float(array.mean()),
            "maximum": float(array.max()),
        }

    return {
        "semantics": "token constructors are coproduct injections selecting coordinates of prefix codata; feature edits probe product projections demanded by that observer; edge evaluations remain separate and are never folded across a sequence",
        "polynomial_interface_slice": {
            "product_projection_probe": "A changes target-number while preserving the typed token square",
            "coproduct_injection_probe": "the retained singular/plural constructor coordinates",
            "main_edge_demand_formula": "D_A[(iota_plural^* - iota_singular^*)q](x)",
            "scope": "one measured cell of the projection-injection demand lattice, not a quotient of complete hidden states",
        },
        "base_edge_maximum_difference_across_extensions": maximum_base_edge_difference,
        "maximum_unconsumed_B_prefix_leak": maximum_counterfactual_prefix_leak,
        "main_verb_company": {
            "summary": margin_summary(main_rows),
            "summary_by_role_and_branch": branch_margin_summaries(main_rows),
            "projection_demand_response_by_role": {
                role: interaction_summary(values) for role, values in interactions.items()
            },
            "decisions": main_rows,
        },
        "future_pronoun_company": {
            "interaction_semantics": "controller compares grammatical x versus AB; attractor compares grammatical x versus A",
            "summary": margin_summary(future_rows),
            "summary_by_role_and_branch": branch_margin_summaries(future_rows),
            "grammatical_path_injection_response_by_role": {
                role: interaction_summary(values)
                for role, values in future_interactions.items()
            },
            "decisions": future_rows,
        },
    }


def main() -> None:
    args = arguments()
    manifest = read_manifest(args.manifest)
    actions = read_actions(args.actions)
    observer_ids, observer_labels = read_observers(args.observers)
    states = expand_states(manifest)
    extensions = [word for word in expand_words(actions) if len(word.factors) == 1]
    composite_extensions = [
        word for word in expand_words(actions) if len(word.factors) == 2
    ]
    require(
        len(states) == 88
        and len(extensions) == 9
        and len(composite_extensions) == 16,
        "unexpected cube dimensions",
    )
    traces: dict[tuple[StateSpec, str], CubeTrace] = {}
    for state in states:
        for extension in extensions:
            traces[(state, extension.name)] = read_cube_trace(
                state,
                extension,
                args.traces,
                require_local_jets=not args.primitive_terminal_only,
            )
    if args.edge_company_only:
        require(
            all(trace.observer_tokens == observer_ids for trace in traces.values()),
            "cube observer token sets differ",
        )
        require(
            {trace.reference_token for trace in traces.values()} == {1},
            "cube reference tokens differ",
        )
        boundary_counts = {trace.typed_boundaries for trace in traces.values()}
        require(len(boundary_counts) == 1, "cube typed-boundary counts differ")
        edge_company = edge_company_evaluation_analysis(
            traces,
            states,
            extensions,
            observer_ids,
            observer_labels,
        )
        result = {
            "schema_version": 2,
            "artifact": "projection_injection_edge_demand_slice",
            "semantics": {
                "firthian_invariant": "the actual computation must respect company; this artifact checks whether constructor/codata edge evaluation exposes it",
                "edge_observer": "complete logit(token)-logit(BOS) codata before every consumed token constructor",
                "composition": "constructor injections select coordinates of codata; feature actions probe demanded projections; observations are zipped and never folded across positions",
                "manifest_comparison": "a negative margin means the measured model demand differs from the supplied grammatical expectation, not that company went unobserved",
                "not_an_inference_reward": True,
            },
            "provenance": {
                "model": args.model_label,
                "evaluator_commit": args.evaluator_commit or "unspecified",
                "analyzer_commit": git_head(),
                "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                "actions_sha256": hashlib.sha256(args.actions.read_bytes()).hexdigest(),
                "observers_sha256": hashlib.sha256(args.observers.read_bytes()).hexdigest(),
                "trace_count": len(traces),
            },
            "validation": {
                "contexts": len(states),
                "extensions": len(extensions),
                "terminal_observer_width": len(observer_ids),
                "typed_boundaries": next(iter(boundary_counts)),
                "maximum_typed_chain_output_l2_defect": max(
                    trace.maximum_chain_defect for trace in traces.values()
                ),
                "maximum_edge_mobius_inverse_absolute_defect": max(
                    trace.maximum_edge_inverse_defect for trace in traces.values()
                ),
                "maximum_stock_hidden_relative_defect": max(
                    trace.maximum_hidden_relative_defect for trace in traces.values()
                ),
                "maximum_stock_logit_contrast_relative_defect": max(
                    trace.maximum_logit_relative_defect for trace in traces.values()
                ),
            },
            "edge_company_evaluation": edge_company,
            "scope": {
                "establishes": "the number-projection demand of retained verb and future-pronoun constructor injections, plus its agreement with the supplied grammar manifest",
                "does_not_yet_recover": "the full projection-injection demand lattice, its recursive polynomial composition, joint completion selection, or inference sharing law",
                "interpretation": "every margin is an observed company preference; disagreement with the manifest localizes the model's different demand rather than negating Firthian semantics",
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(
            "edge validation: "
            f"traces={len(traces)} observer={len(observer_ids)} "
            f"chain={result['validation']['maximum_typed_chain_output_l2_defect']:.8g}"
        )
        for name in ("main_verb_company", "future_pronoun_company"):
            summary = edge_company[name]["summary"]
            print(
                f"  {name:24s} manifest={summary['matches_manifest_expectation']}/{summary['decisions']} "
                f"minimum_margin={summary['minimum_margin']:.6f}"
            )
        return
    composite_traces: dict[tuple[StateSpec, str], CubeTrace] = {}
    for state in states:
        for extension in composite_extensions:
            composite_traces[(state, extension.name)] = read_cube_trace(
                state,
                extension,
                args.composite_traces,
                require_local_jets=False,
            )
    require(
        all(
            trace.observer_tokens == observer_ids
            for trace in (*traces.values(), *composite_traces.values())
        ),
        "cube observer token sets differ",
    )
    require(
        {
            trace.reference_token
            for trace in (*traces.values(), *composite_traces.values())
        }
        == {1},
        "cube reference tokens differ",
    )
    boundary_counts = {
        trace.typed_boundaries
        for trace in (*traces.values(), *composite_traces.values())
    }
    require(len(boundary_counts) == 1, "cube typed-boundary counts differ")
    typed_boundaries = next(iter(boundary_counts))

    tensor = np.empty(
        (len(states), len(extensions), len(SUBSETS), len(observer_ids)),
        dtype=np.float64,
    )
    for state_index, state in enumerate(states):
        for action_index, extension in enumerate(extensions):
            trace = traces[(state, extension.name)]
            for subset_index, subset in enumerate(SUBSETS):
                tensor[state_index, action_index, subset_index] = trace.coefficients[subset]

    under_c_ab = tensor[:, :, 3, :] + tensor[:, :, 7, :]
    abc = tensor[:, :, 7, :]
    c_jet = tensor[:, :, 4:, :]
    representations = {
        "AB_under_extension": (under_c_ab.reshape(len(states), -1), len(observer_ids)),
        "ABC_third_order": (abc.reshape(len(states), -1), len(observer_ids)),
        "C_conditioned_jet": (c_jet.reshape(len(states), -1), 4 * len(observer_ids)),
    }
    hankel: dict[str, Any] = {}
    for name, (matrix, block_width) in representations.items():
        analysis = double_holdout_analysis(
            matrix,
            states,
            extensions,
            block_width,
        )
        analysis["matrix_rank"] = numerical_row_rank(matrix)
        analysis["confirmation_pareto_ranks"] = pareto_ranks(analysis["confirmation"])
        hankel[name] = analysis

    terminal_ablation = terminal_action_jet_ablation(tensor, states, extensions)
    terminal_atlas = terminal_finite_atlas_analysis(tensor, states, extensions)
    edge_company = edge_company_evaluation_analysis(
        traces,
        states,
        extensions,
        observer_ids,
        observer_labels,
    )
    local_ablation = local_transition_ablation(
        traces,
        states,
        extensions[0],
    )
    tree, composable_alphabet = behavioral_tree(
        tensor,
        states,
        extensions,
        composite_extensions,
        traces,
        composite_traces,
    )
    partition_refinement = predictive_partition_refinement(
        tree,
        states,
        composable_alphabet,
    )
    exact_classes = {
        name: exact_signature_classes(matrix)
        for name, (matrix, _) in representations.items()
    }
    result: dict[str, Any] = {
        "schema_version": 2,
        "artifact": "carrier_conditioned_predictive_grammar_cube",
        "semantics": {
            "firthian_invariant": "observational equivalence must be a congruence under every retained company action",
            "terminal_observer": "fixed vector logit(token)-logit(BOS); no softmax or scalar reward",
            "edge_observer": "before each consumed constructor, retain the complete token-indexed observation and the constructor ID as separate zipped data",
            "edge_evaluation": "a constructor selects one coordinate of the preceding codata; no edge values are summed",
            "hankel": "endpoint diagnostic H[i,(C,t)]=Delta_A Delta_B(q_t after C)(x_i)",
            "terminal_mobius": "eight corners transformed into carrier,A,B,AB,C,AC,BC,ABC",
            "local_jet": "carrier,A,B,AB retained separately within each position-indexed C fiber",
            "local_C": "not subtracted because suffix extension changes the frontier type",
            "heldout_validation": "the held-context x held-family block is unseen; held-family loadings are identified only on training contexts",
            "labels": "controller/attractor pairing enters only future-distinction diagnostics",
            "not_an_inference_reward": True,
        },
        "provenance": {
            "model": args.model_label,
            "evaluator_commit": args.evaluator_commit or "unspecified",
            "analyzer_commit": git_head(),
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "actions_sha256": hashlib.sha256(args.actions.read_bytes()).hexdigest(),
            "observers_sha256": hashlib.sha256(args.observers.read_bytes()).hexdigest(),
            "trace_count": len(traces) + len(composite_traces),
            "primitive_trace_count": len(traces),
            "composite_trace_count": len(composite_traces),
            "raw_local_jets_embedded": False,
        },
        "validation": {
            "contexts": len(states),
            "extensions": len(extensions),
            "composite_extensions": len(composite_extensions),
            "extension_families": sorted({extension.family for extension in extensions}),
            "terminal_observer_width": len(observer_ids),
            "typed_boundaries": typed_boundaries,
            "maximum_typed_chain_output_l2_defect": max(
                trace.maximum_chain_defect
                for trace in (*traces.values(), *composite_traces.values())
            ),
            "maximum_local_mobius_inverse_absolute_defect": max(
                trace.maximum_local_inverse_defect for trace in traces.values()
            ),
            "maximum_edge_mobius_inverse_absolute_defect": max(
                trace.maximum_edge_inverse_defect
                for trace in (*traces.values(), *composite_traces.values())
            ),
            "maximum_terminal_mobius_inverse_absolute_defect": max(
                trace.terminal_inverse_defect
                for trace in (*traces.values(), *composite_traces.values())
            ),
            "maximum_stock_hidden_relative_defect": max(
                trace.maximum_hidden_relative_defect
                for trace in (*traces.values(), *composite_traces.values())
            ),
            "maximum_stock_logit_contrast_relative_defect": max(
                trace.maximum_logit_relative_defect
                for trace in (*traces.values(), *composite_traces.values())
            ),
        },
        "terminal_tensor": {
            "shape": list(tensor.shape),
            "mode_ranks": tensor_mode_ranks(tensor),
            "exact_signature_partitions": exact_classes,
        },
        "behavioral_hankel_double_holdout": hankel,
        "terminal_uniform_action_jet_ablation": terminal_ablation,
        "terminal_carrier_conditioned_finite_atlas": terminal_atlas,
        "edge_company_evaluation": edge_company,
        "local_typed_transition_jet_ablation": local_ablation,
        "predictive_partition_refinement": partition_refinement,
        "scope": {
            "establishes": f"which sampled observer/state/action factorizations preserve or lose finite future-company distinctions on {args.model_label}",
            "does_not_yet_recover": "the exhaustive root-reachable observational quotient, its completion selection, or its inference sharing law",
            "interpretation": "a failed congruence test refines the proposed representation or transport; it is not evidence against Firthian semantics",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(
        "cube validation: "
        f"traces={len(traces) + len(composite_traces)} "
        f"observer={len(observer_ids)} "
        f"chain={result['validation']['maximum_typed_chain_output_l2_defect']:.8g} "
        f"logit_relative={result['validation']['maximum_stock_logit_contrast_relative_defect']:.8g}"
    )
    print("terminal tensor mode ranks:")
    for name, row in result["terminal_tensor"]["mode_ranks"].items():
        print(
            f"  {name:16s} rank={row['rank']:2d}/{row['shape'][0]} "
            f"r99={row['energy_rank_99_percent']:2d}"
        )
    print("double holdout confirmation:")
    for name, row in hankel.items():
        confirmation = row["confirmation"]
        full_rank = str(confirmation["training_rank_range"][1])
        full = confirmation["curve"][full_rank]
        print(
            f"  {name:24s} rank={full_rank:>2s} "
            f"block={full['held_block_relative_error']['mean']:.6f} "
            f"distinction={full['future_distinction_relative_error']['mean']:.6f} "
            f"nearest={confirmation['nearest_source_baseline_relative_error']:.6f}"
        )
    print("local full-rank confirmation median errors:")
    for name, row in local_ablation["aggregate"].items():
        print(
            f"  {name:20s} median={row['error_median']:.6f} "
            f"best={row['best_transition_count']}/{local_ablation['transition_count']}"
        )
    print("edge company decisions:")
    for name in ("main_verb_company", "future_pronoun_company"):
        summary = edge_company[name]["summary"]
        print(
            f"  {name:24s} manifest={summary['matches_manifest_expectation']}/{summary['decisions']} "
            f"minimum_margin={summary['minimum_margin']:.6f}"
        )


if __name__ == "__main__":
    main()

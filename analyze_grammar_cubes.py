#!/usr/bin/env python3
"""Analyze terminal behavior and carrier-conditioned local action jets.

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
SUBSETS = ("carrier", "A", "B", "AB", "C", "AC", "BC", "ABC")
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


@dataclass
class CubeTrace:
    state: StateSpec
    extension: ActionWord
    path: Path
    jet_path: Path
    observer_tokens: tuple[int, ...]
    reference_token: int
    coefficients: dict[str, np.ndarray]
    local: dict[str, list[LocalReference]]
    maximum_chain_defect: float
    maximum_local_inverse_defect: float
    terminal_inverse_defect: float
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
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--model-label", default="Stories15M")
    parser.add_argument("--evaluator-commit")
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


def read_observer_ids(path: Path) -> tuple[int, ...]:
    ids: list[int] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            ids.append(int(stripped.split(maxsplit=1)[0]))
        except ValueError as error:
            raise SystemExit(f"{path}:{line_number}: invalid token ID") from error
    require(bool(ids) and len(ids) == len(set(ids)), "observer IDs are empty or duplicated")
    return tuple(ids)


def trace_stem(state: StateSpec, extension: ActionWord) -> str:
    diagram = state.diagram.trace_name.removesuffix(".jsonl")
    return f"{diagram}--c-{extension.name}"


def read_cube_trace(
    state: StateSpec,
    extension: ActionWord,
    directory: Path,
) -> CubeTrace:
    stem = trace_stem(state, extension)
    path = directory / f"{stem}.jsonl"
    jet_path = directory / f"{stem}.f32"
    require(path.is_file() and jet_path.is_file(), f"missing cube artifacts: {stem}")
    retained: dict[str, list[dict[str, Any]]] = defaultdict(list)
    allowed = {
        "grammatical_cube_meta",
        "grammatical_cube_context",
        "grammatical_cube_local_jet",
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
    require(meta.get("schema_version") == 1, f"{path}: wrong cube schema")
    require(
        meta.get("semantics") == "carrier_conditioned_action_jet_with_terminal_token_contrasts",
        f"{path}: wrong cube semantics",
    )
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
    require(set(local) == {"without_C", "with_C"}, f"{path}: local fibers differ")
    for fiber, rows in local.items():
        rows.sort(key=lambda row: row.boundary_index)
        require(len(rows) == 80, f"{path}: {fiber} boundary count differs")
        require([row.boundary_index for row in rows] == list(range(80)), f"{path}: unordered boundaries")
    for left, right in zip(local["without_C"], local["with_C"]):
        require(
            (left.boundary_index, left.layer, left.phase, left.boundary)
            == (right.boundary_index, right.layer, right.phase, right.boundary),
            f"{path}: C fiber boundaries do not align",
        )
    require(maximum_end == jet_path.stat().st_size, f"{path}: jet sidecar size differs")
    require(int(check["typed_boundaries"]) == 80, f"{path}: check boundary count differs")
    require(float(check["maximum_typed_chain_output_l2_defect"]) == 0.0, f"{path}: typed chain defect")
    for field in (
        "maximum_local_mobius_inverse_absolute_defect",
        "terminal_mobius_inverse_absolute_defect",
        "maximum_stock_hidden_relative_defect",
        "maximum_stock_logit_contrast_relative_defect",
    ):
        require(np.isfinite(float(check[field])), f"{path}: nonfinite {field}")
    return CubeTrace(
        state=state,
        extension=extension,
        path=path,
        jet_path=jet_path,
        observer_tokens=observer_tokens,
        reference_token=int(meta["observer_reference_token"]),
        coefficients=coefficients,
        local=dict(local),
        maximum_chain_defect=float(check["maximum_typed_chain_output_l2_defect"]),
        maximum_local_inverse_defect=float(check["maximum_local_mobius_inverse_absolute_defect"]),
        terminal_inverse_defect=float(check["terminal_mobius_inverse_absolute_defect"]),
        maximum_hidden_relative_defect=float(check["maximum_stock_hidden_relative_defect"]),
        maximum_logit_relative_defect=float(check["maximum_stock_logit_contrast_relative_defect"]),
    )


def load_local(trace: CubeTrace, fiber: str, boundary: int) -> np.ndarray:
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
    require(norm > 0.0, "relative error target is zero")
    return float(np.linalg.norm(predicted - actual) / norm)


def rank_values(maximum: int) -> list[int]:
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
    require(sampled_rank > 0, "training Hankel block has zero rank")
    mean_prediction = np.repeat(np.mean(b, axis=0, keepdims=True), len(held_rows), axis=0)
    distances = np.sum((c[:, None, :] - a[None, :, :]) ** 2, axis=2)
    nearest = np.argmin(distances, axis=1)
    nearest_prediction = b[nearest]
    true_distinction = paired_differences(d, held_states)
    true_distinction_norm = float(np.linalg.norm(true_distinction))
    require(true_distinction_norm > 0.0, "held future distinctions are zero")
    curve: dict[str, Any] = {}
    for rank in rank_values(sampled_rank):
        row_coordinates = u[:, :rank] * singular[:rank]
        held_loadings = np.linalg.lstsq(row_coordinates, b, rcond=None)[0]
        held_row_coordinates = c @ vh[:rank].T
        predicted = held_row_coordinates @ held_loadings
        predicted_distinction = paired_differences(predicted, held_states)
        predicted_norm = float(np.linalg.norm(predicted_distinction))
        dot = float(np.sum(predicted_distinction * true_distinction))
        cosine = 0.0 if predicted_norm == 0.0 else dot / (
            predicted_norm * true_distinction_norm
        )
        curve[str(rank)] = {
            "rank": rank,
            "held_block_relative_error": relative_frobenius(d, predicted),
            "future_distinction_relative_error": relative_frobenius(
                true_distinction,
                predicted_distinction,
            ),
            "future_distinction_norm_ratio": predicted_norm / true_distinction_norm,
            "future_distinction_cosine": cosine,
        }
    return {
        "training_shape": list(a.shape),
        "held_shape": list(d.shape),
        "training_rank": sampled_rank,
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
            values = [row[field] for row in available]
            curve[key][field] = {
                "minimum": min(values),
                "mean": float(np.mean(values)),
                "maximum": max(values),
            }
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
    u, singular, vh = np.linalg.svd(x_train, full_matrices=False)
    tolerance = float(max(x_train.shape) * np.finfo(np.float64).eps * singular[0])
    sampled_rank = int(np.count_nonzero(singular > tolerance))
    require(sampled_rank > 0, "jet feature matrix has zero rank")
    distances = np.sum((x_held[:, None, :] - x_train[None, :, :]) ** 2, axis=2)
    nearest = np.argmin(distances, axis=1)
    nearest_prediction = y_train[nearest]
    true_distinction = paired_differences(y_held, held_states)
    true_distinction_norm = float(np.linalg.norm(true_distinction))
    curve: dict[str, Any] = {}
    for rank in rank_values(sampled_rank):
        coordinates = (x_held @ vh[:rank].T) / singular[:rank]
        predicted = coordinates @ u[:, :rank].T @ y_train
        predicted_distinction = paired_differences(predicted, held_states)
        predicted_norm = float(np.linalg.norm(predicted_distinction))
        cosine = 0.0
        if predicted_norm > 0.0 and true_distinction_norm > 0.0:
            cosine = float(np.sum(predicted_distinction * true_distinction)) / (
                predicted_norm * true_distinction_norm
            )
        curve[str(rank)] = {
            "rank": rank,
            "held_target_relative_error": relative_frobenius(y_held, predicted),
            "future_distinction_relative_error": relative_frobenius(
                true_distinction,
                predicted_distinction,
            ),
            "future_distinction_norm_ratio": predicted_norm / true_distinction_norm,
            "future_distinction_cosine": cosine,
        }
    return {
        "feature_width": features.shape[1],
        "target_width": targets.shape[1],
        "training_rank": sampled_rank,
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
            values = [row[field] for row in available]
            curve[key][field] = {
                "minimum": min(values),
                "mean": float(np.mean(values)),
                "maximum": max(values),
            }
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


def main() -> None:
    args = arguments()
    manifest = read_manifest(args.manifest)
    actions = read_actions(args.actions)
    observer_ids = read_observer_ids(args.observers)
    states = expand_states(manifest)
    extensions = [word for word in expand_words(actions) if len(word.factors) == 1]
    require(len(states) == 88 and len(extensions) == 9, "unexpected cube dimensions")
    traces: dict[tuple[StateSpec, str], CubeTrace] = {}
    for state in states:
        for extension in extensions:
            traces[(state, extension.name)] = read_cube_trace(
                state,
                extension,
                args.traces,
            )
    require(
        all(trace.observer_tokens == observer_ids for trace in traces.values()),
        "cube observer token sets differ",
    )
    require(
        {trace.reference_token for trace in traces.values()} == {1},
        "cube reference tokens differ",
    )

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
    local_ablation = local_transition_ablation(
        traces,
        states,
        extensions[0],
    )
    exact_classes = {
        name: exact_signature_classes(matrix)
        for name, (matrix, _) in representations.items()
    }
    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact": "carrier_conditioned_predictive_grammar_cube",
        "semantics": {
            "terminal_observer": "fixed vector logit(token)-logit(BOS); no softmax or scalar reward",
            "hankel": "H[i,(C,t)]=Delta_A Delta_B(q_t after C)(x_i)",
            "terminal_mobius": "eight corners transformed into carrier,A,B,AB,C,AC,BC,ABC",
            "local_jet": "carrier,A,B,AB retained separately within each position-indexed C fiber",
            "local_C": "not subtracted because suffix extension changes the frontier type",
            "heldout_validation": "both context rows and complete extension-family column blocks are unseen",
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
            "trace_count": len(traces),
            "raw_local_jets_embedded": False,
        },
        "validation": {
            "contexts": len(states),
            "extensions": len(extensions),
            "extension_families": sorted({extension.family for extension in extensions}),
            "terminal_observer_width": len(observer_ids),
            "typed_boundaries": 80,
            "maximum_typed_chain_output_l2_defect": max(
                trace.maximum_chain_defect for trace in traces.values()
            ),
            "maximum_local_mobius_inverse_absolute_defect": max(
                trace.maximum_local_inverse_defect for trace in traces.values()
            ),
            "maximum_terminal_mobius_inverse_absolute_defect": max(
                trace.terminal_inverse_defect for trace in traces.values()
            ),
            "maximum_stock_hidden_relative_defect": max(
                trace.maximum_hidden_relative_defect for trace in traces.values()
            ),
            "maximum_stock_logit_contrast_relative_defect": max(
                trace.maximum_logit_relative_defect for trace in traces.values()
            ),
        },
        "terminal_tensor": {
            "shape": list(tensor.shape),
            "mode_ranks": tensor_mode_ranks(tensor),
            "exact_signature_partitions": exact_classes,
        },
        "behavioral_hankel_double_holdout": hankel,
        "terminal_uniform_action_jet_ablation": terminal_ablation,
        "local_typed_transition_jet_ablation": local_ablation,
        "scope": {
            "establishes": "finite terminal predictive tests and carrier-conditioned jet transition diagnostics on Stories15M",
            "does_not_establish": "an exhaustive observer family, a stable semantic quotient, a completion rule, or an inference speedup",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(
        "cube validation: "
        f"traces={len(traces)} observer={len(observer_ids)} "
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


if __name__ == "__main__":
    main()

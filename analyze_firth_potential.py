#!/usr/bin/env python3
"""Measure whether recursive-company Firth ballots form an exact potential.

The retained recursive company supplies two local constructor observations:

    q_D(root)[d]
    q_E(root, d)[e]

For every two outer injections d0,d1 and two inner injections e0,e1, these
observations define a binary two-player preference one-form.  The outer player
uses its own constructor coordinate in q_D; the inner player uses its own
coordinate in q_E.  The directed square closes exactly when the local game is
an exact potential game:

    D(d0->d1 | e0) + E(e0->e1 | d1)
      = E(e0->e1 | d0) + D(d0->d1 | e1).

All reported quantities are differences of token-logit contrasts, so the
common logit gauge cancels.  No probabilities, path likelihoods, completion
reward, or winner are introduced.  The detailed JSONL is line-buffered so
each completed cell remains inspectable if analysis is interrupted.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from gather_recursive_company import DEFAULT_OUTPUT as DEFAULT_RECURSIVE_DIRECTORY


ROOT = Path(__file__).resolve().parent
DEFAULT_TRACE = DEFAULT_RECURSIVE_DIRECTORY / "recursive-company.jsonl"
DEFAULT_CELLS = ROOT / "work_traces" / "firth_potential" / "cells.jsonl"
DEFAULT_RESULT = ROOT / "outputs" / "cps-stories15m-firth-potential.json"


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
    parser.add_argument("--trace", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--cells", type=Path, default=DEFAULT_CELLS)
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--model-label", default="Stories15M")
    parser.add_argument("--evaluator-commit")
    parser.add_argument(
        "--retained-extrema",
        type=int,
        default=32,
        help="number of largest and smallest nonzero cells embedded in the compact report",
    )
    return parser.parse_args()


@dataclass(frozen=True)
class Candidate:
    token: int
    piece: str
    contrast: float


@dataclass(frozen=True)
class Demand:
    root: str
    depth: int
    path: tuple[int, ...]
    text: str
    candidates: tuple[Candidate, ...]

    @property
    def tokens(self) -> tuple[int, ...]:
        return tuple(candidate.token for candidate in self.candidates)

    def candidate(self, token: int) -> Candidate:
        matches = [candidate for candidate in self.candidates if candidate.token == token]
        require(len(matches) == 1, f"{self.root} {self.path}: token {token} is not unique")
        return matches[0]


def read_demands(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Demand], dict[tuple[str, int], Demand], dict[str, Any] | None]:
    meta: dict[str, Any] | None = None
    check: dict[str, Any] | None = None
    outer: dict[str, Demand] = {}
    inner: dict[tuple[str, int], Demand] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            kind = row.get("kind")
            if kind == "recursive_company_meta":
                require(meta is None, f"{path}:{line_number}: duplicate meta")
                meta = row
                continue
            if kind == "recursive_company_check":
                require(check is None, f"{path}:{line_number}: duplicate check")
                check = row
                continue
            if kind != "recursive_company_demand":
                continue
            candidates = tuple(
                Candidate(
                    token=int(candidate["token"]),
                    piece=str(candidate["piece"]),
                    contrast=float(candidate["contrast"]),
                )
                for candidate in row["candidates"]
            )
            require(
                candidates
                and len({candidate.token for candidate in candidates}) == len(candidates)
                and all(math.isfinite(candidate.contrast) for candidate in candidates),
                f"{path}:{line_number}: invalid candidate codata",
            )
            demand = Demand(
                root=str(row["root"]),
                depth=int(row["depth"]),
                path=tuple(int(token) for token in row["path_tokens"]),
                text=str(row["text"]),
                candidates=candidates,
            )
            require(len(demand.path) == demand.depth, f"{path}:{line_number}: path depth differs")
            if demand.depth == 0:
                require(demand.root not in outer, f"duplicate root demand {demand.root}")
                outer[demand.root] = demand
            elif demand.depth == 1:
                key = (demand.root, demand.path[0])
                require(key not in inner, f"duplicate inner demand {key}")
                inner[key] = demand
            else:
                raise SystemExit(f"{path}:{line_number}: expected retained depth two term")
    require(meta is not None, f"{path}: missing recursive-company meta")
    return meta, outer, inner, check


def finite_summary(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    require(array.size > 0 and np.all(np.isfinite(array)), "empty or non-finite summary family")
    absolute = np.abs(array)
    return {
        "count": int(array.size),
        "exact_zero": int(np.sum(array == 0.0)),
        "negative": int(np.sum(array < 0.0)),
        "positive": int(np.sum(array > 0.0)),
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "maximum": float(np.max(array)),
        "minimum_absolute": float(np.min(absolute)),
        "median_absolute": float(np.median(absolute)),
        "mean_absolute": float(np.mean(absolute)),
        "maximum_absolute": float(np.max(absolute)),
        "absolute_quantiles": {
            "q01": float(np.quantile(absolute, 0.01)),
            "q10": float(np.quantile(absolute, 0.10)),
            "q25": float(np.quantile(absolute, 0.25)),
            "q50": float(np.quantile(absolute, 0.50)),
            "q75": float(np.quantile(absolute, 0.75)),
            "q90": float(np.quantile(absolute, 0.90)),
            "q99": float(np.quantile(absolute, 0.99)),
        },
    }


def candidate_json(candidate: Candidate) -> dict[str, Any]:
    return {
        "token": candidate.token,
        "piece": candidate.piece,
        "contrast": candidate.contrast,
    }


def split_root(root: str) -> tuple[str, str]:
    fields = root.rsplit("::", 1)
    require(len(fields) == 2, f"root lacks grammatical corner suffix: {root}")
    return fields[0], fields[1]


def retained_record(record: dict[str, Any]) -> dict[str, Any]:
    """Compact a detailed cell without hiding the candidate identities or scores."""
    return {
        "root": record["root"],
        "case": record["case"],
        "corner": record["corner"],
        "outer_transition": record["outer_transition"],
        "inner_transition": record["inner_transition"],
        "local_ballots": record["local_ballots"],
        "preference_one_form": record["preference_one_form"],
        "path_preference_gains": record["path_preference_gains"],
        "preference_circulation": record["preference_circulation"],
        "circulation_fraction": record["hodge_decomposition"]["circulation_fraction"],
        "actual_later_codata_pullback": record["actual_later_codata_pullback"],
    }


def main() -> None:
    args = arguments()
    require(args.retained_extrema > 0, "--retained-extrema must be positive")
    require(args.trace.is_file(), f"missing trace: {args.trace}")
    meta, outer, inner, check = read_demands(args.trace)
    require(meta.get("schema_version") == 1, "recursive-company schema differs")
    require(meta.get("semantics") == "dependent_polynomial_company_tree", "term semantics differ")
    require(meta.get("depth") == 2, "potential analysis currently requires depth two")
    require(meta.get("probabilities_used") is False, "source trace used probabilities")
    require(meta.get("scalar_reward_used") is False, "source trace used a scalar reward")
    require(meta.get("whole_completion_argmax_used") is False, "source trace selected a completion")
    require(meta.get("complete_paths_flattened") is False, "source trace flattened branches")

    root_tokens: tuple[int, ...] | None = None
    inner_tokens: tuple[int, ...] | None = None
    for root, demand in outer.items():
        if root_tokens is None:
            root_tokens = demand.tokens
        require(demand.tokens == root_tokens, f"outer constructor family changes at {root}")
        for token in demand.tokens:
            child = inner.get((root, token))
            require(child is not None, f"missing inner demand {(root, token)}")
            if inner_tokens is None:
                inner_tokens = child.tokens
            require(child.tokens == inner_tokens, f"inner constructor family changes at {(root, token)}")
    require(root_tokens is not None and inner_tokens is not None, "recursive term has no demands")
    require(len(inner) == len(outer) * len(root_tokens), "inner demand coverage differs")
    require(tuple(meta.get("family_widths", ())) == (len(root_tokens), len(inner_tokens)), "family widths differ")

    args.cells.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    circulations: list[float] = []
    fractions: list[float] = []
    outer_future_dependence: list[float] = []
    pullback_closure_defects: list[float] = []
    pullback_reciprocity_defects: list[float] = []
    per_root: dict[str, list[float]] = defaultdict(list)
    per_corner: dict[str, list[float]] = defaultdict(list)
    per_outer_pair: dict[tuple[int, int], list[float]] = defaultdict(list)
    per_inner_pair: dict[tuple[int, int], list[float]] = defaultdict(list)
    largest: list[tuple[float, int, dict[str, Any]]] = []
    smallest_nonzero: list[tuple[float, int, dict[str, Any]]] = []
    sequence = 0

    with args.cells.open("w", encoding="utf-8", buffering=1) as cells:
        for root in sorted(outer):
            root_demand = outer[root]
            case, corner = split_root(root)
            for d0, d1 in combinations(root_tokens, 2):
                outer0 = root_demand.candidate(d0)
                outer1 = root_demand.candidate(d1)
                demand0 = inner[(root, d0)]
                demand1 = inner[(root, d1)]
                outer_gain = outer1.contrast - outer0.contrast
                for e0, e1 in combinations(inner_tokens, 2):
                    inner00 = demand0.candidate(e0)
                    inner01 = demand0.candidate(e1)
                    inner10 = demand1.candidate(e0)
                    inner11 = demand1.candidate(e1)
                    inner_gain_at_d0 = inner01.contrast - inner00.contrast
                    inner_gain_at_d1 = inner11.contrast - inner10.contrast

                    # The actual continuation available to the outer choice is
                    # the complete later codata as a function of d.  Holding e
                    # fixed and changing d supplies the missing reciprocal
                    # edges directly; no scalar path fold is involved.
                    continuation_outer_at_e0 = (
                        inner10.contrast - inner00.contrast
                    )
                    continuation_outer_at_e1 = (
                        inner11.contrast - inner01.contrast
                    )
                    continuation_inner_at_d0 = inner_gain_at_d0
                    continuation_inner_at_d1 = inner_gain_at_d1
                    continuation_closure = (
                        continuation_outer_at_e0
                        + continuation_inner_at_d1
                        - continuation_outer_at_e1
                        - continuation_inner_at_d0
                    )
                    continuation_outer_mixed = (
                        continuation_outer_at_e1
                        - continuation_outer_at_e0
                    )
                    continuation_inner_mixed = (
                        continuation_inner_at_d1
                        - continuation_inner_at_d0
                    )
                    continuation_reciprocity = (
                        continuation_outer_mixed - continuation_inner_mixed
                    )

                    # The outer observation is made before either inner injection,
                    # so both copies are exactly the same retained operand.  Keep
                    # both edges explicit to expose the square being tested.
                    outer_gain_at_e0 = outer_gain
                    outer_gain_at_e1 = outer_gain
                    path_outer_then_inner = outer_gain_at_e0 + inner_gain_at_d1
                    path_inner_then_outer = inner_gain_at_d0 + outer_gain_at_e1
                    circulation = path_outer_then_inner - path_inner_then_outer

                    edge = np.asarray(
                        (
                            outer_gain_at_e0,
                            outer_gain_at_e1,
                            inner_gain_at_d0,
                            inner_gain_at_d1,
                        ),
                        dtype=np.float64,
                    )
                    cycle_basis = np.asarray((1.0, -1.0, -1.0, 1.0))
                    circulating = (circulation / 4.0) * cycle_basis
                    conservative = edge - circulating
                    conservative_closure = (
                        conservative[0]
                        + conservative[3]
                        - conservative[1]
                        - conservative[2]
                    )
                    edge_norm = float(np.linalg.norm(edge))
                    circulating_norm = float(np.linalg.norm(circulating))
                    fraction = 0.0 if edge_norm == 0.0 else circulating_norm / edge_norm
                    potential = (
                        0.0,
                        float(conservative[0]),
                        float(conservative[2]),
                        float(conservative[0] + conservative[3]),
                    )
                    # If the later selector is held fixed, exact potential
                    # compatibility requires the earlier selector to acquire
                    # precisely the same mixed response.  Splitting that
                    # correction symmetrically between the two future fibers
                    # is the unique minimum-L2 change to the earlier edges.
                    reciprocal_outer = np.asarray(
                        (
                            outer_gain - circulation / 2.0,
                            outer_gain + circulation / 2.0,
                        ),
                        dtype=np.float64,
                    )
                    backward_completed = np.asarray(
                        (
                            reciprocal_outer[0],
                            reciprocal_outer[1],
                            inner_gain_at_d0,
                            inner_gain_at_d1,
                        ),
                        dtype=np.float64,
                    )
                    backward_closure = (
                        backward_completed[0]
                        + backward_completed[3]
                        - backward_completed[1]
                        - backward_completed[2]
                    )
                    backward_potential = (
                        0.0,
                        float(backward_completed[0]),
                        float(backward_completed[2]),
                        float(backward_completed[0] + backward_completed[3]),
                    )
                    require(
                        abs(conservative_closure)
                        <= 32.0 * np.finfo(np.float64).eps * max(1.0, edge_norm),
                        "Hodge projection did not close its conservative square",
                    )
                    require(
                        abs(backward_closure)
                        <= 32.0 * np.finfo(np.float64).eps * max(1.0, edge_norm),
                        "backward reciprocal completion did not close its square",
                    )

                    record = {
                        "kind": "firth_potential_cell",
                        "root": root,
                        "case": case,
                        "corner": corner,
                        "outer_transition": {
                            "from": {"token": d0, "piece": outer0.piece},
                            "to": {"token": d1, "piece": outer1.piece},
                        },
                        "inner_transition": {
                            "from": {"token": e0, "piece": inner00.piece},
                            "to": {"token": e1, "piece": inner01.piece},
                        },
                        "local_ballots": {
                            "outer": {
                                "at_inner_from": [candidate_json(outer0), candidate_json(outer1)],
                                "at_inner_to": [candidate_json(outer0), candidate_json(outer1)],
                            },
                            "inner_at_outer_from": [candidate_json(inner00), candidate_json(inner01)],
                            "inner_at_outer_to": [candidate_json(inner10), candidate_json(inner11)],
                        },
                        "preference_one_form": {
                            "outer_at_inner_from": outer_gain_at_e0,
                            "outer_at_inner_to": outer_gain_at_e1,
                            "inner_at_outer_from": inner_gain_at_d0,
                            "inner_at_outer_to": inner_gain_at_d1,
                        },
                        "path_preference_gains": {
                            "outer_then_inner": path_outer_then_inner,
                            "inner_then_outer": path_inner_then_outer,
                        },
                        "preference_circulation": circulation,
                        "dislike_energy_circulation": -circulation,
                        "actual_later_codata_pullback": {
                            "scalar_cell": "Phi_E(d,e)=q_E(root,d)[e]",
                            "preference_one_form": {
                                "outer_at_inner_from": continuation_outer_at_e0,
                                "outer_at_inner_to": continuation_outer_at_e1,
                                "inner_at_outer_from": continuation_inner_at_d0,
                                "inner_at_outer_to": continuation_inner_at_d1,
                            },
                            "outer_mixed_response": continuation_outer_mixed,
                            "inner_mixed_response": continuation_inner_mixed,
                            "reciprocity_defect": continuation_reciprocity,
                            "closure_defect": continuation_closure,
                            "gauge_scope": (
                                "individual outer edges use the retained reference-token "
                                "section; equality of mixed responses is invariant under "
                                "an arbitrary common translation of each q_E(d) fiber"
                            ),
                        },
                        "hodge_decomposition": {
                            "edge_order": [
                                "outer_at_inner_from",
                                "outer_at_inner_to",
                                "inner_at_outer_from",
                                "inner_at_outer_to",
                            ],
                            "conservative_preference_one_form": conservative.tolist(),
                            "circulating_preference_one_form": circulating.tolist(),
                            "integrated_conservative_potential": {
                                "outer_from_inner_from": potential[0],
                                "outer_to_inner_from": potential[1],
                                "outer_from_inner_to": potential[2],
                                "outer_to_inner_to": potential[3],
                            },
                            "conservative_closure_defect": float(conservative_closure),
                            "circulating_l2": circulating_norm,
                            "complete_one_form_l2": edge_norm,
                            "circulation_fraction": fraction,
                        },
                        "required_backward_continuation": {
                            "measured_outer_mixed_response": (
                                outer_gain_at_e1 - outer_gain_at_e0
                            ),
                            "measured_inner_mixed_response": (
                                inner_gain_at_d1 - inner_gain_at_d0
                            ),
                            "required_outer_mixed_response": circulation,
                            "minimum_l2_outer_edge_correction": [
                                -circulation / 2.0,
                                circulation / 2.0,
                            ],
                            "completed_preference_one_form": backward_completed.tolist(),
                            "completed_closure_defect": float(backward_closure),
                            "integrated_completed_potential": {
                                "outer_from_inner_from": backward_potential[0],
                                "outer_to_inner_from": backward_potential[1],
                                "outer_from_inner_to": backward_potential[2],
                                "outer_to_inner_to": backward_potential[3],
                            },
                            "status": (
                                "absent when the outer choice observes only q_D; "
                                "present exactly as the mixed response of q_E(d) when "
                                "the later codata is retained as its continuation"
                            ),
                        },
                    }
                    cells.write(json.dumps(record, separators=(",", ":")) + "\n")

                    circulations.append(circulation)
                    fractions.append(fraction)
                    outer_future_dependence.append(outer_gain_at_e1 - outer_gain_at_e0)
                    pullback_closure_defects.append(continuation_closure)
                    pullback_reciprocity_defects.append(continuation_reciprocity)
                    per_root[root].append(circulation)
                    per_corner[corner].append(circulation)
                    per_outer_pair[(d0, d1)].append(circulation)
                    per_inner_pair[(e0, e1)].append(circulation)
                    compact = retained_record(record)
                    absolute = abs(circulation)
                    sequence += 1
                    if len(largest) < args.retained_extrema:
                        heapq.heappush(largest, (absolute, sequence, compact))
                    elif absolute > largest[0][0]:
                        heapq.heapreplace(largest, (absolute, sequence, compact))
                    if absolute > 0.0:
                        negative_absolute = -absolute
                        if len(smallest_nonzero) < args.retained_extrema:
                            heapq.heappush(
                                smallest_nonzero,
                                (negative_absolute, sequence, compact),
                            )
                        elif negative_absolute > smallest_nonzero[0][0]:
                            heapq.heapreplace(
                                smallest_nonzero,
                                (negative_absolute, sequence, compact),
                            )

    expected_cells = (
        len(outer)
        * math.comb(len(root_tokens), 2)
        * math.comb(len(inner_tokens), 2)
    )
    require(len(circulations) == expected_cells, "potential-cell coverage differs")
    largest_records = [
        record
        for _absolute, _sequence, record in sorted(
            largest,
            key=lambda item: (-item[0], item[1]),
        )
    ]
    smallest_records = [
        record
        for _negative_absolute, _sequence, record in sorted(
            smallest_nonzero,
            key=lambda item: (-item[0], item[1]),
        )
    ]

    def grouped_summary(groups: dict[Any, list[float]]) -> list[dict[str, Any]]:
        rows = []
        for key, values in groups.items():
            summary = finite_summary(values)
            rows.append(
                {
                    "key": list(key) if isinstance(key, tuple) else key,
                    "count": summary["count"],
                    "exact_zero": summary["exact_zero"],
                    "mean_absolute": summary["mean_absolute"],
                    "median_absolute": summary["median_absolute"],
                    "maximum_absolute": summary["maximum_absolute"],
                }
            )
        return sorted(rows, key=lambda row: (-row["mean_absolute"], str(row["key"])))

    check_fields = {} if check is None else {
        key: check[key]
        for key in (
            "roots",
            "depth",
            "demand_nodes",
            "complete_branches",
            "maximum_calls_per_filler",
            "root_observer_runs",
            "composed_observations",
            "composition_steps",
        )
        if key in check
    }
    result = {
        "schema_version": 1,
        "artifact": "recursive_company_firth_exact_potential_audit",
        "semantics": {
            "outer_local_utility": "u_D(d,e)=q_D(root)[d]",
            "inner_local_utility": "u_E(d,e)=q_E(root,d)[e]",
            "preference_one_form": "each oriented edge is the active player's own constructor-coordinate contrast difference",
            "exact_potential_law": "D(d0->d1|e0)+E(e0->e1|d1)=E(e0->e1|d0)+D(d0->d1|e1)",
            "gauge": "every edge subtracts two coordinates of one token-codata vector, eliminating its common logit gauge",
            "hodge": "equal-edge-metric orthogonal decomposition on each four-edge constructor square",
            "backward_completion": (
                "hold the measured later-selector edges fixed and add the "
                "minimum-L2 reciprocal mixed response demanded of the earlier continuation"
            ),
            "actual_backward_continuation": (
                "retain q_E(d) as the result of each outer candidate; its "
                "cross-d coordinate differences supply the reciprocal outer edges"
            ),
            "probabilities_used": False,
            "absolute_logits_combined": False,
            "path_likelihoods_summed": False,
            "completion_reward_defined": False,
            "selection_performed": False,
        },
        "provenance": {
            "model": args.model_label,
            "evaluator_commit": args.evaluator_commit or "unspecified",
            "analyzer_commit": git_head(),
            "source_trace": str(args.trace.resolve()),
            "source_trace_sha256": sha256(args.trace),
            "detailed_cells": str(args.cells.resolve()),
            "detailed_cells_sha256": sha256(args.cells),
        },
        "term": {
            "roots": len(outer),
            "outer_family": [
                {
                    "token": token,
                    "piece": outer[next(iter(sorted(outer)))].candidate(token).piece,
                }
                for token in root_tokens
            ],
            "inner_family": [
                {
                    "token": token,
                    "piece": inner[(next(iter(sorted(outer))), root_tokens[0])].candidate(token).piece,
                }
                for token in inner_tokens
            ],
            "outer_pairs_per_root": math.comb(len(root_tokens), 2),
            "inner_pairs_per_outer_pair": math.comb(len(inner_tokens), 2),
            "constructor_squares": expected_cells,
        },
        "source_integrity": {
            "meta": {
                key: meta[key]
                for key in (
                    "company_rows",
                    "demand_nodes",
                    "complete_branches",
                    "learned_fillers",
                    "family_filler_calls",
                    "maximum_calls_per_filler",
                    "family_scalar_reads",
                    "backend",
                    "root_observer_runs",
                    "observations_composed",
                )
                if key in meta
            },
            "check": check_fields,
            "outer_demands": len(outer),
            "inner_demands": len(inner),
            "nonfinite_coordinates": 0,
        },
        "potential_audit": {
            "preference_circulation": finite_summary(circulations),
            "circulation_fraction": finite_summary(fractions),
            "outer_response_to_unobserved_future_constructor": finite_summary(
                outer_future_dependence
            ),
            "actual_later_codata_pullback_closure_defect": finite_summary(
                pullback_closure_defects
            ),
            "actual_later_codata_pullback_reciprocity_defect": finite_summary(
                pullback_reciprocity_defects
            ),
            "exactly_closed_cells": int(np.sum(np.asarray(circulations) == 0.0)),
            "nonclosed_cells": int(np.sum(np.asarray(circulations) != 0.0)),
            "by_grammatical_corner": grouped_summary(per_corner),
            "by_root": grouped_summary(per_root),
            "by_outer_pair": grouped_summary(per_outer_pair),
            "by_inner_pair": grouped_summary(per_inner_pair),
            "largest_absolute_cells": largest_records,
            "smallest_nonzero_absolute_cells": smallest_records,
        },
        "scope": {
            "establishes": (
                "whether the two edge-local final-unembedding Firth selectors in "
                "the retained recursive constructor company are restrictions of "
                "one exact scalar potential on each sampled D/E square"
            ),
            "does_not_establish": (
                "whether the full scale-indexed CPS ballot family becomes a "
                "potential after higher-scale continuations vote"
            ),
            "next_if_nonclosed": (
                "compose the full nested codata through selection strength without "
                "discarding the cross-candidate continuation coordinates"
            ),
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    circulation_summary = result["potential_audit"]["preference_circulation"]
    fraction_summary = result["potential_audit"]["circulation_fraction"]
    print(
        "firth_potential "
        f"roots={len(outer)} cells={expected_cells} "
        f"closed={result['potential_audit']['exactly_closed_cells']} "
        f"nonclosed={result['potential_audit']['nonclosed_cells']}"
    )
    print(
        "  circulation "
        f"min={circulation_summary['minimum']:.9g} "
        f"median_abs={circulation_summary['median_absolute']:.9g} "
        f"mean_abs={circulation_summary['mean_absolute']:.9g} "
        f"max={circulation_summary['maximum']:.9g}"
    )
    print(
        "  circulation_fraction "
        f"median={fraction_summary['median']:.9g} "
        f"mean={fraction_summary['mean']:.9g} "
        f"max={fraction_summary['maximum']:.9g}"
    )
    pullback_closure = result["potential_audit"][
        "actual_later_codata_pullback_closure_defect"
    ]
    pullback_reciprocity = result["potential_audit"][
        "actual_later_codata_pullback_reciprocity_defect"
    ]
    print(
        "  actual_q_E_pullback "
        f"closure_median_abs={pullback_closure['median_absolute']:.9g} "
        f"closure_max_abs={pullback_closure['maximum_absolute']:.9g} "
        f"reciprocity_median_abs={pullback_reciprocity['median_absolute']:.9g} "
        f"reciprocity_max_abs={pullback_reciprocity['maximum_absolute']:.9g}"
    )
    print(f"  detailed_cells={args.cells}")


if __name__ == "__main__":
    main()

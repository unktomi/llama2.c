#!/usr/bin/env python3
"""Validate recursive-company structure and parent-observation parity.

The prior two-level polynomial trace independently observed root verb codata
and every verb-indexed hole codata on the CPU path. This analyzer verifies
that the recursive Metal company preserves those parent observations while
actually consuming every hole constructor and retaining every resulting
terminal codata vector. It does not order complete branches.
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
    require(
        meta.get("completion_selected") is False,
        "not an observation-only trace; the final-row selection implementation was removed",
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
        },
        "codata_composition": {
            "codata_constructed_before_observation": True,
            "root_observer_runs": int(check["root_observer_runs"]),
            "composed_observations": int(check["composed_observations"]),
            "continuation_bind_steps": int(check["composition_steps"]),
            "edge_observation_defects": 0,
            "terminal_observation_defects": 0,
        },
        "selection_boundary": {
            "completion_selected": False,
            "reason": (
                "Observation-only diagnostic. The rejected final-row selector "
                "was deleted; this executable does not perform inference."
            ),
        },
        "sample_terminal_observations": sample_terminals,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        "recursive_company_analysis "
        f"roots=176 demands={sum(depth_counts.values())} "
        f"leaves={len(terminal_paths)} nonfinite={nonfinite_coordinates} "
        f"composed={len(composed_paths)} "
        f"max_parent_relative_defect={maximum_relative_parent_defect:.9g}"
    )


if __name__ == "__main__":
    main()

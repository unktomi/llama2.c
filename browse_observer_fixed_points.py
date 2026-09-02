#!/usr/bin/env python3
"""Browse retained corpus fixed-point measurements, not generated completions.

All vocabulary scores come from the flushed float32 sidecar. --top limits only
display; --top 0 displays the entire vocabulary. No model execution occurs.
"""

from __future__ import annotations

import argparse
import json
import mmap
import re
import struct
from collections import Counter, defaultdict
from pathlib import Path


def decoded_display(text):
    """Render named ASCII byte tokens, including newlines, as actual text."""
    def replace(match):
        byte = int(match.group(1), 16)
        return chr(byte) if byte in (9, 10, 13) or 32 <= byte < 127 else match[0]
    return re.sub(r"<0x([0-9a-fA-F]{2})>", replace, text)


def load_trace(path: Path):
    meta = None
    summary = None
    vocabulary = {}
    samples = {}
    events = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            # A live writer may have flushed part of its final JSONL record.
            if not line.endswith("\n"):
                break
            record = json.loads(line)
            kind = record["kind"]
            if kind == "meta":
                meta = record
            elif kind == "vocabulary":
                vocabulary[record["token"]] = decoded_display(record["piece"])
            elif kind == "sample":
                samples[record["sample"]] = record
            elif kind in {"fixed_observation", "joint_omission"}:
                events.append(record)
            elif kind == "summary":
                summary = record
    if not meta or meta["measurement"] != "individual_residual_observer_fixed_points":
        raise SystemExit("not an observer fixed-point trace")
    return meta, vocabulary, samples, events, summary


def counts(events):
    result = Counter()
    for event in events:
        result["operations"] += 1
        result["codata_fixed"] += event["codata_fixed"]
        result["choice_fixed"] += event["choice_fixed"]
        result["target_order_fixed"] += event["target_order_changed_count"] == 0
        result["target_order_changed"] += event["target_order_changed_count"] != 0
    return dict(result)


def summarize(path, meta, samples, events, summary):
    singles = [e for e in events if e["kind"] == "fixed_observation"]
    reachable = [e for e in singles if not e["structurally_unreachable"]]
    joints = [e for e in events if e["kind"] == "joint_omission"]
    grouped = defaultdict(list)
    for event in reachable:
        grouped[(event["layer"], event["operation"])].append(event)
    changed = [e for e in reachable if e["target_order_changed_count"]]
    fixed_but_changed = [e for e in reachable if e["choice_fixed"] and not e["codata_fixed"]]
    return {
        "trace": str(path.resolve()),
        "meta": meta,
        "complete": summary is not None,
        "summary": summary,
        "all_individual_updates": counts(singles),
        "excluding_structurally_unreachable": counts(reachable),
        "by_layer_and_operation": [
            {"layer": layer, "operation": operation, **counts(records)}
            for (layer, operation), records in sorted(grouped.items())
        ],
        "joint_groups": len(joints),
        "joint_choice_changed": sum(not e["choice_fixed"] for e in joints),
        "samples": list(samples.values()),
        # First chronological examples; not selected for grammatical quality.
        "first_changed_comparisons": changed[:8],
        "first_fixed_choice_changed_codata": fixed_but_changed[:8],
        "joint_choice_changes": [e for e in joints if not e["choice_fixed"]],
        "scope": "Observed corpus points only. No grammar labels, inference reward, "
                 "global eigenspace, or deployed rewrite are claimed.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--sample", type=int)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--position", type=int)
    parser.add_argument("--operation", choices=["attention_residual", "ffn_residual"])
    parser.add_argument("--joint", action="store_true")
    parser.add_argument("--changed", action="store_true",
                        help="display only target comparisons whose ordering changed")
    parser.add_argument("--top", type=int, default=8)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--output", type=Path, help="save the compact summary, without overwriting")
    args = parser.parse_args()
    if args.top < 0 or args.limit < 1:
        parser.error("--top must be nonnegative; --limit must be positive")
    meta, vocabulary, samples, events, summary = load_trace(args.trace)
    report = summarize(args.trace, meta, samples, events, summary)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    print(f"Model: {meta['model']} | samples={len(samples)} | "
          f"context={meta['positions']} | complete={summary is not None}")
    print("Individual updates, excluding final-layer rows with no causal route to root:")
    print(json.dumps(report["excluding_structurally_unreachable"]))
    print(f"Combined independently winner-preserving omissions changed the winner in "
          f"{report['joint_choice_changed']}/{report['joint_groups']} groups.")
    if args.summary:
        return
    selected = []
    for event in events:
        if (event["kind"] == "joint_omission") != args.joint:
            continue
        if args.sample is not None and event["sample"] != args.sample:
            continue
        if args.layer is not None and event["layer"] != args.layer:
            continue
        if args.position is not None and event["position"] != args.position:
            continue
        if args.operation is not None and event["operation"] != args.operation:
            continue
        if args.changed and not event["target_order_changed_count"]:
            continue
        selected.append(event)
    sidecar = Path(meta["logits_file"])
    if not sidecar.is_file():
        sidecar = args.trace.parent / sidecar.name
    endian = "<" if meta["byte_order"] == "little" else ">"
    vector = struct.Struct(f"{endian}{meta['vocabulary']}f")
    with sidecar.open("rb") as binary, mmap.mmap(binary.fileno(), 0, access=mmap.ACCESS_READ) as data:
        for event in selected[: args.limit]:
            sample = samples[event["sample"]]
            on = vector.unpack_from(data, sample["logits_offset"])
            off = vector.unpack_from(data, event["logits_offset"])
            target = sample["target"]
            print(f"\nSample {event['sample']}: {decoded_display(sample['text'])!r}")
            print(f"Corpus next token: {decoded_display(sample['target_piece'])!r}")
            print(f"Layer {event['layer']} {event['operation']} "
                  f"at {event['term_piece']!r} (position {event['position']})")
            if args.joint:
                print(f"Omitted positions: {event['omitted_positions']}")
            print(f"All contrasts fixed={event['codata_fixed']}; "
                  f"winner set fixed={event['choice_fixed']}; "
                  f"changed target comparisons={event['target_order_changed_count']}")
            print("Alternative                 target margin WITH   WITHOUT     same order")
            shown = 0
            for token in sorted(range(len(on)), key=lambda t: (-on[t], t)):
                if token == target:
                    continue
                a, b = on[target] - on[token], off[target] - off[token]
                equal_order = ((a > 0) - (a < 0)) == ((b > 0) - (b < 0))
                if args.changed and equal_order:
                    continue
                print(f"{vocabulary[token]!r:28} {a:14.7f} {b:12.7f}  {equal_order}")
                shown += 1
                if args.top and shown >= args.top:
                    break
            for label, values in [("With update", on), ("Without update", off)]:
                maximum = max(values)
                winners = [vocabulary[t] for t in range(len(values)) if values[t] == maximum]
                print(f"{label} winner set: {winners!r}")
    print(f"\nShowing {min(len(selected), args.limit)}/{len(selected)} matching measurements. "
          "These are next-token observations, not generated completions.")


if __name__ == "__main__":
    main()

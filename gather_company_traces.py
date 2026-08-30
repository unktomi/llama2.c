#!/usr/bin/env python3
"""Collect neutral, inspectable long-context traces from TinyStories.

Each case uses every non-final paragraph as prompt and the final paragraph as
the observed continuation.  The script does not label either the source text
or any later comparison as coherent/corrupt; it only records provenance and
measurements emitted by the C probe.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT.parent / "data" / "TinyStories-valid.txt"
DEFAULT_PROBE = ROOT / "company_probe"
DEFAULT_MODEL = ROOT / "test" / "stories260K.bin"
DEFAULT_TOKENIZER = ROOT / "test" / "tok512.bin"
DEFAULT_OUTPUT = ROOT / "work_traces" / "long_context"


@dataclass(frozen=True)
class SourceCase:
    source_index: int
    prompt: str
    completion: str

    @property
    def characters(self) -> int:
        return len(self.prompt) + len(self.completion)


@dataclass(frozen=True)
class Counts:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    maximum_tokens: int
    fits: bool


COUNT_PATTERN = re.compile(
    r"prompt_tokens=(?P<prompt>\d+) completion_tokens=(?P<completion>\d+) "
    r"total_tokens=(?P<total>\d+) max_context_tokens=(?P<maximum>\d+) "
    r"fits=(?P<fits>[01])"
)

SUMMARY_PATTERN = re.compile(
    r"prompt_tokens=(?P<prompt>\d+) completion_tokens=(?P<completion>\d+) "
    r"completion_log_probability=(?P<logp>[^ ]+) "
    r"delimiter_log_probability=(?P<delimiter>[^ ]+) "
    r"delimiter_rank=(?P<rank>\d+)"
)


def source_cases(data_path: Path) -> list[SourceCase]:
    records = data_path.read_text(encoding="utf-8").split("<|endoftext|>")
    cases: list[SourceCase] = []
    for source_index, record in enumerate(records):
        paragraphs = [line.strip() for line in record.splitlines() if line.strip()]
        if len(paragraphs) < 2:
            continue
        cases.append(
            SourceCase(
                source_index=source_index,
                prompt="\n".join(paragraphs[:-1]),
                completion="\n" + paragraphs[-1],
            )
        )
    return cases


def count_case(
    probe: Path,
    model: Path,
    tokenizer: Path,
    prompt_path: Path,
    completion_path: Path,
) -> Counts:
    result = subprocess.run(
        [
            str(probe),
            str(model),
            str(tokenizer),
            "--files",
            str(prompt_path),
            str(completion_path),
            "--count-only",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    match = COUNT_PATTERN.search(result.stdout)
    if match is None:
        raise RuntimeError(f"unrecognized count output: {result.stdout!r}")
    return Counts(
        prompt_tokens=int(match["prompt"]),
        completion_tokens=int(match["completion"]),
        total_tokens=int(match["total"]),
        maximum_tokens=int(match["maximum"]),
        fits=match["fits"] == "1",
    )


def write_case_inputs(case: SourceCase, directory: Path) -> tuple[Path, Path]:
    prompt_path = directory / "prompt.txt"
    completion_path = directory / "completion.txt"
    prompt_path.write_text(case.prompt, encoding="utf-8")
    completion_path.write_text(case.completion, encoding="utf-8")
    return prompt_path, completion_path


def run_trace(
    probe: Path,
    model: Path,
    tokenizer: Path,
    prompt_path: Path,
    completion_path: Path,
    trace_path: Path,
    checkpoint_every: int,
) -> str:
    result = subprocess.run(
        [
            str(probe),
            str(model),
            str(tokenizer),
            "--files",
            str(prompt_path),
            str(completion_path),
            "--trace",
            str(trace_path),
            "--checkpoint-every",
            str(checkpoint_every),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cases", type=int, default=4)
    parser.add_argument("--min-total-tokens", type=int, default=384)
    parser.add_argument("--max-total-tokens", type=int, default=512)
    parser.add_argument("--checkpoint-every", type=int, default=32)
    parser.add_argument(
        "--target-characters-per-token",
        type=float,
        default=2.0,
        help="scan-order estimate only; exact eligibility uses the C tokenizer",
    )
    parser.add_argument(
        "--scan-limit",
        type=int,
        default=2000,
        help="maximum source records to tokenize while locating long cases",
    )
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if args.cases <= 0 or args.scan_limit <= 0:
        raise SystemExit("--cases and --scan-limit must be positive")
    if args.min_total_tokens <= 0 or args.max_total_tokens < args.min_total_tokens:
        raise SystemExit("invalid token interval")
    if args.checkpoint_every <= 0:
        raise SystemExit("--checkpoint-every must be positive")
    if args.target_characters_per_token <= 0:
        raise SystemExit("--target-characters-per-token must be positive")
    required = (args.data, args.probe, args.model, args.tokenizer)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("missing required paths: " + ", ".join(missing))

    # Character length is used only to start near the requested token range.
    # Eligibility is always determined by the exact C tokenizer count below.
    target_characters = (
        args.max_total_tokens * args.target_characters_per_token
    )
    candidates = sorted(
        source_cases(args.data),
        key=lambda case: abs(case.characters - target_characters),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    scanned = 0
    with tempfile.TemporaryDirectory(prefix="company-trace-") as temporary:
        scratch = Path(temporary)
        for case in candidates:
            if scanned >= args.scan_limit or len(rows) >= args.cases:
                break
            scanned += 1
            if scanned % 100 == 0:
                print(
                    f"scanned={scanned} collected={len(rows)}",
                    flush=True,
                )
            prompt_path, completion_path = write_case_inputs(case, scratch)
            counts = count_case(
                args.probe,
                args.model,
                args.tokenizer,
                prompt_path,
                completion_path,
            )
            if not counts.fits:
                continue
            if not (
                args.min_total_tokens
                <= counts.total_tokens
                <= args.max_total_tokens
            ):
                continue

            case_number = len(rows)
            case_dir = args.output / (
                f"case_{case_number:03d}_source_{case.source_index}_"
                f"tokens_{counts.total_tokens}"
            )
            case_dir.mkdir(parents=True, exist_ok=True)
            case_prompt, case_completion = write_case_inputs(case, case_dir)
            trace_path = case_dir / "trace.jsonl"
            summary = run_trace(
                args.probe,
                args.model,
                args.tokenizer,
                case_prompt,
                case_completion,
                trace_path,
                args.checkpoint_every,
            )
            (case_dir / "summary.txt").write_text(summary, encoding="utf-8")
            match = SUMMARY_PATTERN.search(summary)
            if match is None:
                raise RuntimeError(f"unrecognized probe summary for {case_dir}")
            row: dict[str, object] = {
                "case": case_number,
                "source_index": case.source_index,
                "prompt_tokens": counts.prompt_tokens,
                "completion_tokens": counts.completion_tokens,
                "total_tokens": counts.total_tokens,
                "characters": case.characters,
                "completion_log_probability": float(match["logp"]),
                "mean_completion_log_probability": (
                    float(match["logp"]) / counts.completion_tokens
                ),
                "delimiter_log_probability": float(match["delimiter"]),
                "delimiter_rank": int(match["rank"]),
                "trace": str(trace_path),
            }
            rows.append(row)
            print(
                f"case={case_number} source={case.source_index} "
                f"prompt_tokens={counts.prompt_tokens} "
                f"completion_tokens={counts.completion_tokens} "
                f"total_tokens={counts.total_tokens} trace={trace_path}",
                flush=True,
            )

    if not rows:
        raise SystemExit(
            f"no cases found in [{args.min_total_tokens}, "
            f"{args.max_total_tokens}] after scanning {scanned} records"
        )
    manifest = args.output / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} cases after scanning {scanned}; manifest={manifest}")
    if len(rows) < args.cases:
        print(
            f"warning: requested {args.cases} cases but found {len(rows)}",
            flush=True,
        )


if __name__ == "__main__":
    main()

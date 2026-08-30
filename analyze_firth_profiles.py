#!/usr/bin/env python3
"""Compare retained C company profiles under named text transformations.

The transformation names are descriptive, not judgments of coherence.
"""

from __future__ import annotations

import re
import statistics
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data" / "TinyStories-valid.txt"
PROBE = ROOT / "company_probe"
MODEL = ROOT / "test" / "stories260K.bin"
TOKENIZER = ROOT / "test" / "tok512.bin"


@dataclass
class Profile:
    label: str
    tokens: int
    completion_log_probability: float
    delimiter_log_probability: float
    layer4_dimension: float
    layer4_same_token_similarity: float
    layer4_prior_similarity: float
    company_gain: float = 0.0


def stories(limit: int) -> list[tuple[str, str]]:
    records = DATA.read_text(encoding="utf-8").split("<|endoftext|>")
    selected: list[tuple[str, str]] = []
    for record in records:
        paragraphs = [line.strip() for line in record.splitlines() if line.strip()]
        if len(paragraphs) < 3:
            continue
        prompt = "\n".join(paragraphs[:-1])
        completion = "\n" + paragraphs[-1]
        selected.append((prompt, completion))
        if len(selected) == limit:
            return selected
    raise RuntimeError("not enough multi-paragraph validation stories")


def value(pattern: str, output: str) -> str:
    match = re.search(pattern, output, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"missing profile field: {pattern}")
    return match.group(1)


def probe(label: str, prompt: str, completion: str) -> Profile:
    output = subprocess.run(
        [str(PROBE), str(MODEL), str(TOKENIZER), prompt, completion],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    layer4 = value(r"^(  layer=4 positions=.*)$", output)
    affine4 = value(r"^(  layer=4 affine_effective_dimension=.*)$", output)
    return Profile(
        label=label,
        tokens=int(value(r"completion_tokens=(\d+)", output)),
        completion_log_probability=float(
            value(r"completion_log_probability=([^ ]+)", output)
        ),
        delimiter_log_probability=float(
            value(r"delimiter_log_probability=([^ ]+)", output)
        ),
        layer4_dimension=float(
            value(r"affine_effective_dimension=([^ ]+)", affine4)
        ),
        layer4_same_token_similarity=float(
            value(r"same_token_state_similarity=([^ ]+)", layer4)
        ),
        layer4_prior_similarity=float(
            value(r"prior_state_similarity=([^ ]+)", layer4)
        ),
    )


def contextual_probe(label: str, prompt: str, completion: str) -> Profile:
    actual = probe(label, prompt, completion)
    neutral = probe(label + "-neutral", "Once upon a time.", completion)
    if actual.tokens != neutral.tokens:
        raise RuntimeError("company contrast changed filler tokenization")
    actual.company_gain = (
        actual.completion_log_probability - neutral.completion_log_probability
    ) / actual.tokens
    return actual


def sentences(paragraph: str) -> list[str]:
    return [part for part in re.split(r"(?<=[.!?])\s+", paragraph.strip()) if part]


def shuffled(paragraph: str) -> str:
    parts = sentences(paragraph)
    if len(parts) < 2:
        return paragraph
    return "\n" + " ".join(parts[1:] + parts[:1])


def repeated(paragraph: str) -> str:
    first = sentences(paragraph)[0]
    target = len(paragraph)
    parts: list[str] = []
    while len(" ".join(parts)) < target:
        parts.append(first)
    return "\n" + " ".join(parts)


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def report_group(name: str, profiles: list[Profile]) -> None:
    print(
        f"{name:9s} count={len(profiles):2d} "
        f"tokens={mean([item.tokens for item in profiles]):7.2f} "
        f"mean_logp={mean([item.completion_log_probability / item.tokens for item in profiles]):9.5f} "
        f"delimiter={mean([item.delimiter_log_probability for item in profiles]):9.5f} "
        f"gain={mean([item.company_gain for item in profiles]):9.5f} "
        f"dim4={mean([item.layer4_dimension for item in profiles]):9.5f} "
        f"same4={mean([item.layer4_same_token_similarity for item in profiles]):9.5f} "
        f"prior4={mean([item.layer4_prior_similarity for item in profiles]):9.5f}"
    )


def main() -> None:
    if not all(path.exists() for path in (DATA, PROBE, MODEL, TOKENIZER)):
        raise SystemExit("build company_probe and provide Stories260K artifacts first")
    cases = stories(24)
    references: list[Profile] = []
    rotated_profiles: list[Profile] = []
    cycled_profiles: list[Profile] = []
    next_story_profiles: list[Profile] = []
    for index, (prompt, completion) in enumerate(cases):
        references.append(
            contextual_probe(f"reference-{index}", prompt, completion)
        )
        rotated_profiles.append(
            contextual_probe(f"sentence-rotation-{index}", prompt, shuffled(completion))
        )
        cycled_profiles.append(
            contextual_probe(f"first-sentence-cycle-{index}", prompt, repeated(completion))
        )
        next_story_profiles.append(
            contextual_probe(
                f"next-story-paragraph-{index}",
                prompt,
                cases[(index + 1) % len(cases)][1],
            )
        )

    report_group("reference", references)
    report_group("rotation", rotated_profiles)
    report_group("cycle", cycled_profiles)
    report_group("next-story", next_story_profiles)
    print("paired_directions:")
    for name, variants in (
        ("rotation", rotated_profiles),
        ("cycle", cycled_profiles),
        ("next-story", next_story_profiles),
    ):
        print(
            f"  {name:8s} "
            f"delimiter_reference_higher={sum(a.delimiter_log_probability > b.delimiter_log_probability for a, b in zip(references, variants))}/24 "
            f"gain_reference_higher={sum(a.company_gain > b.company_gain for a, b in zip(references, variants))}/24 "
            f"dim_reference_higher={sum(a.layer4_dimension > b.layer4_dimension for a, b in zip(references, variants))}/24 "
            f"same_reference_lower={sum(a.layer4_same_token_similarity < b.layer4_same_token_similarity for a, b in zip(references, variants))}/24 "
            f"prior_reference_lower={sum(a.layer4_prior_similarity < b.layer4_prior_similarity for a, b in zip(references, variants))}/24"
        )


if __name__ == "__main__":
    main()

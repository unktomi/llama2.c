#!/usr/bin/env python3
"""Train a tiny Llama transformer on a finite agreement grammar.

The generated language has a known predictive state at the main-verb and
cross-sentence-pronoun choices: controller number, together with the finite
construction phase supplied by the generator.  Controller and PP-attractor
templates use the same lexical substitutions as the real grammar cubes.

This program trains and exports a real Transformer from model.py. It does not
implement the projection--injection demand analysis itself; the exported
checkpoint and stock tokenizer are consumed by cps_grammar_cube,
gather_grammar_cubes.py, and analyze_grammar_cubes.py so the known-law control
traverses the same evaluator.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sentencepiece as spm
import torch

from export import legacy_export
from gather_grammar_behaviors import expand_words, read_actions
from gather_grammar_relations import DEFAULT_MANIFEST, read_manifest
from model import ModelArgs, Transformer


ROOT = Path(__file__).resolve().parent
DEFAULT_ACTIONS = ROOT / "grammar_future_actions.json"
DEFAULT_OUTPUT = ROOT / "work_traces" / "synthetic_grammar_stock"


@dataclass(frozen=True)
class ChoiceTest:
    kind: str
    controller_number: str
    prefix: tuple[int, ...]
    expected: int
    alternative: int


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--actions", type=Path, default=DEFAULT_ACTIONS)
    parser.add_argument(
        "--tokenizer-model",
        type=Path,
        default=ROOT / "tokenizer.model",
        help="aligned SentencePiece model whose sibling .bin is used by llama2.c",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--force-train", action="store_true")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def all_templates_and_families(
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    templates = [
        *manifest["exploration_templates"],
        *manifest["confirmation_templates"],
    ]
    families = [
        *manifest["exploration_families"],
        *manifest["confirmation_families"],
    ]
    return templates, families


def plural_dictionary(families: list[dict[str, Any]]) -> dict[str, str]:
    result = {
        str(family["target"]): str(family["target_plural"])
        for family in families
    }
    missing = sorted(
        {
            str(family["context"])
            for family in families
            if str(family["context"]) not in result
        }
    )
    require(not missing, "synthetic grammar lacks plurals for: " + ", ".join(missing))
    return result


def sentence(
    template: dict[str, Any],
    family: dict[str, Any],
    role: str,
    context_number: str,
    target_number: str,
    plurals: dict[str, str],
) -> tuple[str, str]:
    context = str(family["context"])
    target = str(family["target"])
    if context_number == "plural":
        context = plurals[context]
    if target_number == "plural":
        target = str(family["target_plural"])
    controller_number = target_number if role == "controller" else context_number
    verb = str(
        family[
            "verb_plural" if controller_number == "plural" else "verb_singular"
        ]
    )
    rendered = str(template[role]).format(
        context=context,
        target=target,
        verb=verb,
    )
    return rendered, controller_number


def build_corpus(
    manifest: dict[str, Any],
    actions: dict[str, Any],
) -> tuple[list[str], list[dict[str, str]]]:
    templates, families = all_templates_and_families(manifest)
    plurals = plural_dictionary(families)
    words = expand_words(actions)
    primitive = {word.name: word for word in words if len(word.factors) == 1}
    composed = [word for word in words if len(word.factors) == 2]
    independent = [
        word
        for word in primitive.values()
        if word.name not in {"plural_pronoun", "singular_pronoun"}
    ]
    examples: set[str] = set()
    records: list[dict[str, str]] = []
    for template in templates:
        for family in families:
            for role in ("controller", "attractor"):
                for context_number in ("singular", "plural"):
                    for target_number in ("singular", "plural"):
                        base, controller_number = sentence(
                            template,
                            family,
                            role,
                            context_number,
                            target_number,
                            plurals,
                        )
                        examples.add(base)
                        records.append(
                            {
                                "base": base,
                                "controller_number": controller_number,
                                "role": role,
                                "template": str(template["name"]),
                                "family": str(family["name"]),
                                "verb_singular": str(family["verb_singular"]),
                                "verb_plural": str(family["verb_plural"]),
                            }
                        )
                        pronoun_name = (
                            "plural_pronoun"
                            if controller_number == "plural"
                            else "singular_pronoun"
                        )
                        allowed = [primitive[pronoun_name], *independent]
                        for word in allowed:
                            examples.add(base + word.text)
                        allowed_names = {word.name for word in allowed}
                        for word in composed:
                            if set(word.factors) <= allowed_names:
                                examples.add(base + word.text)
    return sorted(examples), records


def encode_corpus(
    processor: spm.SentencePieceProcessor,
    corpus: list[str],
) -> list[list[int]]:
    sequences = [
        [processor.bos_id(), *processor.encode(text), processor.eos_id()]
        for text in corpus
    ]
    require(all(len(sequence) >= 3 for sequence in sequences), "empty synthetic sequence")
    return sequences


def differing_choice(
    processor: spm.SentencePieceProcessor,
    left: str,
    right: str,
) -> tuple[tuple[int, ...], int, int]:
    left_tokens = [processor.bos_id(), *processor.encode(left)]
    right_tokens = [processor.bos_id(), *processor.encode(right)]
    common = 0
    while (
        common < len(left_tokens)
        and common < len(right_tokens)
        and left_tokens[common] == right_tokens[common]
    ):
        common += 1
    require(common > 0, "choice variants have no token prefix")
    require(
        common < len(left_tokens) and common < len(right_tokens),
        "choice variants tokenize identically",
    )
    return (
        tuple(left_tokens[:common]),
        int(left_tokens[common]),
        int(right_tokens[common]),
    )


def replace_last_word(text: str, old: str, new: str) -> str:
    position = text.rfind(old)
    require(position >= 0, f"{old!r} is absent from {text!r}")
    return text[:position] + new + text[position + len(old) :]


def build_choice_tests(
    processor: spm.SentencePieceProcessor,
    records: list[dict[str, str]],
    actions: dict[str, Any],
) -> list[ChoiceTest]:
    primitive = {
        word.name: word
        for word in expand_words(actions)
        if len(word.factors) == 1
    }
    tests: dict[tuple[Any, ...], ChoiceTest] = {}
    for record in records:
        number = record["controller_number"]
        singular_text = replace_last_word(
            record["base"],
            record[
                "verb_plural" if number == "plural" else "verb_singular"
            ],
            record["verb_singular"],
        )
        plural_text = replace_last_word(
            record["base"],
            record[
                "verb_plural" if number == "plural" else "verb_singular"
            ],
            record["verb_plural"],
        )
        prefix, singular_token, plural_token = differing_choice(
            processor,
            singular_text,
            plural_text,
        )
        expected, alternative = (
            (plural_token, singular_token)
            if number == "plural"
            else (singular_token, plural_token)
        )
        test = ChoiceTest("main_verb", number, prefix, expected, alternative)
        tests[(test.kind, test.controller_number, test.prefix, expected, alternative)] = test

        singular_extension = record["base"] + primitive["singular_pronoun"].text
        plural_extension = record["base"] + primitive["plural_pronoun"].text
        prefix, singular_token, plural_token = differing_choice(
            processor,
            singular_extension,
            plural_extension,
        )
        expected, alternative = (
            (plural_token, singular_token)
            if number == "plural"
            else (singular_token, plural_token)
        )
        test = ChoiceTest("future_pronoun", number, prefix, expected, alternative)
        tests[(test.kind, test.controller_number, test.prefix, expected, alternative)] = test
    return list(tests.values())


def padded_batch(
    sequences: list[list[int]],
    indices: list[int],
    pad: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected = [sequences[index] for index in indices]
    maximum = max(len(sequence) for sequence in selected) - 1
    inputs = torch.full((len(selected), maximum), pad, dtype=torch.long)
    targets = torch.full((len(selected), maximum), -1, dtype=torch.long)
    for row, sequence in enumerate(selected):
        length = len(sequence) - 1
        inputs[row, :length] = torch.asarray(sequence[:-1], dtype=torch.long)
        targets[row, :length] = torch.asarray(sequence[1:], dtype=torch.long)
    return inputs, targets


@torch.no_grad()
def evaluate_choices(
    model: Transformer,
    tests: list[ChoiceTest],
    device: torch.device,
) -> dict[str, Any]:
    grouped: dict[int, list[tuple[int, ChoiceTest]]] = {}
    for index, test in enumerate(tests):
        grouped.setdefault(len(test.prefix), []).append((index, test))
    margins = np.empty(len(tests), dtype=np.float64)
    correct = np.empty(len(tests), dtype=np.bool_)
    model.eval()
    for group in grouped.values():
        tokens = torch.tensor([test.prefix for _, test in group], device=device)
        logits = model(tokens)[:, 0, :]
        for row, (index, test) in enumerate(group):
            margin = float(
                (logits[row, test.expected] - logits[row, test.alternative])
                .detach()
                .cpu()
            )
            margins[index] = margin
            correct[index] = margin > 0.0
    result: dict[str, Any] = {}
    for kind in sorted({test.kind for test in tests}):
        selected = np.asarray(
            [index for index, test in enumerate(tests) if test.kind == kind]
        )
        values = margins[selected]
        result[kind] = {
            "tests": len(selected),
            "accuracy": float(np.mean(correct[selected])),
            "minimum_margin": float(values.min()),
            "mean_margin": float(values.mean()),
            "maximum_margin": float(values.max()),
        }
    return result


def observer_file(
    processor: spm.SentencePieceProcessor,
    grammar_tokens: set[int],
    output: Path,
) -> Path:
    path = output / "synthetic-observers.tsv"
    with path.open("w", encoding="utf-8") as destination:
        destination.write("# Every token constructor occurring in the finite grammar except BOS.\n")
        for token in sorted(grammar_tokens - {processor.bos_id()}):
            destination.write(f"{token}\t{processor.id_to_piece(token)}\n")
    return path


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "mps":
        require(torch.backends.mps.is_available(), "MPS was requested but is unavailable")
    return torch.device(requested)


def main() -> None:
    args = arguments()
    require(args.steps > 0 and args.batch_size > 0, "training parameters must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(args.manifest)
    actions = read_actions(args.actions)
    corpus, records = build_corpus(manifest, actions)
    tokenizer_model = args.tokenizer_model.resolve()
    tokenizer_bin = tokenizer_model.with_suffix(".bin")
    require(tokenizer_model.is_file(), f"missing tokenizer model: {tokenizer_model}")
    require(tokenizer_bin.is_file(), f"missing tokenizer binary: {tokenizer_bin}")
    (args.output / "synthetic-corpus.txt").write_text(
        "\n".join(corpus) + "\n",
        encoding="utf-8",
    )
    processor = spm.SentencePieceProcessor(model_file=str(tokenizer_model))
    sequences = encode_corpus(processor, corpus)
    tests = build_choice_tests(processor, records, actions)
    maximum_length = max(len(sequence) for sequence in sequences)
    require(maximum_length <= 64, f"synthetic sequence length {maximum_length} exceeds 64")

    params = ModelArgs(
        dim=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=4,
        vocab_size=processor.vocab_size(),
        hidden_dim=176,
        multiple_of=16,
        max_seq_len=64,
        dropout=0.0,
    )
    device = choose_device(args.device)
    checkpoint = args.output / "synthetic.pt"
    model = Transformer(params)
    history: list[dict[str, Any]] = []
    if checkpoint.is_file() and not args.force_train:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(saved)
    else:
        model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=3e-3,
            betas=(0.9, 0.95),
            weight_decay=0.01,
        )
        generator = random.Random(args.seed)
        model.train()
        for step in range(1, args.steps + 1):
            indices = [generator.randrange(len(sequences)) for _ in range(args.batch_size)]
            # The llama2.c tokenizer ABI reserves only UNK/BOS/EOS before the
            # byte-fallback tokens, so this tokenizer deliberately has no PAD
            # id.  These input slots are loss-masked; EOS is a valid inert fill.
            inputs, targets = padded_batch(sequences, indices, processor.eos_id())
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            model(inputs, targets)
            loss = model.last_loss
            require(loss is not None, "training forward did not produce loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step == 1 or step % 100 == 0 or step == args.steps:
                sampled_tests = tests[:: max(1, len(tests) // 128)]
                metrics = evaluate_choices(model, sampled_tests, device)
                row = {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "choice_metrics": metrics,
                }
                history.append(row)
                print(json.dumps(row), flush=True)
                model.train()
        model.to("cpu")
        torch.save(model.state_dict(), checkpoint)

    model.to(device)
    final_metrics = evaluate_choices(model, tests, device)
    require(
        all(row["accuracy"] == 1.0 for row in final_metrics.values()),
        "trained control did not learn every exhaustive grammatical choice",
    )
    model.to("cpu")
    model.eval()
    model_path = args.output / "synthetic-model.bin"
    legacy_export(model, str(model_path))
    observers = observer_file(
        processor,
        {token for sequence in sequences for token in sequence},
        args.output,
    )
    report = {
        "schema_version": 1,
        "artifact": "trained_finite_agreement_grammar_control",
        "seed": args.seed,
        "device": str(device),
        "corpus_sentences": len(corpus),
        "choice_tests": len(tests),
        "known_predictive_state": {
            "controller_number_values": ["singular", "plural"],
            "construction_phase": "finite state supplied mechanically by the grammar generator",
            "role_rule": "target number controls in controller templates; context number controls in attractor templates",
        },
        "model": {
            "dim": params.dim,
            "layers": params.n_layers,
            "heads": params.n_heads,
            "vocab": params.vocab_size,
            "maximum_sequence_length_observed": maximum_length,
        },
        "training_history": history,
        "exhaustive_choice_metrics": final_metrics,
        "paths": {
            "model": str(model_path.resolve()),
            "tokenizer": str(tokenizer_bin),
            "observer_tokens": str(observers.resolve()),
        },
    }
    (args.output / "training-report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(final_metrics, indent=2))


if __name__ == "__main__":
    main()

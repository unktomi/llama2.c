# Reverse-causal company readout

`reverse_company_readout.c` tests whether frozen llama activations contain a
company distinction that eager next-token projection misses. It does not use
AR likelihood as a target and does not assign a scalar score to a sequence.

## Observation

Each example contains two complete token companies that differ at one hole:

- the word occurring in a real TinyStories validation story;
- another complete corpus word that the frozen model's local AR logit ranks
  above it.

AR is used only to select a difficult alternative. The label comes from the
observed corpus filler. An alternative must occur as a complete word at least
twice in the corpus; tokenizer fragments are excluded.

The frozen transformer is applied separately to the two instantiated
companies. The executable retains the embedding residual and every layer's
residual stream at every position. No gradient enters the transformer.

For a tested filler at position `i`, a learned reverse recurrence reads the
frozen residuals at positions `j > i` from right to left. Those downstream
states have causally incorporated the consequences of the filler. Its reverse
message is combined with the frozen left-context residuals and the frozen
embedding of the tested filler. The head emits one compatibility coordinate
for that instantiated filler.

Training uses only the paired difference

```text
compatibility(corpus filler) - compatibility(replacement filler).
```

No coordinates from distinct holes are added or averaged. Three readouts use
the same head width, optimizer, and allocated parameter layout:

- `left`: left context and tested filler, with no suffix;
- `embedding`: the same inputs plus the future token embeddings;
- `reverse`: the same inputs plus the embedding and every frozen transformer
  layer at every future position.

The controls leave some allocated matrix columns inactive, so equal allocation
is not an equal-effective-capacity claim. The executable prints the active
coefficient counts to make that distinction explicit.

## Reproduction

Build:

```sh
make reversecompany CC=clang
```

The measured 260K run used:

```sh
./reverse_company_readout \
  ../llama2.c/test/stories260K.bin \
  ../llama2.c/test/tok512.bin \
  ../data/TinyStories-valid.txt \
  --train 1024 --validation 256 --tokens 64 --top-k 8 \
  --head-dim 32 --epochs 20 --batch 16 --shown 24 \
  --trace reverse-company-260k-final.jsonl \
  --save reverse-company-260k-head.bin
```

The checkpoint and trace paths are local artifacts and are intentionally not
committed. Every JSONL observation is flushed immediately.

The base model had 260K parameters. Each readout allocated 27,745 coefficients.
The active counts were 14,401 for `left`, 17,505 for `embedding`, and 27,745
for `reverse`. The best validation-loss checkpoints produced:

```text
frozen local AR preference:   10 / 256   ( 3.91%)
left-only learned control:   179 / 256   (69.92%)
embedding-suffix control:    185 / 256   (72.27%)
reverse-company readout:     209 / 256   (81.64%)
```

Paired `reverse` versus `embedding` outcomes were:

```text
both correct:        176
reverse only:         33
embedding only:        9
both wrong:           38
```

The AR number is low by construction: the data builder preferentially retains
holes where AR ranks the replacement above the corpus word. The relevant
comparisons are among the three learned readouts.

## What this establishes

On this split and training procedure, the all-layer trace decodes the held-out
corpus-vs-replacement distinction better than either left context alone or
future token identity. The paired result is not just a tie shift: `reverse`
corrects 33 decisions missed by `embedding`, while losing 9 in the opposite
direction. This is evidence consistent with useful information in the frozen
model's later reactions, rather than a claim that the control has eliminated
every capacity confound.

It does **not** yet establish that:

- the readout generates coherent complete stories;
- a corpus filler is uniquely preferable to every replacement (some labels
  remain genuinely ambiguous);
- the gain over the embedding control is entirely representational rather than
  partly due to the larger active input matrix;
- the head is already composed into selection strength;
- the hidden-feedback tape alone supplies the same candidate-specific
  downstream reactions;
- the current recurrence is the final multiscale architecture.

The immediate integration target is to expose the per-filler compatibility
coordinate as a structured observer for each instantiated branch. Phrase and
longer-span fillers should be trained and retained as their own coordinates,
not reduced into token-score sums.

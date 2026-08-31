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

For long-continuation experiments, `--min-suffix N` requires at least `N`
tokens after the tested filler. The trace records both `token_count` and
`suffix_tokens`, so increasing the context ceiling cannot silently select only
short continuations.

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

## Long-continuation measurements

Two further runs constrained the tested filler to have a genuinely long
continuation. They retained the same 1,024/256 split sizes, top-8 hard-negative
construction, width 32, 20 epochs, batch 16, and seed 42.

```text
context cap  required suffix  observed suffix       left    embedding  reverse
128 tokens   >= 64 tokens     64..125, mean 89.46   72.27%  75.00%     82.42%
256 tokens   >= 128 tokens    128..253, mean 183.73 64.06%  61.33%     72.66%
```

For 128 tokens, `reverse` made 36 decisions that `embedding` missed and lost
17 in the other direction. For 256 tokens those counts were 45 and 16.

The lower absolute 256-token result is not a monotonic suffix-length collapse
inside that validation set. `reverse` accuracy by actual suffix bin was
54/76, 47/69, 59/77, and 26/34 for suffixes 128–159, 160–191, 192–223, and
224–253 respectively. The 128- and 256-token builders necessarily select
different stories and holes, so this experiment does not by itself separate a
harder example distribution from optimization or recurrence limitations.

These are complete, teacher-forced corpus/corruption companies. They test the
observer across long future context; they are not generated completions from
an integrated selection decoder.

## Candidate-wide trace

The evaluator retains the observed filler plus every complete-word alternative
in the requested `--top-k` support. After training, each alternative is
instantiated in the same hole and receives its own frozen forward continuation.
The JSONL contains one immediately flushed `company_candidate` event per
alternative with:

- fully decoded candidate text;
- raw `reverse`, `embedding`, `left`, and AR coordinates;
- deltas from the observed filler;
- within-hole ranks for all four coordinates;
- observed and selected-training-negative flags.

Raw coordinates are comparable only inside one hole. To browse a hole in
reverse-score order:

```sh
jq -s '[.[] | select(.event == "company_candidate" and .pair == 0)] |
       sort_by(.reverse_rank) |
       map({rank: .reverse_rank, piece, score: .reverse_score,
            observed, text})' TRACE.jsonl
```

Coverage runs used 1,024 held-out holes at each of three context scales and
top-16 alternatives plus the observed filler:

```text
context/suffix  holes  candidate companies  binary win  observed top-1  top-3  mean rank/17
64 / >=16       1024   17408                83.69%      29.20%          56.74% 4.59
128 / >=64      1024   17408                83.30%      28.22%          54.79% 4.68
256 / >=128     1024   17408                78.52%      21.68%          46.58% 5.45
```

The support-wide ordering exposes a bug hidden by the binary metric. For
example, the 64-token run scores `room for see` at 4.263080, above observed
`room to see` at 3.315738. The 256-token run ranks `visit he library` first at
-2.589438, observed `visit the library` fourteenth at -4.820989, and grammatical
`visit a library` seventeenth at -5.680669.

This is not a conclusion from a handful of displayed examples. Across the
three traces, the observed filler ranks first in 810/3,072 holes (26.37%). A
stable filler bias is also visible: `for` wins 299 supports, whereas `a` wins
16 despite occurring in 3,040 supports and being observed in 116 holes.

The concrete objective mismatch is in training: each hole constrains only
`observed_score - selected_negative_score`. The other support alternatives
receive no loss at all. Thus 81.84% success against the single sampled negative
coexists with the incorrect support-wide ordering above. These measurements
invalidate the current scorer as the requested Firth observer; they do not
show that the frozen transformer lacks the desired information.

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

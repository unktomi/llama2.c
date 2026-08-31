# Reverse-causal company readout

`reverse_company_readout.c` tests whether frozen llama activations contain a
company distinction that eager next-token projection misses. It does not use
AR likelihood as a target and does not assign a scalar score to a sequence.

## Observation

Each example contains a support of complete token companies that differ at one
hole:

- the word occurring in a real TinyStories validation story;
- the frozen model's top-k alternative complete corpus words.

AR is used only to select a difficult hole and its finite candidate support.
The label comes from the observed corpus filler. An alternative must occur as
a complete word at least twice in the corpus; tokenizer fragments are excluded.

The frozen transformer is applied separately to every instantiated company.
The executable retains the embedding residual and every layer's residual
stream at every position. No gradient enters the transformer.

For a tested filler at position `i`, a learned reverse recurrence reads the
frozen residuals at positions `j > i` from right to left. Those downstream
states have causally incorporated the consequences of the filler. Its reverse
message is combined with the frozen left-context residuals and the frozen
embedding of the tested filler. The head emits one compatibility coordinate
for that instantiated filler.

The default training objective compares every filler in one hole at once:

```text
loss = logsumexp(compatibility(each supported filler))
       - compatibility(observed filler).
```

Adding a constant to all coordinates in the hole leaves this loss unchanged.
No coordinates from distinct holes or token positions are composed into a
sequence reward. `--objective pairwise` retains the former one-rival loss for
an exact baseline. Three readouts use the same head width and allocated
parameter layout:

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

The corrected 260K support-wide run used:

```sh
./reverse_company_readout \
  ../llama2.c/test/stories260K.bin \
  ../llama2.c/test/tok512.bin \
  ../data/TinyStories-valid.txt \
  --train 1024 --validation 1024 --tokens 64 --min-suffix 16 --top-k 16 \
  --head-dim 32 --epochs 20 --batch 16 --shown 1 --seed 42 \
  --objective support \
  --trace reverse-company-support-64-s16-k16-seed42-1024.jsonl \
  --save reverse-company-support-64-s16-k16-seed42-1024-head.bin
```

The checkpoint and trace paths are local artifacts and are intentionally not
committed. Every JSONL observation is flushed immediately.

For long-continuation experiments, `--min-suffix N` requires at least `N`
tokens after the tested filler. The trace records both `token_count` and
`suffix_tokens`, so increasing the context ceiling cannot silently select only
short continuations.

The base model has 260K parameters. Each readout allocates 27,745 coefficients.
The active counts were 14,401 for `left`, 17,505 for `embedding`, and 27,745
for `reverse`. The earlier top-8 pairwise experiment produced:

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

These historical pairwise-objective runs constrained the tested filler to have
a genuinely long continuation. They retained the same 1,024/256 split sizes,
top-8 hard-negative construction, width 32, 20 epochs, batch 16, and seed 42.

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
- reverse-head hybrid coordinates that change only the filler embedding or
  only the candidate-specific suffix reaction;
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

The pairwise-objective coverage runs used 1,024 held-out holes at each of three
context scales and top-16 alternatives plus the observed filler:

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
invalidate the pairwise scorer as the requested Firth observer; they do not
show that the frozen transformer lacks the desired information.

The original support builder also allocated one choice slot per sequence token
while appending up to `top_k` choices per token. A corrected 64-token/top-16
replay measured 67 choices in one training story, exceeding the old 64-slot
allocation. The storage is now sized to `tokens * top_k`, guarded at every
append, and the actual count is recorded as `viable_choices`. Address and
undefined-behavior sanitizers pass the corrected 256-token/top-16 path. After
removing the new diagnostic field, the complete corrected 64-token trace is
identical to the earlier trace, so the overflow was real undefined behavior but
did not cause the ranking failure measured here.

The corrected trace localizes that failure more sharply. The selected training
negative is below the observed filler in 857/1,024 holes (83.69%). Yet the
observed filler is top-1 in only 299/1,024 holes. Of the 725 wrong winners, 700
are alternatives that were not the selected training negative. Across all
unselected alternatives, 3,507/15,360 outrank the observed filler.

For a 256-hole diagnostic slice, two hybrid scores separate the tested filler
from its downstream reaction:

- `reverse_token_only_score` changes the filler while retaining the observed
  company's suffix activations;
- `reverse_suffix_only_score` retains the observed filler while substituting
  the candidate company's suffix activations.

In `room to see`, the malformed `for` filler receives +2.006783 from the
filler-only hybrid while the suffix-only hybrid changes the score by -1.088884.
These are nonlinear interventions, not additive components. The combined head
still gives `for see` +0.947342 over `to see`. Across the 181
wrong top-1 decisions in this slice, 64 have this same sign pattern: the suffix
reaction rejects the rival but the filler branch overrules it. The suffix
hybrid also prefers the wrong rival in the other 117 cases, so deleting the
filler branch is not sufficient; it raises observed top-1 only from 75/256 to
86/256.

`--load HEAD.bin` reloads a compatible saved reverse head for repeatable
hybrid tracing without retraining that head. The controls are still trained
for the requested run, and their epoch count remains controlled by `--epochs`.

## Full-support objective

The default `--objective support` retains every candidate-specific frozen
continuation during dataset construction and trains the reverse head with the
single within-hole categorical loss shown above. `--objective pairwise`
reproduces the former selected-negative loss. On an identical 256/256 split,
width, support, seed, and optimization schedule, the change produced:

```text
objective  selected-negative win  observed top-1
pairwise                    73.44%           23.05% (59/256)
support                     74.22%           40.62% (104/256)
```

The nearly unchanged binary result demonstrates why it was not a useful proxy
for the actual selection. Hole by hole, the support objective fixes 59
pairwise failures and regresses 14 pairwise successes.

On the original 1,024/1,024 split, the corrected objective can be compared
directly to the complete corrected pairwise trace:

```text
objective  selected-negative win  observed top-1  top-3  mean rank/17
pairwise                    83.69%           29.20% 56.74%         4.588
support                     82.03%           51.66% 76.66%         2.707
```

The support objective fixes 288 pairwise failures and regresses 58 pairwise
successes, a net gain of 230 top-1 holes. It also fixes the concrete
`room for see` failure: observed `to` moves from rank 3 to rank 1 and `for`
moves from rank 1 to rank 4.

This repairs the demonstrated support-coverage bug; it does not make the
observer sound. At the selected epoch, training top-1 is 96.58% while held-out
top-1 is 51.66%. Of the 495 remaining wrong winners, the suffix-only hybrid
also prefers the wrong rival in 359 cases. The worst retained candidates still
include malformed companies such as `made the stretching` where the corpus has
`made for stretching`, and `park with her mom was too busy` where the corpus
has `park but her mom was too busy`. Those failures now occur after every
rival in each training-hole support has participated in the objective. They
point to generalization and the flat reverse fold as the next fault boundary,
rather than the repaired single-rival loss.

## What this establishes

In the historical top-8 paired experiment, the all-layer trace decodes the
held-out corpus-vs-selected-replacement distinction better than either left
context alone or future token identity. The paired result is not just a tie
shift: `reverse` corrects 33 decisions missed by `embedding`, while losing 9 in
the opposite direction. This is evidence consistent with useful information
in the frozen model's later reactions, rather than a claim that the control
has eliminated every capacity confound. The full-support experiment separately
establishes that the one-rival loss caused a large part of the malformed
ordering: matching the loss to the retained support raises observed top-1 from
29.20% to 51.66% on identical holes.

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

The next diagnostic target is the scale at which the remaining flat reverse
fold loses a correct ordering. Phrase and longer-span fillers need retained
observers of their own rather than reduction into token-score sums; that
hierarchical composition is not implemented here yet.

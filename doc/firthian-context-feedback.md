# History-dependent company selection

`run_hidden_feedback_select.c` is the active finite, sampled selection-product
path. It does not insert a final-layer hidden state directly at the next
position's embedding boundary. That identity crossing is quarantined because
it preserved coarse subject matter while producing low-scale failures such as
`a a a sick sick`.

## The term

The prompt positions are `Select` units. A generated position is a
history-dependent selection whose available carrier is the vocabulary:

```text
epsilon_h : (Token -> Outcome) -> Token
```

The model covector at prefix `h` is used only to propose the next previously
undemanded argument of `epsilon_h`. It does not emit that token. Applying the
continuation embeds the proposed constructor, extends the causal company, and
suspends at the next selection. The longer prefix must be observed before that
next selection can demand an argument.

This is the strict-C organization of Escardo's history-dependent product:

```text
a    = epsilon_h (x -> p (x : b(x)))
b(x) = product_(h ++ [x]) (xs -> p (x : xs))
result = a : b(a)
```

The demanded finite subtree is memoized by `SelectNode`. A sweep adds one
previously undemanded continuation to every selection already reached at the
start of that sweep. A selection first reached during the sweep demands one
continuation so its caller can be rated. `-k K` bounds demands at any one
selection; it is not an exhaustive top-K grid and does not claim exact search
of the vocabulary product.

## Company observation and strength

`llama_company_evaluate` lowers every currently demanded prefix to one causal
row family. All rows in that batch cross each learned filler together. A new
causal frontier requires another family application; the executable reports
`company_batches` and `maximum_calls_per_filler` explicitly.

For each complete leaf `xs`, the observer retains the full position-indexed
outcome:

```text
Q_i(xs) = logsoftmax(logits(context_i))[token_i]
q(xs)   = [Q_0(xs), ..., Q_(n-1)(xs)]
```

Prompt and completion coordinates remain in the same outcome. Coordinates are
not summed or averaged. The current total order sorts each outcome from its
least approving coordinate upward, then compares those vectors
lexicographically (leximin). This is still a causal model observer, not a
claim that every possible Firthian, bidirectional company judgment has been
recovered.

`force_select` recursively applies every reached local selection to the backed
outcomes of its children. Each node memoizes the selected child and leaf.
Strength performs no model call or learned-weight read. Only the synthetic root
turns the retained outcome into a token sequence.

## Measured boundary correction

For Stories15M, prompt `Lily was`, six completion positions, two demand sweeps,
and seed 42:

```sh
./run_hidden_feedback_select test/stories15M.bin \
  -z tokenizer.bin -i "Lily was" -l 6 -k 2 -s 42 \
  -o candidates.jsonl
```

the demanded subtree has seven leaves and selects:

```text
Lily was a little girl who liked to
```

The complete worst-first ordering is:

```text
a little girl who liked to
a little girl who loved to
very happy. She was going
a big girl who liked to
a little bird who liked to
a little girl who liked c
a little girl. She liked
```

This run uses 15 company batches (13 frontier refreshes and two terminal
observations), so the maximum learned-filler call count is 15. It is a semantic
correction and an inspectable sampled result, not the original one-physical-
read performance goal.

The corresponding 16-position Stories260K control must use its 512-token
tokenizer:

```sh
./run_hidden_feedback_select test/stories260K.bin \
  -z test/tok512.bin -i "Lily was" -l 16 -k 2 -s 42 \
  -o candidates-260k.jsonl
```

It retains 17 complete leaves and selects:

```text
Lily was a little girl named Lily who loved to play with her
```

The executable rejects prompt token IDs outside the checkpoint vocabulary, so
accidentally pairing Stories260K with the 32K tokenizer now reports an
incompatibility instead of dereferencing token `-1`.

## Trace

Every JSONL record is flushed immediately. Relevant records are:

- `continuation_demand`: decoded token, owning prefix, causal position, and
  its rank in that prefix's proposal covector;
- `company_run`: phase, rows, learned-filler counts, and model time;
- `company_outcome`: every decoded complete candidate, its position scores,
  and the full worst-first vector;
- `candidate_rated`: every local continuation application and the complete
  backed leaf it returned;
- `select`: the retained continuation at each memoized selection;
- `root_terminalized`: the one emitted witness and its complete outcome.

The trace is the acceptance evidence. A fluent selected string does not hide
bad alternatives or an implausible ordering; if the recorded order disagrees
with the expected company judgment, that is an observer/runtime bug to expose.

## Quarantined failures

`prebuilt-sampled-path-runtime-don't-do-this.c` samples complete paths before
local selections receive their observer. `resumed-neutral-suffix-runtime-
don't-do-this.c` resumes earlier selections but fills new prefixes with an
unobserved fixed-tape suffix. `intermediate-layer-unembedding-observer-
don't-do-this.c` shows why applying the final output head at every intermediate
layer is not a valid lower-scale reward: the embedding projection nearly
assigns probability one to repeated copies of the current token.

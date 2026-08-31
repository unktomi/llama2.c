# Firthian context feedback

`run_hidden_feedback_select.c` contains the active finite selection-product
path. The hidden-feedback recurrence proposes a finite carrier; it does not
select or feed back a token. The demanded token term is evaluated separately
by `llama_company_evaluate`.

## Completion-indexed outcomes

The global callback does not return a path sum or one terminal covector. For
each complete demanded branch `xs`, it returns a position-indexed outcome:

```text
q(xs) = [Q_0(xs), ..., Q_(n-1)(xs)]
```

Each `Q_i(xs)` is one normalized constructor/company coordinate from the
model-produced covector at that position. At an internal position it observes
the selected successor in the current occurrence's covector. At the finite
right boundary it observes the final constructor in its incoming covector.
The trace records the complete vector and every context/company binding in a
flushed `company_outcome` event.

The outcome is retained as `ProjectionTermOutcome`. A local Select at position
`i` forces its candidate-specific suffix and reads only coordinate `i` from
the complete outcome `q(x : b(x))`. It then propagates the winning outcome
unchanged. Coordinates from different positions are never added, averaged,
geometrically combined, or converted to a path score.

## Memoized strength

`ProjectionTermNode.selection_state` is the memo cell for the recursive
selection rooted at that node:

```text
b(x)   = force the selection below x once
a      = Select_i (x -> q(x : b(x))[i])
result = a : b(a)
```

The node stores both `selected_leaf` and `selected_child`. Consequently the
suffix and complete outcome used while comparing `a` are the exact objects
returned after `a` is selected; neither is recomputed. Sampling changes which
token-indexed cells exist in the finite term, not this recursion.

## One family application

The complete demanded occurrence tree is lowered to `LlamaCompanyShape`.
`llama_company_evaluate` applies embedding, RMS, Q/K/V, attention output, MLP,
final RMS, and output-head fillers to their whole row families. The runtime
checks `maximum_calls_per_filler == 1` for this phase. Strength runs afterward
over immutable outcomes and aborts if it causes any learned-filler call or
scalar weight read.

## Finite support

`-k` bounds the local proposal carrier and `-b` bounds complete demanded
leaves. They are distinct. For example, a three-token completion with `-k 4
-b 64` contains the complete `4 x 4 x 4` product: 4 ratings at the first
position, 16 at the second, and 64 at the third.

```sh
./run_hidden_feedback_select test/stories260K.bin \
  -z test/tok512.bin -i "Lily was" -n 6 \
  -r company -k 4 -b 64 -s 42 -o candidates.jsonl
```

The proposal carrier is still produced by independently unembedding the fixed
hidden-feedback tape. That is a known semantic limitation, not a solved
quality result. In the measured 15-token Stories260K run, exact binary support
formed all `2^15` leaves and strength was genuinely non-unary at every
position, but the later carrier contained mostly character fragments such as
`c` and `w`; backward induction cannot select a coherent token absent from its
carrier. Making demand continuation-sensitive without returning to repeated
eager model passes is the next representation problem.

## Trace

The JSONL trace is flushed after every event. Relevant records are:

- `selection_term_built`: demanded rows and leaves;
- `company_run`: learned-filler counts and model time;
- `company_outcome`: decoded branch, full coordinate vector, and bindings;
- `candidate_rated`: every local candidate and the coordinate actually used;
- `select`: the retained child and outcome at one memoized node;
- `root_terminalized`: the single emitted witness and its full outcome;
- `strength_run`: proof that strength performed no learned-filler work.

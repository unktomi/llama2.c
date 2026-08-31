# Firthian context feedback

`run_hidden_feedback_select.c` contains the active finite selection-product
path. The hidden-feedback recurrence proposes a finite carrier; it does not
select or feed back a token. The demanded token term is evaluated separately
by `llama_company_evaluate`.

## Completion-indexed outcomes

For each complete demanded branch `xs`, the model callback returns the entire
position-indexed outcome

```text
q(xs) = [Q_0(xs), ..., Q_(n-1)(xs)].
```

For every non-root occurrence, `Q_i(xs)` is the normalized observation of the
candidate token in the covector at its incoming causal context:

```text
Q_i(xs) = logsoftmax(logits(context_i))[token_i].
```

The first occurrence has no incoming context inside the term and retains a
neutral coordinate. Prefill coordinates and completion coordinates remain in
the same outcome. The trace records the complete vector and every
context/company binding in a flushed `company_outcome` event.

The callback result is not reduced to the final token and its coordinates are
never added or averaged. `logsoftmax` supplies the explicit probability
normalization used to compare them. Outcomes are ordered by the literal
least-disliked-company order: sort each outcome's coordinates from least
approving to most approving, maximize the least approving coordinate, then the
second least, and so on. This is a leximin order on the complete outcome, not
a scalar path reward.

## Memoized strength

`ProjectionTermNode.selection_state` is the memo cell for the recursive
selection rooted at that node:

```text
b(x)   = force the selection below x once
a      = Select(x -> q(x : b(x))) under the common leximin outcome order
result = a : b(a)
```

Every local Select receives the same kind of complete callback result. A node
stores both `selected_leaf` and `selected_child`, so the suffix and outcome
used while comparing `a` are the exact objects returned after `a` is selected;
neither is recomputed. Only the root terminalizes the retained outcome to the
token witness. The scalar printed as `selected_terminal_diagnostic` is the
worst coordinate of that already-selected outcome and did not drive strength
as an isolated reward.

## One family application

The complete demanded occurrence tree is lowered to `LlamaCompanyShape`.
`llama_company_evaluate` applies embedding, RMS, Q/K/V, attention output, MLP,
final RMS, and output-head fillers to their whole row families. The runtime
checks `maximum_calls_per_filler == 1` for this phase. Strength runs afterward
over immutable outcomes and aborts if it causes any learned-filler call or
scalar weight read.

## Sampled dependent demand tree

Every generated frame retains the full 32,000-token vocabulary. `-k 0` leaves
the proposal distribution over that full carrier. A positive `-k K` is an
explicit top-K proposal restriction and is reported as such; it is never
described as the semantic carrier or as exact full-vocabulary search.

`-b N` demands `N` complete hypothetical paths before the learned observer is
run. Shared prefixes are represented once and repeated paths increment their
reachability multiplicity. The resulting irregular prefix tree is a finite
dependent selection term. After one whole-family `llama_company_evaluate`,
memoized strength applies every local `Select` to the recursively selected
suffix outcome. Sampling constructs the observed subtree; it does not choose
the returned witness.

The finite recursion is exact over the observed dependent subtree, but the
subtree is only a sampled approximation to the full vocabulary product. The
trace therefore records `exact:false`, `path_demands`, `unique_leaves`,
`root_reachability`, and any positive `proposal_top_k`.

For six completion positions with 1,024 full-vocabulary path demands:

```sh
./run_hidden_feedback_select test/stories15M.bin \
  -z tokenizer.bin -i "Lily was" -l 6 \
  -r company -k 0 -b 1024 -s 42 -o candidates.jsonl
```

Use `-l N` to request exactly `N` completion positions independently of the
tokenized prompt length. This overrides the older total-position `-n` option
and refuses prompts that would exceed the model context.

The sampling proposal is still produced by independently unembedding the
fixed hidden-feedback tape. That is a known coverage limitation, not a solved
quality result. Backward induction cannot select a coherent completion that
was never demanded. The active implementation reports the actual weak leaves;
it does not repair them with an AR rollout, scalar likelihood sum, terminal
token score, or truncated suffix.

The first full-vocabulary system run above retained 1,023 unique leaves from
1,024 path demands and selected:

```text
Lily was a a sad a very sad
```

None of those leaves contained `girl`, `young`, `little`, `child`, or
`person`. At the relevant fixed-tape frame, `girl` had local rank 516 and
proposal probability `2.34597073e-06`, so 1,024 draws had expected occurrence
count `0.00240227`. This is a measured failure of the fixed hidden-feedback
proposal, not evidence that the recursive observer rejected a coherent leaf.

## Trace

Every JSONL record is flushed immediately. Relevant records are:

- `selection_term_built`: exact rows, leaves, and root reachability;
- `company_run`: learned-filler counts and model time;
- `company_outcome`: decoded branch, complete coordinate vector, leximin order,
  and context/company bindings;
- `candidate_rated`: every local candidate and the complete backed outcome's
  diagnostic worst coordinate;
- `select`: the retained child and outcome at one memoized node;
- `root_terminalized`: the single emitted witness and complete outcome;
- `strength_run`: proof that strength performed no learned-filler work.

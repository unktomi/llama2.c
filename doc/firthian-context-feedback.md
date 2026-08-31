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

## Exact finite sampled support

`-k` is the number of token-indexed cells sampled without replacement at each
prefix. Once those local carriers have been demanded, their finite selection
product is exact: every sampled alternative remains available at every
completion position.

`-b` is only an optional resource safety limit. If the exact sampled product
would exceed it, the run fails before model evaluation. It never spends the
limit near the root and silently turns later Selects into unary nodes. The
default is unlimited.

For a two-token completion with `-k 4`, the term has all `4^2 = 16` leaves:

```sh
./run_hidden_feedback_select test/stories15M.bin \
  -z tokenizer.bin -i "Lily was" -n 6 \
  -r company -k 4 -b 16 -s 42 -o candidates.jsonl
```

Use `-l N` to request exactly `N` completion positions independently of the
tokenized prompt length. This overrides the older total-position `-n` option
and refuses prompts that would exceed the model context.

The proposal carrier is still produced by independently unembedding the fixed
hidden-feedback tape. That is a known semantic limitation, not a solved
quality result. Backward induction cannot select a coherent completion absent
from this carrier. The active implementation reports the actual weak leaves;
it does not repair them with an AR rollout, scalar likelihood sum, terminal
token score, or truncated suffix.

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

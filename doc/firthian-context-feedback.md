# Firthian context feedback

`run_hidden_feedback_select.c` contains the active finite selection-product
path.  Its company evaluator is separate from the carrier that proposes a
finite token support.

## Structured outcome

For each demanded token occurrence `n`, `llama_company_evaluate` produces one
contextual hidden state and applies the learned output head to the entire
family:

```text
Q(n) = E^T h(n) : Token -> Logit
```

`Q(n)` is retained as a token-indexed observation.  It is not reduced to a
path score.  `ProjectionTermNode` retains the demanded continuation children,
the child selected by backward induction, the selected leaf, and the row's
log-partition.

The selection at a non-terminal occurrence `x` first recursively selects its
continuation.  If the selected immediate successor is `y`, it rates `x` by

```text
log_softmax(Q(x))[y]
```

and chooses the `x` whose selected company has the greatest learned
observation.  The selected `y` was itself chosen against its selected `z`, so
the operation recurses backward through the term.

The recursion is the literal memoized product:

```text
b(x)   = force the selection rooted below x once
a      = Select(x -> score(x, b(x)))
result = a : b(a)
```

Each occurrence records whether its `b(x)` is unforced, currently forcing, or
forced.  The selected suffix stored while rating `a` is returned directly; it
is not recomputed.

The finite right boundary does not invent EOS.  With no right-hand company,
the last filler is rated by the incoming contextual row that reaches it.  This
is recorded as `observer_direction: "incoming_boundary"`; all internal
pairings are recorded as `observer_direction: "outgoing"`.

The old sum of company density ratios remains in the trace only as
`path_density_log_ratio_diagnostic`.  It does not participate in selection.

## One family application

The demanded occurrence term is converted to a `LlamaCompanyShape` and passed
once to `llama_company_evaluate`.  Embedding, RMS, attention projections, MLP
projections, final RMS, and the output head each receive their complete row
family in one call.  `maximum_calls_per_filler` checks that property for this
family evaluation.

That call completes before selection strength begins.  Strength then receives
only the immutable `LlamaCompanyResult` logit table.  The executable snapshots
all learned-filler counters at this boundary and aborts if either a filler call
or scalar weight read occurs during the memoized product.  The trace records
the two phases separately as `company_run` and `strength_run`, including their
independent timings.

This does not yet establish that the carrier recurrence preceding the family
evaluation is one-shot.  That recurrence still uses the reference numerical
path to construct its fixed proposal frames.

## Trace

Every demanded candidate is appended and flushed immediately.  A
`candidate_rated` event records:

- the decoded complete continuation in `text`;
- the candidate token and local carrier rank;
- the contextual row and selected company occurrence;
- the company token coordinate, raw logit, and log-partition;
- whether the pairing is outgoing or the incoming finite boundary;
- the exact normalized rating used by the local `Select`.

Only the root emits `root_terminalization: true`.

## Current support limitation

The carrier currently provides one fixed top-k frame per absolute token
position.  These frames are not dependent on the prefix selected by strength.
Consequently, a coherent continuation may be absent even when every one of
its internal learned edge scores is good.  In the measured four-token
`Lily was` term, `a little girl .` is impossible because the final fixed frame
contains `She`, `It`, `Today`, and `Her`; `.` is below rank 512 in that frame.

Likewise, a 32-leaf/top-16 term explores 16 alternatives at the first variable
slot, 2 at the second, and 1 at every later slot.  Text after that point is an
unchallenged sampled continuation.  Making sampled support continuation-
dependent without reintroducing repeated eager model runs is the remaining
representation problem; changing the Firthian rating cannot manufacture a
token that is absent from the term.

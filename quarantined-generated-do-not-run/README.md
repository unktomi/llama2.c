# Generated artifacts from rejected paths

Do not run these executables as inference implementations. They are stale
build products of the corresponding `*-don't-do-this.c` sources in the parent
directory. They were moved here only to keep the repository root and active
test/build surface free of rejected algorithms; the opening comment of each C
source records the specific semantic failure.

The `fixed-token-*` artifacts came from the later fixed-family evaluator. That
evaluator required concrete completion tokens before composing the model, so
it again placed selection outside the term. Its numerical parity result was
not evidence for Escardo inference and the entire slice has been quarantined.

The `token-prefix-field-*` artifacts came from the subsequent active evaluator.
That evaluator used `int token` as the selection carrier and forced
`sample_field` from every recursive selection frame.  It therefore performed
strict per-prefix model observations and repeatedly applied the learned
fillers; moving those applications below one outer `run_pcont` call did not
compose them.

The `resident-weight-scope-*` artifacts are stale products of an earlier build
that was incorrectly presented as one-shot physical weight use. A 256-token
run invoked each layer's family kernel in 258 causal waves; installing a weight
pointer once did not make those numerical applications one-shot. The active
non-greedy path makes no physical weight-reuse claim, so these old executable
and trace files remain archived only as evidence of that rejected claim.

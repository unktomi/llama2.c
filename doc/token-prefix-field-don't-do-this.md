# DO NOT USE: token-carrier/prefix-field evaluator

This document describes a rejected evaluator.  Its recursive selection carrier
is a token, and its observer calls `sample_field` at every forced prefix.  It
therefore terminalizes model observations during the product and applies each
learned filler repeatedly.  A single outer call named `run_pcont` did not make
those inner observations composed or the weights one-shot.

The files are retained under explicit `*-don't-do-this` names for audit only.

# Historical description

The alternative inference path is now entirely C:

- `atkey_term.c` forms and evaluates the continuation/selection term.
- `atkey_term_c.c` exposes the original llama2.c numerical leaves and model
  weights. It does not expose or call `forward()`.
- `atkey_term_c.h` is the boundary between the term and those leaves.

`make runatkeyterm` needs a C compiler only. Haskell is not part of this path.

## The C representation of the term

The mathematical family context has the shape

```text
S -> (A -> B) -> T
```

`LearnedFiller` is the defunctionalized C representation of one learned
`A -> B`. `family_context(input, filler)` receives that filler once and
constructs a family of output `Field`s. Every output field refers to the same
filler object; it does not contain a copied weight matrix.

A `Field` is a causal prefix function:

```c
struct Field {
    uint32_t id;
    int dependency;
    FieldCompute compute;
    void *environment;
};
```

Its value at a completion prefix is memoized by `(field id, prefix id)`. The
`dependency` is its prefix modulus: a field with dependency `d` can observe
exactly the first `d` selected completion tokens. Sampling canonicalizes a
longer prefix to depth `d` before consulting the memo table.

`model_fields_term` first creates token fields for all prompt and completion
positions and then composes embedding, every transformer layer, final RMS, and
the classifier. It retains only the causal logit fields needed by selection.
Logit field `i` is checked to have dependency `i`. Term construction calls no
numerical leaf; the executable verifies that every learned-filler counter is
zero immediately before `run_pcont`.

`layer_company` turns one input field company into the next. A completed layer
is therefore the filler of the following layer scale. `network_company`
composes these companies before selection begins; it is not a grid of eager
token-by-layer forward calls.

SwiGLU is a nonlinear context. `compute_swiglu` samples its gate field once and
passes the resulting activation as both the raw gate and sigmoid input. The
activation is duplicated; the learned gate filler is not.

## Escardó's dependent product

For a selection function `epsilon : (X -> R) -> X`, Escardó's dependent
product can be written

```text
b(x) = delta(x) (xs -> p(x : xs))
a    = epsilon   (x  -> p(x : b(x)))
result = a : b(a)
```

The C translation is direct:

- `history_product_select` is the recursive product.
- `product_suffix(frame, x)` is `b(x)`.
- `product_suffix` binds the selected head to its lazy, memoized suffix.
- `observe_candidate(frame, x)` constructs `x : b(x)` and applies `p`.
- the local top-k argmax is `epsilon`.

`ProductFrame` memoizes one suffix thunk for each candidate `x`. This is the C
counterpart of Escardó's `where`-bound `b x`: if a candidate suffix is demanded
again, the already selected function-tree branch is reused.

### Non-forcing strength log

`--strength-log PATH` records the operational form of this equation. It is
available on the exact product path and emits a monotonically numbered,
line-buffer-independent event stream. Each event is explicitly flushed.

The event sequence distinguishes:

- `select_enter`: entry into one recursive Bellman/selection frame;
- `observer_apply` and `observer_return`: application of `p` to `x : b(x)`;
- `suffix_bind`, `suffix_force_*`, and `suffix_reuse`: creation and demand of
  Escardó's where-bound `b(x)`;
- `bellman_demand` and `bellman_return`: the downstream value requested for a
  local move and the exact value or valid upper bound returned;
- `token_role role=island`: a token occurrence presented as the object of the
  current selection;
- `token_role role=bridge`: that same occurrence, linked by
  `occurrence_prefix`, used as the context edge entering its continuation;
- `select_choose` and `compose_return`: the local choice and its composition
  into the enclosing selection;
- `tau_*`: the sole final `J_R -> K_R` application.

The strength stream never decodes or traverses a lazy completion. This matters
because the separate candidate audit (`-a`) prints whole candidate text and can
therefore force suffixes that the bounded selector itself did not demand.

The selector is history dependent. It first observes the current causal logit
field, forms its local top-k support, and compares each supported token by the
reward of its whole recursively selected continuation. It is therefore not
beam search and not repeated greedy decoding.

For a full-horizon completion, the reward is unchanged model
log-probability:

```text
sum_t log softmax(logits_t)[token_t]
```

If a delimiter ends the outcome early, its sum is projected to the requested
horizon before comparison:

```text
horizon * sum_t log softmax(logits_t)[token_t] / observed_token_count
```

This makes variable-length outcomes comparable. Without it, an early
delimiter receives every missing suffix position for free and can beat a much
more probable full continuation merely because fewer negative terms were
added. A full-horizon outcome is numerically unchanged.

Every step is nonpositive. The raw partial sum remains an upper bound on the
final horizon-equivalent reward: its best hypothetical suffix adds zero until
the horizon. With `-b 1` (the default), a candidate whose partial sum is already
no greater than the incumbent cannot recover, so its remaining suffix need not
be forced. This is only a demand shortcut for the same dependent-product
result. `-b 0` disables it. `-d 1` retains and later forces every local
candidate outcome so the printed top-k audit contains exact whole continuation
scores.

### Boolean satisfaction instead of numeric certification

The default goal remains the exact numeric argmax. Supplying `-s SCORE`
changes the answer type to the Boolean predicate

```text
p(completion) = horizon_equivalent_reward(completion) >= SCORE
```

and uses the same dependent product to return the first satisfying whole
continuation. This is Escardo-style searchable selection with `R = Bool`, not
a beam or prefix queue. It deliberately does not certify that the witness is
the numeric maximum; the output labels the goal and threshold explicitly.

On Stories260K with prompt `Lily was`, horizon 48, and local top-k 4:

```text
-s -17: 0.60 seconds, 837 selection frames, reward -16.857306938428646
exact:  441.93 seconds, 855857 selection frames, reward -16.549512049150444
```

The satisfying completion is a coherent 48-token witness. A threshold just
below the certified optimum (`-s -16.55`) finds that same optimum in about
27.62 seconds and 45300 frames, without claiming it has proved maximality.

### Time-bounded recursive continuation demand

`--sample-ms MILLISECONDS` and `--sample-demands COUNT` do not switch to a
second inference algorithm. `run_pcont` still invokes `history_product_select`.
The only approximation is inside each local selection frame: it samples which
member of the model's top-`k` support to present to the observer next.
Candidates are drawn without replacement with weights derived from their local
logits. Their comparison value remains the unchanged whole-completion model
log-probability.

Operationally, one candidate demand has this order:

```text
continuation_sample x
observer_apply (x : b(x))
  recursively select b(x)
  return the suffix's backed observer value
bellman_return x
```

Only then may the local `Select` update its incumbent. A fixed demand budget is
passed as an index through the recursive product. A frame with budget `B` and
remaining horizon `h` admits at most `min(k, floor(B/h))` sampled local
continuations. Until all `k` are admitted, each admitted continuation receives
the minimum complete suffix budget `h-1` and surplus below the next admission
threshold remains unused. Once all `k` are present, additional budget is split
monotonically among their suffix selections. The sampling order is derived
from the prompt and prefix, rather than a traversal-global random stream, so a
larger budget never reshuffles or shrinks already demanded support.

The wall-clock budget is checked between observer applications. It can stop a
frame from asking about another candidate, but an in-flight recursive demand
is completed and every newly entered frame is allowed one candidate so the
selection remains total. There is no phase that first produces complete
ancestral paths, and there is no separate leaf-maximum backup.

The fixed random seed makes `--sample-demands` reproducible. With Stories260K,
`Lily was`, horizon 8, and top-4, the measured nested sweep is:

```text
budget 8:     -6.2908930099474425  " a little girl named J"
budget 32:    -5.1156563480122523  " walking in the par"
budget 512:   -4.6115352978197520  " a boy who lo"
budget 16384: -3.7059803957940889  " a little girl who"
```

The rewards are nondecreasing because these are nested demanded supports. A
limit of 87380 exhausts the h8 top-4 selection support after 84 actual demands
and matches the exact product. `--strength-log` is available in sampled mode;
its system test verifies nested Bellman demand/return order and rejects the old
`sample_edge`, `sample_rollout`, and `sample_backup` event order.

For the longer h48 run with a 20000-demand root budget, the four root
continuations are all observed after their recursive suffixes. The selected
result is:

```text
completion:  a quiet dog named Max who liked to play in the park. One day,
             Max went to the park with his mommy and daddy.
selected_reward=-27.56077465986171
sampled_candidate_demands=5290
```

## Continuations and the one final run

`Observer` is the C specialization of the active continuation type. Its answer
is an `Outcome`, which pairs the already-composed model fields with a lazy
completion term. `observe_candidate` extends that completion term and applies
the observer; it does not construct a model layer or run an eager model step.

The `J_R -> K_R` morphism has the familiar behavior

```text
p(selection(e, p))
```

and is closed by `run_pcont`. There is one source-level call to `run_pcont`, in
`main`, after the model term, selection, and observer have all been assembled.
During that run, selection may invoke its composed observer many times, as the
selection-function equation requires. There is no hidden eager whole-model run
inside the observer.

## Weight multiplicity versus physical execution

There are 48 learned filler objects in Stories260K: embedding, nine maps for
each of five layers, final RMS, and classifier. Each learned map occurs once as
a filler of its context in the source term. Continuation-use multiplicity lives
in field families and completion prefixes.

The portable CPU evaluator still applies a filler for each demanded
`(field, prefix)` value, so it does not claim that a matrix coefficient is
physically fetched only once. The executable reports learned filler
applications and scalar accesses to keep that lowering cost visible. Fusing a
field family into a weight-stationary batched kernel is a later lowering; it is
not simulated by delaying an eager `forward()` call.

### Quarantined lowerings and rewards

The exhaustive vocabulary-prefix company, complete ancestral rollout samplers,
and entropy-distance reward have been removed from every active build target.
They remain as `exhaustive-prefix-company-don't-do-this.c`,
`exhaustive-logit-grid-don't-do-this.c`,
`categorical-rollout-shortcut-don't-do-this.c`,
`batched-ancestral-rollouts-don't-do-this.c`, and
`entropy-distance-reward-don't-do-this.c`; each begins with a comment recording
why it was rejected. Their archived Python regressions no longer match
`test_*.py`, so the default suite cannot execute them accidentally.

The exhaustive company can still serve as a tiny numerical oracle, but it
materializes `V + V^2 + ...` contexts and therefore cannot answer a requirement
to sample only demanded continuations. The ancestral samplers chose complete
AR paths before composing selections; a later maximum backup did not repair
that order. The entropy-distance score changed the observer without a
derivation from the model or the supplied Firthian calculus.

## Stories260K evidence

For prompt `Lily was`, horizon 8, local top-k 4, the composed product selects:

```text
completion:  a little girl who
selected_tokens=[261,376,298,315,421,263,415,414]
selected_reward=-3.7059803957940889
```

At depth 5 the local rank-0 token is `named` (395), but its fully selected
continuation scores `-4.3616329666282496`. Local rank 1, `who` (263), leads to
`-3.7059803957940889`, so backward induction chooses `who`.

The C top-4, horizon-8 run takes about 0.08 seconds and peaks near 19 MB on the
development machine. The prior Haskell closure representation used about
0.96 seconds and 577 MB for this same case; Haskell is no longer used by the
build or evaluator.

## Build and verification

```sh
make testatkeyterm CC=clang
make testcc CC=clang
```

The system checks run actual Stories260K completions. They cover exact top-4
candidate records, the non-greedy rank-1 decision above, sampled recursive
demand/return order, convergence to the exact bounded product when sampled
support is exhausted, top-1 numerical parity with stock llama2.c, delimiter
termination, zero learned-kernel calls before the final run, and structural
rejection of an upstream `forward()` call or an eager model construction hidden
in the final observer.

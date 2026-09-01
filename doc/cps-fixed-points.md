# Exact hidden-continuation pullbacks

`cps_fixed_points.c` evaluates a frozen llama2.c transformer as a typed term
and constructs its suffix continuations mechanically. The CPS term does not
supply a linguistic parse, choose a scalar reward, or use the classifier. The
root observation is the complete hidden frontier after the trained final
RMSNorm.

This is a finite experiment on four token-constructor variants. It asks where
two initially independent edits first interact, whether the untouched
remainder of the transformer can still observe that interaction, and whether
an actual suffix continuation is fixed by an endomorphic block.

## Continuation IR

A numerical operation is represented by:

```c
typedef struct {
    const char *name;
    int input_width;
    int output_width;
    FrontierMapApply apply;
    void *environment;
} FrontierMap;
```

A continuation is vector-valued:

```c
typedef struct {
    int input_width;
    int result_width;
    ContinuationApply apply;
    void *environment;
} Continuation;
```

`make_pullback` is the literal action of a map on a continuation:

```text
pullback(F, k)(x) = k(F(x)).
```

Starting with identity on the final normalized hidden frontier, the program
builds every suffix by repeated pullback. If the numerical term is

```text
F0 ; F1 ; ... ; Fn ; final_rms
```

then the continuation at boundary `s` is exactly

```text
ks = final_rms . Fn . ... . Fs.
```

There is no logit projection in this term. `Continuation.result_width` remains
`positions * dim` all the way to the root.

This realizes the linear continuation-space operator

```text
U_F(k) = k . F
```

by its action on a continuation, without claiming that nonlinear `F` is a
linear map on hidden states.

## Typed kernel term

The attention residual is factored as the actual seven-stage computation:

```text
X
 -> (X, rms(X))
 -> (X, Q, K, V)
 -> (X, V, QK^T)
 -> (X, V, softmax(QK^T))
 -> (X, attention-values)
 -> (X, Wo(attention-values))
 -> X + update
```

The SwiGLU residual is similarly factored:

```text
X
 -> (X, rms(X))
 -> (X, W1 rms(X), W3 rms(X))
 -> (X, SiLU(W1 rms(X)), W3 rms(X))
 -> (X, gated-product)
 -> (X, W2 gated-product)
 -> X + update
```

The explicit product carriers retain the residual and branch values until the
operation that consumes them. The trace checks both the forward factorization
and its CPS composition against the monolithic reference block.

## Discovering an interaction without a parse

Four equally tokenized strings supply a commuting constructor square:

```text
x00 -- A --> x10
 |              |
 B              B
 v              v
x01 -- A --> x11
```

The program is not told that either edit is a determiner, noun, phrase, or
hole. It locates the two changed token positions mechanically and carries the
complete `positions * dim` frontier through every layer.

At an operation input, existing interaction is

```text
prior = x11 - (x10 + x01 - x00).
```

To ask whether the current operation creates a new interaction independently
of that prior term, form the torsor completion

```text
x_independent = x10 + x01 - x00
```

and compare

```text
local = F(x_independent) - (F(x10) + F(x01) - F(x00)).
```

The exact higher-scale observation is not a norm propagated layer by layer.
The untouched suffix continuation is applied to both endpoints:

```text
visible = k(F(x_independent))
        - k(F(x10) + F(x01) - F(x00)).
```

The trace reports L2 norms of these vector differences for browsing. They are
not rewards and are never added across operations.

For an endomorphism `F : H -> H`, the sampled fixed-continuation defect is

```text
|| k(F(x)) - k(x) || / || k(F(x)) ||.
```

It tests the actual generated suffix `k` on the four sampled states. It does
not purport to enumerate every eigenfunction of `U_F`.

## Stories15M run

```bash
make cpsfixedpoints CC=clang
./cps_fixed_points ../llama2.c/test/stories15M.bin tokenizer.bin \
  "The cat walked across the path." \
  "The cat walked across my path." \
  "The cat walked across the foot." \
  "The cat walked across my foot." \
  --trace outputs/cps-fixed-the-my-path-foot.jsonl
```

All four inputs have eight tokens. The two automatically located edits are
`" the" -> " my"` at position 5 and `" path" -> " foot"` at position 6.

At layer 0, the operation-level measurements are:

| Operation | Local interaction L2 | Suffix-visible L2 |
|---|---:|---:|
| attention RMS pair | 0.00000157 | 0.00018686 |
| Q/K/V projection | 0.00000636 | 0.00020364 |
| QK contraction | 1.3331234 | 11.332332 |
| softmax | 0.11648931 | 3.0692596 |
| attention/value contraction | 0.28336127 | 9.4663313 |
| attention output projection | 0.00000035 | 0.00016795 |
| attention residual addition | 0.00000011 | 0.00016657 |
| FFN RMS pair | 0.35465067 | 2.1289764 |
| W1/W3 projection | 0.00001255 | 0.00018422 |
| SiLU | 0.50753585 | 4.3099672 |
| SwiGLU product | 0.81317001 | 16.729052 |
| FFN output projection | 0.00000134 | 0.00018851 |
| FFN residual addition | 0.00000020 | 0.00017353 |

The separation is several orders of magnitude. Before attention combines the
edited positions, tokenwise RMSNorm and the learned linear projections do not
materially couple the edits. The first substantial layer-0 join is the learned
QK contraction. Softmax and the value contraction transform that relation.
After attention has put both distinctions into a common token state, the FFN's
tokenwise RMSNorm can couple them; SiLU and the gated product add further
interaction. The later learned linear projections and residual additions are
again at floating-point cancellation scale.

In later layers, attention RMSNorm is no longer near zero because earlier
layers have already consolidated the distinctions into individual token
states. QK, softmax, value contraction, FFN RMSNorm, SiLU, and the gated
product remain interaction sites. This is a model-derived operation structure,
not a supplied constituency tree.

Every attention and FFN typed chain produced exactly the same frontier as its
monolithic block, and every CPS chain matched the corresponding whole-block
suffix, with maximum measured composition defect `0`. The layer-frontier
evaluator is also compared with stock token-scheduled `forward()` using only
its post-final-RMS hidden state; the small remaining float32 discrepancy is
reported in both absolute and relative form. Stock `forward()` also executes
its classifier during this independent parity check, but its returned logits
are discarded and are not reachable from any continuation in the CPS term.

## Scope

This establishes an exact CPS representation of the sampled transformer term
and a concrete way to find continuation-visible joins without projecting to
tokens. It does not yet discover a closed basis for the full continuation
space, infer a global grammar, or perform non-greedy completion. Those require
sampling more constructor alternatives and retaining the resulting pullback
incidence/fixed subspaces; the present trace is the smallest nontrivial case.

## Torsor-safe affine continuation arrangement

`cps_affine_spectrum.c` expands the four-context experiment without turning
hidden-state points into vectors. It reads real prompt text recursively from a
corpus directory, tokenizes it, and takes distinct fixed-width constructor
windows. No token category or constituency label is supplied.

For an endomorphic operation `F` and the exact suffix continuation `k`, sample
`i` retains both root points

```text
after_i  = k(F(x_i))
bypass_i = k(x_i).
```

The mapped pair file declares an implicit homogeneous coordinate. The program
never pseudoinverts either matrix of points. Its fixed calculation uses only
the torsor displacement

```text
d_i = after_i - bypass_i.
```

Each row represents one contextual fixed hyperplane:

```text
E_i = { c | d_i c = 0 }.
```

For a family `I` of contexts, the simultaneous fixed covectors are

```text
E_I = kernel(stack(d_i, i in I)).
```

The constant affine observation is stored as a separate one-dimensional mode.
It is not represented by selecting zero in the hidden-state torsor.

The pair file is updated after every completed sample and supports `--resume`.
Its complete row arrangement is the semantic artifact: contexts are not added
to produce a reward. The companion basis file contains the complete SVD of the
stacked displacement matrix for rank/nullspace calculation and numerical
browsing. Singular ordering uses the model's float32 coordinate metric; it is
not an inference score or a replacement for the individual contextual rows.

Both binary files are native-endian and self-describing. `CPSAFF1` contains
its header followed by one row per context:

```text
[after_0 ... after_(r-1), bypass_0 ... bypass_(r-1)].
```

Its header records `samples_written`, so an interrupted deterministic corpus
scan can continue with `--resume`. `CPSBAS2` contains its header, every
singular value, and the complete `r * r` right-singular basis in row-major
order. The header separately records exact displacement nullity and the
one-dimensional affine constant.

On macOS this target uses Accelerate LAPACK:

```bash
make cpsaffinespectrum CC=clang
./cps_affine_spectrum ../llama2.c/test/stories15M.bin tokenizer.bin \
  --corpus-dir ../llama2.c/work_traces/long_context_32 \
  --positions 3 --samples 928 --layer 0 --operation attention \
  --matrix outputs/cps-affine-l0-attention-p3-s928.bin \
  --basis outputs/cps-affine-l0-attention-p3-s928-basis.bin \
  --trace outputs/cps-affine-l0-attention-p3-s928.jsonl
```

This run uses a root frontier of `3 * 288 = 864` coordinates. All 928 decoded
windows are distinct and come from retained TinyStories prompts. The
displacement matrix has numerical rank 864:

```text
largest singular value:       3129.1079
smallest singular value:         1.2354311
float32 rank tolerance:           0.3461614
global covector nullity:          0
affine constant dimension:        1
```

Therefore no nonconstant affine combination of the 864 exact suffix-coordinate
continuations is fixed by layer-0 attention across every sampled context. Only
the affine constant is globally fixed. This does **not** collapse the
contextual result: every nonzero individual displacement row has an
863-dimensional covector kernel. Rather, those hyperplanes vary with context
enough that their global intersection is trivial. The next composition must
retain this family of contextual subspaces instead of replacing it with that
single intersection.

## Root-reachable pullback spectrum

`cps_pullback_spectrum.c` tests the larger continuation-space premise directly.
For the selected endomorphism `F` and its mechanically generated suffix `k`,
let `phi` be the coordinate continuations of the requested root frontier.  At
dictionary depth `p` the program retains the actual block-Krylov family

```text
Psi_p = [1, phi, U_F phi, ..., U_F^(p-1) phi]
U_F(g) = g . F.
```

For every real token context `x_i`, the mapped evaluation file stores all root
points

```text
k(x_i), k(F(x_i)), ..., k(F^p(x_i)).
```

It flushes a complete row before advancing `samples_written`, can resume, and
can grow its capacity without recomputing earlier rows.  The JSONL trace is
also flushed after every decoded context.  No logits, token classifier, scalar
reward, or supplied parse occurs in this term.

The hidden states in these expressions remain points.  Linear algebra is
performed on columns of scalar functions evaluated at those points.  The
constant function is explicit, so translating the root hidden-state chart is
an affine change of dictionary basis rather than a choice of a meaningful
hidden-state zero.

### When a sampled operator exists

Write the fit evaluations as

```text
X[i,:] = Psi_p(x_i)
Y[i,:] = U_F(Psi_p)(x_i).
```

If `X = U Sigma V^T`, two different conditions must be checked:

```text
representation defect = ||(I - U U^T) Y|| / ||Y||
descent defect        = ||Y (I - V V^T)|| / ||Y||.
```

The first asks whether every pulled dictionary function can be represented by
the original sampled functions.  The second asks whether a coefficient
combination that is zero as a sampled function remains zero after pullback:

```text
kernel(X) subset kernel(Y).
```

Without the second condition, pullback does not descend to the sampled
function quotient.  Diagonalizing a fitted coefficient matrix in that case
does not produce continuation eigenfunctions.  This distinction rejected the
initial whole-frontier fit: 9,217 functions on 640 fit contexts had a `0.6321`
descent defect even though its representation residual was small.

When both conditions are numerically supported, the operator on the sampled
function-value basis is

```text
M = U^T Y V Sigma^-1.
```

For every right eigenvector `w`, the corresponding dictionary continuation has
coefficients

```text
c = V Sigma^-1 w.
```

The program evaluates the defining equation on contexts excluded from the
fit:

```text
Y_validation c = lambda X_validation c.
```

Every eigenvalue, its fit residual, its held-out residual, and the variation of
the resulting function across held-out contexts is written to JSONL.  A mode
that diagonalizes `M` but fails this equation is recorded, not accepted.

`CPSKRY1` is the native-endian mapped evaluation format.  `CPSKOP1` version 2
stores, in order: its header, all dictionary singular values, the retained
right dictionary basis as logical rows, the column-major reduced operator,
real and imaginary eigenvalues, column-major right eigenvectors, and singular
values of `M-I`.

### Stories15M sixteen-token result

The retained system run used layer-0 attention, sixteen-token contexts, the
last root position's complete 288-dimensional hidden state, three pullback
generations, and no token projection:

```bash
make cpspullbackspectrum CC=clang
./cps_pullback_spectrum ../llama2.c/test/stories15M.bin tokenizer.bin \
  --corpus-dir ../llama2.c/work_traces/long_context_32 \
  --evaluations outputs/cps-pullback-l0-attention-p16-s2048-d3.bin \
  --spectrum outputs/cps-pullback-l0-attention-p16-s2048-d3-last-spectrum-v2.bin \
  --positions 16 --samples 2048 --fit-samples 1792 \
  --pullback-depth 3 --layer 0 --operation attention --root last \
  --trace outputs/cps-pullback-l0-attention-p16-s2048-d3-last-resume.jsonl \
  --resume
```

The first 1,024 rows were reused from the earlier persisted run.  The complete
evaluation has 2,048 distinct contexts; the operator was fitted on 1,792 and
checked on the remaining 256:

```text
dictionary columns:                 865
sampled rank:                       865
fit representation defect:       0.098371779
fit descent defect:               0.000000402
held-out dictionary defect:       0.23208011
fitted fixed dimension:                    1
```

The fixed mode is the constant continuation: its held-out eigen-equation
residual is `1.12e-14`.  The best genuinely varying nonconstant mode has
`lambda = 0.7051078942`, held-out residual `0.30905124`, and held-out
constant-variation ratio `0.61850862`.  It is not accurate enough to use as an
exact transport rule.

A controlled sweep on the same first 1,024 contexts (`896` fit, `128`
held-out) was:

| Dictionary depth | Columns | Fit representation | Fit descent | Held-out dictionary |
|---:|---:|---:|---:|---:|
| 1 | 289 | 0.52835 | 3.29e-7 | 0.87973 |
| 2 | 577 | 0.15633 | 3.64e-7 | 0.52695 |
| 3 | 865 | 0.02363 | 0.00231 | 0.89669 |

The shallow third-depth fit appeared to have seven fixed directions, but all
nonconstant ones failed held-out evaluation.  Doubling coverage restored full
dictionary rank and reduced the fitted fixed space to the constant alone.
Those six modes were sampling artifacts.

This result does not establish that the complete continuation operator has no
nonconstant eigenspaces.  It establishes the narrower, actionable fact that
the depth-three coordinate-generated root-reachable space is not closed well
enough to compile into exact fast inference.  No weight bypass or inference
speedup is claimed from this run.

## Grammatical-action continuations

`cps_grammar_actions.c` supplies semantic coordinates from grammatical
constructor actions rather than from arbitrary hidden-state coordinates.  Its
five command-line terms are ordered as

```text
x, a(x), b(x), b(a(x)), a(b(x)).
```

Thus `abx` means “apply `a`, then `b`.”  With pullback
`U_a(k) = k . a`, the two literal root observations are

```text
(U_a - I)(U_b - I)k(x)
    = k(abx) - k(ax) - k(bx) + k(x)

(U_a U_b - U_b U_a)k(x)
    = k(abx) - k(bax).
```

The first detects symmetric coupling such as agreement even when the second
is zero.  The second detects order-sensitive actions.  Both complete vectors
are retained.  Their L2 norms are emitted only to make the trace browsable;
they are never added or used as rewards.

### Separating root effect from the interaction boundary

There are three different quantities in each boundary record and they must
not be conflated.

For boundary states `h_x`, `h_a`, `h_b`, `h_ab`, and `h_ba`, the local mixed
vector is

```text
m = h_ab - h_a - h_b + h_x.
```

It locates where an initially factorized constructor square stops being an
affine parallelogram in the transformer's typed state.  The local commutator
is `h_ab - h_ba`.

The literal pulled-back mixed observation applies the exact remaining suffix
to each real point separately:

```text
k(h_ab) - k(h_a) - k(h_b) + k(h_x).
```

Because every `h` is the actual state of its term, this vector is invariant as
the same transformer computation is reassociated into CPS.  It establishes
the eventual root-visible joint effect; it cannot by itself locate the first
join.

The boundary-local root-visible witness instead constructs the torsor point

```text
h_independent = h_a + h_b - h_x
```

and retains

```text
k(h_ab) - k(h_independent).
```

This is zero up to floating-point cancellation while `h_ab` is still the
independent torsor completion, then becomes nonzero when a kernel has actually
coupled the two distinctions in a way visible to the untouched suffix.  It is
not a sum of completion probabilities and `h_independent` is not treated as a
hidden-state origin.

Every boundary trace row contains all five vectors: local mixed, local
commutator, pulled-back mixed, pulled-back commutator, and torsor-visible
difference.  The root may be the complete final hidden frontier or only its
last position.  It is always post-final-RMS hidden state, never logits.  The
first implementation requires equally tokenized variants so every local
subtraction is typed; it refuses padding.

### Exact transition attribution

Let `F_l` map one recorded boundary to the next and let

```text
k_l = k_(l+1) . F_l
s_l = a_l + b_l - x_l
tau_l = k_l(ab_l) - k_l(s_l).
```

Since `s_(l+1) = F_l(a_l) + F_l(b_l) - F_l(x_l)`, direct cancellation gives

```text
delta_tau_l = tau_(l+1) - tau_l
            = k_(l+1)(F_l(s_l)) - k_(l+1)(s_(l+1)).
```

Every `grammatical_action_transition` record contains this complete vector.
It is the root-visible failure of that one typed map to preserve the sampled
torsor parallelogram.  Its L2 norm is reported separately, but the norms do
not telescope and are not attribution weights.  Only the vectors obey

```text
tau_last - tau_first = sum_l delta_tau_l.
```

The retained runs check this identity componentwise.  Maximum absolute
float32 telescoping defects are between `1.12e-8` and `5.96e-8`.

### QK bilinear reconstruction and causal removal

At a QKV boundary, construct the actual torsor point

```text
s = a + b - x.
```

The typed QK map receives `(X,Q,K,V)` and returns `(X,V,scores)`.  For the Q
and K components of `s`, define

```text
delta_a_Q = Q_a - Q_x       delta_a_K = K_a - K_x
delta_b_Q = Q_b - Q_x       delta_b_K = K_b - K_x.
```

In real arithmetic the complete score-table defect is exactly

```text
Q_s K_s^T - (Q_a K_a^T + Q_b K_b^T - Q_x K_x^T)
  = delta_a_Q delta_b_K^T + delta_b_Q delta_a_K^T,
```

including the head-size scale and causal mask.  This formula deliberately uses
the synthesized `s`, not the almost-additive real `ab` state.  The copied `X`
and `V` prefix has no bilinear defect.

Each `grammatical_qk_causal` record independently retains:

1. the score defect measured by running `F(s)`;
2. `delta_a_Q delta_b_K^T` and `delta_b_Q delta_a_K^T` as separate complete
   score-table vectors;
3. the score vector reconstructed from their sum;
4. its complete residual vector;
5. the adjacent transition `delta_tau`;
6. the same root observation calculated directly from the two QK outputs;
7. the root residual after subtracting only the analytic cross terms from
   `F(s)` and running the untouched suffix.

For layer 0, the real-model results are:

| Model and action square | `aQ,bK` | `bQ,aK` | Measured score defect | Reconstruction defect | QK `delta_tau` | Fraction after removal |
|---|---:|---:|---:|---:|---:|---:|
| Stories15M dog/dogs × runs/run | 0 | 1.6305688 | 1.6305690 | 2.90e-6 | 3.8370042 | 2.53e-5 |
| Stories15M the/my × path/foot | 0 | 1.3331223 | 1.3331234 | 3.08e-6 | 3.6344222 | 2.78e-5 |
| Stories260K He/They × is/are | 0 | 2.1544359 | 2.1544336 | 1.74e-5 | 0.0484339 | 1.94e-4 |
| Stories15M dog/cat × runs/plays control | 0 | 1.0102187 | 1.0102196 | 3.41e-6 | 1.6753862 | 8.00e-5 |

In every retained rectangle, action `a` changes the earlier token and action
`b` the later token. At layer 0 the causal mask therefore makes
`delta_a_Q delta_b_K^T` exactly zero: an earlier query cannot inspect a later
key. The entire first-layer cross term is the opposite direction, in which
the later edited token queries the earlier edited token. This is useful typed
structure for subsequent basis discovery, but it is generic causal
directionality rather than evidence of grammar selectivity. At later layers
both directed terms can be nonzero because earlier attention has transported
the edits across positions.

The copied `(X,V)` prefix defect is exactly zero in all retained records.  The
direct QK root difference and independently recorded transition vector agree
to at worst `4.51e-7` across every layer in these runs.  Across every layer,
the largest analytic score reconstruction relative defect is `2.44e-5`, and
the largest fraction of a QK root increment remaining after cross removal is
`1.94e-4`.

This establishes the causal architectural result: the two bilinear QK cross
terms account for essentially the entire root-visible QK transition, and
removing only those terms returns the propagated effect to the prior numerical
floor.  It does not establish grammar selectivity.  The all-grammatical
dog/cat × runs/plays control obeys the same identity, and its final mixed norm
is larger than either Stories15M grammatical example.  Norm magnitude is
therefore not a Firthian score.  Selectivity must be tested in the retained
vector directions and their action/continuation closure across many positive
and controlled rectangles.

The grammar executable constructs its self-referential continuation term
directly in caller-owned storage. Its maps and pullbacks therefore retain
valid environments rather than pointers into a returned local structure.
Trace schema 3 also gives the analytic formula and tensor distinct JSON keys.
All four retained traces were regenerated after these corrections; schema 2
traces are superseded.
For each of the five supplied terms, the executable runs stock `forward()`
and compares every post-final-RMS token state with the complete-frontier term.
Across the four retained runs the largest relative stock-forward defect is
`1.04e-6`. This parity concerns hidden states; logits are still outside the
root observer.

### Retained system cases

The Stories15M agreement case is:

```bash
make cpsgrammaractions CC=clang
./cps_grammar_actions ../llama2.c/test/stories15M.bin tokenizer.bin \
  "The dog runs." "The dogs runs." "The dog run." \
  "The dogs run." "The dogs run." --root last \
  --trace outputs/cps-grammar-agreement-dog-runs-15m.jsonl
```

The constructor edits commute exactly, so every commutator vector is zero.
The last-position mixed root effect has L2 norm `13.616778`.  At the first
three boundaries, the root-visible torsor difference remains at the numerical
floor:

| Boundary | Local mixed L2 | Torsor-visible L2 |
|---|---:|---:|
| token embedding | 5.54e-9 | 9.52e-5 |
| attention RMS pair | 1.82e-7 | 9.25e-5 |
| Q/K/V projection | 2.54e-7 | 1.11e-4 |
| QK contraction | 1.630569 | 3.836970 |

The QK contraction is therefore the first substantial join in this typed
term, roughly `3.5e4` times the preceding root-visible cancellation level.
Softmax and value contraction transform the joint relation; later layers act
on already contextualized states.

The earlier possessive/noun square gives the same layer-0 boundary:

```text
The cat walked across the path.
The cat walked across my path.
The cat walked across the foot.
The cat walked across my foot.
```

Its embedding, RMS, QKV, and QK torsor-visible norms are respectively
`1.04e-4`, `9.62e-5`, `1.21e-4`, and `3.6344`.  The complete last-position
mixed root norm is `14.473706`.

An independently token-aligned Stories260K agreement square uses `He/They`
and `is/are`.  Its corresponding torsor-visible values are `1.12e-5`,
`1.04e-5`, `1.05e-5`, and `4.84e-2`; the same QK boundary is visible despite
the much smaller five-layer, width-64 model.

All four runs report zero typed-chain output defect: each complete factored
attention or FFN chain reproduces its monolithic map for every supplied
constructor term. This is chain-level parity, not an independently captured
reference after each intermediate stage. These finite cases support the
operational boundary measurement. They do not yet recover a global grammar,
establish an exhaustive continuation eigenspace, or compile a faster inference
schedule. The next basis should be generated from many such retained action
differences and their higher-order Möbius compositions, with closure checked
on unseen actions before any transport rule is accepted.

### Matched controller-attractor continuation geometry

The first controlled relation-family experiment keeps the two action tokens
at exactly the same positions while changing their grammatical role.  A
controller square has the form

```text
x   = Near the bird the dog runs.
ax  = Near the bird the dogs runs.
bx  = Near the bird the dog run.
abx = Near the bird the dogs run.
```

Here either single edit breaks subject--verb agreement and the joint edit
repairs it.  Its matched attractor square is

```text
x   = The bird near the dog runs.
ax  = The bird near the dogs runs.
bx  = The bird near the dog run.
abx = The bird near the dogs run.
```

The edited target is now inside the intervening PP, so pluralizing it does not
repair the verb edit.  In both squares the target-number action is at token
position 5 and the verb-number action is at position 6.  The corresponding
token constructors at those positions are identical.  The program rejects,
rather than pads, any diagram that does not have this type.

At layer 0, QKV projection is tokenwise.  Consequently, the two retained
directed QK cross tensors are bit-identical between every matched controller
and attractor case.  Across all 44 matched pairs their maximum absolute
difference is exactly `0`.  The local bridge therefore contains no controller
label.  What changes is the continuation in which that bridge is observed.
Even the layer-0 QK `delta_tau`, which sends the same local interaction through
the different learned suffixes, need no longer agree.

The analysis retains four representations:

1. both local directed layer-0 QK cross tensors;
2. the final post-RMS mixed pullback vector;
3. the six layer-indexed QK `delta_tau` vectors, concatenated rather than
   summed;
4. all 79 ordered typed-boundary `delta_tau` vectors, likewise concatenated.

These are torsor differences, so their collection is a vector space without
choosing an origin in hidden-state space.  For training vectors `v_i` of one
relation, an uncentered SVD supplies an orthonormal row-span `B_R`.  The only
diagnostic is

```text
epsilon_R(v) = ||v - B_R^T B_R v|| / ||v||.
```

No residual, norm, or sum becomes a completion reward.  The primary held-out
comparison asks whether a controller vector is closer to the controller span
than its matched attractor vector.  A second, symmetric diagnostic fits both
spans and asks whether each held-out vector is nearer its own span.  Strict
ties count as failures.

Exploration uses the `near`, `with`, and `under` templates and twelve lexical
families split into three disjoint folds.  Each of nine splits holds out one
complete template and one lexical fold, trains on the remaining 16 vectors per
role, and tests four matched pairs.  The confirmation was frozen afterward:
fit all 36 exploration vectors per role, then test an unseen `By/by` template
with eight new target inflections and contexts.

| Retained representation | Exploration relation | Exploration nearest | Confirmation relation | Confirmation nearest |
|---|---:|---:|---:|---:|
| local layer-0 directed QK | 0/36 | 0/72 | 0/8 | 0/16 |
| final mixed root | 25/36 | 53/72 | 7/8 | 13/16 |
| layer-indexed QK transitions | 32/36 | 62/72 | 6/8 | 14/16 |
| all typed transitions, scale-indexed | 34/36 | 62/72 | 8/8 | 16/16 |

For the scale-indexed representation, the primary exploration comparison is
`12/12` on `near`, `12/12` on `under`, and `10/12` on `with`.  All eight
confirmation margins have the expected sign; they range from `0.00371` to
`0.08412`.  A post-hoc per-layer view of the suffix-pulled QK transition gives
relation-basis results of `21`, `24`, `29`, `24`, `32`, and `25` wins out of
36 from layers 0 through 5.  Layer 4 is the largest in this finite sample, but
it was not a prespecified layer and is not reported as a discovered universal
boundary.

The distinction between the endpoint and the scale-indexed result is the
important one: folding the observations to one final mixed vector loses
relation information that remains available when each learned scale keeps its
own observation and the observations are composed as a product.  This is
finite evidence for the proposed continuation role of context, not yet a
complete grammar or an inference rule.

#### Recursive residual-block pullback closure

The next experiment asks a narrower closure question without replacing the
retained vectors by norms.  For each of the six attention residual maps and
six SwiGLU residual maps `F`, let `k` be the exact suffix continuation from
that block to the final post-RMS hidden state, and write

```text
C_F^g(x) = (U_a-I)(U_b-I)(k after F^g)(x).
```

The evaluator retains `C_F^0` through `C_F^3` as complete 288-dimensional
vectors.  It independently constructs `make_pullback(F,k)` and verifies that
its mixed observation is exactly `C_F^1`; the maximum composition defect over
all 1,056 block/case pairs is `0` at recorded precision.  This is a check of
the CPS term, not evidence of closure.

At dictionary depth `d`, the scale-indexed function tables are

```text
X_d = [C_F^0, ..., C_F^(d-1)] over all twelve blocks,
Y_d = [C_F^1, ..., C_F^d]     over all twelve blocks.
```

The analysis tests both ambient root coordinates and coordinates projected
through a relation basis fitted only on the training split.  It separately
measures representation, descent through the sampled-function quotient, and
prediction of the complete shifted table on unseen templates and lexemes.
The latter two tests prevent an underdetermined fit from being called an
operator.

For the controller confirmation split, the training-relation projection gives:

| Depth | Function columns | Sampled rank | Fit descent | Held-out shifted table | Identity baseline | Exploration CV held-out mean |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 432 | 36/36 rows | 0.71899 | 0.59165 | 1.01173 | 0.40918 |
| 2 | 864 | 36/36 rows | 0.67192 | 0.51224 | 0.86152 | 0.39986 |
| 3 | 1296 | 36/36 rows | 0.65174 | 0.75847 | 0.82096 | 0.56440 |

All nine exploration folds likewise saturate their 16 available controller
rows at every depth.  The attractor fits behave similarly: confirmation
descent is `0.71201`, `0.67036`, and `0.64629`, while held-out error is
`0.60894`, `0.63913`, and `0.77067`.  In ambient coordinates, controller
held-out errors are `0.88144`, `0.85145`, and `0.92396`.  Although adding a
second generation helps the projected controller fit modestly, the third
generation reverses that improvement.  Descent remains large throughout.

The sampled input tables have full row rank, so their approximately
`1e-15` fit representation defects are interpolation artifacts, not evidence
for a small quotient.  This identifies the chosen residual-block pullbacks
and hidden-coordinate observer as insufficient coordinates for the small,
stable, pullback-closed continuation algebra.

The mismatch is localized to the operation actually tested: repeatedly
pulling the suffix through one endomorphic residual block.  It does not test
the stronger grammatical-action closure law

```text
U_a K_grammar subset K_grammar,
```

where `a` extends or transforms a grammatical context.  A residual block and
a token/context action are not interchangeable.  The following experiment
therefore tests one real grammatical action directly on the scale-indexed zip.

#### Direct grammatical role-action quotient

Every matched pair supplies an exact involutive context action `R`: exchange
the controller diagram with its PP-attractor diagram while keeping the two
number edits and their token positions fixed.  Thus

```text
R(controller) = attractor,
R(attractor) = controller,
R^2 = I.
```

Let `Z(x)` be the complete 79-boundary typed-transition zip, of width 22,752.
On exploration pairs only, an uncentered SVD supplies rank-`r` continuation
coefficients `P_r`.  The induced action `M_r` is fitted from

```text
q_r(x)  = Z(x) P_r,
q_r(Rx) = q_r(x) M_r.
```

The frozen confirmation tests the same equation on the unseen `by` template
and eight unseen lexical families.  It also tests `M_r^2=I` and measures how
much of both the complete zip and the controller--attractor difference the
quotient discarded.  No rank is selected post hoc; the artifact retains the
whole prespecified curve.

| Rank | Unseen zip residual | Unseen role-difference residual | Action error | Identity error | Involution error |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.98777 | 0.99426 | 0.62129 | 0.62896 | 0.05122 |
| 3 | 0.96746 | 0.99170 | 0.46002 | 0.46597 | 0.05445 |
| 8 | 0.94951 | 0.97558 | 0.59441 | 0.64184 | 0.26171 |
| 16 | 0.92601 | 0.94184 | 0.62609 | 0.81619 | 0.23072 |
| 32 | 0.90335 | 0.90383 | 0.80771 | 0.91467 | 0.29959 |
| 72 | 0.87897 | 0.87577 | 1.01490 | 0.92805 | 4.64e-15 |

The low-rank action errors are misleading in isolation.  At rank 3 the
quotient discards `99.17%` of the unseen role-difference norm, and its action
prediction improves on identity by only `0.00595`.  At full training rank the
fitted map is an exact involution, but its unseen action prediction is worse
than identity.  All nine exploration folds show the same obstruction: their
32-row tables have rank 32.  At rank 1 the mean held-out role-difference
residual is `0.99784`; at rank 32 it remains `0.93544`, while action error has
risen to `0.89975` versus a `0.67368` identity baseline.

To avoid assuming that the relevant grammatical mode is among the largest
singular directions, the same test also uses the exact action-adapted parity
decomposition

```text
Z_even = (Z_controller + Z_attractor) / 2,   R Z_even = +Z_even,
Z_odd  = (Z_controller - Z_attractor) / 2,   R Z_odd  = -Z_odd.
```

Both training families have rank 36/36.  Their smallest retained singular
values are still respectively `0.0383` and `0.0896` of the largest, so there
is no numerical rank collapse.  Even their complete training spans leave
confirmation residuals of `0.90397` and `0.90324`.

This direct action result therefore identifies the current uncentered linear
zip as a more specific observer interface than the tested role action can
carry uniformly: on these 44 matched diagrams it cannot simultaneously retain
the grammatical role distinction and transport that action to unseen
contexts. The earlier 8/8 relative classification of the large zip is real,
but relative separation is not closure. This measurement does not say that a
smaller company-demand interface is absent; it says the complete zip does not
factor through the proposed uniform linear transport. No completion observer
or inference speedup follows from this particular factorization.

Absolute confirmation residuals for the all-transition controller span remain
between about `0.81` and `0.96`; only their matched relative ordering is being
tested.  Some margins are small.  More templates, lexical families, and other
dependencies are therefore required before treating the sampled span as
closed.  The experiment makes no speedup claim.

All 88 source traces are real Stories15M executions.  Their maximum complete
typed-chain output defect is `0`, their maximum relative stock-`forward()`
hidden defect is `1.48e-6`, and their maximum componentwise telescoping defect
is `1.19e-7`.  The largest QK analytic reconstruction relative defect is
`5.37e-5`; after removing the analytic cross terms, the largest remaining root
fraction is `6.31e-4`.

The manifest, collector, analyzer, and compact result are respectively
`grammar_relation_cases.json`, `gather_grammar_relations.py`,
`analyze_grammar_relations.py`, and
`outputs/cps-grammar-relations-analysis.json`.  The raw vectors are omitted
from Git because of their size and can be regenerated with:

```bash
make grammarrelations CC=clang
```

#### Future-company observational refinement

The held-action discrepancy in the role-action regression above asked a chosen
geometric zip to carry an action. The next experiment changes the object being
represented. For a
grammatical interaction diagram `x`, a future action word `w`, and the complete
288-coordinate post-final-RMS observation at the last token, it retains

```text
V_x(w) = (U_a-I)(U_b-I)(k after w)(x).
```

The same suffix `w` is appended to all five terms `x`, `a(x)`, `b(x)`,
`b(a(x))`, and `a(b(x))`; no role label enters the transformer or the
observation. The retained five corner roots reconstruct `V_x(w)` exactly. For
a primitive future action `c`, the analyzer then forms

```text
T_x(c) = V_x(c) - V_x(epsilon)
       = (U_c-I)(U_a-I)(U_b-I)k(x).
```

This is the explicit eight-corner, third-order Möbius observation. For an
ordered depth-two word `cd`, it separately retains

```text
Q_x(c,d) = V_x(cd) - V_x(c) - V_x(d) + V_x(epsilon).
```

Nothing is summed across mutually exclusive completions, and none of these
vectors becomes a completion reward.

The fixed action manifest contains an identity, an independently executed
identity repeat, nine primitive future contexts, and all sixteen ordered
depth-two compositions of four of those primitives. The primitive families
include number-sensitive pronouns, a number-neutral witness, a clause
extension, later attachments, lexical noun phrases, and a modifier. Applying
all 27 words to all 88 controller/attractor diagrams produced 2,376 real
Stories15M traces.

Partition refinement happens before SVD. The initial matched-pair partition
has 44 two-member blocks. Within each block, the controller and attractor have
bit-identical layer-0 directed QK tensors, as established by the preceding
experiment. At every refinement step the analyzer searches the retained
future words for one whose **complete vector** separates a current block. It
does not use controller/attractor labels, vector norms, or a learned
classifier. Exact comparison is meaningful here because all 88 independent
identity reruns are bit-identical, with maximum componentwise difference `0`.

The exact refinement is decisive about this observer family: complete hidden
root coordinates retain contextual identity rather than the coarser
observational equivalence needed for computational sharing.

* the no-extension observation `V(epsilon)` already splits all 44 matched
  blocks;
* every one of the nine primitive third-order observations splits all 44;
* every one of the sixteen fourth-order conditional observations also splits
  all 44;
* starting from one global block, any primitive third-order observation splits
  all 88 diagrams into singletons.

Number-neutral, lexical, and modifier controls split the blocks just as
completely as the number-sensitive continuations. Exact separability is
therefore generic contextual fingerprinting, not by itself a grammatical
result. The grammatical question is addressed only by the disjoint held-out
role comparison below.

Consequently, equality under all primitive observations has no nontrivial
pairs on which to test extension congruence. The recorded congruence check has
zero violations but is explicitly marked vacuous; it is not closure evidence.

Only after that exact result does the analyzer inspect the behavior matrices:

| Behavior rows | Matrix | Numerical row rank | 99% energy rank | Smallest/largest retained singular value |
|---|---:|---:|---:|---:|
| all absolute future observations | 88 x 7,488 | 88/88 | 63 | 0.01895 |
| all future increments over identity | 88 x 7,200 | 88/88 | 64 | 0.02108 |
| nine primitive third-order observations | 88 x 2,592 | 88/88 | 65 | 0.02276 |
| sixteen fourth-order conditional observations | 88 x 4,608 | 88/88 | 66 | 0.02483 |

There is no numerical rank collapse in this sample. In particular, moving
from the geometric zip to complete-coordinate future behavior still retains a
fully specific interface; it does not by itself expose which projections the
current company actually demands.

There is nevertheless a reusable relative signal in how future company
changes the interaction. Labels are introduced only after behavior
construction and counterexample selection, using the same nine exploration
folds and frozen unseen `by` confirmation as the earlier experiment:

| Behavior representation | Exploration relation | Exploration nearest | Confirmation relation | Confirmation nearest |
|---|---:|---:|---:|---:|
| absolute future behavior | 19/36 | 56/72 | 3/8 | 13/16 |
| all future increments | 25/36 | 56/72 | 8/8 | 13/16 |
| primitive third-order zip | 24/36 | 56/72 | 7/8 | 14/16 |
| fourth-order conditional zip | 25/36 | 51/72 | 4/8 | 14/16 |

For the all-future-increment representation, all eight primary confirmation
margins have the expected sign, from `0.00581` through `0.08803`. An
independent direct SVD of the 36 controller and 36 attractor exploration rows
reproduces the same 8/8 and 13/16 results. The imperfect exploration and
symmetric results matter: this is evidence that response to future company
carries grammatical-role information, not evidence that the sampled spans are
semantic states.

The experiment therefore resolves the immediate ambiguity. The behavioral
object exposes distinctions that the identical local bridge cannot contain,
and subtracting the identity behavior transfers substantially better than the
absolute endpoint behavior. But complete hidden-root coordinates distinguish
every sampled context, so their exact observational quotient is the discrete
88-state partition. More future words can only refine that complete interface;
they cannot reveal which projections were unnecessary to a particular
constructor. Sharing must instead be derived by factoring each active
continuation through the projections it demands, as in the edge experiment
below. It cannot be obtained by fitting another global matrix to these rows or
by treating the relative span residual as an inference score.

The maximum stock-`forward()` hidden relative defect over all 2,376 traces is
`1.57e-6`. Mixed and commutator vectors reconstruct from the retained roots
with maximum defect `0`; the independently ordered eight-corner third-order
expansion differs by at most `3.81e-6`, at float32 cancellation scale. The
manifest, collector, analyzer, and compact result are
`grammar_future_actions.json`, `gather_grammar_behaviors.py`,
`analyze_grammar_behaviors.py`, and
`outputs/cps-grammar-behaviors-analysis.json`. Raw vectors remain outside Git
and can be regenerated with:

```bash
make grammarbehaviors CC=clang
```

### Projection--injection demand at token edges

The preceding endpoint experiments asked complete hidden coordinates to
identify complete contexts. That retains every projection. The resulting
singleton partitions and full ranks are facts about that observer interface,
but they are not a test of whether two computations can share the smaller
interface demanded by their current company.

The relevant algebra is contravariant. If a product projection

```text
pi_S : A -> A_S
```

forgets features, then precomposition embeds exactly the continuations that do
not demand them:

```text
pi_S^* : R^(A_S) -> R^A,    k |-> k . pi_S.
```

On the filler side, token alternatives are coproduct injections. On the
observer side, the same coproduct is a product of token-indexed observations:

```text
R^(sum_t Token_t) = product_t R^(Token_t).
```

Consequently a token constructor and a hidden observation must be kept as two
typed roles. At prefix `p`, the transformer supplies the complete codata

```text
q(h_p) = [logit_t(h_p) - logit_BOS(h_p)]_t.
```

Consuming constructor `t` selects `iota_t^* q(h_p)` and advances the hidden
state. A word is therefore a Mealy-style zip of edge observations and
constructors, not a fold of token probabilities:

```text
(q(h_p), t_{p+1}), (q(h_{p+1}), t_{p+2}), ...
```

Only after an edge is consumed may the next state forget the projection that
was needed to choose it. Observing only the final post-suffix state can
therefore miss a law that the model used correctly at an earlier constructor.

Schema 2 of `cps_grammar_cube.c` emits one flushed
`grammatical_cube_edge_zip` record for every real token edge in both cube
fibers. Each record contains the four corner constructor IDs and the complete
carrier/A/B/AB Moebius coefficients of the pre-constructor token-contrast
codata. The checks establish:

* the B/AB state coefficients before the unconsumed B constructor are exactly
  zero;
* every edge's four raw codata corners reconstruct from its Moebius
  coefficients;
* the factored typed chains still reproduce the stock computation;
* observations at distinct positions remain separate and no scalar sequence
  score or probability is introduced.

For the main verb, let `A` pluralize the target noun and let the singular and
plural verb constructors be `iota_s` and `iota_p`. The first measured cell of
the projection--injection demand lattice is

```text
D_A[(iota_p^* - iota_s^*)q](x)
 = (q_p(Ax)-q_s(Ax)) - (q_p(x)-q_s(x)).
```

This is not a completion reward. It asks whether this particular constructor
contrast demands the number feature exposed by `A`. Higher feature actions
and their mixed differences will add further cells; composing those demand
interfaces recursively is the polynomial-grammar task.

#### Known-law trained control

`synthetic_grammar_control.py` trains the repository's real `model.py`
transformer on the finite language generated by the same controller/attractor
manifest and the stock tokenizer. Its exhaustive teacher-forced checks cover
all generated choices before the C trace is inspected:

| Choice | Matches | Minimum margin |
|---|---:|---:|
| main verb | 640/640 | 6.891069 |
| future pronoun | 640/640 | 5.660831 |

The unchanged schema-2 C evaluator then applies all 88 contexts and nine
future actions (792 cubes). Its edge analysis recovers every retained choice:

| Edge choice | Manifest matches | Minimum margin |
|---|---:|---:|
| main verb | 176/176 | 6.891069 |
| future pronoun | 176/176 | 5.660831 |

The main-verb demand response separates the intended interfaces. Controller
contexts have mean response `29.502194` (minimum `22.231576`); attractor
contexts have mean `1.484175` (range `-0.517457` to `4.539331`). At the future
pronoun edge, the grammatical path response is `12.692550` for controllers and
`0.003335` for attractors. The independent exhaustive and C-edge minimum
margins agree exactly. The maximum edge Moebius inverse defect is
`9.54e-7`, the maximum stock logit-contrast relative defect is `3.53e-8`, and
typed-chain output defect is zero.

This also explains why its former endpoint fit looked poor: after the model
has emitted the correct verb or pronoun constructor, the terminal state need
not retain the number demand that was just discharged.

#### Stories15M demand slice

The identical 792-cube system run on Stories15M produces the following
constructor decisions. “Match” means agreement with the supplied grammatical
manifest; every row still records the model's actual decoded constructors and
both competing logit contrasts.

| Branch | Manifest matches | Mean expected-minus-alternative margin |
|---|---:|---:|
| controller baseline `x` | 27/44 | -0.036468 |
| controller plural `AB` | 44/44 | 2.042212 |
| attractor baseline `x` | 13/44 | -0.886815 |
| attractor target-plural `A` | 4/44 | -1.544800 |

The number-projection response is positive in both roles: mean `2.005744` for
controllers and `0.657985` for attractors. Thus the small model's verb
observer strongly demands plural target number even when that target occupies
the PP-attractor position. In the clearest comparison, every edited controller
selects its plural verb, but 40/44 edited attractors also prefer the plural
alternative instead of retaining the singular controller agreement. This
localizes the familiar attraction behavior as a demand-interface fact: this
model has not made the target-number projection irrelevant in the attractor
construction.

Future pronoun constructors match the manifest on 110/176 branches. Their
controller grammatical-path response has mean `3.007364`; the attractor path
response remains `1.534084`, again showing that the supposedly irrelevant
edit is still observed by this model.

The Stories run has zero typed-chain output defect, maximum edge Moebius
inverse defect `9.54e-7`, maximum stock hidden relative defect `1.48e-6`, and
maximum stock logit-contrast relative defect `1.76e-6`. The two compact,
decoded artifacts are
`outputs/cps-synthetic-grammar-edge-demand.json` and
`outputs/cps-stories15m-edge-company-analysis.json`.

This establishes one number/constructor slice, not the whole polynomial
grammar. The next inference task is to add independent feature actions and
recover the minimal demanded projection sets for each constructor family,
then compose those sum-of-products interfaces upward. Complete hidden states
may remain unique throughout; sharing depends on common demanded projections,
not equality of full values.

#### Joint controller--attractor number demand

The separate controller and attractor diagrams above measure one axis at a
time. `gather_number_demand_cubes.py` now constructs the decisive aligned
eight-corner cube inside each attractor sentence:

| Corner | Controller | Attractor | Verb constructor |
|---|---|---|---|
| `x` | singular | singular | singular |
| `A` | singular | plural | singular |
| `B` | singular | singular | plural |
| `AB` | singular | plural | plural |
| `C` | plural | singular | singular |
| `AC` | plural | plural | singular |
| `BC` | plural | singular | plural |
| `ABC` | plural | plural | plural |

All three actions must occupy independent aligned token positions. The C
evaluator's aligned-constructor mode validates the complete factorized cube;
it does not pad, retokenize, or weaken the existing square check. All 44
manifest cases pass this exact token typing.

At the pre-verb edge, B is still unconsumed. Its state coefficients are
therefore exactly zero, while its two constructor coordinates define

```text
L = q_plural - q_singular.
```

The analyzer retains all 83 or 84 coordinates of

```text
q, D_A q, D_C q, D_C D_A q
```

as well as the complete number-demand spectrum

```text
D_C L, D_A L, D_C D_A L.
```

The distinction requested by the interface algebra remains explicit:

* exact factorization requires every coefficient involving a forgotten
  projection to vanish;
* decision-preserving factorization requires only that deleting that
  projection never change the selected injection.

No epsilon is chosen. Every complete coefficient and every decoded decision
margin is retained in the output.

For the known-law trained control:

| Coefficient | Minimum | Mean | Maximum |
|---|---:|---:|---:|
| `D_C L` | 21.493332 | 29.562254 | 36.056793 |
| `D_A L` | -0.517457 | 1.484175 | 4.539331 |
| `D_C D_A L` | -3.143678 | -0.275321 | 3.166681 |

Thus the trained network does not have exact attractor-number factorization:
its nuisance first- and second-order coefficients are nonzero. Nevertheless,
all four branches match the generated grammar in all 44 cubes (176/176), and
dropping attractor number preserves the verb constructor for both controller
numbers in 44/44 cases. On the unseen `by` template, controller response has
mean `30.654978`, compared with attractor mean `1.864542` and mixed mean
`0.098305`.

Stories15M gives:

| Coefficient | Minimum | Mean | Maximum | Positive cases |
|---|---:|---:|---:|---:|
| `D_C L` | -2.457022 | 0.404440 | 2.171251 | 32/44 |
| `D_A L` | -2.342069 | 0.657985 | 3.029380 | 34/44 |
| `D_C D_A L` | -0.580619 | 0.707283 | 3.318953 | 37/44 |

Its branch decisions make the interface failure concrete:

| Branch | Manifest matches | Meaning |
|---|---:|---|
| `x` | 13/44 | both nouns singular |
| `A` | 4/44 | only attractor plural |
| `C` | 34/44 | only controller plural |
| `AC` | 44/44 | both nouns plural |

Changing attractor number preserves the selected verb in 31/44
controller-singular strata and 34/44 controller-plural strata; it preserves
both in only 28/44 cubes. The mean mixed coefficient `0.707283` is larger than
either mean first-order coefficient, so the result is not adequately
described as independent additive contamination. Controller demand itself is
being changed by attractor number.

This repeats on the held-out `by` construction. Its eight cases have mean
controller, attractor, and mixed responses `0.565103`, `1.582944`, and
`0.452243`; all eight attractor responses are positive. The smallest absolute
mixed coefficient across all 44 cases is `0.025945`, while the maximum stock
logit-contrast L2 defect is `8.44e-5`. The typed-chain output defect and
unconsumed-B leak are both exactly zero; the maximum cross-fiber Moebius
inverse defect is `1.78e-15`.

The compact artifacts are
`outputs/cps-synthetic-grammar-number-demand-analysis.json` and
`outputs/cps-stories15m-number-demand-analysis.json`. Raw traces remain outside
Git and the Stories experiment is reproduced by:

```bash
make numberdemandcubes CC=clang
```

This recovers the complete number-demand support for one verb-constructor
family. The next closure question is not a matrix action on hidden states. It
is whether directly measured composite demand agrees with polynomial
substitution of the recovered component interfaces on unseen lexemes,
templates, and company.

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
2. the score vector reconstructed from the two analytic cross terms;
3. their complete residual vector;
4. the adjacent transition `delta_tau`;
5. the same root observation calculated directly from the two QK outputs;
6. the root residual after subtracting only the analytic cross terms from
   `F(s)` and running the untouched suffix.

For layer 0, the real-model results are:

| Model and action square | Measured score defect | Reconstruction defect | QK `delta_tau` | Root after cross removal | Fraction remaining |
|---|---:|---:|---:|---:|---:|
| Stories15M dog/dogs × runs/run | 1.6305690 | 2.90e-6 | 3.8370042 | 9.72e-5 | 2.53e-5 |
| Stories15M the/my × path/foot | 1.3331234 | 3.08e-6 | 3.6344222 | 1.01e-4 | 2.78e-5 |
| Stories260K He/They × is/are | 2.1544336 | 1.74e-5 | 0.0484339 | 9.39e-6 | 1.94e-4 |
| Stories15M dog/cat × runs/plays control | 1.0102196 | 3.41e-6 | 1.6753862 | 1.34e-4 | 8.00e-5 |

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

All three runs report zero typed-stage output defect: the factored attention
and FFN chains reproduce their monolithic maps for every supplied constructor
term.  These finite cases support the operational boundary measurement.  They
do not yet recover a global grammar, establish an exhaustive continuation
eigenspace, or compile a faster inference schedule.  The next basis should be
generated from many such retained action differences and their higher-order
Möbius compositions, with closure checked on unseen actions before any
transport rule is accepted.

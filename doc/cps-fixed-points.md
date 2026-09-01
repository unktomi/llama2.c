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

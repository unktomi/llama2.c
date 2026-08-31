# Contextual coupling and fixed-mode trace

`semantic_fixed_points.c` asks where two concrete grammatical edits first
meet in the frozen transformer and how their actual directional derivatives
are transported afterward. It is a diagnostic around the reference kernels,
not a completion scorer or an alternative inference policy.

## The semantic square

Four equally tokenized strings supply two commuting token-constructor edits:

```text
text00 -- A --> text10
  |                 |
  B                 B
  v                 v
text01 -- A --> text11
```

At each captured boundary, the program records

```text
edit A = h10 - h00
edit B = h01 - h00
omega  = h11 - h10 - h01 + h00
```

The `stage` records contain the norms of these finite arrows and of `omega`.
A nonzero `omega` says that the two edits have coupled by that boundary. It
does not by itself say whether any later continuation retains the coupling.

The per-head `attention_relation` records retain all four raw QK scores and
softmax probabilities at both edited key positions. At the first layer, the
mixed raw score is the learned comparison

```text
(q_noun1 - q_noun0)^T (k_determiner1 - k_determiner0) / sqrt(head_size).
```

## Analytic tangent path

The base execution is `text00`. Two forward tangent states are seeded with
the actual embedding arrows for A and B. Their own K/V caches carry edits from
earlier tokens into later causal attention. The derivative path implements the
JVP of every operation used by the reference layer:

- RMSNorm, including its radial derivative;
- Wq, Wk, Wv, Wo, W1, W3, and W2;
- RoPE;
- QK contraction;
- softmax via `p * (ds - dot(p, ds))`;
- the value sum via `dp * v + p * dv`;
- SiLU and the SwiGLU product;
- both residual additions.

`tangent_stage` records contain the two derivative norms and their cosine at
every numerical boundary. They are analytic derivatives at `text00`, not
finite differences between the other three corners.

For a same-width transition, let D contain its two captured input tangent
directions and let E contain the corresponding output tangents. The
`tangent_transition` record contains

```text
M = D^+ E
R = E - D M
```

When the captured input is the complete domain of that occurrence,
`E = J_F D`, so `M = D^+ J_F D`. The closest fixed defect is

```text
min_c ||E c - D c|| / ||D c||,
```

and the closest discarded gain is

```text
min_c ||E c|| / ||D c||.
```

The reported coefficients identify the corresponding combination of the A
and B directions. Rank is computed from the two-direction Gram matrix, so a
zero direction is not regularized into a fictitious second mode. The
`edit_a_defined`, `edit_b_defined`, and `modes_defined` flags distinguish an
undefined ratio from a numerical zero. For a rank-deficient span,
`transfer_fixed_defect` compares the transfer with the represented-space
projector `D^+ D`, rather than incorrectly comparing it with the full
coefficient-space identity.

At an observed-token attention boundary, the displayed noun vector is only a
projection of the complete causal state: earlier-token K/V entries are also
inputs. A rank increase or a large `subspace_escape` can therefore identify a
contextual direction arriving from the cache. It must not be described as a
full-layer eigenmode or as information created from nothing.

`secant_transition` applies the same small-space calculation to the two finite
corner arrows. It is labeled separately because `D^+ D_out` is a finite
secant transport, not `D^+ J_F D`.

## Run

The carpet/foot example does not align under the bundled 32k tokenizer, so the
checked trace uses the one-token noun alternatives `path` and `foot`:

```bash
make semanticfixedpoints CC=clang
./semantic_fixed_points ../llama2.c/test/stories15M.bin tokenizer.bin \
  "The cat walked across the path." \
  "The cat walked across my path." \
  "The cat walked across the foot." \
  "The cat walked across my foot." \
  --trace outputs/semantic-fixed-the-my-path-foot.jsonl
```

For this checkpoint, all four strings contain eight tokens. Edit A is
`" the" -> " my"` at position 5, edit B is `" path" -> " foot"` at
position 6, and position 6 is observed. In layer 0, the finite interaction is
zero at `layer_input`, `attention_rms`, and `query`; its norms are 1.3331225 at
the QK scores, 0.26783409 after softmax, 0.10466128 after the attention
residual, and 0.62536335 at the layer output. The largest edited-slot raw QK
interaction is head 4 against the determiner position, at +0.81148934.

The tangent trace also shows why current-token state alone is insufficient:
at the layer-0 noun input, A has norm zero and B has norm 0.48417742. At the
QK scores, their derivative norms are 4.2651102 and 12.495343. A reached the
noun through the retained determiner key.

The program deliberately omits final RMSNorm, the token classifier, and all
completion rewards. It also makes no one-shot physical-weight-read claim: the
extra model executions and JVP matmuls are measurement work. A continuation
pullback/Koopman trace is a separate next calculation needed to decide which
of these numerical couplings remain observationally visible.

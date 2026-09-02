# Modal composition: discussion checkpoint

This is a post-compaction record of the current discussion, not an
implementation, model measurement, or completed inference result.

## Current objective and user premise

The user wants one compiled operator that acts on a prompt representation
and yields the complete completion, without hiding the original eager
transformer evaluation in that representation or its readout. Fixed prefill
means fixed token commitments, not hidden states frozen across layers.

Use the actual model layers as operational scales; do not impose a
traditional linguistic parse tree as the computation. The user proposes
viewing the nonlinearities as copy/operate/reduce terms and, in modes, as
phase changes rather than energy gain or loss. This is a proposal to develop,
not a measured property of the current implementation.

The optimization condition remains observer-relative functional invariance:

    p . o = p

Pointwise equalities on a sample do not establish this identity. Interchange
must respect the remaining composed observation, not only an isolated readout.

## Code checked for this checkpoint

`run.c` computes the actual RMS normalization, Q/K/V, RoPE, causal score
contraction, softmax, value contraction, output projection, residual addition,
and SwiGLU. The SwiGLU update is

    W2 (SiLU(W1 N(x)) * W3 N(x)).

RoPE explicitly rotates coordinate pairs. The other kernels are not declared
norm-preserving. Equal input/output dimensions do not establish preservation
of an energy or inner product. Conversely, changing the ordinary hidden-state
norm does not disprove conservation in a larger representation of modes.

`cps_fixed_points.c:pullback_apply` still runs its stored map, then its stored
continuation. It is not a compiled modal matrix that eliminates those calls.
No such replacement was implemented or tested in this checkpoint.

## Exact linear lifting versus lossless lifting

In a free vector space with one basis element [x] per complete value, define

    copy [x] = [x] tensor [x]
    delete [x] = 1
    lift(f) [x] = [f(x)].

Extend these definitions linearly. Copying a linear combination retains
correlated copies, not every independent cross-combination. Composition is
exact: lift(g . f) = lift(g) lift(f). A lifted learned numerical matrix is
not the same object as its original small coordinate matrix.

Copy/delete laws by themselves do not imply unitarity. If f(x) = f(y) for
distinct values, lift(f)([x] - [y]) = 0. For instance, ordinary softmax
identifies score vectors differing by a common offset. Such identification
may be appropriate for the observer, but is not a phase rotation of that
nonzero difference in this value basis.

There is an exact lossless construction for deterministic finite-bit
primitives. Retain the input and an auxiliary register, and define

    R_f(x, z) = (x, z XOR f(x)).

This is its own inverse. Its value-basis matrix is a permutation, hence
unitary with the orthonormal basis inner product. Reversible primitive
implementations can be composed, rather than treating the whole model as an
oracle. The finite composed permutation has unit-modulus eigenvalues: in
its eigenbasis the action is by phases.

This establishes a mathematical realization, not a physical conserved energy
for the trained network, nor a small matrix or a faster evaluation. Reading
only f(x) omits the information retained by the larger reversible state.
History can also be uncomputed, but that requires computation rather than
free deletion. An observer-sufficient quotient could discard distinctions
earlier if its factorization is actually established.

The unresolved task is a compact representation closed under the actual
operations and sufficient for the demanded composed observations. A huge
value-basis matrix, reversible circuit, or unevaluated CPS expression alone
does not deliver the requested speedup or new completion criterion.

## Provenance and execution limits

The preceding committed model measurements are described in
`cps-fixed-points.md` under the residual-observer experiment. They concern
specific pointwise omissions and their joint effects, not a modal energy
law or a certified whole-model rewrite.

No new model runs, unit tests, probes, or fixtures were written or executed
here. The user requires permission for new tests/probes/fixtures. No
subagents or outside AI assistance were used. Rejected scorers were not
restored. The unrelated untracked synthetic-analysis output was left alone.

Primary mathematical references consulted:

* Coecke, Pavlovic, Vicary, [A new description of orthogonal bases](https://arxiv.org/abs/0810.0812): copying/deleting basis structure.
* Bennett, [Logical Reversibility of Computation](https://www.cs.princeton.edu/courses/archive/fall06/cos576/papers/bennett73.html): reversible realization, retained history, and uncomputation.

These references support the stated constructions, not an empirical claim
that this transformer already has a compact lossless modal realization.

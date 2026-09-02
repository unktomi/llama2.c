# Authored samples for observer-relative commutation

These are requested linguistic sample families, not model results. No model
run, tokenizer-alignment check, or neural fixed-point certificate is claimed.
The observations below specify what meaning should be retained. They do not
declare a hidden coordinate, attention head, or logit to be that observation.

Each positive family specifies two transformations A and B. On the stated
family, the intended observation satisfies

```text
p(A(x)) = p(B(x)) = p(A(B(x))) = p(B(A(x))) = p(x).
```

The transformations also have explicit joint realizations; this is not just
a claim about two independent single-input measurements. Each family includes
a nearby transformation C to which p must be sensitive.

## 1. Agreement: retain controller number, ignore color and attractor number

The hole is the present-tense copula, with alternatives `is` and `are`.
The observation p is the grammatically agreeing copula, not the full vector
of the model's preferences.

* A replaces the controller's color `black` with `white`.
* B pluralizes the noun inside the `near` phrase: `dog` becomes `dogs`.
* C pluralizes the controller: `cat` becomes `cats`.

| Variant | Text | Expected p |
|---|---|---|
| x | The black cat near the dog ___ asleep. | is |
| A(x) | The white cat near the dog ___ asleep. | is |
| B(x) | The black cat near the dogs ___ asleep. | is |
| A(B(x)) = B(A(x)) | The white cat near the dogs ___ asleep. | is |
| C(x) | The black cats near the dog ___ asleep. | are |
| A(C(x)) | The white cats near the dog ___ asleep. | are |
| B(C(x)) | The black cats near the dogs ___ asleep. | are |
| A(B(C(x))) | The white cats near the dogs ___ asleep. | are |

Thus A and B remain invisible across both controller-number strata. C does
not. An observer of color or the number of nearby dogs would instead detect
A or B. The invariance belongs to p, not to the complete sentence meaning.

## 2. Voice and tense: retain the roles, ignore their surface realization

The observation p returns the agent and recipient of the described thanking.
It does not assert that a future event has already occurred.

* A changes active voice to passive voice, retaining participant roles.
* B changes past tense to future tense with `will`.
* C exchanges the two participant roles.

| Variant | Text | Expected p: agent, recipient |
|---|---|---|
| x | Roxy thanked Billy. | Roxy, Billy |
| A(x) | Billy was thanked by Roxy. | Roxy, Billy |
| B(x) | Roxy will thank Billy. | Roxy, Billy |
| A(B(x)) = B(A(x)) | Billy will be thanked by Roxy. | Roxy, Billy |
| C(x) | Billy thanked Roxy. | Billy, Roxy |
| A(C(x)) | Roxy was thanked by Billy. | Billy, Roxy |
| B(C(x)) | Billy will thank Roxy. | Billy, Roxy |
| A(B(C(x))) | Roxy will be thanked by Billy. | Billy, Roxy |

Unlike edits in disjoint character slots, A and B both reconstruct the verb
phrase. Their composition preserves the role observation even though word
order, auxiliaries, and tense change. A first-noun observer would fail here.
An observation of event time should detect B; an observation of grammatical
voice should detect A.

## 3. Longer company: retain participants across intervening descriptions

Base text:

> Roxy and Billy found a heavy box beside the river. Roxy wore a red hat.
> Billy wore a blue coat. The path was muddy, so they moved slowly. When they
> reached the bridge, they carried the box across together.

The observation p is the set of participants in the explicitly described
final carrying event: `{Roxy, Billy}`.

* A replaces `red hat` with `green hat`.
* B replaces `blue coat` with `yellow coat`.
* C replaces the final sentence with `When they reached the bridge, Roxy
  carried the box across alone.`

The four garment sentences are:

| Variant | Intervening description | Expected p |
|---|---|---|
| x | Roxy wore a red hat. Billy wore a blue coat. | Roxy and Billy |
| A(x) | Roxy wore a green hat. Billy wore a blue coat. | Roxy and Billy |
| B(x) | Roxy wore a red hat. Billy wore a yellow coat. | Roxy and Billy |
| A(B(x)) = B(A(x)) | Roxy wore a green hat. Billy wore a yellow coat. | Roxy and Billy |

The surrounding sentences remain exactly the base text. Applying C changes
p to `{Roxy}` in all four variants. The colors do not alter the stated
participants, but removing Billy's participation does. This tests the same
invariance with intervening material rather than an isolated short clause.

## How these samples relate to the requested optimization

The linguistic expectation is inspectable. The remaining model-side work is
to locate a composed observation that realizes it, then determine which
actual typed operations preserve that observation over these variations.

Preserving the expected answer once is not sufficient. Nor does an unchanged
semantic role imply identical full hidden states, identical probabilities,
or a license to omit an entire attention block. The sought rewrite is tied to
the information demanded by the particular observation and must survive the
compositions shown above.

# Training method

Symbolic-U is not trained with gradients or reinforcement learning. The current
research method is closer to a combination of symbolic program induction,
inductive logic programming, hypothesis search, contrastive curriculum design,
explanation-based reuse and ablation.

## 1. Separate the OS from learned content

Before an experiment, explicitly list what the kernel already knows. A desired
semantic result does not count as learned if it was hidden in a parser, feature
name, sensor output or candidate rule.

This is why the research repeatedly removed conveniences such as fixed action
heads, relation names, entity types, named operations, lemma normalization,
case labels and POS categories.

## 2. Curriculum by identifiability

Training examples are selected so competing explanations can be distinguished.
The vision curriculum is a simple example:

```text
red triangle
blue triangle
green triangle
red quadrilateral
blue quadrilateral
green quadrilateral
```

Changing color while holding shape fixed and vice versa gives the learner a
chance to separate those factors. If two hypotheses remain observationally
indistinguishable, the desired result should remain `UNKNOWN` rather than being
invented by fiat.

## 3. Candidate symbolic structures

The learner searches a bounded family of candidate U/program structures. This is
still an inductive bias and must be documented. Symbolic-U does **not** currently
synthesize arbitrary algorithms from no structural priors.

A count example from K24c compared candidates such as:

```text
constant-one
nonempty-means-one
recursive-member-count
```

Only the recursive candidate was consistent with all count curricula.

## 4. Support and conflict

Candidate structures are tested against consequences. A typical accepted U has
supporting examples and zero relevant conflicts under the current policy.
Candidates that are refuted may receive `U=-1`.

Crucially:

```text
candidate U rejected
    does not imply
output proposition false
```

The latter requires positive support for an explicit opposite.

## 5. Backward validation

Forward enumeration can propose many candidate U. Query-guided backward proof
asks whether a candidate can actually support a target from available evidence.
This reduces irrelevant work and prevents a query from becoming evidence.

## 6. Blind evaluation

The research distinguishes curriculum examples from held-out tests. K24 used
new mixed image compositions. K25 went further by holding out disturbance
**combinations** while using the four existing test scenes only for evaluation.

A blind result is recorded as correct, false commit, contradiction or `UNKNOWN`.
Coverage and conditional accuracy should be reported separately when abstention
is possible.

## 7. Ablation as learning research

A central method in K1-K18 was to remove a formerly fixed distinction and ask:

- can the behavior be relearned from consequences?
- does another hidden cue leak the answer?
- is the distinction identifiable at all?
- which minimal assumption must remain?

A failure can therefore be useful evidence that a distinction is not currently
learnable from the provided curriculum.

## 8. Reuse and compilation

Repeated confirmed proofgraphs can become learned reuse structures and later
Macro-U. Compilation is therefore itself a form of experience-driven control
optimization, but it remains symbolic and dependency-aware.

## 9. Sensor selection training

K25 applies the same idea to perception. The learner receives:

```text
anonymous context measurements F1..F6
query identity
sensor provenance
training consequence
sensor cost
```

It does **not** receive rules like `IF DARK USE GAIN`. Confirmed `TRUST-U`
associate sensor/query/context combinations with successful proof paths. The
current conservative policy improves false-commit rate and cost but has low
coverage on hard unseen disturbance combinations.

## 10. What is still manually supplied?

The project is not assumption-free. Current manual priors include:

- token boundaries for language experiments;
- generic candidate-family grammars;
- the existence of identity/order/binding/search in the kernel;
- discrete progress assumptions for current arithmetic;
- classical visual preprocessing algorithms;
- target consequences in supervised curricula;
- selected resource costs in K25.

The research goal is to make these assumptions explicit and then test which can
be removed—not to hide them.

# Architecture

## 1. Hard semantic contract

Symbolic-U has two distinct ternary domains.

### Proposition (`KEY`) state

- `+1`: the proposition is positively provable;
- `0`: undecided / insufficient evidence;
- `-1`: an explicitly registered opposite is positively provable.

### Derivation (`U`) state

- `+1`: this derivation/link/operator is confirmed;
- `0`: it is pending/open;
- `-1`: this derivation/link/operator is rejected.

The architecture must preserve:

```text
U=-1 != KEY=-1
```

A rejected proof route is not negative evidence for its target. Negative belief
comes from positive proof of an explicit opposite Key.

If both a proposition and its opposite have positive support, the result is
represented as `KEY=0` plus `contradiction=True`. This is intentionally close to
a four-status evidence interpretation: true-only, false-only, neither, both.

## 2. Graph model

The conceptual structure is a typed, hypergraph-like network:

```text
KEY -> U -> KEY -> U -> KEY
```

A U may consume multiple Keys and produce a Key. Variables/ports allow reuse
across grounded instances. Context is part of proposition identity, and
provenance remains attached to proofs.

## 3. Query-guided execution

Backward proof is the semantic execution mechanism:

```text
QUERY target
  -> enumerate U whose output can match target
  -> bind variables
  -> recursively prove required input Keys
  -> combine positive/opposite support
  -> return +1 / 0 / -1 and contradiction metadata
```

Forward processing is used mainly to generate candidate structure or observations;
it is not allowed to turn the Query itself into evidence.

Cycles are detected and resource/budget limits bound search.

## 4. Minimal kernel hypothesis

The strongest current working hypothesis for the fixed substrate is:

```text
SYMBOL
PERSISTENT IDENTITY
ORDER

KEY
U

VARIABLE / BIND

CONTEXT / PROVENANCE

OPPOSITION
TERNARY TRUTH
  KEY +1/0/-1
  U   +1/0/-1
  contradiction separate

MATCH / COMPOSE
SEARCH
BACKWARD PROVE
ACTIVE PROOF CONTEXT / termination
RESOURCE / BUDGET
```

Domain categories such as `PERSON`, `OBJECT`, `GIVE`, POS tags, case labels,
color names, shape names and named arithmetic successors are not intended to be
kernel primitives.

## 5. Arithmetic and anonymous progress

The clean runtime builds an anonymous discrete domain from generic `ORDER`
fragments. A unique minimum and immediate cover relation provide a progress
structure. Recursive ADD/MUL rule shapes operate over this domain.

K22 showed that named `ZERO/SUCC/PRED` ontology is not necessary for the tested
behavior; the current runtime still assumes a finite discrete ordered chain and
immediate-cover structure.

## 6. Time and state

Temporal U can learn cue orientation and unit scale. Nested time expressions are
composed as DAGs. State queries use temporal order to find the latest applicable
ADD/REMOVE event.

Unknown or rejected intermediate derivations remain unknown rather than being
silently negated.

## 7. Macro-U compilation

Repeated stable proofgraphs can be compiled into `Macro-U` objects. A Macro-U is
not an opaque answer cache: it keeps context, dependencies, provenance,
decomposition and proof shape.

```text
modular proof
 -> stable repeated graph
 -> compile Macro-U
 -> fast reuse
 -> dependency changes
 -> Macro-U state -1 / retired
 -> fall back to micro-U backward proof
```

Retirement never makes the output Key negative.

## 8. Vision boundary

Pixel arrays are handled by classical OpenCV/Pillow preprocessing. The sensor
emits low-level regions, individual vertices/edges and measurements, not semantic
color/shape labels and not object/corner counts.

The symbolic layer then performs curriculum grounding, recursive counting and
composed queries.

K25 adds multiple classical sensor/filter paths. Context observations remain
anonymous (`F1..F6`), and `TRUST-U` rules learn which path is justified for a
query under observed conditions.

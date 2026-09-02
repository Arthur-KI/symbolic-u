# Glossary

**KEY** — symbolic proposition/state/fact that can be queried.

**U** — symbolic derivation/operator/hyperedge consuming Keys and producing a Key.

**KEY +1** — target proposition positively provable.

**KEY 0** — undecided/unknown; contradiction may additionally be flagged.

**KEY -1** — explicit opposite proposition positively provable.

**U +1** — derivation/link confirmed.

**U 0** — derivation/link pending/open.

**U -1** — derivation/link rejected; does not negate its output Key.

**Opposition** — explicit map between a proposition predicate and its negative or
opposite counterpart.

**Contradiction** — positive proof exists for both a proposition and its explicit
opposite.

**Backward proof** — query-guided recursive search from a target to U and their
required input Keys.

**Provenance** — source identity and proof lineage retained with observations and
derivations.

**Context** — story/image/source scope that is part of proposition identity.

**Progress** — anonymous immediate step relation derived over an ordered domain.

**Macro-U** — compiled stable proofgraph retaining dependencies, context,
provenance and decomposition.

**Micro-U** — ordinary small derivations used for modular proof and fallback.

**Visual tokenizer/sensor** — classical OpenCV/Pillow front end converting pixels
to region/vertex/measurement observations without semantic class labels.

**TRUST-U** — K25 derivation expressing that a sensor path is confirmed for a
query under an observed anonymous context condition.

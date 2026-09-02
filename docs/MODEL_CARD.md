# Symbolic-U research model card

## Model type

Non-neural symbolic learning/reasoning research prototype.

## Intended use

- controlled experiments in symbolic induction and reasoning;
- studying explicit unknown/contradiction/provenance semantics;
- comparing modular proof with compiled proof reuse;
- classical-sensor-to-symbolic-world experiments;
- educational/research exploration of alternative AI architectures.

## Out of scope

- production autonomous robots;
- safety-critical decisions;
- unrestricted natural-language assistant use;
- natural-image recognition at modern neural-model quality;
- claims of human-level/general intelligence.

## Inputs

Depending on module: grounded symbolic facts, controlled text curricula, ordered
symbol domains, temporal facts, or synthetic images passed through classical
OpenCV/Pillow sensors.

## Outputs

Symbolic query results (`+1`, `0`, `-1`), contradiction metadata, proof/provenance
structures, learned U, compiled Macro-U, or visual semantic/count query results.

## Learning

No gradient training and no reinforcement learning. Candidate symbolic structures
are evaluated from curricula and consequences; some stable proof patterns are
compiled for reuse.

## Known failure modes

See `LIMITATIONS.md`; particularly candidate-space dependence, uncontrolled
language complexity, natural-image perception and K25's low coverage under hard
unseen disturbance combinations.

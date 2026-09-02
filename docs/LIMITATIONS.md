# Limitations and non-claims

This file is intentionally explicit. A research result should remain easy to
falsify and difficult to overstate.

## Language

- Controlled grounded curricula, not unrestricted German.
- Token boundaries are assumed.
- Some experiments use persistent grounded participant IDs and explicit
  before/after state Keys.
- Unrestricted coreference, pragmatics, metaphor, irony and broad polysemy are
  unsolved.
- Natural-language nested syntax is not yet fully integrated end-to-end with the
  K22 temporal DAG runtime.

## Learning

- Candidate hypothesis families are bounded generic priors.
- The clean runtime reuses some previously learned rule shapes rather than
  jointly relearning every component on each launch.
- Identifiability depends strongly on curriculum design.
- There is no claim of arbitrary program synthesis from no bias.

## Arithmetic and time

- Anonymous progress still assumes a finite discrete ordered chain and useful
  immediate-cover structure.
- Current arithmetic is a research demonstration, not a high-performance
  numerical engine.

## Macro-U

- Current Macro-U are primarily context-grounded compiled proofgraphs, not yet
  general parameterized macro-program synthesis across arbitrary unseen stories.
- Performance results are symbolic operation counts, not production latency.
- Micro-U fallback remains necessary for safe revision.

## Vision

- OpenCV/Pillow perform classical image tokenization; raw pixel perception is not
  learned.
- K24/K25 images are synthetic geometric scenes.
- K25 sensor paths are algorithmic variants sharing low-level machinery, not
  independent physical sensors.
- K25 has high `UNKNOWN` rate under difficult held-out perturbation combinations.

## Robotics

- Robotics is an application direction, not a deployed system.
- No functional-safety, collision-avoidance or real-world reliability claim is
  made.

## Generality

Using one execution abstraction across several domains is evidence of reusable
software structure, not proof of general intelligence.

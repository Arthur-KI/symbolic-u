# Project structure

The repository deliberately separates **current software** from **research
history**.

## Current software

### `symbolic_u/`

The clean implementation:

- `core.py` — propositions, patterns, rules, world, backward prover, opposition;
- `arithmetic.py` — anonymous progress and recursive arithmetic;
- `temporal.py` — temporal lexicon, nested time graph and state reasoning;
- `monolith.py` — decomposable Macro-U compiler;
- `vision_sensor.py` — classical low-level visual tokenizer;
- `vision_reasoner.py` — curriculum grounding and symbolic visual queries;
- `helpers.py` — small construction helpers.

### `CURRENT_MODEL.py`

Thin integrated entry point. It intentionally does not collapse the modules into
one giant source file.

### `tests/`

Regression tests for the supported current behavior.

### `demos/` and `tools/`

Small demonstrations and image/sensor inspection utilities.

## Data

### `data/vision/`

Synthetic K24/K25 images and the main curriculum file.

## Research

### `research/language/`

Curated K1-K17 language-grounding and ablation snapshots.

### `research/math/`

Curated recursive arithmetic/program-learning snapshots.

### `research/kernel_time/`

K18 minimal OS through K23 Macro-U compilation plus production-monolith artifacts.

### `research/vision/`

K24/K24c and K25 scripts, checks and reports.

### `research/archive_full/flat/`

Preservation copy of every matching early `symbolic*` script/report/check recovered
from the working research directory when this package was created. This includes
many experiments that predate the named K-series.

## Documentation and project files

- `docs/` — consolidated architecture, training, experiments, prior art and limits;
- `.github/` — CI and issue/PR templates;
- `LICENSE`, `NOTICE`, `CITATION.cff` — public repository metadata;
- `scripts/verify_repository.py` — invariant and packaging sanity check.

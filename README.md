# Symbolic-U

[![CI](https://github.com/Arthur-KI/symbolic-u/actions/workflows/ci.yml/badge.svg)](https://github.com/Arthur-KI/symbolic-u/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Neural networks](https://img.shields.io/badge/neural_networks-none-success.svg)](docs/LIMITATIONS.md)

**Symbolic-U is an experimental, non-neural symbolic learning and reasoning system.**
It uses a small fixed symbolic substrate (`KEY`, `U`, identity, order, binding,
context/provenance, ternary state, backward proof, search and resource control)
and learns or composes domain structure above it.

The repository contains both the **clean current runtime** and the **full recovered research
archive available from the working project history**.

> Current status: controlled research prototype. It is not an unrestricted
> language model, a natural-image recognition system, or production-certified
> robotics software.

Deutsch: [README_DE.md](README_DE.md)

## Core contract

Two different ternary states are deliberately kept separate:

```text
KEY state
 +1  proposition positively provable
  0  unknown / undecided
 -1  explicit opposite positively provable

U state
 +1  derivation/link confirmed
  0  derivation/link pending/open
 -1  derivation/link rejected
```

The invariant is fundamental:

```text
U = -1  !=  KEY = -1
```

A rejected derivation never proves the opposite proposition. Contradiction is
represented as positive support for both a proposition and its explicit
opposite; the query result remains `0` with a separate contradiction flag.

Reasoning is query-guided:

```text
QUERY target KEY
        |
        v
which U could produce it?
        |
        v
which input Keys does that U require?
        |
        v
recurse through +1 / 0 / -1
```

A useful project shorthand is:

> **Forward generates hypotheses. Backward gives them meaning.**

## What is learned vs fixed

The fixed substrate is intentionally small. It contains symbol identity,
ordering, `KEY/U`, variables/binding, context/provenance, explicit opposition,
ternary state, matching/composition/search, backward proof, resource/budget and
termination control.

The research line progressively moved structure above that substrate. Controlled
experiments have learned or induced, among other things:

- participant/event bindings and semantic families from grounded language curricula;
- morphology and grouping hypotheses without fixed POS/case/lemma labels;
- recursive symbolic arithmetic structures over anonymous ordered progress;
- temporal cue direction, unit scale, intervals and state persistence behavior;
- visual color and shape grounding above classical region/vertex observations;
- a generic recursive count program over membership facts;
- query/context-conditioned trust in multiple classical sensor/filter paths;
- reusable proof structures and decomposable compiled `Macro-U`.

The system is **not reinforcement learning** and does not use gradient training.
Its learning is based on candidate symbolic structures, curriculum consequences,
support/conflict tests, identifiability, blind evaluation and ablation.

See [Training method](docs/TRAINING_METHOD.md).

## Architecture

```text
raw text / numeric facts / image pixels / sensor measurements
                         |
        +----------------+----------------+
        |                                 |
  symbolic tokenization          classical visual sensors
        |                                 |
        +--------------- observations ----+
                         |
                    KEY / U world
                         |
                 context + provenance
                         |
                 backward proof search
                         |
             arithmetic / time / state
                         |
            learned control / sensor U
                         |
             stable repeated proof?
                         |
                compile Macro-U
                         |
       dependency change -> retire -> micro-U fallback
```

The visual boundary is explicit: OpenCV/Pillow are used as classical visual
front ends. Raw pixel perception itself is **not** claimed to be learned.

## Current experimental line

| Milestone | Main result | Status |
|---|---|---|
| K17 | ternary backward reasoning + reuse | 21/21 checks |
| K18 | minimal symbolic OS | 19/19 checks |
| K18b | drop-one kernel audit | 11/11 checks |
| K19 | temporal reasoning on minimal OS | 23/23 checks |
| K20 | learned temporal arithmetic mappings | 29/29 checks |
| K21 | recursive ADD/MUL U used by time | 25/25 checks |
| K22 | anonymous progress + nested temporal composition | 32/32 checks |
| K23 | context-safe decomposable symbolic Macro-U compiler | 23/23 checks |
| K24c | classical vision tokens + real symbolic counting | 15/15 checks |
| K25 | learned query/context sensor-path selection | partial pass; next bottleneck identified |

K25 is intentionally not presented as solved. On its held-out disturbance
combinations, the conservative learned selector reduced false commitments and
sensor cost but returned many `UNKNOWN` answers. See
[Vision and sensor selection](docs/VISION_AND_SENSORS.md).

## Repository layout

```text
symbolic-u/
├── symbolic_u/              # clean current runtime
├── tests/                   # integrated regression tests
├── demos/                   # small runnable demonstrations
├── tools/                   # vision inspection / CLI tools
├── data/vision/             # synthetic K24/K25 image curriculum
├── research/
│   ├── language/            # curated K1-K17 language milestones
│   ├── math/                # curated arithmetic/program-learning milestones
│   ├── kernel_time/         # K18-K23 kernel/time/compiler milestones
│   ├── vision/              # K24/K25 experiments and reports
│   └── archive_full/        # untouched recovered full experiment archive
├── docs/                    # consolidated research documentation
├── scripts/                 # repository verification helpers
├── CURRENT_MODEL.py         # thin integrated entry point
└── run_all.py               # clean regression suite
```

See [Project structure](docs/PROJECT_STRUCTURE.md) and
[Full research index](research/archive_full/INDEX.md).

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate              # Linux/macOS
# .venv\Scripts\Activate.ps1           # Windows PowerShell

python -m pip install --upgrade pip
pip install -r requirements.txt
python run_all.py
```

Expected clean-runtime result:

```text
Ran 15 tests
OK
```

Run demos:

```bash
python -m demos.demo_core
python -m demos.demo_temporal
python -m demos.demo_vision
python -m demos.demo_monolith
```

Inspect the deliberately "dumb" visual sensor:

```bash
python tools/inspect_sensor.py data/vision/test_01.png
```

Run K25:

```bash
python research/vision/k25_multi_sensor_selection.py
```

For historical snapshots that contain old absolute `/mnt/data/` paths, use:

```bash
python research/run_snapshot.py kernel_time/symbolic_v90_k18_minimal_os.py
python research/run_snapshot.py kernel_time/symbolic_v94_k22_nested_temporal_progress.py
python research/run_snapshot.py kernel_time/symbolic_v100_k23_monolith_compiler.py
```

## Research philosophy

The project deliberately asks a narrow question:

> How much domain structure can be moved out of a symbolic kernel and learned
> from identifiable curricula while preserving explicit proof, uncertainty,
> contradiction, provenance and revision?

A failed experiment is kept when it exposes an identifiability or scaling
boundary. `UNKNOWN` is an intended result when evidence is insufficient.

## Related work

The project is **not** based on a claim that symbolic learning, selective
perception, truth maintenance, rule compilation or sensor-to-symbol pipelines
are new individually. Relevant historical lines include STRIPS/Shakey,
Ulysses selective perception, Soar chunking, TMS/ATMS, Belnap-Dunn style
four-valued evidence, ILP/meta-interpretive learning, perceptual anchoring,
DyKnow, RoboSherlock/KnowRob, metareasoning and the Apperception Engine.

The current research question is whether this particular set of behaviors can
be usefully unified under the same small `KEY/U` execution model.

See [Prior art and related work](docs/PRIOR_ART.md) and the
[claims/evidence map](docs/CLAIMS_AND_EVIDENCE.md).

## Reproducibility

The repository ships the current tests, reports, image data, curated milestone
scripts and a recovered flat archive of earlier experiment artifacts. See
[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) and run:

```bash
python scripts/verify_repository.py
```

## License

Unless a file says otherwise, original project code and documentation are
released under the **Apache License 2.0**. See [LICENSE](LICENSE),
[NOTICE](NOTICE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The included Grimm source texts are public-domain literary material and are not
claimed as original project authorship.

## Citation

GitHub can render the included [CITATION.cff](CITATION.cff). Until a formal
paper/DOI exists, cite the repository as **Arthur-KI, Symbolic-U** with the
release/tag or commit used.

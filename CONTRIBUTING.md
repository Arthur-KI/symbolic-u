# Contributing

Contributions are welcome. This project values reproducible negative results as
much as successful ones.

## Non-negotiable semantic invariants

Do not merge changes that collapse these concepts:

```text
U = -1  !=  KEY = -1
UNKNOWN != FALSE
contradiction != ordinary UNKNOWN
QUERY != evidence
```

A rejected derivation may block one proof path; it must not automatically prove
the explicit opposite proposition.

## Experiment protocol

New research experiments should normally:

1. state the hypothesis before running;
2. identify what is fixed vs learned;
3. include at least one meaningful baseline;
4. separate training/curriculum data from blind evaluation;
5. test `+1 / 0 / -1` semantics and provenance where relevant;
6. preserve failures and limitations in the report;
7. avoid moving behavior into the clean runtime until it survives regressions;
8. add or update tests if the clean runtime changes.

Number substantial experiments (`K26`, `K27`, ...), keep raw results, and link
them from `docs/EXPERIMENTS.md`.

## Pull requests

Run:

```bash
python run_all.py
python scripts/verify_repository.py
```

If an experiment changes a historical snapshot, explain why. Prefer adding a new
experiment over rewriting old evidence.

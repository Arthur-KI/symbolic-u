# Release verification

Public-package verification performed on **2026-09-02**.

## Clean runtime

- Integrated regression suite: **15/15 PASS**
- Demo modules: core, temporal, vision, monolith — **PASS**
- Python source syntax audit — **0 syntax errors**
- Editable package build in offline mode (`--no-build-isolation --no-deps`) — **PASS**

## Historical smoke tests

- K18 minimal symbolic OS — **19/19 PASS**
- K22 nested temporal/progress — **32/32 PASS**
- K23 monolith compiler — **23/23 PASS**

## Vision

- K24c clean-runtime reproduction harness — **PASS** on all four blind scenes and composed `test_03` queries
- K25 multi-sensor selection fresh run — **12/12 invariant checks PASS**
- K25 status remains **PARTIAL_PASS_NEXT_BOTTLENECK_FOUND**

## Packaging invariants

- no neural runtime dependency declared;
- 16 vision curriculum images and 4 blind test images present;
- recovered flat historical archive contains **372 files**;
- no `__pycache__` directories are included in the release archive;
- `U=-1 != KEY=-1` is covered by clean tests and historical checks.

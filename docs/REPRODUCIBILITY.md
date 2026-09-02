# Reproducibility

## Supported clean runtime

Install dependencies and run:

```bash
pip install -r requirements.txt
python run_all.py
```

The packaged clean regression suite currently contains 15 tests covering core
ternary semantics, contradiction, cycles, backward distractor avoidance,
arithmetic, nested time/state, Macro-U behavior and vision.

The repository verification script adds packaging/invariant checks:

```bash
python scripts/verify_repository.py
```

## Tested build

The packaging build was verified with:

```text
Python 3.13.5
NumPy 2.3.5
Pillow 12.3.0
OpenCV 4.13.0
```

`requirements-tested.txt` records exact package versions available in that build.
CI intentionally tests a broader Python 3.10-3.13 range using minimum-compatible
requirements rather than pinning every platform to one wheel set.

## Historical scripts

Many old snapshots were written in a temporary environment and contain absolute
`/mnt/data/` paths. The curated runner copies a snapshot to a temporary workspace
and rewrites those paths only in the execution copy:

```bash
python research/run_snapshot.py kernel_time/symbolic_v90_k18_minimal_os.py
```

The archived source is not edited.

## Full archive

`research/archive_full/flat/` is a preservation layer containing the recovered
historical `symbolic*` Python scripts, CSV results and JSON reports found in the
research workspace when this public package was prepared.

It is intentionally flat so filenames and historical cross-references remain
recognizable. It is not the supported API.

## Vision data

`data/vision/` contains the 16 curriculum images, 4 blind scene images and the
curriculum CSV used by the current visual reasoner. K25 creates disturbance
variants deterministically in memory rather than requiring a manually prepared
image set.

## Reproducibility caveat

Some earlier experiments are research snapshots rather than hermetic packages.
They may depend on sibling historical files or assumptions from their period.
The clean runtime and selected K18/K22/K23 snapshots are the primary reproducible
entry points; failures in an archival exploratory script should not be silently
rewritten into passing behavior.

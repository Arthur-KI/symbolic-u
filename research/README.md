# Research archive

This directory has two roles:

1. **Curated milestones** in `language/`, `math/`, `kernel_time/` and `vision/`.
2. **Preservation archive** in `archive_full/flat/`, containing the recovered
   earlier scripts/results with original filenames and contents.

Historical snapshots may contain old absolute `/mnt/data/` paths. Use the portable
runner for curated scripts:

```bash
python research/run_snapshot.py kernel_time/symbolic_v90_k18_minimal_os.py
python research/run_snapshot.py kernel_time/symbolic_v94_k22_nested_temporal_progress.py
python research/run_snapshot.py kernel_time/symbolic_v100_k23_monolith_compiler.py
```

The runner modifies only a temporary execution copy. Do not rewrite archival
sources merely to make them look cleaner.

The supported current implementation is `symbolic_u/`.

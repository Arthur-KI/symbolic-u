from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "README.md", "README_DE.md", "LICENSE", "NOTICE", "CITATION.cff",
    "symbolic_u/core.py", "symbolic_u/arithmetic.py", "symbolic_u/temporal.py",
    "symbolic_u/monolith.py", "symbolic_u/vision_sensor.py",
    "symbolic_u/vision_reasoner.py", "run_all.py",
    "research/vision/K25_MULTI_SENSOR/K25_report.json",
    "research/archive_full/INDEX.md",
    "research/vision/k24c_reproduce.py",
]

NEURAL_MARKERS = {
    "torch", "tensorflow", "keras", "jax", "flax", "transformers",
    "onnxruntime", "pytorch", "mxnet"
}


def check(condition: bool, name: str, failures: list[str]) -> None:
    print(("PASS" if condition else "FAIL"), name)
    if not condition:
        failures.append(name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()
    failures: list[str] = []

    for rel in REQUIRED:
        check((ROOT / rel).exists(), f"required:{rel}", failures)

    tracked = subprocess.run(
    ["git", "ls-files"],
    cwd=ROOT,
    capture_output=True,
    text=True,
    check=True,
).stdout.splitlines()

tracked_cache = [
    p for p in tracked
    if "__pycache__/" in p.replace("\\", "/")
    or p.endswith((".pyc", ".pyo"))
]

check(
    not tracked_cache,
    "no Python cache artifacts committed",
    failures,
)

    req = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    check(not any(x in req for x in NEURAL_MARKERS), "no neural runtime dependency", failures)

    report = json.loads((ROOT / "research/vision/K25_MULTI_SENSOR/K25_report.json").read_text(encoding="utf-8"))
    check(report.get("status") == "PARTIAL_PASS_NEXT_BOTTLENECK_FOUND", "K25 status preserved", failures)
    checks = report.get("checks", {})
    check(bool(checks) and all(checks.values()), "K25 recorded invariant checks", failures)
    check(report.get("rejected_u_audit", {}).get("answer_key_state") == 0,
          "K25 rejected U does not create KEY -1", failures)

    vision = ROOT / "data/vision"
    check(len(list(vision.glob("train_*.png"))) == 16, "16 vision train images", failures)
    check(len(list(vision.glob("test_*.png"))) == 4, "4 vision blind images", failures)

    archive_count = len(list((ROOT / "research/archive_full/flat").glob("*")))
    check(archive_count >= 350, f"full archive preserved ({archive_count} files)", failures)

    if not args.skip_tests:
        cp = subprocess.run([sys.executable, "run_all.py"], cwd=ROOT)
        check(cp.returncode == 0, "clean runtime test suite", failures)

    print("\nRepository verification:", "PASS" if not failures else "FAIL")
    if failures:
        for f in failures:
            print(" -", f)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Reproduce the supported K24c behavior using the clean current runtime.

This is a reconstruction harness created for the public repository. It is not
claimed to be the original historical K24c driver (which was not present in the
recovered workspace).
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CURRENT_MODEL import train_default_vision
from symbolic_u.vision_sensor import observe_image

EXPECTED = {
    1: [("green", "quadrilateral"), ("red", "triangle")],
    2: [("blue", "quadrilateral")] * 3,
    3: [
        ("blue", "triangle"),
        ("red", "quadrilateral"),
        ("green", "quadrilateral"),
        ("blue", "quadrilateral"),
    ],
    4: [("red", "quadrilateral"), ("red", "triangle")],
}


def main() -> int:
    learner = train_default_vision(ROOT / "data" / "vision")
    ok = True
    for idx, expected in EXPECTED.items():
        obs = observe_image(ROOT / "data" / "vision" / f"test_{idx:02d}.png", f"TEST_{idx:02d}")
        got = []
        for rid in learner.image_members(obs):
            got.append((learner.query_color(obs, rid).value, learner.query_shape(obs, rid).value))
        count = learner.query_count(obs).value
        passed = sorted(got) == sorted(expected) and count == len(expected)
        print("PASS" if passed else "FAIL", f"test_{idx:02d}", "count=", count, "objects=", got)
        ok &= passed

    # Composed query from the original K24c discussion.
    obs = observe_image(ROOT / "data" / "vision" / "test_03.png", "TEST_03")
    composed = {
        "red_quadrilateral": learner.query_filtered_count(obs, "red", "quadrilateral").value,
        "blue_quadrilateral": learner.query_filtered_count(obs, "blue", "quadrilateral").value,
        "blue_triangle": learner.query_filtered_count(obs, "blue", "triangle").value,
        "green_quadrilateral": learner.query_filtered_count(obs, "green", "quadrilateral").value,
        "red_triangle": learner.query_filtered_count(obs, "red", "triangle").value,
    }
    expected_composed = {
        "red_quadrilateral": 1,
        "blue_quadrilateral": 1,
        "blue_triangle": 1,
        "green_quadrilateral": 1,
        "red_triangle": 0,
    }
    cpass = composed == expected_composed
    print("PASS" if cpass else "FAIL", "composed test_03", composed)
    ok &= cpass
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

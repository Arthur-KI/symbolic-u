# Claims and evidence map

This file maps current project statements to executable or machine-readable
artifacts. It is meant to make overclaiming harder.

| Claim | Primary evidence |
|---|---|
| Rejected U does not imply negative Key | `tests/test_core.py`, K17/K18 checks |
| Explicit positive+opposite support becomes contradiction | `tests/test_core.py`, K17b/K18 |
| Backward proof ignores disconnected distractors | `tests/test_core.py`, K22 checks |
| Minimal symbolic OS survives controlled language ablations | K18 report/checks and K18b drop-one audit |
| Recursive symbolic arithmetic is usable by temporal reasoning | K21 report/checks and clean arithmetic/temporal tests |
| Anonymous progress supports nested temporal composition | K22 report/checks |
| Stable proofgraphs can compile to context-safe Macro-U | K23 script/report/checks and `tests/test_monolith.py` |
| Retired Macro-U do not create negative Keys | K23 and `tests/test_monolith.py` |
| K24c sensor does not emit semantic count/shape/color labels | `tests/test_vision.py`, `symbolic_u/vision_sensor.py` |
| Generic symbolic count is used for object and vertex membership | K24c report/checks, `symbolic_u/vision_reasoner.py` |
| K24c blind synthetic scene outputs match the curriculum semantics | `tests/test_vision.py` |
| K25 learns query/context-conditioned sensor-path trust rules | `research/vision/k25_multi_sensor_selection.py` and K25 report |
| K25 reduces false commitments and sensor cost vs fixed sensor baseline | K25 report; partial result only |

## Statements not established by current evidence

The repository does **not** establish unrestricted language understanding,
robust natural-image recognition, autonomous general-purpose program synthesis,
production robotics safety, or general intelligence.

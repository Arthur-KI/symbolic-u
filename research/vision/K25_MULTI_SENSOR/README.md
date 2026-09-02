# K25 — Multi-sensor selection

This experiment is fully automatic and non-neural.

It takes the existing K24 geometric PNGs, creates deterministic disturbed variants in memory, runs six classical OpenCV sensor/filter paths, learns sensor-specific color/shape calibration on the clean curriculum, and then learns context/query-conditioned `TRUST-U` rules.

The learner never receives labels such as `dark`, `blur`, `noise`, `lowcontrast`, `rotate`, or `occlude`. It sees only six anonymous image measurements `F1..F6`, query identity, sensor provenance, and training consequences.

## Train vs blind stress

Training uses clean/single disturbances plus selected disturbance pairs. The blind stress suite uses only disturbance combinations absent from training and only the four held-out `test_*.png` scene compositions.

## Current result

`PARTIAL_PASS_NEXT_BOTTLENECK_FOUND`

The conservative query-guided gate strongly reduces false commits and sensor cost, but leaves many hard unseen combinations as `KEY 0 / UNKNOWN`. This is considered preferable to forcing an answer, but coverage is too low to call compositional sensor reliability solved.

Run:

```bash
python research/vision/k25_multi_sensor_selection.py
```

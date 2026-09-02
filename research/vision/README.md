# Vision research

## K24 original

`K24_ORIGINAL/` contains recovered early visual-curriculum source and the v1b
report/checks. The first color classifier exposed an exact-RGB assumption.

## K24c

K24c removed semantic count outputs from the visual sensor. Recovered raw result
artifacts live in `K24C_RAW/`, while the current clean implementation is in
`symbolic_u/vision_sensor.py` and `symbolic_u/vision_reasoner.py`.

`k24c_reproduce.py` is a modern reproduction harness, not the missing original
historical driver.

## K25

`k25_multi_sensor_selection.py` automatically creates held-out disturbance
combinations and learns query/context-conditioned `TRUST-U` over six classical
sensor paths. See `K25_MULTI_SENSOR/README.md` and `docs/VISION_AND_SENSORS.md`.

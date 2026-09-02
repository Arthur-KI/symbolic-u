# Vision and multi-sensor reasoning

## K24c: deliberately dumb visual sensor

The current classical visual tokenizer emits facts analogous to:

```text
MEMBER(region, image)
VERTEX_OF(vertex, region)
POSITION(vertex, ...)
EDGE(vertex_a, vertex_b, region)
CLOSED(region)
COLOR_MEASURE(region, rgb)
AREA_MEASURE(region, ...)
CENTROID_MEASURE(region, ...)
BACKGROUND_MEASURE(image, rgb)
```

It deliberately does **not** emit semantic color names, shape names, object count
or corner count.

The symbolic layer uses a learned recursive Count-U:

```text
COUNT(empty) = minimum
MEMBER(x,S) + COUNT(S\{x},n) + PROGRESS(n,n1)
    -> COUNT(S,n1)
```

The same program counts regions and vertices. Curriculum then grounds symbolic
vertex counts to labels such as `triangle` and `quadrilateral`.

A synthetic five-vertex region is therefore counted as five but remains an
unknown shape because no five-vertex shape label was taught.

## Why the first color learner failed

The early exact-RGB hypothesis failed on the red curriculum because nominal red
examples contained more than one first-channel value. The corrected learner
estimated an intra-class measurement tolerance from the curriculum instead of
hard-coding the red values.

This is preserved as a useful example of a failure exposing a hidden assumption.

## Blind K24c scene composition

The held-out scenes combine colors/shapes in mixtures not used as simple training
scene templates. The system composes learned color, shape and count U to answer
filtered queries such as `count(red AND quadrilateral)`.

## K25: multiple classical sensor paths

K25 adds six non-neural paths:

```text
RAW
DENOISE
GAIN
CONSERVATIVE
OTSU_RGB
OTSU_CHROMA
```

Synthetic disturbances are created automatically. Training includes clean,
single disturbances and selected pairs; blind evaluation uses different pairs
and triples.

The learner never receives labels such as `dark`, `blur`, `noise`, `rotate` or
`occlude`. It receives anonymous measurements `F1..F6`, query identity, sensor
provenance, training consequence and a symbolic cost.

Confirmed `TRUST-U` are zero-conflict rules in the current training set.
Backward query execution activates only rules compatible with the query and
observed context, preferring cheaper confirmed paths.

## K25 result

The learned gate substantially reduces false commits and sensor cost compared to
a globally fixed sensor. Its weakness is coverage on unseen combined
perturbations. Many hard cases become `KEY 0` rather than forced answers.

That is useful safety behavior but not yet robust perception.

## Important non-claims

- The scenes are clean synthetic geometry, not natural photographs.
- The sensor paths are multiple classical processing routes, not independent
  physical sensors.
- OpenCV preprocessing is engineered, not learned.
- K25 does not solve general active perception or robotics sensor fusion.

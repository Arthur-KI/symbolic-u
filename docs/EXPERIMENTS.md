# Experiment history

This is the consolidated high-level map. Exact scripts and machine-readable
reports live under `research/`; the recovered earlier archive is under
`research/archive_full/flat/`.

## Pre-K series / exploratory archive

The full archive contains earlier versions from the initial mini symbolic LM,
search-scaling and bottleneck experiments through arithmetic, adaptive U
libraries, structured text, event-family learning and raw-language curricula.
These experiments are kept for archaeology rather than presented as the clean
current API.

Broad phases visible in the archive include:

- early Key/Query and ternary language prototypes;
- search/bottleneck scaling and raw-scene experiments;
- recursive arithmetic and cross-domain generic U;
- adaptive U libraries, dependency discovery and abstraction invention;
- structured-text and Grimm-story curricula;
- semantic-family invention and leakage/adversarial audits;
- the K1-K17 explicit ablation line.

See `research/archive_full/INDEX.md`.

## K1-K7: remove semantic conveniences

- **K1** remove semantic surface cues.
- **K2** remove fixed action heads.
- **K3** remove fixed relation names.
- **K4** remove concrete semantic types.
- **K5** remove named operations.
- **K6** remove universal persistence.
- **K7** remove semantic context labels.

The purpose was not merely to improve a benchmark but to expose which structure
was genuinely necessary for identifiability.

## K8-K10: binding and event meaning

- **K8** binder abstraction.
- **K9** consequence-supervised participant binding.
- **K10** autonomous event meaning from recurring grounded state-change motifs.

## K11-K16: rawer language and learned search

- **K11/K11b** remove lemma normalization and case labels.
- **K12** productive morphology.
- **K13** remove coarse POS labels.
- **K14/K14b** learned token-to-mention attachment and raw-token entity anchors.
- **K15/K15b** event-to-argument U and learned clause/group search.
- **K16** curriculum-learned search composition under generic resource cost.

## K17: ternary backward reasoning

K17 formalized the key architecture invariant and backward relevance:

```text
U=-1 != KEY=-1
```

The main K17 suite recorded 21/21 checks. K17b added explicit opposition.

## K18/K18b: minimal symbolic OS

K18 reduced the working kernel to generic symbolic mechanisms and tested the
result (19/19 checks). K18b performed a drop-one audit (11/11 checks).

## K19-K22: time and arithmetic

- **K19** temporal order and state reasoning: 23/23.
- **K20** learned temporal directions/unit scales: 29/29.
- **K21** recursive learned ADD/MUL U used by temporal calculations: 25/25.
- **K22** anonymous ordered progress, nested temporal DAGs, contradiction and
  distractor tests: 32/32.

## K23: symbolic monolith compiler

Stable repeated proofgraphs are compiled into context-scoped, provenance-
preserving, decomposable Macro-U. Dependency changes retire affected Macro-U
without making their output propositions false. Main experiment: 23/23 checks;
a separate production-safety check file is also archived.

## K24/K24c: vision bridge

K24 established:

```text
pixels -> classical visual tokenizer -> symbolic observations -> Key/U reasoning
```

K24c removed semantic counts from the sensor. The same generic recursive Count-U
counts both object-membership and vertex-membership, after which shape words are
grounded from curriculum. K24c: 15/15 checks.

## K25: learned multi-sensor selection

K25 automatically disturbs the synthetic scenes and runs six classical sensor
paths. The learner receives anonymous global context features, query identity,
sensor provenance and training outcomes.

Held-out evaluation uses disturbance combinations absent from training and the
four held-out test scene compositions.

Recorded K25 result:

| Strategy | Correct | False commits | Unknown | Mean sensor cost |
|---|---:|---:|---:|---:|
| learned query/context gate | 150 | 11 | 271 | 2.09 |
| best fixed sensor | 361 | 71 | 0 | 4.25 |
| strict all-sensor consensus | 262 | 64 | 106 | 17.00 |

The learned gate's conditional accuracy when it committed was about 93.17%, but
coverage was only about 37.27%. This is recorded as a **partial pass**, not as a
solution to robust sensor fusion.

## Current next bottleneck

The most immediate research question is compositional sensor reliability:
learning how evidence quality changes under combinations of conditions without
requiring hand-written disturbance semantics or sacrificing most coverage.

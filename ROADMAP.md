# Research roadmap

This roadmap is exploratory, not a promise of delivery. New experiments should
remain compatible with the hard `KEY/U` semantic contract.

## K26 candidates

1. **Compositional sensor reliability** — improve K25 coverage on combinations of
   disturbances without hard-coding disturbance names or global sensor ranking.
2. **Parameterized Macro-U** — generalize repeated grounded proofgraph shapes into
   variable-bearing symbolic macros reusable across unseen contexts while keeping
   provenance and decomposition.
3. **Visual spatial relations** — learn/derive LEFT/RIGHT/ABOVE/BELOW from raw
   centroid observations and answer filtered relational count queries.
4. **Robotics-style sensor fusion simulator** — camera, range, force and encoder
   observations feeding a shared Key/U world model with action/state transitions.
5. **Query-complete semantic compression** — compare raw+U vs U-only state across
   broad semantic queries and explicitly record which raw-text queries become
   impossible after compression.
6. **Nested raw-language integration** — connect the K10-K16 language induction
   line end-to-end with K22 temporal/state composition.

## Long-term questions

- How far can candidate families themselves be learned rather than provided?
- Which parts of `ORDER`, identity and binding are genuinely kernel-near?
- Can sensor choice become active information acquisition under resource budgets?
- Can learned Macro-U remain reusable while still safely invalidating dependencies?
- Where does candidate explosion become the dominant scaling limit?

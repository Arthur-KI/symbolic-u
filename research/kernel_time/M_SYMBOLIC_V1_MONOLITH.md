# M_SYMBOLIC_V1 — compiled symbolic monolith
## Contract
- Learn with small U; execute stable proof graphs as Macro-U.
- KEY and Macro-U keep separate +1/0/-1 semantics.
- Macro-U -1 retires one compiled proof only; it never proves the opposite Key.
- Every Macro-U retains dependencies and provenance and can be decomposed.
- Dependency change retires only affected Macro-U; runtime falls back to micro-U and may recompile.

## Active Macro-U
- M0001 [MAIN] POS TIME|["B"] support=3 hits=42
- M0002 [MAIN] POS TIME|["C"] support=3 hits=42
- M0015 [CON] POS TIME|["B"] support=3 hits=1
- M0016 [CON] NEG TIME|["B"] support=3 hits=1
- M0017 [MAIN] POS STATE|["wolf","house","X"] support=3 hits=2
- M0018 [MAIN] POS TIME|["D"] support=3 hits=1
- M0019 [MAIN] POS TIME|["E"] support=3 hits=1
- M0020 [MAIN] POS TIME|["GQ"] support=3 hits=1
- M0021 [MAIN] POS DURING|["B","INNER"] support=3 hits=1
- M0022 [MAIN] POS DURING|["X","INNER"] support=3 hits=1
- M0023 [MAIN] POS CONTAINS|["OUTER","INNER"] support=3 hits=1
- M0024 [MAIN] NEG DURATION|["INNER",45,"minuten"] support=3 hits=1
- M0025 [MAIN] POS STATE|["wolf","house","B"] support=3 hits=1
- M0026 [MAIN] POS STATE|["anna","house","B"] support=3 hits=1
- M0027 [MAIN] POS MEET|["wolf","anna","house","B"] support=3 hits=1
- M0028 [MAIN] POS MEET|["wolf","anna","house","X"] support=3 hits=1

## Mined recurring proof shapes
- 3× TIME -> ANCHOR -> CUE -> U -> UNIT -> ANCHOR -> MUL -> ADD
- 3× TIME -> ANCHOR -> CUE -> CUE -> U -> U -> UNIT -> UNIT -> ANCHOR -> MUL -> ADD -> MUL -> ADD_BACK
- 3× TIME -> ANCHOR -> CUE -> CUE -> U -> U -> U -> UNIT -> UNIT -> ANCHOR -> MUL -> ADD -> MUL -> ADD_BACK -> MUL -> ADD
- 3× TIME -> ANCHOR -> CUE -> CUE -> U -> U -> U -> U -> UNIT -> UNIT -> UNIT -> ANCHOR -> MUL -> ADD -> MUL -> ADD_BACK -> MUL -> ADD -> MUL -> ADD
- 3× TIME -> ANCHOR -> CUE -> CUE -> U -> U -> U -> U -> U -> U -> UNIT -> UNIT -> UNIT -> ANCHOR -> MUL -> ADD -> MUL -> ADD_BACK -> MUL -> ADD -> MUL -> ADD -> MUL -> ADD_BACK -> MUL -> ADD
- 3× DURING -> ANCHOR -> CUE -> CUE -> INTERVAL -> U -> U -> U -> UNIT -> UNIT -> INTERVAL_TEST
- 3× DURING -> ANCHOR -> CUE -> CUE -> INTERVAL -> U -> U -> U -> U -> UNIT -> UNIT -> INTERVAL_TEST
- 3× CONTAINS -> ANCHOR -> CUE -> CUE -> INTERVAL -> INTERVAL -> U -> U -> U -> U -> UNIT -> UNIT -> UNIT -> CONTAIN_TEST
- 3× DURATION -> ANCHOR -> COUNT -> CUE -> CUE -> INTERVAL -> U -> U -> U -> UNIT -> UNIT -> DURATION_TEST
- 3× STATE -> ANCHOR -> CUE -> CUE -> STATE -> STATE -> U -> U -> U -> UNIT -> UNIT -> ADD

## Performance
- repeated micro symbolic cost: 6605720
- repeated monolith cost: 560
- ratio: 11795.93x

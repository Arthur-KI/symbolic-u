from symbolic_u.helpers import opaque_chain
from symbolic_u.arithmetic import ProgressDomain, RecursiveArithmetic
from symbolic_u.temporal import *

nodes, fragments = opaque_chain(2300, "T")
d = ProgressDomain.from_order_fragments(fragments)
a = RecursiveArithmetic(d)

lex = TemporalLexicon.learned(
    cue_direction={"später": 1, "nach": 1, "vor": -1},
    unit_scale={"minute": nodes[1], "minuten": nodes[1],
                "stunde": nodes[60], "stunden": nodes[60],
                "tag": nodes[1440]}
)

g = TimeGraph("DEMO", a, lex)
g.add_anchor("A", nodes[600], "10:00")
g.add_relative("UB", "A", "B", nodes[2], "stunden", "später")
g.add_relative("UC", "B", "C", nodes[30], "minuten", "vor")
g.add_relative("UD", "C", "D", nodes[45], "minuten", "später")

for ev in ("A","B","C","D"):
    r = g.time(ev)
    print(ev, r.state, None if r.value is None else nodes.index(r.value), r.trace)

sr = StateReasoner(g, [
    StateEvent("C", "ADD", "wolf", "house"),
    StateEvent("D", "REMOVE", "wolf", "house"),
])
print("wolf@B:", sr.state_at("wolf", "house", "B"))

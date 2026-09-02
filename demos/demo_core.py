from symbolic_u.core import *

w = World()
w.set_opposition("P", "NOT_P")
a = Proposition("A", (), "S")
p = Proposition("P", (), "S")
not_p = Proposition("NOT_P", (), "S")

w.add_fact(a, "source")
w.add_rule(Rule("U_GOOD", (Pattern("A", (), "S"),), Pattern("P", (), "S"), Truth.TRUE))
w.add_rule(Rule("U_REJECTED", (), Pattern("P", (), "S"), Truth.FALSE))

prover = BackwardProver(w)
print("P:", prover.query(p))

w.add_fact(not_p, "explicit opposite")
print("P after opposite:", prover.query(p))

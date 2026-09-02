import unittest
from symbolic_u.core import *

class CoreTests(unittest.TestCase):
    def test_ternary_opposition_rejected_u(self):
        w = World()
        w.set_opposition("P","NOT_P")
        p = Proposition("P",(), "S")
        np = Proposition("NOT_P",(), "S")

        w.add_rule(Rule("BAD",(),Pattern("P",(),"S"),Truth.FALSE))
        r = BackwardProver(w).query(p)
        self.assertEqual(r.state, Truth.UNKNOWN)

        w.add_fact(np,"explicit")
        r = BackwardProver(w).query(p)
        self.assertEqual(r.state, Truth.FALSE)

    def test_contradiction(self):
        w = World(); w.set_opposition("P","NOT_P")
        p = Proposition("P",(), "S")
        w.add_fact(p,"a")
        w.add_fact(Proposition("NOT_P",(),"S"),"b")
        r = BackwardProver(w).query(p)
        self.assertEqual(r.state, Truth.UNKNOWN)
        self.assertTrue(r.contradiction)

    def test_cycle_terminates_unknown(self):
        w = World()
        w.add_rule(Rule("U1",(Pattern("B",(),"S"),),Pattern("A",(),"S")))
        w.add_rule(Rule("U2",(Pattern("A",(),"S"),),Pattern("B",(),"S")))
        r = BackwardProver(w,budget=100).query(Proposition("A",(),"S"))
        self.assertEqual(r.state, Truth.UNKNOWN)

    def test_backward_ignores_distractors(self):
        w = World()
        w.add_fact(Proposition("BASE",(),"S"),"src")
        w.add_rule(Rule("GOOD",(Pattern("BASE",(),"S"),),Pattern("Q",(),"S")))
        for i in range(1000):
            w.add_rule(Rule(f"Z{i}",(),Pattern(f"ZP{i}",(),"S")))
        r = BackwardProver(w).query(Proposition("Q",(),"S"))
        self.assertEqual(r.state, Truth.TRUE)
        self.assertEqual(r.touched_rules, ("GOOD",))

if __name__ == "__main__":
    unittest.main()

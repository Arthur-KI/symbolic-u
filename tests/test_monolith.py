import unittest
from symbolic_u.monolith import *

class MonolithTests(unittest.TestCase):
    def test_compile_context_and_retire(self):
        c = MonolithCompiler(threshold=3)
        q = qid("P",("x",))
        a = Answer(1,False,"v",("U:a",))
        for _ in range(3):
            c.observe("A",q,a,{"U:a"})
        self.assertEqual(c.query("A",q).state,1)
        self.assertEqual(c.query("B",q).state,0)

        retired = c.retire_dependency("U:a")
        self.assertTrue(retired)
        # Macro-U -1 does not create KEY -1.
        self.assertEqual(c.query("A",q).state,0)

    def test_negative_is_positive_opposite_macro(self):
        c = MonolithCompiler(threshold=3)
        q = qid("P",("x",))
        a = Answer(-1,False,None,("OPPOSITE",))
        for _ in range(3):
            c.observe("A",q,a,{"U:opp"})
        self.assertEqual(c.query("A",q).state,-1)
        macros = c.decompose("A",q)
        self.assertEqual(macros[0].state,1)
        self.assertEqual(macros[0].polarity,"NEG")

    def test_unknown_not_compiled(self):
        c = MonolithCompiler(threshold=2)
        q = qid("P",())
        for _ in range(5):
            c.observe("A",q,Answer(0,False),{"U:x"})
        self.assertEqual(c.query("A",q).state,0)
        self.assertEqual(c.decompose("A",q),[])

if __name__ == "__main__":
    unittest.main()

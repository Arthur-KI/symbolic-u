import unittest
from symbolic_u.helpers import opaque_chain
from symbolic_u.arithmetic import *
from symbolic_u.temporal import *

class ArithmeticTemporalTests(unittest.TestCase):
    def setUp(self):
        self.nodes, frags = opaque_chain(2300, "T")
        self.d = ProgressDomain.from_order_fragments(frags)
        self.a = RecursiveArithmetic(self.d)
        self.lex = TemporalLexicon.learned(
            {"später":1,"vor":-1,"nach":1},
            {"minute":self.nodes[1],"minuten":self.nodes[1],
             "stunde":self.nodes[60],"stunden":self.nodes[60],
             "tag":self.nodes[1440]}
        )

    def test_arithmetic(self):
        self.assertEqual(self.a.add_output(self.nodes[2],self.nodes[3]),self.nodes[5])
        self.assertEqual(self.a.mul_output(self.nodes[4],self.nodes[3]),self.nodes[12])

    def test_nested_time(self):
        g = TimeGraph("S",self.a,self.lex)
        g.add_anchor("A",self.nodes[600],"a")
        g.add_relative("UB","A","B",self.nodes[2],"stunden","später")
        g.add_relative("UC","B","C",self.nodes[30],"minuten","vor")
        g.add_relative("UD","C","D",self.nodes[45],"minuten","später")
        self.assertEqual(g.time("B").value,self.nodes[720])
        self.assertEqual(g.time("C").value,self.nodes[690])
        self.assertEqual(g.time("D").value,self.nodes[735])

    def test_unknown_middle_propagates(self):
        g = TimeGraph("S",self.a,self.lex)
        g.add_anchor("A",self.nodes[600],"a")
        g.add_relative("U1","A","B",self.nodes[1],"stunde","später")
        g.add_relative("U2","B","C",self.nodes[1],"stunden","unknown-cue")
        g.add_relative("U3","C","D",self.nodes[30],"minuten","später")
        self.assertEqual(g.time("C").state,Truth.UNKNOWN)
        self.assertEqual(g.time("D").state,Truth.UNKNOWN)

    def test_state(self):
        g = TimeGraph("S",self.a,self.lex)
        g.add_anchor("ENTER",self.nodes[600],"a")
        g.add_relative("LEAVEU","ENTER","LEAVE",self.nodes[2],"stunden","später")
        g.add_relative("TARGETU","ENTER","TARGET",self.nodes[3],"stunden","später")
        sr = StateReasoner(g,[
            StateEvent("ENTER","ADD","wolf","house"),
            StateEvent("LEAVE","REMOVE","wolf","house")
        ])
        self.assertEqual(sr.state_at("wolf","house","TARGET").state,Truth.FALSE)

if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path
from symbolic_u.helpers import opaque_chain
from symbolic_u.arithmetic import ProgressDomain, RecursiveArithmetic
from symbolic_u.vision_sensor import observe_image
from symbolic_u.vision_reasoner import VisionCurriculumLearner

TARGETS = {
  1:("white","red","triangle",1),2:("white","red","triangle",2),
  3:("white","red","quadrilateral",1),4:("white","red","quadrilateral",2),
  5:("white","blue","triangle",1),6:("white","blue","quadrilateral",1),
  7:("white","green","triangle",1),8:("white","green","quadrilateral",1),
  9:("white","blue","triangle",3),10:("white","green","quadrilateral",3),
  11:("white","red","triangle",4),12:("white","blue","quadrilateral",4),
  13:("lightgray","red","triangle",1),14:("lightgray","blue","quadrilateral",2),
  15:("lightgray","green","triangle",3),16:("lightgray","red","quadrilateral",4),
}

class VisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = Path(__file__).resolve().parents[1]/"data"/"vision"
        nodes, frags = opaque_chain(32,"C")
        cls.nodes = nodes
        cls.model = VisionCurriculumLearner(
            RecursiveArithmetic(ProgressDomain.from_order_fragments(frags)),
            {node:i for i,node in enumerate(nodes)}
        )
        examples=[]
        for i in range(1,17):
            obs=observe_image(cls.data/f"train_{i:02d}.png",f"TRAIN_{i:02d}")
            bg,c,s,n=TARGETS[i]
            examples.append((obs,{"background":bg,"color":c,"shape":s,"count":n}))
        cls.model.fit(examples)

    def test_count_program(self):
        self.assertEqual(self.model.count_program,"MEMBER_RECURSE")
        self.assertGreater(self.model.count_scores["CONST_ONE"][1],0)

    def test_blind_outputs(self):
        expected = {
            1:[("green","quadrilateral"),("red","triangle")],
            2:[("blue","quadrilateral")]*3,
            3:[("blue","triangle"),("red","quadrilateral"),
               ("green","quadrilateral"),("blue","quadrilateral")],
            4:[("red","quadrilateral"),("red","triangle")]
        }
        for i in range(1,5):
            obs=observe_image(self.data/f"test_{i:02d}.png",f"TEST_{i:02d}")
            got=[]
            for rid in self.model.image_members(obs):
                got.append((
                    self.model.query_color(obs,rid).value,
                    self.model.query_shape(obs,rid).value
                ))
            self.assertCountEqual(got,expected[i])

    def test_filtered_count(self):
        obs=observe_image(self.data/"test_03.png","TEST_03")
        self.assertEqual(self.model.query_filtered_count(obs,"red","quadrilateral").value,1)
        self.assertEqual(self.model.query_filtered_count(obs,"blue","triangle").value,1)

    def test_sensor_does_not_emit_counts_or_semantics(self):
        obs=observe_image(self.data/"test_03.png","TEST_03")
        predicates={f[0] for f in obs.facts}
        forbidden={"REGION_COUNT","CORNER_COUNT","COUNT","TRIANGLE",
                   "QUADRILATERAL","RED","BLUE","GREEN"}
        self.assertFalse(predicates & forbidden)

if __name__ == "__main__":
    unittest.main()

from pathlib import Path
from symbolic_u.helpers import opaque_chain
from symbolic_u.arithmetic import ProgressDomain, RecursiveArithmetic
from symbolic_u.vision_sensor import observe_image
from symbolic_u.vision_reasoner import VisionCurriculumLearner

root = Path(__file__).resolve().parents[1]
data = root / "data" / "vision"

targets = {
  1:("white","red","triangle",1),2:("white","red","triangle",2),
  3:("white","red","quadrilateral",1),4:("white","red","quadrilateral",2),
  5:("white","blue","triangle",1),6:("white","blue","quadrilateral",1),
  7:("white","green","triangle",1),8:("white","green","quadrilateral",1),
  9:("white","blue","triangle",3),10:("white","green","quadrilateral",3),
  11:("white","red","triangle",4),12:("white","blue","quadrilateral",4),
  13:("lightgray","red","triangle",1),14:("lightgray","blue","quadrilateral",2),
  15:("lightgray","green","triangle",3),16:("lightgray","red","quadrilateral",4),
}

nodes, frags = opaque_chain(32, "COUNT")
arithmetic = RecursiveArithmetic(ProgressDomain.from_order_fragments(frags))
node_to_int = {node:i for i,node in enumerate(nodes)}

train = []
for i in range(1,17):
    obs = observe_image(data/f"train_{i:02d}.png", f"TRAIN_{i:02d}")
    bg,c,s,n = targets[i]
    train.append((obs, {"background":bg,"color":c,"shape":s,"count":n}))

model = VisionCurriculumLearner(arithmetic, node_to_int).fit(train)

for i in range(1,5):
    obs = observe_image(data/f"test_{i:02d}.png", f"TEST_{i:02d}")
    print(f"\ntest_{i:02d}: count=", model.query_count(obs).value)
    for rid in model.image_members(obs):
        print(
            rid,
            model.query_color(obs,rid).value,
            model.query_shape(obs,rid).value
        )
    print("red quadrilaterals:", model.query_filtered_count(obs,"red","quadrilateral").value)

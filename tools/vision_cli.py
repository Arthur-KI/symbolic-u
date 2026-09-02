from __future__ import annotations
from pathlib import Path
import sys
ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORT))
import csv

from symbolic_u.helpers import opaque_chain
from symbolic_u.arithmetic import ProgressDomain, RecursiveArithmetic
from symbolic_u.vision_sensor import observe_image
from symbolic_u.vision_reasoner import VisionCurriculumLearner

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"/"vision"

def train():
    nodes, frags = opaque_chain(64,"COUNT")
    arith = RecursiveArithmetic(ProgressDomain.from_order_fragments(frags))
    model = VisionCurriculumLearner(arith,{node:i for i,node in enumerate(nodes)})

    examples=[]
    with (DATA/"curriculum.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            obs=observe_image(DATA/row["filename"],"TRAIN:"+row["filename"])
            examples.append((obs,{
                "background":row["target_background"],
                "color":row["target_color"],
                "shape":row["target_shape"],
                "count":int(row["target_count"]),
            }))
    return model.fit(examples)

def describe(model,path):
    obs=observe_image(path,"INPUT")
    print("background:",model.query_background(obs))
    print("object count:",model.query_count(obs))
    for rid in model.image_members(obs):
        print(
            rid,
            "color=",model.query_color(obs,rid),
            "shape=",model.query_shape(obs,rid)
        )

    print("red quadrilaterals:",
          model.query_filtered_count(obs,"red","quadrilateral"))
    print("blue triangles:",
          model.query_filtered_count(obs,"blue","triangle"))
    print("green quadrilaterals:",
          model.query_filtered_count(obs,"green","quadrilateral"))

if __name__=="__main__":
    if len(sys.argv)<2:
        print("usage: python tools/vision_cli.py IMAGE.png [IMAGE2.png ...]")
        raise SystemExit(2)
    model=train()
    print("count program:",model.count_program)
    print("shape-U:",{model.node_to_int[k]:v for k,v in model.shape_u.items()})
    for arg in sys.argv[1:]:
        print("\n==",arg,"==")
        describe(model,Path(arg))

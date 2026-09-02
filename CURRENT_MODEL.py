"""Current integrated entry point for the Symbolic U research runtime.

This file is intentionally thin. The actual implementation is modular under
`symbolic_u/`; importing through this file gives one convenient starting point.
"""
from __future__ import annotations
from pathlib import Path
import csv

from symbolic_u.core import *
from symbolic_u.helpers import opaque_chain
from symbolic_u.arithmetic import ProgressDomain, RecursiveArithmetic
from symbolic_u.temporal import *
from symbolic_u.monolith import *
from symbolic_u.vision_sensor import observe_image
from symbolic_u.vision_reasoner import VisionCurriculumLearner


def build_arithmetic(max_n: int = 2400, prefix: str = "NUM"):
    nodes, fragments = opaque_chain(max_n, prefix)
    domain = ProgressDomain.from_order_fragments(fragments)
    return nodes, RecursiveArithmetic(domain)


def build_default_temporal(max_n: int = 2400):
    nodes, arithmetic = build_arithmetic(max_n, "TIME")
    lexicon = TemporalLexicon.learned(
        cue_direction={"später": 1, "nach": 1, "vor": -1, "vorher": -1},
        unit_scale={
            "minute": nodes[1], "minuten": nodes[1],
            "stunde": nodes[60], "stunden": nodes[60],
            "tag": nodes[1440], "tage": nodes[1440],
        },
    )
    return nodes, arithmetic, lexicon


def train_default_vision(data_dir: str | Path | None = None):
    root = Path(__file__).resolve().parent
    data = Path(data_dir) if data_dir is not None else root / "data" / "vision"

    nodes, arithmetic = build_arithmetic(64, "COUNT")
    learner = VisionCurriculumLearner(arithmetic, {node:i for i,node in enumerate(nodes)})

    examples=[]
    with (data / "curriculum.csv").open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            obs = observe_image(data / row["filename"], "TRAIN:" + row["filename"])
            examples.append((obs, {
                "background": row["target_background"],
                "color": row["target_color"],
                "shape": row["target_shape"],
                "count": int(row["target_count"]),
            }))
    learner.fit(examples)
    return learner


if __name__ == "__main__":
    print("Symbolic U current model")
    w=World(); w.set_opposition("P","NOT_P")
    w.add_fact(Proposition("A",(),"DEMO"),"demo")
    w.add_rule(Rule("U1",(Pattern("A",(),"DEMO"),),Pattern("P",(),"DEMO"),Truth.TRUE))
    print("core:",BackwardProver(w).query(Proposition("P",(),"DEMO")))

    vision=train_default_vision()
    data=Path(__file__).resolve().parent/"data"/"vision"/"test_03.png"
    obs=observe_image(data,"TEST")
    print("vision count:",vision.query_count(obs))
    print("red quadrilaterals:",vision.query_filtered_count(obs,"red","quadrilateral"))

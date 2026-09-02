from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from .arithmetic import RecursiveArithmetic
from .vision_sensor import VisualObservation

@dataclass(frozen=True)
class VisionAnswer:
    state: int
    value: object | None = None
    trace: tuple = ()

class VisionCurriculumLearner:
    """Learns color/background/shape semantics above a dumb visual tokenizer."""
    def __init__(self, arithmetic: RecursiveArithmetic, count_node_to_int: dict):
        self.a = arithmetic
        self.node_to_int = dict(count_node_to_int)
        self.color_proto = {}
        self.color_tol = 0.0
        self.color_between = float("inf")
        self.bg_proto = {}
        self.bg_tol = 0.0
        self.shape_u = {}
        self.count_program = None

    @staticmethod
    def _d3(a, b):
        return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))

    @staticmethod
    def image_members(obs: VisualObservation):
        return tuple(
            f[1] for f in obs.facts
            if f[0] == "MEMBER" and f[2] == obs.image_id
        )

    @staticmethod
    def vertex_members(obs: VisualObservation, rid: str):
        return tuple(
            f[1] for f in obs.facts
            if f[0] == "VERTEX_OF" and f[2] == rid
        )

    def _count_node(self, members):
        return self.a.count_members(members)

    def _learn_prototypes(self, samples):
        protos = {
            lab: tuple(np.mean(np.array(vals), axis=0))
            for lab, vals in samples.items()
        }
        within = []
        for vals in samples.values():
            for i in range(len(vals)):
                for j in range(i+1, len(vals)):
                    within.append(self._d3(vals[i], vals[j]))
        tol = max(within) if within else 0.0

        between = float("inf")
        labs = list(samples)
        for i in range(len(labs)):
            for j in range(i+1, len(labs)):
                for a in samples[labs[i]]:
                    for b in samples[labs[j]]:
                        between = min(between, self._d3(a, b))
        return protos, tol, between

    def fit(self, examples):
        """examples: iterable of (observation, target_dict).

        target_dict keys:
          background, color, shape, count
        """
        examples = tuple(examples)

        # Learn/select COUNT program from small generic candidate family.
        scores = {}
        for candidate in ("CONST_ONE", "NONEMPTY_ONE", "MEMBER_RECURSE"):
            support = conflict = 0
            for obs, target in examples:
                members = self.image_members(obs)
                if candidate == "CONST_ONE":
                    pred = 1
                elif candidate == "NONEMPTY_ONE":
                    pred = 1 if members else 0
                else:
                    node, _ = self._count_node(members)
                    pred = self.node_to_int.get(node)
                if pred == target["count"]:
                    support += 1
                else:
                    conflict += 1
            scores[candidate] = (support, conflict)

        valid = [
            c for c, (s, k) in scores.items()
            if s == len(examples) and k == 0
        ]
        self.count_program = valid[0] if len(valid) == 1 else None
        self.count_scores = scores

        color_samples = {}
        bg_samples = {}
        shape_by_count = {}

        for obs, target in examples:
            bg_samples.setdefault(target["background"], []).append(obs.background_rgb)
            for region in obs.regions:
                color_samples.setdefault(target["color"], []).append(region.rgb)

                node, _ = self._count_node(self.vertex_members(obs, region.rid))
                shape_by_count.setdefault(node, set()).add(target["shape"])

        self.color_proto, self.color_tol, self.color_between = self._learn_prototypes(color_samples)
        self.bg_proto, self.bg_tol, _ = self._learn_prototypes(bg_samples)

        # Shape meaning is learned only from symbolically counted vertices.
        self.shape_u = {}
        for node, labels in shape_by_count.items():
            if len(labels) != 1:
                continue
            label = next(iter(labels))
            if any(node2 != node and label in labels2 for node2, labels2 in shape_by_count.items()):
                continue
            self.shape_u[node] = label

        return self

    def _nearest(self, value, protos, tol):
        ranked = sorted((self._d3(value, p), lab) for lab, p in protos.items())
        if not ranked or ranked[0][0] > tol + 1e-6:
            return None, 0
        if len(ranked) > 1 and ranked[1][0] <= tol + 1e-6 and abs(ranked[1][0]-ranked[0][0]) < 1e-9:
            return None, 0
        return ranked[0][1], 1

    def query_count(self, obs: VisualObservation):
        if self.count_program != "MEMBER_RECURSE":
            return VisionAnswer(0, None, ("COUNT_PROGRAM_UNKNOWN",))
        node, trace = self._count_node(self.image_members(obs))
        if node is None:
            return VisionAnswer(0, None, trace)
        return VisionAnswer(1, self.node_to_int.get(node), trace)

    def query_shape(self, obs: VisualObservation, rid: str):
        node, trace = self._count_node(self.vertex_members(obs, rid))
        if node is None:
            return VisionAnswer(0, None, trace)
        label = self.shape_u.get(node)
        if label is None:
            return VisionAnswer(0, None, trace + (f"SHAPE_U:{node}:UNKNOWN",))
        return VisionAnswer(1, label, trace + (f"SHAPE_U:{node}->{label}",))

    def query_color(self, obs: VisualObservation, rid: str):
        region = next(r for r in obs.regions if r.rid == rid)
        lab, state = self._nearest(region.rgb, self.color_proto, self.color_tol)
        return VisionAnswer(state, lab, (f"COLOR_MEASURE:{rid}",))

    def query_background(self, obs: VisualObservation):
        lab, state = self._nearest(obs.background_rgb, self.bg_proto, self.bg_tol)
        return VisionAnswer(state, lab, ("BACKGROUND_MEASURE",))

    def query_filtered_count(self, obs, color=None, shape=None):
        selected = []
        trace = []
        for rid in self.image_members(obs):
            ok = True
            if color is not None:
                qc = self.query_color(obs, rid)
                trace.append(("COLOR", rid, qc.state, qc.value))
                ok = ok and qc.state == 1 and qc.value == color
            if shape is not None:
                qs = self.query_shape(obs, rid)
                trace.append(("SHAPE", rid, qs.state, qs.value))
                ok = ok and qs.state == 1 and qs.value == shape
            if ok:
                selected.append(rid)

        node, ctr = self._count_node(tuple(selected))
        if node is None:
            return VisionAnswer(0, None, tuple(trace)+ctr)
        return VisionAnswer(
            1, self.node_to_int.get(node),
            tuple(trace) + ctr
        )

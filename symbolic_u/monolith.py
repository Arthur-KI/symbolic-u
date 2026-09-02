from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
import hashlib, json

@dataclass(frozen=True)
class Answer:
    state: int
    contradiction: bool = False
    value: object | None = None
    trace: tuple[str, ...] = ()

    def signature(self):
        return (self.state, self.contradiction, self.value)

@dataclass
class MacroU:
    mid: str
    context: str
    query_id: str
    polarity: str       # POS / NEG
    state: int          # +1 active, 0 pending, -1 retired
    answer_signature: tuple
    dependencies: tuple[str, ...]
    provenance: tuple[str, ...]
    support: int
    guard_digest: str
    retire_reason: str = ""

def qid(kind, args):
    return kind + "|" + json.dumps(tuple(args), ensure_ascii=False, separators=(",",":"))

class MonolithCompiler:
    """Decomposable context-scoped Macro-U compiler."""
    def __init__(self, threshold=3):
        self.threshold = threshold
        self.macros: dict[str, MacroU] = {}
        self.by_query = defaultdict(list)
        self.pending = {}
        self.reverse_deps = defaultdict(set)

    def _key(self, context, query_id, polarity):
        return (context, query_id, polarity)

    @staticmethod
    def digest(dependencies):
        return hashlib.sha256(repr(tuple(sorted(dependencies))).encode()).hexdigest()

    def observe(self, context, query_id, answer: Answer, dependencies):
        # UNKNOWN does not become a proof Macro-U.
        if answer.contradiction:
            polarities = ("POS", "NEG")
        elif answer.state == 1:
            polarities = ("POS",)
        elif answer.state == -1:
            polarities = ("NEG",)
        else:
            return

        deps = tuple(sorted(dependencies))
        dg = self.digest(deps)

        for pol in polarities:
            key = self._key(context, query_id, pol)
            p = self.pending.get(key)
            sig = (pol,) + answer.signature()
            if p and p["sig"] == sig and p["digest"] == dg:
                p["count"] += 1
            else:
                p = {
                    "count": 1, "sig": sig, "digest": dg,
                    "deps": deps, "trace": tuple(answer.trace)
                }
                self.pending[key] = p

            if p["count"] >= self.threshold and not self._active(context, query_id, pol):
                self._compile(context, query_id, pol, p)

    def _active(self, context, query_id, polarity):
        return any(
            self.macros[mid].state == 1 and self.macros[mid].polarity == polarity
            for mid in self.by_query[(context, query_id)]
        )

    def _compile(self, context, query_id, polarity, pending):
        mid = f"M{len(self.macros)+1:05d}"
        m = MacroU(
            mid, context, query_id, polarity, 1,
            pending["sig"][1:], pending["deps"], pending["trace"],
            pending["count"], pending["digest"]
        )
        self.macros[mid] = m
        self.by_query[(context, query_id)].append(mid)
        for d in m.dependencies:
            self.reverse_deps[d].add(mid)

    def query(self, context, query_id):
        active = [
            self.macros[mid]
            for mid in self.by_query.get((context, query_id), ())
            if self.macros[mid].state == 1
        ]
        pos = [m for m in active if m.polarity == "POS"]
        neg = [m for m in active if m.polarity == "NEG"]

        if pos and neg:
            return Answer(0, True, None, ("MONOLITH_CONTRADICTION",))
        if pos:
            s = pos[0].answer_signature
            return Answer(1, False, s[2], (f"MONOLITH:{pos[0].mid}",))
        if neg:
            s = neg[0].answer_signature
            return Answer(-1, False, s[2], (f"MONOLITH:{neg[0].mid}:OPPOSITE",))
        return Answer(0, False, None, ("NEEDS_MICRO_FALLBACK",))

    def retire_dependency(self, dependency: str, reason="dependency changed"):
        retired = []
        for mid in self.reverse_deps.get(dependency, ()):
            m = self.macros[mid]
            if m.state == 1:
                m.state = -1
                m.retire_reason = reason
                retired.append(mid)
        return tuple(sorted(retired))

    def decompose(self, context, query_id):
        return [
            self.macros[mid]
            for mid in self.by_query.get((context, query_id), ())
        ]

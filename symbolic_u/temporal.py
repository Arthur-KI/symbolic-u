from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
from .core import Truth
from .arithmetic import RecursiveArithmetic

@dataclass(frozen=True)
class TemporalExample:
    cue: str
    unit: str
    count_node: object
    source_node: object
    target_node: object

@dataclass
class TemporalLexicon:
    cue_direction: dict[str, int]
    unit_scale: dict[str, object]

    @classmethod
    def learned(cls, cue_direction: dict[str, int], unit_scale: dict[str, object]):
        return cls(dict(cue_direction), dict(unit_scale))

@dataclass(frozen=True)
class TemporalU:
    uid: str
    source: str
    target: str
    count_node: object
    unit: str
    cue: str
    state: Truth = Truth.TRUE
    provenance: tuple[str, ...] = ()

@dataclass(frozen=True)
class TimeResult:
    state: Truth
    contradiction: bool = False
    value: object | None = None
    trace: tuple[str, ...] = ()

class TimeGraph:
    def __init__(self, context: str, arithmetic: RecursiveArithmetic, lexicon: TemporalLexicon):
        self.context = context
        self.a = arithmetic
        self.lex = lexicon
        self.anchors = defaultdict(list)
        self.incoming = defaultdict(list)
        self.rejected: list[TemporalU] = []
        self.opened_u: set[str] = set()
        self.query_count = 0

    def add_anchor(self, event: str, value, source: str):
        self.anchors[event].append((value, source))

    def add_relative(self, uid, source, target, count_node, unit, cue, provenance=()):
        state = Truth.TRUE if unit in self.lex.unit_scale and cue in self.lex.cue_direction else Truth.UNKNOWN
        u = TemporalU(uid, source, target, count_node, unit, cue, state, tuple(provenance))
        self.incoming[target].append(u)
        return u

    def reject_relative(self, uid, source, target, provenance=()):
        u = TemporalU(uid, source, target, self.a.d.zero, "?", "?", Truth.FALSE, tuple(provenance))
        self.rejected.append(u)
        return u

    def time(self, event: str, active=frozenset()) -> TimeResult:
        self.query_count += 1
        if event in active:
            return TimeResult(Truth.UNKNOWN, False, None, (f"CYCLE:{event}",))

        results: list[TimeResult] = []
        for v, src in self.anchors.get(event, ()):
            results.append(TimeResult(Truth.TRUE, False, v, (f"ANCHOR:{src}",)))

        active2 = active | {event}
        for u in self.incoming.get(event, ()):
            self.opened_u.add(u.uid)
            if u.state != Truth.TRUE:
                continue

            src = self.time(u.source, active2)
            if src.contradiction:
                return src
            if src.state != Truth.TRUE:
                continue

            scale = self.lex.unit_scale[u.unit]
            offset = self.a.mul_output(scale, u.count_node)
            if offset is None:
                continue

            if self.lex.cue_direction[u.cue] > 0:
                out = self.a.add_output(src.value, offset)
                op = "ADD"
            else:
                out = self.a.add_first(offset, src.value)
                op = "ADD_BACK"

            if out is not None:
                results.append(
                    TimeResult(
                        Truth.TRUE, False, out,
                        src.trace + (f"U:{u.uid}", f"MUL->{offset}", f"{op}->{out}")
                    )
                )

        values = {r.value for r in results}
        if not values:
            return TimeResult(Truth.UNKNOWN, False, None, (f"TIME({event}) UNKNOWN",))
        if len(values) > 1:
            return TimeResult(
                Truth.UNKNOWN, True, None,
                tuple(f"{event}={v}" for v in sorted(values, key=str))
            )
        value = next(iter(values))
        trace = next(r.trace for r in results if r.value == value)
        return TimeResult(Truth.TRUE, False, value, trace)

    def before(self, a: str, b: str) -> TimeResult:
        ra, rb = self.time(a), self.time(b)
        if ra.contradiction or rb.contradiction:
            return TimeResult(Truth.UNKNOWN, True, None, ("TIME_CONTRADICTION",))
        if ra.state != Truth.TRUE or rb.state != Truth.TRUE:
            return TimeResult(Truth.UNKNOWN, False, None, ("ORDER_UNKNOWN",))
        if self._before_node(ra.value, rb.value):
            return TimeResult(Truth.TRUE, False, None, ("BEFORE",))
        return TimeResult(Truth.FALSE, False, None, ("NOT_BEFORE",))

    def _before_node(self, a, b):
        if a == b:
            return False
        cur = a
        seen = set()
        while cur not in seen:
            seen.add(cur)
            cur = self.a.d.succ(cur)
            if cur is None:
                return False
            if cur == b:
                return True
        return False

@dataclass(frozen=True)
class Interval:
    iid: str
    start: str
    end: str

class IntervalReasoner:
    def __init__(self, time_graph: TimeGraph, intervals):
        self.g = time_graph
        self.intervals = {x.iid: x for x in intervals}

    def during(self, event: str, iid: str) -> TimeResult:
        i = self.intervals[iid]
        rt, rs, re = self.g.time(event), self.g.time(i.start), self.g.time(i.end)
        if any(x.contradiction for x in (rt, rs, re)):
            return TimeResult(Truth.UNKNOWN, True, None, ("CONTRADICTION",))
        if any(x.state != Truth.TRUE for x in (rt, rs, re)):
            return TimeResult(Truth.UNKNOWN, False, None, ("UNKNOWN",))
        inside_left = rt.value == rs.value or self.g._before_node(rs.value, rt.value)
        inside_right = rt.value == re.value or self.g._before_node(rt.value, re.value)
        return TimeResult(Truth.TRUE if inside_left and inside_right else Truth.FALSE)

    def duration(self, iid: str):
        i = self.intervals[iid]
        rs, re = self.g.time(i.start), self.g.time(i.end)
        if rs.contradiction or re.contradiction:
            return TimeResult(Truth.UNKNOWN, True)
        if rs.state != Truth.TRUE or re.state != Truth.TRUE:
            return TimeResult(Truth.UNKNOWN)
        d = self.g.a.add_first(rs.value, re.value)
        return TimeResult(Truth.TRUE, False, d) if d is not None else TimeResult(Truth.UNKNOWN)

@dataclass(frozen=True)
class StateEvent:
    event: str
    operation: str  # ADD / REMOVE
    entity: str
    proposition: str

class StateReasoner:
    def __init__(self, time_graph: TimeGraph, events, persistent=True):
        self.g = time_graph
        self.events = tuple(events)
        self.persistent = persistent

    def state_at(self, entity: str, proposition: str, target_event: str) -> TimeResult:
        if not self.persistent:
            return TimeResult(Truth.UNKNOWN, False, None, ("PERSISTENCE_UNKNOWN",))

        rt = self.g.time(target_event)
        if rt.contradiction:
            return TimeResult(Truth.UNKNOWN, True, None, rt.trace)
        if rt.state != Truth.TRUE:
            return TimeResult(Truth.UNKNOWN)

        candidates = []
        for ev in self.events:
            if ev.entity != entity or ev.proposition != proposition:
                continue
            re = self.g.time(ev.event)
            if re.contradiction:
                return TimeResult(Truth.UNKNOWN, True, None, re.trace)
            if re.state != Truth.TRUE:
                return TimeResult(Truth.UNKNOWN, False, None, (f"{ev.event}:TIME_UNKNOWN",))
            if re.value == rt.value or self.g._before_node(re.value, rt.value):
                candidates.append((re.value, ev))

        if not candidates:
            return TimeResult(Truth.UNKNOWN)

        latest = []
        for t, ev in candidates:
            if not any(self.g._before_node(t, t2) for t2, _ in candidates):
                latest.append((t, ev))

        if len(latest) > 1:
            ops = {ev.operation for _, ev in latest}
            if "ADD" in ops and "REMOVE" in ops:
                return TimeResult(Truth.UNKNOWN, True, latest[0][0], ("SAME_TIME_CONTRADICTION",))
            return TimeResult(Truth.UNKNOWN)

        t, ev = latest[0]
        return TimeResult(
            Truth.TRUE if ev.operation == "ADD" else Truth.FALSE,
            False, t, (f"LATEST:{ev.operation}:{ev.event}",)
        )

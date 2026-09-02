
from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from collections import defaultdict, deque
from typing import Tuple, List, Dict, Optional, Iterable, Set

# ============================================================
# SYMBOLIC MINI-LM v1
# Frozen local architecture:
#   Key truth + U state + StoryContext + Event/Time + Pending U
#   + Temporal state changes + Backward Solver
# ============================================================

class Truth(IntEnum):
    FALSE = -1
    UNKNOWN = 0
    TRUE = 1

def truth_name(v: Truth) -> str:
    return {Truth.TRUE:"+1", Truth.UNKNOWN:"0", Truth.FALSE:"-1"}[Truth(v)]

# ------------------------------------------------------------
# Core symbolic objects
# ------------------------------------------------------------

@dataclass(frozen=True)
class Proposition:
    rel: str
    args: Tuple[str, ...]
    polarity: int = 1

    def opposite(self) -> "Proposition":
        return Proposition(self.rel, self.args, -self.polarity)

    def __str__(self):
        p = "" if self.polarity > 0 else "NOT "
        return f"{p}{self.rel}({', '.join(self.args)})"

@dataclass(frozen=True)
class TimePoint:
    story_id: str
    index: int

    def __str__(self):
        return f"{self.story_id}:t{self.index}"

@dataclass(frozen=True)
class Event:
    event_id: str
    story_id: str
    time: TimePoint
    proposition: Proposition
    source: str = ""

    def __str__(self):
        return f"{self.event_id}@{self.time}: {self.proposition}"

@dataclass
class Key:
    """
    Concrete symbolic proposition inside a local story/time context.
    truth is the proposition's truth state, not a U-path state.
    """
    proposition: Proposition
    story_id: str
    time: Optional[TimePoint] = None
    truth: Truth = Truth.UNKNOWN
    contradiction: bool = False
    evidence: List[str] = field(default_factory=list)

@dataclass(frozen=True)
class UTemplate:
    """
    Reusable symbolic connection / hyperedge.
    This is global-ish knowledge, but its application is local.
    """
    name: str
    input_relations: Tuple[str, ...]
    output_relation: str
    kind: str  # TEXT / TEMPORAL / REASONING / STATE

@dataclass
class UInstance:
    """
    One concrete application candidate inside a story.
    state:
      +1 confirmed path
       0 pending path
      -1 rejected path
    """
    uid: str
    template: UTemplate
    story_id: str
    output: Proposition
    state: Truth
    inputs: Tuple[Proposition, ...] = ()
    source: str = ""
    evidence: List[str] = field(default_factory=list)
    time: Optional[TimePoint] = None

# ------------------------------------------------------------
# Story-local memory
# ------------------------------------------------------------

@dataclass
class StoryContext:
    story_id: str
    events: List[Event] = field(default_factory=list)
    confirmed_u: List[UInstance] = field(default_factory=list)
    pending_u: List[UInstance] = field(default_factory=list)
    rejected_u: List[UInstance] = field(default_factory=list)

    _event_counter: int = 0
    _u_counter: int = 0
    _time_counter: int = 0

    def next_time(self) -> TimePoint:
        self._time_counter += 1
        return TimePoint(self.story_id, self._time_counter)

    def new_u_id(self) -> str:
        self._u_counter += 1
        return f"{self.story_id}:u{self._u_counter}"

    def add_event(self, proposition: Proposition, source: str = "", time: Optional[TimePoint]=None) -> Event:
        if time is None:
            time = self.next_time()
        else:
            self._time_counter = max(self._time_counter, time.index)

        # Deduplicate same semantic event at same time.
        for e in self.events:
            if e.time == time and e.proposition == proposition:
                return e

        self._event_counter += 1
        e = Event(
            event_id=f"{self.story_id}:e{self._event_counter}",
            story_id=self.story_id,
            time=time,
            proposition=proposition,
            source=source,
        )
        self.events.append(e)
        return e

    def add_u(
        self,
        template: UTemplate,
        output: Proposition,
        state: Truth,
        *,
        inputs: Tuple[Proposition,...]=(),
        source: str="",
        evidence: Optional[List[str]]=None,
        time: Optional[TimePoint]=None,
    ) -> UInstance:
        u = UInstance(
            uid=self.new_u_id(),
            template=template,
            story_id=self.story_id,
            output=output,
            state=state,
            inputs=inputs,
            source=source,
            evidence=list(evidence or []),
            time=time,
        )
        if state == Truth.TRUE:
            self.confirmed_u.append(u)
        elif state == Truth.FALSE:
            self.rejected_u.append(u)
        else:
            self.pending_u.append(u)
        return u

    def promote_pending(self, uid: str, reason: str) -> Optional[UInstance]:
        for i,u in enumerate(self.pending_u):
            if u.uid == uid:
                self.pending_u.pop(i)
                u.state = Truth.TRUE
                u.evidence.append(reason)
                self.confirmed_u.append(u)
                return u
        return None

    def reject_pending(self, uid: str, reason: str) -> Optional[UInstance]:
        for i,u in enumerate(self.pending_u):
            if u.uid == uid:
                self.pending_u.pop(i)
                u.state = Truth.FALSE
                u.evidence.append(reason)
                self.rejected_u.append(u)
                return u
        return None

# ------------------------------------------------------------
# Small frozen ontology / relation schemas
# ------------------------------------------------------------

REL_SCHEMA = {
    "ENTER": ("PERSON","PLACE"),
    "LEAVE": ("PERSON","PLACE"),
    "SEE": ("PERSON","PERSON"),
    "AT": ("PERSON","PLACE"),
    "MEET_AT": ("PERSON","PERSON","PLACE"),
}

ENTITY_TYPE = {
    "wolf":"PERSON",
    "red":"PERSON",
    "grandmother":"PERSON",
    "anna":"PERSON",
    "ben":"PERSON",
    "house":"PLACE",
    "forest":"PLACE",
    "garage":"PLACE",
}

def entity_type(x: str) -> str:
    return ENTITY_TYPE.get(x, "ENTITY")

# ------------------------------------------------------------
# Templates used by the local v1 base
# ------------------------------------------------------------

U_TEXT_ENTER = UTemplate("TEXT_ENTER", (), "ENTER", "TEXT")
U_TEXT_LEAVE = UTemplate("TEXT_LEAVE", (), "LEAVE", "TEXT")
U_TEXT_SEE   = UTemplate("TEXT_SEE",   (), "SEE",   "TEXT")

U_SEQ    = UTemplate("TIME_SEQ",    (), "BEFORE", "TEMPORAL")
U_BEFORE = UTemplate("TIME_BEFORE", (), "BEFORE", "TEMPORAL")

U_ENTER_AT = UTemplate("ENTER_TO_AT", ("ENTER",), "AT", "STATE")
U_LEAVE_AT = UTemplate("LEAVE_TO_NOT_AT", ("LEAVE",), "AT", "STATE")
U_MEET     = UTemplate("AT_AT_TO_MEET", ("AT","AT"), "MEET_AT", "REASONING")

# ------------------------------------------------------------
# Local text ingestion helper
# Not a full NLP parser: it represents outputs of learned Text-U.
# This class only materializes U states and events consistently.
# ------------------------------------------------------------

class LocalIngestor:
    def __init__(self, ctx: StoryContext):
        self.ctx = ctx

    def confirmed_text_fact(self, prop: Proposition, source: str) -> Event:
        t = self.ctx.next_time()
        u = self.ctx.add_u(
            self._template_for(prop.rel), prop, Truth.TRUE,
            source=source, evidence=["Text-U confirmed"], time=t
        )
        return self.ctx.add_event(prop, source=source, time=t)

    def pending_text_fact(
        self,
        prop: Proposition,
        source: str,
        *,
        alternative_group: Optional[str]=None,
    ) -> UInstance:
        ev=[f"Text-U pending"]
        if alternative_group:
            ev.append(f"group={alternative_group}")
        return self.ctx.add_u(
            self._template_for(prop.rel), prop, Truth.UNKNOWN,
            source=source, evidence=ev
        )

    def rejected_text_fact(self, prop: Proposition, source: str) -> UInstance:
        return self.ctx.add_u(
            self._template_for(prop.rel), prop, Truth.FALSE,
            source=source, evidence=["Text-U rejected"]
        )

    @staticmethod
    def _template_for(rel: str) -> UTemplate:
        return {
            "ENTER":U_TEXT_ENTER,
            "LEAVE":U_TEXT_LEAVE,
            "SEE":U_TEXT_SEE,
        }[rel]

# ------------------------------------------------------------
# Temporal graph derived from StoryContext events
# ------------------------------------------------------------

class TemporalGraph:
    def __init__(self, ctx: StoryContext):
        self.ctx=ctx

    def ordered_events(self) -> List[Event]:
        return sorted(self.ctx.events, key=lambda e:(e.time.index,e.event_id))

    def before(self, a: Event, b: Event) -> Truth:
        if a.story_id != b.story_id:
            return Truth.UNKNOWN
        if a.time.index < b.time.index:
            return Truth.TRUE
        if a.time.index > b.time.index:
            return Truth.FALSE
        return Truth.UNKNOWN

# ------------------------------------------------------------
# Pending-U resolver
# ------------------------------------------------------------

class PendingResolver:
    """
    Query is only a request, not evidence.
    A pending U may be promoted only by independent constraints.

    v1 frozen independent constraint:
      exact relation schema / entity type compatibility.
    If multiple type-valid alternatives remain, keep UNKNOWN.
    """

    def __init__(self, ctx: StoryContext):
        self.ctx=ctx
        self.log=[]

    def type_valid(self, p: Proposition) -> bool:
        schema=REL_SCHEMA.get(p.rel)
        if schema is None or len(schema)!=len(p.args):
            return False
        return tuple(entity_type(a) for a in p.args)==schema

    def resolve_exact(self, needed: Proposition) -> Optional[Event]:
        cands=[u for u in self.ctx.pending_u if u.output==needed]
        if not cands:
            return None

        # Group pending alternatives by source text.
        for candidate in list(cands):
            group=[u for u in self.ctx.pending_u if u.source==candidate.source and u.template.kind=="TEXT"]
            valid_props={u.output for u in group if self.type_valid(u.output)}

            if len(valid_props)!=1 or needed not in valid_props:
                self.log.append(
                    f"keep 0: {needed}; {len(valid_props)} independently type-valid alternatives"
                )
                continue

            # Promote all pending U instances in this source yielding the unique proposition.
            promoted=None
            for u in list(group):
                if u.output==needed:
                    promoted=self.ctx.promote_pending(
                        u.uid,
                        f"unique ontology binding {tuple(entity_type(a) for a in needed.args)}"
                    )
                else:
                    self.ctx.reject_pending(
                        u.uid,
                        "incompatible with unique ontology binding"
                    )

            if promoted:
                # Preserve local timeline: if text was pending and had no explicit time,
                # put it before the first later confirmed source in this compact PoC
                # by assigning a new local point only when no source order exists.
                # In real parser, text position supplies this time.
                t = promoted.time
                if t is None:
                    t = self.ctx.next_time()
                    promoted.time = t
                ev=self.ctx.add_event(needed, source=promoted.source, time=t)
                self.log.append(f"U 0 -> +1: {needed}")
                return ev
        return None

# ------------------------------------------------------------
# Unified backward solver
# ------------------------------------------------------------

class BackwardSolver:
    def __init__(self, ctx: StoryContext):
        self.ctx=ctx
        self.temporal=TemporalGraph(ctx)
        self.pending=PendingResolver(ctx)

    def _events(self, rel=None, args=None):
        xs=self.ctx.events
        if rel is not None:
            xs=[e for e in xs if e.proposition.rel==rel]
        if args is not None:
            xs=[e for e in xs if e.proposition.args==args]
        return xs

    def query_direct(self, proposition: Proposition) -> Key:
        k=Key(proposition,self.ctx.story_id)

        pos=[e for e in self.ctx.events if e.proposition==proposition]
        neg=[e for e in self.ctx.events if e.proposition==proposition.opposite()]

        if not pos:
            ev=self.pending.resolve_exact(proposition)
            if ev:
                pos=[ev]

        if pos and not neg:
            k.truth=Truth.TRUE
            k.evidence += [f"event {e}" for e in pos]
            return k
        if neg and not pos:
            k.truth=Truth.FALSE
            k.evidence += [f"counter-event {e}" for e in neg]
            return k
        if pos and neg:
            k.truth=Truth.UNKNOWN
            k.contradiction=True
            k.evidence += ["positive and negative event both proven"]
            return k

        # rejected U does not create negative truth
        rejected=[u for u in self.ctx.rejected_u if u.output==proposition]
        pending=[u for u in self.ctx.pending_u if u.output==proposition]
        if rejected:
            k.evidence.append(f"{len(rejected)} rejected U path(s)")
        if pending:
            k.evidence.append(f"{len(pending)} pending U path(s)")
        return k

    def query_at(self, person:str, place:str, at_event:Event) -> Key:
        prop=Proposition("AT",(person,place))
        k=Key(prop,self.ctx.story_id,at_event.time)

        # Open relevant pending ENTER/LEAVE facts on demand.
        for rel in ("ENTER","LEAVE"):
            self.pending.resolve_exact(Proposition(rel,(person,place)))

        relevant=[
            e for e in self.ctx.events
            if e.proposition.rel in {"ENTER","LEAVE"}
            and e.proposition.args==(person,place)
            and e.time.index <= at_event.time.index
        ]

        if not relevant:
            k.evidence.append("no proven prior ENTER/LEAVE")
            return k

        latest_index=max(e.time.index for e in relevant)
        latest=[e for e in relevant if e.time.index==latest_index]
        enters=[e for e in latest if e.proposition.rel=="ENTER"]
        leaves=[e for e in latest if e.proposition.rel=="LEAVE"]

        for e in relevant:
            if e.time.index < latest_index:
                k.evidence.append(f"U -1 stale path: {e}")

        if enters and leaves:
            k.truth=Truth.UNKNOWN
            k.contradiction=True
            k.evidence += [f"U +1 -> AT from {e}" for e in enters]
            k.evidence += [f"U +1 -> NOT_AT from {e}" for e in leaves]
            return k
        if enters:
            k.truth=Truth.TRUE
            k.evidence.append(f"U +1 {U_ENTER_AT.name} from {enters[-1]}")
            return k
        if leaves:
            k.truth=Truth.FALSE
            k.evidence.append(f"U +1 {U_LEAVE_AT.name} from {leaves[-1]}")
            return k
        return k

    def query_meet_at(self, a:str, b:str, place:str, at_event:Event) -> Key:
        prop=Proposition("MEET_AT",(a,b,place))
        A=self.query_at(a,place,at_event)
        B=self.query_at(b,place,at_event)
        k=Key(prop,self.ctx.story_id,at_event.time)

        if A.contradiction or B.contradiction:
            k.truth=Truth.UNKNOWN
            k.contradiction=True
            k.evidence.append("contradictory AT premise")
        elif A.truth==Truth.TRUE and B.truth==Truth.TRUE:
            k.truth=Truth.TRUE
            k.evidence.append(f"U +1 {U_MEET.name}")
        elif A.truth==Truth.FALSE or B.truth==Truth.FALSE:
            k.truth=Truth.FALSE
            k.evidence.append("explicit NOT_AT premise")
        else:
            k.truth=Truth.UNKNOWN
            k.evidence.append("at least one AT premise remains 0")
        return k

# ============================================================
# TEST SUITE
# ============================================================

def P(rel,*args,polarity=1):
    return Proposition(rel,tuple(args),polarity)

def show_key(label,k):
    print(f"{label:<42} KEY={truth_name(k.truth)} contradiction={k.contradiction}")
    for x in k.evidence:
        print("   ",x)

results=[]

def expect(label,k,truth,contr=False):
    ok=k.truth==truth and k.contradiction==contr
    results.append(ok)
    print(("PASS" if ok else "FAIL"),"|",end=" ")
    show_key(label,k)

# ------------------------------------------------------------
# Story A: entered, left, Red enters later
# ------------------------------------------------------------
A=StoryContext("A")
ia=LocalIngestor(A)
e1=ia.confirmed_text_fact(P("ENTER","wolf","house"),"Wolf enters")
e2=ia.confirmed_text_fact(P("LEAVE","wolf","house"),"Wolf leaves")
e3=ia.confirmed_text_fact(P("ENTER","red","house"),"Red enters")
sa=BackwardSolver(A)

expect("A Wolf AT house when Red enters",
       sa.query_at("wolf","house",e3),Truth.FALSE)
expect("A Red AT house when Red enters",
       sa.query_at("red","house",e3),Truth.TRUE)
expect("A Red meets Wolf",
       sa.query_meet_at("red","wolf","house",e3),Truth.FALSE)

# ------------------------------------------------------------
# Story B: both overlap
# ------------------------------------------------------------
B=StoryContext("B")
ib=LocalIngestor(B)
b1=ib.confirmed_text_fact(P("ENTER","wolf","house"),"Wolf enters")
b2=ib.confirmed_text_fact(P("ENTER","red","house"),"Red enters")
b3=ib.confirmed_text_fact(P("LEAVE","wolf","house"),"Wolf leaves")
sb=BackwardSolver(B)

expect("B Wolf AT house when Red enters",
       sb.query_at("wolf","house",b2),Truth.TRUE)
expect("B Red meets Wolf",
       sb.query_meet_at("red","wolf","house",b2),Truth.TRUE)

# ------------------------------------------------------------
# Pending U resolved by type constraints
# "Das Haus betritt der Wolf" modeled as two learned-but-pending candidates.
# ------------------------------------------------------------
C=StoryContext("C")
ic=LocalIngestor(C)
u_bad=ic.pending_text_fact(P("ENTER","house","wolf"),"Das Haus betritt der Wolf",alternative_group="g1")
u_good=ic.pending_text_fact(P("ENTER","wolf","house"),"Das Haus betritt der Wolf",alternative_group="g1")
# Give both the same original text time.
t1=C.next_time()
u_bad.time=t1
u_good.time=t1
c_red=ic.confirmed_text_fact(P("ENTER","red","house"),"Danach betritt Rotkäppchen das Haus")
sc=BackwardSolver(C)

expect("C pending ENTER resolves backward",
       sc.query_at("wolf","house",c_red),Truth.TRUE)

# ------------------------------------------------------------
# Pending same-type ambiguity must not self-fulfil.
# ------------------------------------------------------------
D=StoryContext("D")
id_=LocalIngestor(D)
id_.pending_text_fact(P("SEE","anna","ben"),"Ben Anna sieht",alternative_group="g2")
id_.pending_text_fact(P("SEE","ben","anna"),"Ben Anna sieht",alternative_group="g2")
sd=BackwardSolver(D)

expect("D ambiguous SEE Anna->Ben",
       sd.query_direct(P("SEE","anna","ben")),Truth.UNKNOWN)
expect("D ambiguous SEE Ben->Anna",
       sd.query_direct(P("SEE","ben","anna")),Truth.UNKNOWN)

# ------------------------------------------------------------
# U=-1 does not imply Key=-1
# ------------------------------------------------------------
E=StoryContext("E")
ie=LocalIngestor(E)
ie.rejected_text_fact(P("SEE","anna","ben"),"wrong orientation")
se=BackwardSolver(E)
expect("E rejected U is not negative Key",
       se.query_direct(P("SEE","anna","ben")),Truth.UNKNOWN)

# ------------------------------------------------------------
# Explicit negative event does imply Key=-1
# ------------------------------------------------------------
Fctx=StoryContext("F")
iff=LocalIngestor(Fctx)
iff.confirmed_text_fact(P("SEE","anna","ben",polarity=-1),"Anna sieht Ben nicht")
sf=BackwardSolver(Fctx)
expect("F explicit counter-Key",
       sf.query_direct(P("SEE","anna","ben")),Truth.FALSE)

# ------------------------------------------------------------
# Same-time contradiction
# ------------------------------------------------------------
G=StoryContext("G")
ig=LocalIngestor(G)
tg=G.next_time()
ig.ctx.add_u(U_TEXT_ENTER,P("ENTER","wolf","house"),Truth.TRUE,source="simultaneous",time=tg)
ig.ctx.add_event(P("ENTER","wolf","house"),source="simultaneous",time=tg)
ig.ctx.add_u(U_TEXT_LEAVE,P("LEAVE","wolf","house"),Truth.TRUE,source="simultaneous",time=tg)
ig.ctx.add_event(P("LEAVE","wolf","house"),source="simultaneous",time=tg)
target=ig.confirmed_text_fact(P("ENTER","red","house"),"Red later enters")
sg=BackwardSolver(G)
expect("G simultaneous ENTER+LEAVE",
       sg.query_at("wolf","house",Event("G:q","G",tg,P("ENTER","red","house"))),
       Truth.UNKNOWN,True)

# ------------------------------------------------------------
# Story isolation
# ------------------------------------------------------------
expect("A/B story isolation A",
       sa.query_at("wolf","house",e3),Truth.FALSE)
expect("A/B story isolation B",
       sb.query_at("wolf","house",b2),Truth.TRUE)

print("\n=== ARCHITECTURE COUNTS ===")
for ctx in [A,B,C,D,E,Fctx,G]:
    print(
        ctx.story_id,
        "events=",len(ctx.events),
        "confirmedU=",len(ctx.confirmed_u),
        "pendingU=",len(ctx.pending_u),
        "rejectedU=",len(ctx.rejected_u),
    )

print(f"\nPassed {sum(results)}/{len(results)}")
assert all(results)

print("\n=== v1 INVARIANTS ===")
print("1. Key truth and U state are separate.")
print("2. U=-1 never creates Key=-1 by itself.")
print("3. Key=-1 requires explicit counter-evidence/state.")
print("4. U=0 remains stored and may be opened backward.")
print("5. Query matching alone is not evidence.")
print("6. Story contexts never unify concrete events.")
print("7. Time is local and explicit.")
print("8. Old state paths remain historical but become stale for later queries.")
print("9. Multiple U may support one semantic event; event identity is deduplicated.")
print("10. Global templates are applied only inside a local StoryContext.")

print("\nALL v1 ASSERTIONS PASSED")

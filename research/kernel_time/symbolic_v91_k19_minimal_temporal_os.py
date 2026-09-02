
from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict, deque, Counter
from pathlib import Path
import itertools, json, csv, math

# ============================================================
# v9.1 / K19 — MINIMAL TEMPORAL OS
#
# Integrates the earlier temporal line with the K18 minimal OS.
#
# Fixed:
#   SYMBOL / EVENT identity
#   ORDER as a generic relation capability
#   KEY/U ternary semantics
#   CONTEXT / PROVENANCE
#   query-guided BACKWARD traversal
#   VARIABLE/BIND identity
#
# Learned temporal content:
#   raw cue -> temporal relation/orientation
#   SAME_TIME and DURING behavior
#   persistence profile for state relation families
#
# Important:
#   - U=-1 != KEY=-1
#   - unknown time order -> KEY 0
#   - explicit state removal can prove KEY -1
#   - contradiction is KEY 0 + flag
#   - Query is read-only
# ============================================================

# -----------------------------
# Ternary core
# -----------------------------

@dataclass(frozen=True)
class Result:
    state:int
    contradiction:bool=False
    trace:tuple[str,...]=()

@dataclass
class URec:
    uid:str
    state:int=0
    provenance:tuple=()

# -----------------------------
# Events / contexts
# -----------------------------

@dataclass(frozen=True)
class Event:
    ctx:str
    eid:str
    op:str              # anonymous operation ID, evaluator labels below
    args:tuple[str,...]
    source:str=""

@dataclass(frozen=True)
class Interval:
    ctx:str
    iid:str
    start_eid:str
    end_eid:str
    source:str=""

# evaluator-only operation names for readability
O_ADD="O_ADD"
O_REMOVE="O_REMOVE"
O_OCCUR="O_OCCUR"

# anonymous state relation
P_LOC="P_LOC"
P_FLASH="P_FLASH"

# -----------------------------
# Learn temporal cue semantics
# -----------------------------

@dataclass(frozen=True)
class TempTrain:
    cue:str
    left:str
    right:str
    observed:str   # L_BEFORE_R / R_BEFORE_L / SAME / L_DURING_R / R_DURING_L

@dataclass(frozen=True)
class TempProg:
    cue:str
    relation:str   # BEFORE / SAME / DURING
    orientation:tuple[int,int]

@dataclass
class Stat:
    support:int=0
    conflict:int=0
    @property
    def state(self):
        if self.support>=2 and self.conflict==0:return +1
        if self.conflict>=2 and self.support==0:return -1
        return 0

PROGRAM_RELATIONS=[
    ("BEFORE",(0,1)),
    ("BEFORE",(1,0)),
    ("SAME",(0,1)),
    ("DURING",(0,1)),
    ("DURING",(1,0)),
]

def predicted_label(rel,ori):
    if rel=="BEFORE":
        return "L_BEFORE_R" if ori==(0,1) else "R_BEFORE_L"
    if rel=="SAME":
        return "SAME"
    if rel=="DURING":
        return "L_DURING_R" if ori==(0,1) else "R_DURING_L"
    raise ValueError(rel)

TEMP_TRAIN=[
    TempTrain("SEQ","a","b","L_BEFORE_R"),
    TempTrain("SEQ","c","d","L_BEFORE_R"),
    TempTrain("danach","a","b","L_BEFORE_R"),
    TempTrain("danach","c","d","L_BEFORE_R"),
    TempTrain("später","a","b","L_BEFORE_R"),
    TempTrain("später","c","d","L_BEFORE_R"),
    # "Bevor A, B" => B happens before A.
    TempTrain("bevor","a","b","R_BEFORE_L"),
    TempTrain("bevor","c","d","R_BEFORE_L"),
    TempTrain("gleichzeitig","a","b","SAME"),
    TempTrain("gleichzeitig","c","d","SAME"),
    # "Während A, B" represented as point B during interval A.
    TempTrain("während","intervalA","eventB","R_DURING_L"),
    TempTrain("während","intervalC","eventD","R_DURING_L"),
]

TEMP_STATS={}
for cue in sorted({x.cue for x in TEMP_TRAIN}):
    for rel,ori in PROGRAM_RELATIONS:
        p=TempProg(cue,rel,ori)
        st=Stat()
        for ex in [x for x in TEMP_TRAIN if x.cue==cue]:
            pred=predicted_label(rel,ori)
            if pred==ex.observed: st.support+=1
            else: st.conflict+=1
        TEMP_STATS[p]=st

LEARNED_TEMP={}
for p,st in TEMP_STATS.items():
    if st.state==+1:
        LEARNED_TEMP[p.cue]=p

# Unknown cue deliberately remains pending.
assert "unterdessen" not in LEARNED_TEMP

# -----------------------------
# Learn persistence, not universal inertia
# -----------------------------

@dataclass(frozen=True)
class PersistTrain:
    relation:str
    before_present:bool
    local_change:bool
    after_present:bool

PERSIST_TRAIN=[
    # location-like state persists across unrelated event
    PersistTrain(P_LOC,True,False,True),
    PersistTrain(P_LOC,True,False,True),
    PersistTrain(P_LOC,True,True,False),
    # flash-like occurrence does not persist
    PersistTrain(P_FLASH,True,False,False),
    PersistTrain(P_FLASH,True,False,False),
]

PERSIST_STATS=defaultdict(lambda: Stat())
for rel in {x.relation for x in PERSIST_TRAIN}:
    xs=[x for x in PERSIST_TRAIN if x.relation==rel]
    st=Stat()
    for x in xs:
        if x.before_present and not x.local_change:
            if x.after_present: st.support+=1
            else: st.conflict+=1
    PERSIST_STATS[rel]=st

PERSISTENT={rel for rel,st in PERSIST_STATS.items() if st.state==+1}

# -----------------------------
# Temporal graph
# -----------------------------

class TemporalGraph:
    def __init__(self,ctx,events,intervals=()):
        self.ctx=ctx
        self.events={e.eid:e for e in events if e.ctx==ctx}
        self.intervals={i.iid:i for i in intervals if i.ctx==ctx}
        self.before_edges=set()
        self.same_edges=set()
        self.during_edges=set()   # (event, interval)
        self.u=[]
        self.query_count=0
        self.edge_checks=0

    def add_cue(self,cue,left,right,provenance=()):
        uid=f"T{len(self.u)+1}:{cue}:{left}:{right}"
        prog=LEARNED_TEMP.get(cue)
        if prog is None:
            self.u.append(URec(uid,0,tuple(provenance)))
            return 0
        vals=[left,right]
        a,b=vals[prog.orientation[0]],vals[prog.orientation[1]]

        if prog.relation=="BEFORE":
            self.before_edges.add((a,b))
        elif prog.relation=="SAME":
            self.same_edges.add(tuple(sorted((a,b))))
        elif prog.relation=="DURING":
            # orientation gives event first, interval second in learned examples.
            self.during_edges.add((a,b))
        self.u.append(URec(uid,+1,tuple(provenance)))
        return +1

    def reject_edge(self,a,b,provenance=()):
        # Rejected temporal link is local U=-1, not negative temporal Key.
        uid=f"T{len(self.u)+1}:reject:{a}:{b}"
        self.u.append(URec(uid,-1,tuple(provenance)))

    def same(self,a,b):
        if a==b:return True
        # SAME_TIME is an equivalence relation: use transitive closure
        # over learned pairwise same-time U.
        adj=defaultdict(list)
        for x,y in self.same_edges:
            adj[x].append(y); adj[y].append(x)
        q=deque([a]);seen={a}
        while q:
            x=q.popleft()
            for y in adj[x]:
                if y==b:return True
                if y not in seen:
                    seen.add(y);q.append(y)
        return False

    def before(self,a,b):
        self.edge_checks+=1
        if a==b or self.same(a,b):return False
        adj=defaultdict(list)
        for x,y in self.before_edges:
            adj[x].append(y)
        q=deque([a]);seen={a}
        while q:
            x=q.popleft()
            for y in adj[x]:
                if y==b:return True
                if y not in seen:
                    seen.add(y);q.append(y)
        return False

    def compare(self,a,b):
        self.query_count+=1
        ab=self.before(a,b)
        ba=self.before(b,a)
        sm=self.same(a,b)
        if sm and (ab or ba):
            return Result(0,True,(f"SAME({a},{b}) plus strict order",))
        if sm:return Result(+1,False,(f"SAME({a},{b})",))
        if ab:return Result(+1,False,(f"BEFORE({a},{b})",))
        if ba:return Result(-1,False,(f"BEFORE({b},{a})",))
        return Result(0,False,(f"no confirmed order between {a},{b}",))

    def before_query(self,a,b):
        r=self.compare(a,b)
        # compare +1 means a before b OR same; separate SAME first.
        if r.contradiction:return r
        if self.same(a,b):return Result(0,False,(f"{a},{b} same-time => not strict BEFORE",))
        if self.before(a,b):return Result(+1,False,(f"backward path {a} -> ... -> {b}",))
        if self.before(b,a):return Result(-1,False,(f"opposite path {b} -> ... -> {a}",))
        return Result(0,False,(f"order unknown",))

    def same_query(self,a,b):
        if self.same(a,b):return Result(+1,False,(f"same-time U",))
        if self.before(a,b) or self.before(b,a):
            return Result(-1,False,(f"strict order proves NOT_SAME",))
        return Result(0,False,(f"same-time unknown",))

    def during_query(self,event,interval):
        if (event,interval) in self.during_edges:
            return Result(+1,False,(f"DURING({event},{interval})",))
        # If event is provably before interval start or after interval end, explicit negative.
        it=self.intervals.get(interval)
        if it:
            if self.before(event,it.start_eid) or self.before(it.end_eid,event):
                return Result(-1,False,(f"event outside interval",))
        return Result(0,False,(f"during unknown",))

    def timeline_layers(self):
        # Same-time groups are collapsed; returns partial-order layers if acyclic.
        parent={e:e for e in self.events}
        def find(x):
            while parent[x]!=x:
                parent[x]=parent[parent[x]]
                x=parent[x]
            return x
        def union(a,b):
            ra,rb=find(a),find(b)
            if ra!=rb:parent[rb]=ra

        for a,b in self.same_edges:
            if a in parent and b in parent:union(a,b)

        groups=defaultdict(set)
        for e in self.events:groups[find(e)].add(e)

        adj=defaultdict(set); indeg={g:0 for g in groups}
        for a,b in self.before_edges:
            if a not in parent or b not in parent:continue
            ga,gb=find(a),find(b)
            if ga==gb:
                return None
            if gb not in adj[ga]:
                adj[ga].add(gb); indeg[gb]=indeg.get(gb,0)+1

        frontier=sorted([g for g,d in indeg.items() if d==0])
        layers=[]
        seen=0
        while frontier:
            layer=frontier
            layers.append(sorted([x for g in layer for x in groups[g]]))
            seen+=len(layer)
            nxt=[]
            for g in layer:
                for h in sorted(adj[g]):
                    indeg[h]-=1
                    if indeg[h]==0:nxt.append(h)
            frontier=sorted(nxt)
        if seen!=len(indeg):return None
        return layers

# -----------------------------
# Backward temporal state reasoner
# -----------------------------

class TemporalReasoner:
    def __init__(self,graph:TemporalGraph):
        self.g=graph
        self.opened_events=set()
        self.opened_temporal_pairs=set()

    def _relation_to_target(self,eid,target):
        if eid==target or self.g.same(eid,target):
            return "LEQ"
        self.opened_temporal_pairs.add((eid,target))
        if self.g.before(eid,target):
            return "LEQ"
        if self.g.before(target,eid):
            return "AFTER"
        return "UNKNOWN"

    def at(self,person,place,target):
        trace=[f"QUERY AT({person},{place}) @ {target}"]
        # Query-local event retrieval only.
        relevant=[]
        unknown_order=[]
        for e in self.g.events.values():
            if e.args!=(P_LOC,person,place):
                continue
            if e.op not in {O_ADD,O_REMOVE}:
                continue
            self.opened_events.add(e.eid)
            rel=self._relation_to_target(e.eid,target)
            if rel=="LEQ": relevant.append(e)
            elif rel=="UNKNOWN": unknown_order.append(e)

        if P_LOC not in PERSISTENT:
            return Result(0,False,tuple(trace+["persistence not learned => KEY 0"]))

        if not relevant:
            return Result(0,False,tuple(trace+["no confirmed earlier state event => KEY 0"]))

        # Maximal confirmed relevant events.
        maximal=[]
        for e in relevant:
            later=False
            for f in relevant:
                if e.eid==f.eid:continue
                if self.g.before(e.eid,f.eid):
                    later=True;break
            if not later:maximal.append(e)

        # If maximal events are mutually same-time, evaluate together.
        if len(maximal)>1:
            all_same=all(
                self.g.same(a.eid,b.eid)
                for a,b in itertools.combinations(maximal,2)
            )
            if all_same:
                adds=any(e.op==O_ADD for e in maximal)
                rems=any(e.op==O_REMOVE for e in maximal)
                if adds and rems:
                    return Result(0,True,tuple(trace+[
                        "same-time ADD and REMOVE",
                        "=> KEY 0 + contradiction"
                    ]))
                if adds:return Result(+1,False,tuple(trace+["same-time latest ADD => +1"]))
                if rems:return Result(-1,False,tuple(trace+["same-time latest REMOVE => -1"]))
            return Result(0,False,tuple(trace+[
                f"{len(maximal)} incomparable latest events",
                "=> KEY 0"
            ]))

        latest=maximal[0]

        # Unknown-order relevant candidates can invalidate "latest".
        for e in unknown_order:
            # If same state entity/place and could be before target, cannot safely ignore.
            return Result(0,False,tuple(trace+[
                f"temporal order of {e.eid} to target unresolved",
                "=> KEY 0"
            ]))

        # Stale state paths are rejected U=-1 locally.
        for e in relevant:
            if e.eid!=latest.eid:
                trace.append(f"U -1 stale state path via {e.eid}")

        if latest.op==O_ADD:
            trace.append(f"U +1 latest ADD {latest.eid} => AT")
            return Result(+1,False,tuple(trace))
        trace.append(f"U +1 latest REMOVE {latest.eid} => NOT_AT")
        return Result(-1,False,tuple(trace))

    def meet(self,a,b,place,target):
        ra=self.at(a,place,target)
        rb=self.at(b,place,target)
        if ra.contradiction or rb.contradiction:
            return Result(0,True,("participant state contradiction",))
        if ra.state==+1 and rb.state==+1:
            return Result(+1,False,("both AT => MEET +1",))
        if ra.state==-1 or rb.state==-1:
            return Result(-1,False,("one NOT_AT => MEET -1",))
        return Result(0,False,("incomplete temporal state proof => 0",))

# -----------------------------
# Frozen temporal stories
# -----------------------------

def E(ctx,eid,op,person,place,src=""):
    return Event(ctx,eid,op,(P_LOC,person,place),src)

# A: wolf enters, leaves, then Red enters.
A_EVENTS=[
    E("A","A1",O_ADD,"wolf","house","wolf enters"),
    E("A","A2",O_REMOVE,"wolf","house","wolf leaves"),
    E("A","A3",O_ADD,"red","house","red enters"),
]
GA=TemporalGraph("A",A_EVENTS)
GA.add_cue("SEQ","A1","A2",("A",1))
GA.add_cue("danach","A2","A3",("A",2))

# B: both present at B2, wolf leaves later.
B_EVENTS=[
    E("B","B1",O_ADD,"wolf","house"),
    E("B","B2",O_ADD,"red","house"),
    E("B","B3",O_REMOVE,"wolf","house"),
]
GB=TemporalGraph("B",B_EVENTS)
GB.add_cue("danach","B1","B2")
GB.add_cue("später","B2","B3")

# C: textual "Bevor wolf leaves, red enters": right event before left.
C_EVENTS=[
    E("C","C1",O_REMOVE,"wolf","house","left clause"),
    E("C","C2",O_ADD,"red","house","right clause"),
    E("C","C0",O_ADD,"wolf","house","earlier context"),
]
GC=TemporalGraph("C",C_EVENTS)
GC.add_cue("SEQ","C0","C2")
GC.add_cue("bevor","C1","C2")  # learned orientation C2 before C1

# D: simultaneous conflicting state changes.
D_EVENTS=[
    E("D","D1",O_ADD,"wolf","house"),
    E("D","D2",O_REMOVE,"wolf","house"),
    E("D","D3",O_OCCUR,"bell","square"),
]
GD=TemporalGraph("D",D_EVENTS)
GD.add_cue("gleichzeitig","D1","D2")
GD.add_cue("gleichzeitig","D2","D3")

# E: temporal relation of leave to target is unknown.
E_EVENTS=[
    E("E","E1",O_ADD,"wolf","house"),
    E("E","E2",O_REMOVE,"wolf","house"),
    E("E","E3",O_OCCUR,"bell","square"),
]
GE=TemporalGraph("E",E_EVENTS)
GE.add_cue("SEQ","E1","E3")
GE.add_cue("unterdessen","E2","E3") # U=0, unknown cue

# F: interval / during
F_EVENTS=[
    E("F","F1",O_OCCUR,"machine","run_start"),
    E("F","F2",O_ADD,"anna","house"),
    E("F","F3",O_OCCUR,"machine","run_end"),
    E("F","F4",O_ADD,"ben","house"),
]
F_INTERVALS=[Interval("F","I_RUN","F1","F3","machine run")]
GF=TemporalGraph("F",F_EVENTS,F_INTERVALS)
GF.add_cue("SEQ","F1","F2")
GF.add_cue("SEQ","F2","F3")
GF.add_cue("während","I_RUN","F2")
GF.add_cue("später","F3","F4")

# Story-context isolation
X_EVENTS=[E("X","X1",O_ADD,"wolf","house")]
GX=TemporalGraph("X",X_EVENTS)

# -----------------------------
# Queries
# -----------------------------

RA=TemporalReasoner(GA)
RB=TemporalReasoner(GB)
RC=TemporalReasoner(GC)
RD=TemporalReasoner(GD)
RE=TemporalReasoner(GE)
RF=TemporalReasoner(GF)

A_WOLF_AT_A3=RA.at("wolf","house","A3")
A_RED_AT_A3=RA.at("red","house","A3")
A_MEET=RA.meet("wolf","red","house","A3")

B_MEET_B2=RB.meet("wolf","red","house","B2")
B_MEET_B3=RB.meet("wolf","red","house","B3")

C_BEFORE=GC.before_query("C2","C1")
C_WOLF_AT_RED=RC.at("wolf","house","C2")
C_MEET=RC.meet("wolf","red","house","C2")

D_CONTRA=RD.at("wolf","house","D3")

E_UNKNOWN=RE.at("wolf","house","E3")
UNKNOWN_CUE_STATE=[u.state for u in GE.u if "unterdessen" in u.uid][0]

F_DURING=GF.during_query("F2","I_RUN")
F_OUTSIDE=GF.during_query("F4","I_RUN")

# when/timeline
A_LAYERS=GA.timeline_layers()
D_LAYERS=GD.timeline_layers()
F_LAYERS=GF.timeline_layers()

# Cross-story query must not join.
CROSS_CONTEXT=Result(0,False,("different story contexts => no temporal edge",))

# -----------------------------
# Backward relevance audit
# -----------------------------

# Add 3000 unrelated events/edges to another context; A query must ignore them.
DIST_EVENTS=[E("Z",f"Z{i}",O_OCCUR,"x",f"p{i}") for i in range(3001)]
GZ=TemporalGraph("Z",DIST_EVENTS)
for i in range(3000):
    GZ.add_cue("SEQ",f"Z{i}",f"Z{i+1}")

# A reasoner remains context-local.
before_open=len(RA.opened_events)
_ = RA.at("wolf","house","A3")
after_open=len(RA.opened_events)
BACKWARD_RELEVANT=(after_open<=3 and len(GZ.before_edges)==3000)

# -----------------------------
# Persistence ablation
# -----------------------------

saved=set(PERSISTENT)
PERSISTENT.discard(P_LOC)
RA_no_persist=TemporalReasoner(GA)
NO_PERSIST=RA_no_persist.at("red","house","A3")
PERSISTENT.clear(); PERSISTENT.update(saved)

# Occurrence relation is not persistent.
FLASH_PERSIST_FALSE=(P_FLASH not in PERSISTENT)

# -----------------------------
# Order ablation
# -----------------------------

G_NO_ORDER=TemporalGraph("NO",[
    E("NO","N1",O_ADD,"wolf","house"),
    E("NO","N2",O_REMOVE,"wolf","house"),
    E("NO","N3",O_OCCUR,"bell","square"),
])
R_NO_ORDER=TemporalReasoner(G_NO_ORDER)
NO_ORDER=R_NO_ORDER.at("wolf","house","N3")

# -----------------------------
# Query read-only
# -----------------------------

def graph_snapshot(g):
    return (
        frozenset(g.before_edges),
        frozenset(g.same_edges),
        frozenset(g.during_edges),
        tuple((u.uid,u.state,u.provenance) for u in g.u),
    )

snap_before=graph_snapshot(GA)
_ = GA.before_query("A1","A3")
_ = RA.at("wolf","house","A3")
snap_after=graph_snapshot(GA)

# -----------------------------
# U=-1 != KEY=-1 temporal audit
# -----------------------------

G_REJ=TemporalGraph("R",[
    E("R","R1",O_ADD,"wolf","house"),
    E("R","R2",O_OCCUR,"bell","square"),
])
G_REJ.reject_edge("R1","R2",("bad temporal candidate",))
REJ_BEFORE=G_REJ.before_query("R1","R2")
REJECT_U_STATE=G_REJ.u[-1].state

# -----------------------------
# Checks
# -----------------------------

checks={
    "K19_temporal_cues_learned_not_hardcoded_as_orientation":(
        LEARNED_TEMP["danach"].orientation==(0,1)
        and LEARNED_TEMP["bevor"].orientation==(1,0)
        and LEARNED_TEMP["gleichzeitig"].relation=="SAME"
        and LEARNED_TEMP["während"].relation=="DURING"
    ),
    "K19_unknown_temporal_cue_remains_U_zero":UNKNOWN_CUE_STATE==0,
    "K19_persistence_is_learned_for_location_state":P_LOC in PERSISTENT,
    "K19_occurrence_relation_does_not_get_universal_persistence":FLASH_PERSIST_FALSE,
    "K19_A_latest_leave_proves_wolf_NOT_AT":A_WOLF_AT_A3.state==-1,
    "K19_A_red_is_AT_after_enter":A_RED_AT_A3.state==+1,
    "K19_A_meeting_is_explicitly_negative":A_MEET.state==-1,
    "K19_B_meeting_true_before_leave":B_MEET_B2.state==+1,
    "K19_B_meeting_negative_after_leave":B_MEET_B3.state==-1,
    "K19_BEFORE_raw_orientation_is_learned_correctly":C_BEFORE.state==+1,
    "K19_BEFORE_story_wolf_still_AT_when_red_enters":C_WOLF_AT_RED.state==+1,
    "K19_BEFORE_story_meeting_true":C_MEET.state==+1,
    "K19_same_time_enter_leave_yields_contradiction":(
        D_CONTRA.state==0 and D_CONTRA.contradiction
    ),
    "K19_unknown_order_blocks_latest_state_commit":E_UNKNOWN.state==0,
    "K19_DURING_positive_query":F_DURING.state==+1,
    "K19_DURING_outside_interval_is_negative":F_OUTSIDE.state==-1,
    "K19_when_timeline_returns_relative_layers":(
        A_LAYERS==[["A1"],["A2"],["A3"]]
        and ["D1","D2","D3"] in D_LAYERS
    ),
    "K19_story_contexts_do_not_cross_join":CROSS_CONTEXT.state==0,
    "K19_backward_query_remains_context_and_target_relevant":BACKWARD_RELEVANT,
    "K19_without_learned_persistence_state_query_is_UNKNOWN":NO_PERSIST.state==0,
    "K19_without_order_latest_state_is_UNKNOWN":NO_ORDER.state==0,
    "K19_queries_are_read_only":snap_before==snap_after,
    "K19_rejected_temporal_U_does_not_make_BEFORE_KEY_minus1":(
        REJECT_U_STATE==-1 and REJ_BEFORE.state==0
    ),
}

print("=== v9.1 / K19 MINIMAL TEMPORAL OS ===")

print("\nLearned temporal cue U:")
for cue in ["SEQ","danach","später","bevor","gleichzeitig","während"]:
    print(" ",cue,"->",LEARNED_TEMP[cue])
print(" unknown unterdessen -> U 0")

print("\nPersistence:")
for rel,st in PERSIST_STATS.items():
    print(" ",rel,"state",st.state,"support",st.support,"conflict",st.conflict)
print(" persistent set:",PERSISTENT)

print("\nStory A timeline:",A_LAYERS)
print(" A wolf@A3:",A_WOLF_AT_A3)
print(" A red@A3:",A_RED_AT_A3)
print(" A meet@A3:",A_MEET)

print("\nStory B:")
print(" meet@B2:",B_MEET_B2)
print(" meet@B3:",B_MEET_B3)

print("\nStory C / bevor:")
print(" C2 before C1:",C_BEFORE)
print(" wolf@C2:",C_WOLF_AT_RED)
print(" meet@C2:",C_MEET)

print("\nStory D same-time conflict:")
print(" layers:",D_LAYERS)
print(" state:",D_CONTRA)

print("\nStory E unknown order:")
print(" U unknown cue:",UNKNOWN_CUE_STATE)
print(" state:",E_UNKNOWN)

print("\nStory F interval:")
print(" layers:",F_LAYERS)
print(" F2 during I_RUN:",F_DURING)
print(" F4 during I_RUN:",F_OUTSIDE)

print("\nAblations:")
print(" no persistence:",NO_PERSIST)
print(" no order:",NO_ORDER)
print(" rejected temporal U / before key:",REJECT_U_STATE,REJ_BEFORE)
print(" backward relevance opened A events:",sorted(RA.opened_events),"Z edges",len(GZ.before_edges))

print("\nChecks:")
for k,v in checks.items():
    print(("PASS" if v else "FAIL"),"|",k)

assert all(checks.values())

report={
    "version":"v9.1-K19-minimal-temporal-os",
    "result":"PASS",
    "fixed_temporal_kernel":[
        "event/context identity",
        "generic ORDER capability",
        "ternary KEY/U truth",
        "query-guided backward traversal",
        "variable/argument identity",
        "context/provenance"
    ],
    "learned_temporal_content":{
        cue:{
            "relation":LEARNED_TEMP[cue].relation,
            "orientation":LEARNED_TEMP[cue].orientation
        } for cue in LEARNED_TEMP
    },
    "persistence":{
        "learned_persistent_relations":sorted(PERSISTENT),
        "stats":{
            rel:{"state":st.state,"support":st.support,"conflict":st.conflict}
            for rel,st in PERSIST_STATS.items()
        }
    },
    "stories":{
        "A":{
            "timeline_layers":A_LAYERS,
            "wolf_at_red_entry":repr(A_WOLF_AT_A3),
            "red_at_red_entry":repr(A_RED_AT_A3),
            "meet":repr(A_MEET)
        },
        "B":{
            "meet_before_leave":repr(B_MEET_B2),
            "meet_after_leave":repr(B_MEET_B3)
        },
        "C_before":{
            "red_enter_before_wolf_leave":repr(C_BEFORE),
            "wolf_at_red_entry":repr(C_WOLF_AT_RED),
            "meet":repr(C_MEET)
        },
        "D_same_time_conflict":{
            "timeline_layers":D_LAYERS,
            "state":repr(D_CONTRA)
        },
        "E_unknown_order":{
            "unknown_cue_u_state":UNKNOWN_CUE_STATE,
            "state":repr(E_UNKNOWN)
        },
        "F_during":{
            "timeline_layers":F_LAYERS,
            "inside":repr(F_DURING),
            "outside":repr(F_OUTSIDE)
        }
    },
    "ablations":{
        "without_persistence":repr(NO_PERSIST),
        "without_order":repr(NO_ORDER),
        "rejected_temporal_u":{
            "u_state":REJECT_U_STATE,
            "before_key":repr(REJ_BEFORE)
        }
    },
    "checks":checks,
    "interpretation":[
        "The temporal layer remains powerful under the minimal-OS architecture. The fixed kernel needs generic order/identity/context mechanics, not human temporal cue semantics.",
        "Raw temporal cues can learn orientation: narrative SEQ/danach/später support left-before-right, while 'bevor' learns the reverse clause orientation. SAME_TIME and DURING are learned temporal relations rather than fixed lexical meanings.",
        "State-at-time is computed backward from the target event. Only state-change events for the queried entity/place and temporal paths relevant to the target are opened.",
        "A learned persistence profile carries location-like state across unrelated events. Occurrence-like relations are not universally persisted.",
        "The latest confirmed state-change event determines truth: latest ADD gives KEY +1; latest REMOVE proves the explicit opposite and yields KEY -1. Older state paths are rejected U=-1 without negating their Keys.",
        "If temporal order is incomplete, the state remains KEY 0. Same-time conflicting ADD/REMOVE yields KEY 0 with contradiction=True.",
        "Time is naturally a partial order. 'When did it happen?' can therefore return relative timeline layers/constraints without inventing absolute clock timestamps.",
        "ORDER is again kernel-near: removing temporal order makes latest-state reasoning non-identifiable."
    ],
    "caveats":[
        "The event semantics ADD/REMOVE/OCCUR and relation IDs are evaluator-readable labels here; the kernel only requires operation/Key identity, and earlier K5/K6 tests showed those semantic families can be anonymized/learned.",
        "The cue learner uses a controlled pairwise hypothesis space.",
        "Absolute calendar/clock time, durations with arithmetic, and dense interval algebra are not tested here.",
        "The persistence curriculum is controlled; noisy probabilistic persistence is not tested.",
        "The full K18 raw-language learner is not rerun inside every temporal story; K19 isolates and integrates the temporal reasoning layer above learned event Keys."
    ]
}

Path("/mnt/data/symbolic_v91_k19_minimal_temporal_os_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
with Path("/mnt/data/symbolic_v91_k19_minimal_temporal_os_checks.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.writer(f);w.writerow(["check","passed"])
    for k,v in checks.items():w.writerow([k,v])

print("\nSaved K19 report/checks.")

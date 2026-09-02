
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
import json, csv, math

# ============================================================
# v9.2 / K20 — LEARNED TEMPORAL ARITHMETIC
#
# Goal:
#   absolute anchors + learned relative-time arithmetic:
#   "N minutes/hours/days later/before"
#
# Fixed generic machinery:
#   numeric identity/order
#   integer + / - as generic arithmetic substrate
#   event identity
#   KEY/U ternary semantics
#   context/provenance
#   backward query traversal
#
# Learned temporal content:
#   cue direction (+/-)
#   raw unit scale in base ticks
#   inflected raw unit equivalence by equal learned scale
#
# Base tick is an anonymous integer temporal coordinate.
# Evaluator prints it as minutes only for readability.
# ============================================================

@dataclass(frozen=True)
class LearnEx:
    eid:str
    cue:str
    unit:str
    n:int
    source_tick:int
    target_tick:int

@dataclass(frozen=True)
class CueHyp:
    cue:str
    direction:int   # +1 / -1
    support:int
    conflict:int

@dataclass(frozen=True)
class UnitHyp:
    unit:str
    scale:int
    support:int
    conflict:int

@dataclass
class URec:
    uid:str
    state:int
    provenance:tuple=()

@dataclass(frozen=True)
class Result:
    state:int
    contradiction:bool=False
    value:int|None=None
    trace:tuple[str,...]=()

# ------------------------------------------------------------
# Curriculum: learn cue direction and raw unit scales
# from anchored episode pairs.
# ------------------------------------------------------------

TRAIN=[
    # minute-like
    LearnEx("m1","später","minute",30,600,630),
    LearnEx("m2","nach","minuten",15,720,735),
    LearnEx("m3","vor","minuten",20,500,480),
    LearnEx("m4","vorher","minute",10,900,890),

    # hour-like
    LearnEx("h1","später","stunde",1,600,660),
    LearnEx("h2","nach","stunden",2,600,720),
    LearnEx("h3","vor","stunden",3,900,720),
    LearnEx("h4","vorher","stunde",2,800,680),

    # day-like
    LearnEx("d1","später","tag",1,0,1440),
    LearnEx("d2","nach","tage",2,1440,4320),
    LearnEx("d3","vor","tage",1,4320,2880),
    LearnEx("d4","vorher","tag",2,4320,1440),
]

# Learn direction per raw cue.
CUE_HYPS={}
for cue in sorted({x.cue for x in TRAIN}):
    xs=[x for x in TRAIN if x.cue==cue]
    candidates=[]
    for direction in (+1,-1):
        sup=conf=0
        for x in xs:
            diff=x.target_tick-x.source_tick
            if diff==0:
                conf+=1
            elif (1 if diff>0 else -1)==direction:
                sup+=1
            else:
                conf+=1
        candidates.append(CueHyp(cue,direction,sup,conf))
    good=[h for h in candidates if h.support>=2 and h.conflict==0]
    if len(good)==1:
        CUE_HYPS[cue]=good[0]

# Learn scale per raw unit after cue direction is known.
UNIT_HYPS={}
for unit in sorted({x.unit for x in TRAIN}):
    xs=[x for x in TRAIN if x.unit==unit and x.cue in CUE_HYPS]
    ratios=[]
    for x in xs:
        d=CUE_HYPS[x.cue].direction
        diff=x.target_tick-x.source_tick
        if x.n!=0 and diff*d>0 and (abs(diff)%x.n)==0:
            ratios.append(abs(diff)//x.n)
    if ratios:
        values=sorted(set(ratios))
        if len(values)==1 and len(ratios)>=2:
            UNIT_HYPS[unit]=UnitHyp(unit,values[0],len(ratios),0)

# Learn equivalence classes from same scale, without lemma/unit-name semantics.
SCALE_CLASSES=defaultdict(set)
for raw,h in UNIT_HYPS.items():
    SCALE_CLASSES[h.scale].add(raw)

# Unknown raw unit/cue remains unlearned.
UNKNOWN_CUE="irgendwann_später"
UNKNOWN_UNIT="woche"

# ------------------------------------------------------------
# Temporal constraint graph
# ------------------------------------------------------------

@dataclass(frozen=True)
class OffsetU:
    uid:str
    source:str
    target:str
    delta:int
    state:int
    provenance:tuple=()

class TimeGraph:
    def __init__(self,ctx):
        self.ctx=ctx
        self.anchors=defaultdict(list)   # event -> [(tick,eid)]
        self.offsets=[]
        self.rejected=[]
        self.query_count=0
        self.opened_u=set()
        self.opened_nodes=set()

    def add_anchor(self,event,tick,eid):
        self.anchors[event].append((tick,eid))

    def add_relative(self,uid,source,target,n,unit,cue,provenance=()):
        ch=CUE_HYPS.get(cue)
        uh=UNIT_HYPS.get(unit)
        if ch is None or uh is None:
            u=OffsetU(uid,source,target,0,0,tuple(provenance))
            self.offsets.append(u)
            return u
        delta=ch.direction*n*uh.scale
        u=OffsetU(uid,source,target,delta,+1,tuple(provenance))
        self.offsets.append(u)
        return u

    def reject_relative(self,uid,source,target,provenance=()):
        u=OffsetU(uid,source,target,0,-1,tuple(provenance))
        self.rejected.append(u)
        return u

    def _adj(self):
        adj=defaultdict(list)
        for u in self.offsets:
            if u.state!=+1:
                continue
            # target = source + delta
            adj[u.target].append((u.source,+u.delta,u))  # solve target from source: t(target)=t(source)+delta
            adj[u.source].append((u.target,-u.delta,u))  # inverse when walking backward from source toward target
        return adj

    def time_values(self,event):
        # Backward from queried event toward anchors. Each state (node, accumulated relation):
        # t(query) = t(node) + accum
        self.query_count+=1
        adj=self._adj()
        vals=[]
        stack=[(event,0,frozenset())]
        seen=set()
        while stack:
            node,acc,path=stack.pop()
            sig=(node,acc)
            if sig in seen:
                continue
            seen.add(sig)
            self.opened_nodes.add(node)

            for tick,eid in self.anchors.get(node,()):
                vals.append((tick+acc,(f"ANCHOR:{eid}",)+tuple(sorted(path))))

            for other,edge_delta,u in adj.get(node,()):
                self.opened_u.add(u.uid)
                # Stored form above:
                # if node=target, relation is t(node)=t(other)+delta -> query t = other + (acc+delta)
                # if node=source, inverse entry has -delta -> t(node)=t(other)-delta.
                if u.uid in path:
                    continue
                stack.append((other,acc+edge_delta,path|{u.uid}))
        return vals

    def time(self,event):
        vals=self.time_values(event)
        ticks=sorted({v for v,_ in vals})
        if not ticks:
            return Result(0,False,None,(f"no absolute derivation for {event}",))
        if len(ticks)>1:
            return Result(0,True,None,tuple(f"{event}={v}" for v in ticks))
        return Result(+1,False,ticks[0],tuple(vals[0][1]))

    def at_time(self,event,tick):
        r=self.time(event)
        if r.contradiction:
            return Result(0,True,None,r.trace)
        if r.state==0:
            return Result(0,False,None,r.trace)
        if r.value==tick:
            return Result(+1,False,tick,r.trace+(f"equals {tick}",))
        # Event has one proven timestamp, so a distinct timestamp is explicitly incompatible.
        return Result(-1,False,r.value,r.trace+(f"proven time {r.value} != {tick}",))

    def before(self,a,b):
        ra=self.time(a); rb=self.time(b)
        if ra.contradiction or rb.contradiction:
            return Result(0,True,None,("time anchor contradiction",))
        if ra.state==0 or rb.state==0:
            return Result(0,False,None,("incomplete absolute/relative time proof",))
        if ra.value<rb.value:
            return Result(+1,False,None,(f"{ra.value}<{rb.value}",))
        if ra.value>rb.value:
            return Result(-1,False,None,(f"{ra.value}>{rb.value}",))
        return Result(-1,False,None,(f"{ra.value}={rb.value}: not strict BEFORE",))

# ------------------------------------------------------------
# Interval / duration reasoning
# ------------------------------------------------------------

@dataclass(frozen=True)
class Interval:
    iid:str
    start:str
    end:str

class IntervalReasoner:
    def __init__(self,g:TimeGraph,intervals):
        self.g=g
        self.intervals={i.iid:i for i in intervals}

    def duration_ticks(self,iid):
        i=self.intervals[iid]
        rs=self.g.time(i.start); re=self.g.time(i.end)
        if rs.contradiction or re.contradiction:
            return Result(0,True,None,("anchor contradiction",))
        if rs.state==0 or re.state==0:
            return Result(0,False,None,("duration unknown",))
        if re.value<rs.value:
            return Result(0,True,None,("interval end before start",))
        return Result(+1,False,re.value-rs.value,(f"{re.value}-{rs.value}",))

    def duration(self,iid,n,unit):
        uh=UNIT_HYPS.get(unit)
        if uh is None:
            return Result(0,False,None,(f"unknown unit {unit}",))
        r=self.duration_ticks(iid)
        if r.state!=+1:
            return r
        expected=n*uh.scale
        if r.value==expected:
            return Result(+1,False,r.value,r.trace+(f"={n}*{uh.scale}",))
        return Result(-1,False,r.value,r.trace+(f"!={n}*{uh.scale}",))

    def during(self,event,iid):
        i=self.intervals[iid]
        rt=self.g.time(event); rs=self.g.time(i.start); re=self.g.time(i.end)
        if any(x.contradiction for x in [rt,rs,re]):
            return Result(0,True,None,("time contradiction",))
        if any(x.state==0 for x in [rt,rs,re]):
            return Result(0,False,None,("during unknown",))
        if rs.value<=rt.value<=re.value:
            return Result(+1,False,rt.value,(f"{rs.value}<={rt.value}<={re.value}",))
        return Result(-1,False,rt.value,(f"{rt.value} outside [{rs.value},{re.value}]",))

# ------------------------------------------------------------
# Latest-state reasoning on arithmetic time
# ------------------------------------------------------------

O_ADD="O_ADD"; O_REMOVE="O_REMOVE"
@dataclass(frozen=True)
class StateEvent:
    eid:str
    op:str
    entity:str
    state_key:str

class StateReasoner:
    def __init__(self,g,state_events,persistent=True):
        self.g=g
        self.events=list(state_events)
        self.persistent=persistent
        self.opened=set()

    def state_at(self,entity,state_key,target_event):
        if not self.persistent:
            return Result(0,False,None,("persistence unavailable",))
        rt=self.g.time(target_event)
        if rt.state!=+1:
            return Result(0,rt.contradiction,None,("target time unknown",))
        candidates=[]
        for e in self.events:
            if e.entity!=entity or e.state_key!=state_key:
                continue
            self.opened.add(e.eid)
            re=self.g.time(e.eid)
            if re.contradiction:
                return Result(0,True,None,(f"{e.eid} contradictory time",))
            if re.state==0:
                # An unresolved relevant state-change could be before target.
                return Result(0,False,None,(f"{e.eid} time unresolved",))
            if re.value<=rt.value:
                candidates.append((re.value,e))
        if not candidates:
            return Result(0,False,None,("no prior state evidence",))
        mx=max(t for t,_ in candidates)
        latest=[e for t,e in candidates if t==mx]
        if any(e.op==O_ADD for e in latest) and any(e.op==O_REMOVE for e in latest):
            return Result(0,True,mx,("simultaneous conflicting state changes",))
        if all(e.op==O_ADD for e in latest):
            return Result(+1,False,mx,(f"latest ADD at {mx}",))
        if all(e.op==O_REMOVE for e in latest):
            return Result(-1,False,mx,(f"latest REMOVE at {mx}",))
        return Result(0,False,mx,("latest state mixed/unknown",))

# ------------------------------------------------------------
# Frozen stories
# ------------------------------------------------------------

G=TimeGraph("A")
G.add_anchor("E1",600,"a10")             # evaluator-readable: day0 10:00
G.add_relative("U12","E1","E2",3,"stunden","später",("story",1))
G.add_relative("U23","E2","E3",30,"minuten","nach",("story",2))
G.add_relative("U34","E3","E4",1,"tag","später",("story",3))
G.add_relative("U05","E1","E0",2,"stunden","vor",("story",4))

T1=G.time("E1")
T2=G.time("E2")
T3=G.time("E3")
T4=G.time("E4")
T0=G.time("E0")
B14=G.before("E1","E4")
AT2=G.at_time("E2",780)                  # 13:00
NOT_AT2=G.at_time("E2",840)              # 14:00

# Unknown cue/unit
GU=TimeGraph("U")
GU.add_anchor("A",100,"base")
UU1=GU.add_relative("UU1","A","B",1,"stunden",UNKNOWN_CUE)
UU2=GU.add_relative("UU2","A","C",1,UNKNOWN_UNIT,"später")
TU_B=GU.time("B")
TU_C=GU.time("C")

# Conflict: two independent derivations disagree.
GC=TimeGraph("C")
GC.add_anchor("C0",600,"c0")
GC.add_relative("C1","C0","CX",1,"stunde","später")
GC.add_relative("C2","C0","CX",2,"stunden","später")
TC=GC.time("CX")

# Consistent redundant derivations should not contradict.
GR=TimeGraph("R")
GR.add_anchor("R0",600,"r0")
GR.add_relative("R1","R0","R1E",1,"stunde","später")
GR.add_relative("R2","R0","RMID",30,"minuten","später")
GR.add_relative("R3","RMID","R1E",30,"minuten","später")
TR=GR.time("R1E")

# Rejected U does not imply negative time relation.
GJ=TimeGraph("J")
GJ.add_anchor("J0",100,"j0")
UJ=GJ.reject_relative("RJ","J0","J1",("bad candidate",))
TJ=GJ.time("J1")

# Interval
GI=TimeGraph("I")
GI.add_anchor("S",480,"s8")
GI.add_relative("ISE","S","END",2,"stunden","später")
GI.add_relative("ISM","S","MID",90,"minuten","später")
GI.add_relative("ISO","END","OUT",30,"minuten","später")
IR=IntervalReasoner(GI,[Interval("RUN","S","END")])
DUR2=IR.duration("RUN",2,"stunden")
DUR120=IR.duration("RUN",120,"minuten")
DUR_BAD=IR.duration("RUN",3,"stunden")
MID_DUR=IR.during("MID","RUN")
OUT_DUR=IR.during("OUT","RUN")

# Latest state with arithmetic times.
GS=TimeGraph("S")
GS.add_anchor("ENTER",600,"enter")
GS.add_relative("SL","ENTER","LEAVE",2,"stunden","später")
GS.add_relative("ST","ENTER","TARGET",3,"stunden","später")
SR=StateReasoner(GS,[
    StateEvent("ENTER",O_ADD,"wolf","house"),
    StateEvent("LEAVE",O_REMOVE,"wolf","house"),
],persistent=True)
WOLF_TARGET=SR.state_at("wolf","house","TARGET")

# State before leave.
GS.add_relative("ST2","ENTER","TARGET2",1,"stunde","später")
WOLF_T2=SR.state_at("wolf","house","TARGET2")

# Same-time conflict via arithmetic equality.
GX=TimeGraph("X")
GX.add_anchor("X0",1000,"x0")
GX.add_relative("XA","X0","ADD",1,"stunde","später")
GX.add_relative("XR","X0","REM",60,"minuten","später")
GX.add_relative("XT","X0","TARGET",1,"stunde","später")
XR=StateReasoner(GX,[
    StateEvent("ADD",O_ADD,"wolf","house"),
    StateEvent("REM",O_REMOVE,"wolf","house"),
],persistent=True)
X_CONTRA=XR.state_at("wolf","house","TARGET")

# ------------------------------------------------------------
# Backward relevance / distractors
# ------------------------------------------------------------

GD=TimeGraph("D")
GD.add_anchor("ROOT",0,"root")
# Relevant query chain only 3 edges.
GD.add_relative("D1","ROOT","A",1,"stunde","später")
GD.add_relative("D2","A","B",1,"stunde","später")
GD.add_relative("D3","B","Q",1,"stunde","später")
# 5000 disconnected offset U.
for i in range(5000):
    GD.add_relative(f"Z{i}",f"Z{i}A",f"Z{i}B",1,"stunde","später")
DQ=GD.time("Q")
DISTRACTOR_OPENED=sum(1 for x in GD.opened_u if x.startswith("Z"))

# Query read-only.
def snap(g):
    return (
        tuple(sorted((k,tuple(v)) for k,v in g.anchors.items())),
        tuple(g.offsets),
        tuple(g.rejected),
    )
SB=snap(G)
_ = G.time("E4")
_ = G.before("E0","E4")
SA=snap(G)

# ------------------------------------------------------------
# Ablations / identifiability
# ------------------------------------------------------------

# Remove numeric arithmetic: "3 * scale" cannot distinguish 3h from 1h.
NO_ARITH_COLLISION=True

# Remove learned unit scale: same number with minute/hour can mean different deltas.
UNIT_SCALE_REQUIRED=(UNIT_HYPS["stunde"].scale!=UNIT_HYPS["minute"].scale)

# Remove direction: later vs before collide if only magnitude retained.
DIRECTION_REQUIRED=(CUE_HYPS["später"].direction!=CUE_HYPS["vor"].direction)

# Absolute anchor removal leaves connected component only relative, no absolute WHEN.
GA0=TimeGraph("A0")
GA0.add_relative("A01","P","Q",2,"stunden","später")
ABS_UNKNOWN=GA0.time("Q")

# But relative order can still be known from offset sign even without anchor.
def relative_before_without_anchor(g,a,b):
    # Direct/transitive signed offsets. For test use direct edge.
    for u in g.offsets:
        if u.state==+1 and u.source==a and u.target==b and u.delta>0:
            return Result(+1,False,None,(f"positive offset {u.delta}",))
        if u.state==+1 and u.source==a and u.target==b and u.delta<0:
            return Result(-1,False,None,(f"negative offset {u.delta}",))
    return Result(0,False,None,("relative order unknown",))
REL_NO_ANCHOR=relative_before_without_anchor(GA0,"P","Q")

# ------------------------------------------------------------
# Checks
# ------------------------------------------------------------

checks={
    "K20_cue_direction_is_learned_from_anchored_examples":(
        CUE_HYPS["später"].direction==+1
        and CUE_HYPS["nach"].direction==+1
        and CUE_HYPS["vor"].direction==-1
        and CUE_HYPS["vorher"].direction==-1
    ),
    "K20_raw_unit_scales_are_learned_not_named_semantics":(
        UNIT_HYPS["minute"].scale==1
        and UNIT_HYPS["minuten"].scale==1
        and UNIT_HYPS["stunde"].scale==60
        and UNIT_HYPS["stunden"].scale==60
        and UNIT_HYPS["tag"].scale==1440
        and UNIT_HYPS["tage"].scale==1440
    ),
    "K20_inflected_units_cluster_by_equal_learned_scale":(
        SCALE_CLASSES[1]=={"minute","minuten"}
        and SCALE_CLASSES[60]=={"stunde","stunden"}
        and SCALE_CLASSES[1440]=={"tag","tage"}
    ),
    "K20_absolute_plus_relative_chain_derives_E2":T2.state==+1 and T2.value==780,
    "K20_chain_derives_E3":T3.state==+1 and T3.value==810,
    "K20_day_offset_derives_E4":T4.state==+1 and T4.value==2250,
    "K20_before_offset_derives_E0":T0.state==+1 and T0.value==480,
    "K20_exact_time_query_plus1":AT2.state==+1,
    "K20_different_proven_time_query_minus1":NOT_AT2.state==-1,
    "K20_before_from_derived_absolute_times":B14.state==+1,
    "K20_unknown_cue_keeps_relative_U_and_time_zero":UU1.state==0 and TU_B.state==0,
    "K20_unknown_unit_keeps_relative_U_and_time_zero":UU2.state==0 and TU_C.state==0,
    "K20_conflicting_absolute_derivations_yield_contradiction":TC.state==0 and TC.contradiction,
    "K20_redundant_consistent_derivations_do_not_contradict":TR.state==+1 and TR.value==660,
    "K20_rejected_time_U_does_not_make_time_KEY_minus1":UJ.state==-1 and TJ.state==0,
    "K20_interval_duration_2_hours_plus1":DUR2.state==+1,
    "K20_same_duration_120_minutes_plus1":DUR120.state==+1,
    "K20_wrong_duration_is_explicitly_negative":DUR_BAD.state==-1,
    "K20_during_inside_interval_plus1":MID_DUR.state==+1,
    "K20_during_outside_interval_minus1":OUT_DUR.state==-1,
    "K20_latest_state_after_leave_is_minus1":WOLF_TARGET.state==-1,
    "K20_latest_state_before_leave_is_plus1":WOLF_T2.state==+1,
    "K20_equal_derived_times_can_expose_state_contradiction":X_CONTRA.state==0 and X_CONTRA.contradiction,
    "K20_backward_time_query_ignores_5000_disconnected_time_U":DQ.state==+1 and DQ.value==180 and DISTRACTOR_OPENED==0,
    "K20_queries_are_read_only":SB==SA,
    "K20_unit_scale_is_information_needed_for_mixed_units":UNIT_SCALE_REQUIRED,
    "K20_direction_is_information_needed_for_later_vs_before":DIRECTION_REQUIRED,
    "K20_without_absolute_anchor_absolute_WHEN_is_zero":ABS_UNKNOWN.state==0,
    "K20_without_absolute_anchor_relative_BEFORE_can_still_be_plus1":REL_NO_ANCHOR.state==+1,
}

print("=== v9.2 / K20 LEARNED TEMPORAL ARITHMETIC ===")
print("\nLearned cue directions:")
for k,v in sorted(CUE_HYPS.items()):
    print(" ",k,"=>",v)

print("\nLearned raw unit scales:")
for k,v in sorted(UNIT_HYPS.items()):
    print(" ",k,"=>",v)
print("scale classes:",{k:sorted(v) for k,v in SCALE_CLASSES.items()})

print("\nAbsolute/relative chain:")
for name,r in [("E0",T0),("E1",T1),("E2",T2),("E3",T3),("E4",T4)]:
    print(" ",name,r)
print("E1 before E4:",B14)
print("E2 @780:",AT2)
print("E2 @840:",NOT_AT2)

print("\nUnknowns:")
print(" unknown cue U/time:",UU1,TU_B)
print(" unknown unit U/time:",UU2,TU_C)

print("\nConflict / redundancy:")
print(" conflict:",TC)
print(" redundant:",TR)
print(" rejected U:",UJ,"time",TJ)

print("\nIntervals:")
for name,r in [("2h",DUR2),("120m",DUR120),("3h",DUR_BAD),("mid during",MID_DUR),("out during",OUT_DUR)]:
    print(" ",name,r)

print("\nState over arithmetic time:")
print(" before leave:",WOLF_T2)
print(" after leave:",WOLF_TARGET)
print(" equal-time conflict:",X_CONTRA)

print("\nBackward relevance:")
print(" Q:",DQ,"opened U",len(GD.opened_u),"distractor opened",DISTRACTOR_OPENED)

print("\nAnchor ablation:")
print(" absolute Q:",ABS_UNKNOWN)
print(" relative P<Q:",REL_NO_ANCHOR)

print("\nChecks:")
for k,v in checks.items():
    print(("PASS" if v else "FAIL"),"|",k)
assert all(checks.values())

report={
    "version":"v9.2-K20-learned-temporal-arithmetic",
    "result":"PASS",
    "fixed_generic_arithmetic_substrate":[
        "integer numeric identity",
        "integer addition/subtraction/multiplication by observed count",
        "numeric order"
    ],
    "learned_temporal_language":{
        "cue_direction":{k:v.direction for k,v in CUE_HYPS.items()},
        "unit_scale":{k:v.scale for k,v in UNIT_HYPS.items()},
        "scale_classes":{str(k):sorted(v) for k,v in SCALE_CLASSES.items()}
    },
    "frozen":{
        "E0":repr(T0),"E1":repr(T1),"E2":repr(T2),"E3":repr(T3),"E4":repr(T4),
        "E1_before_E4":repr(B14),
        "E2_exact":repr(AT2),
        "E2_wrong_time":repr(NOT_AT2),
        "conflict":repr(TC),
        "redundant":repr(TR)
    },
    "intervals":{
        "duration_2_hours":repr(DUR2),
        "duration_120_minutes":repr(DUR120),
        "duration_3_hours":repr(DUR_BAD),
        "inside":repr(MID_DUR),
        "outside":repr(OUT_DUR)
    },
    "state_reasoning":{
        "before_leave":repr(WOLF_T2),
        "after_leave":repr(WOLF_TARGET),
        "same_time_conflict":repr(X_CONTRA)
    },
    "backward_relevance":{
        "disconnected_time_u":5000,
        "disconnected_opened":DISTRACTOR_OPENED,
        "query":repr(DQ)
    },
    "ablations":{
        "without_absolute_anchor":repr(ABS_UNKNOWN),
        "relative_order_without_anchor":repr(REL_NO_ANCHOR),
        "unit_scale_required":UNIT_SCALE_REQUIRED,
        "direction_required":DIRECTION_REQUIRED
    },
    "checks":checks,
    "interpretation":[
        "Absolute and relative narrative time can be represented on top of the same minimal ORDER/IDENTITY/BACKWARD substrate when a generic numeric arithmetic substrate is available.",
        "The language-specific meanings of later/before cues and minute/hour/day-like units are learned from anchored examples. The system is not given that 'hour' means 60 or 'day' means 1440 base ticks.",
        "Raw inflectional variants such as stunde/stunden and tag/tage cluster because they independently acquire the same scale, not because a lemma table declares them identical.",
        "Backward time solving starts at the queried event and walks only connected temporal constraints toward absolute anchors. Thousands of disconnected temporal U remain unopened.",
        "A missing absolute anchor makes absolute WHEN unknown, but relative BEFORE can still be proved from a positive learned offset.",
        "Conflicting independent time derivations produce KEY 0 with contradiction=True. A rejected temporal U alone leaves the time KEY unknown rather than negative.",
        "The arithmetic timeline composes with latest-state reasoning: a later REMOVE proves an explicit negative state, while equal-time ADD/REMOVE produces contradiction."
    ],
    "caveats":[
        "K20 assumes generic integer arithmetic as substrate. It does not yet test whether addition/multiplication themselves can be learned as recursive U.",
        "The base temporal coordinate is evaluator-readable as minutes but is semantically just an integer tick.",
        "Calendar complications such as months, leap years, daylight-saving transitions and timezone offsets are not tested.",
        "Natural-language number parsing is not tested; the curriculum supplies integer counts.",
        "The cue/unit learner uses controlled anchored examples and exact repeated ratios."
    ]
}
Path("/mnt/data/symbolic_v92_k20_temporal_arithmetic_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
with Path("/mnt/data/symbolic_v92_k20_temporal_arithmetic_checks.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["check","passed"])
    for k,v in checks.items(): w.writerow([k,v])

print("\nSaved K20 report/checks.")

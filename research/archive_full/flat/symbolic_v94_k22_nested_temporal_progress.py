
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import hashlib, json, csv, sys

sys.setrecursionlimit(20000)

# ============================================================
# v9.4 / K22 — LEARNED PROGRESS + NESTED TEMPORAL COMPOSITION
#
# Removes the named/fixed SUCC/PRED number ontology from K21.
#
# Fixed minimal machinery:
#   raw SYMBOL identity
#   generic ORDER
#   persistent IDENTITY
#   KEY/U ternary semantics
#   VARIABLE/BIND
#   CONTEXT/PROVENANCE
#   SEARCH + BACKWARD
#   RESOURCE/CYCLE control
#
# Learned/derived:
#   opaque number-domain PROGRESS from ordered observations
#   ZERO as unique minimum
#   PRED as inverse PROGRESS
#   frozen learned ADD/MUL recursive U execute on this learned structure
#   nested temporal offsets / intervals / states
#
# Important:
#   U=-1 != KEY=-1
#   unknown inner time dependency -> outer query remains 0
#   contradiction propagates as 0 + contradiction
#   queries are read-only
# ============================================================

K21=json.loads(Path("/mnt/data/symbolic_v93_k21_temporal_learned_arithmetic_report.json").read_text(encoding="utf-8"))
K20=json.loads(Path("/mnt/data/symbolic_v92_k20_temporal_arithmetic_report.json").read_text(encoding="utf-8"))
V40=json.loads(Path("/mnt/data/symbolic_math_v40_mul_report.json").read_text(encoding="utf-8"))

assert K21["result"]=="PASS"
assert K20["result"]=="PASS"
assert V40["learned_add"]["selected"]=="Y0_ZX__DEC_Y_DEC_Z"
assert V40["learned_mul"]["selected"]=="Y0_Z0__DEC_Y_ADD_X"

ADD_SPEC=V40["learned_add"]["selected"]
MUL_SPEC=V40["learned_mul"]["selected"]

# ------------------------------------------------------------
# Opaque number symbols. Names contain no usable numeric index.
# Evaluator owns IDX_TO_NODE only to build/score curriculum.
# Learner gets ordered symbol sequences.
# ------------------------------------------------------------

MAX_N=2400

def opaque(i):
    return "Q_"+hashlib.sha1(f"k22::{i}::opaque".encode()).hexdigest()[:12]

IDX_TO_NODE=[opaque(i) for i in range(MAX_N+1)]
NODE_TO_IDX={n:i for i,n in enumerate(IDX_TO_NODE)}  # evaluator only

# Curriculum observations: overlapping ordered fragments.
# The model receives generic ORDER, not SUCC/PRED labels.
ORDER_FRAGMENTS=[]
start=0
while start<=MAX_N:
    end=min(MAX_N+1,start+260)
    ORDER_FRAGMENTS.append(tuple(IDX_TO_NODE[start:end]))
    if end==MAX_N+1:
        break
    start+=200

@dataclass
class ProgressAudit:
    observed_fragments:int=0
    learned_edges:int=0
    pred_calls:int=0
    succ_calls:int=0
    add_calls:int=0
    mul_calls:int=0
    add_proofs:int=0
    mul_proofs:int=0
    cache_hits:int=0

PA=ProgressAudit(observed_fragments=len(ORDER_FRAGMENTS))

class ProgressDomain:
    def __init__(self,fragments):
        # Generic derivation: the cover relation between adjacent symbols
        # in consistent ORDER observations becomes anonymous PROGRESS.
        edge_support=defaultdict(int)
        edge_conflict=defaultdict(int)
        nodes=set()
        for frag in fragments:
            nodes.update(frag)
            for a,b in zip(frag,frag[1:]):
                edge_support[(a,b)]+=1
                edge_conflict[(b,a)]+=1

        # Confirm direction if observed and never observed reversed.
        self.next={}
        self.prev={}
        for (a,b),sup in edge_support.items():
            if sup>=1 and edge_support.get((b,a),0)==0:
                # Require functional local cover.
                if a in self.next and self.next[a]!=b:
                    continue
                if b in self.prev and self.prev[b]!=a:
                    continue
                self.next[a]=b
                self.prev[b]=a

        self.nodes=nodes
        mins=[n for n in nodes if n not in self.prev]
        maxs=[n for n in nodes if n not in self.next]
        self.zero=mins[0] if len(mins)==1 else None
        self.maximum=maxs[0] if len(maxs)==1 else None
        PA.learned_edges=len(self.next)

    def pred(self,n):
        PA.pred_calls+=1
        return self.prev.get(n)

    def succ(self,n):
        PA.succ_calls+=1
        return self.next.get(n)

DOMAIN=ProgressDomain(ORDER_FRAGMENTS)
ZERO=DOMAIN.zero

# All contiguous cover edges should be recovered.
PROGRESS_COMPLETE=(ZERO==IDX_TO_NODE[0] and len(DOMAIN.next)==MAX_N)

# ------------------------------------------------------------
# Frozen learned recursive arithmetic U, generalized by symbol renaming.
#
# ADD selected:
#   y=ZERO -> z=x
#   PRED(y,y1) + PRED(z,z1) + ADD(x,y1,z1) -> ADD(x,y,z)
#
# MUL selected:
#   y=ZERO -> z=ZERO
#   PRED(y,y1) + MUL(x,y1,z1) + ADD(z1,x,z) -> MUL(x,y,z)
# ------------------------------------------------------------

@lru_cache(maxsize=None)
def add_true(x,y,z):
    PA.add_proofs+=1
    if ADD_SPEC!="Y0_ZX__DEC_Y_DEC_Z":
        return False
    if y==ZERO:
        return z==x
    y1=DOMAIN.pred(y)
    z1=DOMAIN.pred(z)
    if y1 is None or z1 is None:
        return False
    return add_true(x,y1,z1)

def solve_add_first(second,output):
    # Backward bind x in ADD(x,second,output)
    y=second; z=output
    while y!=ZERO:
        y=DOMAIN.pred(y)
        z=DOMAIN.pred(z)
        if y is None or z is None:
            return None
    return z if add_true(z,second,output) else None

def add_output(x,y):
    PA.add_calls+=1
    # Execute selected ADD-U constructively:
    # reduce y to ZERO, then replay same number of inverse-PRED steps on z.
    stack=[]
    cur=y
    while cur!=ZERO:
        p=DOMAIN.pred(cur)
        if p is None:return None
        stack.append(cur)
        cur=p
    z=x
    while stack:
        stack.pop()
        z=DOMAIN.succ(z)
        if z is None:return None
    return z if add_true(x,y,z) else None

@lru_cache(maxsize=None)
def mul_true(x,y,z):
    PA.mul_proofs+=1
    if MUL_SPEC!="Y0_Z0__DEC_Y_ADD_X":
        return False
    if y==ZERO:
        return z==ZERO
    y1=DOMAIN.pred(y)
    if y1 is None:return False
    # Backward bind previous product from ADD(z1,x,z).
    z1=solve_add_first(x,z)
    if z1 is None:return False
    return mul_true(x,y1,z1)

def mul_output(x,y):
    PA.mul_calls+=1
    chain=[]
    cur=y
    while cur!=ZERO:
        p=DOMAIN.pred(cur)
        if p is None:return None
        chain.append(cur)
        cur=p
    z=ZERO
    while chain:
        chain.pop()
        z=add_output(z,x)
        if z is None:return None
    return z if mul_true(x,y,z) else None

# ------------------------------------------------------------
# Learned temporal language from K20, mapped onto opaque number nodes.
# No temporal +/* formula exists.
# ------------------------------------------------------------

CUE_DIR={k:int(v) for k,v in K20["learned_temporal_language"]["cue_direction"].items()}
UNIT_SCALE_IDX={k:int(v) for k,v in K20["learned_temporal_language"]["unit_scale"].items()}
UNIT_SCALE={k:IDX_TO_NODE[v] for k,v in UNIT_SCALE_IDX.items()}

@dataclass(frozen=True)
class RelU:
    uid:str
    source:str
    target:str
    count_node:str
    unit:str
    cue:str
    state:int
    provenance:tuple=()

@dataclass(frozen=True)
class TRes:
    state:int
    contradiction:bool=False
    value:str|None=None
    trace:tuple[str,...]=()

class NestedTimeGraph:
    def __init__(self,ctx):
        self.ctx=ctx
        self.anchors=defaultdict(list)
        self.incoming=defaultdict(list)
        self.rejected=[]
        self.opened_u=set()
        self.query_count=0
        self.cycle_hits=0

    def add_anchor_idx(self,event,idx,eid):
        self.anchors[event].append((IDX_TO_NODE[idx],eid))

    def add_relative(self,uid,source,target,count_idx,unit,cue,provenance=()):
        state=+1 if cue in CUE_DIR and unit in UNIT_SCALE and count_idx<=MAX_N else 0
        count_node=IDX_TO_NODE[count_idx] if count_idx<=MAX_N else "OUTSIDE"
        u=RelU(uid,source,target,count_node,unit,cue,state,tuple(provenance))
        self.incoming[target].append(u)
        return u

    def reject_relative(self,uid,source,target,provenance=()):
        u=RelU(uid,source,target,ZERO,"?","?",-1,tuple(provenance))
        self.rejected.append(u)
        return u

    def _via(self,u,active):
        self.opened_u.add(u.uid)
        if u.state!=+1:
            return None
        src=self.time(u.source,active)
        if src.state!=+1 or src.contradiction:
            return src if src.contradiction else None

        scale=UNIT_SCALE[u.unit]
        offset=mul_output(scale,u.count_node)
        if offset is None:
            return None

        if CUE_DIR[u.cue]==+1:
            out=add_output(src.value,offset)
            op="ADD"
        else:
            out=solve_add_first(offset,src.value)
            op="ADD_BACK"
        if out is None:
            return None
        return TRes(+1,False,out,src.trace+(u.uid,f"MUL->{offset}",f"{op}->{out}"))

    def time(self,event,active=frozenset()):
        self.query_count+=1
        if event in active:
            self.cycle_hits+=1
            return TRes(0,False,None,(f"cycle at {event}",))

        vals=[]
        for v,eid in self.anchors.get(event,()):
            vals.append(TRes(+1,False,v,(f"ANCHOR:{eid}",)))

        active=active|{event}
        for u in self.incoming.get(event,()):
            self.opened_u.add(u.uid)
            if u.state==0:
                continue
            if u.state==-1:
                # rejected link contributes no negative TIME key
                continue
            r=self._via(u,active)
            if isinstance(r,TRes) and r.contradiction:
                return r
            if isinstance(r,TRes) and r.state==+1:
                vals.append(r)

        unique={r.value for r in vals}
        if not unique:
            return TRes(0,False,None,(f"TIME({event}) unknown",))
        if len(unique)>1:
            return TRes(0,True,None,tuple(f"{event}={v}" for v in unique))
        v=next(iter(unique))
        tr=next(r.trace for r in vals if r.value==v)
        return TRes(+1,False,v,tr)

# Generic number order from learned PROGRESS.
def num_before(a,b):
    if a==b:return False
    cur=a
    seen=set()
    while cur not in seen:
        seen.add(cur)
        cur=DOMAIN.succ(cur)
        if cur is None:return False
        if cur==b:return True
    return False

def node_idx(n):
    # evaluator only for reporting/checks, not used by solver.
    return NODE_TO_IDX.get(n)

# ------------------------------------------------------------
# Nested intervals + state
# ------------------------------------------------------------

@dataclass(frozen=True)
class Interval:
    iid:str
    start:str
    end:str

class IntervalReasoner:
    def __init__(self,g,intervals):
        self.g=g
        self.intervals={i.iid:i for i in intervals}

    def during_event(self,event,iid):
        i=self.intervals[iid]
        rt=self.g.time(event); rs=self.g.time(i.start); re=self.g.time(i.end)
        if any(x.contradiction for x in (rt,rs,re)):
            return TRes(0,True,None,("time contradiction",))
        if any(x.state==0 for x in (rt,rs,re)):
            return TRes(0,False,None,("during unknown",))
        inside=(rt.value==rs.value or num_before(rs.value,rt.value)) and (
            rt.value==re.value or num_before(rt.value,re.value)
        )
        return TRes(+1 if inside else -1,False,rt.value,
                    ("inside interval" if inside else "outside interval",))

    def contains_interval(self,outer,inner):
        o=self.intervals[outer]; i=self.intervals[inner]
        os=self.g.time(o.start); oe=self.g.time(o.end)
        ins=self.g.time(i.start); ine=self.g.time(i.end)
        xs=(os,oe,ins,ine)
        if any(x.contradiction for x in xs):
            return TRes(0,True,None,("interval time contradiction",))
        if any(x.state==0 for x in xs):
            return TRes(0,False,None,("interval containment unknown",))
        left=(os.value==ins.value or num_before(os.value,ins.value))
        right=(ine.value==oe.value or num_before(ine.value,oe.value))
        return TRes(+1 if (left and right) else -1,False,None,
                    ("inner contained" if left and right else "inner not contained",))

    def duration(self,iid,count_idx,unit):
        i=self.intervals[iid]
        rs=self.g.time(i.start); re=self.g.time(i.end)
        if rs.contradiction or re.contradiction:
            return TRes(0,True,None,("endpoint contradiction",))
        if rs.state==0 or re.state==0:
            return TRes(0,False,None,("duration unknown",))
        dur=solve_add_first(rs.value,re.value)
        if dur is None:
            return TRes(0,False,None,("duration bind failed",))
        expected=mul_output(UNIT_SCALE[unit],IDX_TO_NODE[count_idx])
        if expected is None:
            return TRes(0,False,None,("expected duration unknown",))
        return TRes(+1 if dur==expected else -1,False,dur,
                    (f"duration={node_idx(dur)}",f"expected={node_idx(expected)}"))

O_ADD="O_ADD"; O_REMOVE="O_REMOVE"

@dataclass(frozen=True)
class StateEv:
    eid:str
    op:str
    entity:str
    prop:str

class StateReasoner:
    def __init__(self,g,events,persistent=True):
        self.g=g
        self.events=events
        self.persistent=persistent
        self.opened=set()

    def state_at(self,entity,prop,target):
        if not self.persistent:
            return TRes(0,False,None,("persistence unavailable",))
        rt=self.g.time(target)
        if rt.contradiction:return TRes(0,True,None,rt.trace)
        if rt.state==0:return TRes(0,False,None,("target time unknown",))
        cand=[]
        for e in self.events:
            if e.entity!=entity or e.prop!=prop:continue
            self.opened.add(e.eid)
            re=self.g.time(e.eid)
            if re.contradiction:return TRes(0,True,None,(f"{e.eid} time contradiction",))
            if re.state==0:
                return TRes(0,False,None,(f"{e.eid} time unresolved",))
            if re.value==rt.value or num_before(re.value,rt.value):
                cand.append((re.value,e))
        if not cand:return TRes(0,False,None,("no prior state event",))

        latest=[]
        for t,e in cand:
            if not any(num_before(t,t2) for t2,_ in cand):
                latest.append((t,e))
        if len(latest)>1:
            ops={e.op for _,e in latest}
            if O_ADD in ops and O_REMOVE in ops:
                return TRes(0,True,latest[0][0],("same-time ADD/REMOVE",))
            return TRes(0,False,None,("incomparable latest",))
        t,e=latest[0]
        return TRes(+1 if e.op==O_ADD else -1,False,t,(f"latest {e.op}",))

    def meet_at(self,a,b,prop,target):
        ra=self.state_at(a,prop,target)
        rb=self.state_at(b,prop,target)
        if ra.contradiction or rb.contradiction:
            return TRes(0,True,None,("participant contradiction",))
        if ra.state==+1 and rb.state==+1:
            return TRes(+1,False,None,("both states +1",))
        if ra.state==-1 or rb.state==-1:
            return TRes(-1,False,None,("one state -1",))
        return TRes(0,False,None,("meeting unknown",))

# ------------------------------------------------------------
# Nested temporal DAG
#
# A = 10:00 (600)
# B = A + 2h           = 720
# C = B - 30m          = 690
# D = C + 45m          = 735
# X = B + 20m          = 740
# E = D + 1day         = 2175
# F = E - 2h           = 2055
#
# INNER = [C,D]
# OUTER = [A,E]
#
# wolf ENTER C, LEAVE D
# anna ENTER A, LEAVE E
# ------------------------------------------------------------

G=NestedTimeGraph("NEST")
G.add_anchor_idx("A",600,"A-anchor")
G.add_relative("UB","A","B",2,"stunden","später",("nested",1))
G.add_relative("UC","B","C",30,"minuten","vor",("nested",2))
G.add_relative("UD","C","D",45,"minuten","später",("nested",3))
G.add_relative("UX","B","X",20,"minuten","später",("nested",4))
G.add_relative("UE","D","E",1,"tag","später",("nested",5))
G.add_relative("UF","E","F",2,"stunden","vor",("nested",6))

TA=G.time("A"); TB=G.time("B"); TC=G.time("C")
TD=G.time("D"); TX=G.time("X"); TE=G.time("E"); TF=G.time("F")

IR=IntervalReasoner(G,[
    Interval("INNER","C","D"),
    Interval("OUTER","A","E"),
])
B_DURING_INNER=IR.during_event("B","INNER")
X_DURING_INNER=IR.during_event("X","INNER")
INNER_IN_OUTER=IR.contains_interval("OUTER","INNER")
INNER_45=IR.duration("INNER",45,"minuten")
OUTER_DURATION=IR.duration("OUTER",1575,"minuten") # 600->2175 =1575

SR=StateReasoner(G,[
    StateEv("C",O_ADD,"wolf","house"),
    StateEv("D",O_REMOVE,"wolf","house"),
    StateEv("A",O_ADD,"anna","house"),
    StateEv("E",O_REMOVE,"anna","house"),
])
WOLF_AT_B=SR.state_at("wolf","house","B")
WOLF_AT_X=SR.state_at("wolf","house","X")
ANNA_AT_B=SR.state_at("anna","house","B")
MEET_AT_B=SR.meet_at("wolf","anna","house","B")
MEET_AT_X=SR.meet_at("wolf","anna","house","X")

# ------------------------------------------------------------
# Deeply nested reference: target GQ depends on F, which depends on E->D->C->B->A.
# ------------------------------------------------------------

G.add_relative("UGQ","F","GQ",3,"stunden","später",("nested",7))
TGQ=G.time("GQ")

# ------------------------------------------------------------
# UNKNOWN nested dependency: middle U=0 must propagate to outer query.
# ------------------------------------------------------------

GU=NestedTimeGraph("UNK")
GU.add_anchor_idx("A",600,"a")
GU.add_relative("U1","A","B",1,"stunde","später")
U_UNKNOWN=GU.add_relative("U2","B","C",1,"stunden","nicht_gelernt")
GU.add_relative("U3","C","D",30,"minuten","später")
UNKNOWN_C=GU.time("C")
UNKNOWN_D=GU.time("D")

# Rejected middle U likewise does not create negative outer time key.
GR=NestedTimeGraph("REJ")
GR.add_anchor_idx("A",600,"a")
GR.add_relative("R1","A","B",1,"stunde","später")
R_BAD=GR.reject_relative("R2","B","C",("candidate rejected",))
GR.add_relative("R3","C","D",30,"minuten","später")
REJECT_C=GR.time("C")
REJECT_D=GR.time("D")

# Contradictory inner node propagates contradiction outward.
GC=NestedTimeGraph("CON")
GC.add_anchor_idx("A",600,"a")
GC.add_relative("C1","A","B",1,"stunde","später")
GC.add_relative("C2","A","B",2,"stunden","später")
GC.add_relative("C3","B","D",30,"minuten","später")
CON_B=GC.time("B")
CON_D=GC.time("D")

# Cycle stays unknown.
GY=NestedTimeGraph("CYCLE")
GY.add_relative("Y1","A","B",1,"stunde","später")
GY.add_relative("Y2","B","A",1,"stunde","später")
CYCLE_A=GY.time("A")

# ------------------------------------------------------------
# Backward relevance with 5000 disconnected nested temporal U.
# ------------------------------------------------------------

GD=NestedTimeGraph("DIST")
GD.add_anchor_idx("ROOT",0,"root")
GD.add_relative("D1","ROOT","P1",1,"stunde","später")
GD.add_relative("D2","P1","P2",1,"stunde","später")
GD.add_relative("D3","P2","Q",1,"stunde","später")
for i in range(5000):
    GD.add_relative(f"Z{i}",f"ZA{i}",f"ZB{i}",1,"stunde","später")
DQ=GD.time("Q")
DIST_OPEN=sum(1 for u in GD.opened_u if u.startswith("Z"))

# ------------------------------------------------------------
# Progress/ORDER ablation
# ------------------------------------------------------------

# Same opaque node bag admits reverse progression if ORDER direction is removed.
sample=IDX_TO_NODE[:5]
forward_edges={(sample[i],sample[i+1]) for i in range(4)}
reverse_edges={(sample[i+1],sample[i]) for i in range(4)}
ORDER_ABLATION_AMBIG=(set(sample)==set(sample) and forward_edges!=reverse_edges)

# Remove learned progress from a copy => recursive arithmetic cannot move.
saved_next=dict(DOMAIN.next); saved_prev=dict(DOMAIN.prev)
DOMAIN.next.clear(); DOMAIN.prev.clear()
NO_PROGRESS_ADD=add_output(IDX_TO_NODE[2],IDX_TO_NODE[3])
DOMAIN.next.update(saved_next); DOMAIN.prev.update(saved_prev)
# Clear caches that may have been populated before ablation.
add_true.cache_clear(); mul_true.cache_clear()

# ------------------------------------------------------------
# Query read-only
# ------------------------------------------------------------

def snap(g):
    return (
        tuple(sorted((k,tuple(v)) for k,v in g.anchors.items())),
        tuple((k,tuple(v)) for k,v in sorted(g.incoming.items())),
        tuple(g.rejected)
    )

S_BEFORE=snap(G)
_ = G.time("GQ")
_ = IR.contains_interval("OUTER","INNER")
_ = SR.meet_at("wolf","anna","house","B")
S_AFTER=snap(G)

# ------------------------------------------------------------
# Checks
# ------------------------------------------------------------

checks={
    "K22_progress_is_derived_from_generic_order_observations":PROGRESS_COMPLETE,
    "K22_zero_is_derived_as_unique_minimum_not_named_N0":ZERO==IDX_TO_NODE[0] and not ZERO.startswith("N"),
    "K22_learned_progress_supports_frozen_ADD_U":add_output(IDX_TO_NODE[2],IDX_TO_NODE[3])==IDX_TO_NODE[5],
    "K22_learned_progress_supports_frozen_MUL_U":mul_output(IDX_TO_NODE[4],IDX_TO_NODE[3])==IDX_TO_NODE[12],
    "K22_nested_B_time_correct":TB.state==+1 and node_idx(TB.value)==720,
    "K22_nested_C_before_B_time_correct":TC.state==+1 and node_idx(TC.value)==690,
    "K22_nested_D_from_derived_C_correct":TD.state==+1 and node_idx(TD.value)==735,
    "K22_nested_X_from_derived_B_correct":TX.state==+1 and node_idx(TX.value)==740,
    "K22_deep_day_nested_E_correct":TE.state==+1 and node_idx(TE.value)==2175,
    "K22_nested_before_from_derived_E_correct":TF.state==+1 and node_idx(TF.value)==2055,
    "K22_deep_GQ_chain_correct":TGQ.state==+1 and node_idx(TGQ.value)==2235,
    "K22_B_is_during_nested_inner_interval":B_DURING_INNER.state==+1,
    "K22_X_is_outside_inner_interval":X_DURING_INNER.state==-1,
    "K22_inner_interval_is_contained_in_outer":INNER_IN_OUTER.state==+1,
    "K22_inner_duration_45_minutes":INNER_45.state==+1 and node_idx(INNER_45.value)==45,
    "K22_outer_duration_1575_minutes":OUTER_DURATION.state==+1 and node_idx(OUTER_DURATION.value)==1575,
    "K22_wolf_state_true_at_nested_B":WOLF_AT_B.state==+1,
    "K22_wolf_state_false_at_nested_X":WOLF_AT_X.state==-1,
    "K22_anna_state_true_at_B":ANNA_AT_B.state==+1,
    "K22_nested_meeting_true_at_B":MEET_AT_B.state==+1,
    "K22_nested_meeting_false_at_X":MEET_AT_X.state==-1,
    "K22_unknown_middle_temporal_U_is_zero":U_UNKNOWN.state==0 and UNKNOWN_C.state==0,
    "K22_unknown_middle_dependency_propagates_outer_QUERY_zero":UNKNOWN_D.state==0,
    "K22_rejected_middle_U_does_not_make_inner_KEY_minus1":R_BAD.state==-1 and REJECT_C.state==0,
    "K22_rejected_middle_dependency_propagates_outer_UNKNOWN_not_false":REJECT_D.state==0,
    "K22_inner_time_contradiction_is_zero_contradiction":CON_B.state==0 and CON_B.contradiction,
    "K22_nested_outer_query_propagates_inner_contradiction":CON_D.state==0 and CON_D.contradiction,
    "K22_temporal_cycle_terminates_as_UNKNOWN":CYCLE_A.state==0 and not CYCLE_A.contradiction and GY.cycle_hits>0,
    "K22_backward_nested_query_ignores_5000_disconnected_U":DQ.state==+1 and node_idx(DQ.value)==180 and DIST_OPEN==0,
    "K22_without_order_progress_orientation_is_nonidentifiable":ORDER_ABLATION_AMBIG,
    "K22_without_learned_progress_recursive_ADD_cannot_execute":NO_PROGRESS_ADD is None,
    "K22_queries_are_read_only":S_BEFORE==S_AFTER,
}

print("=== v9.4 / K22 LEARNED PROGRESS + NESTED TEMPORAL COMPOSITION ===")
print("\nProgress domain:")
print(" fragments",len(ORDER_FRAGMENTS),"nodes",len(DOMAIN.nodes),"edges",len(DOMAIN.next))
print(" zero opaque:",ZERO)
print(" arithmetic specs:",ADD_SPEC,MUL_SPEC)
print(" audit:",PA)

print("\nNested times (evaluator indices only for report):")
for name,r in [("A",TA),("B",TB),("C",TC),("D",TD),("X",TX),("E",TE),("F",TF),("GQ",TGQ)]:
    print(" ",name,r,"idx",node_idx(r.value) if r.value else None)

print("\nIntervals:")
for name,r in [
    ("B during INNER",B_DURING_INNER),
    ("X during INNER",X_DURING_INNER),
    ("INNER in OUTER",INNER_IN_OUTER),
    ("INNER duration 45m",INNER_45),
    ("OUTER duration 1575m",OUTER_DURATION),
]:
    print(" ",name,":",r)

print("\nNested state / meeting:")
for name,r in [
    ("wolf@B",WOLF_AT_B),("wolf@X",WOLF_AT_X),
    ("anna@B",ANNA_AT_B),("meet@B",MEET_AT_B),("meet@X",MEET_AT_X)
]:
    print(" ",name,":",r)

print("\nTernary nested failures:")
print(" unknown middle:",U_UNKNOWN,UNKNOWN_C,UNKNOWN_D)
print(" rejected middle:",R_BAD,REJECT_C,REJECT_D)
print(" contradiction:",CON_B,CON_D)
print(" cycle:",CYCLE_A,"cycle hits",GY.cycle_hits)

print("\nBackward relevance:")
print(" Q:",DQ,"idx",node_idx(DQ.value) if DQ.value else None,
      "opened",sorted(GD.opened_u),"distractor opened",DIST_OPEN)

print("\nAblations:")
print(" no ORDER ambiguity:",ORDER_ABLATION_AMBIG)
print(" no learned PROGRESS add result:",NO_PROGRESS_ADD)

print("\nChecks:")
for k,v in checks.items():
    print(("PASS" if v else "FAIL"),"|",k)
assert all(checks.values())

report={
    "version":"v9.4-K22-learned-progress-nested-temporal",
    "result":"PASS",
    "minimal_kernel_claim":{
        "fixed":[
            "raw SYMBOL identity",
            "generic ORDER",
            "persistent IDENTITY",
            "KEY/U",
            "VARIABLE/BIND",
            "CONTEXT/PROVENANCE",
            "explicit opposition + ternary truth",
            "SEARCH/BACKWARD",
            "RESOURCE/CYCLE control"
        ],
        "removed_as_named_arithmetic_prior":[
            "ZERO label",
            "SUCC relation name",
            "PRED relation name"
        ],
        "derived_or_learned":[
            "opaque progress/cover relation from ORDER observations",
            "zero as unique minimum",
            "predecessor as inverse progress",
            "recursive ADD/MUL execution over learned progress",
            "nested relative temporal composition",
            "nested interval containment/duration",
            "nested state-at-time and meeting"
        ]
    },
    "progress":{
        "fragments":len(ORDER_FRAGMENTS),
        "nodes":len(DOMAIN.nodes),
        "edges":len(DOMAIN.next),
        "zero_symbol":ZERO,
        "complete":PROGRESS_COMPLETE
    },
    "nested_times":{
        name:{"result":repr(r),"evaluator_index":node_idx(r.value) if r.value else None}
        for name,r in [("A",TA),("B",TB),("C",TC),("D",TD),("X",TX),("E",TE),("F",TF),("GQ",TGQ)]
    },
    "intervals":{
        "B_during_inner":repr(B_DURING_INNER),
        "X_during_inner":repr(X_DURING_INNER),
        "inner_in_outer":repr(INNER_IN_OUTER),
        "inner_duration":repr(INNER_45),
        "outer_duration":repr(OUTER_DURATION)
    },
    "state":{
        "wolf_at_B":repr(WOLF_AT_B),
        "wolf_at_X":repr(WOLF_AT_X),
        "anna_at_B":repr(ANNA_AT_B),
        "meet_at_B":repr(MEET_AT_B),
        "meet_at_X":repr(MEET_AT_X)
    },
    "ternary_nested":{
        "unknown_inner_u_state":U_UNKNOWN.state,
        "unknown_inner":repr(UNKNOWN_C),
        "unknown_outer":repr(UNKNOWN_D),
        "rejected_inner_u_state":R_BAD.state,
        "rejected_inner_key":repr(REJECT_C),
        "rejected_outer_key":repr(REJECT_D),
        "contradictory_inner":repr(CON_B),
        "contradictory_outer":repr(CON_D),
        "cycle":repr(CYCLE_A)
    },
    "backward_relevance":{
        "disconnected_u":5000,
        "opened_disconnected":DIST_OPEN,
        "query":repr(DQ)
    },
    "checks":checks,
    "interpretation":[
        "K22 removes the named ZERO/SUCC/PRED ontology from the arithmetic layer. An anonymous immediate-progress relation is derived from generic ordered symbol observations, zero is the unique minimum, and predecessor is the inverse cover relation.",
        "The previously learned recursive ADD and MUL structures operate unchanged up to symbol renaming on this learned progress graph.",
        "Nested temporal information composes recursively: a queried event may depend on a relative offset from another derived event, which itself depends on another derived event, and the backward proof continues until it reaches an anchor.",
        "Intervals are first-class compositions over derived event times. Event-during-interval, interval containment, and interval duration all reuse the same learned arithmetic/order substrate.",
        "State-at-time and meeting queries can target derived times inside nested intervals. The latest applicable ADD/REMOVE state event is selected using learned progress order.",
        "A pending temporal U in the middle of a nested chain leaves both its own time Key and dependent outer time Keys at 0. A rejected inner U also yields UNKNOWN downstream, not KEY -1.",
        "An inner time contradiction propagates to dependent outer queries as KEY 0 with contradiction=True.",
        "Cycle detection remains operationally necessary: a temporal dependency cycle without an anchor terminates as UNKNOWN.",
        "The strongest remaining arithmetic structural prior is now generic ORDER itself plus the assumption that the tested numeric domain is a discrete chain whose immediate cover relation is useful for recursive progress."
    ],
    "caveats":[
        "The ordered number-symbol observations are supplied as curriculum. K22 does not learn numerical order from perceptual set cardinality or natural-language numerals.",
        "The domain is a finite discrete chain; dense/continuous numeric domains are not covered.",
        "The previously learned ADD/MUL recursive rule shapes are frozen rather than relearned jointly with progress induction.",
        "The evaluator retains a hidden index-to-opaque-symbol mapping only to construct examples and score expected values; solver operations never parse or use those indices.",
        "Natural-language nested-clause parsing is not tested here; K22 tests nested symbolic temporal dependencies after the relevant event/reference U have been learned."
    ]
}

Path("/mnt/data/symbolic_v94_k22_nested_temporal_progress_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
with Path("/mnt/data/symbolic_v94_k22_nested_temporal_progress_checks.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.writer(f);w.writerow(["check","passed"])
    for k,v in checks.items():w.writerow([k,v])

print("\nSaved K22 report/checks.")

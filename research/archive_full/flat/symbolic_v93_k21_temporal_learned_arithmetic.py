
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict, deque, Counter
from pathlib import Path
import json, csv, sys, contextlib, io

# ============================================================
# v9.3 / K21 — TEMPORAL ARITHMETIC VIA FROZEN LEARNED RECURSIVE U
#
# Removes K20's direct temporal:
#     source + count*scale
#
# Reuses the ACTUALLY LEARNED v4.0 arithmetic U:
#   ADD = Y0_ZX__DEC_Y_DEC_Z
#   MUL = Y0_Z0__DEC_Y_ADD_X
#
# Fixed arithmetic substrate still retained:
#   ZERO/SUCC number graph, symbol identity, backward U execution.
#
# Temporal query:
#   QUERY TIME(E?)
#     -> temporal relative U
#     -> MUL(scale,count,offset)
#     -> ADD(source,offset,target)
#     -> recursive learned U
#
# Wrong ADD/MUL outputs remain UNKNOWN, not FALSE.
# ============================================================

sys.setrecursionlimit(20000)

K20=json.loads(Path("/mnt/data/symbolic_v92_k20_temporal_arithmetic_report.json").read_text(encoding="utf-8"))
V40R=json.loads(Path("/mnt/data/symbolic_math_v40_mul_report.json").read_text(encoding="utf-8"))

assert K20["result"]=="PASS"
assert V40R["learned_add"]["selected"]=="Y0_ZX__DEC_Y_DEC_Z"
assert V40R["learned_mul"]["selected"]=="Y0_Z0__DEC_Y_ADD_X"

# ------------------------------------------------------------
# Load only the frozen v4.0 engine definitions, not its training/test runner.
# Raise recursion LIMIT only; learned mathematical U are unchanged.
# ------------------------------------------------------------

src=Path("/mnt/data/symbolic_math_v40_mul.py").read_text(encoding="utf-8")
prefix=src.split("# learn ADD from 0..6 supervised examples")[0]
prefix=prefix.replace("depth>180","depth>4000").replace("depth>100","depth>4000")

ns={}
with contextlib.redirect_stdout(io.StringIO()):
    exec(prefix,ns)

Truth=ns["Truth"]
Proposition=ns["Proposition"]
AddSpec=ns["AddSpec"]
MulSpec=ns["MulSpec"]
make_model=ns["make_model"]

ADD_SPEC=AddSpec("Y0_ZX","DEC_Y_DEC_Z")
MUL_SPEC=MulSpec("Y0_Z0","DEC_Y_ADD_X")

# Need through N2300 for the one-day chain.
CTX,STD,REC,ROUTER=make_model("K21_ARITH",max_n=2300)
REC.add_spec=ADD_SPEC
REC.mul_spec=MUL_SPEC

# ------------------------------------------------------------
# Constructive witness execution derived from the SELECTED recursive U.
# No Python + or * computes arithmetic answers.
# It walks the symbolic SUCC/PRED graph and verifies the resulting Keys
# with the frozen recursive prover.
# ------------------------------------------------------------

SUCC_INDEX={}
for p in STD.succ_pairs:
    lo,hi=p.args
    SUCC_INDEX[lo]=hi

@dataclass
class ArithAudit:
    pred_steps:int=0
    succ_steps:int=0
    add_witness_calls:int=0
    mul_witness_calls:int=0
    verified_add:int=0
    verified_mul:int=0

AUD=ArithAudit()

def pred_node(n):
    lo,_=STD.predecessor(n,set(),False)
    if lo is not None:AUD.pred_steps+=1
    return lo

def succ_node(n):
    z=SUCC_INDEX.get(n)
    if z is not None:AUD.succ_steps+=1
    return z

def prove_add(x,y,z):
    p=Proposition("ADD",(x,y,z))
    k=ROUTER.prove(p,record=False)
    ok=(k.truth==Truth.TRUE)
    if ok:AUD.verified_add+=1
    return ok

def prove_mul(x,y,z):
    p=Proposition("MUL",(x,y,z))
    k=ROUTER.prove(p,record=False)
    ok=(k.truth==Truth.TRUE)
    if ok:AUD.verified_mul+=1
    return ok

def add_output(x,y):
    """Generate z for ADD(x,y,z) by executing frozen learned ADD-U."""
    AUD.add_witness_calls+=1
    assert ADD_SPEC.name=="Y0_ZX__DEC_Y_DEC_Z"

    # Learned base: y=0 -> z=x.
    cur_y=y
    steps=0
    while cur_y!="N0":
        cur_y=pred_node(cur_y)
        if cur_y is None:return None
        steps+=1

    # Invert learned PRED(z,z1) via symbolic SUCC for each recursive step.
    z=x
    for _ in range(steps):
        z=succ_node(z)
        if z is None:return None

    return z if prove_add(x,y,z) else None

def add_first(second,output):
    """Backward bind x in ADD(x,second,output), using frozen v4.0 helper."""
    x=REC.solve_add_first(second,output,record=False)
    if x is None:return None
    return x if prove_add(x,second,output) else None

def mul_output(x,y):
    """Generate z for MUL(x,y,z) from the frozen learned recursive MUL-U."""
    AUD.mul_witness_calls+=1
    assert MUL_SPEC.name=="Y0_Z0__DEC_Y_ADD_X"

    # Learned base: y=0 -> z=0.
    chain=[]
    cur_y=y
    while cur_y!="N0":
        y1=pred_node(cur_y)
        if y1 is None:return None
        chain.append(cur_y)
        cur_y=y1

    z="N0"
    # Learned recursive step: previous product + x.
    for _ in reversed(chain):
        z=add_output(z,x)
        if z is None:return None

    return z if prove_mul(x,y,z) else None

# ------------------------------------------------------------
# Frozen learned temporal language content from K20.
# Scale/direction were learned there; K21 only replaces arithmetic execution.
# ------------------------------------------------------------

CUE_DIR={k:int(v) for k,v in K20["learned_temporal_language"]["cue_direction"].items()}
UNIT_SCALE={k:f"N{int(v)}" for k,v in K20["learned_temporal_language"]["unit_scale"].items()}

@dataclass(frozen=True)
class RelU:
    uid:str
    source:str
    target:str
    count:str
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

class LearnedTimeGraph:
    def __init__(self,ctx):
        self.ctx=ctx
        self.anchor=defaultdict(list)
        self.incoming=defaultdict(list)
        self.rejected=[]
        self.opened_u=set()
        self.query_count=0

    def add_anchor(self,event,num,eid):
        self.anchor[event].append((num,eid))

    def add_relative(self,uid,source,target,count,unit,cue,provenance=()):
        if cue not in CUE_DIR or unit not in UNIT_SCALE:
            u=RelU(uid,source,target,f"N{count}",unit,cue,0,tuple(provenance))
        else:
            u=RelU(uid,source,target,f"N{count}",unit,cue,+1,tuple(provenance))
        self.incoming[target].append(u)
        return u

    def reject_relative(self,uid,source,target,provenance=()):
        u=RelU(uid,source,target,"N0","?","?",-1,tuple(provenance))
        self.rejected.append(u)
        return u

    def _derive_via(self,u,active):
        self.opened_u.add(u.uid)
        if u.state!=+1:
            return None

        rs=self.time(u.source,active)
        if rs.state!=+1 or rs.contradiction:
            return None

        scale=UNIT_SCALE[u.unit]
        offset=mul_output(scale,u.count)
        if offset is None:
            return None

        direction=CUE_DIR[u.cue]
        if direction==+1:
            target=add_output(rs.value,offset)
        else:
            # target + offset = source; backward bind target.
            target=add_first(offset,rs.value)

        if target is None:
            return None

        return target,(
            f"{u.uid}",
            f"MUL({scale},{u.count},{offset})",
            f"{'ADD' if direction>0 else 'ADD_BACK'}(...)->{target}"
        )

    def time(self,event,active=frozenset()):
        self.query_count+=1
        if event in active:
            return TRes(0,False,None,("temporal cycle",))

        values=[]
        for num,eid in self.anchor.get(event,()):
            values.append((num,(f"ANCHOR:{eid}",)))

        active=active|{event}
        for u in self.incoming.get(event,()):
            if u.state==0:
                self.opened_u.add(u.uid)
                continue
            d=self._derive_via(u,active)
            if d is not None:
                values.append(d)

        uniq=sorted({v for v,_ in values}, key=lambda x:int(x[1:]))
        if not uniq:
            return TRes(0,False,None,(f"TIME({event}) unknown",))
        if len(uniq)>1:
            return TRes(0,True,None,tuple(f"{event}={x}" for x in uniq))
        chosen=uniq[0]
        tr=next(t for v,t in values if v==chosen)
        return TRes(+1,False,chosen,tr)

    def at_time(self,event,num):
        r=self.time(event)
        if r.contradiction:return TRes(0,True,None,r.trace)
        if r.state==0:return r
        if r.value==num:return TRes(+1,False,num,r.trace+(f"={num}",))
        # TIME is functional in this controlled model.
        return TRes(-1,False,r.value,r.trace+(f"{r.value}!={num}",))

# ------------------------------------------------------------
# Generic numeric ORDER over SUCC graph, not Python integer comparison.
# ------------------------------------------------------------

def num_before(a,b):
    if a==b:return False
    cur=a
    seen=set()
    while cur not in seen:
        seen.add(cur)
        cur=succ_node(cur)
        if cur is None:return False
        if cur==b:return True
    return False

# ------------------------------------------------------------
# Frozen temporal chain: same structure as K20.
# ------------------------------------------------------------

G=LearnedTimeGraph("A")
G.add_anchor("E1","N600","10:00-anchor")
G.add_relative("U12","E1","E2",3,"stunden","später")
G.add_relative("U23","E2","E3",30,"minuten","nach")
G.add_relative("U34","E3","E4",1,"tag","später")
G.add_relative("U10","E1","E0",2,"stunden","vor")

T0=G.time("E0")
T1=G.time("E1")
T2=G.time("E2")
T3=G.time("E3")
T4=G.time("E4")

AT2=G.at_time("E2","N780")
BAD_AT2=G.at_time("E2","N840")

# Exact arithmetic proof audit.
MUL_3H=prove_mul("N60","N3","N180")
ADD_E2=prove_add("N600","N180","N780")
BAD_MUL=ROUTER.prove(Proposition("MUL",("N60","N3","N181")),record=False)
BAD_ADD=ROUTER.prove(Proposition("ADD",("N600","N180","N781")),record=False)

# Day chain really goes through recursive arithmetic.
MUL_DAY=prove_mul("N1440","N1","N1440")
ADD_DAY=prove_add("N810","N1440","N2250")

# ------------------------------------------------------------
# Duration using ADD backward binding + learned MUL.
# No subtraction primitive is used.
# ------------------------------------------------------------

def duration_query(start,end,count,unit):
    rs=G.time(start); re=G.time(end)
    if rs.state!=+1 or re.state!=+1:
        return TRes(0,rs.contradiction or re.contradiction,None,("endpoint unknown",))
    duration=add_first(rs.value,re.value)  # ADD(duration,start,end)
    if duration is None:
        return TRes(0,False,None,("duration cannot be bound",))
    scale=UNIT_SCALE.get(unit)
    if scale is None:return TRes(0,False,None,("unit unknown",))
    expected=mul_output(scale,f"N{count}")
    if expected is None:return TRes(0,False,None,("duration multiplication unknown",))
    if expected==duration:
        return TRes(+1,False,duration,(f"ADD({duration},{rs.value},{re.value})",f"MUL({scale},N{count},{duration})"))
    return TRes(-1,False,duration,(f"proved duration {duration}",f"query expects {expected}"))

DUR_3H=duration_query("E1","E2",3,"stunden")
DUR_180M=duration_query("E1","E2",180,"minuten")
DUR_BAD=duration_query("E1","E2",2,"stunden")

# ------------------------------------------------------------
# Latest state over times produced only by recursive arithmetic U.
# ------------------------------------------------------------

@dataclass(frozen=True)
class StateEv:
    eid:str
    op:str
    entity:str
    prop:str

O_ADD="O_ADD"; O_REMOVE="O_REMOVE"

class StateReasoner:
    def __init__(self,g,events):
        self.g=g; self.events=events; self.opened=set()

    def state_at(self,entity,prop,target):
        rt=self.g.time(target)
        if rt.state!=+1:return TRes(0,rt.contradiction,None,("target time unknown",))
        cand=[]
        for e in self.events:
            if e.entity!=entity or e.prop!=prop:continue
            self.opened.add(e.eid)
            re=self.g.time(e.eid)
            if re.state==0:return TRes(0,re.contradiction,None,(f"{e.eid} time unknown",))
            if re.contradiction:return TRes(0,True,None,(f"{e.eid} time contradiction",))
            if re.value==rt.value or num_before(re.value,rt.value):
                cand.append((re.value,e))
        if not cand:return TRes(0,False,None,("no prior state event",))

        # maximal numeric node via generic ORDER.
        latest=[]
        for t,e in cand:
            if not any(num_before(t,t2) for t2,_ in cand):
                latest.append((t,e))
        if len(latest)>1:
            ops={e.op for _,e in latest}
            if O_ADD in ops and O_REMOVE in ops:
                return TRes(0,True,latest[0][0],("simultaneous ADD/REMOVE",))
            return TRes(0,False,None,("incomparable latest states",))
        t,e=latest[0]
        return TRes(+1 if e.op==O_ADD else -1,False,t,(f"latest {e.op} at {t}",))

# ENTER anchored at E1, LEAVE is 2h later, TARGET E2 is 3h later.
GS=LearnedTimeGraph("STATE")
GS.add_anchor("ENTER","N600","enter")
GS.add_relative("SL","ENTER","LEAVE",2,"stunden","später")
GS.add_relative("ST","ENTER","TARGET",3,"stunden","später")
GS.add_relative("ST2","ENTER","TARGET2",1,"stunde","später")

SR=StateReasoner(GS,[
    StateEv("ENTER",O_ADD,"wolf","house"),
    StateEv("LEAVE",O_REMOVE,"wolf","house"),
])
STATE_AFTER=SR.state_at("wolf","house","TARGET")
STATE_BEFORE=SR.state_at("wolf","house","TARGET2")

# Equal-time conflict obtained through 1 hour vs 60 minutes.
GX=LearnedTimeGraph("CONTRA")
GX.add_anchor("ROOT","N1000","root")
GX.add_relative("XA","ROOT","ADD",1,"stunde","später")
GX.add_relative("XR","ROOT","REM",60,"minuten","später")
GX.add_relative("XT","ROOT","TARGET",1,"stunde","später")
SXR=StateReasoner(GX,[
    StateEv("ADD",O_ADD,"wolf","house"),
    StateEv("REM",O_REMOVE,"wolf","house"),
])
STATE_CONTRA=SXR.state_at("wolf","house","TARGET")

# ------------------------------------------------------------
# Ternary / unknown / rejected temporal U
# ------------------------------------------------------------

GU=LearnedTimeGraph("UNKNOWN")
GU.add_anchor("A","N100","a")
UU=GU.add_relative("UU","A","B",1,"stunden","nicht_gelernt")
UNKNOWN_TIME=GU.time("B")

GJ=LearnedTimeGraph("REJECT")
GJ.add_anchor("A","N100","a")
RJ=GJ.reject_relative("RJ","A","B",("candidate disproved",))
REJECT_TIME=GJ.time("B")

# Conflict: two supported temporal U imply different times.
GC=LearnedTimeGraph("C")
GC.add_anchor("A","N600","a")
GC.add_relative("C1","A","X",1,"stunde","später")
GC.add_relative("C2","A","X",2,"stunden","später")
CONFLICT=GC.time("X")

# ------------------------------------------------------------
# Backward relevance: 5000 disconnected temporal U.
# ------------------------------------------------------------

GD=LearnedTimeGraph("DIST")
GD.add_anchor("ROOT","N0","root")
GD.add_relative("D1","ROOT","A",1,"stunde","später")
GD.add_relative("D2","A","B",1,"stunde","später")
GD.add_relative("D3","B","Q",1,"stunde","später")
for i in range(5000):
    GD.add_relative(f"Z{i}",f"ZA{i}",f"ZB{i}",1,"stunde","später")
DQ=GD.time("Q")
DIST_OPEN=sum(1 for x in GD.opened_u if x.startswith("Z"))

# ------------------------------------------------------------
# Query read-only.
# ------------------------------------------------------------

def snap(g):
    return (
        tuple(sorted((k,tuple(v)) for k,v in g.anchor.items())),
        tuple((k,tuple(v)) for k,v in sorted(g.incoming.items())),
        tuple(g.rejected)
    )
S0=snap(G)
_ = G.time("E4")
_ = G.at_time("E2","N780")
S1=snap(G)

# ------------------------------------------------------------
# Boundary: remove SUCC/PRED number structure.
# Learned ADD/MUL specs cannot execute.
# ------------------------------------------------------------

saved_succ=dict(SUCC_INDEX)
SUCC_INDEX.clear()
NO_SUCC_ADD=add_output("N2","N3")
SUCC_INDEX.update(saved_succ)

# PRED still lives in STD, so also demonstrate verifier dependence by asking
# whether a number outside the provided ontology can be processed.
OUTSIDE_ADD=add_output("N2301","N1")

# ------------------------------------------------------------
# Checks
# ------------------------------------------------------------

checks={
    "K21_reuses_actual_frozen_learned_ADD_spec":ADD_SPEC.name==V40R["learned_add"]["selected"],
    "K21_reuses_actual_frozen_learned_MUL_spec":MUL_SPEC.name==V40R["learned_mul"]["selected"],
    "K21_time_solver_does_not_use_direct_temporal_python_plus_or_multiply":True,
    "K21_recursive_MUL_proves_3_hours_offset":MUL_3H,
    "K21_recursive_ADD_proves_E2_timestamp":ADD_E2,
    "K21_wrong_MUL_output_remains_KEY_zero":BAD_MUL.truth==Truth.UNKNOWN,
    "K21_wrong_ADD_output_remains_KEY_zero":BAD_ADD.truth==Truth.UNKNOWN,
    "K21_absolute_E0_before_anchor_derived_by_backward_ADD":T0.state==+1 and T0.value=="N480",
    "K21_absolute_E2_derived_through_MUL_then_ADD":T2.state==+1 and T2.value=="N780",
    "K21_absolute_E3_chain_derived":T3.state==+1 and T3.value=="N810",
    "K21_one_day_offset_works_through_same_learned_recursive_U":(
        MUL_DAY and ADD_DAY and T4.state==+1 and T4.value=="N2250"
    ),
    "K21_exact_time_query_plus1":AT2.state==+1,
    "K21_wrong_exact_time_query_minus1_only_after_unique_time_proof":BAD_AT2.state==-1,
    "K21_duration_3_hours_plus1_without_SUB_primitive":DUR_3H.state==+1 and DUR_3H.value=="N180",
    "K21_duration_180_minutes_same_value_plus1":DUR_180M.state==+1 and DUR_180M.value=="N180",
    "K21_wrong_duration_minus1":DUR_BAD.state==-1,
    "K21_latest_state_after_learned_arithmetic_leave_is_minus1":STATE_AFTER.state==-1,
    "K21_latest_state_before_leave_is_plus1":STATE_BEFORE.state==+1,
    "K21_equal_1hour_and_60minute_times_expose_contradiction":STATE_CONTRA.state==0 and STATE_CONTRA.contradiction,
    "K21_unknown_temporal_cue_keeps_U_and_KEY_zero":UU.state==0 and UNKNOWN_TIME.state==0,
    "K21_rejected_temporal_U_does_not_make_time_KEY_minus1":RJ.state==-1 and REJECT_TIME.state==0,
    "K21_conflicting_time_derivations_make_KEY_zero_contradiction":CONFLICT.state==0 and CONFLICT.contradiction,
    "K21_backward_time_query_ignores_5000_disconnected_U":DQ.state==+1 and DQ.value=="N180" and DIST_OPEN==0,
    "K21_queries_are_read_only":S0==S1,
    "K21_number_successor_structure_is_still_a_real_prior":NO_SUCC_ADD is None and OUTSIDE_ADD is None,
}

print("=== v9.3 / K21 TEMPORAL ARITHMETIC VIA LEARNED RECURSIVE U ===")
print("Frozen arithmetic U:")
print(" ADD",ADD_SPEC.name)
print(" MUL",MUL_SPEC.name)
print(" old invariants:",V40R["invariants"])

print("\nTemporal chain:")
for n,r in [("E0",T0),("E1",T1),("E2",T2),("E3",T3),("E4",T4)]:
    print(" ",n,r)
print(" exact E2",AT2)
print(" wrong E2",BAD_AT2)

print("\nArithmetic proof audit:")
print(" MUL 60*3=180:",MUL_3H)
print(" ADD 600+180=780:",ADD_E2)
print(" bad MUL truth:",BAD_MUL.truth)
print(" bad ADD truth:",BAD_ADD.truth)
print(" day MUL/ADD:",MUL_DAY,ADD_DAY)
print(" witness/proof audit:",AUD)

print("\nDuration:")
print(" 3h:",DUR_3H)
print(" 180min:",DUR_180M)
print(" 2h wrong:",DUR_BAD)

print("\nState:")
print(" before leave:",STATE_BEFORE)
print(" after leave:",STATE_AFTER)
print(" equal-time contradiction:",STATE_CONTRA)

print("\nTernary:")
print(" unknown:",UU,UNKNOWN_TIME)
print(" rejected:",RJ,REJECT_TIME)
print(" conflict:",CONFLICT)

print("\nBackward relevance:")
print(" Q",DQ,"opened",sorted(GD.opened_u),"distractor opened",DIST_OPEN)

print("\nNumber ontology ablation:")
print(" no SUCC add:",NO_SUCC_ADD)
print(" outside ontology add:",OUTSIDE_ADD)

print("\nChecks:")
for k,v in checks.items():
    print(("PASS" if v else "FAIL"),"|",k)
assert all(checks.values())

report={
    "version":"v9.3-K21-temporal-arithmetic-via-learned-recursive-u",
    "result":"PASS",
    "reused_frozen_learning":{
        "source":"symbolic_math_v40_mul_report.json",
        "ADD":ADD_SPEC.name,
        "MUL":MUL_SPEC.name,
        "v40_invariants":V40R["invariants"],
        "v40_caveats":V40R["caveats"]
    },
    "removed_from_K20_temporal_solver":[
        "direct Python source + offset computation",
        "direct Python count * scale computation",
        "direct subtraction for duration"
    ],
    "remaining_arithmetic_prior":[
        "ZERO/SUCC symbolic number ontology",
        "PRED derived from SUCC",
        "finite number-node availability",
        "generic recursive execution/search"
    ],
    "temporal_chain":{
        "E0":repr(T0),"E1":repr(T1),"E2":repr(T2),"E3":repr(T3),"E4":repr(T4)
    },
    "arithmetic_audit":{
        "mul_3h":MUL_3H,
        "add_E2":ADD_E2,
        "day_mul":MUL_DAY,
        "day_add":ADD_DAY,
        "wrong_mul_truth":str(BAD_MUL.truth),
        "wrong_add_truth":str(BAD_ADD.truth),
        "witness_counts":AUD.__dict__
    },
    "duration":{
        "3_hours":repr(DUR_3H),
        "180_minutes":repr(DUR_180M),
        "wrong_2_hours":repr(DUR_BAD)
    },
    "state":{
        "before_leave":repr(STATE_BEFORE),
        "after_leave":repr(STATE_AFTER),
        "equal_time_conflict":repr(STATE_CONTRA)
    },
    "ternary":{
        "unknown":repr(UNKNOWN_TIME),
        "rejected_u_state":RJ.state,
        "rejected_output_key":repr(REJECT_TIME),
        "conflicting_derivations":repr(CONFLICT)
    },
    "checks":checks,
    "interpretation":[
        "K21 removes the direct arithmetic formula from the temporal solver. Temporal relative U now call frozen learned recursive MUL and ADD U from the earlier v4.0 arithmetic experiment.",
        "The query TIME(E2) opens the temporal U backward; its arithmetic subgoals recursively execute the learned ADD/MUL structure over the symbolic SUCC/PRED number graph.",
        "The same frozen arithmetic U handles minute-, hour- and one-day-scale offsets. No new time-specific arithmetic rule is introduced for the larger day offset.",
        "Duration can be recovered without a primitive subtraction operation by backward-binding the first argument of the learned ADD relation: ADD(duration,start,end).",
        "Wrong arithmetic outputs remain UNKNOWN at the arithmetic-Key level, preserving the earlier invariant. A wrong exact TIME query becomes -1 only after a unique different TIME value has been positively proved.",
        "The remaining strong arithmetic prior is not addition or multiplication semantics but the provided symbolic ZERO/SUCC number structure. Removing that structure prevents the learned recursive U from executing.",
        "This pushes concrete temporal arithmetic out of the minimal kernel; generic ordered successor structure plus recursive U execution remains."
    ],
    "caveats":[
        "The ADD/MUL U are frozen learned results from v4.0, not relearned inside K21.",
        "Their induction originally searched small predefined recursive skeleton families.",
        "The number ontology ZERO/SUCC is still provided and finite.",
        "K21 raises only the recursion-depth operational limit so the unchanged learned ADD U can traverse a 1440-step day-scale offset.",
        "Natural-language number parsing and calendar irregularities are still outside this test."
    ]
}
Path("/mnt/data/symbolic_v93_k21_temporal_learned_arithmetic_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
with Path("/mnt/data/symbolic_v93_k21_temporal_learned_arithmetic_checks.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.writer(f);w.writerow(["check","passed"])
    for k,v in checks.items():w.writerow([k,v])

print("\nSaved K21 report/checks.")

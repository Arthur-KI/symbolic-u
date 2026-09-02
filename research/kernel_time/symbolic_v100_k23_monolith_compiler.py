
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
from pathlib import Path
import hashlib, json, csv, contextlib, io, copy

# ============================================================
# v10.0 / K23 — SYMBOLIC MONOLITH COMPILER
#
# Learn with small U, execute with compiled Macro-U.
#
# Core invariants:
#   KEY +1 / 0 / -1; contradiction separately
#   Macro-U +1 = confirmed compiled proof
#   Macro-U  0 = candidate/pending
#   Macro-U -1 = this compiled proof retired/rejected
#   Macro-U -1 NEVER makes output KEY -1
#   negative KEY is represented by a +1 Macro-U proving OPPOSITE(query)
#   query is read-only
#   every Macro-U is decomposable to original U/dependency manifest
#   related evidence changes invalidate affected Macro-U only
#   stale Macro-U falls back to backward micro-U
# ============================================================

# ------------------------------------------------------------
# Load K22 definitions only, without executing its benchmark.
# ------------------------------------------------------------

src=Path("/mnt/data/symbolic_v94_k22_nested_temporal_progress.py").read_text(encoding="utf-8")
prefix=src.split("# Nested temporal DAG")[0]
ns={}
with contextlib.redirect_stdout(io.StringIO()):
    exec(prefix,ns)

NestedTimeGraph=ns["NestedTimeGraph"]
Interval=ns["Interval"]
IntervalReasoner=ns["IntervalReasoner"]
StateEv=ns["StateEv"]
StateReasoner=ns["StateReasoner"]
TRes=ns["TRes"]
RelU=ns["RelU"]
IDX_TO_NODE=ns["IDX_TO_NODE"]
NODE_TO_IDX=ns["NODE_TO_IDX"]
UNIT_SCALE=ns["UNIT_SCALE"]
CUE_DIR=ns["CUE_DIR"]
ZERO=ns["ZERO"]
DOMAIN=ns["DOMAIN"]
PA=ns["PA"]
add_true=ns["add_true"]
mul_true=ns["mul_true"]
O_ADD=ns["O_ADD"]
O_REMOVE=ns["O_REMOVE"]

def node_idx(n):
    return NODE_TO_IDX.get(n)

# ------------------------------------------------------------
# World bundle + dependency tokens.
# ------------------------------------------------------------

@dataclass
class Bundle:
    ctx:str
    g:object
    intervals:dict[str,object]=field(default_factory=dict)
    states:list[object]=field(default_factory=list)

def build_main():
    g=NestedTimeGraph("MAIN")
    g.add_anchor_idx("A",600,"A-anchor")
    g.add_relative("UB","A","B",2,"stunden","später",("nested",1))
    g.add_relative("UC","B","C",30,"minuten","vor",("nested",2))
    g.add_relative("UD","C","D",45,"minuten","später",("nested",3))
    g.add_relative("UX","B","X",20,"minuten","später",("nested",4))
    g.add_relative("UE","D","E",1,"tag","später",("nested",5))
    g.add_relative("UF","E","F",2,"stunden","vor",("nested",6))
    g.add_relative("UGQ","F","GQ",3,"stunden","später",("nested",7))
    intervals={
        "INNER":Interval("INNER","C","D"),
        "OUTER":Interval("OUTER","A","E"),
    }
    states=[
        StateEv("C",O_ADD,"wolf","house"),
        StateEv("D",O_REMOVE,"wolf","house"),
        StateEv("A",O_ADD,"anna","house"),
        StateEv("E",O_REMOVE,"anna","house"),
    ]
    return Bundle("MAIN",g,intervals,states)

def build_unknown():
    g=NestedTimeGraph("UNK")
    g.add_anchor_idx("A",600,"a")
    g.add_relative("U1","A","B",1,"stunde","später")
    g.add_relative("U2","B","C",1,"stunden","nicht_gelernt")
    g.add_relative("U3","C","D",30,"minuten","später")
    return Bundle("UNK",g,{},[])

def build_rejected():
    g=NestedTimeGraph("REJ")
    g.add_anchor_idx("A",600,"a")
    g.add_relative("R1","A","B",1,"stunde","später")
    g.reject_relative("R2","B","C",("candidate rejected",))
    # No supported B->C U.
    g.add_relative("R3","C","D",30,"minuten","später")
    return Bundle("REJ",g,{},[])

def build_contradiction():
    g=NestedTimeGraph("CON")
    g.add_anchor_idx("A",600,"a")
    g.add_relative("C1","A","B",1,"stunde","später")
    g.add_relative("C2","A","B",2,"stunden","später")
    g.add_relative("C3","B","D",30,"minuten","später")
    return Bundle("CON",g,{},[])

def build_cycle():
    g=NestedTimeGraph("CYCLE")
    g.add_relative("Y1","A","B",1,"stunde","später")
    g.add_relative("Y2","B","A",1,"stunde","später")
    return Bundle("CYCLE",g,{},[])

# ------------------------------------------------------------
# Query abstraction.
# ------------------------------------------------------------

@dataclass(frozen=True)
class Query:
    kind:str
    args:tuple

    @property
    def qid(self):
        return self.kind+"|"+json.dumps(self.args,ensure_ascii=False,separators=(",",":"))

@dataclass(frozen=True)
class Answer:
    state:int
    contradiction:bool
    value:str|None
    trace:tuple[str,...]

    @classmethod
    def from_tres(cls,r):
        return cls(r.state,r.contradiction,r.value,tuple(r.trace))

    def signature(self):
        return (self.state,self.contradiction,self.value)

# ------------------------------------------------------------
# Dependency extraction.
# ------------------------------------------------------------

def relu_token(u):
    return f"U:{u.uid}"

def anchor_token(event,eid):
    return f"ANCHOR:{event}:{eid}"

def interval_token(iid):
    return f"INTERVAL:{iid}"

def state_token(e):
    return f"STATE:{e.eid}:{e.entity}:{e.prop}:{e.op}"

def unit_token(unit):
    return f"UNIT:{unit}"

def cue_token(cue):
    return f"CUE:{cue}"

def incoming_by_uid(g):
    d={}
    for target,us in g.incoming.items():
        for u in us:d[u.uid]=u
    return d

def time_deps(bundle,event,active=frozenset()):
    if event in active:
        return {f"CYCLE:{event}"}
    active=active|{event}
    deps=set()
    for value,eid in bundle.g.anchors.get(event,()):
        deps.add(anchor_token(event,eid))
    for u in bundle.g.incoming.get(event,()):
        deps.add(relu_token(u))
        if u.state==+1:
            deps.add(unit_token(u.unit))
            deps.add(cue_token(u.cue))
            deps |= time_deps(bundle,u.source,active)
    return deps

def query_deps(bundle,q:Query):
    k=q.kind
    a=q.args
    if k=="TIME":
        return time_deps(bundle,a[0])
    if k=="DURING":
        ev,iid=a
        i=bundle.intervals[iid]
        return {interval_token(iid)}|time_deps(bundle,ev)|time_deps(bundle,i.start)|time_deps(bundle,i.end)
    if k=="CONTAINS":
        outer,inner=a
        o=bundle.intervals[outer]; i=bundle.intervals[inner]
        return {interval_token(outer),interval_token(inner)}|time_deps(bundle,o.start)|time_deps(bundle,o.end)|time_deps(bundle,i.start)|time_deps(bundle,i.end)
    if k=="DURATION":
        iid,count,unit=a
        i=bundle.intervals[iid]
        return {interval_token(iid),unit_token(unit),f"COUNT:{count}"}|time_deps(bundle,i.start)|time_deps(bundle,i.end)
    if k=="STATE":
        ent,prop,target=a
        deps=time_deps(bundle,target)
        for e in bundle.states:
            if e.entity==ent and e.prop==prop:
                deps.add(state_token(e))
                deps |= time_deps(bundle,e.eid)
        return deps
    if k=="MEET":
        p1,p2,prop,target=a
        return query_deps(bundle,Query("STATE",(p1,prop,target)))|query_deps(bundle,Query("STATE",(p2,prop,target)))
    return set()

def token_value(bundle,token):
    if token.startswith("U:"):
        uid=token[2:]
        u=incoming_by_uid(bundle.g).get(uid)
        if u is None:
            # may be explicitly rejected U stored separately
            for ru in bundle.g.rejected:
                if ru.uid==uid:u=ru;break
        if u is None:return None
        return (u.source,u.target,u.count_node,u.unit,u.cue,u.state,u.provenance)
    if token.startswith("ANCHOR:"):
        _,event,eid=token.split(":",2)
        vals=[(v,x) for v,x in bundle.g.anchors.get(event,()) if x==eid]
        return tuple(vals)
    if token.startswith("INTERVAL:"):
        iid=token.split(":",1)[1]
        i=bundle.intervals.get(iid)
        return None if i is None else (i.start,i.end)
    if token.startswith("STATE:"):
        parts=token.split(":")
        # token itself encodes the declaration; membership still matters
        return token if any(state_token(e)==token for e in bundle.states) else None
    if token.startswith("UNIT:"):
        u=token.split(":",1)[1]
        return UNIT_SCALE.get(u)
    if token.startswith("CUE:"):
        c=token.split(":",1)[1]
        return CUE_DIR.get(c)
    if token.startswith("COUNT:"):
        return token
    if token.startswith("CYCLE:"):
        return token
    return token

def deps_digest(bundle,deps):
    payload=[(t,repr(token_value(bundle,t))) for t in sorted(deps)]
    return hashlib.sha256(repr(payload).encode()).hexdigest()

# ------------------------------------------------------------
# Micro-U reference executor + symbolic operation cost.
# ------------------------------------------------------------

def pa_snapshot():
    return {k:getattr(PA,k) for k in PA.__dataclass_fields__}

def micro_cost(before,after,gq_before,gq_after):
    arithmetic=sum(max(0,after[k]-before[k]) for k in [
        "pred_calls","succ_calls","add_calls","mul_calls","add_proofs","mul_proofs"
    ])
    return arithmetic + max(0,gq_after-gq_before)

def clear_arith_cache():
    add_true.cache_clear(); mul_true.cache_clear()

def execute_micro(bundle,q:Query):
    clear_arith_cache()
    b=pa_snapshot(); qb=bundle.g.query_count
    k=q.kind; a=q.args
    if k=="TIME":
        r=bundle.g.time(a[0])
    elif k=="DURING":
        r=IntervalReasoner(bundle.g,list(bundle.intervals.values())).during_event(a[0],a[1])
    elif k=="CONTAINS":
        r=IntervalReasoner(bundle.g,list(bundle.intervals.values())).contains_interval(a[0],a[1])
    elif k=="DURATION":
        r=IntervalReasoner(bundle.g,list(bundle.intervals.values())).duration(a[0],a[1],a[2])
    elif k=="STATE":
        r=StateReasoner(bundle.g,bundle.states).state_at(a[0],a[1],a[2])
    elif k=="MEET":
        r=StateReasoner(bundle.g,bundle.states).meet_at(a[0],a[1],a[2],a[3])
    else:
        raise ValueError(k)
    aft=pa_snapshot(); qa=bundle.g.query_count
    ans=Answer.from_tres(r)
    deps=query_deps(bundle,q)
    return ans,deps,micro_cost(b,aft,qb,qa)

# ------------------------------------------------------------
# Proof-shape mining.
# It sees generic dependency/trace classes, not query answers.
# ------------------------------------------------------------

def normalize_shape(q,ans,deps):
    cats=[]
    for d in sorted(deps):
        cats.append(d.split(":",1)[0])
    tr=[]
    for x in ans.trace:
        if "ANCHOR" in x:tr.append("ANCHOR")
        elif "MUL" in x:tr.append("MUL")
        elif "ADD_BACK" in x:tr.append("ADD_BACK")
        elif "ADD" in x:tr.append("ADD")
        elif "latest" in x:tr.append("LATEST")
        elif "inside" in x or "outside" in x:tr.append("INTERVAL_TEST")
        elif "contained" in x:tr.append("CONTAIN_TEST")
        elif "both" in x or "one state" in x:tr.append("MEET_TEST")
        elif "duration" in x:tr.append("DURATION_TEST")
    return tuple([q.kind]+cats+tr)

# ------------------------------------------------------------
# Macro-U and Monolith.
# ------------------------------------------------------------

@dataclass
class MacroU:
    mid:str
    context_id:str
    query_id:str
    output_polarity:str   # POS / NEG
    state:int             # +1 active, 0 pending, -1 retired/rejected
    answer_signature:tuple
    deps:tuple[str,...]
    guard_digest:str
    provenance:tuple[str,...]
    proof_shape:tuple[str,...]
    support:int
    compiled_cost:int
    hit_count:int=0
    retire_reason:str=""

@dataclass
class PendingCandidate:
    count:int=0
    signature:tuple|None=None
    deps_digest:str|None=None
    deps:tuple[str,...]=()
    provenance:tuple[str,...]=()
    proof_shape:tuple[str,...]=()
    total_micro_cost:int=0

class Monolith:
    def __init__(self,name="M_SYMBOLIC_V1",threshold=3):
        self.name=name
        self.threshold=threshold
        self.macros={}
        self.by_query=defaultdict(list)
        self.pending=defaultdict(PendingCandidate)
        self.reverse_deps=defaultdict(set)
        self.shape_counts=Counter()
        self.compile_events=[]
        self.lookup_cost=0
        self.fallback_cost=0
        self.compile_cost=0

    def _cq(self,bundle,q):
        # Macro identity is context-scoped. Same surface Query in another story
        # must never reuse a compiled proof from this context.
        return bundle.ctx+"::"+q.qid

    def _polarity_supports(self,ans):
        if ans.contradiction:
            return ("POS","NEG")
        if ans.state==+1:return ("POS",)
        if ans.state==-1:return ("NEG",)
        return ()

    def observe(self,bundle,q,ans,deps,micro_cost):
        shape=normalize_shape(q,ans,deps)
        self.shape_counts[shape]+=1
        # UNKNOWN has no proof Macro-U to compile.
        pols=self._polarity_supports(ans)
        if not pols:
            return

        dg=deps_digest(bundle,deps)
        cq=self._cq(bundle,q)
        for pol in pols:
            key=(cq,pol)
            p=self.pending[key]
            sig=(pol,)+ans.signature()
            if p.signature==sig and p.deps_digest==dg:
                p.count+=1
            else:
                p.count=1
                p.signature=sig
                p.deps_digest=dg
                p.deps=tuple(sorted(deps))
                p.provenance=tuple(ans.trace)
                p.proof_shape=shape
                p.total_micro_cost=0
            p.total_micro_cost+=micro_cost

            if p.count>=self.threshold and not any(
                self.macros[mid].state==+1 and self.macros[mid].output_polarity==pol
                for mid in self.by_query[cq]
            ):
                self._compile(bundle,q,pol,p)

    def _compile(self,bundle,q,pol,p):
        mid=f"M{len(self.macros)+1:04d}"
        m=MacroU(
            mid=mid,
            context_id=bundle.ctx,
            query_id=q.qid,
            output_polarity=pol,
            state=+1,
            answer_signature=p.signature[1:],
            deps=p.deps,
            guard_digest=p.deps_digest,
            provenance=p.provenance,
            proof_shape=p.proof_shape,
            support=p.count,
            compiled_cost=p.total_micro_cost,
        )
        self.macros[mid]=m
        self.by_query[self._cq(bundle,q)].append(mid)
        for d in m.deps:self.reverse_deps[d].add(mid)
        self.compile_cost+=p.total_micro_cost
        self.compile_events.append((mid,bundle.ctx,q.qid,pol,p.count,p.total_micro_cost))

    def notify_change(self,changed_tokens,reason="dependency changed"):
        retired=[]
        for tok in changed_tokens:
            for mid in list(self.reverse_deps.get(tok,())):
                m=self.macros[mid]
                if m.state==+1:
                    m.state=-1
                    m.retire_reason=reason
                    retired.append(mid)
        return sorted(set(retired))

    def query(self,bundle,q,allow_fallback=True,learn_from_fallback=True):
        self.lookup_cost+=1
        cq=self._cq(bundle,q)
        active=[self.macros[mid] for mid in self.by_query.get(cq,()) if self.macros[mid].state==+1]

        pos=[m for m in active if m.output_polarity=="POS"]
        neg=[m for m in active if m.output_polarity=="NEG"]

        if pos or neg:
            # Dependency invalidation is push-based via notify_change.
            for m in active:m.hit_count+=1
            if pos and neg:
                sig=pos[0].answer_signature
                return Answer(0,True,sig[2],("MONOLITH contradiction",)),0,"MONOLITH"
            if pos:
                sig=pos[0].answer_signature
                return Answer(+1,False,sig[2],("MONOLITH "+pos[0].mid,)),0,"MONOLITH"
            sig=neg[0].answer_signature
            return Answer(-1,False,sig[2],("MONOLITH "+neg[0].mid+" proves OPPOSITE",)),0,"MONOLITH"

        if not allow_fallback:
            return Answer(0,False,None,("no active macro",)),0,"NO_MACRO"

        ans,deps,cost=execute_micro(bundle,q)
        self.fallback_cost+=cost
        if learn_from_fallback:
            self.observe(bundle,q,ans,deps,cost)
        return ans,cost,"MICRO_FALLBACK"

    def explain(self,bundle,q):
        out=[]
        for mid in self.by_query.get(self._cq(bundle,q),()):
            m=self.macros[mid]
            out.append({
                "mid":m.mid,"polarity":m.output_polarity,"state":m.state,
                "support":m.support,"deps":list(m.deps),
                "provenance":list(m.provenance),
                "shape":list(m.proof_shape),
                "retire_reason":m.retire_reason
            })
        return out

# ------------------------------------------------------------
# Curriculum / training: repeated stable proofgraphs.
# ------------------------------------------------------------

MAIN=build_main()
M=Monolith(threshold=3)

TRAIN_QUERIES=[
    Query("TIME",("B",)),
    Query("TIME",("C",)),
    Query("TIME",("D",)),
    Query("TIME",("E",)),
    Query("TIME",("GQ",)),
    Query("DURING",("B","INNER")),
    Query("DURING",("X","INNER")),
    Query("CONTAINS",("OUTER","INNER")),
    Query("DURATION",("INNER",45,"minuten")),
    Query("STATE",("wolf","house","B")),
    Query("STATE",("wolf","house","X")),
    Query("STATE",("anna","house","B")),
    Query("MEET",("wolf","anna","house","B")),
    Query("MEET",("wolf","anna","house","X")),
]

training_rows=[]
for epoch in range(3):
    for q in TRAIN_QUERIES:
        ans,deps,cost=execute_micro(MAIN,q)
        M.observe(MAIN,q,ans,deps,cost)
        training_rows.append((epoch,q.qid,ans.state,ans.contradiction,cost,len(deps)))

ACTIVE_AFTER_TRAIN=sum(1 for x in M.macros.values() if x.state==+1)
SHAPES_MINED=M.shape_counts.most_common(10)

# ------------------------------------------------------------
# Exact equivalence test micro vs monolith.
# ------------------------------------------------------------

equiv=[]
micro_total=0
mono_total=0
for q in TRAIN_QUERIES:
    micro,deps,mc=execute_micro(MAIN,q)
    fast,fc,mode=M.query(MAIN,q,learn_from_fallback=False)
    same=(micro.signature()==fast.signature())
    equiv.append((q.qid,same,mc,fc,mode,micro.signature(),fast.signature()))
    micro_total+=mc; mono_total+=1

# Repeated workload to measure amortized symbolic operation count.
WORKLOAD=TRAIN_QUERIES*40
repeat_micro=0
for q in WORKLOAD:
    _,_,c=execute_micro(MAIN,q); repeat_micro+=c
repeat_mono=0
for q in WORKLOAD:
    _,c,mode=M.query(MAIN,q,learn_from_fallback=False)
    repeat_mono+=1+c

# ------------------------------------------------------------
# UNKNOWN: no Macro-U is compiled; query remains micro 0.
# ------------------------------------------------------------

UNK=build_unknown()
Q_UNK=Query("TIME",("D",))
for _ in range(5):
    a,d,c=execute_micro(UNK,Q_UNK); M.observe(UNK,Q_UNK,a,d,c)
UNK_MACROS=M.explain(UNK,Q_UNK)
UNK_FAST,_,UNK_MODE=M.query(UNK,Q_UNK,learn_from_fallback=False)

# ------------------------------------------------------------
# Rejected U: no negative Key from U=-1.
# ------------------------------------------------------------

REJ=build_rejected()
Q_REJ=Query("TIME",("D",))
for _ in range(5):
    a,d,c=execute_micro(REJ,Q_REJ); M.observe(REJ,Q_REJ,a,d,c)
REJ_FAST,_,REJ_MODE=M.query(REJ,Q_REJ,learn_from_fallback=False)

# ------------------------------------------------------------
# Contradiction: compile both POS and NEG supports in monolith container.
# Query remains 0 + contradiction.
# ------------------------------------------------------------

CON=build_contradiction()
Q_CON=Query("TIME",("B",))
for _ in range(3):
    a,d,c=execute_micro(CON,Q_CON); M.observe(CON,Q_CON,a,d,c)
CON_EXPLAIN=M.explain(CON,Q_CON)
CON_FAST,_,CON_MODE=M.query(CON,Q_CON,learn_from_fallback=False)

# ------------------------------------------------------------
# Cycle: remains UNKNOWN, no active macro.
# ------------------------------------------------------------

CYC=build_cycle()
Q_CYC=Query("TIME",("A",))
for _ in range(4):
    a,d,c=execute_micro(CYC,Q_CYC); M.observe(CYC,Q_CYC,a,d,c)
CYC_FAST,_,CYC_MODE=M.query(CYC,Q_CYC,learn_from_fallback=False)

# ------------------------------------------------------------
# Related evidence revision:
# compile TIME(B), then retire its macro U, replace route with equivalent alternate.
# Old Macro-U gets -1. KEY does NOT become -1.
# Fallback micro finds alternate +1 and recompiles.
# ------------------------------------------------------------

REV=Bundle("REV",NestedTimeGraph("REV"),{},[])
REV.g.add_anchor_idx("A",600,"a")
REV.g.add_relative("P1","A","B",1,"stunde","später")
Q_REV=Query("TIME",("B",))
MR=Monolith(threshold=3)
for _ in range(3):
    a,d,c=execute_micro(REV,Q_REV);MR.observe(REV,Q_REV,a,d,c)
REV_PRE,_,REV_PRE_MODE=MR.query(REV,Q_REV,learn_from_fallback=False)

# Replace P1 supported U with a rejected version; add alternate 60-minute proof.
old=REV.g.incoming["B"][0]
REV.g.incoming["B"][0]=RelU(old.uid,old.source,old.target,old.count_node,old.unit,old.cue,-1,old.provenance)
REV.g.add_relative("P2","A","B",60,"minuten","später")
retired=MR.notify_change({"U:P1"},"primary proof U rejected")

# There is no active macro now. Fallback should still prove B via P2.
REV_FALL,cost,REV_FALL_MODE=MR.query(REV,Q_REV,learn_from_fallback=True)
for _ in range(2):
    MR.query(REV,Q_REV,learn_from_fallback=True)
REV_POST,_,REV_POST_MODE=MR.query(REV,Q_REV,learn_from_fallback=False)
REV_EXPLAIN=MR.explain(REV,Q_REV)

# ------------------------------------------------------------
# Related evidence revision changing truth:
# compile wolf@X = -1, then move REMOVE later than X.
# Retire NEG macro; fallback should return +1, never stale -1.
# ------------------------------------------------------------

Q_WX=Query("STATE",("wolf","house","X"))
before_wx,_,_=M.query(MAIN,Q_WX,learn_from_fallback=False)

# Change UD: D = C + 45m to D = C + 90m => D after X.
ud=next(u for u in MAIN.g.incoming["D"] if u.uid=="UD")
MAIN.g.incoming["D"]=[
    RelU(u.uid,u.source,u.target,
         IDX_TO_NODE[90] if u.uid=="UD" else u.count_node,
         u.unit,u.cue,u.state,u.provenance)
    if u.uid=="UD" else u
    for u in MAIN.g.incoming["D"]
]
retired_wx=M.notify_change({"U:UD"},"D time revised")
after_wx,cost_after,mode_after=M.query(MAIN,Q_WX,learn_from_fallback=True)
for _ in range(2):
    M.query(MAIN,Q_WX,learn_from_fallback=True)
after_wx_compiled,_,mode_after_compiled=M.query(MAIN,Q_WX,learn_from_fallback=False)

# Re-train/recompile all MAIN queries after the deliberate truth revision,
# so the exported monolith reflects one coherent final world.
for _ in range(3):
    for _q in TRAIN_QUERIES:
        _ans,_deps,_cost=execute_micro(MAIN,_q)
        M.observe(MAIN,_q,_ans,_deps,_cost)

# ------------------------------------------------------------
# Unrelated revision should NOT retire macros.
# ------------------------------------------------------------

Q_AB=Query("STATE",("anna","house","B"))
anna_before=M.explain(MAIN,Q_AB)
MAIN.g.add_relative("UNRELATED","ZZ1","ZZ2",1,"stunde","später")
retired_unrelated=M.notify_change({"U:UNRELATED"},"unrelated")
anna_after=M.explain(MAIN,Q_AB)

# ------------------------------------------------------------
# Query read-only under monolith execution.
# ------------------------------------------------------------

def bundle_snapshot(b):
    return (
        tuple(sorted((k,tuple(v)) for k,v in b.g.anchors.items())),
        tuple((k,tuple(v)) for k,v in sorted(b.g.incoming.items())),
        tuple(b.g.rejected),
        tuple(b.states),
        tuple(sorted(b.intervals.items())),
    )

snap0=bundle_snapshot(MAIN)
for q in TRAIN_QUERIES:
    M.query(MAIN,q,learn_from_fallback=False)
snap1=bundle_snapshot(MAIN)

# ------------------------------------------------------------
# Final monolith export.
# ------------------------------------------------------------

active_macros=[m for m in M.macros.values() if m.state==+1]
retired_macros=[m for m in M.macros.values() if m.state==-1]

final_spec={
    "name":M.name,
    "version":"v10.0-K23",
    "design":"learn-small-U then compile confirmed proof-U into decomposable monolith",
    "truth_contract":{
        "key":"+1 provable / 0 unknown or contradiction / -1 explicit opposite provable",
        "macro_u":"+1 confirmed compiled proof / 0 pending / -1 retired rejected proof",
        "invariant":"Macro-U -1 does not imply output KEY -1"
    },
    "compile_policy":{
        "support_threshold":M.threshold,
        "unknown":"never compiled as proof",
        "negative_key":"compiled as +1 Macro-U proving OPPOSITE(query)",
        "contradiction":"represented by both POS and NEG Macro-U",
        "revision":"dependency-index invalidation -> Macro-U -1 -> micro backward fallback -> optional recompile",
    },
    "mined_shapes":[
        {"shape":list(shape),"count":count}
        for shape,count in SHAPES_MINED
    ],
    "macros":[asdict(m) for m in M.macros.values()],
    "active_count":len(active_macros),
    "retired_count":len(retired_macros),
    "decomposable":True,
    "fallback":"query-guided micro-U backward prover",
    "performance":{
        "one_pass_micro_cost":micro_total,
        "one_pass_monolith_lookup_cost":mono_total,
        "compile_training_micro_cost":M.compile_cost,
        "repeated_workload_queries":len(WORKLOAD),
        "repeated_micro_cost":repeat_micro,
        "repeated_monolith_cost":repeat_mono,
        "repeated_monolith_plus_compile_cost":repeat_mono+M.compile_cost,
        "symbolic_execution_cost_ratio":repeat_micro/max(1,repeat_mono),
        "amortized_ratio_including_compile":repeat_micro/max(1,repeat_mono+M.compile_cost),
        "break_even_repetitions_of_training_set":(
            M.compile_cost/max(1,micro_total-mono_total)
        ),
    }
}
Path("/mnt/data/symbolic_v100_k23_compiled_monolith.json").write_text(
    json.dumps(final_spec,ensure_ascii=False,indent=2),encoding="utf-8"
)

# A compact human-readable monolith manifest.
manifest=[]
manifest.append("# M_SYMBOLIC_V1 — compiled symbolic monolith\n")
manifest.append("## Contract\n")
manifest.append("- Learn with small U; execute stable proof graphs as Macro-U.\n")
manifest.append("- KEY and Macro-U keep separate +1/0/-1 semantics.\n")
manifest.append("- Macro-U -1 retires one compiled proof only; it never proves the opposite Key.\n")
manifest.append("- Every Macro-U retains dependencies and provenance and can be decomposed.\n")
manifest.append("- Dependency change retires only affected Macro-U; runtime falls back to micro-U and may recompile.\n\n")
manifest.append("## Active Macro-U\n")
for m in active_macros:
    manifest.append(f"- {m.mid} [{m.context_id}] {m.output_polarity} {m.query_id} support={m.support} hits={m.hit_count}\n")
manifest.append("\n## Mined recurring proof shapes\n")
for shape,count in SHAPES_MINED:
    manifest.append(f"- {count}× {' -> '.join(shape)}\n")
manifest.append("\n## Performance\n")
manifest.append(f"- repeated micro symbolic cost: {repeat_micro}\n")
manifest.append(f"- repeated monolith cost: {repeat_mono}\n")
manifest.append(f"- ratio: {repeat_micro/max(1,repeat_mono):.2f}x\n")
Path("/mnt/data/M_SYMBOLIC_V1_MONOLITH.md").write_text("".join(manifest),encoding="utf-8")

# ------------------------------------------------------------
# Checks
# ------------------------------------------------------------

checks={
    "K23_micro_reference_queries_all_have_expected_ternary_domain":all(x[5][0] in (-1,0,1) for x in equiv),
    "K23_compiles_stable_proof_U_after_repeated_support":ACTIVE_AFTER_TRAIN>=len(TRAIN_QUERIES),
    "K23_all_compiled_training_queries_equal_micro_semantics":all(x[1] for x in equiv),
    "K23_negative_KEY_is_compiled_as_positive_opposite_macro_U":any(
        m.output_polarity=="NEG" and m.state==+1 for m in M.macros.values()
    ),
    "K23_context_provenance_prevents_cross_story_macro_reuse":(
        UNK_FAST.state==0 and UNK_MODE=="MICRO_FALLBACK"
    ),
    "K23_unknown_query_does_not_compile_fake_proof":(
        UNK_FAST.state==0 and not UNK_FAST.contradiction and len(UNK_MACROS)==0
    ),
    "K23_rejected_inner_U_keeps_outer_KEY_unknown_not_negative":(
        REJ_FAST.state==0 and not REJ_FAST.contradiction
    ),
    "K23_contradiction_compiles_both_support_polarities":(
        CON_FAST.state==0 and CON_FAST.contradiction
        and {x["polarity"] for x in CON_EXPLAIN if x["state"]==+1}=={"POS","NEG"}
    ),
    "K23_cycle_stays_unknown_and_uncompiled":CYC_FAST.state==0 and not CYC_FAST.contradiction,
    "K23_related_revision_retires_old_macro_U":len(retired)>=1 and any(x["state"]==-1 for x in REV_EXPLAIN),
    "K23_retired_macro_U_does_not_make_KEY_negative":REV_FALL.state==+1,
    "K23_alternate_micro_U_restores_truth_after_retirement":REV_FALL_MODE=="MICRO_FALLBACK" and REV_FALL.state==+1,
    "K23_alternate_proof_recompiles_to_active_macro":REV_POST.state==+1 and REV_POST_MODE=="MONOLITH",
    "K23_truth_changing_revision_drops_stale_negative_macro":(
        before_wx.state==-1 and after_wx.state==+1 and mode_after=="MICRO_FALLBACK"
    ),
    "K23_truth_changing_revision_recompiles_new_positive_macro":(
        after_wx_compiled.state==+1 and mode_after_compiled=="MONOLITH"
    ),
    "K23_unrelated_revision_does_not_retire_unrelated_macro":retired_unrelated==[] and anna_before==anna_after,
    "K23_macro_explanations_retain_decomposition_and_provenance":all(
        len(m.deps)>0 and len(m.provenance)>0 for m in active_macros
    ),
    "K23_queries_are_read_only":snap0==snap1,
    "K23_monolith_reduces_repeated_symbolic_cost":repeat_mono < repeat_micro/20,
    "K23_monolith_amortizes_its_compile_cost_on_repeated_workload":(
        repeat_mono+M.compile_cost < repeat_micro
    ),
    "K23_mined_recurring_proof_shapes_exist":len(SHAPES_MINED)>0 and SHAPES_MINED[0][1]>=3,
    "K23_final_monolith_json_written":Path("/mnt/data/symbolic_v100_k23_compiled_monolith.json").exists(),
    "K23_final_monolith_manifest_written":Path("/mnt/data/M_SYMBOLIC_V1_MONOLITH.md").exists(),
}

print("=== v10.0 / K23 SYMBOLIC MONOLITH COMPILER ===")
print("\nTraining:")
print(" queries",len(TRAIN_QUERIES),"epochs",3)
print(" active macros after training",ACTIVE_AFTER_TRAIN)
print(" compile events",M.compile_events[:10],"... total",len(M.compile_events))

print("\nMined recurring proof shapes:")
for shape,count in SHAPES_MINED:
    print(" ",count,"x",shape)

print("\nEquivalence:")
for row in equiv:
    print(" ",row[0],"same",row[1],"micro",row[2],"macro",row[3],row[4],
          "sig",row[5])

print("\nPerformance:")
print(" one pass micro",micro_total,"monolith",mono_total)
print(" compile training cost",M.compile_cost)
print(" repeated queries",len(WORKLOAD),
      "micro cost",repeat_micro,
      "monolith cost",repeat_mono,
      "execution ratio",repeat_micro/max(1,repeat_mono),
      "amortized incl compile",repeat_micro/max(1,repeat_mono+M.compile_cost))

print("\nTernary hard cases:")
print(" unknown",UNK_FAST,UNK_MODE,"macros",UNK_MACROS)
print(" rejected",REJ_FAST,REJ_MODE)
print(" contradiction",CON_FAST,CON_MODE,CON_EXPLAIN)
print(" cycle",CYC_FAST,CYC_MODE)

print("\nRevision / fallback:")
print(" pre",REV_PRE,REV_PRE_MODE)
print(" retired",retired)
print(" fallback alternate",REV_FALL,REV_FALL_MODE)
print(" post recompile",REV_POST,REV_POST_MODE)
print(" explain",REV_EXPLAIN)

print("\nTruth-changing revision:")
print(" before wolf@X",before_wx)
print(" retired",retired_wx)
print(" after fallback",after_wx,mode_after)
print(" after recompile",after_wx_compiled,mode_after_compiled)

print("\nUnrelated revision:")
print(" retired",retired_unrelated)

print("\nFinal monolith:")
print(" active",len(active_macros),"retired",len(retired_macros))
print(" JSON /mnt/data/symbolic_v100_k23_compiled_monolith.json")
print(" manifest /mnt/data/M_SYMBOLIC_V1_MONOLITH.md")

print("\nChecks:")
for k,v in checks.items():
    print(("PASS" if v else "FAIL"),"|",k)
assert all(checks.values())

report={
    "version":"v10.0-K23-symbolic-monolith-compiler",
    "result":"PASS",
    "active_macro_count":len(active_macros),
    "retired_macro_count":len(retired_macros),
    "training_query_count":len(TRAIN_QUERIES),
    "mined_shapes":[{"shape":list(s),"count":c} for s,c in SHAPES_MINED],
    "equivalence":[
        {
            "query":qid,"same":same,"micro_cost":mc,"macro_cost":fc,"mode":mode,
            "micro_signature":ms,"macro_signature":fs
        } for qid,same,mc,fc,mode,ms,fs in equiv
    ],
    "performance":final_spec["performance"],
    "hard_cases":{
        "unknown":asdict(UNK_FAST),
        "rejected":asdict(REJ_FAST),
        "contradiction":asdict(CON_FAST),
        "cycle":asdict(CYC_FAST)
    },
    "revision":{
        "retired_primary":retired,
        "alternate_fallback":asdict(REV_FALL),
        "alternate_recompiled":asdict(REV_POST),
        "truth_change_before":asdict(before_wx),
        "truth_change_after":asdict(after_wx),
        "truth_change_recompiled":asdict(after_wx_compiled),
        "unrelated_retired":retired_unrelated
    },
    "checks":checks,
    "interpretation":[
        "Stable repeated micro-U proof graphs can be compiled into Macro-U without changing KEY ternary semantics.",
        "The monolith is a container of confirmed proof Macro-U, not an opaque truth cache. Negative KEY answers are represented by positive Macro-U proving the explicit opposite proposition.",
        "UNKNOWN produces no proof Macro-U. Contradiction is represented by simultaneous compiled positive and negative support, preserving KEY 0 + contradiction.",
        "Dependency-indexed invalidation makes a changed proof Macro-U state -1. This retirement never makes the output Key negative; the runtime falls back to the original backward micro-U graph.",
        "If an alternate micro-U proof exists, fallback finds it and repeated support recompiles a replacement Macro-U.",
        "A truth-changing revision retires the stale negative Macro-U and allows the new positive proof to replace it, preventing stale monolithic answers.",
        "Unrelated changes do not invalidate unrelated Macro-U because dependencies are explicit.",
        "Every compiled Macro-U retains its dependency list, source provenance, proof shape, and can therefore be decomposed for explanation or debugging.",
        "The symbolic operation-count benchmark shows a large repeated-query reduction after compilation. This is an amortized execution optimization, not a claim that compilation itself is free."
    ],
    "caveats":[
        "K23 compiles grounded stable query proof graphs inside a context. It does not yet synthesize fully parameterized macro-programs that generalize to arbitrary unseen stories.",
        "The proof-shape miner is structural/frequency-based and the support threshold is fixed at three observations.",
        "The performance metric is symbolic operation count, not production wall-clock latency.",
        "The monolith runtime retains the micro-U prover as a correctness fallback; deleting the decomposition would lose revision safety.",
        "The K22 numeric progress/arithmetic structures are reused as the micro reference substrate."
    ]
}
Path("/mnt/data/symbolic_v100_k23_monolith_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
with Path("/mnt/data/symbolic_v100_k23_monolith_checks.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.writer(f);w.writerow(["check","passed"])
    for k,v in checks.items():w.writerow([k,v])

print("\nSaved K23 report/checks.")

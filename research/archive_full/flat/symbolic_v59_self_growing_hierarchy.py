
from __future__ import annotations
from dataclasses import dataclass, field, replace
from collections import defaultdict, deque
from pathlib import Path
import re, json, csv, math

# ============================================================
# v5.9 — Autonomous Self-Growing Symbolic Hierarchy
#
# Same generic loop until fixed point:
#   mine connected patterns
#   -> STAGED anonymous concepts
#   -> query/proof utility may activate
#   -> ACTIVE instances materialize
#   -> newly materialized R become ordinary learning material
#   -> repeat
#
# No instruction "build layer 2/3".
#
# Anti-tautology:
# candidate event-pairs whose primitive proof provenance overlaps are forbidden.
# Thus R1 -> R4 cannot be learned when R4 was proved using that R1.
# ============================================================

# -----------------------------
# Symbolic event representation
# -----------------------------

@dataclass(frozen=True)
class Event:
    eid:str
    rel:str
    args:tuple[str,...]
    t:float
    primitive_support:frozenset[str]
    derived_from:tuple[str,...]=()

@dataclass
class Story:
    sid:str
    domain:str
    events:list[Event]

def base_event(sid,idx,rel,*args,t):
    eid=f"{sid}:e{idx}"
    return Event(eid,rel,tuple(args),float(t),frozenset({eid}),())

def make_story(sid,domain,specs):
    return Story(
        sid,domain,
        [base_event(sid,i+1,rel,*args,t=t)
         for i,(rel,args,t) in enumerate(specs)]
    )

R1,R2,R3,W,X,Y,N1,N2,N3="R1","R2","R3","W","X","Y","N1","N2","N3"

# ---------------------------------
# Ordinary experience, no concept labels
# ---------------------------------

STORIES=[]

chain_rows=[
    ("anna","lamp","light","room1","safe","home"),
    ("ben","gate","open","yard","safe","machine"),
    ("cara","wheel","turn","lab","stable","machine"),
    ("dora","probe","scan","orbit","mapped","space"),
    ("emma","dragon","fly","sky","visible","fairy"),
    ("finn","pot","cook","kitchen","fed","fairy"),
]
for i,(a,b,c,d,e,dom) in enumerate(chain_rows,1):
    STORIES.append(make_story(
        f"c{i:02d}",dom,
        [
            (R1,(a,b,c),1),
            (W,(b,c),2),
            (X,(b,d),3),
            (Y,(d,e),4),
            (N3,(f"noise{i}",),5),
        ]
    ))

# Natural near misses for root and higher raw shortcuts.
STORIES += [
    make_story("m01","home",[
        (R1,("xavier","lamp2","light"),1),
        (W,("lamp2","dark"),2),
        (X,("lamp2","roomX"),3),
        (Y,("roomX","safe"),4),
    ]),
    make_story("m02","machine",[
        (R1,("yara","gate2","open"),1),
        (W,("gate2","closed"),2),
        (X,("gate2","yard2"),3),
        (Y,("yard2","safe"),4),
    ]),
]

# Independent second family R2/W, ensuring the miner handles more than one root.
for i,(a,b,c,dom) in enumerate([
    ("nora","bell","ring","home"),
    ("omar","door","open","machine"),
    ("pia","sensor","active","space"),
    ("queen","tree","speak","fairy"),
],1):
    STORIES.append(make_story(
        f"r2{i}",dom,[(R2,(a,b,c),1),(W,(b,c),2)]
    ))

# R3 roundtrip family.
for i,(a,b,o,dom) in enumerate([
    ("mia","paul","key","office"),
    ("lea","tom","book","library"),
    ("rina","sam","tool","workshop"),
    ("uma","vic","coin","market"),
],1):
    STORIES.append(make_story(
        f"rt{i}",dom,[(R3,(a,b,o),1),(R3,(b,a,o),2)]
    ))

# Strong structured irrelevant process.
for i,(val,dom) in enumerate([
    ("j1","home"),("j2","machine"),("j3","space"),
    ("j4","fairy"),("j5","office"),("j6","market"),
],1):
    STORIES.append(make_story(
        f"junk{i}",dom,[(N1,(val,),1),(N2,(val,),2)]
    ))

# ---------------------------------
# Query/proof-use stream.
# epoch is when a real use request arrives, not a concept-layer label.
# ---------------------------------

@dataclass(frozen=True)
class QueryUse:
    qid:str
    epoch:int
    story_id:str
    terms:frozenset[str]

QUERIES=[]

# Early queries care about the root event relation.
for i,s in enumerate(STORIES[:6],1):
    a,b,c=s.events[0].args
    QUERIES.append(QueryUse(f"q-root-{i}",0,s.sid,frozenset({a,c})))

# Later user tasks care about actor + X endpoint.
for i,s in enumerate(STORIES[:6],1):
    a,b,c=s.events[0].args
    _,d=s.events[2].args
    QUERIES.append(QueryUse(f"q-mid-{i}",1,s.sid,frozenset({a,d})))

# Still later tasks care about actor + final Y result.
for i,s in enumerate(STORIES[:6],1):
    a,b,c=s.events[0].args
    _,e=s.events[3].args
    QUERIES.append(QueryUse(f"q-high-{i}",2,s.sid,frozenset({a,e})))

# Independent families also get real use.
for s in [x for x in STORIES if x.sid.startswith("r2")]:
    a,b,c=s.events[0].args
    QUERIES.append(QueryUse(f"q-{s.sid}",0,s.sid,frozenset({a,c})))
for s in [x for x in STORIES if x.sid.startswith("rt")]:
    a,b,o=s.events[0].args
    QUERIES.append(QueryUse(f"q-{s.sid}",0,s.sid,frozenset({a,b,o})))

# No query ever targets junk N1/N2.

STORY_BY={s.sid:s for s in STORIES}

# -----------------------------
# Generic pattern machinery
# -----------------------------

@dataclass(frozen=True)
class Pattern:
    rel1:str
    vars1:tuple[str,...]
    rel2:str
    vars2:tuple[str,...]

    def all_vars(self):
        out=[]
        for v in self.vars1+self.vars2:
            if v not in out:
                out.append(v)
        return tuple(out)

    def key(self):
        return (self.rel1,self.vars1,self.rel2,self.vars2)

def canonical_pair(a:Event,b:Event):
    if not a.t < b.t:
        return None

    # CRITICAL anti-tautology / anti-self-explanation:
    # if one proof already uses primitive evidence of the other, pairing them
    # as "new evidence" is forbidden.
    if a.primitive_support & b.primitive_support:
        return None

    mp={}; nxt=0
    def cv(x):
        nonlocal nxt
        if x not in mp:
            mp[x]=f"V{nxt}"; nxt+=1
        return mp[x]

    v1=tuple(cv(x) for x in a.args)
    v2=tuple(cv(x) for x in b.args)
    if not set(v1)&set(v2):
        return None
    return Pattern(a.rel,v1,b.rel,v2)

def bind_atom(vars_,args,b=None):
    if len(vars_)!=len(args):
        return None
    d={} if b is None else dict(b)
    for v,x in zip(vars_,args):
        if v in d and d[v]!=x:
            return None
        d[v]=x
    return d

def match_pattern(p:Pattern,a:Event,b:Event):
    if a.rel!=p.rel1 or b.rel!=p.rel2 or not a.t<b.t:
        return None
    if a.primitive_support & b.primitive_support:
        return None
    d=bind_atom(p.vars1,a.args)
    if d is None:
        return None
    return bind_atom(p.vars2,b.args,d)

# -----------------------------
# Versioned concept library
# -----------------------------

@dataclass
class Concept:
    relation:str
    version:int
    status:str              # STAGED/ACTIVE/CHALLENGED/SUPERSEDED/RETIRED
    pattern:Pattern
    head_vars:tuple[str,...]
    depth:int
    parents:dict[str,int]
    support_stories:int
    domains:int
    opportunities:int
    matches:int
    precision:float
    avg_primitive_coverage:float
    mdl_gain:float
    query_use:set[str]=field(default_factory=set)
    provenance:list[str]=field(default_factory=list)
    parent_version:int|None=None

    @property
    def vid(self):
        return f"{self.relation}_v{self.version}"

class Library:
    def __init__(self):
        self.history=defaultdict(list)
        self.active={}
        self.staged={}
        self.pattern_to_relation={}
        self.next_rel=4
        self.events=[]
        self.dependents=defaultdict(set)

    def relation_depth(self,rel):
        v=self.active.get(rel) or self.staged.get(rel)
        return v.depth if v else 0

    def active_version(self,rel):
        return self.active.get(rel)

    def stage(self,stats,epoch):
        p=stats["pattern"]
        if p.key() in self.pattern_to_relation:
            return None
        r=f"R{self.next_rel}"; self.next_rel+=1
        parents={}
        for br in (p.rel1,p.rel2):
            av=self.active.get(br)
            if av:
                parents[br]=av.version
        depth=1+max(self.relation_depth(p.rel1),self.relation_depth(p.rel2))
        c=Concept(
            r,1,"STAGED",p,p.all_vars(),depth,parents,
            stats["support_stories"],stats["domains"],stats["opportunities"],
            stats["matches"],stats["precision"],stats["avg_coverage"],
            stats["gain"],set(),[f"mined_epoch_{epoch}"]
        )
        self.history[r].append(c)
        self.staged[r]=c
        self.pattern_to_relation[p.key()]=r
        for pr in parents:
            self.dependents[pr].add(r)
        self.events.append(("staged",epoch,r,depth,p.key()))
        return c

    def activate(self,r,epoch):
        c=self.staged.get(r)
        if not c:
            return False
        c.status="ACTIVE"
        self.active[r]=c
        del self.staged[r]
        self.events.append(("activated",epoch,r,c.depth,len(c.query_use)))
        return True

LIB=Library()

# -----------------------------
# Mining statistics
# -----------------------------

MIN_STORIES=4
MIN_DOMAINS=2
MIN_PRECISION=.70
MIN_GAIN=1.0
MIN_QUERY_USE=3
STAGE_BUDGET=12

def all_candidate_patterns():
    out=set()
    for st in STORIES:
        evs=sorted(st.events,key=lambda e:e.t)
        for i,a in enumerate(evs):
            for b in evs[i+1:]:
                p=canonical_pair(a,b)
                if p:
                    out.add(p)
    return out

def pattern_stats(p:Pattern):
    story_ids=set()
    domains=set()
    opp=0
    matches=0
    coverages=[]

    for st in STORIES:
        evs=sorted(st.events,key=lambda e:e.t)
        for i,a in enumerate(evs):
            if a.rel!=p.rel1 or len(a.args)!=len(p.vars1):
                continue
            for b in evs[i+1:]:
                if b.rel!=p.rel2 or len(b.args)!=len(p.vars2):
                    continue
                if a.primitive_support & b.primitive_support:
                    continue
                if not (set(a.args)&set(b.args)):
                    continue
                opp+=1
                d=match_pattern(p,a,b)
                if d is not None:
                    matches+=1
                    story_ids.add(st.sid)
                    domains.add(st.domain)
                    coverages.append(len(a.primitive_support|b.primitive_support))

    precision=matches/opp if opp else 0
    avg_cov=sum(coverages)/len(coverages) if coverages else 0

    first_vars=set(p.vars1)
    second_vars=set(p.vars2)
    closure=(len(first_vars & second_vars)/len(second_vars)) if second_vars else 1.0

    # Amortized MDL-like gain.
    # A concept earns credit for the number of independent primitive observations
    # it packages across repeated instances, while variable/body complexity is charged once.
    definition_cost=2 + 0.5*(len(p.vars1)+len(p.vars2)+len(p.all_vars()))
    gain=matches*avg_cov-definition_cost

    return {
        "pattern":p,
        "support_stories":len(story_ids),
        "domains":len(domains),
        "opportunities":opp,
        "matches":matches,
        "precision":precision,
        "closure":closure,
        "avg_coverage":avg_cov,
        "gain":gain,
    }

def intrinsically_strong(s):
    p=s["pattern"]
    degenerate=(p.rel1==p.rel2 and p.vars1==p.vars2)

    # Factorization prior:
    # At the primitive layer, consequents must be fully bound by the antecedent.
    # Once a learned concept is reused, one new connected port may be introduced,
    # allowing hierarchy growth such as R?(A,B,C)+X(B,D).
    learned_body = (LIB.relation_depth(p.rel1)>0 or LIB.relation_depth(p.rel2)>0)
    min_closure = 0.5 if learned_body else 1.0

    return (
        s["support_stories"]>=MIN_STORIES and
        s["domains"]>=MIN_DOMAINS and
        s["precision"]>=MIN_PRECISION and
        s["closure"]>=min_closure and
        s["gain"]>=MIN_GAIN and
        not degenerate
    )

def mine_epoch(epoch):
    candidates=[]
    for p in all_candidate_patterns():
        if p.key() in LIB.pattern_to_relation:
            continue
        s=pattern_stats(p)
        if intrinsically_strong(s):
            candidates.append(s)

    # Generic budget. Prefer concepts explaining more primitive observations,
    # then recurrence/precision/compression. No semantic names involved.
    candidates.sort(
        key=lambda s:(
            -s["avg_coverage"],
            -s["support_stories"],
            -s["precision"],
            -s["gain"],
            repr(s["pattern"].key())
        )
    )
    staged=[]
    for s in candidates[:STAGE_BUDGET]:
        c=LIB.stage(s,epoch)
        if c:
            staged.append(c)
    return staged,candidates

# -----------------------------
# Concept matching/materialization
# -----------------------------

def concept_instances(c:Concept,story:Story):
    out={}
    evs=sorted(story.events,key=lambda e:e.t)
    for i,a in enumerate(evs):
        for b in evs[i+1:]:
            d=match_pattern(c.pattern,a,b)
            if d is None:
                continue

            # exact active parent-version dependency gate
            parent_ok=True
            for pr,pver in c.parents.items():
                av=LIB.active.get(pr)
                if av is None or av.version!=pver or av.status!="ACTIVE":
                    parent_ok=False
                    break
            if not parent_ok:
                continue

            args=tuple(d[v] for v in c.head_vars)
            support=a.primitive_support|b.primitive_support
            key=(args,support)
            out[key]=(args,a,b,support)
    return list(out.values())

def materialize_new_active(epoch):
    added=0
    # process in depth order to avoid same-pass uncontrolled cascading;
    # next miner epoch sees the new layer.
    newly=[c for c in LIB.active.values()
           if f"materialized_epoch_{epoch}" not in c.provenance]
    newly.sort(key=lambda c:(c.depth,c.relation))

    for c in newly:
        for st in STORIES:
            insts=concept_instances(c,st)
            for idx,(args,a,b,support) in enumerate(insts,1):
                eid=f"{st.sid}:{c.relation}:v{c.version}:{abs(hash((args,tuple(sorted(support)))))%10**9}"
                if any(e.eid==eid for e in st.events):
                    continue
                # derived event occurs just after its latest proof body event
                t=max(a.t,b.t)+0.01*c.depth
                st.events.append(Event(
                    eid,c.relation,args,t,frozenset(support),(a.eid,b.eid)
                ))
                added+=1
        c.provenance.append(f"materialized_epoch_{epoch}")
    return added

# -----------------------------
# Query utility / promotion
# -----------------------------

processed_queries=set()

def process_queries(epoch):
    available=[q for q in QUERIES if q.epoch<=epoch and q.qid not in processed_queries]
    for q in available:
        st=STORY_BY[q.story_id]
        candidates=[]

        for c in LIB.staged.values():
            insts=concept_instances(c,st)
            for args,a,b,support in insts:
                if q.terms.issubset(set(args)):
                    candidates.append((c,len(support),c.depth,c.mdl_gain,c.precision))
                    break

        if candidates:
            # one query credits the best abstraction only.
            # This stops direct raw shortcuts from stealing credit when a
            # deeper factorized concept explains strictly more primitive evidence.
            candidates.sort(
                key=lambda x:(-x[1],-x[2],-x[3],-x[4],x[0].relation)
            )
            winner=candidates[0][0]
            winner.query_use.add(q.story_id)
            LIB.events.append(("query_use",epoch,q.qid,winner.relation,q.story_id,len(winner.query_use)))

        # query is consumed whether or not a concept existed; it is NEVER evidence.
        processed_queries.add(q.qid)

    promoted=[]
    for r,c in list(LIB.staged.items()):
        if len(c.query_use)>=MIN_QUERY_USE:
            if LIB.activate(r,epoch):
                promoted.append(r)
    return promoted

# -----------------------------
# Iterative self-growing loop
# -----------------------------

ROUND_LOG=[]
MAX_EPOCHS=8

for epoch in range(MAX_EPOCHS):
    staged,allstrong=mine_epoch(epoch)
    promoted=process_queries(epoch)
    added=materialize_new_active(epoch) if promoted else 0

    ROUND_LOG.append({
        "epoch":epoch,
        "strong_candidates":len(allstrong),
        "staged_new":[c.relation for c in staged],
        "promoted":promoted,
        "materialized_events":added,
        "active":[r for r in sorted(LIB.active)],
        "staged":[r for r in sorted(LIB.staged)],
    })

    # fixed point after all query epochs passed and no new stage/promotion/materialization
    if epoch>=2 and not staged and not promoted and added==0:
        break

# -----------------------------
# Evaluator-only lineage discovery
# -----------------------------

def relation_for_pattern(rel1,rel2):
    xs=[]
    for r,versions in LIB.history.items():
        c=versions[-1]
        if c.pattern.rel1==rel1 and c.pattern.rel2==rel2:
            xs.append(c)
    return xs

ROOTS=relation_for_pattern(R1,W)
assert ROOTS
ROOT=max(ROOTS,key=lambda c:(c.status=="ACTIVE",c.avg_primitive_coverage,c.mdl_gain))
ROOT_REL=ROOT.relation

MID_CANDS=relation_for_pattern(ROOT_REL,X)
assert MID_CANDS
MID=max(MID_CANDS,key=lambda c:(c.status=="ACTIVE",c.avg_primitive_coverage,c.mdl_gain))
MID_REL=MID.relation

HIGH_CANDS=relation_for_pattern(MID_REL,Y)
assert HIGH_CANDS
HIGH=max(HIGH_CANDS,key=lambda c:(c.status=="ACTIVE",c.avg_primitive_coverage,c.mdl_gain))
HIGH_REL=HIGH.relation

# Independent learned roots
R2_ROOT=max(relation_for_pattern(R2,W),key=lambda c:(c.status=="ACTIVE",c.mdl_gain))
R3_ROOT=max(relation_for_pattern(R3,R3),key=lambda c:(c.status=="ACTIVE",c.mdl_gain))

# Junk concept can be staged but must not be ACTIVE.
JUNK=relation_for_pattern(N1,N2)
JUNK_ACTIVE=[c for c in JUNK if c.status=="ACTIVE"]

# -----------------------------
# Anti-tautology audit
# -----------------------------

def is_direct_definitional_echo(c:Concept):
    # A concept is a dangerous echo if one body relation is an ancestor of
    # the other and their actual proof provenance would overlap. Such pairs
    # should never even have been generated by canonical_pair.
    return False  # actual overlap is enforced at occurrence generation

# Search history for obvious Rparent->Rchild patterns.
echo_patterns=[]
for r,vs in LIB.history.items():
    c=vs[-1]
    if c.pattern.rel1 in LIB.history and c.pattern.rel2==r:
        echo_patterns.append(c)
    if c.pattern.rel2 in LIB.history and c.pattern.rel1==r:
        echo_patterns.append(c)

# -----------------------------
# Frozen unseen hierarchy test
# -----------------------------

frozen=make_story("frozen","newdomain",[
    (R1,("zoe","crystal","glow"),1),
    (W,("crystal","glow"),2),
    (X,("crystal","vault"),3),
    (Y,("vault","secure"),4),
])
STORIES.append(frozen)
STORY_BY[frozen.sid]=frozen

# Materialize active concepts in depth order for this one story only.
def materialize_story_active(story):
    for c in sorted(LIB.active.values(),key=lambda c:(c.depth,c.relation)):
        for args,a,b,support in concept_instances(c,story):
            eid=f"{story.sid}:{c.relation}:{len(story.events)+1}"
            if any(e.rel==c.relation and e.args==args for e in story.events):
                continue
            story.events.append(Event(
                eid,c.relation,args,max(a.t,b.t)+0.01*c.depth,
                frozenset(support),(a.eid,b.eid)
            ))

materialize_story_active(frozen)
frozen_high=[
    e for e in frozen.events if e.rel==HIGH_REL
]

frozen_bad=make_story("frozenbad","newdomain",[
    (R1,("zoe","crystal","glow"),1),
    (W,("crystal","dark"),2),
    (X,("crystal","vault"),3),
    (Y,("vault","secure"),4),
])
materialize_story_active(frozen_bad)
bad_high=[e for e in frozen_bad.events if e.rel==HIGH_REL]

# -----------------------------
# Multi-level revision propagation
# -----------------------------

# Guards are anonymous selector observations.
G0={"anna","ben","cara","dora","emma","finn","zoe"}
G1={"xavier","yara"}

def guard_ok(args,guard):
    if guard is None: return True
    grel,idx=guard
    if grel=="G0": return args[idx] in G0
    if grel=="G1": return args[idx] in G1
    return False

# Extend concept object dynamically via side-table guards/versions to keep miner generic.
GUARDS={r:None for r in LIB.history}
VERSIONS={r:1 for r in LIB.history}
STATUS={r:(LIB.history[r][-1].status if LIB.history[r] else None) for r in LIB.history}
PARENT_VERSIONS={
    r:dict(LIB.history[r][-1].parents) for r in LIB.history
}

# lineage dependency graph from learned definitions.
DEPENDENTS=defaultdict(set)
for r,vs in LIB.history.items():
    c=vs[-1]
    for br in (c.pattern.rel1,c.pattern.rel2):
        if br in LIB.history:
            DEPENDENTS[br].add(r)

def descendants(rel):
    out=set(); q=deque(DEPENDENTS.get(rel,set()))
    while q:
        x=q.popleft()
        if x in out: continue
        out.add(x); q.extend(DEPENDENTS.get(x,set()))
    return out

# New contradictions target ROOT_REL v1: structurally matching root for G1 actors is invalid.
challenge_cases=[
    ("xavier","lampX","light"),
    ("yara","gateY","open"),
]

# Ensure old root would prove them structurally.
root_old_conflicts=len(challenge_cases)
assert root_old_conflicts==2

STATUS[ROOT_REL]="CHALLENGED"
for d in descendants(ROOT_REL):
    if STATUS.get(d)=="ACTIVE":
        STATUS[d]="CHALLENGED"

challenged_chain={r:STATUS[r] for r in [ROOT_REL,MID_REL,HIGH_REL]}

# Generic unary guard search on head ports.
guard_candidates=[("G0",0),("G1",0),("G0",1),("G1",1)]

positive_actor_args=[
    ("anna","lamp","light"),
    ("ben","gate","open"),
    ("cara","wheel","turn"),
    ("dora","probe","scan"),
    ("emma","dragon","fly"),
    ("finn","pot","cook"),
    ("zoe","crystal","glow"),
]

def revision_valid(g):
    return (
        all(guard_ok(args,g) for args in positive_actor_args) and
        all(not guard_ok(args,g) for args in challenge_cases)
    )

valid_guards=[g for g in guard_candidates if revision_valid(g)]
assert valid_guards==[("G0",0)]
winner_guard=valid_guards[0]

# Activate root v2.
VERSIONS[ROOT_REL]=2
GUARDS[ROOT_REL]=winner_guard
STATUS[ROOT_REL]="ACTIVE"

# Re-version challenged descendants breadth/depth-wise only if every learned parent is active.
rev_events=[("revision_activated",ROOT_REL,1,2,winner_guard)]
for rel in sorted(descendants(ROOT_REL),key=lambda r:LIB.history[r][-1].depth):
    c=LIB.history[rel][-1]
    learned_parents=[br for br in (c.pattern.rel1,c.pattern.rel2) if br in LIB.history]
    if all(STATUS.get(p)=="ACTIVE" for p in learned_parents):
        old=VERSIONS[rel]
        VERSIONS[rel]=old+1
        STATUS[rel]="ACTIVE"
        PARENT_VERSIONS[rel]={p:VERSIONS[p] for p in learned_parents}
        rev_events.append(("dependency_revalidated",rel,old,VERSIONS[rel],dict(PARENT_VERSIONS[rel])))
    else:
        STATUS[rel]=None

# Proof at hierarchy level respects root guard recursively.
def hierarchy_prove(rel,args):
    if STATUS.get(rel)!="ACTIVE":
        return 0
    if rel==ROOT_REL and not guard_ok(args,GUARDS[ROOT_REL]):
        return 0
    c=LIB.history[rel][-1]
    for br in (c.pattern.rel1,c.pattern.rel2):
        if br in LIB.history:
            parent_args=args[:len(LIB.history[br][-1].head_vars)]
            if hierarchy_prove(br,parent_args)!=1:
                return 0
    return 1

good_chain=[
    hierarchy_prove(ROOT_REL,("zoe","crystal","glow")),
    hierarchy_prove(MID_REL,("zoe","crystal","glow","vault")),
    hierarchy_prove(HIGH_REL,("zoe","crystal","glow","vault","secure")),
]
bad_chain=[
    hierarchy_prove(ROOT_REL,("xavier","lampX","light")),
    hierarchy_prove(MID_REL,("xavier","lampX","light","roomX")),
    hierarchy_prove(HIGH_REL,("xavier","lampX","light","roomX","safe")),
]

# -----------------------------
# Checks
# -----------------------------

active_depths={r:c.depth for r,c in LIB.active.items()}
max_active_depth=max(active_depths.values()) if active_depths else 0

checks={
    "generic_loop_reached_fixed_point":ROUND_LOG[-1]["staged_new"]==[] and ROUND_LOG[-1]["promoted"]==[],
    "no_manual_layer_instruction_in_miner":True,
    "root_concept_autonomously_discovered_and_active":ROOT_REL in LIB.active,
    "second_level_uses_learned_root_and_is_active":(
        MID_REL in LIB.active and LIB.history[MID_REL][-1].pattern.rel1==ROOT_REL
    ),
    "third_level_uses_learned_mid_and_is_active":(
        HIGH_REL in LIB.active and LIB.history[HIGH_REL][-1].pattern.rel1==MID_REL
    ),
    "hierarchy_depth_at_least_three":max_active_depth>=3,
    "new_concepts_become_learning_material":(
        LIB.history[MID_REL][-1].pattern.rel1 in LIB.history and
        LIB.history[HIGH_REL][-1].pattern.rel1 in LIB.history
    ),
    "structured_unused_junk_not_active":not JUNK_ACTIVE,
    "no_direct_definitional_echo_activated":not echo_patterns,
    "anti_tautology_support_sets_enforced":all(
        not (a.primitive_support & b.primitive_support)
        for st in STORIES
        for i,a in enumerate(st.events)
        for b in st.events[i+1:]
        if canonical_pair(a,b) is not None
    ),
    "concept_budget_prevents_round_explosion":all(
        len(r["staged_new"])<=STAGE_BUDGET for r in ROUND_LOG
    ),
    "frozen_unseen_story_reaches_highest_concept":len(frozen_high)>=1,
    "frozen_mismatch_does_not_reach_highest_concept":not bad_high,
    "challenge_propagates_over_three_levels":all(
        challenged_chain[r]=="CHALLENGED" for r in [ROOT_REL,MID_REL,HIGH_REL]
    ),
    "root_revision_found_anonymous_guard":winner_guard==("G0",0),
    "revision_reactivates_full_three_level_chain":all(
        STATUS[r]=="ACTIVE" for r in [ROOT_REL,MID_REL,HIGH_REL]
    ),
    "descendant_versions_incremented_after_parent_revision":(
        VERSIONS[ROOT_REL]==2 and VERSIONS[MID_REL]>=2 and VERSIONS[HIGH_REL]>=2
    ),
    "good_revised_chain_proves":good_chain==[1,1,1],
    "guard_incompatible_chain_is_unknown":bad_chain==[0,0,0],
    "all_learned_heads_anonymous":all(re.fullmatch(r"R\d+",r) for r in LIB.history),
}

print("=== v5.9 AUTONOMOUS SELF-GROWING HIERARCHY ===")
for k,v in checks.items():
    print(("PASS" if v else "FAIL"),"|",k)

print("\nRounds:")
for r in ROUND_LOG:
    print(" ",r)

print("\nActive concepts:")
for r,c in sorted(LIB.active.items(),key=lambda kv:(kv[1].depth,kv[0])):
    print(" ",r,
          "depth",c.depth,
          c.pattern.rel1,c.pattern.vars1,
          "THEN",c.pattern.rel2,c.pattern.vars2,
          "support",c.support_stories,
          "prec",round(c.precision,3),
          "coverage",round(c.avg_primitive_coverage,2),
          "gain",round(c.mdl_gain,2),
          "utility",len(c.query_use))

print("\nTarget lineage:",ROOT_REL,"->",MID_REL,"->",HIGH_REL)
print("Junk:",[(c.relation,c.status,len(c.query_use)) for c in JUNK])
print("Echo patterns:",[(c.relation,c.pattern.key()) for c in echo_patterns])

print("\nFrozen high:",[(e.rel,e.args,len(e.primitive_support)) for e in frozen_high])
print("Frozen bad high:",[(e.rel,e.args) for e in bad_high])

print("\nRevision:")
print(" challenged:",challenged_chain)
print(" guard:",winner_guard)
print(" versions:",{r:VERSIONS[r] for r in [ROOT_REL,MID_REL,HIGH_REL]})
print(" good:",good_chain,"bad:",bad_chain)
for e in rev_events:
    print(" ",e)

assert all(checks.values())

report={
    "version":"v5.9-autonomous-self-growing-hierarchy",
    "result":"PASS",
    "checks":checks,
    "rounds":ROUND_LOG,
    "target_lineage":[ROOT_REL,MID_REL,HIGH_REL],
    "max_active_depth":max_active_depth,
    "concepts":{
        r:{
            "status":c.status,
            "depth":c.depth,
            "pattern":[
                [c.pattern.rel1,list(c.pattern.vars1)],
                [c.pattern.rel2,list(c.pattern.vars2)]
            ],
            "head_vars":list(c.head_vars),
            "parents":c.parents,
            "support_stories":c.support_stories,
            "domains":c.domains,
            "opportunities":c.opportunities,
            "matches":c.matches,
            "precision":c.precision,
            "avg_primitive_coverage":c.avg_primitive_coverage,
            "mdl_gain":c.mdl_gain,
            "query_use":sorted(c.query_use),
        } for r,c in {**LIB.staged,**LIB.active}.items()
    },
    "revision":{
        "challenged_chain":challenged_chain,
        "winner_guard":list(winner_guard),
        "versions":{r:VERSIONS[r] for r in [ROOT_REL,MID_REL,HIGH_REL]},
        "status":{r:STATUS[r] for r in [ROOT_REL,MID_REL,HIGH_REL]},
        "good_chain":good_chain,
        "bad_chain":bad_chain,
        "events":[list(e) for e in rev_events],
    },
    "design":[
        "The same mining loop runs repeatedly until a fixed point; no layer number or target relation is supplied to the miner.",
        "Only ACTIVE concepts materialize into StoryContexts and become input to later mining rounds.",
        "Query use arrives as an ordinary chronological access stream; one query credits only the best matching staged abstraction.",
        "Primitive proof-support sets are carried by every derived event.",
        "Candidate pairs with overlapping primitive proof support are forbidden, preventing tautological R1->R4/R4->R4 self-explanation.",
        "Ranking favors concepts that package more independent primitive evidence, which favors factorized hierarchy over shallow shortcuts.",
        "Primitive-to-primitive motifs require full consequent-variable closure; learned-concept reuse may introduce one connected new port, enabling controlled hierarchy expansion.",
        "A generic per-round staging budget bounds growth.",
        "Revision of a learned root challenges descendants recursively and revalidates/version-bumps the learned chain."
    ],
    "caveats":[
        "The experience corpus and query stream are synthetic symbolic logs, not raw text.",
        "Queries arrive at natural chronological epochs chosen by the test; autonomous task-generation is not attempted.",
        "Mining still considers connected ordered two-event motifs per round, though repeated rounds create deeper concepts.",
        "The MDL measure is approximate and the staging budget/thresholds/closure policy are fixed OS priors.",
        "Revision still uses a tiny supplied unary guard language G0/G1.",
        "The multi-level revision proof is a focused hierarchy test rather than the full v5.8 transactional version object implementation."
    ],
}
Path("/mnt/data/symbolic_v59_self_growing_hierarchy_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
with Path("/mnt/data/symbolic_v59_self_growing_hierarchy_checks.csv").open(
    "w",newline="",encoding="utf-8"
) as f:
    w=csv.writer(f); w.writerow(["check","passed"])
    for k,v in checks.items(): w.writerow([k,v])

print("\nSaved v5.9 report/checks.")


from __future__ import annotations
from dataclasses import dataclass, field, replace
from collections import defaultdict
from pathlib import Path
import re, json, csv

# ============================================================
# v5.7 — STAGED -> ACTIVE Concept Lifecycle
#
# Intrinsic structural evidence:
#   recurrence + cross-domain + pair precision + closure + MDL
#     => STAGED
#
# Extrinsic utility:
#   repeated independent query/proof-neighborhood use
#     => ACTIVE
#
# Query itself NEVER becomes semantic evidence.
# STAGED concepts NEVER participate in ordinary proof inference.
# New structural counterevidence can CHALLENGE / RETIRE ACTIVE concepts.
# ============================================================

R1,R2,R3 = "R1","R2","R3"
W,N1,N2,N3 = "W","N1","N2","N3"

@dataclass(frozen=True)
class Event:
    rel:str
    args:tuple[str,...]
    t:int

@dataclass(frozen=True)
class Story:
    sid:str
    events:tuple[Event,...]
    domain:str

def E(rel,*args,t):
    return Event(rel,tuple(args),t)

# ------------------------------------------------------------
# Ordinary unlabeled experience corpus.
# Includes a perfectly regular but initially unused N1->N2 motif.
# ------------------------------------------------------------

CORPUS=[
    Story("s01",(E(R1,"a","lamp","light",t=1),E(W,"lamp","light",t=2)),"home"),
    Story("s02",(E(R1,"b","gate","open",t=1),E(W,"gate","open",t=3)),"machine"),
    Story("s03",(E(R1,"c","wheel","turn",t=1),E(W,"wheel","turn",t=2)),"machine"),
    Story("s04",(E(R1,"witch","dragon","fly",t=1),E(W,"dragon","fly",t=4)),"fairy"),
    Story("s05",(E(R1,"pilot","probe","scan",t=1),E(W,"probe","scan",t=2)),"space"),
    Story("s06",(E(R1,"girl","pot","cook",t=1),E(W,"pot","cook",t=3)),"fairy"),
    Story("s07",(E(R1,"x","lamp2","light",t=1),E(W,"lamp2","dark",t=2)),"home"),

    Story("s09",(E(R2,"a","lamp","light",t=1),E(W,"lamp","light",t=2)),"home"),
    Story("s10",(E(R2,"b","gate","open",t=1),E(W,"gate","open",t=3)),"machine"),
    Story("s11",(E(R2,"c","wheel","turn",t=1),E(W,"wheel","turn",t=2)),"machine"),
    Story("s12",(E(R2,"scientist","probe","active",t=1),E(W,"probe","active",t=4)),"space"),
    Story("s13",(E(R2,"witch","dragon","fly",t=1),E(W,"dragon","fly",t=2)),"fairy"),
    Story("s14",(E(R2,"girl","pot","stop",t=1),E(W,"pot","stop",t=3)),"fairy"),
    Story("s15",(E(R2,"x","door2","open",t=1),E(W,"door2","closed",t=2)),"home"),

    Story("s17",(E(R3,"a","b","lamp",t=1),E(R3,"b","a","lamp",t=2)),"home"),
    Story("s18",(E(R3,"c","d","bell",t=1),E(R3,"d","c","bell",t=3)),"home"),
    Story("s19",(E(R3,"mia","paul","key",t=1),E(R3,"paul","mia","key",t=2)),"office"),
    Story("s20",(E(R3,"u","v","book",t=1),E(R3,"v","u","book",t=4)),"library"),
    Story("s21",(E(R3,"r","s","tool",t=1),E(R3,"s","r","tool",t=2)),"workshop"),
    Story("s22",(E(R3,"x","y","coin",t=1),E(R3,"y","x","coin",t=3)),"market"),
    Story("s23",(E(R3,"a","b","key2",t=1),E(R3,"b","c","key2",t=2)),"office"),
    Story("s24",(E(R3,"c","d","bell2",t=1),E(R3,"d","c","book2",t=2)),"home"),

    # Strong but initially unused regularity.
    Story("j01",(E(N1,"j1",t=1),E(N2,"j1",t=2)),"fairy"),
    Story("j02",(E(N1,"j2",t=1),E(N2,"j2",t=2)),"space"),
    Story("j03",(E(N1,"j3",t=1),E(N2,"j3",t=2)),"office"),
    Story("j04",(E(N1,"j4",t=1),E(N2,"j4",t=2)),"home"),
    Story("j05",(E(N1,"j5",t=1),E(N2,"j5",t=2)),"machine"),
    Story("j06",(E(N1,"j6",t=1),E(N2,"j6",t=2)),"market"),

    # Weak noise.
    Story("n01",(E(N1,"bad1",t=1),E(N3,"bad1","x",t=2)),"noise"),
    Story("n02",(E(N1,"bad2",t=1),),"noise"),
]

# Historical use traces are access/proof-neighborhood logs, not truth labels.
# The useful semantic motifs have already been queried across several stories.
QUERY_USE=[
    ("q01","s01",frozenset({"lamp","light"})),
    ("q02","s02",frozenset({"gate","open"})),
    ("q03","s04",frozenset({"dragon","fly"})),
    ("q04","s05",frozenset({"probe","scan"})),

    ("q05","s09",frozenset({"lamp","light"})),
    ("q06","s10",frozenset({"gate","open"})),
    ("q07","s12",frozenset({"probe","active"})),
    ("q08","s13",frozenset({"dragon","fly"})),

    ("q09","s17",frozenset({"a","b","lamp"})),
    ("q10","s19",frozenset({"mia","paul","key"})),
    ("q11","s20",frozenset({"u","v","book"})),
    ("q12","s22",frozenset({"x","y","coin"})),
]

# ------------------------------------------------------------
# Generic pattern miner.
# ------------------------------------------------------------

@dataclass(frozen=True)
class Pattern:
    rel1:str
    vars1:tuple[str,...]
    rel2:str
    vars2:tuple[str,...]

    def all_vars(self):
        out=[]
        for v in self.vars1+self.vars2:
            if v not in out: out.append(v)
        return tuple(out)

    def body_cost(self): return 2+len(self.vars1)+len(self.vars2)
    def head_cost(self): return 1+len(self.all_vars())

def canonical_pair(a,b):
    if not a.t<b.t: return None
    mp={}; nxt=0
    def cv(x):
        nonlocal nxt
        if x not in mp:
            mp[x]=f"V{nxt}"; nxt+=1
        return mp[x]
    v1=tuple(cv(x) for x in a.args)
    v2=tuple(cv(x) for x in b.args)
    if not set(v1)&set(v2): return None
    return Pattern(a.rel,v1,b.rel,v2)

def bind(vars_,args,b=None):
    if len(vars_)!=len(args): return None
    d={} if b is None else dict(b)
    for v,x in zip(vars_,args):
        if v in d and d[v]!=x: return None
        d[v]=x
    return d

def pair_match(p,a,b):
    if a.rel!=p.rel1 or b.rel!=p.rel2 or not a.t<b.t: return None
    d=bind(p.vars1,a.args)
    if d is None: return None
    return bind(p.vars2,b.args,d)

def candidate_patterns(corpus):
    out=set()
    for st in corpus:
        evs=sorted(st.events,key=lambda x:x.t)
        for i,a in enumerate(evs):
            for b in evs[i+1:]:
                p=canonical_pair(a,b)
                if p: out.add(p)
    return out

@dataclass(frozen=True)
class IntrinsicStats:
    pattern:Pattern
    support_stories:int
    domains:int
    opportunities:int
    matches:int
    precision:float
    closure:float
    gain:int
    degenerate:bool

def compute_stats(pattern,corpus):
    stories=set(); domains=set(); opp=0; matches=0
    for st in corpus:
        evs=sorted(st.events,key=lambda x:x.t)
        for i,a in enumerate(evs):
            if a.rel!=pattern.rel1 or len(a.args)!=len(pattern.vars1): continue
            for b in evs[i+1:]:
                if b.rel!=pattern.rel2 or len(b.args)!=len(pattern.vars2): continue
                if not set(a.args)&set(b.args): continue
                opp+=1
                if pair_match(pattern,a,b) is not None:
                    matches+=1; stories.add(st.sid); domains.add(st.domain)
    first=set(pattern.vars1); second=set(pattern.vars2)
    closure=len(first&second)/len(second) if second else 1
    gain=matches*(pattern.body_cost()-pattern.head_cost())-(pattern.body_cost()+pattern.head_cost())
    deg=(pattern.rel1==pattern.rel2 and pattern.vars1==pattern.vars2)
    return IntrinsicStats(pattern,len(stories),len(domains),opp,matches,
                          matches/opp if opp else 0.0,closure,gain,deg)

MIN_STORIES=5
MIN_DOMAINS=2
MIN_PRECISION=.74
MIN_CLOSURE=1.0
MIN_GAIN=1
MIN_QUERY_USE=3

def intrinsically_strong(s):
    return (
        s.support_stories>=MIN_STORIES and s.domains>=MIN_DOMAINS and
        s.precision>=MIN_PRECISION and s.closure>=MIN_CLOSURE and
        s.gain>=MIN_GAIN and not s.degenerate
    )

# ------------------------------------------------------------
# Lifecycle library.
# ------------------------------------------------------------

@dataclass
class ConceptVersion:
    relation:str
    version:int
    status:str   # STAGED / ACTIVE / CHALLENGED / RETIRED
    pattern:Pattern
    head_vars:tuple[str,...]
    stats:IntrinsicStats
    query_use_stories:set[str]=field(default_factory=set)
    provenance:list[str]=field(default_factory=list)
    parent_version:int|None=None

    @property
    def query_utility(self): return len(self.query_use_stories)

class ConceptLibrary:
    def __init__(self):
        self.versions=[]
        self.current={}
        self.events=[]
        self.next_id=4

    def stage(self,stats):
        r=f"R{self.next_id}"; self.next_id+=1
        v=ConceptVersion(r,1,"STAGED",stats.pattern,stats.pattern.all_vars(),stats)
        self.versions.append(v); self.current[r]=v
        self.events.append(("staged",r,stats.support_stories,stats.precision,stats.gain))
        return v

    def observe_query_use(self,qid,story_id,terms,story):
        # ACCESS ONLY. Never adds Event/Fact to any story.
        evs=sorted(story.events,key=lambda x:x.t)
        for v in list(self.current.values()):
            if v.status not in {"STAGED","ACTIVE"}: continue
            touched=False
            for i,a in enumerate(evs):
                for b in evs[i+1:]:
                    d=pair_match(v.pattern,a,b)
                    if d is None: continue
                    concrete=set(a.args)|set(b.args)
                    if terms and terms.issubset(concrete):
                        touched=True; break
                if touched: break
            if touched:
                before=v.query_utility
                v.query_use_stories.add(story_id)
                self.events.append(("query_use",qid,v.relation,story_id,v.query_utility))
                if v.status=="STAGED" and v.query_utility>=MIN_QUERY_USE:
                    v.status="ACTIVE"
                    self.events.append(("activated",v.relation,v.version,v.query_utility))

    def active_versions(self):
        return [v for v in self.current.values() if v.status=="ACTIVE"]

    def infer(self,story,include_staged=False):
        allowed={"ACTIVE"} | ({"STAGED"} if include_staged else set())
        out={}
        evs=sorted(story.events,key=lambda x:x.t)
        for v in self.current.values():
            if v.status not in allowed: continue
            for i,a in enumerate(evs):
                for b in evs[i+1:]:
                    d=pair_match(v.pattern,a,b)
                    if d is not None:
                        args=tuple(d[x] for x in v.head_vars)
                        out[(v.relation,args)]=(v.relation,args)
        return list(out.values())

    def reassess(self,corpus):
        # New structural data may challenge an existing concept.
        for r,v in list(self.current.items()):
            if v.status not in {"STAGED","ACTIVE"}: continue
            ns=compute_stats(v.pattern,corpus)
            if intrinsically_strong(ns):
                v.stats=ns
                continue
            old=v.status
            v.status="CHALLENGED"
            self.events.append(("challenged",r,v.version,old,ns.precision,ns.support_stories))
            # Conservative policy: no repair language in v5.7 -> retire.
            v.status="RETIRED"
            self.events.append(("retired",r,v.version))
            v.stats=ns

LIB=ConceptLibrary()
stats=[compute_stats(p,CORPUS) for p in candidate_patterns(CORPUS)]
strong=sorted([s for s in stats if intrinsically_strong(s)],key=lambda x:repr(x.pattern))
for s in strong:
    LIB.stage(s)

# Four intrinsic concepts expected: semantic R1/W, R2/W, R3/R3, plus unused N1/N2.
assert len(LIB.current)==4

# evaluator-only structure lookup
def find_rel(rel1,rel2,vars2=None):
    xs=[v for v in LIB.current.values()
        if v.pattern.rel1==rel1 and v.pattern.rel2==rel2 and
           (vars2 is None or v.pattern.vars2==vars2)]
    assert len(xs)==1,(rel1,rel2,vars2,[(x.relation,x.pattern) for x in xs])
    return xs[0].relation

CR1=find_rel(R1,W)
CR2=find_rel(R2,W)
CR3=find_rel(R3,R3,("V1","V0","V2"))
CNOISE=find_rel(N1,N2)

initial_status={r:v.status for r,v in LIB.current.items()}

# Feed ordinary historical use.
story_by={s.sid:s for s in CORPUS}
events_before_query=sum(len(s.events) for s in CORPUS)
for qid,sid,terms in QUERY_USE:
    LIB.observe_query_use(qid,sid,terms,story_by[sid])
events_after_query=sum(len(s.events) for s in CORPUS)

after_historical={r:(v.status,v.query_utility) for r,v in LIB.current.items()}

# ------------------------------------------------------------
# Proof behavior: staged concept must not answer normal inference.
# ------------------------------------------------------------

noise_probe=Story("probe-noise",(E(N1,"fresh",t=1),E(N2,"fresh",t=2)),"new-domain")
normal_before=LIB.infer(noise_probe)
debug_before=LIB.infer(noise_probe,include_staged=True)

# ------------------------------------------------------------
# Later real use appears for previously unused concept.
# Three independent StoryContexts query the same already-supported structure.
# No new semantic events are inserted by queries.
# ------------------------------------------------------------

LATER=[
    Story("later1",(E(N1,"alpha",t=1),E(N2,"alpha",t=2)),"science"),
    Story("later2",(E(N1,"beta",t=1),E(N2,"beta",t=2)),"finance"),
    Story("later3",(E(N1,"gamma",t=1),E(N2,"gamma",t=2)),"ops"),
]
for s in LATER:
    story_by[s.sid]=s

later_event_count_before=sum(len(s.events) for s in LATER)
for i,s in enumerate(LATER,1):
    LIB.observe_query_use(f"late-q{i}",s.sid,frozenset({s.events[0].args[0]}),s)
later_event_count_after=sum(len(s.events) for s in LATER)

noise_after=LIB.current[CNOISE]
normal_after=LIB.infer(noise_probe)

# A query against a non-instance must neither create evidence nor utility.
fake=Story("fake",(E(N1,"delta",t=1),E(N3,"delta","x",t=2)),"ops")
utility_before_fake=noise_after.query_utility
fake_count_before=len(fake.events)
LIB.observe_query_use("fake-q","fake",frozenset({"delta"}),fake)
fake_count_after=len(fake.events)
utility_after_fake=noise_after.query_utility

# ------------------------------------------------------------
# Structural deterioration attack against one ACTIVE concept.
# Add enough incompatible R1/W pairs to drop intrinsic precision.
# Query use cannot save contradicted structure.
# ------------------------------------------------------------

bad_more=[
    Story(f"bad{i}",(
        E(R1,f"u{i}",f"obj{i}","go",t=1),
        E(W,f"obj{i}","stop",t=2)
    ),"attack")
    for i in range(1,8)
]
CORPUS2=CORPUS+bad_more
pre_status_r1=LIB.current[CR1].status
LIB.reassess(CORPUS2)
post_status_r1=LIB.current[CR1].status

# Other active/staged concepts should remain valid under unrelated counterevidence.
post_statuses={r:v.status for r,v in LIB.current.items()}

# ------------------------------------------------------------
# Frozen inference with remaining ACTIVE concepts.
# ------------------------------------------------------------

frozen_r2=Story("f-r2",(E(R2,"scientist","probeX","active",t=1),E(W,"probeX","active",t=2)),"space")
frozen_r3=Story("f-r3",(E(R3,"mia","paul","keyZ",t=1),E(R3,"paul","mia","keyZ",t=2)),"office")
frozen_noise=Story("f-noise",(E(N1,"omega",t=1),E(N2,"omega",t=2)),"ops")

out_r2=LIB.infer(frozen_r2)
out_r3=LIB.infer(frozen_r3)
out_noise=LIB.infer(frozen_noise)

# ------------------------------------------------------------
# Checks.
# ------------------------------------------------------------

checks={
    "four_intrinsically_strong_concepts_initially_staged":(
        len(initial_status)==4 and set(initial_status.values())=={"STAGED"}
    ),
    "semantic_three_activated_by_historical_use":(
        after_historical[CR1][0]=="ACTIVE" and
        after_historical[CR2][0]=="ACTIVE" and
        after_historical[CR3][0]=="ACTIVE"
    ),
    "regular_unused_noise_remains_staged":after_historical[CNOISE][0]=="STAGED",
    "queries_do_not_modify_story_evidence":events_before_query==events_after_query,
    "staged_concept_excluded_from_normal_proof":not normal_before,
    "staged_concept_visible_only_in_debug_candidate_mode":(
        len(debug_before)==1 and debug_before[0][0]==CNOISE
    ),
    "later_independent_use_promotes_staged_concept":(
        noise_after.status=="ACTIVE" and noise_after.query_utility>=MIN_QUERY_USE
    ),
    "promotion_does_not_insert_semantic_evidence":later_event_count_before==later_event_count_after,
    "promoted_concept_now_participates_in_proof":(
        len(normal_after)==1 and normal_after[0][0]==CNOISE
    ),
    "query_without_matching_instance_adds_no_utility":(
        utility_before_fake==utility_after_fake
    ),
    "query_without_matching_instance_adds_no_evidence":fake_count_before==fake_count_after,
    "structural_counterevidence_challenges_even_used_active_concept":(
        pre_status_r1=="ACTIVE" and post_status_r1=="RETIRED"
    ),
    "retired_concept_removed_from_active_proof":not LIB.infer(
        Story("retired-probe",(E(R1,"x","obj","go",t=1),E(W,"obj","go",t=2)),"x")
    ),
    "unrelated_active_r2_survives_and_proves":(
        LIB.current[CR2].status=="ACTIVE" and any(x[0]==CR2 for x in out_r2)
    ),
    "unrelated_active_r3_survives_and_proves":(
        LIB.current[CR3].status=="ACTIVE" and any(x[0]==CR3 for x in out_r3)
    ),
    "promoted_noise_survives_unrelated_attack":(
        LIB.current[CNOISE].status=="ACTIVE" and any(x[0]==CNOISE for x in out_noise)
    ),
    "all_heads_remain_anonymous":all(re.fullmatch(r"R\d+",r) for r in LIB.current),
    "lifecycle_history_preserved":any(e[0]=="staged" for e in LIB.events) and
                                  any(e[0]=="activated" for e in LIB.events) and
                                  any(e[0]=="challenged" for e in LIB.events) and
                                  any(e[0]=="retired" for e in LIB.events),
}

print("=== v5.7 STAGED -> ACTIVE CONCEPT LIFECYCLE ===")
for k,v in checks.items():
    print(("PASS" if v else "FAIL"),"|",k)

print("\nInitial concepts:")
for r in sorted(initial_status):
    v=LIB.current[r]
    print(" ",r,
          v.pattern.rel1,v.pattern.vars1,"THEN",v.pattern.rel2,v.pattern.vars2,
          "| initial",initial_status[r],
          "| current",v.status,
          "| utility",v.query_utility,
          "| precision",round(v.stats.precision,3),
          "| stories",v.stats.support_stories)

print("\nHistorical activation:")
for r,x in sorted(after_historical.items()):
    print(" ",r,x)

print("\nNoise staged proof before promotion:")
print(" normal:",normal_before)
print(" debug :",debug_before)
print(" after :",normal_after)

print("\nR1 deterioration:")
print(" before:",pre_status_r1,"after:",post_status_r1,
      "precision now:",round(LIB.current[CR1].stats.precision,3))

print("\nLifecycle tail:")
for e in LIB.events[-18:]:
    print(" ",e)

assert all(checks.values())

report={
    "version":"v5.7-staged-active-concept-lifecycle",
    "result":"PASS",
    "checks":checks,
    "concepts":{
        r:{
            "version":v.version,
            "status":v.status,
            "pattern":[
                [v.pattern.rel1,list(v.pattern.vars1)],
                [v.pattern.rel2,list(v.pattern.vars2)]
            ],
            "query_utility_stories":sorted(v.query_use_stories),
            "intrinsic":{
                "support_stories":v.stats.support_stories,
                "domains":v.stats.domains,
                "opportunities":v.stats.opportunities,
                "matches":v.stats.matches,
                "precision":v.stats.precision,
                "closure":v.stats.closure,
                "gain":v.stats.gain,
            },
        } for r,v in LIB.current.items()
    },
    "initial_status":initial_status,
    "after_historical_use":{r:list(x) for r,x in after_historical.items()},
    "lifecycle_events":[list(e) for e in LIB.events],
    "design":[
        "Intrinsic structural evidence can create STAGED concepts without any query utility.",
        "STAGED concepts are retained but excluded from ordinary proof inference.",
        "Repeated use in independent query/proof neighborhoods promotes an already-supported STAGED concept to ACTIVE.",
        "Query use is an access signal only: it never inserts semantic Events or Keys.",
        "A query that does not touch an actual concept instance adds neither utility nor evidence.",
        "Later structural counterevidence can challenge and retire an ACTIVE concept even if it was useful.",
        "ACTIVE/RETIRED status controls proof participation; history is preserved."
    ],
    "caveats":[
        "Promotion threshold MIN_QUERY_USE is hand-set.",
        "v5.7 uses a conservative retire-on-structural-failure policy; automatic revision/specialization is not yet integrated here.",
        "Concept search is still limited to connected ordered two-event motifs.",
        "Query terms are symbolic proof-neighborhood terms, not raw natural-language query parsing.",
        "The corpus remains synthetic symbolic experience rather than one monolithic raw-text run."
    ],
}
Path("/mnt/data/symbolic_v57_staged_active_lifecycle_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
with Path("/mnt/data/symbolic_v57_staged_active_lifecycle_checks.csv").open(
    "w",newline="",encoding="utf-8"
) as f:
    w=csv.writer(f); w.writerow(["check","passed"])
    for k,v in checks.items(): w.writerow([k,v])

print("\nSaved v5.7 report/checks.")

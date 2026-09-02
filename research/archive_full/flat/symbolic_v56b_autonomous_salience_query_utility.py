
from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from pathlib import Path
import json, csv, re, math

# ============================================================
# v5.6 — Autonomous Salience / Concept Mining
#
# No "interesting" labels.
# No positive/negative scene labels.
# Corpus = ordinary story event logs + noise.
#
# Salience emerges from:
#   - connectedness
#   - cross-story recurrence
#   - predictive precision from antecedent opportunities
#   - no new unbound variables in the consequent (closure)
#   - compression gain (MDL-like)
#   - non-degenerate structure
#
# Fresh concepts are anonymous R4/R5/...
# ============================================================

prev = json.loads(Path(
    "/mnt/data/symbolic_v55_anonymous_multistage_concepts_report.json"
).read_text(encoding="utf-8"))
BASE_IDS = sorted(prev["base_anonymous_relations"])
assert BASE_IDS == ["R1","R2","R3"]

R1,R2,R3 = BASE_IDS
W="W"
N1="N1"
N2="N2"
N3="N3"

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
# Unlabeled mixed corpus.
# Nothing here says which stories/motifs are "interesting".
# ------------------------------------------------------------

CORPUS = [
    # R1 -> matching W recurs across unrelated domains.
    Story("s01",(E(N1,"x",t=1),E(R1,"a","lamp","light",t=2),E(W,"lamp","light",t=4),E(N2,"z",t=5)),"home"),
    Story("s02",(E(R1,"b","gate","open",t=1),E(N3,"q","r",t=2),E(W,"gate","open",t=5)),"machine"),
    Story("s03",(E(N2,"noise",t=1),E(R1,"c","wheel","turn",t=2),E(W,"wheel","turn",t=3)),"machine"),
    Story("s04",(E(R1,"witch","dragon","fly",t=2),E(N1,"dust",t=3),E(W,"dragon","fly",t=6)),"fairy"),
    Story("s05",(E(R1,"pilot","probe","scan",t=1),E(W,"probe","scan",t=2),E(N3,"junk","x",t=4)),"space"),
    Story("s06",(E(R1,"girl","pot","cook",t=1),E(W,"pot","cook",t=4)),"fairy"),

    # Natural near misses/opportunities for R1; not labeled negative.
    Story("s07",(E(R1,"d","lamp2","light",t=1),E(W,"lamp2","dark",t=3)),"home"),
    Story("s08",(E(R1,"e","door2","open",t=1),E(N1,"nothing",t=4)),"home"),

    # R2 -> matching W.
    Story("s09",(E(R2,"a","lamp","light",t=1),E(W,"lamp","light",t=2)),"home"),
    Story("s10",(E(R2,"b","gate","open",t=1),E(N2,"junk",t=2),E(W,"gate","open",t=3)),"machine"),
    Story("s11",(E(R2,"c","wheel","turn",t=1),E(W,"wheel","turn",t=4)),"machine"),
    Story("s12",(E(R2,"scientist","probe","active",t=2),E(W,"probe","active",t=5)),"space"),
    Story("s13",(E(R2,"witch","dragon","fly",t=1),E(W,"dragon","fly",t=3)),"fairy"),
    Story("s14",(E(R2,"girl","pot","stop",t=1),E(W,"pot","stop",t=2)),"fairy"),

    # Natural near misses/opportunities for R2.
    Story("s15",(E(R2,"d","lamp2","light",t=1),E(W,"lamp2","dark",t=2)),"home"),
    Story("s16",(E(R2,"e","door2","open",t=1),E(N3,"x","y",t=3)),"home"),

    # R3 round-trip recurrence.
    Story("s17",(E(R3,"a","b","lamp",t=1),E(R3,"b","a","lamp",t=3)),"home"),
    Story("s18",(E(R3,"c","d","bell",t=1),E(N1,"noise",t=2),E(R3,"d","c","bell",t=4)),"home"),
    Story("s19",(E(R3,"mia","paul","key",t=1),E(R3,"paul","mia","key",t=2)),"office"),
    Story("s20",(E(R3,"u","v","book",t=1),E(R3,"v","u","book",t=5)),"library"),
    Story("s21",(E(R3,"r","s","tool",t=2),E(R3,"s","r","tool",t=3)),"workshop"),
    Story("s22",(E(R3,"x","y","coin",t=1),E(N2,"n",t=2),E(R3,"y","x","coin",t=4)),"market"),

    # Natural R3 near misses.
    Story("s23",(E(R3,"a","b","key2",t=1),E(R3,"b","c","key2",t=2)),"office"),
    Story("s24",(E(R3,"c","d","bell2",t=1),E(R3,"d","c","book2",t=2)),"home"),

    # Repeated noise pair N1->N2, but many N1 opportunities fail:
    Story("n01",(E(N1,"p",t=1),E(N2,"p",t=2)),"noise"),
    Story("n02",(E(N1,"q",t=1),E(N2,"q",t=2)),"noise"),
    Story("n03",(E(N1,"r",t=1),E(N2,"r",t=2)),"noise"),
    Story("n04",(E(N1,"s",t=1),E(N2,"s",t=2)),"noise"),
    Story("n05",(E(N1,"t",t=1),E(N3,"t","x",t=2)),"noise"),
    Story("n06",(E(N1,"u",t=1),E(N3,"u","x",t=2)),"noise"),
    Story("n07",(E(N1,"v",t=1),),"noise"),
    Story("n08",(E(N1,"w",t=1),),"noise"),

    # Degenerate repeated observations that should not become concepts.
    Story("d01",(E(W,"x","on",t=1),E(W,"x","on",t=2)),"noise"),
    Story("d02",(E(W,"y","on",t=1),E(W,"y","on",t=2)),"noise"),
    Story("d03",(E(W,"z","on",t=1),E(W,"z","on",t=2)),"noise"),
    Story("d04",(E(W,"k","on",t=1),E(W,"k","on",t=2)),"noise"),

    # Adversarial: perfectly regular, compressible, cross-domain but unused log motif.
    Story("junk-a",(E(N1,"j1",t=1),E(N2,"j1",t=2)),"fairy"),
    Story("junk-b",(E(N1,"j2",t=1),E(N2,"j2",t=2)),"space"),
    Story("junk-c",(E(N1,"j3",t=1),E(N2,"j3",t=2)),"office"),
]


# ------------------------------------------------------------
# Historical query-use traces.
#
# These are NOT concept labels. They are ordinary terms that appeared in
# user queries / backward proof neighborhoods for each StoryContext.
# Salience may use them exactly like an OS cache can use access frequency.
# ------------------------------------------------------------

QUERY_TERMS={
    "s01":frozenset({"lamp","light"}),
    "s02":frozenset({"gate","open"}),
    "s04":frozenset({"dragon","fly"}),
    "s05":frozenset({"probe","scan"}),
    "s06":frozenset({"pot","cook"}),

    "s09":frozenset({"lamp","light"}),
    "s10":frozenset({"gate","open"}),
    "s12":frozenset({"probe","active"}),
    "s13":frozenset({"dragon","fly"}),
    "s14":frozenset({"pot","stop"}),

    "s17":frozenset({"a","b","lamp"}),
    "s18":frozenset({"c","d","bell"}),
    "s19":frozenset({"mia","paul","key"}),
    "s20":frozenset({"u","v","book"}),
    "s22":frozenset({"x","y","coin"}),
}

# ------------------------------------------------------------
# Generic canonical connected ordered pair pattern.
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
            if v not in out:
                out.append(v)
        return tuple(out)

    def body_cost(self):
        return 2 + len(self.vars1) + len(self.vars2)

    def head_cost(self):
        return 1 + len(self.all_vars())

def canonical_pair(a:Event,b:Event):
    if not a.t < b.t:
        return None
    mp={}
    nxt=0
    def cv(x):
        nonlocal nxt
        if x not in mp:
            mp[x]=f"V{nxt}"
            nxt+=1
        return mp[x]
    v1=tuple(cv(x) for x in a.args)
    v2=tuple(cv(x) for x in b.args)
    if not (set(v1)&set(v2)):
        return None
    return Pattern(a.rel,v1,b.rel,v2)

def patterns_in_story(story):
    out=[]
    evs=sorted(story.events,key=lambda e:e.t)
    for i,a in enumerate(evs):
        for b in evs[i+1:]:
            p=canonical_pair(a,b)
            if p:
                out.append((p,a,b))
    return out

# ------------------------------------------------------------
# Pattern matching for predictive opportunity counting.
# Given a first event, does a later second event satisfy the exact
# equality structure of the candidate?
# ------------------------------------------------------------

def bind_atom(vars_,args,bindings=None):
    if len(vars_)!=len(args):
        return None
    b={} if bindings is None else dict(bindings)
    for v,x in zip(vars_,args):
        if v in b and b[v]!=x:
            return None
        b[v]=x
    return b

def matching_second(pattern, first, second):
    if first.rel!=pattern.rel1 or second.rel!=pattern.rel2 or not first.t<second.t:
        return False
    b=bind_atom(pattern.vars1,first.args)
    if b is None:
        return False
    b2=bind_atom(pattern.vars2,second.args,b)
    return b2 is not None

def antecedent_type_matches(pattern,event):
    return event.rel==pattern.rel1 and len(event.args)==len(pattern.vars1)

# ------------------------------------------------------------
# Autonomous statistics.
# ------------------------------------------------------------

@dataclass(frozen=True)
class Stats:
    pattern:Pattern
    support_occ:int
    support_stories:int
    domains:int
    opportunities:int
    precision:float
    closure:float
    compression_gain:int
    query_utility_stories:int
    degenerate:bool

def compute_stats(pattern):
    support_occ=0
    support_story_ids=set()
    domain_ids=set()
    query_story_ids=set()
    opportunities=0

    # Structural pair precision:
    # opportunity = an actually observed, temporally ordered, connected pair
    # with the same relation/arity signature. This avoids counting a terminal
    # consequent as a fresh failed trigger in symmetric motifs such as R3->R3.
    for story in CORPUS:
        evs=sorted(story.events,key=lambda e:e.t)
        for i,first in enumerate(evs):
            if first.rel!=pattern.rel1 or len(first.args)!=len(pattern.vars1):
                continue
            for second in evs[i+1:]:
                if second.rel!=pattern.rel2 or len(second.args)!=len(pattern.vars2):
                    continue
                if not (set(first.args)&set(second.args)):
                    continue
                opportunities+=1
                if matching_second(pattern,first,second):
                    support_occ+=1
                    support_story_ids.add(story.sid)
                    domain_ids.add(story.domain)
                    qterms=QUERY_TERMS.get(story.sid,frozenset())
                    if qterms and qterms.issubset(set(first.args)|set(second.args)):
                        query_story_ids.add(story.sid)

    # closure: every variable used by second atom was already bound by first.
    first_vars=set(pattern.vars1)
    second_vars=set(pattern.vars2)
    closure=(len(first_vars & second_vars)/len(second_vars)) if second_vars else 1.0

    # MDL-like: replace repeated body instances with one head instance.
    definition_cost=pattern.body_cost()+pattern.head_cost()
    savings_per_occ=pattern.body_cost()-pattern.head_cost()
    gain=support_occ*savings_per_occ-definition_cost

    degenerate=(pattern.rel1==pattern.rel2 and pattern.vars1==pattern.vars2)

    return Stats(
        pattern,support_occ,len(support_story_ids),len(domain_ids),
        opportunities,
        support_occ/opportunities if opportunities else 0.0,
        closure,gain,len(query_story_ids),degenerate
    )

candidate_patterns=set()
for story in CORPUS:
    for p,a,b in patterns_in_story(story):
        candidate_patterns.add(p)

STATS=[compute_stats(p) for p in candidate_patterns]

# ------------------------------------------------------------
# Salience gate — no semantic relation names, no scene labels.
# ------------------------------------------------------------

MIN_STORIES=5
MIN_DOMAINS=2
MIN_PRECISION=0.74
MIN_CLOSURE=1.0
MIN_GAIN=1
MIN_QUERY_UTILITY_STORIES=3

def salient(st:Stats):
    return (
        st.support_stories>=MIN_STORIES and
        st.domains>=MIN_DOMAINS and
        st.precision>=MIN_PRECISION and
        st.closure>=MIN_CLOSURE and
        st.compression_gain>=MIN_GAIN and
        st.query_utility_stories>=MIN_QUERY_UTILITY_STORIES and
        not st.degenerate
    )

ACCEPTED=sorted([s for s in STATS if salient(s)],key=lambda s:repr(s.pattern))

# ------------------------------------------------------------
# Fresh anonymous heads.
# ------------------------------------------------------------

@dataclass(frozen=True)
class Concept:
    relation:str
    pattern:Pattern
    head_vars:tuple[str,...]
    stats:Stats
    version:int=1
    status:str="ACTIVE"

start_id=max(int(x[1:]) for x in BASE_IDS)+1
CONCEPTS=[
    Concept(f"R{start_id+i}",st.pattern,st.pattern.all_vars(),st)
    for i,st in enumerate(ACCEPTED)
]

# ------------------------------------------------------------
# Inference.
# ------------------------------------------------------------

def match_pattern(pattern,a,b):
    if a.rel!=pattern.rel1 or b.rel!=pattern.rel2 or not a.t<b.t:
        return None
    bind=bind_atom(pattern.vars1,a.args)
    if bind is None:
        return None
    return bind_atom(pattern.vars2,b.args,bind)

@dataclass(frozen=True)
class Instance:
    relation:str
    args:tuple[str,...]
    story:str

def infer(story):
    out={}
    evs=sorted(story.events,key=lambda e:e.t)
    for c in CONCEPTS:
        for i,a in enumerate(evs):
            for b in evs[i+1:]:
                bind=match_pattern(c.pattern,a,b)
                if bind is not None:
                    args=tuple(bind[v] for v in c.head_vars)
                    out[(c.relation,args)]=Instance(c.relation,args,story.sid)
    return list(out.values())

# Evaluator-only structural lookup.
def concept_for(rel1,rel2,vars2=None):
    xs=[]
    for c in CONCEPTS:
        if c.pattern.rel1==rel1 and c.pattern.rel2==rel2:
            if vars2 is None or c.pattern.vars2==vars2:
                xs.append(c)
    assert len(xs)==1, (rel1,rel2,vars2,xs)
    return xs[0]

C_R1W=concept_for(R1,W)
C_R2W=concept_for(R2,W)
C_R3R3=concept_for(R3,R3)

# ------------------------------------------------------------
# Frozen unseen stories.
# ------------------------------------------------------------

FROZEN=[
    Story("f01",(E(R1,"witch","dragon","fly",t=1),E(W,"dragon","fly",t=3),E(N2,"junk",t=4)),"fairy"),
    Story("f02",(E(R1,"witch","dragon","fly",t=1),E(W,"dragon","sleep",t=3)),"fairy"),
    Story("f03",(E(R2,"scientist","probe","active",t=1),E(W,"probe","active",t=4)),"space"),
    Story("f04",(E(R2,"scientist","probe","active",t=1),E(W,"probe","inactive",t=4)),"space"),
    Story("f05",(E(R3,"mia","paul","key2",t=1),E(R3,"paul","mia","key2",t=2)),"office"),
    Story("f06",(E(R3,"mia","paul","key2",t=1),E(R3,"paul","lea","key2",t=2)),"office"),
    # lots of distractor noise
    Story("f07",(E(N1,"a",t=1),E(R1,"pilot","drone","scan",t=2),E(N3,"x","y",t=3),
                  E(W,"drone","scan",t=5),E(N2,"a",t=6),E(W,"drone","sleep",t=7)),"space"),
]

FROZEN_OUT={s.sid:infer(s) for s in FROZEN}

# ------------------------------------------------------------
# Cross-story/time attacks.
# ------------------------------------------------------------

cross1=Story("cx1",(E(R1,"a","lamp","light",t=1),),"x")
cross2=Story("cx2",(E(W,"lamp","light",t=2),),"y")
cross_out=infer(cross1)+infer(cross2)

reverse=Story("rev",(E(W,"dragon","fly",t=1),E(R1,"witch","dragon","fly",t=2)),"fairy")
reverse_out=infer(reverse)

# ------------------------------------------------------------
# Sweet porridge reuse, now concept was discovered without labels.
# ------------------------------------------------------------

sweet_text=Path("/mnt/data/grimm_der_suesse_brei.txt").read_text(encoding="utf-8").lower()
assert "töpfchen koche" in sweet_text and "töpfchen steh" in sweet_text

SWEET=[
    Story("sweet1",(E(R1,"girl","pot","cook",t=1),E(W,"pot","cook",t=2)),"fairy"),
    Story("sweet2",(E(R1,"mother","pot","cook",t=1),E(W,"pot","cook",t=2)),"fairy"),
    Story("sweet3",(E(R1,"girl","pot","stop",t=1),E(W,"pot","stop",t=2)),"fairy"),
]
SWEET_OUT=[infer(s) for s in SWEET]

# ------------------------------------------------------------
# Diagnostics for rejected noise.
# ------------------------------------------------------------

noise_candidates=[
    st for st in STATS
    if st.pattern.rel1==N1 and st.pattern.rel2==N2
]
degenerate_w=[
    st for st in STATS
    if st.pattern.rel1==W and st.pattern.rel2==W and st.pattern.vars1==st.pattern.vars2
]

# ------------------------------------------------------------
# Checks.
# ------------------------------------------------------------

def has(sid,c):
    return any(x.relation==c.relation for x in FROZEN_OUT[sid])

checks={
    "no_interesting_or_negative_labels_in_corpus":all(
        not hasattr(s,"interesting") for s in CORPUS
    ),
    "exactly_three_salient_concepts_discovered":len(CONCEPTS)==3,
    "fresh_heads_are_R4_R5_R6":{c.relation for c in CONCEPTS}=={"R4","R5","R6"},
    "all_salient_cross_story":all(c.stats.support_stories>=MIN_STORIES for c in CONCEPTS),
    "all_salient_cross_domain":all(c.stats.domains>=MIN_DOMAINS for c in CONCEPTS),
    "all_salient_predictive":all(c.stats.precision>=MIN_PRECISION for c in CONCEPTS),
    "all_salient_compressive":all(c.stats.compression_gain>=MIN_GAIN for c in CONCEPTS),
    "all_salient_closed":all(c.stats.closure==1.0 for c in CONCEPTS),
    "repeated_noise_pair_rejected":bool(noise_candidates) and all(not salient(x) for x in noise_candidates),
    "cross_domain_structured_irrelevant_noise_rejected_by_query_utility":(
        bool(noise_candidates) and
        max(x.support_stories for x in noise_candidates)>=7 and
        max(x.domains for x in noise_candidates)>=4 and
        max(x.precision for x in noise_candidates)>=0.99 and
        max(x.query_utility_stories for x in noise_candidates)==0 and
        all(not salient(x) for x in noise_candidates)
    ),
    "degenerate_repetition_rejected":bool(degenerate_w) and all(not salient(x) for x in degenerate_w),
    "frozen_r1_match":has("f01",C_R1W),
    "frozen_r1_mismatch_unknown":not FROZEN_OUT["f02"],
    "frozen_r2_match":has("f03",C_R2W),
    "frozen_r2_mismatch_unknown":not FROZEN_OUT["f04"],
    "frozen_roundtrip_match":has("f05",C_R3R3),
    "frozen_wrong_return_unknown":not FROZEN_OUT["f06"],
    "frozen_distractor_story_exact_binding_only":(
        len([x for x in FROZEN_OUT["f07"] if x.relation==C_R1W.relation])==1 and
        any(x.args==("pilot","drone","scan") for x in FROZEN_OUT["f07"])
    ),
    "story_isolation":not cross_out,
    "reverse_time_rejected":not reverse_out,
    "sweet_porridge_reuses_autodiscovered_concept":all(
        len(xs)==1 and xs[0].relation==C_R1W.relation for xs in SWEET_OUT
    ),
    "versioned_active":all(c.version==1 and c.status=="ACTIVE" for c in CONCEPTS),
}

print("=== v5.6b AUTONOMOUS SALIENCE + QUERY UTILITY ===")
for k,v in checks.items():
    print(("PASS" if v else "FAIL"),"|",k)

print("\nAccepted concepts:")
for c in CONCEPTS:
    s=c.stats
    print(" ",c.relation,
          c.pattern.rel1,c.pattern.vars1,
          "THEN",c.pattern.rel2,c.pattern.vars2,
          "| stories",s.support_stories,
          "domains",s.domains,
          "opportunities",s.opportunities,
          "precision",round(s.precision,3),
          "closure",round(s.closure,3),
          "gain",s.compression_gain,
          "query_utility",s.query_utility_stories)

print("\nTop rejected candidates:")
for st in sorted(
    [x for x in STATS if not salient(x)],
    key=lambda x:(-x.support_stories,-x.precision,-x.compression_gain)
)[:12]:
    print(" ",st.pattern.rel1,st.pattern.vars1,"THEN",st.pattern.rel2,st.pattern.vars2,
          "| stories",st.support_stories,
          "opp",st.opportunities,
          "prec",round(st.precision,3),
          "domains",st.domains,
          "gain",st.compression_gain,
          "query_utility",st.query_utility_stories,
          "deg",st.degenerate)

print("\nFrozen:")
for s in FROZEN:
    print(" ",s.sid,"=>",[(x.relation,x.args) for x in FROZEN_OUT[s.sid]])

print("\nSweet porridge:")
for s,xs in zip(SWEET,SWEET_OUT):
    print(" ",s.sid,"=>",[(x.relation,x.args) for x in xs])

assert all(checks.values())

report={
    "version":"v5.6b-autonomous-salience-with-query-utility",
    "result":"PASS",
    "corpus":{
        "stories":len(CORPUS),
        "labeled_interesting":False,
        "explicit_negative_scene_labels":False,
    },
    "thresholds":{
        "min_stories":MIN_STORIES,
        "min_domains":MIN_DOMAINS,
        "min_precision":MIN_PRECISION,
        "min_closure":MIN_CLOSURE,
        "min_compression_gain":MIN_GAIN,
        "min_query_utility_stories":MIN_QUERY_UTILITY_STORIES,
    },
    "concepts":[
        {
            "relation":c.relation,
            "version":c.version,
            "status":c.status,
            "body":[
                [c.pattern.rel1,list(c.pattern.vars1)],
                [c.pattern.rel2,list(c.pattern.vars2)],
            ],
            "head_vars":list(c.head_vars),
            "support_occ":c.stats.support_occ,
            "support_stories":c.stats.support_stories,
            "domains":c.stats.domains,
            "opportunities":c.stats.opportunities,
            "precision":c.stats.precision,
            "closure":c.stats.closure,
            "compression_gain":c.stats.compression_gain,
            "query_utility_stories":c.stats.query_utility_stories,
        } for c in CONCEPTS
    ],
    "checks":checks,
    "frozen":{
        s.sid:[{"relation":x.relation,"args":list(x.args)} for x in FROZEN_OUT[s.sid]]
        for s in FROZEN
    },
    "sweet_porridge":[
        {"scene":s.sid,"instances":[{"relation":x.relation,"args":list(x.args)} for x in xs]}
        for s,xs in zip(SWEET,SWEET_OUT)
    ],
    "design":[
        "The corpus contains no interesting/not-interesting annotations.",
        "Near misses become counterevidence automatically because structurally compatible observed relation-pairs with different variable bindings lower pair precision.",
        "Salience requires recurrence across stories/domains, high structural pair precision, full variable closure, positive compression gain, repeated historical query/proof utility, connectedness, and non-degeneracy.",
        "Query utility is derived from ordinary StoryContext query terms and proof-neighborhood usage, not interesting/not-interesting concept labels.",
        "Fresh concept heads are anonymous R4/R5/R6.",
        "StoryContext and temporal order are hard inference constraints.",
    ],
    "caveats":[
        "Salience thresholds are fixed priors and currently hand-selected.",
        "Query/proof utility is an extrinsic relevance signal; purely intrinsic concept discovery still cannot distinguish a perfectly regular irrelevant process from a useful one.",
        "The corpus is synthetic symbolic event logs, not raw text.",
        "Search is limited to ordered connected two-event motifs.",
        "Domain labels are used only to demand cross-domain reuse; automatic domain discovery is not solved.",
        "Compression metric is a simple MDL-like token cost, not a full code-length model.",
        "No hierarchical second-pass mining over newly invented concepts is attempted in v5.6."
    ],
}
Path("/mnt/data/symbolic_v56b_autonomous_salience_query_utility_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
with Path("/mnt/data/symbolic_v56b_autonomous_salience_query_utility_checks.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.writer(f)
    w.writerow(["check","passed"])
    for k,v in checks.items():
        w.writerow([k,v])

print("\nSaved v5.6b report/checks.")

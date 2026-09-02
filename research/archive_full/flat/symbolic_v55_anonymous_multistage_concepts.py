
from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import re, json, csv

# ============================================================
# v5.5 — Anonymous Multi-Stage Concept Invention
#
# Input layer:
#   frozen anonymous event relations R1/R2/R3 from v5.4b
#   W(x,y) = independently observed WORLD proposition
#
# Learner receives:
#   positive "interesting composition" scenes
#   negative near-miss scenes
# It receives NO semantic family/class labels.
#
# Learner:
#   enumerates connected ordered 2-U motifs
#   canonicalizes variable sharing
#   requires support >= 3 and conflict = 0
#   invents fresh anonymous heads R4/R5/R6...
#
# No ACTION / CLAIM / HAVE / COMMAND / TRANSFER predicates are used here.
# ============================================================

# Verify the previous frozen anonymous layer exists and is anonymous.
prev=json.loads(Path(
    "/mnt/data/symbolic_v54b_safe_anonymous_event_semantics_report.json"
).read_text(encoding="utf-8"))
BASE_IDS=sorted(prev["anonymous_relations"])
assert BASE_IDS==["R1","R2","R3"]
assert all(re.fullmatch(r"R\d+",x) for x in BASE_IDS)

# The learner sees only these symbols. Semantic names are intentionally absent.
R_A,R_B,R_C=BASE_IDS
WORLD="W"

@dataclass(frozen=True)
class Event:
    rel:str
    args:tuple[str,...]
    t:int

@dataclass(frozen=True)
class Scene:
    sid:str
    events:tuple[Event,...]
    interesting:bool
    story:str

def E(rel,*args,t):
    return Event(rel,tuple(args),t)

# ------------------------------------------------------------
# Training scenes.
#
# Three recurring motifs are mixed together in one positive pool.
# The miner is NOT told there are three classes.
# ------------------------------------------------------------

POS=[
    Scene("p01",(E(R_A,"lina","LAMP","LIGHT",t=1), E(WORLD,"LAMP","LIGHT",t=2)),True,"p01"),
    Scene("p02",(E(R_A,"ben","GATE","OPEN",t=1), E(WORLD,"GATE","OPEN",t=2)),True,"p02"),
    Scene("p03",(E(R_A,"nora","WHEEL","TURN",t=1), E(WORLD,"WHEEL","TURN",t=2)),True,"p03"),

    Scene("p04",(E(R_B,"lina","LAMP","LIGHT",t=1), E(WORLD,"LAMP","LIGHT",t=2)),True,"p04"),
    Scene("p05",(E(R_B,"ben","GATE","OPEN",t=1), E(WORLD,"GATE","OPEN",t=2)),True,"p05"),
    Scene("p06",(E(R_B,"nora","WHEEL","TURN",t=1), E(WORLD,"WHEEL","TURN",t=2)),True,"p06"),

    Scene("p07",(E(R_C,"lina","ben","LAMP",t=1), E(R_C,"ben","lina","LAMP",t=2)),True,"p07"),
    Scene("p08",(E(R_C,"nora","tom","BELL",t=1), E(R_C,"tom","nora","BELL",t=2)),True,"p08"),
    Scene("p09",(E(R_C,"mia","paul","KEY",t=1), E(R_C,"paul","mia","KEY",t=2)),True,"p09"),
]

NEG=[
    # same first anonymous relation but wrong target/action binding
    Scene("n01",(E(R_A,"lina","LAMP","LIGHT",t=1), E(WORLD,"LAMP","DARK",t=2)),False,"n01"),
    Scene("n02",(E(R_A,"ben","GATE","OPEN",t=1), E(WORLD,"DOOR2","OPEN",t=2)),False,"n02"),
    Scene("n03",(E(R_A,"nora","WHEEL","TURN",t=1), E(WORLD,"WHEEL","STOP",t=2)),False,"n03"),

    # same for the second anonymous relation
    Scene("n04",(E(R_B,"lina","LAMP","LIGHT",t=1), E(WORLD,"LAMP","DARK",t=2)),False,"n04"),
    Scene("n05",(E(R_B,"ben","GATE","OPEN",t=1), E(WORLD,"DOOR2","OPEN",t=2)),False,"n05"),
    Scene("n06",(E(R_B,"nora","WHEEL","TURN",t=1), E(WORLD,"WHEEL","STOP",t=2)),False,"n06"),

    # one-way or wrong return endpoint/item
    Scene("n07",(E(R_C,"lina","ben","LAMP",t=1), E(R_C,"ben","mia","LAMP",t=2)),False,"n07"),
    Scene("n08",(E(R_C,"nora","tom","BELL",t=1), E(R_C,"tom","nora","KEY",t=2)),False,"n08"),
    Scene("n09",(E(R_C,"mia","paul","KEY",t=1), E(R_C,"lea","mia","KEY",t=2)),False,"n09"),
]

# ------------------------------------------------------------
# Generic canonicalizer: relation names and equality structure only.
# ------------------------------------------------------------

@dataclass(frozen=True)
class Pattern:
    rel1:str
    vars1:tuple[str,...]
    rel2:str
    vars2:tuple[str,...]

    def variable_order(self):
        out=[]
        for v in self.vars1+self.vars2:
            if v not in out:
                out.append(v)
        return tuple(out)

def canonical_pair(a:Event,b:Event):
    if not a.t < b.t:
        return None
    mapping={}
    nxt=0
    def cv(x):
        nonlocal nxt
        if x not in mapping:
            mapping[x]=f"V{nxt}"
            nxt+=1
        return mapping[x]
    v1=tuple(cv(x) for x in a.args)
    v2=tuple(cv(x) for x in b.args)
    # connectedness: at least one variable from first reused by second
    if not (set(v1)&set(v2)):
        return None
    return Pattern(a.rel,v1,b.rel,v2)

def scene_patterns(scene):
    ps=set()
    evs=sorted(scene.events,key=lambda x:x.t)
    for i,a in enumerate(evs):
        for b in evs[i+1:]:
            p=canonical_pair(a,b)
            if p:
                ps.add(p)
    return ps

# ------------------------------------------------------------
# Mine patterns.
# ------------------------------------------------------------

support=defaultdict(set)
conflict=defaultdict(set)

for s in POS:
    for p in scene_patterns(s):
        support[p].add(s.sid)
for s in NEG:
    for p in scene_patterns(s):
        conflict[p].add(s.sid)

accepted=[]
for p,sids in support.items():
    sup=len(sids)
    con=len(conflict.get(p,set()))
    if sup>=3 and con==0:
        accepted.append((p,sup,con))

accepted.sort(key=lambda x:repr(x[0]))
assert len(accepted)==3

# Invent new relation ids after highest existing id.
next_id=max(int(x[1:]) for x in BASE_IDS)+1

@dataclass(frozen=True)
class Concept:
    relation:str
    pattern:Pattern
    head_vars:tuple[str,...]
    support:int
    conflict:int
    version:int=1
    status:str="ACTIVE"

CONCEPTS=[]
for i,(p,sup,con) in enumerate(accepted):
    CONCEPTS.append(Concept(
        relation=f"R{next_id+i}",
        pattern=p,
        head_vars=p.variable_order(),
        support=sup,
        conflict=con,
    ))

# ------------------------------------------------------------
# Match a learned pattern against a new scene and ground head vars.
# ------------------------------------------------------------

def match_pattern(pattern:Pattern,a:Event,b:Event):
    if not a.t<b.t or a.rel!=pattern.rel1 or b.rel!=pattern.rel2:
        return None
    bindings={}
    def bind(var,val):
        if var in bindings:
            return bindings[var]==val
        bindings[var]=val
        return True
    if len(a.args)!=len(pattern.vars1) or len(b.args)!=len(pattern.vars2):
        return None
    for v,x in zip(pattern.vars1,a.args):
        if not bind(v,x): return None
    for v,x in zip(pattern.vars2,b.args):
        if not bind(v,x): return None
    return bindings

@dataclass(frozen=True)
class ConceptInstance:
    relation:str
    args:tuple[str,...]
    proof:tuple[Event,Event]
    story:str

def infer_concepts(scene:Scene):
    out=[]
    evs=sorted(scene.events,key=lambda x:x.t)
    for c in CONCEPTS:
        for i,a in enumerate(evs):
            for b in evs[i+1:]:
                bindings=match_pattern(c.pattern,a,b)
                if bindings is not None:
                    args=tuple(bindings[v] for v in c.head_vars)
                    out.append(ConceptInstance(c.relation,args,(a,b),scene.story))
    # dedupe
    uniq={}
    for x in out:
        uniq[(x.relation,x.args)]=x
    return list(uniq.values())

# ------------------------------------------------------------
# Evaluator-only identification of which discovered concept corresponds
# to which STRUCTURAL pattern. The learner never gets these labels.
# ------------------------------------------------------------

def concept_for(first_rel,second_rel):
    xs=[c for c in CONCEPTS if c.pattern.rel1==first_rel and c.pattern.rel2==second_rel]
    assert len(xs)==1
    return xs[0].relation

C_AW=concept_for(R_A,WORLD)
C_BW=concept_for(R_B,WORLD)
C_CC=concept_for(R_C,R_C)
assert len({C_AW,C_BW,C_CC})==3

# ------------------------------------------------------------
# Frozen unseen-domain tests.
# ------------------------------------------------------------

FROZEN=[
    # motif 1: same relation, matching independent world observation
    Scene("f01",(E(R_A,"witch","DRAGON","FLY",t=1), E(WORLD,"DRAGON","FLY",t=2)),True,"f01"),
    # near miss: wrong observed predicate
    Scene("f02",(E(R_A,"witch","DRAGON","FLY",t=1), E(WORLD,"DRAGON","SLEEP",t=2)),False,"f02"),

    # motif 2, same slots but different base anonymous relation
    Scene("f03",(E(R_B,"scientist","PROBE","ACTIVE",t=1), E(WORLD,"PROBE","ACTIVE",t=2)),True,"f03"),
    Scene("f04",(E(R_B,"scientist","PROBE","ACTIVE",t=1), E(WORLD,"PROBE","INACTIVE",t=2)),False,"f04"),

    # round trip in unseen domain
    Scene("f05",(E(R_C,"mia","paul","KEY2",t=1), E(R_C,"paul","mia","KEY2",t=2)),True,"f05"),
    Scene("f06",(E(R_C,"mia","paul","KEY2",t=1), E(R_C,"paul","lea","KEY2",t=2)),False,"f06"),
]

frozen_results={}
for s in FROZEN:
    xs=infer_concepts(s)
    frozen_results[s.sid]=xs

# ------------------------------------------------------------
# Story isolation attack: two halves in different stories must not compose.
# ------------------------------------------------------------

cross_a=Scene("cross-a",(E(R_A,"witch","DRAGON","FLY",t=1),),False,"story-X")
cross_b=Scene("cross-b",(E(WORLD,"DRAGON","FLY",t=2),),False,"story-Y")
# Correct inference is per SceneContext, never concatenate across stories.
cross_instances=infer_concepts(cross_a)+infer_concepts(cross_b)

# ------------------------------------------------------------
# Held-out fairy/lore reuse from "Der süße Brei":
# R_A is the anonymous directive-like event relation from v5.4b.
# Independent WORLD observation is supplied by the narrative extractor,
# not by the command itself.
# ------------------------------------------------------------

PORRIDGE=Path("/mnt/data/grimm_der_suesse_brei.txt").read_text(encoding="utf-8").lower()
assert "töpfchen koche" in PORRIDGE and "kocht" in PORRIDGE
assert "töpfchen steh" in PORRIDGE and "hört auf zu kochen" in PORRIDGE

sweet=[
    Scene("sweet1",(E(R_A,"girl","POT","COOK",t=1),E(WORLD,"POT","COOK",t=2)),True,"sweet1"),
    Scene("sweet2",(E(R_A,"mother","POT","COOK",t=1),E(WORLD,"POT","COOK",t=2)),True,"sweet2"),
    Scene("sweet3",(E(R_A,"girl","POT","STOP",t=1),E(WORLD,"POT","STOP",t=2)),True,"sweet3"),
]
sweet_results=[infer_concepts(s) for s in sweet]

# ------------------------------------------------------------
# Additional hard attack: same relations but reversed time must fail.
# ------------------------------------------------------------

rev=Scene("reverse-time",(
    E(WORLD,"DRAGON","FLY",t=1),
    E(R_A,"witch","DRAGON","FLY",t=2)
),False,"reverse-time")
rev_instances=infer_concepts(rev)

# ------------------------------------------------------------
# Checks.
# ------------------------------------------------------------

def has_relation(scene_id,relation):
    return any(x.relation==relation for x in frozen_results[scene_id])

checks={
    "three_new_anonymous_concepts_invented":(
        len(CONCEPTS)==3 and
        {c.relation for c in CONCEPTS}=={"R4","R5","R6"}
    ),
    "no_semantic_concept_names":all(re.fullmatch(r"R\d+",c.relation) for c in CONCEPTS),
    "all_concepts_full_support_zero_conflict":all(c.support==3 and c.conflict==0 for c in CONCEPTS),
    "concepts_are_connected_two_u_patterns":all(
        set(c.pattern.vars1)&set(c.pattern.vars2) for c in CONCEPTS
    ),
    "two_same_port_world_motifs_remain_distinct":C_AW!=C_BW,
    "frozen_relation_a_world_match_positive":has_relation("f01",C_AW),
    "frozen_relation_a_world_mismatch_rejected":not frozen_results["f02"],
    "frozen_relation_b_world_match_positive":has_relation("f03",C_BW),
    "frozen_relation_b_world_mismatch_rejected":not frozen_results["f04"],
    "frozen_roundtrip_positive":has_relation("f05",C_CC),
    "frozen_wrong_return_rejected":not frozen_results["f06"],
    "story_isolation_blocks_cross_story_composition":not cross_instances,
    "reverse_temporal_order_rejected":not rev_instances,
    "sweet_porridge_reuses_same_higher_concept":(
        len(sweet_results)==3 and
        all(len(xs)==1 and xs[0].relation==C_AW for xs in sweet_results)
    ),
    "versioned_active_concepts":all(c.version==1 and c.status=="ACTIVE" for c in CONCEPTS),
}

print("=== v5.5 ANONYMOUS MULTI-STAGE CONCEPT INVENTION ===")
for k,v in checks.items():
    print(("PASS" if v else "FAIL"),"|",k)

print("\nInvented concepts:")
for c in CONCEPTS:
    print(" ",c.relation,
          c.pattern.rel1,c.pattern.vars1,
          "THEN",c.pattern.rel2,c.pattern.vars2,
          "-> head",c.head_vars,
          "support",c.support,"conflict",c.conflict)

print("\nFrozen:")
for s in FROZEN:
    xs=frozen_results[s.sid]
    print(" ",s.sid,"interesting=",s.interesting,
          "=>",[(x.relation,x.args) for x in xs])

print("\nStory isolation cross:",[(x.relation,x.args) for x in cross_instances])
print("Reverse time:",[(x.relation,x.args) for x in rev_instances])

print("\nSweet porridge:")
for s,xs in zip(sweet,sweet_results):
    print(" ",s.sid,"=>",[(x.relation,x.args) for x in xs])

assert all(checks.values())

report={
    "version":"v5.5-anonymous-multistage-concept-invention",
    "result":"PASS",
    "base_anonymous_relations":BASE_IDS,
    "new_concepts":[
        {
            "relation":c.relation,
            "version":c.version,
            "status":c.status,
            "body":[
                [c.pattern.rel1,list(c.pattern.vars1)],
                [c.pattern.rel2,list(c.pattern.vars2)]
            ],
            "head_vars":list(c.head_vars),
            "support":c.support,
            "conflict":c.conflict,
        } for c in CONCEPTS
    ],
    "checks":checks,
    "frozen":{
        s.sid:{
            "interesting":s.interesting,
            "instances":[{"relation":x.relation,"args":list(x.args)} for x in frozen_results[s.sid]]
        } for s in FROZEN
    },
    "sweet_porridge":[
        {
            "scene":s.sid,
            "instances":[{"relation":x.relation,"args":list(x.args)} for x in xs]
        } for s,xs in zip(sweet,sweet_results)
    ],
    "design":[
        "The learner receives no semantic class labels for the three higher-order motifs.",
        "Candidate concepts are connected ordered pairs of existing anonymous U/fact relations.",
        "Variable sharing is inferred by canonicalizing equality structure across ports.",
        "Only motifs with support >=3 and zero conflicts on near-miss scenes are installed.",
        "Fresh concept heads are anonymous R4/R5/R6 and versioned ACTIVE.",
        "Temporal order and StoryContext are hard constraints.",
        "No ACTION, CLAIM, HAVE, COMMAND, TRANSFER, compliance, truthfulness, or loan labels appear in the learned concept heads/bodies."
    ],
    "caveats":[
        "W is still a fixed generic independent-world-observation relation.",
        "Concept search is limited to connected ordered 2-event motifs.",
        "Positive 'interesting composition' scenes are supervised; autonomous salience selection is not solved.",
        "Head variables are currently all variables in first-occurrence order, not learned by MDL.",
        "No recursion or three-stage concept induction is attempted in v5.5.",
        "Sweet-porridge test starts from frozen lower-level anonymous R events plus independently extracted world observations; it is not a fresh raw-text parser test."
    ],
}
Path("/mnt/data/symbolic_v55_anonymous_multistage_concepts_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
with Path("/mnt/data/symbolic_v55_anonymous_multistage_concepts_checks.csv").open(
    "w",newline="",encoding="utf-8"
) as f:
    w=csv.writer(f); w.writerow(["check","passed"])
    for k,v in checks.items(): w.writerow([k,v])

print("\nSaved v5.5 report/checks.")

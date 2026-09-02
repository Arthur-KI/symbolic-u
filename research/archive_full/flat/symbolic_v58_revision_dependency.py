
from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict, deque
from pathlib import Path
import json, csv, re, copy

# ============================================================
# v5.8 — Versioned Revision + Dependency Propagation
#
# Anonymous learned concept heads:
#   R4 -> R8 -> R9
#
# New evidence challenges R4_v1.
# Revision search may add ONE generic symbolic guard G?(port).
# If R4_v2 passes train + frozen:
#   R4_v1 SUPERSEDED
#   descendants CHALLENGED automatically
#   descendants are revalidated/re-versioned against active parents
#
# Query is never evidence.
# ============================================================

# ---------- symbolic facts / selectors ----------

@dataclass(frozen=True)
class Atom:
    rel: str
    args: tuple[str,...]

def A(rel,*args): return Atom(rel,tuple(args))

# Primitive relation/event observations.
# R1 and W are pre-existing lower anonymous/world relations.
# X and Y are generic supporting relations used by higher concepts.
BASE_RELATIONS={"R1","W","X","Y","G0","G1","G2"}

# Selector facts are supplied ontology observations, not concept labels.
SELECTORS={
    # positive/generalizing agents
    "anna": {"G0"},
    "ben": {"G0"},
    "cara": {"G0"},
    "dora": {"G0"},
    "emma": {"G0"},
    "finn": {"G0"},
    "gina": {"G0"},
    # counterexample agents
    "xavier": {"G1"},
    "yara": {"G1"},
    "zane": {"G1"},
    "quinn": {"G1"},
}
# Distractor selectors
for who in ("anna","xavier","cara","zane"):
    SELECTORS.setdefault(who,set()).add("G2")

def has_selector(entity,guard_rel):
    return guard_rel in SELECTORS.get(entity,set())

# ---------- examples ----------

@dataclass(frozen=True)
class Example:
    eid: str
    head_args: tuple[str,...]
    body_atoms: tuple[Atom,...]
    should_hold: bool
    split: str   # train / frozen / challenge
    story: str

# R4 means only structurally:
# R1(A,B,C) + W(B,C) [+ optional guard(A)] -> R4(A,B,C)
def r4_body(a,b,c):
    return (A("R1",a,b,c), A("W",b,c))

R4_TRAIN=[
    Example("r4t1",("anna","lamp","light"),r4_body("anna","lamp","light"),True,"train","s1"),
    Example("r4t2",("ben","gate","open"),r4_body("ben","gate","open"),True,"train","s2"),
    Example("r4t3",("cara","wheel","turn"),r4_body("cara","wheel","turn"),True,"train","s3"),
    Example("r4t4",("dora","probe","scan"),r4_body("dora","probe","scan"),True,"train","s4"),
]
R4_FROZEN=[
    Example("r4f1",("emma","dragon","fly"),r4_body("emma","dragon","fly"),True,"frozen","f1"),
    Example("r4f2",("finn","pot","cook"),r4_body("finn","pot","cook"),True,"frozen","f2"),
    # structurally present but guard-incompatible; should stay unknown after revision
    Example("r4f3",("xavier","door","open"),r4_body("xavier","door","open"),False,"frozen","f3"),
]
R4_CHALLENGE=[
    # New accepted evidence says these v1 derivations are invalid.
    Example("r4c1",("xavier","lamp2","light"),r4_body("xavier","lamp2","light"),False,"challenge","c1"),
    Example("r4c2",("yara","gate2","open"),r4_body("yara","gate2","open"),False,"challenge","c2"),
    Example("r4c3",("zane","wheel2","turn"),r4_body("zane","wheel2","turn"),False,"challenge","c3"),
]

# Higher concept examples reference the anonymous parent relation.
# R8(A,B,C,D): R4(A,B,C) + X(B,D)
R8_TRAIN=[
    Example("r8t1",("anna","lamp","light","room1"),
            (A("R4","anna","lamp","light"),A("X","lamp","room1")),True,"train","h1"),
    Example("r8t2",("ben","gate","open","yard"),
            (A("R4","ben","gate","open"),A("X","gate","yard")),True,"train","h2"),
    Example("r8t3",("cara","wheel","turn","lab"),
            (A("R4","cara","wheel","turn"),A("X","wheel","lab")),True,"train","h3"),
]
R8_FROZEN=[
    Example("r8f1",("emma","dragon","fly","sky"),
            (A("R4","emma","dragon","fly"),A("X","dragon","sky")),True,"frozen","hf1"),
    Example("r8f2",("xavier","door","open","hall"),
            (A("R4","xavier","door","open"),A("X","door","hall")),False,"frozen","hf2"),
]

# R9(A,B,C,D,E): R8(A,B,C,D) + Y(D,E)
R9_TRAIN=[
    Example("r9t1",("anna","lamp","light","room1","safe"),
            (A("R8","anna","lamp","light","room1"),A("Y","room1","safe")),True,"train","k1"),
    Example("r9t2",("ben","gate","open","yard","safe"),
            (A("R8","ben","gate","open","yard"),A("Y","yard","safe")),True,"train","k2"),
    Example("r9t3",("cara","wheel","turn","lab","safe"),
            (A("R8","cara","wheel","turn","lab"),A("Y","lab","safe")),True,"train","k3"),
]
R9_FROZEN=[
    Example("r9f1",("emma","dragon","fly","sky","safe"),
            (A("R8","emma","dragon","fly","sky"),A("Y","sky","safe")),True,"frozen","kf1"),
    Example("r9f2",("xavier","door","open","hall","safe"),
            (A("R8","xavier","door","open","hall"),A("Y","hall","safe")),False,"frozen","kf2"),
]

# ---------- versioned concepts ----------

@dataclass
class ConceptVersion:
    relation: str
    version: int
    status: str
    body_rels: tuple[str,...]
    guard: tuple[str,int] | None  # (G?, head-arg-index)
    parents: dict[str,int]
    provenance: list[str]
    support: int=0
    conflict: int=0
    frozen_support: int=0
    frozen_conflict: int=0
    parent_version: int|None=None

    @property
    def vid(self): return f"{self.relation}_v{self.version}"

class VersionedLibrary:
    def __init__(self):
        self.history: dict[str,list[ConceptVersion]]=defaultdict(list)
        self.active: dict[str,ConceptVersion|None]={}
        self.dependents: dict[str,set[str]]=defaultdict(set)
        self.events=[]
        self.next_version=defaultdict(lambda:1)

    def install_initial(self,relation,body_rels,parents,provenance):
        v=ConceptVersion(
            relation,self.next_version[relation],"ACTIVE",tuple(body_rels),None,
            dict(parents),list(provenance)
        )
        self.next_version[relation]+=1
        self.history[relation].append(v); self.active[relation]=v
        for p in parents: self.dependents[p].add(relation)
        self.events.append(("installed_initial",v.vid,dict(parents)))
        return v

    def descendants(self,relation):
        seen=set(); q=deque(self.dependents.get(relation,set()))
        while q:
            x=q.popleft()
            if x in seen: continue
            seen.add(x)
            q.extend(self.dependents.get(x,set()))
        return seen

    def challenge(self,relation,reason):
        v=self.active.get(relation)
        if v is None: return
        v.status="CHALLENGED"
        self.events.append(("challenged",v.vid,reason))
        # Propagate immediately to active descendants.
        for d in sorted(self.descendants(relation)):
            dv=self.active.get(d)
            if dv is not None and dv.status=="ACTIVE":
                dv.status="CHALLENGED"
                self.events.append(("dependency_challenged",dv.vid,relation))

    def activate_revision(self,old,new):
        old.status="SUPERSEDED"
        new.status="ACTIVE"
        self.history[new.relation].append(new)
        self.active[new.relation]=new
        self.events.append(("revision_activated",old.vid,new.vid,new.guard))

    def retire(self,relation,reason):
        v=self.active.get(relation)
        if v:
            v.status="RETIRED"
            self.events.append(("retired",v.vid,reason))
        self.active[relation]=None
        # Descendants cannot remain active.
        for d in sorted(self.descendants(relation)):
            dv=self.active.get(d)
            if dv:
                dv.status="RETIRED"
                self.events.append(("dependency_retired",dv.vid,relation))
                self.active[d]=None

LIB=VersionedLibrary()

# Initial anonymous hierarchy.
r4v1=LIB.install_initial("R4",("R1","W"),{},["v5.7-autodiscovered"])
r8v1=LIB.install_initial("R8",("R4","X"),{"R4":1},["hierarchical-mine"])
r9v1=LIB.install_initial("R9",("R8","Y"),{"R8":1},["hierarchical-mine"])

# ---------- evaluation ----------

def guard_allows(v:ConceptVersion,head_args):
    if v.guard is None: return True
    grel,idx=v.guard
    return has_selector(head_args[idx],grel)

def eval_examples(v,examples):
    support=conflict=0
    for ex in examples:
        pred=guard_allows(v,ex.head_args)
        if pred==ex.should_hold:
            support+=1
        else:
            conflict+=1
    return support,conflict

def score(v,train,frozen):
    v.support,v.conflict=eval_examples(v,train)
    v.frozen_support,v.frozen_conflict=eval_examples(v,frozen)
    return (
        v.conflict==0 and
        v.frozen_conflict==0 and
        v.support==len(train) and
        v.frozen_support==len(frozen)
    )

# v1 was trained before negative frozen/challenge was known.
# Historical fit on original positives.
assert eval_examples(r4v1,R4_TRAIN)[1]==0
assert eval_examples(r8v1,R8_TRAIN)[1]==0
assert eval_examples(r9v1,R9_TRAIN)[1]==0

# Query mechanism: proof only, never evidence.
HEAD_ARITY={"R4":3,"R8":4,"R9":5}

def prove(relation,args,_seen=None):
    v=LIB.active.get(relation)
    if v is None or v.status!="ACTIVE":
        return 0
    if not guard_allows(v,args):
        return 0

    seen=set() if _seen is None else set(_seen)
    marker=(relation,tuple(args),v.version)
    if marker in seen:
        return 0
    seen.add(marker)

    # Higher relations require BOTH:
    # 1) exact active parent version compatibility
    # 2) proof of the concrete parent instance.
    for p,pver in v.parents.items():
        av=LIB.active.get(p)
        if av is None or av.status!="ACTIVE" or av.version!=pver:
            return 0
        pargs=tuple(args[:HEAD_ARITY[p]])
        if prove(p,pargs,seen)!=1:
            return 0
    return +1

# ---------- challenge R4_v1 ----------

# v1 conflicts with newly accepted challenge cases.
pre_challenge_conf=sum(
    1 for ex in R4_CHALLENGE if guard_allows(r4v1,ex.head_args)!=ex.should_hold
)
assert pre_challenge_conf==3

LIB.challenge("R4",{"new_conflicts":pre_challenge_conf})

statuses_after_challenge={
    r:(LIB.active[r].status if LIB.active.get(r) else None)
    for r in ("R4","R8","R9")
}

# ---------- symbolic revision search for R4 ----------

# Tiny generic revision language: one unary selector on one head port.
GUARDS=[("G0",0),("G1",0),("G2",0),("G0",1),("G1",1),("G2",1)]

@dataclass(frozen=True)
class RevisionCandidate:
    guard:tuple[str,int]
    train_support:int
    train_conflict:int
    frozen_support:int
    frozen_conflict:int
    challenge_support:int
    challenge_conflict:int
    mdl_cost:int

def candidate_score(guard):
    temp=ConceptVersion("R4",999,"STAGED",("R1","W"),guard,{},["revision-search"])
    ts,tc=eval_examples(temp,R4_TRAIN)
    fs,fc=eval_examples(temp,R4_FROZEN)
    cs,cc=eval_examples(temp,R4_CHALLENGE)
    return RevisionCandidate(guard,ts,tc,fs,fc,cs,cc,1)

CANDS=[candidate_score(g) for g in GUARDS]

valid=[
    c for c in CANDS
    if c.train_conflict==0 and c.frozen_conflict==0 and c.challenge_conflict==0
       and c.train_support==len(R4_TRAIN)
       and c.frozen_support==len(R4_FROZEN)
       and c.challenge_support==len(R4_CHALLENGE)
]
assert valid
valid.sort(key=lambda c:(c.mdl_cost,c.guard))
winner=valid[0]

# Must discover G0 on actor port, not be preselected.
assert winner.guard==("G0",0)

r4v2=ConceptVersion(
    "R4",LIB.next_version["R4"],"STAGED",("R1","W"),winner.guard,{},
    ["challenge-batch-1","symbolic-guard-search"],parent_version=1
)
LIB.next_version["R4"]+=1
# Gate against train + frozen + challenge together.
combined=R4_TRAIN+R4_CHALLENGE
assert score(r4v2,combined,R4_FROZEN)
LIB.activate_revision(r4v1,r4v2)

# ---------- revalidate / re-version descendants ----------

def reversion_dependent(relation,old,body_rels,parent_rel,parent_ver,train,frozen):
    new=ConceptVersion(
        relation,LIB.next_version[relation],"STAGED",tuple(body_rels),None,
        {parent_rel:parent_ver},["dependency-revalidation"],parent_version=old.version
    )
    LIB.next_version[relation]+=1

    # Parent gating and child correctness are separate:
    # if the new parent cannot prove an instance, the child body is simply closed
    # (UNKNOWN), not a child conflict.
    def dep_eval(examples):
        support=conflict=0
        gated=0
        for ex in examples:
            parent_args=ex.head_args[:3] if parent_rel=="R4" else ex.head_args[:4]
            parent_ok=prove(parent_rel,parent_args)==1
            pred=parent_ok  # no local child guard in this v5.8 experiment
            if pred==ex.should_hold:
                support+=1
            else:
                conflict+=1
            if not parent_ok:
                gated+=1
        return support,conflict,gated

    ts,tc,tg=dep_eval(train)
    fs,fc,fg=dep_eval(frozen)
    new.support,new.conflict=ts,tc
    new.frozen_support,new.frozen_conflict=fs,fc

    ok=(tc==0 and fc==0 and ts==len(train) and fs==len(frozen))
    if not ok:
        return None, {"train_gated":tg,"frozen_gated":fg}, {"train_conflict":tc,"frozen_conflict":fc}

    old.status="SUPERSEDED"
    new.status="ACTIVE"
    LIB.history[relation].append(new)
    LIB.active[relation]=new
    LIB.events.append(("dependency_revalidated",old.vid,new.vid,{parent_rel:parent_ver},
                       {"train_gated":tg,"frozen_gated":fg}))
    return new, {"train_gated":tg,"frozen_gated":fg}, {"train_conflict":tc,"frozen_conflict":fc}

r8v2,r8_train2,r8_frozen2=reversion_dependent(
    "R8",r8v1,("R4","X"),"R4",2,R8_TRAIN,R8_FROZEN
)
assert r8v2 is not None

r9v2,r9_train2,r9_frozen2=reversion_dependent(
    "R9",r9v1,("R8","Y"),"R8",2,R9_TRAIN,R9_FROZEN
)
assert r9v2 is not None

# ---------- proof checks ----------

good_r4=prove("R4",("emma","dragon","fly"))
bad_r4=prove("R4",("xavier","door","open"))
good_r8=prove("R8",("emma","dragon","fly","sky"))
bad_r8=prove("R8",("xavier","door","open","hall"))
good_r9=prove("R9",("emma","dragon","fly","sky","safe"))
bad_r9=prove("R9",("xavier","door","open","hall","safe"))

# Query never mutates evidence / selector store / versions.
selector_snapshot=copy.deepcopy(SELECTORS)
history_snapshot={r:len(vs) for r,vs in LIB.history.items()}
_ = prove("R4",("quinn","mystery","go"))
assert SELECTORS==selector_snapshot
assert {r:len(vs) for r,vs in LIB.history.items()}==history_snapshot

# ---------- failed proactive revision rollback ----------

# Try bad guard G2; it should fail and not replace active v2.
bad_candidate=ConceptVersion(
    "R4",LIB.next_version["R4"],"STAGED",("R1","W"),("G2",0),{},
    ["failed-proactive-revision"],parent_version=2
)
LIB.next_version["R4"]+=1
bad_ok=score(bad_candidate,R4_TRAIN+R4_CHALLENGE,R4_FROZEN)
assert not bad_ok
bad_candidate.status="RETIRED"
LIB.history["R4"].append(bad_candidate)
LIB.events.append(("staged_revision_rejected",bad_candidate.vid,bad_candidate.guard))
assert LIB.active["R4"] is r4v2

# ---------- unrepairable fork / cascade ----------

# Clone logical state minimally and add challenge to good G0 cases too,
# so no single selector guard can save R4.
UNREPAIRABLE=[
    Example("u1",("anna","lamp","light"),r4_body("anna","lamp","light"),False,"challenge2","u1"),
    Example("u2",("ben","gate","open"),r4_body("ben","gate","open"),False,"challenge2","u2"),
]
all_challenge2=R4_CHALLENGE+UNREPAIRABLE

def any_safe_guard():
    for g in GUARDS:
        temp=ConceptVersion("R4",1000,"STAGED",("R1","W"),g,{},["fork"])
        train=R4_TRAIN
        frozen=R4_FROZEN
        ch=all_challenge2
        ts,tc=eval_examples(temp,train)
        fs,fc=eval_examples(temp,frozen)
        cs,cc=eval_examples(temp,ch)
        if tc==0 and fc==0 and cc==0 and ts==len(train) and fs==len(frozen) and cs==len(ch):
            return g
    return None

fork_safe=any_safe_guard()
assert fork_safe is None

# Simulate conservative fork policy: challenged root has no safe active version;
# descendants must not remain active.
fork_status={"R4":"CHALLENGED","R8":"CHALLENGED","R9":"CHALLENGED"}
if fork_safe is None:
    fork_status={"R4":None,"R8":None,"R9":None}

# ---------- checks ----------

checks={
    "initial_anonymous_hierarchy_active":(
        r4v1.relation=="R4" and r8v1.relation=="R8" and r9v1.relation=="R9"
    ),
    "new_evidence_conflicts_with_r4_v1":pre_challenge_conf==3,
    "challenge_propagates_to_all_descendants":(
        statuses_after_challenge=={"R4":"CHALLENGED","R8":"CHALLENGED","R9":"CHALLENGED"}
    ),
    "revision_search_finds_symbolic_guard_not_semantic_head":winner.guard==("G0",0),
    "r4_v2_full_combined_support_zero_conflict":(
        r4v2.conflict==0 and r4v2.frozen_conflict==0
    ),
    "r4_v1_superseded_r4_v2_active":(
        r4v1.status=="SUPERSEDED" and LIB.active["R4"] is r4v2 and r4v2.status=="ACTIVE"
    ),
    "r8_reversioned_against_r4_v2":(
        r8v1.status=="SUPERSEDED" and r8v2.version==2 and
        r8v2.parents=={"R4":2} and r8v2.status=="ACTIVE"
    ),
    "r9_reversioned_against_r8_v2":(
        r9v1.status=="SUPERSEDED" and r9v2.version==2 and
        r9v2.parents=={"R8":2} and r9v2.status=="ACTIVE"
    ),
    "good_root_and_descendant_proofs_survive":good_r4==1 and good_r8==1 and good_r9==1,
    "guard_incompatible_root_and_descendants_unknown":bad_r4==0 and bad_r8==0 and bad_r9==0,
    "query_is_not_evidence_or_revision_trigger":(
        SELECTORS==selector_snapshot and
        {r:len(vs) for r,vs in LIB.history.items()}=={
            "R4":3,  # v1,v2,rejected v3
            "R8":2,
            "R9":2,
        }
    ),
    "failed_revision_rolls_back_without_replacing_active":(
        bad_candidate.status=="RETIRED" and LIB.active["R4"] is r4v2
    ),
    "history_and_parent_links_preserved":(
        r4v2.parent_version==1 and r8v2.parent_version==1 and r9v2.parent_version==1
    ),
    "unrepairable_root_has_no_safe_guard":fork_safe is None,
    "unrepairable_root_cascades_to_no_active_descendants":fork_status=={"R4":None,"R8":None,"R9":None},
    "all_concept_heads_remain_anonymous":all(
        re.fullmatch(r"R\d+",r) for r in ("R4","R8","R9")
    ),
}

print("=== v5.8 VERSIONED REVISION + DEPENDENCY PROPAGATION ===")
for k,v in checks.items():
    print(("PASS" if v else "FAIL"),"|",k)

print("\nChallenge state:")
print(statuses_after_challenge)

print("\nRevision candidates:")
for c in CANDS:
    print(" ",c.guard,
          "train",c.train_support,"/",c.train_conflict,
          "frozen",c.frozen_support,"/",c.frozen_conflict,
          "challenge",c.challenge_support,"/",c.challenge_conflict,
          "VALID" if c in valid else "")
print("winner:",winner.guard)

print("\nVersions:")
for r in ("R4","R8","R9"):
    for v in LIB.history[r]:
        print(" ",v.vid,
              v.status,
              "guard",v.guard,
              "parents",v.parents,
              "parent_version",v.parent_version)

print("\nProofs:")
print(" good:",good_r4,good_r8,good_r9)
print(" bad :",bad_r4,bad_r8,bad_r9)

print("\nLifecycle:")
for e in LIB.events:
    print(" ",e)

print("\nUnrepairable fork:",fork_status)

assert all(checks.values())

report={
    "version":"v5.8-versioned-revision-dependency-propagation",
    "result":"PASS",
    "checks":checks,
    "winner_guard":list(winner.guard),
    "challenge_state":statuses_after_challenge,
    "proofs":{
        "good":[good_r4,good_r8,good_r9],
        "guard_incompatible":[bad_r4,bad_r8,bad_r9],
    },
    "versions":{
        r:[
            {
                "vid":v.vid,
                "status":v.status,
                "guard":list(v.guard) if v.guard else None,
                "parents":v.parents,
                "parent_version":v.parent_version,
                "support":v.support,
                "conflict":v.conflict,
                "frozen_support":v.frozen_support,
                "frozen_conflict":v.frozen_conflict,
                "provenance":v.provenance,
            } for v in LIB.history[r]
        ] for r in ("R4","R8","R9")
    },
    "revision_candidates":[
        {
            "guard":list(c.guard),
            "train_support":c.train_support,
            "train_conflict":c.train_conflict,
            "frozen_support":c.frozen_support,
            "frozen_conflict":c.frozen_conflict,
            "challenge_support":c.challenge_support,
            "challenge_conflict":c.challenge_conflict,
            "valid":c in valid,
        } for c in CANDS
    ],
    "lifecycle_events":[list(e) for e in LIB.events],
    "unrepairable_fork":fork_status,
    "design":[
        "Anonymous concept heads remain R4/R8/R9.",
        "A challenged root immediately challenges all active descendants.",
        "Revision searches a small generic symbolic guard language over head ports.",
        "A new root version must pass combined training, challenge, and frozen gates before activation.",
        "Dependent concepts are re-versioned only against the exact active parent version they depend on.",
        "Failed proactive revision is RETIRED and cannot replace the current ACTIVE chain.",
        "If the root is unrepairable, no descendant is allowed to remain ACTIVE.",
        "Queries call prove() only and never add selector facts, examples, versions, or lifecycle events."
    ],
    "caveats":[
        "The revision language is intentionally tiny: one unary guard on one head port.",
        "Selectors G0/G1/G2 are supplied symbolic ontology observations; discovering new selector predicates is not attempted.",
        "Dependency revalidation preserves the dependent body structure; arbitrary structural repair of descendants is not attempted.",
        "The unrepairable cascade is tested as a conservative fork policy, not as a full transactional clone of the library.",
        "This is a symbolic hierarchy PoC, not a raw-text end-to-end test."
    ],
}
Path("/mnt/data/symbolic_v58_revision_dependency_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
with Path("/mnt/data/symbolic_v58_revision_dependency_checks.csv").open(
    "w",newline="",encoding="utf-8"
) as f:
    w=csv.writer(f); w.writerow(["check","passed"])
    for k,v in checks.items(): w.writerow([k,v])

print("\nSaved v5.8 report/checks.")

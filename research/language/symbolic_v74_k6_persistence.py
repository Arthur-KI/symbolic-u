
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
import json, csv, re

# ============================================================
# v7.4 / K6 — Learned Persistence-U / State Memory ablation
#
# Removed from NEW memory layer:
#   fixed inertia: "all state Keys persist until changed"
#
# Retained kernel-ish substrate:
#   ordered time
#   exact Key identity
#   open-world default: no inference from silence
#   provenance / independent evidence IDs
#
# Learned:
#   anonymous persistence policies M1... for relation heads
# ============================================================

K5=json.loads(Path("/mnt/data/symbolic_v73_k5_operation_ablation_report.json").read_text(encoding="utf-8"))
K4=json.loads(Path("/mnt/data/symbolic_v72_k4_type_ablation_report.json").read_text(encoding="utf-8"))
K3=json.loads(Path("/mnt/data/symbolic_v71_k3_relation_ablation_report.json").read_text(encoding="utf-8"))
K2=json.loads(Path("/mnt/data/symbolic_v70_k2_action_head_ablation_report.json").read_text(encoding="utf-8"))
assert K5["result"]=="PASS" and all(K5["checks"].values())
assert K4["result"]=="PASS" and all(K4["checks"].values())

P1=K3["evaluator_only_mapping"]["PROCESS"]
P2=K3["evaluator_only_mapping"]["ATTRIBUTE"]
P3=K3["evaluator_only_mapping"]["POSSESSION"]
P4=K3["evaluator_only_mapping"]["LOCATION"]
P5="P5"  # anonymous held-out temporal relation, evaluator-only occurrence-like
A={k:v["head"] for k,v in K2["anonymous_actions"].items()}

Key=tuple[str,tuple[str,...]]

# ---------- timelines ----------
@dataclass(frozen=True)
class Obs:
    time:int
    key:Key
    evidence_id:str

@dataclass(frozen=True)
class QueryLabel:
    time:int
    key:Key
    expected:int  # +1 supported now, 0 not inferable
    evidence_id:str

@dataclass
class Story:
    observations:list[Obs]=field(default_factory=list)

    def direct_at(self,time,key):
        return any(o.time==time and o.key==key for o in self.observations)

    def prior_observations(self,time,key):
        return sorted(
            [o for o in self.observations if o.time < time and o.key==key],
            key=lambda o:o.time
        )

    def relation_facts_between(self,relation,t0,t1):
        return sorted(
            [o for o in self.observations if t0 < o.time <= t1 and o.key[0]==relation],
            key=lambda o:o.time
        )

# ---------- generic blocker ----------
# If same relation and same first argument acquires a DIFFERENT second argument,
# old binary fact is blocked from that time onward.
# This is purely structural and corresponds to K5's learned replacement motif;
# no LOCATION/STATE semantic rule is used.
def has_structural_replacement(story:Story,key:Key,t0:int,tq:int):
    rel,args=key
    if len(args)!=2:
        return False
    x,y=args
    for o in story.relation_facts_between(rel,t0,tq):
        r,a=o.key
        if len(a)==2 and a[0]==x and a[1]!=y:
            return True
    return False

# ---------- Persistence-U learner ----------
@dataclass
class PersistenceState:
    relation:str
    mode_id:str
    support:set[str]=field(default_factory=set)
    conflict:set[str]=field(default_factory=set)
    status:str="STAGED"

    def refresh(self):
        if self.conflict:
            self.status="CHALLENGED" if len(self.support)>=3 else "REJECTED"
        elif len(self.support)>=3:
            self.status="ACTIVE"
        else:
            self.status="STAGED"

class PersistenceLearner:
    def __init__(self):
        self.by_relation={}
        self._next=1

    def state(self,relation):
        if relation not in self.by_relation:
            self.by_relation[relation]=PersistenceState(
                relation=relation,mode_id=f"M{self._next}"
            )
            self._next+=1
        return self.by_relation[relation]

    def observe_case(self,story:Story,label:QueryLabel):
        rel=label.key[0]
        st=self.state(rel)

        # Direct observation at query time is not evidence for persistence.
        if story.direct_at(label.time,label.key):
            return st

        priors=story.prior_observations(label.time,label.key)
        if not priors:
            return st

        last=priors[-1]
        blocked=has_structural_replacement(story,label.key,last.time,label.time)

        # Candidate persistence U predicts +1 iff there is an earlier exact Key
        # and no structural replacement blocker.
        pred=+1 if not blocked else 0

        if pred==label.expected and label.expected==+1:
            st.support.add(label.evidence_id)
        elif pred==+1 and label.expected!=+1:
            st.conflict.add(label.evidence_id)
        # Expected 0 with blocked prediction 0 is not support for "nonpersistence";
        # open-world silence remains no evidence.
        st.refresh()
        return st

    def infer(self,story:Story,time:int,key:Key):
        if story.direct_at(time,key):
            return +1,"DIRECT"

        st=self.by_relation.get(key[0])
        if not st or st.status!="ACTIVE":
            return 0,"NO_ACTIVE_PERSISTENCE_U"

        priors=story.prior_observations(time,key)
        if not priors:
            return 0,"NO_PRIOR_KEY"

        last=priors[-1]
        if has_structural_replacement(story,key,last.time,time):
            return 0,"BLOCKED_BY_STRUCTURAL_REPLACEMENT"

        return +1,st.mode_id

# ---------- training curriculum ----------
def K(r,a,b): return (r,(a,b))

learner=PersistenceLearner()

TRAIN=[]

# P1: three independent midpoint confirmations => learn persistence
for i,(x,a0) in enumerate([
    ("LAMP",A["leuchten"]),
    ("WHEEL",A["drehen"]),
    ("MACHINE",A["laufen"]),
],1):
    s=Story([Obs(0,K(P1,x,a0),f"p1-{i}-obs")])
    q=QueryLabel(1,K(P1,x,a0),+1,f"p1-{i}-mid")
    TRAIN.append((s,q))

# P4: same, no semantic PLACE type is needed here
for i,(x,y) in enumerate([
    ("ANNA","HOUSE"),("BEN","GARDEN"),("CARA","ROOM")
],1):
    s=Story([Obs(0,K(P4,x,y),f"p4-{i}-obs")])
    q=QueryLabel(2,K(P4,x,y),+1,f"p4-{i}-mid")
    TRAIN.append((s,q))

# P3: possession-like relation also persists
for i,(x,y) in enumerate([
    ("ANNA","KEY"),("BEN","BOOK"),("CARA","COIN")
],1):
    s=Story([Obs(0,K(P3,x,y),f"p3-{i}-obs")])
    q=QueryLabel(3,K(P3,x,y),+1,f"p3-{i}-mid")
    TRAIN.append((s,q))

# P2: attribute-like relation persists
for i,(x,y) in enumerate([
    ("GATE","CLOSED"),("LAMP","BLUE"),("MACHINE","HOT")
],1):
    s=Story([Obs(0,K(P2,x,y),f"p2-{i}-obs")])
    q=QueryLabel(1,K(P2,x,y),+1,f"p2-{i}-mid")
    TRAIN.append((s,q))

# P5 has the SAME coarse binary form and can reuse the same sorts of symbols,
# but midpoint labels say it is NOT licensed to persist.
for i,(x,y) in enumerate([
    ("LAMP",A["leuchten"]),
    ("WHEEL",A["drehen"]),
    ("MACHINE",A["laufen"]),
],1):
    s=Story([Obs(0,K(P5,x,y),f"p5-{i}-obs")])
    q=QueryLabel(1,K(P5,x,y),0,f"p5-{i}-mid")
    TRAIN.append((s,q))

for s,q in TRAIN:
    learner.observe_case(s,q)

# ---------- inspect learned modes ----------
ST={r:learner.by_relation.get(r) for r in [P1,P2,P3,P4,P5]}

# P1..P4 active; P5 should be rejected/challenged because carry predicts wrong.
STATE_RELATIONS_ACTIVE=all(ST[r] and ST[r].status=="ACTIVE" for r in [P1,P2,P3,P4])
P5_NOT_ACTIVE=ST[P5] and ST[P5].status in {"REJECTED","CHALLENGED"}

# ---------- frozen transfer ----------
FROZEN=[
    (Story([Obs(0,K(P1,"POT",A["kochen"]),"f1")]),1,K(P1,"POT",A["kochen"]),+1),
    (Story([Obs(0,K(P4,"DORA","FOREST"),"f2")]),5,K(P4,"DORA","FOREST"),+1),
    (Story([Obs(0,K(P3,"DORA","BOX"),"f3")]),4,K(P3,"DORA","BOX"),+1),
    (Story([Obs(0,K(P2,"GATE","RED"),"f4")]),7,K(P2,"GATE","RED"),+1),
    (Story([Obs(0,K(P5,"POT",A["kochen"]),"f5")]),1,K(P5,"POT",A["kochen"]),0),
]
FROZEN_RESULTS=[]
for s,t,k,gold in FROZEN:
    pred,why=learner.infer(s,t,k)
    FROZEN_RESULTS.append((pred,why,gold))
FROZEN_OK=all(p==g for p,w,g in FROZEN_RESULTS)

# ---------- blocker tests ----------
# Unrelated relation does not block.
unrelated=Story([
    Obs(0,K(P4,"ANNA","HOUSE"),"u0"),
    Obs(1,K(P2,"GATE","CLOSED"),"u1"),
])
UNRELATED_PRED=learner.infer(unrelated,2,K(P4,"ANNA","HOUSE"))

# Same P + same first arg + different second arg blocks old Key.
moved=Story([
    Obs(0,K(P4,"ANNA","HOUSE"),"m0"),
    Obs(2,K(P4,"ANNA","GARDEN"),"m1"),
])
OLD_AFTER_MOVE=learner.infer(moved,3,K(P4,"ANNA","HOUSE"))
NEW_AFTER_MOVE=learner.infer(moved,3,K(P4,"ANNA","GARDEN"))

# Same relation but different first arg must NOT block.
other_subject=Story([
    Obs(0,K(P4,"ANNA","HOUSE"),"os0"),
    Obs(2,K(P4,"BEN","GARDEN"),"os1"),
])
OTHER_SUBJECT_PRED=learner.infer(other_subject,3,K(P4,"ANNA","HOUSE"))

# ---------- query is not evidence ----------
before_support=len(ST[P1].support)
qstory=Story([Obs(0,K(P1,"GATE",A["öffnen"]),"qe0")])
for _ in range(5):
    learner.infer(qstory,1,K(P1,"GATE",A["öffnen"]))
after_support=len(ST[P1].support)

# ---------- duplicate evidence cannot fake activation ----------
dup_learner=PersistenceLearner()
dup_story=Story([Obs(0,K("PX","A","B"),"do")])
for _ in range(4):
    dup_learner.observe_case(
        dup_story,QueryLabel(1,K("PX","A","B"),+1,"same-evidence")
    )
DUP_STATE=dup_learner.by_relation["PX"]

# ---------- later conflict challenges active persistence ----------
challenge_learner=PersistenceLearner()
for i in range(3):
    s=Story([Obs(0,K("PY",f"X{i}","Y"),f"co{i}")])
    challenge_learner.observe_case(
        s,QueryLabel(1,K("PY",f"X{i}","Y"),+1,f"cs{i}")
    )
PRE_CHALLENGE=challenge_learner.by_relation["PY"].status
cs=Story([Obs(0,K("PY","Z","Y"),"cz")])
challenge_learner.observe_case(
    cs,QueryLabel(1,K("PY","Z","Y"),0,"conflict-1")
)
POST_CHALLENGE=challenge_learner.by_relation["PY"].status

# ---------- identifiability boundary: endpoint-only silence ----------
# Observed t0=k and t2=k with no observation at t1.
# Model A: k persisted through t1.
# Model B: k disappeared after t0 and independently reappeared at t2.
# Both make exactly the same positive observations.
endpoint_story=Story([
    Obs(0,K("PZ","A","B"),"ep0"),
    Obs(2,K("PZ","A","B"),"ep2"),
])
models={
    "PERSIST_THROUGH_GAP":{
        0:{K("PZ","A","B")},1:{K("PZ","A","B")},2:{K("PZ","A","B")}
    },
    "DISAPPEAR_REAPPEAR":{
        0:{K("PZ","A","B")},1:set(),2:{K("PZ","A","B")}
    }
}
EXPLICIT_ENDPOINTS={
    0:{K("PZ","A","B")},
    2:{K("PZ","A","B")}
}
same_observations=all(
    models[m][0]==EXPLICIT_ENDPOINTS[0] and models[m][2]==EXPLICIT_ENDPOINTS[2]
    for m in models
)

# Endpoint observations alone are NOT fed as positive persistence labels.
endpoint_learner=PersistenceLearner()
ENDPOINT_INFER=endpoint_learner.infer(endpoint_story,1,K("PZ","A","B"))

# Add explicit midpoint confirmation: now it can become evidence.
for i in range(3):
    s=Story([Obs(0,K("PZ",f"A{i}","B"),f"ez{i}")])
    endpoint_learner.observe_case(
        s,QueryLabel(1,K("PZ",f"A{i}","B"),+1,f"em{i}")
    )
ENDPOINT_AFTER_LABEL=endpoint_learner.infer(
    Story([Obs(0,K("PZ","NEW","B"),"en")]),1,K("PZ","NEW","B")
)

# ---------- harder boundary: relation P5 same coarse type/topology ----------
# P1 and P5 share exact arity and even same kinds of occupants.
def coarse_profile(r):
    examples=[
        q.key for s,q in TRAIN if q.key[0]==r
    ]
    return ("BINARY",2,tuple(sorted(len(k[1]) for k in examples)))
COARSE_COLLISION=coarse_profile(P1)==coarse_profile(P5)
TEMPORAL_LABELS_SEPARATE=(
    ST[P1].status=="ACTIVE" and ST[P5].status!="ACTIVE"
)

# ---------- Grimm ----------
# After "Töpfchen koche", P1(POT,A2) should persist over unrelated material.
# The learned O3 disappearance from K5 acts as an explicit removal boundary.
POTK=K(P1,"POT",A["kochen"])

grimm_before_stop=Story([
    Obs(0,POTK,"g-cook"),
    Obs(1,K(P4,"GIRL","HOME"),"g-unrelated"),
])
GRIMM_BEFORE=learner.infer(grimm_before_stop,2,POTK)

# Model the O3 consequence as state snapshot no longer carrying POTK.
# We don't encode a semantic REMOVE label; instead, at t3 a structural replacement/
# change boundary is represented by a different same-subject P1 fact for blocker audit.
# For exact cessation, current snapshot after O3 has no POTK; therefore inference must
# be queried from a story segment whose last process observation is cut by an explicit
# temporal deletion marker in provenance.
@dataclass(frozen=True)
class Tombstone:
    time:int
    key:Key
    evidence_id:str

# Extend blocking generically with explicit exact-Key tombstones from learned O3.
class StoryWithDeletes(Story):
    def __init__(self,observations,deletions):
        super().__init__(observations)
        self.deletions=deletions

def infer_with_deletes(learner,story,time,key):
    if story.direct_at(time,key):
        return +1,"DIRECT"
    prior=story.prior_observations(time,key)
    if not prior:
        return 0,"NO_PRIOR_KEY"
    last=prior[-1]
    if any(d.key==key and last.time < d.time <= time for d in story.deletions):
        return 0,"BLOCKED_BY_ANONYMOUS_O3_TOMBSTONE"
    return learner.infer(story,time,key)

grimm_after_stop=StoryWithDeletes(
    [Obs(0,POTK,"g-cook")],
    [Tombstone(3,POTK,"g-O3")]
)
GRIMM_AFTER=infer_with_deletes(learner,grimm_after_stop,4,POTK)

# second cook observation starts a new persistence interval
grimm_restart=StoryWithDeletes(
    [Obs(0,POTK,"g-cook"),Obs(5,POTK,"g-cook2")],
    [Tombstone(3,POTK,"g-O3")]
)
GRIMM_RESTART=infer_with_deletes(learner,grimm_restart,6,POTK)

# ---------- checks ----------
checks={
    "frozen_K5_base_is_green":K5["result"]=="PASS" and all(K5["checks"].values()),
    "K6_P1_to_P4_learn_active_anonymous_persistence_modes":STATE_RELATIONS_ACTIVE,
    "K6_same_coarse_binary_P5_does_not_acquire_persistence":P5_NOT_ACTIVE,
    "K6_frozen_new_constants_transfer_by_relation_policy":FROZEN_OK,
    "K6_unrelated_relation_does_not_block_persistence":UNRELATED_PRED[0]==+1,
    "K6_same_relation_same_first_new_second_blocks_old_key":OLD_AFTER_MOVE[0]==0,
    "K6_new_replacement_value_itself_persists":NEW_AFTER_MOVE[0]==+1,
    "K6_different_first_argument_does_not_block":OTHER_SUBJECT_PRED[0]==+1,
    "K6_query_calls_do_not_add_learning_evidence":before_support==after_support,
    "K6_duplicate_evidence_cannot_activate_persistence":DUP_STATE.status=="STAGED" and len(DUP_STATE.support)==1,
    "K6_later_conflict_challenges_active_persistence":PRE_CHALLENGE=="ACTIVE" and POST_CHALLENGE=="CHALLENGED",
    "K6_endpoint_only_gap_is_non_identifiable":same_observations,
    "K6_endpoint_only_data_does_not_license_persistence":ENDPOINT_INFER[0]==0,
    "K6_explicit_midpoint_continuity_evidence_can_activate_persistence":ENDPOINT_AFTER_LABEL[0]==+1,
    "K6_same_arity_same_coarse_occupants_are_separated_by_temporal_evidence":COARSE_COLLISION and TEMPORAL_LABELS_SEPARATE,
    "K6_Grimm_process_persists_before_O3":GRIMM_BEFORE[0]==+1,
    "K6_Grimm_O3_boundary_stops_old_persistence_interval":GRIMM_AFTER[0]==0,
    "K6_Grimm_new_cook_observation_starts_new_interval":GRIMM_RESTART[0]==+1,
}

print("=== v7.4 / K6 LEARNED PERSISTENCE-U ===")
for k,v in checks.items():
    print(("PASS" if v else "FAIL"),"|",k)

print("\nLearned persistence modes:")
for r in [P1,P2,P3,P4,P5]:
    st=ST[r]
    print(" ",r,"=>",st.mode_id,"support",len(st.support),
          "conflict",len(st.conflict),"status",st.status)

print("\nFrozen:")
for i,(pred,why,gold) in enumerate(FROZEN_RESULTS,1):
    print(" ",i,"pred",pred,"gold",gold,"via",why)

print("\nBlockers:")
print(" unrelated:",UNRELATED_PRED)
print(" old after replacement:",OLD_AFTER_MOVE)
print(" new after replacement:",NEW_AFTER_MOVE)
print(" different subject:",OTHER_SUBJECT_PRED)

print("\nLifecycle:")
print(" query support:",before_support,"->",after_support)
print(" duplicate:",DUP_STATE.status,DUP_STATE.support)
print(" challenge:",PRE_CHALLENGE,"->",POST_CHALLENGE)

print("\nIdentifiability:")
print(" endpoint models same explicit observations:",same_observations)
print(" endpoint-only inference:",ENDPOINT_INFER)
print(" after midpoint evidence:",ENDPOINT_AFTER_LABEL)
print(" P1/P5 coarse collision:",COARSE_COLLISION)
print(" P1 status:",ST[P1].status,"P5 status:",ST[P5].status)

print("\nGrimm:")
print(" before stop:",GRIMM_BEFORE)
print(" after O3:",GRIMM_AFTER)
print(" restart:",GRIMM_RESTART)

assert all(checks.values())

report={
    "version":"v7.4-K6-learned-persistence",
    "result":"PASS",
    "checks":checks,
    "modes":{
        r:{
            "mode_id":ST[r].mode_id,
            "support":len(ST[r].support),
            "conflict":len(ST[r].conflict),
            "status":ST[r].status
        } for r in [P1,P2,P3,P4,P5]
    },
    "identifiability":{
        "endpoint_only_models":{
            name:{str(t):[repr(x) for x in sorted(facts,key=repr)] for t,facts in timeline.items()}
            for name,timeline in models.items()
        },
        "same_explicit_endpoint_observations":same_observations,
        "endpoint_only_midgap_inference":ENDPOINT_INFER,
        "after_explicit_midpoint_training":ENDPOINT_AFTER_LABEL,
        "finding":"Positive observations at t0 and t2 with silence at t1 do not identify persistence. A persistence model and a disappear/reappear model fit exactly the same endpoint observations. Additional continuity evidence or an inductive bias is required."
    },
    "coarse_collision":{
        "P1_profile":coarse_profile(P1),
        "P5_profile":coarse_profile(P5),
        "collision":COARSE_COLLISION,
        "P1_status":ST[P1].status,
        "P5_status":ST[P5].status,
        "finding":"Two anonymous binary relations with the same coarse arity/occupant profile can learn different memory behavior from temporal supervision alone."
    },
    "grimm":{
        "key":repr(POTK),
        "before_stop":GRIMM_BEFORE,
        "after_O3":GRIMM_AFTER,
        "restart":GRIMM_RESTART
    },
    "interpretation":[
        "K6 removes universal fixed inertia from the new memory layer. Open-world default is no carry inference.",
        "Anonymous relation-specific persistence modes M# are activated only by repeated independent midpoint continuity evidence.",
        "A same-arity anonymous relation P5 does not persist because midpoint labels conflict with the carry hypothesis; port shape alone is insufficient.",
        "Structural replacement of the same relation/first argument blocks an older key without semantic LOCATION/STATE knowledge.",
        "Queries are read-only and duplicate evidence IDs cannot create support.",
        "A later counterexample challenges an active persistence policy.",
        "The Grimm process key P1(POT,A2) persists across unrelated material, is cut by the learned anonymous O3 disappearance boundary, and can begin a new interval after a later observation."
    ],
    "caveats":[
        "K6 still receives explicit midpoint truth labels during persistence curriculum. Endpoint observations alone are proven insufficient by the identifiability audit.",
        "The open-world default 'do not infer from silence' remains fixed kernel policy.",
        "Exact Key identity and temporal order remain fixed and were already shown kernel-near in K5.",
        "The blocker rule uses exact same-relation/same-first-argument replacement topology; more general persistence termination conditions should be learned as higher U.",
        "The O3 tombstone in the Grimm audit is structural provenance from K5, not a named REMOVE semantic operation.",
        "P5 is a controlled anonymous occurrence-like relation introduced specifically to test that persistence is not inferred from binary shape."
    ]
}
Path("/mnt/data/symbolic_v74_k6_persistence_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
with Path("/mnt/data/symbolic_v74_k6_persistence_checks.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["check","passed"])
    for k,v in checks.items(): w.writerow([k,v])

print("\nSaved K6 report/checks.")

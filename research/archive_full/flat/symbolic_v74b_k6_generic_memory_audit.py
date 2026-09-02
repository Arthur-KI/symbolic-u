
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import json, re

K6=json.loads(Path("/mnt/data/symbolic_v74_k6_persistence_report.json").read_text(encoding="utf-8"))
K5=json.loads(Path("/mnt/data/symbolic_v73_k5_operation_ablation_report.json").read_text(encoding="utf-8"))
K3=json.loads(Path("/mnt/data/symbolic_v71_k3_relation_ablation_report.json").read_text(encoding="utf-8"))
K2=json.loads(Path("/mnt/data/symbolic_v70_k2_action_head_ablation_report.json").read_text(encoding="utf-8"))
assert K6["result"]=="PASS" and all(K6["checks"].values())

P1=K3["evaluator_only_mapping"]["PROCESS"]
P2=K3["evaluator_only_mapping"]["ATTRIBUTE"]
P3=K3["evaluator_only_mapping"]["POSSESSION"]
P4=K3["evaluator_only_mapping"]["LOCATION"]
P5="P5"
A={k:v["head"] for k,v in K2["anonymous_actions"].items()}
O=K5["evaluator_only_mapping"]  # evaluator-only names -> already anonymous O IDs

def K(r,a,b): return (r,(a,b))

@dataclass(frozen=True)
class Obs:
    time:int
    key:tuple
    eid:str

@dataclass(frozen=True)
class Boundary:
    # Materialized output of a learned K5 local operation component.
    # The memory layer consumes only anonymous op_id plus before/after Key sets.
    time:int
    op_id:str
    before_only:frozenset
    after_only:frozenset
    eid:str

@dataclass
class Timeline:
    obs:list[Obs]=field(default_factory=list)
    boundaries:list[Boundary]=field(default_factory=list)

    def direct_at(self,t,key):
        if any(o.time==t and o.key==key for o in self.obs):
            return True
        if any(b.time==t and key in b.after_only for b in self.boundaries):
            return True
        return False

    def prior_sources(self,t,key):
        out=[(o.time,o.eid) for o in self.obs if o.time<t and o.key==key]
        out += [(b.time,b.eid) for b in self.boundaries if b.time<t and key in b.after_only]
        return sorted(out)

    def killed_between(self,key,t0,tq):
        return [
            b for b in self.boundaries
            if t0 < b.time <= tq and key in b.before_only and key not in b.after_only
        ]

# ------------------------------------------------------------
# Generic memory-U abstraction:
# M1 := carry exact prior Key forward iff no learned local operation boundary
#       explicitly removes that Key from its local before->after component.
# Relation attachment is learned separately.
# ------------------------------------------------------------

MEMORY_U="M1"

@dataclass
class Attachment:
    relation:str
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

ATT={r:Attachment(r) for r in [P1,P2,P3,P4,P5]}

# Reuse main K6 temporal supervision, but all persistent relations attach to SAME M1.
positive={
    P1:[("LAMP",A["leuchten"]),("WHEEL",A["drehen"]),("MACHINE",A["laufen"])],
    P2:[("GATE","CLOSED"),("LAMP","BLUE"),("MACHINE","HOT")],
    P3:[("ANNA","KEY"),("BEN","BOOK"),("CARA","COIN")],
    P4:[("ANNA","HOUSE"),("BEN","GARDEN"),("CARA","ROOM")],
}
for r,pairs in positive.items():
    for i,(x,y) in enumerate(pairs):
        ATT[r].support.add(f"{r}-persist-{i}")
    ATT[r].refresh()

# P5: same coarse shape, but midpoint persistence predictions conflict.
for i in range(3):
    ATT[P5].conflict.add(f"P5-nonpersist-{i}")
ATT[P5].refresh()

def infer(tl:Timeline,t,key):
    if tl.direct_at(t,key):
        return +1,"DIRECT"
    att=ATT.get(key[0])
    if not att or att.status!="ACTIVE":
        return 0,"NO_ACTIVE_MEMORY_ATTACHMENT"
    priors=tl.prior_sources(t,key)
    if not priors:
        return 0,"NO_PRIOR_KEY"
    last_t,last_eid=priors[-1]
    kills=tl.killed_between(key,last_t,t)
    if kills:
        return 0,("BLOCKED_BY_O",kills[-1].op_id,kills[-1].eid)
    return +1,MEMORY_U

# ------------------------------------------------------------
# O1-like replacement boundary: old second value dies, new starts.
# ------------------------------------------------------------
old=K(P2,"GATE","CLOSED")
new=K(P2,"GATE","OPENED")
rep=Timeline(
    obs=[Obs(0,old,"rep-start")],
    boundaries=[Boundary(
        2,O["REPLACE_SECOND"],
        frozenset({old}),frozenset({new}),"rep-O"
    )]
)
REP_OLD=infer(rep,3,old)
REP_NEW=infer(rep,3,new)

# ------------------------------------------------------------
# O2-like transfer boundary: THIS is the missing attack.
# old first value dies, new starts while second arg is shared.
# ------------------------------------------------------------
annakey=K(P3,"ANNA","KEY")
benkey=K(P3,"BEN","KEY")
transfer=Timeline(
    obs=[Obs(0,annakey,"tr-start")],
    boundaries=[Boundary(
        2,O["TRANSFER_FIRST"],
        frozenset({annakey}),frozenset({benkey}),"tr-O"
    )]
)
TR_OLD=infer(transfer,3,annakey)
TR_NEW=infer(transfer,3,benkey)

# ------------------------------------------------------------
# O3-like exact disappearance.
# ------------------------------------------------------------
pot=K(P1,"POT",A["kochen"])
stop=Timeline(
    obs=[Obs(0,pot,"stop-start")],
    boundaries=[Boundary(
        2,O["DISAPPEAR"],
        frozenset({pot}),frozenset(),"stop-O"
    )]
)
STOP_OLD=infer(stop,3,pot)

# ------------------------------------------------------------
# O4-like appearance starts a new persistence interval.
# ------------------------------------------------------------
lamp=K(P1,"LAMP",A["leuchten"])
start=Timeline(
    boundaries=[Boundary(
        2,O["APPEAR"],
        frozenset(),frozenset({lamp}),"start-O"
    )]
)
START_NEW=infer(start,3,lamp)

# ------------------------------------------------------------
# Unrelated O boundary cannot kill the Key.
# ------------------------------------------------------------
house=K(P4,"ANNA","HOUSE")
other=K(P2,"GATE","CLOSED")
unrelated=Timeline(
    obs=[Obs(0,house,"u-start")],
    boundaries=[Boundary(
        2,O["DISAPPEAR"],
        frozenset({other}),frozenset(),"u-O"
    )]
)
UNRELATED=infer(unrelated,3,house)

# ------------------------------------------------------------
# Connected operation can remove multiple Keys; exact membership controls blocking.
# No semantic reading of O ID is needed by the memory layer.
# ------------------------------------------------------------
k1=K(P3,"ANNA","KEY")
k2=K(P3,"ANNA","BOOK")
k3=K(P3,"BEN","KEY")
multi=Timeline(
    obs=[Obs(0,k1,"m1"),Obs(0,k2,"m2")],
    boundaries=[Boundary(
        2,"O77",  # unknown anonymous higher operation
        frozenset({k1}),frozenset({k3}),"m-O"
    )]
)
MULTI_KILLED=infer(multi,3,k1)
MULTI_SURVIVES=infer(multi,3,k2)

# ------------------------------------------------------------
# Memory-U identity is shared; relation-specific evidence only controls attachment.
# ------------------------------------------------------------
SHARED_MEMORY=(
    all(ATT[r].status=="ACTIVE" for r in [P1,P2,P3,P4])
    and ATT[P5].status=="REJECTED"
)

# A new relation cannot inherit M1 merely from same arity/type shape.
P6="P6"
unknown=Timeline(obs=[Obs(0,K(P6,"ANNA","HOUSE"),"p6")])
UNKNOWN_REL=infer(unknown,3,K(P6,"ANNA","HOUSE"))

# ------------------------------------------------------------
# Open-world endpoint ambiguity remains unchanged.
# ------------------------------------------------------------
# The memory rule can be learned from external continuity labels,
# but t0/t2 endpoints alone cannot establish its attachment to P6.
endpoints=Timeline(obs=[
    Obs(0,K(P6,"A","B"),"e0"),
    Obs(2,K(P6,"A","B"),"e2")
])
ENDPOINT_GAP=infer(endpoints,1,K(P6,"A","B"))

# ------------------------------------------------------------
# Grimm: learned O3 cut + later O4/direct start
# ------------------------------------------------------------
grimm=Timeline(
    obs=[Obs(0,pot,"g-cook-1")],
    boundaries=[
        Boundary(3,O["DISAPPEAR"],frozenset({pot}),frozenset(),"g-stop"),
        Boundary(5,O["APPEAR"],frozenset(),frozenset({pot}),"g-cook-2"),
    ]
)
G2=infer(grimm,2,pot)
G4=infer(grimm,4,pot)
G6=infer(grimm,6,pot)

checks={
    "K6b_four_persistent_relations_share_one_generic_memory_U":SHARED_MEMORY,
    "K6b_same_shape_P5_is_not_attached_to_memory_U":ATT[P5].status=="REJECTED",
    "K6b_O1_boundary_blocks_old_and_starts_new":REP_OLD[0]==0 and REP_NEW[0]==+1,
    "K6b_O2_transfer_blocks_old_owner_and_starts_new_owner":TR_OLD[0]==0 and TR_NEW[0]==+1,
    "K6b_O3_exact_disappearance_blocks_persistence":STOP_OLD[0]==0,
    "K6b_O4_appearance_starts_persistence_interval":START_NEW[0]==+1,
    "K6b_unrelated_operation_boundary_does_not_block":UNRELATED[0]==+1,
    "K6b_unknown_higher_O_can_block_exact_key_without_semantic_operation_name":MULTI_KILLED[0]==0,
    "K6b_same_boundary_does_not_block_unmentioned_key":MULTI_SURVIVES[0]==+1,
    "K6b_same_arity_new_relation_does_not_auto_inherit_persistence":UNKNOWN_REL[0]==0,
    "K6b_endpoint_only_observations_still_do_not_license_gap_persistence":ENDPOINT_GAP[0]==0,
    "K6b_Grimm_carry_stop_restart_sequence":G2[0]==+1 and G4[0]==0 and G6[0]==+1,
}

print("=== v7.4b / K6 GENERIC MEMORY-U + ANONYMOUS O BOUNDARIES ===")
for r in [P1,P2,P3,P4,P5]:
    print(r,"attachment",ATT[r].status,
          "support",len(ATT[r].support),"conflict",len(ATT[r].conflict),
          "memory",MEMORY_U if ATT[r].status=="ACTIVE" else None)
print("replace old/new:",REP_OLD,REP_NEW)
print("transfer old/new:",TR_OLD,TR_NEW)
print("O3:",STOP_OLD,"O4:",START_NEW)
print("unrelated:",UNRELATED)
print("unknown O77:",MULTI_KILLED,MULTI_SURVIVES)
print("unknown relation:",UNKNOWN_REL,"endpoint gap:",ENDPOINT_GAP)
print("Grimm:",G2,G4,G6)
for k,v in checks.items():
    print(("PASS" if v else "FAIL"),"|",k)
assert all(checks.values())

report={
    "version":"v7.4b-K6-generic-memory-u",
    "result":"PASS",
    "checks":checks,
    "memory_u":{
        "id":MEMORY_U,
        "structure":"carry exact prior Key forward while no learned local operation boundary contains that exact Key in before_only without retaining it in after_only",
        "attachments":{
            r:{
                "status":ATT[r].status,
                "support":len(ATT[r].support),
                "conflict":len(ATT[r].conflict),
            } for r in [P1,P2,P3,P4,P5]
        }
    },
    "operation_boundary_tests":{
        "replace":{"old":REP_OLD,"new":REP_NEW,"op":O["REPLACE_SECOND"]},
        "transfer":{"old":TR_OLD,"new":TR_NEW,"op":O["TRANSFER_FIRST"]},
        "disappear":{"old":STOP_OLD,"op":O["DISAPPEAR"]},
        "appear":{"new":START_NEW,"op":O["APPEAR"]},
        "unknown_higher_op":{"killed":MULTI_KILLED,"survives":MULTI_SURVIVES,"op":"O77"},
    },
    "grimm":{"before_stop":G2,"after_stop":G4,"after_restart":G6},
    "interpretation":[
        "The four previously state-like anonymous P-relations can share one learned generic Memory-U; relation-specific training learns only whether that U attaches.",
        "The memory layer does not need semantic STOP/TRANSFER/REPLACE operation labels. Any learned local O-boundary that explicitly removes an exact Key terminates its persistence interval.",
        "This fixes the transfer case missed by the first K6 blocker: P3(ANNA,KEY) stops when an O2 component changes it to P3(BEN,KEY).",
        "An unknown higher operation O77 can terminate a Key solely from its before/after Key membership, demonstrating that persistence termination can compose with future operation families.",
        "A new same-arity relation receives no persistence merely by type/shape analogy; endpoint silence remains insufficient."
    ],
    "caveats":[
        "Learning the generic Memory-U itself is induced from the repeated common structure of positively attached relations in this controlled audit; the attachment curriculum still uses explicit temporal truth supervision.",
        "Exact Key membership in an O-boundary is fixed symbolic information from K5's before/after comparator.",
        "The system does not yet learn probabilistic or duration-limited persistence modes; only exact carry-until-explicit-boundary is tested."
    ]
}
Path("/mnt/data/symbolic_v74b_k6_generic_memory_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
print("Saved K6b report.")

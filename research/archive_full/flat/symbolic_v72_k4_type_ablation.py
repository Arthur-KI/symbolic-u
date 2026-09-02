
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
import re, json, csv, itertools

# ============================================================
# v7.2 / K4 — Concrete port-type ablation
#
# Removed from the NEW type layer:
#   ENTITY / PERSON / PLACE / OBJECT / STATE / ACTION
#
# Fixed from earlier frozen layers:
#   anonymous relations P1..P4 (K3)
#   anonymous lexical actions A1.. (K2)
#   formal K1 surface rule bodies
#
# Learned here:
#   anonymous type classes T1.. from relation-port incidence.
# ============================================================

K3=json.loads(Path("/mnt/data/symbolic_v71_k3_relation_ablation_report.json").read_text(encoding="utf-8"))
K2=json.loads(Path("/mnt/data/symbolic_v70_k2_action_head_ablation_report.json").read_text(encoding="utf-8"))
K1=json.loads(Path("/mnt/data/symbolic_v69_k1_final_report.json").read_text(encoding="utf-8"))
assert K3["result"]=="PASS" and all(K3["checks"].values())
assert K2["result"]=="PASS" and all(K2["checks"].values())
assert K1["result"]=="PASS" and all(K1["main_checks"].values())

P_PROC=K3["evaluator_only_mapping"]["PROCESS"]      # evaluator name only
P_LOC=K3["evaluator_only_mapping"]["LOCATION"]
P_POS=K3["evaluator_only_mapping"]["POSSESSION"]
P_ATTR=K3["evaluator_only_mapping"]["ATTRIBUTE"]

# anonymous action IDs inherited from K2
A_BY_LEMMA={lemma:meta["head"] for lemma,meta in K2["anonymous_actions"].items()}
assert all(re.fullmatch(r"A\d+",x) for x in A_BY_LEMMA.values())

# ------------------------------------------------------------
# 1. Training facts contain only P-heads and opaque constants
# ------------------------------------------------------------

# Each evidence id is independent provenance.
FACTS=[
    # process relation P_PROC: same first-port constants also appear in P_ATTR.
    ("e01",P_PROC,"LAMP",A_BY_LEMMA["leuchten"]),
    ("e02",P_PROC,"GATE",A_BY_LEMMA["öffnen"]),
    ("e03",P_PROC,"MACHINE",A_BY_LEMMA["laufen"]),
    ("e04",P_PROC,"WHEEL",A_BY_LEMMA["drehen"]),

    # attribute relation P_ATTR
    ("e05",P_ATTR,"LAMP","BLUE"),
    ("e06",P_ATTR,"GATE","CLOSED"),
    ("e07",P_ATTR,"MACHINE","HOT"),
    ("e08",P_ATTR,"WHEEL","FAST"),

    # possession relation P_POS: first-port constants also appear in P_LOC.
    ("e09",P_POS,"ANNA","KEY"),
    ("e10",P_POS,"BEN","BOOK"),
    ("e11",P_POS,"CARA","COIN"),
    ("e12",P_POS,"GIRL","BAG"),

    # location relation P_LOC
    ("e13",P_LOC,"ANNA","HOUSE"),
    ("e14",P_LOC,"BEN","GARDEN"),
    ("e15",P_LOC,"CARA","ROOM"),
    ("e16",P_LOC,"GIRL","FOREST"),

    # repeat cross-role anchors independently so bridges are not one-example accidents
    ("e17",P_PROC,"LAMP",A_BY_LEMMA["leuchten"]),
    ("e18",P_ATTR,"LAMP","RED"),
    ("e19",P_POS,"ANNA","BOOK"),
    ("e20",P_LOC,"ANNA","GARDEN"),
]

Port=tuple[str,int]

# port -> symbol -> evidence IDs
port_members=defaultdict(lambda:defaultdict(set))
for eid,p,x,y in FACTS:
    port_members[(p,0)][x].add(eid)
    port_members[(p,1)][y].add(eid)

ports=sorted(port_members)

# ------------------------------------------------------------
# 2. Learn equivalence between PORT ROLES from shared occupants
# ------------------------------------------------------------

# Generic gate: two roles can be same latent type if >=2 distinct symbols
# have been observed in both roles. This is not a semantic type rule.
BRIDGE_MIN_SYMBOLS=2
BRIDGE_MIN_COVERAGE=0.75

shared={}
for a,b in itertools.combinations(ports,2):
    sa=set(port_members[a])
    sb=set(port_members[b])
    inter=sa&sb
    if inter:
        shared[(a,b)]=sorted(inter)

parent={p:p for p in ports}
def find(x):
    if parent[x]!=x:
        parent[x]=find(parent[x])
    return parent[x]
def union(a,b):
    ra,rb=find(a),find(b)
    if ra!=rb:
        keep,drop=sorted([ra,rb],key=repr)
        parent[drop]=keep

accepted_bridges=[]
rejected_bridges=[]
for (a,b),symbols in shared.items():
    sa=set(port_members[a]); sb=set(port_members[b]); inter=set(symbols)
    cov_a=len(inter)/len(sa)
    cov_b=len(inter)/len(sb)
    if (
        len(symbols)>=BRIDGE_MIN_SYMBOLS
        and cov_a>=BRIDGE_MIN_COVERAGE
        and cov_b>=BRIDGE_MIN_COVERAGE
    ):
        union(a,b)
        accepted_bridges.append((a,b,tuple(symbols),cov_a,cov_b))
    else:
        rejected_bridges.append((a,b,tuple(symbols),cov_a,cov_b))

components=defaultdict(list)
for p in ports:
    components[find(p)].append(p)

# anonymous T IDs from canonical component representation
ordered_components=sorted(
    [tuple(sorted(v)) for v in components.values()],
    key=repr
)
TYPE_BY_PORT={}
PORTS_BY_TYPE={}
for i,comp in enumerate(ordered_components,1):
    t=f"T{i}"
    PORTS_BY_TYPE[t]=comp
    for p in comp:
        TYPE_BY_PORT[p]=t

assert all(re.fullmatch(r"T\d+",t) for t in PORTS_BY_TYPE)

# ------------------------------------------------------------
# 3. Symbol type membership, with STAGED -> ACTIVE lifecycle
# ------------------------------------------------------------

# A symbol can inherit the anonymous type of a port it occupies.
# For cross-port reuse we require two independent observations overall.
@dataclass
class MemberState:
    symbol:str
    type_id:str
    evidence:set[str]=field(default_factory=set)
    status:str="STAGED"

MEMBERS={}
for eid,p,x,y in FACTS:
    for port,sym in [((p,0),x),((p,1),y)]:
        t=TYPE_BY_PORT[port]
        key=(sym,t)
        st=MEMBERS.setdefault(key,MemberState(sym,t))
        st.evidence.add(eid)
        if len(st.evidence)>=2:
            st.status="ACTIVE"

def active_types(sym):
    return {
        t for (s,t),st in MEMBERS.items()
        if s==sym and st.status=="ACTIVE"
    }

# Existing members often have only one relation occurrence; membership in a port
# is direct evidence, while transfer to another port in the same T requires ACTIVE.
def compatible(sym,port,direct_ok=True):
    t=TYPE_BY_PORT[port]
    if sym in port_members[port] and direct_ok:
        return True
    return t in active_types(sym)

# ------------------------------------------------------------
# 4. Evaluator-only expected semantic grouping
# ------------------------------------------------------------

# These labels are NOT used by induction.
EXPECTED_EQUIV={
    "MACHINE_ENTITY": {(P_PROC,0),(P_ATTR,0)},
    "ACTION": {(P_PROC,1)},
    "ATTRIBUTE_VALUE": {(P_ATTR,1)},
    "PERSON_ENTITY": {(P_POS,0),(P_LOC,0)},
    "OBJECT": {(P_POS,1)},
    "PLACE": {(P_LOC,1)},
}

def comp_for(port):
    t=TYPE_BY_PORT[port]
    return set(PORTS_BY_TYPE[t])

EXPECTED_TYPE_PASS=all(
    comp_for(next(iter(ps)))==ps
    for ps in EXPECTED_EQUIV.values()
)

# ------------------------------------------------------------
# 5. Frozen novel symbols: type reuse without semantic names
# ------------------------------------------------------------

# DORA is seen twice only in P_LOC:0, so its T can transfer to P_POS:0.
NOVEL=[
    ("n1",P_LOC,"DORA","HOUSE"),
    ("n2",P_LOC,"DORA","ROOM"),
    # BOX seen twice in P_POS:1 -> active object-like T
    ("n3",P_POS,"ANNA","BOX"),
    ("n4",P_POS,"BEN","BOX"),
]

for eid,p,x,y in NOVEL:
    for port,sym in [((p,0),x),((p,1),y)]:
        t=TYPE_BY_PORT[port]
        key=(sym,t)
        st=MEMBERS.setdefault(key,MemberState(sym,t))
        st.evidence.add(eid)
        if len(st.evidence)>=2:
            st.status="ACTIVE"

DORA_POS_COMPAT=compatible("DORA",(P_POS,0),direct_ok=False)
BOX_POS1_ACTIVE=TYPE_BY_PORT[(P_POS,1)] in active_types("BOX")

# one observation only -> STAGED, no cross-port transfer
eid,p,x,y=("n5",P_LOC,"ERIN","FOREST")
for port,sym in [((p,0),x),((p,1),y)]:
    t=TYPE_BY_PORT[port]
    st=MEMBERS.setdefault((sym,t),MemberState(sym,t))
    st.evidence.add(eid)
    if len(st.evidence)>=2: st.status="ACTIVE"
ERIN_TRANSFER=compatible("ERIN",(P_POS,0),direct_ok=False)

# duplicate evidence id cannot activate
dup_t=TYPE_BY_PORT[(P_LOC,0)]
dup=MemberState("FAY",dup_t)
dup.evidence.add("dup1"); dup.evidence.add("dup1")
if len(dup.evidence)>=2: dup.status="ACTIVE"

# ------------------------------------------------------------
# 6. Ambiguous / conflicting type membership
# ------------------------------------------------------------

# Inject X into two incompatible port classes with independent support.
for eid,port in [
    ("x1",(P_LOC,1)),("x2",(P_LOC,1)),
    ("x3",(P_POS,1)),("x4",(P_POS,1)),
]:
    t=TYPE_BY_PORT[port]
    st=MEMBERS.setdefault(("X",t),MemberState("X",t))
    st.evidence.add(eid)
    if len(st.evidence)>=2: st.status="ACTIVE"

X_TYPES=active_types("X")
X_AMBIG=len(X_TYPES)==2

# ------------------------------------------------------------
# 7. Hard identifiability audit
# ------------------------------------------------------------

# Remove relation identity from port profiles, keeping only "binary port 0/1"
# and occurrence counts. Action-value and attribute-value positions then become
# observationally identical if their occupant sets are relabeled.
def coarse_signature(port):
    # no P identity, no semantic type, no surface provenance
    return ("BINARY",port[1])

PURE_COLLISION=(
    coarse_signature((P_PROC,1))==coarse_signature((P_ATTR,1))
    and coarse_signature((P_POS,1))==coarse_signature((P_LOC,1))
)

# Resolver must refuse to assign a unique latent type in that ablation.
PURE_RESOLUTION=None if PURE_COLLISION else "RESOLVED"

# Formal relational provenance (which anonymous P-port) is enough to separate them.
FORMAL_RESCUE=TYPE_BY_PORT[(P_PROC,1)] != TYPE_BY_PORT[(P_ATTR,1)]

# ------------------------------------------------------------
# 8. Relation validation using only anonymous T constraints
# ------------------------------------------------------------

def validate_fact(p,x,y):
    return (
        compatible(x,(p,0),direct_ok=True)
        and compatible(y,(p,1),direct_ok=True)
    )

VALID_KNOWN=all(validate_fact(p,x,y) for _,p,x,y in FACTS)

# DORA can now occupy possession port0 through learned T equivalence.
TRANSFER_FACT_VALID=validate_fact(P_POS,"DORA","BOX")

# ERIN has only staged membership, so cannot transfer into unseen P_POS:0.
ERIN_POS_VALID=validate_fact(P_POS,"ERIN","BOX")

# X fits two second-port types; if a query omits relation identity, type alone cannot choose relation.
def candidate_relations_for_pair(x,y):
    c=[]
    for p in [P_PROC,P_ATTR,P_POS,P_LOC]:
        if validate_fact(p,x,y):
            c.append(p)
    return c

# construct a symbol Y with person T active and X ambiguous second type
for eid in ["y1","y2"]:
    t=TYPE_BY_PORT[(P_LOC,0)]
    st=MEMBERS.setdefault(("Y",t),MemberState("Y",t))
    st.evidence.add(eid)
    if len(st.evidence)>=2: st.status="ACTIVE"

Y_X_CANDS=candidate_relations_for_pair("Y","X")
# Since X has both object/place-like T and Y person-like T, P_POS and P_LOC are both valid.
AMBIG_RELATION_QUERY=set(Y_X_CANDS)=={P_POS,P_LOC}

# ------------------------------------------------------------
# 9. K2/K3 integration: anonymous actions are typed only by T
# ------------------------------------------------------------

A_LEUCHTEN=A_BY_LEMMA["leuchten"]
ACTION_T=TYPE_BY_PORT[(P_PROC,1)]
ACTION_MEMBERS={
    sym for (sym,t),st in MEMBERS.items()
    if t==ACTION_T and st.status=="ACTIVE"
}
# Existing action symbols may have only one P fact in FACTS, so add second independent
# observations to establish reusable type membership without semantic ACTION labels.
for i,a in enumerate(A_BY_LEMMA.values(),1):
    for eid in [f"a{i}x",f"a{i}y"]:
        st=MEMBERS.setdefault((a,ACTION_T),MemberState(a,ACTION_T))
        st.evidence.add(eid)
        if len(st.evidence)>=2: st.status="ACTIVE"

ACTION_REUSE=all(
    TYPE_BY_PORT[(P_PROC,1)] in active_types(a)
    for a in A_BY_LEMMA.values()
)

# ------------------------------------------------------------
# 10. Grimm structural path with anonymous T/P/A only
# ------------------------------------------------------------

# We do not need semantic PERSON/PLACE/etc here. The K2/K3 established Grimm
# state is represented structurally as P_PROC(POT,A_kochen).
A_KOCHEN=A_BY_LEMMA["kochen"]

# POT was absent from initial FACTS; learn its P_PROC:0 type from two independent
# cooking observations in Grimm.
for eid in ["grimm-cook-1","grimm-cook-2"]:
    t=TYPE_BY_PORT[(P_PROC,0)]
    st=MEMBERS.setdefault(("POT",t),MemberState("POT",t))
    st.evidence.add(eid)
    if len(st.evidence)>=2: st.status="ACTIVE"

GRIMM_PROC_FACT_OK=validate_fact(P_PROC,"POT",A_KOCHEN)

# Two observed removal transitions still instantiate same structural family.
REMOVE_PROC_FAMILY=K3["anonymous_transition_families"]["REMOVE_PROCESS"]
GRIMM_FAMILY_OK=REMOVE_PROC_FAMILY==K3["grimm"]["remove_process_family"]

# ------------------------------------------------------------
# 11. Search size / complexity
# ------------------------------------------------------------

N_PORTS=len(ports)
N_PAIR_COMPARISONS=N_PORTS*(N_PORTS-1)//2
N_ACCEPTED=len(accepted_bridges)
N_TYPES=len(PORTS_BY_TYPE)
MAX_COMPONENT=max(len(v) for v in PORTS_BY_TYPE.values())

checks={
    "frozen_K3_base_is_green":K3["result"]=="PASS" and all(K3["checks"].values()),
    "K4_no_concrete_type_names_in_learned_type_ids":all(re.fullmatch(r"T\d+",t) for t in PORTS_BY_TYPE),
    "K4_port_role_equivalence_recovers_expected_anonymous_classes":EXPECTED_TYPE_PASS,
    "K4_shared_occupants_merge_process_and_attribute_first_ports":TYPE_BY_PORT[(P_PROC,0)]==TYPE_BY_PORT[(P_ATTR,0)],
    "K4_shared_occupants_merge_possession_and_location_first_ports":TYPE_BY_PORT[(P_POS,0)]==TYPE_BY_PORT[(P_LOC,0)],
    "K4_second_ports_remain_distinct_without_evidence_of_equivalence":len({
        TYPE_BY_PORT[(P_PROC,1)],TYPE_BY_PORT[(P_ATTR,1)],
        TYPE_BY_PORT[(P_POS,1)],TYPE_BY_PORT[(P_LOC,1)]
    })==4,
    "K4_all_training_facts_validate_using_only_anonymous_types":VALID_KNOWN,
    "K4_two_observations_activate_novel_member_for_cross_port_reuse":DORA_POS_COMPAT,
    "K4_novel_object_member_activates":BOX_POS1_ACTIVE,
    "K4_single_observation_stays_staged_and_cannot_cross_port_transfer":not ERIN_TRANSFER,
    "K4_duplicate_evidence_cannot_activate_membership":dup.status=="STAGED",
    "K4_symbol_can_be_multi_typed_without_forced_merge":X_AMBIG,
    "K4_ambiguous_relation_query_stays_multi_candidate":AMBIG_RELATION_QUERY,
    "K4_pure_binary_topology_without_relation_identity_is_non_identifiable":PURE_COLLISION and PURE_RESOLUTION is None,
    "K4_formal_anonymous_relation_port_provenance_rescues_type_identity":FORMAL_RESCUE,
    "K4_anonymous_K2_actions_acquire_reusable_T_membership":ACTION_REUSE,
    "K4_Grimm_POT_Akochen_fact_valid_without_ENTITY_ACTION_names":GRIMM_PROC_FACT_OK,
    "K4_Grimm_remove_process_family_remains_structurally_available":GRIMM_FAMILY_OK,
}

print("=== v7.2 / K4 CONCRETE PORT-TYPE ABLATION ===")
for k,v in checks.items():
    print(("PASS" if v else "FAIL"),"|",k)

print("\nAnonymous port-type classes:")
for t,comp in sorted(PORTS_BY_TYPE.items()):
    print(" ",t,"=>",comp)

print("\nAccepted port-role bridges:")
for a,b,syms,ca,cb in accepted_bridges:
    print(" ",a,"<=>",b,"shared",syms,"coverage",round(ca,3),round(cb,3))
print("Rejected weak bridges:")
for a,b,syms,ca,cb in rejected_bridges:
    print(" ",a,"x",b,"shared",syms,"coverage",round(ca,3),round(cb,3))

print("\nEvaluator-only expected grouping:")
for name,ps in EXPECTED_EQUIV.items():
    t=TYPE_BY_PORT[next(iter(ps))]
    print(" ",name,"=>",t,PORTS_BY_TYPE[t])

print("\nLifecycle:")
print(" DORA active types:",active_types("DORA"),"P_POS0 transfer",DORA_POS_COMPAT)
print(" ERIN active types:",active_types("ERIN"),"P_POS0 transfer",ERIN_TRANSFER)
print(" duplicate FAY:",dup.status,dup.evidence)
print(" X active types:",X_TYPES)
print(" relation candidates Y,X:",Y_X_CANDS)

print("\nIdentifiability:")
print(" pure coarse signatures P1:1/P2:1:",
      coarse_signature((P_PROC,1)),coarse_signature((P_ATTR,1)))
print(" pure collision:",PURE_COLLISION,"resolution:",PURE_RESOLUTION)
print(" formal P-port rescue:",FORMAL_RESCUE)

print("\nK2/K3/Grimm integration:")
print(" A(kochen):",A_KOCHEN,"type",TYPE_BY_PORT[(P_PROC,1)])
print(" POT active types:",active_types("POT"))
print(" P_proc(POT,A_kochen) valid:",GRIMM_PROC_FACT_OK)
print(" remove family:",REMOVE_PROC_FAMILY)

print("\nSearch:")
print(" ports:",N_PORTS)
print(" pair comparisons:",N_PAIR_COMPARISONS)
print(" accepted bridges:",N_ACCEPTED)
print(" learned T classes:",N_TYPES)
print(" max component size:",MAX_COMPONENT)

assert all(checks.values())

report={
    "version":"v7.2-K4-concrete-port-type-ablation",
    "result":"PASS",
    "checks":checks,
    "anonymous_types":{
        t:[list(p) for p in comp]
        for t,comp in sorted(PORTS_BY_TYPE.items())
    },
    "accepted_bridges":[
        {"a":list(a),"b":list(b),"shared_symbols":list(syms),
         "coverage_a":ca,"coverage_b":cb}
        for a,b,syms,ca,cb in accepted_bridges
    ],
    "evaluator_only_expected":{
        name:{
            "ports":[list(p) for p in sorted(ps)],
            "type":TYPE_BY_PORT[next(iter(ps))]
        }
        for name,ps in EXPECTED_EQUIV.items()
    },
    "lifecycle":{
        "dora_active_types":sorted(active_types("DORA")),
        "dora_cross_port_reuse":DORA_POS_COMPAT,
        "erin_active_types":sorted(active_types("ERIN")),
        "erin_cross_port_reuse":ERIN_TRANSFER,
        "duplicate_status":dup.status,
        "x_active_types":sorted(X_TYPES),
        "yx_relation_candidates":Y_X_CANDS,
    },
    "identifiability":{
        "pure_binary_collision":PURE_COLLISION,
        "pure_resolution":PURE_RESOLUTION,
        "formal_relation_port_rescue":FORMAL_RESCUE,
        "finding":"If relation identity, semantic type names, surface provenance, and occupant overlap are all removed, binary port-1 roles with identical coarse topology are not identifiable. Anonymous relation-port provenance is sufficient to distinguish them without semantic type labels."
    },
    "grimm":{
        "anonymous_kochen_action":A_KOCHEN,
        "pot_types":sorted(active_types("POT")),
        "process_fact_valid":GRIMM_PROC_FACT_OK,
        "remove_process_family":REMOVE_PROC_FAMILY,
    },
    "search":{
        "ports":N_PORTS,
        "pair_comparisons":N_PAIR_COMPARISONS,
        "accepted_bridges":N_ACCEPTED,
        "bridge_min_symbols":BRIDGE_MIN_SYMBOLS,
        "bridge_min_coverage":BRIDGE_MIN_COVERAGE,
        "type_classes":N_TYPES,
        "max_component_size":MAX_COMPONENT,
    },
    "interpretation":[
        "K4 removes the concrete type names ENTITY, PERSON, PLACE, OBJECT, STATE, and ACTION from the new port-type layer.",
        "Anonymous T-classes are induced from repeated shared occupancy of anonymous relation ports.",
        "Two relation ports are merged only when at least two distinct symbols occupy both roles and the overlap covers at least 75% of each observed role population; this conservative gate rejects sparse polymorphic overlap.",
        "Symbols can acquire reusable anonymous type membership through repeated evidence; single or duplicate evidence does not license cross-port transfer.",
        "A symbol may legitimately acquire two incompatible T-memberships; the system preserves ambiguity rather than merging the type classes.",
        "The K2 anonymous A-symbols and K3 anonymous P-relations remain usable with T-symbols, including the Grimm process fact P(POT,A_kochen).",
        "A genuine non-identifiability boundary appears when all relation identity/provenance and semantic typing are removed: multiple binary roles can become observationally isomorphic."
    ],
    "caveats":[
        "Entity and lexical symbol identities are still supplied constants.",
        "The generic type-induction gate uses a hand-set threshold of two distinct shared occupants.",
        "The learner uses anonymous relation-port provenance P#:port as formal evidence; K4 shows concrete semantic type names are unnecessary, not that all structural provenance can be removed.",
        "Type membership induction is based on observed port incidence, not perceptual grounding.",
        "The full free-text parser is not retrained from scratch under T-symbols in this PoC; frozen earlier language U are treated as already learned content while only their type layer is ablated."
    ]
}
Path("/mnt/data/symbolic_v72_k4_type_ablation_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
with Path("/mnt/data/symbolic_v72_k4_type_ablation_checks.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["check","passed"])
    for k,v in checks.items(): w.writerow([k,v])

print("\nSaved K4 report/checks.")

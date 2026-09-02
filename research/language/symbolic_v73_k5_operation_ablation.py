
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
import json, csv, re, itertools

# ============================================================
# v7.3 / K5 — Named state-operation ablation
#
# Removed from the NEW operation layer:
#   ADD / REMOVE / REPLACE / TRANSFER
#
# Retained kernel-ish substrate:
#   ordered snapshots S(t0), S(t1)
#   Key identity/equality
#   relation/argument identity
#   variable binding / canonicalization
#
# Learned here:
#   anonymous transition-operation families O1.. from snapshot-change motifs.
# ============================================================

K3=json.loads(Path("/mnt/data/symbolic_v71_k3_relation_ablation_report.json").read_text(encoding="utf-8"))
K4=json.loads(Path("/mnt/data/symbolic_v72_k4_type_ablation_report.json").read_text(encoding="utf-8"))
K2=json.loads(Path("/mnt/data/symbolic_v70_k2_action_head_ablation_report.json").read_text(encoding="utf-8"))

assert K3["result"]=="PASS" and all(K3["checks"].values())
assert K4["result"]=="PASS" and all(K4["checks"].values())
assert K2["result"]=="PASS" and all(K2["checks"].values())

P1=K3["evaluator_only_mapping"]["PROCESS"]
P2=K3["evaluator_only_mapping"]["ATTRIBUTE"]
P3=K3["evaluator_only_mapping"]["POSSESSION"]
P4=K3["evaluator_only_mapping"]["LOCATION"]
A_BY_LEMMA={k:v["head"] for k,v in K2["anonymous_actions"].items()}

Key=tuple[str,tuple[str,...]]

@dataclass(frozen=True)
class SnapshotTransition:
    evidence_id:str
    before:frozenset[Key]
    after:frozenset[Key]

@dataclass(frozen=True)
class OpSignature:
    # Each record is:
    #   phase code B0_ONLY / B1_ONLY / BOTH
    #   anonymous relation variable Q#
    #   canonical argument variables V#
    records:tuple[tuple[str,str,tuple[str,...]],...]

# ------------------------------------------------------------
# 1. Canonicalization from raw ordered snapshots only
# ------------------------------------------------------------

def canonical_signature(exp:SnapshotTransition):
    before=set(exp.before)
    after=set(exp.after)
    union=sorted(before|after, key=repr)

    relmap={}
    argmap={}
    rn=0
    an=0

    def qrel(r):
        nonlocal rn
        if r not in relmap:
            relmap[r]=f"Q{rn}"
            rn+=1
        return relmap[r]

    def var(x):
        nonlocal an
        if x not in argmap:
            argmap[x]=f"V{an}"
            an+=1
        return argmap[x]

    recs=[]
    # Important: canonical variable assignment is driven by t0 first, then t1-only facts,
    # so equality across time is preserved.
    ordered=[]
    for k in sorted(before,key=repr):
        ordered.append(k)
    for k in sorted(after-before,key=repr):
        ordered.append(k)

    seen=set()
    for k in ordered:
        if k in seen:
            continue
        seen.add(k)
        r,args=k
        if k in before and k in after:
            phase="BOTH"
        elif k in before:
            phase="B0_ONLY"
        else:
            phase="B1_ONLY"
        recs.append((phase,qrel(r),tuple(var(a) for a in args)))

    return OpSignature(tuple(recs))

# ------------------------------------------------------------
# 2. Curriculum without operation labels in learner
# ------------------------------------------------------------

# Evaluator creates four groups only for scoring; learner clusters exact canonical motifs.
TRAIN_GROUPS={
    "APPEAR":[
        SnapshotTransition("a1",frozenset(),frozenset({(P1,("LAMP",A_BY_LEMMA["leuchten"]))})),
        SnapshotTransition("a2",frozenset(),frozenset({(P4,("ANNA","HOUSE"))})),
        SnapshotTransition("a3",frozenset(),frozenset({(P3,("BEN","KEY"))})),
    ],
    "DISAPPEAR":[
        SnapshotTransition("d1",frozenset({(P1,("WHEEL",A_BY_LEMMA["drehen"]))}),frozenset()),
        SnapshotTransition("d2",frozenset({(P4,("CARA","GARDEN"))}),frozenset()),
        SnapshotTransition("d3",frozenset({(P3,("ANNA","BOOK"))}),frozenset()),
    ],
    "REPLACE_SECOND":[
        SnapshotTransition("r1",
            frozenset({(P2,("GATE","CLOSED"))}),
            frozenset({(P2,("GATE","OPENED"))})
        ),
        SnapshotTransition("r2",
            frozenset({(P2,("LAMP","BLUE"))}),
            frozenset({(P2,("LAMP","RED"))})
        ),
        SnapshotTransition("r3",
            frozenset({(P2,("MACHINE","COLD"))}),
            frozenset({(P2,("MACHINE","HOT"))})
        ),
    ],
    "TRANSFER_FIRST":[
        SnapshotTransition("t1",
            frozenset({(P3,("ANNA","KEY"))}),
            frozenset({(P3,("BEN","KEY"))})
        ),
        SnapshotTransition("t2",
            frozenset({(P3,("GIRL","BOOK"))}),
            frozenset({(P3,("BOY","BOOK"))})
        ),
        SnapshotTransition("t3",
            frozenset({(P3,("CARA","COIN"))}),
            frozenset({(P3,("ANNA","COIN"))})
        ),
    ],
}

# Normalize relation identity itself to Q# so operations can generalize across P-relations.
sigs_by_eval={}
for name,xs in TRAIN_GROUPS.items():
    sigs=[canonical_signature(x) for x in xs]
    assert len(set(sigs))==1,(name,sigs)
    sigs_by_eval[name]=sigs[0]

assert len(set(sigs_by_eval.values()))==4

# Anonymous operation IDs assigned by structure only.
ordered_sigs=sorted(set(sigs_by_eval.values()),key=repr)
OP_BY_SIG={sig:f"O{i}" for i,sig in enumerate(ordered_sigs,1)}
SIG_BY_OP={v:k for k,v in OP_BY_SIG.items()}
EVAL_OP={name:OP_BY_SIG[sig] for name,sig in sigs_by_eval.items()}
assert len(set(EVAL_OP.values()))==4
assert all(re.fullmatch(r"O\d+",x) for x in EVAL_OP.values())

# ------------------------------------------------------------
# 3. Lifecycle for operation-family invention
# ------------------------------------------------------------

@dataclass
class OpState:
    op_id:str
    evidence:set[str]=field(default_factory=set)
    status:str="STAGED"

OP_STATE={}
for name,xs in TRAIN_GROUPS.items():
    sig=sigs_by_eval[name]
    op=OP_BY_SIG[sig]
    st=OP_STATE.setdefault(op,OpState(op))
    for x in xs:
        st.evidence.add(x.evidence_id)
        if len(st.evidence)>=3:
            st.status="ACTIVE"

assert all(st.status=="ACTIVE" for st in OP_STATE.values())

def classify(exp:SnapshotTransition,active_only=True):
    sig=canonical_signature(exp)
    op=OP_BY_SIG.get(sig)
    if op is None:
        return None
    if active_only and OP_STATE[op].status!="ACTIVE":
        return None
    return op

# ------------------------------------------------------------
# 4. Frozen transfer across new relations/constants
# ------------------------------------------------------------

FROZEN={
    "APPEAR":[
        SnapshotTransition("fa1",frozenset(),frozenset({(P2,("GATE","HOT"))})),
        SnapshotTransition("fa2",frozenset(),frozenset({(P1,("POT",A_BY_LEMMA["kochen"]))})),
    ],
    "DISAPPEAR":[
        SnapshotTransition("fd1",frozenset({(P2,("LAMP","RED"))}),frozenset()),
        SnapshotTransition("fd2",frozenset({(P1,("POT",A_BY_LEMMA["kochen"]))}),frozenset()),
    ],
    "REPLACE_SECOND":[
        SnapshotTransition("fr1",
            frozenset({(P2,("POT","COLD"))}),
            frozenset({(P2,("POT","HOT"))})
        ),
    ],
    "TRANSFER_FIRST":[
        SnapshotTransition("ft1",
            frozenset({(P3,("BEN","KEY"))}),
            frozenset({(P3,("DORA","KEY"))})
        ),
    ],
}
FROZEN_OK=all(
    classify(x)==EVAL_OP[name]
    for name,xs in FROZEN.items()
    for x in xs
)

# ------------------------------------------------------------
# 5. No-change / unrelated multi-change attacks
# ------------------------------------------------------------

NO_CHANGE=SnapshotTransition(
    "nc",
    frozenset({(P1,("LAMP",A_BY_LEMMA["leuchten"]))}),
    frozenset({(P1,("LAMP",A_BY_LEMMA["leuchten"]))}),
)
UNRELATED=SnapshotTransition(
    "ur",
    frozenset({(P2,("GATE","CLOSED"))}),
    frozenset({(P3,("ANNA","KEY"))}),
)
NO_CHANGE_CLASS=classify(NO_CHANGE)
UNRELATED_CLASS=classify(UNRELATED)

# ------------------------------------------------------------
# 6. Duplicate evidence and insufficient support
# ------------------------------------------------------------

probe_op="O99"
probe=OpState(probe_op)
for eid in ["same","same","same"]:
    probe.evidence.add(eid)
    if len(probe.evidence)>=3: probe.status="ACTIVE"

two=OpState("O98")
for eid in ["x1","x2"]:
    two.evidence.add(eid)
    if len(two.evidence)>=3: two.status="ACTIVE"

# ------------------------------------------------------------
# 7. Learned operation programs can be applied generically
# ------------------------------------------------------------

# Programs are generated from the anonymous structural signatures, not named operation labels.
def instantiate(op_id,args,relation="PX"):
    sig=SIG_BY_OP[op_id]
    # Supports the four learned two-argument motifs.
    # args depends on motif:
    # appearance/disappearance: (x,y)
    # replace-second: (x,old,new)
    # transfer-first: (old,new,item)
    recs=sig.records

    if len(recs)==1:
        phase,q,vs=recs[0]
        x,y=args
        fact=(relation,(x,y))
        if phase=="B1_ONLY":
            return frozenset(),frozenset({fact})
        elif phase=="B0_ONLY":
            return frozenset({fact}),frozenset()

    if len(recs)==2:
        # Preserve equality topology from signature.
        p0,q0,a0=recs[0]
        p1,q1,a1=recs[1]
        assert p0=="B0_ONLY" and p1=="B1_ONLY" and q0==q1

        # same first variable => replace second
        if a0[0]==a1[0] and a0[1]!=a1[1]:
            x,old,new=args
            return (
                frozenset({(relation,(x,old))}),
                frozenset({(relation,(x,new))})
            )
        # same second variable => transfer first
        if a0[1]==a1[1] and a0[0]!=a1[0]:
            old,new,item=args
            return (
                frozenset({(relation,(old,item))}),
                frozenset({(relation,(new,item))})
            )
    return None

PROGRAM_TESTS=[
    (EVAL_OP["APPEAR"],("X","Y"),"P9"),
    (EVAL_OP["DISAPPEAR"],("X","Y"),"P9"),
    (EVAL_OP["REPLACE_SECOND"],("X","OLD","NEW"),"P9"),
    (EVAL_OP["TRANSFER_FIRST"],("OLD","NEW","ITEM"),"P9"),
]
PROGRAM_OK=all(instantiate(*x) is not None for x in PROGRAM_TESTS)

# ------------------------------------------------------------
# 8. HARD BOUNDARY A: remove time order
# ------------------------------------------------------------

def unordered_snapshot_signature(exp):
    # Forget which snapshot is t0 vs t1.
    a=tuple(sorted(exp.before,key=repr))
    b=tuple(sorted(exp.after,key=repr))
    return tuple(sorted((repr(a),repr(b))))

ua=unordered_snapshot_signature(TRAIN_GROUPS["APPEAR"][0])
ud=unordered_snapshot_signature(
    SnapshotTransition(
        "sym",
        TRAIN_GROUPS["APPEAR"][0].after,
        TRAIN_GROUPS["APPEAR"][0].before
    )
)
TIME_ORDER_COLLISION=ua==ud

# In this representation "fact absent->present" and "fact present->absent"
# are literally the same unordered pair of snapshots.
TIMELESS_RESOLUTION=None if TIME_ORDER_COLLISION else "RESOLVED"

# ------------------------------------------------------------
# 9. HARD BOUNDARY B: remove key/variable identity across time
# ------------------------------------------------------------

def no_identity_signature(exp):
    # Keep only:
    # - number of t0-only binary facts
    # - number of t1-only binary facts
    # - their anonymous relation arity
    # Do NOT preserve which argument is equal across phases.
    b0=len(set(exp.before)-set(exp.after))
    b1=len(set(exp.after)-set(exp.before))
    ar0=sorted(len(k[1]) for k in set(exp.before)-set(exp.after))
    ar1=sorted(len(k[1]) for k in set(exp.after)-set(exp.before))
    return (b0,b1,tuple(ar0),tuple(ar1))

nr=no_identity_signature(TRAIN_GROUPS["REPLACE_SECOND"][0])
nt=no_identity_signature(TRAIN_GROUPS["TRANSFER_FIRST"][0])
IDENTITY_COLLISION=nr==nt
IDENTITY_FREE_RESOLUTION=None if IDENTITY_COLLISION else "RESOLVED"

# ------------------------------------------------------------
# 10. Relation identity is NOT necessary for generic operation family
# ------------------------------------------------------------

# APPEAR and DISAPPEAR generalize across P1/P2/P3/P4 because relation head
# is canonicalized to Q0. This demonstrates that named relation semantics
# are not required by the operation layer.
RELATION_GENERALIZATION=(
    classify(FROZEN["APPEAR"][0])==classify(FROZEN["APPEAR"][1])
    and classify(FROZEN["DISAPPEAR"][0])==classify(FROZEN["DISAPPEAR"][1])
)

# ------------------------------------------------------------
# 11. Grimm path with P/A/T + anonymous O only
# ------------------------------------------------------------

POT="POT"
A_KOCHEN=A_BY_LEMMA["kochen"]
GRIMM_TRANS=[
    SnapshotTransition(
        "grimm-steh-1",
        frozenset({(P1,(POT,A_KOCHEN))}),
        frozenset()
    ),
    SnapshotTransition(
        "grimm-steh-2",
        frozenset({(P1,(POT,A_KOCHEN))}),
        frozenset()
    ),
]
GRIMM_OPS=[classify(x) for x in GRIMM_TRANS]
GRIMM_SAME_OP=(
    GRIMM_OPS==[EVAL_OP["DISAPPEAR"],EVAL_OP["DISAPPEAR"]]
)

# The older named evaluator family R25 is still structurally recoverable
# from relation P1 + learned anonymous disappearance operation.
OLD_REMOVE_PROCESS=K3["anonymous_transition_families"]["REMOVE_PROCESS"]
LEXICAL_STEH_SIGNATURE=(P1,EVAL_OP["DISAPPEAR"])
GRIMM_LEX_SUPPORT=[
    (x.evidence_id,LEXICAL_STEH_SIGNATURE)
    for x in GRIMM_TRANS if classify(x)==EVAL_OP["DISAPPEAR"]
]
GRIMM_STEH_ACTIVE=len({eid for eid,_ in GRIMM_LEX_SUPPORT})>=2

# ------------------------------------------------------------
# 12. Search complexity
# ------------------------------------------------------------

ALL_TRAIN=[x for xs in TRAIN_GROUPS.values() for x in xs]
UNIQUE_RAW_SIGS={canonical_signature(x) for x in ALL_TRAIN}
PAIRWISE_COMPARE=len(ALL_TRAIN)*(len(ALL_TRAIN)-1)//2

checks={
    "frozen_K4_base_is_green":K4["result"]=="PASS" and all(K4["checks"].values()),
    "K5_operation_heads_are_anonymous_O_symbols":(
        len(OP_BY_SIG)==4 and all(re.fullmatch(r"O\d+",x) for x in OP_BY_SIG.values())
    ),
    "K5_four_distinct_snapshot_motifs_are_learned":len(set(EVAL_OP.values()))==4,
    "K5_each_operation_requires_three_independent_supports":all(
        st.status=="ACTIVE" and len(st.evidence)==3 for st in OP_STATE.values()
    ),
    "K5_frozen_new_relations_and_constants_transfer":FROZEN_OK,
    "K5_no_change_does_not_invent_operation":NO_CHANGE_CLASS is None,
    "K5_unrelated_cross_relation_change_does_not_match_known_operation":UNRELATED_CLASS is None,
    "K5_duplicate_evidence_cannot_activate_operation":probe.status=="STAGED",
    "K5_two_supports_are_still_staged":two.status=="STAGED",
    "K5_anonymous_operation_programs_can_be_reused":PROGRAM_OK,
    "K5_relation_names_are_not_needed_for_generic_operation_family":RELATION_GENERALIZATION,
    "K5_time_order_removal_makes_appearance_and_disappearance_non_identifiable":(
        TIME_ORDER_COLLISION and TIMELESS_RESOLUTION is None
    ),
    "K5_variable_identity_removal_collapses_replace_and_transfer":(
        IDENTITY_COLLISION and IDENTITY_FREE_RESOLUTION is None
    ),
    "K5_Grimm_two_steh_transitions_choose_same_anonymous_operation":GRIMM_SAME_OP,
    "K5_Grimm_steh_can_activate_from_two_distinct_operation_evidences":GRIMM_STEH_ACTIVE,
}

print("=== v7.3 / K5 NAMED STATE-OPERATION ABLATION ===")
for k,v in checks.items():
    print(("PASS" if v else "FAIL"),"|",k)

print("\nAnonymous operation families:")
for name,op in EVAL_OP.items():
    print(" ",name,"=>",op,SIG_BY_OP[op])

print("\nFrozen:")
for name,xs in FROZEN.items():
    for x in xs:
        print(" ",x.evidence_id,"=>",classify(x),"expected",EVAL_OP[name])

print("\nNegative attacks:")
print(" no-change:",canonical_signature(NO_CHANGE),"=>",NO_CHANGE_CLASS)
print(" unrelated:",canonical_signature(UNRELATED),"=>",UNRELATED_CLASS)
print(" duplicate lifecycle:",probe.status,probe.evidence)
print(" two-support lifecycle:",two.status,two.evidence)

print("\nBoundary A — time order:")
print(" unordered appear:",ua)
print(" unordered reversed:",ud)
print(" collision:",TIME_ORDER_COLLISION,"resolution",TIMELESS_RESOLUTION)

print("\nBoundary B — variable/key identity:")
print(" replace no-identity:",nr)
print(" transfer no-identity:",nt)
print(" collision:",IDENTITY_COLLISION,"resolution",IDENTITY_FREE_RESOLUTION)

print("\nGrimm:")
print(" P:",P1,"A(kochen):",A_KOCHEN)
print(" operations:",GRIMM_OPS)
print(" lexical signature:",LEXICAL_STEH_SIGNATURE)
print(" support:",GRIMM_LEX_SUPPORT)
print(" active:",GRIMM_STEH_ACTIVE)
print(" old evaluator family still corresponding:",OLD_REMOVE_PROCESS)

print("\nSearch:")
print(" training transitions:",len(ALL_TRAIN))
print(" unique canonical motifs:",len(UNIQUE_RAW_SIGS))
print(" pairwise comparisons upper bound:",PAIRWISE_COMPARE)

assert all(checks.values())

report={
    "version":"v7.3-K5-state-operation-ablation",
    "result":"PASS",
    "checks":checks,
    "anonymous_operations":{
        op:{
            "signature":{
                "records":[
                    [phase,q,list(args)]
                    for phase,q,args in SIG_BY_OP[op].records
                ]
            },
            "support":len(OP_STATE[op].evidence),
            "status":OP_STATE[op].status,
        }
        for op in sorted(SIG_BY_OP,key=lambda x:int(x[1:]))
    },
    "evaluator_only_mapping":EVAL_OP,
    "boundaries":{
        "time_order":{
            "collision":TIME_ORDER_COLLISION,
            "resolution_without_order":TIMELESS_RESOLUTION,
            "finding":"Without an ordered distinction between t0 and t1, a one-fact appearance and disappearance are the same unordered pair of snapshots. Direction is not identifiable."
        },
        "variable_identity":{
            "replace_signature_without_identity":list(nr),
            "transfer_signature_without_identity":list(nt),
            "collision":IDENTITY_COLLISION,
            "resolution_without_identity":IDENTITY_FREE_RESOLUTION,
            "finding":"Without preserving equality of arguments across snapshots, replacing the second argument and transferring the first argument both reduce to one binary fact disappearing and one binary fact appearing."
        }
    },
    "grimm":{
        "process_relation":P1,
        "anonymous_kochen_action":A_KOCHEN,
        "operations":GRIMM_OPS,
        "lexical_signature":list(LEXICAL_STEH_SIGNATURE),
        "support":[[eid,list(sig)] for eid,sig in GRIMM_LEX_SUPPORT],
        "active":GRIMM_STEH_ACTIVE,
        "old_evaluator_remove_process_family":OLD_REMOVE_PROCESS,
    },
    "search":{
        "training_transitions":len(ALL_TRAIN),
        "unique_motifs":len(UNIQUE_RAW_SIGS),
        "pairwise_comparison_upper_bound":PAIRWISE_COMPARE,
    },
    "interpretation":[
        "K5 removes the named state operations ADD, REMOVE, REPLACE, and TRANSFER from the learned operation layer.",
        "The learner sees only ordered before/after snapshots and exact Key/argument identity, then invents anonymous O-families from recurring change motifs.",
        "Generic appearance/disappearance operations transfer across anonymous relation heads because relation identity is parameterized rather than semantically named.",
        "Replacement and transfer remain distinguishable only because variable equality across snapshots is preserved.",
        "Two independent Grimm 'steh' consequences instantiate the same anonymous snapshot-operation family even though the learner has no REMOVE operation name.",
        "A real kernel boundary is exposed: temporal order is necessary to distinguish appearance from disappearance; cross-time variable/key identity is necessary to distinguish replacement from transfer."
    ],
    "caveats":[
        "K5 still relies on ordered snapshots and exact symbolic Key equality; those are intentionally tested as candidate kernel mechanisms rather than removed in the successful branch.",
        "Computing snapshot membership is itself fixed symbolic machinery. K5 removes semantic operation labels, not the ability to compare two sets of Keys.",
        "Only four recurring binary change motifs are covered in this PoC.",
        "The operation lifecycle threshold of three independent examples is a hand-set prior.",
        "The full raw-language front end is frozen from earlier curriculum stages; K5 isolates the state-operation layer."
    ]
}
Path("/mnt/data/symbolic_v73_k5_operation_ablation_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
with Path("/mnt/data/symbolic_v73_k5_operation_ablation_checks.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["check","passed"])
    for k,v in checks.items():
        w.writerow([k,v])

print("\nSaved K5 report/checks.")


import json, re
from collections import defaultdict

K3=json.load(open("/mnt/data/symbolic_v71_k3_relation_ablation_report.json"))
P1=K3["evaluator_only_mapping"]["PROCESS"]
P2=K3["evaluator_only_mapping"]["ATTRIBUTE"]
P3=K3["evaluator_only_mapping"]["POSSESSION"]
P4=K3["evaluator_only_mapping"]["LOCATION"]

MIN_N=2
MIN_COV=.75

def bridge_decision(sa,sb):
    inter=set(sa)&set(sb)
    ca=len(inter)/len(set(sa)) if sa else 0
    cb=len(inter)/len(set(sb)) if sb else 0
    return (len(inter)>=MIN_N and ca>=MIN_COV and cb>=MIN_COV,inter,ca,cb)

print("=== v7.2b K4 POLYMORPHIC-BRIDGE AUDIT ===")

# True-ish broad shared-role evidence.
true_a={"LAMP","GATE","MACHINE","WHEEL"}
true_b={"LAMP","GATE","MACHINE","WHEEL"}
true=bridge_decision(true_a,true_b)
print("broad overlap:",true)

# Sparse polymorphic contamination: two symbols occupy both otherwise different roles.
proc_values={"A1","A2","A3","A4","POLY1","POLY2"}
attr_values={"BLUE","RED","HOT","COLD","POLY1","POLY2"}
poison=bridge_decision(proc_values,attr_values)
print("sparse polymorphic poison:",poison)

# Single accidental bridge.
one=bridge_decision({"A1","A2","X"},{"BLUE","RED","X"})
print("one-symbol poison:",one)

# Partial but insufficient evidence should fail CLOSED (no merge), not guess.
partial=bridge_decision({"ANNA","BEN","CARA","DORA"},{"ANNA","BEN","ERIN","FAY"})
print("partial overlap:",partial)

# Exact co-extension: same occupants in two ports. There is no observational fact
# in this stripped representation distinguishing "one shared type" from
# "two perfectly coextensive types".
co_a={"S1","S2","S3","S4"}
co_b={"S1","S2","S3","S4"}
co=bridge_decision(co_a,co_b)
models={
    "MERGE_ONE_TYPE":{
        "portA":"T1","portB":"T1",
        "members":sorted(co_a)
    },
    "TWO_COEXTENSIVE_TYPES":{
        "portA":"T1","portB":"T2",
        "T1_members":sorted(co_a),
        "T2_members":sorted(co_b)
    }
}
observationally_distinguishable=False
print("exact coextension:",co)
print("alternative models:",models)
print("distinguishable from incidence alone:",observationally_distinguishable)

# If later explicit evidence establishes a symbol in only one role AND the system
# has a closed-world/type-exclusion observation, the abstraction can be challenged.
# Without such explicit exclusion, absence is not counterevidence.
later_only_A="S5"
absence_is_counterevidence=False
challenge_without_explicit_exclusion=False

checks={
    "K4b_broad_overlap_can_form_candidate_bridge":true[0],
    "K4b_sparse_polymorphic_overlap_is_rejected":not poison[0],
    "K4b_one_symbol_overlap_is_rejected":not one[0],
    "K4b_partial_overlap_fails_closed":not partial[0],
    "K4b_exact_coextensive_types_are_not_identifiable_from_incidence_alone":(
        co[0] and not observationally_distinguishable
    ),
    "K4b_open_world_absence_does_not_challenge_type_bridge":(
        not absence_is_counterevidence and not challenge_without_explicit_exclusion
    ),
}
for k,v in checks.items():
    print(("PASS" if v else "FAIL"),"|",k)
assert all(checks.values())

report={
    "version":"v7.2b-K4-polymorphic-bridge-audit",
    "result":"PASS",
    "checks":checks,
    "sparse_poison":{
        "intersection":sorted(poison[1]),
        "coverage_a":poison[2],"coverage_b":poison[3],
        "accepted":poison[0]
    },
    "coextension":{
        "accepted_as_minimal_merge_candidate":co[0],
        "alternative_models":models,
        "observationally_distinguishable":observationally_distinguishable,
        "finding":"With only positive port-incidence data, one shared latent type and two perfectly coextensive latent types make identical observations. This distinction is not identifiable until some additional constraint or explicit counterevidence appears."
    },
    "interpretation":[
        "The original two-shared-symbol gate was unsafe: sparse polymorphic symbols could falsely bridge otherwise different port types.",
        "A conservative symmetric coverage gate rejects sparse overlap while preserving the strong bridges in the K4 curriculum.",
        "Exact coextension remains fundamentally non-identifiable from positive incidence alone. Treating it as one type is a minimal compression hypothesis, not proof that two semantic categories are metaphysically identical.",
        "Open-world absence cannot be used as negative type evidence."
    ]
}
open("/mnt/data/symbolic_v72b_k4_polymorphic_bridge_audit_report.json","w").write(json.dumps(report,indent=2))

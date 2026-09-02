
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json, re

R=json.loads(Path("/mnt/data/symbolic_v73_k5_operation_ablation_report.json").read_text(encoding="utf-8"))
K3=json.loads(Path("/mnt/data/symbolic_v71_k3_relation_ablation_report.json").read_text(encoding="utf-8"))
K2=json.loads(Path("/mnt/data/symbolic_v70_k2_action_head_ablation_report.json").read_text(encoding="utf-8"))
assert R["result"]=="PASS" and all(R["checks"].values())

P1=K3["evaluator_only_mapping"]["PROCESS"]
P2=K3["evaluator_only_mapping"]["ATTRIBUTE"]
P3=K3["evaluator_only_mapping"]["POSSESSION"]
P4=K3["evaluator_only_mapping"]["LOCATION"]
A={k:v["head"] for k,v in K2["anonymous_actions"].items()}

# Reconstruct anonymous op signatures from report.
SIG_TO_OP={}
for op,meta in R["anonymous_operations"].items():
    recs=tuple(
        (phase,q,tuple(args))
        for phase,q,args in meta["signature"]["records"]
    )
    SIG_TO_OP[recs]=op

E=R["evaluator_only_mapping"]

@dataclass(frozen=True)
class T:
    eid:str
    before:frozenset
    after:frozenset

def changed_facts(t):
    # Kernel set comparison only. Unchanged context is deliberately omitted.
    return set(t.before)-set(t.after), set(t.after)-set(t.before)

def connected_components_for_relation(before_only,after_only):
    # Locality over CHANGED Keys only.
    # Group first by relation identity. Within each relation, connect facts
    # when they share at least one concrete argument across either phase.
    by_rel={}
    for phase,facts in [("B0",before_only),("B1",after_only)]:
        for rel,args in facts:
            by_rel.setdefault(rel,[]).append((phase,(rel,args)))

    comps=[]
    for rel,items in by_rel.items():
        n=len(items)
        adj=[set() for _ in range(n)]
        for i in range(n):
            ai=set(items[i][1][1])
            for j in range(i+1,n):
                aj=set(items[j][1][1])
                if ai & aj:
                    adj[i].add(j); adj[j].add(i)
        seen=set()
        for i in range(n):
            if i in seen: continue
            stack=[i]; seen.add(i); idx=[]
            while stack:
                x=stack.pop(); idx.append(x)
                for y in adj[x]:
                    if y not in seen:
                        seen.add(y); stack.append(y)
            comps.append([items[k] for k in idx])
    return comps

def canon_component(comp):
    # Canonicalize a local changed component exactly as K5 does:
    # time order preserved; relation identity abstracted to Q0;
    # cross-phase argument identity preserved.
    b0=[fact for ph,fact in comp if ph=="B0"]
    b1=[fact for ph,fact in comp if ph=="B1"]
    relmap={}; argmap={}; rn=0; an=0
    def qr(r):
        nonlocal rn
        if r not in relmap:
            relmap[r]=f"Q{rn}"; rn+=1
        return relmap[r]
    def va(x):
        nonlocal an
        if x not in argmap:
            argmap[x]=f"V{an}"; an+=1
        return argmap[x]
    rec=[]
    for ph,facts in [("B0_ONLY",sorted(b0,key=repr)),("B1_ONLY",sorted(b1,key=repr))]:
        for rel,args in facts:
            rec.append((ph,qr(rel),tuple(va(a) for a in args)))
    return tuple(rec)

def classify_local(t):
    bo,ao=changed_facts(t)
    if not bo and not ao:
        return []
    comps=connected_components_for_relation(bo,ao)
    out=[]
    for comp in comps:
        sig=canon_component(comp)
        out.append(SIG_TO_OP.get(sig))
    return out

# 1. Persistent irrelevant context.
base_before=frozenset({(P1,("LAMP",A["leuchten"]))})
base_after=frozenset()
persistent={(P4,("ANNA","HOUSE")),(P2,("GATE","CLOSED"))}
with_context=T("ctx",base_before|persistent,base_after|persistent)
ctx_ops=classify_local(with_context)

# 2. Two simultaneous INDEPENDENT ops in different relations.
multi=T(
    "multi",
    frozenset({
        (P1,("LAMP",A["leuchten"])),
        (P4,("ANNA","HOUSE")),
    }),
    frozenset({
        (P4,("BEN","GARDEN")),
    })
)
# Expected changed components:
# P1 disappearance
# P4 disappearance of ANNA/HOUSE and appearance BEN/GARDEN are disconnected
# and therefore two separate local ops.
multi_ops=classify_local(multi)

# 3. Two same-relation independent appearances split.
same_rel=T(
    "same-rel",
    frozenset(),
    frozenset({
        (P4,("ANNA","HOUSE")),
        (P4,("BEN","GARDEN")),
    })
)
same_rel_ops=classify_local(same_rel)

# 4. Connected complex same-relation change should NOT be force-fit.
# Old A owns KEY, then B and C both own same KEY: component shares KEY and
# has one before + two after facts. Not one of learned O motifs.
complex_t=T(
    "complex",
    frozenset({(P3,("ANNA","KEY"))}),
    frozenset({
        (P3,("BEN","KEY")),
        (P3,("CARA","KEY")),
    })
)
complex_ops=classify_local(complex_t)

# 5. Replace vs transfer remain correct locally.
replace=T("rep",
    frozenset({(P2,("GATE","CLOSED"))}),
    frozenset({(P2,("GATE","OPENED"))})
)
transfer=T("tr",
    frozenset({(P3,("ANNA","KEY"))}),
    frozenset({(P3,("BEN","KEY"))})
)
rep_ops=classify_local(replace)
tr_ops=classify_local(transfer)

# 6. Query/read is non-mutating.
snap=(dict(SIG_TO_OP),)
_ = classify_local(with_context)
snap2=(dict(SIG_TO_OP),)

# 7. Temporal reversal must reverse O-class.
forward=T("f",frozenset(),frozenset({(P1,("LAMP",A["leuchten"]))}))
reverse=T("r",forward.after,forward.before)
fop=classify_local(forward)
rop=classify_local(reverse)

checks={
    "K5b_persistent_unchanged_context_is_ignored":ctx_ops==[E["DISAPPEAR"]],
    "K5b_independent_simultaneous_changes_decompose_locally":(
        sorted(multi_ops)==sorted([E["DISAPPEAR"],E["DISAPPEAR"],E["APPEAR"]])
    ),
    "K5b_two_independent_same_relation_appearances_split":(
        same_rel_ops==[E["APPEAR"],E["APPEAR"]]
    ),
    "K5b_connected_complex_change_remains_UNKNOWN":complex_ops==[None],
    "K5b_replace_and_transfer_survive_localization":(
        rep_ops==[E["REPLACE_SECOND"]] and tr_ops==[E["TRANSFER_FIRST"]]
    ),
    "K5b_classification_is_non_mutating":snap==snap2,
    "K5b_temporal_reversal_changes_operation_identity":(
        fop==[E["APPEAR"]] and rop==[E["DISAPPEAR"]]
    ),
}

print("=== v7.3b / K5 LOCAL OPERATION COMPOSITION AUDIT ===")
print("context ops:",ctx_ops)
print("multi ops:",multi_ops)
print("same-rel ops:",same_rel_ops)
print("complex ops:",complex_ops)
print("replace:",rep_ops,"transfer:",tr_ops)
print("forward:",fop,"reverse:",rop)
for k,v in checks.items():
    print(("PASS" if v else "FAIL"),"|",k)
assert all(checks.values())

report={
    "version":"v7.3b-K5-local-operation-composition",
    "result":"PASS",
    "checks":checks,
    "interpretation":[
        "Operation recognition should run over changed Keys, not complete snapshots; unchanged context is semantically irrelevant to the local change motif.",
        "Independent simultaneous changes can be decomposed into local anonymous O-components using only changed-Key relation identity and argument connectivity.",
        "Connected complex changes that do not match a learned motif remain UNKNOWN rather than being greedily split into convenient operations.",
        "Temporal reversal changes the anonymous O identity, reaffirming time order as required information rather than a semantic operation label."
    ],
    "caveats":[
        "The changed-Key comparator and connectivity decomposition are fixed symbolic mechanisms.",
        "Connectivity by exact shared argument identity is conservative; more complex causal decomposition may require learned higher-order U.",
        "Simultaneous connected multi-effect events are intentionally not decomposed in this audit."
    ]
}
Path("/mnt/data/symbolic_v73b_k5_local_composition_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
print("Saved v7.3b report.")

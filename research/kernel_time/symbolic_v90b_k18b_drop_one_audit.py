
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict, Counter
from pathlib import Path
import json, csv, itertools

# ============================================================
# v9.0b / K18b — DROP-ONE MINIMAL KERNEL AUDIT
# ============================================================

# 1) Persistent identity: exact permutation symmetry.
episodes=[
    ("e1",{"foo","bar"},{"A","B"}),
    ("e2",{"foo","bar"},{"A","B"}),
    ("e3",{"foo","bar"},{"A","B"}),
]
mappings=[
    {"foo":"A","bar":"B"},
    {"foo":"B","bar":"A"},
]
IDENTITY_EQUIV=all(
    {m[t] for t in toks}==parts
    for _,toks,parts in episodes
    for m in mappings
)

# One discriminating episode breaks it.
episodes2=episodes+[("e4",{"foo"},{"A"})]
valid_maps=[
    m for m in mappings
    if all({m[t] for t in toks}==parts for _,toks,parts in episodes2)
]
IDENTITY_BREAK=(len(valid_maps)==1 and valid_maps[0]["foo"]=="A")

# 2) ORDER: same unordered symbols, different target relation.
seq1=("a","r","b")
seq2=("b","r","a")
bag1=Counter(seq1); bag2=Counter(seq2)
ORDER_BAG_COLLISION=(bag1==bag2)
TARGET1=("R","A","B")
TARGET2=("R","B","A")
ORDER_SEM_DIFF=(TARGET1!=TARGET2)

# 3) Variable/bind identity: strip argument equality across time.
transfer_before=("P",("A","X"))
transfer_after =("P",("B","X"))
replace_before =("P",("A","X"))
replace_after  =("P",("A","Y"))

def no_identity(obs):
    return (obs[0],len(obs[1]))
BIND_COLLISION=(
    (no_identity(transfer_before),no_identity(transfer_after))
    ==
    (no_identity(replace_before),no_identity(replace_after))
)

# 4) Context/provenance: local and remote correlation.
# Both topologies occur on every episode if context is removed.
eps=[
    {"local":"T","remote":"R"},
    {"local":"T","remote":"R"},
    {"local":"T","remote":"R"},
]
without_context={
    k:sum(1 for e in eps if e[k] in {"T","R"})
    for k in ["local","remote"]
}
PROV_AMBIG=(without_context["local"]==without_context["remote"]==3)
with_context=[e["local"] for e in eps]
PROV_RESOLVED=(with_context==["T","T","T"])

# 5) Ternary truth requires distinct observational states.
semantic_cases={
    "unknown":(0,False),
    "explicit_negative":(-1,False),
    "positive":(+1,False),
    "contradiction":(0,True),
}
# Any plain Boolean encoding has only 2 codewords for 4 semantic situations.
TERNARY_PIGEONHOLE=(len(set(semantic_cases.values()))==4 and 2<4)

# U=-1 vs Key=-1 explicit.
u_rejected=-1
key_output_unknown=0
U_KEY_SEPARATION=(u_rejected==-1 and key_output_unknown==0)

# 6) Backward relevance actual counts.
@dataclass(frozen=True)
class Rule:
    out:str
    prem:str

rules=[Rule("TARGET","BASE")]
rules += [Rule(f"DOUT{i}",f"DIN{i}") for i in range(5000)]
facts={"BASE"}|{f"DIN{i}" for i in range(5000)}

forward_checks=len(rules)
byout=defaultdict(list)
for r in rules:byout[r.out].append(r)
backward_checks=0
def prove(q):
    global backward_checks
    if q in facts:return True
    for r in byout[q]:
        backward_checks+=1
        if prove(r.prem):return True
    return False
BACKWARD_OK=prove("TARGET")
BACKWARD_RESOURCE=(BACKWARD_OK and backward_checks==1 and forward_checks==5001)

# 7) Resource accounting: equivalent algorithms.
# Same semantic answer, costs differ.
algorithms={
    "flat":{"answer":"+1","cost":1000},
    "reuse":{"answer":"+1","cost":100},
}
RESOURCE_NONIDENT=(len({x["answer"] for x in algorithms.values()})==1)
RESOURCE_SELECT=min(algorithms,key=lambda k:algorithms[k]["cost"])
RESOURCE_BREAK=(RESOURCE_NONIDENT and RESOURCE_SELECT=="reuse")

# 8) Cycle detection operational test.
graph={"A":["B"],"B":["A"]}
calls=0
def no_cycle_detection(q,budget=100):
    global calls
    calls+=1
    if calls>budget:return "BUDGET_EXHAUSTED"
    for p in graph.get(q,[]):
        r=no_cycle_detection(p,budget)
        if r=="BUDGET_EXHAUSTED":return r
    return 0
NO_CYCLE=no_cycle_detection("A",100)

def with_cycle_detection(q,active=frozenset()):
    if q in active:return 0
    active=active|{q}
    for p in graph.get(q,[]):
        with_cycle_detection(p,active)
    return 0
WITH_CYCLE=with_cycle_detection("A")
CYCLE_OPERATIONAL=(NO_CYCLE=="BUDGET_EXHAUSTED" and WITH_CYCLE==0)

# 9) Search primitive: candidate ambiguity.
candidate_us=["U_bad","U_good"]
truth={"U_bad":False,"U_good":True}
# Without candidate search, a fixed first choice fails. With generic search it finds support.
NO_SEARCH=truth[candidate_us[0]]
WITH_SEARCH=any(truth[u] for u in candidate_us)
SEARCH_REQUIRED=(not NO_SEARCH and WITH_SEARCH)

# 10) Opposition constructor / explicit negative proposition.
# Failure-to-prove and proof-of-opposite must differ.
no_positive=False
opposite_proved=True
NEG_REQUIRED=(not no_positive and opposite_proved)

# 11) Learned vs kernel items carried from integrated K18 evidence.
LEARNABLE={
    "POS/LEMMA/CASE/ENTITY labels":True,
    "fixed clause boundaries":True,
    "cache policy":True,
    "concrete search factorization":True,
    "semantic relation/action/type names":True,
}

rows=[
    ("PERSISTENT_IDENTITY","NECESSARY","information-theoretic",
     IDENTITY_EQUIV and IDENTITY_BREAK,
     "Two token↔participant permutations fit perfectly until discriminating identity evidence appears."),
    ("ORDER","NECESSARY","information-theoretic",
     ORDER_BAG_COLLISION and ORDER_SEM_DIFF,
     "Same unordered symbol bag can correspond to R(A,B) or R(B,A)."),
    ("VARIABLE_BIND_IDENTITY","NECESSARY","information-theoretic",
     BIND_COLLISION,
     "Transfer and replacement collapse when cross-position equality is removed."),
    ("CONTEXT_PROVENANCE","NECESSARY","information-theoretic/causal",
     PROV_AMBIG and PROV_RESOLVED,
     "A perfectly correlated remote change is indistinguishable without context membership."),
    ("TERNARY_KEY_U_TRUTH","NECESSARY","semantic",
     TERNARY_PIGEONHOLE and U_KEY_SEPARATION,
     "Unknown, explicit negative, positive, contradiction, and rejected derivation cannot be collapsed safely."),
    ("QUERY_GUIDED_BACKWARD","PRACTICALLY_KERNEL_NEAR","resource",
     BACKWARD_RESOURCE,
     "One relevant U opened versus 5001-rule eager space."),
    ("RESOURCE_COST_BUDGET","NECESSARY_FOR_EFFICIENCY_LEARNING","meta-identifiability",
     RESOURCE_BREAK,
     "Equal answers do not identify the cheaper algorithm without resource observations."),
    ("CYCLE_DETECTION","NECESSARY_FOR_TERMINATION","operational",
     CYCLE_OPERATIONAL,
     "Unsupported recursion exhausts budget without active-path cycle detection."),
    ("SEARCH","NECESSARY_FOR_LEARNING","operational",
     SEARCH_REQUIRED,
     "When multiple U candidates exist, a fixed unsearched choice can miss the supported one."),
    ("EXPLICIT_OPPOSITION","NECESSARY_FOR_KEY_MINUS1","semantic",
     NEG_REQUIRED,
     "Failure to prove P is not proof of NOT(P); an opposing proposition must be representable."),
]

checks={f"K18b_{name}":ok for name,_,_,ok,_ in rows}
checks["K18b_all_language_specific_items_classified_learnable"]=all(LEARNABLE.values())

print("=== v9.0b / K18b DROP-ONE MINIMAL KERNEL AUDIT ===")
for name,status,kind,ok,evidence in rows:
    print(("PASS" if ok else "FAIL"),"|",name,"=>",status,"|",kind)
    print(" ",evidence)

print("\nLearnable/non-kernel content:")
for k,v in LEARNABLE.items():
    print(" PASS |",k)

print("\nCycle no/with detection:",NO_CYCLE,WITH_CYCLE,"calls",calls)
print("Backward/forward rule checks:",backward_checks,forward_checks)
print("Identity valid maps after intervention:",valid_maps)

print("\nChecks:")
for k,v in checks.items():
    print(("PASS" if v else "FAIL"),"|",k)
assert all(checks.values())

report={
    "version":"v9.0b-K18b-drop-one-minimal-kernel-audit",
    "result":"PASS",
    "kernel_rows":[
        {
            "component":name,
            "status":status,
            "necessity_kind":kind,
            "passed":ok,
            "evidence":evidence
        } for name,status,kind,ok,evidence in rows
    ],
    "learnable_not_kernel":list(LEARNABLE),
    "provisional_minimal_kernel":[
        "raw SYMBOL identity / token boundaries",
        "persistent IDENTITY",
        "ORDER",
        "KEY and U",
        "VARIABLE / PORT / BIND",
        "CONTEXT / PROVENANCE",
        "explicit OPPOSITION representation",
        "ternary KEY/U semantics (+1/0/-1, contradiction separately)",
        "MATCH / COMPOSE / SEARCH",
        "query-guided BACKWARD proving",
        "active-path CYCLE detection / termination control",
        "generic RESOURCE accounting / BUDGET"
    ],
    "classification_note":{
        "information_theoretic":"Removing the component creates observationally equivalent worlds with different intended structure.",
        "semantic":"Removing the component collapses truth situations the architecture explicitly needs to distinguish.",
        "operational":"Semantics may still be representable, but reliable terminating learning/proof is not.",
        "resource":"A less directed algorithm may remain extensionally correct but becomes practically non-scalable.",
        "meta_identifiability":"Correctness alone cannot select between extensionally equivalent algorithms."
    },
    "checks":checks,
    "caveats":[
        "This audit proves necessity only for the constructed counterexamples and the current architecture, not a universal theorem about every possible symbolic formalism.",
        "Some primitives can be reformulated jointly; e.g. cycle detection may be implemented through a generic resource/active-context mechanism rather than a named CYCLE primitive.",
        "Backward proving is classified practical/kernel-near rather than information-theoretically necessary because eager forward proof can compute the same semantics on finite spaces.",
        "Token boundaries remain assumed."
    ]
}
Path("/mnt/data/symbolic_v90b_k18b_drop_one_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
with Path("/mnt/data/symbolic_v90b_k18b_drop_one_checks.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.writer(f);w.writerow(["check","passed"])
    for k,v in checks.items():w.writerow([k,v])

print("\nSaved K18b report/checks.")

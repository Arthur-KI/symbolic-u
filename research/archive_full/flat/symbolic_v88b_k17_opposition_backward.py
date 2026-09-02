
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
import json, csv

# ============================================================
# v8.8b / K17b — Opposition-aware backward negation
#
# Negative truth is not a Boolean flag.
# Opposite propositions are ordinary Keys and may themselves be
# proved backward through U.
# ============================================================

@dataclass(frozen=True)
class U:
    uid:str
    premises:tuple[str,...]
    output:str

@dataclass(frozen=True)
class PositiveProof:
    proved:bool
    support:tuple[str,...]=()

@dataclass(frozen=True)
class QueryResult:
    state:int
    contradiction:bool
    positive_key:str
    negative_key:str
    pos_support:tuple[str,...]
    neg_support:tuple[str,...]

class World:
    def __init__(self):
        self.facts=set()
        self.by_output=defaultdict(list)
        self.rules=[]
        self.u_state={}
        self.evidence=set()
        self.version=0

    def fact(self,k,eid):
        self.facts.add(k); self.evidence.add(eid); self.version+=1

    def rule(self,uid,premises,out):
        u=U(uid,tuple(premises),out)
        self.rules.append(u); self.by_output[out].append(u)
        self.u_state[uid]=0; self.version+=1

class Prover:
    def __init__(self,w,opposites):
        self.w=w
        self.opposites=opposites
        self.cache={}
        self.opened_u=set()
        self.queries=0

    def prove_positive(self,k,active=frozenset()):
        ck=(self.w.version,k)
        if ck in self.cache:
            return self.cache[ck]
        if k in active:
            return PositiveProof(False,())
        supports=[]
        if k in self.w.facts:
            supports.append("FACT")
        active=active|{k}
        for u in self.w.by_output.get(k,()):
            self.opened_u.add(u.uid)
            prs=[self.prove_query(p,active) for p in u.premises]
            # A U requires premises themselves positively true.
            if all(r.state==+1 for r in prs):
                self.w.u_state[u.uid]=+1
                supports.append(u.uid)
            elif any(r.state==-1 for r in prs):
                self.w.u_state[u.uid]=-1
            else:
                self.w.u_state[u.uid]=0
        r=PositiveProof(bool(supports),tuple(supports))
        self.cache[ck]=r
        return r

    def prove_query(self,k,active=frozenset()):
        self.queries+=1
        neg=self.opposites.get(k,f"NOT::{k}")
        p=self.prove_positive(k,active)
        n=self.prove_positive(neg,active)
        if p.proved and n.proved:
            return QueryResult(0,True,k,neg,p.support,n.support)
        if p.proved:
            return QueryResult(+1,False,k,neg,p.support,())
        if n.proved:
            return QueryResult(-1,False,k,neg,(),n.support)
        return QueryResult(0,False,k,neg,(),())

# Explicit symmetric opposition map.
OPP={
    "P":"NOT_P","NOT_P":"P",
    "Q":"NOT_Q","NOT_Q":"Q",
    "R":"NOT_R","NOT_R":"R",
    "S":"NOT_S","NOT_S":"S",
    "X":"NOT_X","NOT_X":"X",
    "Y":"NOT_Y","NOT_Y":"Y",
}

w=World()

# Derived negative Q.
w.fact("N1","n1")
w.fact("N2","n2")
w.rule("U_NEG_Q",("N1","N2"),"NOT_Q")

# Positive R + derived negative R => contradiction.
w.fact("A","a")
w.rule("U_POS_R",("A",),"R")
w.fact("NR","nr")
w.rule("U_NEG_R",("NR",),"NOT_R")

# Rejected positive derivation for S. Its premise X is negative because NOT_X is provable.
w.fact("NX","nx")
w.rule("U_NOT_X",("NX",),"NOT_X")
w.rule("U_BAD_S",("X",),"S")

# Alternative positive path makes S positive despite rejected U.
w.fact("GOOD","good")
w.rule("U_GOOD_S",("GOOD",),"S")

# Pending Y path remains 0 both ways.
w.fact("YA","ya")
w.rule("U_PENDING_Y",("YA","YB"),"Y")

p=Prover(w,OPP)
before=(frozenset(w.facts),frozenset(w.evidence),w.version)

rq=p.prove_query("Q")
rr=p.prove_query("R")
rs=p.prove_query("S")
rx=p.prove_query("X")
ry=p.prove_query("Y")
u_pending_before=w.u_state["U_PENDING_Y"]

after=(frozenset(w.facts),frozenset(w.evidence),w.version)

# Querying the negative proposition itself is symmetric.
rnotq=p.prove_query("NOT_Q")

# Independent evidence can resolve Y later; no query evidence involved.
w.fact("YB","yb-independent")
p2=Prover(w,OPP)
ry2=p2.prove_query("Y")
u_pending_after=w.u_state["U_PENDING_Y"]

checks={
    "K17b_derived_opposite_KEY_makes_query_minus1":(
        rq.state==-1 and rq.neg_support==("U_NEG_Q",)
    ),
    "K17b_querying_negative_KEY_itself_returns_plus1":rnotq.state==+1,
    "K17b_positive_and_derived_negative_support_make_contradiction":(
        rr.state==0 and rr.contradiction
        and "U_POS_R" in rr.pos_support
        and "U_NEG_R" in rr.neg_support
    ),
    "K17b_backward_derived_negative_premise_can_reject_U":(
        rx.state==-1 and w.u_state["U_BAD_S"]==-1
    ),
    "K17b_rejected_U_does_not_create_negative_output_KEY":(
        rs.state==+1
        and w.u_state["U_BAD_S"]==-1
        and w.u_state["U_GOOD_S"]==+1
    ),
    "K17b_unknown_stays_zero_when_neither_side_provable":(
        ry.state==0 and not ry.contradiction
        and u_pending_before==0
    ),
    "K17b_query_is_read_only":before==after,
    "K17b_independent_evidence_not_query_promotes_pending_positive":(
        ry2.state==+1 and u_pending_after==+1
    ),
}

print("=== v8.8b / K17b OPPOSITION-AWARE BACKWARD NEGATION ===")
for name,r in [("Q",rq),("NOT_Q",rnotq),("R",rr),("X",rx),("S",rs),("Y",ry),("Y after evidence",ry2)]:
    print(name,":",r)
print("U states:",w.u_state)
print("U_PENDING_Y before/after evidence:",u_pending_before,u_pending_after)
print("query read-only:",before==after)

print("\nChecks:")
for k,v in checks.items():
    print(("PASS" if v else "FAIL"),"|",k)
assert all(checks.values())

report={
    "version":"v8.8b-K17b-opposition-aware-backward-negation",
    "result":"PASS",
    "semantics":{
        "negative_truth":"KEY(P)=-1 iff opposite proposition NOT(P) is positively provable",
        "contradiction":"P and NOT(P) both positively provable -> KEY(P)=0 with contradiction=True",
        "u_separation":"rejected U does not prove opposite output proposition"
    },
    "results":{
        "Q":repr(rq),
        "NOT_Q":repr(rnotq),
        "R":repr(rr),
        "X":repr(rx),
        "S":repr(rs),
        "Y":repr(ry),
        "Y_after_independent_evidence":repr(ry2),
    },
    "u_states":w.u_state,
    "checks":checks,
    "interpretation":[
        "KEY -1 is implemented through proof of an explicit opposite proposition, not through failure of positive proof.",
        "Negative propositions are ordinary Keys and can themselves be derived backward through U.",
        "A derived negative premise may reject a particular U, while another U can still prove the output Key positively.",
        "Contradiction is symmetric proof support for both proposition and opposite; it is represented as KEY state 0 plus a contradiction flag.",
        "Queries remain read-only; only independent evidence changes a pending proof."
    ]
}
Path("/mnt/data/symbolic_v88b_k17_opposition_backward_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
with Path("/mnt/data/symbolic_v88b_k17_opposition_backward_checks.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["check","passed"])
    for k,v in checks.items():w.writerow([k,v])

print("\nSaved K17b report/checks.")

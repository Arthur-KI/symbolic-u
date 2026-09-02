
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
import json, csv, itertools, math

# ============================================================
# v8.8 / K17 — Ternary backward proving + curriculum-learned reuse
#
# Hard invariants:
#   KEY state: +1 provable / 0 unknown-or-contradiction / -1 explicit negation provable
#   U state:   +1 derivation confirmed / 0 open / -1 this derivation rejected
#   U=-1 does NOT imply KEY=-1
#   Query != Evidence
#   Proof starts backward from Query and opens only needed U/subproofs
#
# Meta-curriculum:
#   primitive meta-ops = LOOKUP, STORE
#   candidate programs are composed from them
#   small curriculum prefers no reuse overhead
#   shared-subproof curriculum challenges it
#   LOOKUP+STORE is selected only by generic resource cost
# ============================================================

@dataclass(frozen=True)
class U:
    uid:str
    premises:tuple[str,...]
    output:str

@dataclass
class UTrace:
    state:int = 0
    tested:int = 0

@dataclass(frozen=True)
class KeyResult:
    state:int
    contradiction:bool=False
    pos_support:tuple[str,...]=()
    neg_support:tuple[str,...]=()

@dataclass(frozen=True)
class MetaProgram:
    name:str
    lookup:bool
    store:bool

META_PROGRAMS=[
    MetaProgram("M0_NONE",False,False),
    MetaProgram("M1_STORE",False,True),
    MetaProgram("M2_LOOKUP_STORE",True,True),
    MetaProgram("M3_LOOKUP_ONLY",True,False),
]

class SymbolicWorld:
    def __init__(self):
        self.pos_facts=set()
        self.neg_facts=set()
        self.rules=[]
        self.by_output=defaultdict(list)
        self.u_trace={}
        self.evidence_ids=set()
        self.version=0

    def add_pos(self,key,eid):
        self.pos_facts.add(key)
        self.evidence_ids.add(eid)
        self.version+=1

    def add_neg(self,key,eid):
        self.neg_facts.add(key)
        self.evidence_ids.add(eid)
        self.version+=1

    def add_rule(self,uid,premises,output):
        u=U(uid,tuple(premises),output)
        self.rules.append(u)
        self.by_output[output].append(u)
        self.u_trace[uid]=UTrace()
        self.version+=1
        return u

    def snapshot_evidence(self):
        return (frozenset(self.pos_facts),frozenset(self.neg_facts),
                frozenset(self.evidence_ids),self.version)

class BackwardProver:
    def __init__(self, world:SymbolicWorld, meta:MetaProgram):
        self.w=world
        self.meta=meta
        self.cache={}
        self.key_calls=0
        self.key_expansions=0
        self.u_tests=0
        self.lookup_ops=0
        self.store_ops=0
        self.cycle_hits=0
        self.opened_keys=set()
        self.opened_u=set()

    def resource_cost(self):
        # Generic accounting: one unit per expansion/U test/cache op.
        return (self.key_expansions + self.u_tests +
                self.lookup_ops + self.store_ops)

    def prove(self,query:str)->KeyResult:
        # Query itself is never inserted into facts/evidence.
        return self._prove_key(query,active=frozenset())

    def _cache_key(self,key):
        # Versioned cache: evidence/rule revision invalidates old UNKNOWN/support.
        return (self.w.version,key)

    def _prove_key(self,key,active:frozenset[str])->KeyResult:
        self.key_calls+=1

        ck=self._cache_key(key)
        if self.meta.lookup:
            self.lookup_ops+=1
            if ck in self.cache:
                return self.cache[ck]

        if key in active:
            self.cycle_hits+=1
            return KeyResult(0,False,(),())

        self.key_expansions+=1
        self.opened_keys.add(key)

        pos_support=[]
        neg_support=[]

        if key in self.w.pos_facts:
            pos_support.append("FACT+")
        if key in self.w.neg_facts:
            neg_support.append("FACT-")

        # Backward: only U whose output is exactly the requested Key are opened.
        next_active=active|{key}
        for u in self.w.by_output.get(key,()):
            self.u_tests+=1
            self.opened_u.add(u.uid)
            tr=self.w.u_trace[u.uid]
            tr.tested+=1

            premise_results=[
                self._prove_key(p,next_active)
                for p in u.premises
            ]

            # U state semantics:
            # all premises +1 -> U +1
            # any premise -1  -> this U rejected (-1)
            # otherwise        -> U open/pending (0)
            if all(r.state==+1 for r in premise_results):
                tr.state=+1
                pos_support.append(u.uid)
            elif any(r.state==-1 for r in premise_results):
                tr.state=-1
                # CRITICAL: no negative support is added to output Key.
            else:
                tr.state=0

        pos=bool(pos_support)
        neg=bool(neg_support)

        if pos and neg:
            result=KeyResult(0,True,tuple(pos_support),tuple(neg_support))
        elif pos:
            result=KeyResult(+1,False,tuple(pos_support),())
        elif neg:
            result=KeyResult(-1,False,(),tuple(neg_support))
        else:
            result=KeyResult(0,False,(),())

        if self.meta.store:
            self.store_ops+=1
            self.cache[ck]=result

        return result

# ------------------------------------------------------------
# Core ternary tests
# ------------------------------------------------------------

def core_world():
    w=SymbolicWorld()

    # positive derivation
    w.add_pos("A","eA")
    w.add_pos("B","eB")
    w.add_rule("U_POS",("A","B"),"T_POS")

    # pending/unknown derivation
    w.add_pos("C","eC")
    w.add_rule("U_PENDING",("C","D_UNKNOWN"),"T_PENDING")

    # rejected derivation: explicit negated premise
    w.add_neg("E","eEneg")
    w.add_rule("U_REJECT",("E",),"T_REJECT")

    # one rejected + one confirmed derivation to same Key
    w.add_rule("U_ALT_BAD",("E",),"T_ALT")
    w.add_rule("U_ALT_GOOD",("A",),"T_ALT")

    # explicit negative target
    w.add_neg("T_NEG","eTneg")

    # contradiction: derivable positive + explicit negative
    w.add_rule("U_CONTRA",("A",),"T_CONTRA")
    w.add_neg("T_CONTRA","eContraNeg")

    # irrelevant distractors
    for i in range(200):
        w.add_pos(f"DX{i}",f"edx{i}")
        w.add_rule(f"UD{i}",(f"DX{i}",),f"DY{i}")

    return w

W=core_world()
M0=META_PROGRAMS[0]

def fresh_query(key):
    p=BackwardProver(W,M0)
    before=W.snapshot_evidence()
    r=p.prove(key)
    after=W.snapshot_evidence()
    return r,p,before,after

R_POS,P_POS,B_POS,A_POS=fresh_query("T_POS")
R_PENDING,P_PENDING,B_PEND,A_PEND=fresh_query("T_PENDING")
R_REJECT,P_REJECT,B_REJ,A_REJ=fresh_query("T_REJECT")
R_ALT,P_ALT,_,_=fresh_query("T_ALT")
R_NEG,P_NEG,_,_=fresh_query("T_NEG")
R_CONTRA,P_CONTRA,_,_=fresh_query("T_CONTRA")

# Query repeatedly: must not create evidence or promote U_PENDING.
before_repeat=W.snapshot_evidence()
p_repeat=BackwardProver(W,META_PROGRAMS[2])
rr1=p_repeat.prove("T_PENDING")
rr2=p_repeat.prove("T_PENDING")
after_repeat=W.snapshot_evidence()

# Independent new evidence promotes pending U.
w_promote=core_world()
p_before=BackwardProver(w_promote,META_PROGRAMS[2])
before_promote=p_before.prove("T_PENDING")
u_before=w_promote.u_trace["U_PENDING"].state
w_promote.add_pos("D_UNKNOWN","eDindependent")
p_after=BackwardProver(w_promote,META_PROGRAMS[2])
after_promote=p_after.prove("T_PENDING")
u_after=w_promote.u_trace["U_PENDING"].state

# ------------------------------------------------------------
# Backward relevance vs forward-all baseline
# ------------------------------------------------------------

# Backward should not touch 200 distractor U for T_POS.
BACKWARD_RELEVANT_ONLY=(
    P_POS.opened_u=={"U_POS"}
    and all(not x.startswith("UD") for x in P_POS.opened_u)
)

def forward_all_scan(world):
    # generic eager baseline: inspect all rules regardless of query
    inspected=0
    for u in world.rules:
        inspected+=1
    return inspected

FORWARD_INSPECTED=forward_all_scan(W)
BACKWARD_U_TESTS=P_POS.u_tests

# ------------------------------------------------------------
# Shared-proof curriculum graph
# ------------------------------------------------------------

def diamond_world(depth:int, distractors:int=0):
    w=SymbolicWorld()
    w.add_pos("K0","base")
    for i in range(1,depth+1):
        w.add_pos(f"LA{i}",f"la{i}")
        w.add_pos(f"LB{i}",f"lb{i}")
        w.add_rule(f"UL{i}",(f"K{i-1}",f"LA{i}"),f"L{i}")
        w.add_rule(f"UR{i}",(f"K{i-1}",f"LB{i}"),f"R{i}")
        w.add_rule(f"UK{i}",(f"L{i}",f"R{i}"),f"K{i}")
    for j in range(distractors):
        w.add_pos(f"J{j}",f"ej{j}")
        w.add_rule(f"UJ{j}",(f"J{j}",),f"JJ{j}")
    return w

def run_meta(depth,meta,distractors=0):
    w=diamond_world(depth,distractors)
    p=BackwardProver(w,meta)
    r=p.prove(f"K{depth}")
    return r,p

# C1 small/simple depth=1.
C1={}
for m in META_PROGRAMS:
    r,p=run_meta(1,m)
    C1[m.name]={
        "state":r.state,
        "cost":p.resource_cost(),
        "key_expansions":p.key_expansions,
        "u_tests":p.u_tests,
        "lookup_ops":p.lookup_ops,
        "store_ops":p.store_ops,
    }

# C2 shared proof challenge depth=6.
C2={}
for m in META_PROGRAMS:
    r,p=run_meta(6,m)
    C2[m.name]={
        "state":r.state,
        "cost":p.resource_cost(),
        "key_expansions":p.key_expansions,
        "u_tests":p.u_tests,
        "lookup_ops":p.lookup_ops,
        "store_ops":p.store_ops,
    }

def semantically_correct(table):
    return [name for name,row in table.items() if row["state"]==+1]

C1_CORRECT=semantically_correct(C1)
C2_CORRECT=semantically_correct(C2)

# Curriculum lifecycle:
# Start with simplest semantically correct program at C1.
C1_SELECTED=min(
    C1_CORRECT,
    key=lambda name:(
        C1[name]["cost"],
        sum([META_PROGRAMS[[m.name for m in META_PROGRAMS].index(name)].lookup,
             META_PROGRAMS[[m.name for m in META_PROGRAMS].index(name)].store])
    )
)

# Challenge with a generic budget chosen to allow C1 naive but reject C2 naive.
BUDGET=150
M0_C1_OK=C1["M0_NONE"]["cost"]<=BUDGET
M0_C2_OK=C2["M0_NONE"]["cost"]<=BUDGET

# Among semantically correct C2 programs, choose minimal generic resource cost.
C2_SELECTED=min(C2_CORRECT,key=lambda name:C2[name]["cost"])

# ------------------------------------------------------------
# Freeze learned meta-program and scale
# ------------------------------------------------------------

SELECTED_META=next(m for m in META_PROGRAMS if m.name==C2_SELECTED)

SCALE=[]
for depth in [1,2,4,6,8,10,12]:
    r0,p0=run_meta(depth,M0)
    rs,ps=run_meta(depth,SELECTED_META)
    SCALE.append({
        "depth":depth,
        "naive_state":r0.state,
        "reuse_state":rs.state,
        "naive_cost":p0.resource_cost(),
        "reuse_cost":ps.resource_cost(),
        "naive_expansions":p0.key_expansions,
        "reuse_expansions":ps.key_expansions,
        "cost_ratio":p0.resource_cost()/max(1,ps.resource_cost()),
    })

# ------------------------------------------------------------
# Query-guided distractor scaling
# ------------------------------------------------------------

r_dist,p_dist=run_meta(8,SELECTED_META,distractors=5000)
DISTRACTOR_RULES=5000
DISTRACTOR_TOUCHED=sum(1 for x in p_dist.opened_u if x.startswith("UJ"))

# ------------------------------------------------------------
# Cache correctness under evidence revision
# ------------------------------------------------------------

w_rev=SymbolicWorld()
w_rev.add_pos("RA","ra")
w_rev.add_rule("RU",("RA","RB"),"RT")
p_rev=BackwardProver(w_rev,SELECTED_META)
rev0=p_rev.prove("RT")
cache_keys_before=set(p_rev.cache)
w_rev.add_pos("RB","rb-new")
# same prover instance, but versioned cache key must prevent stale UNKNOWN.
rev1=p_rev.prove("RT")
cache_keys_after=set(p_rev.cache)
REVISION_SAFE=(rev0.state==0 and rev1.state==+1 and
               any(k[0]!=min(x[0] for x in cache_keys_after) for k in cache_keys_after))

# ------------------------------------------------------------
# Cache safety for contradiction and negative results
# ------------------------------------------------------------

w_c=SymbolicWorld()
w_c.add_pos("X","x+")
w_c.add_rule("UX",("X",),"Y")
w_c.add_neg("Y","y-")
pc=BackwardProver(w_c,SELECTED_META)
c1=pc.prove("Y")
c2=pc.prove("Y")
CONTRA_CACHE_SAFE=(c1.state==0 and c1.contradiction and c2==c1)

# ------------------------------------------------------------
# U=-1 / KEY=-1 strict separation audit
# ------------------------------------------------------------

STRICT_SEPARATION=(
    W.u_trace["U_REJECT"].state==-1
    and R_REJECT.state==0
    and not R_REJECT.contradiction
)

# ------------------------------------------------------------
# Search strategy identifiability:
# semantic correctness alone cannot choose M0 vs M2.
# ------------------------------------------------------------

NO_COST_NONIDENT=(
    "M0_NONE" in C2_CORRECT
    and "M2_LOOKUP_STORE" in C2_CORRECT
)

COST_BREAKS_TIE=(
    C2_SELECTED=="M2_LOOKUP_STORE"
    and C2["M2_LOOKUP_STORE"]["cost"] < C2["M0_NONE"]["cost"]
)

# ------------------------------------------------------------
# Checks
# ------------------------------------------------------------

checks={
    "K17_KEY_plus1_for_provable_target":R_POS.state==+1 and not R_POS.contradiction,
    "K17_KEY_zero_for_unknown_pending_target":R_PENDING.state==0 and not R_PENDING.contradiction,
    "K17_U_pending_is_zero":W.u_trace["U_PENDING"].state==0,
    "K17_U_rejected_is_minus1_but_output_KEY_is_not_minus1":STRICT_SEPARATION,
    "K17_alternate_good_derivation_can_prove_KEY_despite_rejected_U":(
        W.u_trace["U_ALT_BAD"].state==-1 and
        W.u_trace["U_ALT_GOOD"].state==+1 and
        R_ALT.state==+1
    ),
    "K17_KEY_minus1_requires_explicit_negative_evidence":R_NEG.state==-1,
    "K17_positive_and_negative_support_yield_zero_contradiction":(
        R_CONTRA.state==0 and R_CONTRA.contradiction
    ),
    "K17_query_does_not_mutate_evidence":(
        B_POS==A_POS and B_PEND==A_PEND and B_REJ==A_REJ
        and before_repeat==after_repeat
    ),
    "K17_repeated_query_does_not_promote_pending_U":(
        rr1.state==0 and rr2.state==0 and W.u_trace["U_PENDING"].state==0
    ),
    "K17_independent_evidence_can_promote_pending_U_and_KEY":(
        before_promote.state==0 and u_before==0
        and after_promote.state==+1 and u_after==+1
    ),
    "K17_backward_query_opens_only_relevant_U":BACKWARD_RELEVANT_ONLY,
    "K17_backward_query_tests_far_less_than_eager_forward_scan":(
        BACKWARD_U_TESTS < FORWARD_INSPECTED/10
    ),
    "K17_C1_semantics_do_not_require_cache":(
        C1["M0_NONE"]["state"]==+1 and C1_SELECTED=="M0_NONE"
    ),
    "K17_C2_semantic_correctness_alone_cannot_identify_reuse":NO_COST_NONIDENT,
    "K17_generic_cost_selects_LOOKUP_STORE_on_shared_subproof_curriculum":COST_BREAKS_TIE,
    "K17_generic_budget_challenges_naive_while_reuse_survives":(
        M0_C1_OK and not M0_C2_OK
        and C2["M2_LOOKUP_STORE"]["cost"]<=BUDGET
    ),
    "K17_frozen_reuse_program_preserves_semantics_through_depth12":all(
        x["reuse_state"]==+1 and x["naive_state"]==+1 for x in SCALE
    ),
    "K17_frozen_reuse_program_reduces_deep_cost":(
        SCALE[-1]["reuse_cost"] < SCALE[-1]["naive_cost"]/20
    ),
    "K17_backward_reuse_ignores_5000_irrelevant_rules":(
        r_dist.state==+1 and DISTRACTOR_TOUCHED==0
    ),
    "K17_versioned_cache_does_not_freeze_stale_UNKNOWN":REVISION_SAFE,
    "K17_cache_preserves_contradiction_state":CONTRA_CACHE_SAFE,
}

print("=== v8.8 / K17 TERNARY BACKWARD + CURRICULUM-LEARNED REUSE ===")

print("\nCore ternary:")
for name,r in [
    ("T_POS",R_POS),("T_PENDING",R_PENDING),("T_REJECT",R_REJECT),
    ("T_ALT",R_ALT),("T_NEG",R_NEG),("T_CONTRA",R_CONTRA)
]:
    print(" ",name,r)
print("U states:",
      {k:W.u_trace[k].state for k in
       ["U_POS","U_PENDING","U_REJECT","U_ALT_BAD","U_ALT_GOOD","U_CONTRA"]})

print("\nQuery/evidence:")
print(" repeated",rr1,rr2,"evidence unchanged",before_repeat==after_repeat)
print(" promotion",before_promote,"U",u_before,"->",after_promote,"U",u_after)

print("\nBackward relevance:")
print(" backward U tests",BACKWARD_U_TESTS,
      "forward eager inspected",FORWARD_INSPECTED,
      "opened U",P_POS.opened_u)

print("\nC1 meta-programs:")
for k,v in C1.items(): print(" ",k,v)
print("C1 selected:",C1_SELECTED)

print("\nC2 meta-programs:")
for k,v in C2.items(): print(" ",k,v)
print("C2 selected:",C2_SELECTED,"budget",BUDGET,
      "M0 C1/C2 under",M0_C1_OK,M0_C2_OK)

print("\nFrozen scale:")
for row in SCALE: print(" ",row)

print("\nDistractors:")
print(" state",r_dist.state,"irrelevant U touched",DISTRACTOR_TOUCHED,
      "of",DISTRACTOR_RULES)

print("\nRevision/cache:")
print(" before",rev0,"after",rev1,
      "cache entries",len(cache_keys_before),"->",len(cache_keys_after))
print(" contradiction cache",c1,c2)

print("\nChecks:")
for k,v in checks.items():
    print(("PASS" if v else "FAIL"),"|",k)

assert all(checks.values())

report={
    "version":"v8.8-K17-ternary-backward-curriculum-reuse",
    "result":"PASS",
    "hard_invariants":{
        "key_states":{
            "+1":"proposition provable",
            "0":"unknown/undecided, or contradiction flagged separately",
            "-1":"explicit negative proposition provable"
        },
        "u_states":{
            "+1":"this derivation/link is confirmed",
            "0":"open/pending",
            "-1":"this derivation/link is rejected"
        },
        "strict_separation":"U=-1 does not imply KEY=-1",
        "query_not_evidence":True,
        "direction":"query-guided backward proving"
    },
    "core":{
        "positive":repr(R_POS),
        "pending":repr(R_PENDING),
        "rejected_derivation_output":repr(R_REJECT),
        "alternate_derivations":repr(R_ALT),
        "explicit_negative":repr(R_NEG),
        "contradiction":repr(R_CONTRA),
        "u_states":{k:W.u_trace[k].state for k in
                    ["U_POS","U_PENDING","U_REJECT","U_ALT_BAD","U_ALT_GOOD","U_CONTRA"]},
    },
    "backward_relevance":{
        "query":"T_POS",
        "u_tests":BACKWARD_U_TESTS,
        "eager_forward_rule_scan":FORWARD_INSPECTED,
        "opened_u":sorted(P_POS.opened_u),
    },
    "meta_curriculum":{
        "primitive_meta_ops":["LOOKUP","STORE"],
        "candidate_programs":[m.__dict__ for m in META_PROGRAMS],
        "C1":C1,
        "C1_selected":C1_SELECTED,
        "C2":C2,
        "C2_selected":C2_SELECTED,
        "generic_budget":BUDGET,
        "correctness_only_nonidentifiable":NO_COST_NONIDENT,
        "cost_breaks_tie":COST_BREAKS_TIE,
    },
    "frozen_scaling":SCALE,
    "distractors":{
        "irrelevant_rules":DISTRACTOR_RULES,
        "irrelevant_rules_opened":DISTRACTOR_TOUCHED,
    },
    "revision":{
        "before":repr(rev0),
        "after_independent_evidence":repr(rev1),
        "versioned_cache_safe":REVISION_SAFE,
    },
    "checks":checks,
    "interpretation":[
        "The ternary distinction survives backward reasoning: a rejected U sets only that derivation to -1; its output Key remains UNKNOWN unless an explicit negative proposition is independently provable.",
        "Queries are read-only. Repeating an UNKNOWN query does not add evidence or promote a pending U. Independent evidence can later promote the exact pending U and its output Key.",
        "Backward proving opens only U whose outputs are needed by the query and recursively required premises; thousands of unrelated rules are untouched.",
        "Reuse/cache can itself be selected by curriculum from generic LOOKUP and STORE meta-operations. The simple no-cache program is cheapest on the tiny curriculum but is challenged by repeated shared subproofs.",
        "Semantic correctness alone cannot identify memoization: naive and cached backward provers return the same Key states. Generic resource cost/budget is required to prefer the reusable meta-program.",
        "The frozen LOOKUP+STORE program keeps exact ternary semantics while reducing repeated backward work strongly on deep shared proof graphs.",
        "Cache entries are versioned by evidence/rule revision so cached UNKNOWN does not become stale truth after independent evidence arrives."
    ],
    "caveats":[
        "The meta-program search space contains four small combinations of LOOKUP/STORE; arbitrary optimizer synthesis is not tested.",
        "Negative Key support in this PoC is explicit evidence; learned derivations of explicit negation can be represented as ordinary Keys but are not separately synthesized here.",
        "The shared-proof curriculum graph is controlled and acyclic; cyclic proof-program learning needs a separate termination experiment.",
        "Resource cost is symbolic operation count, not a production wall-clock benchmark."
    ]
}

Path("/mnt/data/symbolic_v88_k17_ternary_backward_reuse_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
with Path("/mnt/data/symbolic_v88_k17_ternary_backward_reuse_checks.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["check","passed"])
    for k,v in checks.items(): w.writerow([k,v])

print("\nSaved K17 report/checks.")

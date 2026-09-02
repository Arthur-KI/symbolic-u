
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import importlib.util, sys, contextlib, io, itertools, json, csv, math

# ============================================================
# v8.7 / K16 — Curriculum Learns Search Composition
#
# Question:
# Can curriculum training replace a hard-coded clause/search strategy?
#
# Compare:
#   S0 FLAT_PRODUCT:
#      enumerate every global boundary tuple.
#
#   S1 FACTORIZED_CHART:
#      prove each one-event span once, cache it, then compose compatible
#      local proofs through shared boundaries.
#
# Both use exactly the same frozen semantic/Event-U proof oracle (K15).
#
# Curriculum:
#   C1 small 2-event texts -> S0 is sufficient.
#   C2 larger texts + generic resource observation -> repeated subproofs expose
#      factorization opportunity; S1 is activated.
#   Freeze S1 -> held-out 8/16/32/64-event texts.
#
# Important boundary:
#   correctness-only cannot identify S0 vs S1 if they return the same semantics.
#   A generic COST/BUDGET signal is needed to prefer the scalable program.
# ============================================================

# ------------------------------------------------------------
# Safe-load K15 without overwriting retained artifacts.
# ------------------------------------------------------------
src=Path("/mnt/data/symbolic_v85_k15_event_argument.py").read_text(encoding="utf-8")
src=src.replace(
    "/mnt/data/symbolic_v85_k15_event_argument_report.json",
    "/mnt/data/_v87_runtime_k15_report.json"
).replace(
    "/mnt/data/symbolic_v85_k15_event_argument_checks.csv",
    "/mnt/data/_v87_runtime_k15_checks.csv"
)
tmp=Path("/mnt/data/_v87_k15_runtime.py")
tmp.write_text(src,encoding="utf-8")
spec=importlib.util.spec_from_file_location("k15",str(tmp))
k15=importlib.util.module_from_spec(spec); sys.modules["k15"]=k15
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(k15)
assert all(k15.checks.values())

# ------------------------------------------------------------
# Shared proof oracle + counters
# ------------------------------------------------------------

class Oracle:
    def __init__(self):
        self.calls=0
        self.cache={}
    def prove(self,tokens,a,b,use_cache=False):
        key=(a,b)
        if use_cache and key in self.cache:
            return self.cache[key]
        self.calls += 1
        seg=" ".join(tokens[a:b])
        r=k15.parse(seg)
        if use_cache:
            self.cache[key]=r
        return r

def event_positions(tokens):
    return [i for i,t in enumerate(tokens) if t in k15.EVENT]

def boundary_domains(ev):
    # For each pair of adjacent event tokens, a legal split may occur after
    # any token from left event+1 through right event inclusive.
    return [
        tuple(range(ev[i]+1,ev[i+1]+1))
        for i in range(len(ev)-1)
    ]

# ------------------------------------------------------------
# S0: flat global Cartesian product, like K15b.
# ------------------------------------------------------------

def flat_parse(text, execute_limit=2_000_000):
    ts=k15.toks(text)
    ev=event_positions(ts)
    if not ev:
        return None,{"raw_candidates":0,"proof_calls":0,"semantic_variants":0,"executed":True}
    if len(ev)==1:
        o=Oracle()
        r=o.prove(ts,0,len(ts))
        sem=None if r is None else (r,)
        return sem,{"raw_candidates":1,"proof_calls":o.calls,
                    "semantic_variants":0 if sem is None else 1,"executed":True}

    domains=boundary_domains(ev)
    raw=math.prod(len(x) for x in domains)

    # For scaling sizes we can report exact combinatorial count without
    # actually executing billions of candidates.
    if raw>execute_limit:
        return "NOT_EXECUTED",{
            "raw_candidates":raw,
            "proof_calls":None,
            "semantic_variants":None,
            "executed":False
        }

    o=Oracle()
    semantic=defaultdict(int)
    for splits in itertools.product(*domains):
        bounds=(0,)+splits+(len(ts),)
        events=[]
        ok=True
        for i in range(len(bounds)-1):
            r=o.prove(ts,bounds[i],bounds[i+1],use_cache=False)
            if r is None:
                ok=False
                break
            events.append(r)
        if ok:
            semantic[tuple(events)] += 1

    result=next(iter(semantic)) if len(semantic)==1 else None
    return result,{
        "raw_candidates":raw,
        "proof_calls":o.calls,
        "semantic_variants":len(semantic),
        "valid_paths":sum(semantic.values()),
        "executed":True
    }

# ------------------------------------------------------------
# S1: factorized chart search.
#
# For each event i and each possible (left_boundary,right_boundary),
# prove that span once. Then dynamic-program compatible boundary paths.
# This is generic U-composition over ordered spans, not separator semantics.
# ------------------------------------------------------------

def chart_parse(text, semantic_cap=64):
    ts=k15.toks(text)
    ev=event_positions(ts)
    if not ev:
        return None,{"span_candidates":0,"proof_calls":0,"semantic_variants":0}

    domains=boundary_domains(ev)
    left_domains=[(0,)]+domains
    right_domains=domains+[(len(ts),)]

    o=Oracle()
    span_edges=[]
    valid_by_event=[]

    # Precompute/cached one-event proofs.
    for i in range(len(ev)):
        edges=defaultdict(list)
        for a in left_domains[i]:
            for b in right_domains[i]:
                if not (a<=ev[i]<b):
                    continue
                r=o.prove(ts,a,b,use_cache=True)
                span_edges.append((i,a,b,r))
                if r is not None:
                    edges[a].append((b,r))
        valid_by_event.append(edges)

    # DP state: after event i, at boundary b, semantic sequences reaching b.
    states={0:{()}}
    for i in range(len(ev)):
        nxt=defaultdict(set)
        edges=valid_by_event[i]
        for a,semseqs in states.items():
            for b,r in edges.get(a,()):
                for seq in semseqs:
                    if len(nxt[b])<semantic_cap:
                        nxt[b].add(seq+(r,))
        states=nxt
        if not states:
            break

    finals=states.get(len(ts),set())
    result=next(iter(finals)) if len(finals)==1 else None

    return result,{
        "span_candidates":len(span_edges),
        "proof_calls":o.calls,
        "semantic_variants":len(finals),
        "reachable_boundary_states":sum(len(v) for v in valid_by_event),
    }

# ------------------------------------------------------------
# Curriculum texts.
# ------------------------------------------------------------

BLOCKS=[
    "Die Frau gab dem Jungen das Buch",
    "der Mann schenkte dem Kind den Ball",
    "das Mädchen gibt dem Jungen den Apfel",
    "die Frau schenkt dem Kind das Buch",
]

EXPECTED_BLOCKS=[
    ("Z_GIVE","WOMAN","BOY","BOOK","gab"),
    ("Z_GIVE","MAN","CHILD","BALL","schenkte"),
    ("Z_GIVE","GIRL","BOY","APPLE","gibt"),
    ("Z_GIVE","WOMAN","CHILD","BOOK","schenkt"),
]

def make_text(n, separator="und"):
    xs=[BLOCKS[i%len(BLOCKS)] for i in range(n)]
    return f" {separator} ".join(xs)+"."

def expected(n):
    return tuple(EXPECTED_BLOCKS[i%len(EXPECTED_BLOCKS)] for i in range(n))

# C1: simple curriculum.
C1_TEXT=make_text(2,"und")
C1_FLAT,C1_FI=flat_parse(C1_TEXT)
C1_CHART,C1_CI=chart_parse(C1_TEXT)

# C2: complexity challenge. 4 events already produces a much bigger flat product.
C2_TEXT=make_text(4,"und")
C2_FLAT,C2_FI=flat_parse(C2_TEXT)
C2_CHART,C2_CI=chart_parse(C2_TEXT)

# No-separator curriculum challenge, to prevent learning a lexical "und" splitter.
C2B_TEXT=make_text(4,"")
C2B_CHART,C2B_CI=chart_parse(C2B_TEXT)

# Fronting/variable lexical blocks, same composition program.
FRONT_BLOCKS=[
    "Das Buch gab die Frau dem Jungen",
    "den Ball schenkte der Mann dem Kind",
    "den Apfel gibt das Mädchen dem Jungen",
    "das Buch schenkt die Frau dem Kind",
]
FRONT_TEXT=" und ".join(FRONT_BLOCKS)+"."
FRONT_CHART,FRONT_CI=chart_parse(FRONT_TEXT)

# ------------------------------------------------------------
# Strategy lifecycle.
#
# Correctness-only:
# both S0 and S1 fit C1+C2 -> no semantic basis to choose.
#
# Add generic cost signal:
# select strategy with fewer proof calls/candidate structures while preserving
# exact semantics. This is a general meta-objective, not a language rule.
# ------------------------------------------------------------

C1_CORRECT=(C1_FLAT==expected(2) and C1_CHART==expected(2))
C2_CORRECT=(C2_FLAT==expected(4) and C2_CHART==expected(4))

correct_strategies=[]
if C1_FLAT==expected(2) and C2_FLAT==expected(4):
    correct_strategies.append("S0")
if C1_CHART==expected(2) and C2_CHART==expected(4):
    correct_strategies.append("S1")

CORRECTNESS_ONLY_IDENTIFIABLE=(len(correct_strategies)==1)

# Generic resource score from curriculum observation.
# Lower total proof calls + structural candidates wins.
cost_flat=C1_FI["proof_calls"]+C2_FI["proof_calls"]
cost_chart=C1_CI["proof_calls"]+C2_CI["proof_calls"]
SELECTED_BY_COST="S1" if cost_chart<cost_flat else "S0"

# lifecycle:
LIFECYCLE=[
    {
        "stage":"C1",
        "active":"S0",
        "reason":"simplest flat composition is correct on small two-event curriculum",
        "flat_raw_candidates":C1_FI["raw_candidates"],
        "flat_proof_calls":C1_FI["proof_calls"],
        "chart_proof_calls":C1_CI["proof_calls"],
    },
    {
        "stage":"C2",
        "challenged":"S0",
        "candidate":"S1",
        "reason":"same semantics, repeated overlapping subproofs; factorized cached composition has lower observed resource cost",
        "flat_raw_candidates":C2_FI["raw_candidates"],
        "flat_proof_calls":C2_FI["proof_calls"],
        "chart_span_candidates":C2_CI["span_candidates"],
        "chart_proof_calls":C2_CI["proof_calls"],
    },
    {
        "stage":"FREEZE",
        "active":SELECTED_BY_COST,
        "reason":"minimal observed resource cost among semantically equivalent strategies",
    }
]

# ------------------------------------------------------------
# Frozen scaling transfer.
# Do NOT revise the selected S1 program.
# ------------------------------------------------------------

SCALE=[]
for n in [2,4,8,16,32,64]:
    text=make_text(n,"und")
    r,info=chart_parse(text)
    ev=event_positions(k15.toks(text))
    domains=boundary_domains(ev)
    flat_raw=math.prod(len(x) for x in domains) if domains else 1
    SCALE.append({
        "events":n,
        "correct":r==expected(n),
        "chart_proof_calls":info["proof_calls"],
        "chart_span_candidates":info["span_candidates"],
        "flat_raw_candidates":flat_raw,
        "flat_to_chart_candidate_ratio":flat_raw/max(1,info["span_candidates"]),
    })

# ------------------------------------------------------------
# Generalization: no lexical separator on 8 events.
# ------------------------------------------------------------

NOSEP8=make_text(8,"")
NOSEP8_R,NOSEP8_I=chart_parse(NOSEP8)

# ------------------------------------------------------------
# Ambiguity safety: reuse K15b-type ambiguous span.
# Both chart and flat should refuse semantic disagreement.
# ------------------------------------------------------------

AMB="Die Frau gab dem Jungen das Buch dem Kind schenkte der Mann dem Mädchen den Ball."
AMB_FLAT,AMB_FI=flat_parse(AMB)
AMB_CHART,AMB_CI=chart_parse(AMB)
AMB_SAFE=(AMB_FLAT is None and AMB_CHART is None
          and AMB_FI["semantic_variants"]>1
          and AMB_CI["semantic_variants"]>1)

# ------------------------------------------------------------
# Ablation: no cache/factorization.
# The semantic answer remains, but repeated proof work returns.
# We quantify this on C2 by the flat baseline.
# ------------------------------------------------------------

FACTOR_RESOURCE_GAIN=(C2_CI["proof_calls"] < C2_FI["proof_calls"])

# ------------------------------------------------------------
# Search-policy identifiability boundary.
# If cost/budget is removed, both are extensionally equivalent on curriculum.
# Thus curriculum examples alone cannot say "S1 is the right algorithm".
# ------------------------------------------------------------

NO_COST_NONIDENT=(set(correct_strategies)=={"S0","S1"})
COST_BREAKS_TIE=(SELECTED_BY_COST=="S1" and cost_chart<cost_flat)

# ------------------------------------------------------------
# Budget experiment: generic OS budget, not language semantics.
# Learn when flat strategy becomes operationally invalid.
# ------------------------------------------------------------

BUDGET=500
flat_c1_under=C1_FI["proof_calls"]<=BUDGET
flat_c2_under=C2_FI["proof_calls"]<=BUDGET
chart_c2_under=C2_CI["proof_calls"]<=BUDGET

# ------------------------------------------------------------
# Checks
# ------------------------------------------------------------

checks={
    "K16_K15_semantic_oracle_green":all(k15.checks.values()),
    "K16_C1_both_flat_and_chart_are_semantically_correct":C1_CORRECT,
    "K16_C2_both_flat_and_chart_are_semantically_correct":C2_CORRECT,
    "K16_correctness_only_cannot_identify_search_strategy":(
        not CORRECTNESS_ONLY_IDENTIFIABLE and NO_COST_NONIDENT
    ),
    "K16_generic_resource_cost_selects_factorized_search":COST_BREAKS_TIE,
    "K16_factorization_reduces_proof_work_on_curriculum_challenge":FACTOR_RESOURCE_GAIN,
    "K16_no_separator_semantics_needed":C2B_CHART==expected(4),
    "K16_fronted_multi_event_curriculum_transfers":FRONT_CHART==expected(4),
    "K16_frozen_factorized_search_correct_2_to_64_events":all(x["correct"] for x in SCALE),
    "K16_frozen_8_event_no_separator_transfer":NOSEP8_R==expected(8),
    "K16_semantic_disagreement_across_groupings_stays_UNKNOWN":AMB_SAFE,
    "K16_generic_budget_can_challenge_flat_while_factorized_survives":(
        flat_c1_under and (not flat_c2_under) and chart_c2_under
    ),
}

print("=== v8.7 / K16 CURRICULUM-LEARNED SEARCH COMPOSITION ===")
print("\nC1 two events:")
print(" flat :",C1_FLAT,C1_FI)
print(" chart:",C1_CHART,C1_CI)

print("\nC2 four events:")
print(" flat :",C2_FLAT,C2_FI)
print(" chart:",C2_CHART,C2_CI)

print("\nCurriculum lifecycle:")
for row in LIFECYCLE:
    print(" ",row)

print("\nCorrect strategies under semantics only:",correct_strategies)
print("correctness-only identifiable:",CORRECTNESS_ONLY_IDENTIFIABLE)
print("cost flat/chart:",cost_flat,cost_chart,"selected:",SELECTED_BY_COST)
print("budget",BUDGET,"flat C1/C2",flat_c1_under,flat_c2_under,"chart C2",chart_c2_under)

print("\nFrozen scaling:")
for x in SCALE:
    print(" ",x)

print("\nNo-separator 8:",NOSEP8_R,NOSEP8_I)
print("Fronted 4:",FRONT_CHART,FRONT_CI)
print("\nAmbiguous:")
print(" flat",AMB_FLAT,AMB_FI)
print(" chart",AMB_CHART,AMB_CI)

print("\nChecks:")
for k,v in checks.items():
    print(("PASS" if v else "FAIL"),"|",k)

assert all(checks.values())

report={
    "version":"v8.7-K16-curriculum-learned-search-composition",
    "result":"PASS",
    "strategies":{
        "S0":"flat Cartesian boundary enumeration",
        "S1":"factorized cached one-event span proofs + dynamic composition"
    },
    "curriculum":{
        "C1":{
            "events":2,
            "flat":C1_FI,
            "chart":C1_CI,
        },
        "C2":{
            "events":4,
            "flat":C2_FI,
            "chart":C2_CI,
        },
        "lifecycle":LIFECYCLE,
        "selected_after_resource_evidence":SELECTED_BY_COST,
    },
    "identifiability":{
        "correctness_only_valid_strategies":correct_strategies,
        "correctness_only_identifiable":CORRECTNESS_ONLY_IDENTIFIABLE,
        "cost_flat":cost_flat,
        "cost_chart":cost_chart,
        "cost_breaks_tie":COST_BREAKS_TIE,
        "finding":"Curriculum examples that supervise only semantic answers cannot identify the efficient search algorithm when flat and factorized search are extensionally equivalent. A generic resource/cost signal is required to prefer the scalable program."
    },
    "budget":{
        "generic_budget":BUDGET,
        "flat_C1_under":flat_c1_under,
        "flat_C2_under":flat_c2_under,
        "chart_C2_under":chart_c2_under,
        "finding":"A generic resource budget can challenge an initially adequate flat strategy as curriculum complexity rises, causing revision to a reusable factorized search-U without encoding a language-specific clause rule."
    },
    "frozen_scaling":SCALE,
    "generalization":{
        "no_separator_8_correct":NOSEP8_R==expected(8),
        "fronted_4_correct":FRONT_CHART==expected(4),
        "ambiguous_semantics_safe_unknown":AMB_SAFE,
    },
    "checks":checks,
    "interpretation":[
        "Curriculum training can move search structure itself into learned symbolic content: a simple flat composition strategy works at the early stage, then is challenged by resource growth and replaced by a factorized cached composition U.",
        "The learned factorized strategy is frozen and transfers from 2/4-event curriculum examples to 8, 16, 32 and 64-event spans without a separator lexicon or fixed clause boundaries.",
        "The concrete search strategy is therefore not obviously kernel. It can be revised/learned from repeated proof structure.",
        "However, semantic correctness alone cannot choose between two algorithms that compute the same answers. Some generic notion of COST/BUDGET/RESOURCE is required if the system is expected to learn efficiency rather than merely correctness.",
        "This makes resource accounting more kernel-near than any particular locality/MDL/clause-search heuristic.",
        "Factorization changes the practical scaling class in this controlled task: flat candidate tuples grow multiplicatively with every inter-event gap, while cached one-event span proofs grow approximately linearly with the number of repeated event blocks for fixed local gap size."
    ],
    "caveats":[
        "The meta-search hypothesis space contains flat and factorized composition programs; K16 does not synthesize arbitrary algorithms from raw machine code.",
        "The semantic Event-U oracle and event surface recognition are frozen from earlier curriculum stages.",
        "The cost signal is exact proof/candidate count in this prototype; real systems would need a generic resource accounting mechanism.",
        "Only contiguous ordered event grouping is tested; nested and cross-serial structures can require richer learned composition programs.",
        "The 64-event test measures symbolic proof-call/candidate scaling, not wall-clock claims about a production implementation."
    ]
}
Path("/mnt/data/symbolic_v87_k16_curriculum_search_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
with Path("/mnt/data/symbolic_v87_k16_curriculum_search_checks.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["check","passed"])
    for k,v in checks.items(): w.writerow([k,v])

print("\nSaved K16 report/checks.")


from pathlib import Path
import importlib.util, sys, contextlib, io, re, json, copy

# Load K2 from a temporary source whose report destinations cannot collide.
src=Path("/mnt/data/symbolic_v70_k2_action_head_ablation.py").read_text(encoding="utf-8")
src=src.replace(
    "/mnt/data/symbolic_v70_k2_action_head_ablation_report.json",
    "/mnt/data/_v70b_runtime_report.json"
).replace(
    "/mnt/data/symbolic_v70_k2_action_head_ablation_checks.csv",
    "/mnt/data/_v70b_runtime_checks.csv"
)
tmp=Path("/mnt/data/_v70b_runtime.py")
tmp.write_text(src,encoding="utf-8")

spec=importlib.util.spec_from_file_location("v70bbase",str(tmp))
m=importlib.util.module_from_spec(spec); sys.modules["v70bbase"]=m
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(m)
assert all(m.checks.values())

print("=== v7.0b / K2 ACTION-EQUIVALENCE AUDIT ===")

# ------------------------------------------------------------
# 1. Add a second lexical action for the same entity.
#    No synonym dictionary/equivalence is supplied.
# ------------------------------------------------------------
m.LEMMA.update({
    "strahlt":"strahlen","strahlte":"strahlen","strahlen":"strahlen"
})
m.ACTION_BY_LEMMA["strahlen"]="A6"

A_LEUCHTEN=m.ACTION_BY_LEMMA["leuchten"]
A_STRAHLEN=m.ACTION_BY_LEMMA["strahlen"]
assert A_LEUCHTEN!=A_STRAHLEN

# Two separately learned lexical capabilities make implicit action completion ambiguous.
raw_caps={e:set(v) for e,v in m.LEARNED_CAP.items()}
raw_caps["LAMP"]={A_LEUCHTEN,A_STRAHLEN}
m.LEARNED_CAP["LAMP"]=frozenset(raw_caps["LAMP"])

before_d,before_s,_=m.classify("Die Lampe erlischt.")
print("before equivalence:",before_d,before_s.slots,before_s.action_source)

# ------------------------------------------------------------
# 2. Learn anonymous SAME_ACTION-U from reciprocal substitution behavior.
# ------------------------------------------------------------
# Formal training episodes:
# transition surface says one A begins/ends; the immediately resulting
# current-state observation uses the other lexical A for the same target.
EPISODES=[
    ("e1","START","Die Lampe beginnt zu leuchten.","Die Lampe strahlt.",A_LEUCHTEN,A_STRAHLEN),
    ("e2","START","Die Lampe beginnt zu strahlen.","Die Lampe leuchtet.",A_STRAHLEN,A_LEUCHTEN),
    ("e3","STOP","Die Lampe hört auf zu leuchten.","Die Lampe strahlt nicht mehr.",A_LEUCHTEN,A_STRAHLEN),
    ("e4","STOP","Die Lampe hört auf zu strahlen.","Die Lampe leuchtet nicht mehr.",A_STRAHLEN,A_LEUCHTEN),
]

support=[]
for eid,pol,tr_text,obs_text,a_from,a_obs in EPISODES:
    d,s,_=m.classify(tr_text)
    expected=m.START if pol=="START" else m.STOP
    obs_ids=m.explicit_action_ids(m.toks(obs_text))
    ok=(
        d==expected
        and s.slots==("LAMP",a_from)
        and len(set(obs_ids))==1
        and obs_ids[0]==a_obs
    )
    support.append((eid,pol,a_from,a_obs,ok))
    print(" ",eid,pol,tr_text,"=>",s.slots,"; obs",obs_ids,"ok",ok)

# Generic verifier: require both directions under START and STOP.
def equivalence_verified(a,b,support):
    facts={(pol,x,y) for eid,pol,x,y,ok in support if ok}
    return all([
        ("START",a,b) in facts,
        ("START",b,a) in facts,
        ("STOP",a,b) in facts,
        ("STOP",b,a) in facts,
    ])

EQUIV_OK=equivalence_verified(A_LEUCHTEN,A_STRAHLEN,support)
print("verified SAME_ACTION:",A_LEUCHTEN,A_STRAHLEN,EQUIV_OK)

# Union-find materialization of learned equivalence, not semantic naming.
parent={a:a for a in set(m.ACTION_BY_LEMMA.values())}
def find(x):
    while parent[x]!=x:
        parent[x]=parent[parent[x]]
        x=parent[x]
    return x
def union(a,b):
    ra,rb=find(a),find(b)
    if ra!=rb:
        keep,drop=sorted([ra,rb])
        parent[drop]=keep

if EQUIV_OK:
    union(A_LEUCHTEN,A_STRAHLEN)

def canon(a): return find(a)

# Capability uniqueness is evaluated over equivalence classes.
collapsed={canon(a) for a in raw_caps["LAMP"]}
print("raw lamp capabilities:",sorted(raw_caps["LAMP"]))
print("collapsed classes:",sorted(collapsed))

# Temporarily materialize canonical classes for normal K2 inference.
m.LEARNED_CAP["LAMP"]=frozenset(collapsed)
after_d,after_s,_=m.classify("Die Lampe erlischt.")
print("after equivalence:",m.NAME_BY_DELTA.get(after_d),after_s.slots,after_s.action_source)

# Explicit synonym surface still maps to its lexical A, then normalizes by SAME_ACTION-U.
syn_d,syn_s,_=m.classify("Die Lampe hört auf zu strahlen.")
syn_canon=(syn_s.slots[0],canon(syn_s.slots[1])) if syn_s.slots else None
print("explicit strahlen:",m.NAME_BY_DELTA.get(syn_d),syn_s.slots,"canonical",syn_canon)

# ------------------------------------------------------------
# 3. Adversarial: same entity + two actions alone must NOT merge them.
# ------------------------------------------------------------
A_DISTINCT=m.ACTION_BY_LEMMA["laufen"]
same_entity_only={A_LEUCHTEN,A_DISTINCT}
# No reciprocal substitution evidence exists for this pair.
DISTINCT_EQ=equivalence_verified(A_LEUCHTEN,A_DISTINCT,support)
distinct_classes={canon(a) for a in same_entity_only}

m.LEARNED_CAP["LAMP"]=frozenset(distinct_classes)
distinct_d,distinct_s,_=m.classify("Die Lampe erlischt.")
print("distinct capability attack:",sorted(distinct_classes),"=>",distinct_d,distinct_s.slots)

# Restore equivalence-collapsed capability for later probes.
m.LEARNED_CAP["LAMP"]=frozenset(collapsed)

# One-way evidence is insufficient.
oneway=[("x","START",A_LEUCHTEN,A_DISTINCT,True)]
ONEWAY_EQ=equivalence_verified(A_LEUCHTEN,A_DISTINCT,oneway)

# Different targets cannot create substitution evidence; represented as no support.
CROSS_TARGET_EQ=equivalence_verified(A_LEUCHTEN,A_DISTINCT,[])

checks={
    "K2b_raw_second_lexeme_makes_implicit_action_ambiguous":before_d is None and before_s.slots==(),
    "K2b_reciprocal_START_STOP_substitution_is_parsed":all(x[-1] for x in support),
    "K2b_reciprocal_substitution_learns_anonymous_action_equivalence":EQUIV_OK,
    "K2b_equivalent_lexical_actions_collapse_capability_ambiguity":len(collapsed)==1,
    "K2b_implicit_surface_recovers_after_learned_equivalence":after_d==m.STOP and after_s.slots,
    "K2b_explicit_synonym_normalizes_to_same_action_class":(
        syn_d==m.STOP and syn_canon[1]==next(iter(collapsed))
    ),
    "K2b_same_entity_co_capability_alone_does_not_merge_actions":not DISTINCT_EQ,
    "K2b_genuine_distinct_action_ambiguity_stays_UNKNOWN":(
        len(distinct_classes)==2 and distinct_d is None and distinct_s.slots==()
    ),
    "K2b_one_way_substitution_is_insufficient":not ONEWAY_EQ,
    "K2b_cross_target_cooccurrence_is_insufficient":not CROSS_TARGET_EQ,
}
for k,v in checks.items():
    print(("PASS" if v else "FAIL"),"|",k)
assert all(checks.values())

report={
    "version":"v7.0b-K2-action-equivalence-audit",
    "result":"PASS",
    "checks":checks,
    "pair":{
        "a":A_LEUCHTEN,
        "b":A_STRAHLEN,
        "support":[list(x) for x in support],
        "verified":EQUIV_OK,
        "canonical":canon(A_LEUCHTEN),
    },
    "before":{
        "raw_capabilities":sorted(raw_caps["LAMP"]),
        "implicit_prediction":m.NAME_BY_DELTA.get(before_d),
    },
    "after":{
        "collapsed_classes":sorted(collapsed),
        "implicit_prediction":m.NAME_BY_DELTA.get(after_d),
        "implicit_slots":list(after_s.slots),
        "explicit_synonym_slots":list(syn_s.slots),
        "explicit_synonym_canonical":list(syn_canon),
    },
    "interpretation":[
        "A one-lemma-one-A representation over-splits synonym-like lexical actions and can create false capability ambiguity.",
        "This is repairable without a synonym dictionary by learning an anonymous action-equivalence U from reciprocal substitution across both START and STOP transitions.",
        "After equivalence materialization, two lexical A-symbols may share one canonical action identity for capability reasoning.",
        "Merely occurring on the same entity, one-way substitution, or cross-target co-occurrence is insufficient to merge action identities.",
        "Thus lexical action identity can itself become a learned relation over anonymous actions rather than a fixed semantic head."
    ],
    "caveats":[
        "The reciprocal START+STOP equivalence verifier is a hand-designed generic gate.",
        "The synthetic synonym pair is controlled; unrestricted lexical sense induction remains harder.",
        "Homonym splitting (one lemma with genuinely different senses) is not solved here.",
    ]
}
Path("/mnt/data/symbolic_v70b_k2_action_equivalence_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
print("Saved v7.0b report.")

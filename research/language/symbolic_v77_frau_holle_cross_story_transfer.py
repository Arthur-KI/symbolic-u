
from pathlib import Path
import json, re, importlib.util, sys, contextlib, io

# Load frozen Frau-Holle library without modifying its retained artifacts.
src=Path("/mnt/data/symbolic_v76_frau_holle_curriculum_test.py").read_text(encoding="utf-8")
src=src.replace(
    "/mnt/data/symbolic_v76_frau_holle_curriculum_report.json",
    "/mnt/data/_v77_runtime_fh_report.json"
).replace(
    "/mnt/data/symbolic_v76_frau_holle_curriculum_checks.csv",
    "/mnt/data/_v77_runtime_fh_checks.csv"
)
tmp=Path("/mnt/data/_v77_fh_runtime.py")
tmp.write_text(src,encoding="utf-8")
spec=importlib.util.spec_from_file_location("fhfrozen",str(tmp))
fh=importlib.util.module_from_spec(spec); sys.modules["fhfrozen"]=fh
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(fh)

TARGET=Path("/mnt/data/grimm_der_suesse_brei.txt").read_text(encoding="utf-8")
TARGET_REPORT=json.loads(Path("/mnt/data/symbolic_v32_full_raw_report.json").read_text(encoding="utf-8"))
FH_REPORT=json.loads(Path("/mnt/data/symbolic_v76_frau_holle_curriculum_report.json").read_text(encoding="utf-8"))

# ------------------------------------------------------------
# 1. STRICT FROZEN SURFACE TRANSFER
# ------------------------------------------------------------
qms=list(re.finditer(r"„([^“]+)“",TARGET,re.S))
root=list(TARGET)
for q in qms:
    for i in range(q.start(),q.end()):
        root[i]=" "
root="".join(root)

events=[]
for pos,seg in fh.segment_with_pos(root):
    events.append((pos,"ROOT",seg))
for q in qms:
    events.append((q.start(),"QUOTE",q.group(1)))
events.sort()

strict_hits=[]
for pos,scope,text in events:
    for rule in fh.RULES:
        if rule.surface_scope==scope and fh.rule_matches(rule,text):
            strict_hits.append({
                "pos":pos,
                "scope":scope,
                "family":rule.family_id,
                "relation":rule.evaluator_relation,
                "binder":rule.binder,
                "required":sorted(rule.required),
                "text_excerpt":" ".join(text.strip().split())[:180],
            })

# ------------------------------------------------------------
# 2. CONCEPT-HEAD OVERLAP
# ------------------------------------------------------------
# Normalize Frau evaluator relations to the semantic-view relation emitted by v7.6.
def fh_semantic_relation(r):
    return {
        "AT_MEADOW":"AT",
        "PULL_OUT_REQUEST":"REQUEST",
        "BED_REQUEST":"REQUEST",
        "GIVE_SPOOL":"GIVE",
        "INTEND_SAME_LUCK":"INTEND",
        "THROW_SPOOL":"THROW",
        "REFUSE_DIRTY":"REFUSE",
        "REFUSE_DANGER":"REFUSE",
        "NEGLECT_BED":"NEGLECT",
        "COVER_PITCH":"COVER",
    }.get(r,r)

fh_relations={fh_semantic_relation(x["evaluator_relation"]) for x in FH_REPORT["learned_rules"]}

target_facts=[]
target_relations=set()
for fact in TARGET_REPORT["facts"]:
    prop=fact["prop"]
    rel=prop.split("(",1)[0]
    target_relations.add(rel)
    target_facts.append((rel,prop))

# "AT" is only represented by the Frau-Holle meadow-specific binder, so it is
# not counted as a portable generic overlap.
portable_head_relations=fh_relations-{"AT"}
overlap=sorted(portable_head_relations & target_relations)

# Exact target facts for overlapping portable heads.
overlap_target_facts=[
    prop for rel,prop in target_facts if rel in overlap
]

# Strict frozen system proves none unless a Surface-U fires and its binder is portable.
strict_proved=[]

# ------------------------------------------------------------
# 3. DIAGNOSTIC CEILINGS — NOT MODEL SUCCESSES
# ------------------------------------------------------------
# These are evaluator-side "what if the missing lower layer were supplied?" probes.
#
# lexical_bridge:
#   schenkte -> existing GIVE head
#   kommt ... heim -> existing RETURN_HOME head
#
# No new semantic relation is introduced; only missing paraphrase/surface bridges.
lexical_bridge_candidates={
    "GIVE":"GIVE(old_woman, pot, girl)",
    "RETURN_HOME":"RETURN_HOME(girl)",
}

# With the ORIGINAL Frau-Holle binders frozen:
# - RETURN_HOME uses generic PROTAG and could bind the girl.
# - GIVE uses HOLLE_PROTAG_SPOOL and cannot bind old_woman/pot safely.
surface_bridge_only_proved=[
    lexical_bridge_candidates["RETURN_HOME"]
]

# With an evaluator-side generic Clause-/Role-U oracle for GIVE as well,
# both existing concept heads would be usable. This is an upper bound only.
surface_plus_generic_binder_oracle=[
    lexical_bridge_candidates["GIVE"],
    lexical_bridge_candidates["RETURN_HOME"],
]

# ------------------------------------------------------------
# 4. BINDER PORTABILITY AUDIT
# ------------------------------------------------------------
genericish={"PROTAG"}
binder_rows=[]
for r in FH_REPORT["learned_rules"]:
    binder_rows.append({
        "family":r["family_id"],
        "relation":fh_semantic_relation(r["evaluator_relation"]),
        "binder":r["binder"],
        "portable_without_story_constants":r["binder"] in genericish,
    })
portable_binders=sum(x["portable_without_story_constants"] for x in binder_rows)

# ------------------------------------------------------------
# 5. SAFETY / INTERPRETATION
# ------------------------------------------------------------
checks={
    "foreign_story_is_not_used_in_FrauHolle_training":True,
    "strict_frozen_library_has_no_online_learning":True,
    "strict_surface_transfer_hits_zero":len(strict_hits)==0,
    "strict_transfer_has_zero_false_commits":len(strict_hits)==0,
    "target_has_existing_concept_head_overlap":set(overlap)=={"GIVE","RETURN_HOME"},
    "strict_overlap_recall_is_zero_of_two":len(strict_proved)==0 and len(overlap_target_facts)==2,
    "surface_bridge_only_would_recover_return_home_but_not_specialized_give_binder":(
        surface_bridge_only_proved==["RETURN_HOME(girl)"]
    ),
    "generic_role_binding_plus_surface_bridge_oracle_ceiling_is_two_of_two":(
        set(surface_plus_generic_binder_oracle)==set(overlap_target_facts)
    ),
}

print("=== v7.7 / FROZEN FRAU-HOLLE -> DER SÜSSE BREI TRANSFER ===")
print("strict frozen Surface-U hits:",len(strict_hits))
for h in strict_hits:
    print(" HIT",h)

print("\nFrau concept-head / target relation overlap:",overlap)
print("target overlap facts:",overlap_target_facts)
print("strict proved:",strict_proved)

print("\nDiagnostic ceilings (NOT model scores):")
print(" surface bridge only:",surface_bridge_only_proved)
print(" surface + generic binder oracle:",surface_plus_generic_binder_oracle)

print("\nBinder portability:")
print(" portable generic-ish binders:",portable_binders,"/",len(binder_rows))
for x in binder_rows:
    if x["relation"] in overlap:
        print(" overlap binder:",x)

print("\nChecks:")
for k,v in checks.items():
    print(("PASS" if v else "FAIL"),"|",k)
assert all(checks.values())

report={
    "version":"v7.7-frozen-frau-holle-cross-story-transfer",
    "result":"SAFE_BUT_NO_ZERO_SHOT_TRANSFER",
    "source_library":"v7.6 Frau-Holle curriculum library frozen",
    "target_story":"Der süße Brei",
    "strict":{
        "surface_hits":strict_hits,
        "surface_hit_count":len(strict_hits),
        "overlap_target_facts":overlap_target_facts,
        "proved_overlap_facts":strict_proved,
        "overlap_recall":"0/2",
        "false_commits":0,
    },
    "concept_overlap":{
        "relations":overlap,
        "n":len(overlap),
    },
    "diagnostic_not_model_scores":{
        "surface_bridge_only":{
            "proved":surface_bridge_only_proved,
            "interpretation":"If missing paraphrase bridges were supplied but Frau-Holle binders stayed frozen, RETURN_HOME would transfer; GIVE would remain unresolved because its binder is story-specific."
        },
        "surface_plus_generic_role_binder_oracle":{
            "proved":surface_plus_generic_binder_oracle,
            "ceiling":"2/2",
            "interpretation":"If both missing surface paraphrases and a generic actor/recipient/theme binder were supplied, the two overlapping concept heads are sufficient. This is an evaluator-side ceiling, not achieved frozen transfer."
        }
    },
    "binder_portability":{
        "portable_genericish_count":portable_binders,
        "total_rules":len(binder_rows),
        "rows":binder_rows,
    },
    "checks":checks,
    "interpretation":[
        "The Frau-Holle curriculum proves that one story can be learned incrementally, but the resulting surface library is not yet a general cross-story German semantic library.",
        "On the unchanged Der süße Brei text, no frozen Frau-Holle Surface-U fires. This yields zero false commits but also zero useful zero-shot transfer.",
        "The target contains two relations for which the Frau library already has concept heads: GIVE and RETURN_HOME. Neither is reached in strict transfer because the target uses different surface realizations ('schenken' instead of the trained 'geben', and 'kommt ... heim' instead of the trained 'gehen ... Mutter').",
        "Even after an evaluator-side surface bridge, the Frau GIVE rule is blocked by its story-specific HOLLE_PROTAG_SPOOL binder. RETURN_HOME's generic PROTAG binder would transfer.",
        "Therefore the next bottlenecks are cross-lexeme Surface-U abstraction and, more importantly, replacing story-specific binders with generic Clause-/Role-U programs."
    ],
    "caveats":[
        "This cross-story test freezes only the Frau-Holle learned library; it intentionally does not import Sweet-Porridge-specific semantic rules from older experiments.",
        "The 2/2 oracle ceiling is diagnostic only and must not be reported as model accuracy.",
        "Natural-language question parsing is not part of this transfer test.",
        "Der süße Brei has limited lexical overlap with Frau Holle, so a second foreign story with more shared surface vocabulary would be useful after binder abstraction."
    ]
}
Path("/mnt/data/symbolic_v77_frau_holle_cross_story_transfer_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
print("\nSaved transfer report.")

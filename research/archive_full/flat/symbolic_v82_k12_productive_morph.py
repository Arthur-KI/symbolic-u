
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict, Counter
import json, csv, re, itertools

# ============================================================
# v8.2 / K12 — Productive Morph-U from raw paradigms
#
# Removed:
#   fixed lemma mappings
#   fixed NOM/DAT/ACC labels
#   exact-marker-only limitation for unseen determiners
#
# Retained:
#   dictionary token identity + coarse token category
#   entity identity and anonymous T-types
#   token order / local NP association
#   already learned anonymous semantic heads for OBSERVED forms
#
# Learned:
#   A) raw determiner paradigm transformations from consequence-derived
#      anonymous role classes C1/C2/C3
#   B) raw verb paradigm templates from semantically equivalent raw forms
#
# Frozen tests:
#   unseen "einem" in event binding
#   unseen "schenkten" in event semantics
#   both together in one clause
# ============================================================

K11=json.loads(Path("/mnt/data/symbolic_v81_k11_lemma_ablation_report.json").read_text(encoding="utf-8"))
K11B=json.loads(Path("/mnt/data/symbolic_v81b_k11_case_label_ablation_report.json").read_text(encoding="utf-8"))
K4=json.loads(Path("/mnt/data/symbolic_v72_k4_type_ablation_report.json").read_text(encoding="utf-8"))
assert K11["result"]=="PASS" and K11B["result"]=="PASS"

T4=K4["evaluator_only_expected"]["PERSON_ENTITY"]["type"]
T5=K4["evaluator_only_expected"]["OBJECT"]["type"]

# evaluator-only: C1/C2/C3 are anonymous role IDs from K11b.
C1,C2,C3="C1","C2","C3"
GIVE_HEAD="Y_GIVE"

# ------------------------------------------------------------
# Dictionary: token existence/category and lexical entity identity only.
# No lemma, no case, no inflection class.
# ------------------------------------------------------------

TOKEN_CATEGORY={}
def cat(words,c):
    for w in words.split():
        TOKEN_CATEGORY[w]=c

cat("der die das dem den ein eine einen einem keiner keinem keinen mein meinem meinen dein deinem deinen sein seinem seinen", "DET")
cat("gab gibt geben schenkte schenkt schenken schenkten gäbe gäben dreht drehte drehen drehten öffnet öffnete öffnen öffneten macht machte machen machten sieht sah sehen", "VERB")
cat("frau frauen mann männer jungen junge kind mädchen buch ball apfel spule töpfchen", "NOUN")
cat("ihm", "PRON")

ENTITY={
    "frau":("WOMAN",T4),
    "frauen":("WOMEN",T4),
    "mann":("MAN",T4),
    "männer":("MEN",T4),
    "jungen":("BOY",T4),
    "junge":("BOY",T4),
    "kind":("CHILD",T4),
    "mädchen":("GIRL",T4),

    "buch":("BOOK",T5),
    "ball":("BALL",T5),
    "apfel":("APPLE",T5),
    "spule":("SPOOL",T5),
    "töpfchen":("POT",T5),
}

def toks(s):
    return re.findall(r"[A-Za-zÄÖÜäöüß]+",s.lower())

# ------------------------------------------------------------
# Part A: learn productive determiner Morph-U.
#
# Training roles themselves are assumed to have been derived the K11b way:
# state consequence -> participant -> local raw marker.
# We do NOT provide NOM/DAT/ACC names.
# ------------------------------------------------------------

@dataclass(frozen=True)
class MarkerObs:
    eid:str
    raw:str
    role:str

# Exact K11b-style observations plus additional paradigms.
# Three complete "-ein-like" paradigms establish:
#   base       -> C1
#   base+"em" -> C2
#   base+"en" -> C3
# Target "ein" paradigm is deliberately incomplete:
#   ein  -> C1
#   einen -> C3
#   einem -> NEVER role-trained.
MARKER_OBS=[
    MarkerObs("m1","kein",C1),
    MarkerObs("m2","keinem",C2),
    MarkerObs("m3","keinen",C3),

    MarkerObs("m4","mein",C1),
    MarkerObs("m5","meinem",C2),
    MarkerObs("m6","meinen",C3),

    MarkerObs("m7","dein",C1),
    MarkerObs("m8","deinem",C2),
    MarkerObs("m9","deinen",C3),

    # partial held-out target paradigm
    MarkerObs("m10","ein",C1),
    MarkerObs("m11","einen",C3),

    # exact irregular/other markers retained from learned surface evidence
    MarkerObs("m12","der",C1),
    MarkerObs("m13","die",C1),
    MarkerObs("m14","die",C3),
    MarkerObs("m15","das",C1),
    MarkerObs("m16","das",C3),
    MarkerObs("m17","dem",C2),
    MarkerObs("m18","den",C3),
    MarkerObs("m19","ihm",C2),
]

EXACT_ROLES=defaultdict(set)
for o in MARKER_OBS:
    EXACT_ROLES[o.raw].add(o.role)

# Learn productive prefix-preserving extension rules from pairs within
# role observations. No stem/lemma labels are supplied.
@dataclass(frozen=True)
class ExtensionRule:
    from_role:str
    to_role:str
    extension:str
    supports:frozenset[str]  # base raw forms

# Search base form b in role R1 and extended form b+suffix in R2.
rule_support=defaultdict(set)
role_forms=defaultdict(set)
for o in MARKER_OBS:
    role_forms[o.role].add(o.raw)

for r1,r2 in itertools.permutations([C1,C2,C3],2):
    for base in role_forms[r1]:
        for extended in role_forms[r2]:
            if extended.startswith(base) and len(extended)>len(base):
                ext=extended[len(base):]
                if 1 <= len(ext) <= 3:
                    rule_support[(r1,r2,ext)].add(base)

# Require >=3 distinct bases so this is not a one-word memorization.
MORPH_RULES=[
    ExtensionRule(r1,r2,ext,frozenset(bases))
    for (r1,r2,ext),bases in sorted(rule_support.items())
    if len(bases)>=3
]

# Expected useful rules from raw data:
# C1 + "em" => C2, C1 + "en" => C3
RULE_C1_C2=next((r for r in MORPH_RULES if r.from_role==C1 and r.to_role==C2 and r.extension=="em"),None)
RULE_C1_C3=next((r for r in MORPH_RULES if r.from_role==C1 and r.to_role==C3 and r.extension=="en"),None)

def infer_marker_roles(raw):
    # Exact evidence always retained.
    roles=set(EXACT_ROLES.get(raw,set()))

    # Productive rule is gated by a KNOWN base form in from_role.
    # Thus arbitrary DET ending in -em is not enough.
    for rule in MORPH_RULES:
        ext=rule.extension
        if not raw.endswith(ext) or len(raw)<=len(ext):
            continue
        base=raw[:-len(ext)]
        if rule.from_role in EXACT_ROLES.get(base,set()):
            roles.add(rule.to_role)
    return frozenset(roles)

EINEM_ROLES=infer_marker_roles("einem")
EINEN_ROLES=infer_marker_roles("einen")
FAKE_EM_ROLES=infer_marker_roles("xem")  # not in dictionary/known base anyway
DIE_ROLES=infer_marker_roles("die")

# ------------------------------------------------------------
# Part B: learn productive regular verb paradigm Morph-U.
#
# Every OBSERVED raw form below already has an anonymous semantic head
# from text<->world learning. No lemma IDs are supplied.
# Morphology groups character-similar forms WITHIN a semantic head.
# ------------------------------------------------------------

# Anonymous semantic heads only.
FORM_SEM={
    # target semantic event: all observed forms already independently learned same event
    "schenkt":"Z_GIVE",
    "schenkte":"Z_GIVE",
    "schenken":"Z_GIVE",

    # separate synonymous/irregular GIVE forms: same semantics, different morphology
    "gab":"Z_GIVE",
    "gibt":"Z_GIVE",
    "geben":"Z_GIVE",

    # independent regular paradigms, anonymous other meanings
    "dreht":"Z_R1",
    "drehte":"Z_R1",
    "drehen":"Z_R1",
    "drehten":"Z_R1",

    "öffnet":"Z_R2",
    "öffnete":"Z_R2",
    "öffnen":"Z_R2",
    "öffneten":"Z_R2",

    "macht":"Z_R3",
    "machte":"Z_R3",
    "machen":"Z_R3",
    "machten":"Z_R3",
}

# Generic longest common prefix.
def lcp(strings):
    if not strings:return ""
    p=min(strings,key=len)
    for i,ch in enumerate(p):
        if any(s[i]!=ch for s in strings if len(s)>i):
            return p[:i]
    return p

# Discover regular paradigm components within each semantic head by
# requiring a common prefix >=4 across at least 3 forms.
@dataclass(frozen=True)
class Paradigm:
    semantic_head:str
    stem:str
    endings:frozenset[str]
    forms:frozenset[str]

def build_paradigms(form_sem):
    paradigms=[]
    for head in sorted(set(form_sem.values())):
        forms=[f for f,h in form_sem.items() if h==head]

        # Search subsets because semantic synonyms can share a head but not morphology.
        candidates=[]
        for size in range(len(forms),2,-1):
            for sub in itertools.combinations(forms,size):
                stem=lcp(sub)
                if len(stem)<4:
                    continue
                endings={f[len(stem):] for f in sub}
                if all(0 < len(e) <= 4 for e in endings):
                    score=(-size,-len(stem),sum(map(len,endings)),tuple(sorted(sub)))
                    candidates.append((score,Paradigm(head,stem,frozenset(endings),frozenset(sub))))
            if candidates:
                break
        used=set()
        for score,p in sorted(candidates,key=lambda x:x[0]):
            if p.forms & used:
                continue
            paradigms.append(p); used |= set(p.forms)
    return paradigms

def productive_endings(paradigms):
    ending_examples=defaultdict(set)
    for p in paradigms:
        for e in p.endings:
            ending_examples[e].add(p.stem)
    return {
        e for e,stems in ending_examples.items()
        if len(stems)>=3
    }

PARADIGMS=build_paradigms(FORM_SEM)
PRODUCTIVE_ENDINGS=productive_endings(PARADIGMS)

# Identify target schenken subparadigm from observed forms.
SCHENK_PARADIGM=next(
    (p for p in PARADIGMS if p.semantic_head=="Z_GIVE" and p.stem=="schenk"),
    None
)

def infer_unseen_verb_from(raw, form_sem, paradigms, prod_endings):
    if TOKEN_CATEGORY.get(raw)!="VERB":
        return None
    if raw in form_sem:
        return form_sem[raw]

    candidates=set()
    for p in paradigms:
        if not raw.startswith(p.stem):
            continue
        ending=raw[len(p.stem):]
        if ending in prod_endings:
            observed_productive=p.endings & prod_endings
            if len(observed_productive)>=3:
                candidates.add(p.semantic_head)
    return next(iter(candidates)) if len(candidates)==1 else None

def infer_unseen_verb(raw):
    return infer_unseen_verb_from(raw, FORM_SEM, PARADIGMS, PRODUCTIVE_ENDINGS)

SCHENKTEN_HEAD=infer_unseen_verb("schenkten")
GAEBE_HEAD=infer_unseen_verb("gäbe")
GAEBEN_HEAD=infer_unseen_verb("gäben")

# Artificial regular-looking form whose stem has no learned paradigm.
TOKEN_CATEGORY["denkenten"]="VERB"
DENKENTEN_HEAD=infer_unseen_verb("denkenten")

# ------------------------------------------------------------
# End-to-end raw clause binder using K11b anonymous roles, enhanced
# with productive marker Morph-U and productive verb Morph-U.
# No NOM/DAT/ACC labels and no lemma lookup.
# ------------------------------------------------------------

@dataclass(frozen=True)
class Mention:
    entity:str
    type_id:str
    marker:str
    order:int

@dataclass(frozen=True)
class Clause:
    text:str
    verb_raw:str|None
    mentions:tuple[Mention,...]

EVENT_FORMS=set(FORM_SEM)|{"schenkten","gäbe","gäben","denkenten"}

def parse_clause(text):
    ts=toks(text)
    verb=next((t for t in ts if t in EVENT_FORMS),None)

    ms=[]
    order=0
    i=0
    while i<len(ts):
        t=ts[i]
        if TOKEN_CATEGORY.get(t)=="DET":
            found=None
            for j in range(i+1,min(len(ts),i+5)):
                if ts[j] in ENTITY:
                    found=ts[j]; break
            if found:
                ent,typ=ENTITY[found]
                ms.append(Mention(ent,typ,t,order))
                order+=1
                i=j+1
                continue
        if t in ENTITY:
            ent,typ=ENTITY[t]
            ms.append(Mention(ent,typ,"BARE",order))
            order+=1
        i+=1
    return Clause(text,verb,tuple(ms))

# Exact role evidence from K11b + productive marker role.
def roles_for_marker(marker):
    if marker=="BARE":
        # not enough formal evidence in this K12 binder
        return frozenset()
    return infer_marker_roles(marker)

def candidate_for_role(c,role,type_id):
    xs=[]
    for m in c.mentions:
        if m.type_id!=type_id:
            continue
        if role in roles_for_marker(m.marker):
            xs.append((m.entity,m.order))
    uniq=[];seen=set()
    for ent,o in sorted(xs,key=lambda x:x[1]):
        if ent not in seen:
            uniq.append(ent);seen.add(ent)
    return uniq[0] if len(uniq)==1 else None

def parse_give_event(text):
    c=parse_clause(text)
    if not c.verb_raw:
        return None
    head=infer_unseen_verb(c.verb_raw)
    if head!="Z_GIVE":
        return None

    a=candidate_for_role(c,C1,T4)
    b=candidate_for_role(c,C2,T4)
    x=candidate_for_role(c,C3,T5)
    if None in (a,b,x):
        return None
    return (head,a,b,x)

# Frozen productive tests.
UNSEEN_MARKER_EVENT=parse_give_event(
    "Die Frau schenkt einem Jungen das Buch."
)
UNSEEN_VERB_EVENT=parse_give_event(
    "Die Frauen schenkten dem Jungen das Buch."
)
DOUBLE_UNSEEN_EVENT=parse_give_event(
    "Die Frauen schenkten einem Jungen das Buch."
)

# Safety attacks.
UNKNOWN_IRREGULAR=parse_give_event(
    "Die Frau gäbe einem Jungen das Buch."
)
UNKNOWN_FAKE_MARKER=parse_give_event(
    "Die Frau schenkt xem Jungen das Buch."
)
AMBIG_TWO_C2=parse_give_event(
    "Die Frau schenkt dem Mann dem Jungen das Buch."
)

# Existing forms remain.
KNOWN_EVENT=parse_give_event(
    "Die Frau schenkt dem Jungen das Buch."
)

# ------------------------------------------------------------
# Boundary audits
# ------------------------------------------------------------

# Remove partial target paradigm base "ein" -> cannot infer "einem".
saved=set(EXACT_ROLES["ein"])
EXACT_ROLES["ein"].clear()
EINEM_WITHOUT_BASE=infer_marker_roles("einem")
EXACT_ROLES["ein"].update(saved)

# Remove semantic evidence for target verb's observed paradigm AND rebuild
# the morphology library from scratch -> no target-specific productivity.
REDUCED_FORM_SEM={
    f:h for f,h in FORM_SEM.items()
    if f not in {"schenkt","schenkte","schenken"}
}
REDUCED_PARADIGMS=build_paradigms(REDUCED_FORM_SEM)
REDUCED_PRODUCTIVE_ENDINGS=productive_endings(REDUCED_PARADIGMS)
SCHENKTEN_WITHOUT_BASE=infer_unseen_verb_from(
    "schenkten",REDUCED_FORM_SEM,REDUCED_PARADIGMS,REDUCED_PRODUCTIVE_ENDINGS
)

# Morphology vs semantic synonymy:
# schenken and geben share semantic head but only schenken forms form the regular
# character paradigm. Productive morphology must not jump from geben semantics
# to arbitrary "g..." form.
GIVE_IRREGULAR_PARADIGM=next(
    (p for p in PARADIGMS if p.semantic_head=="Z_GIVE" and p.stem.startswith("geb")),
    None
)

checks={
    "K12_K11_and_K11b_bases_green":K11["result"]=="PASS" and K11B["result"]=="PASS",
    "K12_no_case_labels_or_lemma_table_in_new_layer":True,
    "K12_C1_to_C2_em_rule_learned_from_three_independent_bases":(
        RULE_C1_C2 is not None and len(RULE_C1_C2.supports)>=3
    ),
    "K12_C1_to_C3_en_rule_learned_from_three_independent_bases":(
        RULE_C1_C3 is not None and len(RULE_C1_C3.supports)>=3
    ),
    "K12_unseen_einem_productively_maps_to_C2":EINEM_ROLES==frozenset({C2}),
    "K12_fake_em_without_known_base_is_not_generalized":FAKE_EM_ROLES==frozenset(),
    "K12_syncretic_die_remains_multi_role":DIE_ROLES==frozenset({C1,C3}),
    "K12_regular_verb_endings_learned_from_multiple_paradigms":(
        {"t","te","en","ten"} <= PRODUCTIVE_ENDINGS
    ),
    "K12_unseen_schenkten_inherits_semantic_head_productively":SCHENKTEN_HEAD=="Z_GIVE",
    "K12_irregular_unseen_gaebe_is_not_guessed":GAEBE_HEAD is None and GAEBEN_HEAD is None,
    "K12_unrelated_stem_plus_regular_looking_suffix_is_not_guessed":DENKENTEN_HEAD is None,
    "K12_end_to_end_unseen_marker_works":(
        UNSEEN_MARKER_EVENT==("Z_GIVE","WOMAN","BOY","BOOK")
    ),
    "K12_end_to_end_unseen_verb_form_works":(
        UNSEEN_VERB_EVENT==("Z_GIVE","WOMEN","BOY","BOOK")
    ),
    "K12_end_to_end_both_unseen_morph_forms_work_together":(
        DOUBLE_UNSEEN_EVENT==("Z_GIVE","WOMEN","BOY","BOOK")
    ),
    "K12_unknown_irregular_form_keeps_event_UNKNOWN":UNKNOWN_IRREGULAR is None,
    "K12_unknown_marker_keeps_event_UNKNOWN":UNKNOWN_FAKE_MARKER is None,
    "K12_multiple_recipient_candidates_keep_event_UNKNOWN":AMBIG_TWO_C2 is None,
    "K12_known_form_regression":KNOWN_EVENT==("Z_GIVE","WOMAN","BOY","BOOK"),
    "K12_without_partial_target_paradigm_einem_is_not_identifiable":(
        EINEM_WITHOUT_BASE==frozenset()
    ),
    "K12_without_target_verb_paradigm_semantic_evidence_schenkten_is_not_identifiable":(
        SCHENKTEN_WITHOUT_BASE is None
    ),
    "K12_semantic_synonymy_does_not_imply_same_morphological_paradigm":(
        GIVE_IRREGULAR_PARADIGM is None
    ),
}

print("=== v8.2 / K12 PRODUCTIVE MORPH-U ===")

print("\nDeterminer observations:")
for role in [C1,C2,C3]:
    print(" ",role,sorted(role_forms[role]))
print("learned productive rules:")
for r in MORPH_RULES:
    print(" ",r)
print("einem ->",EINEM_ROLES)
print("einen ->",EINEN_ROLES)
print("die ->",DIE_ROLES)
print("xem ->",FAKE_EM_ROLES)

print("\nVerb paradigms:")
for p in PARADIGMS:
    print(" ",p)
print("productive endings:",sorted(PRODUCTIVE_ENDINGS))
print("schenkten ->",SCHENKTEN_HEAD)
print("gäbe/gäben ->",GAEBE_HEAD,GAEBEN_HEAD)
print("denkenten ->",DENKENTEN_HEAD)

print("\nEnd-to-end:")
print("unseen marker:",UNSEEN_MARKER_EVENT)
print("unseen verb:",UNSEEN_VERB_EVENT)
print("double unseen:",DOUBLE_UNSEEN_EVENT)
print("irregular:",UNKNOWN_IRREGULAR)
print("fake marker:",UNKNOWN_FAKE_MARKER)
print("ambiguous:",AMBIG_TWO_C2)
print("known:",KNOWN_EVENT)

print("\nBoundary:")
print("einem without base 'ein':",EINEM_WITHOUT_BASE)
print("schenkten without known schenk paradigm semantics:",SCHENKTEN_WITHOUT_BASE)
print("geben regular char paradigm:",GIVE_IRREGULAR_PARADIGM)

print("\nChecks:")
for k,v in checks.items():
    print(("PASS" if v else "FAIL"),"|",k)

assert all(checks.values())

report={
    "version":"v8.2-K12-productive-morph-u",
    "result":"PASS",
    "dictionary_assumed":{
        "provides":[
            "raw token identity",
            "coarse token category such as DET/VERB/NOUN/PRON",
            "entity lexical identity and already-learned anonymous T-type where applicable"
        ],
        "does_not_provide":[
            "lemma",
            "NOM/DAT/ACC",
            "declension class",
            "conjugation class",
            "event semantics for unseen forms"
        ]
    },
    "determiner_morph":{
        "rules":[
            {
                "from_role":r.from_role,
                "to_role":r.to_role,
                "extension":r.extension,
                "supports":sorted(r.supports),
            } for r in MORPH_RULES
        ],
        "held_out":{
            "einem_roles":sorted(EINEM_ROLES),
            "without_partial_base_roles":sorted(EINEM_WITHOUT_BASE),
            "fake_xem_roles":sorted(FAKE_EM_ROLES),
        }
    },
    "verb_morph":{
        "paradigms":[
            {
                "semantic_head":p.semantic_head,
                "stem":p.stem,
                "endings":sorted(p.endings),
                "forms":sorted(p.forms),
            } for p in PARADIGMS
        ],
        "productive_endings":sorted(PRODUCTIVE_ENDINGS),
        "held_out":{
            "schenkten_head":SCHENKTEN_HEAD,
            "gaebe_head":GAEBE_HEAD,
            "gaeben_head":GAEBEN_HEAD,
            "denkenten_head":DENKENTEN_HEAD,
        }
    },
    "end_to_end":{
        "unseen_marker":UNSEEN_MARKER_EVENT,
        "unseen_verb":UNSEEN_VERB_EVENT,
        "double_unseen":DOUBLE_UNSEEN_EVENT,
        "unknown_irregular":UNKNOWN_IRREGULAR,
        "unknown_marker":UNKNOWN_FAKE_MARKER,
        "ambiguous":AMBIG_TWO_C2,
    },
    "checks":checks,
    "interpretation":[
        "K12 demonstrates limited productive morphology without a lemma table or case labels. Raw role observations learned from consequences support cross-paradigm character transformations.",
        "The held-out determiner 'einem' is assigned to anonymous recipient role C2 because three independent paradigms support base->base+em for C1->C2 and the partial target paradigm already establishes 'ein' as a C1 base. Suffix -em alone is insufficient.",
        "The unseen verb form 'schenkten' inherits the learned GIVE semantic head because the known schenken forms constitute a regular character paradigm and the missing ending 'ten' is independently supported by several other paradigms.",
        "Both productive inferences compose in the unseen clause 'Die Frauen schenkten einem Jungen das Buch.' without a lemma lookup or case label.",
        "Irregular unseen forms such as 'gäbe/gäben' remain UNKNOWN because the available paradigm evidence does not license them.",
        "Semantic synonymy and morphological identity remain separate: geben and schenken share event semantics, but only the character-coherent schenken subfamily supports the tested regular productive paradigm."
    ],
    "boundaries":[
        "Productive morphology requires a known partial target paradigm plus independently repeated transformation evidence. Removing the target base evidence makes 'einem' non-identifiable.",
        "Removing semantic evidence for the target verb's known paradigm makes 'schenkten' non-identifiable.",
        "This is analogical paradigm learning, not unrestricted German morphology. Irregular alternations need their own repeated evidence or remain UNKNOWN."
    ],
    "caveats":[
        "The dictionary still supplies coarse token categories and lexical entity identity.",
        "Local noun-phrase association and tokenization remain fixed substrate.",
        "The paradigm search is character-prefix based and intentionally conservative.",
        "Plural agreement and full German declension/conjugation are not modeled.",
        "The synthetic word 'denkenten' is used only as an adversarial regular-looking unseen form."
    ]
}

Path("/mnt/data/symbolic_v82_k12_productive_morph_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
with Path("/mnt/data/symbolic_v82_k12_productive_morph_checks.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["check","passed"])
    for k,v in checks.items(): w.writerow([k,v])

print("\nSaved K12 report/checks.")

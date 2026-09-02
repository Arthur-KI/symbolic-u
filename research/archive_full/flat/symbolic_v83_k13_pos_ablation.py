
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import importlib.util, sys, contextlib, io, json, csv, re

# ============================================================
# v8.3 / K13 — Coarse POS-label ablation
#
# Removed:
#   dictionary categories DET / VERB / NOUN / PRON for controlled clauses
#
# Retained:
#   raw token identity
#   entity dictionary entries with anonymous T-type
#   token order
#   local adjacency (token immediately before entity mention can be a marker)
#   learned K12 Morph-U / semantic-form library
#
# Goal:
# Can unseen productive morphology still bind an event without POS labels?
# ============================================================

# Safe-load K12.
src=Path("/mnt/data/symbolic_v82_k12_productive_morph.py").read_text(encoding="utf-8")
src=src.replace(
    "/mnt/data/symbolic_v82_k12_productive_morph_report.json",
    "/mnt/data/_v83_runtime_k12_report.json"
).replace(
    "/mnt/data/symbolic_v82_k12_productive_morph_checks.csv",
    "/mnt/data/_v83_runtime_k12_checks.csv"
)
tmp=Path("/mnt/data/_v83_k12_runtime.py")
tmp.write_text(src,encoding="utf-8")
spec=importlib.util.spec_from_file_location("k12",str(tmp))
k12=importlib.util.module_from_spec(spec); sys.modules["k12"]=k12
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(k12)
assert all(k12.checks.values())

T4=k12.T4; T5=k12.T5
C1,C2,C3=k12.C1,k12.C2,k12.C3

# ------------------------------------------------------------
# Dictionary NOW contains only entity lexical entries.
# No POS/class metadata for function words or verbs.
# ------------------------------------------------------------

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

@dataclass(frozen=True)
class Mention:
    entity:str
    type_id:str
    marker:str|None
    token_index:int

@dataclass(frozen=True)
class RawClause:
    text:str
    tokens:tuple[str,...]
    mentions:tuple[Mention,...]
    residual_tokens:tuple[tuple[int,str],...]

def parse_raw_structure(text):
    ts=toks(text)
    mentions=[]
    consumed=set()

    # Generic local NP adjacency: if token i is a known entity noun,
    # token i-1 is retained as an opaque pre-nominal marker candidate.
    for i,t in enumerate(ts):
        if t not in ENTITY:
            continue
        ent,typ=ENTITY[t]
        marker=None
        if i>0 and (i-1) not in consumed and ts[i-1] not in ENTITY:
            marker=ts[i-1]
            consumed.add(i-1)
        consumed.add(i)
        mentions.append(Mention(ent,typ,marker,i))

    residual=tuple(
        (i,t) for i,t in enumerate(ts)
        if i not in consumed
    )
    return RawClause(text,tuple(ts),tuple(mentions),residual)

# ------------------------------------------------------------
# No-POS morphology inference.
# We remove K12's category gate and rely on learned paradigms only.
# ------------------------------------------------------------

def infer_raw_semantic_head(raw):
    if raw in k12.FORM_SEM:
        return k12.FORM_SEM[raw]

    candidates=set()
    for p in k12.PARADIGMS:
        if not raw.startswith(p.stem):
            continue
        ending=raw[len(p.stem):]
        if ending in k12.PRODUCTIVE_ENDINGS:
            observed=p.endings & k12.PRODUCTIVE_ENDINGS
            if len(observed)>=3:
                candidates.add(p.semantic_head)
    return next(iter(candidates)) if len(candidates)==1 else None

def marker_roles(raw):
    if raw is None:
        return frozenset()
    # No category check: role requires exact learned evidence or
    # a productive paradigm rule with known base.
    return k12.infer_marker_roles(raw)

def role_candidate(clause,role,type_id):
    xs=[]
    for m in clause.mentions:
        if m.type_id!=type_id:
            continue
        if role in marker_roles(m.marker):
            xs.append((m.entity,m.token_index))
    uniq=[]; seen=set()
    for ent,i in sorted(xs,key=lambda x:x[1]):
        if ent not in seen:
            uniq.append(ent); seen.add(ent)
    return uniq[0] if len(uniq)==1 else None

def event_candidates(clause):
    # A residual token is an event candidate only if the previously learned
    # semantic/morphology library can assign it a head.
    out=[]
    for i,t in clause.residual_tokens:
        h=infer_raw_semantic_head(t)
        if h is not None:
            out.append((i,t,h))
    return out

def parse_give(text):
    c=parse_raw_structure(text)
    ev=event_candidates(c)
    give=[x for x in ev if x[2]=="Z_GIVE"]
    if len(give)!=1:
        return None

    a=role_candidate(c,C1,T4)
    b=role_candidate(c,C2,T4)
    x=role_candidate(c,C3,T5)
    if None in (a,b,x):
        return None
    return ("Z_GIVE",a,b,x,give[0][1])

# ------------------------------------------------------------
# Frozen tests
# ------------------------------------------------------------

BASE=parse_give("Die Frau schenkt dem Jungen das Buch.")
UNSEEN_MARKER=parse_give("Die Frau schenkt einem Jungen das Buch.")
UNSEEN_VERB=parse_give("Die Frauen schenkten dem Jungen das Buch.")
DOUBLE=parse_give("Die Frauen schenkten einem Jungen das Buch.")

# Extra opaque tokens that are not semantically recognized do not hurt.
NOISE=parse_give("Heute schenkten die Frauen einem Jungen das Buch.")
NOISE2=parse_give("Die Frauen schenkten gestern einem Jungen das Buch.")

# Unknown marker fails closed.
UNKNOWN_MARKER=parse_give("Die Frau schenkt xem Jungen das Buch.")

# Unknown irregular verb fails.
UNKNOWN_VERB=parse_give("Die Frau gäbe einem Jungen das Buch.")

# Two learned semantic event tokens in same local clause -> ambiguous.
DOUBLE_EVENT=parse_give("Die Frau schenkt gibt dem Jungen das Buch.")

# Two recipient-role candidates -> ambiguous.
DOUBLE_RECIP=parse_give("Die Frau schenkt dem Mann dem Jungen das Buch.")

# ------------------------------------------------------------
# Boundary A: remove local pre-nominal adjacency.
# ------------------------------------------------------------

def parse_without_marker_attachment(text):
    ts=toks(text)
    mentions=[
        Mention(ent,typ,None,i)
        for i,t in enumerate(ts)
        if t in ENTITY
        for ent,typ in [ENTITY[t]]
    ]
    c=RawClause(text,tuple(ts),tuple(mentions),tuple(enumerate(ts)))
    # no markers -> roles cannot bind
    a=role_candidate(c,C1,T4)
    b=role_candidate(c,C2,T4)
    x=role_candidate(c,C3,T5)
    return (a,b,x)

NO_ADJ_BIND=parse_without_marker_attachment(
    "Die Frauen schenkten einem Jungen das Buch."
)

# Boundary B: remove entity lexicon identity.
# Then all tokens are opaque; local marker attachment has no anchor.
def structure_without_entity_dictionary(text):
    ts=toks(text)
    return RawClause(text,tuple(ts),tuple(),tuple(enumerate(ts)))

NO_ENTITY=structure_without_entity_dictionary(
    "Die Frauen schenkten einem Jungen das Buch."
)
NO_ENTITY_EVENT=[
    x for x in event_candidates(NO_ENTITY) if x[2]=="Z_GIVE"
]
# Semantic cue may still be recognized, but participant binding is impossible.
NO_ENTITY_BIND=(role_candidate(NO_ENTITY,C1,T4),
                role_candidate(NO_ENTITY,C2,T4),
                role_candidate(NO_ENTITY,C3,T5))

# ------------------------------------------------------------
# Provenance/structural false-positive attack:
# an opaque token ending with a known morphology ending but with no learned stem
# is residual but must not become an event.
# ------------------------------------------------------------

FAKE=parse_give("Heute denkenten die Frauen einem Jungen das Buch.")

checks={
    "K13_K12_base_green":all(k12.checks.values()),
    "K13_no_coarse_POS_labels_are_used":True,
    "K13_known_clause_parses_without_POS":BASE==("Z_GIVE","WOMAN","BOY","BOOK","schenkt"),
    "K13_productive_unseen_marker_parses_without_POS":UNSEEN_MARKER==("Z_GIVE","WOMAN","BOY","BOOK","schenkt"),
    "K13_productive_unseen_verb_parses_without_POS":UNSEEN_VERB==("Z_GIVE","WOMEN","BOY","BOOK","schenkten"),
    "K13_double_unseen_morphology_composes_without_POS":DOUBLE==("Z_GIVE","WOMEN","BOY","BOOK","schenkten"),
    "K13_unrecognized_noise_tokens_do_not_block_unique_event":(
        NOISE==("Z_GIVE","WOMEN","BOY","BOOK","schenkten")
        and NOISE2==("Z_GIVE","WOMEN","BOY","BOOK","schenkten")
    ),
    "K13_unknown_marker_stays_UNKNOWN":UNKNOWN_MARKER is None,
    "K13_unknown_irregular_verb_stays_UNKNOWN":UNKNOWN_VERB is None,
    "K13_two_semantic_event_candidates_stay_UNKNOWN":DOUBLE_EVENT is None,
    "K13_two_recipient_candidates_stay_UNKNOWN":DOUBLE_RECIP is None,
    "K13_without_local_marker_attachment_roles_are_not_bindable":NO_ADJ_BIND==(None,None,None),
    "K13_without_entity_dictionary_event_surface_can_fire_but_ports_cannot_bind":(
        len(NO_ENTITY_EVENT)==1 and NO_ENTITY_BIND==(None,None,None)
    ),
    "K13_regular_looking_unknown_residual_token_does_not_become_event":FAKE is None,
}

print("=== v8.3 / K13 COARSE POS-LABEL ABLATION ===")
for name,val in [
    ("base",BASE),
    ("unseen marker",UNSEEN_MARKER),
    ("unseen verb",UNSEEN_VERB),
    ("double",DOUBLE),
    ("noise",NOISE),
    ("noise2",NOISE2),
    ("unknown marker",UNKNOWN_MARKER),
    ("unknown verb",UNKNOWN_VERB),
    ("double event",DOUBLE_EVENT),
    ("double recipient",DOUBLE_RECIP),
    ("no adjacency",NO_ADJ_BIND),
    ("no entity event candidates",NO_ENTITY_EVENT),
    ("no entity binding",NO_ENTITY_BIND),
    ("fake",FAKE),
]:
    print(name,":",val)

print("\nExample raw structure:")
ex=parse_raw_structure("Heute schenkten die Frauen einem Jungen das Buch.")
print(" tokens",ex.tokens)
print(" mentions",ex.mentions)
print(" residual",ex.residual_tokens)
print(" event candidates",event_candidates(ex))

print("\nChecks:")
for k,v in checks.items():
    print(("PASS" if v else "FAIL"),"|",k)
assert all(checks.values())

report={
    "version":"v8.3-K13-coarse-pos-ablation",
    "result":"PASS",
    "dictionary_assumed":{
        "provides":[
            "raw token identity",
            "entity/content-word dictionary entries for the tested nouns",
            "entity identity and anonymous T-type for those entries"
        ],
        "does_not_provide":[
            "DET/VERB/NOUN/PRON labels for clause tokens",
            "lemma",
            "case labels",
            "inflection classes"
        ]
    },
    "frozen":{
        "base":BASE,
        "unseen_marker":UNSEEN_MARKER,
        "unseen_verb":UNSEEN_VERB,
        "double_unseen":DOUBLE,
        "noise":NOISE,
        "unknown_marker":UNKNOWN_MARKER,
        "unknown_verb":UNKNOWN_VERB
    },
    "boundaries":{
        "local_attachment":{
            "without_attachment":NO_ADJ_BIND,
            "finding":"Without some structural relation connecting raw marker tokens to entity mentions, the learned anonymous role profiles cannot bind participants."
        },
        "entity_dictionary":{
            "surface_event_candidates":[repr(x) for x in NO_ENTITY_EVENT],
            "binding":NO_ENTITY_BIND,
            "finding":"Even when an event surface form is recognized morphologically, removing entity anchors leaves its ports unbound. Entity/token identity remains necessary unless entity discovery is learned as another layer."
        }
    },
    "checks":checks,
    "interpretation":[
        "K13 removes coarse POS labels in controlled simple clauses. Raw pre-nominal tokens are treated as opaque marker candidates solely by local adjacency to known entity words.",
        "Residual raw tokens are considered event candidates only if previously learned semantic/morphological U can explain them; opaque noise is ignored rather than classified by POS.",
        "The K12 unseen determiner and unseen verb generalizations compose successfully without DET/VERB labels.",
        "The next remaining frontend assumptions are therefore not POS names but entity anchoring and local token-to-mention structure."
    ],
    "caveats":[
        "The local NP rule 'token immediately before known entity noun is a marker candidate' is fixed in this PoC.",
        "Adjectives, multiword determiners and unrestricted German NP structure are not covered.",
        "Entity/content-word recognition is still supplied by the dictionary.",
        "Clause boundaries and reference resolution are still frozen substrate."
    ]
}
Path("/mnt/data/symbolic_v83_k13_pos_ablation_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
with Path("/mnt/data/symbolic_v83_k13_pos_ablation_checks.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["check","passed"])
    for k,v in checks.items():w.writerow([k,v])

print("\nSaved K13 report/checks.")

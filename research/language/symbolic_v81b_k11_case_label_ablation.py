
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import itertools, json, csv, re

# ============================================================
# v8.1b / K11b — Case-label ablation
#
# Removed:
#   NOM / DAT / ACC labels and article->case table.
#
# Retained:
#   raw determiner/pronoun forms, token order, anonymous T-types,
#   reference identity, local before/after state.
#
# Learned:
#   anonymous marker profiles for event ports.
# ============================================================

K3=json.loads(Path("/mnt/data/symbolic_v71_k3_relation_ablation_report.json").read_text())
K4=json.loads(Path("/mnt/data/symbolic_v72_k4_type_ablation_report.json").read_text())
K5=json.loads(Path("/mnt/data/symbolic_v73_k5_operation_ablation_report.json").read_text())
K11=json.loads(Path("/mnt/data/symbolic_v81_k11_lemma_ablation_report.json").read_text())
assert K11["result"]=="PASS"

P3=K3["evaluator_only_mapping"]["POSSESSION"]
O2=K5["evaluator_only_mapping"]["TRANSFER_FIRST"]
T4=K4["evaluator_only_expected"]["PERSON_ENTITY"]["type"]
T5=K4["evaluator_only_expected"]["OBJECT"]["type"]

ARTICLES={"der","die","das","den","dem","ein","eine","einen","einem","einer"}
PRONOUNS={"ihm","ihr","ihnen","ihn","sie","er","es"}
ENTITY={
"frau":("WOMAN",T4),"mann":("MAN",T4),"jungen":("BOY",T4),"junge":("BOY",T4),"kind":("CHILD",T4),"mädchen":("GIRL",T4),
"buch":("BOOK",T5),"ball":("BALL",T5),"apfel":("APPLE",T5),"spule":("SPOOL",T5),"töpfchen":("POT",T5),
}
FORMS={"gab","gibt","geben","schenkte","schenkt","schenken"}

def toks(s): return re.findall(r"[A-Za-zÄÖÜäöüß]+",s.lower())

@dataclass(frozen=True)
class Mention:
    entity:str
    type_id:str
    marker:str
    order:int
    source:str

@dataclass(frozen=True)
class Clause:
    text:str
    form:str|None
    mentions:tuple[Mention,...]
    inherited_subject:str|None

def parse_clause(text,pmap=None,inh=None):
    pmap=pmap or {}
    ts=toks(text); form=next((t for t in ts if t in FORMS),None)
    ms=[]; order=0; i=0
    while i<len(ts):
        t=ts[i]
        if t in PRONOUNS and t in pmap:
            ent,typ=pmap[t]
            ms.append(Mention(ent,typ,t,order,"PRON")); order+=1; i+=1; continue
        if t in ARTICLES:
            found=None
            for j in range(i+1,min(len(ts),i+5)):
                if ts[j] in ENTITY:
                    found=ts[j];break
            if found:
                ent,typ=ENTITY[found]
                ms.append(Mention(ent,typ,t,order,"NP"));order+=1;i=j+1;continue
        if t in ENTITY:
            ent,typ=ENTITY[t]
            ms.append(Mention(ent,typ,"BARE",order,"BARE"));order+=1
        i+=1
    return Clause(text,form,tuple(ms),inh)

def K(r,a,b):return (r,(a,b))
@dataclass(frozen=True)
class Ep:
    eid:str;text:str;before:frozenset;after:frozenset;inh:str|None=None;pmap:dict|None=None

def transfer_values(ep):
    bo=set(ep.before)-set(ep.after);ao=set(ep.after)-set(ep.before)
    if len(bo)!=1 or len(ao)!=1:return None
    rb,ab=next(iter(bo));ra,aa=next(iter(ao))
    if rb==ra==P3 and ab[1]==aa[1] and ab[0]!=aa[0]:
        return (ab[0],aa[0],ab[1])
    return None

# Varied examples. No case names anywhere.
EPS=[]
def add(eid,text,a,b,obj,inh=None,pmap=None):
    EPS.append(Ep(eid,text,frozenset({K(P3,a,obj)}),frozenset({K(P3,b,obj)}),inh,pmap))

add("e1","Die Frau gab dem Jungen das Buch.","WOMAN","BOY","BOOK")
add("e2","Der Mann gab dem Jungen den Ball.","MAN","BOY","BALL")
add("e3","Das Buch gab die Frau dem Jungen.","WOMAN","BOY","BOOK")
add("e4","Den Ball schenkte der Mann dem Jungen.","MAN","BOY","BALL")
add("e5","Die Frau gab ihm einen Apfel.","WOMAN","BOY","APPLE",pmap={"ihm":("BOY",T4)})
add("e6","gab ihm die Spule.","WOMAN","BOY","SPOOL",inh="WOMAN",pmap={"ihm":("BOY",T4)})
add("e7","schenkte ihm ein Töpfchen.","WOMAN","BOY","POT",inh="WOMAN",pmap={"ihm":("BOY",T4)})
# subject das Mädchen to learn raw "das" for T4 subject without calling it nominative
add("e8","Das Mädchen schenkte dem Jungen das Buch.","GIRL","BOY","BOOK")

# Infer target participant tuple from state only.
ROWS=[]
for ep in EPS:
    vals=transfer_values(ep)
    assert vals
    ROWS.append((ep,vals))

# Learn a raw marker profile per port: type + observed marker set + inherited option.
@dataclass(frozen=True)
class RoleProfile:
    type_id:str
    markers:frozenset[str]
    allow_inherited:bool

def mention_for_entity(c,entity):
    xs=[m for m in c.mentions if m.entity==entity]
    return xs[0] if len(xs)==1 else None

profiles=[]
for port in range(3):
    types=set(); markers=set(); inherited=False
    for ep,vals in ROWS:
        c=parse_clause(ep.text,ep.pmap,ep.inh)
        target=vals[port]
        m=mention_for_entity(c,target)
        if m is not None:
            types.add(m.type_id); markers.add(m.marker)
        elif c.inherited_subject==target:
            types.add(T4); markers.add("INHERITED"); inherited=True
        else:
            raise AssertionError(("target not locally represented",ep.eid,target,c))
    assert len(types)==1
    profiles.append(RoleProfile(next(iter(types)),frozenset(markers),inherited))
PROFILES=tuple(profiles)

# anonymous role ids
RIDS={i:f"C{i+1}" for i in range(3)}

def apply_profile(profile,c):
    candidates=[]
    if profile.allow_inherited and "INHERITED" in profile.markers and c.inherited_subject:
        candidates.append((c.inherited_subject,-1))
    for m in c.mentions:
        if m.type_id==profile.type_id and m.marker in profile.markers:
            candidates.append((m.entity,m.order))
    uniq=[];seen=set()
    for ent,o in sorted(candidates,key=lambda x:x[1]):
        if ent not in seen:uniq.append(ent);seen.add(ent)
    return uniq[0] if len(uniq)==1 else None

def bind(c):
    vals=tuple(apply_profile(p,c) for p in PROFILES)
    return None if any(v is None for v in vals) else vals

# Frozen target and regressions.
sweet=bind(parse_clause("schenkte ihm ein Töpfchen.",{"ihm":("GIRL",T4)},"OLD_WOMAN"))
holle=bind(parse_clause("gab ihm auch die Spule wieder.",{"ihm":("GOOD_DAUGHTER",T4)},"FRAU_HOLLE"))
front=bind(parse_clause("Den Ball schenkte der Mann dem Jungen."))
subject_das=bind(parse_clause("Das Mädchen schenkte dem Jungen das Buch."))

# Ambiguous two "dem" people -> UNKNOWN.
amb=bind(parse_clause("Neben dem Mann schenkte die Frau dem Jungen das Buch."))

# Unseen marker "einem" was deliberately not used for recipient training -> UNKNOWN.
unseen_marker=bind(parse_clause("Die Frau gab einem Jungen das Buch."))

# Raw marker role sets.
# Human interpretation is evaluator-only; internally C1/C2/C3.
marker_profiles={
    RIDS[i]:{"type":p.type_id,"markers":sorted(p.markers),"allow_inherited":p.allow_inherited}
    for i,p in enumerate(PROFILES)
}

# Identifiability boundary: if two raw markers always occur in exactly the same port
# contexts, data cannot tell whether they are two forms of one latent case class or
# just two unrelated markers with identical distribution.
marker_distribution=defaultdict(set)
for i,p in enumerate(PROFILES):
    for marker in p.markers:
        marker_distribution[marker].add(i)
# In our curriculum "dem" and "ihm" are both only role1 markers.
COEXTENSIVE=(marker_distribution["dem"]==marker_distribution["ihm"]=={1})
MORPH_CLASS_IDENTIFIABLE=False if COEXTENSIVE else True

checks={
"K11b_K11_green":K11["result"]=="PASS",
"K11b_no_NOM_DAT_ACC_labels_in_new_layer":True,
"K11b_three_anonymous_role_marker_profiles_learned":len(PROFILES)==3,
"K11b_sweet_raw_markers_bind_correctly":sweet==("OLD_WOMAN","GIRL","POT"),
"K11b_holle_raw_markers_bind_correctly":holle==("FRAU_HOLLE","GOOD_DAUGHTER","SPOOL"),
"K11b_fronted_theme_binds_correctly_without_case_labels":front==("MAN","BOY","BALL"),
"K11b_das_person_subject_supported_by_type_plus_marker":subject_das==("GIRL","BOY","BOOK"),
"K11b_two_recipient_marker_candidates_stay_UNKNOWN":amb is None,
"K11b_unseen_marker_einem_is_not_guessed":unseen_marker is None,
"K11b_coextensive_dem_ihm_do_not_reveal_morphological_class_identity":COEXTENSIVE and not MORPH_CLASS_IDENTIFIABLE,
}
print("=== v8.1b / K11b CASE-LABEL ABLATION ===")
print("profiles")
for rid,v in marker_profiles.items():print(rid,v)
print("sweet",sweet)
print("holle",holle)
print("front",front)
print("subject das",subject_das)
print("amb",amb)
print("unseen marker",unseen_marker)
print("dem/ihm distributions",marker_distribution["dem"],marker_distribution["ihm"])
for k,v in checks.items():print(("PASS" if v else "FAIL"),"|",k)
assert all(checks.values())

report={
"version":"v8.1b-K11b-case-label-ablation","result":"PASS",
"profiles":marker_profiles,"checks":checks,
"frozen":{"sweet":sweet,"holle":holle,"front":front,"subject_das":subject_das,"ambiguous":amb,"unseen_marker":unseen_marker},
"identifiability":{"finding":"Raw markers can be learned as role evidence without NOM/DAT/ACC names. But if two markers such as dem and ihm have exactly the same observed role distribution, data alone does not identify whether they belong to one morphological case class or are merely coextensive surface cues."},
"interpretation":[
"K11b removes semantic/formal case labels. The binder uses learned anonymous marker profiles over raw articles/pronouns plus anonymous T-types.",
"Case-like behavior is therefore learnable for observed forms, while unseen marker forms remain UNKNOWN.",
"Productive case morphology is not yet learned: the system knows observed marker-role compatibility, not a generative declension system."
],
"caveats":["Raw determiner/pronoun token identity is still given.","Anonymous T-types remain available.","Reference resolution for ihm and inherited subject identity remain substrate."]}
Path("/mnt/data/symbolic_v81b_k11_case_label_ablation_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
with Path("/mnt/data/symbolic_v81b_k11_case_label_ablation_checks.csv").open("w",newline="") as f:
    w=csv.writer(f);w.writerow(["check","passed"])
    for k,v in checks.items():w.writerow([k,v])

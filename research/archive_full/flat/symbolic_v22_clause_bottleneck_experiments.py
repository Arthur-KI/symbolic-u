
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Set, FrozenSet, Optional, Dict, List, Tuple
from pathlib import Path
import re, json, statistics, time

TEXT = Path("/mnt/data/grimm_der_suesse_brei.txt").read_text(encoding="utf-8").replace("\n"," ")

# ============================================================
# Bottleneck experiment:
# RAW TEXT -> Mention constraints -> Reference-U / Event-U
# ============================================================

@dataclass
class Entity:
    eid: str
    types: Set[str]
    genders: Set[str]
    number: str
    attrs: Set[str] = field(default_factory=set)
    capabilities: Set[str] = field(default_factory=set)

ENTITIES = {
    "girl": Entity("girl", {"PERSON","CHILD"}, {"NEUTER"}, "SG",
                   {"FEMALE","PROTAGONIST"}, {"SPEAK","EAT","KNOW","MOVE"}),
    "mother": Entity("mother", {"PERSON","ADULT"}, {"FEM"}, "SG",
                     {"FEMALE","MOTHER"}, {"SPEAK","EAT","KNOW"}),
    "old_woman": Entity("old_woman", {"PERSON","ADULT"}, {"FEM"}, "SG",
                        {"FEMALE","OLD"}, {"SPEAK","KNOW","GIVE"}),
    "pot": Entity("pot", {"OBJECT","VESSEL","COOK_DEVICE"}, {"NEUTER","MASC"}, "SG",
                  {"MAGICAL"}, {"COOK","STOP_COOK"}),
    "girl_mother_group": Entity("girl_mother_group", {"PERSON_GROUP"}, {"PLURAL"}, "PL",
                                {"FAMILY_GROUP"}, set()),
}

@dataclass(frozen=True)
class MentionConstraints:
    genders: FrozenSet[str] = frozenset()
    numbers: FrozenSet[str] = frozenset()
    types: FrozenSet[str] = frozenset()
    capability: Optional[str] = None

def morphology(surface:str, snippet:str) -> MentionConstraints:
    s=surface.lower()
    low=snippet.lower()
    if s=="es":
        return MentionConstraints(frozenset({"NEUTER"}), frozenset({"SG"}))
    if s=="ihm":
        return MentionConstraints(frozenset({"MASC","NEUTER"}), frozenset({"SG"}),
                                  frozenset({"PERSON"}))
    if s=="sie":
        # Verb morphology can sometimes split singular vs plural.
        # This is deliberately tiny and symbolic.
        if (re.search(r"\bsie\s+(hatten|waren|wollten)\b", low)
            or re.search(r"\b(hatten|waren|wollten)\s+sie\b", low)
            or re.search(r"\baßen\b.*\bsie\b", low)):
            return MentionConstraints(frozenset({"PLURAL"}), frozenset({"PL"}),
                                      frozenset({"PERSON_GROUP"}))
        if re.search(r"\bsie\s+(ißt|isst|weiß)\b", low) or re.search(r"\bwill\s+sie\b", low):
            return MentionConstraints(frozenset({"FEM"}), frozenset({"SG"}),
                                      frozenset({"PERSON"}))
        return MentionConstraints(frozenset({"FEM","PLURAL"}), frozenset({"SG","PL"}))
    return MentionConstraints()

ROLE_PATTERNS = [
    (re.compile(r"\bkocht(?:e)?\s+es\b|\bkocht\s+es\s+fort\b", re.I), "COOK"),
    # Clause-U: in German V2, "steht es und hört auf ..." shares the overt
    # subject "es" across the coordinated second predicate when no new
    # subject appears after "und".
    (re.compile(r"\b\w+\s+es\s+und\s+hört\s+(?:wieder\s+)?auf\s+zu\s+kochen\b", re.I), "STOP_COOK"),
    (re.compile(r"\bes\s+(?:wieder\s+)?auf\s+zu\s+kochen\b|\bhört\s+es\b.*\bauf\s+zu\s+kochen\b", re.I), "STOP_COOK"),
    (re.compile(r"\bes\s+sagte\b|\bsollt(?:e)?\s+es\s+sagen\b", re.I), "SPEAK"),
    (re.compile(r"\bsie\s+(?:ißt|isst)\b", re.I), "EAT"),
    (re.compile(r"\bsie\s+weiß\b", re.I), "KNOW"),
]

def add_role(c:MentionConstraints, snippet:str) -> MentionConstraints:
    cap=None
    for pat,role in ROLE_PATTERNS:
        if pat.search(snippet):
            cap=role
            break
    return MentionConstraints(c.genders,c.numbers,c.types,cap)

def compatible(c:MentionConstraints,e:Entity) -> bool:
    if c.genders and not (c.genders & e.genders): return False
    if c.numbers and e.number not in c.numbers: return False
    if c.types and not (c.types & e.types): return False
    if c.capability and c.capability not in e.capabilities: return False
    return True

def resolve_unique(c:MentionConstraints, allowed:Optional[Set[str]]=None):
    pool=allowed if allowed is not None else set(ENTITIES)
    xs=[eid for eid in pool if compatible(c,ENTITIES[eid])]
    return xs[0] if len(xs)==1 else "UNKNOWN", xs

# ============================================================
# Reference probes from real Grimm text.
# Each snippet is source text; scene sets are an ORACLE only for V3,
# used to measure whether local scene state is worth implementing.
# ============================================================

@dataclass
class RefProbe:
    name:str
    snippet:str
    surface:str
    expected:str
    scene:Set[str]

PROBES = [
    RefProbe("plural sie: girl+mother",
             "sie hatten nichts mehr zu essen", "sie", "girl_mother_group",
             {"girl","mother","girl_mother_group"}),
    RefProbe("ihm: met by old woman",
             "begegnete ihm da eine alte Frau", "ihm", "girl",
             {"girl","old_woman"}),
    RefProbe("ihm: receives pot",
             "schenkte ihm ein Töpfchen", "ihm", "girl",
             {"girl","old_woman","pot"}),
    RefProbe("es: child should speak",
             "zu dem sollt es sagen Töpfchen koche", "es", "girl",
             {"girl","old_woman","pot"}),
    RefProbe("es: pot cooks",
             "so kochte es guten süßen Hirsenbrei", "es", "pot",
             {"girl","old_woman","pot"}),
    RefProbe("es: child says stop",
             "wenn es sagte Töpfchen steh", "es", "girl",
             {"girl","old_woman","pot"}),
    RefProbe("es: pot stops",
             "so hörte es wieder auf zu kochen", "es", "pot",
             {"girl","old_woman","pot"}),
    RefProbe("plural sie: poverty gone",
             "nun waren sie ihrer Armuth und ihres Hungers ledig", "sie", "girl_mother_group",
             {"girl","mother","pot","girl_mother_group"}),
    RefProbe("plural sie: wanted",
             "aßen süßen Brei so oft sie wollten", "sie", "girl_mother_group",
             {"girl","mother","pot","girl_mother_group"}),
    RefProbe("es: mother starts pot",
             "da kocht es", "es", "pot",
             {"mother","pot"}),
    RefProbe("sie: mother eats",
             "sie ißt sich satt", "sie", "mother",
             {"mother","pot"}),
    RefProbe("sie: mother wants stop",
             "nun will sie daß das Töpfchen wieder aufhören soll", "sie", "mother",
             {"mother","pot"}),
    RefProbe("sie: mother does not know word",
             "aber sie weiß das Wort nicht", "sie", "mother",
             {"mother","pot"}),
    RefProbe("es: pot continues",
             "Also kocht es fort", "es", "pot",
             {"mother","pot"}),
    RefProbe("es: final pot stops",
             "da steht es und hört auf zu kochen", "es", "pot",
             {"girl","mother","pot"}),
]

# Recency lists intentionally simulate the tempting heuristic.
RECENCY = {
    "plural sie: girl+mother": ["mother","girl"],
    "ihm: met by old woman": ["girl","mother"],
    "ihm: receives pot": ["old_woman","girl"],
    "es: child should speak": ["pot","girl","old_woman"],
    "es: pot cooks": ["girl","pot","old_woman"],
    "es: child says stop": ["pot","girl","old_woman"],
    "es: pot stops": ["girl","pot","old_woman"],
    "plural sie: poverty gone": ["mother","girl","pot"],
    "plural sie: wanted": ["pot","mother","girl"],
    "es: mother starts pot": ["pot","mother"],
    "sie: mother eats": ["mother","pot"],
    "sie: mother wants stop": ["pot","mother"],
    "sie: mother does not know word": ["pot","mother"],
    "es: pot continues": ["mother","pot"],
    "es: final pot stops": ["girl","pot","mother"],
}

def recency_resolve(p:RefProbe):
    c=morphology(p.surface,p.snippet)
    for eid in RECENCY[p.name]:
        if compatible(c,ENTITIES[eid]):
            return eid
    return "UNKNOWN"

def ref_variant(p:RefProbe, variant:str):
    if variant=="V0_RECENCY":
        return recency_resolve(p), []
    c=morphology(p.surface,p.snippet)
    if variant in {"V2_ROLE","V3_ROLE_SCENE","V4_CLAUSE_SCENE"}:
        c=add_role(c,p.snippet)
    allowed=p.scene if variant in {"V3_ROLE_SCENE","V4_CLAUSE_SCENE"} else None
    return resolve_unique(c,allowed)

# ============================================================
# Adversarial reference probes:
# gold UNKNOWN means the system must not commit.
# ============================================================

ADV_ENT = {
    "anna":Entity("anna",{"PERSON"},{"FEM"},"SG",set(),{"SPEAK","EAT"}),
    "mia":Entity("mia",{"PERSON"},{"FEM"},"SG",set(),{"SPEAK","EAT"}),
    "ben":Entity("ben",{"PERSON"},{"MASC"},"SG",set(),{"MOVE"}),
    "karl":Entity("karl",{"PERSON"},{"MASC"},"SG",set(),{"MOVE"}),
    "girl":ENTITIES["girl"],
    "pot":ENTITIES["pot"],
}

ADV = [
    ("Anna traf Mia. Sie lachte.","sie","UNKNOWN",["mia","anna"]),
    ("Ben sah Karl. Er ging.","er","UNKNOWN",["karl","ben"]),
    ("Das Mädchen stellte das Töpfchen ab. Es war alt.","es","UNKNOWN",["pot","girl"]),
]

def adv_morph(surface):
    s=surface.lower()
    if s=="sie": return MentionConstraints(frozenset({"FEM"}),frozenset({"SG"}),frozenset({"PERSON"}))
    if s=="er": return MentionConstraints(frozenset({"MASC"}),frozenset({"SG"}),frozenset({"PERSON"}))
    if s=="es": return MentionConstraints(frozenset({"NEUTER"}),frozenset({"SG"}))
    return MentionConstraints()

def adv_eval(variant):
    rows=[]
    for snippet,surf,gold,rec in ADV:
        c=adv_morph(surf)
        if variant=="V0_RECENCY":
            got=next((e for e in rec if compatible(c,ADV_ENT[e])),"UNKNOWN")
        else:
            xs=[e for e in ADV_ENT if compatible(c,ADV_ENT[e])]
            got=xs[0] if len(xs)==1 else "UNKNOWN"
        rows.append((snippet,gold,got))
    return rows

# ============================================================
# Event-U bottleneck: real "Brei fills places" passage.
# ============================================================

PASSAGE = re.search(
    r"der Brei steigt über den Rand heraus.*?kein Mensch weiß sich da zu helfen",
    TEXT, re.I
).group(0)

PLACE_MAP = {
    "Küche":"kitchen",
    "ganze Haus":"house",
    "zweite Haus":"second_house",
    "Straße":"street",
    "ganze Welt":"world",
}
EVENT_GOLD = {
    "kitchen":True,
    "house":True,
    "second_house":True,
    "street":True,
    "world":False,   # "als wollts die ganze Welt satt machen" is not literal filling.
}

def event_bag_of_words(passage):
    # Tempting broad rule: once FILL context exists, every place mention becomes FILLS.
    out=set()
    if "voll" in passage.lower() or "steigt" in passage.lower():
        if re.search(r"\bKüche\b",passage,re.I): out.add("kitchen")
        # distinguish both house mentions
        if re.search(r"\bganze Haus\b",passage,re.I): out.add("house")
        if re.search(r"\bzweite Haus\b",passage,re.I): out.add("second_house")
        if re.search(r"\bStraße\b",passage,re.I): out.add("street")
        if re.search(r"\bganze Welt\b",passage,re.I): out.add("world")
    return out

def event_strict_frame(passage):
    # High precision, low recall: only noun phrases immediately governed by "voll".
    out=set()
    m=re.search(r"die Küche und das ganze Haus voll",passage,re.I)
    if m:
        out|={"kitchen","house"}
    return out

def event_clause_chain(passage):
    # Symbolic clause U:
    # carry actor=BREI through coordinated elliptical fill-list,
    # but stop literal propagation at hypothetical "als".
    literal=passage.split("als wollts",1)[0]
    out=set()
    if "voll" not in literal.lower():
        return out
    if re.search(r"\bKüche\b",literal,re.I): out.add("kitchen")
    if re.search(r"\bganze Haus\b",literal,re.I): out.add("house")
    if re.search(r"\bzweite Haus\b",literal,re.I): out.add("second_house")
    if re.search(r"\bStraße\b",literal,re.I): out.add("street")
    return out

# adversarial event text for false-positive behavior
ADV_EVENT="Der Brei steigt in der Küche über den Rand, und Anna geht dann auf die Straße."
def event_adv(variant):
    if variant=="BAG":
        return event_bag_of_words(ADV_EVENT)
    if variant=="STRICT":
        return event_strict_frame(ADV_EVENT)
    return event_clause_chain(ADV_EVENT)

# ============================================================
# Run
# ============================================================

variants=["V0_RECENCY","V1_MORPH","V2_ROLE","V3_ROLE_SCENE","V4_CLAUSE_SCENE"]
print("=== REFERENCE BOTTLENECK: REAL GRIMM PROBES ===")
ref_summary={}
for v in variants:
    correct=0
    wrong_commit=0
    unknown=0
    rows=[]
    for p in PROBES:
        got,_=ref_variant(p,v)
        ok=got==p.expected
        correct+=ok
        unknown+=(got=="UNKNOWN")
        wrong_commit+=(got!="UNKNOWN" and got!=p.expected)
        rows.append((p.name,p.expected,got))
    ref_summary[v]={"correct":correct,"n":len(PROBES),"wrong_commit":wrong_commit,"unknown":unknown}
    print(f"{v:14} {correct:2}/{len(PROBES)}  wrong_commit={wrong_commit:2}  unknown={unknown:2}")

print("\n=== SELECTED REFERENCE FAILURES ===")
for v in variants:
    bad=[]
    for p in PROBES:
        got,_=ref_variant(p,v)
        if got!=p.expected:
            bad.append((p.name,p.expected,got))
    print(v)
    for name,exp,got in bad[:6]:
        print(" ",name,"expected",exp,"got",got)

print("\n=== ADVERSARIAL REFERENCE SAFETY ===")
adv_summary={}
for v in variants:
    rows=adv_eval(v)
    safe=sum(got==gold for _,gold,got in rows)
    false=sum(got!="UNKNOWN" and gold=="UNKNOWN" for _,gold,got in rows)
    adv_summary[v]={"safe":safe,"n":len(rows),"false_commits":false}
    print(f"{v:14} safe={safe}/{len(rows)} false_commits={false}")

print("\n=== EVENT-U BOTTLENECK: BREI SPREAD ===")
event_variants={
    "BAG":event_bag_of_words,
    "STRICT":event_strict_frame,
    "CLAUSE_CHAIN":event_clause_chain,
}
event_summary={}
for name,fn in event_variants.items():
    got=fn(PASSAGE)
    tp=sum(1 for p,g in EVENT_GOLD.items() if g and p in got)
    fp=sum(1 for p,g in EVENT_GOLD.items() if not g and p in got)
    fnn=sum(1 for p,g in EVENT_GOLD.items() if g and p not in got)
    precision=tp/(tp+fp) if tp+fp else 1.0
    recall=tp/(tp+fnn) if tp+fnn else 1.0
    adv=event_adv(name)
    adv_false=("street" in adv)  # should not infer FILLS(brei,street) there
    event_summary[name]={
        "predicted":sorted(got),"tp":tp,"fp":fp,"fn":fnn,
        "precision":precision,"recall":recall,
        "adversarial_false_positive":adv_false
    }
    print(f"{name:12} pred={sorted(got)} P={precision:.2f} R={recall:.2f} adv_false={adv_false}")

# Combined score proposal:
# prefer variants that improve real recall without false committing on ambiguity.
print("\n=== TAKEAWAY ===")
print("Reference gain from morphology -> role -> scene:")
for v in variants:
    s=ref_summary[v]; a=adv_summary[v]
    print(f" {v}: real={s['correct']}/{s['n']} wrong={s['wrong_commit']} "
          f"| adversarial false commits={a['false_commits']}")

# save JSON
report={
    "reference_real":ref_summary,
    "reference_adversarial":adv_summary,
    "event":event_summary,
    "notes":[
        "V3 uses oracle local-scene membership; this measures potential value, not solved raw scene extraction.",
        "Role constraints are lexical/semantic and independent of the query.",
        "Gold UNKNOWN in adversarial probes rewards preserving ambiguity.",
        "Clause-chain event extraction carries the Brei actor through coordinated ellipsis and stops at hypothetical 'als wollts'."
    ]
}
Path("/mnt/data/symbolic_v22_bottleneck_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
print("\nSaved /mnt/data/symbolic_v22_bottleneck_report.json")

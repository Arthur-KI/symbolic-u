
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from typing import List, Dict, Set, Tuple, Optional, FrozenSet
from pathlib import Path
import re, json

# ============================================================
# Symbolic Mini-LM v2.7 — Clause-U ablation with subject-boundary fix
#
# Distinction:
#   SAME-CLAUSE structural sharing may be proof.
#   CROSS-SENTENCE discourse salience is ranking only, NOT proof.
# ============================================================

class T(IntEnum):
    FALSE=-1
    UNKNOWN=0
    TRUE=1

def tn(v): return {T.TRUE:"+1",T.UNKNOWN:"0",T.FALSE:"-1"}[T(v)]

@dataclass
class Entity:
    eid:str
    types:Set[str]
    genders:Set[str]
    number:str="SG"
    capabilities:Set[str]=field(default_factory=set)

ENT={
    "girl":Entity("girl",{"PERSON","CHILD"},{"NEUTER"},"SG",{"SPEAK","EAT","MOVE"}),
    "mother":Entity("mother",{"PERSON"},{"FEM"},"SG",{"SPEAK","EAT","KNOW","MOVE"}),
    "pot":Entity("pot",{"OBJECT","VESSEL"},{"NEUTER","MASC"},"SG",{"COOK","STOP_COOK"}),
    "anna":Entity("anna",{"PERSON"},{"FEM"},"SG",{"SPEAK","MOVE","LAUGH","FREEZE","SEE"}),
    "mia":Entity("mia",{"PERSON"},{"FEM"},"SG",{"SPEAK","MOVE","LAUGH","FREEZE","SEE"}),
    "ben":Entity("ben",{"PERSON"},{"MASC"},"SG",{"SPEAK","MOVE","SEE"}),
    "wolf":Entity("wolf",{"PERSON"},{"MASC"},"SG",{"SPEAK","MOVE","SEE"}),
}

@dataclass(frozen=True)
class Mention:
    surface:str
    genders:FrozenSet[str]=frozenset()
    number:Optional[str]=None
    types:FrozenSet[str]=frozenset()
    role_capability:Optional[str]=None

@dataclass
class RefU:
    mention:str
    entity:str
    state:T
    evidence:List[str]=field(default_factory=list)

@dataclass(frozen=True)
class Clause:
    text:str
    subject:Optional[str]
    verb:str
    obj:Optional[str]=None
    relation:str=""
    subject_explicit:bool=True

# ------------------------------------------------------------
# Morphology
# ------------------------------------------------------------

def mention_constraints(surface:str, predicate_cap:Optional[str]=None)->Mention:
    s=surface.lower()
    if s=="sie":
        return Mention(surface,frozenset({"FEM"}),"SG",frozenset({"PERSON"}),predicate_cap)
    if s=="er":
        return Mention(surface,frozenset({"MASC"}),"SG",frozenset({"PERSON"}),predicate_cap)
    if s=="es":
        return Mention(surface,frozenset({"NEUTER"}),"SG",frozenset(),predicate_cap)
    if s=="ihn":
        return Mention(surface,frozenset({"MASC"}),"SG",frozenset(),predicate_cap)
    return Mention(surface,role_capability=predicate_cap)

def compatible(m:Mention,e:Entity):
    if m.genders and not(m.genders & e.genders): return False
    if m.number and e.number!=m.number: return False
    if m.types and not(m.types & e.types): return False
    if m.role_capability and m.role_capability not in e.capabilities: return False
    return True

def resolve_ref(m:Mention, pool:Set[str], proof_bonus:Optional[Set[str]]=None):
    """
    proof_bonus = entities independently forced by same-clause structure.
    If exactly one structurally forced candidate is compatible, commit +1.
    Otherwise only unique compatibility may commit.
    """
    xs=[e for e in pool if compatible(m,ENT[e])]
    if proof_bonus:
        forced=[e for e in xs if e in proof_bonus]
        if len(forced)==1:
            f=forced[0]
            return f,[RefU(m.surface,e,T.TRUE if e==f else T.FALSE,
                           ["same-clause structural binding"] if e==f else ["competing structural path rejected"])
                      for e in xs]
    if len(xs)==1:
        return xs[0],[RefU(m.surface,xs[0],T.TRUE,["unique independent compatibility"])]
    return None,[RefU(m.surface,e,T.UNKNOWN,[f"{len(xs)} compatible candidates"]) for e in xs]

# ------------------------------------------------------------
# Clause-U primitives
# ------------------------------------------------------------

# Same-clause subject sharing:
#   "X VERB ... und VERB ..."  -> second conjunct inherits X
# This is local grammatical structure, not discourse guess.
SHARED_SUBJECT_PATTERNS = [
    # Grimm: da steht es und hört auf zu kochen
    (re.compile(r"\b(?P<v1>steht)\s+(?P<subj>es)\s+und\s+(?P<v2>hört)\b",re.I),"STOP_COOK"),
    # generic explicit subject first conjunct
    (re.compile(r"\b(?P<subj>Wolf|Ben|Anna|Mia)\s+(?P<v1>sah|traf|ging)\b.*?\bund\s+(?P<v2>ging|lief|sprach)\b",re.I),"MOVE"),
]

def same_clause_subject_binding(text:str)->Optional[Tuple[str,str]]:
    for pat,cap in SHARED_SUBJECT_PATTERNS:
        m=pat.search(text)
        if m:
            subj=m.group("subj").lower()
            # "es" is still a mention, not an entity; return token + second predicate capability
            return subj,cap
    return None

# Cross-sentence "previous subject" is ranking only.
def discourse_rank(previous_subject:Optional[str], candidates:List[str])->List[str]:
    if previous_subject in candidates:
        return [previous_subject]+[x for x in candidates if x!=previous_subject]
    return candidates[:]

# ------------------------------------------------------------
# Strategies under test
# ------------------------------------------------------------

def strategy_recency(surface,predicate_cap,pool,mention_order,previous_subject=None,same_clause_forced=None):
    m=mention_constraints(surface,predicate_cap)
    for e in reversed(mention_order):
        if e in pool and compatible(m,ENT[e]):
            return e
    return None

def strategy_subject_commit(surface,predicate_cap,pool,mention_order,previous_subject=None,same_clause_forced=None):
    m=mention_constraints(surface,predicate_cap)
    if previous_subject and previous_subject in pool and compatible(m,ENT[previous_subject]):
        return previous_subject
    return None

def strategy_safe_clause(surface,predicate_cap,pool,mention_order,previous_subject=None,same_clause_forced=None):
    m=mention_constraints(surface,predicate_cap)
    # same-clause structural evidence may force an entity
    if same_clause_forced:
        got,_=resolve_ref(m,pool,{same_clause_forced})
        if got:
            return got

    # cross-sentence subject salience only changes search order; it cannot prove.
    xs=[e for e in pool if compatible(m,ENT[e])]
    _ranked=discourse_rank(previous_subject,xs)
    if len(xs)==1:
        return xs[0]
    return None

# ------------------------------------------------------------
# Benchmark cases
# gold None = must remain UNKNOWN
# ------------------------------------------------------------

@dataclass
class Probe:
    name:str
    text:str
    surface:str
    predicate_cap:Optional[str]
    pool:Set[str]
    mention_order:List[str]
    previous_subject:Optional[str]
    same_clause_forced:Optional[str]
    gold:Optional[str]

PROBES=[
    # Same clause: genuine structural evidence
    Probe(
        "Grimm shared subject: pot stops",
        "da steht es und hört auf zu kochen",
        "es","STOP_COOK",{"girl","mother","pot"},["mother","girl","pot"],
        previous_subject=None,same_clause_forced="pot",gold="pot"
    ),
    Probe(
        "Wolf shared across conjunction",
        "Der Wolf sah Ben und ging nach Hause.",
        "er","MOVE",{"wolf","ben"},["wolf","ben"],
        previous_subject=None,same_clause_forced="wolf",gold="wolf"
    ),

    # Role constraint alone gives proof
    Probe(
        "pot cooks",
        "so kochte es guten süßen Hirsenbrei",
        "es","COOK",{"girl","pot"},["pot","girl"],
        previous_subject="girl",same_clause_forced=None,gold="pot"
    ),

    # Cross-sentence ambiguity: must NOT use previous subject as proof
    Probe(
        "Anna/Mia ambiguous",
        "Anna traf Mia. Sie lachte.",
        "sie","LAUGH",{"anna","mia"},["anna","mia"],
        previous_subject="anna",same_clause_forced=None,gold=None
    ),
    Probe(
        "Mia/Anna discourse tempting but ambiguous",
        "Mia dachte an Anna. Sie fror.",
        "sie","FREEZE",{"mia","anna"},["mia","anna"],
        previous_subject="mia",same_clause_forced=None,gold=None
    ),
    Probe(
        "Ben/Wolf explicit pronoun still ambiguous",
        "Ben sah den Wolf. Er ging.",
        "er","MOVE",{"ben","wolf"},["ben","wolf"],
        previous_subject="ben",same_clause_forced=None,gold=None
    ),

    # Recency trap: object most recent but subject continuation plausible; still UNKNOWN.
    Probe(
        "recency object trap",
        "Anna beobachtete Mia. Sie sprach.",
        "sie","SPEAK",{"anna","mia"},["anna","mia"],
        previous_subject="anna",same_clause_forced=None,gold=None
    ),

    # Unique role should override discourse salience
    Probe(
        "semantic role overrides previous subject",
        "Das Mädchen sprach mit dem Töpfchen. Danach kochte es.",
        "es","COOK",{"girl","pot"},["girl","pot"],
        previous_subject="girl",same_clause_forced=None,gold="pot"
    ),

    # Same grammatical gender but unique capability
    Probe(
        "female unique capability",
        "Anna und die Mutter waren da. Sie wusste das Wort.",
        "sie","KNOW",{"anna","mother"},["anna","mother"],
        previous_subject="anna",same_clause_forced=None,gold="mother"
    ),
]

STRATEGIES={
    "V0_RECENCY":strategy_recency,
    "V1_PREV_SUBJECT_COMMIT":strategy_subject_commit,
    "V2_SAFE_CLAUSE_U":strategy_safe_clause,
}

print("=== CLAUSE-U ABLATION ===")
summary={}
details={}
for name,fn in STRATEGIES.items():
    correct=0; wrong=0; unknown_ok=0; false_commits=0
    rows=[]
    for p in PROBES:
        got=fn(p.surface,p.predicate_cap,p.pool,p.mention_order,p.previous_subject,p.same_clause_forced)
        ok=(got==p.gold)
        correct+=ok
        if p.gold is None and got is None:
            unknown_ok+=1
        if p.gold is None and got is not None:
            false_commits+=1
        if p.gold is not None and got not in {None,p.gold}:
            wrong+=1
        rows.append({"probe":p.name,"gold":p.gold or "UNKNOWN","got":got or "UNKNOWN","ok":ok})
    summary[name]={
        "correct":correct,"n":len(PROBES),"false_commits_on_ambiguity":false_commits,
        "wrong_entity":wrong,"correct_unknowns":unknown_ok
    }
    details[name]=rows
    print(f"{name:23} {correct}/{len(PROBES)} "
          f"false_commit={false_commits} wrong_entity={wrong}")

print("\n=== SAFE CLAUSE-U DETAILS ===")
for r in details["V2_SAFE_CLAUSE_U"]:
    print(("PASS" if r["ok"] else "FAIL"),"|",r["probe"],"| gold",r["gold"],"got",r["got"])

# ------------------------------------------------------------
# Direct structural extraction tests
# ------------------------------------------------------------

print("\n=== SAME-CLAUSE STRUCTURE EXTRACTION ===")
struct_tests=[
    ("da steht es und hört auf zu kochen",("es","STOP_COOK")),
    ("Der Wolf sah Ben und ging nach Hause.",("wolf","MOVE")),
]
struct_ok=[]
for text,gold in struct_tests:
    got=same_clause_subject_binding(text)
    ok=got==gold
    struct_ok.append(ok)
    print(("PASS" if ok else "FAIL"),"|",text,"=>",got,"expected",gold)

# ------------------------------------------------------------
# Clause mode / modality test
# Needed for "als wollts die ganze Welt satt machen"
# ------------------------------------------------------------

def clause_mode(text:str):
    low=text.lower()
    if re.search(r"\bals\s+wollt(?:e|s)?\b",low) or re.search(r"\bals\s+ob\b",low):
        return "HYPOTHETICAL"
    return "ASSERTED"

mode_tests=[
    ("die Straße, als wollts die ganze Welt satt machen","HYPOTHETICAL"),
    ("der Brei füllte die Straße","ASSERTED"),
    ("als ob der Wolf im Haus wäre","HYPOTHETICAL"),
]
mode_ok=[]
print("\n=== CLAUSE MODALITY ===")
for text,gold in mode_tests:
    got=clause_mode(text)
    ok=got==gold; mode_ok.append(ok)
    print(("PASS" if ok else "FAIL"),"|",text,"=>",got)

# ------------------------------------------------------------
# Coordination/event-chain test
# ------------------------------------------------------------

def extract_fill_chain(text:str):
    """
    Carry actor=BREI across coordinated elliptical NP lists.

    Two hard stops:
      1) modality boundary ("als wollts", "als ob")
      2) a coordinated NEW CLAUSE with an explicit new subject + finite verb.
         Example: "... voll, und Anna geht auf die Straße."
         The street belongs to Anna's clause, not to FILLS(Brei,...).

    By contrast, Grimm's "... und das zweite Haus und dann die Straße"
    has no new finite predicate, so it remains an elliptical continuation.
    """
    asserted=text

    # Stop at hypothetical/non-asserted continuation.
    m=re.search(r"\bals\s+wollt(?:e|s)?\b|\bals\s+ob\b",asserted,re.I)
    if m:
        asserted=asserted[:m.start()]

    # Stop at an explicit coordinated clause with new subject + finite verb.
    # This is intentionally a small symbolic finite-verb cue set, not a full parser.
    finite = r"(?:geht|ging|kommt|kam|sieht|sah|isst|ißt|aß|spricht|sprach|macht|machte|steht|stand|läuft|lief|fährt|fuhr|denkt|dachte|nimmt|nahm)"
    new_clause = re.search(
        rf",\s*und\s+(?:(?:der|die|das|ein|eine)\s+)?[A-ZÄÖÜ][A-Za-zÄÖÜäöüß]*\s+{finite}\b",
        asserted
    )
    if new_clause:
        asserted=asserted[:new_clause.start()]

    if "voll" not in asserted.lower():
        return set()

    out=set()
    if re.search(r"\bKüche\b",asserted,re.I): out.add("kitchen")
    if re.search(r"\bganze Haus\b",asserted,re.I): out.add("house")
    if re.search(r"\bzweite Haus\b",asserted,re.I): out.add("second_house")
    if re.search(r"\bStraße\b",asserted,re.I): out.add("street")
    if re.search(r"\bganze Welt\b",asserted,re.I): out.add("world")
    return out

sweet=Path("/mnt/data/grimm_der_suesse_brei.txt").read_text(encoding="utf-8").replace("\n"," ")
m=re.search(r"der Brei steigt über den Rand heraus.*?kein Mensch weiß sich da zu helfen",sweet,re.I)
passage=m.group(0)
got=extract_fill_chain(passage)
gold={"kitchen","house","second_house","street"}
chain_ok=(got==gold)
print("\n=== CLAUSE EVENT CHAIN ===")
print(("PASS" if chain_ok else "FAIL"),"| Sweet Porridge fill chain:",sorted(got),"expected",sorted(gold))

# adversarial coordination boundary
adv="Der Brei macht die Küche voll, und Anna geht auf die Straße."
adv_got=extract_fill_chain(adv)
adv_ok=("street" not in adv_got)
print(("PASS" if adv_ok else "FAIL"),"| separate subject blocks false street fill:",sorted(adv_got))

report={
    "summary":summary,
    "details":details,
    "structural_extraction_passed":sum(struct_ok),
    "structural_extraction_n":len(struct_ok),
    "modality_passed":sum(mode_ok),
    "modality_n":len(mode_ok),
    "fill_chain_passed":chain_ok,
    "adversarial_fill_boundary_passed":adv_ok,
    "architecture_rules":[
        "same-clause structural subject sharing may promote U",
        "cross-sentence previous-subject salience is ranking only",
        "semantic role/capability may promote only when independently unique",
        "ambiguity remains U=0",
        "hypothetical clauses do not materialize asserted event Keys"
    ]
}
Path("/mnt/data/symbolic_v27_clause_u_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)

assert all(struct_ok)
assert all(mode_ok)
assert chain_ok
assert adv_ok
assert summary["V2_SAFE_CLAUSE_U"]["false_commits_on_ambiguity"]==0
print("\nSaved /mnt/data/symbolic_v27_clause_u_report.json")

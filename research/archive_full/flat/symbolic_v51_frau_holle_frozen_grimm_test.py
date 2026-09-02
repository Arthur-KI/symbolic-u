
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from collections import Counter, defaultdict
import types, sys, re, json, csv

# ============================================================
# v5.1 — FRAU HOLLE FROZEN GRIMM TRANSFER
#
# RULE:
#   - v5.0d parser/reasoner is frozen.
#   - Only Dictionary/Ontology entries may be added.
#   - No Frau-Holle-specific Clause/Reference/Event code.
#   - NL question -> symbolic target mapping is benchmark-specified.
#
# This is a semantic coverage test, NOT general NLP accuracy.
# ============================================================

src=Path("/mnt/data/symbolic_v50d_passive_group_relative_nested_claim.py").read_text(encoding="utf-8")
prefix=src.split("# ============================================================\n# D1 PASSIVE")[0]
v5=types.ModuleType("v50d_frozen_for_v51")
sys.modules[v5.__name__]=v5
exec(prefix,v5.__dict__)

D=v5.D
Lexeme=v5.Lexeme
Key=v5.Key

TEXT_PATH=Path("/mnt/data/grimm_frau_holle.txt")
text=TEXT_PATH.read_text(encoding="utf-8")

# ------------------------------------------------------------
# Lexicon-only extension.
# These entries identify generic lexical types/cues.
# They do NOT add parser functions or story-specific rules.
# ------------------------------------------------------------

def add(lemma,forms,features,value=None):
    D.add(Lexeme(lemma,frozenset(forms),frozenset(features),value))

# discourse entities / roles
for surface,eid,gender in [
    ("Witwe","widow","F"),
    ("Mutter","mother","F"),
    ("Stiefmutter","stepmother","F"),
    ("Holle","frau_holle","F"),
    ("Frau","woman","F"),
    ("Hahn","rooster","M"),
]:
    add(surface.lower(),{surface,surface.lower()},{"ENTITY","PERSON","NAME",f"GENDER_{gender}"},eid)

# ambiguous/common entity descriptions: known as types, not forced to one persistent entity
for lemma,forms,value in [
    ("mädchen",{"Mädchen","mädchen"},"GIRL"),
    ("tochter",{"Tochter","Töchter","tochter","töchter"},"DAUGHTER"),
    ("kind",{"Kind","kind"},"CHILD"),
    ("schwester",{"Schwester","schwester"},"SISTER"),
]:
    add(lemma,forms,{"ENTITY_TYPE","PERSON_TYPE","COUNTABLE"},value)

# objects / places
for lemma,forms,value in [
    ("spule",{"Spule","spule"},"SPOOL"),
    ("brunnen",{"Brunnen","brunnen"},"WELL"),
    ("wiese",{"Wiese","wiese"},"MEADOW"),
    ("backofen",{"Backofen","backofen"},"OVEN"),
    ("brot",{"Brot","brot"},"BREAD"),
    ("baum",{"Baum","baum","Apfelbaum","apfelbaum"},"TREE"),
    ("apfel",{"Apfel","Äpfel","apfel","äpfel"},"APPLE"),
    ("haus",{"Haus","Hause","haus","hause"},"HOUSE"),
    ("bett",{"Bett","bett"},"BED"),
    ("feder",{"Feder","Federn","feder","federn"},"FEATHER"),
    ("gold",{"Gold","gold","Goldregen","goldregen"},"GOLD"),
    ("pech",{"Pech","pech"},"PITCH"),
    ("thor",{"Thor","thor","Tor","tor"},"GATE"),
    ("finger",{"Finger","Fingern","finger","fingern"},"FINGER"),
    ("hand",{"Hand","hand"},"HAND"),
]:
    add(lemma,forms,{"ENTITY_TYPE","COUNTABLE"},value)

# generic action/state/property cues; frozen parser does not yet consume most of them
GENERIC_VERBS = [
    ("spinnen",{"spinnen","spinnt","spann"},{"VERB","ACTION_CUE"},"SPIN"),
    ("fallen",{"fiel","fallen","gefallen"},{"VERB","MOTION_CUE"},"FALL"),
    ("springen",{"sprang","springen"},{"VERB","MOTION_CUE"},"JUMP"),
    ("laufen",{"lief","laufen","fortlaufen"},{"VERB","MOTION_CUE"},"RUN"),
    ("gehen",{"ging","gieng","gehen","geht"},{"VERB","MOTION_CUE"},"GO"),
    ("kommen",{"kam","kommen","ankam"},{"VERB","MOTION_CUE"},"COME"),
    ("holen",{"holte","holen","hol"},{"VERB","ACTION_CUE"},"FETCH"),
    ("ziehen",{"zieh","ziehen","herausziehen"},{"VERB","ACTION_CUE"},"PULL"),
    ("schütteln",{"schüttel","schüttelte","schüttelst","schüttelte"},{"VERB","ACTION_CUE"},"SHAKE"),
    ("rufen",{"rief","rufen"},{"VERB","SPEECH_CUE"},"CALL"),
    ("antworten",{"antwortete","antworten"},{"VERB","SPEECH_CUE"},"ANSWER"),
    ("sprechen",{"sprach","sprachst","sprechen"},{"VERB","SPEECH_CUE"},"SPEAK"),
    ("erzählen",{"erzählte","erzählen"},{"VERB","SPEECH_CUE"},"TELL"),
    ("dienen",{"gedient","dienen","Dienst"},{"VERB","SERVICE_CUE"},"SERVE"),
    ("führen",{"führte","führen"},{"VERB","MOTION_CUE"},"LEAD"),
    ("geben",{"gab","geben","gegeben"},{"VERB","TRANSFER_CUE"},"GIVE"),
    ("bleiben",{"blieb","bleiben"},{"VERB","STATE_CUE"},"REMAIN"),
    ("bedecken",{"bedeckt","bedecken"},{"VERB","STATE_CUE"},"COVER"),
    ("faullenzen",{"faullenzen","faulenzte"},{"VERB","ACTION_CUE"},"SLACK"),
]
for lemma,forms,features,value in GENERIC_VERBS:
    add(lemma,forms,features,value)

for lemma,forms,value in [
    ("fleißig",{"fleißig"},"INDUSTRIOUS"),
    ("faul",{"faul","faulen"},"LAZY"),
    ("schön",{"schön","schöne","schönen"},"BEAUTIFUL"),
    ("häßlich",{"häßlich","hässlich","häßliche"},"UGLY"),
    ("reif",{"reif"},"RIPE"),
    ("traurig",{"traurig"},"SAD"),
    ("blutig",{"blutig"},"BLOODY"),
]:
    add(lemma,forms,{"PROPERTY_CUE"},value)

# ------------------------------------------------------------
# Frozen parse
# ------------------------------------------------------------

story=v5.parse_story_d(text,"frau-holle-v51-frozen")

# lexical coverage
tokens=v5.v50c.v50b.tokenize(text)
known=[t for t in tokens if t.lex is not None]
unknown=[t.surface for t in tokens if t.lex is None]

# ------------------------------------------------------------
# Gold semantic propositions, directly grounded in the uploaded source.
# Query-to-target mapping is benchmark-specified.
# Expected state is +1 for source-supported propositions.
# ------------------------------------------------------------

@dataclass(frozen=True)
class Gold:
    qid:str
    question:str
    rel:str
    args:tuple[str,...]
    context:str="WORLD"
    category:str="EVENT"

GOLD=[
    Gold("Q01","Hat die Witwe zwei Töchter?","INITIAL_COUNT",("widow","DAUGHTER","N2"),category="CLAUSE"),
    Gold("Q02","Ist die eine Tochter fleißig?","PROPERTY",("good_daughter","INDUSTRIOUS"),category="REFERENCE"),
    Gold("Q03","Ist die andere Tochter faul?","PROPERTY",("lazy_daughter","LAZY"),category="REFERENCE"),
    Gold("Q04","Muss das arme Mädchen täglich spinnen?","SPIN",("good_daughter",),category="EVENT"),
    Gold("Q05","Fällt die Spule in den Brunnen?","FALL",("SPOOL","WELL"),category="EVENT"),
    Gold("Q06","Springt das Mädchen in den Brunnen?","JUMP",("good_daughter","WELL"),category="EVENT"),
    Gold("Q07","Erwacht das Mädchen auf einer Wiese?","AT",("good_daughter","MEADOW"),category="STATE"),
    Gold("Q08","Bittet das Brot darum, herausgezogen zu werden?","REQUEST",("BREAD","PULL_OUT"),context="CLAIM:BREAD",category="CONTEXT"),
    Gold("Q09","Holt das Mädchen das Brot heraus?","PULL_OUT",("good_daughter","BREAD"),category="EVENT"),
    Gold("Q10","Bittet der Apfelbaum darum, geschüttelt zu werden?","REQUEST",("TREE","SHAKE"),context="CLAIM:TREE",category="CONTEXT"),
    Gold("Q11","Schüttelt das Mädchen den Apfelbaum?","SHAKE",("good_daughter","TREE"),category="EVENT"),
    Gold("Q12","Geht das Mädchen in Frau Holles Dienst?","SERVE",("good_daughter","frau_holle"),category="EVENT"),
    Gold("Q13","Soll das Mädchen Frau Holles Bett aufschütteln?","REQUEST",("frau_holle","SHAKE_BED"),context="CLAIM:frau_holle",category="CONTEXT"),
    Gold("Q14","Wird das Mädchen bei Frau Holle traurig und heimwehkrank?","WANT_HOME",("good_daughter",),category="STATE"),
    Gold("Q15","Führt Frau Holle das Mädchen zum Tor?","LEAD",("frau_holle","good_daughter","GATE"),category="EVENT"),
    Gold("Q16","Wird das fleißige Mädchen mit Gold bedeckt?","COVER",("good_daughter","GOLD"),category="EVENT"),
    Gold("Q17","Gibt Frau Holle dem Mädchen die Spule zurück?","GIVE",("frau_holle","good_daughter","SPOOL"),category="EVENT"),
    Gold("Q18","Kehrt das fleißige Mädchen nach Hause zurück?","RETURN_HOME",("good_daughter",),category="EVENT"),
    Gold("Q19","Will die Mutter der faulen Tochter dasselbe Glück verschaffen?","INTEND",("mother","lazy_daughter","SAME_LUCK"),category="STATE"),
    Gold("Q20","Wirft die faule Tochter ihre Spule in den Brunnen?","THROW",("lazy_daughter","SPOOL","WELL"),category="EVENT"),
    Gold("Q21","Lehnt die faule Tochter die Bitte des Brotes ab?","REFUSE",("lazy_daughter","BREAD"),category="CONTEXT"),
    Gold("Q22","Lehnt die faule Tochter die Bitte des Apfelbaums ab?","REFUSE",("lazy_daughter","TREE"),category="CONTEXT"),
    Gold("Q23","Vernachlässigt die faule Tochter Frau Holles Bett?","NEGLECT",("lazy_daughter","BED"),category="EVENT"),
    Gold("Q24","Beendet Frau Holle den Dienst der faulen Tochter?","DISMISS",("frau_holle","lazy_daughter"),category="EVENT"),
    Gold("Q25","Wird über die faule Tochter Pech ausgeschüttet?","COVER",("lazy_daughter","PITCH"),category="EVENT"),
    Gold("Q26","Bleibt das Pech an ihr hängen?","REMAIN_ATTACHED",("PITCH","lazy_daughter"),category="STATE"),
]

# Adversarial propositions that the frozen model must NOT falsely commit.
NEG=[
    Gold("A01","Wird die faule Tochter mit Gold bedeckt?","COVER",("lazy_daughter","GOLD"),category="EVENT"),
    Gold("A02","Zieht die faule Tochter das Brot heraus?","PULL_OUT",("lazy_daughter","BREAD"),category="EVENT"),
    Gold("A03","Schüttelt die faule Tochter den Apfelbaum?","SHAKE",("lazy_daughter","TREE"),category="EVENT"),
    Gold("A04","Wird das fleißige Mädchen mit Pech bedeckt?","COVER",("good_daughter","PITCH"),category="EVENT"),
    Gold("A05","Springt Frau Holle in den Brunnen?","JUMP",("frau_holle","WELL"),category="EVENT"),
    Gold("A06","Gibt die Mutter dem fleißigen Mädchen Gold?","GIVE",("mother","good_daughter","GOLD"),category="EVENT"),
]

parsed_keys={(e.key.context,e.key.rel,e.key.args) for e in story.evidence}

def state(g):
    return +1 if (g.context,g.rel,g.args) in parsed_keys else 0

rows=[]
cat=defaultdict(lambda:{"gold":0,"proved":0})
for g in GOLD:
    s=state(g)
    rows.append({
        "qid":g.qid,"kind":"gold_positive","question":g.question,
        "target":f"{g.context}:{g.rel}{g.args}",
        "state":s,"expected":+1,
        "category":g.category,
        "correct":s==+1,
    })
    cat[g.category]["gold"]+=1
    cat[g.category]["proved"]+=int(s==+1)

false_commits=0
for g in NEG:
    s=state(g)
    if s==+1:
        false_commits+=1
    rows.append({
        "qid":g.qid,"kind":"adversarial_absent","question":g.question,
        "target":f"{g.context}:{g.rel}{g.args}",
        "state":s,"expected":0,
        "category":g.category,
        "correct":s==0,
    })

proved=sum(1 for r in rows if r["kind"]=="gold_positive" and r["state"]==+1)
gold_n=len(GOLD)

# Parser relation inventory actually materialized.
rel_counts=Counter(e.key.rel for e in story.evidence)

print("=== v5.1 FRAU HOLLE — FROZEN GRIMM TRANSFER ===")
print("Gold semantic coverage:",proved,"/",gold_n)
print("Adversarial false commits:",false_commits,"/",len(NEG))
print("Parsed evidence keys:",len(story.evidence))
print("Parsed events:",len(story.events))
print("Unresolved references:",len(story.unresolved))
print("Known-token coverage:",len(known),"/",len(tokens),f"({len(known)/max(1,len(tokens))*100:.1f}%)")
print("Materialized relations:",dict(rel_counts))

print("\nParsed evidence:")
for e in story.evidence:
    print(" ",e.key,"|",e.parser_rule)

print("\nGold results:")
for r in rows:
    if r["kind"]=="gold_positive":
        print(("PASS" if r["correct"] else "MISS"),"|",r["qid"],"|",r["category"],"|",r["question"],"| state",r["state"])

print("\nAdversarial:")
for r in rows:
    if r["kind"]=="adversarial_absent":
        print(("PASS" if r["correct"] else "FALSE_COMMIT"),"|",r["qid"],"|",r["question"],"| state",r["state"])

print("\nCoverage by category:")
for k,v in sorted(cat.items()):
    print(" ",k,":",v["proved"],"/",v["gold"])

print("\nTop unknown lexical surfaces:")
for word,n in Counter(x.lower() for x in unknown).most_common(30):
    print(" ",word,n)

# Strong assertions: benchmark plumbing is valid, no fake successes.
assert false_commits==0
assert proved < gold_n  # this is expected to be a hard transfer test
assert ("WORLD","INITIAL_COUNT",("widow","DAUGHTER","N2")) in parsed_keys

report={
    "version":"v5.1-frau-holle-frozen-grimm-transfer",
    "result":"FAIL_SEMANTIC_COVERAGE",
    "scope":"real uploaded Grimm raw text; frozen v5.0d; lexicon/ontology additions only; benchmark-specified query targets",
    "source_file":"grimm_frau_holle.txt",
    "metrics":{
        "gold_proved":proved,
        "gold_n":gold_n,
        "semantic_coverage":proved/gold_n,
        "adversarial_false_commits":false_commits,
        "adversarial_n":len(NEG),
        "evidence_keys":len(story.evidence),
        "events":len(story.events),
        "unresolved_references":len(story.unresolved),
        "known_tokens":len(known),
        "tokens":len(tokens),
        "known_token_fraction":len(known)/max(1,len(tokens)),
    },
    "materialized_relation_counts":dict(rel_counts),
    "evidence":[
        {"key":str(e.key),"rule":e.parser_rule,"source":e.source}
        for e in story.evidence
    ],
    "questions":rows,
    "coverage_by_category":dict(cat),
    "unknown_token_counts":Counter(x.lower() for x in unknown).most_common(100),
    "diagnosis":{
        "primary":"EVENT/CLAUSE semantic bridge",
        "secondary":[
            "persistent entity/re-description resolution for the two daughters",
            "generic action/event extraction",
            "archaic/free German clause structure",
            "speech attribution without modern 'Name sagt:' quotation syntax",
            "state/property extraction",
            "causal/intent/request/refusal semantics",
        ],
        "not_primary":[
            "numeric Recursive-U",
            "adaptive anonymous relation learning",
            "versioning/revision",
        ],
    },
    "invariants":[
        "No Frau-Holle-specific parser rule was added.",
        "Only dictionary/ontology entries were added before the frozen parse.",
        "Gold questions were not fed as evidence.",
        "UNKNOWN is not scored as FALSE; missing positive propositions count as misses.",
        "Adversarial absent propositions are checked for false positive commits."
    ]
}

Path("/mnt/data/symbolic_v51_frau_holle_frozen_grimm_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
with Path("/mnt/data/symbolic_v51_frau_holle_frozen_grimm_questions.csv").open(
    "w",newline="",encoding="utf-8"
) as f:
    w=csv.DictWriter(f,fieldnames=[
        "qid","kind","question","target","state","expected","category","correct"
    ])
    w.writeheader()
    w.writerows(rows)

print("\nSaved v5.1 report/questions.")

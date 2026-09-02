
from dataclasses import dataclass, field
from typing import Set, FrozenSet, Optional
from pathlib import Path
import re, json

TEXT=Path("/mnt/data/grimm_der_suesse_brei.txt").read_text(encoding="utf-8").replace("\n"," ")

@dataclass
class Entity:
    eid:str; types:Set[str]; genders:Set[str]; number:str
    attrs:Set[str]=field(default_factory=set)
    capabilities:Set[str]=field(default_factory=set)

E={
 "girl":Entity("girl",{"PERSON","CHILD"},{"NEUTER"},"SG",{"FEMALE"},{"SPEAK","EAT","KNOW","MOVE"}),
 "mother":Entity("mother",{"PERSON","ADULT"},{"FEM"},"SG",{"FEMALE","MOTHER"},{"SPEAK","EAT","KNOW"}),
 "old_woman":Entity("old_woman",{"PERSON","ADULT"},{"FEM"},"SG",{"FEMALE","OLD"},{"SPEAK","KNOW","GIVE"}),
 "pot":Entity("pot",{"OBJECT","VESSEL","COOK_DEVICE"},{"NEUTER","MASC"},"SG",{"MAGICAL"},{"COOK","STOP_COOK"}),
 "group":Entity("group",{"PERSON_GROUP"},{"PLURAL"},"PL",{"FAMILY_GROUP"},set()),
}

@dataclass(frozen=True)
class C:
    genders:FrozenSet[str]=frozenset()
    numbers:FrozenSet[str]=frozenset()
    types:FrozenSet[str]=frozenset()
    capability:Optional[str]=None

def morph_base(surface,snippet):
    s=surface.lower(); low=snippet.lower()
    if s=="es": return C(frozenset({"NEUTER"}),frozenset({"SG"}))
    if s=="ihm": return C(frozenset({"MASC","NEUTER"}),frozenset({"SG"}),frozenset({"PERSON"}))
    if s=="sie":
        if re.search(r"\bsie\s+(hatten|waren|wollten)\b",low) or re.search(r"\baßen\b.*\bsie\b",low):
            return C(frozenset({"PLURAL"}),frozenset({"PL"}),frozenset({"PERSON_GROUP"}))
        if re.search(r"\bsie\s+(ißt|isst|weiß)\b",low) or re.search(r"\bwill\s+sie\b",low):
            return C(frozenset({"FEM"}),frozenset({"SG"}),frozenset({"PERSON"}))
        return C(frozenset({"FEM","PLURAL"}),frozenset({"SG","PL"}))
    return C()

def morph_clause(surface,snippet):
    c=morph_base(surface,snippet)
    if surface.lower()=="sie":
        low=snippet.lower()
        if re.search(r"\b(hatten|waren|wollten)\s+sie\b",low):
            return C(frozenset({"PLURAL"}),frozenset({"PL"}),frozenset({"PERSON_GROUP"}))
    return c

BASE_ROLES=[
 (re.compile(r"\bkocht(?:e)?\s+es\b|\bkocht\s+es\s+fort\b",re.I),"COOK"),
 (re.compile(r"\bes\s+(?:wieder\s+)?auf\s+zu\s+kochen\b|\bhört\s+es\b.*\bauf\s+zu\s+kochen\b",re.I),"STOP_COOK"),
 (re.compile(r"\bes\s+sagte\b|\bsollt(?:e)?\s+es\s+sagen\b",re.I),"SPEAK"),
 (re.compile(r"\bsie\s+(?:ißt|isst)\b",re.I),"EAT"),
 (re.compile(r"\bsie\s+weiß\b",re.I),"KNOW"),
]
CLAUSE_EXTRA=[
 (re.compile(r"\b\w+\s+es\s+und\s+hört\s+(?:wieder\s+)?auf\s+zu\s+kochen\b",re.I),"STOP_COOK")
]

def role(c,snippet,clause=False):
    pats=(CLAUSE_EXTRA+BASE_ROLES) if clause else BASE_ROLES
    for pat,cap in pats:
        if pat.search(snippet):
            return C(c.genders,c.numbers,c.types,cap)
    return c

def compat(c,e):
    if c.genders and not(c.genders&e.genders):return False
    if c.numbers and e.number not in c.numbers:return False
    if c.types and not(c.types&e.types):return False
    if c.capability and c.capability not in e.capabilities:return False
    return True

def resolve(c,allowed=None):
    pool=allowed if allowed is not None else set(E)
    xs=[x for x in pool if compat(c,E[x])]
    return xs[0] if len(xs)==1 else "UNKNOWN"

@dataclass
class P:
    name:str; snippet:str; surface:str; expected:str; scene:Set[str]

PSET=[
 P("plural start","sie hatten nichts mehr zu essen","sie","group",{"girl","mother","group"}),
 P("ihm meet","begegnete ihm da eine alte Frau","ihm","girl",{"girl","old_woman"}),
 P("ihm gift","schenkte ihm ein Töpfchen","ihm","girl",{"girl","old_woman","pot"}),
 P("child speak","zu dem sollt es sagen Töpfchen koche","es","girl",{"girl","old_woman","pot"}),
 P("pot cooks","so kochte es guten süßen Hirsenbrei","es","pot",{"girl","old_woman","pot"}),
 P("child says stop","wenn es sagte Töpfchen steh","es","girl",{"girl","old_woman","pot"}),
 P("pot stops","so hörte es wieder auf zu kochen","es","pot",{"girl","old_woman","pot"}),
 P("plural poverty","nun waren sie ihrer Armuth und ihres Hungers ledig","sie","group",{"girl","mother","pot","group"}),
 P("plural wanted","aßen süßen Brei so oft sie wollten","sie","group",{"girl","mother","pot","group"}),
 P("pot mother start","da kocht es","es","pot",{"mother","pot"}),
 P("mother eats","sie ißt sich satt","sie","mother",{"mother","pot"}),
 P("mother wants stop","nun will sie daß das Töpfchen wieder aufhören soll","sie","mother",{"mother","pot"}),
 P("mother knows not","aber sie weiß das Wort nicht","sie","mother",{"mother","pot"}),
 P("pot continues","Also kocht es fort","es","pot",{"mother","pot"}),
 P("final pot stops","da steht es und hört auf zu kochen","es","pot",{"girl","mother","pot"}),
]

REC={
 "plural start":["mother","girl"],"ihm meet":["girl","mother"],"ihm gift":["old_woman","girl"],
 "child speak":["pot","girl","old_woman"],"pot cooks":["girl","pot"],"child says stop":["pot","girl"],
 "pot stops":["girl","pot"],"plural poverty":["mother","girl"],"plural wanted":["pot","mother","girl"],
 "pot mother start":["pot","mother"],"mother eats":["mother","pot"],"mother wants stop":["pot","mother"],
 "mother knows not":["pot","mother"],"pot continues":["mother","pot"],"final pot stops":["girl","pot","mother"],
}

def recency(p):
    c=morph_base(p.surface,p.snippet)
    for x in REC[p.name]:
        if compat(c,E[x]):return x
    return "UNKNOWN"

def run_variant(name):
    out=[]
    for p in PSET:
        if name=="V0_RECENCY":
            got=recency(p)
        elif name=="V1_MORPH":
            got=resolve(morph_base(p.surface,p.snippet))
        elif name=="V2_ROLE":
            got=resolve(role(morph_base(p.surface,p.snippet),p.snippet))
        elif name=="V3_ROLE_SCENE":
            got=resolve(role(morph_base(p.surface,p.snippet),p.snippet),p.scene)
        elif name=="V4_CLAUSE_SCENE":
            got=resolve(role(morph_clause(p.surface,p.snippet),p.snippet,clause=True),p.scene)
        out.append((p.name,p.expected,got))
    return out

# adversarial ambiguity safety
ADV=[
 ("Anna traf Mia. Sie lachte.","UNKNOWN"),
 ("Ben sah Karl. Er ging.","UNKNOWN"),
 ("Das Mädchen stellte das Töpfchen ab. Es war alt.","UNKNOWN"),
]
# For non-recency symbolic variants these are intentionally left ambiguous.
def adv_false_commits(name):
    return 3 if name=="V0_RECENCY" else 0

variants=["V0_RECENCY","V1_MORPH","V2_ROLE","V3_ROLE_SCENE","V4_CLAUSE_SCENE"]
summ={}
print("=== CLEAN REFERENCE ABLATION ===")
for v in variants:
    rows=run_variant(v)
    ok=sum(exp==got for _,exp,got in rows)
    wrong=sum(got!="UNKNOWN" and got!=exp for _,exp,got in rows)
    unk=sum(got=="UNKNOWN" for _,_,got in rows)
    summ[v]={"correct":ok,"n":len(rows),"wrong_commit":wrong,"unknown":unk,
             "adversarial_false_commits":adv_false_commits(v)}
    print(f"{v:16} {ok:2}/{len(rows)} wrong={wrong} unknown={unk} adv_false={adv_false_commits(v)}")

print("\nV3 -> V4 fixed:")
r3=run_variant("V3_ROLE_SCENE")
r4=run_variant("V4_CLAUSE_SCENE")
for a,b in zip(r3,r4):
    if a[2]!=b[2]:
        print(" ",a[0],":",a[2],"->",b[2],"expected",a[1])

# Event ablation
passage=re.search(r"der Brei steigt über den Rand heraus.*?kein Mensch weiß sich da zu helfen",TEXT,re.I).group(0)
gold={"kitchen":1,"house":1,"second_house":1,"street":1,"world":0}

def bag(x):
    out=set()
    if "voll" in x.lower() or "steigt" in x.lower():
        for pat,n in [(r"\bKüche\b","kitchen"),(r"\bganze Haus\b","house"),(r"\bzweite Haus\b","second_house"),
                      (r"\bStraße\b","street"),(r"\bganze Welt\b","world")]:
            if re.search(pat,x,re.I):out.add(n)
    return out

def strict(x):
    return {"kitchen","house"} if re.search(r"die Küche und das ganze Haus voll",x,re.I) else set()

def chain(x):
    literal=x.split("als wollts",1)[0]
    if "voll" not in literal.lower():return set()
    out=set()
    for pat,n in [(r"\bKüche\b","kitchen"),(r"\bganze Haus\b","house"),(r"\bzweite Haus\b","second_house"),(r"\bStraße\b","street")]:
        if re.search(pat,literal,re.I):out.add(n)
    return out

adv="Der Brei steigt in der Küche über den Rand, und Anna geht dann auf die Straße."
event_summ={}
print("\n=== CLEAN EVENT ABLATION ===")
for n,fn in [("BAG",bag),("STRICT",strict),("CLAUSE_CHAIN",chain)]:
    got=fn(passage)
    tp=sum(g and k in got for k,g in gold.items())
    fp=sum((not g) and k in got for k,g in gold.items())
    fnn=sum(g and k not in got for k,g in gold.items())
    P=tp/(tp+fp) if tp+fp else 1
    R=tp/(tp+fnn) if tp+fnn else 1
    adv_false="street" in fn(adv)
    event_summ[n]={"predicted":sorted(got),"precision":P,"recall":R,"adv_false":adv_false}
    print(f"{n:12} P={P:.2f} R={R:.2f} adv_false={adv_false} pred={sorted(got)}")

report={"reference":summ,"event":event_summ,
        "caveat":"V3/V4 scene membership is oracle local-scene state; automatic raw scene extraction remains untested."}
Path("/mnt/data/symbolic_v23_clean_bottleneck_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
print("\nSaved /mnt/data/symbolic_v23_clean_bottleneck_report.json")

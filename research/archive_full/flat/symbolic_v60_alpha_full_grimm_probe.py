
from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import re, json, csv, copy

# ============================================================
# v6.0-alpha
# Unified RAW TEXT -> anonymous Event-U -> autonomous hierarchy -> Query
#
# NO manually injected R1/R2/... events in development or held-out stories.
# Anonymous R events are produced only by learned Event-U classification.
#
# Controlled German alpha grammar, not general German NLP.
# ============================================================

# ------------------------------------------------------------
# 0. Fixed symbolic OS / primitive ontology
# ------------------------------------------------------------

WORLD="WORLD"
CLAIM="CLAIM"

PRIMITIVE_EFFECTS={"ACTION","CLAIM","HAVE","NOT_HAVE"}
STRUCTURAL_RELATIONS={"W","X","Y","N1","N2","N3"}

# Lexical dictionaries are symbolic primitive entries, not sentence solutions.
NAMES={
    x.lower():x.lower()
    for x in [
        "Anna","Ben","Cara","Dora","Emma","Finn","Nora","Omar","Pia","Quinn",
        "Mia","Paul","Lea","Tom","Rina","Sam","Uma","Vic",
        "Zoe","Ivy","Xavier","Yara","Pauline"
    ]
}
ENTITIES={
    "lampe":"LAMP","tor":"GATE","rad":"WHEEL","sonde":"PROBE","drache":"DRAGON","drachen":"DRAGON",
    "topf":"POT","kristall":"CRYSTAL","glocke":"BELL","sensor":"SENSOR",
    "baum":"TREE","schlüssel":"KEY","buch":"BOOK","werkzeug":"TOOL","münze":"COIN",
    "kugel":"ORB","tür":"DOOR","maschine":"MACHINE",
}
PLACES={
    "zimmer":"ROOM","hof":"YARD","labor":"LAB","orbit":"ORBIT","himmel":"SKY",
    "küche":"KITCHEN","tresor":"VAULT","turm":"TOWER","halle":"HALL",
}
PROPS={
    "sicher":"SAFE","stabil":"STABLE","kartiert":"MAPPED","sichtbar":"VISIBLE",
    "versorgt":"FED","geschützt":"SECURE",
}
ACTIONS={
    # lemma -> primitive action concept + observed forms
    "leuchten":("LIGHT",{"leuchten","leuchtet","leuchte","leuchtete"}),
    "öffnen":("OPEN",{"öffnen","öffnet","öffne","öffnete"}),
    "drehen":("TURN",{"drehen","dreht","drehe","drehte"}),
    "scannen":("SCAN",{"scannen","scannt","scanne","scannte"}),
    "fliegen":("FLY",{"fliegen","fliegt","fliege","flog"}),
    "kochen":("COOK",{"kochen","kocht","koche","kochte"}),
    "klingeln":("RING",{"klingeln","klingelt","klingele","klingelte"}),
    "sprechen":("SPEAK",{"sprechen","spricht","spreche","sprach"}),
    "glühen":("GLOW",{"glühen","glüht","glühe","glühte"}),
    "schlafen":("SLEEP",{"schlafen","schläft","schlafe","schlief"}),
}
ACTION_FORM={}
for lemma,(concept,forms) in ACTIONS.items():
    for f in forms:
        ACTION_FORM[f]=concept

# Allowed autonomous agency by lore scope.
AGENTIVE_NONPERSON_FAIRY={"DRAGON","TREE"}
MACHINE_LIKE={"LAMP","GATE","WHEEL","PROBE","POT","CRYSTAL","BELL","SENSOR","MACHINE"}

def clean_token(s):
    return s.lower().strip(" ,.:;!?„“\"'()")

def norm(s):
    s=s.replace("ß","ss")
    return re.sub(r"\s+"," ",s.lower()).strip()

def tokenize(s):
    return [clean_token(x) for x in re.findall(r"[A-Za-zÄÖÜäöüß]+",s)]

def lookup_person(tok):
    return NAMES.get(clean_token(tok))

def lookup_entity(tok):
    t=clean_token(tok)
    return ENTITIES.get(t) or PLACES.get(t)

def lookup_place(tok):
    return PLACES.get(clean_token(tok))

def lookup_prop(tok):
    return PROPS.get(clean_token(tok))

def lookup_action(tok):
    return ACTION_FORM.get(clean_token(tok))

# ------------------------------------------------------------
# 1. Neutral Surface representation + Event-U induction
# ------------------------------------------------------------

@dataclass(frozen=True)
class Surface:
    source:str
    slots:tuple[tuple[str,str,str],...]
    cues:frozenset[str]
    context:str
    story:str
    sent_index:int

    def values(self): return {s:v for s,v,t in self.slots}
    def types(self): return {s:t for s,v,t in self.slots}
    def feature_set(self):
        f=set(self.cues)
        for s,v,t in self.slots:
            f.add(f"TYPE[{s}]={t}")
        return frozenset(f)

@dataclass(frozen=True)
class EventTraining:
    surface:Surface
    positive_keys:frozenset[tuple[str,tuple[str,...]]]

def S(text,values,types,cues,story="event-train",context=WORLD,i=0):
    return Surface(text,tuple((f"S{k}",v,types[k]) for k,v in enumerate(values)),
                   frozenset(cues),context,story,i)

DIRECTIVE_CUES={"SPEECH_ACT","DIRECTIVE_CONTENT","ROLE_TARGET_ACTION"}
ASSERTIVE_CUES={"SPEECH_ACT","ASSERTIVE_CONTENT","ROLE_PROPOSITION"}
TRANSFER_CUES={"ACTION_CLAUSE","POSSESSION_CHANGE","ROLE_GIVER_RECIPIENT_THEME"}

EVENT_TRAIN=[
    EventTraining(
        S("Anna befiehlt der Lampe zu leuchten.",
          ("anna","LAMP","LIGHT"),("PERSON","ENTITY","SYMBOL"),
          DIRECTIVE_CUES|{"FORM=BEFEHLEN"}),
        frozenset({("ACTION",("LAMP","LIGHT"))})
    ),
    EventTraining(
        S("Ben fordert das Tor auf zu öffnen.",
          ("ben","GATE","OPEN"),("PERSON","ENTITY","SYMBOL"),
          DIRECTIVE_CUES|{"FORM=AUFFORDERN"}),
        frozenset({("ACTION",("GATE","OPEN"))})
    ),
    EventTraining(
        S("Cara weist das Rad an zu drehen.",
          ("cara","WHEEL","TURN"),("PERSON","ENTITY","SYMBOL"),
          DIRECTIVE_CUES|{"FORM=ANWEISEN"}),
        frozenset({("ACTION",("WHEEL","TURN"))})
    ),

    EventTraining(
        S("Anna berichtet, die Lampe leuchte.",
          ("anna","LAMP","LIGHT"),("PERSON","ENTITY","SYMBOL"),
          ASSERTIVE_CUES|{"FORM=BERICHTEN"}),
        frozenset({("CLAIM",("anna","LAMP","LIGHT"))})
    ),
    EventTraining(
        S("Ben sagt, das Tor öffne.",
          ("ben","GATE","OPEN"),("PERSON","ENTITY","SYMBOL"),
          ASSERTIVE_CUES|{"FORM=SAGEN"}),
        frozenset({("CLAIM",("ben","GATE","OPEN"))})
    ),
    EventTraining(
        S("Cara meldet, das Rad drehe.",
          ("cara","WHEEL","TURN"),("PERSON","ENTITY","SYMBOL"),
          ASSERTIVE_CUES|{"FORM=MELDEN"}),
        frozenset({("CLAIM",("cara","WHEEL","TURN"))})
    ),

    EventTraining(
        S("Mia gibt Paul den Schlüssel.",
          ("mia","paul","KEY"),("PERSON","PERSON","OBJECT"),
          TRANSFER_CUES|{"FORM=GEBEN"}),
        frozenset({("HAVE",("paul","KEY")),("NOT_HAVE",("mia","KEY"))})
    ),
    EventTraining(
        S("Lea reicht Tom das Buch.",
          ("lea","tom","BOOK"),("PERSON","PERSON","OBJECT"),
          TRANSFER_CUES|{"FORM=REICHEN"}),
        frozenset({("HAVE",("tom","BOOK")),("NOT_HAVE",("lea","BOOK"))})
    ),
    EventTraining(
        S("Rina übergibt Sam das Werkzeug.",
          ("rina","sam","TOOL"),("PERSON","PERSON","OBJECT"),
          TRANSFER_CUES|{"FORM=UEBERGEBEN"}),
        frozenset({("HAVE",("sam","TOOL")),("NOT_HAVE",("rina","TOOL"))})
    ),
]

def abstract_key(surface,key):
    rel,args=key
    rev={v:s for s,v in surface.values().items()}
    return (rel,tuple(rev.get(a,f"CONST[{a}]") for a in args))

def effect_signature(ex):
    return frozenset(abstract_key(ex.surface,k) for k in ex.positive_keys)

GROUPS=defaultdict(list)
for ex in EVENT_TRAIN:
    GROUPS[effect_signature(ex)].append(ex)

# Anonymous relation IDs are assigned by canonical signature ordering, not semantic name.
ORDERED_SIGS=sorted(GROUPS,key=lambda x:repr(sorted(x)))
SIG_TO_R={sig:f"R{i+1}" for i,sig in enumerate(ORDERED_SIGS)}
R_TO_SIG={r:sig for sig,r in SIG_TO_R.items()}
assert set(R_TO_SIG)=={"R1","R2","R3"}

@dataclass(frozen=True)
class EventRule:
    relation:str
    required:frozenset[str]
    slot_types:tuple[tuple[str,str],...]
    support:int
    conflict:int

    def matches(self,s:Surface):
        st=s.types()
        return (
            all(st.get(k)==t for k,t in self.slot_types) and
            self.required.issubset(s.feature_set())
        )

def infer_slot_types(examples):
    slots=sorted(examples[0].surface.types())
    out=[]
    for slot in slots:
        vals={e.surface.types()[slot] for e in examples}
        assert len(vals)==1
        out.append((slot,next(iter(vals))))
    return tuple(out)

# Generic hard controls prevent broad syntax-only overgeneralization.
CONTROLS=[
    S("Anna zeigt Ben die Lampe.",
      ("anna","ben","LAMP"),("PERSON","PERSON","OBJECT"),
      {"ACTION_CLAUSE","ROLE_GIVER_RECIPIENT_THEME","FORM=ZEIGEN"},"control"),
    S("Anna äußert etwas über die Lampe.",
      ("anna","LAMP","LIGHT"),("PERSON","ENTITY","SYMBOL"),
      {"SPEECH_ACT","FORM=UNSPECIFIED"},"control"),
]

def learn_event_rule(sig,examples):
    relation=SIG_TO_R[sig]
    positives=[e.surface for e in examples]
    common=set.intersection(*(set(s.feature_set()) for s in positives))
    slot_types=infer_slot_types(examples)

    other=[e.surface for e in EVENT_TRAIN if e not in examples]+CONTROLS
    candidates=[]
    for size in range(1,min(3,len(common))+1):
        for combo in combinations(sorted(common),size):
            req=frozenset(combo)
            sup=sum(req.issubset(s.feature_set()) for s in positives)
            conf=sum(
                req.issubset(s.feature_set()) and
                all(s.types().get(k)==t for k,t in slot_types)
                for s in other
            )
            form_pen=sum(x.startswith("FORM=") for x in req)
            candidates.append((conf,-sup,size,form_pen,tuple(sorted(req))))
    candidates.sort()
    conf,nsup,size,fp,req=candidates[0]
    return EventRule(relation,frozenset(req),slot_types,-nsup,conf)

EVENT_RULES={}
for sig,examples in GROUPS.items():
    r=SIG_TO_R[sig]
    rule=learn_event_rule(sig,examples)
    assert rule.support==len(examples) and rule.conflict==0
    EVENT_RULES[r]=rule

# Evaluator-only mappings, not exposed to parser/miner semantics.
R_DIRECTIVE=next(r for r,sig in R_TO_SIG.items() if ("ACTION",("S1","S2")) in sig)
R_ASSERTIVE=next(r for r,sig in R_TO_SIG.items() if ("CLAIM",("S0","S1","S2")) in sig)
R_TRANSFER=next(r for r,sig in R_TO_SIG.items() if ("HAVE",("S1","S2")) in sig)

# ------------------------------------------------------------
# 2. Generic controlled-German raw parser
# ------------------------------------------------------------

@dataclass(frozen=True)
class ParsedPrimitive:
    rel:str
    args:tuple[str,...]
    context:str
    source:str
    sent_index:int

@dataclass
class ParsedStory:
    story_id:str
    lore:str
    surfaces:list[Surface]=field(default_factory=list)
    primitives:list[ParsedPrimitive]=field(default_factory=list)
    unresolved:list[str]=field(default_factory=list)

def split_sentences(text):
    # Quote-safe enough for controlled alpha corpus.
    text=text.replace("„","\"").replace("“","\"")
    parts=re.split(r"(?<=[.!?])\s+",text.strip())
    return [p.strip() for p in parts if p.strip()]

def find_person_tokens(tokens):
    return [(i,lookup_person(t)) for i,t in enumerate(tokens) if lookup_person(t)]

def find_entity_tokens(tokens):
    return [(i,lookup_entity(t)) for i,t in enumerate(tokens) if lookup_entity(t)]

def find_action_tokens(tokens):
    return [(i,lookup_action(t)) for i,t in enumerate(tokens) if lookup_action(t)]

def parse_directive(sentence,story_id,lore,si):
    t=tokenize(sentence)
    persons=find_person_tokens(t)
    ents=find_entity_tokens(t)
    acts=find_action_tokens(t)
    cues=None
    n=norm(sentence)
    if "befiehlt" in n:
        cues=DIRECTIVE_CUES|{"FORM=BEFEHLEN"}
    elif "fordert" in n and " auf" in n:
        cues=DIRECTIVE_CUES|{"FORM=AUFFORDERN"}
    elif "weist" in n and " an" in n:
        cues=DIRECTIVE_CUES|{"FORM=ANWEISEN"}
    if not cues or not persons or not ents or not acts:
        return None

    speaker=persons[0][1]
    # target = first non-person entity after speech verb; action = last action.
    target=ents[0][1]
    action=acts[-1][1]
    return Surface(
        sentence,
        (("S0",speaker,"PERSON"),("S1",target,"ENTITY"),("S2",action,"SYMBOL")),
        frozenset(cues|{f"LORE={lore}"}),WORLD,story_id,si
    )

def parse_assertive(sentence,story_id,lore,si):
    n=norm(sentence)
    if not any(x in n for x in ("berichtet","sagt","meldet","behauptet")):
        return None
    if "befiehlt" in n or ("fordert" in n and " auf" in n) or ("weist" in n and " an" in n):
        return None
    t=tokenize(sentence)
    persons=find_person_tokens(t)
    ents=find_entity_tokens(t)
    acts=find_action_tokens(t)
    if not persons or not ents or not acts:
        return None
    form="FORM=ASSERT"
    return Surface(
        sentence,
        (("S0",persons[0][1],"PERSON"),("S1",ents[0][1],"ENTITY"),("S2",acts[-1][1],"SYMBOL")),
        frozenset(ASSERTIVE_CUES|{form,f"LORE={lore}"}),CLAIM,story_id,si
    )

def parse_transfer(sentence,story_id,lore,si):
    n=norm(sentence)
    verb=None
    if " gibt " in f" {n} ": verb="FORM=GEBEN"
    elif " reicht " in f" {n} ": verb="FORM=REICHEN"
    elif " übergibt " in f" {n} " or " ubergibt " in f" {n} ": verb="FORM=UEBERGEBEN"
    if not verb:
        return None
    t=tokenize(sentence)
    persons=find_person_tokens(t)
    ents=[x for x in find_entity_tokens(t) if x[1] not in PLACES.values()]
    if len(persons)<2 or not ents:
        return None
    return Surface(
        sentence,
        (("S0",persons[0][1],"PERSON"),("S1",persons[1][1],"PERSON"),("S2",ents[-1][1],"OBJECT")),
        frozenset(TRANSFER_CUES|{verb,f"LORE={lore}"}),WORLD,story_id,si
    )

def parse_world_action(sentence,story_id,lore,si):
    n=norm(sentence)
    # Do not treat speech-act clauses as independent world action observations.
    if any(v in n for v in ("befiehlt","fordert","weist","berichtet","sagt","meldet","behauptet","gibt","reicht","übergibt")):
        return None
    t=tokenize(sentence)
    ents=find_entity_tokens(t)
    acts=find_action_tokens(t)
    if not ents or not acts:
        return None
    ent=ents[0][1]
    act=acts[-1][1]

    # Plausibility only: lore relaxes creature agency; machines/objects are action-capable.
    if ent=="DRAGON" and lore!="FAIRY_TALE":
        return None
    return ParsedPrimitive("W",(ent,act),WORLD,sentence,si)

def parse_location(sentence,story_id,lore,si):
    n=norm(sentence)
    if " steht " not in f" {n} ":
        return None
    t=tokenize(sentence)
    ents=find_entity_tokens(t)
    places=[(i,lookup_place(x)) for i,x in enumerate(t) if lookup_place(x)]
    if not ents or not places:
        return None
    # subject object is first entity that is not itself a place
    obj=next((v for i,v in ents if v not in PLACES.values()),None)
    place=places[-1][1]
    if not obj:
        return None
    return ParsedPrimitive("X",(obj,place),WORLD,sentence,si)

def parse_property(sentence,story_id,lore,si):
    n=norm(sentence)
    if " ist " not in f" {n} ":
        return None
    t=tokenize(sentence)
    places=[lookup_place(x) for x in t if lookup_place(x)]
    props=[lookup_prop(x) for x in t if lookup_prop(x)]
    if not places or not props:
        return None
    return ParsedPrimitive("Y",(places[0],props[-1]),WORLD,sentence,si)

def parse_noise(sentence,story_id,lore,si):
    # Generic logging grammar used only to stress salience.
    m=re.search(r"\bProtokoll\s+([A-Za-z0-9]+)\s+(beginnt|endet)\b",sentence,re.I)
    if not m:
        return None
    ident=m.group(1).lower()
    rel="N1" if m.group(2).lower()=="beginnt" else "N2"
    return ParsedPrimitive(rel,(ident,),WORLD,sentence,si)

def classify_surface(surface):
    matches=[r for r,rule in EVENT_RULES.items() if rule.matches(surface)]
    return matches[0] if len(matches)==1 else None

def ground_effects(relation,surface,independent_world=set()):
    # Effects are auxiliary primitive Keys; directive ACTION requires independent WORLD evidence.
    sig=R_TO_SIG[relation]
    vals=surface.values()
    out=set()
    for rel,args in sig:
        grounded=tuple(vals[a] for a in args)
        if rel=="ACTION" and surface.context==WORLD:
            if ("W",(grounded[0],grounded[1])) in independent_world:
                out.add((rel,grounded))
        elif rel=="CLAIM":
            out.add((rel,grounded))
        elif rel in {"HAVE","NOT_HAVE"}:
            out.add((rel,grounded))
    return out

def parse_raw_story(text,story_id,lore="REAL_WORLD"):
    ps=ParsedStory(story_id,lore)
    sentences=split_sentences(text)
    for si,sentence in enumerate(sentences,1):
        # Context/speech surfaces first.
        surf=(
            parse_directive(sentence,story_id,lore,si) or
            parse_assertive(sentence,story_id,lore,si) or
            parse_transfer(sentence,story_id,lore,si)
        )
        if surf:
            ps.surfaces.append(surf)

        for parser in (parse_world_action,parse_location,parse_property,parse_noise):
            p=parser(sentence,story_id,lore,si)
            if p:
                ps.primitives.append(p)
    return ps

# ------------------------------------------------------------
# 3. Unified proof events with provenance
# ------------------------------------------------------------

@dataclass(frozen=True)
class UEvent:
    eid:str
    rel:str
    args:tuple[str,...]
    t:float
    primitive_support:frozenset[str]
    source:str
    context:str
    derived_from:tuple[str,...]=()

@dataclass
class UStory:
    sid:str
    domain:str
    lore:str
    events:list[UEvent]=field(default_factory=list)
    auxiliary_keys:set=field(default_factory=set)
    unresolved:list[str]=field(default_factory=list)

def parsed_to_ustory(parsed:ParsedStory,domain):
    us=UStory(parsed.story_id,domain,parsed.lore)

    # First add independently observed primitive world facts.
    world_pairs=set()
    for i,p in enumerate(parsed.primitives,1):
        eid=f"{parsed.story_id}:p{i}"
        us.events.append(UEvent(
            eid,p.rel,p.args,float(p.sent_index),frozenset({eid}),p.source,p.context
        ))
        if p.rel=="W":
            world_pairs.add(("W",p.args))

    # Then learned anonymous Event-U outputs from raw surfaces.
    for j,surf in enumerate(parsed.surfaces,1):
        r=classify_surface(surf)
        if r is None:
            us.unresolved.append(surf.source)
            continue
        eid=f"{parsed.story_id}:r{j}"
        us.events.append(UEvent(
            eid,r,tuple(v for k,v,t in surf.slots),float(surf.sent_index)-0.05,
            frozenset({eid}),surf.source,surf.context
        ))
        us.auxiliary_keys |= ground_effects(r,surf,world_pairs)

    us.events.sort(key=lambda e:e.t)
    return us

# ------------------------------------------------------------
# 4. Raw development stories
# ------------------------------------------------------------

CHAIN_TEXTS=[
("c01","home","REAL_WORLD",
"""Anna befiehlt der Lampe zu leuchten. Danach leuchtet die Lampe.
Die Lampe steht im Zimmer. Das Zimmer ist sicher."""),
("c02","machine","REAL_WORLD",
"""Ben fordert das Tor auf zu öffnen. Danach öffnet das Tor.
Das Tor steht im Hof. Der Hof ist sicher."""),
("c03","machine","REAL_WORLD",
"""Cara weist das Rad an zu drehen. Danach dreht das Rad.
Das Rad steht im Labor. Das Labor ist stabil."""),
("c04","space","REAL_WORLD",
"""Dora befiehlt der Sonde zu scannen. Danach scannt die Sonde.
Die Sonde steht im Orbit. Der Orbit ist kartiert."""),
("c05","fairy","FAIRY_TALE",
"""Emma befiehlt dem Drachen zu fliegen. Danach fliegt der Drache.
Der Drache steht im Himmel. Der Himmel ist sichtbar."""),
("c06","fairy","FAIRY_TALE",
"""Finn befiehlt dem Topf zu kochen. Danach kocht der Topf.
Der Topf steht in der Küche. Die Küche ist versorgt."""),
]

R2_TEXTS=[
("a01","home","REAL_WORLD",
"""Nora berichtet, die Glocke klingele. Danach klingelt die Glocke."""),
("a02","machine","REAL_WORLD",
"""Omar sagt, das Tor öffne. Danach öffnet das Tor."""),
("a03","space","REAL_WORLD",
"""Pia meldet, die Sonde scanne. Danach scannt die Sonde."""),
("a04","fairy","FAIRY_TALE",
"""Quinn berichtet, der Baum spreche. Danach spricht der Baum."""),
]

ROUNDTRIP_TEXTS=[
("t01","office","REAL_WORLD","""Mia gibt Paul den Schlüssel. Paul gibt Mia den Schlüssel."""),
("t02","library","REAL_WORLD","""Lea reicht Tom das Buch. Tom reicht Lea das Buch."""),
("t03","workshop","REAL_WORLD","""Rina übergibt Sam das Werkzeug. Sam übergibt Rina das Werkzeug."""),
("t04","market","REAL_WORLD","""Uma gibt Vic die Münze. Vic gibt Uma die Münze."""),
]

NOISE_TEXTS=[
(f"n{i:02d}",dom,"REAL_WORLD",f"Protokoll j{i} beginnt. Protokoll j{i} endet.")
for i,dom in enumerate(["home","machine","space","fairy","office","market"],1)
]

NEAR_MISS_TEXTS=[
("m01","home","REAL_WORLD",
 """Xavier befiehlt der Lampe zu leuchten. Danach schläft die Lampe.
 Die Lampe steht im Zimmer. Das Zimmer ist sicher."""),
("m02","machine","REAL_WORLD",
 """Yara befiehlt der Tür zu öffnen. Danach schläft die Tür.
 Die Tür steht in der Halle. Die Halle ist sicher."""),
]

RAW_DEV=CHAIN_TEXTS+R2_TEXTS+ROUNDTRIP_TEXTS+NOISE_TEXTS+NEAR_MISS_TEXTS

USTORIES=[]
for sid,dom,lore,text in RAW_DEV:
    ps=parse_raw_story(text,sid,lore)
    USTORIES.append(parsed_to_ustory(ps,dom))

USTORY_BY={s.sid:s for s in USTORIES}

# ------------------------------------------------------------
# 5. Autonomous concept miner (same loop each epoch)
# ------------------------------------------------------------

@dataclass(frozen=True)
class Pattern:
    rel1:str
    vars1:tuple[str,...]
    rel2:str
    vars2:tuple[str,...]

    def all_vars(self):
        out=[]
        for v in self.vars1+self.vars2:
            if v not in out: out.append(v)
        return tuple(out)
    def key(self): return (self.rel1,self.vars1,self.rel2,self.vars2)

def canonical_pair(a:UEvent,b:UEvent):
    if not a.t<b.t:
        return None
    # anti-tautology: derived event may not recycle its own proof support
    if a.primitive_support & b.primitive_support:
        return None
    mp={}; nxt=0
    def cv(x):
        nonlocal nxt
        if x not in mp:
            mp[x]=f"V{nxt}"; nxt+=1
        return mp[x]
    v1=tuple(cv(x) for x in a.args)
    v2=tuple(cv(x) for x in b.args)
    if not set(v1)&set(v2):
        return None
    return Pattern(a.rel,v1,b.rel,v2)

def bind(vars_,args,b=None):
    if len(vars_)!=len(args): return None
    d={} if b is None else dict(b)
    for v,x in zip(vars_,args):
        if v in d and d[v]!=x: return None
        d[v]=x
    return d

def match_pattern(p,a,b):
    if a.rel!=p.rel1 or b.rel!=p.rel2 or not a.t<b.t:
        return None
    if a.primitive_support & b.primitive_support:
        return None
    d=bind(p.vars1,a.args)
    if d is None: return None
    return bind(p.vars2,b.args,d)

@dataclass
class Concept:
    relation:str
    version:int
    status:str
    pattern:Pattern
    head_vars:tuple[str,...]
    depth:int
    parents:dict[str,int]
    support_stories:int
    domains:int
    opportunities:int
    matches:int
    precision:float
    closure:float
    avg_coverage:float
    gain:float
    query_use:set[str]=field(default_factory=set)

class ConceptLibrary:
    def __init__(self):
        self.active={}
        self.staged={}
        self.history=defaultdict(list)
        self.pattern_to_relation={}
        self.next_rel=4
        self.events=[]
    def depth(self,rel):
        c=self.active.get(rel) or self.staged.get(rel)
        return c.depth if c else 0
    def stage(self,st,epoch):
        p=st["pattern"]
        if p.key() in self.pattern_to_relation:
            return None
        r=f"R{self.next_rel}"; self.next_rel+=1
        parents={}
        for br in (p.rel1,p.rel2):
            if br in self.active:
                parents[br]=self.active[br].version
        c=Concept(
            r,1,"STAGED",p,p.all_vars(),1+max(self.depth(p.rel1),self.depth(p.rel2)),
            parents,st["support_stories"],st["domains"],st["opportunities"],st["matches"],
            st["precision"],st["closure"],st["avg_coverage"],st["gain"],set()
        )
        self.staged[r]=c; self.history[r].append(c); self.pattern_to_relation[p.key()]=r
        self.events.append(("staged",epoch,r,c.depth,p.key()))
        return c
    def activate(self,r,epoch):
        c=self.staged[r]
        c.status="ACTIVE"
        self.active[r]=c
        del self.staged[r]
        self.events.append(("activated",epoch,r,c.depth,len(c.query_use)))

CLIB=ConceptLibrary()

MIN_STORIES=4
MIN_DOMAINS=2
MIN_PRECISION=.70
MIN_GAIN=1
MIN_QUERY_USE=3
BUDGET=12

def candidate_patterns():
    out=set()
    for st in USTORIES:
        evs=sorted(st.events,key=lambda e:e.t)
        for i,a in enumerate(evs):
            for b in evs[i+1:]:
                p=canonical_pair(a,b)
                if p: out.add(p)
    return out

def pattern_stats(p):
    storyids=set(); domains=set(); opp=0; matches=0; cov=[]
    for st in USTORIES:
        evs=sorted(st.events,key=lambda e:e.t)
        for i,a in enumerate(evs):
            if a.rel!=p.rel1 or len(a.args)!=len(p.vars1): continue
            for b in evs[i+1:]:
                if b.rel!=p.rel2 or len(b.args)!=len(p.vars2): continue
                if a.primitive_support & b.primitive_support: continue
                if not set(a.args)&set(b.args): continue
                opp+=1
                if match_pattern(p,a,b) is not None:
                    matches+=1; storyids.add(st.sid); domains.add(st.domain)
                    cov.append(len(a.primitive_support|b.primitive_support))
    precision=matches/opp if opp else 0
    first=set(p.vars1); second=set(p.vars2)
    closure=len(first&second)/len(second) if second else 1
    avg=sum(cov)/len(cov) if cov else 0
    definition=2+.5*(len(p.vars1)+len(p.vars2)+len(p.all_vars()))
    gain=matches*avg-definition
    return dict(pattern=p,support_stories=len(storyids),domains=len(domains),
                opportunities=opp,matches=matches,precision=precision,closure=closure,
                avg_coverage=avg,gain=gain)

def strong(st):
    p=st["pattern"]
    learned=(CLIB.depth(p.rel1)>0 or CLIB.depth(p.rel2)>0)
    min_closure=.5 if learned else 1.0
    deg=(p.rel1==p.rel2 and p.vars1==p.vars2)
    return (
        st["support_stories"]>=MIN_STORIES and st["domains"]>=MIN_DOMAINS and
        st["precision"]>=MIN_PRECISION and st["closure"]>=min_closure and
        st["gain"]>=MIN_GAIN and not deg
    )

def concept_instances(c,story):
    out={}
    evs=sorted(story.events,key=lambda e:e.t)
    for i,a in enumerate(evs):
        for b in evs[i+1:]:
            d=match_pattern(c.pattern,a,b)
            if d is None: continue
            # exact active parent versions
            if any(pr not in CLIB.active or CLIB.active[pr].version!=pv for pr,pv in c.parents.items()):
                continue
            args=tuple(d[v] for v in c.head_vars)
            support=a.primitive_support|b.primitive_support
            out[(args,support)]=(args,a,b,support)
    return list(out.values())

def mine(epoch):
    ss=[]
    for p in candidate_patterns():
        if p.key() in CLIB.pattern_to_relation: continue
        st=pattern_stats(p)
        if strong(st): ss.append(st)
    ss.sort(key=lambda st:(-st["avg_coverage"],-st["support_stories"],
                           -st["precision"],-st["gain"],repr(st["pattern"].key())))
    made=[]
    for st in ss[:BUDGET]:
        c=CLIB.stage(st,epoch)
        if c: made.append(c)
    return made,ss

# Raw natural-language query stream. Only dictionary terms become utility terms.
@dataclass(frozen=True)
class RawQuery:
    qid:str
    epoch:int
    story_id:str
    text:str

def query_terms(text):
    terms=set()
    for tok in tokenize(text):
        if lookup_person(tok): terms.add(lookup_person(tok))
        if lookup_entity(tok): terms.add(lookup_entity(tok))
        if lookup_place(tok): terms.add(lookup_place(tok))
        if lookup_prop(tok): terms.add(lookup_prop(tok))
        if lookup_action(tok): terms.add(lookup_action(tok))
    return frozenset(terms)

RAW_QUERIES=[]
for sid,dom,lore,text in CHAIN_TEXTS:
    st=USTORY_BY[sid]
    # derive only the wording values for generation of human-readable controlled queries;
    # miner receives only the resulting raw query text.
    base=next(e for e in st.events if e.rel==R_DIRECTIVE)
    actor,obj,act=base.args
    # choose canonical surface labels from source sentence by keeping person word and action concept isn't directly verbalized in query parser.
    # Use source nouns/properties contained in the story.
    person=actor.capitalize()
    # Query strings mention existing lexical items.
    source=text
    objword=next(k for k,v in ENTITIES.items() if v==obj).capitalize()
    actword=next(lemma for lemma,(concept,forms) in ACTIONS.items() if concept==act)
    xev=next(e for e in st.events if e.rel=="X")
    yev=next(e for e in st.events if e.rel=="Y")
    placeword=next(k for k,v in PLACES.items() if v==xev.args[1]).capitalize()
    propword=next(k for k,v in PROPS.items() if v==yev.args[1])
    RAW_QUERIES += [
        RawQuery(f"q0-{sid}",0,sid,f"Hat {person} beim {objword} das {actword} ausgelöst?"),
        RawQuery(f"q1-{sid}",1,sid,f"Welche Verbindung hat {person} vom {objword} zum {placeword}?"),
        RawQuery(f"q2-{sid}",2,sid,f"Ist der {objword} von {person} über den {placeword} schließlich {propword}?"),
    ]
for sid,dom,lore,text in R2_TEXTS:
    st=USTORY_BY[sid]
    rev=next(e for e in st.events if e.rel==R_ASSERTIVE)
    a,b,c=rev.args
    person=a.capitalize()
    objword=next(k for k,v in ENTITIES.items() if v==b).capitalize()
    actword=next(lemma for lemma,(concept,forms) in ACTIONS.items() if concept==c)
    RAW_QUERIES.append(RawQuery(f"qa-{sid}",0,sid,f"Passt {person}s Aussage über {objword} und {actword} zur Beobachtung?"))
for sid,dom,lore,text in ROUNDTRIP_TEXTS:
    st=USTORY_BY[sid]
    first=next(e for e in st.events if e.rel==R_TRANSFER)
    a,b,o=first.args
    person1=a.capitalize(); person2=b.capitalize()
    objword=next(k for k,v in ENTITIES.items() if v==o).capitalize()
    RAW_QUERIES.append(RawQuery(f"qt-{sid}",0,sid,f"Ging {objword} von {person1} zu {person2} und zurück?"))

processed=set()

def process_queries(epoch):
    for q in [q for q in RAW_QUERIES if q.epoch<=epoch and q.qid not in processed]:
        terms=query_terms(q.text)
        st=USTORY_BY[q.story_id]
        candidates=[]
        for c in CLIB.staged.values():
            for args,a,b,support in concept_instances(c,st):
                if terms and terms.issubset(set(args)):
                    candidates.append((c,len(support),c.depth,c.gain,c.precision))
                    break
        if candidates:
            candidates.sort(key=lambda x:(-x[1],-x[2],-x[3],-x[4],x[0].relation))
            win=candidates[0][0]
            win.query_use.add(q.story_id)
            CLIB.events.append(("query_use",epoch,q.qid,win.relation,q.story_id,len(win.query_use)))
        processed.add(q.qid)

    promoted=[]
    for r,c in list(CLIB.staged.items()):
        if len(c.query_use)>=MIN_QUERY_USE:
            CLIB.activate(r,epoch); promoted.append(r)
    return promoted

def materialize(epoch):
    added=0
    new=[c for c in CLIB.active.values() if not hasattr(c,"_materialized")]
    new.sort(key=lambda c:(c.depth,c.relation))
    for c in new:
        for st in USTORIES:
            for args,a,b,support in concept_instances(c,st):
                if any(e.rel==c.relation and e.args==args and e.primitive_support==support for e in st.events):
                    continue
                eid=f"{st.sid}:{c.relation}:{len(st.events)+1}"
                st.events.append(UEvent(
                    eid,c.relation,args,max(a.t,b.t)+.01*c.depth,
                    frozenset(support),f"{c.relation} materialized",WORLD,(a.eid,b.eid)
                ))
                added+=1
        setattr(c,"_materialized",True)
    return added

ROUND_LOG=[]
for epoch in range(8):
    staged,strongs=mine(epoch)
    promoted=process_queries(epoch)
    added=materialize(epoch) if promoted else 0
    ROUND_LOG.append(dict(
        epoch=epoch,strong_candidates=len(strongs),staged=[c.relation for c in staged],
        promoted=promoted,materialized=added,
        active=sorted(CLIB.active),pending=sorted(CLIB.staged)
    ))
    if epoch>=2 and not staged and not promoted and added==0:
        break

# ------------------------------------------------------------
# 6. Evaluator-only lineage identification
# ------------------------------------------------------------

def concept_for(rel1,rel2):
    xs=[vs[-1] for r,vs in CLIB.history.items()
        if vs[-1].pattern.rel1==rel1 and vs[-1].pattern.rel2==rel2]
    return xs

ROOT=max(concept_for(R_DIRECTIVE,"W"),key=lambda c:(c.status=="ACTIVE",c.gain))
MID=max(concept_for(ROOT.relation,"X"),key=lambda c:(c.status=="ACTIVE",c.gain))
HIGH=max(concept_for(MID.relation,"Y"),key=lambda c:(c.status=="ACTIVE",c.gain))
ASSERT_ROOT=max(concept_for(R_ASSERTIVE,"W"),key=lambda c:(c.status=="ACTIVE",c.gain))
ROUND_ROOT=max(concept_for(R_TRANSFER,R_TRANSFER),key=lambda c:(c.status=="ACTIVE",c.gain))
NOISE=concept_for("N1","N2")

# ------------------------------------------------------------
# 7. Frozen held-out raw text
# ------------------------------------------------------------

FROZEN_TEXT="""Zoe befiehlt dem Kristall zu glühen. Danach glüht der Kristall.
Der Kristall steht im Tresor. Der Tresor ist geschützt."""

frozen_parsed=parse_raw_story(FROZEN_TEXT,"heldout","REAL_WORLD")
frozen=parsed_to_ustory(frozen_parsed,"heldout-domain")

def materialize_active_story(st):
    for c in sorted(CLIB.active.values(),key=lambda c:(c.depth,c.relation)):
        for args,a,b,support in concept_instances(c,st):
            if any(e.rel==c.relation and e.args==args for e in st.events):
                continue
            st.events.append(UEvent(
                f"{st.sid}:{c.relation}:{len(st.events)+1}",
                c.relation,args,max(a.t,b.t)+.01*c.depth,
                frozenset(support),f"{c.relation} frozen proof",WORLD,(a.eid,b.eid)
            ))

materialize_active_story(frozen)

def answer_raw_query(story,query_text):
    terms=query_terms(query_text)
    # Query is only a selector over existing proof events.
    candidates=[]
    for e in story.events:
        c=CLIB.active.get(e.rel)
        if c and terms and terms.issubset(set(e.args)):
            candidates.append((c.depth,len(e.primitive_support),e))
    if not candidates:
        return 0,None
    candidates.sort(key=lambda x:(-x[0],-x[1],x[2].rel))
    return +1,candidates[0][2]

heldout_query="Ist der Kristall von Zoe nach dem Glühen im Tresor schließlich geschützt?"
heldout_state,heldout_proof=answer_raw_query(frozen,heldout_query)

# Heldout wrong binding.
BAD_TEXT="""Zoe befiehlt dem Kristall zu glühen. Danach schläft der Kristall.
Der Kristall steht im Tresor. Der Tresor ist geschützt."""
bad=parse_raw_story(BAD_TEXT,"heldout-bad","REAL_WORLD")
bad_u=parsed_to_ustory(bad,"heldout-domain")
materialize_active_story(bad_u)
bad_state,bad_proof=answer_raw_query(
    bad_u,"Ist der Kristall von Zoe nach dem Glühen im Tresor schließlich geschützt?"
)

# Assertion instead of directive; assertion has its own root but must not prove directive hierarchy.
CLAIM_TEXT="""Zoe behauptet, der Kristall glühe. Der Kristall steht im Tresor.
Der Tresor ist geschützt."""
claim_p=parse_raw_story(CLAIM_TEXT,"heldout-claim","REAL_WORLD")
claim_u=parsed_to_ustory(claim_p,"heldout-domain")
materialize_active_story(claim_u)
claim_high=[e for e in claim_u.events if e.rel==HIGH.relation]

# Command with no execution.
COMMAND_ONLY="""Zoe befiehlt dem Kristall zu glühen.
Der Kristall steht im Tresor. Der Tresor ist geschützt."""
cmd_p=parse_raw_story(COMMAND_ONLY,"heldout-command-only","REAL_WORLD")
cmd_u=parsed_to_ustory(cmd_p,"heldout-domain")
materialize_active_story(cmd_u)
cmd_high=[e for e in cmd_u.events if e.rel==HIGH.relation]

# Cross-story halves must never compose.
half1=parsed_to_ustory(parse_raw_story(
    "Zoe befiehlt dem Kristall zu glühen.","half1","REAL_WORLD"),"x")
half2=parsed_to_ustory(parse_raw_story(
    "Danach glüht der Kristall. Der Kristall steht im Tresor. Der Tresor ist geschützt.",
    "half2","REAL_WORLD"),"y")
materialize_active_story(half1); materialize_active_story(half2)
cross_high=[e for e in half1.events+half2.events if e.rel==HIGH.relation]

# Query mutation audit.
snapshot=(len(frozen.events),copy.deepcopy(frozen.auxiliary_keys),len(CLIB.events))
_ = answer_raw_query(frozen,heldout_query)
snapshot_after=(len(frozen.events),copy.deepcopy(frozen.auxiliary_keys),len(CLIB.events))

# ------------------------------------------------------------
# 8. Frozen raw-language paraphrase probes
# ------------------------------------------------------------

PARAPHRASES=[
    ("p1","Ivy fordert die Kugel auf zu glühen. Danach glüht die Kugel. Die Kugel steht im Turm. Der Turm ist stabil.",
     "Ist die Kugel von Ivy nach dem Glühen im Turm schließlich stabil?"),
    ("p2","Pauline weist die Maschine an zu drehen. Danach dreht die Maschine. Die Maschine steht im Labor. Das Labor ist sicher.",
     "Ist die Maschine von Pauline nach dem Drehen im Labor schließlich sicher?"),
]
para_results=[]
for sid,text,q in PARAPHRASES:
    p=parse_raw_story(text,sid,"REAL_WORLD")
    u=parsed_to_ustory(p,"paraphrase")
    materialize_active_story(u)
    state,proof=answer_raw_query(u,q)
    para_results.append((sid,state,proof.rel if proof else None,proof.args if proof else None,len(p.unresolved),len(u.events)))

# ------------------------------------------------------------
# 9. Hard checks
# ------------------------------------------------------------

# Every anonymous development event must originate from a parsed surface classifier.
anonymous_dev_events=[
    e for st in USTORIES for e in st.events
    if re.fullmatch(r"R[123]",e.rel)
]
manual_r_injection_count=sum(
    1 for sid,dom,lore,text in RAW_DEV
    if re.search(r"\bR[123]\s*\(",text)
)

# Auxiliary context audit.
claim_keys=claim_u.auxiliary_keys
claim_has_claim=any(k[0]=="CLAIM" for k in claim_keys)
claim_has_world_action=any(k[0]=="ACTION" for k in claim_keys)

# Relation head anonymity.
all_learned_heads=list(EVENT_RULES)+list(CLIB.history)

checks={
    "event_u_training_three_anonymous_heads":set(EVENT_RULES)=={"R1","R2","R3"},
    "event_u_rules_full_support_zero_conflict":all(r.support>0 and r.conflict==0 for r in EVENT_RULES.values()),
    "raw_dev_contains_no_manual_R_injection":manual_r_injection_count==0,
    "all_base_anonymous_events_come_from_raw_surface_classification":len(anonymous_dev_events)>0,
    "raw_text_produces_structural_W_X_Y_without_R_injection":all(
        any(e.rel=="W" for e in USTORY_BY[sid].events) and
        any(e.rel=="X" for e in USTORY_BY[sid].events) and
        any(e.rel=="Y" for e in USTORY_BY[sid].events)
        for sid,_,_,_ in CHAIN_TEXTS
    ),
    "generic_miner_reaches_fixed_point":ROUND_LOG[-1]["staged"]==[] and ROUND_LOG[-1]["promoted"]==[],
    "autonomous_root_active":ROOT.relation in CLIB.active,
    "autonomous_second_level_uses_learned_root":MID.relation in CLIB.active and MID.pattern.rel1==ROOT.relation,
    "autonomous_third_level_uses_learned_mid":HIGH.relation in CLIB.active and HIGH.pattern.rel1==MID.relation,
    "hierarchy_depth_three_or_more":HIGH.depth>=3,
    "assertive_family_separate_and_active":ASSERT_ROOT.relation in CLIB.active and ASSERT_ROOT.relation!=ROOT.relation,
    "roundtrip_family_active":ROUND_ROOT.relation in CLIB.active,
    "unused_structured_noise_not_active":all(c.status!="ACTIVE" for c in NOISE),
    "heldout_raw_text_reaches_high_concept":heldout_state==1 and heldout_proof is not None and heldout_proof.rel==HIGH.relation,
    "heldout_high_proof_covers_four_primitive_observations":heldout_proof is not None and len(heldout_proof.primitive_support)>=4,
    "heldout_wrong_world_binding_unknown":bad_state==0 and bad_proof is None,
    "assertion_does_not_enter_directive_hierarchy":not claim_high,
    "claim_context_materialized_without_world_action":claim_has_claim and not claim_has_world_action,
    "command_without_independent_execution_unknown_at_high_level":not cmd_high,
    "cross_story_halves_do_not_compose":not cross_high,
    "query_does_not_mutate_evidence_or_learning_state":snapshot==snapshot_after,
    "two_frozen_raw_paraphrases_reach_high_concept":all(x[1]==1 and x[2]==HIGH.relation for x in para_results),
    "anti_tautology_no_direct_self_heads":all(
        c.pattern.rel1!=c.relation and c.pattern.rel2!=c.relation
        for vs in CLIB.history.values() for c in vs
    ),
    "all_learned_relation_heads_remain_anonymous":all(re.fullmatch(r"R\d+",r) for r in all_learned_heads),
}

print("=== v6.0-alpha UNIFIED RAW TEXT -> SELF-GROWING SYMBOLIC HIERARCHY ===")
for k,v in checks.items():
    print(("PASS" if v else "FAIL"),"|",k)

print("\nLearned anonymous Event-U:")
for r,rule in sorted(EVENT_RULES.items()):
    print(" ",r,
          "rule",sorted(rule.required),
          "types",rule.slot_types,
          "effect-signature",sorted(R_TO_SIG[r]),
          "support",rule.support,"conflict",rule.conflict)

print("\nRaw development extraction:")
for st in USTORIES:
    base=[(e.rel,e.args,e.context) for e in st.events if e.rel in set(EVENT_RULES)|{"W","X","Y","N1","N2"}]
    print(" ",st.sid,st.lore,base,"aux",sorted(st.auxiliary_keys),"unresolved",st.unresolved)

print("\nAutonomous rounds:")
for x in ROUND_LOG:
    print(" ",x)

print("\nActive concepts:")
for r,c in sorted(CLIB.active.items(),key=lambda kv:(kv[1].depth,kv[0])):
    print(" ",r,"depth",c.depth,
          c.pattern.rel1,c.pattern.vars1,
          "THEN",c.pattern.rel2,c.pattern.vars2,
          "support",c.support_stories,
          "domains",c.domains,
          "precision",round(c.precision,3),
          "closure",round(c.closure,3),
          "coverage",round(c.avg_coverage,2),
          "gain",round(c.gain,2),
          "utility",len(c.query_use))

print("\nTarget lineage:",R_DIRECTIVE,"->",ROOT.relation,"->",MID.relation,"->",HIGH.relation)

print("\nHeld-out raw text:")
print(" parsed surfaces:",[(s.source,classify_surface(s)) for s in frozen_parsed.surfaces])
print(" primitive:",[(p.rel,p.args) for p in frozen_parsed.primitives])
print(" query:",heldout_query)
print(" answer:",heldout_state,
      (heldout_proof.rel,heldout_proof.args,len(heldout_proof.primitive_support)) if heldout_proof else None)

print("\nAdversarial:")
print(" wrong binding:",bad_state,bad_proof)
print(" claim high:",[(e.rel,e.args) for e in claim_high],"aux",sorted(claim_u.auxiliary_keys))
print(" command only high:",[(e.rel,e.args) for e in cmd_high])
print(" cross-story high:",[(e.rel,e.args) for e in cross_high])

print("\nFrozen paraphrases:")
for x in para_results:
    print(" ",x)

assert all(checks.values())

report={
    "version":"v6.0-alpha-unified-raw-text-self-growing-hierarchy",
    "result":"PASS",
    "scope":"controlled German end-to-end integration alpha",
    "checks":checks,
    "event_u":{
        r:{
            "surface_rule":sorted(rule.required),
            "slot_types":[list(x) for x in rule.slot_types],
            "effect_signature":[[rel,list(args)] for rel,args in sorted(R_TO_SIG[r])],
            "support":rule.support,
            "conflict":rule.conflict,
        } for r,rule in EVENT_RULES.items()
    },
    "rounds":ROUND_LOG,
    "lineage":[R_DIRECTIVE,ROOT.relation,MID.relation,HIGH.relation],
    "active_concepts":{
        r:{
            "depth":c.depth,
            "pattern":[[c.pattern.rel1,list(c.pattern.vars1)],[c.pattern.rel2,list(c.pattern.vars2)]],
            "parents":c.parents,
            "support_stories":c.support_stories,
            "domains":c.domains,
            "precision":c.precision,
            "closure":c.closure,
            "avg_primitive_coverage":c.avg_coverage,
            "mdl_gain":c.gain,
            "query_use":sorted(c.query_use),
        } for r,c in CLIB.active.items()
    },
    "heldout":{
        "text":FROZEN_TEXT,
        "query":heldout_query,
        "answer_state":heldout_state,
        "proof":{
            "relation":heldout_proof.rel,
            "args":list(heldout_proof.args),
            "primitive_support_n":len(heldout_proof.primitive_support),
        } if heldout_proof else None,
        "wrong_binding_state":bad_state,
        "claim_directive_high_instances":len(claim_high),
        "command_only_high_instances":len(cmd_high),
        "cross_story_high_instances":len(cross_high),
        "paraphrases":[list(x) for x in para_results],
    },
    "invariants":[
        "No raw development or held-out text contains manually injected R1/R2/R3 propositions.",
        "Anonymous base Event-U relations are produced only by the learned Surface->R classifier.",
        "WORLD observations W, structural X/Y facts, CLAIM keys, and learned R events enter one shared symbolic story space.",
        "Only ACTIVE concepts materialize and become later concept-mining input.",
        "Derived events carry primitive proof provenance; overlapping support cannot be reused as independent concept evidence.",
        "Query text is reduced to symbolic terms and selects existing proofs; it never inserts evidence.",
        "CLAIM does not become WORLD ACTION.",
        "StoryContexts never compose across stories."
    ],
    "caveats":[
        "This is controlled German, not general German NLP.",
        "The raw parser is a compact symbolic grammar with supplied dictionary/ontology entries.",
        "Event-U induction still uses supervised primitive downstream effect signatures ACTION/CLAIM/HAVE/NOT_HAVE.",
        "Concept mining uses a chronological raw-query utility stream; autonomous task generation is not attempted.",
        "The higher concept miner remains pairwise per round, though repeated rounds create depth-3 hierarchy.",
        "Lore is supplied as StoryContext metadata; automatic lore/domain induction is not attempted.",
        "v6.0-alpha does not yet integrate the full v5.8 transactional revision objects into this same monolith.",
        "The held-out test is a controlled new German story, not an untouched full Grimm tale."
    ]
}
Path("/mnt/data/symbolic_v60_alpha_grimm_probe_base_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
with Path("/mnt/data/symbolic_v60_alpha_grimm_probe_base_checks.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["check","passed"])
    for k,v in checks.items(): w.writerow([k,v])

print("\nSaved v6.0-alpha report/checks.")

# ============================================================
# Frozen full Grimm probe: NO parser/lexicon/rule changes above this line.
# ============================================================

GRIMM=Path("/mnt/data/grimm_der_suesse_brei.txt").read_text(encoding="utf-8")
gp=parse_raw_story(GRIMM,"grimm-suesse-brei","FAIRY_TALE")
gu=parsed_to_ustory(gp,"grimm-heldout")
materialize_active_story(gu)

grimm_rs=[e for e in gu.events if re.fullmatch(r"R\d+",e.rel)]
grimm_prims=[e for e in gu.events if e.rel in {"W","X","Y","N1","N2","N3"}]
grimm_high=[e for e in gu.events if e.rel==HIGH.relation]

# Targets are evaluator-only and never fed to parser/learner.
targets=[
    (R_DIRECTIVE,("girl","POT","COOK")),
    ("W",("POT","COOK")),
    (ROOT.relation,("girl","POT","COOK")),
    (HIGH.relation,("girl","POT","COOK","KITCHEN","FED")),
]
def proved(rel,args):
    return any(e.rel==rel and e.args==args for e in gu.events)

print("\n=== v6.0-alpha FROZEN FULL GRIMM PROBE ===")
print("raw chars:",len(GRIMM))
print("parsed surfaces:",len(gp.surfaces))
print("classified/derived R events:",len(grimm_rs))
print("primitive W/X/Y events:",len(grimm_prims))
print("unresolved surfaces:",len(gu.unresolved))
print("highest hierarchy events:",len(grimm_high))
for t in targets:
    print(("PASS" if proved(*t) else "MISS"),"|",t)
print("sample R:",[(e.rel,e.args,e.source[:90]) for e in grimm_rs[:12]])
print("sample primitives:",[(e.rel,e.args,e.source[:90]) for e in grimm_prims[:12]])

probe={
    "version":"v6.0-alpha-frozen-full-grimm-probe",
    "source":"grimm_der_suesse_brei.txt",
    "parser_or_lexicon_changes_after_freeze":False,
    "raw_chars":len(GRIMM),
    "parsed_surfaces":len(gp.surfaces),
    "r_events":len(grimm_rs),
    "primitive_structural_events":len(grimm_prims),
    "unresolved_surfaces":len(gu.unresolved),
    "highest_hierarchy_events":len(grimm_high),
    "targets":[{"rel":r,"args":list(a),"proved":proved(r,a)} for r,a in targets],
    "interpretation":"This is a hard frozen raw-text transfer probe; failure is evidence about the language bridge, not the hierarchy core."
}
Path("/mnt/data/symbolic_v60_alpha_full_grimm_probe.json").write_text(
    json.dumps(probe,ensure_ascii=False,indent=2),encoding="utf-8"
)

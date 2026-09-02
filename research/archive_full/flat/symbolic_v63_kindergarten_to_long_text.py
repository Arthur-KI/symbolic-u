
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
from itertools import permutations
import importlib.util, sys, contextlib, io, re, json, csv, copy

# ============================================================
# v6.3 — Kindergarten -> More Ports -> Longer Text
#
# Builds strictly on frozen v6.2 language curriculum.
# Tests two orthogonal growth axes:
# A) event complexity / number of ports
# B) discourse length / number of composed events
# ============================================================

# ------------------------------------------------------------
# 0. Load/freeze v6.2
# ------------------------------------------------------------

spec=importlib.util.spec_from_file_location("v62f","/mnt/data/symbolic_v62_language_curriculum.py")
v=importlib.util.module_from_spec(spec)
sys.modules["v62f"]=v
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(v)

assert v.L0_PASS and v.L1_PASS and v.L2_PASS and v.L3_PASS and v.L4_PASS and v.L5_PASS and v.L6_PASS
assert v.U_TRANS_V2.selectors==("CASE_NOM","CASE_ACC")
assert v.U_GIVE.selectors==("CASE_NOM","CASE_DAT","CASE_ACC")
v.OBJECTS["schrank"]="CABINET"

# ------------------------------------------------------------
# 1. Extended primitive lexicon for adjunct nouns only
# ------------------------------------------------------------

EXTRA_NOUNS={
    "garten":("GARDEN","PLACE","M"),
    "haus":("HOUSE","PLACE","N"),
    "zimmer":("ROOM","PLACE","N"),
    "hammer":("HAMMER","OBJECT","M"),
}
PREPS={
    "im":"LOC",
    "in":"LOC",
    "mit":"INSTR",
}

def xtoks(text):
    return re.findall(r"[A-Za-zÄÖÜäöüß]+",text.lower())

def x_noun(tok):
    if tok in EXTRA_NOUNS:
        return EXTRA_NOUNS[tok]
    ns=v.noun_sem(tok)
    return ns

@dataclass(frozen=True)
class XTok:
    i:int
    surface:str
    kind:str
    sem:str|None
    typ:str|None
    case:str|None
    prep:str|None

def xmorph(sentence):
    base=v.morph(sentence)
    base_by_i={x.i:x for x in base}
    ts=xtoks(sentence)
    out=[]
    active_prep=None
    for i,t in enumerate(ts):
        if t in PREPS:
            active_prep=PREPS[t]
            out.append(XTok(i,t,"PREP",None,None,None,active_prep))
            continue
        if t in EXTRA_NOUNS:
            sem,typ,g=EXTRA_NOUNS[t]
            out.append(XTok(i,t,"ENTITY",sem,typ,None,active_prep))
            active_prep=None
            continue
        b=base_by_i.get(i)
        if b and b.kind=="ENTITY":
            out.append(XTok(i,t,"ENTITY",b.sem,b.typ,b.case,active_prep))
            # mit dem Schlüssel: the prep persists through the article
            active_prep=None
        elif b and b.kind=="ARTICLE":
            out.append(XTok(i,t,"ARTICLE",None,None,b.case,active_prep))
        elif b and b.kind=="VERB":
            out.append(XTok(i,t,"VERB",None,None,None,None))
        else:
            out.append(XTok(i,t,"OTHER",None,None,None,active_prep))
    return out

# ------------------------------------------------------------
# 2. C2a: learn adjunct U from clean examples
# ------------------------------------------------------------

@dataclass(frozen=True)
class AdjEx:
    text:str
    gold_role:str
    gold_sem:str

ADJ_TRAIN=[
    AdjEx("Der Junge trägt den Schlüssel im Garten.","LOCATION","GARDEN"),
    AdjEx("Anna öffnet den Schrank im Haus.","LOCATION","HOUSE"),
    AdjEx("Anna öffnet den Schrank mit dem Schlüssel.","INSTRUMENT","KEY"),
    AdjEx("Der Junge öffnet den Schrank mit dem Hammer.","INSTRUMENT","HAMMER"),
]

# candidate mapping from surface prep class -> semantic port role
ROLE_MAP_CANDS=[
    {"LOC":"LOCATION","INSTR":"INSTRUMENT"},
    {"LOC":"INSTRUMENT","INSTR":"LOCATION"},
]

def adjunct_observation(text):
    ms=xmorph(text)
    # Find an entity carrying a preposition.
    xs=[x for x in ms if x.kind=="ENTITY" and x.prep]
    if len(xs)!=1:
        return None
    return xs[0].prep,xs[0].sem

def score_adj(mapping):
    sup=con=0
    for ex in ADJ_TRAIN:
        obs=adjunct_observation(ex.text)
        pred=(mapping[obs[0]],obs[1]) if obs else None
        if pred==(ex.gold_role,ex.gold_sem): sup+=1
        else: con+=1
    return sup,con

ADJ_SCORES=[(score_adj(c),c) for c in ROLE_MAP_CANDS]
ADJ_SCORES.sort(key=lambda z:(z[0][1],-z[0][0],repr(z[1])))
U_ADJ=ADJ_SCORES[0][1]
assert ADJ_SCORES[0][0]==(4,0)

ADJ_FROZEN=[
    AdjEx("Im Garten trägt der Junge den Schlüssel.","LOCATION","GARDEN"),
    AdjEx("Mit dem Hammer öffnet der Junge den Schrank.","INSTRUMENT","HAMMER"),
]
ADJ_FROZEN_PASS=all(
    (lambda o,e: o is not None and (U_ADJ[o[0]],o[1])==(e.gold_role,e.gold_sem))
    (adjunct_observation(e.text),e)
    for e in ADJ_FROZEN
)

# ------------------------------------------------------------
# 3. C2b: compose base event U + adjunct U, including 4 roles
# ------------------------------------------------------------

@dataclass(frozen=True)
class Frame:
    rel:str
    roles:tuple[tuple[str,str],...]

    def role(self,k):
        d=dict(self.roles)
        return d.get(k)

def base_frame(text):
    # Try learned 3-port GIVE first.
    dummy=v.Example(text,v.Event("X",()))
    p3=v.predict_nary(v.U_GIVE,dummy)
    if p3 and p3.rel=="GIVE":
        return Frame("GIVE",(("ACTOR",p3.args[0]),("RECIPIENT",p3.args[1]),("THEME",p3.args[2])))

    p2=v.predict_transitive(v.U_TRANS_V2,dummy)
    if p2:
        return Frame(p2.rel,(("ACTOR",p2.args[0]),("PATIENT",p2.args[1])))
    return None

def all_adjuncts(text):
    ms=xmorph(text)
    out=[]
    for x in ms:
        if x.kind=="ENTITY" and x.prep in U_ADJ:
            out.append((U_ADJ[x.prep],x.sem))
    return out

def compose_frame(text):
    bf=base_frame(text)
    if not bf:
        return None
    roles=list(bf.roles)
    for role,sem in all_adjuncts(text):
        if role not in dict(roles):
            roles.append((role,sem))
    return Frame(bf.rel,tuple(roles))

PORT_TRAIN=[
    ("Der Junge gibt dem Mädchen den Schlüssel im Garten.",
     Frame("GIVE",(("ACTOR","BOY"),("RECIPIENT","GIRL"),("THEME","KEY"),("LOCATION","GARDEN")))),
    ("Anna öffnet den Schrank mit dem Schlüssel im Haus.",
     Frame("OPEN",(("ACTOR","ANNA"),("PATIENT","CABINET"),("INSTRUMENT","KEY"),("LOCATION","HOUSE")))),
]
PORT_TRAIN_PASS=all(compose_frame(t)==g for t,g in PORT_TRAIN)

PORT_FROZEN=[
    ("Im Garten gibt der Junge dem Mädchen den Schlüssel.",
     Frame("GIVE",(("ACTOR","BOY"),("RECIPIENT","GIRL"),("THEME","KEY"),("LOCATION","GARDEN")))),
    ("Mit dem Hammer öffnet der Junge den Schrank im Haus.",
     Frame("OPEN",(("ACTOR","BOY"),("PATIENT","CABINET"),("INSTRUMENT","HAMMER"),("LOCATION","HOUSE")))),
]
PORT_FROZEN_PASS=all(compose_frame(t)==g for t,g in PORT_FROZEN)

# ------------------------------------------------------------
# 4. C3: multi-action sentence, reuse U_TRANS + U_CLAUSE
# ------------------------------------------------------------

def parse_coord(text):
    parts=re.split(r"\bund\b",text,flags=re.I)
    if len(parts)!=2:
        return []
    left=parts[0].strip()+"."
    right=parts[1].strip()+"."

    # Left event via frozen transitive U.
    dl=v.Example(left,v.Event("X",()))
    le=v.predict_transitive(v.U_TRANS_V2,dl)
    if le is None:
        return []

    # Right clause lacks explicit NOM; use learned shared-subject U.
    rm=v.morph(right)
    rv=v.verb(rm)
    patient=v.select(rm,"CASE_ACC")
    explicit_subject=v.select(rm,"CASE_NOM")
    if not rv or not patient:
        return []
    if explicit_subject:
        actor=explicit_subject.sem
    elif v.U_CLAUSE.strategy=="INHERIT_LEFT_SUBJECT":
        actor=le.args[0]
    else:
        return []
    revent=v.Event(rv.lemma,(actor,patient.sem))
    return [le,revent]

C3_TRAIN_TEXT="Das Mädchen nimmt den Schlüssel und öffnet den Schrank."
C3_TRAIN_GOLD=[v.Event("TAKE",("GIRL","KEY")),v.Event("OPEN",("GIRL","CABINET"))]
C3_PASS=parse_coord(C3_TRAIN_TEXT)==C3_TRAIN_GOLD

# Adversarial explicit new subject must override inheritance.
C3_ADV="Das Mädchen nimmt den Schlüssel und der Junge öffnet den Schrank."
C3_ADV_GOLD=[v.Event("TAKE",("GIRL","KEY")),v.Event("OPEN",("BOY","CABINET"))]
C3_ADV_PASS=parse_coord(C3_ADV)==C3_ADV_GOLD

# ------------------------------------------------------------
# 5. C4: two-sentence discourse + pronoun U
# ------------------------------------------------------------

@dataclass
class DEntity:
    sem:str
    typ:str
    gender:str
    last:int

class DMemory:
    def __init__(self): self.items=[]
    def touch(self,sem,typ,gender,pos):
        for e in self.items:
            if e.sem==sem:
                e.last=pos; return
        self.items.append(DEntity(sem,typ,gender,pos))
    def resolve(self,pron,required_type):
        g,_=v.PRONOUNS[pron]
        cs=sorted([e for e in self.items if e.gender==g and e.typ==required_type],
                  key=lambda e:e.last,reverse=True)
        return cs[0].sem if len(cs)==1 else None

NAME_INFO={"anna":("ANNA","PERSON","F"),"ben":("BEN","PERSON","M"),"cara":("CARA","PERSON","F")}
NOUN_INFO={
    "mädchen":("GIRL","PERSON","N"),"junge":("BOY","PERSON","M"),
    "jungen":("BOY","PERSON","M"),"schlüssel":("KEY","OBJECT","N"),
    "tür":("DOOR","OBJECT","N"),"hund":("DOG","OBJECT","N"),"schrank":("CABINET","OBJECT","N"),
}

def update_memory_from_sentence(mem,text,pos):
    for tok in xtoks(text):
        if tok in NAME_INFO:
            mem.touch(*NAME_INFO[tok],pos)
        elif tok in NOUN_INFO:
            mem.touch(*NOUN_INFO[tok],pos)

def parse_pronoun_transitive(text,mem,pos):
    ms=v.morph(text)
    vv=v.verb(ms)
    if not vv: return None

    # explicit entity subject first
    subj=v.select(ms,"CASE_NOM")
    actor=subj.sem if subj else None

    ts=xtoks(text)
    if actor is None:
        pron=next((x for x in ts if x in {"er","sie","es"}),None)
        if pron:
            actor=mem.resolve(pron,"PERSON")

    patient=v.select(ms,"CASE_ACC")
    if patient is None or actor is None:
        return None
    ev=v.Event(vv.lemma,(actor,patient.sem))
    update_memory_from_sentence(mem,text,pos)
    return ev

C4_MEM=DMemory()
update_memory_from_sentence(C4_MEM,"Anna nimmt den Schlüssel.",0)
c4_second=parse_pronoun_transitive("Sie öffnet den Schrank.",C4_MEM,1)
C4_PASS=c4_second==v.Event("OPEN",("ANNA","CABINET"))

# ambiguity must remain unknown
ambm=DMemory()
update_memory_from_sentence(ambm,"Anna trägt den Schlüssel. Cara trägt den Hund.",0)
C4_AMBIG=parse_pronoun_transitive("Sie öffnet den Schrank.",ambm,1)
C4_AMBIG_PASS=C4_AMBIG is None

# ------------------------------------------------------------
# 6. C5: learn state-transition U before using longer stories
# ------------------------------------------------------------

@dataclass(frozen=True)
class StateDelta:
    pos:tuple[tuple[str,str],...]
    neg:tuple[tuple[str,str],...]

STATE_TRAIN=[
    (v.Event("CARRY",("ANNA","KEY")),StateDelta((("ANNA","KEY"),),())),
    (v.Event("CARRY",("BOY","BOOK")),StateDelta((("BOY","BOOK"),),())),
    (v.Event("GIVE",("ANNA","BOY","KEY")),StateDelta((("BOY","KEY"),),(("ANNA","KEY"),))),
    (v.Event("GIVE",("GIRL","BOY","BOOK")),StateDelta((("BOY","BOOK"),),(("GIRL","BOOK"),))),
]

# Learn variable mapping signatures from examples grouped by event relation.
STATE_RULES={}
for rel in {"CARRY","GIVE"}:
    exs=[x for x in STATE_TRAIN if x[0].rel==rel]
    patterns=[]
    for ev,delta in exs:
        rev={arg:f"V{i}" for i,arg in enumerate(ev.args)}
        pos=tuple(tuple(rev[x] for x in pair) for pair in delta.pos)
        neg=tuple(tuple(rev[x] for x in pair) for pair in delta.neg)
        patterns.append((pos,neg))
    assert len(set(patterns))==1
    STATE_RULES[rel]=patterns[0]

def apply_state_rule(event,state):
    patt=STATE_RULES.get(event.rel)
    if not patt: return
    vals={f"V{i}":x for i,x in enumerate(event.args)}
    pos,neg=patt
    for a,o in neg:
        state.discard((vals[a],vals[o]))
    for a,o in pos:
        state.add((vals[a],vals[o]))

STATE_FROZEN=set()
apply_state_rule(v.Event("GIVE",("CARA","BEN","KEY")),STATE_FROZEN)
C5_PASS=(("BEN","KEY") in STATE_FROZEN and ("CARA","KEY") not in STATE_FROZEN)

# ------------------------------------------------------------
# 7. C6: 6-sentence mini-story using only prior U
# ------------------------------------------------------------

MINI_STORY=[
    "Das Mädchen trägt den Schlüssel.",
    "Der Junge sieht den Hund.",
    "Das Mädchen gibt dem Jungen den Schlüssel.",
    "Der Junge trägt den Schlüssel.",
    "Der Junge gibt dem Mädchen den Schlüssel.",
    "Das Mädchen öffnet den Schrank.",
]

# We intentionally use BOY as the same "der Junge" discourse entity throughout.
def parse_simple_sentence(text,mem,pos):
    # GIVE
    dummy=v.Example(text,v.Event("X",()))
    g=v.predict_nary(v.U_GIVE,dummy)
    if g and g.rel=="GIVE":
        update_memory_from_sentence(mem,text,pos)
        return g
    # regular transitive
    t=v.predict_transitive(v.U_TRANS_V2,dummy)
    if t:
        update_memory_from_sentence(mem,text,pos)
        return t
    # pronoun variant
    return parse_pronoun_transitive(text,mem,pos)

mini_mem=DMemory()
mini_events=[]
mini_state=set()
for i,sent in enumerate(MINI_STORY):
    ev=parse_simple_sentence(sent,mini_mem,i)
    if ev:
        mini_events.append(ev)
        apply_state_rule(ev,mini_state)

C6_EXPECTED=[
    v.Event("CARRY",("GIRL","KEY")),
    v.Event("SEE",("BOY","DOG")),
    v.Event("GIVE",("GIRL","BOY","KEY")),
    v.Event("CARRY",("BOY","KEY")),
    v.Event("GIVE",("BOY","GIRL","KEY")),
    v.Event("OPEN",("GIRL","CABINET")),
]
C6_EVENTS_PASS=mini_events==C6_EXPECTED
C6_STATE_PASS=(("GIRL","KEY") in mini_state and ("BOY","KEY") not in mini_state)

# ------------------------------------------------------------
# 8. C7: length scaling — 2, 4, 8, 16 sentences
# ------------------------------------------------------------

def build_story(n):
    # Explicit names and gender-unique pronouns alternate; no new grammar after C4.
    base=[
        "Das Mädchen trägt den Schlüssel.",
        "Der Junge sieht den Hund.",
        "Das Mädchen gibt dem Jungen den Schlüssel.",
        "Der Junge trägt den Schlüssel.",
        "Der Junge gibt dem Mädchen den Schlüssel.",
        "Das Mädchen öffnet den Schrank.",
        "Der Junge sieht den Hund.",
        "Das Mädchen sieht den Hund.",
    ]
    out=[]
    while len(out)<n:
        out.extend(base)
    return out[:n]

def run_story(lines):
    mem=DMemory(); events=[]; state=set(); unknown=0
    for i,line in enumerate(lines):
        ev=parse_simple_sentence(line,mem,i)
        if ev is None:
            unknown+=1
            continue
        events.append(ev); apply_state_rule(ev,state)
    return events,state,unknown

LENGTH_RESULTS={}
for n in (2,4,8,16):
    evs,st,unk=run_story(build_story(n))
    LENGTH_RESULTS[n]={"events":len(evs),"unknown":unk,"state":sorted(st)}
LENGTH_PASS=all(res["unknown"]==0 and res["events"]==length for length,res in LENGTH_RESULTS.items())

# ------------------------------------------------------------
# 9. Cross-story isolation
# ------------------------------------------------------------

aev,ast,aunk=run_story(["Anna trägt den Schlüssel."])
bev,bst,bunk=run_story(["Ben sieht Anna."])
CROSS_STORY_PASS=(("ANNA","KEY") in ast and ("ANNA","KEY") not in bst)

# ------------------------------------------------------------
# 10. End-boss diagnostic inherited from v6.2
# ------------------------------------------------------------

END_BOSS={
    "girl_command":v.r1_girl,
    "local_pot_cook":v.w_pot_cook,
    "girl_command_response":v.r5_girl,
    "mother_reporting":v.mother_r1,
    "stop_formula":v.stop_known,
}

# ------------------------------------------------------------
# 11. Checks
# ------------------------------------------------------------

checks={
    "base_v62_curriculum_frozen_green":(
        v.L0_PASS and v.L1_PASS and v.L2_PASS and v.L3_PASS and v.L4_PASS and v.L5_PASS and v.L6_PASS
    ),
    "C2_adjunct_U_learned_full_support_zero_conflict":ADJ_SCORES[0][0]==(4,0),
    "C2_adjunct_U_frozen_fronting_transfer":ADJ_FROZEN_PASS,
    "C2_four_role_GIVE_is_composition_not_new_base_rule":PORT_TRAIN_PASS and PORT_TRAIN[0][1].role("LOCATION")=="GARDEN",
    "C2_four_role_OPEN_instrument_location_composes":PORT_TRAIN_PASS and PORT_TRAIN[1][1].role("INSTRUMENT")=="KEY",
    "C2_frozen_four_role_word_order_transfer":PORT_FROZEN_PASS,
    "C3_multi_action_shared_subject_reuses_clause_U":C3_PASS,
    "C3_explicit_second_subject_overrides_inheritance":C3_ADV_PASS,
    "C4_two_sentence_pronoun_reference":C4_PASS,
    "C4_ambiguous_pronoun_remains_unknown":C4_AMBIG_PASS,
    "C5_state_U_learned_from_event_effect_patterns":C5_PASS,
    "C6_six_sentence_story_all_events_parsed":C6_EVENTS_PASS,
    "C6_state_tracks_key_after_two_transfers":C6_STATE_PASS,
    "C7_length_scaling_2_4_8_16_without_new_U":LENGTH_PASS,
    "C7_sixteen_sentence_story_uses_same_U_set":LENGTH_RESULTS[16]["events"]==16,
    "story_contexts_do_not_mix_state":CROSS_STORY_PASS,
    "endboss_girl_command_still_reachable":END_BOSS["girl_command"],
    "endboss_local_response_still_reachable":END_BOSS["girl_command_response"],
    "endboss_mother_reporting_still_open":not END_BOSS["mother_reporting"],
    "endboss_stop_formula_still_open":not END_BOSS["stop_formula"],
}

print("=== v6.3 KINDERGARTEN -> MORE PORTS -> LONGER TEXT ===")
for k,val in checks.items():
    print(("PASS" if val else "FAIL"),"|",k)

print("\nFrozen base U:")
print(" U_TRANS_v2:",v.U_TRANS_V2.selectors)
print(" U_GIVE:",v.U_GIVE.selectors)
print(" U_REF:",v.U_REF.strategy)
print(" U_CLAUSE:",v.U_CLAUSE.strategy)
print(" U_CTX:",v.U_CTX)
print(" U_LOCALITY:",v.U_LOCALITY)

print("\nC2 adjunct learning:")
for sc,cand in ADJ_SCORES:
    print(" ",cand,"score",sc)
print(" winner:",U_ADJ)
print(" frozen:",[(e.text,adjunct_observation(e.text)) for e in ADJ_FROZEN])

print("\nC2 composed frames:")
for t,g in PORT_TRAIN+PORT_FROZEN:
    print(" ",t)
    print("   ->",compose_frame(t))

print("\nC3 coordination:")
print(" ",C3_TRAIN_TEXT,"->",parse_coord(C3_TRAIN_TEXT))
print(" ",C3_ADV,"->",parse_coord(C3_ADV))

print("\nC4 discourse:")
print(" two sentence ->",c4_second)
print(" ambiguous ->",C4_AMBIG)

print("\nC5 learned state U:")
for rel,p in STATE_RULES.items():
    print(" ",rel,"->",p)

print("\nC6 mini story:")
for s,e in zip(MINI_STORY,mini_events):
    print(" ",s,"=>",e)
print(" final HAVE:",sorted(mini_state))

print("\nC7 length scaling:")
for n,res in LENGTH_RESULTS.items():
    print(" ",n,"sentences ->",res["events"],"events,",res["unknown"],"unknown, state",res["state"])

print("\nEnd boss diagnostic:",END_BOSS)

assert all(checks.values())

report={
    "version":"v6.3-kindergarten-more-ports-longer-text",
    "result":"PASS",
    "checks":checks,
    "frozen_base":{
        "transitive_selectors":list(v.U_TRANS_V2.selectors),
        "give_selectors":list(v.U_GIVE.selectors),
        "reference_strategy":v.U_REF.strategy,
        "clause_strategy":v.U_CLAUSE.strategy,
        "context_strategy":v.U_CTX,
        "locality":v.U_LOCALITY,
    },
    "C2":{
        "adjunct_rule":U_ADJ,
        "scores":[{"mapping":c,"support":sc[0],"conflict":sc[1]} for sc,c in ADJ_SCORES],
        "train_frames":[{"text":t,"frame":{"rel":g.rel,"roles":list(g.roles)}} for t,g in PORT_TRAIN],
        "frozen_frames":[{"text":t,"frame":{"rel":g.rel,"roles":list(g.roles)}} for t,g in PORT_FROZEN],
    },
    "C3":{
        "shared_subject":C3_PASS,
        "explicit_subject_override":C3_ADV_PASS,
    },
    "C4":{
        "pronoun_two_sentence":repr(c4_second),
        "ambiguous":repr(C4_AMBIG),
    },
    "C5":{
        "state_rules":{r:{"pos":list(p[0]),"neg":list(p[1])} for r,p in STATE_RULES.items()},
    },
    "C6":{
        "story":MINI_STORY,
        "events":[{"rel":e.rel,"args":list(e.args)} for e in mini_events],
        "final_have":[list(x) for x in sorted(mini_state)],
    },
    "C7":{
        "length_results":LENGTH_RESULTS,
    },
    "end_boss":END_BOSS,
    "interpretation":[
        "Event complexity can increase by composing small previously learned U rather than replacing them with larger sentence-specific rules.",
        "A 4-role GIVE/OPEN frame is built from frozen base role-U plus a separately learned adjunct-U.",
        "Only after local event/role U are stable does the curriculum increase discourse length.",
        "The same U set handles 2, 4, 8, and 16 sentence controlled stories without new language rules.",
        "Longer text therefore tests memory/composition rather than simultaneously introducing new syntax.",
        "The Grimm diagnostic remains partially solved: the first girl command/response works; mother reporting inversion and the magical stop formula remain future curriculum stages."
    ],
    "caveats":[
        "This remains controlled symbolic German with supplied morphology/lexicon.",
        "The 16-sentence scaling story repeats a known structural vocabulary; it is a composition/memory test, not broad linguistic generalization.",
        "Adjunct learning uses supervised semantic role labels LOCATION/INSTRUMENT.",
        "State-effect learning is supervised by state deltas.",
        "Syncretism and richer German prepositional case are still separate future curriculum stages."
    ]
}
Path("/mnt/data/symbolic_v63_kindergarten_to_long_text_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
with Path("/mnt/data/symbolic_v63_kindergarten_to_long_text_checks.csv").open(
    "w",newline="",encoding="utf-8"
) as f:
    w=csv.writer(f); w.writerow(["check","passed"])
    for k,val in checks.items(): w.writerow([k,val])

print("\nSaved v6.3 report/checks.")

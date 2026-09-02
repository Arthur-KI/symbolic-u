
from dataclasses import dataclass, field
from enum import IntEnum
from collections import defaultdict, deque
from typing import Tuple, List, Dict, Set, Optional
import re

# ============================================================
# Symbolic Mini-LM v1.1 — Dictionary/Ontology experiment
# Goal:
#   1) symbolic descriptions reduce bad Text-U candidates
#   2) unseen lemma can generalize through semantic descriptors
#   3) true lexical ambiguity stays UNKNOWN when context is weak
# ============================================================

class T(IntEnum):
    FALSE=-1
    UNKNOWN=0
    TRUE=1

def tn(x):
    return {T.TRUE:"+1",T.UNKNOWN:"0",T.FALSE:"-1"}[T(x)]

@dataclass(frozen=True)
class Atom:
    rel:str
    args:Tuple[str,...]
    def __str__(self):
        return f"{self.rel}({', '.join(self.args)})"

@dataclass(frozen=True)
class Sense:
    sid:str
    features:frozenset
    symbolic:Tuple[Atom,...]=()

@dataclass
class Lexeme:
    lemma:str
    forms:Set[str]
    pos:str
    senses:List[Sense]

class SymbolicDictionary:
    def __init__(self):
        self.form_index:Dict[str,Lexeme]={}
        self.parents=defaultdict(set)
        self.entries=[]

    def add(self,entry:Lexeme):
        self.entries.append(entry)
        for f in entry.forms|{entry.lemma}:
            self.form_index[f.lower()]=entry

    def add_is_a(self,child,parent):
        self.parents[child].add(parent)

    def ancestors(self,x):
        out=set()
        q=deque([x])
        while q:
            a=q.popleft()
            for p in self.parents[a]:
                if p not in out:
                    out.add(p);q.append(p)
        return out

    def lookup(self,w):
        return self.form_index.get(w.lower())

    def sense_features(self,w):
        e=self.lookup(w)
        return [set(s.features) for s in e.senses] if e else []

# ------------------------------------------------------------
# Dictionary contents
# ------------------------------------------------------------

D=SymbolicDictionary()

# ontology
D.add_is_a("WOLF","ANIMAL")
D.add_is_a("ANIMAL","LIVING_ENTITY")
D.add_is_a("PERSON","LIVING_ENTITY")
D.add_is_a("HOUSE","PLACE")
D.add_is_a("GARAGE","PLACE")
D.add_is_a("BENCH","OBJECT")
D.add_is_a("FINANCIAL_INSTITUTION","ORGANIZATION")

# nouns / names
for name in ["Paul","Anna","Ben","Karl","Mia"]:
    D.add(Lexeme(
        name.lower(),{name,name.lower()},"PROPN",
        [Sense(f"{name.lower()}:person",frozenset({"ENTITY","PERSON"}),
               (Atom("IS_A",(name.lower(),"PERSON")),))]
    ))

D.add(Lexeme(
    "wolf",{"Wolf","wolf"},"NOUN",
    [Sense("wolf:animal",frozenset({"ENTITY","ANIMAL","LIVING"}),
           (Atom("IS_A",("wolf","WOLF")),
            Atom("CAN",("wolf","MOVE")),
            Atom("CAN",("wolf","SEE")),
            Atom("CAN",("wolf","EAT"))))]
))

D.add(Lexeme(
    "haus",{"Haus","haus"},"NOUN",
    [Sense("haus:place",frozenset({"ENTITY","PLACE","CONTAINER_PLACE"}),
           (Atom("IS_A",("house","HOUSE")),))]
))
D.add(Lexeme(
    "garage",{"Garage","garage"},"NOUN",
    [Sense("garage:place",frozenset({"ENTITY","PLACE","CONTAINER_PLACE"}),
           (Atom("IS_A",("garage","GARAGE")),))]
))

# SEE verbs
for lemma,forms in [
    ("sehen",{"sieht","sehen"}),
    ("beobachten",{"beobachtet","beobachten"}),
]:
    D.add(Lexeme(
        lemma,forms,"VERB",
        [Sense(f"{lemma}:see",frozenset({"ACTION","PERCEPTION","RELATION_CUE","SEE_ACTION"}),
               (Atom("POSSIBLE_RELATION",(lemma,"SEE")),
                Atom("PORT_TYPE",(lemma,"1","LIVING_ENTITY")),
                Atom("PORT_TYPE",(lemma,"2","LIVING_ENTITY"))))]
    ))

# movement verbs share description. "marschiert" is never used in training.
for lemma,forms,manner in [
    ("gehen",{"geht","gehen"},"WALK"),
    ("laufen",{"läuft","laufen"},"RUN"),
    ("marschieren",{"marschiert","marschieren"},"MARCH"),
]:
    D.add(Lexeme(
        lemma,forms,"VERB",
        [Sense(f"{lemma}:move",frozenset({"ACTION","MOVE_ACTION","MOTION"}),
               (Atom("ACTION_CLASS",(lemma,"MOVE")),
                Atom("MANNER",(lemma,manner)),
                Atom("PORT_TYPE",(lemma,"actor","LIVING_ENTITY"))))]
    ))

# path cues: semantic description rather than full sentence rule
D.add(Lexeme(
    "in",{"in","ins"},"ADP",
    [Sense("in:path_inside",frozenset({"PATH_CUE","TO_INSIDE"}),
           (Atom("PATH_CHANGE",("OUTSIDE","INSIDE")),))]
))

# WITH is a relation cue / phrase boundary
D.add(Lexeme(
    "mit",{"mit"},"ADP",
    [Sense("mit:association",frozenset({"RELATION_CUE","WITH_CUE","PHRASE_BOUNDARY"}),
           (Atom("POSSIBLE_RELATION",("mit","WITH")),))]
))

# sitting/on for bank disambiguation
D.add(Lexeme(
    "sitzen",{"sitzt","sitzen"},"VERB",
    [Sense("sitzen:posture",frozenset({"POSTURE","SIT_ACTION"}),
           (Atom("PREFERS_OBJECT_CLASS",("SIT_ACTION","SEAT")),))]
))
D.add(Lexeme(
    "auf",{"auf"},"ADP",
    [Sense("auf:surface",frozenset({"SURFACE_REL","LOCATION_CUE"}))]
))
D.add(Lexeme(
    "überweisen",{"überweist","überweisen"},"VERB",
    [Sense("ueberweisen:finance",frozenset({"FINANCE_ACTION","TRANSFER_MONEY"}),
           (Atom("PREFERS_OBJECT_CLASS",("TRANSFER_MONEY","FINANCIAL_INSTITUTION")),))]
))
D.add(Lexeme(
    "an",{"an"},"ADP",
    [Sense("an:target",frozenset({"TARGET_CUE"}))]
))

# ambiguous bank
D.add(Lexeme(
    "bank",{"Bank","bank"},"NOUN",
    [
        Sense("bank:finance",
              frozenset({"ENTITY","ORGANIZATION","FINANCIAL_INSTITUTION"}),
              (Atom("IS_A",("bank_finance","FINANCIAL_INSTITUTION")),)),
        Sense("bank:bench",
              frozenset({"ENTITY","OBJECT","SEAT","BENCH"}),
              (Atom("IS_A",("bank_bench","BENCH")),)),
    ]
))

# ------------------------------------------------------------
# Primitive token analysis
# ------------------------------------------------------------

@dataclass
class Tok:
    i:int
    surface:str
    entry:Optional[Lexeme]
    sense_features:List[Set[str]]

def analyze(text):
    ws=re.findall(r"[A-Za-zÄÖÜäöüß]+",text)
    return [Tok(i,w,D.lookup(w),D.sense_features(w)) for i,w in enumerate(ws)]

def has_any(tok,feature):
    return any(feature in fs for fs in tok.sense_features)

def entity_type_options(tok):
    opts=set()
    for fs in tok.sense_features:
        if "PERSON" in fs: opts.add("PERSON")
        if "ANIMAL" in fs: opts.add("ANIMAL")
        if "PLACE" in fs: opts.add("PLACE")
        if "FINANCIAL_INSTITUTION" in fs: opts.add("FINANCIAL_INSTITUTION")
        if "BENCH" in fs: opts.add("BENCH")
    return opts

def canon_entity(tok):
    lw=tok.surface.lower()
    return {"haus":"house","garage":"garage"}.get(lw,lw)

# ============================================================
# TEST 1: Minimal baseline vs semantic dictionary for SEE
# ============================================================

@dataclass(frozen=True)
class Cand:
    fact:Atom
    state:T
    why:str

def baseline_see_candidates(text):
    toks=analyze(text)
    see=[t for t in toks if has_any(t,"SEE_ACTION")]
    people=[t for t in toks if "PERSON" in entity_type_options(t)]
    out=[]
    for v in see:
        left=[p for p in people if p.i<v.i]
        right=[p for p in people if p.i>v.i]
        for a in left:
            for b in right:
                out.append(Cand(Atom("SEE",(canon_entity(a),canon_entity(b))),T.UNKNOWN,
                                "baseline: PERSON left + PERSON right"))
    return out

def enhanced_see_candidates(text):
    toks=analyze(text)
    out=[]
    for v in [t for t in toks if has_any(t,"SEE_ACTION")]:
        left=[t for t in toks if t.i<v.i and ("PERSON" in entity_type_options(t) or "ANIMAL" in entity_type_options(t))]
        right=[t for t in toks if t.i>v.i and ("PERSON" in entity_type_options(t) or "ANIMAL" in entity_type_options(t))]
        if not left: continue
        actor=max(left,key=lambda t:t.i)

        for obj in right:
            between=[x for x in toks if v.i < x.i < obj.i]
            blocked=any(has_any(x,"PHRASE_BOUNDARY") for x in between)
            if blocked:
                out.append(Cand(Atom("SEE",(canon_entity(actor),canon_entity(obj))),T.UNKNOWN,
                                "relation cue/phrase boundary intervenes"))
            else:
                # learned generic SEE-U: actor left, nearest living object before another relation cue
                out.append(Cand(Atom("SEE",(canon_entity(actor),canon_entity(obj))),T.TRUE,
                                "semantic Text-U: SEE_ACTION + living actor/object, no boundary"))
                # Do not stop here: later candidates across a relation cue must
                # remain explicitly represented as U=0 rather than disappearing.
    return out

# ============================================================
# TEST 2: learned semantic descriptor U for motion -> ENTER
# ============================================================

@dataclass
class MotionPatternStat:
    support:int=0
    conflict:int=0
    @property
    def state(self):
        if self.support>=2 and self.conflict==0:return T.TRUE
        if self.conflict>=2 and self.support==0:return T.FALSE
        return T.UNKNOWN

def semantic_motion_signature(text):
    toks=analyze(text)
    moves=[t for t in toks if has_any(t,"MOVE_ACTION")]
    paths=[t for t in toks if has_any(t,"TO_INSIDE")]
    actors=[t for t in toks if "PERSON" in entity_type_options(t) or "ANIMAL" in entity_type_options(t)]
    places=[t for t in toks if "PLACE" in entity_type_options(t)]
    if not moves or not paths or not actors or not places:
        return None
    m=moves[0]
    left=[a for a in actors if a.i<m.i]
    right_places=[p for p in places if p.i>m.i]
    if not left or not right_places:
        return None
    return canon_entity(max(left,key=lambda t:t.i)), canon_entity(min(right_places,key=lambda t:t.i))

motion_train=[
    ("Paul geht ins Haus.",Atom("ENTER",("paul","house"))),
    ("Anna läuft in die Garage.",Atom("ENTER",("anna","garage"))),
]

motion_stat=MotionPatternStat()
for text,gold in motion_train:
    sig=semantic_motion_signature(text)
    if sig and Atom("ENTER",sig)==gold:
        motion_stat.support+=1
    elif sig:
        motion_stat.conflict+=1

def infer_motion_enter(text):
    sig=semantic_motion_signature(text)
    if sig is None:
        return Cand(Atom("ENTER",("?","?")),T.UNKNOWN,"no semantic motion pattern")
    return Cand(
        Atom("ENTER",sig),
        motion_stat.state,
        f"learned on descriptors MOVE_ACTION + TO_INSIDE; S={motion_stat.support},C={motion_stat.conflict}"
    )

# ============================================================
# TEST 3: ambiguous lexical sense
# ============================================================

@dataclass
class SenseEval:
    sid:str
    state:T
    score:int
    why:str

def bank_senses(text):
    toks=analyze(text)
    banktok=next((t for t in toks if t.surface.lower()=="bank"),None)
    if not banktok:return []
    context=set()
    for t in toks:
        for fs in t.sense_features:
            context |= fs

    out=[]
    entry=banktok.entry
    for s in entry.senses:
        score=0
        reasons=[]
        if "SEAT" in s.features and "SIT_ACTION" in context:
            score+=2; reasons.append("SIT_ACTION prefers SEAT")
        if "FINANCIAL_INSTITUTION" in s.features and "TRANSFER_MONEY" in context:
            score+=2; reasons.append("TRANSFER_MONEY prefers FINANCIAL_INSTITUTION")

        # Strong opposite-context conflict.
        if "SEAT" in s.features and "TRANSFER_MONEY" in context:
            score-=2; reasons.append("finance context conflicts with SEAT")
        if "FINANCIAL_INSTITUTION" in s.features and "SIT_ACTION" in context:
            score-=2; reasons.append("sitting context conflicts with finance sense")

        state=T.TRUE if score>=2 else T.FALSE if score<=-2 else T.UNKNOWN
        out.append(SenseEval(s.sid,state,score,", ".join(reasons) or "no decisive context"))
    return out

# ============================================================
# Ontology-description probes
# ============================================================

def wolf_description():
    e=D.lookup("Wolf")
    return e.senses[0].symbolic

# ============================================================
# RUN TESTS
# ============================================================

results=[]

def ok(name,cond,detail=""):
    results.append(bool(cond))
    print(f"{'PASS' if cond else 'FAIL'} | {name}")
    if detail:
        print("   ",detail)

print("=== SYMBOLIC DICTIONARY SAMPLE ===")
for a in wolf_description():
    print(" ",a)
print(" WOLF ancestors:",sorted(D.ancestors("WOLF")))

print("\n=== TEST 1: LESS TEXT-U NONSENSE ===")
txt="Paul sieht Anna mit Ben."
base=baseline_see_candidates(txt)
enh=enhanced_see_candidates(txt)
print("baseline:")
for c in base: print(" ",tn(c.state),c.fact,"|",c.why)
print("enhanced:")
for c in enh: print(" ",tn(c.state),c.fact,"|",c.why)

base_sem={c.fact for c in base}
enh_true={c.fact for c in enh if c.state==T.TRUE}
enh_zero={c.fact for c in enh if c.state==T.UNKNOWN}

ok("baseline generates SEE(Paul,Ben) candidate",
   Atom("SEE",("paul","ben")) in base_sem)
ok("enhanced confirms only SEE(Paul,Anna)",
   enh_true=={Atom("SEE",("paul","anna"))},
   f"confirmed={sorted(map(str,enh_true))}")
ok("Ben path is preserved as 0, not falsely true",
   Atom("SEE",("paul","ben")) in enh_zero)

print("\n=== TEST 2: UNSEEN LEMMA VIA SYMBOLIC DESCRIPTION ===")
for text in ["Paul geht ins Haus.","Anna läuft in die Garage.","Ben marschiert ins Haus."]:
    c=infer_motion_enter(text)
    print(" ",text,"=>",tn(c.state),c.fact,"|",c.why)

unseen=infer_motion_enter("Ben marschiert ins Haus.")
ok("marschiert was never in training examples",
   all("marschiert" not in t for t,_ in motion_train))
ok("unseen lemma generalizes through MOVE_ACTION descriptor",
   unseen.state==T.TRUE and unseen.fact==Atom("ENTER",("ben","house")))

print("\n=== TEST 3: AMBIGUOUS BANK ===")
weak=bank_senses("Die Bank ist alt.")
seat=bank_senses("Anna sitzt auf der Bank.")
fin=bank_senses("Anna überweist an die Bank.")

for label,vals in [("weak",weak),("seat",seat),("finance",fin)]:
    print(label)
    for x in vals:
        print(" ",x.sid,tn(x.state),"score",x.score,"|",x.why)

ok("weak context leaves BOTH Bank senses at 0",
   all(x.state==T.UNKNOWN for x in weak))
ok("sitting context selects bench sense",
   next(x for x in seat if x.sid=="bank:bench").state==T.TRUE and
   next(x for x in seat if x.sid=="bank:finance").state==T.FALSE)
ok("finance context selects finance sense",
   next(x for x in fin if x.sid=="bank:finance").state==T.TRUE and
   next(x for x in fin if x.sid=="bank:bench").state==T.FALSE)

print("\n=== TEST 4: DESCRIPTION / ONTOLOGY IS ACTUALLY SYMBOLIC ===")
wolf_atoms=set(wolf_description())
ok("Wolf carries symbolic CAN(MOVE)",
   Atom("CAN",("wolf","MOVE")) in wolf_atoms)
ok("ontology derives WOLF -> LIVING_ENTITY transitively",
   "LIVING_ENTITY" in D.ancestors("WOLF"))

print(f"\nPassed {sum(results)}/{len(results)}")
assert all(results)
print("ALL v1.1 ASSERTIONS PASSED")

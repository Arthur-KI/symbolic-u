
from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from collections import defaultdict, deque
from typing import Tuple, List, Dict, Set, Optional
import re

# ============================================================
# Symbolic Mini-LM v1.2
# Integrated local end-to-end stress test
# Dictionary/Ontology + Text-U + Pending-U + Story/Time + Backward Solver
# ============================================================

class T(IntEnum):
    FALSE=-1
    UNKNOWN=0
    TRUE=1

def tn(x): return {T.TRUE:"+1", T.UNKNOWN:"0", T.FALSE:"-1"}[T(x)]

@dataclass(frozen=True)
class Prop:
    rel:str
    args:Tuple[str,...]
    polarity:int=1
    def opposite(self): return Prop(self.rel,self.args,-self.polarity)
    def __str__(self):
        return ("" if self.polarity>0 else "NOT ")+f"{self.rel}({', '.join(self.args)})"

@dataclass(frozen=True)
class Atom:
    rel:str
    args:Tuple[str,...]

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

class Dictionary:
    def __init__(self):
        self.form_index={}
        self.parents=defaultdict(set)
    def add(self,e:Lexeme):
        for f in e.forms|{e.lemma}: self.form_index[f.lower()]=e
    def add_is_a(self,a,b): self.parents[a].add(b)
    def lookup(self,w): return self.form_index.get(w.lower())
    def ancestors(self,x):
        out=set(); q=deque([x])
        while q:
            a=q.popleft()
            for p in self.parents[a]:
                if p not in out: out.add(p); q.append(p)
        return out

D=Dictionary()
D.add_is_a("WOLF","ANIMAL"); D.add_is_a("ANIMAL","LIVING_ENTITY")
D.add_is_a("PERSON","LIVING_ENTITY"); D.add_is_a("HOUSE","PLACE")
D.add_is_a("GARAGE","PLACE"); D.add_is_a("BENCH","OBJECT")
D.add_is_a("FINANCIAL_INSTITUTION","ORGANIZATION")

# names/entities
for n in ["Anna","Ben","Karl","Mia","Paul","Rotkäppchen"]:
    canon={"Rotkäppchen":"red"}.get(n,n.lower())
    D.add(Lexeme(canon,{n,n.lower()},"PROPN",[
        Sense(canon+":person",frozenset({"ENTITY","PERSON","LIVING_ENTITY"}),(Atom("IS_A",(canon,"PERSON")),))
    ]))
D.add(Lexeme("wolf",{"Wolf","wolf"},"NOUN",[
    Sense("wolf:animal",frozenset({"ENTITY","ANIMAL","LIVING_ENTITY"}),
          (Atom("IS_A",("wolf","WOLF")),Atom("CAN",("wolf","MOVE")),Atom("CAN",("wolf","SEE"))))
]))
D.add(Lexeme("haus",{"Haus","haus"},"NOUN",[
    Sense("house:place",frozenset({"ENTITY","PLACE","CONTAINER_PLACE"}),(Atom("IS_A",("house","HOUSE")),))
]))
D.add(Lexeme("garage",{"Garage","garage"},"NOUN",[
    Sense("garage:place",frozenset({"ENTITY","PLACE","CONTAINER_PLACE"}),(Atom("IS_A",("garage","GARAGE")),))
]))

# verbs/cues
for lemma,forms,features,atoms in [
    ("sehen",{"sieht","sehen"},{"ACTION","SEE_ACTION","RELATION_CUE"},
     (Atom("POSSIBLE_RELATION",("sehen","SEE")),Atom("PORT_TYPE",("sehen","1","LIVING_ENTITY")),Atom("PORT_TYPE",("sehen","2","LIVING_ENTITY")))),
    ("beobachten",{"beobachtet","beobachten"},{"ACTION","SEE_ACTION","RELATION_CUE"},
     (Atom("POSSIBLE_RELATION",("beobachten","SEE")),)),
    ("betreten",{"betritt","betreten"},{"ACTION","ENTER_ACTION","MOVE_ACTION"},
     (Atom("POSSIBLE_RELATION",("betreten","ENTER")),)),
    ("verlassen",{"verlässt","verlassen"},{"ACTION","LEAVE_ACTION","MOVE_ACTION"},
     (Atom("POSSIBLE_RELATION",("verlassen","LEAVE")),)),
    ("gehen",{"geht","gehen"},{"ACTION","MOVE_ACTION","MOTION"},
     (Atom("ACTION_CLASS",("gehen","MOVE")),)),
    ("laufen",{"läuft","laufen"},{"ACTION","MOVE_ACTION","MOTION"},
     (Atom("ACTION_CLASS",("laufen","MOVE")),)),
    ("marschieren",{"marschiert","marschieren"},{"ACTION","MOVE_ACTION","MOTION"},
     (Atom("ACTION_CLASS",("marschieren","MOVE")),)),
    ("sitzen",{"sitzt","sitzen"},{"POSTURE","SIT_ACTION"},
     (Atom("PREFERS_OBJECT_CLASS",("SIT_ACTION","SEAT")),)),
    ("überweisen",{"überweist","überweisen"},{"FINANCE_ACTION","TRANSFER_MONEY"},
     (Atom("PREFERS_OBJECT_CLASS",("TRANSFER_MONEY","FINANCIAL_INSTITUTION")),)),
]:
    D.add(Lexeme(lemma,forms,"VERB",[Sense(lemma+":sense",frozenset(features),atoms)]))

for lemma,forms,features,atoms in [
    ("in",{"in","ins"},{"PATH_CUE","TO_INSIDE"},(Atom("PATH_CHANGE",("OUTSIDE","INSIDE")),)),
    ("mit",{"mit"},{"RELATION_CUE","WITH_CUE","PHRASE_BOUNDARY"},(Atom("POSSIBLE_RELATION",("mit","WITH")),)),
    ("auf",{"auf"},{"LOCATION_CUE","SURFACE_REL"},()),
    ("an",{"an"},{"TARGET_CUE"},()),
    ("danach",{"Danach","danach"},{"AFTER_CUE"},()),
    ("später",{"Später","später"},{"AFTER_CUE"},()),
    ("bevor",{"Bevor","bevor"},{"BEFORE_CUE"},()),
]:
    D.add(Lexeme(lemma,forms,"ADP",[Sense(lemma+":sense",frozenset(features),atoms)]))

D.add(Lexeme("bank",{"Bank","bank"},"NOUN",[
    Sense("bank:finance",frozenset({"ENTITY","FINANCIAL_INSTITUTION","ORGANIZATION"}),
          (Atom("IS_A",("bank_finance","FINANCIAL_INSTITUTION")),)),
    Sense("bank:bench",frozenset({"ENTITY","BENCH","SEAT","OBJECT"}),
          (Atom("IS_A",("bank_bench","BENCH")),)),
]))

FUNCTION_WORDS={"der","die","das","den","dem","ein","eine","einen","ist","zur","zum"}

@dataclass
class Tok:
    i:int; surface:str; entry:Optional[Lexeme]; features:Set[str]; sense_features:List[Set[str]]

def analyze(text):
    ws=re.findall(r"[A-Za-zÄÖÜäöüß0-9]+",text)
    out=[]
    for i,w in enumerate(ws):
        e=D.lookup(w)
        sfs=[set(s.features) for s in e.senses] if e else []
        fs=set().union(*sfs) if sfs else set()
        out.append(Tok(i,w,e,fs,sfs))
    return out

def canon(tok:Tok):
    lw=tok.surface.lower()
    return {"rotkäppchen":"red","haus":"house","garage":"garage"}.get(lw,lw)

def type_options(tok:Tok):
    out=set()
    for fs in tok.sense_features:
        for t in ["PERSON","ANIMAL","PLACE","FINANCIAL_INSTITUTION","BENCH"]:
            if t in fs: out.add(t)
    return out

def is_living(tok): return bool(type_options(tok)&{"PERSON","ANIMAL"})
def is_place(tok): return "PLACE" in type_options(tok)

# ============================================================
# Text-U
# ============================================================

@dataclass(frozen=True)
class TextCand:
    prop:Prop
    state:T
    source:str
    why:str

def parse_event_candidates(sentence:str)->List[TextCand]:
    toks=analyze(sentence)
    out=[]

    # SEE supports learned SVO and VSO variants.
    # A WITH phrase after the first object creates later U=0 alternatives.
    for v in [t for t in toks if "SEE_ACTION" in t.features]:
        living=[x for x in toks if is_living(x)]
        left=[x for x in living if x.i<v.i]
        right=[x for x in living if x.i>v.i]

        actor=None; objects=[]
        if left and right:  # SVO
            actor=max(left,key=lambda x:x.i)
            objects=right
        elif len(right)>=2: # VSO after temporal fronting: sieht der Wolf Red ...
            actor=right[0]
            objects=right[1:]

        if actor:
            confirmed_one=False
            for obj in objects:
                between=[x for x in toks if min(v.i,actor.i)<x.i<obj.i]
                blocked=any("PHRASE_BOUNDARY" in x.features for x in between)
                st=T.UNKNOWN if blocked or confirmed_one else T.TRUE
                why="WITH/phrase boundary" if blocked else ("later SEE alternative" if confirmed_one else "learned SEE role pattern")
                out.append(TextCand(Prop("SEE",(canon(actor),canon(obj))),st,sentence,why))
                if st==T.TRUE:
                    confirmed_one=True

    def parse_change(action_feature, rel):
        for v in [t for t in toks if action_feature in t.features]:
            living=[x for x in toks if is_living(x)]
            places=[x for x in toks if is_place(x)]
            left_l=[x for x in living if x.i<v.i]
            right_l=[x for x in living if x.i>v.i]
            left_p=[x for x in places if x.i<v.i]
            right_p=[x for x in places if x.i>v.i]

            # SVO: Wolf betritt/verlaesst Haus
            if left_l and right_p:
                actor=max(left_l,key=lambda x:x.i)
                place=min(right_p,key=lambda x:x.i)
                out.append(TextCand(Prop(rel,(canon(actor),canon(place))),T.TRUE,sentence,"learned SVO state-change pattern"))
                continue

            # VSO: Danach betritt/verlaesst der Wolf das Haus
            if right_l and right_p:
                actor=min(right_l,key=lambda x:x.i)
                later_places=[p for p in right_p if p.i>actor.i]
                if later_places:
                    place=min(later_places,key=lambda x:x.i)
                    out.append(TextCand(Prop(rel,(canon(actor),canon(place))),T.TRUE,sentence,"learned VSO state-change pattern"))
                    continue

            # SOV: ... der Wolf das Haus verlaesst
            if left_l and left_p:
                actor=min(left_l,key=lambda x:x.i)
                place=max(left_p,key=lambda x:x.i)
                out.append(TextCand(Prop(rel,(canon(actor),canon(place))),T.TRUE,sentence,"learned SOV state-change pattern"))
                continue

            # OVS/fronted object: Das Haus betritt der Wolf -> keep U=0.
            if left_p and right_l:
                actor=min(right_l,key=lambda x:x.i)
                place=max(left_p,key=lambda x:x.i)
                out.append(TextCand(Prop(rel,(canon(actor),canon(place))),T.UNKNOWN,sentence,"object-fronted pending"))
                out.append(TextCand(Prop(rel,(canon(place),canon(actor))),T.UNKNOWN,sentence,"object-fronted inverse pending"))

    parse_change("ENTER_ACTION","ENTER")
    parse_change("LEAVE_ACTION","LEAVE")

    # Semantic motion generalization: MOVE_ACTION + TO_INSIDE => ENTER.
    moves=[t for t in toks if "MOVE_ACTION" in t.features and "ENTER_ACTION" not in t.features and "LEAVE_ACTION" not in t.features]
    path=[t for t in toks if "TO_INSIDE" in t.features]
    living=[t for t in toks if is_living(t)]
    places=[t for t in toks if is_place(t)]
    if moves and path and living and places:
        m=moves[0]
        left=[x for x in living if x.i<m.i]
        rightp=[x for x in places if x.i>m.i]
        if left and rightp:
            out.append(TextCand(
                Prop("ENTER",(canon(max(left,key=lambda x:x.i)),canon(min(rightp,key=lambda x:x.i)))),
                T.TRUE,sentence,"descriptor MOVE_ACTION + TO_INSIDE"
            ))

    return out

# ============================================================
# Lexical ambiguity
# ============================================================

@dataclass(frozen=True)
class SenseResult:
    sid:str
    state:T

def bank_senses(sentence):
    toks=analyze(sentence)
    bank=next((t for t in toks if t.surface.lower()=="bank"),None)
    if not bank:return []
    ctx=set().union(*(t.features for t in toks))
    out=[]
    for s in bank.entry.senses:
        score=0
        if "SEAT" in s.features and "SIT_ACTION" in ctx: score+=2
        if "FINANCIAL_INSTITUTION" in s.features and "TRANSFER_MONEY" in ctx: score+=2
        if "SEAT" in s.features and "TRANSFER_MONEY" in ctx: score-=2
        if "FINANCIAL_INSTITUTION" in s.features and "SIT_ACTION" in ctx: score-=2
        st=T.TRUE if score>=2 else T.FALSE if score<=-2 else T.UNKNOWN
        out.append(SenseResult(s.sid,st))
    return out

# ============================================================
# Local story/time memory
# ============================================================

@dataclass(frozen=True)
class TimePoint:
    story:str
    index:int

@dataclass(frozen=True)
class Event:
    eid:str
    story:str
    time:TimePoint
    prop:Prop
    source:str
    def __str__(self):
        return f"{self.eid}@t{self.time.index}:{self.prop}"

@dataclass
class UInst:
    uid:str
    output:Prop
    state:T
    source:str
    time:TimePoint
    why:str

@dataclass
class Story:
    sid:str
    events:List[Event]=field(default_factory=list)
    confirmed:List[UInst]=field(default_factory=list)
    pending:List[UInst]=field(default_factory=list)
    rejected:List[UInst]=field(default_factory=list)
    sentence_times:Dict[int,TimePoint]=field(default_factory=dict)
    _eid:int=0
    _uid:int=0

    def add_event(self,p,t,source):
        for e in self.events:
            if e.prop==p and e.time==t:return e
        self._eid+=1
        e=Event(f"{self.sid}:e{self._eid}",self.sid,t,p,source)
        self.events.append(e); return e

    def add_u(self,p,st,t,source,why):
        self._uid+=1
        u=UInst(f"{self.sid}:u{self._uid}",p,st,source,t,why)
        (self.confirmed if st==T.TRUE else self.pending if st==T.UNKNOWN else self.rejected).append(u)
        if st==T.TRUE:self.add_event(p,t,source)
        return u

# sentence order is relative time; BEFORE clauses get two ordered sub-times
def ingest_story(sid,text):
    story=Story(sid)
    sents=[s.strip() for s in re.split(r"[.!?]+",text) if s.strip()]
    next_t=1

    for si,s in enumerate(sents):
        low=s.lower()
        if low.startswith("bevor") and "," in s:
            a,b=s.split(",",1)
            a=re.sub(r"^\s*Bevor\s+","",a,flags=re.I).strip()
            b=b.strip()

            # semantic BEFORE: main clause b happens before subordinate a.
            t_b=TimePoint(sid,next_t); next_t+=1
            t_a=TimePoint(sid,next_t); next_t+=1
            story.sentence_times[si]=t_b

            for cand in parse_event_candidates(b):
                story.add_u(cand.prop,cand.state,t_b,b,cand.why)
            for cand in parse_event_candidates(a):
                story.add_u(cand.prop,cand.state,t_a,a,cand.why)
        else:
            t=TimePoint(sid,next_t); next_t+=1
            story.sentence_times[si]=t
            for cand in parse_event_candidates(s):
                story.add_u(cand.prop,cand.state,t,s,cand.why)
    return story

REL_SCHEMA={
    "ENTER":("LIVING","PLACE"),
    "LEAVE":("LIVING","PLACE"),
    "SEE":("LIVING","LIVING"),
}

def arg_class(x):
    e=D.lookup({"red":"Rotkäppchen","house":"Haus","garage":"Garage"}.get(x,x))
    if not e:
        return "UNKNOWN"
    f=set().union(*(set(s.features) for s in e.senses))
    if f&{"PERSON","ANIMAL"}:return "LIVING"
    if "PLACE" in f:return "PLACE"
    return "UNKNOWN"

class Solver:
    def __init__(self,story:Story):
        self.s=story
        self.trace=[]

    def resolve_pending_exact(self,p:Prop):
        candidates=[u for u in self.s.pending if u.output==p]
        for u in list(candidates):
            pending_same=[x for x in self.s.pending if x.source==u.source and x.time==u.time and x.output.rel==p.rel]
            confirmed_same=[x for x in self.s.confirmed if x.source==u.source and x.time==u.time and x.output.rel==p.rel]
            competing=pending_same+confirmed_same
            valid={x.output for x in competing if tuple(arg_class(a) for a in x.output.args)==REL_SCHEMA.get(x.output.rel)}
            if len(valid)==1 and p in valid:
                # promote p, reject pending alternatives
                for x in list(pending_same):
                    self.s.pending.remove(x)
                    if x.output==p:
                        x.state=T.TRUE; self.s.confirmed.append(x); self.s.add_event(p,x.time,x.source)
                    else:
                        x.state=T.FALSE; self.s.rejected.append(x)
                self.trace.append(f"U 0 -> +1 {p} by unique schema binding")
                return True
            else:
                self.trace.append(f"keep U=0 {p}; {len(valid)} schema-valid alternatives")
        return False

    def find_event(self,rel,args):
        xs=[e for e in self.s.events if e.prop==Prop(rel,tuple(args))]
        return xs[0] if xs else None

    def direct(self,p:Prop):
        self.trace=[]
        if any(e.prop==p for e in self.s.events): return T.TRUE
        if any(e.prop==p.opposite() for e in self.s.events): return T.FALSE
        self.resolve_pending_exact(p)
        if any(e.prop==p for e in self.s.events): return T.TRUE
        return T.UNKNOWN

    def at(self,person,place,target_event:Event):
        self.trace=[]
        for rel in ("ENTER","LEAVE"):
            self.resolve_pending_exact(Prop(rel,(person,place)))

        relevant=[e for e in self.s.events
                  if e.prop.rel in {"ENTER","LEAVE"} and e.prop.args==(person,place)
                  and e.time.index<=target_event.time.index]
        if not relevant:
            self.trace.append("no prior proven state-event")
            return T.UNKNOWN
        latest=max(e.time.index for e in relevant)
        last=[e for e in relevant if e.time.index==latest]
        en=[e for e in last if e.prop.rel=="ENTER"]
        le=[e for e in last if e.prop.rel=="LEAVE"]
        if en and le:
            self.trace.append("contradictory same-time state")
            return T.UNKNOWN
        for e in relevant:
            if e.time.index<latest:self.trace.append(f"U -1 stale {e}")
        if en:
            self.trace.append(f"U +1 ENTER->AT from {en[-1]}")
            return T.TRUE
        if le:
            self.trace.append(f"U +1 LEAVE->NOT_AT from {le[-1]}")
            return T.FALSE
        return T.UNKNOWN

    def meet_at(self,a,b,place,target_event):
        A=self.at(a,place,target_event); trA=list(self.trace)
        B=self.at(b,place,target_event); trB=list(self.trace)
        self.trace=trA+trB
        if A==T.TRUE and B==T.TRUE:
            self.trace.append("U +1 AT+AT->MEET")
            return T.TRUE
        if A==T.FALSE or B==T.FALSE:
            self.trace.append("explicit NOT_AT premise")
            return T.FALSE
        return T.UNKNOWN

# ============================================================
# LONGER LOCAL STORY
# ============================================================

story_text = """
Der Wolf geht ins Haus.
Anna sitzt auf der Bank.
Danach sieht der Wolf Rotkäppchen mit Ben.
Später verlässt der Wolf das Haus.
Ben marschiert in die Garage.
Das Haus betritt Rotkäppchen.
Bevor Rotkäppchen das Haus verlässt, betritt der Wolf das Haus.
"""

story=ingest_story("story1",story_text)
solver=Solver(story)

print("=== INGESTED EVENTS ===")
for e in sorted(story.events,key=lambda x:(x.time.index,x.eid)):
    print(" ",e,"|",e.source)

print("\n=== PENDING U ===")
for u in story.pending:
    print(" ",u.uid,"t",u.time.index,u.output,"|",u.why,"|",u.source)

# lexical sense checks
print("\n=== BANK SENSES ===")
bank_sentence="Anna sitzt auf der Bank."
for x in bank_senses(bank_sentence):
    print(" ",x.sid,tn(x.state))

# locate target events
red_enter = solver.find_event("ENTER",("red","house"))
ben_enter = solver.find_event("ENTER",("ben","garage"))
wolf_enter_first = solver.find_event("ENTER",("wolf","house"))
wolf_leave = solver.find_event("LEAVE",("wolf","house"))

results=[]
def check(name,got,exp,trace=False):
    ok=got==exp; results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'} | {name:<54} got={tn(got)} expected={tn(exp)}")
    if trace:
        for z in solver.trace: print("   ",z)
    return got

print("\n=== QUERIES ===")

# 1) semantic dictionary generalization: "geht ins Haus" => ENTER
check("Wolf entered house via MOVE_ACTION+TO_INSIDE",
      T.TRUE if wolf_enter_first else T.UNKNOWN,T.TRUE)

# 2) SEE with WITH barrier
check("Wolf sees Red",
      solver.direct(Prop("SEE",("wolf","red"))),T.TRUE,True)
check("Wolf sees Ben stays unresolved",
      solver.direct(Prop("SEE",("wolf","ben"))),T.UNKNOWN,True)

# 3) unseen motion lemma in training-level sense
check("Ben entered garage via 'marschiert'",
      T.TRUE if ben_enter else T.UNKNOWN,T.TRUE)

# 4) object-fronted pending should resolve by schema at query-time
check("Red entered house from object-fronted syntax",
      solver.direct(Prop("ENTER",("red","house"))),T.TRUE,True)
red_enter = solver.find_event("ENTER",("red","house"))

# 5) temporal state when Red enters after Wolf left
check("Wolf is NOT in house when Red first enters",
      solver.at("wolf","house",red_enter),T.FALSE,True)

# 6) BEFORE sentence: wolf re-enters before Red leaves
# Find latest Wolf ENTER and Red LEAVE from last sentence
wolf_enters=[e for e in story.events if e.prop==Prop("ENTER",("wolf","house"))]
red_leaves=[e for e in story.events if e.prop==Prop("LEAVE",("red","house"))]
latest_wolf_enter=max(wolf_enters,key=lambda e:e.time.index) if wolf_enters else None
red_leave=max(red_leaves,key=lambda e:e.time.index) if red_leaves else None

check("BEFORE created later Wolf re-entry",
      T.TRUE if latest_wolf_enter and latest_wolf_enter.time.index<red_leave.time.index else T.FALSE,
      T.TRUE)

# 7) meeting at Wolf re-entry event: Red should still be inside
check("Red and Wolf meet when Wolf re-enters",
      solver.meet_at("red","wolf","house",latest_wolf_enter),T.TRUE,True)

# 8) after Red leaves, Red no longer inside
check("Red NOT in house at her leave-time",
      solver.at("red","house",red_leave),T.FALSE,True)

# 9) lexical ambiguity
bs=bank_senses("Die Bank ist alt.")
check("Weak Bank context remains unknown",
      T.UNKNOWN if all(x.state==T.UNKNOWN for x in bs) else T.FALSE,T.UNKNOWN)

# 10) Story isolation
other=ingest_story("story2","Der Wolf geht ins Haus. Rotkäppchen geht ins Haus.")
s2=Solver(other)
red2=s2.find_event("ENTER",("red","house"))
check("Story2 has Wolf inside when Red enters",
      s2.at("wolf","house",red2),T.TRUE)
check("Story1 answer remains different",
      solver.at("wolf","house",red_enter),T.FALSE)

# 11) Query shouldn't self-fulfil arbitrary ambiguous same-type syntax
amb=ingest_story("amb","Ben Anna sieht.")
sa=Solver(amb)
check("Ambiguous same-type SEE Anna->Ben remains 0",
      sa.direct(Prop("SEE",("anna","ben"))),T.UNKNOWN)
check("Ambiguous same-type SEE Ben->Anna remains 0",
      sa.direct(Prop("SEE",("ben","anna"))),T.UNKNOWN)

print("\n=== COUNTS ===")
print("events",len(story.events),
      "confirmedU",len(story.confirmed),
      "pendingU",len(story.pending),
      "rejectedU",len(story.rejected))

print(f"\nPassed {sum(results)}/{len(results)}")
assert all(results)
print("ALL v1.2 END-TO-END ASSERTIONS PASSED")


from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import types, sys, re, json, csv

# ============================================================
# v5.0b — Controlled German:
# pronouns + "davon" + transfer roles + temporal order
# + ambiguity + CLAIM references + coordination + ellipsis
#
# The numeric U is reused from v5.0a. No Python +/- is used for proof.
# ============================================================

# Load v5.0a definitions only, not its benchmark body.
src = Path("/mnt/data/symbolic_v50a_unified_structured_text.py").read_text(encoding="utf-8")
prefix = src.split("# ------------------------------------------------------------\n# Execute v5.0a")[0]
v50a = types.ModuleType("v50a_core_for_v50b")
sys.modules[v50a.__name__] = v50a
exec(prefix, v50a.__dict__)

NUM=v50a.NUM
Lexeme=v50a.Lexeme
Dictionary=v50a.Dictionary
Key=v50a.Key
Evidence=v50a.Evidence
AdaptiveVersionedLibrary=v50a.AdaptiveVersionedLibrary
training_examples=v50a.training_examples
frozen_examples=v50a.frozen_examples
extract_relation_tuple=v50a.extract_relation_tuple
split_sentences=v50a.split_sentences

# ------------------------------------------------------------
# Extended symbolic lexicon
# ------------------------------------------------------------

D=Dictionary()

PERSONS = {
    "Anna":("anna","F"),
    "Mia":("mia","F"),
    "Lea":("lea","F"),
    "Paul":("paul","M"),
    "Ben":("ben","M"),
    "Karl":("karl","M"),
}
for surface,(eid,gender) in PERSONS.items():
    D.add(Lexeme(
        eid, frozenset({surface,surface.lower()}),
        frozenset({"ENTITY","PERSON","NAME",f"GENDER_{gender}"}), eid
    ))

for lemma,forms,value in [
    ("apfel",{"Apfel","Äpfel","apfel","äpfel"},"APPLE"),
    ("birne",{"Birne","Birnen","birne","birnen"},"PEAR"),
    ("stein",{"Stein","Steine","stein","steine"},"STONE"),
]:
    D.add(Lexeme(lemma,frozenset(forms),frozenset({"ENTITY_TYPE","COUNTABLE"}),value))

for n,w in v50a.NUMBER_WORDS.items():
    D.add(Lexeme(w,frozenset({w,str(n)}),frozenset({"NUMBER"}),f"N{n}"))

for lemma,forms,features,value in [
    ("haben",{"hat","hatte","haben"}, {"VERB","COUNT_STATE"}, "COUNT_STATE"),
    ("geben",{"gibt","gab","geben"}, {"VERB","TRANSFER"}, "TRANSFER"),
    ("sagen",{"sagt","sagte","sagen"}, {"VERB","SPEECH"}, "CLAIM"),
    ("treffen",{"trifft","traf","treffen"}, {"VERB","MEET"}, "MEET"),
    ("noch",{"noch"}, {"QUERY_CUE"}, None),
    ("danach",{"danach"}, {"TEMP_AFTER"}, None),
    ("vorher",{"vorher"}, {"TEMP_BEFORE"}, None),
    ("davon",{"davon"}, {"ANAPHOR_ITEM"}, None),
    ("weiter",{"weiter"}, {"ELLIPSIS_RECIPIENT"}, None),
    ("an",{"an"}, {"RECIPIENT_MARKER"}, None),
    ("und",{"und"}, {"COORD"}, None),
]:
    D.add(Lexeme(lemma,frozenset(forms),frozenset(features),value))

# pronouns are constraints, not entities
D.add(Lexeme("sie",frozenset({"Sie","sie"}),frozenset({"PRONOUN","PERSON","GENDER_F"}),"PRON_F"))

TOKEN_RE=re.compile(r"[A-Za-zÄÖÜäöüß0-9]+",re.UNICODE)

@dataclass(frozen=True)
class Token:
    surface:str
    lex:Lexeme|None

def tokenize(text):
    return [Token(x,D.lookup(x)) for x in TOKEN_RE.findall(text)]

def indices_with(tokens,feature):
    return [(i,t) for i,t in enumerate(tokens) if t.lex and feature in t.lex.features]

def names(tokens):
    return [(i,t.lex.value) for i,t in enumerate(tokens) if t.lex and "NAME" in t.lex.features]

def items(tokens):
    return [(i,t.lex.value) for i,t in enumerate(tokens) if t.lex and "COUNTABLE" in t.lex.features]

def nums(tokens):
    return [(i,t.lex.value) for i,t in enumerate(tokens) if t.lex and "NUMBER" in t.lex.features]

def has(tokens,feature):
    return any(t.lex and feature in t.lex.features for t in tokens)

# ------------------------------------------------------------
# Mentions / events / explicit unresolved state
# ------------------------------------------------------------

@dataclass(frozen=True)
class Mention:
    entity:str
    gender:str
    context:str
    sentence_index:int

@dataclass
class Event:
    eid:str
    kind:str
    actor:str|None
    recipient:str|None
    item:str|None
    count:str|None
    context:str
    surface_index:int
    time_rank:float
    source:str

@dataclass
class Story:
    story_id:str
    evidence:list[Evidence]=field(default_factory=list)
    mentions:list[Mention]=field(default_factory=list)
    events:list[Event]=field(default_factory=list)
    unresolved:list[dict]=field(default_factory=list)
    last_item_by_context:dict[str,str]=field(default_factory=dict)

    def add_key(self,key,source,rule):
        self.evidence.append(Evidence(key,source,rule))

    def facts(self,context="WORLD",rel=None):
        return {
            e.key.args for e in self.evidence
            if e.key.context==context and (rel is None or e.key.rel==rel)
        }

    def add_mention(self,entity,context,si):
        gender=next((g for _,(eid,g) in PERSONS.items() if eid==entity),"?")
        self.mentions.append(Mention(entity,gender,context,si))

    def candidate_mentions(self,gender,context,si):
        # Same proposition context first; WORLD antecedents are visible inside CLAIM
        allowed={context}
        if context.startswith("CLAIM:"):
            allowed.add("WORLD")
        seen={}
        for m in self.mentions:
            if m.sentence_index>=si or m.context not in allowed or m.gender!=gender:
                continue
            # latest mention per entity
            seen[m.entity]=m
        return sorted(seen.values(),key=lambda m:m.sentence_index,reverse=True)

# ------------------------------------------------------------
# Reference-U helpers
# ------------------------------------------------------------

def resolve_subject(tokens,story,context,si,verb_i):
    explicit=[(i,e) for i,e in names(tokens) if i<verb_i]
    if explicit:
        ent=explicit[-1][1]
        story.add_mention(ent,context,si)
        return ent

    pron=[(i,t) for i,t in indices_with(tokens,"PRONOUN") if i<verb_i]
    if pron:
        # only feminine singular in this benchmark
        cands=story.candidate_mentions("F",context,si)
        if len(cands)==1:
            return cands[0].entity
        story.unresolved.append({
            "type":"REFERENCE",
            "surface":pron[-1][1].surface,
            "context":context,
            "candidates":[m.entity for m in cands],
            "sentence_index":si,
        })
        return None

    # German V2 with a fronted temporal constituent:
    # "Vorher gibt Anna Paul drei Äpfel."
    # "Danach gibt Paul Mia zwei Äpfel."
    if verb_i>0 and (
        (tokens[0].lex and "TEMP_BEFORE" in tokens[0].lex.features) or
        (tokens[0].lex and "TEMP_AFTER" in tokens[0].lex.features)
    ):
        after=[(i,e) for i,e in names(tokens) if i>verb_i]
        if after:
            ent=after[0][1]
            story.add_mention(ent,context,si)
            return ent
    return None

def resolve_item(tokens,story,context):
    its=items(tokens)
    if its:
        item=its[0][1]
        story.last_item_by_context[context]=item
        return item

    if has(tokens,"ANAPHOR_ITEM"):
        candidates=[]
        if context in story.last_item_by_context:
            candidates.append(story.last_item_by_context[context])
        if context.startswith("CLAIM:") and "WORLD" in story.last_item_by_context:
            candidates.append(story.last_item_by_context["WORLD"])
        candidates=list(dict.fromkeys(candidates))
        if len(candidates)==1:
            return candidates[0]
        story.unresolved.append({
            "type":"ITEM_REFERENCE",
            "surface":"davon",
            "context":context,
            "candidates":candidates,
        })
    return None

# ------------------------------------------------------------
# Clause/Event-U parsing
# ------------------------------------------------------------

QUOTE_RE=re.compile(r'[„"“](.+?)[”"“]',re.DOTALL)

def parse_count_clause(sentence,story,context,si):
    toks=tokenize(sentence)
    vis=indices_with(toks,"COUNT_STATE")
    if not vis:
        return False
    vi=vis[0][0]
    actor=resolve_subject(toks,story,context,si,vi)
    item=resolve_item(toks,story,context)
    ns=[x for x in nums(toks) if x[0]>vi]
    if actor and item and ns:
        n=ns[0][1]
        story.add_key(Key("INITIAL_COUNT",(actor,item,n),context),sentence,
                      "COUNT_STATE(actor,item,count)")
        return True
    return False

def parse_meet_clause(sentence,story,context,si):
    toks=tokenize(sentence)
    if not indices_with(toks,"MEET"):
        return False
    for _,ent in names(toks):
        story.add_mention(ent,context,si)
    return True

def transfer_events_from_clause(sentence,story,context,si,last_event):
    toks=tokenize(sentence)
    vis=indices_with(toks,"TRANSFER")
    if not vis:
        return []
    vi=vis[0][0]
    actor=resolve_subject(toks,story,context,si,vi)
    if actor is None:
        return []

    item=resolve_item(toks,story,context)
    ns=[x for x in nums(toks) if x[0]>vi]
    ents_after=[x for x in names(toks) if x[0]>vi]

    # If V2 fronting put the explicit subject after the verb, remove that
    # occurrence from the recipient candidates.
    explicit_before=[x for x in names(toks) if x[0]<vi]
    if not explicit_before and actor is not None:
        removed_subject=False
        filtered=[]
        for pos,ent in ents_after:
            if not removed_subject and ent==actor:
                removed_subject=True
                continue
            filtered.append((pos,ent))
        ents_after=filtered

    # Explicit recipient patterns:
    # "Anna gibt Paul drei Äpfel"
    # "Paul gibt zwei Äpfel an Mia"
    recipient=None
    markers=indices_with(toks,"RECIPIENT_MARKER")
    if markers:
        mi=markers[0][0]
        aft=[e for i,e in ents_after if i>mi]
        if aft: recipient=aft[0]
    elif ents_after:
        recipient=ents_after[0][1]

    events=[]
    base_rank=float(si)
    if has(toks,"TEMP_AFTER") and last_event is not None:
        base_rank=last_event.time_rank+0.5
    elif has(toks,"TEMP_BEFORE") and last_event is not None:
        base_rank=last_event.time_rank-0.5

    # Coordination: "Anna gibt Paul drei Äpfel und Mia zwei."
    coord=indices_with(toks,"COORD")
    if coord and item and len(ns)>=2 and len(ents_after)>=2:
        ci=coord[0][0]
        left_recips=[e for i,e in ents_after if i<ci]
        right_recips=[e for i,e in ents_after if i>ci]
        left_nums=[n for i,n in ns if i<ci]
        right_nums=[n for i,n in ns if i>ci]
        if left_recips and right_recips and left_nums and right_nums:
            for j,(rec,n) in enumerate([
                (left_recips[0],left_nums[0]),
                (right_recips[0],right_nums[0])
            ]):
                ev=Event(
                    f"E{len(story.events)+len(events)+1}","TRANSFER",
                    actor,rec,item,n,context,si,base_rank+j*0.01,sentence
                )
                events.append(ev)
            return events

    # Elliptic "Paul gibt zwei davon weiter." = actor loses count, recipient unknown.
    if has(toks,"ELLIPSIS_RECIPIENT"):
        recipient=None

    if item and ns:
        n=ns[0][1]
        events.append(Event(
            f"E{len(story.events)+1}","TRANSFER",
            actor,recipient,item,n,context,si,base_rank,sentence
        ))
    return events

def parse_story(text,story_id):
    story=Story(story_id)
    last_event=None
    sentences=split_sentences(text)

    for si,sentence in enumerate(sentences):
        toks=tokenize(sentence)
        speech=indices_with(toks,"SPEECH")
        if speech:
            vi=speech[0][0]
            speaker_names=[e for i,e in names(toks) if i<vi]
            speaker=speaker_names[-1] if speaker_names else "UNKNOWN_SPEAKER"
            if speaker!="UNKNOWN_SPEAKER":
                story.add_mention(speaker,"WORLD",si)
            m=QUOTE_RE.search(sentence)
            if m:
                quoted=m.group(1).strip()
                qctx=f"CLAIM:{speaker}"
                # Parse quoted clause with sentence index slightly later than outer antecedent.
                if not parse_count_clause(quoted,story,qctx,si):
                    qevs=transfer_events_from_clause(quoted,story,qctx,si,last_event)
                    story.events.extend(qevs)
                continue

        if parse_meet_clause(sentence,story,"WORLD",si):
            continue
        if parse_count_clause(sentence,story,"WORLD",si):
            continue

        evs=transfer_events_from_clause(sentence,story,"WORLD",si,last_event)
        if evs:
            story.events.extend(evs)
            last_event=sorted(evs,key=lambda e:e.time_rank)[-1]

    # Materialize event Keys and temporal relations.
    for ev in story.events:
        story.add_key(
            Key("TRANSFER",(ev.eid,ev.actor or "?",ev.recipient or "?",ev.item or "?",ev.count or "?"),ev.context),
            ev.source,
            "EVENT_U(TRANSFER actor recipient item count)"
        )
    ordered=sorted([e for e in story.events if e.context=="WORLD"],key=lambda e:(e.time_rank,e.surface_index))
    for a,b in zip(ordered,ordered[1:]):
        story.add_key(Key("BEFORE",(a.eid,b.eid),"WORLD"),
                      f"{a.source} || {b.source}","TEMPORAL_ORDER")
    return story

# ------------------------------------------------------------
# Query parser
# ------------------------------------------------------------

@dataclass(frozen=True)
class Query:
    owner:str
    item:str
    source:str

def parse_query(text):
    toks=tokenize(text)
    ens=names(toks); its=items(toks)
    if has(toks,"QUERY_CUE") and ens and its:
        return Query(ens[-1][1],its[0][1],text)
    return None

# ------------------------------------------------------------
# Numeric U: learn/reuse v5.0a anonymous relation from structured text bank
# ------------------------------------------------------------

def build_numeric_library():
    lib=AdaptiveVersionedLibrary()
    train=[
        extract_relation_tuple(ex,f"v50b-train-{i}")
        for i,ex in enumerate(training_examples())
    ]
    frozen=[
        extract_relation_tuple(ex,f"v50b-frozen-{i}")
        for i,ex in enumerate(frozen_examples())
    ]
    r1=lib.invent_numeric_relation(train,frozen)
    return lib,r1

lib,r1=build_numeric_library()

def solve_sub(old,delta):
    # Find new: R1(old,delta,new)
    w=v50a.v44.num_world(45,[],[],lt=True)
    sols=[]
    for i in range(46):
        z=f"N{i}"
        r1.program.reset()
        if r1.program.prove((old,delta,z),w):
            sols.append(z)
    return sols[0] if len(sols)==1 else None

def solve_add(old,delta):
    # Find new: R1(new,delta,old)
    w=v50a.v44.num_world(45,[],[],lt=True)
    sols=[]
    for i in range(46):
        z=f"N{i}"
        r1.program.reset()
        if r1.program.prove((z,delta,old),w):
            sols.append(z)
    return sols[0] if len(sols)==1 else None

# ------------------------------------------------------------
# State-U over temporally ordered WORLD events
# ------------------------------------------------------------

def answer(story,query):
    if query is None:
        return None

    initial=[
        x for x in story.facts("WORLD","INITIAL_COUNT")
        if x[0]==query.owner and x[1]==query.item
    ]
    if len(initial)!=1:
        return None

    state=initial[0][2]

    events=sorted(
        [e for e in story.events if e.context=="WORLD" and e.item==query.item],
        key=lambda e:(e.time_rank,e.surface_index,e.eid)
    )

    # Need state for all persons because transfers can add then subtract.
    states={}
    for x in story.facts("WORLD","INITIAL_COUNT"):
        owner,item,n=x
        if item==query.item:
            if owner in states and states[owner]!=n:
                return None
            states[owner]=n

    for ev in events:
        if not ev.count or not ev.actor:
            continue
        if ev.actor not in states:
            # no closed-world zero assumption
            return None
        new_actor=solve_sub(states[ev.actor],ev.count)
        if new_actor is None:
            return None
        states[ev.actor]=new_actor

        if ev.recipient:
            if ev.recipient not in states:
                return None
            new_rec=solve_add(states[ev.recipient],ev.count)
            if new_rec is None:
                return None
            states[ev.recipient]=new_rec

    return states.get(query.owner)

# ------------------------------------------------------------
# Tests B1-B8
# ------------------------------------------------------------

# B1/B2/B3/B4-after: pronoun, davon, role ports, chained state, danach.
story_chain=parse_story(
    "Anna hat sieben Äpfel. Paul hat null Äpfel. "
    "Sie gibt Paul drei davon. Mia hat null Äpfel. Paul gibt danach zwei Äpfel an Mia.",
    "chain"
)
q_paul=parse_query("Wie viele Äpfel hat Paul noch?")
q_anna=parse_query("Wie viele Äpfel hat Anna noch?")
q_mia=parse_query("Wie viele Äpfel hat Mia noch?")
ans_paul=answer(story_chain,q_paul)
ans_anna=answer(story_chain,q_anna)
ans_mia=answer(story_chain,q_mia)

# B4-before: surface order differs from temporal order.
story_before=parse_story(
    "Anna hat sieben Äpfel. Paul hat null Äpfel. Mia hat null Äpfel. "
    "Paul gibt zwei Äpfel an Mia. Vorher gibt Anna Paul drei Äpfel.",
    "before"
)
before_paul=answer(story_before,q_paul)
before_mia=answer(story_before,q_mia)

# B5: two compatible female antecedents -> unresolved, no count commit.
story_amb=parse_story(
    "Anna traf Mia. Sie hatte drei Äpfel.",
    "ambiguous"
)
q_amb=parse_query("Wie viele Äpfel hat Anna noch?")
amb_answer=answer(story_amb,q_amb)

# B6: reference and davon inside CLAIM; WORLD unchanged.
story_claim=parse_story(
    'Anna hat elf Äpfel. Paul sagt: „Sie gibt Mia drei davon.“',
    "claim-ref"
)
claim_anna=answer(story_claim,parse_query("Wie viele Äpfel hat Anna noch?"))

# B7: coordination with shared actor/item.
story_coord=parse_story(
    "Anna hat zehn Äpfel. Paul hat null Äpfel. Mia hat null Äpfel. "
    "Anna gibt Paul drei Äpfel und Mia zwei.",
    "coord"
)
coord_anna=answer(story_coord,q_anna)
coord_paul=answer(story_coord,q_paul)
coord_mia=answer(story_coord,q_mia)

# B8: ellipsis recipient unknown but actor decrement is still meaningful.
story_ell=parse_story(
    "Paul hat fünf Äpfel. Paul gibt zwei davon weiter.",
    "ellipsis"
)
ell_answer=answer(story_ell,q_paul)

# Audit event roles and contexts.
chain_world_transfers=[e for e in story_chain.events if e.context=="WORLD"]
claim_transfers=[e for e in story_claim.events if e.context.startswith("CLAIM:")]
coord_transfers=[e for e in story_coord.events if e.context=="WORLD"]

checks={
    "B1_pronoun_sie_resolves_to_anna":(
        len(chain_world_transfers)>=1 and chain_world_transfers[0].actor=="anna"
    ),
    "B1_davon_resolves_item_type":(
        chain_world_transfers[0].item=="APPLE"
    ),
    "B2_transfer_roles_actor_recipient":(
        chain_world_transfers[0].actor=="anna" and
        chain_world_transfers[0].recipient=="paul" and
        chain_world_transfers[1].actor=="paul" and
        chain_world_transfers[1].recipient=="mia"
    ),
    "B3_chained_state_updates":(
        ans_anna=="N4" and ans_paul=="N1" and ans_mia=="N2"
    ),
    "B4_danach_temporal_order":(
        ("E1","E2") in story_chain.facts("WORLD","BEFORE")
    ),
    "B4_vorher_reorders_surface_events":(
        before_paul=="N1" and before_mia=="N2" and
        any(a[0]!="E1" or a[1]!="E2" for a in story_before.facts("WORLD","BEFORE"))
    ),
    "B5_ambiguous_pronoun_stays_unknown":(
        amb_answer is None and
        len(story_amb.unresolved)==1 and
        set(story_amb.unresolved[0]["candidates"])=={"anna","mia"} and
        not story_amb.facts("WORLD","INITIAL_COUNT")
    ),
    "B6_claim_internal_reference_resolves":(
        len(claim_transfers)==1 and
        claim_transfers[0].actor=="anna" and
        claim_transfers[0].recipient=="mia" and
        claim_transfers[0].item=="APPLE"
    ),
    "B6_claim_does_not_change_world_state":(
        claim_anna=="N11" and
        not [e for e in story_claim.events if e.context=="WORLD"]
    ),
    "B7_coordination_creates_two_events":(
        len(coord_transfers)==2 and
        {(e.recipient,e.count) for e in coord_transfers}=={("paul","N3"),("mia","N2")}
    ),
    "B7_coordination_state_correct":(
        coord_anna=="N5" and coord_paul=="N3" and coord_mia=="N2"
    ),
    "B8_ellipsis_actor_state_correct":(
        ell_answer=="N3" and
        len(story_ell.events)==1 and
        story_ell.events[0].recipient is None and
        story_ell.events[0].item=="APPLE"
    ),
    "numeric_u_reused_not_relearned":(
        dict(lib.learn_attempts)=={"R1":1}
    ),
    "wrong_chain_answer_stays_unknown":(
        ans_paul!="N2"
    ),
}

print("=== v5.0b REFERENCE / TRANSFER / TIME / CLAIM ===")
for k,v in checks.items():
    print(("PASS" if v else "FAIL"),"|",k)

print("\nCHAIN events:")
for e in story_chain.events:
    print(" ",e.eid,e.context,e.kind,e.actor,"->",e.recipient,e.item,e.count,"t",e.time_rank)
print("CHAIN answers: Anna",ans_anna,"Paul",ans_paul,"Mia",ans_mia)

print("\nBEFORE surface-reversal events:")
for e in story_before.events:
    print(" ",e.eid,e.actor,"->",e.recipient,e.count,"surface",e.surface_index,"t",e.time_rank)
print(" temporal keys:",story_before.facts("WORLD","BEFORE"))
print(" answers: Paul",before_paul,"Mia",before_mia)

print("\nAMBIGUITY:")
print(" unresolved:",story_amb.unresolved)
print(" answer:",amb_answer)

print("\nCLAIM:")
for e in story_claim.events:
    print(" ",e.context,e.actor,"->",e.recipient,e.item,e.count)
print(" WORLD answer: Anna",claim_anna)

print("\nCOORD:")
for e in story_coord.events:
    print(" ",e.actor,"->",e.recipient,e.count,e.item)
print(" answers:",coord_anna,coord_paul,coord_mia)

print("\nELLIPSIS:")
for e in story_ell.events:
    print(" ",e.actor,"->",e.recipient,e.count,e.item)
print(" answer:",ell_answer)

assert all(checks.values())

report={
    "version":"v5.0b-reference-transfer-time-claim",
    "result":"PASS",
    "scope":"controlled synthetic German; not general NLP accuracy",
    "checks":checks,
    "numeric_u":{
        "version":r1.id,
        "learn_attempts":dict(lib.learn_attempts),
        "base_rule":r1.meta["base_rule"],
        "recursive_rule":r1.meta["recursive_rule"],
    },
    "chain":{
        "events":[e.__dict__ for e in story_chain.events],
        "answers":{"anna":ans_anna,"paul":ans_paul,"mia":ans_mia},
    },
    "before":{
        "events":[e.__dict__ for e in story_before.events],
        "temporal_keys":[list(x) for x in story_before.facts("WORLD","BEFORE")],
        "answers":{"paul":before_paul,"mia":before_mia},
    },
    "ambiguity":{
        "unresolved":story_amb.unresolved,
        "answer":amb_answer,
    },
    "claim":{
        "events":[e.__dict__ for e in story_claim.events],
        "world_answers":{"anna":claim_anna},
    },
    "coordination":{
        "events":[e.__dict__ for e in story_coord.events],
        "answers":{"anna":coord_anna,"paul":coord_paul,"mia":coord_mia},
    },
    "ellipsis":{
        "events":[e.__dict__ for e in story_ell.events],
        "answer":ell_answer,
    },
    "architecture":[
        "Pronouns are resolved by explicit symbolic gender/context constraints; ambiguous compatible antecedents remain unresolved.",
        "Davon is a symbolic item-reference constraint resolved from context-local item salience.",
        "Transfers are Event-U objects with actor, recipient, item, count, context, and explicit temporal rank.",
        "Danach/vorher create symbolic temporal ordering independent of surface sentence order.",
        "State progression applies the already learned anonymous numeric U R1; no Python addition/subtraction is used as the proof function.",
        "Receiving uses R1 backwards: find NEW such that R1(NEW, RECEIVED, OLD).",
        "CLAIM events remain outside WORLD state progression.",
        "Coordination expands to two Event-U instances with shared actor/item.",
        "Ellipsis 'weiter' preserves actor/item/count while leaving recipient unresolved."
    ],
    "caveats":[
        "Pronoun morphology currently covers only feminine singular 'sie'.",
        "Reference salience is a small symbolic context store, not a general discourse model.",
        "Temporal language currently covers sentence order plus explicit danach/vorher; no interval algebra yet.",
        "Transfers require explicit initial counts for both sender and known recipient; there is no closed-world assumption that an unseen recipient starts at zero.",
        "Coordination handles one tested shared-actor/shared-item pattern.",
        "Ellipsis supports an unknown recipient only when the query depends on the sender's remaining state.",
        "This is still controlled German, not Grimm prose."
    ],
}

Path("/mnt/data/symbolic_v50b_reference_transfer_time_claim_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
with Path("/mnt/data/symbolic_v50b_reference_transfer_time_claim_checks.csv").open(
    "w",newline="",encoding="utf-8"
) as f:
    w=csv.writer(f)
    w.writerow(["check","passed"])
    for k,v in checks.items():
        w.writerow([k,v])

print("\nSaved v5.0b report/checks.")

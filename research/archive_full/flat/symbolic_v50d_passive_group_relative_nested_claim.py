
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import types, sys, re, json, csv

# ============================================================
# v5.0d — final controlled-language layer before Grimm
#
# Adds:
#   1) German passive transfer
#   2) plural "sie" / explicit GROUP reference via agreement
#   3) relative-clause boundary
#   4) recursively nested CLAIM contexts
#
# Purely symbolic. Same R1 numeric U as v5.0a-c.
# ============================================================

src=Path("/mnt/data/symbolic_v50c_case_multiitem_claimchain_v2.py").read_text(encoding="utf-8")
prefix=src.split("# ------------------------------------------------------------\n# C1 —")[0]
v50c=types.ModuleType("v50c_core_for_v50d")
sys.modules[v50c.__name__]=v50c
exec(prefix,v50c.__dict__)

# Short aliases.
D=v50c.D
Lexeme=v50c.Lexeme
Key=v50c.Key
Event=v50c.Event
Story=v50c.Story
Query=v50c.Query
lib=v50c.lib
r1=v50c.r1

# ------------------------------------------------------------
# Symbolic morphology / new relation cues
# ------------------------------------------------------------

for lemma,forms,features,value in [
    ("werden",{"werden","wird","wurden"}, {"VERB","PASSIVE_AUX"}, "PASSIVE"),
    ("gegeben",{"gegeben"}, {"VERB_PARTICIPLE","TRANSFER_PARTICIPLE"}, "TRANSFER"),
    ("von",{"von"}, {"FROM_MARKER"}, None),
    ("zusammen",{"zusammen"}, {"GROUP_CUE"}, None),
]:
    D.add(Lexeme(lemma,frozenset(forms),frozenset(features),value))

# German "sie" is syncretic. Do not hard-code singular feminine anymore.
D.add(Lexeme(
    "sie",frozenset({"Sie","sie"}),
    frozenset({"PRONOUN","NOM_PRONOUN","SIE_SYNCRETIC"}),"SIE"
))

# Keep explicit forms from v5.0c.
# er/ihm/ihr already remain in D.

def group_id(members):
    return "GROUP[" + "+".join(sorted(set(members))) + "]"

def ensure_group_store(story):
    if not hasattr(story,"group_mentions"):
        story.group_mentions=[]
    return story.group_mentions

def add_group_mention(story,gid,members,context,si):
    store=ensure_group_store(story)
    rec={"group":gid,"members":tuple(sorted(set(members))),"context":context,"si":si}
    if rec not in store:
        store.append(rec)

def group_candidates(story,context,si):
    allowed={context}
    if context.startswith("CLAIM:"):
        allowed.add("WORLD")
    # Nested claims can see ancestor proposition contexts, but not siblings.
    if ">CLAIM:" in context:
        parts=context.split(">")
        for i in range(1,len(parts)):
            allowed.add(">".join(parts[:i]))
        allowed.add("WORLD")
    return [
        g for g in ensure_group_store(story)
        if g["context"] in allowed and g["si"]<si
    ]

def individual_candidates(story,gender,context,si):
    # Extend v5.0c context visibility to nested claim ancestors.
    allowed={context}
    if context.startswith("CLAIM:"):
        allowed.add("WORLD")
    if ">CLAIM:" in context:
        parts=context.split(">")
        for i in range(1,len(parts)):
            allowed.add(">".join(parts[:i]))
        allowed.add("WORLD")

    seen={}
    for m in story.mentions:
        if m.sentence_index>=si or m.context not in allowed or m.gender!=gender:
            continue
        seen[m.entity]=m
    return sorted(seen.values(),key=lambda m:m.sentence_index,reverse=True)

def verb_agreement(tok_surface):
    s=tok_surface.lower()
    if s in {"geben","haben","werden"}:
        return "PL"
    if s in {"gibt","gab","hat","hatte","wird"}:
        return "SG"
    return None

def token_gender(tok):
    if tok.lex:
        if "GENDER_F" in tok.lex.features: return "F"
        if "GENDER_M" in tok.lex.features: return "M"
    return None

# ------------------------------------------------------------
# Subject Reference-U with agreement.
# ------------------------------------------------------------

def resolve_subject_d(tokens,story,context,si,verb_i):
    explicit_before=[(i,e) for i,e in v50c.v50b.names(tokens) if i<verb_i]
    if explicit_before:
        ent=explicit_before[-1][1]
        story.add_mention(ent,context,si)
        return ent

    verb_tok=tokens[verb_i]
    agreement=verb_agreement(verb_tok.surface)

    nom_prons=v50c.v50b.indices_with(tokens,"NOM_PRONOUN")
    if nom_prons:
        pre=[x for x in nom_prons if x[0]<verb_i]
        chosen=pre[-1] if pre else nom_prons[0]
        tok=chosen[1]

        if tok.lex and "SIE_SYNCRETIC" in tok.lex.features:
            if agreement=="PL":
                gc=group_candidates(story,context,si)
                if len(gc)==1:
                    return gc[0]["group"]
                story.unresolved.append({
                    "type":"REFERENCE","surface":tok.surface,
                    "number":"PL","context":context,
                    "candidates":[g["group"] for g in gc],
                    "sentence_index":si,
                })
                return None
            if agreement=="SG":
                cands=individual_candidates(story,"F",context,si)
                if len(cands)==1:
                    return cands[0].entity
                story.unresolved.append({
                    "type":"REFERENCE","surface":tok.surface,
                    "number":"SG","gender":"F","context":context,
                    "candidates":[m.entity for m in cands],
                    "sentence_index":si,
                })
                return None

        gender=token_gender(tok)
        cands=individual_candidates(story,gender,context,si)
        if len(cands)==1:
            return cands[0].entity
        story.unresolved.append({
            "type":"REFERENCE","surface":tok.surface,
            "case":"NOM","context":context,
            "candidates":[m.entity for m in cands],
            "sentence_index":si,
        })
        return None

    # V2 fronting with explicit post-verbal subject.
    fronted=(
        verb_i>0 and (
            v50c.v50b.nums(tokens[:verb_i]) or
            v50c.v50b.items(tokens[:verb_i]) or
            v50c.v50b.has(tokens[:verb_i],"TEMP_BEFORE") or
            v50c.v50b.has(tokens[:verb_i],"TEMP_AFTER")
        )
    )
    if fronted:
        post=[(i,e) for i,e in v50c.v50b.names(tokens) if i>verb_i]
        if post:
            ent=post[0][1]
            story.add_mention(ent,context,si)
            return ent
    return None

# ------------------------------------------------------------
# Item resolution, now group-aware through ordinary evidence.
# ------------------------------------------------------------

resolve_item_safe=v50c.resolve_item_safe

# ------------------------------------------------------------
# Count clauses including coordinated plural group subject.
# ------------------------------------------------------------

def parse_count_clause_d(sentence,story,context,si):
    toks=v50c.v50b.tokenize(sentence)
    vis=v50c.v50b.indices_with(toks,"COUNT_STATE")
    if not vis:
        return False
    vi=vis[0][0]
    verb=toks[vi].surface.lower()

    its=v50c.v50b.items(toks)
    ns=v50c.v50b.nums(toks)
    if not its or not ns:
        return False
    item=its[0][1]
    n=(next((x for x in ns if x[0]>vi),ns[0]))[1]

    pre_names=[e for i,e in v50c.v50b.names(toks) if i<vi]
    if verb=="haben" and len(pre_names)>=2 and v50c.v50b.has(toks,"COORD"):
        gid=group_id(pre_names)
        for ent in pre_names:
            story.add_mention(ent,context,si)
        add_group_mention(story,gid,pre_names,context,si)
        story.add_key(
            Key("GROUP",(gid,*tuple(sorted(pre_names))),context),
            sentence,"GROUP_U(coordination)"
        )
        story.add_key(
            Key("INITIAL_COUNT",(gid,item,n),context),
            sentence,"COUNT_STATE(group,item,count)"
        )
        story.last_item_by_context[context]=item
        return True

    actor=resolve_subject_d(toks,story,context,si,vi)
    if actor:
        story.last_item_by_context[context]=item
        story.add_key(
            Key("INITIAL_COUNT",(actor,item,n),context),
            sentence,"COUNT_STATE(actor,item,count)"
        )
        return True
    return False

# ------------------------------------------------------------
# Active transfer, using the new subject resolver.
# ------------------------------------------------------------

def resolve_dative_recipient_d(tokens,story,context,si):
    dprons=v50c.v50b.indices_with(tokens,"DAT_PRONOUN")
    if not dprons:
        return None,False
    _,tok=dprons[0]
    gender=v50c.pronoun_gender(tok)
    cands=individual_candidates(story,gender,context,si)
    if len(cands)==1:
        return cands[0].entity,True
    story.unresolved.append({
        "type":"REFERENCE","case":"DAT","surface":tok.surface,
        "context":context,"candidates":[m.entity for m in cands],
        "sentence_index":si,
    })
    return None,True

def transfer_active_d(sentence,story,context,si,last_event):
    toks=v50c.v50b.tokenize(sentence)
    vis=v50c.v50b.indices_with(toks,"TRANSFER")
    if not vis:
        return []
    vi=vis[0][0]
    actor=resolve_subject_d(toks,story,context,si,vi)
    if actor is None:
        return []

    item=resolve_item_safe(toks,story,context)
    ns=v50c.v50b.nums(toks)
    if not item or not ns:
        return []

    recipient,had_dat=resolve_dative_recipient_d(toks,story,context,si)

    ents_after=[x for x in v50c.v50b.names(toks) if x[0]>vi]
    filtered=[]
    removed_actor=False
    for pos,ent in ents_after:
        if not removed_actor and ent==actor:
            removed_actor=True
            continue
        filtered.append((pos,ent))
    ents_after=filtered

    if not had_dat:
        markers=v50c.v50b.indices_with(toks,"RECIPIENT_MARKER")
        if markers:
            mi=markers[0][0]
            aft=[e for i,e in ents_after if i>mi]
            if aft: recipient=aft[0]
        elif ents_after:
            recipient=ents_after[0][1]

    rank=float(si)
    if v50c.v50b.has(toks,"TEMP_AFTER") and last_event is not None:
        rank=last_event.time_rank+0.5
    elif v50c.v50b.has(toks,"TEMP_BEFORE") and last_event is not None:
        rank=last_event.time_rank-0.5

    n=ns[0][1]
    if v50c.v50b.has(toks,"ELLIPSIS_RECIPIENT"):
        recipient=None

    return [Event(
        f"E{len(story.events)+1}","TRANSFER",actor,recipient,item,n,
        context,si,rank,sentence
    )]

# ------------------------------------------------------------
# Passive Clause-U
#
# "Zwei Äpfel werden Paul von Anna gegeben."
# object/count -> transferred theme
# name after von -> actor/source
# remaining name between auxiliary and von -> recipient
# ------------------------------------------------------------

def transfer_passive_d(sentence,story,context,si,last_event):
    toks=v50c.v50b.tokenize(sentence)
    aux=v50c.v50b.indices_with(toks,"PASSIVE_AUX")
    part=v50c.v50b.indices_with(toks,"TRANSFER_PARTICIPLE")
    frm=v50c.v50b.indices_with(toks,"FROM_MARKER")
    if not aux or not part or not frm:
        return []

    ai=aux[0][0]; fi=frm[0][0]
    its=v50c.v50b.items(toks)
    ns=v50c.v50b.nums(toks)
    ens=v50c.v50b.names(toks)
    if not its or not ns or not ens:
        return []

    actors=[e for i,e in ens if i>fi]
    recipients=[e for i,e in ens if ai<i<fi]
    if len(actors)!=1 or len(recipients)!=1:
        story.unresolved.append({
            "type":"PASSIVE_ROLES","context":context,
            "actors":actors,"recipients":recipients,"sentence_index":si,
        })
        return []

    actor=actors[0]
    recipient=recipients[0]
    item=its[0][1]
    n=ns[0][1]
    story.add_mention(actor,context,si)
    story.add_mention(recipient,context,si)
    story.last_item_by_context[context]=item

    rank=float(si)
    return [Event(
        f"E{len(story.events)+1}","TRANSFER",actor,recipient,item,n,
        context,si,rank,sentence
    )]

# ------------------------------------------------------------
# Relative Clause-U
#
# Controlled pattern:
#   "Anna, die sieben Äpfel hat, gibt Paul zwei Äpfel."
# Relative clause and matrix clause become separate clause structures.
# ------------------------------------------------------------

REL_RE=re.compile(
    r"^\s*([A-Za-zÄÖÜäöüß]+)\s*,\s*die\s+(.+?)\s*,\s*(.+?)\s*$",
    re.UNICODE
)

def parse_relative_d(sentence,story,context,si,last_event):
    m=REL_RE.match(sentence.rstrip("."))
    if not m:
        return None
    antecedent=m.group(1)
    rel=m.group(2)
    matrix=m.group(3)

    # Relative German order: "... sieben Äpfel hat"
    rtoks=v50c.v50b.tokenize(rel)
    if not v50c.v50b.indices_with(rtoks,"COUNT_STATE"):
        return None
    its=v50c.v50b.items(rtoks)
    ns=v50c.v50b.nums(rtoks)
    if not its or not ns:
        return None

    # Materialize the relative clause with explicit antecedent.
    synthesized_rel=f"{antecedent} hat {ns[0][1][1:]} {rtoks[its[0][0]].surface}."
    # Avoid numeral surface reconstruction; write Key directly from lexicon values.
    ent_lex=D.lookup(antecedent)
    if ent_lex is None or "NAME" not in ent_lex.features:
        return None
    actor=ent_lex.value
    item=its[0][1]
    n=ns[0][1]
    story.add_mention(actor,context,si)
    story.last_item_by_context[context]=item
    story.add_key(
        Key("INITIAL_COUNT",(actor,item,n),context),
        sentence,
        "RELATIVE_CLAUSE_U(antecedent -> COUNT_STATE)"
    )
    story.add_key(
        Key("RELATIVE_OF",(f"RC{si}",actor),context),
        sentence,
        "RELATIVE_BINDING_U"
    )

    matrix_sentence=f"{antecedent} {matrix}."
    events=transfer_active_d(matrix_sentence,story,context,si,last_event)
    return events

# ------------------------------------------------------------
# Quote-aware sentence splitter and recursive CLAIM parser.
# ------------------------------------------------------------

def split_top_level(text):
    pairs={"„":"“","‚":"‘"}
    stack=[]
    out=[]
    buf=[]
    for ch in text.strip():
        if ch in pairs:
            stack.append(pairs[ch])
            buf.append(ch)
            continue
        if stack and ch==stack[-1]:
            stack.pop()
            buf.append(ch)
            continue
        buf.append(ch)
        if ch in ".!?" and not stack:
            s="".join(buf).strip()
            if s: out.append(s)
            buf=[]
    tail="".join(buf).strip()
    if tail: out.append(tail)
    return out

SPEECH_RE=re.compile(
    r"^\s*([A-Za-zÄÖÜäöüß]+)\s+sagt\s*:\s*([„‚])(.+)([“‘])\s*\.?\s*$",
    re.UNICODE|re.DOTALL
)

def parse_one_clause_d(sentence,story,context,si,last_event):
    rel=parse_relative_d(sentence,story,context,si,last_event)
    if rel is not None:
        return rel
    if parse_count_clause_d(sentence,story,context,si):
        return []
    p=transfer_passive_d(sentence,story,context,si,last_event)
    if p:
        return p
    return transfer_active_d(sentence,story,context,si,last_event)

def parse_speech_recursive(sentence,story,context,si):
    m=SPEECH_RE.match(sentence.strip())
    if not m:
        return False
    speaker_surface=m.group(1)
    content=m.group(3).strip()
    lex=D.lookup(speaker_surface)
    if lex is None or "NAME" not in lex.features:
        return False
    speaker=lex.value
    story.add_mention(speaker,context,si)

    newctx=(f"CLAIM:{speaker}" if context=="WORLD"
            else f"{context}>CLAIM:{speaker}")

    inner_last=None
    for qi,inner in enumerate(split_top_level(content)):
        inner_si=si*100+qi+1
        if parse_speech_recursive(inner,story,newctx,inner_si):
            continue
        events=parse_one_clause_d(inner,story,newctx,inner_si,inner_last)
        if events:
            story.events.extend(events)
            inner_last=events[-1]
    return True

def parse_story_d(text,story_id):
    story=Story(story_id)
    ensure_group_store(story)
    last_world=None

    for si,sentence in enumerate(split_top_level(text)):
        if parse_speech_recursive(sentence,story,"WORLD",si):
            continue
        events=parse_one_clause_d(sentence,story,"WORLD",si,last_world)
        if events:
            story.events.extend(events)
            last_world=events[-1]

    # Materialize events / temporal relations.
    for ev in story.events:
        story.add_key(
            Key("TRANSFER",(
                ev.eid,ev.actor or "?",ev.recipient or "?",
                ev.item or "?",ev.count or "?"
            ),ev.context),
            ev.source,
            "EVENT_U(TRANSFER actor recipient item count)"
        )

    ordered=sorted(
        [e for e in story.events if e.context=="WORLD"],
        key=lambda e:(e.time_rank,e.surface_index,e.eid)
    )
    for a,b in zip(ordered,ordered[1:]):
        story.add_key(
            Key("BEFORE",(a.eid,b.eid),"WORLD"),
            f"{a.source} || {b.source}",
            "TEMPORAL_ORDER"
        )
    return story

# ------------------------------------------------------------
# Query parser, extended for coordinated GROUP owner.
# ------------------------------------------------------------

def parse_query_d(text):
    toks=v50c.v50b.tokenize(text)
    its=v50c.v50b.items(toks)
    ens=v50c.v50b.names(toks)
    if not v50c.v50b.has(toks,"QUERY_CUE") or not its:
        return None
    if len(ens)>=2 and v50c.v50b.has(toks,"COORD"):
        return Query(group_id([e for _,e in ens]),its[0][1],text)
    if ens:
        return Query(ens[-1][1],its[0][1],text)
    return None

answer=v50c.v50b.answer

# ============================================================
# D1 PASSIVE
# ============================================================

passive=parse_story_d(
    "Anna hat fünf Äpfel. Paul hat null Äpfel. "
    "Zwei Äpfel werden Paul von Anna gegeben.",
    "passive"
)
q_anna=parse_query_d("Wie viele Äpfel hat Anna noch?")
q_paul=parse_query_d("Wie viele Äpfel hat Paul noch?")
passive_anna=answer(passive,q_anna)
passive_paul=answer(passive,q_paul)

# ============================================================
# D2 PLURAL "SIE" -> GROUP by verb agreement
# ============================================================

group_story=parse_story_d(
    "Anna und Mia haben zusammen sechs Äpfel. Paul hat null Äpfel. "
    "Sie geben Paul zwei davon.",
    "group"
)
q_group=parse_query_d("Wie viele Äpfel haben Anna und Mia noch?")
group_remaining=answer(group_story,q_group)
group_paul=answer(group_story,q_paul)

# Singular "sie" must NOT resolve to plural group, and Anna/Mia are ambiguous.
group_sg=parse_story_d(
    "Anna und Mia haben zusammen sechs Äpfel. Paul hat null Äpfel. "
    "Sie gibt Paul zwei davon.",
    "group-singular"
)
group_sg_events=[e for e in group_sg.events if e.context=="WORLD"]

# ============================================================
# D3 RELATIVE CLAUSE
# ============================================================

relative=parse_story_d(
    "Paul hat null Äpfel. Anna, die sieben Äpfel hat, gibt Paul zwei Äpfel.",
    "relative"
)
relative_anna=answer(relative,q_anna)
relative_paul=answer(relative,q_paul)

# ============================================================
# D4 NESTED CLAIM
# ============================================================

nested=parse_story_d(
    "Anna hat fünf Äpfel. Mia hat null Äpfel. "
    "Paul sagt: „Mia sagt: ‚Anna gibt Mia zwei Äpfel.‘“",
    "nested"
)
nested_world_anna=answer(nested,q_anna)
q_mia=parse_query_d("Wie viele Äpfel hat Mia noch?")
nested_world_mia=answer(nested,q_mia)
nested_events=[e for e in nested.events if ">CLAIM:" in e.context]

# Direct CLAIM plus nested CLAIM remain distinct.
nested_contexts={e.context for e in nested.events}

# ============================================================
# D5 CLAIM PASSIVE also stays scoped
# ============================================================

claim_passive=parse_story_d(
    "Anna hat fünf Äpfel. Paul hat null Äpfel. "
    "Mia sagt: „Zwei Äpfel werden Paul von Anna gegeben.“",
    "claim-passive"
)
claim_passive_anna=answer(claim_passive,q_anna)
claim_passive_paul=answer(claim_passive,q_paul)
claim_passive_events=[e for e in claim_passive.events if e.context.startswith("CLAIM:")]

# ============================================================
# Checks
# ============================================================

passive_events=[e for e in passive.events if e.context=="WORLD"]
group_events=[e for e in group_story.events if e.context=="WORLD"]
relative_events=[e for e in relative.events if e.context=="WORLD"]

checks={
    "D1_passive_roles_from_structure":(
        len(passive_events)==1 and
        passive_events[0].actor=="anna" and
        passive_events[0].recipient=="paul" and
        passive_events[0].item=="APPLE" and
        passive_events[0].count=="N2"
    ),
    "D1_passive_state_correct":(
        passive_anna=="N3" and passive_paul=="N2"
    ),
    "D2_group_entity_created":(
        any(e.key.rel=="GROUP" for e in group_story.evidence) and
        q_group.owner==group_id(["anna","mia"])
    ),
    "D2_plural_sie_resolves_group_by_agreement":(
        len(group_events)==1 and
        group_events[0].actor==group_id(["anna","mia"]) and
        group_events[0].recipient=="paul"
    ),
    "D2_group_state_correct":(
        group_remaining=="N4" and group_paul=="N2"
    ),
    "D2_singular_sie_does_not_fake_group":(
        len(group_sg_events)==0 and
        any(
            u.get("surface","").lower()=="sie" and u.get("number")=="SG"
            for u in group_sg.unresolved
        )
    ),
    "D3_relative_clause_binds_antecedent":(
        ("anna","APPLE","N7") in relative.facts("WORLD","INITIAL_COUNT") and
        any(e.key.rel=="RELATIVE_OF" for e in relative.evidence)
    ),
    "D3_relative_matrix_clause_separate_event":(
        len(relative_events)==1 and
        relative_events[0].actor=="anna" and
        relative_events[0].recipient=="paul" and
        relative_events[0].count=="N2"
    ),
    "D3_relative_clause_state_correct":(
        relative_anna=="N5" and relative_paul=="N2"
    ),
    "D4_nested_claim_context_created":(
        len(nested_events)==1 and
        nested_events[0].context=="CLAIM:paul>CLAIM:mia"
    ),
    "D4_nested_claim_does_not_leak_world":(
        nested_world_anna=="N5" and
        nested_world_mia=="N0" and
        len([e for e in nested.events if e.context=="WORLD"])==0
    ),
    "D5_passive_inside_claim_parses":(
        len(claim_passive_events)==1 and
        claim_passive_events[0].actor=="anna" and
        claim_passive_events[0].recipient=="paul"
    ),
    "D5_passive_claim_does_not_change_world":(
        claim_passive_anna=="N5" and claim_passive_paul=="N0"
    ),
    "D6_same_numeric_u_reused":(
        dict(lib.learn_attempts)=={"R1":1}
    ),
    "D6_wrong_passive_answer_stays_unknown":(
        passive_paul!="N3"
    ),
}

print("=== v5.0d PASSIVE / GROUP-SIE / RELATIVE / NESTED CLAIM ===")
for k,v in checks.items():
    print(("PASS" if v else "FAIL"),"|",k)

print("\nPASSIVE:")
for e in passive.events:
    print(" ",e.context,e.actor,"->",e.recipient,e.item,e.count)
print(" states Anna",passive_anna,"Paul",passive_paul)

print("\nGROUP:")
print(" query owner:",q_group.owner)
for e in group_story.evidence:
    if e.key.rel in {"GROUP","INITIAL_COUNT"}:
        print(" ",e.key)
for e in group_story.events:
    print(" ",e.context,e.actor,"->",e.recipient,e.item,e.count)
print(" states Group",group_remaining,"Paul",group_paul)
print(" singular-sie unresolved:",group_sg.unresolved)

print("\nRELATIVE:")
for e in relative.evidence:
    print(" ",e.key,"|",e.parser_rule)
print(" states Anna",relative_anna,"Paul",relative_paul)

print("\nNESTED CLAIM:")
for e in nested.events:
    print(" ",e.context,e.actor,"->",e.recipient,e.item,e.count)
print(" WORLD states Anna",nested_world_anna,"Mia",nested_world_mia)

print("\nCLAIM PASSIVE:")
for e in claim_passive.events:
    print(" ",e.context,e.actor,"->",e.recipient,e.item,e.count)
print(" WORLD states Anna",claim_passive_anna,"Paul",claim_passive_paul)

print("\nNumeric U attempts:",dict(lib.learn_attempts))

assert all(checks.values())

report={
    "version":"v5.0d-passive-group-relative-nested-claim",
    "result":"PASS",
    "scope":"controlled synthetic German; not general NLP accuracy",
    "checks":checks,
    "numeric_u":{
        "version":r1.id,
        "attempts":dict(lib.learn_attempts),
        "base_rule":r1.meta["base_rule"],
        "recursive_rule":r1.meta["recursive_rule"],
    },
    "passive":{
        "events":[e.__dict__ for e in passive.events],
        "answers":{"anna":passive_anna,"paul":passive_paul},
    },
    "group":{
        "events":[e.__dict__ for e in group_story.events],
        "unresolved_singular":group_sg.unresolved,
        "answers":{"group":group_remaining,"paul":group_paul},
    },
    "relative":{
        "keys":[str(e.key) for e in relative.evidence],
        "events":[e.__dict__ for e in relative.events],
        "answers":{"anna":relative_anna,"paul":relative_paul},
    },
    "nested_claim":{
        "events":[e.__dict__ for e in nested.events],
        "world_answers":{"anna":nested_world_anna,"mia":nested_world_mia},
    },
    "claim_passive":{
        "events":[e.__dict__ for e in claim_passive.events],
        "world_answers":{"anna":claim_passive_anna,"paul":claim_passive_paul},
    },
    "architecture":[
        "Passive transfer roles are derived from symbolic clause cues: 'von' marks source/actor and the recipient region marks addressee/recipient.",
        "Syncretic German 'sie' is constrained by finite-verb agreement: plural 'geben' may resolve to a GROUP mention, singular 'gibt' may resolve only to a singular feminine entity.",
        "Coordinated names can create a persistent GROUP entity with a shared count state.",
        "Relative clauses create a separate clause Key linked to the antecedent; the matrix clause creates its own Event-U.",
        "Speech parsing is recursive: every nested speaker pushes a new CLAIM context.",
        "Nested or passive CLAIM events never enter WORLD state progression.",
        "All state arithmetic continues to use the same learned anonymous R1 relation."
    ],
    "caveats":[
        "Passive coverage is limited to a controlled 'werden ... von ... gegeben' transfer construction.",
        "Plural-group semantics currently models a shared group inventory, not distribution of items among members.",
        "German 'sie' syncretism is constrained here only by singular/plural verb agreement and discourse candidates.",
        "Relative-clause coverage is limited to one antecedent-linked count-state pattern.",
        "Nested CLAIM parsing is recursive but uses controlled quotation syntax.",
        "No general subordinate-clause grammar, case morphology, or arbitrary German scrambling is claimed.",
        "Still a controlled benchmark; Grimm prose is the next qualitatively different test."
    ],
}

Path("/mnt/data/symbolic_v50d_passive_group_relative_nested_claim_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
with Path("/mnt/data/symbolic_v50d_passive_group_relative_nested_claim_checks.csv").open(
    "w",newline="",encoding="utf-8"
) as f:
    w=csv.writer(f)
    w.writerow(["check","passed"])
    for k,v in checks.items():
        w.writerow([k,v])

print("\nSaved v5.0d report/checks.")

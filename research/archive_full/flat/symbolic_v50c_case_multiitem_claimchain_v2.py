
from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import types, sys, re, json, csv

# ============================================================
# v5.0c — controlled German, harder reference/word-order layer
#
# Adds:
# - nominative pronouns er/sie
# - dative pronouns ihm/ihr
# - multiple active object types
# - safe ambiguity for "davon" and dative references
# - multi-clause local reference chains inside CLAIM
# - V2 object-fronting: "Zwei Äpfel gibt sie ihm."
#
# Numeric reasoning is the SAME anonymous R1 from v5.0a/b.
# ============================================================

src = Path("/mnt/data/symbolic_v50b_reference_transfer_time_claim.py").read_text(encoding="utf-8")
prefix = src.split("# ------------------------------------------------------------\n# Tests B1-B8")[0]
v50b = types.ModuleType("v50b_core_for_v50c")
sys.modules[v50b.__name__] = v50b
exec(prefix, v50b.__dict__)

# Reuse symbols.
D=v50b.D
Lexeme=v50b.Lexeme
Key=v50b.Key
Evidence=v50b.Evidence
Event=v50b.Event
Story=v50b.Story
Query=v50b.Query
PERSONS=v50b.PERSONS
NUM=v50b.NUM
lib=v50b.lib
r1=v50b.r1

# ------------------------------------------------------------
# Extend symbolic morphology/case inventory.
# ------------------------------------------------------------

D.add(Lexeme(
    "er",frozenset({"Er","er"}),
    frozenset({"PRONOUN","NOM_PRONOUN","PERSON","GENDER_M"}),"PRON_M_NOM"
))
# Replace/augment "sie" as nominative feminine.
D.add(Lexeme(
    "sie",frozenset({"Sie","sie"}),
    frozenset({"PRONOUN","NOM_PRONOUN","PERSON","GENDER_F"}),"PRON_F_NOM"
))
D.add(Lexeme(
    "ihm",frozenset({"Ihm","ihm"}),
    frozenset({"PRONOUN","DAT_PRONOUN","PERSON","GENDER_M"}),"PRON_M_DAT"
))
D.add(Lexeme(
    "ihr",frozenset({"Ihr","ihr"}),
    frozenset({"PRONOUN","DAT_PRONOUN","PERSON","GENDER_F"}),"PRON_F_DAT"
))

# ------------------------------------------------------------
# Generic symbolic reference helpers.
# ------------------------------------------------------------

def pronoun_gender(tok):
    if tok.lex is None:
        return None
    if "GENDER_F" in tok.lex.features:
        return "F"
    if "GENDER_M" in tok.lex.features:
        return "M"
    return None

def resolve_nominal_subject(tokens,story,context,si,verb_i):
    explicit_before=[(i,e) for i,e in v50b.names(tokens) if i<verb_i]
    if explicit_before:
        ent=explicit_before[-1][1]
        story.add_mention(ent,context,si)
        return ent

    # Nominative pronoun can occur pre-verb ("Sie gibt ...") or after V2 verb
    # if another constituent is fronted ("Zwei Äpfel gibt sie ...").
    nom_prons=[(i,t) for i,t in v50b.indices_with(tokens,"NOM_PRONOUN")]
    if nom_prons:
        # Prefer a pronoun in the syntactic subject region:
        # pre-verb, otherwise first post-verb pronoun under fronting.
        pre=[x for x in nom_prons if x[0]<verb_i]
        chosen=pre[-1] if pre else nom_prons[0]
        gender=pronoun_gender(chosen[1])
        cands=story.candidate_mentions(gender,context,si)
        if len(cands)==1:
            return cands[0].entity
        story.unresolved.append({
            "type":"REFERENCE",
            "case":"NOM",
            "surface":chosen[1].surface,
            "context":context,
            "candidates":[m.entity for m in cands],
            "sentence_index":si,
        })
        return None

    # V2 fronting with explicit post-verbal subject:
    # "Drei Äpfel gibt Anna Paul."
    fronted = (
        verb_i>0 and (
            v50b.nums(tokens[:verb_i]) or
            v50b.items(tokens[:verb_i]) or
            v50b.has(tokens[:verb_i],"TEMP_BEFORE") or
            v50b.has(tokens[:verb_i],"TEMP_AFTER")
        )
    )
    if fronted:
        post=[(i,e) for i,e in v50b.names(tokens) if i>verb_i]
        if post:
            ent=post[0][1]
            story.add_mention(ent,context,si)
            return ent
    return None

def resolve_dative_recipient(tokens,story,context,si):
    dprons=v50b.indices_with(tokens,"DAT_PRONOUN")
    if not dprons:
        return None,False
    # One dative pronoun in this controlled grammar.
    _,tok=dprons[0]
    gender=pronoun_gender(tok)
    cands=story.candidate_mentions(gender,context,si)
    if len(cands)==1:
        return cands[0].entity,True
    story.unresolved.append({
        "type":"REFERENCE",
        "case":"DAT",
        "surface":tok.surface,
        "context":context,
        "candidates":[m.entity for m in cands],
        "sentence_index":si,
    })
    return None,True

def context_item_candidates(story,context):
    def collect(ctx):
        out=[]
        for e in story.evidence:
            if e.key.context!=ctx:
                continue
            if e.key.rel=="INITIAL_COUNT" and len(e.key.args)>=2:
                out.append(e.key.args[1])
            elif e.key.rel=="TRANSFER" and len(e.key.args)>=5 and e.key.args[3]!="?":
                out.append(e.key.args[3])
        for ev in story.events:
            if ev.context==ctx and ev.item:
                out.append(ev.item)
        return list(dict.fromkeys(out))

    local=collect(context)
    if local:
        return local
    if context.startswith("CLAIM:"):
        return collect("WORLD")
    return []

def resolve_item_safe(tokens,story,context):
    explicit=v50b.items(tokens)
    if explicit:
        item=explicit[0][1]
        story.last_item_by_context[context]=item
        return item

    if v50b.has(tokens,"ANAPHOR_ITEM"):
        cands=context_item_candidates(story,context)
        if len(cands)==1:
            return cands[0]
        story.unresolved.append({
            "type":"ITEM_REFERENCE",
            "surface":"davon",
            "context":context,
            "candidates":cands,
        })
    return None

# Override reference hooks used below.
v50b.resolve_subject=resolve_nominal_subject
v50b.resolve_item=resolve_item_safe

# ------------------------------------------------------------
# Extended count / transfer parser.
# ------------------------------------------------------------

def parse_count_clause(sentence,story,context,si):
    toks=v50b.tokenize(sentence)
    vis=v50b.indices_with(toks,"COUNT_STATE")
    if not vis:
        return False
    vi=vis[0][0]
    actor=resolve_nominal_subject(toks,story,context,si,vi)
    item=resolve_item_safe(toks,story,context)

    # Number may be after verb in ordinary order.
    ns=v50b.nums(toks)
    if actor and item and ns:
        # For count state pick the number closest after verb, else nearest.
        after=[x for x in ns if x[0]>vi]
        n=(after[0] if after else ns[0])[1]
        story.add_key(
            Key("INITIAL_COUNT",(actor,item,n),context),
            sentence,
            "COUNT_STATE(actor,item,count)"
        )
        return True
    return False

def transfer_events_from_clause(sentence,story,context,si,last_event):
    toks=v50b.tokenize(sentence)
    vis=v50b.indices_with(toks,"TRANSFER")
    if not vis:
        return []
    vi=vis[0][0]

    actor=resolve_nominal_subject(toks,story,context,si,vi)
    if actor is None:
        return []

    item=resolve_item_safe(toks,story,context)
    ns=v50b.nums(toks)  # supports fronted numeric object
    if not item or not ns:
        return []

    # Dative pronoun has explicit case-role priority.
    recipient,had_dat_pron=resolve_dative_recipient(toks,story,context,si)

    ents_after=[x for x in v50b.names(toks) if x[0]>vi]
    # Remove post-verbal explicit actor under fronting.
    filtered=[]
    removed_actor=False
    for pos,ent in ents_after:
        if not removed_actor and ent==actor:
            removed_actor=True
            continue
        filtered.append((pos,ent))
    ents_after=filtered

    if not had_dat_pron:
        markers=v50b.indices_with(toks,"RECIPIENT_MARKER")
        if markers:
            mi=markers[0][0]
            aft=[e for i,e in ents_after if i>mi]
            if aft:
                recipient=aft[0]
        elif ents_after:
            recipient=ents_after[0][1]

    base_rank=float(si)
    if v50b.has(toks,"TEMP_AFTER") and last_event is not None:
        base_rank=last_event.time_rank+0.5
    elif v50b.has(toks,"TEMP_BEFORE") and last_event is not None:
        base_rank=last_event.time_rank-0.5

    # Coordination stays delegated to v5.0b for explicit-name pattern.
    coord=v50b.indices_with(toks,"COORD")
    if coord and not had_dat_pron:
        # Only use old coordination path if count/item are not fronted in this test.
        old=v50b.transfer_events_from_clause(sentence,story,context,si,last_event)
        if len(old)>1:
            return old

    # Pick one numeric quantity. Under fronting it's normally before the verb.
    n=ns[0][1]
    if v50b.has(toks,"ELLIPSIS_RECIPIENT"):
        recipient=None

    return [Event(
        f"E{len(story.events)+1}","TRANSFER",
        actor,recipient,item,n,context,si,base_rank,sentence
    )]

# ------------------------------------------------------------
# Multi-clause CLAIM parser.
# ------------------------------------------------------------

QUOTE_RE=re.compile(r'[„"“](.+?)[”"“]',re.DOTALL)

def parse_clause(sentence,story,context,si,last_event):
    if v50b.parse_meet_clause(sentence,story,context,si):
        return []
    if parse_count_clause(sentence,story,context,si):
        return []
    return transfer_events_from_clause(sentence,story,context,si,last_event)

def parse_story(text,story_id):
    story=Story(story_id)
    last_world_event=None
    sentences=v50b.split_sentences(text)

    for si,sentence in enumerate(sentences):
        toks=v50b.tokenize(sentence)
        speech=v50b.indices_with(toks,"SPEECH")
        if speech:
            vi=speech[0][0]
            speaker_names=[e for i,e in v50b.names(toks) if i<vi]
            speaker=speaker_names[-1] if speaker_names else "UNKNOWN_SPEAKER"
            if speaker!="UNKNOWN_SPEAKER":
                story.add_mention(speaker,"WORLD",si)
            m=QUOTE_RE.search(sentence)
            if m:
                quoted=m.group(1).strip()
                qctx=f"CLAIM:{speaker}"
                inner_last=None
                for qi,qsentence in enumerate(v50b.split_sentences(quoted)):
                    inner_si=si*100+qi+1
                    qevents=parse_clause(qsentence,story,qctx,inner_si,inner_last)
                    if qevents:
                        story.events.extend(qevents)
                        inner_last=sorted(qevents,key=lambda e:e.time_rank)[-1]
                continue

        events=parse_clause(sentence,story,"WORLD",si,last_world_event)
        if events:
            story.events.extend(events)
            last_world_event=sorted(events,key=lambda e:e.time_rank)[-1]

    # Materialize Event-U and time Keys.
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
# Query helper and same frozen numeric State-U.
# ------------------------------------------------------------

parse_query=v50b.parse_query
answer=v50b.answer

# ------------------------------------------------------------
# C1 — nominative + dative pronouns.
# ------------------------------------------------------------

story_pron=parse_story(
    "Paul hat fünf Äpfel. Anna hat null Äpfel. Er gibt ihr zwei davon.",
    "pronouns"
)
q_paul=parse_query("Wie viele Äpfel hat Paul noch?")
q_anna=parse_query("Wie viele Äpfel hat Anna noch?")
pron_paul=answer(story_pron,q_paul)
pron_anna=answer(story_pron,q_anna)

# ------------------------------------------------------------
# C2 — opposite genders, female actor + male dative recipient.
# ------------------------------------------------------------

story_pron2=parse_story(
    "Anna hat fünf Birnen. Paul hat null Birnen. Sie gibt ihm zwei davon.",
    "pronouns2"
)
q_anna_pear=parse_query("Wie viele Birnen hat Anna noch?")
q_paul_pear=parse_query("Wie viele Birnen hat Paul noch?")
pron2_anna=answer(story_pron2,q_anna_pear)
pron2_paul=answer(story_pron2,q_paul_pear)

# ------------------------------------------------------------
# C3 — ambiguous dative recipient stays unresolved.
# Actor loss is still a valid partial world consequence.
# ------------------------------------------------------------

story_dat_amb=parse_story(
    "Anna hat null Äpfel. Mia hat null Äpfel. Paul hat fünf Äpfel. "
    "Er gibt ihr zwei davon.",
    "dat-amb"
)
dat_paul=answer(story_dat_amb,q_paul)
dat_anna=answer(story_dat_amb,q_anna)
q_mia=parse_query("Wie viele Äpfel hat Mia noch?")
dat_mia=answer(story_dat_amb,q_mia)

# ------------------------------------------------------------
# C4 — multiple object types make bare "davon" ambiguous.
# ------------------------------------------------------------

story_item_amb=parse_story(
    "Anna hat sieben Äpfel. Anna hat vier Birnen. Paul hat null Äpfel. "
    "Sie gibt Paul zwei davon.",
    "item-amb"
)
item_amb_apple=answer(story_item_amb,q_anna)
item_amb_pear=answer(story_item_amb,q_anna_pear)

# Explicit noun disambiguates.
story_item_clear=parse_story(
    "Anna hat sieben Äpfel. Anna hat vier Birnen. Paul hat null Äpfel. "
    "Sie gibt Paul zwei Äpfel.",
    "item-clear"
)
item_clear_anna=answer(story_item_clear,q_anna)
item_clear_paul=answer(story_item_clear,q_paul)
item_clear_pear=answer(story_item_clear,q_anna_pear)

# ------------------------------------------------------------
# C5 — local reference chain entirely inside CLAIM.
# ------------------------------------------------------------

story_claim=parse_story(
    'Paul sagt: „Mia hat drei Äpfel. Sie gibt Anna zwei davon.“',
    "claim-chain"
)
claim_events=[e for e in story_claim.events if e.context=="CLAIM:paul"]
claim_world_events=[e for e in story_claim.events if e.context=="WORLD"]

# ------------------------------------------------------------
# C6 — V2 object fronting with BOTH pronouns after verb.
# "Zwei Äpfel gibt sie ihm."
# ------------------------------------------------------------

story_v2=parse_story(
    "Anna hat sieben Äpfel. Paul hat null Äpfel. Zwei Äpfel gibt sie ihm.",
    "v2-front"
)
v2_anna=answer(story_v2,q_anna)
v2_paul=answer(story_v2,q_paul)

# Explicit names with object fronting too.
story_v2_names=parse_story(
    "Anna hat sieben Äpfel. Paul hat null Äpfel. Drei Äpfel gibt Anna Paul.",
    "v2-names"
)
v2n_anna=answer(story_v2_names,q_anna)
v2n_paul=answer(story_v2_names,q_paul)

# ------------------------------------------------------------
# C7 — wrong/reference-ambiguous outcomes remain UNKNOWN.
# ------------------------------------------------------------

def query_state(story,q,candidate):
    got=answer(story,q)
    return +1 if got==candidate else 0

# ------------------------------------------------------------
# Checks
# ------------------------------------------------------------

dat_unresolved=[
    x for x in story_dat_amb.unresolved
    if x.get("case")=="DAT"
]
item_unresolved=[
    x for x in story_item_amb.unresolved
    if x.get("type")=="ITEM_REFERENCE"
]

checks={
    "C1_er_resolves_male_actor":(
        len(story_pron.events)==1 and story_pron.events[0].actor=="paul"
    ),
    "C1_ihr_resolves_female_recipient":(
        story_pron.events[0].recipient=="anna"
    ),
    "C1_pronoun_transfer_state":(
        pron_paul=="N3" and pron_anna=="N2"
    ),
    "C2_sie_and_ihm_case_roles":(
        len(story_pron2.events)==1 and
        story_pron2.events[0].actor=="anna" and
        story_pron2.events[0].recipient=="paul" and
        pron2_anna=="N3" and pron2_paul=="N2"
    ),
    "C3_ambiguous_dative_reference_recorded":(
        len(dat_unresolved)==1 and
        set(dat_unresolved[0]["candidates"])=={"anna","mia"}
    ),
    "C3_ambiguous_recipient_not_guessed":(
        len(story_dat_amb.events)==1 and
        story_dat_amb.events[0].recipient is None and
        dat_paul=="N3" and dat_anna=="N0" and dat_mia=="N0"
    ),
    "C4_multiple_item_types_make_davon_ambiguous":(
        len(item_unresolved)==1 and
        set(item_unresolved[0]["candidates"])=={"APPLE","PEAR"} and
        len(story_item_amb.events)==0
    ),
    "C4_item_ambiguity_does_not_change_state":(
        item_amb_apple=="N7" and item_amb_pear=="N4"
    ),
    "C4_explicit_item_disambiguates":(
        item_clear_anna=="N5" and
        item_clear_paul=="N2" and
        item_clear_pear=="N4"
    ),
    "C5_claim_local_reference_chain_resolves":(
        len(claim_events)==1 and
        claim_events[0].actor=="mia" and
        claim_events[0].recipient=="anna" and
        claim_events[0].item=="APPLE" and
        claim_events[0].count=="N2"
    ),
    "C5_claim_chain_does_not_leak_world":(
        len(claim_world_events)==0 and
        not story_claim.facts("WORLD","INITIAL_COUNT")
    ),
    "C6_v2_fronted_object_with_pronouns":(
        len(story_v2.events)==1 and
        story_v2.events[0].actor=="anna" and
        story_v2.events[0].recipient=="paul" and
        story_v2.events[0].count=="N2" and
        v2_anna=="N5" and v2_paul=="N2"
    ),
    "C6_v2_fronted_object_with_names":(
        len(story_v2_names.events)==1 and
        story_v2_names.events[0].actor=="anna" and
        story_v2_names.events[0].recipient=="paul" and
        v2n_anna=="N4" and v2n_paul=="N3"
    ),
    "C7_wrong_answer_stays_unknown":(
        query_state(story_v2,q_paul,"N3")==0
    ),
    "C7_numeric_u_still_reused_once":(
        dict(lib.learn_attempts)=={"R1":1}
    ),
}

print("=== v5.0c CASE / MULTI-ITEM / CLAIM-CHAIN / V2 ===")
for k,v in checks.items():
    print(("PASS" if v else "FAIL"),"|",k)

print("\nC1 pronouns:")
for e in story_pron.events:
    print(" ",e.actor,"->",e.recipient,e.item,e.count)
print(" states: Paul",pron_paul,"Anna",pron_anna)

print("\nC2 opposite pronouns:")
for e in story_pron2.events:
    print(" ",e.actor,"->",e.recipient,e.item,e.count)
print(" states: Anna",pron2_anna,"Paul",pron2_paul)

print("\nC3 ambiguous dative:")
print(" unresolved:",story_dat_amb.unresolved)
for e in story_dat_amb.events:
    print(" ",e.actor,"->",e.recipient,e.item,e.count)
print(" states:",dat_paul,dat_anna,dat_mia)

print("\nC4 item ambiguity:")
print(" unresolved:",story_item_amb.unresolved)
print(" events:",len(story_item_amb.events))
print(" states APPLE/PEAR:",item_amb_apple,item_amb_pear)
print(" explicit item states:",item_clear_anna,item_clear_paul,item_clear_pear)

print("\nC5 CLAIM chain:")
for e in story_claim.evidence:
    print(" ",e.key)
for e in claim_events:
    print(" event",e.context,e.actor,"->",e.recipient,e.item,e.count)

print("\nC6 V2 fronting:")
for e in story_v2.events:
    print(" pronouns:",e.actor,"->",e.recipient,e.item,e.count)
print(" states:",v2_anna,v2_paul)
for e in story_v2_names.events:
    print(" names:",e.actor,"->",e.recipient,e.item,e.count)
print(" states:",v2n_anna,v2n_paul)

print("\nNumeric U attempts:",dict(lib.learn_attempts))

assert all(checks.values())

report={
    "version":"v5.0c-case-multiitem-claimchain-v2",
    "result":"PASS",
    "scope":"controlled synthetic German; not general NLP accuracy",
    "checks":checks,
    "numeric_u":{
        "version":r1.id,
        "attempts":dict(lib.learn_attempts),
        "base_rule":r1.meta["base_rule"],
        "recursive_rule":r1.meta["recursive_rule"],
    },
    "pronouns":{
        "events":[e.__dict__ for e in story_pron.events],
        "answers":{"paul":pron_paul,"anna":pron_anna},
    },
    "ambiguous_dative":{
        "unresolved":story_dat_amb.unresolved,
        "events":[e.__dict__ for e in story_dat_amb.events],
        "answers":{"paul":dat_paul,"anna":dat_anna,"mia":dat_mia},
    },
    "multiple_items":{
        "ambiguous_unresolved":story_item_amb.unresolved,
        "ambiguous_events":[e.__dict__ for e in story_item_amb.events],
        "clear_events":[e.__dict__ for e in story_item_clear.events],
    },
    "claim_chain":{
        "keys":[str(e.key) for e in story_claim.evidence],
        "events":[e.__dict__ for e in claim_events],
    },
    "v2_fronting":{
        "pronoun_events":[e.__dict__ for e in story_v2.events],
        "name_events":[e.__dict__ for e in story_v2_names.events],
        "pronoun_answers":{"anna":v2_anna,"paul":v2_paul},
        "name_answers":{"anna":v2n_anna,"paul":v2n_paul},
    },
    "architecture":[
        "Nominative and dative pronoun forms contribute explicit symbolic case and gender constraints.",
        "Reference-U selects only a unique compatible entity; multiple compatible entities remain unresolved.",
        "A transfer with unresolved recipient may still commit the independently supported sender-loss event, while no recipient state is guessed.",
        "Bare 'davon' is resolved only when one item type is compatible in the proposition context; multiple item types remain unresolved.",
        "Quoted multi-clause content has a local CLAIM context in which mentions and item references can chain without becoming WORLD facts.",
        "German V2 object fronting is handled by clause-role structure rather than a fixed subject-before-verb heuristic.",
        "The same ACTIVE anonymous numeric U R1 is reused; no new arithmetic relation is learned."
    ],
    "caveats":[
        "Case inventory is deliberately small: nominative er/sie and dative ihm/ihr.",
        "German 'sie' syncretism (nominative/accusative/plural) is not generally solved; this benchmark constrains it to feminine singular nominative.",
        "Item anaphora deliberately prefers UNKNOWN over recency when multiple item types are active.",
        "Partial-event commitment is conservative only for independently supported roles; a richer proposition dependency graph is future work.",
        "V2 fronting covers tested numeric-object and temporal fronting patterns, not arbitrary German scrambling.",
        "Still controlled German, not Grimm prose."
    ],
}

Path("/mnt/data/symbolic_v50c_case_multiitem_claimchain_v2_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
with Path("/mnt/data/symbolic_v50c_case_multiitem_claimchain_v2_checks.csv").open(
    "w",newline="",encoding="utf-8"
) as f:
    w=csv.writer(f)
    w.writerow(["check","passed"])
    for k,v in checks.items():
        w.writerow([k,v])

print("\nSaved v5.0c report/checks.")

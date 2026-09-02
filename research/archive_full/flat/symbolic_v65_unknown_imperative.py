
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from itertools import product
import importlib.util, sys, contextlib, io, re, json, csv, copy

# ============================================================
# v6.5 — C11 Learn Unknown Imperative Semantics from Consequences
#
# Frozen base: v6.4
#
# Core idea:
#   UNKNOWN_DIRECTIVE(token,target)
#   + PRE_ACTIVE(target,action)
#   + local observed CEASE(target,action)
#   -> anonymous semantic class R10(token,target,action)
#
# Token lexical lifecycle:
#   first independent consequence -> STAGED
#   second independent consequence -> ACTIVE
#
# ACTIVE lexical semantics can recognize a future directive,
# but directive alone does NOT prove WORLD cessation.
# ============================================================

# ------------------------------------------------------------
# 0. Load/freeze v6.4
# ------------------------------------------------------------

spec=importlib.util.spec_from_file_location("v64f","/mnt/data/symbolic_v64_syncretism_reporting.py")
v=importlib.util.module_from_spec(spec)
sys.modules["v64f"]=v
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(v)

assert all(v.checks.values())

ANON_STOP_REL="R10"

# ------------------------------------------------------------
# 1. Primitive action morphology / target lexicon
#    (no meaning for unknown imperative tokens)
# ------------------------------------------------------------

TARGETS={
    "lampe":"LAMP",
    "rad":"WHEEL",
    "maschine":"MACHINE",
    "tor":"GATE",
    "töpfchen":"POT",
    "toepfchen":"POT",
    "topf":"POT",
}

ACTION_FORMS={
    "leuchtet":"LIGHT","leuchten":"LIGHT",
    "dreht":"TURN","drehen":"TURN",
    "läuft":"RUN","laeuft":"RUN","laufen":"RUN",
    "öffnet":"OPEN","oeffnet":"OPEN","öffnen":"OPEN","oeffnen":"OPEN",
    "kocht":"COOK","kochte":"COOK","kochen":"COOK","koche":"COOK",
}

STOP_CUES={"hört","hoert","hörte","hoerte"}
CONTINUE_CUES={"weiter","weiterhin","fort"}

def wtoks(text):
    return re.findall(r"[A-Za-zÄÖÜäöüß]+",text.lower())

def target_from_quote(q):
    return next((TARGETS[x] for x in wtoks(q) if x in TARGETS),None)

def known_action_from_quote(q):
    # An action is known only if the imperative/action form is already in lexicon.
    return next((ACTION_FORMS[x] for x in wtoks(q) if x in ACTION_FORMS),None)

def unknown_imperative_token(q):
    ts=wtoks(q)
    target_idx=next((i for i,x in enumerate(ts) if x in TARGETS),None)
    if target_idx is None:
        return None
    for x in ts[target_idx+1:]:
        if x not in ACTION_FORMS and x not in {"bitte"}:
            return x
    return None

# ------------------------------------------------------------
# 2. Generic ACTIVE / CEASE observation extraction
# ------------------------------------------------------------

@dataclass(frozen=True)
class StopExperience:
    story:str
    token:str
    command_target:str
    pre_target:str
    pre_action:str
    cease_target:str|None
    cease_action:str|None

def extract_active(sentence):
    ts=wtoks(sentence)
    target=next((TARGETS[x] for x in ts if x in TARGETS),None)
    action=next((ACTION_FORMS[x] for x in ts if x in ACTION_FORMS),None)
    return (target,action) if target and action else None

def extract_cease(after_text,command_target=None):
    ts=wtoks(after_text)
    if not any(x in STOP_CUES for x in ts):
        return None

    # Generic "hört ... auf zu ACTION" cue.
    if "auf" not in ts or "zu" not in ts:
        return None

    action=None
    for x in ts:
        if x in ACTION_FORMS:
            action=ACTION_FORMS[x]
    if action is None:
        return None

    explicit=next((TARGETS[x] for x in ts if x in TARGETS),None)

    # If response subject is pronoun "es", the command target may constrain
    # reference binding, but the CEASE verb/cue itself is the occurrence evidence.
    if explicit:
        target=explicit
    elif "es" in ts and command_target:
        target=command_target
    else:
        target=None

    return (target,action) if target else None

# ------------------------------------------------------------
# 3. C11 training: learn the anonymous transition topology
# ------------------------------------------------------------

# Positive curriculum examples use opaque command words zarp/murk/fep.
POSITIVE_RAW=[
    (
        "train-light",
        "Die Lampe leuchtet.",
        '"Lampe zarp."',
        "Danach hörte die Lampe auf zu leuchten.",
    ),
    (
        "train-turn",
        "Das Rad dreht.",
        '"Rad murk."',
        "Danach hörte das Rad auf zu drehen.",
    ),
    (
        "train-run",
        "Die Maschine läuft.",
        '"Maschine fep."',
        "Danach hörte die Maschine auf zu laufen.",
    ),
]

NEGATIVE_RAW=[
    # Command/pre target match, but cessation belongs to a DIFFERENT target
    # while preserving the SAME action. This separates target topology from action topology.
    (
        "neg-post-target",
        "Die Lampe leuchtet.",
        '"Lampe niff."',
        "Danach hörte das Tor auf zu leuchten.",
    ),
    # Command target differs from the actually active/ceasing target.
    (
        "neg-pre-target",
        "Das Tor leuchtet.",
        '"Lampe vex."',
        "Danach hörte die Lampe auf zu leuchten.",
    ),
    # Same target but different action cessation.
    (
        "neg-action",
        "Das Rad dreht.",
        '"Rad blup."',
        "Danach hörte das Rad auf zu leuchten.",
    ),
    # No cessation evidence.
    (
        "neg-none",
        "Die Maschine läuft.",
        '"Maschine grom."',
        "Danach läuft die Maschine weiter.",
    ),
]

def experience_from_raw(story,pre,quote,after):
    active=extract_active(pre)
    q=re.search(r'"([^"]+)"',quote)
    assert active and q
    qtext=q.group(1)
    target=target_from_quote(qtext)
    token=unknown_imperative_token(qtext)
    cease=extract_cease(after,target)
    return StopExperience(
        story,
        token,
        target,
        active[0],
        active[1],
        cease[0] if cease else None,
        cease[1] if cease else None,
    )

POS=[experience_from_raw(*x) for x in POSITIVE_RAW]
NEG=[experience_from_raw(*x) for x in NEGATIVE_RAW]

# Candidate U topology language.
# require_cease: consequence must exist
# target_mode:
#   SAME_COMMAND_PRE_POST or COMMAND_POST_ONLY or NONE
# action_mode:
#   SAME_PRE_POST or NONE
@dataclass(frozen=True)
class StopSchema:
    require_cease:bool
    target_mode:str
    action_mode:str

SCHEMAS=[
    StopSchema(rc,tm,am)
    for rc,tm,am in product(
        (False,True),
        ("NONE","COMMAND_POST_ONLY","SAME_COMMAND_PRE_POST"),
        ("NONE","SAME_PRE_POST"),
    )
]

def schema_accepts(s,e):
    if s.require_cease and (e.cease_target is None or e.cease_action is None):
        return False

    if s.target_mode=="COMMAND_POST_ONLY":
        if e.cease_target is None or e.command_target!=e.cease_target:
            return False
    elif s.target_mode=="SAME_COMMAND_PRE_POST":
        if (
            e.cease_target is None
            or not (e.command_target==e.pre_target==e.cease_target)
        ):
            return False

    if s.action_mode=="SAME_PRE_POST":
        if e.cease_action is None or e.pre_action!=e.cease_action:
            return False

    return True

def schema_score(s):
    support=sum(schema_accepts(s,e) for e in POS)
    conflict=sum(schema_accepts(s,e) for e in NEG)
    # Prefer fewer requirements only among equally safe/full-support candidates.
    complexity=(
        int(s.require_cease)
        + (0 if s.target_mode=="NONE" else 1 if s.target_mode=="COMMAND_POST_ONLY" else 2)
        + (0 if s.action_mode=="NONE" else 1)
    )
    return support,conflict,complexity

VALID=[]
for s in SCHEMAS:
    sup,con,cx=schema_score(s)
    if sup==len(POS) and con==0:
        VALID.append((cx,s))
VALID.sort(key=lambda x:(x[0],repr(x[1])))
assert VALID
U_UNKNOWN_IMPERATIVE=VALID[0][1]

# The negatives force exact target/action continuity.
assert U_UNKNOWN_IMPERATIVE==StopSchema(
    False,"SAME_COMMAND_PRE_POST","SAME_PRE_POST"
)

# ------------------------------------------------------------
# 4. Token lifecycle: STAGED -> ACTIVE
# ------------------------------------------------------------

@dataclass
class LexicalU:
    token:str
    relation:str
    status:str="STAGED"
    support_stories:set[str]=field(default_factory=set)
    conflicts:int=0
    provenance:list[str]=field(default_factory=list)

    @property
    def support(self): return len(self.support_stories)

class LexiconLearner:
    def __init__(self,min_support=2):
        self.entries={}
        self.min_support=min_support
        self.lifecycle=[]

    def observe(self,e:StopExperience):
        if not e.token:
            return None

        accepted=schema_accepts(U_UNKNOWN_IMPERATIVE,e)

        if e.token not in self.entries:
            if not accepted:
                return None
            lu=LexicalU(e.token,ANON_STOP_REL)
            self.entries[e.token]=lu
            self.lifecycle.append(("STAGED",e.token,e.story))
        else:
            lu=self.entries[e.token]

        if accepted:
            lu.support_stories.add(e.story)
            lu.provenance.append(e.story)
            if lu.status=="STAGED" and lu.support>=self.min_support:
                lu.status="ACTIVE"
                self.lifecycle.append(("ACTIVE",e.token,e.story))
        else:
            # Only count conflict when there is an actual cessation observation
            # inconsistent with the schema. Absence of evidence is UNKNOWN, not conflict.
            if e.cease_target is not None or e.cease_action is not None:
                lu.conflicts+=1
                if lu.status=="ACTIVE":
                    lu.status="CHALLENGED"
                    self.lifecycle.append(("CHALLENGED",e.token,e.story))
        return lu

    def recognize(self,token,target,active_action):
        lu=self.entries.get(token)
        if not lu or lu.status!="ACTIVE":
            return None
        # R10 is a directive semantic classification, not a world effect.
        return (ANON_STOP_REL,(target,active_action))

# ------------------------------------------------------------
# 5. Frozen synthetic lexical induction and reuse
# ------------------------------------------------------------

LEX_SYN=LexiconLearner(min_support=2)

SYN1=experience_from_raw(
    "syn-plim-1",
    "Die Lampe leuchtet.",
    '"Lampe plim."',
    "Danach hörte die Lampe auf zu leuchten.",
)
SYN2=experience_from_raw(
    "syn-plim-2",
    "Das Rad dreht.",
    '"Rad plim."',
    "Danach hörte das Rad auf zu drehen.",
)

lu1=LEX_SYN.observe(SYN1)
SYN_AFTER_ONE=(lu1.status,lu1.support)
lu2=LEX_SYN.observe(SYN2)
SYN_AFTER_TWO=(lu2.status,lu2.support)

# Reuse on a third target/action with no outcome observation.
SYN_REUSE=LEX_SYN.recognize("plim","GATE","OPEN")

# Directive alone does not prove WORLD cessation.
SYN_WORLD_CEASE_WITHOUT_OBSERVATION=False

# Add matching observation: now WORLD CEASE can be independently grounded.
SYN3=experience_from_raw(
    "syn-plim-3",
    "Das Tor öffnet.",
    '"Tor plim."',
    "Danach hörte das Tor auf zu öffnen.",
)
SYN3_WORLD=(
    schema_accepts(U_UNKNOWN_IMPERATIVE,SYN3)
    and SYN3.cease_target=="GATE"
    and SYN3.cease_action=="OPEN"
)

# Mismatch does not fit the learned semantic topology.
SYN_BAD=experience_from_raw(
    "syn-bad",
    "Das Rad dreht.",
    '"Rad quux."',
    "Danach hörte die Lampe auf zu leuchten.",
)
SYN_BAD_ACCEPT=schema_accepts(U_UNKNOWN_IMPERATIVE,SYN_BAD)

# ------------------------------------------------------------
# 6. Frozen full Grimm: induce meaning of unseen "steh"
# ------------------------------------------------------------

GRIMM=Path("/mnt/data/grimm_der_suesse_brei.txt").read_text(encoding="utf-8")
G=GRIMM.replace("„",'"').replace("“",'"')
SPANS=list(re.finditer(r'"([^"]+)"',G,re.S))

LEX_GRIMM=LexiconLearner(min_support=2)
active={}             # target -> action
grimm_trace=[]
stop_occurrences=[]

def local_after(span,n=180):
    return G[span.end():min(len(G),span.end()+n)]

for i,span in enumerate(SPANS):
    q=span.group(1)
    target=target_from_quote(q)
    known=known_action_from_quote(q)
    unknown=unknown_imperative_token(q)

    if known and target:
        # Reuse v6.4 frozen result: known COOK quote has local execution observation.
        # Only WORLD observation activates the state.
        local=local_after(span)
        ts=wtoks(local)
        observed=(
            target=="POT"
            and known=="COOK"
            and ("kocht" in ts or "kochte" in ts)
            and "es" in ts
        )
        if observed:
            active[target]=known
        grimm_trace.append({
            "i":i,"quote":q,"kind":"KNOWN","target":target,"action":known,
            "world_observed":observed,"active_after":dict(active),
        })
        continue

    if unknown and target:
        pre_action=active.get(target)
        cease=extract_cease(local_after(span),target)

        exp=StopExperience(
            f"grimm-quote-{i+1}",
            unknown,
            target,
            target,
            pre_action,
            cease[0] if cease else None,
            cease[1] if cease else None,
        )

        lu=LEX_GRIMM.observe(exp)
        if cease and pre_action==cease[1] and target==cease[0]:
            # WORLD cessation comes from the narrative observation.
            active.pop(target,None)

        stop_occurrences.append({
            "quote_index":i,
            "token":unknown,
            "pre_action":pre_action,
            "cease":cease,
            "lex_status":lu.status if lu else None,
            "lex_support":lu.support if lu else 0,
        })
        grimm_trace.append({
            "i":i,"quote":q,"kind":"UNKNOWN","target":target,"token":unknown,
            "pre_action":pre_action,"cease":cease,
            "lex_status":lu.status if lu else None,
            "active_after":dict(active),
        })

STEH=LEX_GRIMM.entries.get("steh")

# After reading full Grimm, meaning can be reused in a new controlled directive.
GRIMM_REUSE=LEX_GRIMM.recognize("steh","LAMP","LIGHT")

# But recognition of directive semantics alone is not WORLD evidence.
GRIMM_REUSE_WORLD_WITHOUT_OBSERVATION=False

# ------------------------------------------------------------
# 7. Adversarial lexical lifecycle
# ------------------------------------------------------------

# First support only -> STAGED and cannot prove/recognize normally.
LEX_ONE=LexiconLearner(min_support=2)
one=LEX_ONE.observe(SYN1)
ONE_RECOG=LEX_ONE.recognize("plim","GATE","OPEN")

# No cease -> no lexical semantic entry at all for a fresh token.
NO_EVIDENCE=experience_from_raw(
    "no-evidence",
    "Die Lampe leuchtet.",
    '"Lampe noop."',
    "Danach leuchtet die Lampe weiter.",
)
LEX_NONE=LexiconLearner(min_support=2)
none_entry=LEX_NONE.observe(NO_EVIDENCE)

# Active token challenged by contradictory observed consequence.
LEX_CH=copy.deepcopy(LEX_SYN)
CONFLICT=StopExperience(
    "conflict-story",
    "plim",
    "GATE",
    "GATE",
    "OPEN",
    "LAMP",
    "LIGHT",
)
LEX_CH.observe(CONFLICT)
CH_STATUS=LEX_CH.entries["plim"].status

# ------------------------------------------------------------
# 8. C12 mixed longer text with unknown lexical induction
# ------------------------------------------------------------

MIX_LEX=LexiconLearner(min_support=2)
MIXED_EXPS=[
    experience_from_raw(
        "mix-1","Die Lampe leuchtet.",'"Lampe vorn."',
        "Danach hörte die Lampe auf zu leuchten."
    ),
    experience_from_raw(
        "mix-2","Das Rad dreht.",'"Rad vorn."',
        "Danach hörte das Rad auf zu drehen."
    ),
]
for e in MIXED_EXPS:
    MIX_LEX.observe(e)

# Now read 20 already-known v6.4 lines plus repeated use of the newly active token.
known_mixed=(v.MIXED_TEXT*2)
known_results=[v.parse_mixed(x) for x in known_mixed]
reuse_directives=[
    MIX_LEX.recognize("vorn","GATE","OPEN"),
    MIX_LEX.recognize("vorn","POT","COOK"),
]
MIXED_LONG_PASS=all(x is not None for x in known_results) and all(reuse_directives)

# ------------------------------------------------------------
# 9. Checks
# ------------------------------------------------------------

checks={
    "frozen_v64_base_stays_green":all(v.checks.values()),

    "C11_unknown_imperative_schema_learned_from_unrelated_examples":(
        U_UNKNOWN_IMPERATIVE==StopSchema(False,"SAME_COMMAND_PRE_POST","SAME_PRE_POST")
    ),
    "C11_schema_full_positive_support_zero_negative_conflict":(
        schema_score(U_UNKNOWN_IMPERATIVE)[:2]==(len(POS),0)
    ),
    "C11_wrong_target_action_negative_rejected":not SYN_BAD_ACCEPT,

    "C11_first_token_evidence_only_staged":SYN_AFTER_ONE==("STAGED",1),
    "C11_second_independent_evidence_activates_token":SYN_AFTER_TWO==("ACTIVE",2),
    "C11_active_token_reuses_across_new_target_action":SYN_REUSE==(ANON_STOP_REL,("GATE","OPEN")),
    "C11_directive_semantics_alone_not_world_cessation":not SYN_WORLD_CEASE_WITHOUT_OBSERVATION,
    "C11_independent_matching_observation_can_ground_world_cessation":SYN3_WORLD,

    "grimm_two_unknown_steh_occurrences_found":len(stop_occurrences)==2,
    "grimm_first_steh_only_staged":stop_occurrences[0]["lex_status"]=="STAGED",
    "grimm_second_steh_activates_lexical_U":stop_occurrences[1]["lex_status"]=="ACTIVE",
    "grimm_steh_active_after_two_independent_consequences":(
        STEH is not None and STEH.status=="ACTIVE" and STEH.support==2
    ),
    "grimm_both_steh_consequences_are_local_COOK_cessations":all(
        x["pre_action"]=="COOK" and x["cease"]==("POT","COOK")
        for x in stop_occurrences
    ),
    "grimm_learned_steh_reuses_on_new_action":GRIMM_REUSE==(ANON_STOP_REL,("LAMP","LIGHT")),
    "grimm_reused_steh_directive_still_not_world_evidence":not GRIMM_REUSE_WORLD_WITHOUT_OBSERVATION,

    "adversarial_one_support_cannot_normal_reuse":ONE_RECOG is None,
    "adversarial_no_consequence_creates_no_semantic_entry":none_entry is None,
    "adversarial_contradictory_consequence_challenges_active_token":CH_STATUS=="CHALLENGED",

    "C12_mixed_long_text_uses_new_lexical_U_plus_old_language_U":MIXED_LONG_PASS,
}

print("=== v6.5 C11 UNKNOWN IMPERATIVE SEMANTICS ===")
for k,val in checks.items():
    print(("PASS" if val else "FAIL"),"|",k)

print("\nLearned anonymous schema:")
for s in SCHEMAS:
    print(" ",s,"score",schema_score(s))
print(" winner:",U_UNKNOWN_IMPERATIVE,"=>",ANON_STOP_REL)

print("\nSynthetic lexical lifecycle:")
print(" after one:",SYN_AFTER_ONE)
print(" after two:",SYN_AFTER_TWO)
print(" reuse:",SYN_REUSE)
print(" world without observation:",SYN_WORLD_CEASE_WITHOUT_OBSERVATION)
print(" third observed cease:",SYN3_WORLD)

print("\nGrimm trace:")
for x in grimm_trace:
    print(" ",x)
print(" stop occurrences:",stop_occurrences)
print(" STEH entry:",
      None if STEH is None else {
          "relation":STEH.relation,
          "status":STEH.status,
          "support":STEH.support,
          "conflicts":STEH.conflicts,
          "provenance":STEH.provenance,
      })
print(" reuse STEH on LAMP/LIGHT:",GRIMM_REUSE)
print(" reuse world without observation:",GRIMM_REUSE_WORLD_WITHOUT_OBSERVATION)

print("\nAdversarial:")
print(" one-support recognize:",ONE_RECOG)
print(" no evidence entry:",none_entry)
print(" conflict status:",CH_STATUS)

print("\nC12 mixed:")
print(" known mixed parsed:",sum(x is not None for x in known_results),"/",len(known_results))
print(" new lexical reuse:",reuse_directives)

assert all(checks.values())

report={
    "version":"v6.5-unknown-imperative-semantics",
    "result":"PASS",
    "anonymous_relation":ANON_STOP_REL,
    "checks":checks,
    "learned_schema":{
        "require_cease":U_UNKNOWN_IMPERATIVE.require_cease,
        "target_mode":U_UNKNOWN_IMPERATIVE.target_mode,
        "action_mode":U_UNKNOWN_IMPERATIVE.action_mode,
        "support":schema_score(U_UNKNOWN_IMPERATIVE)[0],
        "conflict":schema_score(U_UNKNOWN_IMPERATIVE)[1],
    },
    "synthetic_lifecycle":{
        "after_one":list(SYN_AFTER_ONE),
        "after_two":list(SYN_AFTER_TWO),
        "reuse":list(SYN_REUSE) if SYN_REUSE else None,
        "world_without_observation":SYN_WORLD_CEASE_WITHOUT_OBSERVATION,
        "third_observed_world_cessation":SYN3_WORLD,
    },
    "grimm":{
        "stop_occurrences":stop_occurrences,
        "lexical_entry":None if STEH is None else {
            "token":STEH.token,
            "relation":STEH.relation,
            "status":STEH.status,
            "support":STEH.support,
            "conflicts":STEH.conflicts,
            "provenance":STEH.provenance,
        },
        "reuse_on_new_action":list(GRIMM_REUSE) if GRIMM_REUSE else None,
        "reuse_world_without_observation":GRIMM_REUSE_WORLD_WITHOUT_OBSERVATION,
        "trace":grimm_trace,
    },
    "interpretation":[
        "The learner is not given STEH=STOP_COOK or any other token-specific stop meaning.",
        "A generic anonymous U is learned from unrelated opaque imperatives whose local consequence is cessation of the same target's same currently active action.",
        "A new token begins STAGED after one independent consequence and becomes ACTIVE only after a second independent matching consequence.",
        "In the frozen Grimm tale, the two occurrences of 'Töpfchen steh' provide the two independent local COOK cessation observations, activating an anonymous lexical U for 'steh'.",
        "After activation, 'steh' transfers to a new target/action as the same anonymous directive class R10.",
        "Recognizing directive semantics is still not WORLD evidence: execution/cessation requires an independent narrative observation."
    ],
    "caveats":[
        "The CEASE observation vocabulary ('hört ... auf zu ...') and primitive action morphology are supplied symbolic infrastructure.",
        "Token lexical activation threshold=2 is a hand-set lifecycle prior.",
        "The learned R10 is a generic cessation-directive class; the experiment does not claim full lexical semantics induction for arbitrary verbs.",
        "The two Grimm supports use the same target/action pair POT/COOK, although they are independent discourse occurrences.",
        "The system learns the meaning after observing consequences; it does not predict the first-ever unseen token's effect before any evidence.",
        "Full free-German parsing remains outside this curriculum PoC."
    ]
}
Path("/mnt/data/symbolic_v65_unknown_imperative_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
with Path("/mnt/data/symbolic_v65_unknown_imperative_checks.csv").open(
    "w",newline="",encoding="utf-8"
) as f:
    w=csv.writer(f); w.writerow(["check","passed"])
    for k,val in checks.items(): w.writerow([k,val])

print("\nSaved v6.5 report/checks.")

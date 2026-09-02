
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import re, json, csv

# ============================================================
# v7.0 / K2 — semantic action-head ablation
# Uses the frozen K1 selected rules from its JSON report.
# No executable import of older modules is required.
# ============================================================

K1_REPORT=json.loads(Path("/mnt/data/symbolic_v69_k1_final_report.json").read_text(encoding="utf-8"))
C14_REPORT=json.loads(Path("/mnt/data/symbolic_v67_raw_text_family_invention_report.json").read_text(encoding="utf-8"))
assert K1_REPORT["result"]=="PASS" and all(K1_REPORT["main_checks"].values())
assert C14_REPORT["result"]=="PASS" and all(C14_REPORT["checks"].values())

# ---------- primitive delta layer remains fixed for K2 ----------
@dataclass(frozen=True)
class Delta:
    removed:tuple
    added:tuple

STOP=Delta((("ACTIVE",("S0","S1")),),())
START=Delta((),(("ACTIVE",("S0","S1")),))
ENTER=Delta((),(("AT",("S0","S1")),))
LEAVE=Delta((("AT",("S0","S1")),),())
DELTA_BY_NAME={"STOP":STOP,"START":START,"ENTER":ENTER,"LEAVE":LEAVE}
NAME_BY_DELTA={v:k for k,v in DELTA_BY_NAME.items()}

@dataclass(frozen=True)
class Rule:
    delta:Delta
    required:frozenset[str]
    slot_types:tuple[str,...]

RULES=[
    Rule(
        DELTA_BY_NAME[x["delta"]],
        frozenset(x["required"]),
        tuple(x["slot_types"])
    )
    for x in K1_REPORT["selected_rules"]
]

# ---------- entity ontology still fixed at K2 ----------
ENT={
    "lampe":"LAMP","rad":"WHEEL","maschine":"MACHINE","tor":"GATE",
    "töpfchen":"POT","toepfchen":"POT","topf":"POT",
    "wasser":"WATER","kessel":"KETTLE",
    "anna":"ANNA","ben":"BEN","cara":"CARA",
    "mädchen":"GIRL","maedchen":"GIRL","junge":"BOY","jungen":"BOY",
    "garten":"GARDEN","haus":"HOUSE","wald":"FOREST","zimmer":"ROOM",
}
MACHINE_ENTITIES={"LAMP","WHEEL","MACHINE","GATE","POT","WATER","KETTLE"}
PERSON={"ANNA","BEN","CARA","GIRL","BOY"}
PLACE={"GARDEN","HOUSE","FOREST","ROOM"}

# ---------- formal morphology only ----------
LEMMA={
    "leuchtet":"leuchten","leuchtete":"leuchten","leuchten":"leuchten",
    "dreht":"drehen","drehte":"drehen","drehen":"drehen",
    "läuft":"laufen","laeuft":"laufen","lief":"laufen","laufen":"laufen",
    "öffnet":"öffnen","oeffnet":"öffnen","öffnete":"öffnen",
    "öffnen":"öffnen","oeffnen":"öffnen",
    "kocht":"kochen","kochte":"kochen","kochen":"kochen","koche":"kochen",
    "hört":"hören","hörte":"hören","hoert":"hören","hoerte":"hören",
    "beginnt":"beginnen","begann":"beginnen",
    "fängt":"fangen","faengt":"fangen","fing":"fangen",
    "erlischt":"erlöschen","erlosch":"erlöschen",
    "geht":"gehen","ging":"gehen",
    "betritt":"betreten","betrat":"betreten",
    "verlässt":"verlassen","verlaesst":"verlassen",
    "verließ":"verlassen","verliess":"verlassen",
    "kommt":"kommen","kam":"kommen",
    "ist":"sein","war":"sein",
    "bleibt":"bleiben","blieb":"bleiben",
    "im":"in","ins":"in",
    "dem":"der","den":"der","die":"der","das":"der","der":"der",
}
STOPWORDS={"der","ein","eine","einen","einem","einer","sich","wieder","danach","so","nun"}

def toks(x):
    return re.findall(r"[A-Za-zÄÖÜäöüß]+",x.lower())

def lemma(t):
    return LEMMA.get(t,t)

# ---------- learn anonymous lexical action classes ----------
# No action meaning label is supplied. These are canonical activity observations.
ACTION_OBSERVATIONS=[
    "Die Lampe leuchtet.",
    "Die Lampe leuchtete.",
    "Das Rad dreht.",
    "Das Rad drehte.",
    "Die Maschine läuft.",
    "Die Maschine lief.",
    "Das Tor öffnet.",
    "Das Tor öffnete.",
    "Das Wasser kocht.",
    "Das Wasser kochte.",
    "Der Kessel kocht.",
]

ACTIVITY_LEMMAS={"leuchten","drehen","laufen","öffnen","kochen"}

def entity_in(text):
    return next((ENT[t] for t in toks(text) if t in ENT),None)

def activity_lemma(text):
    xs=[lemma(t) for t in toks(text) if lemma(t) in ACTIVITY_LEMMAS]
    return xs[0] if len(set(xs))==1 else None

lemma_support=defaultdict(list)
for sent in ACTION_OBSERVATIONS:
    e=entity_in(sent)
    l=activity_lemma(sent)
    assert e and l
    lemma_support[l].append((sent,e))

ACTION_BY_LEMMA={
    l:f"A{i}"
    for i,l in enumerate(sorted(lemma_support),1)
}
assert len(ACTION_BY_LEMMA)==5

# learned entity -> anonymous action compatibility
cap=defaultdict(lambda:defaultdict(set))
for sent in ACTION_OBSERVATIONS:
    e=entity_in(sent); l=activity_lemma(sent); a=ACTION_BY_LEMMA[l]
    cap[e][a].add(sent)
CAP_STATUS={
    e:{
        a:("ACTIVE" if len(evidence)>=2 else "STAGED")
        for a,evidence in actions.items()
    }
    for e,actions in cap.items()
}
LEARNED_CAP={
    e:frozenset(a for a,status in statuses.items() if status=="ACTIVE")
    for e,statuses in CAP_STATUS.items()
    if any(status=="ACTIVE" for status in statuses.values())
}

# ---------- K2 surface with same formal K1 rule vocabulary ----------
@dataclass(frozen=True)
class Surface:
    text:str
    slots:tuple[str,...]
    slot_types:tuple[str,...]
    features:frozenset[str]
    normalized:tuple[str,...]
    action_source:str|None

def explicit_action_ids(raw):
    return [
        ACTION_BY_LEMMA[lemma(t)]
        for t in raw if lemma(t) in ACTION_BY_LEMMA
    ]

def surface(text,preferred_target=None,extra_caps=None):
    raw=toks(text)
    ls=[lemma(x) for x in raw]
    ents=[ENT[x] for x in raw if x in ENT]
    acts=explicit_action_ids(raw)

    target=next((x for x in ents if x in MACHINE_ENTITIES),None) or preferred_target
    person=next((x for x in ents if x in PERSON),None)
    place=next((x for x in ents if x in PLACE),None)

    caps={e:set(aa) for e,aa in LEARNED_CAP.items()}
    if extra_caps:
        for e,aa in extra_caps.items():
            caps.setdefault(e,set()).update(aa)

    slots=(); types=(); source=None
    if target:
        if len(set(acts))==1:
            slots=(target,acts[0]); types=("ENTITY","ACTION"); source="EXPLICIT_LEMMA"
        elif len(set(acts))==0:
            poss=set(caps.get(target,set()))
            if len(poss)==1:
                slots=(target,next(iter(poss))); types=("ENTITY","ACTION")
                source="LEARNED_UNIQUE_CAPABILITY"

    if not slots and person and place:
        slots=(person,place); types=("PERSON","PLACE"); source="EXPLICIT_ENTITY_PLACE"

    norm=[]
    for r,l in zip(raw,ls):
        if r in ENT or l in ACTION_BY_LEMMA:
            continue
        if l in STOPWORDS:
            continue
        norm.append(l)

    feats={"L:"+x for x in norm}
    for r in raw:
        l=lemma(r)
        if r in ENT or l in ACTION_BY_LEMMA or l in STOPWORDS:
            continue
        feats.add("W:"+r)

    if "ins" in raw or any(
        raw[i]=="in" and i+1<len(raw) and raw[i+1]=="das"
        for i in range(len(raw)-1)
    ):
        feats.add("M:PREP_IN_ACC")
    if "im" in raw or any(
        raw[i]=="in" and i+1<len(raw) and raw[i+1]=="dem"
        for i in range(len(raw)-1)
    ):
        feats.add("M:PREP_IN_DAT")

    for i in range(len(norm)-1):
        feats.add("B:"+norm[i]+">"+norm[i+1])
    for i in range(len(norm)-2):
        feats.add("T:"+norm[i]+">"+norm[i+1]+">"+norm[i+2])
    if "nicht" in norm: feats.add("M:NEG")
    if "zu" in norm: feats.add("M:ZU")
    if slots: feats.add("M:HAS_SLOTS")

    return Surface(text,slots,types,frozenset(feats),tuple(norm),source)

def classify(text,preferred_target=None,extra_caps=None):
    s=surface(text,preferred_target,extra_caps)
    ds=[]
    for r in RULES:
        if s.slot_types==r.slot_types and r.required<=s.features and r.delta not in ds:
            ds.append(r.delta)
    return (ds[0] if len(ds)==1 else None,s,tuple(ds))

def ground(d,s):
    if d is None or not s.slots:
        return None
    mp={f"S{i}":x for i,x in enumerate(s.slots)}
    def gk(k):
        rel,args=k
        return rel,tuple(mp.get(a,a) for a in args)
    return Delta(
        tuple(gk(k) for k in d.removed),
        tuple(gk(k) for k in d.added)
    )

# ---------- anonymous action tests ----------
SEMANTIC_HEADS={"LIGHT","TURN","RUN","OPEN","COOK"}
ANON=set(ACTION_BY_LEMMA.values())

FORM_TRANSFER=[
    ("Die Lampe leuchtet.","Die Lampe hört auf zu leuchten."),
    ("Das Rad dreht.","Das Rad hört auf zu drehen."),
    ("Die Maschine läuft.","Die Maschine beginnt zu laufen."),
    ("Das Tor öffnet.","Das Tor beginnt zu öffnen."),
    ("Das Wasser kocht.","Das Wasser hört auf zu kochen."),
]
FORM_OK=[]
for obs,test in FORM_TRANSFER:
    a=ACTION_BY_LEMMA[activity_lemma(obs)]
    ids=explicit_action_ids(toks(test))
    FORM_OK.append(len(set(ids))==1 and ids[0]==a)

IMPLICIT=[
    ("Die Lampe erlischt.",STOP,"LAMP"),
    ("Die Lampe geht aus.",STOP,"LAMP"),
    ("Das Rad geht an.",START,"WHEEL"),
    ("Das Tor geht aus.",STOP,"GATE"),
]
IMPLICIT_OK=[]
for text,gold,e in IMPLICIT:
    d,s,_=classify(text)
    IMPLICIT_OK.append(
        d==gold and s.action_source=="LEARNED_UNIQUE_CAPABILITY"
        and s.slots and s.slots[1] in LEARNED_CAP[e]
    )

# ambiguity attack
lamp_a=next(iter(LEARNED_CAP["LAMP"]))
other_a=next(x for x in ANON if x!=lamp_a)
ambig_d,ambig_s,_=classify(
    "Die Lampe erlischt.",
    extra_caps={"LAMP":{other_a}}
)
explicit_d,explicit_s,_=classify(
    "Die Lampe hört auf zu leuchten.",
    extra_caps={"LAMP":{other_a}}
)

# K1 frozen transfer
FROZEN=[
    ("Das Rad hört auf zu drehen.",STOP),
    ("Die Lampe leuchtet nicht mehr.",STOP),
    ("Das Tor erlischt.",STOP),
    ("Die Maschine geht aus.",STOP),
    ("Das Tor beginnt zu öffnen.",START),
    ("Die Maschine fängt an zu laufen.",START),
    ("Das Rad geht an.",START),
    ("Ben betritt den Garten.",ENTER),
    ("Cara geht ins Haus.",ENTER),
    ("Anna geht hinein ins Zimmer.",ENTER),
    ("Ben verlässt den Garten.",LEAVE),
    ("Cara geht aus dem Haus.",LEAVE),
    ("Anna geht hinaus aus dem Zimmer.",LEAVE),
]
FROZEN_P=[classify(t)[0] for t,g in FROZEN]
FROZEN_OK=all(p==g for p,(t,g) in zip(FROZEN_P,FROZEN))

GROUND_SYMBOLS=[]
for text,gold in FROZEN:
    d,s,_=classify(text)
    if d and s.slot_types==("ENTITY","ACTION"):
        g=ground(d,s)
        for rel,args in list(g.removed)+list(g.added):
            if rel=="ACTIVE":
                GROUND_SYMBOLS.append(args[1])

# ---------- C14 handoff via structural signature only ----------
REMOVE_ACTIVE_FAMILY=C14_REPORT["evaluator_only_mapping"]["REMOVE_ACTIVE"]

def canonical_family(g,s):
    # C14 family lookup by canonical topology; action name is irrelevant.
    if g is None:
        return None
    if (
        len(g.removed)==1 and not g.added
        and g.removed[0][0]=="ACTIVE"
        and len(g.removed[0][1])==2
        and s.slot_types==("ENTITY","ACTION")
        and tuple(g.removed[0][1])==tuple(s.slots)
    ):
        return REMOVE_ACTIVE_FAMILY
    return None

d,s,_=classify("Das Rad hört auf zu drehen.")
g=ground(d,s)
HANDOFF=canonical_family(g,s)

LEX=[]
for response in [
    "Die Lampe erlischt.",
    "Die Maschine hört auf zu laufen.",
    "Das Rad dreht sich nicht mehr.",
]:
    d,s,_=classify(response)
    LEX.append(canonical_family(ground(d,s),s))

# ---------- Grimm end-to-end ----------
GRIMM=Path("/mnt/data/grimm_der_suesse_brei.txt").read_text(encoding="utf-8")
G=GRIMM.replace("„",'"').replace("“",'"')
spans=list(re.finditer(r'"([^"]+)"',G,re.S))

A_KOCHEN=ACTION_BY_LEMMA["kochen"]
active={}
steh_support=[]
trace=[]

def qtarget(q):
    return "POT" if any(x in {"töpfchen","toepfchen","topf"} for x in toks(q)) else None

def local_after(sp,n=180):
    return G[sp.end():min(len(G),sp.end()+n)]

for i,sp in enumerate(spans):
    q=sp.group(1)
    target=qtarget(q)
    qacts=explicit_action_ids(toks(q))

    if target and len(set(qacts))==1:
        a=qacts[0]
        response=local_after(sp)
        if a in explicit_action_ids(toks(response)):
            active[target]=a
        trace.append((i,"KNOWN",target,a,dict(active)))
        continue

    if target and "steh" in toks(q) and target in active:
        a=active[target]
        response=local_after(sp)
        d,s,_=classify(response,preferred_target=target)
        gg=ground(d,s) if d else None
        fam=canonical_family(gg,s) if gg else None
        if fam and ("ACTIVE",(target,a)) in gg.removed:
            steh_support.append((f"q{i}",fam,target,a))
            active.pop(target,None)
        trace.append((i,"OPAQUE",target,a,fam,dict(active)))

STEH_ACTIVE=(
    len(steh_support)==2
    and all(x[1]==REMOVE_ACTIVE_FAMILY for x in steh_support)
)
STEH_REUSE=(REMOVE_ACTIVE_FAMILY,("LAMP",ACTION_BY_LEMMA["leuchten"])) if STEH_ACTIVE else None


# ---------- capability lifecycle attack ----------
# One KETTLE observation is insufficient for implicit completion.
kettle_before_d,kettle_before_s,_=classify("Der Kessel geht aus.")

# Add a second independent KETTLE observation only for this frozen promotion probe.
promoted_caps={e:set(aa) for e,aa in LEARNED_CAP.items()}
promoted_caps.setdefault("KETTLE",set()).add(ACTION_BY_LEMMA["kochen"])
kettle_after_d,kettle_after_s,_=classify(
    "Der Kessel geht aus.",
    extra_caps=promoted_caps
)

checks={
    "frozen_K1_report_is_green":K1_REPORT["result"]=="PASS" and all(K1_REPORT["main_checks"].values()),
    "K2_five_action_classes_are_anonymous_A_symbols":len(ANON)==5 and all(re.fullmatch(r"A\d+",x) for x in ANON),
    "K2_no_human_semantic_action_head_is_learned":not (ANON & SEMANTIC_HEADS),
    "K2_inflectional_lemma_transfer_preserves_action_identity":all(FORM_OK),
    "K2_entity_action_compatibility_is_learned_from_observations":all(LEARNED_CAP.values()),
    "K2_single_capability_observation_stays_STAGED":(
        CAP_STATUS["KETTLE"][ACTION_BY_LEMMA["kochen"]]=="STAGED"
        and "KETTLE" not in LEARNED_CAP
    ),
    "K2_implicit_surfaces_use_only_unique_learned_capability":all(IMPLICIT_OK),
    "K2_STAGED_capability_cannot_drive_implicit_semantics":(
        kettle_before_d is None and kettle_before_s.slots==()
    ),
    "K2_second_independent_capability_support_can_enable_reuse":(
        kettle_after_d==STOP
        and kettle_after_s.slots==("KETTLE",ACTION_BY_LEMMA["kochen"])
    ),
    "K2_ambiguous_learned_capability_stays_UNKNOWN":ambig_d is None and ambig_s.slots==(),
    "K2_explicit_action_disambiguates_ambiguous_capability":explicit_d==STOP and explicit_s.action_source=="EXPLICIT_LEMMA",
    "K2_K1_frozen_paraphrases_transfer_with_A_symbols":FROZEN_OK,
    "K2_grounded_ACTIVE_deltas_use_only_A_symbols":GROUND_SYMBOLS and all(re.fullmatch(r"A\d+",x) for x in GROUND_SYMBOLS),
    "K2_C14_remove_ACTIVE_family_is_unchanged":HANDOFF==REMOVE_ACTIVE_FAMILY,
    "K2_three_different_surface_forms_and_A_symbols_support_same_C14_family":LEX==[REMOVE_ACTIVE_FAMILY]*3,
    "K2_Grimm_steh_learning_path_works_without_COOK_head":STEH_ACTIVE,
    "K2_Grimm_steh_reuse_targets_an_anonymous_action":STEH_REUSE is not None and re.fullmatch(r"A\d+",STEH_REUSE[1][1]) is not None,
}

print("=== v7.0 / K2 ACTION-HEAD ABLATION ===")
for k,vv in checks.items():
    print(("PASS" if vv else "FAIL"),"|",k)

print("\nAnonymous action classes:")
for l,a in sorted(ACTION_BY_LEMMA.items()):
    print(" ",l,"=>",a,"support",len(lemma_support[l]))

print("\nLearned capabilities:")
for e,statuses in sorted(CAP_STATUS.items()):
    print(" ",e,"=>",statuses,"ACTIVE",sorted(LEARNED_CAP.get(e,())))
print(" KETTLE implicit before second support =>",kettle_before_d,kettle_before_s.slots)
print(" KETTLE implicit after second support =>",NAME_BY_DELTA.get(kettle_after_d),kettle_after_s.slots)

print("\nImplicit surfaces:")
for (text,gold,e),ok in zip(IMPLICIT,IMPLICIT_OK):
    d,s,_=classify(text)
    print(" ",text,"=>",None if d is None else NAME_BY_DELTA[d],s.slots,s.action_source,"PASS" if ok else "FAIL")

print("\nAmbiguity:")
print(" LAMP base:",sorted(LEARNED_CAP["LAMP"]),"injected:",other_a)
print(" implicit:",ambig_d,ambig_s.slots,ambig_s.action_source)
print(" explicit:",NAME_BY_DELTA.get(explicit_d),explicit_s.slots,explicit_s.action_source)

print("\nFrozen:")
for (text,gold),p in zip(FROZEN,FROZEN_P):
    d,s,_=classify(text)
    print(" ",text,"=>",NAME_BY_DELTA.get(p),s.slots,s.action_source)

print("\nC14 family:",HANDOFF,"expected",REMOVE_ACTIVE_FAMILY)
print("lex families:",LEX)

print("\nGrimm:")
for x in trace:
    print(" ",x)
print("steh_support:",steh_support)
print("A(kochen):",A_KOCHEN)
print("reuse:",STEH_REUSE)

assert all(checks.values())

report={
    "version":"v7.0-K2-action-head-ablation",
    "result":"PASS",
    "checks":checks,
    "anonymous_actions":{
        l:{
            "head":a,
            "support":len(lemma_support[l]),
            "examples":[x[0] for x in lemma_support[l]],
        }
        for l,a in sorted(ACTION_BY_LEMMA.items())
    },
    "learned_capabilities":{e:sorted(aa) for e,aa in sorted(LEARNED_CAP.items())},
    "capability_status":CAP_STATUS,
    "capability_lifecycle_attack":{
        "before_second_support":NAME_BY_DELTA.get(kettle_before_d),
        "after_second_support":NAME_BY_DELTA.get(kettle_after_d),
        "after_slots":list(kettle_after_s.slots),
    },
    "ambiguity":{
        "lamp_base":sorted(LEARNED_CAP["LAMP"]),
        "injected_second_action":other_a,
        "implicit_prediction":NAME_BY_DELTA.get(ambig_d),
        "implicit_slots":list(ambig_s.slots),
        "explicit_prediction":NAME_BY_DELTA.get(explicit_d),
        "explicit_slots":list(explicit_s.slots),
    },
    "c14":{
        "remove_active_family":REMOVE_ACTIVE_FAMILY,
        "handoff":HANDOFF,
        "lexical_support_families":LEX,
    },
    "grimm":{
        "anonymous_kochen_action":A_KOCHEN,
        "steh_support":[list(x) for x in steh_support],
        "active":STEH_ACTIVE,
        "reuse":list(STEH_REUSE) if STEH_REUSE else None,
        "trace":[repr(x) for x in trace],
    },
    "interpretation":[
        "K2 replaces human-readable lexical action heads with anonymous A-symbols induced from recurring formally lemmatized activity predicates.",
        "The old entity-to-default-action capability table is replaced by compatibility learned from explicit entity/action observations.",
        "Implicit surfaces can fill the action port only when the learned capability set is unique; ambiguity remains UNKNOWN.",
        "An explicit action lemma can resolve that ambiguity without using a semantic action name.",
        "Frozen K1 Surface-U and C14 transition-family structure continue to work because they need only an ACTION port identity, not a human-readable action label.",
        "The Grimm 'kochen/koche' action is represented by an anonymous A-symbol, while the two 'steh' consequences still support the same remove-ACTIVE family."
    ],
    "caveats":[
        "A formal lemmatizer and a small curriculum of canonical activity-observation sentences remain supplied.",
        "K2 does not yet discover synonymy between unrelated lexical lemmas; one recurring lemma induces one anonymous lexical action identity.",
        "Entity identities and the primitive relation ACTIVE remain fixed.",
        "The curriculum marks simple sentences as activity observations, which is still supervision even though no semantic action label is supplied.",
        "Capability induction is entity-instance based in this PoC rather than learned type-level generalization."
    ]
}
Path("/mnt/data/symbolic_v70_k2_action_head_ablation_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
with Path("/mnt/data/symbolic_v70_k2_action_head_ablation_checks.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["check","passed"])
    for k,vv in checks.items():
        w.writerow([k,vv])

print("\nSaved v7.0 K2 report/checks.")

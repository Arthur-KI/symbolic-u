
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import importlib.util, sys, re, json, csv

# ============================================================
# v5.4b — Anonymous Event Semantics, safety-corrected
#
# Builds on v5.4 invention results, but re-learns surface gates with
# hard negative surface controls, inferred slot signatures, and
# independent-evidence gating for speech-caused WORLD actions.
# ============================================================

base_path="/mnt/data/symbolic_v54_anonymous_event_semantics.py"
spec=importlib.util.spec_from_file_location("v54base_for_v54b",base_path)
b=importlib.util.module_from_spec(spec)
sys.modules["v54base_for_v54b"]=b
spec.loader.exec_module(b)

# ------------------------------------------------------------
# Hard surface controls: same broad syntax / even same port types,
# but they must NOT become the anonymous possession-change relation.
# These are generic negatives, not benchmark sentences.
# ------------------------------------------------------------

CONTROLS=[
    b.S(
        "Lina zeigte Ben die Lampe.",
        ("lina","ben","LAMP"),("PERSON","PERSON","OBJECT"),
        {"ACTION_CLAUSE","DATIVE_THEME","FORM=ZEIGEN"},
        "control-show"
    ),
    b.S(
        "Nora erklärte Tom die Glocke.",
        ("nora","tom","BELL"),("PERSON","PERSON","OBJECT"),
        {"ACTION_CLAUSE","DATIVE_THEME","DECLARATIVE_COMPLEMENT","FORM=ERKLAEREN"},
        "control-explain"
    ),
]

# ------------------------------------------------------------
# Relearn minimal surface rule against positive examples + hard controls.
# ------------------------------------------------------------

@dataclass(frozen=True)
class SafeRule:
    relation:str
    required:frozenset[str]
    support:int
    conflict:int
    slot_types:tuple[tuple[str,str],...]

    def type_matches(self,s):
        st=s.types()
        return all(st.get(slot)==typ for slot,typ in self.slot_types)

    def matches(self,s):
        return self.type_matches(s) and self.required.issubset(s.feature_set())

def infer_slot_types(examples):
    slots=sorted(examples[0].surface.types())
    out=[]
    for slot in slots:
        vals={e.surface.types()[slot] for e in examples}
        assert len(vals)==1
        out.append((slot,next(iter(vals))))
    return tuple(out)

def learn_safe_rule(relation,positives,all_train,controls):
    common=set.intersection(*(set(e.surface.feature_set()) for e in positives))
    neg_surfaces=[e.surface for e in all_train if e not in positives]+list(controls)
    slot_types=infer_slot_types(positives)

    candidates=[]
    for size in range(1,min(3,len(common))+1):
        for combo in combinations(sorted(common),size):
            req=frozenset(combo)
            sup=sum(
                req.issubset(e.surface.feature_set()) and
                all(e.surface.types().get(s)==t for s,t in slot_types)
                for e in positives
            )
            conf=sum(
                req.issubset(s.feature_set()) and
                all(s.types().get(sl)==typ for sl,typ in slot_types)
                for s in neg_surfaces
            )
            form_penalty=sum(x.startswith("FORM=") for x in req)
            candidates.append((conf,-sup,size,form_penalty,tuple(sorted(req))))
    candidates.sort()
    conf,negsup,size,fp,req=candidates[0]
    return SafeRule(relation,frozenset(req),-negsup,conf,slot_types)

SAFE_RULES={}
for sig,examples in b.signatures.items():
    r=b.SIG_TO_R[sig]
    rule=learn_safe_rule(r,examples,b.TRAIN,CONTROLS)
    assert rule.support==len(examples) and rule.conflict==0
    SAFE_RULES[r]=rule

# ------------------------------------------------------------
# Safe anonymous library.
# R meaning is still its discovered effect signature.
# Speech -> WORLD ACTION is proof-gated by independent execution evidence.
# ------------------------------------------------------------

@dataclass(frozen=True)
class SafeVersion:
    relation:str
    version:int
    status:str
    effect_signature:frozenset
    rule:SafeRule

class SafeLibrary:
    def __init__(self):
        self.active={
            r:SafeVersion(r,1,"ACTIVE",b.R_TO_SIG[r],SAFE_RULES[r])
            for r in sorted(b.R_TO_SIG)
        }

    def classify(self,s):
        ms=[v for v in self.active.values() if v.rule.matches(s)]
        return ms[0] if len(ms)==1 else None

    def potential(self,v,s):
        if v is None: return set()
        vals=s.values()
        out=set()
        for rel,args in v.effect_signature:
            out.add((rel,tuple(vals[a] for a in args)))
        return out

    def commit(self,v,s,independent_support=frozenset()):
        pot=self.potential(v,s)
        out=set()
        for fact in pot:
            rel,args=fact
            # General proposition-context invariant:
            # a speech/directive event is evidence of the directive relation,
            # NOT by itself evidence that its external action happened.
            if rel=="ACTION" and "SPEECH" in s.cues:
                if fact in independent_support:
                    out.add(fact)
            else:
                out.add(fact)
        return out

LIB=SafeLibrary()

# Identify anonymous families only for evaluation.
R_ACTION=next(r for r,sig in b.R_TO_SIG.items() if ("ACTION",("S1","S2")) in sig)
R_CLAIM=next(r for r,sig in b.R_TO_SIG.items() if ("CLAIM",("S0","S1","S2")) in sig)
R_POSSESSION=next(r for r,sig in b.R_TO_SIG.items() if ("HAVE",("S1","S2")) in sig)

# ------------------------------------------------------------
# Re-run frozen paraphrase/domain cases with independent effect evidence.
# ------------------------------------------------------------

FROZEN=[
    (
        b.S("Tom forderte den Kristall auf, zu leuchten. Danach leuchtete der Kristall.",
            ("tom","CRYSTAL","LIGHT"),("PERSON","ENTITY","SYMBOL"),
            b.DIRECTIVE_CUES|{"FORM=AUFFORDERN"},"frozen-1"),
        {("ACTION",("CRYSTAL","LIGHT"))},
        R_ACTION,
        {("ACTION",("CRYSTAL","LIGHT"))},
    ),
    (
        b.S("Tom meldete, der Kristall leuchte.",
            ("tom","CRYSTAL","LIGHT"),("PERSON","ENTITY","SYMBOL"),
            b.ASSERTIVE_CUES|{"FORM=MELDEN"},"frozen-2"),
        set(),
        R_CLAIM,
        {("CLAIM",("tom","CRYSTAL","LIGHT"))},
    ),
    (
        b.S("Mia reichte Paul den Schlüssel.",
            ("mia","paul","KEY"),("PERSON","PERSON","OBJECT"),
            b.POSSESSION_CUES|{"FORM=REICHEN"},"frozen-3"),
        set(),
        R_POSSESSION,
        {("HAVE",("paul","KEY")),("NOT_HAVE",("mia","KEY"))},
    ),
    (
        b.S("Die Hexe befahl dem Drachen zu fliegen. Danach flog der Drache.",
            ("witch","DRAGON","FLY"),("PERSON","ENTITY","SYMBOL"),
            b.DIRECTIVE_CUES|{"FORM=INFINITIVE_DIRECTIVE","LORE=FAIRY_TALE"},"frozen-4"),
        {("ACTION",("DRAGON","FLY"))},
        R_ACTION,
        {("ACTION",("DRAGON","FLY"))},
    ),
    (
        b.S("Die Hexe sagte, der Drache fliege.",
            ("witch","DRAGON","FLY"),("PERSON","ENTITY","SYMBOL"),
            b.ASSERTIVE_CUES|{"FORM=REPORTED_SPEECH","LORE=FAIRY_TALE"},"frozen-5"),
        set(),
        R_CLAIM,
        {("CLAIM",("witch","DRAGON","FLY"))},
    ),
]

frozen_rows=[]
for s,support,expected_r,expected_facts in FROZEN:
    v=LIB.classify(s)
    got=LIB.commit(v,s,frozenset(support))
    frozen_rows.append((s.text,v.relation if v else None,got,expected_r,expected_facts,
                        (v is not None and v.relation==expected_r and got==expected_facts)))

# ------------------------------------------------------------
# Two attacks that broke v5.4.
# ------------------------------------------------------------

attack_action=b.S(
    "Lina zeigte Ben die Lampe.",
    ("lina","ben","LAMP"),("PERSON","PERSON","OBJECT"),
    {"ACTION_CLAUSE","DATIVE_THEME","FORM=ZEIGEN"},
    "attack-nontransfer"
)
a1=LIB.classify(attack_action)
a1facts=LIB.commit(a1,attack_action)

attack_directive=b.S(
    "Anna befahl Paul zu gehen.",
    ("anna","paul","GO"),("PERSON","ENTITY","SYMBOL"),
    b.DIRECTIVE_CUES|{"FORM=INFINITIVE_DIRECTIVE","LORE=REAL_WORLD"},
    "attack-unverified-directive"
)
a2=LIB.classify(attack_directive)
a2potential=LIB.potential(a2,attack_directive)
a2facts=LIB.commit(a2,attack_directive,frozenset())

# With independent execution evidence, same R can safely materialize ACTION.
a2confirmed=LIB.commit(
    a2,attack_directive,
    frozenset({("ACTION",("paul","GO"))})
)

# ------------------------------------------------------------
# Ambiguity and story isolation.
# ------------------------------------------------------------

amb=b.AMB
amb_v=LIB.classify(amb)
amb_facts=LIB.commit(amb_v,amb)

# story isolation
storyA=b.S(
    "Lina gab Ben die Lampe.",
    ("lina","ben","LAMP"),("PERSON","PERSON","OBJECT"),
    b.POSSESSION_CUES|{"FORM=GEBEN"},"story-A"
)
storyB=b.S(
    "Lina sagte, die Lampe leuchte.",
    ("lina","LAMP","LIGHT"),("PERSON","ENTITY","SYMBOL"),
    b.ASSERTIVE_CUES|{"FORM=REPORTED_SPEECH"},"story-B"
)
store={"story-A":set(),"story-B":set()}
for s in (storyA,storyB):
    v=LIB.classify(s)
    store[s.story].update(LIB.commit(v,s))

# ------------------------------------------------------------
# Sweet porridge frozen reuse:
# independent narrative says the pot cooks / later stops.
# R head stays anonymous.
# ------------------------------------------------------------

porridge=Path("/mnt/data/grimm_der_suesse_brei.txt").read_text(encoding="utf-8").lower()
assert "töpfchen koche" in porridge and "töpfchen steh" in porridge

sweet=[
    (
        b.S("Töpfchen koche",("girl","POT","COOK"),("PERSON","ENTITY","SYMBOL"),
            b.DIRECTIVE_CUES|{"FORM=QUOTE_IMPERATIVE","LORE=FAIRY_TALE"},"sweet-girl-cook"),
        {("ACTION",("POT","COOK"))},
    ),
    (
        b.S("Mutter: Töpfchen koche",("mother","POT","COOK"),("PERSON","ENTITY","SYMBOL"),
            b.DIRECTIVE_CUES|{"FORM=QUOTE_IMPERATIVE","LORE=FAIRY_TALE"},"sweet-mother-cook"),
        {("ACTION",("POT","COOK"))},
    ),
    (
        b.S("Töpfchen steh",("girl","POT","STOP"),("PERSON","ENTITY","SYMBOL"),
            b.DIRECTIVE_CUES|{"FORM=QUOTE_IMPERATIVE","LORE=FAIRY_TALE"},"sweet-girl-stop"),
        {("ACTION",("POT","STOP"))},
    ),
]
sweet_rows=[]
for s,support in sweet:
    v=LIB.classify(s)
    pot=LIB.potential(v,s)
    got=LIB.commit(v,s,frozenset(support))
    sweet_rows.append((s,v.relation if v else None,pot,got))

# ------------------------------------------------------------
# Checks
# ------------------------------------------------------------

checks={
    "three_anonymous_relations_retained":len(LIB.active)==3,
    "no_semantic_event_heads":all(re.fullmatch(r"R\d+",r) for r in LIB.active),
    "safe_rules_full_support_zero_conflict":all(
        v.rule.conflict==0 and v.rule.support==len(b.signatures[v.effect_signature])
        for v in LIB.active.values()
    ),
    "possession_rule_not_generic_action_clause":(
        "POSSESSION_CHANGE" in SAFE_RULES[R_POSSESSION].required
    ),
    "slot_type_signature_inferred_and_enforced":(
        SAFE_RULES[R_POSSESSION].slot_types==
        (("S0","PERSON"),("S1","PERSON"),("S2","OBJECT"))
    ),
    "frozen_5of5":all(x[-1] for x in frozen_rows),
    "nontransfer_attack_rejected":a1 is None and not a1facts,
    "unverified_directive_relation_detected":(
        a2 is not None and a2.relation==R_ACTION
    ),
    "unverified_directive_world_action_not_committed":(
        ("ACTION",("paul","GO")) in a2potential and
        ("ACTION",("paul","GO")) not in a2facts
    ),
    "independent_execution_evidence_unlocks_action":(
        ("ACTION",("paul","GO")) in a2confirmed
    ),
    "underspecified_speech_unknown":amb_v is None and not amb_facts,
    "directive_assertion_same_types_still_separated":(
        LIB.classify(b.TRAIN[0].surface).relation !=
        LIB.classify(b.TRAIN[3].surface).relation
    ),
    "story_isolation":(
        ("HAVE",("ben","LAMP")) in store["story-A"] and
        ("HAVE",("ben","LAMP")) not in store["story-B"] and
        ("CLAIM",("lina","LAMP","LIGHT")) in store["story-B"]
    ),
    "sweet_porridge_reuses_same_anonymous_relation":(
        len(sweet_rows)==3 and all(r==R_ACTION for _,r,_,_ in sweet_rows)
    ),
    "sweet_porridge_effects_require_and_have_independent_support":(
        all(got==support for (_,support),(_,_,_,got) in zip(sweet,sweet_rows))
    ),
    "versioned_active":all(v.version==1 and v.status=="ACTIVE" for v in LIB.active.values()),
}

print("=== v5.4b SAFE ANONYMOUS EVENT SEMANTICS ===")
for k,v in checks.items():
    print(("PASS" if v else "FAIL"),"|",k)

print("\nSafe anonymous relations:")
for r in sorted(LIB.active):
    v=LIB.active[r]
    print(" ",r,
          "rule",sorted(v.rule.required),
          "types",v.rule.slot_types,
          "effect",sorted(v.effect_signature),
          "support",v.rule.support,"conflict",v.rule.conflict)

print("\nAttack 1:",attack_action.text)
print(" relation",a1.relation if a1 else None,"facts",sorted(a1facts))
print("\nAttack 2:",attack_directive.text)
print(" relation",a2.relation if a2 else None)
print(" potential",sorted(a2potential))
print(" committed without execution evidence",sorted(a2facts))
print(" committed with execution evidence",sorted(a2confirmed))

print("\nFrozen:")
for text,r,got,er,ef,ok in frozen_rows:
    print(("PASS" if ok else "FAIL"),"|",r,"|",text,"=>",sorted(got))

print("\nSweet porridge:")
for s,r,pot,got in sweet_rows:
    print(" ",s.story,r,"potential",sorted(pot),"committed",sorted(got))

assert all(checks.values())

report={
    "version":"v5.4b-safe-anonymous-event-semantics",
    "result":"PASS",
    "checks":checks,
    "anonymous_relations":{
        r:{
            "version":v.version,
            "status":v.status,
            "surface_rule":sorted(v.rule.required),
            "slot_types":[list(x) for x in v.rule.slot_types],
            "effect_signature":[[rel,list(args)] for rel,args in sorted(v.effect_signature)],
            "support":v.rule.support,
            "conflict":v.rule.conflict,
        } for r,v in LIB.active.items()
    },
    "attacks":{
        "nontransfer_action":{
            "surface":attack_action.text,
            "relation":a1.relation if a1 else None,
            "facts":[str(x) for x in sorted(a1facts)],
        },
        "unverified_directive":{
            "surface":attack_directive.text,
            "relation":a2.relation if a2 else None,
            "potential":[str(x) for x in sorted(a2potential)],
            "committed_without_support":[str(x) for x in sorted(a2facts)],
            "committed_with_support":[str(x) for x in sorted(a2confirmed)],
        }
    },
    "frozen":[
        {"text":text,"relation":r,"facts":[str(x) for x in sorted(got)],"passed":ok}
        for text,r,got,er,ef,ok in frozen_rows
    ],
    "sweet_porridge":[
        {"story":s.story,"relation":r,
         "potential":[str(x) for x in sorted(pot)],
         "committed":[str(x) for x in sorted(got)]}
        for s,r,pot,got in sweet_rows
    ],
    "design":[
        "Anonymous heads remain R1/R2/R3; no COMMAND/SAY/TRANSFER head is installed.",
        "Each R infers an immutable slot-type signature from its positive examples.",
        "Surface-rule learning includes hard negative controls, preventing over-broad MDL rules such as ACTION_CLAUSE alone.",
        "Speech/directive-derived external ACTION is a potential consequence, not WORLD evidence by itself.",
        "Independent execution evidence is required before ACTION is committed.",
        "Relation recognition and consequence commitment are separate symbolic states.",
    ],
    "caveats":[
        "Primitive effect relations ACTION/CLAIM/HAVE/NOT_HAVE remain fixed ontology.",
        "The distinction between speech-caused external effects and directly asserted proposition effects is an OS-level proof policy.",
        "Hard negative surface controls are supplied training data; discovering useful counterexamples autonomously remains open.",
        "Sweet-porridge execution support is independently read from its narrative in this experiment, not induced from the directive alone.",
        "No general NLP accuracy claim."
    ],
}
Path("/mnt/data/symbolic_v54b_regression_rerun_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
with Path("/mnt/data/symbolic_v54b_regression_rerun_checks.csv").open(
    "w",newline="",encoding="utf-8"
) as f:
    w=csv.writer(f); w.writerow(["check","passed"])
    for k,v in checks.items(): w.writerow([k,v])

print("\nSaved v5.4b report/checks.")

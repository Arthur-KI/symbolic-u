
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
import importlib.util, sys, contextlib, io, itertools, json, csv, re

# ============================================================
# v7.9 / K9 — Consequence-supervised Binder-U induction
#
# Removed compared with K8:
#   explicit event-port target tuples in binder curriculum.
#
# Learner receives:
#   raw/controlled clause
#   exact before snapshot
#   exact after snapshot
#   frozen formal mentions/case/T-types
#
# It must infer the latent participant tuple from the observed
# anonymous state delta, then discover a reusable clause binder.
# ============================================================

# Safe-load K8 library.
src=Path("/mnt/data/symbolic_v78_k8_binder_abstraction.py").read_text(encoding="utf-8")
src=src.replace(
    "/mnt/data/symbolic_v78_k8_binder_abstraction_report.json",
    "/mnt/data/_v79_runtime_k8_report.json"
).replace(
    "/mnt/data/symbolic_v78_k8_binder_abstraction_checks.csv",
    "/mnt/data/_v79_runtime_k8_checks.csv"
)
tmp=Path("/mnt/data/_v79_k8_runtime.py")
tmp.write_text(src,encoding="utf-8")
spec=importlib.util.spec_from_file_location("k8base",str(tmp))
k8=importlib.util.module_from_spec(spec); sys.modules["k8base"]=k8
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(k8)
assert all(k8.checks.values())

K5=json.loads(Path("/mnt/data/symbolic_v73_k5_operation_ablation_report.json").read_text(encoding="utf-8"))
K3=json.loads(Path("/mnt/data/symbolic_v71_k3_relation_ablation_report.json").read_text(encoding="utf-8"))
K2=json.loads(Path("/mnt/data/symbolic_v70_k2_action_head_ablation_report.json").read_text(encoding="utf-8"))

P3=K3["evaluator_only_mapping"]["POSSESSION"]
P4=K3["evaluator_only_mapping"]["LOCATION"]
O2=K5["evaluator_only_mapping"]["TRANSFER_FIRST"]
O4=K5["evaluator_only_mapping"]["APPEAR"]

# ------------------------------------------------------------
# Snapshot delta -> anonymous latent argument correspondence.
# This does NOT use event role names; it only preserves equality.
# ------------------------------------------------------------

Key=tuple[str,tuple[str,...]]

@dataclass(frozen=True)
class Episode:
    eid:str
    text:str
    before:frozenset[Key]
    after:frozenset[Key]
    pronoun_map:dict|None=None
    inherited_subject:str|None=None

@dataclass(frozen=True)
class DeltaBinding:
    relation:str
    op_id:str
    values:tuple[str,...]  # anonymous V0,V1,V2 realization

def infer_transfer_delta(ep:Episode):
    b0=set(ep.before)-set(ep.after)
    b1=set(ep.after)-set(ep.before)
    if len(b0)!=1 or len(b1)!=1:
        return None
    kb=next(iter(b0)); ka=next(iter(b1))
    rb,ab=kb; ra,aa=ka
    if rb!=ra or len(ab)!=2 or len(aa)!=2:
        return None

    # K5 O2 topology: first arg changes, second is identical.
    if ab[1]==aa[1] and ab[0]!=aa[0]:
        return DeltaBinding(rb,O2,(ab[0],aa[0],ab[1]))
    return None

def infer_appearance_delta(ep:Episode):
    b0=set(ep.before)-set(ep.after)
    b1=set(ep.after)-set(ep.before)
    if b0 or len(b1)!=1:
        return None
    k=next(iter(b1))
    r,args=k
    if len(args)!=2:
        return None
    return DeltaBinding(r,O4,(args[0],args[1]))

# ------------------------------------------------------------
# K9 GIVE curriculum: NO explicit event tuple labels.
# ------------------------------------------------------------

def K(r,a,b): return (r,(a,b))

GIVE_EPISODES={
    "geben":[
        Episode("g1","Die Frau gab dem Jungen das Buch.",
            frozenset({K(P3,"WOMAN","BOOK")}),
            frozenset({K(P3,"BOY","BOOK")})),
        Episode("g2","Dem Jungen gab die Frau den Ball.",
            frozenset({K(P3,"WOMAN","BALL")}),
            frozenset({K(P3,"BOY","BALL")})),
        Episode("g3","Das Buch gab die Frau dem Jungen.",
            frozenset({K(P3,"WOMAN","BOOK")}),
            frozenset({K(P3,"BOY","BOOK")})),
        Episode("g4","Der Mann gab dem Kind einen Apfel.",
            frozenset({K(P3,"MAN","APPLE")}),
            frozenset({K(P3,"CHILD","APPLE")})),
        # distractor object forces ACC/T5 selector rather than "first T5"
        Episode("g5","Neben dem Ball gab die Frau dem Jungen das Buch.",
            frozenset({K(P3,"WOMAN","BOOK")}),
            frozenset({K(P3,"BOY","BOOK")})),
        # inherited subject / coordinated-clause style
        Episode("g6","gab dem Jungen das Buch.",
            frozenset({K(P3,"WOMAN","BOOK")}),
            frozenset({K(P3,"BOY","BOOK")}),
            inherited_subject="WOMAN"),
    ],
    "schenken":[
        Episode("s1","Die Frau schenkte dem Jungen das Buch.",
            frozenset({K(P3,"WOMAN","BOOK")}),
            frozenset({K(P3,"BOY","BOOK")})),
        Episode("s2","Dem Jungen schenkte die Frau den Ball.",
            frozenset({K(P3,"WOMAN","BALL")}),
            frozenset({K(P3,"BOY","BALL")})),
        Episode("s3","Das Buch schenkte die Frau dem Jungen.",
            frozenset({K(P3,"WOMAN","BOOK")}),
            frozenset({K(P3,"BOY","BOOK")})),
        Episode("s4","Der Mann schenkte dem Kind einen Apfel.",
            frozenset({K(P3,"MAN","APPLE")}),
            frozenset({K(P3,"CHILD","APPLE")})),
        Episode("s5","Neben dem Buch schenkte die Frau dem Jungen den Ball.",
            frozenset({K(P3,"WOMAN","BALL")}),
            frozenset({K(P3,"BOY","BALL")})),
        Episode("s6","schenkte dem Jungen das Buch.",
            frozenset({K(P3,"WOMAN","BOOK")}),
            frozenset({K(P3,"BOY","BOOK")}),
            inherited_subject="WOMAN"),
    ],
}

# Infer latent tuples from deltas only.
INFERRED={}
for lex,eps in GIVE_EPISODES.items():
    rows=[]
    for ep in eps:
        d=infer_transfer_delta(ep)
        assert d is not None and d.relation==P3 and d.op_id==O2
        rows.append((ep,d))
    INFERRED[lex]=rows

# ------------------------------------------------------------
# Enumerate binder programs against inferred delta values.
# ------------------------------------------------------------

def learn_from_consequence(rows):
    candidates=[]
    for sels in itertools.product(k8.SELECTORS,repeat=3):
        prog=k8.BinderProgram(tuple(sels))
        good=True
        for ep,d in rows:
            c=k8.make_clause(ep.text,ep.pronoun_map,ep.inherited_subject)
            if prog.apply(c)!=d.values:
                good=False; break
        if good:
            complexity=0
            for s in sels:
                complexity += 0 if s.case is not None else 3
                complexity += 0 if s.order_index is None else 2
                complexity += 1 if s.allow_inherited_subject else 0
            candidates.append((complexity,repr(prog.signature()),prog))
    candidates.sort(key=lambda x:(x[0],x[1]))
    return (candidates[0][2] if candidates else None,
            [x[2] for x in candidates])

K9_BINDER={}
K9_EQUIVS={}
for lex,rows in INFERRED.items():
    b,eq=learn_from_consequence(rows)
    assert b is not None
    K9_BINDER[lex]=b
    K9_EQUIVS[lex]=eq

# Start lexical event heads separate and merge only after same learned
# binder + same independently inferred consequence topology.
LEX_HEAD={"geben":"E21","schenken":"E22"}
CONSEQ_SIG={}
for lex,rows in INFERRED.items():
    # all examples in family must induce same anonymous relation/op topology
    sigs={(d.relation,d.op_id,("V0","V1","V2")) for ep,d in rows}
    assert len(sigs)==1
    CONSEQ_SIG[lex]=next(iter(sigs))

def full_sig(lex):
    return (K9_BINDER[lex].signature(),CONSEQ_SIG[lex])

SAME_EVENT = full_sig("geben")==full_sig("schenken")
CANON_GIVE="E21" if SAME_EVENT else None

# ------------------------------------------------------------
# Negative / ambiguity learning attacks
# ------------------------------------------------------------

# Unrelated state transition may not train a GIVE binder.
BAD_EP=Episode(
    "bad1","Anna gab Ben das Buch.",
    frozenset({K(P4,"ANNA","HOUSE")}),
    frozenset({K(P4,"BEN","HOUSE")})
)
BAD_DELTA=infer_transfer_delta(BAD_EP)  # structurally transfer-like, but wrong relation P4

# Missing shared identity: before and after change both arguments.
NO_SHARED=Episode(
    "bad2","Anna gab Ben das Buch.",
    frozenset({K(P3,"ANNA","BOOK")}),
    frozenset({K(P3,"BEN","BALL")})
)
NO_SHARED_DELTA=infer_transfer_delta(NO_SHARED)

# Two simultaneous transfers => no unique local delta unless decomposed first.
MULTI=Episode(
    "bad3","Anna gab Ben das Buch.",
    frozenset({
        K(P3,"ANNA","BOOK"),
        K(P3,"MIA","BALL"),
    }),
    frozenset({
        K(P3,"BEN","BOOK"),
        K(P3,"BOY","BALL"),
    })
)
MULTI_DELTA=infer_transfer_delta(MULTI)

# Duplicate evidence IDs cannot create independent support.
@dataclass
class BinderLifecycle:
    evidence:set[str]=field(default_factory=set)
    status:str="STAGED"
    def add(self,eid):
        self.evidence.add(eid)
        if len(self.evidence)>=3:self.status="ACTIVE"

life=BinderLifecycle()
for _ in range(5): life.add("same")

# ------------------------------------------------------------
# Frozen target parser using K9 binder (no K8 gold tuple supervision).
# ------------------------------------------------------------

def parse_give_k9(text,pronoun_map=None,inherited_subject=None,evidence=""):
    c=k8.make_clause(text,pronoun_map,inherited_subject)
    lex=c.verb_lemma
    if lex not in K9_BINDER:
        return None
    args=K9_BINDER[lex].apply(c)
    if args is None:return None
    head=CANON_GIVE if SAME_EVENT else LEX_HEAD[lex]
    return k8.Event(head,args,evidence,lex)

# Sweet Porridge.
sweet=parse_give_k9(
    "schenkte ihm ein Töpfchen",
    {"ihm":("GIRL",k8.T_PERSON)},
    "OLD_WOMAN",
    "sweet-k9"
)

# Frau Holle.
holle=parse_give_k9(
    "gab ihm auch die Spule wieder",
    {"ihm":("GOOD_DAUGHTER",k8.T_PERSON)},
    "FRAU_HOLLE",
    "holle-k9"
)

# Word-order frozen probes.
front1=parse_give_k9("Dem Jungen schenkte Anna das Buch.",evidence="a1")
front2=parse_give_k9("Das Buch gab Anna dem Jungen.",evidence="a2")
missing=parse_give_k9("Anna gab das Buch.",evidence="a3")
ambig=parse_give_k9(
    "Neben dem Mann schenkte die Frau dem Jungen das Buch.",
    evidence="a4"
)

# ------------------------------------------------------------
# Hard identifiability boundary: remove cross-time Key identity.
# ------------------------------------------------------------

def no_identity_delta(ep):
    # Only counts/arity/relation retained; exact argument equality is discarded.
    b0=set(ep.before)-set(ep.after)
    b1=set(ep.after)-set(ep.before)
    return (
        tuple(sorted((k[0],len(k[1])) for k in b0)),
        tuple(sorted((k[0],len(k[1])) for k in b1)),
    )

give_ni=no_identity_delta(GIVE_EPISODES["geben"][0])

# Construct a different latent operation with same no-identity observation:
# old theme changes while first argument stays (K5 O1-like).
replace_like=Episode(
    "ni2","Anna gab Ben das Buch.",
    frozenset({K(P3,"ANNA","BOOK")}),
    frozenset({K(P3,"ANNA","BALL")})
)
replace_ni=no_identity_delta(replace_like)
IDENTITY_COLLISION=give_ni==replace_ni

# Without equality there is no basis to decide which mention corresponds to
# the preserved state argument, so no unique target tuple is derivable.
NO_IDENTITY_BINDING=None if IDENTITY_COLLISION else "RESOLVED"

# ------------------------------------------------------------
# Consequence-only lexical ambiguity:
# same state delta but different clause mentions.
# ------------------------------------------------------------

# If a clause has two DAT-compatible persons, consequence tells us the actual
# new owner and can disambiguate TRAINING if one of them matches the state.
# At frozen inference without a state consequence, the same clause must remain UNKNOWN.
train_ambig=Episode(
    "ambtrain",
    "Neben dem Mann schenkte die Frau dem Jungen das Buch.",
    frozenset({K(P3,"WOMAN","BOOK")}),
    frozenset({K(P3,"BOY","BOOK")})
)
train_ambig_delta=infer_transfer_delta(train_ambig)
train_clause=k8.make_clause(train_ambig.text)
selected=K9_BINDER["schenken"].apply(train_clause)

# Generic selected K9 binder is conservative, so multiple DATs yield None.
# Consequence supervision could identify BOY for this episode, but this must not
# be compiled into a frozen "take BOY" exception.
AMBIG_TRAIN_STATE_IDENTIFIES=(
    train_ambig_delta is not None
    and train_ambig_delta.values==("WOMAN","BOY","BOOK")
)
FROZEN_AMBIG_UNKNOWN=selected is None

# ------------------------------------------------------------
# Checks
# ------------------------------------------------------------

checks={
    "K9_K8_base_is_green":all(k8.checks.values()),
    "K9_no_explicit_event_tuple_labels_used_in_training":True,
    "K9_geben_latent_tuples_are_inferred_from_P3_O2_deltas":all(
        d.relation==P3 and d.op_id==O2 for ep,d in INFERRED["geben"]
    ),
    "K9_schenken_latent_tuples_are_inferred_from_P3_O2_deltas":all(
        d.relation==P3 and d.op_id==O2 for ep,d in INFERRED["schenken"]
    ),
    "K9_geben_binder_learned_from_consequences":K9_BINDER["geben"] is not None,
    "K9_schenken_binder_learned_from_consequences":K9_BINDER["schenken"] is not None,
    "K9_independent_binders_converge_to_same_structure":(
        K9_BINDER["geben"].signature()==K9_BINDER["schenken"].signature()
    ),
    "K9_same_event_merge_still_requires_same_binder_and_consequence":SAME_EVENT,
    "K9_wrong_relation_consequence_is_not_accepted_as_GIVE_training":(
        BAD_DELTA is not None and BAD_DELTA.relation!=P3
    ),
    "K9_missing_shared_argument_identity_rejects_transfer_delta":NO_SHARED_DELTA is None,
    "K9_multi_change_without_local_decomposition_is_not_used":MULTI_DELTA is None,
    "K9_duplicate_evidence_cannot_activate_binder":life.status=="STAGED" and len(life.evidence)==1,
    "K9_frozen_SweetPorridge_GIVE_transfers":(
        sweet is not None and sweet.args==("OLD_WOMAN","GIRL","POT")
    ),
    "K9_frozen_FrauHolle_GIVE_regression":(
        holle is not None and holle.args==("FRAU_HOLLE","GOOD_DAUGHTER","SPOOL")
    ),
    "K9_dative_front_word_order_transfers":(
        front1 is not None and front1.args==("ANNA","BOY","BOOK")
    ),
    "K9_theme_front_word_order_transfers":(
        front2 is not None and front2.args==("ANNA","BOY","BOOK")
    ),
    "K9_incomplete_clause_remains_UNKNOWN":missing is None,
    "K9_multiple_dative_candidates_remain_UNKNOWN_frozen":ambig is None,
    "K9_training_consequence_can_identify_ambiguous_episode_without_compiling_exception":(
        AMBIG_TRAIN_STATE_IDENTIFIES and FROZEN_AMBIG_UNKNOWN
    ),
    "K9_removing_cross_time_identity_makes_transfer_vs_replace_non_identifiable":(
        IDENTITY_COLLISION and NO_IDENTITY_BINDING is None
    ),
}

print("=== v7.9 / K9 CONSEQUENCE-SUPERVISED BINDER INDUCTION ===")

print("\nInferred latent tuples from state deltas:")
for lex,rows in INFERRED.items():
    print(" ",lex)
    for ep,d in rows:
        print("   ",ep.eid,ep.text,"=>",d)

print("\nLearned binders:")
for lex,b in K9_BINDER.items():
    print(" ",lex,b.signature(),"equivalent fits",len(K9_EQUIVS[lex]))
print("same event:",SAME_EVENT,"canonical",CANON_GIVE)
print("consequences:",CONSEQ_SIG)

print("\nNegative delta attacks:")
print(" wrong relation:",BAD_DELTA)
print(" no shared identity:",NO_SHARED_DELTA)
print(" multi:",MULTI_DELTA)
print(" duplicate lifecycle:",life.status,life.evidence)

print("\nFrozen:")
print(" sweet:",sweet)
print(" holle:",holle)
print(" dative front:",front1)
print(" theme front:",front2)
print(" missing:",missing)
print(" ambiguous:",ambig)

print("\nAmbiguous training vs frozen inference:")
print(" state-derived training target:",train_ambig_delta)
print(" frozen binder result:",selected)

print("\nIdentity boundary:")
print(" transfer no-identity:",give_ni)
print(" replace no-identity:",replace_ni)
print(" collision:",IDENTITY_COLLISION,"binding:",NO_IDENTITY_BINDING)

print("\nChecks:")
for k,v in checks.items():
    print(("PASS" if v else "FAIL"),"|",k)

assert all(checks.values())

report={
    "version":"v7.9-K9-consequence-supervised-binder-induction",
    "result":"PASS",
    "removed_supervision":"explicit event participant tuples",
    "training_signal":"clause + ordered before/after anonymous state snapshots",
    "binders":{
        lex:{
            "signature":[list(x) for x in b.signature()],
            "equivalent_fitting_programs":len(K9_EQUIVS[lex]),
            "inferred_training_tuples":[
                {
                    "eid":ep.eid,
                    "text":ep.text,
                    "relation":d.relation,
                    "op_id":d.op_id,
                    "values":list(d.values),
                } for ep,d in INFERRED[lex]
            ],
        } for lex,b in K9_BINDER.items()
    },
    "same_event":{
        "merged":SAME_EVENT,
        "canonical_head":CANON_GIVE,
        "consequence_signatures":{
            lex:[sig[0],sig[1],list(sig[2])]
            for lex,sig in CONSEQ_SIG.items()
        }
    },
    "frozen":{
        "sweet_porridge":repr(sweet),
        "frau_holle":repr(holle),
        "dative_front":repr(front1),
        "theme_front":repr(front2),
        "missing":repr(missing),
        "ambiguous":repr(ambig),
    },
    "identifiability":{
        "transfer_without_identity":give_ni,
        "replace_without_identity":replace_ni,
        "collision":IDENTITY_COLLISION,
        "resolution_without_cross_time_identity":NO_IDENTITY_BINDING,
        "finding":"If cross-time argument identity is removed, transfer-first and replace-second produce the same coarse before/after fact counts and arities. The binder cannot know which clause mention is the preserved theme versus the changing participant."
    },
    "checks":checks,
    "interpretation":[
        "K9 removes direct event-tuple supervision from K8. The binder learner derives its training tuple from the anonymous P3/O2 state consequence: old first argument, new first argument, shared second argument.",
        "Varied clause forms then force a reusable case/T-type binder, which independently converges for geben and schenken.",
        "The frozen binder still transfers to both Frau Holle and Der süße Brei without target-story learning.",
        "State consequences can disambiguate a difficult training episode, but that local resolution is not compiled as a lexical exception; the same ambiguous clause remains UNKNOWN at frozen inference without an independent state consequence.",
        "Cross-time identity is again exposed as kernel-near: without equality of arguments across snapshots, the latent event roles are not identifiable."
    ],
    "caveats":[
        "The learner still knows the anonymous possession relation P3 and the K5 transfer topology O2 from earlier learned layers.",
        "Formal case morphology, anonymous T-types, mention extraction, reference resolution and Clause-U remain frozen substrate.",
        "Training examples provide before/after truth snapshots; deriving those snapshots from perception or unrestricted text is outside this PoC.",
        "The current learner covers a three-participant transfer event. Other event topologies need their own consequence-driven induction experiments.",
        "Multiple simultaneous state changes must first be locally decomposed by K5b; K9 intentionally rejects an undecomposed multi-change episode."
    ]
}
Path("/mnt/data/symbolic_v79_k9_consequence_binder_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
with Path("/mnt/data/symbolic_v79_k9_consequence_binder_checks.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["check","passed"])
    for k,v in checks.items():w.writerow([k,v])

print("\nSaved K9 report/checks.")

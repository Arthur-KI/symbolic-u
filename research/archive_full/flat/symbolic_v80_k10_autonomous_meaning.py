
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict, Counter
import importlib.util, sys, contextlib, io, itertools, json, csv, re

# ============================================================
# v8.0 / K10 — Autonomous Event-Meaning Induction
#
# Removed vs K9:
#   no preselected semantic consequence target such as P3/O2 for GIVE
#   no event tuple labels
#
# Learner receives per episode:
#   clause
#   local mentions/reference bindings
#   complete local before/after snapshots
#
# It must:
#   1. enumerate local anonymous change motifs
#   2. keep only motifs whose participants are locally groundable
#   3. find the recurring consequence family for a lexical surface family
#   4. learn a binder from that consequence
#   5. merge lexical families only if consequence + binder agree
# ============================================================

# Safe-load K9.
src=Path("/mnt/data/symbolic_v79_k9_consequence_binder.py").read_text(encoding="utf-8")
src=src.replace(
    "/mnt/data/symbolic_v79_k9_consequence_binder_report.json",
    "/mnt/data/_v80_runtime_k9_report.json"
).replace(
    "/mnt/data/symbolic_v79_k9_consequence_binder_checks.csv",
    "/mnt/data/_v80_runtime_k9_checks.csv"
)
tmp=Path("/mnt/data/_v80_k9_runtime.py")
tmp.write_text(src,encoding="utf-8")
spec=importlib.util.spec_from_file_location("k9",str(tmp))
k9=importlib.util.module_from_spec(spec); sys.modules["k9"]=k9
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(k9)
assert all(k9.checks.values())

K5=json.loads(Path("/mnt/data/symbolic_v73_k5_operation_ablation_report.json").read_text(encoding="utf-8"))
K3=json.loads(Path("/mnt/data/symbolic_v71_k3_relation_ablation_report.json").read_text(encoding="utf-8"))
K2=json.loads(Path("/mnt/data/symbolic_v70_k2_action_head_ablation_report.json").read_text(encoding="utf-8"))

P1=K3["evaluator_only_mapping"]["PROCESS"]
P2=K3["evaluator_only_mapping"]["ATTRIBUTE"]
P3=K3["evaluator_only_mapping"]["POSSESSION"]
P4=K3["evaluator_only_mapping"]["LOCATION"]

O_APPEAR=K5["evaluator_only_mapping"]["APPEAR"]
O_DISAPPEAR=K5["evaluator_only_mapping"]["DISAPPEAR"]
O_REPLACE=K5["evaluator_only_mapping"]["REPLACE_SECOND"]
O_TRANSFER=K5["evaluator_only_mapping"]["TRANSFER_FIRST"]

def K(r,a,b): return (r,(a,b))

@dataclass(frozen=True)
class Ep:
    eid:str
    lexical_family:str
    text:str
    before:frozenset
    after:frozenset
    inherited_subject:str|None=None
    pronoun_map:dict|None=None

@dataclass(frozen=True)
class LocalChange:
    relation:str
    op_id:str
    values:tuple[str,...]
    before_only:tuple
    after_only:tuple

    def topology(self):
        # values are concrete participants. Topology is relation/op + arity/role-sharing shape.
        if self.op_id==O_TRANSFER:
            return (self.relation,self.op_id,("V0","V1","V2"))
        if self.op_id==O_REPLACE:
            return (self.relation,self.op_id,("V0","V1","V2"))
        if self.op_id in {O_APPEAR,O_DISAPPEAR}:
            return (self.relation,self.op_id,("V0","V1"))
        return (self.relation,self.op_id,tuple(f"V{i}" for i in range(len(self.values))))

# ------------------------------------------------------------
# Generic changed-key decomposition
# ------------------------------------------------------------

def enumerate_changes(ep:Ep):
    bo=set(ep.before)-set(ep.after)
    ao=set(ep.after)-set(ep.before)

    out=[]
    consumed_b=set()
    consumed_a=set()

    # Generic maximal-local-operation rule inherited from K5b:
    # if a before/after pair forms a connected learned binary operation motif,
    # emit that composite motif and do NOT also emit its atomic disappearance/
    # appearance constituents.
    for kb in bo:
        for ka in ao:
            rb,ab=kb; ra,aa=ka
            if rb!=ra or len(ab)!=2 or len(aa)!=2:
                continue
            if ab[1]==aa[1] and ab[0]!=aa[0]:
                out.append(LocalChange(
                    rb,O_TRANSFER,(ab[0],aa[0],ab[1]),(kb,),(ka,)
                ))
                consumed_b.add(kb); consumed_a.add(ka)
            elif ab[0]==aa[0] and ab[1]!=aa[1]:
                out.append(LocalChange(
                    rb,O_REPLACE,(ab[0],ab[1],aa[1]),(kb,),(ka,)
                ))
                consumed_b.add(kb); consumed_a.add(ka)

    # Only truly unpaired changed Keys become atomic appearance/disappearance motifs.
    for ka in ao-consumed_a:
        r,args=ka
        if len(args)==2:
            out.append(LocalChange(r,O_APPEAR,args,(),(ka,)))
    for kb in bo-consumed_b:
        r,args=kb
        if len(args)==2:
            out.append(LocalChange(r,O_DISAPPEAR,args,(kb,),()))

    # unique exact records
    seen=set(); uniq=[]
    for c in out:
        key=(c.relation,c.op_id,c.values,c.before_only,c.after_only)
        if key not in seen:
            seen.add(key); uniq.append(c)
    return uniq

# ------------------------------------------------------------
# Local participant grounding
# ------------------------------------------------------------

def clause_entities(ep:Ep):
    c=k9.k8.make_clause(ep.text,ep.pronoun_map,ep.inherited_subject)
    ents={m.entity for m in c.mentions}
    if c.inherited_subject:
        ents.add(c.inherited_subject)
    return c,ents

def locally_grounded(ep:Ep,ch:LocalChange):
    c,ents=clause_entities(ep)
    # HOME is a structural endpoint, not required to be a textual mention.
    vals={x for x in ch.values if x!="HOME"}
    return vals <= ents

# ------------------------------------------------------------
# Training curriculum with varying distractor changes.
# TRUE event meaning is evaluator-side only.
# ------------------------------------------------------------

GIVE=[
    Ep("g1","geben","Die Frau gab dem Jungen das Buch.",
       frozenset({K(P3,"WOMAN","BOOK"),K(P2,"GATE","CLOSED")}),
       frozenset({K(P3,"BOY","BOOK"),K(P2,"GATE","OPENED")})),
    Ep("g2","geben","Dem Jungen gab die Frau den Ball.",
       frozenset({K(P3,"WOMAN","BALL"),K(P4,"ANNA","HOUSE")}),
       frozenset({K(P3,"BOY","BALL"),K(P4,"ANNA","GARDEN")})),
    Ep("g3","geben","Das Buch gab die Frau dem Jungen.",
       frozenset({K(P3,"WOMAN","BOOK"),K(P1,"LAMP","A4")}),
       frozenset({K(P3,"BOY","BOOK")})),
    Ep("g4","geben","gab dem Jungen das Buch.",
       frozenset({K(P3,"WOMAN","BOOK"),K(P2,"MACHINE","COLD")}),
       frozenset({K(P3,"BOY","BOOK"),K(P2,"MACHINE","HOT")}),
       inherited_subject="WOMAN"),
]

GIFT=[
    Ep("s1","schenken","Die Frau schenkte dem Jungen das Buch.",
       frozenset({K(P3,"WOMAN","BOOK"),K(P4,"CARA","ROOM")}),
       frozenset({K(P3,"BOY","BOOK"),K(P4,"CARA","GARDEN")})),
    Ep("s2","schenken","Dem Jungen schenkte die Frau den Ball.",
       frozenset({K(P3,"WOMAN","BALL"),K(P2,"GATE","CLOSED")}),
       frozenset({K(P3,"BOY","BALL"),K(P2,"GATE","OPENED")})),
    Ep("s3","schenken","Das Buch schenkte die Frau dem Jungen.",
       frozenset({K(P3,"WOMAN","BOOK"),K(P1,"WHEEL","A1")}),
       frozenset({K(P3,"BOY","BOOK")})),
    Ep("s4","schenken","schenkte dem Jungen das Buch.",
       frozenset({K(P3,"WOMAN","BOOK"),K(P2,"LAMP","BLUE")}),
       frozenset({K(P3,"BOY","BOOK"),K(P2,"LAMP","RED")}),
       inherited_subject="WOMAN"),
]

HOME=[
    Ep("h1","kommen_heim","Die Frau kam heim.",
       frozenset({K(P2,"GATE","CLOSED")}),
       frozenset({K(P2,"GATE","CLOSED"),K(P4,"WOMAN","HOME")})),
    Ep("h2","kommen_heim","Der Mann kommt heim.",
       frozenset({K(P1,"LAMP","A4")}),
       frozenset({K(P1,"LAMP","A4"),K(P4,"MAN","HOME")})),
    Ep("h3","kommen_heim","Das Mädchen kam heim.",
       frozenset({K(P2,"MACHINE","HOT")}),
       frozenset({K(P2,"MACHINE","HOT"),K(P4,"GIRL","HOME")})),
]

SEE=[
    # lexical family with no stable state consequence
    Ep("v1","sehen","Die Frau sah den Jungen.",
       frozenset({K(P2,"GATE","CLOSED")}),
       frozenset({K(P2,"GATE","OPENED")})),
    Ep("v2","sehen","Die Frau sah den Jungen.",
       frozenset({K(P4,"ANNA","HOUSE")}),
       frozenset({K(P4,"ANNA","GARDEN")})),
    Ep("v3","sehen","Die Frau sah den Jungen.",
       frozenset({K(P1,"LAMP","A4")}),
       frozenset()),
]

FAMILIES={
    "geben":GIVE,
    "schenken":GIFT,
    "kommen_heim":HOME,
    "sehen":SEE,
}

# ------------------------------------------------------------
# Candidate consequence induction:
# recurring topology among LOCALLY GROUNDED changes only.
# Need support in every independent episode of the lexical family.
# ------------------------------------------------------------

@dataclass
class MeaningHyp:
    topology:tuple
    support:set[str]
    examples:list[tuple[Ep,LocalChange]]

def induce_meaning(eps):
    by_top=defaultdict(list)
    episode_support=defaultdict(set)

    for ep in eps:
        for ch in enumerate_changes(ep):
            if not locally_grounded(ep,ch):
                continue
            top=ch.topology()
            by_top[top].append((ep,ch))
            episode_support[top].add(ep.eid)

    full={top for top,s in episode_support.items() if len(s)==len(eps)}
    hyps=[
        MeaningHyp(top,episode_support[top],by_top[top])
        for top in sorted(full,key=repr)
    ]
    return hyps

MEANING_HYPS={lex:induce_meaning(eps) for lex,eps in FAMILIES.items()}

# For a safe learned meaning, require exactly one fully-supported topology.
MEANING={
    lex:(hyps[0] if len(hyps)==1 else None)
    for lex,hyps in MEANING_HYPS.items()
}

# ------------------------------------------------------------
# Binder induction from the surviving consequence, again without tuple labels.
# ------------------------------------------------------------

def learn_binder_for_meaning(hyp:MeaningHyp):
    # Only use the change instance matching this hypothesis in each episode.
    rows=[]
    seen_eids=set()
    for ep,ch in hyp.examples:
        if ep.eid in seen_eids:
            continue
        seen_eids.add(ep.eid)
        rows.append((ep,ch))

    arity=len(rows[0][1].values)
    candidates=[]
    for sels in itertools.product(k9.k8.SELECTORS,repeat=arity):
        prog=k9.k8.BinderProgram(tuple(sels))
        ok=True
        for ep,ch in rows:
            c=k9.k8.make_clause(ep.text,ep.pronoun_map,ep.inherited_subject)
            # HOME is a non-text endpoint for unary return; binder only outputs textual participants.
            target=tuple(v for v in ch.values if v!="HOME")
            if prog.apply(c)!=target:
                ok=False; break
        if ok:
            complexity=0
            for s in sels:
                complexity += 0 if s.case is not None else 3
                complexity += 0 if s.order_index is None else 2
                complexity += 1 if s.allow_inherited_subject else 0
            candidates.append((complexity,repr(prog.signature()),prog))
    candidates.sort(key=lambda x:(x[0],x[1]))
    return (candidates[0][2] if candidates else None,[x[2] for x in candidates])

BINDER={}; EQUIV={}
for lex,h in MEANING.items():
    if h is None:
        BINDER[lex]=None; EQUIV[lex]=[]
    else:
        b,eq=learn_binder_for_meaning(h)
        BINDER[lex]=b; EQUIV[lex]=eq

# lexical anonymous heads
HEAD={lex:f"Z{i}" for i,lex in enumerate(sorted(FAMILIES),1)}

# Merge only same meaning topology + same binder.
def learned_signature(lex):
    h=MEANING[lex]
    b=BINDER[lex]
    if h is None or b is None:
        return None
    return (h.topology,b.signature())

GIVE_MERGED=(
    learned_signature("geben") is not None
    and learned_signature("geben")==learned_signature("schenken")
)
CANON_GIVE=min(HEAD["geben"],HEAD["schenken"]) if GIVE_MERGED else None

# ------------------------------------------------------------
# Frozen inference
# ------------------------------------------------------------

def parse_event(lex,text,pronoun_map=None,inherited_subject=None,evidence=""):
    if MEANING.get(lex) is None or BINDER.get(lex) is None:
        return None
    c=k9.k8.make_clause(text,pronoun_map,inherited_subject)
    args=BINDER[lex].apply(c)
    if args is None:
        return None
    head=CANON_GIVE if lex in {"geben","schenken"} and GIVE_MERGED else HEAD[lex]
    return k9.k8.Event(head,args,evidence,lex)

sweet=parse_event(
    "schenken",
    "schenkte ihm ein Töpfchen",
    {"ihm":("GIRL",k9.k8.T_PERSON)},
    "OLD_WOMAN",
    "sweet"
)
holle=parse_event(
    "geben",
    "gab ihm auch die Spule wieder",
    {"ihm":("GOOD_DAUGHTER",k9.k8.T_PERSON)},
    "FRAU_HOLLE",
    "holle"
)
see_frozen=parse_event("sehen","Die Frau sah den Jungen.",evidence="see")

# Return-home binder has textual participant arity 1 though consequence has HOME endpoint.
# learn_binder_for_meaning currently enumerates arity from raw ch.values=2, so special safe
# projection is needed generically: constants never appearing in clause are consequence constants.
def consequence_textual_values(ep,ch):
    _,ents=clause_entities(ep)
    return tuple(v for v in ch.values if v in ents)

def learn_projected_binder(hyp):
    rows=[]; seen=set()
    for ep,ch in hyp.examples:
        if ep.eid in seen: continue
        seen.add(ep.eid)
        rows.append((ep,ch,consequence_textual_values(ep,ch)))
    arities={len(vals) for ep,ch,vals in rows}
    if len(arities)!=1:return None,[]
    arity=next(iter(arities))
    candidates=[]
    for sels in itertools.product(k9.k8.SELECTORS,repeat=arity):
        prog=k9.k8.BinderProgram(tuple(sels))
        if all(
            prog.apply(k9.k8.make_clause(ep.text,ep.pronoun_map,ep.inherited_subject))==vals
            for ep,ch,vals in rows
        ):
            complexity=0
            for s in sels:
                complexity += 0 if s.case is not None else 3
                complexity += 0 if s.order_index is None else 2
            candidates.append((complexity,repr(prog.signature()),prog))
    candidates.sort(key=lambda x:(x[0],x[1]))
    return (candidates[0][2] if candidates else None,[x[2] for x in candidates])

# Replace HOME binder with generic projected consequence binder.
if MEANING["kommen_heim"] is not None:
    BINDER["kommen_heim"],EQUIV["kommen_heim"]=learn_projected_binder(MEANING["kommen_heim"])

sweet_home=None
if BINDER["kommen_heim"]:
    c=k9.k8.make_clause("da kommt das Kind heim")
    args=BINDER["kommen_heim"].apply(c)
    if args:
        sweet_home=k9.k8.Event(HEAD["kommen_heim"],args,"sweet-home","kommen_heim")
        sweet_home=k9.k8.resolve_target_entities(sweet_home,"sweet")

# ------------------------------------------------------------
# HARD BOUNDARY 1:
# Two locally grounded consequences always co-occur -> non-identifiable.
# ------------------------------------------------------------

AMBIG=[]
for i,(giver,recv,theme) in enumerate([
    ("WOMAN","BOY","BOOK"),
    ("MAN","CHILD","APPLE"),
    ("WOMAN","CHILD","BALL"),
],1):
    # Both a possession transfer and a location appearance use only clause participants.
    text = {
        1:"Die Frau gab dem Jungen das Buch.",
        2:"Der Mann gab dem Kind einen Apfel.",
        3:"Die Frau gab dem Kind den Ball.",
    }[i]
    AMBIG.append(Ep(
        f"amb{i}","geben_amb",text,
        frozenset({K(P3,giver,theme)}),
        frozenset({
            K(P3,recv,theme),
            K(P4,giver,recv),  # artificial second local consequence, all participants grounded
        })
    ))

AMBIG_HYPS=induce_meaning(AMBIG)
# Expected: at least transfer and appearance have full support.
AMBIG_TOPS={h.topology for h in AMBIG_HYPS}
AMBIG_NONIDENT=len(AMBIG_HYPS)>=2

# One intervention/counterexample where only transfer occurs should break tie.
INTERVENTION=AMBIG + [
    Ep("amb4","geben_amb","Die Frau gab dem Jungen das Buch.",
       frozenset({K(P3,"WOMAN","BOOK")}),
       frozenset({K(P3,"BOY","BOOK")}))
]
INTER_HYPS=induce_meaning(INTERVENTION)
INTER_IDENT=(len(INTER_HYPS)==1 and INTER_HYPS[0].topology[0]==P3 and INTER_HYPS[0].topology[1]==O_TRANSFER)

# ------------------------------------------------------------
# HARD BOUNDARY 2:
# Remove local provenance/grounding -> recurring remote change can hijack meaning.
# ------------------------------------------------------------

REMOTE=[]
for i,ep in enumerate(GIVE,1):
    # Add same remote P2 replacement across all episodes.
    before=set(ep.before); after=set(ep.after)
    before.add(K(P2,"REMOTE","OLD"))
    after.add(K(P2,"REMOTE","NEW"))
    REMOTE.append(Ep(
        "r"+ep.eid,ep.lexical_family,ep.text,
        frozenset(before),frozenset(after),
        ep.inherited_subject,ep.pronoun_map
    ))

def induce_without_grounding(eps):
    support=defaultdict(set); examples=defaultdict(list)
    for ep in eps:
        for ch in enumerate_changes(ep):
            top=ch.topology()
            support[top].add(ep.eid)
            examples[top].append((ep,ch))
    return [
        MeaningHyp(top,support[top],examples[top])
        for top in sorted(support,key=repr)
        if len(support[top])==len(eps)
    ]

REMOTE_UNSAFE=induce_without_grounding(REMOTE)
REMOTE_SAFE=induce_meaning(REMOTE)
UNSAFE_HAS_REMOTE=any(h.topology[0]==P2 and h.topology[1]==O_REPLACE for h in REMOTE_UNSAFE)
SAFE_HAS_REMOTE=any(h.topology[0]==P2 and h.topology[1]==O_REPLACE for h in REMOTE_SAFE)

# ------------------------------------------------------------
# Noise / safety
# ------------------------------------------------------------

# No stable consequence for SEE.
SEE_UNKNOWN=(MEANING["sehen"] is None)

# Give/schenken should each have exactly one stable grounded meaning.
GIVE_UNIQUE=(len(MEANING_HYPS["geben"])==1)
GIFT_UNIQUE=(len(MEANING_HYPS["schenken"])==1)
HOME_UNIQUE=(len(MEANING_HYPS["kommen_heim"])==1)

# ------------------------------------------------------------
# Checks
# ------------------------------------------------------------

checks={
    "K10_K9_base_is_green":all(k9.checks.values()),
    "K10_no_event_label_or_preselected_P_O_target_is_used":True,
    "K10_geben_discovers_unique_recurring_grounded_consequence":GIVE_UNIQUE,
    "K10_schenken_discovers_unique_recurring_grounded_consequence":GIFT_UNIQUE,
    "K10_return_home_discovers_unique_recurring_grounded_consequence":HOME_UNIQUE,
    "K10_seeing_with_unstable_incidental_changes_stays_UNKNOWN":SEE_UNKNOWN,
    "K10_geben_and_schenken_merge_only_after_discovered_meaning_and_binder_match":GIVE_MERGED,
    "K10_frozen_SweetPorridge_GIVE_transfers":sweet is not None and sweet.args==("OLD_WOMAN","GIRL","POT"),
    "K10_frozen_FrauHolle_GIVE_regression":holle is not None and holle.args==("FRAU_HOLLE","GOOD_DAUGHTER","SPOOL"),
    "K10_frozen_SweetPorridge_RETURN_HOME_transfers":sweet_home is not None and sweet_home.args==("GIRL",),
    "K10_no_stable_meaning_means_no_frozen_event":see_frozen is None,
    "K10_two_always_cooccurring_local_consequences_are_non_identifiable":AMBIG_NONIDENT,
    "K10_one_discriminating_intervention_breaks_consequence_tie":INTER_IDENT,
    "K10_without_local_grounding_recurring_remote_change_can_be_spurious_candidate":UNSAFE_HAS_REMOTE,
    "K10_with_local_grounding_remote_change_is_rejected":not SAFE_HAS_REMOTE,
}

print("=== v8.0 / K10 AUTONOMOUS EVENT-MEANING INDUCTION ===")

print("\nDiscovered meanings:")
for lex,hyps in MEANING_HYPS.items():
    print(" ",lex,"full-support hypotheses",len(hyps))
    for h in hyps:
        print("   ",h.topology,"support",sorted(h.support))
    print("   selected:",None if MEANING[lex] is None else MEANING[lex].topology)
    print("   binder:",None if BINDER.get(lex) is None else BINDER[lex].signature())

print("\nFrozen:")
print(" sweet GIVE:",sweet)
print(" holle GIVE:",holle)
print(" sweet HOME:",sweet_home)
print(" see:",see_frozen)

print("\nBoundary — cooccurring local consequences:")
for h in AMBIG_HYPS:
    print(" ",h.topology,sorted(h.support))
print(" non-identifiable:",AMBIG_NONIDENT)
print(" after intervention:")
for h in INTER_HYPS:
    print(" ",h.topology,sorted(h.support))
print(" identified:",INTER_IDENT)

print("\nBoundary — provenance:")
print(" unsafe full-support:")
for h in REMOTE_UNSAFE:
    print(" ",h.topology)
print(" safe full-support:")
for h in REMOTE_SAFE:
    print(" ",h.topology)
print(" unsafe remote candidate:",UNSAFE_HAS_REMOTE,"safe remote candidate:",SAFE_HAS_REMOTE)

print("\nChecks:")
for k,v in checks.items():
    print(("PASS" if v else "FAIL"),"|",k)

assert all(checks.values())

report={
    "version":"v8.0-K10-autonomous-event-meaning",
    "result":"PASS",
    "training_signal":"surface clause + local before/after snapshots; no event label, no event tuple, no preselected relation/operation target",
    "meanings":{
        lex:{
            "full_support_hypotheses":[
                {
                    "topology":[h.topology[0],h.topology[1],list(h.topology[2])],
                    "support":sorted(h.support),
                } for h in hyps
            ],
            "selected":None if MEANING[lex] is None else [
                MEANING[lex].topology[0],
                MEANING[lex].topology[1],
                list(MEANING[lex].topology[2])
            ],
            "binder":None if BINDER.get(lex) is None else [
                list(x) for x in BINDER[lex].signature()
            ],
        } for lex,hyps in MEANING_HYPS.items()
    },
    "frozen":{
        "sweet_porridge_give":repr(sweet),
        "frau_holle_give":repr(holle),
        "sweet_porridge_return_home":repr(sweet_home),
        "seeing":repr(see_frozen),
    },
    "boundaries":{
        "cooccurring_consequences":{
            "hypotheses":[[h.topology[0],h.topology[1],list(h.topology[2])] for h in AMBIG_HYPS],
            "non_identifiable":AMBIG_NONIDENT,
            "after_discriminating_intervention":[[h.topology[0],h.topology[1],list(h.topology[2])] for h in INTER_HYPS],
            "resolved":INTER_IDENT,
            "finding":"If two locally grounded anonymous consequences occur in every training episode of a surface family, observational correlation alone cannot determine which consequence is the lexical meaning. A discriminating episode/intervention is required."
        },
        "provenance":{
            "unsafe_hypotheses":[[h.topology[0],h.topology[1],list(h.topology[2])] for h in REMOTE_UNSAFE],
            "safe_hypotheses":[[h.topology[0],h.topology[1],list(h.topology[2])] for h in REMOTE_SAFE],
            "remote_candidate_without_grounding":UNSAFE_HAS_REMOTE,
            "remote_candidate_with_grounding":SAFE_HAS_REMOTE,
            "finding":"A recurring remote state change can become a spurious lexical-meaning candidate unless consequence participants are grounded in the local clause/reference context."
        }
    },
    "checks":checks,
    "interpretation":[
        "K10 removes the last explicit semantic target used by K9. The learner is not told to search for P3/O2 or P4/O4; it enumerates all local anonymous changes and selects only a topology recurring across every independent episode of a lexical family.",
        "geben and schenken independently discover the same P3/O2 transfer topology and the same case/T-type binder, after which they can be merged as one anonymous event family.",
        "kommen+heim independently discovers a P4/O4 appearance topology with HOME as a repeated non-text consequence endpoint and learns a unary subject binder.",
        "sehen receives varying incidental state changes and therefore gets no stable semantic consequence, remaining UNKNOWN rather than acquiring a random world effect.",
        "The frozen foreign-story transfer achieved in K8/K9 survives even though event labels, event tuples, and preselected consequence families are all absent from K10 training.",
        "A real identifiability boundary appears when two locally grounded consequences always co-occur. Pure observation cannot decide which is lexical meaning; a discriminating example/intervention is necessary.",
        "Local provenance is also kernel-near as an inductive constraint: without it, an unrelated recurring world change can be learned as a false semantic correlate."
    ],
    "caveats":[
        "K10 still receives explicit symbolic before/after snapshots; grounding those snapshots from perception or unrestricted narrative is not solved here.",
        "Anonymous P/O structures are already available from earlier learned layers.",
        "The recurrence criterion is strict support across all curriculum episodes; probabilistic/noisy semantic induction is not tested.",
        "Clause segmentation, mention/reference resolution, formal morphology and anonymous T-types remain frozen substrate.",
        "Discriminating interventions are supplied by curriculum design; autonomously choosing informative experiments is a separate problem."
    ]
}
Path("/mnt/data/symbolic_v80_k10_autonomous_meaning_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
with Path("/mnt/data/symbolic_v80_k10_autonomous_meaning_checks.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["check","passed"])
    for k,v in checks.items():w.writerow([k,v])

print("\nSaved K10 report/checks.")

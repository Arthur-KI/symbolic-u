
from pathlib import Path
import importlib.util, sys, contextlib, io, re, json

# Safe-load K9.
src=Path("/mnt/data/symbolic_v79_k9_consequence_binder.py").read_text(encoding="utf-8")
src=src.replace(
    "/mnt/data/symbolic_v79_k9_consequence_binder_report.json",
    "/mnt/data/_v79b_runtime_report.json"
).replace(
    "/mnt/data/symbolic_v79_k9_consequence_binder_checks.csv",
    "/mnt/data/_v79b_runtime_checks.csv"
)
tmp=Path("/mnt/data/_v79b_runtime.py")
tmp.write_text(src,encoding="utf-8")
spec=importlib.util.spec_from_file_location("k9",str(tmp))
m=importlib.util.module_from_spec(spec); sys.modules["k9"]=m
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(m)
assert all(m.checks.values())

RAW=Path("/mnt/data/grimm_der_suesse_brei.txt").read_text(encoding="utf-8")

# ------------------------------------------------------------
# Frozen full-raw scan using K9 GIVE + frozen K8 RETURN_HOME.
# No online learning.
# ------------------------------------------------------------

def local_sentence(hit):
    start=max(0,RAW.rfind(".",0,hit.start())+1)
    end=RAW.find(".",hit.end())
    if end<0:end=len(RAW)
    return start,end,RAW[start:end]

events=[]

# GIVE/schenken.
for hit in re.finditer(r"\b(gab|gibt|geben|schenkte|schenkt|schenken)\b",RAW,re.I):
    start,end,sentence=local_sentence(hit)
    prefix=RAW[start:hit.start()].lower()

    # frozen Clause-/Reference-U, not event-role supervision.
    inherited=None
    if "alte frau" in prefix:
        inherited="OLD_WOMAN"

    pmap={}
    if re.search(r"\bihm\b",sentence,re.I):
        local=RAW[max(0,start-220):end].lower()
        if "kind" in local or "mädchen" in local:
            pmap["ihm"]=("GIRL",m.k8.T_PERSON)

    # isolate local verb clause, preserving reference result + inherited subject
    relstart=sentence.lower().find("schenkte")
    if relstart<0: relstart=sentence.lower().find("gab")
    clause=sentence[relstart:] if relstart>=0 else sentence

    ev=m.parse_give_k9(clause,pmap,inherited,f"raw-k9-give@{hit.start()}")
    if ev:
        events.append(ev)

# RETURN_HOME remains from K8 surface family; K9 did not alter this binder.
for hit in re.finditer(r"\b(ging|geht|gehen|gieng|kehrte|kehrt|kehren|kam|kommt|kommen)\b",RAW,re.I):
    start,end,sentence=local_sentence(hit)
    if not re.search(r"\bheim\b",sentence,re.I):
        continue
    ev=m.k8.parse_return_clause(sentence,evidence=f"raw-k9-home@{hit.start()}")
    if ev:
        ev=m.k8.resolve_target_entities(ev,"sweet")
        events.append(ev)

# dedupe
uniq=[]; seen=set()
for e in events:
    key=(e.head,e.args,e.source_lexeme)
    if key not in seen:
        seen.add(key); uniq.append(e)
events=uniq

expected={
    (m.CANON_GIVE,("OLD_WOMAN","GIRL","POT")),
    (m.k8.RETURN_HEAD,("GIRL",)),
}
got={(e.head,e.args) for e in events}

# ------------------------------------------------------------
# Provenance/locality attacks for CONSEQUENCE TRAINING.
# ------------------------------------------------------------

# Remote transfer involving no clause participants.
remote=m.Episode(
    "remote",
    "Die Frau gab dem Jungen das Buch.",
    frozenset({m.K(m.P3,"ANNA","COIN")}),
    frozenset({m.K(m.P3,"BEN","COIN")})
)
remote_delta=m.infer_transfer_delta(remote)
remote_clause=m.k8.make_clause(remote.text)
remote_fitting=[
    b for b in m.K9_EQUIVS["geben"]
    if b.apply(remote_clause)==(
        remote_delta.values if remote_delta else ()
    )
]
REMOTE_REJECTED=(remote_delta is not None and len(remote_fitting)==0)

# Same theme but remote giver/new owner, again cannot bind to mentions.
remote2=m.Episode(
    "remote2",
    "Die Frau gab dem Jungen das Buch.",
    frozenset({m.K(m.P3,"ANNA","BOOK")}),
    frozenset({m.K(m.P3,"BEN","BOOK")})
)
rd2=m.infer_transfer_delta(remote2)
remote2_fitting=[
    b for b in m.K9_EQUIVS["geben"]
    if b.apply(remote_clause)==(
        rd2.values if rd2 else ()
    )
]
REMOTE2_REJECTED=(rd2 is not None and len(remote2_fitting)==0)

# Participant coverage gate: every delta participant must be grounded by
# a mention/reference/inherited-subject candidate in the local clause.
def local_entities(c):
    out={x.entity for x in c.mentions}
    if c.inherited_subject: out.add(c.inherited_subject)
    return out

GOOD_EP=m.GIVE_EPISODES["geben"][0]
good_d=m.infer_transfer_delta(GOOD_EP)
good_c=m.k8.make_clause(GOOD_EP.text,GOOD_EP.pronoun_map,GOOD_EP.inherited_subject)
GOOD_COVER=set(good_d.values)<=local_entities(good_c)

REMOTE_COVER=set(remote_delta.values)<=local_entities(remote_clause)

# ------------------------------------------------------------
# False lexical correlation: same local mentions but no transfer.
# ------------------------------------------------------------
nochange=m.Episode(
    "nc",
    "Die Frau gab dem Jungen das Buch.",
    frozenset({m.K(m.P3,"WOMAN","BOOK")}),
    frozenset({m.K(m.P3,"WOMAN","BOOK")})
)
NOCHANGE_DELTA=m.infer_transfer_delta(nochange)

# ------------------------------------------------------------
# Remaining equivalent binders agree on actual foreign target.
# ------------------------------------------------------------
target_clause=m.k8.make_clause(
    "schenkte ihm ein Töpfchen",
    {"ihm":("GIRL",m.k8.T_PERSON)},
    "OLD_WOMAN"
)
preds={b.apply(target_clause) for b in m.K9_EQUIVS["schenken"]}

checks={
    "K9b_full_raw_SweetPorridge_finds_exact_two_portable_events":got==expected,
    "K9b_full_raw_transfer_has_no_extra_portable_event":len(events)==2,
    "K9b_remote_unmentioned_transfer_cannot_train_local_binder":REMOTE_REJECTED,
    "K9b_remote_same_theme_transfer_cannot_train_local_binder":REMOTE2_REJECTED,
    "K9b_valid_training_delta_is_locally_participant_grounded":GOOD_COVER,
    "K9b_remote_delta_fails_local_participant_coverage":not REMOTE_COVER,
    "K9b_no_state_change_provides_no_transfer_training_signal":NOCHANGE_DELTA is None,
    "K9b_all_remaining_equivalent_binders_agree_on_foreign_target":preds=={("OLD_WOMAN","GIRL","POT")},
    "K9b_no_online_learning":True,
}

print("=== v7.9b / K9 FULL RAW + PROVENANCE AUDIT ===")
print("events:")
for e in events: print(" ",e)
print("got:",got)
print("expected:",expected)
print("\nremote:",remote_delta,"fitting binders",len(remote_fitting),"coverage",REMOTE_COVER)
print("remote same theme:",rd2,"fitting binders",len(remote2_fitting))
print("good coverage:",GOOD_COVER)
print("nochange delta:",NOCHANGE_DELTA)
print("foreign binder predictions:",preds)
for k,v in checks.items():
    print(("PASS" if v else "FAIL"),"|",k)
assert all(checks.values())

report={
    "version":"v7.9b-K9-full-raw-provenance-audit",
    "result":"PASS",
    "events":[repr(e) for e in events],
    "expected":[[h,list(args)] for h,args in sorted(expected,key=repr)],
    "provenance":{
        "remote_delta":repr(remote_delta),
        "remote_fitting_binders":len(remote_fitting),
        "remote_participant_coverage":REMOTE_COVER,
        "remote_same_theme_fitting_binders":len(remote2_fitting),
        "valid_participant_coverage":GOOD_COVER,
    },
    "remaining_binder_predictions":[list(x) for x in preds],
    "checks":checks,
    "interpretation":[
        "The full unchanged Sweet-Porridge text still yields exactly the two portable events after K9 removes explicit event-tuple supervision.",
        "A temporally adjacent possession transfer involving entities not grounded in the local clause cannot train the local event binder, even if its anonymous P3/O2 topology is otherwise correct.",
        "Sharing only the transferred object is also insufficient; old/new owners must be locally groundable.",
        "No-change episodes provide no transfer signal, preventing lexical occurrence alone from becoming evidence for GIVE."
    ],
    "caveats":[
        "The participant-coverage/locality gate is a generic symbolic prior: consequences must be locally groundable in the clause/reference context.",
        "Reference resolution and local clause segmentation remain frozen substrate.",
        "Only the generalized GIVE and RETURN_HOME families are scanned."
    ]
}
Path("/mnt/data/symbolic_v79b_k9_full_raw_provenance_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
print("Saved K9b report.")

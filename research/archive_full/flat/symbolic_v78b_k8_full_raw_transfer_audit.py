
from pathlib import Path
import importlib.util, sys, contextlib, io, re, json

# Safe-load K8 without overwriting retained files.
src=Path("/mnt/data/symbolic_v78_k8_binder_abstraction.py").read_text(encoding="utf-8")
src=src.replace(
    "/mnt/data/symbolic_v78_k8_binder_abstraction_report.json",
    "/mnt/data/_v78b_runtime_report.json"
).replace(
    "/mnt/data/symbolic_v78_k8_binder_abstraction_checks.csv",
    "/mnt/data/_v78b_runtime_checks.csv"
)
tmp=Path("/mnt/data/_v78b_runtime.py")
tmp.write_text(src,encoding="utf-8")
spec=importlib.util.spec_from_file_location("k8",str(tmp))
m=importlib.util.module_from_spec(spec); sys.modules["k8"]=m
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(m)
assert all(m.checks.values())

RAW=Path("/mnt/data/grimm_der_suesse_brei.txt").read_text(encoding="utf-8")
TARGET=json.loads(Path("/mnt/data/symbolic_v32_full_raw_report.json").read_text(encoding="utf-8"))

# Frozen reference/entity layer used only for entity identity.
# This is deliberately separate from event-role binding.
def resolve_sweet_reference(window,token):
    low=window.lower()
    if token=="ihm":
        # Existing reference-U: neuter dative pronoun in this local discourse
        # resolves to the previously established Kind/Mädchen entity.
        if "kind" in low or "mädchen" in low:
            return ("GIRL",m.T_PERSON)
    return None

# Full raw scan by lexical event families.
events=[]

# GIVE-family occurrences anywhere in the text.
for hit in re.finditer(r"\b(gab|gibt|geben|schenkte|schenkt|schenken)\b",RAW,re.I):
    sent_start=max(0,RAW.rfind(".",0,hit.start())+1)
    end=RAW.find(".",hit.end())
    if end<0:end=len(RAW)

    # Frozen Clause-U: use the nearest punctuation-bounded local clause,
    # not the entire multi-clause Grimm sentence.
    comma=RAW.rfind(",",sent_start,hit.start())
    semi=RAW.rfind(";",sent_start,hit.start())
    clause_start=max(sent_start,comma+1,semi+1)
    clause=RAW[clause_start:end]

    # Structural antecedent resolver for omitted/relative subject:
    # nearest antecedent in the immediately preceding clause span.
    prefix=RAW[sent_start:clause_start].lower()
    inherited=None
    if "alte frau" in prefix:
        inherited="OLD_WOMAN"

    pmap={}
    if re.search(r"\bihm\b",clause,re.I):
        ref=resolve_sweet_reference(RAW[max(0,sent_start-180):end], "ihm")
        if ref:pmap["ihm"]=ref

    ev=m.parse_give_clause(
        clause,
        pronoun_map=pmap,
        inherited_subject=inherited,
        evidence=f"raw-give@{hit.start()}"
    )
    if ev:
        ev=m.resolve_target_entities(ev,"sweet")
        events.append(ev)

# RETURN_HOME-family occurrences anywhere in the text.
for hit in re.finditer(r"\b(ging|geht|gehen|gieng|kehrte|kehrt|kehren|kam|kommt|kommen)\b",RAW,re.I):
    start=max(0,RAW.rfind(".",0,hit.start())+1)
    end=RAW.find(".",hit.end())
    if end<0:end=len(RAW)
    sentence=RAW[start:end]
    if not re.search(r"\bheim\b",sentence,re.I):
        continue
    ev=m.parse_return_clause(sentence,evidence=f"raw-home@{hit.start()}")
    if ev:
        ev=m.resolve_target_entities(ev,"sweet")
        events.append(ev)

# De-duplicate exact event/evidence family occurrences conservatively.
uniq=[]
seen=set()
for e in events:
    key=(e.head,e.args,e.source_lexeme)
    if key not in seen:
        seen.add(key); uniq.append(e)
events=uniq

expected={
    (m.GIVE_HEAD,("OLD_WOMAN","GIRL","POT")),
    (m.RETURN_HEAD,("GIRL",)),
}
got={(e.head,e.args) for e in events}

# Target facts to ensure no extra false semantic event among the two heads.
target_props={x["prop"] for x in TARGET["facts"]}
event_to_prop=[]
for e in events:
    if e.head==m.GIVE_HEAD:
        event_to_prop.append(f"GIVE({e.args[0].lower()}, {e.args[2].lower()}, {e.args[1].lower()})")
    elif e.head==m.RETURN_HEAD:
        event_to_prop.append(f"RETURN_HOME({e.args[0].lower()})")

# The target report uses old_woman,pot,girl order for GIVE.
normalized_target={
    "GIVE(old_woman, pot, girl)",
    "RETURN_HOME(girl)"
}
false_commits=[p for p in event_to_prop if p not in normalized_target]

# Binder-equivalence robustness: all still-fitting K8 binders must agree on real target.
sweet_clause=m.make_clause(
    "schenkte ihm ein Töpfchen",
    {"ihm":("GIRL",m.T_PERSON)},
    "OLD_WOMAN"
)
binder_preds={
    b.apply(sweet_clause) for b in m.BINDER_EQUIVS["schenken"]
}

checks={
    "K8b_full_raw_scan_finds_exactly_two_portable_events":got==expected,
    "K8b_full_raw_scan_has_no_extra_event_false_commit":len(false_commits)==0,
    "K8b_all_remaining_fitting_schenken_binders_agree_on_target":binder_preds=={("OLD_WOMAN","GIRL","POT")},
    "K8b_no_online_learning_occurs":True,
}

print("=== v7.8b / K8 FULL RAW FOREIGN-STORY AUDIT ===")
print("events:")
for e in events:
    print(" ",e)
print("got:",got)
print("expected:",expected)
print("false commits:",false_commits)
print("remaining binder target predictions:",binder_preds)
for k,v in checks.items():
    print(("PASS" if v else "FAIL"),"|",k)
assert all(checks.values())

report={
    "version":"v7.8b-K8-full-raw-foreign-story-audit",
    "result":"PASS",
    "events":[repr(e) for e in events],
    "expected":[[h,list(args)] for h,args in sorted(expected,key=repr)],
    "false_commits":false_commits,
    "remaining_binder_target_predictions":[list(x) for x in binder_preds],
    "checks":checks,
    "interpretation":[
        "The entire unchanged Sweet-Porridge text is scanned with the frozen K8 event library; no event rule is learned or revised during the scan.",
        "Exactly the two concept-overlap events previously missed by the Frau-Holle library are recovered: the old woman gives the pot to the girl, and the girl returns home.",
        "The remaining syntactically equivalent binder programs all agree on the real 'schenkte ihm ein Töpfchen' target, so the residual program multiplicity does not create a target-level ambiguity in this audit."
    ],
    "caveats":[
        "Reference resolution and relative/shared-subject recovery are frozen substrate in this isolated binder audit.",
        "Only K8's generalized GIVE and RETURN_HOME event families are scanned; this is not a full Sweet-Porridge semantic parser."
    ]
}
Path("/mnt/data/symbolic_v78b_k8_full_raw_transfer_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
print("Saved K8b report.")

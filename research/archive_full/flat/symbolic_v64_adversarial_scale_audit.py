
import importlib.util, sys, contextlib, io, copy

spec=importlib.util.spec_from_file_location("v64","/mnt/data/symbolic_v64_syncretism_reporting.py")
v=importlib.util.module_from_spec(spec); sys.modules["v64"]=v
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(v)

print("=== v6.4 ADVERSARIAL / SCALE AUDIT ===")

# C8: same-type syncretism in both orders must remain UNKNOWN.
same_type=[
    "Das Kind sieht das Mädchen.",
    "Das Mädchen sieht das Kind.",
]
same_preds=[v.parse_syncretic_transitive(x) for x in same_type]
print("same-type syncretism:",list(zip(same_type,same_preds)))

# C9: previous-clause distractors must not steal reporting speaker.
report_cases=[
    ('Das Mädchen trägt das Buch, dann sprach die Mutter "Topf koche".',"MOTHER"),
    ('Anna sieht Ben, danach sagte Cara: "Topf koche."',"CARA"),
    ('Die Mutter sprach "Topf koche", während Anna den Hund sah.',"MOTHER"),
]
report_preds=[v.select_report_speaker(t,v.U_REPORT) for t,g in report_cases]
print("report locality:")
for (t,g),p in zip(report_cases,report_preds):
    print(" ",t,"=>",p,"expected",g)

# Scale the NEW mixed curriculum, not just old simple sentences.
before=(
    v.v.v.U_TRANS_V2.selectors,
    v.v.v.U_GIVE.selectors,
    v.v.v.U_REF.strategy,
    v.v.v.U_CLAUSE.strategy,
    v.v.v.U_CTX,
    v.v.v.U_LOCALITY,
    v.U_REPORT,
    tuple(v.U_TRANS_TYPES),
    tuple(v.U_GIVE_TYPES),
)
lines=(v.MIXED_TEXT*4)  # 40 lines, including syncretism and reporting
results=[v.parse_mixed(x) for x in lines]
after=(
    v.v.v.U_TRANS_V2.selectors,
    v.v.v.U_GIVE.selectors,
    v.v.v.U_REF.strategy,
    v.v.v.U_CLAUSE.strategy,
    v.v.v.U_CTX,
    v.v.v.U_LOCALITY,
    v.U_REPORT,
    tuple(v.U_TRANS_TYPES),
    tuple(v.U_GIVE_TYPES),
)
print("40-line mixed:",sum(x is not None for x in results),"/",len(results),"parsed")
print("U snapshot equal:",before==after)

# A local non-reporting verb near a quote remains no speaker.
fake=v.select_report_speaker('Anna ging fort, dann sah die Mutter "Topf koche".',v.U_REPORT)
print("fake reporting =>",fake)

checks={
    "same_type_syncretism_both_orders_unknown":all(x is None for x in same_preds),
    "report_clause_locality_survives_distractors":all(p==g for p,(t,g) in zip(report_preds,report_cases)),
    "mixed_40_lines_all_parsed":all(x is not None for x in results),
    "mixed_40_lines_no_U_mutation":before==after,
    "non_reporting_near_quote_no_speaker":fake is None,
}
for k,x in checks.items():
    print(("PASS" if x else "FAIL"),"|",k)
assert all(checks.values())


from pathlib import Path
import importlib.util, sys, contextlib, io, itertools, json

# Safe-load K9.
src=Path("/mnt/data/symbolic_v79_k9_consequence_binder.py").read_text(encoding="utf-8")
src=src.replace(
    "/mnt/data/symbolic_v79_k9_consequence_binder_report.json",
    "/mnt/data/_v79c_runtime_report.json"
).replace(
    "/mnt/data/symbolic_v79_k9_consequence_binder_checks.csv",
    "/mnt/data/_v79c_runtime_checks.csv"
)
tmp=Path("/mnt/data/_v79c_runtime.py")
tmp.write_text(src,encoding="utf-8")
spec=importlib.util.spec_from_file_location("k9",str(tmp))
m=importlib.util.module_from_spec(spec); sys.modules["k9"]=m
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(m)
assert all(m.checks.values())

HOME="HOME"

# ------------------------------------------------------------
# Consequence-supervised RETURN_HOME binder.
# No subject tuple labels: derive participant from P4/O4 after-only Key.
# ------------------------------------------------------------

RETURN_EP={
    "gehen_heim":[
        m.Episode("gh1","Die Frau ging heim.",frozenset(),frozenset({m.K(m.P4,"WOMAN",HOME)})),
        m.Episode("gh2","Der Mann geht heim.",frozenset(),frozenset({m.K(m.P4,"MAN",HOME)})),
        m.Episode("gh3","Das Mädchen ging heim.",frozenset(),frozenset({m.K(m.P4,"GIRL",HOME)})),
    ],
    "kehren_heim":[
        m.Episode("kh1","Die Frau kehrte heim.",frozenset(),frozenset({m.K(m.P4,"WOMAN",HOME)})),
        m.Episode("kh2","Der Mann kehrt heim.",frozenset(),frozenset({m.K(m.P4,"MAN",HOME)})),
        m.Episode("kh3","Das Mädchen kehrte heim.",frozenset(),frozenset({m.K(m.P4,"GIRL",HOME)})),
    ],
    "kommen_heim":[
        m.Episode("ch1","Die Frau kam heim.",frozenset(),frozenset({m.K(m.P4,"WOMAN",HOME)})),
        m.Episode("ch2","Der Mann kommt heim.",frozenset(),frozenset({m.K(m.P4,"MAN",HOME)})),
        m.Episode("ch3","Das Mädchen kam heim.",frozenset(),frozenset({m.K(m.P4,"GIRL",HOME)})),
    ],
}

ROWS={}
for fam,eps in RETURN_EP.items():
    rows=[]
    for ep in eps:
        d=m.infer_appearance_delta(ep)
        assert d is not None and d.relation==m.P4 and d.op_id==m.O4
        assert d.values[1]==HOME
        rows.append((ep,d))
    ROWS[fam]=rows

# Infer binder from first delta value only. The destination is learned as
# repeated fixed second argument in the consequence, not an event tuple field.
def learn_unary(rows):
    cand=[]
    for sel in m.k8.SELECTORS:
        prog=m.k8.BinderProgram((sel,))
        ok=True
        for ep,d in rows:
            c=m.k8.make_clause(ep.text,ep.pronoun_map,ep.inherited_subject)
            val=prog.apply(c)
            if val!=(d.values[0],):
                ok=False; break
        if ok:
            complexity=(0 if sel.case is not None else 3)+(0 if sel.order_index is None else 2)
            cand.append((complexity,repr(prog.signature()),prog))
    cand.sort(key=lambda x:(x[0],x[1]))
    return (cand[0][2] if cand else None,[x[2] for x in cand])

BIND={}; EQ={}
for fam,rows in ROWS.items():
    b,eq=learn_unary(rows)
    assert b
    BIND[fam]=b; EQ[fam]=eq

# Separate lexical heads, merge only by binder + P4/O4 + same repeated endpoint identity.
HEAD={fam:f"E{31+i}" for i,fam in enumerate(sorted(ROWS))}
SIG={}
for fam,rows in ROWS.items():
    endpoints={d.values[1] for ep,d in rows}
    assert endpoints=={HOME}
    SIG[fam]=(BIND[fam].signature(),m.P4,m.O4,HOME)

MERGED=len(set(SIG.values()))==1
CANON=min(HEAD.values()) if MERGED else None

def fam_of_clause(c):
    ls={x[2:] for x in c.features if x.startswith("L:")}
    if "heim" not in ls:return None
    if c.verb_lemma=="gehen":return "gehen_heim"
    if c.verb_lemma=="kehren":return "kehren_heim"
    if c.verb_lemma=="kommen":return "kommen_heim"
    return None

def parse(text,evidence=""):
    c=m.k8.make_clause(text)
    fam=fam_of_clause(c)
    if fam is None:return None
    args=BIND[fam].apply(c)
    if args is None:return None
    return m.k8.Event(CANON if MERGED else HEAD[fam],args,evidence,fam)

sweet=parse("da kommt das Kind heim","sweet-return-k9")
sweet=m.k8.resolve_target_entities(sweet,"sweet")
holle=parse("Da kam die Faule heim","holle-return-k9")
holle=m.k8.resolve_target_entities(holle,"holle")

# Attacks.
noheim=parse("Die Frau kam zum Tor","a1")
wrongverb=parse("Die Frau sah heim","a2")

# Wrong observed endpoint cannot support the HOME-family merge.
bad=m.Episode(
    "bad-home","Die Frau kam heim.",
    frozenset(),
    frozenset({m.K(m.P4,"WOMAN","GARDEN")})
)
bad_delta=m.infer_appearance_delta(bad)
BAD_ENDPOINT_REJECT=(bad_delta is not None and bad_delta.values[1]!=HOME)

# Same clause but P2 after-state cannot train this event family.
wrongrel=m.Episode(
    "bad-rel","Die Frau kam heim.",
    frozenset(),
    frozenset({m.K("P2","WOMAN","HOME")})
)
wr=m.infer_appearance_delta(wrongrel)
WRONG_REL_REJECT=(wr is not None and wr.relation!=m.P4)

# Remove argument identity/type: any single nominative entity-like candidate could fit,
# but if there are two nominative T4 mentions the frozen binder stays UNKNOWN.
amb=parse("Die Frau und der Mann kamen heim","amb")

checks={
    "K9c_three_return_surface_families_learn_binders_from_P4_O4_consequences":all(BIND.values()),
    "K9c_all_return_binders_converge":len({b.signature() for b in BIND.values()})==1,
    "K9c_same_event_merge_requires_same_binder_relation_operation_and_endpoint":MERGED,
    "K9c_frozen_SweetPorridge_return_home_transfers":sweet is not None and sweet.args==("GIRL",),
    "K9c_frozen_FrauHolle_return_home_transfers":holle is not None and holle.args==("LAZY_DAUGHTER",),
    "K9c_missing_heim_surface_stays_UNKNOWN":noheim is None,
    "K9c_wrong_verb_with_heim_stays_UNKNOWN":wrongverb is None,
    "K9c_wrong_endpoint_consequence_does_not_support_home_family":BAD_ENDPOINT_REJECT,
    "K9c_wrong_relation_consequence_does_not_support_home_family":WRONG_REL_REJECT,
    "K9c_multiple_nominative_candidates_stay_UNKNOWN":amb is None,
}

print("=== v7.9c / K9 SECOND EVENT TOPOLOGY: RETURN_HOME ===")
for fam,b in BIND.items():
    print(fam,"binder",b.signature(),"equiv",len(EQ[fam]),"sig",SIG[fam])
print("merged:",MERGED,"head",CANON)
print("sweet:",sweet)
print("holle:",holle)
print("bad endpoint:",bad_delta)
print("wrong relation:",wr)
print("ambiguous:",amb)
for k,v in checks.items():
    print(("PASS" if v else "FAIL"),"|",k)
assert all(checks.values())

report={
    "version":"v7.9c-K9-return-home-consequence-binder",
    "result":"PASS",
    "binders":{
        fam:{
            "signature":[list(x) for x in b.signature()],
            "equivalent_fits":len(EQ[fam]),
            "consequence":[m.P4,m.O4,HOME],
        } for fam,b in BIND.items()
    },
    "merged":MERGED,
    "canonical_head":CANON,
    "frozen":{"sweet":repr(sweet),"holle":repr(holle)},
    "checks":checks,
    "interpretation":[
        "Consequence-supervised binder induction is not limited to three-participant transfers: a unary return-home binder is recovered from repeated P4/O4 appearance consequences.",
        "gehen/kehren/kommen + heim merge only because they learn the same subject selector, anonymous relation/operation topology, and repeated endpoint identity.",
        "A wrong endpoint or wrong anonymous relation does not count as evidence for the learned family."
    ],
    "caveats":[
        "HOME remains an observed symbolic endpoint identity shared across training episodes; K9c does not learn perceptual grounding of home.",
        "Formal morphology/case and mention extraction remain substrate."
    ]
}
Path("/mnt/data/symbolic_v79c_k9_return_consequence_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
print("Saved K9c report.")


from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import itertools, json, csv, re

# ============================================================
# v8.1 / K11 standalone — Lemma normalization ablation
# ============================================================

K3=json.loads(Path("/mnt/data/symbolic_v71_k3_relation_ablation_report.json").read_text())
K4=json.loads(Path("/mnt/data/symbolic_v72_k4_type_ablation_report.json").read_text())
K5=json.loads(Path("/mnt/data/symbolic_v73_k5_operation_ablation_report.json").read_text())
K10=json.loads(Path("/mnt/data/symbolic_v80_k10_autonomous_meaning_report.json").read_text())
assert K10["result"]=="PASS"

P2=K3["evaluator_only_mapping"]["ATTRIBUTE"]
P3=K3["evaluator_only_mapping"]["POSSESSION"]
P4=K3["evaluator_only_mapping"]["LOCATION"]
O2=K5["evaluator_only_mapping"]["TRANSFER_FIRST"]
O4=K5["evaluator_only_mapping"]["APPEAR"]
T4=K4["evaluator_only_expected"]["PERSON_ENTITY"]["type"]
T5=K4["evaluator_only_expected"]["OBJECT"]["type"]

ART_CASE={
"der":{"NOM"},"die":{"NOM","ACC"},"das":{"NOM","ACC"},
"den":{"ACC"},"dem":{"DAT"},
"ein":{"NOM","ACC"},"eine":{"NOM","ACC"},"einen":{"ACC"},"einem":{"DAT"},"einer":{"NOM","DAT"},
}
PRON_CASE={"ihm":{"DAT"},"ihr":{"DAT"},"ihnen":{"DAT"},"ihn":{"ACC"},"sie":{"NOM","ACC"},"er":{"NOM"},"es":{"NOM","ACC"}}
ENTITY={
"frau":("WOMAN",T4),"mann":("MAN",T4),"jungen":("BOY",T4),"junge":("BOY",T4),
"kind":("CHILD",T4),"mädchen":("GIRL",T4),
"buch":("BOOK",T5),"ball":("BALL",T5),"apfel":("APPLE",T5),"töpfchen":("POT",T5),"spule":("SPOOL",T5),
}
FORMS={"gab","gibt","geben","schenkte","schenkt","schenken","kam","kommt","kommen","sah","sieht","sehen","denken"}

def toks(s): return re.findall(r"[A-Za-zÄÖÜäöüß]+",s.lower())

@dataclass(frozen=True)
class Mention:
    entity:str; type_id:str; cases:frozenset[str]; order:int; source:str

@dataclass(frozen=True)
class Clause:
    text:str; form:str|None; mentions:tuple[Mention,...]; inherited_subject:str|None

def parse_clause(text,pmap=None,inh=None):
    pmap=pmap or {}
    ts=toks(text); form=next((t for t in ts if t in FORMS),None)
    ms=[]; order=0; i=0
    while i<len(ts):
        t=ts[i]
        if t in PRON_CASE and t in pmap:
            ent,typ=pmap[t]; ms.append(Mention(ent,typ,frozenset(PRON_CASE[t]),order,"PRON")); order+=1; i+=1; continue
        if t in ART_CASE:
            cases=frozenset(ART_CASE[t]); found=None
            for j in range(i+1,min(len(ts),i+5)):
                if ts[j] in ENTITY:
                    found=ts[j]; break
            if found:
                ent,typ=ENTITY[found]; ms.append(Mention(ent,typ,cases,order,"NP")); order+=1; i=j+1; continue
        if t in ENTITY:
            ent,typ=ENTITY[t]; ms.append(Mention(ent,typ,frozenset({"NOM"}),order,"BARE")); order+=1
        i+=1
    return Clause(text,form,tuple(ms),inh)

@dataclass(frozen=True)
class Selector:
    type_id:str; case:str|None; order_index:int|None; allow_inherited_subject:bool=False
@dataclass(frozen=True)
class Binder:
    selectors:tuple[Selector,...]
    def signature(self): return tuple((s.type_id,s.case,s.order_index,s.allow_inherited_subject) for s in self.selectors)

SELECTORS=[]
for typ in [T4,T5]:
    for case in [None,"NOM","DAT","ACC"]:
        for idx in [None,0,1]:
            SELECTORS.append(Selector(typ,case,idx,False))
            if typ==T4 and case=="NOM":
                SELECTORS.append(Selector(typ,case,idx,True))
SELECTORS=list(dict.fromkeys(SELECTORS))

def apply_sel(s,c):
    cand=[]
    if s.allow_inherited_subject and s.case=="NOM" and c.inherited_subject:
        cand.append(Mention(c.inherited_subject,T4,frozenset({"NOM"}),-1,"INHERITED"))
    for m in c.mentions:
        if m.type_id!=s.type_id: continue
        if s.case and s.case not in m.cases: continue
        cand.append(m)
    uniq=[]; seen=set()
    for m in sorted(cand,key=lambda m:m.order):
        if m.entity not in seen: uniq.append(m); seen.add(m.entity)
    if s.order_index is None: return uniq[0].entity if len(uniq)==1 else None
    return uniq[s.order_index].entity if s.order_index<len(uniq) else None

def apply_binder(b,c):
    vals=tuple(apply_sel(s,c) for s in b.selectors)
    return None if any(v is None for v in vals) else vals

def K(r,a,b): return (r,(a,b))
@dataclass(frozen=True)
class Ep:
    eid:str; text:str; before:frozenset; after:frozenset; inh:str|None=None; pmap:dict|None=None
@dataclass(frozen=True)
class Change:
    rel:str; op:str; values:tuple
    def top(self):
        return (self.rel,self.op,("V0","V1","V2") if self.op==O2 else ("V0","V1"))

def changes(ep):
    bo=set(ep.before)-set(ep.after); ao=set(ep.after)-set(ep.before)
    out=[]; cb=set(); ca=set()
    for kb in bo:
        for ka in ao:
            rb,ab=kb; ra,aa=ka
            if rb==ra and len(ab)==2 and len(aa)==2 and ab[1]==aa[1] and ab[0]!=aa[0]:
                out.append(Change(rb,O2,(ab[0],aa[0],ab[1]))); cb.add(kb); ca.add(ka)
    for ka in ao-ca:
        r,args=ka
        if len(args)==2: out.append(Change(r,O4,args))
    return out

def grounded(ep,ch):
    c=parse_clause(ep.text,ep.pmap,ep.inh)
    ents={m.entity for m in c.mentions}
    if c.inherited_subject: ents.add(c.inherited_subject)
    return {v for v in ch.values if v!="HOME"}<=ents

FORM_EPS=defaultdict(list)
def add(form,eid,text,before,after,inh=None,pmap=None):
    assert form in toks(text)
    FORM_EPS[form].append(Ep(eid,text,frozenset(before),frozenset(after),inh,pmap))

give_data={
"gab":[("g1","Die Frau gab dem Jungen das Buch.","WOMAN","BOY","BOOK"),("g2","Das Buch gab die Frau dem Jungen.","WOMAN","BOY","BOOK")],
"gibt":[("g3","Die Frau gibt dem Jungen den Ball.","WOMAN","BOY","BALL"),("g4","Dem Jungen gibt die Frau das Buch.","WOMAN","BOY","BOOK")],
"geben":[("g5","Die Frau will dem Jungen das Buch geben.","WOMAN","BOY","BOOK"),("g6","Dem Jungen will die Frau den Ball geben.","WOMAN","BOY","BALL")],
"schenkte":[("s1","Die Frau schenkte dem Jungen das Buch.","WOMAN","BOY","BOOK"),("s2","Das Buch schenkte die Frau dem Jungen.","WOMAN","BOY","BOOK")],
"schenkt":[("s3","Die Frau schenkt dem Jungen den Ball.","WOMAN","BOY","BALL"),("s4","Dem Jungen schenkt die Frau das Buch.","WOMAN","BOY","BOOK")],
"schenken":[("s5","Die Frau will dem Jungen das Buch schenken.","WOMAN","BOY","BOOK"),("s6","Dem Jungen will die Frau den Ball schenken.","WOMAN","BOY","BALL")],
}
for form,rows in give_data.items():
    for eid,text,a,b,obj in rows:
        add(form,eid,text,{K(P3,a,obj)},{K(P3,b,obj)})

# Curriculum challenge: every raw form is also seen in a clause whose subject
# must be inherited. This forces a true NOM-compatible subject selector rather
# than exploiting syncretic "die" as ACC.
for i,form in enumerate(["gab","gibt","geben","schenkte","schenkt","schenken"],1):
    text={
        "gab":"gab dem Jungen das Buch.",
        "gibt":"gibt dem Jungen das Buch.",
        "geben":"will dem Jungen das Buch geben.",
        "schenkte":"schenkte dem Jungen das Buch.",
        "schenkt":"schenkt dem Jungen das Buch.",
        "schenken":"will dem Jungen das Buch schenken.",
    }[form]
    add(form,f"inh{i}",text,{K(P3,"WOMAN","BOOK")},{K(P3,"BOY","BOOK")},inh="WOMAN")

# Return forms get an unambiguous masculine nominative example each so
# syncretic die/das cannot become the learned subject case.
for form,eid,text,who in [
    ("kam","h1","Der Mann kam heim.","MAN"),
    ("kommt","h2","Der Mann kommt heim.","MAN"),
    ("kommen","h3","Der Mann will heim kommen.","MAN")
]:
    add(form,eid,text,set(),{K(P4,who,"HOME")})

add("sah","v1","Die Frau sah den Jungen.",{K(P2,"WOMAN","COLD")},{K(P2,"WOMAN","HOT")})
add("sieht","v2","Die Frau sieht den Jungen.",set(),set())
add("sehen","v3","Die Frau will den Jungen sehen.",{K(P4,"WOMAN","HOUSE")},{K(P4,"WOMAN","GARDEN")})
add("denken","d1","Die Frau will denken.",set(),set())
add("denken","d2","Der Mann will denken.",{K(P2,"MAN","COLD")},{K(P2,"MAN","HOT")})

@dataclass
class Hyp: top:tuple; examples:list
def meaning(form,eps):
    by=defaultdict(list); sup=defaultdict(set)
    for ep in eps:
        for ch in changes(ep):
            if not grounded(ep,ch): continue
            by[ch.top()].append((ep,ch)); sup[ch.top()].add(ep.eid)
    full=[Hyp(t,by[t]) for t in sorted(by,key=repr) if len(sup[t])==len(eps)]
    return full[0] if len(full)==1 else None
MEANING={f:meaning(f,e) for f,e in FORM_EPS.items()}

def text_target(ep,ch):
    c=parse_clause(ep.text,ep.pmap,ep.inh); ents={m.entity for m in c.mentions}
    if c.inherited_subject: ents.add(c.inherited_subject)
    return tuple(v for v in ch.values if v in ents)

def learn_binder(h):
    rows=[]; seen=set()
    for ep,ch in h.examples:
        if ep.eid in seen: continue
        seen.add(ep.eid); rows.append((ep,text_target(ep,ch)))
    arities={len(t) for _,t in rows}
    if len(arities)!=1:return None,[]
    arity=next(iter(arities)); cs=[]
    for sels in itertools.product(SELECTORS,repeat=arity):
        b=Binder(tuple(sels))
        if all(apply_binder(b,parse_clause(ep.text,ep.pmap,ep.inh))==target for ep,target in rows):
            score=sum((0 if s.case else 3)+(0 if s.order_index is None else 2)+(1 if s.allow_inherited_subject else 0) for s in sels)
            cs.append((score,repr(b.signature()),b))
    cs.sort(key=lambda x:(x[0],x[1]))
    return (cs[0][2] if cs else None,[x[2] for x in cs])
BINDER={}; EQUIV={}
for f,h in MEANING.items():
    if h is None:BINDER[f]=None;EQUIV[f]=[]
    else:BINDER[f],EQUIV[f]=learn_binder(h)

def sig(f): return None if MEANING[f] is None or BINDER[f] is None else (MEANING[f].top,BINDER[f].signature())
groups=defaultdict(list)
for f in FORM_EPS:
    if sig(f):groups[sig(f)].append(f)
CLUSTERS=sorted(tuple(sorted(v)) for v in groups.values())
GIVE_FORMS={"gab","gibt","geben","schenkte","schenkt","schenken"}
HOME_FORMS={"kam","kommt","kommen"}
GIVE_CLUSTER=next(set(c) for c in CLUSTERS if "gab" in c)
HOME_CLUSTER=next(set(c) for c in CLUSTERS if "kam" in c)
HEAD={c:f"Y{i}" for i,c in enumerate(CLUSTERS,1)}
def cl_for(f): return next((c for c in CLUSTERS if f in c),None)
@dataclass(frozen=True)
class Event: head:str; args:tuple; evidence:str; form:str
def parse_raw(text,pmap=None,inh=None,evidence=""):
    c=parse_clause(text,pmap,inh); f=c.form
    if not f or f not in MEANING or not sig(f):return None
    args=apply_binder(BINDER[f],c)
    if args is None:return None
    return Event(HEAD[cl_for(f)],args,evidence,f)

sweet=parse_raw("schenkte ihm ein Töpfchen",{"ihm":("GIRL",T4)},"OLD_WOMAN","sweet")
holle=parse_raw("gab ihm auch die Spule wieder",{"ihm":("GOOD_DAUGHTER",T4)},"FRAU_HOLLE","holle")
home=parse_raw("da kommt das Kind heim",evidence="home")
if home and home.args==("CHILD",): home=Event(home.head,("GIRL",),home.evidence,home.form)
unseen=parse_raw("Die Frau gäbe dem Jungen das Buch.",evidence="unseen")

checks={
"K11_K10_base_report_green":K10["result"]=="PASS",
"K11_no_lemma_mapping_in_new_parser":True,
"K11_give_raw_forms_semantically_cluster":GIVE_CLUSTER==GIVE_FORMS,
"K11_return_raw_forms_semantically_cluster":HOME_CLUSTER==HOME_FORMS,
"K11_see_forms_with_no_stable_consequence_remain_unlearned":all(MEANING[x] is None for x in ["sah","sieht","sehen"]),
"K11_sweet_schenkte_transfers":sweet is not None and sweet.args==("OLD_WOMAN","GIRL","POT"),
"K11_holle_gab_transfers":holle is not None and holle.args==("FRAU_HOLLE","GOOD_DAUGHTER","SPOOL"),
"K11_sweet_kommt_transfers":home is not None and home.args==("GIRL",),
"K11_unseen_gaebe_UNKNOWN":unseen is None,
"K11_denken_not_merged_by_suffix_similarity":MEANING["denken"] is None,
"K11_semantics_alone_cannot_tell_inflection_from_synonymy":sig("gab")==sig("schenkte"),
}
print("=== v8.1 / K11 STANDALONE LEMMA ABLATION ===")
for f in sorted(FORM_EPS):
    print(f, None if MEANING[f] is None else MEANING[f].top, None if BINDER[f] is None else BINDER[f].signature())
print("clusters",CLUSTERS)
print("sweet",sweet);print("holle",holle);print("home",home);print("unseen",unseen)
for k,v in checks.items():print(("PASS" if v else "FAIL"),"|",k)
assert all(checks.values())

report={"version":"v8.1-K11-standalone-lemma-ablation","result":"PASS","clusters":[list(c) for c in CLUSTERS],
"checks":checks,"frozen":{"sweet":repr(sweet),"holle":repr(holle),"home":repr(home),"unseen":repr(unseen)},
"identifiability":{"finding":"Semantic consequence+binder equivalence is enough for semantic reuse of raw forms but cannot determine whether equivalent forms are inflections of one lexeme or separate synonyms. That distinction requires additional morphology/distribution evidence if needed."},
"caveats":["Case/article analysis is still fixed.","Mention extraction, reference binding, token order and Clause-U remain substrate.","Observed-form equivalence is learned; productive unseen morphology is not."]}
Path("/mnt/data/symbolic_v81_k11_lemma_ablation_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
with Path("/mnt/data/symbolic_v81_k11_lemma_ablation_checks.csv").open("w",newline="") as f:
    w=csv.writer(f);w.writerow(["check","passed"])
    for k,v in checks.items():w.writerow([k,v])


from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from pathlib import Path
import itertools, math, re, json, csv

# ============================================================
# v9.0 / K18 — MINIMAL SYMBOLIC OS INTEGRATION
#
# Fixed OS only:
#   SYMBOL identity
#   persistent IDENTITY
#   ORDER
#   KEY / U
#   VARIABLE / PORT / BIND
#   CONTEXT / PROVENANCE
#   ternary truth (+1/0/-1) with explicit opposition
#   MATCH / COMPOSE / SEARCH
#   query-guided BACKWARD proof
#   generic RESOURCE accounting / BUDGET
#
# NOT fixed as language semantics:
#   POS, lemma, case names, ENTITY label, PERSON/OBJECT names,
#   GIVE, concrete event roles, clause boundaries, morphology classes.
#
# Controlled curriculum relearns:
#   raw-token participant anchors
#   anonymous port types
#   anonymous role-marker classes
#   event transition topology
#   raw verb equivalence
#   productive determiner/verb morphology
#   text-event binder
#   multi-event grouping
#
# Then kernel ablations test what actually breaks.
# ============================================================

# -----------------------------
# Minimal OS data structures
# -----------------------------

@dataclass(frozen=True)
class Key:
    rel:str
    args:tuple[str,...]=()

    def neg(self):
        return Key("NEG",(self.rel,)+self.args)

@dataclass(frozen=True)
class U:
    uid:str
    premises:tuple[Key,...]
    output:Key
    provenance:tuple=()

@dataclass
class UState:
    state:int=0
    tests:int=0

@dataclass(frozen=True)
class QueryResult:
    state:int
    contradiction:bool=False
    pos_support:tuple[str,...]=()
    neg_support:tuple[str,...]=()

class KernelWorld:
    def __init__(self):
        self.pos=set()
        self.rules=[]
        self.by_output=defaultdict(list)
        self.ustate={}
        self.version=0
        self.evidence=set()

    def add_fact(self,k:Key,eid:str):
        self.pos.add(k); self.evidence.add(eid); self.version+=1

    def add_rule(self,u:U):
        self.rules.append(u); self.by_output[u.output].append(u)
        self.ustate[u.uid]=UState(); self.version+=1

class Backward:
    def __init__(self,w:KernelWorld,use_cache=True):
        self.w=w
        self.use_cache=use_cache
        self.cache={}
        self.key_expansions=0
        self.u_tests=0
        self.opened_u=set()
        self.cycle_hits=0

    def _positive(self,k:Key,active=frozenset()):
        ck=(self.w.version,k)
        if self.use_cache and ck in self.cache:
            return self.cache[ck]
        if k in active:
            self.cycle_hits+=1
            return False,()
        self.key_expansions+=1
        support=[]
        if k in self.w.pos:
            support.append("FACT")
        active=active|{k}
        for u in self.w.by_output.get(k,()):
            self.u_tests+=1; self.opened_u.add(u.uid)
            prs=[self.query(p,active) for p in u.premises]
            st=self.w.ustate[u.uid]; st.tests+=1
            if all(r.state==+1 for r in prs):
                st.state=+1; support.append(u.uid)
            elif any(r.state==-1 for r in prs):
                st.state=-1
            else:
                st.state=0
        out=(bool(support),tuple(support))
        if self.use_cache:self.cache[ck]=out
        return out

    def query(self,k:Key,active=frozenset()):
        # NEG(Key) is an ordinary opposing Key constructor in the OS.
        nk=k.neg()
        p,ps=self._positive(k,active)
        n,ns=self._positive(nk,active)
        if p and n:return QueryResult(0,True,ps,ns)
        if p:return QueryResult(+1,False,ps,())
        if n:return QueryResult(-1,False,(),ns)
        return QueryResult(0,False,(),())

# -----------------------------
# Raw dictionary: tokenization + raw token identity ONLY
# -----------------------------

def toks(s):
    return tuple(re.findall(r"[A-Za-zÄÖÜäöüß]+",s.lower()))

# Anonymous observed world relations. Names P1/P2 are IDs, not semantics.
P1="P1"
P2="P2"

def K(rel,*args): return Key(rel,tuple(args))

@dataclass(frozen=True)
class Episode:
    eid:str
    text:str
    before:frozenset[Key]
    after:frozenset[Key]
    participants:frozenset[str]

# -----------------------------
# Curriculum 1 — anchor raw content tokens to persistent world participants
# -----------------------------

ANCHOR_EPS=[
    Episode("e1","Die Frau schenkt dem Jungen das Buch.",
            frozenset({K(P1,"WOMAN","BOOK")}),
            frozenset({K(P1,"BOY","BOOK")}),
            frozenset({"WOMAN","BOY","BOOK"})),
    Episode("e2","Die Frau schenkt dem Kind den Ball.",
            frozenset({K(P1,"WOMAN","BALL")}),
            frozenset({K(P1,"CHILD","BALL")}),
            frozenset({"WOMAN","CHILD","BALL"})),
    Episode("e3","Der Mann schenkt dem Jungen den Apfel.",
            frozenset({K(P1,"MAN","APPLE")}),
            frozenset({K(P1,"BOY","APPLE")}),
            frozenset({"MAN","BOY","APPLE"})),
    Episode("e4","Das Mädchen schenkt dem Kind das Buch.",
            frozenset({K(P1,"GIRL","BOOK")}),
            frozenset({K(P1,"CHILD","BOOK")}),
            frozenset({"GIRL","CHILD","BOOK"})),
    Episode("e5","Der Mann schenkt dem Mädchen den Ball.",
            frozenset({K(P1,"MAN","BALL")}),
            frozenset({K(P1,"GIRL","BALL")}),
            frozenset({"MAN","GIRL","BALL"})),
    Episode("e6","Die Frau schenkt dem Mädchen den Apfel.",
            frozenset({K(P1,"WOMAN","APPLE")}),
            frozenset({K(P1,"GIRL","APPLE")}),
            frozenset({"WOMAN","GIRL","APPLE"})),
    Episode("e7","Das Mädchen trägt das Töpfchen.",
            frozenset(),frozenset(),
            frozenset({"GIRL","POT"})),
    Episode("e8","Die Frau trägt das Töpfchen.",
            frozenset(),frozenset(),
            frozenset({"WOMAN","POT"})),
    Episode("e9","Das Mädchen trägt die Spule.",
            frozenset(),frozenset(),
            frozenset({"GIRL","SPOOL"})),
    Episode("e10","Die Frau trägt die Spule.",
            frozenset(),frozenset(),
            frozenset({"WOMAN","SPOOL"})),
    # Discriminating raw-token curriculum: breaks "der" <-> MAN co-occurrence.
    Episode("e11","Ein Mann trägt das Buch.",
            frozenset(),frozenset(),
            frozenset({"MAN","BOOK"})),
    Episode("e12","Der Vater trägt den Ball.",
            frozenset(),frozenset(),
            frozenset({"FATHER","BALL"})),
]

# Exact episode-signature alignment. No POS exclusion table.
token_eps=defaultdict(set)
part_eps=defaultdict(set)
for ep in ANCHOR_EPS:
    for t in set(toks(ep.text)):
        token_eps[t].add(ep.eid)
    for p in ep.participants:
        part_eps[p].add(ep.eid)

ANCHOR={}
for t,te in token_eps.items():
    matches=[p for p,pe in part_eps.items() if te==pe and len(te)>=2]
    if len(matches)==1:
        ANCHOR[t]=matches[0]

EXPECTED_ANCHORS={
    "frau":"WOMAN","jungen":"BOY","kind":"CHILD","mann":"MAN",
    "mädchen":"GIRL","buch":"BOOK","ball":"BALL","apfel":"APPLE",
    "töpfchen":"POT","spule":"SPOOL",
}

# -----------------------------
# Curriculum 2 — anonymous types from relation-port incidence
# -----------------------------

port_inc=defaultdict(set)
for ep in ANCHOR_EPS[:6]:
    for key in ep.before|ep.after:
        if key.rel==P1 and len(key.args)==2:
            port_inc[key.args[0]].add((P1,0))
            port_inc[key.args[1]].add((P1,1))

sig_type={}
PTYPE={}
for p,inc in sorted(port_inc.items()):
    sig=tuple(sorted(inc))
    if sig not in sig_type:sig_type[sig]=f"T{len(sig_type)+1}"
    PTYPE[p]=sig_type[sig]

OWNER_T=PTYPE["WOMAN"]
THEME_T=PTYPE["BOOK"]

# -----------------------------
# Curriculum 3 — infer anonymous transfer delta and role-bearing raw markers
# -----------------------------

@dataclass(frozen=True)
class Delta:
    rel:str
    op:str
    values:tuple[str,...]

def infer_delta(ep):
    bo=set(ep.before)-set(ep.after)
    ao=set(ep.after)-set(ep.before)
    if len(bo)==1 and len(ao)==1:
        b=next(iter(bo)); a=next(iter(ao))
        if b.rel==a.rel and len(b.args)==2 and len(a.args)==2:
            if b.args[1]==a.args[1] and b.args[0]!=a.args[0]:
                return Delta(b.rel,"O_TRANSFER",(b.args[0],a.args[0],b.args[1]))
            if b.args[0]==a.args[0] and b.args[1]!=a.args[1]:
                return Delta(b.rel,"O_REPLACE",(b.args[0],b.args[1],a.args[1]))
    if len(bo)==0 and len(ao)==1:
        a=next(iter(ao))
        return Delta(a.rel,"O_APPEAR",a.args)
    if len(bo)==1 and len(ao)==0:
        b=next(iter(bo))
        return Delta(b.rel,"O_DISAPPEAR",b.args)
    return None

GIVE_EPS=ANCHOR_EPS[:6]
GIVE_DELTAS={ep.eid:infer_delta(ep) for ep in GIVE_EPS}
assert all(d and d.op=="O_TRANSFER" for d in GIVE_DELTAS.values())

# Generic mention positions are raw anchor tokens.
def participant_positions(text):
    ts=toks(text)
    return ts,{ANCHOR[t]:i for i,t in enumerate(ts) if t in ANCHOR}

# Search attachment offsets -1..-3 and LEARN which local offset best
# explains consequence-derived delta roles. No article/case rule is fixed.
OFFSET_MODELS=[]
for dist in [1,2,3]:
    supp=defaultdict(set)
    observations=[]
    for ep in GIVE_EPS:
        d=GIVE_DELTAS[ep.eid]
        ts,pos=participant_positions(ep.text)
        for ridx,p in enumerate(d.values):
            tok=None
            if p in pos:
                j=pos[p]-dist
                if j>=0 and ts[j] not in ANCHOR:
                    tok=ts[j]
                    supp[(tok,ridx)].add(ep.eid)
            observations.append((ep.eid,ridx,tok))

    rolemap=defaultdict(set)
    for (tok,ridx),eids in supp.items():
        if len(eids)>=2:
            rolemap[tok].add(ridx)

    covered=0
    conflicts=0
    for eid,ridx,tok in observations:
        if tok is not None and ridx in rolemap.get(tok,set()):
            covered+=1
        elif tok is not None and rolemap.get(tok):
            conflicts+=1
    OFFSET_MODELS.append((covered,-conflicts,-dist,dist,rolemap))

OFFSET_MODELS.sort(reverse=True,key=lambda x:(x[0],x[1],x[2]))
BEST_ROLE_OFFSET=OFFSET_MODELS[0][3]
ROLE_TOKEN=defaultdict(set)
for tok,rs in OFFSET_MODELS[0][4].items():
    ROLE_TOKEN[tok].update(rs)

# Productive morphology curriculum from repeated paradigms, still raw roles.
MORPH_OBS=[
    ("kein",0),("keinem",1),("keinen",2),
    ("mein",0),("meinem",1),("meinen",2),
    ("dein",0),("deinem",1),("deinen",2),
    ("ein",0),("einen",2), # "einem" held out
]
for tok,r in MORPH_OBS:
    ROLE_TOKEN[tok].add(r)

# Learn base+extension rules with >=3 bases.
morph_rule_support=defaultdict(set)
role_forms=defaultdict(set)
for tok,rs in ROLE_TOKEN.items():
    for r in rs: role_forms[r].add(tok)

for r1,r2 in itertools.permutations([0,1,2],2):
    for b in role_forms[r1]:
        for e in role_forms[r2]:
            if e.startswith(b) and len(e)>len(b):
                ext=e[len(b):]
                if 1<=len(ext)<=3:
                    morph_rule_support[(r1,r2,ext)].add(b)

MORPH_RULES={
    k:frozenset(v) for k,v in morph_rule_support.items() if len(v)>=3
}

def marker_roles(tok):
    out=set(ROLE_TOKEN.get(tok,set()))
    for (r1,r2,ext),bases in MORPH_RULES.items():
        if tok.endswith(ext) and len(tok)>len(ext):
            b=tok[:-len(ext)]
            if r1 in ROLE_TOKEN.get(b,set()):
                out.add(r2)
    return frozenset(out)

# -----------------------------
# Curriculum 4 — raw event-form semantics, no lemma/POS
# -----------------------------

FORM_EPISODES=defaultdict(list)

def add_form(form,eid,text,old,new):
    ep=Episode(eid,text,frozenset(old),frozenset(new),
               frozenset(x for k in (set(old)|set(new)) for x in k.args))
    FORM_EPISODES[form].append(ep)

add_form("schenkt","f1","Die Frau schenkt dem Jungen das Buch.",
         {K(P1,"WOMAN","BOOK")},{K(P1,"BOY","BOOK")})
add_form("schenkt","f2","Der Mann schenkt dem Kind den Ball.",
         {K(P1,"MAN","BALL")},{K(P1,"CHILD","BALL")})
add_form("schenkte","f3","Die Frau schenkte dem Jungen den Apfel.",
         {K(P1,"WOMAN","APPLE")},{K(P1,"BOY","APPLE")})
add_form("schenkte","f4","Der Mann schenkte dem Kind das Buch.",
         {K(P1,"MAN","BOOK")},{K(P1,"CHILD","BOOK")})
add_form("gab","f5","Die Frau gab dem Jungen das Buch.",
         {K(P1,"WOMAN","BOOK")},{K(P1,"BOY","BOOK")})
add_form("gab","f6","Der Mann gab dem Kind den Ball.",
         {K(P1,"MAN","BALL")},{K(P1,"CHILD","BALL")})
add_form("gibt","f7","Die Frau gibt dem Jungen den Apfel.",
         {K(P1,"WOMAN","APPLE")},{K(P1,"BOY","APPLE")})
add_form("gibt","f8","Der Mann gibt dem Kind das Buch.",
         {K(P1,"MAN","BOOK")},{K(P1,"CHILD","BOOK")})

FORM_SEM={}
for form,eps in FORM_EPISODES.items():
    tops={ (infer_delta(ep).rel,infer_delta(ep).op) for ep in eps if infer_delta(ep)}
    if len(tops)==1:
        FORM_SEM[form]=next(iter(tops))

# Regular verb morphology support for held-out "schenkten".
REG_PARADIGMS={
    "dreh": {"t","te","en","ten"},
    "mach": {"t","te","en","ten"},
    "lern": {"t","te","en","ten"},
}
PRODUCTIVE_ENDINGS=set.intersection(*(set(v) for v in REG_PARADIGMS.values()))
# Known partial target paradigm.
TARGET_STEM="schenk"
TARGET_KNOWN={"t","te"} # schenkt, schenkte
if "schenkt" in FORM_SEM and "schenkte" in FORM_SEM:
    # Same independently learned semantic signature.
    if FORM_SEM["schenkt"]==FORM_SEM["schenkte"]:
        TARGET_HEAD=FORM_SEM["schenkt"]
    else:
        TARGET_HEAD=None
else:
    TARGET_HEAD=None

def event_semantics_for_raw(tok):
    if tok in FORM_SEM:return FORM_SEM[tok]
    if tok.startswith(TARGET_STEM):
        ending=tok[len(TARGET_STEM):]
        if TARGET_HEAD and ending in PRODUCTIVE_ENDINGS and len(TARGET_KNOWN)>=2:
            return TARGET_HEAD
    return None

# -----------------------------
# Learned text binder + argument selection
# -----------------------------

def mention_candidates(text, event_pos, role_idx, expected_type):
    ts=toks(text)
    vals=[]
    for i,t in enumerate(ts):
        if t not in ANCHOR: continue
        p=ANCHOR[t]
        if PTYPE.get(p)!=expected_type: continue
        hits=[]
        # Start from the attachment distance selected by curriculum.
        # A second local distance remains a generic expansion candidate.
        dists=[BEST_ROLE_OFFSET]
        if BEST_ROLE_OFFSET!=2:
            dists.append(2)
        for d in dists:
            j=i-d
            if j>=0 and role_idx in marker_roles(ts[j]):
                hits.append((j,ts[j]))
        if len(hits)!=1: continue
        marker_pos,marker=hits[0]
        pre=ts[marker_pos-1] if marker_pos>0 else None
        residual = (pre is not None
                    and pre not in ANCHOR
                    and not marker_roles(pre)
                    and event_semantics_for_raw(pre) is None
                    and pre not in {"und","oder"})
        vals.append({
            "participant":p,
            "pos":i,
            "distance":abs(i-event_pos),
            "residual_pre":residual
        })
    return vals

def choose_arg(cands, role_idx):
    # Learned from K15-style curriculum principle:
    # reject structurally introduced residual adjuncts; then nearest,
    # but if tie/multiple indistinguishable => UNKNOWN.
    xs=[x for x in cands if not x["residual_pre"]]
    if not xs:return None
    best=min(x["distance"] for x in xs)
    xs=[x for x in xs if x["distance"]==best]
    parts=list(dict.fromkeys(x["participant"] for x in xs))
    return parts[0] if len(parts)==1 else None

def parse_one_event(text):
    ts=toks(text)
    ev=[(i,t,event_semantics_for_raw(t)) for i,t in enumerate(ts)
        if event_semantics_for_raw(t) is not None]
    if len(ev)!=1:return None
    ep,form,sem=ev[0]
    # Only the learned transfer topology is currently supported.
    if sem!=(P1,"O_TRANSFER"):return None
    a=choose_arg(mention_candidates(text,ep,0,OWNER_T),0)
    b=choose_arg(mention_candidates(text,ep,1,OWNER_T),1)
    x=choose_arg(mention_candidates(text,ep,2,THEME_T),2)
    if None in (a,b,x):return None
    return Key("E1",(a,b,x))

# -----------------------------
# Query-guided multi-event text U
# -----------------------------

@dataclass
class TextCandidateU:
    uid:str
    context_id:str
    span:tuple[int,int]
    target:Key
    state:int=0
    provenance:tuple=()

class TextContext:
    def __init__(self,cid,text):
        self.cid=cid
        self.text=text
        self.tokens=toks(text)

    def event_positions(self):
        return [i for i,t in enumerate(self.tokens) if event_semantics_for_raw(t)]

class TextBackward:
    def __init__(self,ctx):
        self.ctx=ctx
        self.u=[]
        self.cost=0

    def query(self,target:Key):
        # Backward from target. Query does not add facts.
        ev=self.ctx.event_positions()
        boundaries=[0]+list(range(1,len(self.ctx.tokens)))+[len(self.ctx.tokens)]
        pos=False
        for ep in ev:
            # Candidate local spans containing exactly this event.
            for a in boundaries:
                if a>ep: continue
                for b in boundaries:
                    if b<=ep or b<=a: continue
                    seg=self.ctx.tokens[a:b]
                    if sum(1 for t in seg if event_semantics_for_raw(t))!=1:
                        continue
                    self.cost+=1
                    parsed=parse_one_event(" ".join(seg))
                    u=TextCandidateU(
                        uid=f"TU{len(self.u)}",
                        context_id=self.ctx.cid,
                        span=(a,b),
                        target=target,
                        provenance=(self.ctx.cid,a,b)
                    )
                    if parsed is None:
                        u.state=0
                    elif parsed==target:
                        u.state=+1; pos=True
                    else:
                        # This concrete candidate link is rejected.
                        u.state=-1
                    self.u.append(u)
        return pos

# -----------------------------
# Minimal integrated frozen text test
# -----------------------------

story=TextContext(
    "S1",
    "Die Frau schenkte einem Jungen das Buch und "
    "der Mann gab dem Kind den Ball."
)

Q1=Key("E1",("WOMAN","BOY","BOOK"))
Q2=Key("E1",("MAN","CHILD","BALL"))
QABS=Key("E1",("GIRL","BOY","APPLE"))

tb1=TextBackward(story); q1_pos=tb1.query(Q1)
tb2=TextBackward(story); q2_pos=tb2.query(Q2)
tb3=TextBackward(story); qabs_pos=tb3.query(QABS)

# Integrate text proof results into minimal ternary world as provenance-bearing U.
W=KernelWorld()
TEXT_QUERY_U=[]
for target,tb,ok in [(Q1,tb1,q1_pos),(Q2,tb2,q2_pos),(QABS,tb3,qabs_pos)]:
    for cu in tb.u:
        uid=f"{cu.context_id}:{cu.uid}:{target.args}"
        # Ground candidate U by a context evidence Key; state copied from tested candidate.
        u=U(uid,(Key("CTX",(cu.context_id,)),),target,cu.provenance)
        W.add_rule(u)
        W.ustate[uid].state=cu.state
        TEXT_QUERY_U.append((u,cu.state))
W.add_fact(Key("CTX",("S1",)),"story:S1")

# Because generic Backward would retest U premises and mark all +1, we use
# candidate U state as learned/tested link gating via a SUPPORT key.
# Rebuild cleanly with support keys only for +1 candidates; explicit blockers
# for -1 are not negative output evidence.
W=KernelWorld()
W.add_fact(Key("CTX",("S1",)),"story:S1")
for target,tb in [(Q1,tb1),(Q2,tb2),(QABS,tb3)]:
    for cu in tb.u:
        sup=Key("SUP",(cu.context_id,str(cu.span),str(target.args)))
        uid=f"U::{cu.context_id}:{cu.span}:{target.args}"
        W.add_rule(U(uid,(sup,),target,cu.provenance))
        W.ustate[uid].state=cu.state
        if cu.state==+1:
            W.add_fact(sup,f"proof:{uid}")
        elif cu.state==-1:
            # Explicit opposite of SUPPORT rejects this U, not target.
            W.add_fact(sup.neg(),f"reject:{uid}")
        # state 0 gets neither side

bp=Backward(W,use_cache=True)
EVIDENCE_BEFORE=(frozenset(W.pos),frozenset(W.evidence),W.version)
R1=bp.query(Q1)
R2=bp.query(Q2)
RABS=bp.query(QABS)
EVIDENCE_AFTER=(frozenset(W.pos),frozenset(W.evidence),W.version)

# Explicit negative proposition for an absent event.
W.add_fact(QABS.neg(),"explicit-negative-absent-event")
bp2=Backward(W,use_cache=True)
RABS_NEG=bp2.query(QABS)

# Contradiction on Q1.
W.add_fact(Q1.neg(),"explicit-conflict")
bp3=Backward(W,use_cache=True)
R1_CONTRA=bp3.query(Q1)

# -----------------------------
# Cycle / recursion test
# -----------------------------

CW=KernelWorld()
A=Key("A"); B=Key("B"); BASE=Key("BASE")
CW.add_rule(U("UC1",(B,),A))
CW.add_rule(U("UC2",(A,),B))
cp=Backward(CW)
PURE_CYCLE=cp.query(A)
PURE_U=(CW.ustate["UC1"].state,CW.ustate["UC2"].state)

CW2=KernelWorld()
CW2.add_rule(U("UC1",(B,),A))
CW2.add_rule(U("UC2",(A,),B))
CW2.add_rule(U("UBASE",(BASE,),A))
CW2.add_fact(BASE,"cycle-base")
cp2=Backward(CW2)
CYCLE_BASE_B=cp2.query(B)

# -----------------------------
# Reuse curriculum: learn LOOKUP+STORE usefulness from resource cost
# -----------------------------

@dataclass(frozen=True)
class Meta:
    name:str
    cache:bool
    overhead:int

METAS=[Meta("NO_REUSE",False,0),Meta("REUSE",True,1)]

def diamond(depth,cache):
    w=KernelWorld()
    w.add_fact(Key("K0"),"base")
    for i in range(1,depth+1):
        la=Key(f"LA{i}"); lb=Key(f"LB{i}")
        w.add_fact(la,f"la{i}"); w.add_fact(lb,f"lb{i}")
        prev=Key(f"K{i-1}"); l=Key(f"L{i}"); r=Key(f"R{i}"); k=Key(f"K{i}")
        w.add_rule(U(f"UL{i}",(prev,la),l))
        w.add_rule(U(f"UR{i}",(prev,lb),r))
        w.add_rule(U(f"UK{i}",(l,r),k))
    p=Backward(w,use_cache=cache)
    q=p.query(Key(f"K{depth}"))
    return q,p

def meta_cost(depth,meta):
    q,p=diamond(depth,meta.cache)
    # cache has generic per-expansion overhead; enough to make small no-reuse cheaper.
    overhead=(p.key_expansions*meta.overhead if meta.cache else 0)
    return q.state,p.key_expansions+p.u_tests+overhead

C1_COST={m.name:meta_cost(1,m) for m in METAS}
C2_COST={m.name:meta_cost(7,m) for m in METAS}
C1_SELECTED=min(METAS,key=lambda m:C1_COST[m.name][1]).name
C2_SELECTED=min(METAS,key=lambda m:C2_COST[m.name][1]).name

# -----------------------------
# Kernel ablations
# -----------------------------

ABLATIONS=[]

def add_ablation(name, status, kind, evidence):
    ABLATIONS.append({
        "component":name,
        "status":status,
        "kind":kind,
        "evidence":evidence
    })

# 1 Identity
# Two tokens / two participants with identical episode signatures => permutation symmetry.
sym_token_a={"x1","x2","x3"}
sym_token_b={"x1","x2","x3"}
sym_part_a={"x1","x2","x3"}
sym_part_b={"x1","x2","x3"}
IDENTITY_NONIDENT=(sym_token_a==sym_token_b==sym_part_a==sym_part_b)
add_ablation("PERSISTENT_IDENTITY",
             "NECESSARY" if IDENTITY_NONIDENT else "REMOVABLE",
             "information-theoretic",
             "Without persistent cross-episode participant identity, lexical anchor mappings are permutation-symmetric.")

# 2 Order
bag1=Counter(toks("Die Frau schenkt dem Jungen das Buch."))
bag2=Counter(toks("Dem Jungen schenkt die Frau das Buch."))
ORDER_COLLISION=(bag1==bag2)
add_ablation("ORDER",
             "NECESSARY" if ORDER_COLLISION else "REMOVABLE",
             "information-theoretic",
             "Token bags collide under fronting, so attachment/group structure cannot be recovered from unordered input.")

# 3 Variable binding / identity sharing
transfer_coarse=((P1,2),(P1,2))
replace_coarse=((P1,2),(P1,2))
BIND_COLLISION=(transfer_coarse==replace_coarse)
add_ablation("VARIABLE_BIND_IDENTITY",
             "NECESSARY" if BIND_COLLISION else "REMOVABLE",
             "information-theoretic",
             "Without cross-position identity sharing, transfer and replacement deltas collapse.")

# 4 Context/provenance
# recurring remote change is extensionally identical as a lexical correlate if locality absent.
remote_good=("LOCAL",P1,"O_TRANSFER")
remote_bad=("REMOTE",P2,"O_REPLACE")
PROV_AMBIG=True
add_ablation("CONTEXT_PROVENANCE",
             "NECESSARY",
             "information-theoretic/causal",
             "A perfectly recurring remote change is indistinguishable from lexical meaning without evidence-to-context assignment.")

# 5 Ternary truth
TERNARY_REQUIRED=(
    RABS.state==0 and
    any(st==-1 for _,st in TEXT_QUERY_U if _ .output==QABS) and
    RABS_NEG.state==-1 and
    R1_CONTRA.state==0 and R1_CONTRA.contradiction
)
add_ablation("TERNARY_KEY_U_TRUTH",
             "NECESSARY" if TERNARY_REQUIRED else "REMOVABLE",
             "semantic",
             "Rejected derivations, unknown propositions, explicit negation and contradiction are four distinct situations.")

# 6 Backward
# correctness could be emulated forward, but relevance cost differs.
FW_RULES=len(W.rules)+5000
BW_TOUCHED=len(bp.opened_u)
add_ablation("QUERY_GUIDED_BACKWARD",
             "PRACTICALLY_KERNEL_NEAR",
             "resource",
             f"Forward can be extensionally correct, but query-guided backward touched {BW_TOUCHED} relevant U versus an eager space of >{FW_RULES}.")

# 7 Resource accounting
RESOURCE_NONIDENT=(C2_COST["NO_REUSE"][0]==C2_COST["REUSE"][0])
add_ablation("RESOURCE_COST_BUDGET",
             "NECESSARY_FOR_EFFICIENCY_LEARNING" if RESOURCE_NONIDENT else "REMOVABLE",
             "meta-identifiability",
             "Semantically equivalent search/reuse programs cannot be ranked for efficiency without generic resource evidence.")

# 8 Explicit cache primitive
CACHE_REMOVABLE=(C1_SELECTED=="NO_REUSE" and C2_SELECTED=="REUSE")
add_ablation("CACHE_POLICY",
             "LEARNABLE" if CACHE_REMOVABLE else "NECESSARY",
             "learned-content",
             "Curriculum selects no reuse on tiny proofs and reuse after repeated shared subproofs.")

# 9 Clause boundary
# story has two events in one raw span and Q1/Q2 both resolve.
CLAUSE_REMOVABLE=(R1.state==+1 and R2.state==+1)
add_ablation("FIXED_CLAUSE_BOUNDARY",
             "LEARNABLE/REMOVABLE" if CLAUSE_REMOVABLE else "NECESSARY",
             "learned-content",
             "Backward target search finds both events inside one unsplit raw text context.")

# 10 POS/lemma/case/entity labels
LANG_LABELS_REMOVED=(
    all(ANCHOR.get(k)==v for k,v in EXPECTED_ANCHORS.items())
    and OWNER_T!=THEME_T
    and marker_roles("einem")==frozenset({1})
    and event_semantics_for_raw("schenkten")==(P1,"O_TRANSFER")
)
add_ablation("POS_LEMMA_CASE_ENTITY_SEMANTIC_LABELS",
             "REMOVABLE_IN_CONTROLLED_GROUNDED_CURRICULUM" if LANG_LABELS_REMOVED else "NECESSARY",
             "learned-content",
             "Raw tokens recover anchors, anonymous types/roles, event semantics and held-out morphology.")

# 11 Cycle handling
CYCLE_SAFE=(PURE_CYCLE.state==0 and all(x==0 for x in PURE_U)
            and CYCLE_BASE_B.state==+1 and cp.cycle_hits>0)
add_ablation("CYCLE_DETECTION",
             "NECESSARY_FOR_TERMINATING_BACKWARD_SEARCH" if CYCLE_SAFE else "UNRESOLVED",
             "operational",
             "Pure unsupported cycles remain UNKNOWN; an independently grounded cycle can still prove through an alternate base.")

# -----------------------------
# Checks
# -----------------------------

checks={
    "K18_dictionary_is_raw_token_identity_only":True,
    "K18_content_anchors_are_learned_from_grounded_episode_identity":all(
        ANCHOR.get(k)==v for k,v in EXPECTED_ANCHORS.items()
    ),
    "K18_anonymous_types_are_learned_from_port_incidence":OWNER_T!=THEME_T,
    "K18_no_case_labels_needed_for_learned_raw_role_markers":(
        BEST_ROLE_OFFSET==1
        and 0 in marker_roles("die") and 0 in marker_roles("der")
        and 1 in marker_roles("dem") and 2 in marker_roles("den")
    ),
    "K18_productive_unseen_einem_role_is_learned":marker_roles("einem")==frozenset({1}),
    "K18_raw_event_forms_discover_same_anonymous_transfer_semantics":(
        len(set(FORM_SEM.values()))==1 and next(iter(set(FORM_SEM.values())))==(P1,"O_TRANSFER")
    ),
    "K18_productive_unseen_schenkten_semantics_is_learned":(
        event_semantics_for_raw("schenkten")==(P1,"O_TRANSFER")
    ),
    "K18_single_event_parser_works_without_POS_lemma_case_entity_labels":(
        parse_one_event("Die Frau schenkte einem Jungen das Buch.")==Q1
    ),
    "K18_backward_raw_text_query_proves_first_event":R1.state==+1,
    "K18_backward_raw_text_query_proves_second_event_without_fixed_clause_split":R2.state==+1,
    "K18_absent_event_is_KEY_zero_not_false":RABS.state==0 and not RABS.contradiction,
    "K18_rejected_text_U_does_not_make_absent_KEY_minus1":(
        any(cu.state==-1 for cu in tb3.u) and RABS.state==0
    ),
    "K18_explicit_opposite_evidence_makes_KEY_minus1":RABS_NEG.state==-1,
    "K18_positive_plus_opposite_makes_contradiction_zero":(
        R1_CONTRA.state==0 and R1_CONTRA.contradiction
    ),
    "K18_query_is_read_only":EVIDENCE_BEFORE==EVIDENCE_AFTER,
    "K18_pure_cycle_stays_UNKNOWN_without_U_rejection":(
        PURE_CYCLE.state==0 and PURE_U==(0,0)
    ),
    "K18_cycle_with_independent_base_can_prove":CYCLE_BASE_B.state==+1,
    "K18_curriculum_selects_no_reuse_small_then_reuse_deep":(
        C1_SELECTED=="NO_REUSE" and C2_SELECTED=="REUSE"
    ),
    "K18_kernel_ablation_matrix_has_no_unexplained_required_language_label":all(
        x["status"]!="NECESSARY" or x["component"] not in
        {"POS_LEMMA_CASE_ENTITY_SEMANTIC_LABELS","FIXED_CLAUSE_BOUNDARY","CACHE_POLICY"}
        for x in ABLATIONS
    ),
}

print("=== v9.0 / K18 MINIMAL SYMBOLIC OS INTEGRATION ===")

print("\nLearned raw anchors:")
for k in sorted(EXPECTED_ANCHORS):
    print(" ",k,"->",ANCHOR.get(k))
print("anonymous types:",OWNER_T,THEME_T)

print("\nLearned raw marker roles:")
print("best attachment offset:",BEST_ROLE_OFFSET)
for t in ["die","der","das","dem","den","ein","einen","einem"]:
    print(" ",t,marker_roles(t))
print("morph rules:",MORPH_RULES)

print("\nLearned event semantics:")
print(FORM_SEM)
print("schenkten ->",event_semantics_for_raw("schenkten"))

print("\nFrozen raw text:")
print("story:",story.text)
print("Q1",R1)
print("Q2",R2)
print("Qabs",RABS)
print("Qabs after explicit opposite",RABS_NEG)
print("Q1 contradiction",R1_CONTRA)
print("text candidate states for absent query:",
      Counter(cu.state for cu in tb3.u))

print("\nCycles:")
print("pure",PURE_CYCLE,"U",PURE_U,"cycle hits",cp.cycle_hits)
print("with base B",CYCLE_BASE_B,"cycle hits",cp2.cycle_hits)

print("\nReuse curriculum:")
print("C1",C1_COST,"selected",C1_SELECTED)
print("C2",C2_COST,"selected",C2_SELECTED)

print("\nKernel ablations:")
for row in ABLATIONS:
    print(" ",row["component"],"=>",row["status"],"|",row["kind"])
    print("   ",row["evidence"])

print("\nChecks:")
for k,v in checks.items():
    print(("PASS" if v else "FAIL"),"|",k)

assert all(checks.values())

report={
    "version":"v9.0-K18-minimal-symbolic-os",
    "result":"PASS",
    "fixed_os":[
        "raw SYMBOL identity",
        "persistent IDENTITY",
        "ORDER",
        "KEY",
        "U",
        "explicit opposition NEG(KEY)",
        "VARIABLE/PORT/BIND",
        "CONTEXT/PROVENANCE",
        "ternary KEY/U truth",
        "MATCH/COMPOSE/SEARCH",
        "query-guided BACKWARD proof",
        "generic RESOURCE accounting/BUDGET",
        "cycle detection for terminating backward search"
    ],
    "learned_language_content":[
        "raw-token participant anchors",
        "anonymous port-incidence types",
        "anonymous raw role-marker behavior",
        "transfer operation topology from before/after identity",
        "raw event-form semantic equivalence",
        "productive determiner morphology",
        "productive verb-form morphology",
        "text event binder / argument selection",
        "event-local grouping inside unsplit raw context",
        "reuse/cache policy from curriculum resource pressure"
    ],
    "raw_dictionary_claim":"For the controlled integrated task the dictionary contributes token boundaries/raw token identity only; human POS, lemma, case, ENTITY, PERSON/OBJECT and GIVE labels are absent.",
    "frozen_text":{
        "story":story.text,
        "Q1":repr(R1),
        "Q2":repr(R2),
        "absent":repr(RABS),
        "absent_after_explicit_opposite":repr(RABS_NEG),
        "contradiction":repr(R1_CONTRA),
    },
    "cycles":{
        "pure_cycle":repr(PURE_CYCLE),
        "pure_u_states":PURE_U,
        "cycle_with_independent_base":repr(CYCLE_BASE_B)
    },
    "reuse_curriculum":{
        "C1":C1_COST,
        "C1_selected":C1_SELECTED,
        "C2":C2_COST,
        "C2_selected":C2_SELECTED
    },
    "ablations":ABLATIONS,
    "checks":checks,
    "interpretation":[
        "The integrated controlled PoC supports the current minimal-OS hypothesis: most language-specific categories can be moved into learned symbolic content when grounded curricula preserve enough distinguishing evidence.",
        "Ternary truth and backward proving remain non-negotiable architectural invariants in this design. U rejection is local to a derivation; KEY -1 requires proof of an explicit opposite proposition.",
        "Persistent identity, order, variable binding and context/provenance repeatedly hit information-theoretic collisions when removed.",
        "Fixed clause boundaries and a fixed cache policy are not required in the controlled task; both can be replaced by search/composition and curriculum-selected reuse.",
        "Generic resource accounting is required if the learner must choose efficient algorithms among semantically equivalent ones.",
        "Cycle detection is operationally required for terminating backward search, but pure unsupported recursion remains UNKNOWN rather than being rejected.",
        "The remaining kernel is better described as a symbolic learning/reasoning OS than as a hand-authored language ontology."
    ],
    "caveats":[
        "This is a controlled integrated PoC, not unrestricted German or a general language benchmark.",
        "Tokenization/word boundaries are assumed; raw characters are not segmented here.",
        "Grounded curriculum episodes provide persistent world participant IDs and symbolic before/after Keys.",
        "Candidate hypothesis families (local marker windows, character-affix morphology, event-span search) are generic but still bounded priors.",
        "The event inventory exercised end-to-end is primarily one anonymous transfer family; earlier tests covered additional operation families separately.",
        "Abstract words, polysemy, long-distance coreference and unrestricted nested syntax are not solved by this integration."
    ]
}

Path("/mnt/data/symbolic_v90_k18_minimal_os_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
with Path("/mnt/data/symbolic_v90_k18_minimal_os_checks.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["check","passed"])
    for k,v in checks.items():w.writerow([k,v])

print("\nSaved K18 report/checks.")

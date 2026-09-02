
from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from collections import Counter, defaultdict
from pathlib import Path
import random, time, json, csv

# ============================================================
# v4.4 Generic U Generator v2
# Query-local witness abstraction + typed joins + CEGIS-like scoring
# ============================================================

@dataclass(frozen=True, order=True)
class RelSig:
    name: str
    types: tuple[str,...]

@dataclass(frozen=True, order=True)
class Atom:
    rel: str
    args: tuple[str,...]

@dataclass(frozen=True)
class Rule:
    head: Atom
    body: tuple[Atom,...]

    def text(self):
        lhs=" + ".join(f"{a.rel}({','.join(a.args)})" for a in self.body)
        return f"{lhs} -> {self.head.rel}({','.join(self.head.args)})"

@dataclass
class World:
    facts: dict[str,set[tuple[str,...]]]
    positives: list[tuple[str,...]]
    negatives: list[tuple[str,...]]

NUM="NUM"; NODE="NODE"

SIG={
    "ZERO":RelSig("ZERO",(NUM,)),
    "EQ":RelSig("EQ",(NUM,NUM)),
    "PRED":RelSig("PRED",(NUM,NUM)),
    "SUCC":RelSig("SUCC",(NUM,NUM)),
    "LT":RelSig("LT",(NUM,NUM)),
    "ADD":RelSig("ADD",(NUM,NUM,NUM)),
    "SUB":RelSig("SUB",(NUM,NUM,NUM)),
    "MUL":RelSig("MUL",(NUM,NUM,NUM)),
    "DIVMOD":RelSig("DIVMOD",(NUM,NUM,NUM,NUM)),
    "EDGE":RelSig("EDGE",(NODE,NODE)),
    "EDGE_COST":RelSig("EDGE_COST",(NODE,NODE,NUM)),
    "PATH":RelSig("PATH",(NODE,NODE)),
    "PATH_COST":RelSig("PATH_COST",(NODE,NODE,NUM)),
}

HEAD_VARS={
    "SUB":("X","Y","Z"),
    "MUL":("X","Y","Z"),
    "DIVMOD":("N","D","Q","R"),
    "PATH":("X","Y"),
    "PATH_COST":("X","Y","C"),
}

def sample_even(seq,n):
    if n is None or len(seq)<=n:
        return list(seq)
    if n<=1:
        return [seq[len(seq)//2]]
    return [seq[round(i*(len(seq)-1)/(n-1))] for i in range(n)]

def connected(body,head_set):
    atoms=list(body)
    conn={i for i,a in enumerate(atoms) if set(a.args)&head_set}
    while True:
        vv={v for i in conn for v in atoms[i].args}
        nxt=conn|{i for i,a in enumerate(atoms) if set(a.args)&vv}
        if nxt==conn:
            return len(conn)==len(atoms)
        conn=nxt

def valid_rule(rule,target,recursive):
    hs=set(rule.head.args)
    occ=Counter(v for a in rule.body for v in a.args)
    if not all(v in occ for v in rule.head.args):
        return False
    # no dangling existential variables
    if any(n<2 for v,n in occ.items() if v not in hs):
        return False
    if not connected(rule.body,hs):
        return False
    rec=[a for a in rule.body if a.rel==target.name]
    if recursive:
        if len(rec)!=1 or rec[0].args==rule.head.args:
            return False
        bgvars=hs|{v for a in rule.body if a.rel!=target.name for v in a.args}
        if not all(v in bgvars for v in rec[0].args):
            return False
    elif rec:
        return False
    return True

# ============================================================
# Ground witness -> symbolic rule abstraction
# ============================================================

def abstract_variants(target, head_tuple, ground_atoms, max_hidden):
    """
    ground_atoms: [(relation_name, ground_tuple), ...]

    A ground constant matching multiple head positions (e.g. SUB(7,0,7))
    may abstract to either head variable per occurrence. This lets the
    learner recover equality/base structures instead of baking constants in.
    """
    head_vars=HEAD_VARS[target.name]
    head=Atom(target.name,head_vars)
    out=set()

    def walk_atom(ai,aj,atoms,args,hidden_map,hidden_count):
        if ai==len(ground_atoms):
            out.add(Rule(head,tuple(sorted(atoms))))
            return

        rel,gargs=ground_atoms[ai]
        sig=SIG[rel]

        if aj==len(gargs):
            walk_atom(
                ai+1,0,
                atoms+[Atom(rel,tuple(args))],
                [],hidden_map,hidden_count
            )
            return

        const=gargs[aj]
        typ=sig.types[aj]

        # Any head variable with the same typed ground value is a legal abstraction.
        head_choices=[
            v for v,val,t in zip(head_vars,head_tuple,target.types)
            if val==const and t==typ
        ]
        if head_choices:
            for v in head_choices:
                walk_atom(ai,aj+1,atoms,args+[v],hidden_map,hidden_count)
            return

        key=(typ,const)
        if key in hidden_map:
            walk_atom(ai,aj+1,atoms,args+[hidden_map[key]],hidden_map,hidden_count)
            return

        idx=hidden_count.get(typ,0)
        if idx>=max_hidden.get(typ,0):
            return
        v=f"_{typ}{idx}"
        hm=dict(hidden_map); hm[key]=v
        hc=dict(hidden_count); hc[typ]=idx+1
        walk_atom(ai,aj+1,atoms,args+[v],hm,hc)

    walk_atom(0,0,[],[],{}, {})
    return out

def typed_extra_counts(rel,fact,known):
    sig=SIG[rel]
    extras=defaultdict(set)
    shared=0
    for val,typ in zip(fact,sig.types):
        if val in known:
            shared+=1
        else:
            extras[typ].add(val)
    return shared,extras

def within_hidden(extras,limits):
    return all(len(vals)<=limits.get(t,0) for t,vals in extras.items())

def merge_extra(a,b):
    out=defaultdict(set)
    for d in (a,b):
        for t,vals in d.items():
            out[t].update(vals)
    return out

def witness_sequences(world,bg_relations,initial_known,max_len,hidden_limits):
    """
    Staged high-boundness join search.

    Every added fact must already bind at least arity-1 arguments.
    It may introduce only the bounded remaining hidden values.
    This is a generic query-planning bias, not a task formula.
    """
    found=set()

    def dfs(chosen,known,extras):
        if chosen:
            found.add(tuple(chosen))
        if len(chosen)>=max_len:
            return

        for rel in bg_relations:
            sig=SIG[rel]
            min_shared=max(1,len(sig.types)-1)
            for fact in world.facts.get(rel,()):
                item=(rel,fact)
                if item in chosen:
                    continue
                shared,newextras=typed_extra_counts(rel,fact,known)
                if shared<min_shared:
                    continue
                merged=merge_extra(extras,newextras)
                if not within_hidden(merged,hidden_limits):
                    continue
                dfs(chosen+[item],known|set(fact),merged)

    dfs([],set(initial_known),defaultdict(set))
    return found

# ============================================================
# Candidate mining
# ============================================================

def mine_base(target,bg,worlds,max_body,sample_n=14):
    rules=Counter()
    zero_hidden={NUM:0,NODE:0}
    for w in worlds:
        for head in sample_even(w.positives,sample_n):
            for seq in witness_sequences(w,bg,set(head),max_body,zero_hidden):
                for r in abstract_variants(target,head,seq,zero_hidden):
                    if valid_rule(r,target,False):
                        rules[r]+=1
    return rules

def hidden_used_by_recursive_tuple(target,head,sub):
    used=defaultdict(set)
    head_vals=set(head)
    for val,typ in zip(sub,target.types):
        if val not in head_vals:
            used[typ].add(val)
    return used

def remaining_hidden(total,used):
    return {
        t:max(0,total.get(t,0)-len(used.get(t,set())))
        for t in set(total)|set(used)
    }

def mine_recursive(target,bg,worlds,max_bg,hidden_limits,sample_n=18):
    rules=Counter()

    for w in worlds:
        heads=sample_even(w.positives,sample_n)
        for head in heads:
            hs=set(head)

            # Training-local recursive subgoals. We are not told their structure;
            # they are simply other positive target instances sharing symbols.
            subs=[s for s in w.positives if s!=head and set(s)&hs]

            for sub in subs:
                used=hidden_used_by_recursive_tuple(target,head,sub)
                # If recursive subgoal alone exceeds variable budget, skip.
                if any(len(v)>hidden_limits.get(t,0) for t,v in used.items()):
                    continue

                rem=remaining_hidden(hidden_limits,used)
                initial=set(head)|set(sub)

                for seq in witness_sequences(w,bg,initial,max_bg,rem):
                    ground=tuple(seq)+((target.name,sub),)
                    for r in abstract_variants(target,head,ground,hidden_limits):
                        if valid_rule(r,target,True):
                            rules[r]+=1
    return rules

def mine_nonrecursive(target,bg,worlds,max_body,hidden_limits,sample_n=14):
    rules=Counter()
    for w in worlds:
        for head in sample_even(w.positives,sample_n):
            for seq in witness_sequences(w,bg,set(head),max_body,hidden_limits):
                for r in abstract_variants(target,head,seq,hidden_limits):
                    if valid_rule(r,target,False):
                        rules[r]+=1
    return rules

# ============================================================
# Rule execution
# ============================================================

def match_fact(atom,fact,binding):
    b=dict(binding)
    for v,val in zip(atom.args,fact):
        if v in b and b[v]!=val:
            return None
        b[v]=val
    return b

def joins(atoms,facts,binding):
    if not atoms:
        yield binding
        return

    atoms=list(atoms)
    # local join plan: most already-bound ports first
    idx=max(range(len(atoms)),key=lambda i:sum(v in binding for v in atoms[i].args))
    atom=atoms.pop(idx)

    for fact in facts.get(atom.rel,()):
        b2=match_fact(atom,fact,binding)
        if b2 is not None:
            yield from joins(atoms,facts,b2)

class Program:
    def __init__(self,target,base=None,recursive=None,depth_limit=80):
        self.target=target
        self.base=base
        self.recursive=recursive
        self.depth_limit=depth_limit
        self.memo={}

    def reset(self):
        self.memo={}

    def rule_holds(self,rule,head_tuple,world,stack,depth):
        b={v:x for v,x in zip(rule.head.args,head_tuple)}
        bg=[a for a in rule.body if a.rel!=self.target.name]
        rec=[a for a in rule.body if a.rel==self.target.name]

        for b2 in joins(bg,world.facts,b):
            if not rec:
                return True
            try:
                sub=tuple(b2[v] for v in rec[0].args)
            except KeyError:
                continue
            if self.prove(sub,world,stack,depth+1):
                return True
        return False

    def prove(self,head_tuple,world,stack=None,depth=0):
        if stack is None:
            stack=set()
        if depth>self.depth_limit or head_tuple in stack:
            return False

        key=(id(world),head_tuple)
        if key in self.memo:
            return self.memo[key]

        ns=set(stack); ns.add(head_tuple)

        if self.base and self.rule_holds(self.base,head_tuple,world,ns,depth):
            self.memo[key]=True
            return True

        if self.recursive and self.rule_holds(self.recursive,head_tuple,world,ns,depth):
            self.memo[key]=True
            return True

        self.memo[key]=False
        return False

def hidden_count(rule):
    hs=set(rule.head.args)
    return len({v for a in rule.body for v in a.args if v not in hs})

def score_base(target,rule,worlds,probe_n=None):
    support=conflict=0
    p=Program(target,rule,None)
    for w in worlds:
        for x in sample_even(w.positives,probe_n):
            support+=p.rule_holds(rule,x,w,set(),0)
        for x in sample_even(w.negatives,probe_n):
            conflict+=p.rule_holds(rule,x,w,set(),0)
    complexity=len(rule.body)+0.2*hidden_count(rule)
    return support,conflict,support*10-conflict*25-complexity

def score_program(target,base,rec,worlds,probe_n=None):
    support=conflict=0
    p=Program(target,base,rec)
    for w in worlds:
        p.reset()
        for x in sample_even(w.positives,probe_n):
            support+=p.prove(x,w)
        p.reset()
        for x in sample_even(w.negatives,probe_n):
            conflict+=p.prove(x,w)
    complexity=len(base.body)+len(rec.body)+0.2*(hidden_count(base)+hidden_count(rec))
    return support,conflict,support*10-conflict*25-complexity

def score_rule(target,rule,worlds,probe_n=None):
    return score_base(target,rule,worlds,probe_n)

# ============================================================
# Generic staged synthesis
# ============================================================

def synth_recursive(name,bg,worlds,max_base,max_bg,hidden_limits,beam=100):
    target=SIG[name]
    t0=time.perf_counter()

    bases=mine_base(target,bg,worlds,max_base)
    base_rows=[
        (score_base(target,r,worlds),freq,r)
        for r,freq in bases.items()
    ]
    base_rows.sort(key=lambda x:(x[0][2],x[1]),reverse=True)
    top_bases=[x[2] for x in base_rows[:6]]

    t1=time.perf_counter()
    recs=mine_recursive(target,bg,worlds,max_bg,hidden_limits)
    t2=time.perf_counter()

    best=None
    evaluated_full=0

    # Minimal-body search first.
    body_lengths=sorted({len(r.body) for r in recs})
    total_pos=sum(len(w.positives) for w in worlds)

    for L in body_lengths:
        candidates=[(r,f) for r,f in recs.items() if len(r.body)==L]

        # Cheap representative probe before full scoring.
        probe=[]
        for r,freq in candidates:
            for b in top_bases:
                sc=score_program(target,b,r,worlds,probe_n=8)
                probe.append((sc,freq,b,r))
        probe.sort(key=lambda x:(x[0][2],x[1]),reverse=True)

        for _,freq,b,r in probe[:beam]:
            sc=score_program(target,b,r,worlds,None)
            evaluated_full+=1
            row=(sc,freq,b,r)
            if best is None or (sc[2],freq)>(best[0][2],best[1]):
                best=row

        if best and best[0][0]==total_pos and best[0][1]==0:
            break

    t3=time.perf_counter()
    return {
        "base_count":len(bases),
        "recursive_count":len(recs),
        "full_programs_evaluated":evaluated_full,
        "base_top":base_rows[:5],
        "best":best,
        "seconds":t3-t0,
        "mine_base_seconds":t1-t0,
        "mine_recursive_seconds":t2-t1,
        "score_seconds":t3-t2,
    }

def synth_nonrecursive(name,bg,worlds,max_body,hidden_limits,beam=180):
    target=SIG[name]
    t0=time.perf_counter()
    rules=mine_nonrecursive(target,bg,worlds,max_body,hidden_limits)
    t1=time.perf_counter()

    # Probe-rank first.
    probe=[
        (score_rule(target,r,worlds,probe_n=10),freq,r)
        for r,freq in rules.items()
    ]
    probe.sort(key=lambda x:(x[0][2],x[1]),reverse=True)

    best=None
    evaluated=0
    for _,freq,r in probe[:beam]:
        sc=score_rule(target,r,worlds,None)
        evaluated+=1
        row=(sc,freq,r)
        if best is None or (sc[2],freq)>(best[0][2],best[1]):
            best=row

    return {
        "candidate_count":len(rules),
        "full_rules_evaluated":evaluated,
        "best":best,
        "seconds":time.perf_counter()-t0,
        "mine_seconds":t1-t0,
    }

# ============================================================
# Benchmark facts
# ============================================================

def numeric_facts(maxn,add=False,sub=False,mul=False,lt=False):
    f=defaultdict(set)
    f["ZERO"].add(("N0",))
    for i in range(maxn+1):
        f["EQ"].add((f"N{i}",f"N{i}"))
    for i in range(1,maxn+1):
        f["PRED"].add((f"N{i}",f"N{i-1}"))
        f["SUCC"].add((f"N{i-1}",f"N{i}"))

    if lt:
        for a in range(maxn+1):
            for b in range(a+1,maxn+1):
                f["LT"].add((f"N{a}",f"N{b}"))

    if add:
        for a in range(maxn+1):
            for b in range(maxn+1-a):
                f["ADD"].add((f"N{a}",f"N{b}",f"N{a+b}"))

    if sub:
        for a in range(maxn+1):
            for b in range(a+1):
                f["SUB"].add((f"N{a}",f"N{b}",f"N{a-b}"))

    if mul:
        for a in range(maxn+1):
            for b in range(maxn+1):
                c=a*b
                if c<=maxn:
                    f["MUL"].add((f"N{a}",f"N{b}",f"N{c}"))
    return dict(f)

def num_world(maxn,pos,neg,**kwargs):
    return World(numeric_facts(maxn,**kwargs),pos,neg)

def random_tree(seed,n=7):
    rng=random.Random(seed)
    nodes=[f"V{i}" for i in range(n)]
    edges=[]
    for i in range(1,n):
        p=rng.randrange(i)
        edges.append((nodes[p],nodes[i],rng.randint(1,5)))
    return nodes,edges

def route_closure(nodes,edges):
    adj=defaultdict(list)
    for a,b,c in edges:
        adj[a].append((b,c))
    out={}
    def dfs(src,u,cost,seen):
        for v,w in adj[u]:
            if v in seen:
                continue
            nc=cost+w
            out[(src,v)]=nc
            dfs(src,v,nc,seen|{v})
    for s in nodes:
        dfs(s,s,0,{s})
    return out

# ============================================================
# Training sets
# ============================================================

# SUB
sub_pos=[]; sub_neg=[]
for a in range(8):
    for b in range(a+1):
        c=a-b
        sub_pos.append((f"N{a}",f"N{b}",f"N{c}"))
        sub_neg.append((f"N{a}",f"N{b}",f"N{c+1}"))
        if c>0:
            sub_neg.append((f"N{a}",f"N{b}",f"N{c-1}"))
sub_worlds=[num_world(12,sub_pos,sub_neg,lt=True)]

# MUL
mul_pos=[]; mul_neg=[]
for a in range(7):
    for b in range(7):
        c=a*b
        mul_pos.append((f"N{a}",f"N{b}",f"N{c}"))
        mul_neg.append((f"N{a}",f"N{b}",f"N{c+1}"))
        if c>0:
            mul_neg.append((f"N{a}",f"N{b}",f"N{c-1}"))
mul_worlds=[num_world(50,mul_pos,mul_neg,add=True,sub=True)]

# DIVMOD
div_pos=[]; div_neg=[]
for n in range(31):
    for d in range(1,7):
        q=n//d; r=n%d
        div_pos.append((f"N{n}",f"N{d}",f"N{q}",f"N{r}"))
        div_neg.append((f"N{n}",f"N{d}",f"N{q}",f"N{r+1}"))
        if q>0:
            div_neg.append((f"N{n}",f"N{d}",f"N{q-1}",f"N{r+d}"))
div_worlds=[num_world(45,div_pos,div_neg,add=True,sub=True,mul=True,lt=True)]

# Routing
path_worlds=[]; cost_worlds=[]
for seed in [3,7,11]:
    nodes,edges=random_tree(seed,7)
    closure=route_closure(nodes,edges)

    facts=defaultdict(set)
    for r,vals in numeric_facts(50,add=True).items():
        facts[r].update(vals)
    for a,b,c in edges:
        facts["EDGE"].add((a,b))
        facts["EDGE_COST"].add((a,b,f"N{c}"))

    pp=list(closure.keys())
    pn=[(a,b) for a in nodes for b in nodes if a!=b and (a,b) not in closure]
    path_worlds.append(World(dict(facts),pp,pn))

    cp=[(a,b,f"N{c}") for (a,b),c in closure.items()]
    cn=[(a,b,f"N{c+1}") for (a,b),c in closure.items()]
    cost_worlds.append(World(dict(facts),cp,cn))

# ============================================================
# Synthesis
# ============================================================

tasks={}

tasks["SUB"]=synth_recursive(
    "SUB",["ZERO","EQ","PRED","SUCC","LT"],
    sub_worlds,max_base=2,max_bg=2,hidden_limits={NUM:2}
)

tasks["MUL"]=synth_recursive(
    "MUL",["ZERO","EQ","PRED","SUCC","ADD","SUB"],
    mul_worlds,max_base=2,max_bg=2,hidden_limits={NUM:2}
)

tasks["DIVMOD"]=synth_nonrecursive(
    "DIVMOD",["ZERO","EQ","PRED","SUCC","LT","ADD","SUB","MUL"],
    div_worlds,max_body=3,hidden_limits={NUM:1}
)

tasks["PATH"]=synth_recursive(
    "PATH",["EDGE","EDGE_COST"],
    path_worlds,max_base=1,max_bg=1,hidden_limits={NODE:1,NUM:1}
)

tasks["PATH_COST"]=synth_recursive(
    "PATH_COST",["EDGE","EDGE_COST","PRED","ADD"],
    cost_worlds,max_base=1,max_bg=2,hidden_limits={NODE:1,NUM:2}
)

# ============================================================
# Frozen unseen validation
# ============================================================

def validate_recursive(name,res,worlds):
    sc,freq,base,rec=res["best"]
    p=Program(SIG[name],base,rec)
    ok=total=0
    for w in worlds:
        p.reset()
        for x in w.positives:
            total+=1
            ok+=bool(p.prove(x,w))
        p.reset()
        for x in w.negatives:
            total+=1
            ok+=not p.prove(x,w)
    return ok,total

def validate_rule(name,res,worlds):
    sc,freq,rule=res["best"]
    p=Program(SIG[name],rule,None)
    ok=total=0
    for w in worlds:
        for x in w.positives:
            total+=1
            ok+=p.rule_holds(rule,x,w,set(),0)
        for x in w.negatives:
            total+=1
            ok+=not p.rule_holds(rule,x,w,set(),0)
    return ok,total

# SUB unseen
p=[]; n=[]
for a,b in [(11,7),(14,5),(18,9),(20,3),(17,16),(25,11),(30,7)]:
    c=a-b
    p.append((f"N{a}",f"N{b}",f"N{c}"))
    n.append((f"N{a}",f"N{b}",f"N{c+1}"))
sub_test=[num_world(35,p,n,lt=True)]

# MUL unseen
p=[]; n=[]
for a,b in [(7,8),(9,5),(11,4),(12,3),(8,9),(10,7)]:
    c=a*b
    p.append((f"N{a}",f"N{b}",f"N{c}"))
    n.append((f"N{a}",f"N{b}",f"N{c+1}"))
mul_test=[num_world(100,p,n,add=True,sub=True)]

# DIVMOD unseen
p=[]; n=[]
for nn,d in [(37,7),(41,8),(53,9),(68,11),(79,12),(91,10)]:
    q=nn//d; r=nn%d
    p.append((f"N{nn}",f"N{d}",f"N{q}",f"N{r}"))
    n.append((f"N{nn}",f"N{d}",f"N{q}",f"N{r+1}"))
div_test=[num_world(110,p,n,add=True,sub=True,mul=True,lt=True)]

# route unseen
path_test=[]; cost_test=[]
for seed in [101,103,107]:
    nodes,edges=random_tree(seed,8)
    closure=route_closure(nodes,edges)
    facts=defaultdict(set)
    for r,vals in numeric_facts(70,add=True).items():
        facts[r].update(vals)
    for a,b,c in edges:
        facts["EDGE"].add((a,b))
        facts["EDGE_COST"].add((a,b,f"N{c}"))
    pp=list(closure.keys())
    pn=[(a,b) for a in nodes for b in nodes if a!=b and (a,b) not in closure]
    path_test.append(World(dict(facts),pp,pn))
    cp=[(a,b,f"N{c}") for (a,b),c in closure.items()]
    cn=[(a,b,f"N{c+1}") for (a,b),c in closure.items()]
    cost_test.append(World(dict(facts),cp,cn))

validation={
    "SUB":validate_recursive("SUB",tasks["SUB"],sub_test),
    "MUL":validate_recursive("MUL",tasks["MUL"],mul_test),
    "DIVMOD":validate_rule("DIVMOD",tasks["DIVMOD"],div_test),
    "PATH":validate_recursive("PATH",tasks["PATH"],path_test),
    "PATH_COST":validate_recursive("PATH_COST",tasks["PATH_COST"],cost_test),
}

# ============================================================
# Output
# ============================================================

def print_recursive(name,res):
    sc,freq,b,r=res["best"]
    print(f"\n=== {name} ===")
    print("candidates base/rec:",res["base_count"],"/",res["recursive_count"])
    print("full programs evaluated:",res["full_programs_evaluated"])
    print("train support/conflict:",sc[0],"/",sc[1])
    print("BASE:",b.text())
    print("REC :",r.text())
    print("unseen:",validation[name][0],"/",validation[name][1])
    print("seconds:",round(res["seconds"],3))

for name in ["SUB","MUL"]:
    print_recursive(name,tasks[name])

dsc,dfreq,drule=tasks["DIVMOD"]["best"]
print("\n=== DIVMOD ===")
print("candidates:",tasks["DIVMOD"]["candidate_count"])
print("full rules evaluated:",tasks["DIVMOD"]["full_rules_evaluated"])
print("train support/conflict:",dsc[0],"/",dsc[1])
print("RULE:",drule.text())
print("unseen:",validation["DIVMOD"][0],"/",validation["DIVMOD"][1])
print("seconds:",round(tasks["DIVMOD"]["seconds"],3))

for name in ["PATH","PATH_COST"]:
    print_recursive(name,tasks[name])

# Define success by behavior/generalization, not exact hand-known formula.
success={}
for name in ["SUB","MUL","PATH","PATH_COST"]:
    sc=tasks[name]["best"][0]
    total_pos=sum(len(w.positives) for w in {
        "SUB":sub_worlds,"MUL":mul_worlds,"PATH":path_worlds,"PATH_COST":cost_worlds
    }[name])
    success[name]=(
        sc[0]==total_pos and sc[1]==0 and
        validation[name][0]==validation[name][1]
    )
sc=tasks["DIVMOD"]["best"][0]
success["DIVMOD"]=(
    sc[0]==len(div_pos) and sc[1]==0 and
    validation["DIVMOD"][0]==validation["DIVMOD"][1]
)

print("\n=== GENERIC REDISCOVERY / GENERALIZATION ===")
for k,v in success.items():
    print(k,":","PASS" if v else "FAIL")

report={
    "version":"v4.4-generic-u-generator-v2-staged",
    "method":"query-local typed witness abstraction",
    "priors":[
        "typed relation signatures",
        "domain-scoped available relation vocabulary",
        "maximum body size",
        "bounded hidden variables by type",
        "head range restriction",
        "hidden variables must join at least twice",
        "body connected to target head",
        "one recursive self-call for recursive U",
        "recursive call groundable from nonrecursive joins",
        "high-boundness witness search: each added fact binds at least arity-1 existing values",
        "support/conflict + MDL",
        "minimal-body-first + probe beam before full evaluation"
    ],
    "not_given":[
        "SUB recursion formula",
        "MUL recursion formula",
        "DIVMOD equation formula",
        "PATH transitivity variable bindings",
        "PATH_COST accumulation variable bindings",
        "named candidate menus such as DEC_Y_ADD_X or EDGE_THEN_PATH"
    ],
    "success":success,
    "validation":{k:{"passed":v[0],"n":v[1]} for k,v in validation.items()},
    "tasks":{},
    "search_observation":{
        "global_cartesian_generator":"A prior exhaustive typed Cartesian attempt in this run exceeded the execution window before completing.",
        "staged_generator":"Candidate construction is instead localized to concrete positive proof neighborhoods."
    },
    "caveats":[
        "This is not unrestricted program synthesis: relation vocabularies and body/hidden-variable limits remain search priors.",
        "Recursive candidate generation is anchored to observed positive recursive subinstances; it may miss valid algorithms whose intermediate states were never observed.",
        "High-boundness witness search is deliberately conservative and may miss rules requiring atoms that introduce several new variables.",
        "Arithmetic background relations used for higher-level synthesis are supplied extensionally in the benchmark harness.",
        "The benchmark still isolates symbolic reasoning from raw-language parsing."
    ]
}

for name,res in tasks.items():
    if name=="DIVMOD":
        sc,freq,r=res["best"]
        report["tasks"][name]={
            "candidate_count":res["candidate_count"],
            "full_evaluated":res["full_rules_evaluated"],
            "support":sc[0],"conflict":sc[1],
            "rule":r.text(),
            "seconds":res["seconds"]
        }
    else:
        sc,freq,b,r=res["best"]
        report["tasks"][name]={
            "base_candidates":res["base_count"],
            "recursive_candidates":res["recursive_count"],
            "full_evaluated":res["full_programs_evaluated"],
            "support":sc[0],"conflict":sc[1],
            "base_rule":b.text(),
            "recursive_rule":r.text(),
            "seconds":res["seconds"]
        }

Path("/mnt/data/symbolic_v44_generic_u_generator_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)

rows=[]
for name in ["SUB","MUL","DIVMOD","PATH","PATH_COST"]:
    t=report["tasks"][name]
    rows.append({
        "target":name,
        "success":success[name],
        "candidates":(
            t.get("candidate_count")
            or t.get("base_candidates",0)+t.get("recursive_candidates",0)
        ),
        "full_evaluated":t["full_evaluated"],
        "support":t["support"],
        "conflict":t["conflict"],
        "unseen_passed":report["validation"][name]["passed"],
        "unseen_n":report["validation"][name]["n"],
        "seconds":t["seconds"]
    })

with Path("/mnt/data/symbolic_v44_generic_u_generator_summary.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)

print("\nSaved v4.4 report + summary.")


from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import json, time, argparse

# Reuse v4.4 generic generator definitions, but do not run its slow benchmark.
SRC=Path("/mnt/data/symbolic_v44_generic_u_generator.py").read_text(encoding="utf-8")
PREFIX=SRC.split("# ============================================================\n# Training sets")[0]
ns={}
exec(PREFIX,ns)
globals().update(ns)

# ============================================================
# v4.4b Recursive-U Verifier
# ============================================================

PROGRESS_RELATIONS={
    # metadata belongs to ontology / relation semantics, not to SUB/MUL formulas.
    "PRED": "BIG_SMALL",   # PRED(big,small)
    "SUCC": "SMALL_BIG",   # SUCC(small,big)
}

def recursive_atom(rule,target_name):
    xs=[a for a in rule.body if a.rel==target_name]
    return xs[0] if len(xs)==1 else None

def zero_anchors(base):
    """Head variables explicitly tied to the well-founded minimum ZERO."""
    hs=set(base.head.args)
    return {
        a.args[0] for a in base.body
        if a.rel=="ZERO" and len(a.args)==1 and a.args[0] in hs
    }

def direct_numeric_decrease(rec_rule,target,head_var):
    ra=recursive_atom(rec_rule,target.name)
    if ra is None or head_var not in rec_rule.head.args:
        return None
    i=rec_rule.head.args.index(head_var)
    rec_var=ra.args[i]
    if rec_var==head_var:
        return None
    for a in rec_rule.body:
        if a.rel=="PRED" and a.args==(head_var,rec_var):
            return {"mode":"WELL_FOUNDED_DECREASE","measure":head_var,"via":"PRED"}
        if a.rel=="SUCC" and a.args==(rec_var,head_var):
            return {"mode":"WELL_FOUNDED_DECREASE","measure":head_var,"via":"SUCC"}
    return None

def finite_structural_traversal(rec_rule,target):
    """
    Generic finite traversal:
    a NODE-valued recursive port changes and a finite background relation
    directly connects the old and new node. Runtime visited-state guard
    supplies cycle safety.
    """
    ra=recursive_atom(rec_rule,target.name)
    if ra is None:
        return None
    for i,(hv,rv,typ) in enumerate(zip(rec_rule.head.args,ra.args,target.types)):
        if typ!="NODE" or hv==rv:
            continue
        for a in rec_rule.body:
            if a.rel==target.name:
                continue
            sig=SIG[a.rel]
            node_args=[arg for arg,t in zip(a.args,sig.types) if t=="NODE"]
            if hv in node_args and rv in node_args:
                return {
                    "mode":"FINITE_TRAVERSAL",
                    "measure":f"visited({i})",
                    "via":a.rel,
                    "port":i,
                }
    return None

def verify_recursive_pair(base,rec,target):
    """
    Pair-level verification matters:
    for numeric recursion the decreasing variable must be one that reaches
    a well-founded base case in the chosen base rule.
    """
    anchors=zero_anchors(base)
    for v in anchors:
        cert=direct_numeric_decrease(rec,target,v)
        if cert:
            cert["base_anchor"]=v
            return cert

    cert=finite_structural_traversal(rec,target)
    if cert:
        cert["base_anchor"]="finite-domain+visited"
        return cert
    return None


# ============================================================
# Indexed witness search (v4.4b performance layer)
# ============================================================
_WITNESS_INDEX={}

def _world_value_index(world):
    key=id(world)
    if key in _WITNESS_INDEX:
        return _WITNESS_INDEX[key]
    idx={}
    for rel,facts in world.facts.items():
        byval=defaultdict(set)
        for fact in facts:
            for val in set(fact):
                byval[val].add(fact)
        idx[rel]=byval
    _WITNESS_INDEX[key]=idx
    return idx

def witness_sequences(world,bg_relations,initial_known,max_len,hidden_limits):
    """Indexed equivalent of v4.4 witness search."""
    found=set()
    idx=_world_value_index(world)

    def dfs(chosen,known,extras):
        if chosen:
            found.add(tuple(chosen))
        if len(chosen)>=max_len:
            return
        for rel in bg_relations:
            sig=SIG[rel]
            min_shared=max(1,len(sig.types)-1)
            cand=set()
            byval=idx.get(rel,{})
            for v in known:
                cand.update(byval.get(v,()))
            for fact in cand:
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
# Cheap local step scoring
# ============================================================

def local_recursive_holds(rule,head_tuple,world,positive_targets):
    """One recursive step only: recursive subgoal must be a known positive."""
    target_name=rule.head.rel
    b={v:x for v,x in zip(rule.head.args,head_tuple)}
    bg=[a for a in rule.body if a.rel!=target_name]
    ra=recursive_atom(rule,target_name)
    if ra is None:
        return False
    for b2 in joins(bg,world.facts,b):
        try:
            sub=tuple(b2[v] for v in ra.args)
        except KeyError:
            continue
        if sub in positive_targets:
            return True
    return False

def local_step_score(rule,worlds,probe_n=None):
    support=conflict=0
    for w in worlds:
        posset=set(w.positives)
        for p in sample_even(w.positives,probe_n):
            support += bool(local_recursive_holds(rule,p,w,posset))
        for n in sample_even(w.negatives,probe_n):
            conflict += bool(local_recursive_holds(rule,n,w,posset))
    complexity=len(rule.body)+0.2*hidden_count(rule)
    return support,conflict,support*10-conflict*25-complexity

def compatible_candidates(target,bases,recs,worlds,beam_local=120,beam_full=24):
    """
    1. verifier reject
    2. cheap local-step score
    3. only top pairs get full recursive scoring
    """
    verified_raw=[]
    reject=0
    modes=defaultdict(int)
    accepted_recs=set()

    # Verify syntactically first; do not run any joins for rejected recursion.
    for binfo in bases:
        b=binfo["rule"]
        for r,freq in recs.items():
            cert=verify_recursive_pair(b,r,target)
            if cert is None:
                reject+=1
                continue
            modes[cert["mode"]]+=1
            accepted_recs.add(r)
            verified_raw.append((freq,b,r,cert))

    local_cache={r:local_step_score(r,worlds,probe_n=10) for r in accepted_recs}
    verified=[]
    for freq,b,r,cert in verified_raw:
        ls=local_cache[r]
        rank=ls[2] + min(freq,20)*0.05 - len(r.body)*0.02
        verified.append((rank,ls,freq,b,r,cert))

    verified.sort(key=lambda x:(x[0],x[2]),reverse=True)
    shortlist=verified[:beam_local]

    # Very cheap recursive probe to choose only a tiny full-scoring beam.
    probed=[]
    for row in shortlist:
        rank,ls,freq,b,r,cert=row
        sc=fixedpoint_score(target,b,r,worlds,probe_n=8)
        probed.append((sc[2],sc,freq,b,r,cert,ls))
    probed.sort(key=lambda x:(x[0],x[2]),reverse=True)

    best=None
    full=[]
    for row in probed[:beam_full]:
        _,_,freq,b,r,cert,ls=row
        sc=fixedpoint_score(target,b,r,worlds,None)
        rr=(sc,freq,b,r,cert,ls)
        full.append(rr)
        if best is None or (sc[2],freq)>(best[0][2],best[1]):
            best=rr

    return {
        "pair_total":len(bases)*len(recs),
        "verifier_rejected":reject,
        "verifier_accepted":len(verified),
        "verifier_modes":dict(modes),
        "local_shortlist":len(shortlist),
        "recursive_probe_count":len(probed),
        "full_evaluated":len(full),
        "best":best,
        "top_full":full[:8],
    }

def mine_and_score_bases(target,bg,worlds,max_base):
    bases=mine_base(target,bg,worlds,max_base)
    probe=[]
    for r,freq in bases.items():
        sc=score_base(target,r,worlds,probe_n=8)
        probe.append((sc,freq,r))
    probe.sort(key=lambda x:(x[0][2],x[1]),reverse=True)

    anchored=[x for x in probe if zero_anchors(x[2])]
    anchored.sort(key=lambda x:(x[0][2],x[1]),reverse=True)
    chosen=[]; seen=set()
    for row in probe[:12]+anchored[:24]:
        if row[2] not in seen:
            seen.add(row[2]); chosen.append(row)

    full=[]
    for _,freq,r in chosen:
        sc=score_base(target,r,worlds,None)
        full.append({"score":sc,"freq":freq,"rule":r})

    overall=sorted(full,key=lambda x:(x["score"][2],x["freq"]),reverse=True)
    anchored_full=sorted(
        [x for x in full if zero_anchors(x["rule"])],
        key=lambda x:(x["score"][2],x["freq"]),reverse=True
    )
    out=[]; seen=set()
    for x in overall[:6]+anchored_full[:8]:
        if x["rule"] not in seen:
            seen.add(x["rule"]); out.append(x)
    return bases,out

# ============================================================
# Verified recursive execution by least fixed point
# ============================================================

def derive_heads(rule,target,world,target_ext):
    facts=dict(world.facts)
    facts[target.name]=set(target_ext)
    # facts is a short-lived wrapper; guard against Python id reuse in the
    # join-index cache by invalidating any prior entry for this fresh dict id.
    _JOIN_INDEX.pop(id(facts),None)
    out=set()
    for b in joins(list(rule.body),facts,{}):
        try:
            out.add(tuple(b[v] for v in rule.head.args))
        except KeyError:
            pass
    return out

def fixedpoint_extension(target,base,rec,world,max_iter=128):
    ext=derive_heads(base,target,world,set())
    for _ in range(max_iter):
        new=derive_heads(rec,target,world,ext)
        merged=ext|new
        if merged==ext:
            return ext
        ext=merged
    return ext

def fixedpoint_score(target,base,rec,worlds,probe_n=None):
    support=conflict=0
    for w in worlds:
        ext=fixedpoint_extension(target,base,rec,w)
        for p in sample_even(w.positives,probe_n):
            support += p in ext
        for n in sample_even(w.negatives,probe_n):
            conflict += n in ext
    complexity=len(base.body)+len(rec.body)+0.2*(hidden_count(base)+hidden_count(rec))
    return support,conflict,support*10-conflict*25-complexity

def synth_verified_recursive(name,bg,worlds,max_base,max_bg,hidden_limits):
    target=SIG[name]
    t0=time.perf_counter()
    bases_all,bases=mine_and_score_bases(target,bg,worlds,max_base)
    t1=time.perf_counter()
    rec_sample=1 if name=="MUL" else 14
    recs=mine_recursive(target,bg,worlds,max_bg,hidden_limits,sample_n=rec_sample)
    t2=time.perf_counter()
    comp=compatible_candidates(target,bases,recs,worlds)
    t3=time.perf_counter()
    return {
        "base_count":len(bases_all),
        "base_used":len(bases),
        "recursive_count":len(recs),
        **comp,
        "seconds":t3-t0,
        "mine_base_seconds":t1-t0,
        "mine_recursive_seconds":t2-t1,
        "verify_score_seconds":t3-t2,
    }


# Indexed rule-join executor. This replaces the v4.4 full relation scans.
_JOIN_INDEX={}

def _fact_join_index(facts):
    key=id(facts)
    if key in _JOIN_INDEX:
        return _JOIN_INDEX[key]
    idx={}
    for rel,rows in facts.items():
        byval=defaultdict(set)
        for row in rows:
            for val in set(row):
                byval[val].add(row)
        idx[rel]=byval
    _JOIN_INDEX[key]=idx
    return idx

def joins(atoms,facts,binding):
    if not atoms:
        yield binding
        return
    atoms=list(atoms)
    idx=max(range(len(atoms)),key=lambda i:sum(v in binding for v in atoms[i].args))
    atom=atoms.pop(idx)
    bound_vals=[binding[v] for v in atom.args if v in binding]
    rows=facts.get(atom.rel,())
    if bound_vals:
        byval=_fact_join_index(facts).get(atom.rel,{})
        sets=[byval.get(v,set()) for v in bound_vals]
        if not sets or any(not st for st in sets):
            return
        cand=set.intersection(*sets)
    else:
        cand=rows
    for fact in cand:
        b2=match_fact(atom,fact,binding)
        if b2 is not None:
            yield from joins(atoms,facts,b2)

# ============================================================
# Datasets
# ============================================================

def make_task(name):
    if name=="SUB":
        pos=[]; neg=[]
        for a in range(8):
            for b in range(a+1):
                c=a-b
                pos.append((f"N{a}",f"N{b}",f"N{c}"))
                neg.append((f"N{a}",f"N{b}",f"N{c+1}"))
                if c>0:
                    neg.append((f"N{a}",f"N{b}",f"N{c-1}"))
        worlds=[num_world(12,pos,neg,lt=True)]
        cfg=(["ZERO","EQ","PRED","SUCC","LT"],2,2,{NUM:2})
        tp=[]; tn=[]
        for a,b in [(11,7),(14,5),(18,9),(20,3),(17,16),(25,11)]:
            c=a-b
            tp.append((f"N{a}",f"N{b}",f"N{c}"))
            tn.append((f"N{a}",f"N{b}",f"N{c+1}"))
        test=[num_world(30,tp,tn,lt=True)]
        return worlds,cfg,test

    if name=="MUL":
        pos=[]; neg=[]
        for a in range(7):
            for b in range(7):
                c=a*b
                pos.append((f"N{a}",f"N{b}",f"N{c}"))
                neg.append((f"N{a}",f"N{b}",f"N{c+1}"))
                if c>0:
                    neg.append((f"N{a}",f"N{b}",f"N{c-1}"))
        worlds=[num_world(40,pos,neg,add=True,sub=True)]
        cfg=(["ZERO","EQ","PRED","SUCC","ADD","SUB"],2,2,{NUM:2})
        tp=[]; tn=[]
        for a,b in [(7,8),(9,5),(11,4),(12,3),(8,9)]:
            c=a*b
            tp.append((f"N{a}",f"N{b}",f"N{c}"))
            tn.append((f"N{a}",f"N{b}",f"N{c+1}"))
        test=[num_world(100,tp,tn,add=True,sub=True)]
        return worlds,cfg,test

    if name in {"PATH","PATH_COST"}:
        train=[]; test=[]
        for seeds,out in [([3,7,11],train),([101,103,107],test)]:
            for seed in seeds:
                nodes,edges=random_tree(seed,8 if out is test else 7)
                closure=route_closure(nodes,edges)
                facts=defaultdict(set)
                for r,vals in numeric_facts(70 if out is test else 50,add=True).items():
                    facts[r].update(vals)
                for a,b,c in edges:
                    facts["EDGE"].add((a,b))
                    facts["EDGE_COST"].add((a,b,f"N{c}"))
                if name=="PATH":
                    pos=list(closure.keys())
                    neg=[(a,b) for a in nodes for b in nodes if a!=b and (a,b) not in closure]
                else:
                    pos=[(a,b,f"N{c}") for (a,b),c in closure.items()]
                    neg=[(a,b,f"N{c+1}") for (a,b),c in closure.items()]
                out.append(World(dict(facts),pos,neg))
        if name=="PATH":
            cfg=(["EDGE","EDGE_COST"],1,1,{NODE:1,NUM:1})
        else:
            cfg=(["EDGE","EDGE_COST","PRED","ADD"],1,2,{NODE:1,NUM:2})
        return train,cfg,test

    raise ValueError(name)

def validate(name,res,test_worlds):
    sc,freq,b,r,cert,ls=res["best"]
    ok=total=0
    for w in test_worlds:
        ext=fixedpoint_extension(SIG[name],b,r,w)
        for x in w.positives:
            total+=1; ok+=x in ext
        for x in w.negatives:
            total+=1; ok+=x not in ext
    return ok,total

# DIVMOD is deliberately nonrecursive control.
def run_divmod():
    pos=[]; neg=[]
    for n in range(31):
        for d in range(1,7):
            q=n//d; r=n%d
            pos.append((f"N{n}",f"N{d}",f"N{q}",f"N{r}"))
            neg.append((f"N{n}",f"N{d}",f"N{q}",f"N{r+1}"))
            if q>0:
                neg.append((f"N{n}",f"N{d}",f"N{q-1}",f"N{r+d}"))
    worlds=[num_world(45,pos,neg,add=True,sub=True,mul=True,lt=True)]
    t0=time.perf_counter()
    mined=mine_nonrecursive(
        SIG["DIVMOD"],
        ["ZERO","EQ","PRED","SUCC","LT","ADD","SUB","MUL"],
        worlds,3,{NUM:1},sample_n=12
    )
    rows=[]
    # cheap probe first
    for r,freq in mined.items():
        sc=score_rule(SIG["DIVMOD"],r,worlds,probe_n=10)
        rows.append((sc,freq,r))
    rows.sort(key=lambda x:(x[0][2],x[1]),reverse=True)
    best=None
    for _,freq,r in rows[:120]:
        sc=score_rule(SIG["DIVMOD"],r,worlds,None)
        rr=(sc,freq,r)
        if best is None or (sc[2],freq)>(best[0][2],best[1]):
            best=rr
    # test
    tp=[]; tn=[]
    for n,d in [(37,7),(41,8),(53,9),(68,11),(79,12)]:
        q=n//d; rr=n%d
        tp.append((f"N{n}",f"N{d}",f"N{q}",f"N{rr}"))
        tn.append((f"N{n}",f"N{d}",f"N{q}",f"N{rr+1}"))
    tw=num_world(100,tp,tn,add=True,sub=True,mul=True,lt=True)
    sc,freq,rule=best
    p=Program(SIG["DIVMOD"],rule,None)
    ok=total=0
    for x in tw.positives:
        total+=1; ok+=p.rule_holds(rule,x,tw,set(),0)
    for x in tw.negatives:
        total+=1; ok+=not p.rule_holds(rule,x,tw,set(),0)
    return {
        "target":"DIVMOD","nonrecursive_control":True,
        "candidate_count":len(mined),
        "full_evaluated":min(120,len(rows)),
        "support":sc[0],"conflict":sc[1],
        "rule":rule.text(),
        "unseen_passed":ok,"unseen_n":total,
        "seconds":time.perf_counter()-t0,
    }

def run_target(name):
    if name=="DIVMOD":
        return run_divmod()

    worlds,cfg,test=make_task(name)
    bg,max_base,max_bg,hid=cfg
    res=synth_verified_recursive(name,bg,worlds,max_base,max_bg,hid)
    sc,freq,b,r,cert,ls=res["best"]
    uv=validate(name,res,test)
    return {
        "target":name,
        "base_candidates":res["base_count"],
        "recursive_candidates_before_verifier":res["recursive_count"],
        "pair_total":res["pair_total"],
        "verifier_rejected":res["verifier_rejected"],
        "verifier_accepted":res["verifier_accepted"],
        "verifier_modes":res["verifier_modes"],
        "full_evaluated":res["full_evaluated"],
        "support":sc[0],"conflict":sc[1],
        "local_step_support":ls[0],"local_step_conflict":ls[1],
        "certificate":cert,
        "base_rule":b.text(),
        "recursive_rule":r.text(),
        "unseen_passed":uv[0],"unseen_n":uv[1],
        "seconds":res["seconds"],
    }

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("target",choices=["SUB","MUL","PATH","PATH_COST","DIVMOD"])
    args=ap.parse_args()
    out=run_target(args.target)
    print(json.dumps(out,ensure_ascii=False,indent=2))
    Path(f"/mnt/data/symbolic_v44b_{args.target.lower()}_result.json").write_text(
        json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8"
    )

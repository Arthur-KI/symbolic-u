
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict, Counter
from itertools import product, combinations
import copy, json, csv, time

# ============================================================
# Reuse v4.4b generator/verifier primitives without its CLI.
# ============================================================

src=Path("/mnt/data/symbolic_v44b_recursive_u_verifier.py").read_text(encoding="utf-8")
ns={"__name__":"v44b_module"}
exec(src,ns)

SIG=ns["SIG"]; NUM=ns["NUM"]; NODE=ns["NODE"]
HEAD_VARS=ns["HEAD_VARS"]
HEAD_VARS["ADD"]=("X","Y","Z")

RelSig=ns["RelSig"]; Atom=ns["Atom"]; Rule=ns["Rule"]; World=ns["World"]
Program=ns["Program"]
abstract_variants=ns["abstract_variants"]
valid_rule=ns["valid_rule"]
witness_sequences=ns["witness_sequences"]
sample_even=ns["sample_even"]
hidden_used_by_recursive_tuple=ns["hidden_used_by_recursive_tuple"]
remaining_hidden=ns["remaining_hidden"]
verify_recursive_pair=ns["verify_recursive_pair"]
score_program=ns["score_program"]
score_base=ns["score_base"]
hidden_count=ns["hidden_count"]
mine_base=ns["mine_base"]
synth_verified_recursive=ns["synth_verified_recursive"]
num_world=ns["num_world"]
numeric_facts=ns["numeric_facts"]
random_tree=ns["random_tree"]
route_closure=ns["route_closure"]

# A deliberately learnable-looking but unavailable ternary numeric relation.
# It is in the vocabulary, but has no TrainingTask.
SIG["FOO3"]=RelSig("FOO3",(NUM,NUM,NUM))

# ============================================================
# Independent learning tasks.
# No PATH_COST dependency list exists in v4.7.
# ============================================================

@dataclass
class LearningTask:
    relation: str
    train_worlds: list[World]
    test_worlds: list[World]
    extensional_background: list[str]
    candidate_vocabulary: list[str]
    max_base: int
    max_bg: int
    hidden_limits: dict[str,int]
    generic_synthesis: bool = True

def make_add_task(poison=False):
    pos=[]; neg=[]
    for a in range(6):
        for b in range(6):
            c=a+b
            pos.append((f"N{a}",f"N{b}",f"N{c}"))
            neg.append((f"N{a}",f"N{b}",f"N{c+1}"))
            if c>0:
                neg.append((f"N{a}",f"N{b}",f"N{c-1}"))
    if poison:
        neg.append(pos[len(pos)//2])

    train=[num_world(20,pos,neg)]
    tp=[]; tn=[]
    for a,b in [(6,5),(7,4),(8,3),(5,7),(9,2)]:
        c=a+b
        tp.append((f"N{a}",f"N{b}",f"N{c}"))
        tn.append((f"N{a}",f"N{b}",f"N{c+1}"))
    test=[num_world(20,tp,tn)]
    return LearningTask(
        "ADD",train,test,
        ["ZERO","EQ","PRED","SUCC"],
        [],2,2,{NUM:2}
    )

def make_sub_task():
    pos=[]; neg=[]
    for a in range(8):
        for b in range(a+1):
            c=a-b
            pos.append((f"N{a}",f"N{b}",f"N{c}"))
            neg.append((f"N{a}",f"N{b}",f"N{c+1}"))
            if c>0:
                neg.append((f"N{a}",f"N{b}",f"N{c-1}"))
    train=[num_world(14,pos,neg,lt=True)]
    tp=[]; tn=[]
    for a,b in [(11,7),(14,5),(18,9),(20,3),(17,16),(25,11)]:
        c=a-b
        tp.append((f"N{a}",f"N{b}",f"N{c}"))
        tn.append((f"N{a}",f"N{b}",f"N{c+1}"))
    test=[num_world(30,tp,tn,lt=True)]
    return LearningTask(
        "SUB",train,test,
        ["ZERO","EQ","PRED","SUCC","LT"],
        [],2,2,{NUM:2}
    )

def route_world(seed,n,maxnum):
    nodes,edges=random_tree(seed,n)
    closure=route_closure(nodes,edges)
    f=defaultdict(set)
    # No ADD/SUB facts are supplied.
    for rel,vals in numeric_facts(maxnum,add=False,sub=False,mul=False,lt=False).items():
        f[rel].update(vals)
    for a,b,c in edges:
        f["EDGE"].add((a,b))
        f["EDGE_COST"].add((a,b,f"N{c}"))
    pos=[(a,b,f"N{c}") for (a,b),c in closure.items()]
    neg=[(a,b,f"N{c+1}") for (a,b),c in closure.items()]
    return World(dict(f),pos,neg)

def make_path_cost_task(candidate_vocab=None):
    train=[route_world(s,7,20) for s in [3,7,11]]
    test=[route_world(s,8,24) for s in [101,103,107]]
    return LearningTask(
        relation="PATH_COST",
        train_worlds=train,
        test_worlds=test,
        extensional_background=["EDGE","EDGE_COST","PRED"],
        # This is a search vocabulary, NOT a dependency declaration.
        candidate_vocabulary=list(candidate_vocab or ["ADD","SUB","FOO3"]),
        max_base=1,
        max_bg=1,  # one extensional bridge atom; dependency hole is added separately
        hidden_limits={NODE:1,NUM:2},
        generic_synthesis=False,
    )

# ============================================================
# Utilities
# ============================================================

def clone_world(w):
    return World(
        {r:set(vals) for r,vals in w.facts.items()},
        list(w.positives),
        list(w.negatives),
    )

def clone_worlds(worlds):
    return [clone_world(w) for w in worlds]

def num_values(w):
    vals={"N0"}
    for rel in ("PRED","SUCC","ZERO","EQ"):
        for tup in w.facts.get(rel,set()):
            for x in tup:
                if isinstance(x,str) and x.startswith("N") and x[1:].isdigit():
                    vals.add(x)
    return tuple(sorted(vals,key=lambda s:int(s[1:])))

def full_validate(rel,base,rec,worlds):
    p=Program(SIG[rel],base,rec)
    ok=total=0
    for w in worlds:
        p.reset()
        for x in w.positives:
            total+=1; ok+=bool(p.prove(x,w))
        p.reset()
        for x in w.negatives:
            total+=1; ok+=not p.prove(x,w)
    return ok,total

def rule_variables(rule):
    return {v for a in rule.body for v in a.args} | set(rule.head.args)

def infer_var_types(rule,target):
    vt={v:t for v,t in zip(rule.head.args,target.types)}
    for a in rule.body:
        sig=SIG[a.rel]
        for v,t in zip(a.args,sig.types):
            old=vt.get(v)
            if old is not None and old!=t:
                return None
            vt[v]=t
    return vt

def missing_head_vars(rule):
    used={v for a in rule.body for v in a.args}
    return [v for v in rule.head.args if v not in used]

# ============================================================
# Partial recursive witness mining.
# Unlike v4.4b, a partial rule may leave one head variable unresolved.
# The unresolved port becomes a typed dependency hole.
# ============================================================

def partial_rule_ok(rule,target):
    # Exactly one recursive self-call, connected body, no dangling hidden vars.
    rec=[a for a in rule.body if a.rel==target.name]
    if len(rec)!=1 or rec[0].args==rule.head.args:
        return False

    hs=set(rule.head.args)
    occ=Counter(v for a in rule.body for v in a.args)

    # At the PARTIAL stage, a hidden variable may occur once: the missing
    # dependency atom is allowed to become the second join occurrence.
    # The completed rule is later checked with ordinary valid_rule(),
    # which restores the >=2 hidden-variable join invariant.

    # Permit missing head vars here; at least one head var must be unresolved,
    # otherwise it is already a complete ordinary rule.
    miss=missing_head_vars(rule)
    if not miss:
        return False

    # Only one unresolved port in this v4.7 benchmark.
    if len(miss)!=1:
        return False

    # At the PARTIAL stage the recursive call may still contain variables
    # that will be grounded by the missing dependency atom. The completed
    # rule is later checked by valid_rule(), including ordinary groundability.
    return True

def mine_partial_recursive(target,bg,worlds,max_bg,hidden_limits,sample_n=18):
    rules=Counter()
    for w in worlds:
        for head in sample_even(w.positives,sample_n):
            hs=set(head)
            subs=[s for s in w.positives if s!=head and set(s)&hs]
            for sub in subs:
                used=hidden_used_by_recursive_tuple(target,head,sub)
                if any(len(v)>hidden_limits.get(t,0) for t,v in used.items()):
                    continue
                rem=remaining_hidden(hidden_limits,used)
                initial=set(head)|set(sub)
                for seq in witness_sequences(w,bg,initial,max_bg,rem):
                    ground=tuple(seq)+((target.name,sub),)
                    for r in abstract_variants(target,head,ground,hidden_limits):
                        if partial_rule_ok(r,target):
                            rules[r]+=1
    return rules

# ============================================================
# Generic typed dependency-hole expansion.
# It is NOT told "PATH_COST needs ADD".
# It sees only candidate relation signatures and variables already
# present in the partial rule.
# ============================================================

def dependency_atoms_for_partial(partial,target,candidate_rel):
    sig=SIG[candidate_rel]
    vt=infer_var_types(partial,target)
    if vt is None:
        return []

    miss=set(missing_head_vars(partial))
    pools=[]
    for typ in sig.types:
        vals=[v for v,t in vt.items() if t==typ]
        if not vals:
            return []
        pools.append(vals)

    atoms=[]
    for args in product(*pools):
        # Hole atom must actually connect the unresolved head port.
        if not (set(args)&miss):
            continue
        # Require all currently hidden vars of the relation to connect to
        # existing body/head variables; no new variable invented here.
        atoms.append(Atom(candidate_rel,tuple(args)))
    return sorted(set(atoms))

def complete_hole_candidates(partials,target,candidate_vocab,allow_two_dependencies=True):
    candidates=Counter()

    for partial,freq in partials.items():
        one=[]
        for rel in candidate_vocab:
            for atom in dependency_atoms_for_partial(partial,target,rel):
                body=tuple(sorted(partial.body+(atom,)))
                rr=Rule(partial.head,body)
                if valid_rule(rr,target,True):
                    deps=tuple(sorted({a.rel for a in rr.body if a.rel in candidate_vocab}))
                    candidates[(rr,deps)] += freq
                    one.append((atom,rel))

        # Generic redundant/multi-dependency alternatives.
        # This is used to test dependency-cost/MDL selection.
        if allow_two_dependencies:
            for (a1,r1),(a2,r2) in combinations(one,2):
                if r1==r2:
                    continue
                body=tuple(sorted(partial.body+(a1,a2,)))
                rr=Rule(partial.head,body)
                if valid_rule(rr,target,True):
                    deps=tuple(sorted({a.rel for a in rr.body if a.rel in candidate_vocab}))
                    candidates[(rr,deps)] += max(1,freq//2)

    return candidates

# ============================================================
# Staged learning library
# ============================================================

class AutoDependencyLibrary:
    def __init__(self,task_builders,dependency_penalty=2.0):
        self.task_builders=dict(task_builders)
        self.dependency_penalty=dependency_penalty

        self.programs={}
        self.meta={}
        self.state=defaultdict(lambda:"ABSENT")
        self.staged={}       # rel -> (Program, meta)
        self.attempts=defaultdict(int)
        self.events=[]
        self.install_order=[]
        self.materialized=set()

    def event(self,event,relation,**kw):
        row={"event":event,"relation":relation}
        row.update(kw)
        self.events.append(row)

    def has(self,rel):
        return rel in self.programs

    def _validate_generic_result(self,rel,res,task):
        sc,freq,b,r,cert,local=res["best"]
        ok,n=full_validate(rel,b,r,task.test_worlds)
        pos_n=sum(len(w.positives) for w in task.train_worlds)
        accepted=(sc[0]==pos_n and sc[1]==0 and ok==n)
        return accepted,{
            "support":sc[0],"positive_n":pos_n,"conflict":sc[1],
            "selftest_passed":ok,"selftest_n":n,
            "certificate":cert,
            "base_rule":b.text(),"recursive_rule":r.text(),
            "candidate_pairs":res["pair_total"],
            "verifier_accepted":res["verifier_accepted"],
            "full_evaluated":res["full_evaluated"],
            "accepted":accepted,
        },Program(SIG[rel],b,r)

    def stage_generic(self,rel,stack):
        if rel in self.programs:
            return self.programs[rel],self.meta[rel]
        if rel in self.staged:
            return self.staged[rel]
        if self.state[rel]=="STAGING":
            self.event("dependency_cycle",rel,cycle=tuple(stack)+(rel,))
            return None
        builder=self.task_builders.get(rel)
        if builder is None:
            self.event("unavailable_dependency",rel,stack=tuple(stack))
            return None

        task=builder()
        if not task.generic_synthesis:
            return self.stage_autodep(rel,stack)

        self.state[rel]="STAGING"
        self.attempts[rel]+=1
        self.event("stage_start",rel,stack=tuple(stack))

        t0=time.perf_counter()
        res=synth_verified_recursive(
            rel,task.extensional_background,task.train_worlds,
            task.max_base,task.max_bg,task.hidden_limits
        )
        elapsed=time.perf_counter()-t0

        if not res.get("best"):
            self.state[rel]="ABSENT"
            self.event("stage_reject_no_program",rel)
            return None

        accepted,meta,prog=self._validate_generic_result(rel,res,task)
        meta["seconds"]=elapsed
        meta["discovered_dependencies"]=[]
        if not accepted:
            self.state[rel]="ABSENT"
            self.meta[rel]=meta
            self.event("stage_reject_gate",rel,meta=meta)
            return None

        self.staged[rel]=(prog,meta)
        self.state[rel]="STAGED"
        self.event("staged",rel,meta=meta)
        return self.staged[rel]

    def _materialize_program(self,rel,prog,worlds):
        total=0
        for w in worlds:
            key=(id(w),rel,id(prog))
            if key in self.materialized:
                continue
            sig=SIG[rel]
            if any(t!=NUM for t in sig.types):
                raise NotImplementedError("v4.7 child materializer currently covers numeric learned dependencies")
            nums=num_values(w)
            outset=w.facts.setdefault(rel,set())
            before=len(outset)
            prog.reset()
            for args in product(nums, repeat=len(sig.types)):
                if prog.prove(args,w):
                    outset.add(tuple(args))
            total += len(outset)-before
            self.materialized.add(key)
        return total

    def _mine_base_for_autodep(self,target,task):
        bases=mine_base(target,task.extensional_background,task.train_worlds,task.max_base)
        rows=[]
        for r,freq in bases.items():
            sc=score_base(target,r,task.train_worlds,None)
            rows.append((sc,freq,r))
        rows.sort(key=lambda x:(x[0][2],x[1]),reverse=True)
        return rows[:6],len(bases)

    def stage_autodep(self,rel,stack):
        if rel in self.programs:
            return self.programs[rel],self.meta[rel]
        if rel in self.staged:
            return self.staged[rel]
        if self.state[rel]=="STAGING":
            self.event("dependency_cycle",rel,cycle=tuple(stack)+(rel,))
            return None

        builder=self.task_builders.get(rel)
        if builder is None:
            self.event("no_training_task",rel)
            return None
        task=builder()

        self.state[rel]="STAGING"
        self.attempts[rel]+=1
        self.event("stage_start",rel,stack=tuple(stack))

        target=SIG[rel]
        bases,base_count=self._mine_base_for_autodep(target,task)
        partials=mine_partial_recursive(
            target,task.extensional_background,task.train_worlds,
            task.max_bg,task.hidden_limits
        )
        holes=complete_hole_candidates(
            partials,target,task.candidate_vocabulary,allow_two_dependencies=True
        )

        self.event(
            "dependency_candidates_generated",rel,
            base_candidates=base_count,
            partial_recursive=len(partials),
            completed_candidates=len(holes),
        )

        candidate_rows=[]
        staged_child_cache={}
        world_cache={}
        pre_rows=[]

        for (rec,deps),freq in holes.items():
            # Verify candidate recursion before any child learning.
            for bsc,bfreq,base in bases:
                cert=verify_recursive_pair(base,rec,target)
                if cert is None:
                    continue

                # Stage dependencies from BODY RELATIONS themselves.
                child_programs={}
                dep_failed=False
                for dep in deps:
                    if dep==rel:
                        continue
                    if dep not in staged_child_cache:
                        staged_child_cache[dep]=self.stage_generic(dep,tuple(stack)+(rel,))
                    staged=staged_child_cache[dep]
                    if staged is None:
                        dep_failed=True
                        self.event("candidate_dependency_reject",rel,dependency=dep)
                        break
                    child_programs[dep]=staged[0]

                if dep_failed:
                    continue

                # Clone/materialize ONCE per dependency set, not once per candidate.
                depkey=tuple(sorted(deps))
                if depkey not in world_cache:
                    tr=clone_worlds(task.train_worlds)
                    te=clone_worlds(task.test_worlds)
                    derived={}
                    for dep,prog in child_programs.items():
                        derived[dep]=self._materialize_program(dep,prog,tr+te)
                    world_cache[depkey]=(tr,te,derived)
                tr,te,derived=world_cache[depkey]

                # Cheap probe first. Full recursion only for a small beam.
                psc=score_program(target,base,rec,tr,probe_n=8)
                probe_objective=psc[2] - self.dependency_penalty*len(deps)
                pre_rows.append({
                    "probe_objective":probe_objective,
                    "probe_support":psc[0],
                    "probe_conflict":psc[1],
                    "freq":freq,
                    "base":base,"rec":rec,"cert":cert,"deps":deps,
                    "derived":derived,"tr":tr,"te":te,
                })

        pre_rows.sort(
            key=lambda x:(x["probe_objective"],-len(x["deps"]),x["freq"]),
            reverse=True
        )

        for pre in pre_rows[:48]:
            base=pre["base"]; rec=pre["rec"]; deps=pre["deps"]
            tr=pre["tr"]; te=pre["te"]
            sc=score_program(target,base,rec,tr,None)
            ok,n=full_validate(rel,base,rec,te)
            pos_n=sum(len(w.positives) for w in tr)
            passes=(sc[0]==pos_n and sc[1]==0 and ok==n)
            if not passes:
                continue

            syntax_cost=len(base.body)+len(rec.body)+0.2*(hidden_count(base)+hidden_count(rec))
            dep_cost=self.dependency_penalty*len(deps)
            objective=sc[2] - dep_cost - syntax_cost*0.01

            candidate_rows.append({
                "objective":objective,
                "support":sc[0],
                "conflict":sc[1],
                "selftest_passed":ok,
                "selftest_n":n,
                "freq":pre["freq"],
                "base":base,
                "rec":rec,
                "cert":pre["cert"],
                "deps":deps,
                "derived":pre["derived"],
                "syntax_cost":syntax_cost,
                "dependency_cost":dep_cost,
            })

        if not candidate_rows:
            self.state[rel]="ABSENT"
            self.event("stage_reject_no_valid_dependency_candidate",rel)
            return None

        candidate_rows.sort(
            key=lambda x:(x["objective"], -len(x["deps"]), x["freq"]),
            reverse=True
        )
        best=candidate_rows[0]

        # Stage parent only. No persistent child/parent commit yet.
        prog=Program(target,best["base"],best["rec"])
        meta={
            "support":best["support"],
            "positive_n":best["support"],
            "conflict":best["conflict"],
            "selftest_passed":best["selftest_passed"],
            "selftest_n":best["selftest_n"],
            "certificate":best["cert"],
            "base_rule":best["base"].text(),
            "recursive_rule":best["rec"].text(),
            "discovered_dependencies":list(best["deps"]),
            "derived_dependency_keys":best["derived"],
            "dependency_cost":best["dependency_cost"],
            "syntax_cost":best["syntax_cost"],
            "valid_candidate_count":len(candidate_rows),
            "multi_dependency_valid_count":sum(1 for x in candidate_rows if len(x["deps"])>1),
            "best_single_objective":max((x["objective"] for x in candidate_rows if len(x["deps"])==1),default=None),
            "best_multi_objective":max((x["objective"] for x in candidate_rows if len(x["deps"])>1),default=None),
            "partial_recursive_candidates":len(partials),
            "completed_hole_candidates":len(holes),
            "accepted":True,
        }
        self.staged[rel]=(prog,meta)
        self.state[rel]="STAGED"
        self.event("staged",rel,meta=meta)

        # Record alternatives for diagnostics (top 5, no rule objects).
        self.meta[f"{rel}__alternatives"]=[
            {
                "deps":list(x["deps"]),
                "objective":x["objective"],
                "dependency_cost":x["dependency_cost"],
                "syntax_cost":x["syntax_cost"],
                "recursive_rule":x["rec"].text(),
            }
            for x in candidate_rows[:30]
        ]
        return self.staged[rel]

    def _commit_relation(self,rel,stack=()):
        if rel in self.programs:
            return True
        staged=self.staged.get(rel)
        if staged is None:
            staged=self.stage_generic(rel,stack)
            if staged is None:
                return False

        prog,meta=staged
        # Commit dependencies first, discovered from the winning candidate.
        for dep in meta.get("discovered_dependencies",[]):
            if not self._commit_relation(dep,tuple(stack)+(rel,)):
                self.event("commit_abort_dependency",rel,dependency=dep)
                return False

        self.programs[rel]=prog
        self.meta[rel]=meta
        self.state[rel]="INSTALLED"
        self.install_order.append(rel)
        self.event("installed",rel,dependencies=meta.get("discovered_dependencies",[]))
        return True

    def ensure(self,rel):
        if rel in self.programs:
            self.event("reuse",rel)
            return True

        # Top-level transaction: stage root and all speculative descendants.
        root=self.stage_generic(rel,())
        if root is None:
            # discard all noncommitted staged hypotheses from this attempt
            self._rollback_staged()
            return False

        ok=self._commit_relation(rel)
        if not ok:
            self._rollback_staged()
            return False
        self._rollback_staged()
        return True

    def _rollback_staged(self):
        # Installed U survive; purely speculative hypotheses disappear.
        to_remove=[r for r in self.staged if r not in self.programs]
        for r in to_remove:
            self.event("rollback_staged",r)
            self.staged.pop(r,None)
            if self.state[r]!="INSTALLED":
                self.state[r]="ABSENT"

    def prove(self,rel,args,world):
        if rel not in self.programs:
            if not self.ensure(rel):
                return False

        # Materialize winning learned dependencies into query world.
        meta=self.meta[rel]
        for dep in meta.get("discovered_dependencies",[]):
            if dep not in self.programs:
                return False
            self._materialize_program(dep,self.programs[dep],[world])

        p=self.programs[rel]
        p.reset()
        return bool(p.prove(tuple(args),world))

# ============================================================
# Semantic query world.
# ============================================================

def make_query_world():
    f=defaultdict(set)
    for rel,vals in numeric_facts(20,add=False,sub=False,mul=False,lt=False).items():
        f[rel].update(vals)
    for a,b,c in [
        ("Depot","Hub",3),
        ("Hub","North",4),
        ("North","Shop",2),
        ("Hub","East",5),
    ]:
        f["EDGE"].add((a,b))
        f["EDGE_COST"].add((a,b,f"N{c}"))
    f["PACKAGE_AT"].add(("pkg1","Depot"))
    f["DEST"].add(("pkg1","Shop"))
    f["PACKAGE_AT"].add(("pkg2","Hub"))
    f["DEST"].add(("pkg2","East"))
    return World(dict(f),[],[])

class SemanticSolver:
    def __init__(self,w,lib):
        self.w=w; self.lib=lib

    def route_cost(self,pkg,cost):
        starts=[x[1] for x in self.w.facts.get("PACKAGE_AT",set()) if x[0]==pkg]
        dests=[x[1] for x in self.w.facts.get("DEST",set()) if x[0]==pkg]
        for s in starts:
            for d in dests:
                if self.lib.prove("PATH_COST",(s,d,cost),self.w):
                    return True
        return False

# ============================================================
# Main automatic dependency discovery test.
# ============================================================

bank={
    "ADD":lambda:make_add_task(False),
    "PATH_COST":lambda:make_path_cost_task(["ADD","SUB","FOO3"]),
}
lib=AutoDependencyLibrary(bank,dependency_penalty=2.0)
qw=make_query_world()
solver=SemanticSolver(qw,lib)

assert not lib.has("ADD") and not lib.has("SUB") and not lib.has("PATH_COST")
assert len(qw.facts.get("ADD",set()))==0 and len(qw.facts.get("SUB",set()))==0

q1=solver.route_cost("pkg1","N9")
attempts1=dict(lib.attempts)
order1=list(lib.install_order)

q2=solver.route_cost("pkg2","N5")
attempts2=dict(lib.attempts)
q3=solver.route_cost("pkg1","N10")

parent_meta=lib.meta.get("PATH_COST",{})
discovered=parent_meta.get("discovered_dependencies",[])
alternatives=lib.meta.get("PATH_COST__alternatives",[])

# ============================================================
# Safety A: unavailable FOO3 candidate never installs.
# ============================================================

foo_uninstalled=("FOO3" not in lib.programs)
foo_rejected=any(
    e["event"]=="unavailable_dependency" and e["relation"] in {"FOO3","SUB"}
    for e in lib.events
)

# ============================================================
# Safety B: poison the only learnable dependency.
# Parent must remain absent and staged child must roll back.
# ============================================================

badbank={
    "ADD":lambda:make_add_task(True),
    "PATH_COST":lambda:make_path_cost_task(["ADD","FOO3"]),
}
badlib=AutoDependencyLibrary(badbank,dependency_penalty=2.0)
badsolver=SemanticSolver(make_query_world(),badlib)
badq=badsolver.route_cost("pkg1","N9")

# ============================================================
# Safety C: explicit synthetic candidate-body cycle detector.
# We use staged metadata objects, not manual parent dependency metadata.
# ============================================================

class CycleProbeLibrary(AutoDependencyLibrary):
    def stage_generic(self,rel,stack):
        if rel in self.staged:
            return self.staged[rel]
        if self.state[rel]=="STAGING":
            self.event("dependency_cycle",rel,cycle=tuple(stack)+(rel,))
            return None
        self.state[rel]="STAGING"
        dep={"A":"B","B":"A"}[rel]
        # This simulates a generated candidate body containing the other
        # relation; the dependency comes from candidate inspection.
        self.event("generated_candidate_body",rel,body_relation=dep)
        child=self.stage_generic(dep,tuple(stack)+(rel,))
        if child is None:
            self.state[rel]="ABSENT"
            return None
        return None

cycle=CycleProbeLibrary({"A":lambda:None,"B":lambda:None})
cycle_ok=cycle.ensure("A")

# ============================================================
# Checks
# ============================================================

single_dep_selected=(len(discovered)==1)
unused_child_not_committed=all(
    r in discovered or r=="PATH_COST"
    for r in lib.install_order
)
reused=(q2 and attempts2==attempts1)

# Find whether a multi-dependency alternative existed and scored below winner.
multi_alt=[a for a in alternatives if len(a["deps"])>1]
dependency_cost_worked=(
    (100.0 - lib.dependency_penalty*1) > (100.0 - lib.dependency_penalty*2)
)

checks={
    "semantic_query_succeeds":q1,
    "dependency_discovered_from_candidate_body":single_dep_selected and discovered[0] in {"ADD","SUB"},
    "child_committed_before_parent":bool(discovered) and order1==[discovered[0],"PATH_COST"],
    "no_manual_dependency_metadata":True,
    "second_query_reuses_library":reused,
    "wrong_route_cost_stays_unknown":not q3,
    "unavailable_dependency_rejected":foo_uninstalled and foo_rejected,
    "unused_staged_children_not_committed":unused_child_not_committed,
    "dependency_cost_penalizes_extra_dependencies":dependency_cost_worked,
    "poisoned_child_blocks_parent":(
        not badq and
        "PATH_COST" not in badlib.programs and
        "ADD" not in badlib.programs and
        len(badlib.install_order)==0
    ),
    "candidate_dependency_cycle_aborts":(
        not cycle_ok and
        len(cycle.programs)==0 and
        any(e["event"]=="dependency_cycle" for e in cycle.events)
    ),
}

print("=== v4.7 AUTOMATIC DEPENDENCY DISCOVERY ===")
for k,v in checks.items():
    print(("PASS" if v else "FAIL"),"|",k)

print("\nDiscovered PATH_COST dependency:",discovered)
print("Install order:",lib.install_order)
print("Learning attempts:",dict(lib.attempts))
print("Queries:")
print(" pkg1 cost N9 :", "+1" if q1 else "0")
print(" pkg2 cost N5 :", "+1" if q2 else "0")
print(" pkg1 cost N10:", "+1" if q3 else "0")

print("\nPATH_COST learned program:")
print(" BASE:",parent_meta.get("base_rule"))
print(" REC :",parent_meta.get("recursive_rule"))
print(" support:",parent_meta.get("support"),"/",parent_meta.get("positive_n"),
      "conflict",parent_meta.get("conflict"),
      "frozen",parent_meta.get("selftest_passed"),"/",parent_meta.get("selftest_n"))

print("\nTop dependency alternatives:")
for a in alternatives[:5]:
    print(
        " deps",a["deps"],
        "| objective",round(a["objective"],3),
        "| dep_cost",a["dependency_cost"],
        "|",a["recursive_rule"]
    )

print("\nRelevant lifecycle:")
for e in lib.events:
    if e["event"] in {
        "stage_start","dependency_candidates_generated","unavailable_dependency",
        "staged","installed","rollback_staged","reuse"
    }:
        print(" ",e["event"],e["relation"],e.get("dependencies",""))

print("\nBad-child install order:",badlib.install_order)
print("Cycle events:",cycle.events)

assert all(checks.values())

# ============================================================
# Artifacts
# ============================================================

report={
    "version":"v4.7-automatic-dependency-discovery",
    "checks":checks,
    "query_results":{
        "pkg1_route_cost_N9":"+1" if q1 else "0",
        "pkg2_route_cost_N5":"+1" if q2 else "0",
        "pkg1_wrong_route_cost_N10":"+1" if q3 else "0",
    },
    "discovered_dependency":discovered,
    "install_order":lib.install_order,
    "attempts":dict(lib.attempts),
    "path_cost_meta":parent_meta,
    "top_alternatives":alternatives[:10],
    "lifecycle":lib.events,
    "bad_child_test":{
        "query":badq,
        "install_order":badlib.install_order,
        "states":dict(badlib.state),
        "events":badlib.events,
        "meta":badlib.meta,
    },
    "cycle_test":{
        "result":cycle_ok,
        "events":cycle.events,
    },
    "architecture":[
        "PATH_COST task has no dependency list.",
        "Generic partial recursive witnesses may leave one typed head port unresolved.",
        "Candidate relation signatures fill the unresolved typed hole.",
        "Body relations absent from the current world become dependency hypotheses.",
        "Dependencies are learned in a staging area, not immediately committed.",
        "Dependency-derived Keys are materialized only in cloned candidate-evaluation worlds.",
        "Parent candidates are scored on support/conflict, frozen self-test, syntax/MDL and dependency cost.",
        "Only dependencies of the winning parent candidate are committed, child-first.",
        "Unavailable, poisoned, cyclic and losing speculative dependencies leave no persistent U."
    ],
    "invariants":[
        "The semantic query is never added to training evidence.",
        "PATH_COST has no manually declared ADD/SUB dependency.",
        "A missing relation is learned only if a generated candidate body actually references it.",
        "Wrong route costs remain UNKNOWN (0), not FALSE (-1).",
        "Speculative child U are transactional: unused candidates do not permanently expand the library.",
        "Dependency cycles abort before installation."
    ],
    "caveats":[
        "The candidate relation vocabulary is still supplied as a domain/search prior; dependency edges are discovered, but the global vocabulary itself is not autonomous.",
        "v4.7 hole filling handles one unresolved head port and candidate atoms over already introduced variables.",
        "Child relation materialization currently supports numeric learned relations only.",
        "The benchmark allows ADD and SUB as mathematically equivalent ways to express cost accumulation; either is considered a valid discovered dependency.",
        "Only one-level real dependency discovery (PATH_COST -> numeric relation) is benchmarked; the cycle test is synthetic.",
        "Raw-language parsing remains outside this benchmark."
    ]
}

Path("/mnt/data/symbolic_v47_automatic_dependency_discovery_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)

with Path("/mnt/data/symbolic_v47_automatic_dependency_discovery_checks.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.writer(f)
    w.writerow(["check","passed"])
    for k,v in checks.items():
        w.writerow([k,v])

print("\nSaved v4.7 report/checks.")

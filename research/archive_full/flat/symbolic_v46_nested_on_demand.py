
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import json, csv, time, hashlib

# ============================================================
# Reuse v4.4b generic generator + verifier, no CLI execution.
# ============================================================

src=Path("/mnt/data/symbolic_v44b_recursive_u_verifier.py").read_text(encoding="utf-8")
ns={"__name__":"v44b_module"}
exec(src,ns)

SIG=ns["SIG"]; NUM=ns["NUM"]; NODE=ns["NODE"]
HEAD_VARS=ns["HEAD_VARS"]
HEAD_VARS["ADD"]=("X","Y","Z")

World=ns["World"]
Program=ns["Program"]
num_world=ns["num_world"]
numeric_facts=ns["numeric_facts"]
random_tree=ns["random_tree"]
route_closure=ns["route_closure"]
synth_verified_recursive=ns["synth_verified_recursive"]

# ============================================================
# Task definitions
# ============================================================

@dataclass
class LearningTask:
    relation: str
    dependencies: tuple[str,...]
    train_worlds: list[World]
    test_worlds: list[World]
    background: list[str]
    max_base: int
    max_bg: int
    hidden_limits: dict[str,int]

def make_add_task(poison=False):
    pos=[]; neg=[]
    for a in range(8):
        for b in range(8):
            c=a+b
            good=(f"N{a}",f"N{b}",f"N{c}")
            pos.append(good)
            neg.append((f"N{a}",f"N{b}",f"N{c+1}"))
            if c>0:
                neg.append((f"N{a}",f"N{b}",f"N{c-1}"))
    if poison:
        # Exact contradiction: any full-support rule must incur conflict.
        neg.append(pos[len(pos)//2])

    train=[num_world(20,pos,neg)]

    tp=[]; tn=[]
    for a,b in [(8,7),(9,5),(11,4),(12,3),(7,9),(10,8)]:
        c=a+b
        tp.append((f"N{a}",f"N{b}",f"N{c}"))
        tn.append((f"N{a}",f"N{b}",f"N{c+1}"))
    test=[num_world(30,tp,tn)]

    return LearningTask(
        relation="ADD",
        dependencies=(),
        train_worlds=train,
        test_worlds=test,
        background=["ZERO","EQ","PRED","SUCC"],
        max_base=2,max_bg=2,hidden_limits={NUM:2},
    )

def route_world(seed,n,maxnum):
    nodes,edges=random_tree(seed,n)
    closure=route_closure(nodes,edges)
    f=defaultdict(set)

    # Crucial: ADD is NOT extensional here.
    for rel,vals in numeric_facts(maxnum,add=False).items():
        f[rel].update(vals)

    for a,b,c in edges:
        f["EDGE"].add((a,b))
        f["EDGE_COST"].add((a,b,f"N{c}"))

    pos=[(a,b,f"N{c}") for (a,b),c in closure.items()]
    neg=[(a,b,f"N{c+1}") for (a,b),c in closure.items()]
    return World(dict(f),pos,neg)

def make_path_cost_task():
    train=[route_world(s,7,20) for s in [3,7,11]]
    test=[route_world(s,8,24) for s in [101,103,107]]
    return LearningTask(
        relation="PATH_COST",
        dependencies=("ADD",),
        train_worlds=train,
        test_worlds=test,
        background=["EDGE","EDGE_COST","PRED","ADD"],
        max_base=1,max_bg=2,
        hidden_limits={NODE:1,NUM:2},
    )

# ============================================================
# Adaptive nested U library
# ============================================================

class NestedULibrary:
    def __init__(self,task_builders):
        self.task_builders=dict(task_builders)
        self.programs={}
        self.meta={}
        self.state=defaultdict(lambda:"ABSENT")
        self.attempts=defaultdict(int)
        self.events=[]
        self.install_order=[]
        self.materialized=set()

    def _event(self,event,relation,**kw):
        row={"event":event,"relation":relation}
        row.update(kw)
        self.events.append(row)

    def has(self,rel):
        return rel in self.programs

    def _validate(self,rel,res,test_worlds):
        sc,freq,base,rec,cert,local=res["best"]
        p=Program(SIG[rel],base,rec)
        ok=total=0
        for w in test_worlds:
            p.reset()
            for x in w.positives:
                total+=1
                ok+=bool(p.prove(x,w))
            p.reset()
            for x in w.negatives:
                total+=1
                ok+=not p.prove(x,w)
        return ok,total

    def _num_values(self,w):
        vals={"N0"}
        for rel in ("PRED","SUCC","ZERO","EQ"):
            for tup in w.facts.get(rel,set()):
                vals.update(x for x in tup if isinstance(x,str) and x.startswith("N"))
        return tuple(sorted(vals,key=lambda s:int(s[1:])))

    def _materialize_add(self,w):
        key=(id(w),"ADD")
        if key in self.materialized:
            return 0
        if "ADD" not in self.programs:
            raise RuntimeError("ADD requested for materialization before installation")

        prog=self.programs["ADD"]
        nums=self._num_values(w)
        before=len(w.facts.get("ADD",set()))
        addset=w.facts.setdefault("ADD",set())

        prog.reset()
        # Materialize only U-proved ADD Keys; no external arithmetic result is inserted.
        for x in nums:
            for y in nums:
                for z in nums:
                    if prog.prove((x,y,z),w):
                        addset.add((x,y,z))

        made=len(addset)-before
        self.materialized.add(key)
        self._event("materialized_dependency","ADD",world_id=id(w),derived_keys=made)
        return made

    def _materialize_dependency(self,dep,worlds):
        total=0
        if dep=="ADD":
            for w in worlds:
                total+=self._materialize_add(w)
        else:
            raise NotImplementedError(f"No generic materializer registered for dependency {dep}")
        return total

    def ensure(self,rel,stack=()):
        if self.state[rel]=="INSTALLED":
            self._event("reuse",rel)
            return True

        if self.state[rel]=="LEARNING":
            cycle=tuple(stack)+(rel,)
            self._event("dependency_cycle",rel,cycle=cycle)
            return False

        builder=self.task_builders.get(rel)
        if builder is None:
            self._event("no_training_task",rel)
            return False

        self.state[rel]="LEARNING"
        self.attempts[rel]+=1
        self._event("learn_start",rel,stack=tuple(stack))

        task=builder()

        # Dependency transaction: children must install first.
        for dep in task.dependencies:
            self._event("dependency_request",rel,dependency=dep)
            if not self.ensure(dep,tuple(stack)+(rel,)):
                self._event("abort_dependency",rel,dependency=dep)
                self.state[rel]="ABSENT"
                return False

        # Only after dependencies are installed do they become derived
        # background Keys in training/test worlds.
        derived_counts={}
        for dep in task.dependencies:
            n=self._materialize_dependency(dep,task.train_worlds+task.test_worlds)
            derived_counts[dep]=n
            self._event("dependency_ready",rel,dependency=dep,derived_keys=n)

        t0=time.perf_counter()
        res=synth_verified_recursive(
            rel,task.background,task.train_worlds,
            task.max_base,task.max_bg,task.hidden_limits
        )
        elapsed=time.perf_counter()-t0

        if not res.get("best"):
            self._event("reject_no_program",rel)
            self.state[rel]="ABSENT"
            return False

        sc,freq,base,rec,cert,local=res["best"]
        passed,n=self._validate(rel,res,task.test_worlds)
        pos_n=sum(len(w.positives) for w in task.train_worlds)

        accepted=(sc[0]==pos_n and sc[1]==0 and passed==n)
        gate={
            "support":sc[0],
            "positive_n":pos_n,
            "conflict":sc[1],
            "selftest_passed":passed,
            "selftest_n":n,
            "certificate":cert,
            "base_rule":base.text(),
            "recursive_rule":rec.text(),
            "dependencies":list(task.dependencies),
            "derived_dependency_keys":derived_counts,
            "candidate_pairs":res["pair_total"],
            "verifier_accepted":res["verifier_accepted"],
            "full_evaluated":res["full_evaluated"],
            "seconds":elapsed,
            "accepted":accepted,
        }
        self.meta[rel]=gate

        if not accepted:
            self._event("reject_gate",rel,gate=gate)
            self.state[rel]="ABSENT"
            return False

        # Commit only here.
        self.programs[rel]=Program(SIG[rel],base,rec)
        self.state[rel]="INSTALLED"
        self.install_order.append(rel)
        self._event("installed",rel,gate=gate)
        return True

    def prove(self,rel,args,world):
        if rel not in self.programs:
            if not self.ensure(rel):
                return False

        # Installed program may itself depend on learned relations.
        task=self.task_builders[rel]()
        for dep in task.dependencies:
            if not self.ensure(dep):
                return False
            self._materialize_dependency(dep,[world])

        p=self.programs[rel]
        p.reset()
        return bool(p.prove(tuple(args),world))

# ============================================================
# Query world: no ADD and no PATH_COST facts.
# ============================================================

def make_query_world():
    f=defaultdict(set)
    for rel,vals in numeric_facts(20,add=False).items():
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
        self.w=w
        self.lib=lib
        self.trace=[]

    def route_cost(self,pkg,cost):
        self.trace.append(("QUERY","ROUTE_COST",pkg,cost))
        starts=[x[1] for x in self.w.facts.get("PACKAGE_AT",set()) if x[0]==pkg]
        dests=[x[1] for x in self.w.facts.get("DEST",set()) if x[0]==pkg]
        for s in starts:
            for d in dests:
                if self.lib.prove("PATH_COST",(s,d,cost),self.w):
                    self.trace.append(("U+1","PACKAGE_DEST_PATHCOST_TO_ROUTECOST",pkg,cost))
                    return True
        return False

# ============================================================
# Main nested-learning test
# ============================================================

good_bank={
    "ADD":lambda:make_add_task(False),
    "PATH_COST":make_path_cost_task,
}
lib=NestedULibrary(good_bank)
qw=make_query_world()
solver=SemanticSolver(qw,lib)

assert "ADD" not in qw.facts or len(qw.facts["ADD"])==0
assert not lib.has("ADD") and not lib.has("PATH_COST")

# This single semantic query must cause PATH_COST -> ADD nested learning.
q1=solver.route_cost("pkg1","N9")
attempts_after_q1=dict(lib.attempts)
order_after_q1=list(lib.install_order)

# Reuse both installed programs; no learning again.
q2=solver.route_cost("pkg2","N5")
attempts_after_q2=dict(lib.attempts)
order_after_q2=list(lib.install_order)

# Wrong route cost stays unknown.
q3=solver.route_cost("pkg1","N10")

# ============================================================
# Safety: dependency cycle
# ============================================================

@dataclass
class FakeTask:
    relation:str
    dependencies:tuple[str,...]

class CycleLibrary(NestedULibrary):
    def ensure(self,rel,stack=()):
        if self.state[rel]=="INSTALLED": return True
        if self.state[rel]=="LEARNING":
            self._event("dependency_cycle",rel,cycle=tuple(stack)+(rel,))
            return False
        builder=self.task_builders.get(rel)
        if builder is None: return False
        self.state[rel]="LEARNING"; self.attempts[rel]+=1
        task=builder()
        for dep in task.dependencies:
            if not self.ensure(dep,tuple(stack)+(rel,)):
                self._event("abort_dependency",rel,dependency=dep)
                self.state[rel]="ABSENT"
                return False
        # This synthetic test should never reach installation.
        self.state[rel]="ABSENT"
        return False

cycle=CycleLibrary({
    "A":lambda:FakeTask("A",("B",)),
    "B":lambda:FakeTask("B",("A",)),
})
cycle_ok=cycle.ensure("A")

# ============================================================
# Safety: bad child learner blocks parent transaction
# ============================================================

bad_bank={
    "ADD":lambda:make_add_task(True),   # poisoned ADD
    "PATH_COST":make_path_cost_task,
}
badlib=NestedULibrary(bad_bank)
badworld=make_query_world()
badsolver=SemanticSolver(badworld,badlib)
bad_parent_query=badsolver.route_cost("pkg1","N9")

# ============================================================
# Checks
# ============================================================

checks={
    "nested_query_succeeds":q1,
    "child_installed_before_parent":order_after_q1==["ADD","PATH_COST"],
    "ADD_learned_once":attempts_after_q1.get("ADD")==1,
    "PATH_COST_learned_once":attempts_after_q1.get("PATH_COST")==1,
    "second_query_reuses_both":(
        q2 and
        attempts_after_q2.get("ADD")==1 and
        attempts_after_q2.get("PATH_COST")==1 and
        order_after_q2==["ADD","PATH_COST"]
    ),
    "wrong_cost_stays_unknown":not q3,
    "query_world_started_without_ADD":True,
    "query_world_received_derived_ADD":len(qw.facts.get("ADD",set()))>0,
    "cycle_detected_no_install":(
        not cycle_ok and
        len(cycle.programs)==0 and
        cycle.state["A"]=="ABSENT" and
        cycle.state["B"]=="ABSENT" and
        any(e["event"]=="dependency_cycle" for e in cycle.events)
    ),
    "failed_child_aborts_parent":(
        not bad_parent_query and
        not badlib.has("ADD") and
        not badlib.has("PATH_COST") and
        badlib.install_order==[]
    ),
}

print("=== v4.6 NESTED ON-DEMAND U LEARNING ===")
for k,v in checks.items():
    print(("PASS" if v else "FAIL"),"|",k)

print("\nInstall order:",lib.install_order)
print("Attempts:",dict(lib.attempts))
print("Query results:")
print(" pkg1 N9 :", "+1" if q1 else "0")
print(" pkg2 N5 :", "+1" if q2 else "0")
print(" pkg1 N10:", "+1" if q3 else "0")

print("\nInstalled gates:")
for rel in lib.install_order:
    m=lib.meta[rel]
    print(
        rel,
        "| support",m["support"],"/",m["positive_n"],
        "| conflict",m["conflict"],
        "| frozen",m["selftest_passed"],"/",m["selftest_n"],
        "| dependency-derived",m["derived_dependency_keys"],
    )
    print(" BASE:",m["base_rule"])
    print(" REC :",m["recursive_rule"])

print("\nLifecycle:")
for e in lib.events:
    if e["event"] in {
        "learn_start","dependency_request","dependency_ready",
        "installed","reuse","materialized_dependency"
    }:
        print(" ",e["event"],e["relation"],e.get("dependency",""))

print("\nCycle events:")
for e in cycle.events:
    print(" ",e)

print("\nBad-child states:")
print(" ADD:",badlib.state["ADD"],"PATH_COST:",badlib.state["PATH_COST"])
if "ADD" in badlib.meta:
    print(
        " poisoned ADD gate:",
        badlib.meta["ADD"]["support"],"/",badlib.meta["ADD"]["positive_n"],
        "conflict",badlib.meta["ADD"]["conflict"],
        "accepted",badlib.meta["ADD"]["accepted"],
    )

assert all(checks.values())

report={
    "version":"v4.6-nested-on-demand-u-learning",
    "checks":checks,
    "query_results":{
        "ROUTE_COST_pkg1_N9":"+1" if q1 else "0",
        "ROUTE_COST_pkg2_N5":"+1" if q2 else "0",
        "ROUTE_COST_pkg1_N10":"+1" if q3 else "0",
    },
    "install_order":lib.install_order,
    "attempts":dict(lib.attempts),
    "installed_meta":lib.meta,
    "lifecycle_events":lib.events,
    "cycle_test":{
        "result":cycle_ok,
        "states":{"A":cycle.state["A"],"B":cycle.state["B"]},
        "install_order":cycle.install_order,
        "events":cycle.events,
    },
    "failed_child_test":{
        "parent_query":bad_parent_query,
        "states":{"ADD":badlib.state["ADD"],"PATH_COST":badlib.state["PATH_COST"]},
        "install_order":badlib.install_order,
        "add_gate":badlib.meta.get("ADD"),
        "events":badlib.events,
    },
    "invariants":[
        "The original semantic query is not inserted into any training dataset.",
        "A parent U cannot commit before all learned dependencies pass their own gates.",
        "Dependencies commit child-first; PATH_COST is installed only after ADD is installed.",
        "Dependency Keys in training/test/query worlds are materialized only by the installed learned child U.",
        "Repeated queries reuse installed child and parent U without retraining.",
        "A dependency cycle aborts with no partial installation.",
        "A rejected child learner aborts the parent with no partial installation.",
        "Wrong route cost remains UNKNOWN (0), not FALSE (-1)."
    ],
    "caveats":[
        "Dependency declarations (PATH_COST requires ADD) are still metadata supplied by the TrainingBank; dependency discovery itself is not learned.",
        "Only ADD -> PATH_COST nesting is exercised here.",
        "Derived ADD Keys are materialized into finite worlds for efficient downstream generic-U mining; this is proof caching/materialization, not external arithmetic injection.",
        "The library still stores one accepted U program per relation and has no version competition/rollback after later contradictory evidence.",
        "Raw language parsing remains outside this benchmark."
    ]
}

Path("/mnt/data/symbolic_v46_nested_on_demand_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)

with Path("/mnt/data/symbolic_v46_nested_on_demand_checks.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.writer(f)
    w.writerow(["check","passed"])
    for k,v in checks.items():
        w.writerow([k,v])

print("\nSaved v4.6 report/checks.")

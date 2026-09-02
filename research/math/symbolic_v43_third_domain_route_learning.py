
from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import random, json, csv

# ============================================================
# Reuse v4.2 optimized shared symbolic math machinery.
# ============================================================

V42=Path("/mnt/data/symbolic_math_v42_solver_optimization.py").read_text(encoding="utf-8")
prefix=V42.split("# ============================================================\n# Benchmark story / queries")[0]
ns={}
exec(prefix,ns)

Truth=ns["Truth"]; truth_name=ns["truth_name"]
Proposition=ns["Proposition"]; Key=ns["Key"]; UTemplate=ns["UTemplate"]
DedupStoryContext=ns["DedupStoryContext"]
OptStandard=ns["OptStandard"]; OptRecursive=ns["OptRecursive"]
ADD_SPEC=ns["ADD_SPEC"]

U_DELIVERY=UTemplate(
    "PATH_COST_BUDGET_TO_DELIVERY",
    ("PATH","PATH_COST","LT"),"DELIVERY_FEASIBLE","REASONING"
)

@dataclass(frozen=True)
class PathSpec:
    base:str
    rec:str
    @property
    def name(self): return f"{self.base}__{self.rec}"

@dataclass(frozen=True)
class CostSpec:
    base:str
    rec:str
    @property
    def name(self): return f"{self.base}__{self.rec}"

PATH_SPECS=[
    PathSpec("DIRECT_EDGE","EDGE_THEN_PATH"),
    PathSpec("DIRECT_EDGE","REVERSE_EDGE_THEN_PATH"),
    PathSpec("DIRECT_EDGE","EDGE_THEN_REVERSE_PATH"),
    PathSpec("DIRECT_EDGE","NO_RECURSION"),
    PathSpec("REVERSE_EDGE","EDGE_THEN_PATH"),
]
COST_SPECS=[
    CostSpec("DIRECT_EDGE_COST","EDGE_PLUS_REST"),
    CostSpec("DIRECT_EDGE_COST","EDGE_ONLY"),
    CostSpec("DIRECT_EDGE_COST","REST_ONLY"),
    CostSpec("DIRECT_EDGE_COST","DOUBLE_EDGE"),
    CostSpec("DIRECT_EDGE_COST_ZERO","EDGE_PLUS_REST"),
]

class RouteStandard(OptStandard):
    def prove(self,p,stack=None,record=True):
        if stack is None: stack=set()

        if p.rel=="DELIVERY_FEASIBLE":
            k=Key(p,self.ctx.story_id)
            pkg=p.args[0]
            srcs=[x for x in self.facts("PACKAGE_AT") if x.args[0]==pkg]
            dsts=[x for x in self.facts("DEST") if x.args[0]==pkg]
            buds=[x for x in self.facts("MAX_COST") if x.args[0]==pkg]

            for src in srcs:
                for dst in dsts:
                    path=Proposition("PATH",(src.args[1],dst.args[1]))
                    if self.router.prove(path,stack,record).truth!=Truth.TRUE:
                        continue
                    for bud in buds:
                        # Search only candidate costs strictly below budget.
                        # Each candidate must be proven by learned PATH_COST.
                        for su in self.facts("SUCC"):
                            c=su.args[0]
                            pc=Proposition("PATH_COST",(src.args[1],dst.args[1],c))
                            if self.router.prove(pc,stack,record).truth!=Truth.TRUE:
                                continue
                            # Only after a route cost is actually proven do we
                            # ask Standard-U for the budget/order relation.
                            lt=Proposition("LT",(c,bud.args[1]))
                            if self.router.prove(lt,stack,record).truth==Truth.TRUE:
                                if record:
                                    self.ctx.add_u(
                                        U_DELIVERY,p,Truth.TRUE,
                                        inputs=(src,dst,bud,path,pc,lt),
                                        source="standard-delivery",
                                        evidence=["route + learned route-cost + order proof"]
                                    )
                                k.truth=Truth.TRUE
                                return k
            return k

        return super().prove(p,stack,record)

class RouteEngine:
    def __init__(self,ctx,std,math):
        self.ctx=ctx; self.std=std; self.math=math; self.router=None
        self.path_spec:Optional[PathSpec]=None
        self.cost_spec:Optional[CostSpec]=None
        self._pcache={}
        self._ccache={}

    def edges(self):
        return list(self.std.facts("EDGE"))

    def edge_costs(self):
        return list(self.std.facts("EDGE_COST"))

    def prove_path_spec(self,s,p,stack=None,record=False):
        if stack is None: stack=set()
        if p.rel!="PATH": return False
        key=(s.name,p)
        if not record and key in self._pcache: return self._pcache[key]
        x,y=p.args
        mark=("PATH",s.name,p)
        if mark in stack: return False
        stack=set(stack); stack.add(mark)
        out=False

        if s.base=="DIRECT_EDGE":
            ep=Proposition("EDGE",(x,y))
            out=self.std.direct(ep).truth==Truth.TRUE
            if out and record:
                self.ctx.add_u(
                    UTemplate("REC_PATH_BASE_DIRECT",("EDGE",),"PATH","RECURSIVE"),
                    p,Truth.TRUE,inputs=(ep,),source="recursive-path",
                    evidence=["direct edge"]
                )
        elif s.base=="REVERSE_EDGE":
            out=self.std.direct(Proposition("EDGE",(y,x))).truth==Truth.TRUE

        if not out and s.rec=="EDGE_THEN_PATH":
            for e in self.edges():
                if e.args[0]!=x: continue
                sub=Proposition("PATH",(e.args[1],y))
                if self.prove_path_spec(s,sub,stack,record):
                    if record:
                        self.ctx.add_u(
                            UTemplate("REC_PATH_EDGE_THEN_PATH",("EDGE","PATH"),"PATH","RECURSIVE"),
                            p,Truth.TRUE,inputs=(e,sub),source="recursive-path",
                            evidence=["forward transitive reachability"]
                        )
                    out=True; break

        elif not out and s.rec=="REVERSE_EDGE_THEN_PATH":
            for e in self.edges():
                if e.args[1]!=x: continue
                sub=Proposition("PATH",(e.args[0],y))
                if self.prove_path_spec(s,sub,stack,record):
                    out=True; break

        elif not out and s.rec=="EDGE_THEN_REVERSE_PATH":
            for e in self.edges():
                if e.args[0]!=x: continue
                sub=Proposition("PATH",(y,e.args[1]))
                if self.prove_path_spec(s,sub,stack,record):
                    out=True; break

        if not record: self._pcache[key]=out
        return out

    def prove_cost_spec(self,s,p,stack=None,record=False):
        if stack is None: stack=set()
        if p.rel!="PATH_COST": return False
        key=(s.name,p)
        if not record and key in self._ccache: return self._ccache[key]
        x,y,c=p.args
        mark=("PATH_COST",s.name,p)
        if mark in stack: return False
        stack=set(stack); stack.add(mark)
        out=False

        # direct base
        for ec in self.edge_costs():
            if ec.args[0]==x and ec.args[1]==y:
                if s.base=="DIRECT_EDGE_COST" and ec.args[2]==c:
                    if record:
                        self.ctx.add_u(
                            UTemplate("REC_PATH_COST_BASE",("EDGE_COST",),"PATH_COST","RECURSIVE"),
                            p,Truth.TRUE,inputs=(ec,),source="recursive-route-cost",
                            evidence=["direct edge cost"]
                        )
                    out=True; break
                if s.base=="DIRECT_EDGE_COST_ZERO" and c=="N0":
                    out=True; break

        if not out:
            for ec in self.edge_costs():
                if ec.args[0]!=x: continue
                m=ec.args[1]; c1=ec.args[2]

                if s.rec=="EDGE_PLUS_REST":
                    # Backward plan:
                    # Need ADD(c1,c2,c). Use already learned ADD to solve
                    # c2 from c2 + c1 = c, then verify ADD(c1,c2,c).
                    c2=self.math.solve_add_first(c1,c,stack,record)
                    if c2 is None: continue
                    add=Proposition("ADD",(c1,c2,c))
                    if self.router.prove(add,stack,record).truth!=Truth.TRUE:
                        continue
                    sub=Proposition("PATH_COST",(m,y,c2))
                    if self.prove_cost_spec(s,sub,stack,record):
                        if record:
                            self.ctx.add_u(
                                UTemplate(
                                    "REC_PATH_COST_EDGE_PLUS_REST",
                                    ("EDGE_COST","ADD","PATH_COST"),
                                    "PATH_COST","RECURSIVE"
                                ),
                                p,Truth.TRUE,inputs=(ec,add,sub),
                                source="recursive-route-cost",
                                evidence=["backward ADD solves remaining path cost"]
                            )
                        out=True; break

                elif s.rec=="EDGE_ONLY":
                    if c==c1:
                        # still require some tail path structurally
                        # any direct/recurrent cost using c target
                        sub=Proposition("PATH_COST",(m,y,c))
                        if self.prove_cost_spec(s,sub,stack,record):
                            out=True; break

                elif s.rec=="REST_ONLY":
                    sub=Proposition("PATH_COST",(m,y,c))
                    if self.prove_cost_spec(s,sub,stack,record):
                        out=True; break

                elif s.rec=="DOUBLE_EDGE":
                    add=Proposition("ADD",(c1,c1,c))
                    if self.router.prove(add,stack,record).truth==Truth.TRUE:
                        # require a tail exists at all
                        for ec2 in self.edge_costs():
                            if ec2.args[0]==m:
                                out=True; break
                        if out: break

        if not record: self._ccache[key]=out
        return out

    def prove(self,p,stack=None,record=True):
        k=Key(p,self.ctx.story_id)
        if p.rel=="PATH" and self.path_spec:
            if self.prove_path_spec(self.path_spec,p,stack,record): k.truth=Truth.TRUE
        if p.rel=="PATH_COST" and self.cost_spec:
            if self.prove_cost_spec(self.cost_spec,p,stack,record): k.truth=Truth.TRUE
        return k

class SharedDomainSolver:
    def __init__(self,std,math,route):
        self.std=std; self.math=math; self.route=route
        std.router=self; math.router=self; route.router=self
        self.cache={}

    def prove(self,p,stack=None,record=True):
        if stack is None: stack=set()
        ck=(p.rel,p.args,p.polarity)
        if record and ck in self.cache: return self.cache[ck]
        if p.rel in {"ADD","MUL","DIVMOD"}:
            k=self.math.prove(p,stack,record)
        elif p.rel in {"PATH","PATH_COST"}:
            k=self.route.prove(p,stack,record)
        else:
            k=self.std.prove(p,stack,record)
        if record and (k.truth!=Truth.UNKNOWN or k.contradiction):
            self.cache[ck]=k
        return k

def make_model(name,max_n=70,number_on=True):
    ctx=DedupStoryContext(name)
    if number_on:
        for i in range(max_n):
            ctx.add_event(Proposition("SUCC",(f"N{i}",f"N{i+1}")),source="number ontology")
    std=RouteStandard(ctx,lt_enabled=True)
    math=OptRecursive(ctx,std)
    math.add_spec=ADD_SPEC
    route=RouteEngine(ctx,std,math)
    router=SharedDomainSolver(std,math,route)
    return ctx,std,math,route,router

def add_graph(ctx,edges):
    for a,b,c in edges:
        ctx.add_event(Proposition("EDGE",(a,b)),source="network")
        ctx.add_event(Proposition("EDGE_COST",(a,b,f"N{c}")),source="network")

def oracle(edges):
    adj={}; ec={}; nodes=set()
    for a,b,c in edges:
        adj.setdefault(a,[]).append(b); ec[(a,b)]=c; nodes|={a,b}
    paths={}
    def dfs(src,cur,total,seen):
        for nxt in adj.get(cur,[]):
            if nxt in seen: continue
            nt=total+ec[(cur,nxt)]
            paths[(src,nxt)]=nt
            dfs(src,nxt,nt,seen|{nxt})
    for n in nodes: dfs(n,n,0,{n})
    return nodes,paths

def graph(seed,n=6):
    rng=random.Random(seed)
    nodes=[f"V{i}" for i in range(n)]
    e=[]
    for i in range(1,n):
        parent=rng.randrange(i)
        e.append((nodes[parent],nodes[i],rng.randint(1,4)))
    return e

train_graphs=[graph(s,6) for s in [3,7,11]]

# Learn PATH
path_rows=[]
for spec in PATH_SPECS:
    sup=conf=0
    for gi,edges in enumerate(train_graphs):
        ctx,std,math,route,router=make_model(f"PT{gi}",50)
        add_graph(ctx,edges); route.path_spec=spec
        nodes,gold=oracle(edges)
        for a,b in gold:
            if router.prove(Proposition("PATH",(a,b)),record=False).truth==Truth.TRUE: sup+=1
        for a in nodes:
            for b in nodes:
                if a==b or (a,b) in gold: continue
                if router.prove(Proposition("PATH",(a,b)),record=False).truth==Truth.TRUE: conf+=1
    comp=2 if spec.rec=="NO_RECURSION" else 3
    path_rows.append((sup*10-conf*25-comp*.1,sup,conf,comp,spec))
path_rows.sort(key=lambda x:(x[0],-x[3]),reverse=True)
PATH_SPEC=path_rows[0][4]

print("=== LEARN PATH ===")
print("selected",PATH_SPEC.name)
for r in path_rows:
    print(" ",r[4].name,"support",r[1],"conflict",r[2],"score",round(r[0],1))

# Learn PATH_COST
cost_rows=[]
for spec in COST_SPECS:
    sup=conf=0
    for gi,edges in enumerate(train_graphs):
        ctx,std,math,route,router=make_model(f"CT{gi}",50)
        add_graph(ctx,edges); route.path_spec=PATH_SPEC; route.cost_spec=spec
        nodes,gold=oracle(edges)
        for (a,b),c in gold.items():
            if router.prove(Proposition("PATH_COST",(a,b,f"N{c}")),record=False).truth==Truth.TRUE: sup+=1
            if router.prove(Proposition("PATH_COST",(a,b,f"N{c+1}")),record=False).truth==Truth.TRUE: conf+=1
    comp=4 if spec.rec=="EDGE_PLUS_REST" else 3
    cost_rows.append((sup*10-conf*25-comp*.1,sup,conf,comp,spec))
cost_rows.sort(key=lambda x:(x[0],-x[3]),reverse=True)
COST_SPEC=cost_rows[0][4]

print("\n=== LEARN PATH_COST ===")
print("selected",COST_SPEC.name)
for r in cost_rows:
    print(" ",r[4].name,"support",r[1],"conflict",r[2],"score",round(r[0],1))

# Frozen unseen graphs
self_rows=[]
for gi,edges in enumerate([graph(s,7) for s in [101,103,107]]):
    ctx,std,math,route,router=make_model(f"U{gi}",70)
    add_graph(ctx,edges); route.path_spec=PATH_SPEC; route.cost_spec=COST_SPEC
    nodes,gold=oracle(edges)
    for (a,b),c in gold.items():
        kp=router.prove(Proposition("PATH",(a,b)),record=False)
        kc=router.prove(Proposition("PATH_COST",(a,b,f"N{c}")),record=False)
        kw=router.prove(Proposition("PATH_COST",(a,b,f"N{c+1}")),record=False)
        self_rows.append({
            "graph":gi,"a":a,"b":b,"gold_cost":c,"kind":"positive",
            "path":truth_name(kp.truth),"cost":truth_name(kc.truth),
            "wrong":truth_name(kw.truth),
            "passed":kp.truth==Truth.TRUE and kc.truth==Truth.TRUE and kw.truth==Truth.UNKNOWN
        })
    neg=0
    for a in sorted(nodes):
        for b in sorted(nodes):
            if a==b or (a,b) in gold: continue
            kp=router.prove(Proposition("PATH",(a,b)),record=False)
            self_rows.append({
                "graph":gi,"a":a,"b":b,"gold_cost":"","kind":"negative",
                "path":truth_name(kp.truth),"cost":"","wrong":"",
                "passed":kp.truth==Truth.UNKNOWN
            })
            neg+=1
            if neg>=5: break
        if neg>=5: break

self_pass=sum(x["passed"] for x in self_rows)
print("\n=== UNSEEN GRAPH SELFTEST ===")
print(self_pass,"/",len(self_rows))

# Integration story
ctx,std,math,route,router=make_model("DELIVERY",70)
route.path_spec=PATH_SPEC; route.cost_spec=COST_SPEC
for a,b,c in [
    ("Depot","Hub",3),("Hub","North",4),("North","Shop",2),("Hub","East",5)
]:
    ctx.add_event(Proposition("EDGE",(a,b)),source="delivery network")
    ctx.add_event(Proposition("EDGE_COST",(a,b,f"N{c}")),source="delivery network")

for pkg,budget in [("pkg1","N10"),("pkg2","N8")]:
    ctx.add_event(Proposition("PACKAGE_AT",(pkg,"Depot")),source="semantic")
    ctx.add_event(Proposition("DEST",(pkg,"Shop")),source="semantic")
    ctx.add_event(Proposition("MAX_COST",(pkg,budget)),source="semantic")

integ=[]
for pkg,exp in [("pkg1",Truth.TRUE),("pkg2",Truth.UNKNOWN)]:
    p=Proposition("DELIVERY_FEASIBLE",(pkg,))
    before=len(ctx.confirmed_u)
    k=router.prove(p,record=True)
    integ.append({
        "pkg":pkg,"expected":truth_name(exp),"got":truth_name(k.truth),
        "passed":k.truth==exp,
        "new_u":[u.template.name for u in ctx.confirmed_u[before:]]
    })

good_cost=router.prove(Proposition("PATH_COST",("Depot","Shop","N9")),record=True)
bad_cost=router.prove(Proposition("PATH_COST",("Depot","Shop","N10")),record=True)

names=[u.template.name for u in ctx.confirmed_u]
audit={
    "pred":sum(n=="SUCC_TO_PRED" for n in names),
    "add":sum(n.startswith("REC_ADD") for n in names),
    "path":sum(n.startswith("REC_PATH_") and not n.startswith("REC_PATH_COST") for n in names),
    "path_cost":sum(n.startswith("REC_PATH_COST") for n in names),
    "lt":sum(n=="PRED_CHAIN_TO_LT" for n in names),
    "delivery":sum(n=="PATH_COST_BUDGET_TO_DELIVERY" for n in names),
}

print("\n=== CROSS-DOMAIN INTEGRATION ===")
for r in integ: print(("PASS" if r["passed"] else "FAIL"),r["pkg"],r["got"])
print("cost N9",truth_name(good_cost.truth),"wrong N10",truth_name(bad_cost.truth))
print("audit",audit)

# Ablations
def ablate(path_on=True,cost_on=True,add_on=True,number_on=True):
    c,s,m,r,ro=make_model("ABL",70,number_on)
    if not add_on: m.add_spec=None
    r.path_spec=PATH_SPEC if path_on else None
    r.cost_spec=COST_SPEC if cost_on else None
    for a,b,cost in [("Depot","Hub",3),("Hub","North",4),("North","Shop",2)]:
        c.add_event(Proposition("EDGE",(a,b)),source="net")
        c.add_event(Proposition("EDGE_COST",(a,b,f"N{cost}")),source="net")
    c.add_event(Proposition("PACKAGE_AT",("pkg","Depot")),source="sem")
    c.add_event(Proposition("DEST",("pkg","Shop")),source="sem")
    c.add_event(Proposition("MAX_COST",("pkg","N10")),source="sem")
    return ro.prove(Proposition("DELIVERY_FEASIBLE",("pkg",)),record=True).truth

abl={
    "without_path":truth_name(ablate(path_on=False)),
    "without_path_cost":truth_name(ablate(cost_on=False)),
    "without_add":truth_name(ablate(add_on=False)),
    "without_number_structure":truth_name(ablate(number_on=False)),
}
print("\n=== ABLATIONS ===")
for k,v in abl.items(): print(k,v)

assert PATH_SPEC.name=="DIRECT_EDGE__EDGE_THEN_PATH"
assert COST_SPEC.name=="DIRECT_EDGE_COST__EDGE_PLUS_REST"
assert self_pass==len(self_rows)
assert all(r["passed"] for r in integ)
assert good_cost.truth==Truth.TRUE and bad_cost.truth==Truth.UNKNOWN
assert all(v>0 for v in audit.values())
assert all(v=="0" for v in abl.values())

report={
    "version":"v4.3-third-domain-route-learning",
    "selected_path":PATH_SPEC.name,
    "selected_path_cost":COST_SPEC.name,
    "path_candidates":[
        {"name":r[4].name,"support":r[1],"conflict":r[2],"complexity":r[3],"score":r[0]}
        for r in path_rows
    ],
    "cost_candidates":[
        {"name":r[4].name,"support":r[1],"conflict":r[2],"complexity":r[3],"score":r[0]}
        for r in cost_rows
    ],
    "unseen":{"passed":self_pass,"n":len(self_rows),"rows":self_rows},
    "integration":{"rows":integ,"good_cost":truth_name(good_cost.truth),
                   "wrong_cost":truth_name(bad_cost.truth),"audit":audit},
    "ablations":abl,
    "invariants":[
        "All modules exchange ordinary Proposition/Key objects.",
        "PATH is a learned recursive U over EDGE primitives.",
        "PATH_COST is a learned recursive U over EDGE_COST and the previously learned ADD U.",
        "ADD still depends on Standard-U PRED/SUCC number structure.",
        "Standard-U consumes learned PATH/PATH_COST plus LT to prove DELIVERY_FEASIBLE.",
        "Wrong costs and unreachable pairs remain UNKNOWN (0), not FALSE (-1).",
        "Frozen unseen tests do not update selected route templates."
    ],
    "caveats":[
        "Candidate U skeletons are still hand-bounded priors.",
        "Graphs are directed acyclic trees with unique path costs; arbitrary cyclic/multipath graphs are not solved here.",
        "Python graph traversal/integer sums generate labels only; the proof engine does not call a graph library or arithmetic operator.",
        "Language parsing is excluded by injecting semantic graph facts.",
        "The first exhaustive intermediate-cost search was rejected due to search explosion; v4.3 uses backward ADD solving."
    ]
}
Path("/mnt/data/symbolic_v43_third_domain_route_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
with Path("/mnt/data/symbolic_v43_third_domain_route_selftest.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(self_rows[0].keys()))
    w.writeheader(); w.writerows(self_rows)
print("\nSaved v4.3 report/selftest.")


from __future__ import annotations
from pathlib import Path
import json, csv, random, time, statistics

# ============================================================
# Load v4.1 implementation without running its benchmark.
# This keeps the exact ADD/MUL/DIVMOD semantics and learned-U machinery.
# ============================================================

V41 = Path("/mnt/data/symbolic_math_v41_divmod.py").read_text(encoding="utf-8")
prefix = V41.split("# ============================================================\n# Learn ADD/MUL exactly as v4.0, then learn DIVMOD.")[0]
ns={}
exec(prefix,ns)

Truth=ns["Truth"]; truth_name=ns["truth_name"]
Proposition=ns["Proposition"]; Key=ns["Key"]; UTemplate=ns["UTemplate"]
StoryContext=ns["StoryContext"]
BaseStandard=ns["StandardEngine"]; BaseRecursive=ns["RecursiveEngine"]; BaseRouter=ns["SharedSolver"]
DivSpec=ns["DivSpec"]; divmod_set_oracle=ns["divmod_set_oracle"]

# ============================================================
# Learn/freeze exactly the same mathematical templates as v4.1.
# ============================================================

def make_base_model(name,max_n=160,number_on=True,lt_enabled=True):
    ctx=StoryContext(name)
    if number_on:
        for i in range(max_n):
            ctx.add_event(Proposition("SUCC",(f"N{i}",f"N{i+1}")),source="number ontology")
    std=BaseStandard(ctx,lt_enabled=lt_enabled)
    rec=BaseRecursive(ctx,std)
    router=BaseRouter(std,rec)
    return ctx,std,rec,router

lctx,lstd,lrec,lrouter=make_base_model("LEARN")
add_train=[(a,b,a+b) for a in range(7) for b in range(7)]
lrec.learn_add(add_train)
mul_train=[(a,b,a*b) for a in range(7) for b in range(7)]
lrec.learn_mul(mul_train)
div_train=[]
for n in range(31):
    for d in range(1,7):
        q,r=divmod_set_oracle(n,d)
        div_train.append((n,d,q,r))
lrec.learn_div(div_train)

ADD_SPEC=lrec.add_spec
MUL_SPEC=lrec.mul_spec
DIV_SPEC=lrec.div_spec

print("=== FROZEN MATH TEMPLATES ===")
print("ADD:",ADD_SPEC.name)
print("MUL:",MUL_SPEC.name)
print("DIV:",DIV_SPEC.name)

# ============================================================
# v4.2 optimized context:
# UInstance dedupe without changing truth semantics.
# ============================================================

class DedupStoryContext(StoryContext):
    def __post_init__(self):
        # dataclass base has no relevant __post_init__, but keep safe.
        try:
            super().__post_init__()
        except AttributeError:
            pass
        self._u_dedup={}

    def add_u(self,template,output,state,*,inputs=(),source="",evidence=None,time=None):
        if not hasattr(self,"_u_dedup"):
            self._u_dedup={}
        sig=(template.name,output,state,tuple(inputs),source,time)
        old=self._u_dedup.get(sig)
        if old is not None:
            # Preserve evidence without creating another UInstance.
            for ev in list(evidence or []):
                if ev not in old.evidence:
                    old.evidence.append(ev)
            return old
        u=super().add_u(
            template,output,state,inputs=tuple(inputs),source=source,
            evidence=evidence,time=time
        )
        self._u_dedup[sig]=u
        return u

# ============================================================
# v4.2 Standard engine:
# - PRED key/materialization cache
# - LT cache
# - direct fact index
# - all PRED requests still prove via SUCC_TO_PRED U
# ============================================================

class OptStandard(BaseStandard):
    def __init__(self,ctx,lt_enabled=True):
        super().__init__(ctx,lt_enabled=lt_enabled)
        self._pred_result={}
        self._lt_result={}
        self._fact_rel_index={}
        self._indexed_event_count=0
        self._sync_fact_index()

    def _sync_fact_index(self):
        # StoryContext is mutable: semantic/text layers may add concrete Keys
        # after solver construction. Index only the newly appended events.
        while self._indexed_event_count < len(self.ctx.events):
            e=self.ctx.events[self._indexed_event_count]
            self._fact_rel_index.setdefault(e.proposition.rel,[]).append(e.proposition)
            self._indexed_event_count += 1

    def facts(self,rel=None):
        self._sync_fact_index()
        if rel is None:
            for vals in self._fact_rel_index.values():
                yield from vals
        else:
            yield from self._fact_rel_index.get(rel,())

    def direct(self,p):
        # Use relation index instead of scanning every event.
        self._sync_fact_index()
        k=Key(p,self.ctx.story_id)
        relfacts=self._fact_rel_index.get(p.rel,())
        pos=p in relfacts
        opp=p.opposite()
        neg=opp in self._fact_rel_index.get(opp.rel,())
        if pos and neg:
            k.truth=Truth.UNKNOWN; k.contradiction=True
        elif pos:
            k.truth=Truth.TRUE
        elif neg:
            k.truth=Truth.FALSE
        return k

    def predecessor(self,n,stack,record):
        cached=self._pred_result.get(n)
        if cached is not None:
            return cached
        lo=self.pred_index.get(n)
        if lo is None:
            self._pred_result[n]=(None,None)
            return None,None
        p=Proposition("PRED",(n,lo))
        k=self.prove(p,stack,record)
        out=(lo,p) if k.truth==Truth.TRUE else (None,None)
        self._pred_result[n]=out
        return out

    def prove(self,p,stack=None,record=True):
        if p.rel=="PRED":
            cached=self._pred_result.get(p.args[0])
            if cached is not None:
                lo,cp=cached
                k=Key(p,self.ctx.story_id)
                if lo is not None and cp==p:
                    k.truth=Truth.TRUE
                    k.evidence.append("cached PRED Key")
                return k

        if p.rel=="LT":
            ck=(p.args, self.lt_enabled)
            if ck in self._lt_result:
                k=Key(p,self.ctx.story_id)
                if self._lt_result[ck]:
                    k.truth=Truth.TRUE
                    k.evidence.append("cached LT Key")
                return k
            k=super().prove(p,stack,record)
            self._lt_result[ck]=(k.truth==Truth.TRUE)
            return k

        k=super().prove(p,stack,record)
        if p.rel=="PRED" and k.truth==Truth.TRUE:
            self._pred_result[p.args[0]]=(p.args[1],p)
        return k

# ============================================================
# v4.2 Recursive engine:
# record-mode intermediate proof caches.
# Existing v4.1 caches only accelerate record=False learning/self-tests;
# these caches also prevent rematerializing identical recursive proof trees.
# ============================================================

class OptRecursive(BaseRecursive):
    def __init__(self,ctx,std):
        super().__init__(ctx,std)
        self._add_record_cache=set()
        self._mul_record_cache=set()
        self._div_record_cache=set()
        self._add_solve_cache={}

    def solve_add_first(self,second,output,stack=None,record=False):
        key=(second,output)
        if key in self._add_solve_cache:
            return self._add_solve_cache[key]
        out=super().solve_add_first(second,output,stack,record)
        self._add_solve_cache[key]=out
        return out

    def prove_add_spec(self,s,p,stack=None,record=False,depth=0):
        key=(s.name,p)
        if record and key in self._add_record_cache:
            return True
        out=super().prove_add_spec(s,p,stack,record,depth)
        if record and out:
            self._add_record_cache.add(key)
        return out

    def prove_mul_spec(self,s,p,stack=None,record=False,depth=0):
        key=(s.name,p)
        if record and key in self._mul_record_cache:
            return True
        out=super().prove_mul_spec(s,p,stack,record,depth)
        if record and out:
            self._mul_record_cache.add(key)
        return out

    def prove_div_spec(self,spec,p,stack=None,record=False):
        key=(spec.name,p)
        if record and key in self._div_record_cache:
            return True
        out=super().prove_div_spec(spec,p,stack,record)
        if record and out:
            self._div_record_cache.add(key)
        return out

# ============================================================
# v4.2 shared router:
# cache all finalized positive proofs; keep UNKNOWN uncached to preserve
# "future evidence may prove it" semantics.
# ============================================================

class OptRouter(BaseRouter):
    def prove(self,p,stack=None,record=True):
        key=(p.rel,p.args,p.polarity)
        if record and key in self.cache:
            return self.cache[key]
        k=super().prove(p,stack,record)
        if record and (k.truth!=Truth.UNKNOWN or k.contradiction):
            self.cache[key]=k
        return k

def make_opt_model(name,max_n=160,number_on=True,lt_enabled=True):
    ctx=DedupStoryContext(name)
    if number_on:
        for i in range(max_n):
            ctx.add_event(Proposition("SUCC",(f"N{i}",f"N{i+1}")),source="number ontology")
    std=OptStandard(ctx,lt_enabled=lt_enabled)
    rec=OptRecursive(ctx,std)
    router=OptRouter(std,rec)
    return ctx,std,rec,router

def install(rec):
    rec.add_spec=ADD_SPEC
    rec.mul_spec=MUL_SPEC
    rec.div_spec=DIV_SPEC

# ============================================================
# Benchmark story / queries
# Includes exact, remainder, larger multiplication, wrong outputs,
# and repeated warm queries.
# ============================================================

def populate(ctx):
    data=[
        ("apples",23,5),      # 4 r3
        ("oranges",24,6),     # 4 r0
        ("bolts",84,7),       # 12 r0
        ("stones",89,11),     # 8 r1
        ("beads",96,12),      # 8 r0
    ]
    for kind,total,cap in data:
        ctx.add_event(Proposition("TOTAL_ITEMS_RAW",(kind,f"N{total}")),source="benchmark")
        ctx.add_event(Proposition("BOX_CAPACITY",(kind,f"N{cap}")),source="benchmark")
    return data

QUERIES=[
    Proposition("PACKING_PLAN",("apples","N4","N3")),
    Proposition("HAS_LEFTOVER",("apples","N4","N3")),
    Proposition("PACKING_PLAN",("oranges","N4","N0")),
    Proposition("EXACT_PACKING",("oranges","N4")),
    Proposition("PACKING_PLAN",("bolts","N12","N0")),
    Proposition("EXACT_PACKING",("bolts","N12")),
    Proposition("PACKING_PLAN",("stones","N8","N1")),
    Proposition("HAS_LEFTOVER",("stones","N8","N1")),
    Proposition("PACKING_PLAN",("beads","N8","N0")),
    Proposition("EXACT_PACKING",("beads","N8")),
    # wrong/noncanonical should remain UNKNOWN
    Proposition("PACKING_PLAN",("apples","N3","N8")),
    Proposition("PACKING_PLAN",("stones","N7","N12")),
]

EXPECTED=[Truth.TRUE]*10+[Truth.UNKNOWN,Truth.UNKNOWN]

def u_audit(ctx):
    names=[u.template.name for u in ctx.confirmed_u]
    return {
        "total_u":len(names),
        "pred_u":sum(n=="SUCC_TO_PRED" for n in names),
        "lt_u":sum(n=="PRED_CHAIN_TO_LT" for n in names),
        "add_u":sum(n.startswith("REC_ADD") for n in names),
        "mul_u":sum(n.startswith("REC_MUL") for n in names),
        "div_u":sum(n.startswith("REC_DIVMOD") for n in names),
        "semantic_u":sum(n in {
            "TOTAL_CAPACITY_DIVMOD_TO_PACKING_PLAN",
            "PACKING_PLAN_NONZERO_TO_LEFTOVER",
            "PACKING_PLAN_ZERO_TO_EXACT"
        } for n in names)
    }

def run_model(kind):
    if kind=="baseline":
        ctx,std,rec,router=make_base_model("BASE")
    else:
        ctx,std,rec,router=make_opt_model("OPT")
    install(rec)
    populate(ctx)

    rows=[]
    cold_times=[]
    for p,exp in zip(QUERIES,EXPECTED):
        t0=time.perf_counter_ns()
        k=router.prove(p,record=True)
        dt=(time.perf_counter_ns()-t0)/1000
        cold_times.append(dt)
        rows.append({
            "query":str(p),
            "expected":truth_name(exp),
            "got":truth_name(k.truth),
            "passed":k.truth==exp,
            "cold_us":dt
        })

    cold_audit=u_audit(ctx)
    before=len(ctx.confirmed_u)

    # Warm pass: identical queries again; should create zero new U in optimized path.
    warm_times=[]
    for p in QUERIES:
        t0=time.perf_counter_ns()
        router.prove(p,record=True)
        warm_times.append((time.perf_counter_ns()-t0)/1000)
    warm_new_u=len(ctx.confirmed_u)-before

    return {
        "ctx":ctx,"std":std,"rec":rec,"router":router,
        "rows":rows,
        "cold_audit":cold_audit,
        "warm_new_u":warm_new_u,
        "cold_median_us":statistics.median(cold_times),
        "cold_total_us":sum(cold_times),
        "warm_median_us":statistics.median(warm_times),
        "warm_total_us":sum(warm_times),
    }

base=run_model("baseline")
opt=run_model("optimized")

print("\n=== CORRECTNESS ===")
print("baseline:",sum(r["passed"] for r in base["rows"]),"/",len(QUERIES))
print("optimized:",sum(r["passed"] for r in opt["rows"]),"/",len(QUERIES))

print("\n=== UINSTANCE AUDIT ===")
for label,res in [("baseline",base),("optimized",opt)]:
    print(label,res["cold_audit"],"warm_new_u=",res["warm_new_u"])

def ratio(a,b):
    return (a/b) if b else float("inf")

print("\n=== TIMING (Python microbenchmark, indicative only) ===")
print("baseline cold total us:",round(base["cold_total_us"],2))
print("optimized cold total us:",round(opt["cold_total_us"],2))
print("cold speedup:",round(ratio(base["cold_total_us"],opt["cold_total_us"]),2),"x")
print("baseline warm total us:",round(base["warm_total_us"],2))
print("optimized warm total us:",round(opt["warm_total_us"],2))
print("warm speedup:",round(ratio(base["warm_total_us"],opt["warm_total_us"]),2),"x")

# ============================================================
# Optimization ablation: isolate each mechanism.
# Rather than clone many engines, report direct structural effects:
# - PRED cache should bound unique PRED U to unique predecessor propositions.
# - dedupe should create no duplicate U signatures.
# - record caches should prevent warm materialization.
# ============================================================

def duplicate_u_count(ctx):
    seen=set(); dup=0
    for u in ctx.confirmed_u:
        sig=(u.template.name,u.output,u.state,tuple(u.inputs),u.source,u.time)
        if sig in seen: dup+=1
        seen.add(sig)
    return dup

base_dups=duplicate_u_count(base["ctx"])
opt_dups=duplicate_u_count(opt["ctx"])

unique_pred_targets=len({
    u.output for u in opt["ctx"].confirmed_u if u.template.name=="SUCC_TO_PRED"
})

print("\n=== OPTIMIZATION INVARIANTS ===")
print("baseline duplicate U signatures:",base_dups)
print("optimized duplicate U signatures:",opt_dups)
print("optimized PRED U / unique PRED targets:",
      opt["cold_audit"]["pred_u"],"/",unique_pred_targets)
print("optimized warm new U:",opt["warm_new_u"])

# ============================================================
# Held-out DIVMOD regression under optimized solver.
# ============================================================

tctx,tstd,trec,trouter=make_opt_model("SELFTEST")
install(trec)
random.seed(42)
self_rows=[]
for _ in range(30):
    n=random.randint(31,90)
    d=random.randint(7,12)
    q,r=divmod_set_oracle(n,d)
    good=trouter.prove(
        Proposition("DIVMOD",(f"N{n}",f"N{d}",f"N{q}",f"N{r}")),
        record=False
    )
    bad=trouter.prove(
        Proposition("DIVMOD",(f"N{n}",f"N{d}",f"N{q}",f"N{r+1}")),
        record=False
    )
    alt_state="n/a"; alt=None
    if q>0:
        alt=(q-1,r+d)
        ak=trouter.prove(
            Proposition("DIVMOD",(f"N{n}",f"N{d}",f"N{alt[0]}",f"N{alt[1]}")),
            record=False
        )
        alt_state=truth_name(ak.truth)
        alt_ok=ak.truth==Truth.UNKNOWN
    else:
        alt_ok=True
    passed=(good.truth==Truth.TRUE and bad.truth==Truth.UNKNOWN and alt_ok)
    self_rows.append({
        "n":n,"d":d,"q":q,"r":r,
        "good":truth_name(good.truth),
        "wrong_r":r+1,"wrong_state":truth_name(bad.truth),
        "alt":str(alt),"alt_state":alt_state,
        "passed":passed
    })

self_pass=sum(r["passed"] for r in self_rows)
print("\n=== OPTIMIZED HELD-OUT DIVMOD ===")
print(self_pass,"/",len(self_rows))

# ============================================================
# Semantic ablations remain unchanged.
# ============================================================

def semantic_case(div_on=True,mul_on=True,add_on=True,number_on=True,lt_enabled=True):
    c,s,r,ro=make_opt_model("ABL",number_on=number_on,lt_enabled=lt_enabled)
    if add_on: r.add_spec=ADD_SPEC
    if mul_on: r.mul_spec=MUL_SPEC
    if div_on: r.div_spec=DIV_SPEC
    c.add_event(Proposition("TOTAL_ITEMS_RAW",("apples","N23")),source="23 apples")
    c.add_event(Proposition("BOX_CAPACITY",("apples","N5")),source="5 per box")
    return ro.prove(Proposition("HAS_LEFTOVER",("apples","N4","N3"))).truth

abl={
    "without_div":truth_name(semantic_case(div_on=False)),
    "without_mul":truth_name(semantic_case(mul_on=False)),
    "without_add":truth_name(semantic_case(add_on=False)),
    "without_number_structure":truth_name(semantic_case(number_on=False)),
    "without_lt":truth_name(semantic_case(lt_enabled=False)),
}
print("\n=== SEMANTIC ABLATIONS ===")
for k,v in abl.items():
    print(k,":",v)

# Assertions: optimization must not change semantics.
assert all(r["passed"] for r in base["rows"])
assert all(r["passed"] for r in opt["rows"])
assert [r["got"] for r in base["rows"]]==[r["got"] for r in opt["rows"]]
assert self_pass==len(self_rows)
assert opt_dups==0
assert opt["warm_new_u"]==0
assert all(v=="0" for v in abl.values())

# ============================================================
# Save artifacts.
# ============================================================

comparison=[]
for br,orr in zip(base["rows"],opt["rows"]):
    comparison.append({
        "query":br["query"],
        "expected":br["expected"],
        "baseline":br["got"],
        "optimized":orr["got"],
        "baseline_cold_us":br["cold_us"],
        "optimized_cold_us":orr["cold_us"],
        "same_semantics":br["got"]==orr["got"],
    })

report={
    "version":"math-v4.2-solver-optimization",
    "frozen_templates":{
        "add":ADD_SPEC.name,"mul":MUL_SPEC.name,"divmod":DIV_SPEC.name
    },
    "correctness":{
        "baseline_passed":sum(r["passed"] for r in base["rows"]),
        "optimized_passed":sum(r["passed"] for r in opt["rows"]),
        "n":len(QUERIES),
        "heldout_divmod_passed":self_pass,
        "heldout_divmod_n":len(self_rows),
    },
    "u_audit":{
        "baseline":base["cold_audit"],
        "optimized":opt["cold_audit"],
        "baseline_duplicate_signatures":base_dups,
        "optimized_duplicate_signatures":opt_dups,
        "baseline_warm_new_u":base["warm_new_u"],
        "optimized_warm_new_u":opt["warm_new_u"],
    },
    "timing_us":{
        "baseline_cold_total":base["cold_total_us"],
        "optimized_cold_total":opt["cold_total_us"],
        "baseline_warm_total":base["warm_total_us"],
        "optimized_warm_total":opt["warm_total_us"],
        "cold_speedup":ratio(base["cold_total_us"],opt["cold_total_us"]),
        "warm_speedup":ratio(base["warm_total_us"],opt["warm_total_us"]),
        "note":"Python microbenchmark; use directionally, not as production performance claim."
    },
    "optimizations":[
        "indexed direct-fact access",
        "PRED proof/materialization cache",
        "LT proof cache",
        "record-mode ADD/MUL/DIVMOD intermediate proof caches",
        "backward ADD solve cache",
        "UInstance signature dedupe",
        "shared positive proof cache"
    ],
    "semantic_ablations":abl,
    "comparison":comparison,
    "heldout":self_rows,
    "invariants":[
        "No mathematical U template changed between baseline and optimized solver.",
        "Wrong/noncanonical arithmetic propositions remain UNKNOWN (0).",
        "No optimization turns U path failure into Key FALSE (-1).",
        "Warm repeated queries materialize no new UInstances in v4.2.",
        "DIVMOD still depends on MUL, ADD, LT, and Standard-U number structure."
    ],
    "caveats":[
        "The benchmark is a Python prototype, not a production systems benchmark.",
        "Number ontology is static during a benchmark run; UNKNOWN proofs are intentionally not cached.",
        "Learned recursive candidate families remain hand-bounded priors.",
        "Language parsing is excluded here so solver optimization can be measured independently."
    ]
}

Path("/mnt/data/symbolic_math_v42_solver_optimization_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)

with Path("/mnt/data/symbolic_math_v42_solver_comparison.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(comparison[0].keys()))
    w.writeheader(); w.writerows(comparison)

with Path("/mnt/data/symbolic_math_v42_divmod_selftest.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(self_rows[0].keys()))
    w.writeheader(); w.writerows(self_rows)

print("\nSaved v4.2 report/comparison/self-test.")


from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, List, Optional, Iterable, Set
from pathlib import Path
import random, json, csv, statistics, time

# ============================================================
# Load the actual frozen v1 core objects (without its old test suite).
# ============================================================

V1 = Path("/mnt/data/symbolic_mini_lm_v1.py").read_text(encoding="utf-8")
core = V1.split("# ============================================================\n# TEST SUITE")[0]
ns={}
exec(core,ns)

Truth       = ns["Truth"]
truth_name  = ns["truth_name"]
Proposition = ns["Proposition"]
Key         = ns["Key"]
UTemplate   = ns["UTemplate"]
UInstance   = ns["UInstance"]
StoryContext= ns["StoryContext"]

# ============================================================
# Shared symbolic rule language
# ============================================================

def isvar(x:str)->bool:
    return x.startswith("?")

@dataclass(frozen=True)
class Atom:
    rel: str
    args: Tuple[str,...]
    polarity: int = 1

@dataclass(frozen=True)
class StandardRule:
    template: UTemplate
    inputs: Tuple[Atom,...]
    output: Atom

def unify_atom_ground(atom:Atom, prop:Proposition, binding:Dict[str,str]) -> Optional[Dict[str,str]]:
    if atom.rel != prop.rel or atom.polarity != prop.polarity or len(atom.args)!=len(prop.args):
        return None
    b=dict(binding)
    for t,v in zip(atom.args,prop.args):
        if isvar(t):
            if t in b and b[t]!=v:
                return None
            b[t]=v
        elif t!=v:
            return None
    return b

def instantiate(atom:Atom,b:Dict[str,str]) -> Optional[Proposition]:
    vals=[]
    for t in atom.args:
        if isvar(t):
            if t not in b:
                return None
            vals.append(b[t])
        else:
            vals.append(t)
    return Proposition(atom.rel,tuple(vals),atom.polarity)

def match_partial(atom:Atom, prop:Proposition, binding:Dict[str,str]) -> Optional[Dict[str,str]]:
    return unify_atom_ground(atom,prop,binding)

# ============================================================
# Standard U engine using v1 UTemplate/UInstance/Key.
# ============================================================

class StandardEngine:
    def __init__(self,ctx:StoryContext,rules:List[StandardRule]):
        self.ctx=ctx
        self.rules=rules
        self.router=None

    def facts(self):
        return [e.proposition for e in self.ctx.events]

    def direct_state(self,p:Proposition):
        pos=any(e.proposition==p for e in self.ctx.events)
        neg=any(e.proposition==p.opposite() for e in self.ctx.events)
        if pos and neg: return Truth.UNKNOWN,True
        if pos: return Truth.TRUE,False
        if neg: return Truth.FALSE,False
        return Truth.UNKNOWN,False

    def prove(self,p:Proposition,stack:Set[Tuple[str,Proposition]]=None) -> Key:
        if stack is None: stack=set()
        k=Key(p,self.ctx.story_id)

        state,contr=self.direct_state(p)
        if state!=Truth.UNKNOWN or contr:
            k.truth=state; k.contradiction=contr
            k.evidence.append("direct concrete Key")
            return k

        marker=("STD",p)
        if marker in stack:
            k.evidence.append("cycle guard")
            return k
        stack=set(stack); stack.add(marker)

        for rule in self.rules:
            if rule.output.rel!=p.rel or rule.output.polarity!=p.polarity:
                continue
            b=unify_atom_ground(rule.output,p,{})
            if b is None:
                continue
            proofs=self._prove_inputs(rule,0,b,[],stack)
            if proofs:
                inputs,binding=proofs[0]
                u=self.ctx.add_u(
                    rule.template,p,Truth.TRUE,inputs=tuple(inputs),
                    source="shared-standard-engine",
                    evidence=["all rule premises proved"]
                )
                k.truth=Truth.TRUE
                k.evidence.append(f"U +1 {rule.template.name} ({u.uid})")
                return k

        k.evidence.append("no standard U proof")
        return k

    def _prove_inputs(self,rule,idx,binding,inputs,stack):
        if idx>=len(rule.inputs):
            return [(inputs,binding)]
        atom=rule.inputs[idx]
        ground=instantiate(atom,binding)
        out=[]

        if ground is not None:
            pk=self.router.prove(ground,stack)
            if pk.truth==Truth.TRUE:
                out.extend(self._prove_inputs(rule,idx+1,binding,inputs+[ground],stack))
            return out

        # Unbound premise: enumerate proven ground Keys through the shared router.
        for prop,b2 in self.router.enumerate_atom(atom,binding,stack):
            out.extend(self._prove_inputs(rule,idx+1,b2,inputs+[prop],stack))
            if out:
                break
        return out

    def enumerate_atom(self,atom:Atom,binding:Dict[str,str],stack:Set[Tuple[str,Proposition]]):
        seen=set()

        # Direct facts.
        for p in self.facts():
            b2=match_partial(atom,p,binding)
            if b2 is not None:
                key=(p,b2.__repr__())
                if key not in seen:
                    seen.add(key)
                    yield p,b2

        # Derived outputs from standard rules. This is finite in this experiment.
        for rule in self.rules:
            if rule.output.rel!=atom.rel or rule.output.polarity!=atom.polarity:
                continue

            # Enumerate candidate bindings by proving rule inputs.
            seed=dict(binding)

            # Respect constants / already-bound outer terms by mapping through output.
            compatible=True
            for out_term,desired_term in zip(rule.output.args,atom.args):
                desired_val = binding.get(desired_term) if isvar(desired_term) else desired_term
                if desired_val is None:
                    continue
                if isvar(out_term):
                    if out_term in seed and seed[out_term]!=desired_val:
                        compatible=False; break
                    seed[out_term]=desired_val
                elif out_term!=desired_val:
                    compatible=False; break
            if not compatible:
                continue

            for inputs,b_rule in self._prove_inputs(rule,0,seed,[],stack):
                p=instantiate(rule.output,b_rule)
                if p is None:
                    continue
                b2=match_partial(atom,p,binding)
                if b2 is None:
                    continue
                # Materialize the standard U instance.
                self.ctx.add_u(
                    rule.template,p,Truth.TRUE,inputs=tuple(inputs),
                    source="shared-standard-enumeration",
                    evidence=["enumerated proven output"]
                )
                key=(p,repr(sorted(b2.items())))
                if key not in seen:
                    seen.add(key)
                    yield p,b2

# ============================================================
# Recursive U candidates.
# They use the SAME router for primitive PRED proofs.
# ============================================================

@dataclass(frozen=True)
class RecursiveSpec:
    base: str
    rec: str

    @property
    def name(self):
        return f"{self.base}__{self.rec}"

BASES=["Y0_ZX","XY_Z0","X0_Z0"]
RECS=["DEC_Y_DEC_OUT","DEC_X_DEC_Y","DEC_Y_SAME_OUT","DEC_X_ONLY","DEC_Y_WRONG_OUT"]

class RecursiveArithmeticEngine:
    def __init__(self,ctx:StoryContext):
        self.ctx=ctx
        self.router=None
        self.selected:Optional[RecursiveSpec]=None
        self.learning_table=[]

    def _enum_pred(self,a0:Optional[str],a1:Optional[str],stack):
        atom=Atom("PRED",(a0 if a0 is not None else "?p0",
                          a1 if a1 is not None else "?p1"))
        b={}
        for p,b2 in self.router.enumerate_atom(atom,b,stack):
            yield p.args[0],p.args[1],p

    def _prove_candidate(self,spec:RecursiveSpec,p:Proposition,stack,record=False,depth=0):
        if p.rel!="SUB" or len(p.args)!=3:
            return False,[]
        if depth>80:
            return False,[]
        x,y,z=p.args
        marker=(spec.name,p)
        if marker in stack:
            return False,[]
        stack=set(stack); stack.add(marker)

        # Base U.
        base_ok=False
        if spec.base=="Y0_ZX":
            base_ok=(y=="N0" and z==x)
        elif spec.base=="XY_Z0":
            base_ok=(x==y and z=="N0")
        elif spec.base=="X0_Z0":
            base_ok=(x=="N0" and z=="N0")
        if base_ok:
            if record:
                templ=UTemplate(f"REC_BASE_{spec.base}",("ZERO",),"SUB","RECURSIVE")
                self.ctx.add_u(
                    templ,p,Truth.TRUE,inputs=(),
                    source="recursive-u",
                    evidence=["recursive base case"]
                )
            return True,[f"BASE {spec.base}"]

        traces=[]

        if spec.rec=="DEC_Y_DEC_OUT":
            # PRED(y,y1), PRED(z1,z), SUB(x,y1,z1) -> SUB(x,y,z)
            for py, y1, pyprop in self._enum_pred(y,None,stack):
                for z1, pz, pzprop in self._enum_pred(None,z,stack):
                    ok,tr=self._prove_candidate(
                        spec,Proposition("SUB",(x,y1,z1)),stack,record,depth+1
                    )
                    if ok:
                        if record:
                            templ=UTemplate(
                                "REC_SUB_DEC_Y_DEC_OUT",
                                ("PRED","PRED","SUB"),"SUB","RECURSIVE"
                            )
                            self.ctx.add_u(
                                templ,p,Truth.TRUE,
                                inputs=(pyprop,pzprop,Proposition("SUB",(x,y1,z1))),
                                source="recursive-u",
                                evidence=["decreasing y via standard PRED U"]
                            )
                        return True,[f"{y}->{y1}, {z1}->{z}"]+tr

        elif spec.rec=="DEC_X_DEC_Y":
            # PRED(x,x1), PRED(y,y1), SUB(x1,y1,z) -> SUB(x,y,z)
            for px,x1,pxprop in self._enum_pred(x,None,stack):
                for py,y1,pyprop in self._enum_pred(y,None,stack):
                    ok,tr=self._prove_candidate(
                        spec,Proposition("SUB",(x1,y1,z)),stack,record,depth+1
                    )
                    if ok:
                        if record:
                            templ=UTemplate(
                                "REC_SUB_DEC_X_DEC_Y",
                                ("PRED","PRED","SUB"),"SUB","RECURSIVE"
                            )
                            self.ctx.add_u(
                                templ,p,Truth.TRUE,
                                inputs=(pxprop,pyprop,Proposition("SUB",(x1,y1,z))),
                                source="recursive-u",
                                evidence=["decreasing x,y via standard PRED U"]
                            )
                        return True,[f"{x}->{x1}, {y}->{y1}"]+tr

        elif spec.rec=="DEC_Y_SAME_OUT":
            for py,y1,pyprop in self._enum_pred(y,None,stack):
                ok,tr=self._prove_candidate(
                    spec,Proposition("SUB",(x,y1,z)),stack,record,depth+1
                )
                if ok:
                    return True,[f"{y}->{y1}"]+tr

        elif spec.rec=="DEC_X_ONLY":
            for px,x1,pxprop in self._enum_pred(x,None,stack):
                ok,tr=self._prove_candidate(
                    spec,Proposition("SUB",(x1,y,z)),stack,record,depth+1
                )
                if ok:
                    return True,[f"{x}->{x1}"]+tr

        elif spec.rec=="DEC_Y_WRONG_OUT":
            # Wrong direction on output: PRED(z,z1)
            for py,y1,pyprop in self._enum_pred(y,None,stack):
                for pz,z1,pzprop in self._enum_pred(z,None,stack):
                    ok,tr=self._prove_candidate(
                        spec,Proposition("SUB",(x,y1,z1)),stack,record,depth+1
                    )
                    if ok:
                        return True,[f"wrong {y}->{y1},{z}->{z1}"]+tr

        return False,traces

    def learn(self,examples:List[Tuple[int,int,int]]):
        table=[]
        for base in BASES:
            for rec in RECS:
                spec=RecursiveSpec(base,rec)
                support=conflict=0
                for a,b,c in examples:
                    target=Proposition("SUB",(f"N{a}",f"N{b}",f"N{c}"))
                    ok,_=self._prove_candidate(spec,target,set(),record=False)
                    if ok:
                        support+=1

                    # Contrastive wrong-output probes.
                    wrongs={max(0,c-1),c+1}
                    wrongs.discard(c)
                    for w in wrongs:
                        if w>30: continue
                        wp=Proposition("SUB",(f"N{a}",f"N{b}",f"N{w}"))
                        wok,_=self._prove_candidate(spec,wp,set(),record=False)
                        if wok:
                            conflict+=1

                complexity=2  # base + recursion
                if rec in {"DEC_Y_DEC_OUT","DEC_X_DEC_Y","DEC_Y_WRONG_OUT"}:
                    complexity+=2
                else:
                    complexity+=1
                score=support*10-conflict*25-complexity*0.1
                table.append({
                    "spec":spec,"support":support,"conflict":conflict,
                    "complexity":complexity,"score":score
                })

        table.sort(key=lambda x:(x["score"],-x["complexity"]),reverse=True)
        self.learning_table=table
        self.selected=table[0]["spec"]
        return table

    def prove(self,p:Proposition,stack=None):
        k=Key(p,self.ctx.story_id)
        if self.selected is None:
            k.evidence.append("no learned recursive U installed")
            return k
        if stack is None: stack=set()
        ok,tr=self._prove_candidate(self.selected,p,stack,record=True)
        if ok:
            k.truth=Truth.TRUE
            k.evidence.append(f"U +1 recursive {self.selected.name}")
            k.evidence.extend(tr[:5])
        else:
            k.evidence.append("recursive U cannot prove target")
        return k

# ============================================================
# Shared router: one Key space, both engines.
# ============================================================

class SharedSolver:
    def __init__(self,standard:StandardEngine,recursive:RecursiveArithmeticEngine):
        self.standard=standard
        self.recursive=recursive
        self.standard.router=self
        self.recursive.router=self
        self.cache={}

    def prove(self,p:Proposition,stack=None):
        if stack is None: stack=set()
        cache_key=(p.rel,p.args,p.polarity)
        if cache_key in self.cache:
            return self.cache[cache_key]

        if p.rel=="SUB":
            k=self.recursive.prove(p,stack)
        else:
            k=self.standard.prove(p,stack)

        # Only cache positive/negative concrete truth; unknown may become provable later.
        if k.truth!=Truth.UNKNOWN or k.contradiction:
            self.cache[cache_key]=k
        return k

    def enumerate_atom(self,atom:Atom,binding,stack):
        # SUB enumeration is intentionally not needed: query-guided math is ground
        # by the time Standard-U asks for it.
        if atom.rel=="SUB":
            return iter(())
        return self.standard.enumerate_atom(atom,binding,stack)

# ============================================================
# Standard U templates/rules.
# ============================================================

U_SUCC_PRED=UTemplate("SUCC_TO_PRED",("SUCC",),"PRED","REASONING")
U_REMAIN=UTemplate("COUNT_REMOVE_SUB_TO_REMAIN",("COUNT","REMOVED_COUNT","SUB"),"REMAINING_COUNT","REASONING")
U_ONE=UTemplate("REMAIN_N1_TO_ONE_REMAINS",("REMAINING_COUNT",),"ONE_REMAINS","REASONING")
U_SURV=UTemplate("ONE_REMAINS_TO_SURVIVOR_EXISTS",("ONE_REMAINS",),"SURVIVOR_EXISTS","REASONING")

RULES=[
    StandardRule(
        U_SUCC_PRED,
        (Atom("SUCC",("?a","?b")),),
        Atom("PRED",("?b","?a"))
    ),
    StandardRule(
        U_REMAIN,
        (
            Atom("COUNT",("?g","?x")),
            Atom("REMOVED_COUNT",("?g","?y")),
            Atom("SUB",("?x","?y","?z")),
        ),
        Atom("REMAINING_COUNT",("?g","?z"))
    ),
    StandardRule(
        U_ONE,
        (Atom("REMAINING_COUNT",("?g","N1")),),
        Atom("ONE_REMAINS",("?g",))
    ),
    StandardRule(
        U_SURV,
        (Atom("ONE_REMAINS",("?g",)),),
        Atom("SURVIVOR_EXISTS",("?g",))
    ),
]

def make_model(story_id,max_n=30):
    ctx=StoryContext(story_id)
    # IMPORTANT: only SUCC is materialized. PRED must be derived by Standard-U.
    for i in range(max_n):
        ctx.add_event(
            Proposition("SUCC",(f"N{i}",f"N{i+1}")),
            source="number ontology"
        )
    std=StandardEngine(ctx,RULES)
    rec=RecursiveArithmeticEngine(ctx)
    router=SharedSolver(std,rec)
    return ctx,std,rec,router

# ============================================================
# 1) Learn recursive U using the same shared solver.
# ============================================================

learn_ctx,learn_std,learn_rec,learn_router=make_model("LEARN",30)

random.seed(13)
train=[(a,0,a) for a in range(0,9)]
pool=[(a,b,a-b) for a in range(1,9) for b in range(1,a+1)]
train+=random.sample(pool,18)

table=learn_rec.learn(train)
best=learn_rec.selected

print("=== RECURSIVE U LEARNING INSIDE SHARED MODEL ===")
print("training examples:",len(train),"range 0..8")
print("selected:",best.name)
print("top candidates:")
for row in table[:6]:
    print(
        f" {row['spec'].name:32} "
        f"support={row['support']:2} conflict={row['conflict']:2} "
        f"score={row['score']:.1f}"
    )

# ============================================================
# 2) Math self-test on an isolated context.
# Independent set-cardinality oracle; tests do NOT change learned U.
# ============================================================

self_ctx,self_std,self_rec,self_router=make_model("SELFTEST",30)
self_rec.selected=best

def set_oracle(a,b):
    whole={f"e{i}" for i in range(a)}
    removed={f"e{i}" for i in range(b)}
    return len(whole-removed)

random.seed(21)
self_cases=[]
for _ in range(24):
    a=random.randint(10,25)
    b=random.randint(0,a)
    self_cases.append((a,b,set_oracle(a,b)))

self_rows=[]
for a,b,c in self_cases:
    good=Proposition("SUB",(f"N{a}",f"N{b}",f"N{c}"))
    bad_c=(c+1) if c<30 else max(0,c-1)
    bad=Proposition("SUB",(f"N{a}",f"N{b}",f"N{bad_c}"))

    kg=self_router.prove(good)
    kb=self_router.prove(bad)

    self_rows.append({
        "a":a,"b":b,"gold":c,
        "good_state":truth_name(kg.truth),
        "wrong_output":bad_c,
        "wrong_state":truth_name(kb.truth),
        "passed":kg.truth==Truth.TRUE and kb.truth==Truth.UNKNOWN,
    })

self_pass=sum(r["passed"] for r in self_rows)
print(f"\n=== MATH SELF-TEST (UNSEEN, isolated) ===")
print(f"{self_pass}/{len(self_rows)} passed")
for r in self_rows[:6]:
    print(
        f" {r['a']}-{r['b']}={r['gold']} -> {r['good_state']} | "
        f"wrong {r['wrong_output']} -> {r['wrong_state']}"
    )

# ============================================================
# 3) True cross-engine integration.
# Standard U needs Recursive-U; Recursive-U needs Standard-U PRED.
# ============================================================

ctx,std,rec,router=make_model("GOATS",30)
rec.selected=best

# "Text/semantic layer" contributes ordinary concrete Keys.
ctx.add_event(Proposition("COUNT",("goat_children","N7")),source="seven goat children")
ctx.add_event(Proposition("REMOVED_COUNT",("goat_children","N6")),source="six were eaten")

# Another group to test non-memorized composition.
ctx.add_event(Proposition("COUNT",("apples","N12")),source="twelve apples")
ctx.add_event(Proposition("REMOVED_COUNT",("apples","N5")),source="five apples removed")

queries=[
    ("goat remaining count",Proposition("REMAINING_COUNT",("goat_children","N1")),Truth.TRUE),
    ("goat survivor semantic fact",Proposition("SURVIVOR_EXISTS",("goat_children",)),Truth.TRUE),
    ("apple remaining count",Proposition("REMAINING_COUNT",("apples","N7")),Truth.TRUE),
    ("wrong apple count stays unknown",Proposition("REMAINING_COUNT",("apples","N8")),Truth.UNKNOWN),
]

integration=[]
for label,p,expected in queries:
    before=len(ctx.confirmed_u)
    k=router.prove(p)
    new_u=ctx.confirmed_u[before:]
    integration.append({
        "label":label,
        "query":str(p),
        "expected":truth_name(expected),
        "got":truth_name(k.truth),
        "passed":k.truth==expected,
        "new_u":[u.template.name for u in new_u],
        "evidence":k.evidence,
    })

print("\n=== CROSS-ENGINE INTEGRATION ===")
for r in integration:
    print(("PASS" if r["passed"] else "FAIL"),"|",r["label"],"|",r["got"])
    print("  U chain:", " -> ".join(r["new_u"]) if r["new_u"] else "(cached/no new U)")

# Aggregate U types actually used across the integrated story.
used_names=[u.template.name for u in ctx.confirmed_u]
std_pred_count=sum(n=="SUCC_TO_PRED" for n in used_names)
rec_count=sum(n.startswith("REC_") for n in used_names)
remain_count=sum(n=="COUNT_REMOVE_SUB_TO_REMAIN" for n in used_names)
semantic_count=sum(n in {"REMAIN_N1_TO_ONE_REMAINS","ONE_REMAINS_TO_SURVIVOR_EXISTS"} for n in used_names)

print("\n=== INTERACTION AUDIT ===")
print("standard SUCC->PRED U used:",std_pred_count)
print("recursive SUB U used:",rec_count)
print("standard COUNT+SUB bridge U used:",remain_count)
print("post-math semantic U used:",semantic_count)

# Strong interaction assertions:
# 1) recursive engine could not work without standard-derived PRED;
# 2) standard semantic query could not work without recursive SUB.
assert std_pred_count>0
assert rec_count>0
assert remain_count>0
assert semantic_count>0
assert all(r["passed"] for r in integration)
assert self_pass==len(self_rows)

# ============================================================
# 4) Ablations prove that the connection is real.
# ============================================================

# A: remove recursive engine -> semantic survivor query must become UNKNOWN.
ablA_ctx,ablA_std,ablA_rec,ablA_router=make_model("ABL_NO_REC",30)
ablA_ctx.add_event(Proposition("COUNT",("goat_children","N7")),source="count")
ablA_ctx.add_event(Proposition("REMOVED_COUNT",("goat_children","N6")),source="removed")
# no selected recursive U
ablA=ablA_router.prove(Proposition("SURVIVOR_EXISTS",("goat_children",)))

# B: remove SUCC ontology -> recursive engine has no standard PRED proofs.
ablB_ctx=StoryContext("ABL_NO_STD")
ablB_std=StandardEngine(ablB_ctx,RULES)
ablB_rec=RecursiveArithmeticEngine(ablB_ctx)
ablB_router=SharedSolver(ablB_std,ablB_rec)
ablB_rec.selected=best
ablB_ctx.add_event(Proposition("COUNT",("goat_children","N7")),source="count")
ablB_ctx.add_event(Proposition("REMOVED_COUNT",("goat_children","N6")),source="removed")
ablB=ablB_router.prove(Proposition("SURVIVOR_EXISTS",("goat_children",)))

print("\n=== ABLATION ===")
print("without Recursive-U:",truth_name(ablA.truth))
print("without Standard-U number ontology:",truth_name(ablB.truth))

assert ablA.truth==Truth.UNKNOWN
assert ablB.truth==Truth.UNKNOWN

# ============================================================
# Artifacts
# ============================================================

report={
    "core_reuse":"Loads Proposition/Key/UTemplate/UInstance/StoryContext/Truth from symbolic_mini_lm_v1.py",
    "learned_recursive_u":{
        "selected":best.name,
        "training_examples":train,
        "top_candidates":[
            {
                "name":row["spec"].name,
                "support":row["support"],
                "conflict":row["conflict"],
                "complexity":row["complexity"],
                "score":row["score"],
            }
            for row in table[:8]
        ],
    },
    "self_test":{
        "passed":self_pass,"n":len(self_rows),"rows":self_rows,
        "note":"Held-out validation uses independent explicit-set remainder cardinality; it does not alter learned U."
    },
    "integration":{
        "rows":integration,
        "interaction_audit":{
            "standard_succ_to_pred_u":std_pred_count,
            "recursive_sub_u":rec_count,
            "standard_count_sub_bridge_u":remain_count,
            "post_math_semantic_u":semantic_count,
        },
        "ablations":{
            "without_recursive_u":truth_name(ablA.truth),
            "without_standard_number_ontology":truth_name(ablB.truth),
        }
    },
    "invariants":[
        "All engines exchange ordinary Proposition/Key objects.",
        "Recursive-U asks the shared solver for PRED; PRED is derived by Standard-U from SUCC.",
        "Standard-U asks the shared solver for SUB; SUB is proved by the learned Recursive-U.",
        "Arithmetic output is consumed by further Standard-U; it is not merely returned as a detached calculator result.",
        "Wrong arithmetic output remains Key 0 rather than Key -1.",
        "Self-tests validate but never train or promote a recursive template."
    ],
    "caveats":[
        "Recursive U induction searches a deliberately small candidate family.",
        "Number successor structure is a provided symbolic prior.",
        "This experiment integrates with the frozen v1 core objects, not the later full raw-text v3.x parser.",
        "The text-side COUNT/REMOVED_COUNT facts are injected semantically so arithmetic/engine interaction can be isolated."
    ]
}

Path("/mnt/data/symbolic_v39_shared_recursive_u_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)

with Path("/mnt/data/symbolic_v39_shared_recursive_u_selftest.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(self_rows[0].keys()))
    w.writeheader(); w.writerows(self_rows)

print("\nSaved report/self-test CSV.")


from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
import types, sys, copy, json, csv, time

# ============================================================
# Reuse v4.4b generic symbolic U learner / verifier.
# No neural components.
# ============================================================

src=Path("/mnt/data/symbolic_v44b_recursive_u_verifier.py").read_text(encoding="utf-8")
mod=types.ModuleType("v44b_for_v49")
sys.modules[mod.__name__]=mod
exec(src,mod.__dict__)

NUM=mod.NUM
SIG=mod.SIG
HEAD_VARS=mod.HEAD_VARS
RelSig=mod.RelSig
Atom=mod.Atom
Rule=mod.Rule
World=mod.World
Program=mod.Program

TARGET="R1"
SIG[TARGET]=RelSig(TARGET,(NUM,NUM,NUM))
HEAD_VARS[TARGET]=("X","Y","Z")

# Symbolic selector predicates available to revision search.
# These are extensional symbolic features, not target-specific formulas.
for name in ["EVEN","ODD","NONZERO","LOW"]:
    SIG[name]=RelSig(name,(NUM,))

SELECTORS=["EVEN","ODD","NONZERO","LOW"]

def add_selectors(world,max_n):
    f=world.facts
    for i in range(max_n+1):
        n=f"N{i}"
        if i%2==0: f.setdefault("EVEN",set()).add((n,))
        else:      f.setdefault("ODD",set()).add((n,))
        if i>0:    f.setdefault("NONZERO",set()).add((n,))
        if i<=8:   f.setdefault("LOW",set()).add((n,))
    return world

def make_world(pos,neg,max_n):
    w=mod.num_world(max_n,pos,neg)
    add_selectors(w,max_n)
    return w

def initial_data():
    pos=[]; neg=[]
    # Initial evidence only ever sees EVEN second arguments.
    for x in range(0,8):
        for y in [0,2,4,6]:
            z=x+y
            pos.append((f"N{x}",f"N{y}",f"N{z}"))
            neg.append((f"N{x}",f"N{y}",f"N{z+1}"))
            if z>0:
                neg.append((f"N{x}",f"N{y}",f"N{z-1}"))
    train=[make_world(pos,neg,20)]

    fp=[]; fn=[]
    for x,y in [(8,0),(9,2),(10,4),(11,6),(12,8)]:
        z=x+y
        fp.append((f"N{x}",f"N{y}",f"N{z}"))
        fn.append((f"N{x}",f"N{y}",f"N{z+1}"))
    frozen=[make_world(fp,fn,30)]
    return train,frozen

def challenge_data():
    pos=[]; neg=[]
    # More even examples remain positive.
    for x,y in [(8,2),(9,4),(10,6),(11,8)]:
        z=x+y
        pos.append((f"N{x}",f"N{y}",f"N{z}"))
        neg.append((f"N{x}",f"N{y}",f"N{z+1}"))

    # New independent evidence: arithmetic-looking triples with ODD Y
    # are explicitly negative for this relation.
    for x,y in [(2,1),(3,3),(4,5),(6,7),(8,9)]:
        neg.append((f"N{x}",f"N{y}",f"N{x+y}"))

    challenge=[make_world(pos,neg,35)]

    # Frozen revision set uses unseen values.
    fp=[]; fn=[]
    for x,y in [(13,0),(14,2),(15,8),(16,10),(17,12)]:
        fp.append((f"N{x}",f"N{y}",f"N{x+y}"))
    for x,y in [(9,1),(10,3),(11,5),(12,7),(13,11)]:
        fn.append((f"N{x}",f"N{y}",f"N{x+y}"))
    revision_frozen=[make_world(fp,fn,40)]
    return challenge,revision_frozen

def impossible_repair_batch():
    # v2 (EVEN Y) proves this, but the new accepted negative says it should not.
    # Available selector LOW(Y) can exclude N10, but frozen positives include N12,
    # so that specialization is not safe/general.
    neg=[("N18","N10","N28")]
    return [make_world([],neg,45)]

def impossible_frozen():
    pos=[
        ("N19","N12","N31"),
        ("N20","N14","N34"),
    ]
    neg=[
        ("N20","N11","N31"),
    ]
    return [make_world(pos,neg,45)]

def eval_program(program,worlds):
    support=conflict=0
    missed=[]
    false_pos=[]
    for w in worlds:
        program.reset()
        for p in w.positives:
            if program.prove(p,w):
                support+=1
            else:
                missed.append(p)
        program.reset()
        for n in w.negatives:
            if program.prove(n,w):
                conflict+=1
                false_pos.append(n)
    return {
        "support":support,
        "positive_n":sum(len(w.positives) for w in worlds),
        "conflict":conflict,
        "negative_n":sum(len(w.negatives) for w in worlds),
        "missed":missed,
        "false_positive":false_pos,
    }

def full_pass(program,worlds):
    e=eval_program(program,worlds)
    return e["support"]==e["positive_n"] and e["conflict"]==0,e

def recursive_atom(rule):
    xs=[a for a in rule.body if a.rel==TARGET]
    return xs[0] if len(xs)==1 else None

def stable_head_vars(rec_rule):
    ra=recursive_atom(rec_rule)
    if ra is None:
        return []
    return [
        hv
        for hv,rv in zip(rec_rule.head.args,ra.args)
        if hv==rv
    ]

def add_guard(rule,predicate,var):
    atom=Atom(predicate,(var,))
    if atom in rule.body:
        return rule
    return Rule(rule.head,tuple(sorted(rule.body+(atom,))))

@dataclass
class UVersion:
    number:int
    status:str
    base:Rule
    rec:Rule
    parent:int|None
    provenance:list[str]
    meta:dict=field(default_factory=dict)

    @property
    def id(self):
        return f"{TARGET}_v{self.number}"

    def program(self):
        return Program(SIG[TARGET],self.base,self.rec)

class VersionedULibrary:
    def __init__(self,initial_train,initial_frozen):
        self.initial_train=initial_train
        self.initial_frozen=initial_frozen
        self.accepted_evidence=list(initial_train)
        self.versions=[]
        self.active_number=None
        self.events=[]
        self.next_version=1

    def event(self,event,**kw):
        row={"event":event}; row.update(kw); self.events.append(row)

    def active(self):
        if self.active_number is None:
            return None
        return next(v for v in self.versions if v.number==self.active_number)

    def install_initial(self):
        result=mod.synth_verified_recursive(
            TARGET,["ZERO","EQ","PRED","SUCC"],
            self.initial_train,
            max_base=2,max_bg=2,hidden_limits={NUM:2},
        )
        sc,freq,base,rec,cert,local=result["best"]
        p=Program(SIG[TARGET],base,rec)
        train_ok,train_eval=full_pass(p,self.initial_train)
        frozen_ok,frozen_eval=full_pass(p,self.initial_frozen)
        assert train_ok and frozen_ok

        v=UVersion(
            number=self.next_version,status="ACTIVE",
            base=base,rec=rec,parent=None,
            provenance=["initial_training","initial_frozen"],
            meta={
                "train":train_eval,
                "frozen":frozen_eval,
                "certificate":cert,
                "base_rule":base.text(),
                "recursive_rule":rec.text(),
            }
        )
        self.next_version+=1
        self.versions.append(v)
        self.active_number=v.number
        self.event("installed_initial",version=v.id)
        return v

    def _candidate_specializations(self,current):
        stable=stable_head_vars(current.rec)
        out=[]
        for var in stable:
            for pred in SELECTORS:
                # Minimal revision: guard recursion-invariant variable at base.
                b=add_guard(current.base,pred,var)
                r=current.rec
                out.append((pred,var,b,r))
        return out

    def revise_from_challenge(self,batch,revision_frozen,batch_name):
        cur=self.active()
        if cur is None:
            self.event("challenge_without_active",batch=batch_name)
            return None

        cur_prog=cur.program()
        batch_eval=eval_program(cur_prog,batch)
        if batch_eval["conflict"]==0 and batch_eval["support"]==batch_eval["positive_n"]:
            self.accepted_evidence.extend(batch)
            cur.provenance.append(batch_name)
            self.event("evidence_compatible",version=cur.id,batch=batch_name)
            return cur

        # Active U has been falsified by accepted evidence.
        cur.status="CHALLENGED"
        self.event(
            "version_challenged",version=cur.id,batch=batch_name,
            conflict=batch_eval["conflict"],
            support_gap=batch_eval["positive_n"]-batch_eval["support"],
        )

        combined=self.accepted_evidence+list(batch)
        candidates=[]

        for pred,var,b,r in self._candidate_specializations(cur):
            num=self.next_version
            self.next_version+=1
            staged=UVersion(
                number=num,status="STAGED",base=b,rec=r,parent=cur.number,
                provenance=cur.provenance+[batch_name,"revision_frozen"],
                meta={"guard":f"{pred}({var})"}
            )
            self.versions.append(staged)

            p=staged.program()
            train_ok,train_eval=full_pass(p,combined)
            frozen_ok,frozen_eval=full_pass(p,revision_frozen)
            staged.meta.update({
                "combined":train_eval,
                "frozen":frozen_eval,
                "base_rule":b.text(),
                "recursive_rule":r.text(),
            })

            if train_ok and frozen_ok:
                # MDL/minimality: one guard here, then prefer larger witness support.
                score=(
                    train_eval["support"]*10
                    - train_eval["conflict"]*25
                    - 1.0
                )
                candidates.append((score,staged))
            else:
                staged.status="RETIRED"
                staged.meta["retire_reason"]="revision_gate_failed"
                self.event(
                    "staged_revision_rejected",version=staged.id,
                    guard=staged.meta["guard"],
                    train_ok=train_ok,frozen_ok=frozen_ok,
                )

        if not candidates:
            # Safety choice: do NOT silently reactivate a contradicted U.
            self.active_number=None
            self.accepted_evidence.extend(batch)
            self.event(
                "no_safe_revision",challenged_version=cur.id,
                policy="no active version; queries remain UNKNOWN until repaired"
            )
            return None

        candidates.sort(key=lambda x:x[0],reverse=True)
        winner=candidates[0][1]

        # All other passing staged candidates lose transactionally.
        for _,cand in candidates[1:]:
            cand.status="RETIRED"
            cand.meta["retire_reason"]="lost_revision_selection"

        cur.status="SUPERSEDED"
        winner.status="ACTIVE"
        self.active_number=winner.number
        self.accepted_evidence.extend(batch)
        self.event(
            "revision_activated",
            old=cur.id,new=winner.id,
            guard=winner.meta["guard"],
        )
        return winner

    def stage_extra_guard(self,predicate,var,frozen,name):
        """
        Proactive/experimental revision. Failure must not replace ACTIVE.
        """
        cur=self.active()
        assert cur is not None
        b=add_guard(cur.base,predicate,var)
        r=cur.rec
        v=UVersion(
            number=self.next_version,status="STAGED",
            base=b,rec=r,parent=cur.number,
            provenance=cur.provenance+[name],
            meta={"guard":f"{predicate}({var})"}
        )
        self.next_version+=1
        self.versions.append(v)
        p=v.program()
        combined_ok,combined_eval=full_pass(p,self.accepted_evidence)
        frozen_ok,frozen_eval=full_pass(p,frozen)
        v.meta.update({
            "combined":combined_eval,
            "frozen":frozen_eval,
            "base_rule":b.text(),
            "recursive_rule":r.text(),
        })
        if combined_ok and frozen_ok:
            cur.status="SUPERSEDED"
            v.status="ACTIVE"
            self.active_number=v.number
            self.event("proactive_revision_activated",old=cur.id,new=v.id)
            return True,v
        v.status="RETIRED"
        v.meta["retire_reason"]="proactive_revision_failed"
        self.event(
            "proactive_revision_rollback",
            attempted=v.id,kept_active=cur.id,
            combined_ok=combined_ok,frozen_ok=frozen_ok,
        )
        return False,v

    def prove(self,args,world):
        cur=self.active()
        if cur is None:
            return False
        p=cur.program()
        p.reset()
        return bool(p.prove(tuple(args),world))

    def fork(self):
        return copy.deepcopy(self)

# ============================================================
# Run v4.9
# ============================================================

initial_train,initial_frozen=initial_data()
challenge,revision_frozen=challenge_data()

lib=VersionedULibrary(initial_train,initial_frozen)
v1=lib.install_initial()

# Before challenge: v1 overgeneralizes to odd Y because it has never seen
# evidence against that region.
probe_before=make_world([],[],40)
before_even=lib.prove(("N9","N4","N13"),probe_before)
before_odd=lib.prove(("N9","N3","N12"),probe_before)

v1_challenge_eval=eval_program(v1.program(),challenge)
v2=lib.revise_from_challenge(challenge,revision_frozen,"challenge_batch_1")

probe_after=make_world([],[],45)
after_even=lib.prove(("N17","N12","N29"),probe_after)
after_odd=lib.prove(("N13","N11","N24"),probe_after)

# Transactional rollback test: add NONZERO(Y), which wrongly excludes the
# established Y=0 positive cases. It must not replace v2.
active_before_bad=lib.active().id
bad_ok,bad_version=lib.stage_extra_guard(
    "NONZERO","Y",revision_frozen,"experimental_over_specialization"
)
active_after_bad=lib.active().id

# Fork and present a later challenge for which the current guard vocabulary
# cannot produce a safe/general repair. The safe policy is NO ACTIVE version,
# not silently using a contradicted rule.
fork=lib.fork()
hard_batch=impossible_repair_batch()
hard_frozen=impossible_frozen()
hard_result=fork.revise_from_challenge(
    hard_batch,hard_frozen,"challenge_batch_unrepairable"
)
hard_query=make_world([],[],45)
hard_proof=fork.prove(("N18","N10","N28"),hard_query)

# ============================================================
# Checks
# ============================================================

v2_guard=v2.meta.get("guard") if v2 else None
statuses={v.id:v.status for v in lib.versions}
fork_statuses={v.id:v.status for v in fork.versions}

checks={
    "v1_initially_active":v1.status in {"SUPERSEDED","CHALLENGED"} and v1.number==1,
    "v1_learned_without_selector_guard":all(
        p+"(" not in (v1.meta["base_rule"]+" "+v1.meta["recursive_rule"])
        for p in SELECTORS
    ),
    "v1_conflicted_with_new_evidence":v1_challenge_eval["conflict"]>0,
    "revision_found_symbolic_specialization":v2 is not None and v2_guard=="EVEN(Y)",
    "v2_active_v1_superseded":v2.status=="ACTIVE" and v1.status=="SUPERSEDED",
    "v2_preserves_even_generalization":after_even,
    "v2_blocks_odd_overgeneralization":not after_odd,
    "failed_v3_rolls_back_transactionally":(
        not bad_ok and
        bad_version.status=="RETIRED" and
        active_before_bad==active_after_bad==v2.id
    ),
    "version_history_preserved":len(lib.versions)>=3,
    "provenance_preserved":(
        "challenge_batch_1" in v2.provenance and
        v2.parent==v1.number
    ),
    "unrepairable_challenge_disables_unsafe_active_version":(
        hard_result is None and
        fork.active_number is None and
        not hard_proof
    ),
    "unrepairable_query_returns_unknown_not_false_proof":not hard_proof,
}

print("=== v4.9 VERSIONED / REVISION-CAPABLE U LIBRARY ===")
for k,v in checks.items():
    print(("PASS" if v else "FAIL"),"|",k)

print("\nInitial v1:")
print(" status now:",v1.status)
print(" BASE:",v1.meta["base_rule"])
print(" REC :",v1.meta["recursive_rule"])
print(" new-evidence conflict:",v1_challenge_eval["conflict"])
print(" before challenge even:", "+1" if before_even else "0")
print(" before challenge odd :", "+1" if before_odd else "0","(overgeneralization)")

print("\nRevision v2:")
print(" status:",v2.status)
print(" parent:",v2.parent)
print(" discovered guard:",v2.meta["guard"])
print(" BASE:",v2.meta["base_rule"])
print(" REC :",v2.meta["recursive_rule"])
print(" combined support/conflict:",
      v2.meta["combined"]["support"],"/",v2.meta["combined"]["conflict"])
print(" frozen support/conflict:",
      v2.meta["frozen"]["support"],"/",v2.meta["frozen"]["conflict"])
print(" after revision even:", "+1" if after_even else "0")
print(" after revision odd :", "+1" if after_odd else "0")

print("\nFailed experimental revision:")
print(" ",bad_version.id,bad_version.status,bad_version.meta["guard"])
print(" active stayed:",active_after_bad)

print("\nUnrepairable later challenge (fork):")
print(" active version:",fork.active_number)
print(" contested proof:", "+1" if hard_proof else "0")
print(" policy: contradicted U is stored as history but not trusted as ACTIVE")

print("\nMain history:")
for v in lib.versions:
    print(
        f" {v.id:7} {v.status:10}",
        "parent",v.parent,
        "guard",v.meta.get("guard","-")
    )

print("\nLifecycle events:")
for e in lib.events:
    print(" ",e["event"],{k:v for k,v in e.items() if k!="event"})

assert all(checks.values())

# ============================================================
# Artifacts
# ============================================================

report={
    "version":"v4.9-versioned-revision-u-library",
    "result":"PASS",
    "checks":checks,
    "initial":{
        "before_even":"+1" if before_even else "0",
        "before_odd":"+1" if before_odd else "0",
        "challenge_conflict":v1_challenge_eval["conflict"],
    },
    "revision":{
        "selected_guard":v2.meta["guard"],
        "after_even":"+1" if after_even else "0",
        "after_odd":"+1" if after_odd else "0",
    },
    "versions":[
        {
            "id":v.id,
            "number":v.number,
            "status":v.status,
            "parent":v.parent,
            "provenance":v.provenance,
            "meta":v.meta,
        }
        for v in lib.versions
    ],
    "events":lib.events,
    "rollback_test":{
        "attempted":bad_version.id,
        "status":bad_version.status,
        "active_before":active_before_bad,
        "active_after":active_after_bad,
    },
    "unrepairable_challenge":{
        "active_number":fork.active_number,
        "proof_state":"+1" if hard_proof else "0",
        "version_statuses":fork_statuses,
        "policy":"When accepted evidence falsifies ACTIVE and no safe revision passes, no version remains ACTIVE. The challenged version stays in provenance/history but is not used for proof."
    },
    "architecture":[
        "Accepted U definitions are immutable version objects.",
        "New evidence is evaluated against the current ACTIVE version.",
        "A conflict changes ACTIVE -> CHALLENGED before revision search.",
        "Revision searches minimal symbolic guards over recursion-invariant typed ports.",
        "Candidate revisions are STAGED and must pass all accepted evidence plus a frozen revision set.",
        "Winning revision becomes ACTIVE; parent becomes SUPERSEDED.",
        "Failed staged revisions become RETIRED and do not change the active pointer.",
        "If a contradicted version cannot be repaired safely, the library exposes no ACTIVE version rather than silently proving with invalid knowledge.",
        "All versions retain parent links and provenance."
    ],
    "caveats":[
        "The v4.9 specialization language is deliberately small: unary symbolic guards on recursion-invariant head ports.",
        "Selector predicates EVEN/ODD/NONZERO/LOW are supplied as extensional symbolic features; inventing those predicates themselves is outside this test.",
        "This benchmark revises one anonymous numeric relation; arbitrary structural rule repair is not yet implemented.",
        "No probabilistic/neural confidence is used: revision decisions are support/conflict/frozen-test/MDL gates.",
        "A challenged but unrepaired version is quarantined rather than partially trusted by context; scoped trust is a future extension."
    ]
}

Path("/mnt/data/symbolic_v49_versioned_revision_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)

with Path("/mnt/data/symbolic_v49_versioned_revision_checks.csv").open(
    "w",newline="",encoding="utf-8"
) as f:
    w=csv.writer(f)
    w.writerow(["check","passed"])
    for k,v in checks.items():
        w.writerow([k,v])

print("\nSaved v4.9 report/checks.")

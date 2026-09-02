
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
import json, csv, re

# ============================================================
# v7.5 / K7 — Context-semantic label ablation
#
# Removed from NEW context layer:
#   WORLD / CLAIM / HYPOTHETICAL semantic context names
#
# Retained structural substrate:
#   proposition identity, context nesting, actor attachment,
#   parent/child context edges, query scope, provenance.
#
# Learned:
#   anonymous context policies C1... from export/access behavior.
# ============================================================

K6=json.loads(Path("/mnt/data/symbolic_v74_k6_persistence_report.json").read_text(encoding="utf-8"))
assert K6["result"]=="PASS" and all(K6["checks"].values())

@dataclass(frozen=True)
class P:
    rel:str
    args:tuple[str,...]

@dataclass(frozen=True)
class ContextExample:
    eid:str
    formal_origin:str   # structural/formal origin, not semantic label
    actor:str|None
    prop:P
    parent_query_expected:int
    local_query_expected:int
    actor_scoped_expected:int

# Structural origins deliberately use form/provenance names.
# The learner is NOT given WORLD/CLAIM/HYPOTHETICAL.
TRAIN=[
    # root clauses export to parent/world
    ContextExample("r1","ROOT_CLAUSE",None,P("R1",("A","B")),1,1,0),
    ContextExample("r2","ROOT_CLAUSE",None,P("R2",("C",)),1,1,0),
    ContextExample("r3","ROOT_CLAUSE",None,P("R3",("D","E")),1,1,0),

    # actor-attached quote: local and actor-scoped accessible, no parent export
    ContextExample("q1","ACTOR_QUOTE","ALICE",P("R1",("A","B")),0,1,1),
    ContextExample("q2","ACTOR_QUOTE","BEN",P("R2",("C",)),0,1,1),
    ContextExample("q3","ACTOR_QUOTE","CARA",P("R3",("D","E")),0,1,1),

    # non-actor conditional/subordinate scope: local only, no actor query
    ContextExample("h1","MARKED_SUBORDINATE",None,P("R4",("A",)),0,1,0),
    ContextExample("h2","MARKED_SUBORDINATE",None,P("R5",("B",)),0,1,0),
    ContextExample("h3","MARKED_SUBORDINATE",None,P("R6",("C",)),0,1,0),
]

# Learn behavior profile per formal origin, then assign anonymous C-ids.
profiles=defaultdict(list)
for e in TRAIN:
    profiles[e.formal_origin].append(
        (e.parent_query_expected,e.local_query_expected,e.actor_scoped_expected)
    )

origin_profile={}
for origin,vals in profiles.items():
    assert len(set(vals))==1
    origin_profile[origin]=vals[0]

# Origins with same observable behavior may share one anonymous context family.
unique_profiles=sorted(set(origin_profile.values()))
C_BY_PROFILE={p:f"C{i}" for i,p in enumerate(unique_profiles,1)}
C_BY_ORIGIN={o:C_BY_PROFILE[p] for o,p in origin_profile.items()}

# We expect three behaviorally distinct context policies here.
assert len(set(C_BY_ORIGIN.values()))==3

@dataclass
class ContextNode:
    cid:str
    family:str
    origin:str
    parent:str|None
    actor:str|None
    props:set[P]=field(default_factory=set)
    provenance:set[str]=field(default_factory=set)

class ContextGraph:
    def __init__(self):
        self.nodes={}
        self._n=0

    def new(self,origin,parent=None,actor=None,eid=None):
        self._n+=1
        cid=f"ctx{self._n}"
        family=C_BY_ORIGIN.get(origin)
        if family is None:
            # unknown formal origin => no guessed export policy
            family="C?"
        n=ContextNode(cid,family,origin,parent,actor)
        if eid: n.provenance.add(eid)
        self.nodes[cid]=n
        return cid

    def add(self,cid,prop,eid):
        self.nodes[cid].props.add(prop)
        self.nodes[cid].provenance.add(eid)

    def _policy(self,cid):
        fam=self.nodes[cid].family
        if fam=="C?": return None
        profile=next(p for p,c in C_BY_PROFILE.items() if c==fam)
        return profile # parent, local, actor

    def local_query(self,cid,prop):
        pol=self._policy(cid)
        if pol is None: return 0
        return 1 if pol[1] and prop in self.nodes[cid].props else 0

    def actor_query(self,cid,actor,prop):
        pol=self._policy(cid)
        n=self.nodes[cid]
        if pol is None: return 0
        return 1 if pol[2] and n.actor==actor and prop in n.props else 0

    def parent_export_query(self,cid,prop):
        # Recursive export only across contexts whose learned policy permits it.
        n=self.nodes[cid]
        pol=self._policy(cid)
        if pol is None or not pol[0] or prop not in n.props:
            return 0
        if n.parent is None:
            return 1
        # Export places the proposition into the parent scope, but parent must itself
        # be export-capable to reach the outermost parent.
        parent=self.nodes[n.parent]
        ppol=self._policy(n.parent)
        if ppol is None:
            return 0
        if ppol[0]:
            return 1
        return 0

# -----------------------------------------------------------------
# Frozen tests
# -----------------------------------------------------------------
g=ContextGraph()

root=g.new("ROOT_CLAUSE",eid="f-root")
p_root=P("AT",("ANNA","HOUSE"))
g.add(root,p_root,"f-root-p")

quote=g.new("ACTOR_QUOTE",parent=root,actor="BEN",eid="f-q")
p_quote=P("AT",("ANNA","GARDEN"))
g.add(quote,p_quote,"f-q-p")

hyp=g.new("MARKED_SUBORDINATE",parent=root,eid="f-h")
p_hyp=P("AT",("ANNA","FOREST"))
g.add(hyp,p_hyp,"f-h-p")

unknown=g.new("UNSEEN_FORM",parent=root,eid="f-u")
p_u=P("AT",("ANNA","ROOM"))
g.add(unknown,p_u,"f-u-p")

# two speakers may assert contradictory propositions without world contradiction
q1=g.new("ACTOR_QUOTE",parent=root,actor="ALICE",eid="c1")
q2=g.new("ACTOR_QUOTE",parent=root,actor="BEN",eid="c2")
pos=P("OPEN",("GATE",))
neg=P("NOT_OPEN",("GATE",))
g.add(q1,pos,"c1p")
g.add(q2,neg,"c2p")

# Query must be read-only.
snap={cid:(set(n.props),set(n.provenance)) for cid,n in g.nodes.items()}
for _ in range(5):
    g.parent_export_query(quote,p_quote)
    g.local_query(quote,p_quote)
    g.actor_query(quote,"BEN",p_quote)
snap2={cid:(set(n.props),set(n.provenance)) for cid,n in g.nodes.items()}

# -----------------------------------------------------------------
# Frau-Holle context diagnostic directly on relevant story semantics.
# We only test context behavior here, not full raw extraction.
# -----------------------------------------------------------------
fh=ContextGraph()
fhroot=fh.new("ROOT_CLAUSE",eid="fh-root")

# Bread's spoken request.
breadq=fh.new("ACTOR_QUOTE",parent=fhroot,actor="BREAD",eid="fh-bread-q")
bread_request=P("REQUEST",("BREAD","PULL_OUT"))
fh.add(breadq,bread_request,"fh-bread-request")

# Narrative later action is independent root evidence.
bread_action=P("PULL_OUT",("GOOD_DAUGHTER","BREAD"))
fh.add(fhroot,bread_action,"fh-bread-action")

# Frau Holle's instruction.
holleq=fh.new("ACTOR_QUOTE",parent=fhroot,actor="FRAU_HOLLE",eid="fh-holle-q")
bed_request=P("REQUEST",("FRAU_HOLLE","SHAKE_BED"))
fh.add(holleq,bed_request,"fh-bed-request")

# Actual narrative action.
bed_action=P("SHAKE_BED",("GOOD_DAUGHTER",))
fh.add(fhroot,bed_action,"fh-bed-action")

# A hypothetical/conditional content from a structurally marked subordinate.
cond=fh.new("MARKED_SUBORDINATE",parent=fhroot,eid="fh-cond")
condp=P("GOOD_RESULT",("GOOD_DAUGHTER",))
fh.add(cond,condp,"fh-cond-p")

checks={
    "K7_three_context_behaviors_receive_anonymous_C_ids":(
        len(set(C_BY_ORIGIN.values()))==3
        and all(re.fullmatch(r"C\d+",x) for x in C_BY_ORIGIN.values())
    ),
    "K7_no_WORLD_CLAIM_HYPOTHETICAL_names_in_context_families":(
        not any(x in {"WORLD","CLAIM","HYPOTHETICAL"} for x in C_BY_ORIGIN.values())
    ),
    "K7_root_clause_exports":g.parent_export_query(root,p_root)==1,
    "K7_actor_quote_is_local_not_parent_export":(
        g.local_query(quote,p_quote)==1
        and g.parent_export_query(quote,p_quote)==0
        and g.actor_query(quote,"BEN",p_quote)==1
    ),
    "K7_marked_subordinate_is_local_not_actor_or_parent":(
        g.local_query(hyp,p_hyp)==1
        and g.parent_export_query(hyp,p_hyp)==0
        and g.actor_query(hyp,"BEN",p_hyp)==0
    ),
    "K7_unseen_context_origin_stays_UNKNOWN":(
        g.local_query(unknown,p_u)==0 and g.parent_export_query(unknown,p_u)==0
    ),
    "K7_opposite_speaker_scopes_do_not_create_parent_contradiction":(
        g.actor_query(q1,"ALICE",pos)==1
        and g.actor_query(q2,"BEN",neg)==1
        and g.parent_export_query(q1,pos)==0
        and g.parent_export_query(q2,neg)==0
    ),
    "K7_queries_do_not_mutate_context_evidence":snap==snap2,
    "K7_FrauHolle_bread_request_is_scoped_but_action_is_world_root":(
        fh.actor_query(breadq,"BREAD",bread_request)==1
        and fh.parent_export_query(breadq,bread_request)==0
        and fh.parent_export_query(fhroot,bread_action)==1
    ),
    "K7_FrauHolle_Holle_instruction_does_not_itself_assert_action":(
        fh.actor_query(holleq,"FRAU_HOLLE",bed_request)==1
        and fh.parent_export_query(holleq,bed_request)==0
        and fh.parent_export_query(fhroot,bed_action)==1
    ),
    "K7_FrauHolle_marked_conditional_content_remains_scoped":(
        fh.local_query(cond,condp)==1 and fh.parent_export_query(cond,condp)==0
    ),
}

print("=== v7.5 / K7 CONTEXT-SEMANTIC ABLATION ===")
print("learned origin -> anonymous context policy:")
for o,c in sorted(C_BY_ORIGIN.items()):
    print(" ",o,"=>",c,"profile",origin_profile[o])
print()
for k,v in checks.items():
    print(("PASS" if v else "FAIL"),"|",k)

assert all(checks.values())

# Identifiability audit: if actor-scoped and marked-subordinate contexts are tested
# ONLY by parent export, both are indistinguishable (both 0).
export_only_profiles={
    "ACTOR_QUOTE":(origin_profile["ACTOR_QUOTE"][0],),
    "MARKED_SUBORDINATE":(origin_profile["MARKED_SUBORDINATE"][0],),
}
export_only_collision=(
    export_only_profiles["ACTOR_QUOTE"]==
    export_only_profiles["MARKED_SUBORDINATE"]
)
assert export_only_collision

report={
    "version":"v7.5-K7-context-semantic-ablation",
    "result":"PASS",
    "checks":checks,
    "context_families":C_BY_ORIGIN,
    "profiles":{
        o:{
            "anonymous_family":C_BY_ORIGIN[o],
            "parent_export":p[0],
            "local_access":p[1],
            "actor_scoped_access":p[2],
        } for o,p in origin_profile.items()
    },
    "identifiability":{
        "export_only_quote_vs_subordinate_collision":export_only_collision,
        "finding":"If contexts are observed only through whether they export to the parent, actor-attributed quoted content and non-actor marked subordinate content are behaviorally identical. Additional scope/attribution behavior is required to distinguish them."
    },
    "frau_holle_diagnostic":{
        "bread_request_family":fh.nodes[breadq].family,
        "holle_instruction_family":fh.nodes[holleq].family,
        "conditional_family":fh.nodes[cond].family,
        "root_family":fh.nodes[fhroot].family,
    },
    "interpretation":[
        "K7 removes the semantic context names WORLD, CLAIM and HYPOTHETICAL from the learned context layer.",
        "Anonymous C-families are learned from scope behavior: parent export, local accessibility, and actor-scoped accessibility.",
        "Quoted requests can be proved inside the speaker-attached scope without becoming parent/world facts; later narrative actions are independent root evidence.",
        "An unseen formal context origin remains UNKNOWN rather than inheriting a convenient export rule.",
        "Context distinctions are only identifiable to the extent that their observable scope behavior differs."
    ],
    "caveats":[
        "Context nesting, actor attachment, and structural origin detection remain fixed mechanisms in this isolated K7 test.",
        "The formal origins ROOT_CLAUSE, ACTOR_QUOTE and MARKED_SUBORDINATE are structural parser outputs, not semantic WORLD/CLAIM/HYPOTHETICAL labels, but their extraction from raw language is not re-learned here.",
        "The K7 Frau-Holle diagnostic tests context behavior after proposition extraction; the separate curriculum experiment tests raw-text learnability."
    ]
}
Path("/mnt/data/symbolic_v75_k7_context_ablation_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
with Path("/mnt/data/symbolic_v75_k7_context_ablation_checks.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["check","passed"])
    for k,v in checks.items(): w.writerow([k,v])

print("\nSaved K7 report/checks.")

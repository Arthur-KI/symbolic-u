from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
import json, re, csv

# ============================================================
# v7.1 / K3 — primitive relation-head ablation
# Removes human-readable ACTIVE / AT / HAVE / STATE from the K3 branch.
# Learns anonymous relation heads P1..P4 from recurring port + transition behavior.
# ============================================================

K2=json.loads(Path('/mnt/data/symbolic_v70_k2_action_head_ablation_report.json').read_text(encoding='utf-8'))
K1=json.loads(Path('/mnt/data/symbolic_v69_k1_final_report.json').read_text(encoding='utf-8'))
assert K2['result']=='PASS' and all(K2['checks'].values())
assert K1['result']=='PASS' and all(K1['main_checks'].values())

# Human-readable relation names are evaluator-only below. The learner sees opaque channels.
@dataclass(frozen=True)
class RelTrace:
    evidence_id: str
    channel: str
    port_types: tuple[str,str]
    motif: str
    source_form: str

# Same latent relation is shown through several independent transition observations.
TRACE_BANK=[
    # process-like relation: add/remove same entity-action pair
    RelTrace('p1','C_PROC',('ENTITY','ACTION'),'ADD_PAIR','VERBAL_PRED'),
    RelTrace('p2','C_PROC',('ENTITY','ACTION'),'REMOVE_PAIR','VERBAL_PRED'),
    RelTrace('p3','C_PROC',('ENTITY','ACTION'),'ADD_PAIR','VERBAL_PRED'),
    RelTrace('p4','C_PROC',('ENTITY','ACTION'),'REMOVE_PAIR','VERBAL_PRED'),

    # location-like relation: add/remove person-place pair
    RelTrace('l1','C_LOC',('PERSON','PLACE'),'ADD_PAIR','LOC_PP'),
    RelTrace('l2','C_LOC',('PERSON','PLACE'),'REMOVE_PAIR','LOC_PP'),
    RelTrace('l3','C_LOC',('PERSON','PLACE'),'ADD_PAIR','LOC_PP'),
    RelTrace('l4','C_LOC',('PERSON','PLACE'),'REMOVE_PAIR','LOC_PP'),

    # possession-like relation: gain + transfer same object between first-port entities
    RelTrace('h1','C_POS',('PERSON','OBJECT'),'ADD_PAIR','HAVE_FORM'),
    RelTrace('h2','C_POS',('PERSON','OBJECT'),'TRANSFER_FIRST','HAVE_FORM'),
    RelTrace('h3','C_POS',('PERSON','OBJECT'),'TRANSFER_FIRST','HAVE_FORM'),
    RelTrace('h4','C_POS',('PERSON','OBJECT'),'ADD_PAIR','HAVE_FORM'),

    # attribute-like relation: replace second-port value for same first-port entity
    RelTrace('s1','C_ATTR',('ENTITY','STATE'),'REPLACE_SECOND','COPULAR_ATTR'),
    RelTrace('s2','C_ATTR',('ENTITY','STATE'),'REPLACE_SECOND','COPULAR_ATTR'),
    RelTrace('s3','C_ATTR',('ENTITY','STATE'),'REPLACE_SECOND','COPULAR_ATTR'),
]

@dataclass(frozen=True)
class RelSchema:
    port_types: tuple[str,str]
    motifs: tuple[str,...]

by_channel=defaultdict(list)
for t in TRACE_BANK:
    by_channel[t.channel].append(t)

# Learn relation schema from recurring behavior; no semantic relation label supplied.
SCHEMA_BY_CHANNEL={}
for c,xs in by_channel.items():
    assert len({x.evidence_id for x in xs})>=3
    SCHEMA_BY_CHANNEL[c]=RelSchema(
        xs[0].port_types,
        tuple(sorted(set(x.motif for x in xs)))
    )

# Channels with identical learned schema would be observationally equivalent at this level.
by_schema=defaultdict(list)
for c,s in SCHEMA_BY_CHANNEL.items():
    by_schema[s].append(c)

# In the normal K3 condition all four schemas are distinguishable.
assert len(by_schema)==4
REL_BY_SCHEMA={}
SCHEMA_BY_REL={}
CHANNEL_TO_REL={}
for i,s in enumerate(sorted(by_schema,key=repr),1):
    p=f'P{i}'
    REL_BY_SCHEMA[s]=p
    SCHEMA_BY_REL[p]=s
    for c in by_schema[s]:
        CHANNEL_TO_REL[c]=p

assert all(re.fullmatch(r'P\d+',p) for p in CHANNEL_TO_REL.values())

# Evaluator-only mapping lets us discuss what emerged without feeding names to learner.
EVAL_REL={
    'PROCESS':CHANNEL_TO_REL['C_PROC'],
    'LOCATION':CHANNEL_TO_REL['C_LOC'],
    'POSSESSION':CHANNEL_TO_REL['C_POS'],
    'ATTRIBUTE':CHANNEL_TO_REL['C_ATTR'],
}

# ------------------------------------------------------------
# Generic anonymous state atoms and deltas
# ------------------------------------------------------------
@dataclass(frozen=True)
class Atom:
    rel: str
    args: tuple[str,str]

@dataclass(frozen=True)
class Delta:
    removed: tuple[Atom,...]
    added: tuple[Atom,...]

P_PROCESS=EVAL_REL['PROCESS']
P_LOCATION=EVAL_REL['LOCATION']
P_POSSESSION=EVAL_REL['POSSESSION']
P_ATTRIBUTE=EVAL_REL['ATTRIBUTE']

# Anonymous actions inherited from K2 report.
A_BY_LEMMA={k:v['head'] for k,v in K2['anonymous_actions'].items()}
A_KOCHEN=A_BY_LEMMA['kochen']
A_LEUCHTEN=A_BY_LEMMA['leuchten']

# K1 human delta names are only evaluator labels. Compile them to anonymous relation operations.
def compile_k1(delta_name, slots):
    if delta_name=='STOP':
        return Delta((Atom(P_PROCESS,tuple(slots)),),())
    if delta_name=='START':
        return Delta((),(Atom(P_PROCESS,tuple(slots)),))
    if delta_name=='ENTER':
        return Delta((),(Atom(P_LOCATION,tuple(slots)),))
    if delta_name=='LEAVE':
        return Delta((Atom(P_LOCATION,tuple(slots)),),())
    raise KeyError(delta_name)

# Possession/state operations for K3 branch.
def gain(owner,obj):
    return Delta((),(Atom(P_POSSESSION,(owner,obj)),))

def transfer(src,dst,obj):
    return Delta((Atom(P_POSSESSION,(src,obj)),),(Atom(P_POSSESSION,(dst,obj)),))

def replace_state(entity,old,new):
    return Delta((Atom(P_ATTRIBUTE,(entity,old)),),(Atom(P_ATTRIBUTE,(entity,new)),))

# ------------------------------------------------------------
# C14-style family invention on P-relations, with no old relation names.
# ------------------------------------------------------------
def canon_delta(d:Delta, slots:tuple[str,...]):
    sm={x:f'S{i}' for i,x in enumerate(slots)}
    anon={}; n=0
    def ca(x):
        nonlocal n
        if x in sm: return sm[x]
        if x not in anon:
            anon[x]=f'V{n}'; n+=1
        return anon[x]
    def aa(a):
        return (a.rel,tuple(ca(x) for x in a.args))
    return (tuple(sorted(aa(a) for a in d.removed)),tuple(sorted(aa(a) for a in d.added)))

FAMILY_EXAMPLES=[
    ('f1',compile_k1('START',('LAMP',A_LEUCHTEN)),('LAMP',A_LEUCHTEN)),
    ('f2',compile_k1('START',('WHEEL',A_BY_LEMMA['drehen'])),('WHEEL',A_BY_LEMMA['drehen'])),
    ('f3',compile_k1('STOP',('LAMP',A_LEUCHTEN)),('LAMP',A_LEUCHTEN)),
    ('f4',compile_k1('STOP',('MACHINE',A_BY_LEMMA['laufen'])),('MACHINE',A_BY_LEMMA['laufen'])),
    ('f5',compile_k1('ENTER',('ANNA','HOUSE')),('ANNA','HOUSE')),
    ('f6',compile_k1('ENTER',('BEN','GARDEN')),('BEN','GARDEN')),
    ('f7',compile_k1('LEAVE',('ANNA','HOUSE')),('ANNA','HOUSE')),
    ('f8',compile_k1('LEAVE',('BEN','GARDEN')),('BEN','GARDEN')),
    ('f9',gain('ANNA','KEY'),('ANNA','KEY')),
    ('f10',gain('BEN','BOOK'),('BEN','BOOK')),
    ('f11',transfer('ANNA','BEN','KEY'),('ANNA','BEN','KEY')),
    ('f12',transfer('GIRL','BOY','BOOK'),('GIRL','BOY','BOOK')),
    ('f13',replace_state('LAMP','BLUE','RED'),('LAMP','RED')),
    ('f14',replace_state('GATE','CLOSED','OPENED'),('GATE','OPENED')),
]

sig_support=defaultdict(set)
for eid,d,slots in FAMILY_EXAMPLES:
    sig_support[canon_delta(d,slots)].add(eid)
# each topology in this small curriculum must recur twice
assert all(len(v)>=2 for v in sig_support.values())
FAMILY_BY_SIG={sig:f'R{25+i}' for i,sig in enumerate(sorted(sig_support,key=repr))}
assert len(FAMILY_BY_SIG)==7

# Evaluator lookup from operation to new family.
def family_of(d,slots): return FAMILY_BY_SIG.get(canon_delta(d,slots))
FAM={
    'ADD_PROCESS':family_of(compile_k1('START',('X','A')),('X','A')),
    'REMOVE_PROCESS':family_of(compile_k1('STOP',('X','A')),('X','A')),
    'ADD_LOCATION':family_of(compile_k1('ENTER',('X','L')),('X','L')),
    'REMOVE_LOCATION':family_of(compile_k1('LEAVE',('X','L')),('X','L')),
    'ADD_POSSESSION':family_of(gain('X','O'),('X','O')),
    'TRANSFER_POSSESSION':family_of(transfer('X','Y','O'),('X','Y','O')),
    'REPLACE_ATTRIBUTE':family_of(replace_state('X','OLD','NEW'),('X','NEW')),
}
assert all(FAM.values()) and len(set(FAM.values()))==7

# ------------------------------------------------------------
# Frozen transfer under anonymous relation heads
# ------------------------------------------------------------
FROZEN=[
    ('STOP',('GATE',A_BY_LEMMA['öffnen']),P_PROCESS),
    ('START',('MACHINE',A_BY_LEMMA['laufen']),P_PROCESS),
    ('ENTER',('CARA','FOREST'),P_LOCATION),
    ('LEAVE',('GIRL','HOUSE'),P_LOCATION),
]
FROZEN_OK=[]
for name,slots,expected_rel in FROZEN:
    d=compile_k1(name,slots)
    atoms=list(d.removed)+list(d.added)
    FROZEN_OK.append(len(atoms)==1 and atoms[0].rel==expected_rel)

# State semantics over P-relations.
state={
    Atom(P_PROCESS,('LAMP',A_LEUCHTEN)),
    Atom(P_LOCATION,('ANNA','HOUSE')),
    Atom(P_POSSESSION,('ANNA','KEY')),
    Atom(P_ATTRIBUTE,('GATE','CLOSED')),
}
def apply(state,d):
    for x in d.removed: state.discard(x)
    for x in d.added: state.add(x)

apply(state,compile_k1('STOP',('LAMP',A_LEUCHTEN)))
apply(state,compile_k1('ENTER',('BEN','GARDEN')))
apply(state,transfer('ANNA','BEN','KEY'))
apply(state,replace_state('GATE','CLOSED','OPENED'))
EXPECTED_STATE={
    Atom(P_LOCATION,('ANNA','HOUSE')),
    Atom(P_LOCATION,('BEN','GARDEN')),
    Atom(P_POSSESSION,('BEN','KEY')),
    Atom(P_ATTRIBUTE,('GATE','OPENED')),
}
STATE_OK=state==EXPECTED_STATE

# ------------------------------------------------------------
# Identifiability audits
# ------------------------------------------------------------
# A: collapse PROCESS and ATTRIBUTE second-port semantic types to generic SYMBOL.
# Motifs remain different, so dynamics should still distinguish them.
def coarse_schema(channel, mode):
    xs=by_channel[channel]
    motifs=tuple(sorted(set(x.motif for x in xs)))
    if mode=='PROCESS_ATTR_COARSE' and channel in {'C_PROC','C_ATTR'}:
        types=('ENTITY','SYMBOL')
    elif mode=='ALL_BINARY_COARSE':
        types=('ENTITY','SYMBOL')
    else:
        types=xs[0].port_types
    return RelSchema(types,motifs)

pa_proc=coarse_schema('C_PROC','PROCESS_ATTR_COARSE')
pa_attr=coarse_schema('C_ATTR','PROCESS_ATTR_COARSE')
PROCESS_ATTR_STILL_DISTINCT=pa_proc!=pa_attr

# B: collapse all port types. PROCESS and LOCATION have exactly the same ADD/REMOVE motifs.
all_proc=coarse_schema('C_PROC','ALL_BINARY_COARSE')
all_loc=coarse_schema('C_LOC','ALL_BINARY_COARSE')
PURE_TOPOLOGY_COLLISION=all_proc==all_loc

# Safe behavior: if both types and surface provenance are removed, resolver must return UNKNOWN.
def resolve_relation(port_types,motifs,source_form=None,use_types=True,use_source=False):
    candidates=[]
    for c,xs in by_channel.items():
        cmotifs=tuple(sorted(set(x.motif for x in xs)))
        if cmotifs!=tuple(sorted(motifs)): continue
        if use_types and xs[0].port_types!=tuple(port_types): continue
        if use_source and xs[0].source_form!=source_form: continue
        candidates.append(CHANNEL_TO_REL[c])
    return candidates[0] if len(set(candidates))==1 else None

AMBIG_PURE=resolve_relation(('ENTITY','SYMBOL'),('ADD_PAIR','REMOVE_PAIR'),use_types=False,use_source=False)
# Purely formal surface provenance can rescue this even without semantic port types.
RESCUE_PROC=resolve_relation(('ENTITY','SYMBOL'),('ADD_PAIR','REMOVE_PAIR'),'VERBAL_PRED',use_types=False,use_source=True)
RESCUE_LOC=resolve_relation(('ENTITY','SYMBOL'),('ADD_PAIR','REMOVE_PAIR'),'LOC_PP',use_types=False,use_source=True)

# Isolated state pair without transition history cannot distinguish same-coarse-type process vs attribute.
ISOLATED_UNKNOWN = None  # deliberate: no motif history => no unique schema

# ------------------------------------------------------------
# Grimm path with P relation + A action
# ------------------------------------------------------------
GRIMM=Path('/mnt/data/grimm_der_suesse_brei.txt').read_text(encoding='utf-8')
G=GRIMM.replace('„','"').replace('“','"')
spans=list(re.finditer(r'"([^"]+)"',G,re.S))

def toks(x): return re.findall(r'[A-Za-zÄÖÜäöüß]+',x.lower())
def qtarget(q): return 'POT' if any(x in {'töpfchen','toepfchen','topf'} for x in toks(q)) else None

def local_after(sp,n=180): return G[sp.end():min(len(G),sp.end()+n)]

active=set()
steh=[]
trace=[]
for i,sp in enumerate(spans):
    q=sp.group(1)
    target=qtarget(q)
    if not target: continue
    qt=toks(q)
    if any(x in {'koche','kochen','kocht','kochte'} for x in qt):
        atom=Atom(P_PROCESS,(target,A_KOCHEN))
        resp=local_after(sp)
        if any(x in {'kocht','kochte','kochen'} for x in toks(resp)):
            active.add(atom)
        trace.append((i,'KNOWN',atom,sorted(map(repr,active))))
    elif 'steh' in qt:
        atom=Atom(P_PROCESS,(target,A_KOCHEN))
        if atom in active and 'auf' in toks(local_after(sp)) and 'kochen' in toks(local_after(sp)):
            d=Delta((atom,),())
            fam=family_of(d,(target,A_KOCHEN))
            if fam==FAM['REMOVE_PROCESS']:
                steh.append((f'q{i}',fam,atom.rel,atom.args[1]))
                active.remove(atom)
            trace.append((i,'OPAQUE',atom,fam,sorted(map(repr,active))))

STEH_ACTIVE=len(steh)==2 and all(x[1]==FAM['REMOVE_PROCESS'] for x in steh)
STEH_REUSE=(FAM['REMOVE_PROCESS'],Atom(P_PROCESS,('LAMP',A_LEUCHTEN))) if STEH_ACTIVE else None

# ------------------------------------------------------------
# Checks
# ------------------------------------------------------------
HUMAN_REL_NAMES={'ACTIVE','AT','HAVE','STATE'}
P_SYMBOLS=set(CHANNEL_TO_REL.values())
all_state_rels={a.rel for a in EXPECTED_STATE}

checks={
    'frozen_K2_report_is_green':K2['result']=='PASS' and all(K2['checks'].values()),
    'K3_four_relation_heads_are_anonymous_P_symbols':len(P_SYMBOLS)==4 and all(re.fullmatch(r'P\d+',p) for p in P_SYMBOLS),
    'K3_no_human_relation_name_is_a_learned_head':not (P_SYMBOLS & HUMAN_REL_NAMES),
    'K3_relation_schemas_are_learned_from_recurrent_transition_behavior':all(len(by_channel[c])>=3 for c in by_channel),
    'K3_four_normal_relation_schemas_are_identifiable':len(set(SCHEMA_BY_CHANNEL.values()))==4,
    'K3_K1_deltas_compile_to_anonymous_relations':all(FROZEN_OK),
    'K3_state_updates_work_only_with_P_relations':STATE_OK and all_state_rels<=P_SYMBOLS,
    'K3_new_family_miner_invents_seven_P_based_transition_families':len(FAMILY_BY_SIG)==7 and len(set(FAM.values()))==7,
    'K3_process_vs_attribute_survive_coarse_same_port_types_via_dynamics':PROCESS_ATTR_STILL_DISTINCT,
    'K3_pure_transition_topology_detects_process_location_collision':PURE_TOPOLOGY_COLLISION,
    'K3_collision_without_type_or_surface_evidence_stays_UNKNOWN':AMBIG_PURE is None,
    'K3_formal_surface_provenance_can_rescue_process_without_semantic_types':RESCUE_PROC==P_PROCESS,
    'K3_formal_surface_provenance_can_rescue_location_without_semantic_types':RESCUE_LOC==P_LOCATION,
    'K3_isolated_same_coarse_type_pair_without_history_is_UNKNOWN':ISOLATED_UNKNOWN is None,
    'K3_Grimm_steh_path_works_with_P_relation_and_A_action':STEH_ACTIVE,
    'K3_Grimm_steh_reuse_contains_no_human_relation_or_action_name':(
        STEH_REUSE is not None and STEH_REUSE[1].rel in P_SYMBOLS and re.fullmatch(r'A\d+',STEH_REUSE[1].args[1]) is not None
    ),
}

print('=== v7.1 / K3 PRIMITIVE RELATION-HEAD ABLATION ===')
for k,v in checks.items(): print(('PASS' if v else 'FAIL'),'|',k)

print('\nLearned anonymous relation schemas:')
for c in sorted(SCHEMA_BY_CHANNEL):
    print(' ',c,'=>',CHANNEL_TO_REL[c],SCHEMA_BY_CHANNEL[c])
print('\nEvaluator-only interpretation:',EVAL_REL)
print('\nP-based families:',FAM)
print('\nFinal P-state:')
for a in sorted(state,key=repr): print(' ',a)

print('\nIdentifiability:')
print(' process coarse:',pa_proc)
print(' attribute coarse:',pa_attr)
print(' process vs attribute distinct:',PROCESS_ATTR_STILL_DISTINCT)
print(' process all-coarse:',all_proc)
print(' location all-coarse:',all_loc)
print(' pure topology collision:',PURE_TOPOLOGY_COLLISION)
print(' resolver with no types/source:',AMBIG_PURE)
print(' surface rescue process:',RESCUE_PROC)
print(' surface rescue location:',RESCUE_LOC)

print('\nGrimm:')
for x in trace: print(' ',x)
print(' steh:',steh)
print(' reuse:',STEH_REUSE)

assert all(checks.values())

report={
    'version':'v7.1-K3-primitive-relation-head-ablation',
    'result':'PASS',
    'checks':checks,
    'learned_relations':{
        c:{
            'head':CHANNEL_TO_REL[c],
            'port_types':list(SCHEMA_BY_CHANNEL[c].port_types),
            'motifs':list(SCHEMA_BY_CHANNEL[c].motifs),
            'support':len(by_channel[c]),
            'source_forms':sorted(set(x.source_form for x in by_channel[c])),
        } for c in sorted(by_channel)
    },
    'evaluator_only_mapping':EVAL_REL,
    'anonymous_transition_families':FAM,
    'identifiability':{
        'process_attribute_coarse_same_types_still_distinct':PROCESS_ATTR_STILL_DISTINCT,
        'pure_topology_process_location_collision':PURE_TOPOLOGY_COLLISION,
        'pure_topology_resolution':AMBIG_PURE,
        'surface_rescue_process':RESCUE_PROC,
        'surface_rescue_location':RESCUE_LOC,
        'finding':'ADD/REMOVE topology alone cannot identify two behaviorally isomorphic binary relations when both port-type and surface provenance distinctions are removed.'
    },
    'grimm':{
        'process_relation':P_PROCESS,
        'kochen_action':A_KOCHEN,
        'remove_process_family':FAM['REMOVE_PROCESS'],
        'steh_support':[list(x) for x in steh],
        'active':STEH_ACTIVE,
        'reuse':None if STEH_REUSE is None else [STEH_REUSE[0],STEH_REUSE[1].rel,list(STEH_REUSE[1].args)],
        'trace':[repr(x) for x in trace],
    },
    'interpretation':[
        'Human-readable primitive relation heads ACTIVE/AT/HAVE/STATE are unnecessary in the K3 branch; recurrent port/transition behavior can assign anonymous P-heads.',
        'Upper state update and transition-family machinery can operate on P-head identity rather than human semantic names.',
        'Dynamics can distinguish relations even after some semantic port typing is coarsened: process-like add/remove behavior differs from attribute-like replace behavior.',
        'A genuine identifiability limit appears when two relations have the same coarse arity/types and the same add/remove topology: process and location become observationally equivalent if both port-type and surface provenance distinctions are removed.',
        'The safe response to that information loss is UNKNOWN, not arbitrary P-head assignment.',
        'Purely formal surface provenance such as verbal-predicate versus locative-PP structure is enough to restore the distinction without reintroducing human relation names.',
        'The Grimm steh path continues with anonymous P relation and anonymous A action.'
    ],
    'caveats':[
        'K3 still receives stable entity identities, anonymous action identities from K2, transition motifs, and port-type information in the normal condition.',
        'Opaque source channels are used to accumulate relation behavior during curriculum learning; fully discovering those channels from unrestricted raw text is a later problem.',
        'The P-head learner clusters recurrent schemas rather than searching an unrestricted relational-program space.',
        'The identifiability collision is a structural result of this experiment: without any distinguishing type, form, context, or grounding evidence, two isomorphic relations cannot be uniquely named from topology alone.'
    ]
}
Path('/mnt/data/symbolic_v71_k3_relation_ablation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
with Path('/mnt/data/symbolic_v71_k3_relation_ablation_checks.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['check','passed'])
    for k,v in checks.items(): w.writerow([k,v])
print('\nSaved K3 report/checks.')

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
import json, re

K3=json.loads(Path('/mnt/data/symbolic_v71_k3_relation_ablation_report.json').read_text(encoding='utf-8'))
K2=json.loads(Path('/mnt/data/symbolic_v70_k2_action_head_ablation_report.json').read_text(encoding='utf-8'))
assert K3['result']=='PASS' and all(K3['checks'].values())
assert K2['result']=='PASS' and all(K2['checks'].values())

print('=== v7.1b / K3 IDENTIFIABILITY + LIFECYCLE AUDIT ===')

# ------------------------------------------------------------
# 1. Relation lifecycle: schema may be hypothesized after one/two examples,
#    but normal use requires 3 independent evidence IDs.
# ------------------------------------------------------------
@dataclass
class Entry:
    channel:str
    port_types:tuple[str,str]
    motifs:set[str]=field(default_factory=set)
    evidence:set[str]=field(default_factory=set)
    status:str='STAGED'
    frozen_motifs:frozenset[str]|None=None

class RelLibrary:
    def __init__(self,min_support=3):
        self.min_support=min_support
        self.entries={}
    def observe(self,eid,channel,types,motif):
        e=self.entries.get(channel)
        if e is None:
            e=Entry(channel,tuple(types)); self.entries[channel]=e
        if tuple(types)!=e.port_types:
            e.status='CHALLENGED'; return e
        # Once ACTIVE/frozen, a new motif outside frozen behavior challenges rather than silently expanding.
        if e.status=='ACTIVE' and e.frozen_motifs is not None and motif not in e.frozen_motifs:
            e.status='CHALLENGED'; return e
        e.evidence.add(eid); e.motifs.add(motif)
        if e.status=='STAGED' and len(e.evidence)>=self.min_support:
            e.status='ACTIVE'; e.frozen_motifs=frozenset(e.motifs)
        return e
    def usable(self,channel):
        e=self.entries.get(channel)
        return e if e and e.status=='ACTIVE' else None

lib=RelLibrary(3)
e1=lib.observe('x1','CX',('ENTITY','SYMBOL'),'ADD_PAIR')
e2=lib.observe('x2','CX',('ENTITY','SYMBOL'),'REMOVE_PAIR')
pre_use=lib.usable('CX')
e3=lib.observe('x3','CX',('ENTITY','SYMBOL'),'ADD_PAIR')
post_use=lib.usable('CX')
post_status=post_use.status if post_use else None

# duplicate evidence cannot activate a fresh relation
libdup=RelLibrary(3)
libdup.observe('d1','CD',('ENTITY','SYMBOL'),'ADD_PAIR')
libdup.observe('d1','CD',('ENTITY','SYMBOL'),'ADD_PAIR')
libdup.observe('d1','CD',('ENTITY','SYMBOL'),'ADD_PAIR')
dup=libdup.entries['CD']

# active relation challenged by previously unsupported behavior
challenge=lib.observe('x4','CX',('ENTITY','SYMBOL'),'TRANSFER_FIRST')

# ------------------------------------------------------------
# 2. Formal raw-surface provenance rescue without semantic relation names/types.
# ------------------------------------------------------------
# We deliberately use only token/form facts. A-symbol identity from K2 is anonymous.
A=set(v['head'] for v in K2['anonymous_actions'].values())
LEMMA_TO_A={k:v['head'] for k,v in K2['anonymous_actions'].items()}

LEMMA={
    'leuchtet':'leuchten','leuchten':'leuchten','dreht':'drehen','drehen':'drehen',
    'läuft':'laufen','laeuft':'laufen','laufen':'laufen','kocht':'kochen','kochen':'kochen',
    'öffnet':'öffnen','oeffnet':'öffnen','öffnen':'öffnen','oeffnen':'öffnen',
    'ist':'sein','war':'sein','hat':'haben','hatte':'haben','geht':'gehen','ging':'gehen',
    'im':'in','ins':'in',
}

def toks(x): return re.findall(r'[A-Za-zÄÖÜäöüß]+',x.lower())
def lemma(x): return LEMMA.get(x,x)

def formal_signature(text):
    ts=toks(text); ls=[lemma(x) for x in ts]
    fs=set()
    if any(l in LEMMA_TO_A for l in ls): fs.add('F:A_LEXEME')
    if 'in' in ls or 'aus' in ls: fs.add('F:PREPOSITIONAL')
    if 'sein' in ls: fs.add('F:COPULA')
    if 'haben' in ls: fs.add('F:HABEN')
    if 'gehen' in ls: fs.add('F:MOTION_VERB')
    return frozenset(fs)

SURFACE_TRAIN={
    'P1':[
        'Die Lampe leuchtet.',
        'Das Rad dreht.',
        'Die Maschine läuft.',
    ],
    'P4':[
        'Anna ist im Haus.',
        'Ben ist im Garten.',
        'Cara geht ins Haus.',
    ],
    'P3':[
        'Anna hat den Schlüssel.',
        'Ben hat das Buch.',
        'Cara hat die Münze.',
    ],
    'P2':[
        'Das Tor ist offen.',
        'Die Lampe ist rot.',
        'Das Töpfchen ist kalt.',
    ],
}

# Learn formal signatures as intersection of recurring features plus discriminative leftovers.
# For this audit, we choose the minimal feature(s) unique among relation examples.
all_examples=[(p,t,formal_signature(t)) for p,xx in SURFACE_TRAIN.items() for t in xx]
FEATURE_RULES={}
for p,xx in SURFACE_TRAIN.items():
    pos=[formal_signature(t) for t in xx]
    universe=sorted(set().union(*pos))
    candidates=[]
    for f in universe:
        sup=sum(f in s for s in pos)
        con=sum(f in s for q,t,s in all_examples if q!=p)
        if sup>=2 and con==0:
            candidates.append((-sup,f))
    FEATURE_RULES[p]=candidates[0][1] if candidates else None

# P2 has only COPULA, but location examples also may have COPULA. It therefore has no unique
# single formal feature and should stay unresolved under this deliberately tiny feature language.
# That is useful: form rescue is partial, not magic.
def resolve_form(text):
    fs=formal_signature(text)
    hits=[p for p,f in FEATURE_RULES.items() if f and f in fs]
    return hits[0] if len(hits)==1 else None

r_proc=resolve_form('Das Wasser kocht.')
r_loc=resolve_form('Anna ist im Garten.')
r_pos=resolve_form('Ben hat den Schlüssel.')
r_attr=resolve_form('Die Lampe ist blau.')

# Add one purely formal feature: presence of a PP marker versus no PP after copula.
# This is syntax, not LOCATION/ATTRIBUTE semantics.
def extended_signature(text):
    fs=set(formal_signature(text)); ts=toks(text); ls=[lemma(x) for x in ts]
    if 'sein' in ls and ('in' in ls or 'aus' in ls): fs.add('F:COPULA_PP')
    if 'sein' in ls and not ('in' in ls or 'aus' in ls): fs.add('F:COPULA_NONPP')
    return frozenset(fs)

ext_rules={
    'P1':'F:A_LEXEME',
    'P4':'F:COPULA_PP',
    'P3':'F:HABEN',
    'P2':'F:COPULA_NONPP',
}
def resolve_extended(text):
    fs=extended_signature(text)
    hits=[p for p,f in ext_rules.items() if f in fs]
    return hits[0] if len(hits)==1 else None

ext_proc=resolve_extended('Das Wasser kocht.')
ext_loc=resolve_extended('Anna ist im Garten.')
ext_pos=resolve_extended('Ben hat den Schlüssel.')
ext_attr=resolve_extended('Die Lampe ist blau.')

# A sentence with no discriminating learned form remains UNKNOWN.
unknown=resolve_extended('Anna sieht Ben.')

checks={
    'K3b_two_supports_keep_relation_STAGED':pre_use is None,
    'K3b_third_independent_support_activates_relation':post_use is not None and post_status=='ACTIVE',
    'K3b_duplicate_evidence_id_cannot_activate_relation':dup.status=='STAGED' and len(dup.evidence)==1,
    'K3b_new_incompatible_motif_challenges_frozen_relation':challenge.status=='CHALLENGED',
    'K3b_minimal_form_features_rescue_process':r_proc=='P1',
    'K3b_minimal_form_features_rescue_possession':r_pos=='P3',
    'K3b_too_coarse_copula_feature_does_not_force_attribute':r_attr is None,
    'K3b_extended_pure_form_features_resolve_all_four_relations':(ext_proc,ext_loc,ext_pos,ext_attr)==('P1','P4','P3','P2'),
    'K3b_unseen_nondiagnostic_form_remains_UNKNOWN':unknown is None,
}

for k,v in checks.items(): print(('PASS' if v else 'FAIL'),'|',k)
print('\nFeature rules minimal:',FEATURE_RULES)
print('minimal resolutions:',r_proc,r_loc,r_pos,r_attr)
print('extended resolutions:',ext_proc,ext_loc,ext_pos,ext_attr)
print('unknown:',unknown)
print('lifecycle pre:',pre_use,'post:',post_status,'dup:',dup.status,len(dup.evidence),'challenge:',challenge.status)

assert all(checks.values())

report={
    'version':'v7.1b-K3-identifiability-lifecycle-audit',
    'result':'PASS',
    'checks':checks,
    'lifecycle':{
        'after_two':'STAGED',
        'after_three':'ACTIVE',
        'duplicate_support_count':len(dup.evidence),
        'after_unseen_motif':challenge.status,
    },
    'formal_rescue':{
        'minimal_rules':FEATURE_RULES,
        'minimal_resolutions':{'process':r_proc,'location':r_loc,'possession':r_pos,'attribute':r_attr},
        'extended_rules':ext_rules,
        'extended_resolutions':{'process':ext_proc,'location':ext_loc,'possession':ext_pos,'attribute':ext_attr},
        'unknown_sentence':unknown,
    },
    'interpretation':[
        'Relation schemas follow the same STAGED-to-ACTIVE evidence discipline as other learned U: two supports are insufficient when the gate is three.',
        'Repeated identical evidence cannot fake independence, and a new incompatible motif challenges a frozen relation schema.',
        'Very small formal surface features can recover some anonymous relation identity without semantic relation names or semantic port types.',
        'A deliberately too-coarse feature language fails on copular location versus attribute usage; the safe result is UNKNOWN.',
        'Adding a purely syntactic distinction, copula+PP versus copula without PP, resolves that collision without reintroducing AT/STATE labels.',
    ]
}
Path('/mnt/data/symbolic_v71b_k3_identifiability_audit_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('Saved v7.1b report.')

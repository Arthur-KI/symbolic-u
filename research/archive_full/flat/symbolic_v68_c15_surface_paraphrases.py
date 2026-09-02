from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
from itertools import combinations
import importlib.util, sys, contextlib, io, re, json, csv

# v6.8 / C15 — Surface paraphrases -> primitive State-U
# Frozen base: v6.7
spec=importlib.util.spec_from_file_location('v67f','/mnt/data/symbolic_v67_raw_text_family_invention.py')
v=importlib.util.module_from_spec(spec); sys.modules['v67f']=v
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(v)
assert all(v.checks.values())

ENT={'lampe':'LAMP','rad':'WHEEL','maschine':'MACHINE','tor':'GATE','anna':'ANNA','ben':'BEN','cara':'CARA','garten':'GARDEN','haus':'HOUSE','wald':'FOREST','zimmer':'ROOM','töpfchen':'POT','toepfchen':'POT','topf':'POT'}
ACTION={'leuchtet':'LIGHT','leuchten':'LIGHT','leuchtete':'LIGHT','dreht':'TURN','drehen':'TURN','drehte':'TURN','läuft':'RUN','laeuft':'RUN','laufen':'RUN','lief':'RUN','öffnet':'OPEN','oeffnet':'OPEN','öffnen':'OPEN','oeffnen':'OPEN','kocht':'COOK','kochen':'COOK','kochte':'COOK'}
PERSON={'ANNA','BEN','CARA'}; PLACE={'GARDEN','HOUSE','FOREST','ROOM'}; MACHINE={'LAMP','WHEEL','MACHINE','GATE','POT'}
DEFAULT_ACTION={'LAMP':'LIGHT','WHEEL':'TURN','MACHINE':'RUN','GATE':'OPEN','POT':'COOK'}
def toks(x): return re.findall(r'[A-Za-zÄÖÜäöüß]+',x.lower())

@dataclass(frozen=True)
class Surface:
    text:str; slots:tuple[str,...]; slot_types:tuple[str,...]; cues:frozenset[str]

def surf(text):
    ts=toks(text); ents=[ENT[x] for x in ts if x in ENT]; acts=[ACTION[x] for x in ts if x in ACTION]; c=set()
    if any(x in ts for x in {'hört','hoert','hörte','hoerte'}) and 'auf' in ts and 'zu' in ts: c.add('CESSATIVE_PERIPH')
    if 'nicht' in ts and 'mehr' in ts: c.add('NEG_MORE')
    if any(x in ts for x in {'erlischt','erlosch'}): c.add('LEX_CEASE')
    if 'geht' in ts and 'aus' in ts: c.add('PARTICLE_OUT')
    if any(x in ts for x in {'beginnt','begann','anfängt','anfaengt','fängt','faengt'}) and 'zu' in ts: c.add('INCHOATIVE_PERIPH')
    if 'jetzt' in ts: c.add('NOW')
    if 'geht' in ts and 'an' in ts: c.add('PARTICLE_ON')
    if any(x in ts for x in {'betritt','betrat'}): c.add('ENTER_LEX')
    if 'ins' in ts or 'hinein' in ts or 'herein' in ts: c.add('INWARD')
    if any(x in ts for x in {'verlässt','verlaesst','verließ','verliess'}): c.add('LEAVE_LEX')
    if 'hinaus' in ts or 'heraus' in ts or ('aus' in ts and ('geht' in ts or 'ging' in ts)): c.add('OUTWARD')
    if 'geht' in ts or 'ging' in ts: c.add('GO')
    if acts: c.add('HAS_ACTION')
    if ents: c.add('HAS_ENTITY')
    target=next((x for x in ents if x in MACHINE),None)
    if target and len(set(acts))==1:
        return Surface(text,(target,acts[0]),('ENTITY','ACTION'),frozenset(c))
    # Lexical/particle paraphrases may omit the base action verb itself.
    # Use only a primitive capability prior to fill the ACTION port; the
    # transition polarity still comes from the learned Surface-U.
    if target and ({'LEX_CEASE','PARTICLE_OUT','PARTICLE_ON'} & c):
        inferred=DEFAULT_ACTION.get(target)
        if inferred:
            c.add('ACTION_FROM_CAPABILITY')
            return Surface(text,(target,inferred),('ENTITY','ACTION'),frozenset(c))
    person=next((x for x in ents if x in PERSON),None); place=next((x for x in ents if x in PLACE),None)
    if person and place: return Surface(text,(person,place),('PERSON','PLACE'),frozenset(c))
    return Surface(text,(),(),frozenset(c))

@dataclass(frozen=True)
class Delta:
    removed:tuple; added:tuple
STOP=Delta((('ACTIVE',('S0','S1')),),())
START=Delta((),(('ACTIVE',('S0','S1')),))
ENTER=Delta((),(('AT',('S0','S1')),))
LEAVE=Delta((('AT',('S0','S1')),),())
NAME={STOP:'STOP',START:'START',ENTER:'ENTER',LEAVE:'LEAVE'}
@dataclass(frozen=True)
class Ex: text:str; delta:Delta
TRAIN=[
Ex('Die Lampe hört auf zu leuchten.',STOP),Ex('Das Rad dreht sich nicht mehr.',STOP),Ex('Die Lampe erlischt.',STOP),Ex('Die Lampe geht aus.',STOP),
Ex('Die Lampe beginnt zu leuchten.',START),Ex('Das Rad beginnt zu drehen.',START),Ex('Die Lampe fängt an zu leuchten.',START),Ex('Die Lampe geht an.',START),
Ex('Anna betritt das Haus.',ENTER),Ex('Ben geht ins Haus.',ENTER),Ex('Cara geht hinein ins Zimmer.',ENTER),
Ex('Anna verlässt das Haus.',LEAVE),Ex('Ben geht aus dem Haus.',LEAVE),Ex('Cara geht hinaus aus dem Zimmer.',LEAVE)]
SUR=[surf(x.text) for x in TRAIN]

@dataclass(frozen=True)
class Rule:
    delta:Delta; required:frozenset[str]; slot_types:tuple[str,...]; support:int; conflict:int

def learn(delta):
    pos=[s for s,e in zip(SUR,TRAIN) if e.delta==delta and s.slots]; neg=[s for s,e in zip(SUR,TRAIN) if e.delta!=delta and s.slots]
    types=pos[0].slot_types; universe=sorted(set().union(*(p.cues for p in pos))); cand=[]
    for n in range(1,min(3,len(universe))+1):
        for co in combinations(universe,n):
            req=frozenset(co); sup=sum(req<=p.cues for p in pos); con=sum(x.slot_types==types and req<=x.cues for x in neg)
            if sup and con==0: cand.append((n,-sup,tuple(sorted(req)),req))
    uncovered=set(range(len(pos))); out=[]
    while uncovered:
        opts=[]
        for n,ns,key,req in cand:
            covered={i for i,p in enumerate(pos) if req<=p.cues}; gain=len(covered&uncovered)
            if gain: opts.append((-gain,n,key,req,covered))
        assert opts; opts.sort(); _,_,_,req,covered=opts[0]; out.append(Rule(delta,req,types,len(covered),0)); uncovered-=covered
    uniq=[]; seen=set()
    for r in out:
        if r.required not in seen: seen.add(r.required); uniq.append(r)
    return uniq
RULES=sum((learn(d) for d in (STOP,START,ENTER,LEAVE)),[])

def classify(text):
    s=surf(text); ds=[]
    for r in RULES:
        if s.slot_types==r.slot_types and r.required<=s.cues and r.delta not in ds: ds.append(r.delta)
    return (ds[0] if len(ds)==1 else None,s)

def ground(d,s):
    if d is None or not s.slots: return None
    mp={f'S{i}':x for i,x in enumerate(s.slots)}
    def gk(k): rel,args=k; return (rel,tuple(mp.get(a,a) for a in args))
    return Delta(tuple(gk(k) for k in d.removed),tuple(gk(k) for k in d.added))

FROZEN=[
('Die Maschine hört auf zu laufen.',STOP),('Das Rad läuft nicht mehr.',STOP),('Die Lampe erlischt.',STOP),
('Die Maschine beginnt zu laufen.',START),('Das Tor beginnt zu öffnen.',START),
('Cara betritt den Garten.',ENTER),('Anna geht hinein ins Haus.',ENTER),
('Cara verlässt den Garten.',LEAVE),('Anna geht hinaus aus dem Haus.',LEAVE)]
FP=[classify(t)[0] for t,g in FROZEN]; FROZEN_PASS=all(p==g for p,(t,g) in zip(FP,FROZEN))
ADV=[('Die Lampe leuchtet nicht.',None),('Die Lampe leuchtet jetzt.',None),('Anna geht.',None),('Anna ist im Haus.',None),('Die Lampe leuchtet.',None),('Anna kommt aus dem Haus.',None)]
AP=[classify(t)[0] for t,g in ADV]; ADV_PASS=all(p==g for p,(t,g) in zip(AP,ADV))

@dataclass
class State: pos:set=field(default_factory=set)
def apply(st,g):
    for k in g.removed: st.pos.discard(k)
    for k in g.added: st.pos.add(k)
def interpret(st,text):
    d,s=classify(text)
    if d is None: return None
    g=ground(d,s); apply(st,g); return g
state=State({('ACTIVE',('LAMP','LIGHT')),('AT',('ANNA','HOUSE'))})
MIX=['Die Lampe erlischt.','Die Maschine beginnt zu laufen.','Anna verlässt das Haus.','Cara betritt den Garten.']
MG=[interpret(state,x) for x in MIX]
MIX_PASS=all(MG) and state.pos=={('ACTIVE',('MACHINE','RUN')),('AT',('CARA','GARDEN'))}

# C14 handoff: compare learned STOP delta with frozen C14 family topology.
@dataclass(frozen=True)
class FC: token:str; slots:tuple[str,...]
@dataclass(frozen=True)
class FT: command:FC; before:frozenset; after:frozenset

def c14_sig(token,slots,g):
    return v.canonical_signature(FT(FC(token,tuple(slots)),frozenset(g.removed),frozenset(g.added)))
d,s=classify('Die Maschine hört auf zu laufen.'); g=ground(d,s); sig=c14_sig('opaque',s.slots,g); family=v.FAMILY_BY_SIG.get(sig)
eval_stop=v.extract_raw_transitions('Die Lampe leuchtet. "Lampe rul leuchten." Danach hört die Lampe auf zu leuchten.','eval','eval')[0]
expected=v.FAMILY_BY_SIG[v.canonical_signature(eval_stop)]
HANDOFF=(family==expected)

# Two different paraphrases support same opaque lexical family.
def lex_family(token,target,action,response):
    d,s=classify(response)
    if d!=STOP: return None
    g=ground(d,s)
    if ('ACTIVE',(target,action)) not in g.removed: return None
    return v.FAMILY_BY_SIG.get(c14_sig(token,(target,action),g))
LEX=[lex_family('plim','LAMP','LIGHT','Die Lampe erlischt.'),lex_family('plim','MACHINE','RUN','Die Maschine hört auf zu laufen.')]
LEX_PASS=LEX==[expected,expected]

# Grimm diagnosis for "auf zu kochen" clauses. Extend action lexicon locally only with COOK forms already known by prior curriculum.
ACTION.update({'kocht':'COOK','kochen':'COOK','kochte':'COOK'})
GRIMM=Path('/mnt/data/grimm_der_suesse_brei.txt').read_text(encoding='utf-8')
clauses=[x.strip() for x in re.split(r'[.!?]',GRIMM.replace('\n',' ')) if 'auf zu kochen' in x.lower()]
GC=[classify(x)[0] for x in clauses]; GRIMM_PASS=(len(clauses)>=1 and all(x==STOP for x in GC))

checks={
'frozen_v67_base_stays_green':all(v.checks.values()),
'C15_learns_rules_for_four_primitive_delta_types':set(r.delta for r in RULES)=={STOP,START,ENTER,LEAVE},
'C15_all_surface_rules_zero_conflict_on_training':all(r.conflict==0 for r in RULES),
'C15_frozen_paraphrases_transfer':FROZEN_PASS,
'C15_adversarial_minimal_pairs_unknown':ADV_PASS,
'C15_mixed_story_grounding_and_state_update':MIX_PASS,
'C15_handoff_matches_existing_C14_stop_family':HANDOFF,
'C15_distinct_stop_paraphrases_support_same_opaque_family':LEX_PASS,
'grimm_auf_zu_kochen_surface_maps_to_STOP':GRIMM_PASS,
}
print('=== v6.8 C15 SURFACE PARAPHRASES -> PRIMITIVE STATE-U ===')
for k,z in checks.items(): print(('PASS' if z else 'FAIL'),'|',k)
print('\nRules:')
for r in RULES: print(NAME[r.delta],sorted(r.required),r.slot_types,'support',r.support,'conflict',r.conflict)
print('\nFrozen:')
for (t,g),p in zip(FROZEN,FP): print(t,'=>',None if p is None else NAME[p],'expected',NAME[g])
print('\nAdversarial:')
for (t,g),p in zip(ADV,AP): print(t,'=>',None if p is None else NAME[p])
print('\nMixed final state:',sorted(state.pos))
print('C14 handoff:',family,'expected',expected)
print('Lexical paraphrase evidence:',LEX)
print('Grimm clauses:',[(x[:160],None if p is None else NAME[p]) for x,p in zip(clauses,GC)])
assert all(checks.values())
report={'version':'v6.8-c15-surface-paraphrases','result':'PASS','checks':checks,'rules':[{'delta':NAME[r.delta],'required':sorted(r.required),'slot_types':list(r.slot_types),'support':r.support,'conflict':r.conflict} for r in RULES],'c14_handoff':{'family':family,'expected':expected},'grimm':{'clauses':len(clauses),'stop_recall':sum(x==STOP for x in GC)},'caveats':['Cue extraction remains supplied symbolic infrastructure.','Only STOP/START/ENTER/LEAVE are covered.','Primitive ACTIVE/AT and slot types remain fixed ontology anchors.','C15 training is supervised by primitive delta labels.','This is controlled German paraphrase learning, not unrestricted parsing.']}
Path('/mnt/data/symbolic_v68_c15_surface_paraphrases_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
with Path('/mnt/data/symbolic_v68_c15_surface_paraphrases_checks.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['check','passed']); [w.writerow([k,z]) for k,z in checks.items()]

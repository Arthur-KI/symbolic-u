from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
import importlib.util, sys, contextlib, io, re, json, csv, copy

# v6.6 C13: anonymous semantic-family invention from before/after state deltas.
# Frozen base: v6.5.
spec=importlib.util.spec_from_file_location('v65f','/mnt/data/symbolic_v65_unknown_imperative.py')
v=importlib.util.module_from_spec(spec); sys.modules['v65f']=v
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(v)
assert all(v.checks.values())

Key=tuple[str,tuple[str,...]]
@dataclass(frozen=True)
class Command:
    token:str
    slots:tuple[str,...]
@dataclass(frozen=True)
class Experience:
    evidence_id:str
    domain:str
    command:Command
    before:frozenset[Key]
    after:frozenset[Key]
@dataclass(frozen=True)
class Signature:
    removed:tuple[tuple[str,tuple[str,...]],...]
    added:tuple[tuple[str,tuple[str,...]],...]

def E(eid,domain,token,slots,before,after):
    return Experience(eid,domain,Command(token,tuple(slots)),frozenset(before),frozenset(after))

def canonical_signature(ex:Experience):
    removed=sorted(ex.before-ex.after); added=sorted(ex.after-ex.before)
    slotmap={x:f'S{i}' for i,x in enumerate(ex.command.slots)}
    anon={}; nxt=[0]
    def a(x):
        if x in slotmap: return slotmap[x]
        if x not in anon:
            anon[x]=f'V{nxt[0]}'; nxt[0]+=1
        return anon[x]
    def k(item):
        rel,args=item; return (rel,tuple(a(x) for x in args))
    return Signature(tuple(k(x) for x in removed),tuple(k(x) for x in added))

# 7 anonymous families, three cross-domain examples each. Tokens are unique inside training,
# so token identity cannot define a family.
train=[]
train += [
 E('a1','home','dax',('LAMP','LIGHT'),[],[('ACTIVE',('LAMP','LIGHT'))]),
 E('a2','machine','miv',('WHEEL','TURN'),[],[('ACTIVE',('WHEEL','TURN'))]),
 E('a3','factory','sop',('MACHINE','RUN'),[],[('ACTIVE',('MACHINE','RUN'))]),]
train += [
 E('b1','home','rul',('LAMP','LIGHT'),[('ACTIVE',('LAMP','LIGHT'))],[]),
 E('b2','machine','kem',('WHEEL','TURN'),[('ACTIVE',('WHEEL','TURN'))],[]),
 E('b3','factory','nax',('MACHINE','RUN'),[('ACTIVE',('MACHINE','RUN'))],[]),]
train += [
 E('c1','travel','pud',('ANNA','GARDEN'),[],[('AT',('ANNA','GARDEN'))]),
 E('c2','city','zef',('BEN','HOUSE'),[],[('AT',('BEN','HOUSE'))]),
 E('c3','fairy','lom',('GIRL','FOREST'),[],[('AT',('GIRL','FOREST'))]),]
train += [
 E('d1','travel','vek',('ANNA','GARDEN'),[('AT',('ANNA','GARDEN'))],[]),
 E('d2','city','fud',('BEN','HOUSE'),[('AT',('BEN','HOUSE'))],[]),
 E('d3','fairy','raq',('GIRL','FOREST'),[('AT',('GIRL','FOREST'))],[]),]
train += [
 E('e1','office','tir',('ANNA','BEN','KEY'),[('HAVE',('ANNA','KEY'))],[('HAVE',('BEN','KEY'))]),
 E('e2','school','wex',('GIRL','BOY','BOOK'),[('HAVE',('GIRL','BOOK'))],[('HAVE',('BOY','BOOK'))]),
 E('e3','market','jop',('CARA','ANNA','COIN'),[('HAVE',('CARA','COIN'))],[('HAVE',('ANNA','COIN'))]),]
train += [
 E('f1','office','gax',('ANNA','KEY'),[],[('HAVE',('ANNA','KEY'))]),
 E('f2','school','bim',('BOY','BOOK'),[],[('HAVE',('BOY','BOOK'))]),
 E('f3','market','qus',('CARA','COIN'),[],[('HAVE',('CARA','COIN'))]),]
train += [
 E('g1','device','fep',('LAMP','RED'),[('STATE',('LAMP','BLUE'))],[('STATE',('LAMP','RED'))]),
 E('g2','device2','duk',('GATE','OPEN_STATE'),[('STATE',('GATE','CLOSED_STATE'))],[('STATE',('GATE','OPEN_STATE'))]),
 E('g3','fairy','pol',('POT','HOT'),[('STATE',('POT','COLD'))],[('STATE',('POT','HOT'))]),]

by=defaultdict(list)
for ex in train: by[canonical_signature(ex)].append(ex)
elig=[]
for sig,xs in by.items():
    if len({x.evidence_id for x in xs})>=3 and len({x.domain for x in xs})>=2:
        elig.append((sig,xs))
assert len(elig)==7
elig.sort(key=lambda z:repr(z[0]))
FAMILY_BY_SIG={}; SIG_BY_FAMILY={}; META={}
for i,(sig,xs) in enumerate(elig,11):
    r=f'R{i}'; FAMILY_BY_SIG[sig]=r; SIG_BY_FAMILY[r]=sig
    META[r]={'support':len(xs),'domains':len({x.domain for x in xs}),'tokens':sorted(x.command.token for x in xs)}

# Evaluator-only topology names; learner never gets these names.
def fam(ex): return FAMILY_BY_SIG[canonical_signature(ex)]
EVAL={
 'START':fam(E('z','z','x',('LAMP','LIGHT'),[],[('ACTIVE',('LAMP','LIGHT'))])),
 'STOP':fam(E('z','z','x',('LAMP','LIGHT'),[('ACTIVE',('LAMP','LIGHT'))],[])),
 'ENTER':fam(E('z','z','x',('ANNA','GARDEN'),[],[('AT',('ANNA','GARDEN'))])),
 'LEAVE':fam(E('z','z','x',('ANNA','GARDEN'),[('AT',('ANNA','GARDEN'))],[])),
 'TRANSFER':fam(E('z','z','x',('ANNA','BEN','KEY'),[('HAVE',('ANNA','KEY'))],[('HAVE',('BEN','KEY'))])),
 'GAIN':fam(E('z','z','x',('ANNA','KEY'),[],[('HAVE',('ANNA','KEY'))])),
 'CHANGE':fam(E('z','z','x',('LAMP','RED'),[('STATE',('LAMP','BLUE'))],[('STATE',('LAMP','RED'))])),
}
assert len(set(EVAL.values()))==7

@dataclass
class Lex:
    token:str; family:str; status:str='STAGED'
    evidence_ids:set[str]=field(default_factory=set)
    conflicts:set[str]=field(default_factory=set)
    @property
    def support(self): return len(self.evidence_ids)
class FamilyLexicon:
    def __init__(self,min_support=2): self.min_support=min_support; self.entries={}; self.lifecycle=[]
    def observe(self,ex:Experience):
        family=FAMILY_BY_SIG.get(canonical_signature(ex))
        # no observed delta / unknown topology => no entry
        if family is None or (not (ex.before-ex.after) and not (ex.after-ex.before)): return None
        t=ex.command.token
        if t not in self.entries:
            self.entries[t]=Lex(t,family); self.lifecycle.append(('STAGED',t,family,ex.evidence_id))
        ent=self.entries[t]
        if family==ent.family:
            ent.evidence_ids.add(ex.evidence_id)
            if ent.status=='STAGED' and ent.support>=self.min_support:
                ent.status='ACTIVE'; self.lifecycle.append(('ACTIVE',t,family,ex.evidence_id))
        else:
            ent.conflicts.add(family)
            if ent.status=='ACTIVE': ent.status='CHALLENGED'; self.lifecycle.append(('CHALLENGED',t,family,ex.evidence_id))
        return ent
    def recognize(self,token,slots):
        ent=self.entries.get(token)
        return (ent.family,tuple(slots)) if ent and ent.status=='ACTIVE' else None

# Held-out lexical words for all 7 families.
held={
 'START':[
   E('hs1','lab','zor',('LAMP','LIGHT'),[],[('ACTIVE',('LAMP','LIGHT'))]),
   E('hs2','factory','zor',('MACHINE','RUN'),[],[('ACTIVE',('MACHINE','RUN'))])],
 'STOP':[
   E('hp1','lab','plim',('LAMP','LIGHT'),[('ACTIVE',('LAMP','LIGHT'))],[]),
   E('hp2','factory','plim',('MACHINE','RUN'),[('ACTIVE',('MACHINE','RUN'))],[])],
 'ENTER':[
   E('he1','travel','nex',('ANNA','GARDEN'),[],[('AT',('ANNA','GARDEN'))]),
   E('he2','fairy','nex',('GIRL','FOREST'),[],[('AT',('GIRL','FOREST'))])],
 'LEAVE':[
   E('hl1','travel','vak',('ANNA','GARDEN'),[('AT',('ANNA','GARDEN'))],[]),
   E('hl2','fairy','vak',('GIRL','FOREST'),[('AT',('GIRL','FOREST'))],[])],
 'TRANSFER':[
   E('ht1','office','tirx',('ANNA','BEN','KEY'),[('HAVE',('ANNA','KEY'))],[('HAVE',('BEN','KEY'))]),
   E('ht2','school','tirx',('GIRL','BOY','BOOK'),[('HAVE',('GIRL','BOOK'))],[('HAVE',('BOY','BOOK'))])],
 'GAIN':[
   E('hg1','office','mup',('ANNA','KEY'),[],[('HAVE',('ANNA','KEY'))]),
   E('hg2','market','mup',('CARA','COIN'),[],[('HAVE',('CARA','COIN'))])],
 'CHANGE':[
   E('hc1','device','fel',('LAMP','RED'),[('STATE',('LAMP','BLUE'))],[('STATE',('LAMP','RED'))]),
   E('hc2','fairy','fel',('POT','HOT'),[('STATE',('POT','COLD'))],[('STATE',('POT','HOT'))])],
}
lex=FamilyLexicon(); status={}
for h,xs in held.items():
    a=lex.observe(xs[0]); s1=(a.status,a.support,a.family)
    b=lex.observe(xs[1]); s2=(b.status,b.support,b.family)
    status[h]=(s1,s2)
reuse={
 'START':lex.recognize('zor',('WHEEL','TURN')),
 'STOP':lex.recognize('plim',('WHEEL','TURN')),
 'ENTER':lex.recognize('nex',('BEN','HOUSE')),
 'LEAVE':lex.recognize('vak',('BEN','HOUSE')),
 'TRANSFER':lex.recognize('tirx',('CARA','ANNA','COIN')),
 'GAIN':lex.recognize('mup',('BOY','BOOK')),
 'CHANGE':lex.recognize('fel',('GATE','OPEN_STATE')),
}

# Adversarial lifecycle.
conf=copy.deepcopy(lex)
conf.observe(E('conf','travel','plim',('ANNA','GARDEN'),[],[('AT',('ANNA','GARDEN'))]))
plim_conf=conf.entries['plim'].status
none=FamilyLexicon(); no_delta=E('n0','lab','noop',('LAMP','LIGHT'),[('ACTIVE',('LAMP','LIGHT'))],[('ACTIVE',('LAMP','LIGHT'))]); no_ent=none.observe(no_delta)
dup=FamilyLexicon(); dex=held['STOP'][0]; dup.observe(dex); dup.observe(dex); dup_ent=dup.entries['plim']
semantic=lex.recognize('plim',('GATE','OPEN')); world={('ACTIVE',('GATE','OPEN'))}; before=set(world); after=set(world)

# Frozen Grimm integration. Use v6.5 raw helpers; family selection comes from C13, not STOP-specific learner.
GRIMM=Path('/mnt/data/grimm_der_suesse_brei.txt').read_text(encoding='utf-8')
G=GRIMM.replace('„','"').replace('“','"')
spans=list(re.finditer(r'"([^"]+)"',G,re.S))
glex=FamilyLexicon(); active={}; steh_trace=[]
def local_after(sp,n=180): return G[sp.end():min(len(G),sp.end()+n)]
def tt(x): return re.findall(r'[A-Za-zÄÖÜäöüß]+',x.lower())
for i,sp in enumerate(spans):
    q=sp.group(1); target=v.target_from_quote(q); known=v.known_action_from_quote(q); unknown=v.unknown_imperative_token(q)
    if target and known:
        ts=tt(local_after(sp)); observed=(target=='POT' and known=='COOK' and ('kocht' in ts or 'kochte' in ts) and 'es' in ts)
        if observed: active[target]=known
        continue
    if target and unknown:
        pre=active.get(target); cease=v.extract_cease(local_after(sp),target)
        if pre and cease==(target,pre):
            ex=E(f'grimm-{i}','grimm',unknown,(target,pre),[('ACTIVE',(target,pre))],[])
            ent=glex.observe(ex)
            steh_trace.append({'evidence_id':ex.evidence_id,'token':unknown,'family':ent.family,'status':ent.status,'support':ent.support,'pre_action':pre,'cease':cease})
            active.pop(target,None)
STEH=glex.entries.get('steh'); grimm_reuse=glex.recognize('steh',('LAMP','LIGHT'))

# Novel transition classification on unseen arguments.
frozen={
 'START':E('fx1','new','x',('GATE','OPEN'),[],[('ACTIVE',('GATE','OPEN'))]),
 'STOP':E('fx2','new','x',('GATE','OPEN'),[('ACTIVE',('GATE','OPEN'))],[]),
 'ENTER':E('fx3','new','x',('CARA','HOUSE'),[],[('AT',('CARA','HOUSE'))]),
 'LEAVE':E('fx4','new','x',('CARA','HOUSE'),[('AT',('CARA','HOUSE'))],[]),
 'TRANSFER':E('fx5','new','x',('BEN','CARA','KEY'),[('HAVE',('BEN','KEY'))],[('HAVE',('CARA','KEY'))]),
 'GAIN':E('fx6','new','x',('GIRL','COIN'),[],[('HAVE',('GIRL','COIN'))]),
 'CHANGE':E('fx7','new','x',('GATE','OPEN_STATE'),[('STATE',('GATE','CLOSED_STATE'))],[('STATE',('GATE','OPEN_STATE'))]),
}
frozen_class={h:FAMILY_BY_SIG.get(canonical_signature(ex)) for h,ex in frozen.items()}

checks={
 'frozen_v65_base_stays_green':all(v.checks.values()),
 'C13_invents_seven_distinct_anonymous_families':len(FAMILY_BY_SIG)==7,
 'C13_all_family_heads_are_anonymous':all(re.fullmatch(r'R\d+',r) for r in FAMILY_BY_SIG.values()),
 'C13_each_family_cross_domain_support_at_least_three':all(m['support']>=3 and m['domains']>=2 for m in META.values()),
 'C13_family_invention_does_not_use_token_identity':all(len(set(m['tokens']))==m['support'] for m in META.values()),
 'C13_all_seven_heldout_tokens_stage_then_activate':all(a[0]=='STAGED' and a[1]==1 and b[0]=='ACTIVE' and b[1]==2 for a,b in status.values()),
 'C13_all_heldout_tokens_choose_correct_invented_family':all(status[h][1][2]==EVAL[h] for h in held),
 'C13_active_meanings_reuse_on_new_arguments':all(reuse[h] and reuse[h][0]==EVAL[h] for h in reuse),
 'C13_family_conflict_challenges_active_lexical_meaning':plim_conf=='CHALLENGED',
 'C13_no_state_delta_creates_no_family_entry':no_ent is None,
 'C13_duplicate_evidence_id_cannot_activate':dup_ent.status=='STAGED' and dup_ent.support==1,
 'C13_semantic_recognition_does_not_mutate_world':semantic is not None and before==after,
 'grimm_two_steh_evidence_occurrences_found':len(steh_trace)==2,
 'grimm_first_steh_staged_second_active':len(steh_trace)==2 and steh_trace[0]['status']=='STAGED' and steh_trace[1]['status']=='ACTIVE',
 'grimm_steh_selects_independently_invented_remove_ACTIVE_family':STEH is not None and STEH.family==EVAL['STOP'],
 'grimm_steh_reuses_on_new_target_action':grimm_reuse is not None and grimm_reuse[0]==EVAL['STOP'],
 'frozen_novel_transitions_classify_all_seven_families':all(frozen_class[h]==EVAL[h] for h in frozen_class),
}

print('=== v6.6 C13 AUTONOMOUS SEMANTIC-FAMILY INVENTION ===')
for k,x in checks.items(): print(('PASS' if x else 'FAIL'),'|',k)
print('\nInvented anonymous families:')
for r in sorted(SIG_BY_FAMILY,key=lambda x:int(x[1:])): print(' ',r,SIG_BY_FAMILY[r],META[r])
print('\nEvaluator-only mapping:')
for h,r in EVAL.items(): print(' ',h,'=>',r)
print('\nHeld-out lifecycle:')
for h,(a,b) in status.items(): print(' ',h,a,'->',b,'reuse',reuse[h])
print('\nAdversarial: conflict',plim_conf,'no-delta',no_ent,'duplicate',dup_ent.status,dup_ent.support)
print('WORLD semantic-only:',semantic,'before',before,'after',after)
print('\nGrimm STEH:')
for x in steh_trace: print(' ',x)
print(' entry',None if STEH is None else {'family':STEH.family,'status':STEH.status,'support':STEH.support,'evidence_ids':sorted(STEH.evidence_ids)})
print(' evaluator STOP family',EVAL['STOP'],'reuse',grimm_reuse)
print('\nFrozen novel classification:')
for h,r in frozen_class.items(): print(' ',h,'=>',r,'expected',EVAL[h])
assert all(checks.values())

report={
 'version':'v6.6-autonomous-semantic-family-invention','result':'PASS','checks':checks,
 'families':{r:{'signature':{'removed':[[rel,list(args)] for rel,args in SIG_BY_FAMILY[r].removed],'added':[[rel,list(args)] for rel,args in SIG_BY_FAMILY[r].added]},**META[r]} for r in sorted(SIG_BY_FAMILY,key=lambda x:int(x[1:]))},
 'evaluator_only_mapping':EVAL,
 'heldout_lifecycle':{h:{'after_first':list(a),'after_second':list(b),'reuse':list(reuse[h]) if reuse[h] else None} for h,(a,b) in status.items()},
 'grimm':{'steh_occurrences':steh_trace,'entry':None if STEH is None else {'token':STEH.token,'family':STEH.family,'status':STEH.status,'support':STEH.support,'evidence_ids':sorted(STEH.evidence_ids)},'matches_remove_active_family':STEH is not None and STEH.family==EVAL['STOP'],'reuse':list(grimm_reuse) if grimm_reuse else None},
 'interpretation':[
  'The learner is not told that C13 is a STOP task; it clusters canonical before/after state-delta topologies and invents anonymous family heads.',
  'Seven different transition families emerge from recurring cross-domain state changes.',
  'Opaque lexical tokens are learned as members of one invented family via STAGED-to-ACTIVE lifecycle.',
  'Frozen Grimm steh selects the already independently invented remove-ACTIVE topology after two distinct evidence occurrences.',
  'Semantic family recognition never mutates WORLD state by itself.'
 ],
 'caveats':[
  'Primitive state predicates ACTIVE, AT, HAVE, STATE are fixed symbolic anchors.',
  'The delta canonicalizer is an OS prior, not learned from text.',
  'Family training is synthetic symbolic experience, not raw German family discovery.',
  'Family recurrence and lexical activation thresholds are hand-set.',
  'The two Grimm supports are distinct occurrences in one story, not two independent stories.',
  'Automatic extraction of arbitrary state deltas from free German remains open.'
 ]
}
Path('/mnt/data/symbolic_v66_semantic_family_invention_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
with Path('/mnt/data/symbolic_v66_semantic_family_invention_checks.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['check','passed']); [w.writerow([k,x]) for k,x in checks.items()]
print('\nSaved v6.6 report/checks.')

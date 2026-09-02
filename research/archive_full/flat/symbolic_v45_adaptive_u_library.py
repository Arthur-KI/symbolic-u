from __future__ import annotations
from pathlib import Path
from collections import defaultdict
import json, csv, time

# Reuse v4.4b definitions without invoking its CLI.
src=Path('/mnt/data/symbolic_v44b_recursive_u_verifier.py').read_text(encoding='utf-8')
ns={'__name__':'v44b_module'}
exec(src,ns)

SIG=ns['SIG']; Program=ns['Program']; synth_verified_recursive=ns['synth_verified_recursive']
make_task=ns['make_task']; validate=ns['validate']; numeric_facts=ns['numeric_facts']; World=ns['World']

class AdaptiveULibrary:
    def __init__(self, training_bank=None):
        self.programs={}
        self.meta={}
        self.training_bank=training_bank or {}
        self.learn_attempts=defaultdict(int)
        self.install_count=0
        self.events=[]

    def has(self,rel):
        return rel in self.programs

    def ensure(self,rel):
        if rel in self.programs:
            self.events.append({'event':'reuse','relation':rel})
            return True
        factory=self.training_bank.get(rel)
        if factory is None:
            self.events.append({'event':'no_training_bank','relation':rel})
            return False

        self.learn_attempts[rel]+=1
        self.events.append({'event':'learn_start','relation':rel})
        worlds,cfg,test=factory()
        bg,max_base,max_bg,hid=cfg
        t0=time.perf_counter()
        res=synth_verified_recursive(rel,bg,worlds,max_base,max_bg,hid)
        elapsed=time.perf_counter()-t0
        if not res.get('best'):
            self.events.append({'event':'reject_no_program','relation':rel})
            return False

        sc,freq,base,rec,cert,local=res['best']
        passed,n=validate(rel,res,test)
        total_pos=sum(len(w.positives) for w in worlds)
        accepted=(sc[0]==total_pos and sc[1]==0 and passed==n)
        gate={
            'train_support':sc[0], 'train_positive_n':total_pos,
            'train_conflict':sc[1], 'selftest_passed':passed, 'selftest_n':n,
            'certificate':cert, 'seconds':elapsed,
            'base_rule':base.text(), 'recursive_rule':rec.text(),
            'candidate_pairs':res['pair_total'],
            'verifier_accepted':res['verifier_accepted'],
            'full_evaluated':res['full_evaluated'],
            'accepted':accepted,
        }
        self.meta[rel]=gate
        if not accepted:
            self.events.append({'event':'reject_gate','relation':rel})
            return False

        self.programs[rel]=Program(SIG[rel],base,rec)
        self.install_count+=1
        self.events.append({'event':'installed','relation':rel})
        return True

    def prove(self,rel,args,world):
        if rel not in self.programs and not self.ensure(rel):
            return False
        prog=self.programs[rel]
        prog.reset()
        return bool(prog.prove(tuple(args),world))

BANK={
    'SUB':lambda: make_task('SUB'),
    'PATH':lambda: make_task('PATH'),
    'PATH_COST':lambda: make_task('PATH_COST'),
}

# New query world: it contains no SUB/PATH/PATH_COST target facts.
facts=defaultdict(set)
for rel,vals in numeric_facts(40,add=True).items():
    facts[rel].update(vals)
facts['COUNT'].update({('goats','N7'),('apples','N12')})
facts['REMOVED_COUNT'].update({('goats','N6'),('apples','N5')})
for a,b,c in [('Depot','Hub',3),('Hub','North',4),('North','Shop',2),('Hub','East',5)]:
    facts['EDGE'].add((a,b)); facts['EDGE_COST'].add((a,b,f'N{c}'))
facts['PACKAGE_AT'].update({('pkg1','Depot'),('pkg2','Hub')})
facts['DEST'].update({('pkg1','Shop'),('pkg2','East')})
world=World(dict(facts),[],[])

class AdaptiveSolver:
    def __init__(self,world,lib):
        self.world=world; self.lib=lib; self.trace=[]
    def direct(self,rel,args):
        return tuple(args) in self.world.facts.get(rel,set())
    def prove(self,rel,args):
        args=tuple(args); self.trace.append(('QUERY',rel,args))
        if self.direct(rel,args):
            self.trace.append(('DIRECT',rel,args)); return True
        if rel in {'SUB','PATH','PATH_COST'}:
            ok=self.lib.prove(rel,args,self.world); self.trace.append(('LIBRARY',rel,args,ok)); return ok
        if rel=='REMAINING_COUNT':
            g,z=args
            for cg,x in self.world.facts.get('COUNT',set()):
                if cg!=g: continue
                for rg,y in self.world.facts.get('REMOVED_COUNT',set()):
                    if rg==g and self.prove('SUB',(x,y,z)):
                        self.trace.append(('U+1','COUNT_REMOVE_SUB_TO_REMAIN',(g,z))); return True
            return False
        if rel=='ONE_REMAINS':
            g=args[0]
            if self.prove('REMAINING_COUNT',(g,'N1')):
                self.trace.append(('U+1','REMAIN_N1_TO_ONE_REMAINS',(g,))); return True
            return False
        if rel=='DELIVERY_REACHABLE':
            pkg=args[0]
            ss=[x[1] for x in self.world.facts.get('PACKAGE_AT',set()) if x[0]==pkg]
            ds=[x[1] for x in self.world.facts.get('DEST',set()) if x[0]==pkg]
            for s in ss:
                for d in ds:
                    if self.prove('PATH',(s,d)):
                        self.trace.append(('U+1','PACKAGE_DEST_PATH_TO_REACHABLE',(pkg,))); return True
            return False
        if rel=='ROUTE_COST':
            pkg,cost=args
            ss=[x[1] for x in self.world.facts.get('PACKAGE_AT',set()) if x[0]==pkg]
            ds=[x[1] for x in self.world.facts.get('DEST',set()) if x[0]==pkg]
            for s in ss:
                for d in ds:
                    if self.prove('PATH_COST',(s,d,cost)):
                        self.trace.append(('U+1','PACKAGE_DEST_PATHCOST_TO_ROUTECOST',(pkg,cost))); return True
            return False
        # Unknown relation: query does not create training data or a new relation.
        self.trace.append(('UNKNOWN_NO_RULE',rel,args)); return False

lib=AdaptiveULibrary(BANK); solver=AdaptiveSolver(world,lib)

# 1) SUB on demand via semantic query; then reuse.
b0=lib.install_count
q1=solver.prove('ONE_REMAINS',('goats',)); a1=lib.install_count
q2=solver.prove('REMAINING_COUNT',('apples','N7')); a2=lib.install_count

# 2) PATH on demand via semantic query; then reuse.
bp=lib.install_count
q3=solver.prove('DELIVERY_REACHABLE',('pkg1',)); a3=lib.install_count
q4=solver.prove('DELIVERY_REACHABLE',('pkg2',)); a4=lib.install_count

# 3) PATH_COST on demand and semantic consumption.
bc=lib.install_count
q5=solver.prove('ROUTE_COST',('pkg1','N9')); a5=lib.install_count
q6=solver.prove('ROUTE_COST',('pkg1','N10')); a6=lib.install_count

# Safety 1: no training bank => no synthesis.
bi=lib.install_count
q7=solver.prove('MYSTERY',('A','B')); ai=lib.install_count

# Safety 2: poisoned training labels must fail install gate.
def poisoned_path_task():
    train,cfg,test=make_task('PATH')
    poisoned=[]
    for w in train:
        pos=list(w.positives); neg=list(w.negatives)
        if pos: neg.append(pos[0])
        poisoned.append(World(w.facts,pos,neg))
    return poisoned,cfg,test

badlib=AdaptiveULibrary({'PATH':poisoned_path_task})
bad_ok=badlib.prove('PATH',('Depot','Shop'),world)

checks={
    'semantic_query_triggered_SUB_install': q1 and a1==b0+1 and lib.has('SUB'),
    'second_SUB_query_reused': q2 and a2==a1 and lib.learn_attempts['SUB']==1,
    'semantic_query_triggered_PATH_install': q3 and a3==bp+1 and lib.has('PATH'),
    'second_PATH_query_reused': q4 and a4==a3 and lib.learn_attempts['PATH']==1,
    'PATH_COST_installed_and_consumed': q5 and a5==bc+1 and lib.has('PATH_COST'),
    'wrong_PATH_COST_stays_unknown': (not q6) and a6==a5,
    'query_without_training_is_not_evidence': (not q7) and ai==bi,
    'poisoned_training_rejected': (not bad_ok) and not badlib.has('PATH') and badlib.install_count==0,
}

print('=== v4.5 ADAPTIVE U LIBRARY ===')
for k,v in checks.items(): print(('PASS' if v else 'FAIL'),'|',k)
print('\nInstalled library:')
for rel in sorted(lib.programs):
    m=lib.meta[rel]
    print(rel,'support',m['train_support'],'/',m['train_positive_n'],'conflict',m['train_conflict'],'selftest',m['selftest_passed'],'/',m['selftest_n'],'attempts',lib.learn_attempts[rel])
    print('  BASE:',m['base_rule']); print('  REC :',m['recursive_rule'])
print('\nQuery results:')
for k,v in {
    'ONE_REMAINS(goats)':q1,'REMAINING_COUNT(apples,N7)':q2,
    'DELIVERY_REACHABLE(pkg1)':q3,'DELIVERY_REACHABLE(pkg2)':q4,
    'ROUTE_COST(pkg1,N9)':q5,'ROUTE_COST(pkg1,N10)':q6,
    'MYSTERY(A,B)':q7,'poisoned PATH':bad_ok
}.items(): print(k,'=>','+1' if v else '0')
print('\nLifecycle:',[(e['event'],e['relation']) for e in lib.events])
print('Poisoned gate:',json.dumps(badlib.meta.get('PATH',{}),ensure_ascii=False))

assert all(checks.values()), checks

report={
    'version':'v4.5-adaptive-u-library',
    'checks':checks,
    'installed':{r:{**m,'learn_attempts':lib.learn_attempts[r]} for r,m in lib.meta.items()},
    'results':{
        'one_remains_goats':q1,'remaining_apples_7':q2,
        'pkg1_reachable':q3,'pkg2_reachable':q4,
        'pkg1_route_cost_9':q5,'pkg1_wrong_route_cost_10':q6,
        'mystery':q7,'poisoned_path':bad_ok,
    },
    'lifecycle_events':lib.events,
    'poisoned_test':{'installed':badlib.has('PATH'),'query_result':bad_ok,'gate':badlib.meta.get('PATH'),'events':badlib.events},
    'invariants':[
        'Query is a target and is never appended to training evidence.',
        'No TrainingBank entry means no synthesis and the result stays UNKNOWN.',
        'Install requires full training support, zero conflict, and complete frozen self-test success.',
        'Installed U are reused by later queries without relearning.',
        'Standard semantic rules consume learned SUB/PATH/PATH_COST as ordinary premises.',
        'Failed/poisoned learning attempts do not create an installed U.'
    ],
    'caveats':[
        'TrainingBank discovery is explicit; the system does not yet generate its own labeled training curriculum.',
        'Only SUB, PATH and PATH_COST are exercised in the on-demand lifecycle test.',
        'PATH_COST uses extensional ADD background facts in this benchmark; nested on-demand ADD learning is not tested here.',
        'The library stores one accepted program per relation; versioned competing hypotheses are future work.',
        'This is adaptive symbolic learning, not unrestricted self-modifying code.'
    ]
}
Path('/mnt/data/symbolic_v45_adaptive_u_library_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
with Path('/mnt/data/symbolic_v45_adaptive_u_library_checks.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['check','passed']); w.writerows(checks.items())
print('\nSaved report/checks.')

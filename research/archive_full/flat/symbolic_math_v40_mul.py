from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import random, json, csv

# Reuse frozen core objects from the actual model.
V1 = Path('/mnt/data/symbolic_mini_lm_v1.py').read_text(encoding='utf-8')
core = V1.split('# ============================================================\n# TEST SUITE')[0]
ns={}
exec(core,ns)
Truth=ns['Truth']; truth_name=ns['truth_name']; Proposition=ns['Proposition']; Key=ns['Key']; UTemplate=ns['UTemplate']; StoryContext=ns['StoryContext']

U_SUCC_PRED=UTemplate('SUCC_TO_PRED',('SUCC',),'PRED','REASONING')
U_TOTAL=UTemplate('GROUP_COUNT_PER_GROUP_MUL_TO_TOTAL',('GROUP_COUNT','PER_GROUP','MUL'),'TOTAL_ITEMS','REASONING')
U_COMPLETE=UTemplate('TOTAL_TARGET_TO_PACKING_COMPLETE',('TOTAL_ITEMS','TARGET_TOTAL'),'PACKING_COMPLETE','REASONING')

class StandardEngine:
    def __init__(self,ctx):
        self.ctx=ctx; self.router=None
        self.succ_pairs=set()
        self.pred_index={}
        self.number_nodes=set()
        for e in ctx.events:
            q=e.proposition
            if q.rel=='SUCC':
                self.succ_pairs.add(q)
                self.pred_index[q.args[1]]=q.args[0]
                self.number_nodes.update(q.args)
    def facts(self,rel=None):
        for e in self.ctx.events:
            if rel is None or e.proposition.rel==rel: yield e.proposition
    def direct(self,p):
        k=Key(p,self.ctx.story_id)
        pos=any(e.proposition==p for e in self.ctx.events)
        neg=any(e.proposition==p.opposite() for e in self.ctx.events)
        if pos and neg: k.truth=Truth.UNKNOWN; k.contradiction=True
        elif pos: k.truth=Truth.TRUE
        elif neg: k.truth=Truth.FALSE
        return k
    def predecessor(self,n,stack,record):
        # indexed lookup of the concrete SUCC premise; proof is still Standard-U
        lo=self.pred_index.get(n)
        if lo is None: return None,None
        p=Proposition('PRED',(n,lo))
        if self.prove(p,stack,record).truth==Truth.TRUE:
            return lo,p
        return None,None
    def prove(self,p,stack=None,record=True):
        if stack is None: stack=set()
        d=self.direct(p)
        if d.truth!=Truth.UNKNOWN or d.contradiction: return d
        k=Key(p,self.ctx.story_id)
        marker=('STD',p)
        if marker in stack: return k
        stack=set(stack); stack.add(marker)
        if p.rel=='PRED':
            hi,lo=p.args; support=Proposition('SUCC',(lo,hi))
            if support in self.succ_pairs:
                if record: self.ctx.add_u(U_SUCC_PRED,p,Truth.TRUE,inputs=(support,),source='standard-u',evidence=['SUCC inverse'])
                k.truth=Truth.TRUE; return k
        if p.rel=='TOTAL_ITEMS':
            kind,z=p.args
            for gc in self.facts('GROUP_COUNT'):
                if gc.args[0]!=kind: continue
                for pg in self.facts('PER_GROUP'):
                    if pg.args[0]!=kind: continue
                    mul=Proposition('MUL',(gc.args[1],pg.args[1],z))
                    if self.router.prove(mul,stack,record).truth==Truth.TRUE:
                        if record: self.ctx.add_u(U_TOTAL,p,Truth.TRUE,inputs=(gc,pg,mul),source='standard-u',evidence=['MUL premise'])
                        k.truth=Truth.TRUE; return k
        if p.rel=='PACKING_COMPLETE':
            kind=p.args[0]
            for target in self.facts('TARGET_TOTAL'):
                if target.args[0]!=kind: continue
                total=Proposition('TOTAL_ITEMS',(kind,target.args[1]))
                if self.router.prove(total,stack,record).truth==Truth.TRUE:
                    if record: self.ctx.add_u(U_COMPLETE,p,Truth.TRUE,inputs=(total,target),source='standard-u',evidence=['target matches total'])
                    k.truth=Truth.TRUE; return k
        return k

@dataclass(frozen=True)
class AddSpec:
    base:str; rec:str
    @property
    def name(self): return f'{self.base}__{self.rec}'
@dataclass(frozen=True)
class MulSpec:
    base:str; rec:str
    @property
    def name(self): return f'{self.base}__{self.rec}'

ADD_BASES=['Y0_ZX','Y0_Z0','XY_ZX']
ADD_RECS=['DEC_Y_DEC_Z','DEC_Y_INC_Z','DEC_Y_SAME_Z','DEC_X_DEC_Z']
MUL_BASES=['Y0_Z0','Y0_ZX','X0_Z0']
MUL_RECS=['DEC_Y_ADD_X','DEC_Y_ADD_Y','DEC_Y_ADD_ONE','DEC_Y_SAME_Z']

class RecursiveEngine:
    def __init__(self,ctx,std):
        self.ctx=ctx; self.std=std; self.router=None; self.add_spec=None; self.mul_spec=None; self._add_cache={}; self._mul_cache={}
    def pred(self,n,stack,record): return self.std.predecessor(n,stack,record)
    def add_base(self,s,x,y,z):
        return (s.base=='Y0_ZX' and y=='N0' and z==x) or (s.base=='Y0_Z0' and y=='N0' and z=='N0') or (s.base=='XY_ZX' and x==y and z==x)
    def prove_add_spec(self,s,p,stack=None,record=False,depth=0):
        if record:
            return self._prove_add_spec_impl(s,p,stack,record,depth)
        key=(s.name,p)
        if key in self._add_cache:
            return self._add_cache[key]
        out=self._prove_add_spec_impl(s,p,stack,record,depth)
        self._add_cache[key]=out
        return out
    def _prove_add_spec_impl(self,s,p,stack=None,record=False,depth=0):
        if stack is None: stack=set()
        if p.rel!='ADD' or depth>180: return False
        x,y,z=p.args; marker=('ADD',s.name,p)
        if marker in stack: return False
        stack=set(stack); stack.add(marker)
        if self.add_base(s,x,y,z):
            if record:
                t=UTemplate('REC_ADD_BASE_'+s.base,(),'ADD','RECURSIVE')
                self.ctx.add_u(t,p,Truth.TRUE,inputs=(),source='recursive-add',evidence=['base'])
            return True
        if s.rec=='DEC_Y_DEC_Z':
            y1,py=self.pred(y,stack,record); z1,pz=self.pred(z,stack,record)
            if y1 is not None and z1 is not None:
                sub=Proposition('ADD',(x,y1,z1))
                if self.prove_add_spec(s,sub,stack,record,depth+1):
                    if record:
                        t=UTemplate('REC_ADD_DEC_Y_DEC_Z',('PRED','PRED','ADD'),'ADD','RECURSIVE')
                        self.ctx.add_u(t,p,Truth.TRUE,inputs=(py,pz,sub),source='recursive-add',evidence=['decrease y,z'])
                    return True
        elif s.rec=='DEC_Y_INC_Z':
            y1,py=self.pred(y,stack,record)
            if y1 is not None:
                # z1 such that PRED(z1,z), derived from SUCC(z,z1)
                for su in self.std.facts('SUCC'):
                    if su.args[0]!=z: continue
                    z1=su.args[1]; pp=Proposition('PRED',(z1,z))
                    if self.std.prove(pp,stack,record).truth==Truth.TRUE:
                        if self.prove_add_spec(s,Proposition('ADD',(x,y1,z1)),stack,record,depth+1): return True
        elif s.rec=='DEC_Y_SAME_Z':
            y1,py=self.pred(y,stack,record)
            if y1 is not None: return self.prove_add_spec(s,Proposition('ADD',(x,y1,z)),stack,record,depth+1)
        elif s.rec=='DEC_X_DEC_Z':
            x1,px=self.pred(x,stack,record); z1,pz=self.pred(z,stack,record)
            if x1 is not None and z1 is not None: return self.prove_add_spec(s,Proposition('ADD',(x1,y,z1)),stack,record,depth+1)
        return False
    def learn_add(self,examples):
        rows=[]
        for b in ADD_BASES:
            for r in ADD_RECS:
                s=AddSpec(b,r); sup=conf=0
                for a,c,g in examples:
                    if self.prove_add_spec(s,Proposition('ADD',(f'N{a}',f'N{c}',f'N{g}')),record=False): sup+=1
                    for w in {max(0,g-1),g+1}-{g}:
                        if self.prove_add_spec(s,Proposition('ADD',(f'N{a}',f'N{c}',f'N{w}')),record=False): conf+=1
                comp=4 if r in {'DEC_Y_DEC_Z','DEC_Y_INC_Z'} else 3
                rows.append((sup*10-conf*25-comp*.1,sup,conf,comp,s))
        rows.sort(key=lambda x:(x[0],-x[3]),reverse=True); self.add_spec=rows[0][4]; return rows
    def solve_add_first(self,second,output,stack=None,record=False):
        # Backward use of the selected ADD template: ADD(?x, second, output).
        # For the learned DEC_Y_DEC_Z rule, walk second/output down together.
        if stack is None: stack=set()
        if self.add_spec is None or self.add_spec.rec!='DEC_Y_DEC_Z' or self.add_spec.base!='Y0_ZX':
            return None
        y=second; z=output
        while y!='N0':
            y1,py=self.pred(y,stack,record); z1,pz=self.pred(z,stack,record)
            if y1 is None or z1 is None: return None
            y,z=y1,z1
        return z
    def mul_base(self,s,x,y,z):
        return (s.base=='Y0_Z0' and y=='N0' and z=='N0') or (s.base=='Y0_ZX' and y=='N0' and z==x) or (s.base=='X0_Z0' and x=='N0' and z=='N0')
    def prove_mul_spec(self,s,p,stack=None,record=False,depth=0):
        if record:
            return self._prove_mul_spec_impl(s,p,stack,record,depth)
        key=(s.name,p)
        if key in self._mul_cache:
            return self._mul_cache[key]
        out=self._prove_mul_spec_impl(s,p,stack,record,depth)
        self._mul_cache[key]=out
        return out
    def _prove_mul_spec_impl(self,s,p,stack=None,record=False,depth=0):
        if stack is None: stack=set()
        if p.rel!='MUL' or depth>100: return False
        x,y,z=p.args; marker=('MUL',s.name,p)
        if marker in stack: return False
        stack=set(stack); stack.add(marker)
        if self.mul_base(s,x,y,z):
            if record:
                t=UTemplate('REC_MUL_BASE_'+s.base,(),'MUL','RECURSIVE')
                self.ctx.add_u(t,p,Truth.TRUE,inputs=(),source='recursive-mul',evidence=['base'])
            return True
        y1,py=self.pred(y,stack,record)
        if y1 is None: return False
        if s.rec=='DEC_Y_SAME_Z':
            z1=z; add=None
        else:
            second = x if s.rec=='DEC_Y_ADD_X' else y if s.rec=='DEC_Y_ADD_Y' else 'N1'
            z1=self.solve_add_first(second,z,stack,record)
            if z1 is None: return False
            add=Proposition('ADD',(z1,second,z))
            if not self.prove_add_spec(self.add_spec,add,stack,record): return False
        prev=Proposition('MUL',(x,y1,z1))
        if not self.prove_mul_spec(s,prev,stack,record,depth+1): return False
        if record and s.rec=='DEC_Y_ADD_X':
            t=UTemplate('REC_MUL_DEC_Y_ADD_X',('PRED','MUL','ADD'),'MUL','RECURSIVE')
            self.ctx.add_u(t,p,Truth.TRUE,inputs=(py,prev,add),source='recursive-mul',evidence=['previous product + x'])
        return True
    def learn_mul(self,examples):
        rows=[]
        for b in MUL_BASES:
            for r in MUL_RECS:
                s=MulSpec(b,r); sup=conf=0
                for a,c,g in examples:
                    if self.prove_mul_spec(s,Proposition('MUL',(f'N{a}',f'N{c}',f'N{g}')),record=False): sup+=1
                    for w in {max(0,g-1),g+1}-{g}:
                        if self.prove_mul_spec(s,Proposition('MUL',(f'N{a}',f'N{c}',f'N{w}')),record=False): conf+=1
                comp=5 if r!='DEC_Y_SAME_Z' else 3
                rows.append((sup*10-conf*25-comp*.1,sup,conf,comp,s))
        rows.sort(key=lambda x:(x[0],-x[3]),reverse=True); self.mul_spec=rows[0][4]; return rows
    def prove(self,p,stack=None,record=True):
        k=Key(p,self.ctx.story_id)
        if p.rel=='ADD' and self.add_spec and self.prove_add_spec(self.add_spec,p,stack,record): k.truth=Truth.TRUE
        if p.rel=='MUL' and self.mul_spec and self.prove_mul_spec(self.mul_spec,p,stack,record): k.truth=Truth.TRUE
        return k

class SharedSolver:
    def __init__(self,std,rec): self.std=std; self.rec=rec; std.router=self; rec.router=self; self.cache={}
    def prove(self,p,stack=None,record=True):
        key=(p.rel,p.args,p.polarity)
        if record and key in self.cache: return self.cache[key]
        k=self.rec.prove(p,stack,record) if p.rel in {'ADD','MUL'} else self.std.prove(p,stack,record)
        if record and (k.truth!=Truth.UNKNOWN or k.contradiction): self.cache[key]=k
        return k

def make_model(name,max_n=160):
    ctx=StoryContext(name)
    for i in range(max_n): ctx.add_event(Proposition('SUCC',(f'N{i}',f'N{i+1}')),source='number ontology')
    std=StandardEngine(ctx); rec=RecursiveEngine(ctx,std); router=SharedSolver(std,rec); return ctx,std,rec,router

# learn ADD from 0..6 supervised examples
lctx,lstd,lrec,lrouter=make_model('LEARN')
random.seed(5)
add_train=[(a,b,a+b) for a in range(7) for b in range(7)]
add_rows=lrec.learn_add(add_train)
mul_train=[(a,b,a*b) for a in range(7) for b in range(7)]
mul_rows=lrec.learn_mul(mul_train)

print('=== LEARNING ===')
print('ADD selected:',lrec.add_spec.name)
for score,sup,conf,comp,s in add_rows[:4]: print(' ',s.name,'support',sup,'conflict',conf,'score',round(score,1))
print('MUL selected:',lrec.mul_spec.name)
for score,sup,conf,comp,s in mul_rows[:5]: print(' ',s.name,'support',sup,'conflict',conf,'score',round(score,1))

# independent held-out oracle using explicit groups
ctx2,std2,rec2,router2=make_model('SELFTEST')
rec2.add_spec=lrec.add_spec; rec2.mul_spec=lrec.mul_spec

def product_oracle(a,b):
    groups=[]
    for gi in range(b): groups.append([f'g{gi}_e{j}' for j in range(a)])
    flat=[]
    for g in groups: flat.extend(g)
    return len(flat)

random.seed(11); self_rows=[]
for _ in range(20):
    a=random.randint(7,12); b=random.randint(7,12); gold=product_oracle(a,b)
    good=router2.prove(Proposition('MUL',(f'N{a}',f'N{b}',f'N{gold}')),record=False)
    bad=router2.prove(Proposition('MUL',(f'N{a}',f'N{b}',f'N{gold+1}')),record=False)
    self_rows.append({'a':a,'b':b,'gold':gold,'good':truth_name(good.truth),'wrong_output':gold+1,'wrong':truth_name(bad.truth),'passed':good.truth==Truth.TRUE and bad.truth==Truth.UNKNOWN})
self_pass=sum(r['passed'] for r in self_rows)
print('\n=== MUL SELF-TEST 7..12 ===')
print(f'{self_pass}/{len(self_rows)}')
for r in self_rows[:8]: print(f" {r['a']}*{r['b']}={r['gold']} -> {r['good']} | wrong {r['wrong_output']} -> {r['wrong']}")

# commutativity probe, no commutativity rule supplied
comm=[]
for a,b in [(7,11),(8,12),(9,10),(12,7)]:
    g=product_oracle(a,b)
    p1=router2.prove(Proposition('MUL',(f'N{a}',f'N{b}',f'N{g}')),record=False).truth==Truth.TRUE
    p2=router2.prove(Proposition('MUL',(f'N{b}',f'N{a}',f'N{g}')),record=False).truth==Truth.TRUE
    comm.append({'a':a,'b':b,'ab':p1,'ba':p2})
print('\nCommutativity probe (no COMMUTE U):')
for r in comm: print(f" {r['a']}*{r['b']} and {r['b']}*{r['a']} ->",r['ab'] and r['ba'])

# cross-engine semantic integration
ictx,istd,irec,irouter=make_model('PACKING')
irec.add_spec=lrec.add_spec; irec.mul_spec=lrec.mul_spec
ictx.add_event(Proposition('GROUP_COUNT',('apples','N6')),source='six boxes')
ictx.add_event(Proposition('PER_GROUP',('apples','N4')),source='four apples per box')
ictx.add_event(Proposition('TARGET_TOTAL',('apples','N24')),source='target')
queries=[('total 24',Proposition('TOTAL_ITEMS',('apples','N24')),Truth.TRUE),('packing complete',Proposition('PACKING_COMPLETE',('apples',)),Truth.TRUE),('wrong total 25',Proposition('TOTAL_ITEMS',('apples','N25')),Truth.UNKNOWN)]
integration=[]
for label,p,exp in queries:
    before=len(ictx.confirmed_u); k=irouter.prove(p,record=True); new=ictx.confirmed_u[before:]
    integration.append({'label':label,'query':str(p),'expected':truth_name(exp),'got':truth_name(k.truth),'passed':k.truth==exp,'new_u':[u.template.name for u in new]})
print('\n=== SHARED INTERACTION ===')
for r in integration:
    print(('PASS' if r['passed'] else 'FAIL'),'|',r['label'],'|',r['got'])
    if r['new_u']: print('  ',' -> '.join(r['new_u'][:35]),'...' if len(r['new_u'])>35 else '')
used=[u.template.name for u in ictx.confirmed_u]
audit={'standard_pred':sum(n=='SUCC_TO_PRED' for n in used),'recursive_add':sum(n.startswith('REC_ADD') for n in used),'recursive_mul':sum(n.startswith('REC_MUL') for n in used),'standard_total_bridge':sum(n=='GROUP_COUNT_PER_GROUP_MUL_TO_TOTAL' for n in used),'post_math_semantic':sum(n=='TOTAL_TARGET_TO_PACKING_COMPLETE' for n in used)}
print('\nInteraction audit:',audit)

# ablations
def semantic_case(add_on=True,mul_on=True,number_on=True):
    if number_on:
        c,s,r,ro=make_model('ABL')
    else:
        c=StoryContext('ABL'); s=StandardEngine(c); r=RecursiveEngine(c,s); ro=SharedSolver(s,r)
    if add_on: r.add_spec=lrec.add_spec
    if mul_on: r.mul_spec=lrec.mul_spec
    c.add_event(Proposition('GROUP_COUNT',('apples','N6')),source='six boxes')
    c.add_event(Proposition('PER_GROUP',('apples','N4')),source='four per box')
    c.add_event(Proposition('TARGET_TOTAL',('apples','N24')),source='target')
    return ro.prove(Proposition('PACKING_COMPLETE',('apples',))).truth
no_mul=semantic_case(True,False,True); no_add=semantic_case(False,True,True); no_num=semantic_case(True,True,False)
print('\n=== ABLATIONS ===')
print('without MUL-U:',truth_name(no_mul)); print('without ADD-U:',truth_name(no_add)); print('without Standard number structure:',truth_name(no_num))

assert lrec.add_spec.name=='Y0_ZX__DEC_Y_DEC_Z'
assert lrec.mul_spec.name=='Y0_Z0__DEC_Y_ADD_X'
assert self_pass==len(self_rows)
assert all(r['ab'] and r['ba'] for r in comm)
assert all(r['passed'] for r in integration)
assert all(v>0 for v in audit.values())
assert no_mul==Truth.UNKNOWN and no_add==Truth.UNKNOWN and no_num==Truth.UNKNOWN

report={
 'version':'math-v4.0-mul',
 'core_reuse':'Frozen Proposition/Key/UTemplate/StoryContext/Truth from symbolic_mini_lm_v1.py',
 'learned_add':{'selected':lrec.add_spec.name,'training_range':'0..6','top':[{'name':s.name,'support':sup,'conflict':conf,'complexity':comp,'score':score} for score,sup,conf,comp,s in add_rows[:6]]},
 'learned_mul':{'selected':lrec.mul_spec.name,'training_range':'0..6','training_n':len(mul_train),'top':[{'name':s.name,'support':sup,'conflict':conf,'complexity':comp,'score':score} for score,sup,conf,comp,s in mul_rows[:8]]},
 'self_test':{'passed':self_pass,'n':len(self_rows),'rows':self_rows},
 'commutativity_probe':comm,
 'integration':{'rows':integration,'audit':audit},
 'ablations':{'without_mul':truth_name(no_mul),'without_add':truth_name(no_add),'without_standard_number_structure':truth_name(no_num)},
 'invariants':['MUL is an ordinary Proposition/Key','MUL recursively requires ADD','ADD recursively requires PRED','PRED is produced by Standard-U from SUCC','MUL result is consumed by Standard-U semantic reasoning','wrong result remains UNKNOWN, not FALSE','held-out self-tests do not train'],
 'caveats':['ADD/MUL induction searches small predefined recursive skeleton families','SUCC number structure is a symbolic prior','training labels are external supervision; proof engine does not call Python + or *','semantic GROUP_COUNT/PER_GROUP facts are injected to isolate reasoning from language parsing']
}
Path('/mnt/data/symbolic_math_v40_mul_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
with Path('/mnt/data/symbolic_math_v40_mul_selftest.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(self_rows[0].keys())); w.writeheader(); w.writerows(self_rows)
print('\nSaved report/self-test.')

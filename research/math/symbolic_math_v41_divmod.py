from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import random, json, csv

# ============================================================
# Reuse the v4.0 ADD/MUL shared-engine implementation without
# executing its old benchmark section.
# ============================================================

V40 = Path('/mnt/data/symbolic_math_v40_mul.py').read_text(encoding='utf-8')
prefix = V40.split('# learn ADD from 0..6 supervised examples')[0]
ns={}
exec(prefix,ns)

Truth=ns['Truth']; truth_name=ns['truth_name']; Proposition=ns['Proposition']; Key=ns['Key']
UTemplate=ns['UTemplate']; StoryContext=ns['StoryContext']
BaseStandardEngine=ns['StandardEngine']; BaseRecursiveEngine=ns['RecursiveEngine']; BaseSharedSolver=ns['SharedSolver']

U_LT_CHAIN=UTemplate('PRED_CHAIN_TO_LT',('PRED',),'LT','REASONING')
U_PLAN=UTemplate('TOTAL_CAPACITY_DIVMOD_TO_PACKING_PLAN',('TOTAL_ITEMS_RAW','BOX_CAPACITY','DIVMOD'),'PACKING_PLAN','REASONING')
U_LEFT=UTemplate('PACKING_PLAN_NONZERO_TO_LEFTOVER',('PACKING_PLAN','NONZERO'),'HAS_LEFTOVER','REASONING')
U_EXACT=UTemplate('PACKING_PLAN_ZERO_TO_EXACT',('PACKING_PLAN',),'EXACT_PACKING','REASONING')

class StandardEngine(BaseStandardEngine):
    def __init__(self,ctx,lt_enabled=True):
        super().__init__(ctx)
        self.lt_enabled=lt_enabled

    def prove(self,p,stack=None,record=True):
        if stack is None: stack=set()

        # Preserve direct concrete Key semantics.
        d=self.direct(p)
        if d.truth!=Truth.UNKNOWN or d.contradiction:
            return d

        # Standard order relation, derived through PRED chain.
        if p.rel=='LT':
            k=Key(p,self.ctx.story_id)
            if not self.lt_enabled:
                return k
            a,b=p.args
            if a==b:
                return k
            cur=b; inputs=[]; seen=set()
            while cur not in seen:
                seen.add(cur)
                lo,pred_prop=self.predecessor(cur,stack,record)
                if lo is None:
                    return k
                inputs.append(pred_prop)
                if lo==a:
                    if record:
                        self.ctx.add_u(U_LT_CHAIN,p,Truth.TRUE,inputs=tuple(inputs),source='standard-order',evidence=['PRED chain establishes strict order'])
                    k.truth=Truth.TRUE
                    k.evidence.append('U +1 PRED_CHAIN_TO_LT')
                    return k
                cur=lo
            return k

        if p.rel=='NONZERO':
            k=Key(p,self.ctx.story_id)
            n=p.args[0]
            if n!='N0':
                # Prove N0<n rather than special-casing numeric value as arithmetic.
                lt=Proposition('LT',('N0',n))
                if self.router.prove(lt,stack,record).truth==Truth.TRUE:
                    k.truth=Truth.TRUE
                    return k
            return k

        if p.rel=='PACKING_PLAN':
            # PACKING_PLAN(kind,q,r) <- totals/capacity + DIVMOD(total,cap,q,r)
            k=Key(p,self.ctx.story_id)
            kind,q,r=p.args
            for total in self.facts('TOTAL_ITEMS_RAW'):
                if total.args[0]!=kind: continue
                for cap in self.facts('BOX_CAPACITY'):
                    if cap.args[0]!=kind: continue
                    dm=Proposition('DIVMOD',(total.args[1],cap.args[1],q,r))
                    if self.router.prove(dm,stack,record).truth==Truth.TRUE:
                        if record:
                            self.ctx.add_u(U_PLAN,p,Truth.TRUE,inputs=(total,cap,dm),source='standard-semantic',evidence=['DIVMOD premise proved in shared key space'])
                        k.truth=Truth.TRUE
                        k.evidence.append('U +1 TOTAL_CAPACITY_DIVMOD_TO_PACKING_PLAN')
                        return k
            return k

        if p.rel=='HAS_LEFTOVER':
            # HAS_LEFTOVER(kind,q,r) consumes a math-derived PACKING_PLAN and order proof.
            k=Key(p,self.ctx.story_id)
            kind,q,r=p.args
            plan=Proposition('PACKING_PLAN',(kind,q,r))
            nz=Proposition('NONZERO',(r,))
            if self.router.prove(plan,stack,record).truth==Truth.TRUE and self.router.prove(nz,stack,record).truth==Truth.TRUE:
                if record:
                    self.ctx.add_u(U_LEFT,p,Truth.TRUE,inputs=(plan,nz),source='standard-semantic',evidence=['packing remainder is nonzero'])
                k.truth=Truth.TRUE
                return k
            return k

        if p.rel=='EXACT_PACKING':
            # EXACT_PACKING(kind,q) iff the paired remainder is exactly zero.
            k=Key(p,self.ctx.story_id)
            kind,q=p.args
            plan=Proposition('PACKING_PLAN',(kind,q,'N0'))
            if self.router.prove(plan,stack,record).truth==Truth.TRUE:
                if record:
                    self.ctx.add_u(U_EXACT,p,Truth.TRUE,inputs=(plan,),source='standard-semantic',evidence=['zero remainder'])
                k.truth=Truth.TRUE
                return k
            return k

        return super().prove(p,stack,record)

@dataclass(frozen=True)
class DivSpec:
    equation:str
    bound:str
    @property
    def name(self): return f'{self.equation}__{self.bound}'

DIV_EQUATIONS=[
    'D_Q_PLUS_R_EQ_N',  # intended: d*q + r = n
    'D_R_PLUS_Q_EQ_N',  # wrong: d*r + q = n
    'Q_R_PLUS_D_EQ_N',  # wrong: q*r + d = n
]
DIV_BOUNDS=[
    'R_LT_D',
    'R_LE_D',
    'NO_BOUND',
]

class RecursiveEngine(BaseRecursiveEngine):
    def __init__(self,ctx,std):
        super().__init__(ctx,std)
        self.div_spec=None
        self.div_rows=[]
        self._div_cache={}

    def _bound_ok(self,spec,r,d,stack,record):
        if spec.bound=='NO_BOUND':
            return True
        if spec.bound=='R_LE_D' and r==d:
            return True
        return self.router.prove(Proposition('LT',(r,d)),stack,record).truth==Truth.TRUE

    def _solve_equation(self,spec,n,d,q,r,stack,record):
        # Solve the unknown intermediate product p by using learned ADD backward:
        # ADD(p, remainder_like, n) -> p.
        if self.add_spec is None or self.mul_spec is None:
            return False,()

        if spec.equation=='D_Q_PLUS_R_EQ_N':
            mul_a,mul_b,add_second=d,q,r
        elif spec.equation=='D_R_PLUS_Q_EQ_N':
            mul_a,mul_b,add_second=d,r,q
        else: # Q_R_PLUS_D_EQ_N
            mul_a,mul_b,add_second=q,r,d

        p=self.solve_add_first(add_second,n,stack,record)
        if p is None:
            return False,()
        add_prop=Proposition('ADD',(p,add_second,n))
        if not self.prove_add_spec(self.add_spec,add_prop,stack,record):
            return False,()
        mul_prop=Proposition('MUL',(mul_a,mul_b,p))
        if not self.prove_mul_spec(self.mul_spec,mul_prop,stack,record):
            return False,()
        return True,(mul_prop,add_prop)

    def prove_div_spec(self,spec,p,stack=None,record=False):
        if stack is None: stack=set()
        if p.rel!='DIVMOD' or len(p.args)!=4:
            return False
        key=(spec.name,p)
        if not record and key in self._div_cache:
            return self._div_cache[key]
        n,d,q,r=p.args
        marker=('DIV',spec.name,p)
        if marker in stack:
            return False
        stack=set(stack); stack.add(marker)

        # Denominator zero is never accepted. This uses symbolic identity only.
        if d=='N0':
            out=False
        else:
            ok,inputs=self._solve_equation(spec,n,d,q,r,stack,record)
            out=ok and self._bound_ok(spec,r,d,stack,record)
            if out and record:
                t=UTemplate('REC_DIVMOD_'+spec.name,('MUL','ADD','LT' if spec.bound!='NO_BOUND' else 'BOUNDLESS'),'DIVMOD','RECURSIVE')
                inps=list(inputs)
                if spec.bound!='NO_BOUND' and r!=d:
                    inps.append(Proposition('LT',(r,d)))
                self.ctx.add_u(t,p,Truth.TRUE,inputs=tuple(inps),source='recursive-divmod',evidence=['coupled quotient/remainder decomposition'])
        if not record:
            self._div_cache[key]=out
        return out

    def learn_div(self,examples):
        rows=[]
        for eq in DIV_EQUATIONS:
            for bound in DIV_BOUNDS:
                spec=DivSpec(eq,bound); sup=conf=0
                for n,d,q,r in examples:
                    good=Proposition('DIVMOD',(f'N{n}',f'N{d}',f'N{q}',f'N{r}'))
                    if self.prove_div_spec(spec,good,record=False):
                        sup+=1

                    # Contrast 1: same n via non-canonical quotient/remainder pair.
                    # q-1, r+d satisfies d*(q-1)+(r+d)=n whenever q>0.
                    if q>0 and r+d<=160:
                        alt=Proposition('DIVMOD',(f'N{n}',f'N{d}',f'N{q-1}',f'N{r+d}'))
                        if self.prove_div_spec(spec,alt,record=False): conf+=1

                    # Contrast 2/3: simple wrong pair components.
                    if q+1<=160:
                        wrongq=Proposition('DIVMOD',(f'N{n}',f'N{d}',f'N{q+1}',f'N{r}'))
                        if self.prove_div_spec(spec,wrongq,record=False): conf+=1
                    if r+1<=160:
                        wrongr=Proposition('DIVMOD',(f'N{n}',f'N{d}',f'N{q}',f'N{r+1}'))
                        if self.prove_div_spec(spec,wrongr,record=False): conf+=1

                complexity=4 + (0 if bound=='NO_BOUND' else 1)
                score=sup*10-conf*25-complexity*.1
                rows.append((score,sup,conf,complexity,spec))
        rows.sort(key=lambda x:(x[0],-x[3]),reverse=True)
        self.div_rows=rows; self.div_spec=rows[0][4]
        return rows

    def prove(self,p,stack=None,record=True):
        if p.rel=='DIVMOD':
            k=Key(p,self.ctx.story_id)
            if self.div_spec and self.prove_div_spec(self.div_spec,p,stack,record):
                k.truth=Truth.TRUE
                k.evidence.append('U +1 '+self.div_spec.name)
            return k
        return super().prove(p,stack,record)

class SharedSolver(BaseSharedSolver):
    def prove(self,p,stack=None,record=True):
        key=(p.rel,p.args,p.polarity)
        if record and key in self.cache:
            return self.cache[key]
        if p.rel in {'ADD','MUL','DIVMOD'}:
            k=self.rec.prove(p,stack,record)
        else:
            k=self.std.prove(p,stack,record)
        if record and (k.truth!=Truth.UNKNOWN or k.contradiction):
            self.cache[key]=k
        return k

def make_model(name,max_n=160,number_on=True,lt_enabled=True):
    ctx=StoryContext(name)
    if number_on:
        for i in range(max_n):
            ctx.add_event(Proposition('SUCC',(f'N{i}',f'N{i+1}')),source='number ontology')
    std=StandardEngine(ctx,lt_enabled=lt_enabled)
    rec=RecursiveEngine(ctx,std)
    router=SharedSolver(std,rec)
    return ctx,std,rec,router

# ============================================================
# Independent oracle: explicit groups/remainder, no // or %.
# ============================================================

def divmod_set_oracle(n,d):
    if d<=0: raise ValueError('positive divisor required')
    remaining=[f'e{i}' for i in range(n)]
    q=0
    while len(remaining)>=d:
        # Remove one full group of d explicit members.
        group=[]
        for _ in range(d):
            group.append(remaining.pop())
        q+=1
    return q,len(remaining)

# ============================================================
# Learn ADD/MUL exactly as v4.0, then learn DIVMOD.
# ============================================================

lctx,lstd,lrec,lrouter=make_model('LEARN')
add_train=[(a,b,a+b) for a in range(7) for b in range(7)]
add_rows=lrec.learn_add(add_train)
mul_train=[(a,b,a*b) for a in range(7) for b in range(7)]
mul_rows=lrec.learn_mul(mul_train)

# DIV training is deliberately small and bounded.
div_train=[]
for n in range(0,31):
    for d in range(1,7):
        q,r=divmod_set_oracle(n,d)
        div_train.append((n,d,q,r))
div_rows=lrec.learn_div(div_train)

print('=== LEARNING ===')
print('ADD:',lrec.add_spec.name)
print('MUL:',lrec.mul_spec.name)
print('DIVMOD selected:',lrec.div_spec.name)
print('Top DIVMOD candidates:')
for score,sup,conf,comp,s in div_rows[:7]:
    print(' ',f'{s.name:34}', 'support',sup,'conflict',conf,'score',round(score,1))

# ============================================================
# Held-out self-tests: unseen larger divisors and numerators.
# ============================================================

tctx,tstd,trec,trouter=make_model('SELFTEST')
trec.add_spec=lrec.add_spec; trec.mul_spec=lrec.mul_spec; trec.div_spec=lrec.div_spec
random.seed(17)
self_rows=[]
for _ in range(30):
    n=random.randint(31,90)
    d=random.randint(7,12)
    q,r=divmod_set_oracle(n,d)
    good=trouter.prove(Proposition('DIVMOD',(f'N{n}',f'N{d}',f'N{q}',f'N{r}')),record=False)

    # Dangerous noncanonical decomposition, when available.
    if q>0:
        aq,ar=q-1,r+d
        alt=trouter.prove(Proposition('DIVMOD',(f'N{n}',f'N{d}',f'N{aq}',f'N{ar}')),record=False)
        alt_pair=(aq,ar); alt_state=truth_name(alt.truth)
    else:
        alt_pair=None; alt_state='n/a'

    bad=trouter.prove(Proposition('DIVMOD',(f'N{n}',f'N{d}',f'N{q}',f'N{r+1}')),record=False)
    passed=(good.truth==Truth.TRUE and bad.truth==Truth.UNKNOWN and (alt_pair is None or alt.truth==Truth.UNKNOWN))
    self_rows.append({'n':n,'d':d,'q':q,'r':r,'good':truth_name(good.truth),'alt_pair':str(alt_pair),'alt_state':alt_state,'wrong_r':r+1,'wrong_state':truth_name(bad.truth),'passed':passed})
self_pass=sum(r['passed'] for r in self_rows)

print('\n=== DIVMOD SELF-TEST, unseen n=31..90 d=7..12 ===')
print(f'{self_pass}/{len(self_rows)}')
for row in self_rows[:8]:
    print(f" {row['n']} / {row['d']} -> q={row['q']} r={row['r']} [{row['good']}] | alt {row['alt_pair']} -> {row['alt_state']} | wrong-r -> {row['wrong_state']}")

# Explicit exact/non-exact probes.
probes=[(84,12),(83,12),(47,7),(53,8)]
print('\nExact / remainder probes:')
for n,d in probes:
    q,r=divmod_set_oracle(n,d)
    k=trouter.prove(Proposition('DIVMOD',(f'N{n}',f'N{d}',f'N{q}',f'N{r}')),record=False)
    print(f' {n}/{d}: q={q}, r={r}, state={truth_name(k.truth)}')

# ============================================================
# Cross-engine integration: semantics -> DIVMOD -> MUL -> ADD ->
# Standard PRED/LT -> result -> semantic Standard-U.
# ============================================================

ictx,istd,irec,irouter=make_model('PACKING')
irec.add_spec=lrec.add_spec; irec.mul_spec=lrec.mul_spec; irec.div_spec=lrec.div_spec
ictx.add_event(Proposition('TOTAL_ITEMS_RAW',('apples','N23')),source='23 apples')
ictx.add_event(Proposition('BOX_CAPACITY',('apples','N5')),source='5 per box')
ictx.add_event(Proposition('TOTAL_ITEMS_RAW',('oranges','N24')),source='24 oranges')
ictx.add_event(Proposition('BOX_CAPACITY',('oranges','N6')),source='6 per box')

queries=[
    ('apple plan',Proposition('PACKING_PLAN',('apples','N4','N3')),Truth.TRUE),
    ('apple leftover semantics',Proposition('HAS_LEFTOVER',('apples','N4','N3')),Truth.TRUE),
    ('dangerous noncanonical apple plan',Proposition('PACKING_PLAN',('apples','N3','N8')),Truth.UNKNOWN),
    ('orange exact plan',Proposition('PACKING_PLAN',('oranges','N4','N0')),Truth.TRUE),
    ('orange exact semantic',Proposition('EXACT_PACKING',('oranges','N4')),Truth.TRUE),
]
integration=[]
for label,p,exp in queries:
    before=len(ictx.confirmed_u)
    k=irouter.prove(p,record=True)
    new=ictx.confirmed_u[before:]
    integration.append({'label':label,'query':str(p),'expected':truth_name(exp),'got':truth_name(k.truth),'passed':k.truth==exp,'new_u':[u.template.name for u in new]})

print('\n=== SHARED INTEGRATION ===')
for row in integration:
    print(('PASS' if row['passed'] else 'FAIL'),'|',row['label'],'|',row['got'])
    if row['new_u']:
        print('  ',' -> '.join(row['new_u'][:38]),'...' if len(row['new_u'])>38 else '')

used=[u.template.name for u in ictx.confirmed_u]
audit={
    'standard_pred':sum(n=='SUCC_TO_PRED' for n in used),
    'standard_lt':sum(n=='PRED_CHAIN_TO_LT' for n in used),
    'recursive_add':sum(n.startswith('REC_ADD') for n in used),
    'recursive_mul':sum(n.startswith('REC_MUL') for n in used),
    'divmod_u':sum(n.startswith('REC_DIVMOD') for n in used),
    'standard_plan_bridge':sum(n=='TOTAL_CAPACITY_DIVMOD_TO_PACKING_PLAN' for n in used),
    'post_math_semantic':sum(n in {'PACKING_PLAN_NONZERO_TO_LEFTOVER','PACKING_PLAN_ZERO_TO_EXACT'} for n in used),
}
print('\nInteraction audit:',audit)

# ============================================================
# Ablations.
# ============================================================

def semantic_case(div_on=True,mul_on=True,add_on=True,number_on=True,lt_enabled=True):
    c,s,r,ro=make_model('ABL',number_on=number_on,lt_enabled=lt_enabled)
    if add_on: r.add_spec=lrec.add_spec
    if mul_on: r.mul_spec=lrec.mul_spec
    if div_on: r.div_spec=lrec.div_spec
    c.add_event(Proposition('TOTAL_ITEMS_RAW',('apples','N23')),source='23 apples')
    c.add_event(Proposition('BOX_CAPACITY',('apples','N5')),source='5 per box')
    return ro.prove(Proposition('HAS_LEFTOVER',('apples','N4','N3'))).truth

abl={
    'without_div':truth_name(semantic_case(div_on=False)),
    'without_mul':truth_name(semantic_case(mul_on=False)),
    'without_add':truth_name(semantic_case(add_on=False)),
    'without_number_structure':truth_name(semantic_case(number_on=False)),
    'without_lt_bound':truth_name(semantic_case(lt_enabled=False)),
}
print('\n=== ABLATIONS ===')
for k,v in abl.items(): print(k,':',v)

# Demonstrate why the learned remainder bound matters.
nb=DivSpec('D_Q_PLUS_R_EQ_N','NO_BOUND')
danger=Proposition('DIVMOD',('N23','N5','N3','N8'))
no_bound_accept=irec.prove_div_spec(nb,danger,record=False)
learned_accept=irec.prove_div_spec(lrec.div_spec,danger,record=False)
print('\n=== REMAINDER-BOUND ADVERSARIAL ===')
print('23 = 5*3 + 8 is equation-valid but not canonical DIVMOD')
print('NO_BOUND candidate accepts:',no_bound_accept)
print('learned candidate accepts:',learned_accept)

assert self_pass==len(self_rows)
assert all(r['passed'] for r in integration)
assert all(v>0 for v in audit.values())
assert all(v=='0' for v in abl.values())
assert no_bound_accept is True and learned_accept is False

report={
    'version':'math-v4.1-divmod',
    'core_reuse':'v4.0 shared ADD/MUL engine, itself reusing frozen v1 Proposition/Key/UTemplate/StoryContext/Truth',
    'learned':{
        'add':lrec.add_spec.name,
        'mul':lrec.mul_spec.name,
        'divmod':lrec.div_spec.name,
        'div_training_n':len(div_train),
        'div_training_domain':'n=0..30, d=1..6',
        'top_div_candidates':[{'name':s.name,'support':sup,'conflict':conf,'complexity':comp,'score':score} for score,sup,conf,comp,s in div_rows[:8]],
    },
    'self_test':{'passed':self_pass,'n':len(self_rows),'domain':'n=31..90, d=7..12','rows':self_rows},
    'integration':{'rows':integration,'audit':audit},
    'ablations':abl,
    'adversarial':{
        'proposition':'DIVMOD(N23,N5,N3,N8)',
        'equation_valid':'23 = 5*3 + 8',
        'no_bound_candidate_accepts':no_bound_accept,
        'learned_candidate_accepts':learned_accept,
    },
    'invariants':[
        'DIVMOD has two coupled semantic outputs q and r in one Proposition.',
        'DIVMOD is proved through learned MUL and ADD, not Python division/modulo.',
        'MUL/ADD still depend on Standard-U number structure.',
        'Strict remainder bound r<d is separately proved in the shared solver.',
        'A math-derived DIVMOD Key is consumed by Standard-U PACKING_PLAN and later semantic U.',
        'Invalid/noncanonical quotient-remainder pairs remain UNKNOWN (0), not FALSE (-1).',
        'Held-out self-tests validate frozen templates and do not update them.'
    ],
    'caveats':[
        'DIVMOD induction searches a small predefined family of relational skeletons.',
        'The SUCC number ontology is supplied as a prior.',
        'Training labels and held-out oracle are generated externally using explicit grouping; the proof engine itself uses no //, %, or / operator.',
        'Language parsing is intentionally excluded; semantic TOTAL_ITEMS_RAW/BOX_CAPACITY Keys are injected directly.'
    ]
}
Path('/mnt/data/symbolic_math_v41_divmod_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
with Path('/mnt/data/symbolic_math_v41_divmod_selftest.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(self_rows[0].keys())); w.writeheader(); w.writerows(self_rows)

print('\nSaved report/self-test.')

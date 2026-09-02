from dataclasses import dataclass
from pathlib import Path
import itertools, random, json, csv, statistics, time

MAX_N=40

def sym(i): return f'N{i}'
def val(n): return int(n[1:])
def pred(n):
    i=val(n); return sym(i-1) if i>0 else None
def succ(n):
    i=val(n); return sym(i+1) if i<MAX_N else None

@dataclass(frozen=True)
class Base:
    name:str
    def infer(self,x,y):
        if self.name=='Y0_ZX' and y=='N0': return x
        if self.name=='Y0_Z0' and y=='N0': return 'N0'
        if self.name=='X0_Z0' and x=='N0': return 'N0'
        if self.name=='XY_Z0' and x==y: return 'N0'
        return None

@dataclass(frozen=True)
class Rec:
    dx:int; dy:int; mode:str
    @property
    def name(self): return f'REC_dx{self.dx}_dy{self.dy}_{self.mode}'
    def step(self,x,y):
        if self.dx:
            x=pred(x)
            if x is None:return None
        if self.dy:
            y=pred(y)
            if y is None:return None
        return x,y
    def out(self,z):
        if self.mode=='SAME': return z
        if self.mode=='PRED': return pred(z)
        if self.mode=='SUCC': return succ(z)

BASES=[Base(n) for n in ['Y0_ZX','Y0_Z0','X0_Z0','XY_Z0']]
RECS=[Rec(dx,dy,m) for dx,dy,m in itertools.product([0,1],[0,1],['SAME','PRED','SUCC']) if dx or dy]

def infer(x,y,b,r,depth=0,seen=None):
    if depth>MAX_N+3:return None
    seen=set() if seen is None else set(seen)
    if (x,y) in seen:return None
    seen.add((x,y))
    z=b.infer(x,y)
    if z is not None:return z
    st=r.step(x,y)
    if st is None:return None
    zr=infer(st[0],st[1],b,r,depth+1,seen)
    if zr is None:return None
    return r.out(zr)

random.seed(7)
small=[(a,b,a-b) for a in range(11) for b in range(a+1)]
train=[(a,0,a) for a in range(11)] + random.sample([(a,b,c) for a,b,c in small if b>0],24)

scores=[]
for b in BASES:
  for r in RECS:
    sup=conf=unk=0
    for a,y,c in train:
        z=infer(sym(a),sym(y),b,r)
        if z is None:unk+=1
        elif val(z)==c:sup+=1
        else:conf+=1
    complexity=1+r.dx+r.dy+(r.mode!='SAME')
    score=sup*10-conf*20-unk*3-complexity*0.1
    scores.append((score,sup,conf,unk,complexity,b,r))
scores.sort(key=lambda x:x[0],reverse=True)
score,sup,conf,unk,complexity,best_b,best_r=scores[0]

# unseen > 10
TEST=[]
for a in [11,12,14,17,20,23,29,31,37,40]:
    for y in sorted(set([1,2,a//3,a//2,a-1])):
        if 0<=y<=a: TEST.append((a,y,a-y))
lookup={(a,y):c for a,y,c in train}

def learned(a,y):
    z=infer(sym(a),sym(y),best_b,best_r)
    return None if z is None else val(z)

def set_sub(a,y):
    group={f'e{i}' for i in range(a)}
    removed={f'e{i}' for i in range(y)}
    count='N0'
    for _ in group-removed:
        count=succ(count)
    return val(count)

methods={'LOOKUP':lambda a,y:lookup.get((a,y)),'LEARNED_U':learned,'SET':set_sub,'INTEGER':lambda a,y:a-y}
summary={}
for name,fn in methods.items():
    oks=[]; us=[]
    for a,y,c in TEST:
        t0=time.perf_counter_ns(); got=fn(a,y); us.append((time.perf_counter_ns()-t0)/1000); oks.append(got==c)
    summary[name]={'passed':sum(oks),'n':len(oks),'accuracy':sum(oks)/len(oks),'median_us':statistics.median(us)}

# ablation: without PRED graph, only base y=0 works; test intentionally y>0
ablation=sum((a if y==0 else None)==c for a,y,c in TEST)

# controlled language composition
WORDS={'null':0,'eins':1,'ein':1,'eine':1,'einen':1,'zwei':2,'drei':3,'vier':4,'fünf':5,'sechs':6,'sieben':7,'acht':8,'neun':9,'zehn':10,'elf':11,'zwölf':12,'dreizehn':13,'vierzehn':14,'fünfzehn':15,'sechzehn':16,'siebzehn':17,'achtzehn':18,'neunzehn':19,'zwanzig':20,'dreiundzwanzig':23}
def parse_de(text):
    low=text.lower().replace('.','').replace('?','')
    nums=[]
    for tok in low.split():
        if tok.isdigit(): nums.append(int(tok))
        elif tok in WORDS: nums.append(WORDS[tok])
    if len(nums)>=2 and any(x in low for x in ['gegessen','entfernt','weggenommen','fraß','frass']): return nums[0],nums[1]
    return None
lang=[
 ('Von zwölf Äpfeln wurden fünf gegessen. Wie viele blieben übrig?',7),
 ('Von 17 Kugeln wurden acht entfernt. Wie viele bleiben?',9),
 ('Von dreiundzwanzig Steinen wurden sechs weggenommen. Wie viele bleiben?',17),
 ('Von 20 Geißlein fraß der Wolf neun. Wie viele bleiben?',11),
]
lang_rows=[]
for txt,gold in lang:
    p=parse_de(txt); got=None if p is None else learned(*p); lang_rows.append({'text':txt,'parsed':p,'gold':gold,'got':got,'passed':got==gold})

report={
 'training':{'n':len(train),'range':'0..10','examples':train},
 'candidate_programs':len(BASES)*len(RECS),
 'best':{'base':best_b.name,'recursion':best_r.name,'support':sup,'conflict':conf,'unknown':unk,'complexity':complexity},
 'top6':[{'base':x[5].name,'recursion':x[6].name,'score':x[0],'support':x[1],'conflict':x[2],'unknown':x[3]} for x in scores[:6]],
 'generalization':summary,
 'ablation_without_pred':{'passed':ablation,'n':len(TEST)},
 'controlled_language':lang_rows,
 'caveats':['Learner searches a small predefined family of recursive U skeletons; it does not invent arbitrary programs.','PRED/SUCC number structure is provided as a symbolic prior.','Training uses only 0..10; unseen tests use larger number nodes in the same PRED graph.','Controlled German parser is intentionally tiny and separate from arithmetic.']
}
Path('/mnt/data/symbolic_arithmetic_u_test_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
with open('/mnt/data/symbolic_arithmetic_u_generalization.csv','w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['a','b','gold','lookup','learned_u','set','integer'])
    for a,y,c in TEST:w.writerow([a,y,c,lookup.get((a,y)),learned(a,y),set_sub(a,y),a-y])

print('=== LEARNED U SEARCH ===')
print('training examples:',len(train),'range 0..10')
print('candidate programs:',len(BASES)*len(RECS))
print('best:',best_b.name,'+',best_r.name)
print('support/conflict/unknown:',sup,conf,unk)
print('\nTop 6:')
for x in scores[:6]: print(x[5].name,'+',x[6].name,'score',round(x[0],1),'support',x[1],'conflict',x[2],'unknown',x[3])
print('\n=== OUT-OF-RANGE ===')
for name in methods:
    s=summary[name]; print(name,s['passed'],'/',s['n'],f"{s['accuracy']:.1%}",'median_us',round(s['median_us'],3))
print('\nExamples:')
for a,y in [(17,8),(23,6),(31,30),(40,20)]: print(a,'-',y,'lookup',lookup.get((a,y)),'learned',learned(a,y),'set',set_sub(a,y),'integer',a-y)
print('\nAblation without PRED:',ablation,'/',len(TEST))
print('\n=== CONTROLLED LANGUAGE ===')
for r in lang_rows: print('PASS' if r['passed'] else 'FAIL',r['parsed'],'=>',r['got'],'|',r['text'])

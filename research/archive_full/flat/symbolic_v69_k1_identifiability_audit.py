import importlib.util,sys,contextlib,io
spec=importlib.util.spec_from_file_location('k1','/mnt/data/symbolic_v69_k1_surface_cue_ablation.py')
k=importlib.util.module_from_spec(spec); sys.modules['k1']=k
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(k)

def fc(f):
    if f.startswith('L:') or f.startswith('M:'): return 1
    if f.startswith('W:'): return 3
    if f.startswith('B:'): return 3
    if f.startswith('T:'): return 4
    return 3

def tie_trace(delta):
    pos,neg,cand=k.candidate_rules(delta)
    uncovered=set(range(len(pos))); trace=[]
    while uncovered:
        opts=[]
        for n,req,cov in cand:
            gain=len(cov&uncovered)
            if gain:
                score=(-gain,sum(fc(f) for f in req),n,sum(map(len,req)))
                opts.append((score,tuple(sorted(req)),req,cov))
        opts.sort(key=lambda x:(x[0],x[1])); best=opts[0][0]
        tied=[x for x in opts if x[0]==best]
        trace.append([list(x[1]) for x in tied])
        chosen=tied[0]; uncovered-=chosen[3]
    return trace

print('=== K1 IDENTIFIABILITY AUDIT ===')
for d in (k.STOP,k.START,k.ENTER,k.LEAVE):
    tr=tie_trace(d)
    print(k.NAME[d], 'ties:', tr)

# Hard held-out morphology/syntax probe.
CASES=[
 ('Das Rad hörte auf zu drehen.',k.STOP),
 ('Zu drehen hört das Rad auf.',k.STOP),
 ('Ben ging in das Haus.',k.ENTER),
 ('Ben ging in dem Haus.',None),
 ('Ben betrat den Garten.',k.ENTER),
 ('Cara verließ das Haus.',k.LEAVE),
 ('Cara ging aus dem Haus.',k.LEAVE),
 ('Die Lampe erlosch.',k.STOP),
 ('Die Maschine begann zu laufen.',k.START),
 ('Die Lampe fing an zu leuchten.',k.START),
 ('Anna schaut ins Haus.',None),
]
oks=[]
for t,g in CASES:
    p,s,ds=k.classify(t); ok=p==g; oks.append(ok)
    print(('PASS' if ok else 'FAIL'),'|',t,'=>',None if p is None else k.NAME[p], 'expected',None if g is None else k.NAME[g])
print('robustness',sum(oks),'/',len(oks))
assert all(oks)

import importlib.util, sys, contextlib, io, copy
spec=importlib.util.spec_from_file_location('v68','/mnt/data/symbolic_v68_c15_surface_paraphrases.py')
v=importlib.util.module_from_spec(spec); sys.modules['v68']=v
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(v)

print('=== v6.8 C15 ADVERSARIAL / SCALE AUDIT ===')
# Same lexical surface cue OUTWARD must split by slot type: machine STOP vs person LEAVE.
cases=[
    ('Die Lampe geht aus.',v.STOP),
    ('Anna geht aus dem Haus.',v.LEAVE),
    ('Die Lampe geht an.',v.START),
    ('Anna geht ins Haus.',v.ENTER),
    ('Die Lampe leuchtet jetzt.',None),
    ('Die Lampe ist aus.',None),
]
preds=[v.classify(t)[0] for t,g in cases]
for (t,g),p in zip(cases,preds):
    print(t,'=>',None if p is None else v.NAME[p],'expected',None if g is None else v.NAME[g])

# 40-line mixed paraphrase stream with frozen rule set.
base=[
    'Die Lampe hört auf zu leuchten.',
    'Die Maschine beginnt zu laufen.',
    'Anna verlässt das Haus.',
    'Cara betritt den Garten.',
    'Die Lampe geht an.',
    'Das Rad dreht sich nicht mehr.',
    'Ben geht ins Haus.',
    'Cara geht hinaus aus dem Zimmer.',
]
lines=(base*5)[:40]
snap=repr(v.RULES)
results=[v.classify(x)[0] for x in lines]
snap2=repr(v.RULES)
print('40-line classified:',sum(x is not None for x in results),'/',len(results))

# Different paraphrase surfaces must canonicalize to the exact same grounded STOP delta.
g1=v.ground(*v.classify('Die Lampe erlischt.'))
g2=v.ground(*v.classify('Die Lampe hört auf zu leuchten.'))
print('same grounded stop:',g1,g2)

checks={
    'typed_outward_separates_STOP_and_LEAVE':preds[0]==v.STOP and preds[1]==v.LEAVE,
    'typed_inward_start_separates_START_and_ENTER':preds[2]==v.START and preds[3]==v.ENTER,
    'current_state_now_not_transition':preds[4] is None,
    'unlearned_stative_aus_not_transition':preds[5] is None,
    'forty_line_paraphrase_stream_all_classified':all(x is not None for x in results),
    'forty_line_stream_does_not_mutate_rules':snap==snap2,
    'different_stop_surfaces_same_grounded_delta':g1==g2,
}
for k,z in checks.items(): print(('PASS' if z else 'FAIL'),'|',k)
assert all(checks.values())

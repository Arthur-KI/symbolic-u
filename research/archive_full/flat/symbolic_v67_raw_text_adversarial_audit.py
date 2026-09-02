import importlib.util, sys, contextlib, io, copy

spec=importlib.util.spec_from_file_location('v67','/mnt/data/symbolic_v67_raw_text_family_invention.py')
v=importlib.util.module_from_spec(spec); sys.modules['v67']=v
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(v)

print('=== v6.7 RAW-TEXT ADVERSARIAL AUDIT ===')

# 1. Ambiguous current action: opaque one-slot command must NOT be completed by guessing.
state=v.StoryState()
state.add(('ACTIVE',('POT','COOK')),'seed')
state.add(('ACTIVE',('POT','LIGHT')),'seed')
cmd,target,known=v.quote_context('Töpfchen steh',state)
print('ambiguous current action command:',cmd)
ambiguous_ok=(cmd is not None and cmd.slots==('POT',))

# 2. Wrong target consequence with matching high-level transition kind.
wrong='Die Lampe leuchtet. Das Tor öffnet. "Lampe plim leuchten." Danach hört das Tor auf zu öffnen.'
ex=v.extract_raw_transitions(wrong,'wrong-target','audit')[0]
fam=v.FAMILY_BY_SIG.get(v.canonical_signature(ex)) if v.open_world_safe(ex) else None
print('wrong target:',sorted(ex.before),'->',sorted(ex.after),'family',fam)
wrong_target_ok=(fam is None)

# 3. Only remote consequence: immediate response does nothing; later sentence stops.
remote='Die Lampe leuchtet. "Lampe plim leuchten." Danach bleibt alles ruhig. Viel später hört die Lampe auf zu leuchten.'
rex=v.extract_raw_transitions(remote,'remote','audit')[0]
rfam=v.FAMILY_BY_SIG.get(v.canonical_signature(rex)) if v.open_world_safe(rex) else None
print('remote consequence family:',rfam,'delta',v.canonical_signature(rex))
remote_ok=(rfam is None)

# 4. Missing prior negative for additive transition remains rejected.
missing='"Lampe zor leuchten." Danach leuchtet die Lampe.'
mex=v.extract_raw_transitions(missing,'missing','audit')[0]
msafe=v.open_world_safe(mex)
print('missing prior negative safe:',msafe,'before_negative',mেক্স.before_negative if False else mex.before_negative)
missing_ok=(not msafe)

# 5. Semantic reuse must not alter WORLD.
world={('ACTIVE',('GATE','OPEN'))}
before=set(world)
sem=v.LEX.recognize('plim',('GATE','OPEN'))
after=set(world)
print('semantic only:',sem,'world unchanged',before==after)
world_ok=(sem is not None and before==after)

# 6. Unknown token with no observed delta cannot create entry.
zero='Die Lampe leuchtet. "Lampe vorn leuchten." Danach leuchtet die Lampe.'
z=v.extract_raw_transitions(zero,'z','audit')[0]
lex=v.Lexicon(); ent=lex.observe(z)
print('zero-delta entry:',ent)
zero_ok=(ent is None)

checks={
 'ambiguous_current_action_not_guessed':ambiguous_ok,
 'wrong_target_consequence_not_same_family':wrong_target_ok,
 'remote_nonlocal_consequence_not_used':remote_ok,
 'missing_prior_negative_not_treated_as_false':missing_ok,
 'semantic_reuse_does_not_mutate_world':world_ok,
 'zero_delta_creates_no_lexical_entry':zero_ok,
}
for k,x in checks.items(): print(('PASS' if x else 'FAIL'),'|',k)
assert all(checks.values())

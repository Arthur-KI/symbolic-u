import importlib.util, sys, contextlib, io, copy
spec=importlib.util.spec_from_file_location('v66','/mnt/data/symbolic_v66_semantic_family_invention.py')
v=importlib.util.module_from_spec(spec); sys.modules['v66']=v
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(v)

print('=== v6.6 FAMILY ADVERSARIAL AUDIT ===')
# 1) Same relation/polarity but wrong variable topology must not hit TRANSFER.
wrong_transfer=v.E('w1','audit','x',('ANNA','BEN','KEY'),
    [('HAVE',('ANNA','KEY'))],[('HAVE',('ANNA','KEY2'))])
wrong_family=v.FAMILY_BY_SIG.get(v.canonical_signature(wrong_transfer))
print('wrong transfer topology =>',wrong_family)

# 2) ADD ACTIVE and ADD AT must remain different families despite same polarity.
print('add ACTIVE',v.EVAL['START'],'add AT',v.EVAL['ENTER'])

# 3) A lexical token with one STOP and one ENTER observation never activates.
lex=v.FamilyLexicon(min_support=2)
a=v.held['STOP'][0]
b=v.Experience('mixed-family','travel',v.Command('plim',('ANNA','GARDEN')),
    frozenset(),frozenset({('AT',('ANNA','GARDEN'))}))
lex.observe(a); lex.observe(b)
ent=lex.entries['plim']
print('mixed family after 2 observations:',ent.status,ent.support,ent.conflicts)

# 4) Duplicate the same transition under many fresh token names: family is structural, not token-driven.
new=[]
for i in range(5):
    ex=v.E(f'struct{i}','audit',f'tok{i}',('SENSOR','SCAN'),[],[('ACTIVE',('SENSOR','SCAN'))])
    new.append(v.FAMILY_BY_SIG.get(v.canonical_signature(ex)))
print('five fresh tokens same topology =>',new)

# 5) Query/recognition of all active held-out words leaves lexical support and world untouched.
lex2=copy.deepcopy(v.lex)
before={k:(e.status,e.support,set(e.evidence_ids),set(e.conflicts)) for k,e in lex2.entries.items()}
world={('AT',('BEN','HOUSE')),('ACTIVE',('WHEEL','TURN'))}
world0=set(world)
for tok,slots in [('zor',('WHEEL','TURN')),('plim',('WHEEL','TURN')),('nex',('BEN','HOUSE')),('vak',('BEN','HOUSE'))]:
    lex2.recognize(tok,slots)
after={k:(e.status,e.support,set(e.evidence_ids),set(e.conflicts)) for k,e in lex2.entries.items()}
print('recognition mutation:',before==after,'world mutation:',world0==world)

checks={
 'wrong_variable_topology_not_transfer':wrong_family!=v.EVAL['TRANSFER'],
 'same_polarity_different_predicates_stay_distinct':v.EVAL['START']!=v.EVAL['ENTER'],
 'mixed_family_token_does_not_activate':ent.status=='STAGED' and ent.support==1 and len(ent.conflicts)==1,
 'family_assignment_is_structural_across_fresh_tokens':all(x==v.EVAL['START'] for x in new),
 'recognition_does_not_mutate_lexicon_or_world':before==after and world0==world,
}
for k,x in checks.items(): print(('PASS' if x else 'FAIL'),'|',k)
assert all(checks.values())

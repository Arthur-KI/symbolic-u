
import importlib.util, sys, contextlib, io, copy

spec=importlib.util.spec_from_file_location("v63","/mnt/data/symbolic_v63_kindergarten_to_long_text.py")
v=importlib.util.module_from_spec(spec); sys.modules["v63"]=v
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(v)

print("=== v6.3 LONG-DISCOURSE / SCALE AUDIT ===")

# 32/64 sentence length scaling with exactly the same frozen U set.
snap=(
    v.v.U_TRANS_V2.selectors,
    v.v.U_GIVE.selectors,
    v.v.U_REF.strategy,
    v.v.U_CLAUSE.strategy,
    v.v.U_CTX,
    v.v.U_LOCALITY,
    tuple(sorted(v.U_ADJ.items())),
    repr(v.STATE_RULES),
)
scale={}
for n in (32,64):
    evs,state,unknown=v.run_story(v.build_story(n))
    scale[n]=(len(evs),unknown,sorted(state))
    print(n,"sentences ->",len(evs),"events,",unknown,"unknown")
snap2=(
    v.v.U_TRANS_V2.selectors,
    v.v.U_GIVE.selectors,
    v.v.U_REF.strategy,
    v.v.U_CLAUSE.strategy,
    v.v.U_CTX,
    v.v.U_LOCALITY,
    tuple(sorted(v.U_ADJ.items())),
    repr(v.STATE_RULES),
)

# Long-ish discourse with object distractors; gender+role Reference-U should stay stable.
lines=[
    "Anna trägt den Schlüssel.",
    "Ben sieht den Hund.",
    "Sie öffnet den Schrank.",
    "Er sieht den Hund.",
    "Anna sieht den Hund.",
    "Ben trägt den Schlüssel.",
    "Sie trägt den Schlüssel.",
    "Er öffnet den Schrank.",
    "Anna trägt den Schlüssel.",
    "Ben sieht den Hund.",
    "Sie öffnet den Schrank.",
    "Er trägt den Schlüssel.",
]
mem=v.DMemory(); parsed=[]
for i,line in enumerate(lines):
    ev=v.parse_simple_sentence(line,mem,i)
    parsed.append(ev)
print("pronoun chain:")
for line,ev in zip(lines,parsed):
    print(" ",line,"=>",ev)

expected_pronoun=[
    (2,v.v.Event("OPEN",("ANNA","CABINET"))),
    (3,v.v.Event("SEE",("BEN","DOG"))),
    (6,v.v.Event("CARRY",("ANNA","KEY"))),
    (7,v.v.Event("OPEN",("BEN","CABINET"))),
    (10,v.v.Event("OPEN",("ANNA","CABINET"))),
    (11,v.v.Event("CARRY",("BEN","KEY"))),
]
pronoun_ok=all(parsed[i]==gold for i,gold in expected_pronoun)

# Add a second female person: "sie" must become UNKNOWN, not recency guessed.
amb=v.DMemory()
v.update_memory_from_sentence(amb,"Anna trägt den Schlüssel.",0)
v.update_memory_from_sentence(amb,"Cara trägt den Hund.",1)
amb_ev=v.parse_pronoun_transitive("Sie öffnet den Schrank.",amb,2)
print("female ambiguity ->",amb_ev)

checks={
    "scale_32_all_events_no_unknown":scale[32][0]==32 and scale[32][1]==0,
    "scale_64_all_events_no_unknown":scale[64][0]==64 and scale[64][1]==0,
    "length_does_not_mutate_U_set":snap==snap2,
    "long_pronoun_chain_keeps_role_binding":pronoun_ok,
    "late_ambiguity_remains_unknown":amb_ev is None,
}
for k,x in checks.items():
    print(("PASS" if x else "FAIL"),"|",k)
assert all(checks.values())

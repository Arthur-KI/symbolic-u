
import importlib.util, sys, contextlib, io
spec=importlib.util.spec_from_file_location("v61","/mnt/data/symbolic_v61_language_bridge.py")
v=importlib.util.module_from_spec(spec); sys.modules["v61"]=v
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(v)
text='Ein Mädchen traf ein Kind. Es sagte „Topf koche.“ Danach kochte es.'
br=v.parse_with_bridge(text,"amb-final","REAL_WORLD")
us=v.m.parsed_to_ustory(br.parsed,"audit")
v.materialize_local_active(us)
people=[e for e in br.memory.entities.values() if "PERSON" in e.types]
r1=[e for e in us.events if e.rel==v.m.R_DIRECTIVE]
print("=== FINAL AMBIGUITY AUDIT ===")
print("people:",[(e.eid,sorted(e.types),e.gram_gender) for e in people])
print("unresolved:",br.memory.unresolved)
print("R1:",[(e.args,e.source) for e in r1])
ok=(len(people)==2 and len(r1)==0 and len(br.memory.unresolved)>=1)
print("audit:","PASS" if ok else "FAIL")
assert ok

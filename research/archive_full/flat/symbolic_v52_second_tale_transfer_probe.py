
from pathlib import Path
import types, sys, json

# Freeze: reuse v5.2 parser exactly as-is; do not add frames or lexicon.
src=Path("/mnt/data/symbolic_v52_generic_event_fairytale_lore_test.py").read_text(encoding="utf-8")
prefix=src.split("# ---------- build worlds ----------")[0]
m=types.ModuleType("v52_frozen_transfer_probe")
sys.modules[m.__name__]=m
exec(prefix,m.__dict__)

text=Path("/mnt/data/grimm_der_suesse_brei.txt").read_text(encoding="utf-8")
m.TEXT=text
m.N=m.norm(text)
w=m.WorldModel("FAIRY_TALE")
m.bootstrap_entities(w,m.N)
m.extract_dialogue(w,text)
m.extract_generic_events(w,text)

# Small held-out semantic probe from the second tale.
targets=[
    ("GIVE",("old_woman","girl","POT")),
    ("COMMAND",("mother","POT","COOK")),
    ("COOK",("POT","PORRIDGE")),
    ("COMMAND",("girl","POT","STOP")),
    ("STOP_COOK",("POT",)),
]
proved=sum(any(f.rel==rel and f.args==args for f in w.facts) for rel,args in targets)

print("=== v5.2 FROZEN SECOND-TALE TRANSFER PROBE ===")
print("Facts extracted:",len(w.facts))
print("Held-out targets proved:",proved,"/",len(targets))
for rel,args in targets:
    ok=any(f.rel==rel and f.args==args for f in w.facts)
    print(("PASS" if ok else "MISS"),"|",rel,args)
print("No v5.2 rule/lexicon changes were made for this probe.")

Path("/mnt/data/symbolic_v52_second_tale_transfer_probe.json").write_text(
    json.dumps({
        "source":"grimm_der_suesse_brei.txt",
        "frozen_v52":True,
        "facts":len(w.facts),
        "targets_proved":proved,
        "targets_n":len(targets),
        "result":"TRANSFER_WEAK" if proved < len(targets) else "PASS"
    },ensure_ascii=False,indent=2),
    encoding="utf-8"
)

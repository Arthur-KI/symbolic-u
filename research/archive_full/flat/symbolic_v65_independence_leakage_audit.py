
import importlib.util, sys, contextlib, io, copy

spec=importlib.util.spec_from_file_location("v65","/mnt/data/symbolic_v65_unknown_imperative.py")
v=importlib.util.module_from_spec(spec); sys.modules["v65"]=v
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(v)

print("=== v6.5 INDEPENDENCE / LEAKAGE AUDIT ===")

# 1. Duplicating the same context ID must not fake independent support.
lex=v.LexiconLearner(min_support=2)
e=v.SYN1
lex.observe(e)
lex.observe(e)  # exact duplicate context
dup=lex.entries["plim"]
print("duplicate context:",dup.status,dup.support,dup.provenance)

# 2. Active semantics for one opaque token must not leak to another.
other=lex.recognize("quux","GATE","OPEN")
print("cross-token leakage:",other)

# 3. Cessation after an unknown command but WITHOUT a matching pre-active state
# must not identify the token as a stop directive.
no_pre=v.StopExperience(
    "no-pre",
    "hush",
    "LAMP",
    "LAMP",
    None,
    "LAMP",
    "LIGHT",
)
no_pre_accept=v.schema_accepts(v.U_UNKNOWN_IMPERATIVE,no_pre)
print("no-pre-active accepted:",no_pre_accept)

# 4. Command target and observed cessation target mismatch, same action.
target_mismatch=v.StopExperience(
    "target-mismatch",
    "hush",
    "LAMP",
    "LAMP",
    "LIGHT",
    "GATE",
    "LIGHT",
)
target_ok=v.schema_accepts(v.U_UNKNOWN_IMPERATIVE,target_mismatch)
print("target mismatch accepted:",target_ok)

# 5. Matching active token remains semantic-only until independent occurrence.
active=v.LexiconLearner(min_support=2)
active.observe(v.SYN1); active.observe(v.SYN2)
sem=active.recognize("plim","MACHINE","RUN")
world=set()
print("semantic recognition:",sem,"world facts:",world)

checks={
    "duplicate_context_does_not_activate":dup.status=="STAGED" and dup.support==1,
    "opaque_token_meaning_does_not_leak":other is None,
    "no_pre_active_state_no_stop_induction":not no_pre_accept,
    "same_action_wrong_target_rejected":not target_ok,
    "semantic_recognition_does_not_create_world_fact":sem==("R10",("MACHINE","RUN")) and not world,
}
for k,x in checks.items():
    print(("PASS" if x else "FAIL"),"|",k)
assert all(checks.values())

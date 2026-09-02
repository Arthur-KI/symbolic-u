from symbolic_u.monolith import *

c = MonolithCompiler(threshold=3)
q = qid("TIME", ("B",))
ans = Answer(1, False, "T720", ("ANCHOR:A", "U:UB", "ADD"))

for _ in range(3):
    c.observe("STORY_A", q, ans, {"ANCHOR:A", "U:UB"})
print("compiled:", c.query("STORY_A", q))
print("other context:", c.query("STORY_B", q))

print("retired:", c.retire_dependency("U:UB"))
print("after retirement:", c.query("STORY_A", q))


from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import FrozenSet, Tuple, List, Dict, Optional, Set

# ============================================================
# Symbolic Mini-LM v1.3
# Mention -> Reference-U -> Entity
#
# Goal:
# - pronouns, noun re-descriptions and epithets use ONE mechanism
# - ambiguity remains U=0
# - query matching itself is never evidence
# - reference resolution uses independent symbolic constraints
# ============================================================

class T(IntEnum):
    FALSE = -1
    UNKNOWN = 0
    TRUE = 1

def tn(x):
    return {T.TRUE:"+1", T.UNKNOWN:"0", T.FALSE:"-1"}[T(x)]

@dataclass
class Entity:
    eid: str
    types: Set[str]
    grammatical_genders: Set[str] = field(default_factory=set)
    number: str = "SG"
    attrs: Set[str] = field(default_factory=set)
    capabilities: Set[str] = field(default_factory=set)
    members: Tuple[str,...] = ()
    present: bool = True
    aliases: Set[str] = field(default_factory=set)

@dataclass(frozen=True)
class Mention:
    mid: str
    surface: str
    required_types: FrozenSet[str] = frozenset()
    allowed_genders: FrozenSet[str] = frozenset()
    number: Optional[str] = None
    required_attrs: FrozenSet[str] = frozenset()
    required_capability: Optional[str] = None
    require_present: bool = False
    story_id: str = "story"

@dataclass
class ReferenceU:
    mention: Mention
    entity_id: str
    state: T
    evidence: List[str] = field(default_factory=list)

class ReferenceResolver:
    def __init__(self, entities: Dict[str,Entity]):
        self.entities = entities

    def _compatible(self, m:Mention, e:Entity):
        ev=[]

        if m.number and e.number != m.number:
            return False,[f"number {e.number} != {m.number}"]

        if m.required_types and not (m.required_types & e.types):
            return False,[f"type mismatch {sorted(e.types)} vs {sorted(m.required_types)}"]

        if m.allowed_genders:
            if not (m.allowed_genders & e.grammatical_genders):
                return False,[f"gender mismatch {sorted(e.grammatical_genders)} vs {sorted(m.allowed_genders)}"]

        if m.required_attrs and not m.required_attrs.issubset(e.attrs):
            return False,[f"missing attrs {sorted(m.required_attrs - e.attrs)}"]

        if m.required_capability and m.required_capability not in e.capabilities:
            return False,[f"missing capability {m.required_capability}"]

        if m.require_present and not e.present:
            return False,["entity not present in local scene"]

        if m.required_types:
            ev.append("type-compatible")
        if m.allowed_genders:
            ev.append("gender-compatible")
        if m.required_attrs:
            ev.append("attribute-compatible")
        if m.required_capability:
            ev.append(f"role requires capability {m.required_capability}")
        if m.require_present:
            ev.append("present in local scene")
        if m.number:
            ev.append(f"number={m.number}")
        return True,ev

    def resolve(self,m:Mention) -> List[ReferenceU]:
        compat=[]
        rejected=[]
        for eid,e in self.entities.items():
            ok,ev=self._compatible(m,e)
            if ok:
                compat.append((eid,ev))
            else:
                rejected.append(ReferenceU(m,eid,T.FALSE,ev))

        out=[]
        if len(compat)==1:
            eid,ev=compat[0]
            out.append(ReferenceU(m,eid,T.TRUE,ev+["unique independent binding"]))
        else:
            for eid,ev in compat:
                out.append(ReferenceU(m,eid,T.UNKNOWN,ev+[f"{len(compat)} compatible entities remain"]))

        out.extend(rejected)
        return out

    @staticmethod
    def chosen(us:List[ReferenceU]):
        xs=[u for u in us if u.state==T.TRUE]
        return xs[0].entity_id if len(xs)==1 else None

    @staticmethod
    def pending(us:List[ReferenceU]):
        return [u.entity_id for u in us if u.state==T.UNKNOWN]

# ------------------------------------------------------------
# Local entity memory for "Der süße Brei"
# ------------------------------------------------------------

E:Dict[str,Entity]={}

# "ein armes frommes Mädchen ... mit seiner Mutter"
E["girl"]=Entity(
    "girl",
    types={"PERSON","CHILD"},
    grammatical_genders={"NEUTER"},   # "das Mädchen", "das Kind"
    attrs={"POOR","PIOUS","FEMALE"},
    capabilities={"MOVE","EAT","SPEAK"},
    aliases={"Mädchen"}
)
E["mother"]=Entity(
    "mother",
    types={"PERSON","ADULT"},
    grammatical_genders={"FEM"},
    attrs={"FEMALE","MOTHER"},
    capabilities={"EAT","SPEAK"},
    aliases={"Mutter"}
)
E["girl_mother_group"]=Entity(
    "girl_mother_group",
    types={"PERSON_GROUP"},
    grammatical_genders={"PLURAL"},
    number="PL",
    attrs={"FAMILY_GROUP"},
    members=("girl","mother"),
    aliases={"sie"}
)

# "eine alte Frau"
E["old_woman"]=Entity(
    "old_woman",
    types={"PERSON","ADULT"},
    grammatical_genders={"FEM"},
    attrs={"FEMALE","OLD"},
    capabilities={"MOVE","SPEAK"},
    aliases={"alte Frau"}
)

# gifted pot
E["pot"]=Entity(
    "pot",
    types={"OBJECT","VESSEL","COOK_DEVICE"},
    grammatical_genders={"NEUTER","MASC"}, # Töpfchen / Topf
    attrs={"MAGICAL"},
    capabilities={"COOK","STOP_COOK"},
    aliases={"Töpfchen","Topf"}
)

resolver=ReferenceResolver(E)

def show(label,m,expected=None,expected_pending=None):
    us=resolver.resolve(m)
    chosen=resolver.chosen(us)
    pending=resolver.pending(us)
    ok=True
    if expected is not None:
        ok = chosen==expected
    if expected_pending is not None:
        ok = set(pending)==set(expected_pending) and chosen is None
    print(("PASS" if ok else "FAIL"),"|",label)
    print("  mention:",m.surface)
    print("  chosen :",chosen)
    if pending:
        print("  pending:",pending)
    for u in us:
        if u.state != T.FALSE:
            print("   ",tn(u.state),u.entity_id,"|","; ".join(u.evidence))
    return ok

results=[]

# ------------------------------------------------------------
# 1. Noun re-description: "Mädchen" -> "das Kind"
# ------------------------------------------------------------
results.append(show(
    '"das Kind" resolves to same entity as Mädchen',
    Mention("m1","das Kind",
            required_types=frozenset({"CHILD"}),
            allowed_genders=frozenset({"NEUTER"}),
            number="SG"),
    expected="girl"
))

# ------------------------------------------------------------
# 2. "ihm" in "begegnete ihm ... eine alte Frau"
#    PERSON + masc/neuter dative -> girl/Kind; old woman is FEM.
# ------------------------------------------------------------
results.append(show(
    '"ihm" resolves to Kind',
    Mention("m2","ihm",
            required_types=frozenset({"PERSON"}),
            allowed_genders=frozenset({"MASC","NEUTER"}),
            number="SG"),
    expected="girl"
))

# ------------------------------------------------------------
# 3. "sie hatten nichts..." is plural group reference.
# ------------------------------------------------------------
results.append(show(
    'plural "sie" resolves to Mädchen+Mutter group',
    Mention("m3","sie",
            required_types=frozenset({"PERSON_GROUP"}),
            allowed_genders=frozenset({"PLURAL"}),
            number="PL"),
    expected="girl_mother_group"
))

# ------------------------------------------------------------
# 4. "es" after command must fill COOK actor role.
#    Both Mädchen and Töpfchen can be grammatically neuter,
#    but only pot has COOK capability.
# ------------------------------------------------------------
results.append(show(
    '"es" in cooking outcome resolves via event-role constraint',
    Mention("m4","es",
            allowed_genders=frozenset({"NEUTER"}),
            number="SG",
            required_capability="COOK"),
    expected="pot"
))

# ------------------------------------------------------------
# 5. "Topf" is a re-description of earlier "Töpfchen".
# ------------------------------------------------------------
results.append(show(
    '"den Topf" resolves to same pot entity',
    Mention("m5","den Topf",
            required_types=frozenset({"VESSEL"}),
            allowed_genders=frozenset({"MASC"}),
            number="SG"),
    expected="pot"
))

# ------------------------------------------------------------
# 6. Later mother scene:
#    girl is explicitly away; "sie ißt sich satt" must be mother.
# ------------------------------------------------------------
E["girl"].present=False
E["old_woman"].present=False
results.append(show(
    '"sie" while girl is away resolves to Mutter',
    Mention("m6","sie",
            required_types=frozenset({"PERSON"}),
            allowed_genders=frozenset({"FEM"}),
            number="SG",
            required_capability="EAT",
            require_present=True),
    expected="mother"
))

# ------------------------------------------------------------
# 7. Girl returns; "das Kind" again resolves to same persistent entity.
# ------------------------------------------------------------
E["girl"].present=True
results.append(show(
    'later "das Kind" still resolves to persistent girl entity',
    Mention("m7","das Kind",
            required_types=frozenset({"CHILD"}),
            allowed_genders=frozenset({"NEUTER"}),
            number="SG"),
    expected="girl"
))

# ------------------------------------------------------------
# 8. Final "es ... hört auf zu kochen" -> pot via STOP_COOK role.
# ------------------------------------------------------------
results.append(show(
    'final "es" resolves to pot via STOP_COOK role',
    Mention("m8","es",
            allowed_genders=frozenset({"NEUTER"}),
            number="SG",
            required_capability="STOP_COOK"),
    expected="pot"
))

# ============================================================
# Synthetic Frau-Holle-style epithet tests
# (mechanism only; not sourced from uploaded text)
# ============================================================

E2={
    "daughter_A":Entity(
        "daughter_A",{"PERSON"},{"FEM"},"SG",
        {"FEMALE","BEAUTIFUL","DILIGENT"},{"MOVE","SPEAK"}
    ),
    "daughter_B":Entity(
        "daughter_B",{"PERSON"},{"FEM"},"SG",
        {"FEMALE","UGLY","LAZY"},{"MOVE","SPEAK"}
    ),
}
r2=ReferenceResolver(E2)

def show2(label,m,expected=None,expected_pending=None):
    us=r2.resolve(m)
    chosen=r2.chosen(us)
    pending=r2.pending(us)
    ok = (chosen==expected) if expected is not None else (set(pending)==set(expected_pending or []) and chosen is None)
    print(("PASS" if ok else "FAIL"),"|",label,"chosen=",chosen,"pending=",pending)
    return ok

results.append(show2(
    '"die Schöne" -> entity with BEAUTIFUL attribute',
    Mention("f1","die Schöne",
            required_types=frozenset({"PERSON"}),
            allowed_genders=frozenset({"FEM"}),
            required_attrs=frozenset({"BEAUTIFUL"})),
    expected="daughter_A"
))
results.append(show2(
    '"die Häßliche" -> entity with UGLY attribute',
    Mention("f2","die Häßliche",
            required_types=frozenset({"PERSON"}),
            allowed_genders=frozenset({"FEM"}),
            required_attrs=frozenset({"UGLY"})),
    expected="daughter_B"
))

# Ambiguity must stay 0.
E3={
    "a":Entity("a",{"PERSON"},{"FEM"},"SG",{"BEAUTIFUL"},set()),
    "b":Entity("b",{"PERSON"},{"FEM"},"SG",{"BEAUTIFUL"},set()),
}
r3=ReferenceResolver(E3)
amb=r3.resolve(Mention(
    "f3","die Schöne",
    required_types=frozenset({"PERSON"}),
    allowed_genders=frozenset({"FEM"}),
    required_attrs=frozenset({"BEAUTIFUL"})
))
amb_pending=r3.pending(amb)
amb_ok=r3.chosen(amb) is None and set(amb_pending)=={"a","b"}
results.append(amb_ok)
print(("PASS" if amb_ok else "FAIL"),'| ambiguous "die Schöne" stays U=0',amb_pending)

# Query must not resolve an otherwise ambiguous pronoun.
E4={
    "p1":Entity("p1",{"PERSON"},{"FEM"},"SG",{"FEMALE"},{"EAT"},present=True),
    "p2":Entity("p2",{"PERSON"},{"FEM"},"SG",{"FEMALE"},{"EAT"},present=True),
}
r4=ReferenceResolver(E4)
amb2=r4.resolve(Mention(
    "q1","sie",
    required_types=frozenset({"PERSON"}),
    allowed_genders=frozenset({"FEM"}),
    number="SG",
    required_capability="EAT",
    require_present=True
))
query_safe = r4.chosen(amb2) is None and set(r4.pending(amb2))=={"p1","p2"}
results.append(query_safe)
print(("PASS" if query_safe else "FAIL"),"| query cannot self-fulfil ambiguous pronoun",r4.pending(amb2))

print("\nPassed",sum(results),"/",len(results))
assert all(results)
print("ALL v1.3 REFERENCE-U ASSERTIONS PASSED")

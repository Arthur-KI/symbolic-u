from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from collections import defaultdict
from typing import Iterable, Mapping

class Truth(IntEnum):
    FALSE = -1
    UNKNOWN = 0
    TRUE = 1

@dataclass(frozen=True, order=True)
class Proposition:
    name: str
    args: tuple = ()
    context: str = "WORLD"

    def __str__(self):
        a = ",".join(map(str, self.args))
        return f"{self.context}:{self.name}({a})"

@dataclass(frozen=True)
class Pattern:
    name: str
    args: tuple = ()
    context: str | None = None

@dataclass
class Rule:
    uid: str
    premises: tuple[Pattern, ...]
    conclusion: Pattern
    state: Truth = Truth.TRUE
    provenance: tuple[str, ...] = ()

@dataclass(frozen=True)
class QueryResult:
    state: Truth
    contradiction: bool = False
    positive_support: tuple[str, ...] = ()
    negative_support: tuple[str, ...] = ()
    touched_rules: tuple[str, ...] = ()
    cost: int = 0

def is_var(x) -> bool:
    return isinstance(x, str) and x.startswith("?")

def unify(pattern: Pattern, prop: Proposition) -> dict[str, object] | None:
    if pattern.name != prop.name or len(pattern.args) != len(prop.args):
        return None
    if pattern.context is not None and pattern.context != prop.context:
        return None
    env: dict[str, object] = {}
    for p, v in zip(pattern.args, prop.args):
        if is_var(p):
            if p in env and env[p] != v:
                return None
            env[p] = v
        elif p != v:
            return None
    return env

def instantiate(pattern: Pattern, env: Mapping[str, object], default_context: str) -> Proposition:
    args = tuple(env.get(x, x) if is_var(x) else x for x in pattern.args)
    ctx = pattern.context if pattern.context is not None else default_context
    return Proposition(pattern.name, args, ctx)

class World:
    """Context-separated symbolic world.

    Base evidence is positive support for ordinary propositions.
    Explicit negative truth is represented by positive evidence for an
    opposite proposition, never by absence of evidence.
    """
    def __init__(self):
        self.facts: dict[Proposition, set[str]] = defaultdict(set)
        self.rules: list[Rule] = []
        self.opposites: dict[tuple[str, str], str] = {}
        self.version = 0

    def add_fact(self, p: Proposition, source: str = "fact"):
        self.facts[p].add(source)
        self.version += 1

    def remove_fact(self, p: Proposition):
        if p in self.facts:
            del self.facts[p]
            self.version += 1

    def add_rule(self, rule: Rule):
        self.rules.append(rule)
        self.version += 1

    def set_rule_state(self, uid: str, state: Truth):
        for r in self.rules:
            if r.uid == uid:
                r.state = state
                self.version += 1
                return
        raise KeyError(uid)

    def set_opposition(self, name_a: str, name_b: str):
        self.opposites[(name_a, name_b)] = name_b
        self.opposites[(name_b, name_a)] = name_a

    def opposite(self, p: Proposition) -> Proposition | None:
        for (a, _), b in self.opposites.items():
            if a == p.name:
                return Proposition(b, p.args, p.context)
        return None

class BackwardProver:
    """Query-guided backward prover.

    Critical invariant:
        Rule.state == FALSE means only "this derivation is rejected".
        It never creates a negative output proposition.
    """
    def __init__(self, world: World, budget: int = 100_000):
        self.world = world
        self.budget = budget

    def query(self, target: Proposition) -> QueryResult:
        touched: set[str] = set()
        cost = [0]
        pos_ok, pos_trace = self._prove_positive(target, set(), touched, cost)

        opp = self.world.opposite(target)
        neg_ok = False
        neg_trace: tuple[str, ...] = ()
        if opp is not None:
            neg_ok, neg_trace = self._prove_positive(opp, set(), touched, cost)

        if pos_ok and neg_ok:
            return QueryResult(
                Truth.UNKNOWN, True, pos_trace, neg_trace,
                tuple(sorted(touched)), cost[0]
            )
        if pos_ok:
            return QueryResult(
                Truth.TRUE, False, pos_trace, (),
                tuple(sorted(touched)), cost[0]
            )
        if neg_ok:
            return QueryResult(
                Truth.FALSE, False, (), neg_trace,
                tuple(sorted(touched)), cost[0]
            )
        return QueryResult(
            Truth.UNKNOWN, False, (), (),
            tuple(sorted(touched)), cost[0]
        )

    def _prove_positive(
        self,
        target: Proposition,
        active: set[Proposition],
        touched: set[str],
        cost: list[int],
    ) -> tuple[bool, tuple[str, ...]]:
        if cost[0] >= self.budget:
            return False, ("BUDGET",)
        cost[0] += 1

        if target in active:
            return False, (f"CYCLE:{target}",)

        if target in self.world.facts:
            src = sorted(self.world.facts[target])
            return True, tuple(f"FACT:{x}" for x in src)

        active2 = set(active)
        active2.add(target)

        # Only inspect rules whose conclusion can match the queried target.
        for rule in self.world.rules:
            env = unify(rule.conclusion, target)
            if env is None:
                continue
            touched.add(rule.uid)

            if rule.state != Truth.TRUE:
                # 0 = pending; -1 = rejected. Neither is negative evidence.
                continue

            traces = []
            good = True
            for premise in rule.premises:
                p = instantiate(premise, env, target.context)
                ok, tr = self._prove_positive(p, active2, touched, cost)
                if not ok:
                    good = False
                    break
                traces.extend(tr)

            if good:
                return True, tuple(traces) + (f"U:{rule.uid}",)

        return False, ()

__all__ = [
    "Truth", "Proposition", "Pattern", "Rule", "World",
    "BackwardProver", "QueryResult", "unify", "instantiate"
]

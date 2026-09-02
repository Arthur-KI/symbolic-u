from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable

@dataclass
class ProgressDomain:
    """Anonymous discrete progress chain learned/derived from ORDER fragments."""
    next: dict[object, object]
    prev: dict[object, object]
    zero: object | None

    @classmethod
    def from_order_fragments(cls, fragments: Iterable[Iterable[object]]):
        nxt: dict[object, object] = {}
        prv: dict[object, object] = {}
        nodes = set()
        conflicts = set()

        for fragment in fragments:
            xs = tuple(fragment)
            nodes.update(xs)
            for a, b in zip(xs, xs[1:]):
                if a in nxt and nxt[a] != b:
                    conflicts.add(a)
                if b in prv and prv[b] != a:
                    conflicts.add(b)
                nxt[a] = b
                prv[b] = a

        for x in conflicts:
            nxt.pop(x, None)
            prv.pop(x, None)

        minima = [x for x in nodes if x not in prv]
        zero = minima[0] if len(minima) == 1 else None
        return cls(nxt, prv, zero)

    def succ(self, x):
        return self.next.get(x)

    def pred(self, x):
        return self.prev.get(x)

class RecursiveArithmetic:
    """Frozen learned recursive structures from the arithmetic research line.

    ADD rule shape:
        y=zero -> z=x
        pred(y,y1) + pred(z,z1) + add(x,y1,z1) -> add(x,y,z)

    MUL rule shape:
        y=zero -> z=zero
        pred(y,y1) + mul(x,y1,z1) + add(z1,x,z) -> mul(x,y,z)

    The names ADD/MUL are API labels here; the recursive structures are the
    learned rule shapes tested in the research scripts.
    """
    def __init__(self, domain: ProgressDomain):
        if domain.zero is None:
            raise ValueError("Progress domain has no unique minimum/zero")
        self.d = domain

    def add_output(self, x, y):
        steps = 0
        cur = y
        seen = set()
        while cur != self.d.zero:
            if cur in seen:
                return None
            seen.add(cur)
            cur = self.d.pred(cur)
            if cur is None:
                return None
            steps += 1

        z = x
        for _ in range(steps):
            z = self.d.succ(z)
            if z is None:
                return None
        return z

    def add_first(self, second, output):
        """Backward bind x in ADD(x, second, output)."""
        y = second
        z = output
        seen = set()
        while y != self.d.zero:
            if y in seen:
                return None
            seen.add(y)
            y = self.d.pred(y)
            z = self.d.pred(z)
            if y is None or z is None:
                return None
        return z

    def mul_output(self, x, y):
        chain = []
        cur = y
        seen = set()
        while cur != self.d.zero:
            if cur in seen:
                return None
            seen.add(cur)
            p = self.d.pred(cur)
            if p is None:
                return None
            chain.append(cur)
            cur = p

        z = self.d.zero
        while chain:
            chain.pop()
            z = self.add_output(z, x)
            if z is None:
                return None
        return z

    def count_members(self, members):
        """Same recursive progress idea used for visual/member counting."""
        node = self.d.zero
        trace = ["COUNT_BASE_EMPTY"]
        for m in reversed(tuple(members)):
            nxt = self.d.succ(node)
            if nxt is None:
                return None, tuple(trace + ["PROGRESS_MISSING"])
            trace.extend((f"MEMBER:{m}", f"PROGRESS:{node}->{nxt}"))
            node = nxt
        return node, tuple(trace)

from __future__ import annotations
import hashlib

def opaque_chain(n: int, prefix="Q"):
    nodes = [
        f"{prefix}_{hashlib.sha1(f'{prefix}:{i}'.encode()).hexdigest()[:12]}"
        for i in range(n+1)
    ]
    fragments = [tuple(nodes)]
    return nodes, fragments

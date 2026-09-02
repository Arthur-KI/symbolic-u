"""K24 symbolic vision learner scaffold.

IMAGE_CONTEXT -> generic sensor Keys -> learned visual U -> backward query.
Never hardcode RGB->color names or corner-count->shape names.
Hard invariants: U=-1 != KEY=-1; query read-only; context separates images; ambiguous sensor evidence -> KEY 0.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class VisualKey:
    image_id:str
    predicate:str
    args:tuple
    state:int=0

@dataclass(frozen=True)
class VisualU:
    uid:str
    inputs:tuple
    output:VisualKey
    state:int=0

# Actual induction is intentionally deferred until real uploaded images exist.
# The curriculum, not the sensor, teaches color/shape/count semantics.

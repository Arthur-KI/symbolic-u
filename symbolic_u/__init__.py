"""Symbolic U research runtime.

Purely symbolic research prototype:
- ternary KEY/U semantics
- explicit opposition
- query-guided backward proof
- progress/arithmetic
- temporal/state reasoning
- decomposable Macro-U compilation
- classical-vision -> symbolic reasoning adapter
"""
from .core import (
    Truth, Proposition, Pattern, Rule, World, BackwardProver, QueryResult
)
from .arithmetic import ProgressDomain, RecursiveArithmetic
from .temporal import (
    TemporalLexicon, TimeGraph, Interval, IntervalReasoner,
    StateEvent, StateReasoner
)
from .monolith import Answer, MacroU, MonolithCompiler

__version__ = "0.25.0"

from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
import json

@dataclass(frozen=True)
class MonolithAnswer:
    state:int
    contradiction:bool
    value:str|None
    mode:str
    macros:tuple[str,...]=()

class CompiledMonolithRuntime:
    """Fast decomposable Macro-U layer.

    It intentionally does not invent answers on a cache miss or retired Macro-U.
    The caller should route NEEDS_MICRO_FALLBACK to the original backward U engine.
    """
    def __init__(self,spec_path="/mnt/data/M_SYMBOLIC_V1_PRODUCTION.json"):
        self.spec=json.loads(Path(spec_path).read_text(encoding="utf-8"))
        self.context_id=self.spec["context_id"]
        self.macros={m["mid"]:dict(m) for m in self.spec["macros"]}
        self.by_query=defaultdict(list)
        self.reverse_deps=defaultdict(set)
        for mid,m in self.macros.items():
            self.by_query[m["query_id"]].append(mid)
            for d in m["deps"]:
                self.reverse_deps[d].add(mid)

    @staticmethod
    def qid(kind,args):
        return kind+"|"+json.dumps(tuple(args),ensure_ascii=False,separators=(",",":"))

    def query(self,context_id,kind,*args):
        if context_id!=self.context_id:
            return MonolithAnswer(0,False,None,"NEEDS_MICRO_FALLBACK",())
        qid=self.qid(kind,args)
        active=[self.macros[mid] for mid in self.by_query.get(qid,())
                if self.macros[mid]["state"]==1]
        pos=[m for m in active if m["output_polarity"]=="POS"]
        neg=[m for m in active if m["output_polarity"]=="NEG"]
        mids=tuple(m["mid"] for m in active)
        if pos and neg:
            return MonolithAnswer(0,True,None,"MONOLITH",mids)
        if pos:
            sig=pos[0]["answer_signature"]
            return MonolithAnswer(1,False,sig[2],"MONOLITH",mids)
        if neg:
            sig=neg[0]["answer_signature"]
            return MonolithAnswer(-1,False,sig[2],"MONOLITH",mids)
        return MonolithAnswer(0,False,None,"NEEDS_MICRO_FALLBACK",())

    def decompose(self,context_id,kind,*args):
        if context_id!=self.context_id:
            return []
        qid=self.qid(kind,args)
        out=[]
        for mid in self.by_query.get(qid,()):
            m=self.macros[mid]
            out.append({
                "mid":mid,
                "state":m["state"],
                "polarity":m["output_polarity"],
                "dependencies":m["deps"],
                "provenance":m["provenance"],
                "proof_shape":m["proof_shape"],
                "support":m["support"],
                "retire_reason":m.get("retire_reason","")
            })
        return out

    def retire_dependency(self,dependency,reason="dependency changed"):
        retired=[]
        for mid in self.reverse_deps.get(dependency,()):
            m=self.macros[mid]
            if m["state"]==1:
                m["state"]=-1
                m["retire_reason"]=reason
                retired.append(mid)
        return tuple(sorted(retired))

if __name__=="__main__":
    rt=CompiledMonolithRuntime()
    samples=[
        ("MAIN","TIME","B"),
        ("MAIN","STATE","wolf","house","X"),
        ("MAIN","MEET","wolf","anna","house","B"),
        ("OTHER","TIME","B"),
    ]
    for x in samples:
        print(x,"=>",rt.query(x[0],x[1],*x[2:]))

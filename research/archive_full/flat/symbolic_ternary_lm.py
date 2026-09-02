# Symbolic Ternary Language Model – Proof of Concept
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Dict, Set, Optional

class Truth(Enum):
    TRUE=1; UNKNOWN=0; FALSE=-1
    def __str__(self):
        return {Truth.TRUE:"WAHR",Truth.UNKNOWN:"UNBEKANNT",Truth.FALSE:"FALSCH"}[self]

@dataclass(frozen=True)
class Fact:
    relation: str
    args: Tuple[str,...]
    def __str__(self):
        return f"{self.relation}({', '.join(self.args)})"

@dataclass(frozen=True)
class Rule:
    premises: Tuple[Fact,...]
    conclusion: Fact
    name: str

class SymbolicTernaryLM:
    def __init__(self):
        self.facts:Set[Fact]=set()
        self.negative_facts:Set[Fact]=set()
        self.lexicon={"größer":"GREATER","kleiner":"LESS","jagt":"CHASE","verfolgt":"CHASE","mag":"LIKE"}
        self.rules=[
            Rule((Fact("GREATER",("?x","?y")),Fact("GREATER",("?y","?z"))),
                 Fact("GREATER",("?x","?z")),"Transitivität"),
            Rule((Fact("GREATER",("?x","?y")),),
                 Fact("LESS",("?y","?x")),"Inverse größer/kleiner")
        ]

    def is_var(self,x): return x.startswith("?")
    def clean(self,x): return x.strip(" \n\t.,!?").lower()

    def substitute(self,fact,env):
        def resolve(x):
            seen=set()
            while self.is_var(x) and x in env and x not in seen:
                seen.add(x); x=env[x]
            return x
        return Fact(fact.relation,tuple(resolve(a) for a in fact.args))

    def unify_args(self,a,b,env):
        env=dict(env)
        def resolve(x):
            seen=set()
            while self.is_var(x) and x in env and x not in seen:
                seen.add(x); x=env[x]
            return x
        a,b=resolve(a),resolve(b)
        if a==b: return env
        if self.is_var(a): env[a]=b; return env
        if self.is_var(b): env[b]=a; return env
        return None

    def unify(self,a,b,env=None):
        if a.relation!=b.relation or len(a.args)!=len(b.args): return None
        env=dict(env or {})
        for x,y in zip(a.args,b.args):
            env=self.unify_args(x,y,env)
            if env is None: return None
        return env

    def parse_sentence(self,sentence):
        s=sentence.strip().rstrip(".!?"); low=s.lower()
        for word,rel in (("größer","GREATER"),("kleiner","LESS")):
            neg=f" ist nicht {word} als "; pos=f" ist {word} als "
            if neg in low:
                left,right=low.split(neg,1)
                f=Fact(rel,(self.clean(left),self.clean(right)))
                self.negative_facts.add(f); return "NEG",f
            if pos in low:
                left,right=low.split(pos,1)
                f=Fact(rel,(self.clean(left),self.clean(right)))
                self.facts.add(f); return "POS",f

        articles={"der","die","das","den","dem","ein","eine","einen","einem"}
        words=[w.strip(".,!?") for w in low.split()]
        words=[w for w in words if w not in articles]
        if len(words)>=3 and words[1] in self.lexicon:
            f=Fact(self.lexicon[words[1]],(words[0],words[2]))
            self.facts.add(f); return "POS",f
        return "UNPARSED",s

    def ingest(self,text):
        return [self.parse_sentence(s) for s in text.replace("\n"," ").split(".") if s.strip()]

    def direct_truth(self,goal):
        if goal in self.facts: return Truth.TRUE
        if goal in self.negative_facts: return Truth.FALSE
        if goal.relation in {"GREATER","LESS"} and len(goal.args)==2:
            reverse=Fact(goal.relation,(goal.args[1],goal.args[0]))
            if reverse in self.facts: return Truth.FALSE
        return Truth.UNKNOWN

    def solve(self,goal,env=None,depth=0,max_depth=8,trail=None):
        env=dict(env or {}); trail=list(trail or [])
        goal=self.substitute(goal,env)
        if depth>max_depth: return []
        solutions=[]

        for known in self.facts:
            e=self.unify(goal,known,env)
            if e is not None:
                solutions.append((e,trail+[f"Fakt: {self.substitute(goal,e)}"]))

        for idx,rule in enumerate(self.rules):
            suffix=f"@{depth}_{idx}_{len(trail)}"; rename={}
            def fresh(arg):
                if not self.is_var(arg): return arg
                if arg not in rename: rename[arg]=arg+suffix
                return rename[arg]
            conclusion=Fact(rule.conclusion.relation,tuple(fresh(a) for a in rule.conclusion.args))
            premises=[Fact(p.relation,tuple(fresh(a) for a in p.args)) for p in rule.premises]
            e0=self.unify(conclusion,goal,env)
            if e0 is None: continue
            partials=[(e0,trail+[f"Regel: {rule.name}"])]
            for premise in premises:
                nxt=[]
                for pe,ptrace in partials:
                    instantiated=self.substitute(premise,pe)
                    nxt.extend(self.solve(instantiated,pe,depth+1,max_depth,ptrace))
                partials=nxt
                if not partials: break
            solutions.extend(partials)
        return solutions

    def backward_truth(self,goal):
        if self.direct_truth(goal)==Truth.FALSE:
            return Truth.FALSE,[f"Widerspruch: {goal}"]
        sols=self.solve(goal)
        if sols:
            _,trace=sols[0]
            return Truth.TRUE,trace+[f"=> {goal} ist beweisbar"]
        return Truth.UNKNOWN,[f"=> {goal} ist nicht beweisbar und nicht widerlegt"]

    def choose(self,A,B):
        a,ta=self.backward_truth(A); b,tb=self.backward_truth(B)
        if a==Truth.TRUE and b!=Truth.TRUE: choice="A"
        elif b==Truth.TRUE and a!=Truth.TRUE: choice="B"
        else: choice="UNENTSCHIEDEN"
        return choice,a,b,ta,tb

if __name__=="__main__":
    m=SymbolicTernaryLM()
    m.ingest("""
    Paul ist größer als Anna.
    Anna ist größer als Tim.
    Der Hund jagt die Katze.
    """)

    A=Fact("GREATER",("paul","tim"))
    B=Fact("GREATER",("tim","paul"))
    choice,a,b,trace,_=m.choose(A,B)

    print("A:",A,a)
    print("B:",B,b)
    print("Antwort:",choice)
    print("Beweis:")
    for line in trace: print(" ",line)

    unknown=Fact("LIKE",("paul","tim"))
    print("Unbekannt-Test:",unknown,m.backward_truth(unknown)[0])

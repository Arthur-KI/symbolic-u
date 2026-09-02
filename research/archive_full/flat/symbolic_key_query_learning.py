
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, List, Dict, Set

class Truth(Enum):
    TRUE=1
    UNKNOWN=0
    FALSE=-1
    def __str__(self):
        return {Truth.TRUE:"WAHR",Truth.UNKNOWN:"UNBEKANNT",Truth.FALSE:"FALSCH"}[self]

@dataclass(frozen=True)
class Fact:
    relation: str
    args: Tuple[str,str]
    def __str__(self):
        return f"{self.relation}({self.args[0]}, {self.args[1]})"

@dataclass(frozen=True)
class RuleTemplate:
    premise1_rel: str
    premise2_rel: str
    conclusion_rel: str
    def __str__(self):
        return (
            f"{self.premise1_rel}(?a, ?b) & "
            f"{self.premise2_rel}(?b, ?c) -> "
            f"{self.conclusion_rel}(?a, ?c)"
        )

@dataclass
class LearnedConnection:
    rule: RuleTemplate
    support: int=0
    conflicts: int=0
    examples: List[str]=field(default_factory=list)
    @property
    def active(self):
        return self.support>=2 and self.conflicts==0

@dataclass
class TrainingExample:
    name: str
    facts: Set[Fact]
    option_a: Fact
    option_b: Fact
    label: str
    @property
    def chosen(self):
        return self.option_a if self.label=="A" else self.option_b
    @property
    def rejected(self):
        return self.option_b if self.label=="A" else self.option_a

class SymbolicKeyQueryNetwork:
    def __init__(self):
        self.connections={}

    def direct_truth(self,goal,facts):
        if goal in facts:
            return Truth.TRUE
        reverse=Fact(goal.relation,(goal.args[1],goal.args[0]))
        if reverse in facts:
            return Truth.FALSE
        return Truth.UNKNOWN

    def two_hop_paths(self,facts,goal):
        start,end=goal.args
        paths=[]
        for f1 in facts:
            if f1.args[0]!=start:
                continue
            mid=f1.args[1]
            for f2 in facts:
                if f2.args[0]==mid and f2.args[1]==end:
                    paths.append((f1,f2))
        return paths

    def candidate_from_path(self,path,goal):
        f1,f2=path
        return RuleTemplate(f1.relation,f2.relation,goal.relation)

    def rule_proves(self,rule,goal,facts):
        if goal.relation!=rule.conclusion_rel:
            return False
        start,end=goal.args
        for f1 in facts:
            if f1.relation!=rule.premise1_rel or f1.args[0]!=start:
                continue
            mid=f1.args[1]
            if Fact(rule.premise2_rel,(mid,end)) in facts:
                return True
        return False

    def train(self,examples):
        candidates=set()
        for ex in examples:
            if ex.chosen in ex.facts:
                continue
            for path in self.two_hop_paths(ex.facts,ex.chosen):
                candidates.add(self.candidate_from_path(path,ex.chosen))

        for rule in candidates:
            conn=LearnedConnection(rule)
            for ex in examples:
                chosen=self.rule_proves(rule,ex.chosen,ex.facts)
                rejected=self.rule_proves(rule,ex.rejected,ex.facts)
                if chosen and not rejected:
                    conn.support+=1
                    conn.examples.append(ex.name)
                if rejected:
                    conn.conflicts+=1
            self.connections[rule]=conn
        return self.connections

    def backward_prove(self,goal,facts,depth=0,max_depth=6,visited=None):
        visited=set(visited or set())
        indent="  "*depth

        direct=self.direct_truth(goal,facts)
        if direct==Truth.TRUE:
            return Truth.TRUE,[f"{indent}✓ Key vorhanden: {goal}"]
        if direct==Truth.FALSE:
            return Truth.FALSE,[f"{indent}✗ Gegen-Key widerspricht: {goal}"]
        if depth>=max_depth or goal in visited:
            return Truth.UNKNOWN,[f"{indent}? Abbruch/offen: {goal}"]

        visited.add(goal)
        traces=[]

        for conn in self.connections.values():
            if not conn.active or conn.rule.conclusion_rel!=goal.relation:
                continue

            rule=conn.rule
            start,end=goal.args
            traces.append(
                f"{indent}Query {goal}: versuche gelernte Verbindung "
                f"[Support={conn.support}] {rule}"
            )

            firsts=[
                f for f in facts
                if f.relation==rule.premise1_rel and f.args[0]==start
            ]

            if not firsts:
                traces.append(
                    f"{indent}  ? Frage 1 offen: "
                    f"{rule.premise1_rel}({start}, X)?"
                )
                continue

            for first in firsts:
                mid=first.args[1]
                traces.append(
                    f"{indent}  ✓ Frage 1: "
                    f"{rule.premise1_rel}({start}, X)? -> X={mid}"
                )

                second=Fact(rule.premise2_rel,(mid,end))
                traces.append(f"{indent}  ? Frage 2: {second}?")
                state,sub=self.backward_prove(
                    second,facts,depth+1,max_depth,visited.copy()
                )
                traces.extend(sub)

                if state==Truth.TRUE:
                    traces.append(
                        f"{indent}=> {goal} ist über gelernte Key→Query-Verbindung WAHR"
                    )
                    return Truth.TRUE,traces

        traces.append(f"{indent}? {goal} bleibt UNBEKANNT")
        return Truth.UNKNOWN,traces

    def answer_ab(self,facts,a,b):
        sa,ta=self.backward_prove(a,facts)
        sb,tb=self.backward_prove(b,facts)
        if sa==Truth.TRUE and sb!=Truth.TRUE:
            choice="A"
        elif sb==Truth.TRUE and sa!=Truth.TRUE:
            choice="B"
        else:
            choice="UNENTSCHIEDEN"
        return choice,sa,sb,ta,tb

def G(a,b):
    return Fact("GREATER",(a,b))

if __name__=="__main__":
    training=[
        TrainingExample(
            "train_1",
            {G("paul","anna"),G("anna","tim")},
            G("paul","tim"),G("tim","paul"),"A"
        ),
        TrainingExample(
            "train_2",
            {G("lea","max"),G("max","noah")},
            G("lea","noah"),G("noah","lea"),"A"
        ),
        TrainingExample(
            "train_3",
            {G("sara","tom"),G("tom","lina")},
            G("sara","lina"),G("lina","sara"),"A"
        ),
    ]

    net=SymbolicKeyQueryNetwork()
    net.train(training)

    print("Gelernt:")
    for c in net.connections.values():
        print(c.rule,"support=",c.support,"active=",c.active)

    facts={G("mira","jonas"),G("jonas","emil")}
    A=G("mira","emil")
    B=G("emil","mira")

    choice,sa,sb,trace,_=net.answer_ab(facts,A,B)
    print("\nTest:",choice,sa,sb)
    for line in trace:
        print(line)

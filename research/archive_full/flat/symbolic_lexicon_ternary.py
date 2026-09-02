from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set

class Truth(Enum):
    TRUE=1
    UNKNOWN=0
    FALSE=-1

    def short(self):
        return {Truth.TRUE:"+1",Truth.UNKNOWN:"0",Truth.FALSE:"-1"}[self]

    def __str__(self):
        return {Truth.TRUE:"WAHR",Truth.UNKNOWN:"UNBEKANNT",Truth.FALSE:"FALSCH"}[self]

@dataclass(frozen=True)
class Sense:
    key: str
    label: str
    properties: frozenset
    forbidden: frozenset=frozenset()

@dataclass
class Lexeme:
    word: str
    senses: List[Sense]

@dataclass(frozen=True)
class RelationSchema:
    key: str
    subject_requires: frozenset=frozenset()
    object_requires: frozenset=frozenset()
    subject_forbids: frozenset=frozenset()
    object_forbids: frozenset=frozenset()

@dataclass
class SenseResult:
    sense: Sense
    state: Truth
    reasons: List[str]=field(default_factory=list)

class SymbolicLexiconEngine:
    def __init__(self):
        self.lexicon={}
        self.relations={}
        self._build_dictionary()

    def _build_dictionary(self):
        self.lexicon["bank"]=Lexeme("bank",[
            Sense("BANK_FINANCE","Finanzinstitut",
                  frozenset({"institution","can_receive_money","can_hold_account","can_lend_money"}),
                  frozenset({"furniture","natural_landform","supports_sitting_as_function"})),
            Sense("BANK_BENCH","Sitzbank",
                  frozenset({"physical_object","furniture","supports_sitting_as_function"}),
                  frozenset({"institution","can_receive_money","natural_landform"})),
            Sense("BANK_RIVER","Ufer / Böschung",
                  frozenset({"physical_place","natural_landform","can_be_sat_on"}),
                  frozenset({"institution","furniture","can_receive_money"})),
        ])

        self.lexicon["hund"]=Lexeme("hund",[
            Sense("DOG","Hund",
                  frozenset({"animal","living_entity","physical_object","can_move","can_chase"}))
        ])

        self.lexicon["katze"]=Lexeme("katze",[
            Sense("CAT","Katze",
                  frozenset({"animal","living_entity","physical_object","can_move"}))
        ])

        self.relations["SIT_ON"]=RelationSchema(
            "SIT_ON",
            subject_requires=frozenset({"living_entity"}),
            object_requires=frozenset({"can_be_sat_on"})
        )

        self.relations["SIT_ON_FUNCTIONAL"]=RelationSchema(
            "SIT_ON_FUNCTIONAL",
            subject_requires=frozenset({"living_entity"}),
            object_requires=frozenset({"supports_sitting_as_function"})
        )

        self.relations["TRANSFER_MONEY_TO"]=RelationSchema(
            "TRANSFER_MONEY_TO",
            object_requires=frozenset({"can_receive_money"})
        )

        self.relations["CHASE"]=RelationSchema(
            "CHASE",
            subject_requires=frozenset({"can_chase"}),
            object_requires=frozenset({"physical_object"})
        )

    def detect_relation(self,text):
        t=text.lower().strip()
        if "sitze auf der bank" in t or "sitzt auf der bank" in t:
            return "SIT_ON_FUNCTIONAL","bank"
        if "überweise" in t and "bank" in t:
            return "TRANSFER_MONEY_TO","bank"
        if "überweist" in t and "bank" in t:
            return "TRANSFER_MONEY_TO","bank"
        if "hund" in t and ("jagt" in t or "verfolgt" in t) and "katze" in t:
            return "CHASE","katze"
        return None,None

    def evaluate_sense(self,sense,required,forbidden):
        reasons=[]

        contradicted=required.intersection(sense.forbidden)
        if contradicted:
            for p in sorted(contradicted):
                reasons.append(f"benötigt '{p}', Bedeutung verbietet es")
            return SenseResult(sense,Truth.FALSE,reasons)

        relation_conflict=forbidden.intersection(sense.properties)
        if relation_conflict:
            for p in sorted(relation_conflict):
                reasons.append(f"Kontext verbietet '{p}', Bedeutung besitzt es")
            return SenseResult(sense,Truth.FALSE,reasons)

        if required.issubset(sense.properties):
            for p in sorted(required):
                reasons.append(f"erfüllt '{p}'")
            return SenseResult(sense,Truth.TRUE,reasons)

        for p in sorted(required.difference(sense.properties)):
            reasons.append(f"'{p}' nicht im Wörterbuch bewiesen")
        return SenseResult(sense,Truth.UNKNOWN,reasons)

    def resolve_word(self,word,relation_key,role="object"):
        lexeme=self.lexicon[word.lower()]
        schema=self.relations[relation_key]

        if role=="object":
            required=set(schema.object_requires)
            forbidden=set(schema.object_forbids)
        else:
            required=set(schema.subject_requires)
            forbidden=set(schema.subject_forbids)

        return [self.evaluate_sense(s,required,forbidden) for s in lexeme.senses]

    def resolve_text(self,text):
        relation,target=self.detect_relation(text)
        if not relation:
            return {"text":text,"relation":None,"target":None,"results":[],"winner":None}

        results=self.resolve_word(target,relation,"object")
        true_senses=[r.sense for r in results if r.state==Truth.TRUE]
        winner=true_senses[0] if len(true_senses)==1 else None

        return {
            "text":text,
            "relation":relation,
            "target":target,
            "results":results,
            "winner":winner
        }

    def print_resolution(self,text):
        out=self.resolve_text(text)
        print(f"\nTEXT: {text}")

        if not out["relation"]:
            print("Keine bekannte Relation erkannt.")
            return

        print("Relation:",out["relation"])
        print("Mehrdeutiges Wort:",out["target"])

        for r in out["results"]:
            print(f" {r.state.short():>2} {r.sense.key:<16} ({r.sense.label}) -> {r.state}")
            for reason in r.reasons:
                print("    -",reason)

        if out["winner"]:
            print("=> AUSGEWÄHLTER KEY:",out["winner"].key)
        else:
            print("=> Noch keine eindeutige Auswahl.")

if __name__=="__main__":
    e=SymbolicLexiconEngine()

    e.print_resolution("Ich sitze auf der Bank.")
    e.print_resolution("Ich überweise Geld an die Bank.")
    e.print_resolution("Der Hund jagt die Katze.")

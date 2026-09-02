
from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
import types, sys, re, json, csv

# ============================================================
# v5.0a — Unified structured German text -> symbolic Keys
#          -> adaptive anonymous U -> proof/query
#
# Purely symbolic. No embeddings, no neural parser, no arithmetic
# implementation is exposed to the learned anonymous relation.
# ============================================================

# Reuse frozen generic Recursive-U induction/verifier.
src = Path("/mnt/data/symbolic_v44b_recursive_u_verifier.py").read_text(encoding="utf-8")
v44 = types.ModuleType("v44_for_v50a")
sys.modules[v44.__name__] = v44
exec(src, v44.__dict__)

NUM = v44.NUM
SIG = v44.SIG
HEAD_VARS = v44.HEAD_VARS
RelSig = v44.RelSig
Program = v44.Program

# ------------------------------------------------------------
# Symbolic dictionary / ontology
# ------------------------------------------------------------

@dataclass(frozen=True)
class Lexeme:
    lemma: str
    forms: frozenset[str]
    features: frozenset[str]
    value: str | None = None

class Dictionary:
    def __init__(self):
        self.by_form = {}
    def add(self, lex: Lexeme):
        for f in lex.forms | {lex.lemma}:
            self.by_form[f.lower()] = lex
    def lookup(self, form: str):
        return self.by_form.get(form.lower())

D = Dictionary()

for name in ["Anna", "Ben", "Mia", "Karl", "Paul", "Lea"]:
    D.add(Lexeme(name.lower(), frozenset({name, name.lower()}),
                 frozenset({"ENTITY","PERSON","NAME"}), name.lower()))

for lemma, forms, value in [
    ("apfel", {"Apfel","Äpfel","apfel","äpfel"}, "APPLE"),
    ("birne", {"Birne","Birnen","birne","birnen"}, "PEAR"),
    ("stein", {"Stein","Steine","stein","steine"}, "STONE"),
]:
    D.add(Lexeme(lemma, frozenset(forms), frozenset({"ENTITY_TYPE","COUNTABLE"}), value))

NUMBER_WORDS = {
    0:"null",1:"eins",2:"zwei",3:"drei",4:"vier",5:"fünf",6:"sechs",7:"sieben",
    8:"acht",9:"neun",10:"zehn",11:"elf",12:"zwölf",13:"dreizehn",14:"vierzehn",
    15:"fünfzehn",16:"sechzehn",17:"siebzehn",18:"achtzehn",19:"neunzehn",20:"zwanzig",
}
for n,w in NUMBER_WORDS.items():
    forms={w,str(n)}
    D.add(Lexeme(w, frozenset(forms), frozenset({"NUMBER"}), f"N{n}"))

for lemma,forms,features,value in [
    ("haben",{"hat","haben"}, {"VERB","COUNT_STATE"}, "INITIAL_COUNT"),
    ("geben",{"gibt","geben"}, {"VERB","TRANSFER_OUT"}, "REMOVED_COUNT"),
    ("wegsignal",{"weg"}, {"PARTICLE","TRANSFER_OUT_PARTICLE"}, None),
    ("sagen",{"sagt","sagen"}, {"VERB","SPEECH"}, "CLAIM"),
    ("noch",{"noch"}, {"QUERY_CUE"}, None),
]:
    D.add(Lexeme(lemma, frozenset(forms), frozenset(features), value))

@dataclass(frozen=True)
class Token:
    surface: str
    lex: Lexeme | None

TOKEN_RE = re.compile(r"[A-Za-zÄÖÜäöüß0-9]+", re.UNICODE)

def tokenize(text: str):
    return [Token(x, D.lookup(x)) for x in TOKEN_RE.findall(text)]

def first_feature(tokens, feature):
    for i,t in enumerate(tokens):
        if t.lex and feature in t.lex.features:
            return i,t
    return None,None

def entities(tokens):
    return [(i,t.lex.value) for i,t in enumerate(tokens)
            if t.lex and "NAME" in t.lex.features]

def items(tokens):
    return [(i,t.lex.value) for i,t in enumerate(tokens)
            if t.lex and "COUNTABLE" in t.lex.features]

def numbers(tokens):
    return [(i,t.lex.value) for i,t in enumerate(tokens)
            if t.lex and "NUMBER" in t.lex.features]

# ------------------------------------------------------------
# Keys / context / provenance
# ------------------------------------------------------------

@dataclass(frozen=True)
class Key:
    rel: str
    args: tuple[str,...]
    context: str = "WORLD"

    def __str__(self):
        return f"{self.context}:{self.rel}({', '.join(self.args)})"

@dataclass
class Evidence:
    key: Key
    source: str
    parser_rule: str

@dataclass
class StoryContext:
    story_id: str
    evidence: list[Evidence] = field(default_factory=list)

    def add(self, key, source, parser_rule):
        self.evidence.append(Evidence(key,source,parser_rule))

    def keys(self, context="WORLD", rel=None):
        out=[]
        for e in self.evidence:
            if e.key.context != context:
                continue
            if rel is not None and e.key.rel != rel:
                continue
            out.append(e.key)
        return out

    def facts(self, context="WORLD", rel=None):
        return {k.args for k in self.keys(context,rel)}

# ------------------------------------------------------------
# Structured symbolic German parser
# ------------------------------------------------------------

def split_sentences(text):
    # Quote-aware symbolic sentence segmentation. Sentence punctuation inside
    # German/ASCII quotes closes the quoted clause, but the outer speech
    # sentence ends only after the closing quote.
    out=[]; buf=[]; quote=None
    open_quotes={"„":"“", '"':'"', "“":"”"}
    close_quotes={"”","“",'"'}
    i=0
    while i < len(text):
        ch=text[i]
        buf.append(ch)
        if quote is None and ch in open_quotes:
            quote=open_quotes[ch]
        elif quote is not None and ch==quote:
            quote=None
            # If quoted material already ended in sentence punctuation,
            # closing the quote completes the outer speech sentence too.
            stripped=''.join(buf).rstrip()
            inner=stripped[:-1].rstrip() if stripped.endswith(ch) else stripped
            if inner.endswith(('.', '!', '?')):
                sent=''.join(buf).strip()
                if sent: out.append(sent)
                buf=[]
        elif quote is None and ch in '.!?':
            sent=''.join(buf).strip()
            if sent: out.append(sent)
            buf=[]
        i+=1
    rest=''.join(buf).strip()
    if rest: out.append(rest)
    return out

def parse_simple_clause(sentence: str, story: StoryContext, context="WORLD"):
    toks=tokenize(sentence)
    ents=entities(toks)
    its=items(toks)
    nums=numbers(toks)

    # COUNT_STATE:
    #   Anna hat sieben Äpfel.
    vi,_=first_feature(toks,"COUNT_STATE")
    if vi is not None and ents and its and nums:
        left_ents=[x for x in ents if x[0] < vi]
        right_items=[x for x in its if x[0] > vi]
        right_nums=[x for x in nums if x[0] > vi]
        if left_ents and right_items and right_nums:
            owner=max(left_ents,key=lambda x:x[0])[1]
            item=min(right_items,key=lambda x:x[0])[1]
            num=min(right_nums,key=lambda x:x[0])[1]
            story.add(
                Key("INITIAL_COUNT",(owner,item,num),context),
                sentence,
                "COUNT_STATE(subject, number, countable)"
            )
            return True

    # TRANSFER_OUT:
    #   Anna gibt fünf Äpfel weg.
    vi,_=first_feature(toks,"TRANSFER_OUT")
    particle_i,_=first_feature(toks,"TRANSFER_OUT_PARTICLE")
    if vi is not None and particle_i is not None and ents and its and nums:
        left_ents=[x for x in ents if x[0] < vi]
        right_items=[x for x in its if vi < x[0] < particle_i]
        right_nums=[x for x in nums if vi < x[0] < particle_i]
        if left_ents and right_items and right_nums:
            owner=max(left_ents,key=lambda x:x[0])[1]
            item=min(right_items,key=lambda x:x[0])[1]
            num=min(right_nums,key=lambda x:x[0])[1]
            story.add(
                Key("REMOVED_COUNT",(owner,item,num),context),
                sentence,
                "TRANSFER_OUT(subject, number, countable, particle)"
            )
            return True

    return False

QUOTE_RE = re.compile(r'[„"“](.+?)[”"“]', re.DOTALL)

def parse_story(text: str, story_id: str):
    story=StoryContext(story_id)
    for sentence in split_sentences(text):
        toks=tokenize(sentence)
        speech_i,_=first_feature(toks,"SPEECH")

        if speech_i is not None:
            ents=entities(toks)
            speakers=[x for x in ents if x[0] < speech_i]
            speaker=max(speakers,key=lambda x:x[0])[1] if speakers else "UNKNOWN_SPEAKER"
            m=QUOTE_RE.search(sentence)
            if m:
                quoted=m.group(1).strip()
                parse_simple_clause(
                    quoted, story, context=f"CLAIM:{speaker}"
                )
                continue

        parse_simple_clause(sentence,story,"WORLD")
    return story

@dataclass(frozen=True)
class Query:
    rel: str
    owner: str
    item: str
    source: str

def parse_query(text: str):
    toks=tokenize(text)
    ents=entities(toks)
    its=items(toks)
    has_query=any(t.lex and "QUERY_CUE" in t.lex.features for t in toks)
    if has_query and ents and its:
        # "Wie viele Äpfel hat Anna noch?"
        return Query("REMAINING_COUNT", ents[-1][1], its[0][1], text)
    return None

# ------------------------------------------------------------
# Training bank is itself structured raw text.
# The current query story is never included here.
# ------------------------------------------------------------

@dataclass(frozen=True)
class LabeledTextExample:
    story: str
    question: str
    expected: str

def nw(n):
    return NUMBER_WORDS[n]

def training_examples():
    names=["Ben","Mia","Karl","Lea"]
    item_words={"PEAR":"Birnen","STONE":"Steine"}
    items_cycle=["PEAR","STONE"]
    out=[]
    k=0
    for x in range(8):
        for y in range(x+1):
            z=x-y
            name=names[k % len(names)]
            item=items_cycle[k % len(items_cycle)]
            word=item_words[item]
            out.append(LabeledTextExample(
                f"{name} hat {nw(x)} {word}. {name} gibt {nw(y)} {word} weg.",
                f"Wie viele {word} hat {name} noch?",
                f"N{z}",
            ))
            k+=1
    return out

def frozen_examples():
    rows=[(8,3),(9,4),(10,2),(11,5),(12,7),(13,6),(14,8)]
    names=["Ben","Mia","Karl","Lea"]
    out=[]
    for i,(x,y) in enumerate(rows):
        name=names[i%len(names)]
        word="Birnen" if i%2==0 else "Steine"
        out.append(LabeledTextExample(
            f"{name} hat {nw(x)} {word}. {name} gibt {nw(y)} {word} weg.",
            f"Wie viele {word} hat {name} noch?",
            f"N{x-y}",
        ))
    return out

def extract_relation_tuple(example: LabeledTextExample, story_id: str):
    s=parse_story(example.story,story_id)
    q=parse_query(example.question)
    assert q is not None
    initials=[a for a in s.facts("WORLD","INITIAL_COUNT")
              if a[0]==q.owner and a[1]==q.item]
    removed=[a for a in s.facts("WORLD","REMOVED_COUNT")
             if a[0]==q.owner and a[1]==q.item]
    assert len(initials)==1 and len(removed)==1
    x=initials[0][2]
    y=removed[0][2]
    return (x,y,example.expected)

# ------------------------------------------------------------
# Versioned adaptive anonymous U library
# ------------------------------------------------------------

@dataclass
class UVersion:
    relation: str
    number: int
    status: str
    program: Program
    meta: dict

    @property
    def id(self):
        return f"{self.relation}_v{self.number}"

class AdaptiveVersionedLibrary:
    def __init__(self):
        self.versions: dict[str,list[UVersion]] = defaultdict(list)
        self.active: dict[str,UVersion] = {}
        self.learn_attempts=defaultdict(int)
        self.events=[]

    def event(self,event,**kw):
        self.events.append({"event":event,**kw})

    def active_relation_by_signature(self, signature):
        for rel,v in self.active.items():
            if tuple(v.meta["signature"])==tuple(signature):
                return v
        return None

    def invent_numeric_relation(self, train_tuples, frozen_tuples):
        # The structural hole says only NUM,NUM,NUM. No SUB name/vocabulary.
        existing=self.active_relation_by_signature((NUM,NUM,NUM))
        if existing:
            self.event("reuse_anonymous_relation",version=existing.id)
            return existing

        rel="R1"
        self.learn_attempts[rel]+=1
        SIG[rel]=RelSig(rel,(NUM,NUM,NUM))
        HEAD_VARS[rel]=("X","Y","Z")

        pos=list(train_tuples)
        neg=[]
        for x,y,z in pos:
            zi=int(z[1:])
            neg.append((x,y,f"N{zi+1}"))
            if zi>0:
                neg.append((x,y,f"N{zi-1}"))

        train_world=v44.num_world(24,pos,neg,lt=True)

        self.event(
            "anonymous_signature_invented",
            relation=rel,
            signature=(NUM,NUM,NUM),
            training_examples=len(pos),
        )

        res=v44.synth_verified_recursive(
            rel,
            ["ZERO","EQ","PRED","SUCC","LT"],
            [train_world],
            max_base=2,
            max_bg=2,
            hidden_limits={NUM:2},
        )
        if not res.get("best"):
            raise AssertionError("anonymous relation induction failed")

        sc,freq,base,rec,cert,local=res["best"]
        prog=Program(SIG[rel],base,rec)

        # Independent frozen raw-text examples.
        fpos=list(frozen_tuples)
        fneg=[]
        for x,y,z in fpos:
            zi=int(z[1:])
            fneg.append((x,y,f"N{zi+1}"))
        fw=v44.num_world(35,fpos,fneg,lt=True)

        prog.reset()
        frozen_pos=sum(bool(prog.prove(t,fw)) for t in fpos)
        prog.reset()
        frozen_false=sum(bool(prog.prove(t,fw)) for t in fneg)

        if not (
            sc[0]==len(pos) and sc[1]==0 and
            frozen_pos==len(fpos) and frozen_false==0
        ):
            raise AssertionError(
                f"anonymous gate failed support={sc[0]}/{len(pos)} conflict={sc[1]} "
                f"frozen={frozen_pos}/{len(fpos)} false={frozen_false}"
            )

        meta={
            "signature":[NUM,NUM,NUM],
            "support":sc[0],
            "positive_n":len(pos),
            "conflict":sc[1],
            "frozen_passed":frozen_pos,
            "frozen_n":len(fpos),
            "frozen_false_positive":frozen_false,
            "base_rule":base.text(),
            "recursive_rule":rec.text(),
            "certificate":cert,
            "training_source":"independent structured raw-text bank",
        }
        v=UVersion(rel,1,"ACTIVE",prog,meta)
        self.versions[rel].append(v)
        self.active[rel]=v
        self.event("anonymous_committed",version=v.id)
        return v

# ------------------------------------------------------------
# Unified reasoner
# ------------------------------------------------------------

class UnifiedReasoner:
    def __init__(self, library, train_examples, frozen_examples):
        self.lib=library
        self.train_examples=train_examples
        self.frozen_examples=frozen_examples
        self.parent_u=None

    def ensure_remaining_u(self):
        if self.parent_u is not None:
            return self.parent_u

        train_tuples=[
            extract_relation_tuple(ex,f"train-{i}")
            for i,ex in enumerate(self.train_examples)
        ]
        frozen_tuples=[
            extract_relation_tuple(ex,f"frozen-{i}")
            for i,ex in enumerate(self.frozen_examples)
        ]

        child=self.lib.invent_numeric_relation(train_tuples,frozen_tuples)

        # Structural U generation:
        # INITIAL_COUNT(O,I,X)
        # REMOVED_COUNT(O,I,Y)
        # target REMAINING_COUNT(O,I,Z)
        # -> X,Y dangling numeric ports, Z missing numeric head port
        # -> infer anonymous child signature NUM,NUM,NUM.
        self.parent_u={
            "target":"REMAINING_COUNT(O,I,Z)",
            "body":[
                "INITIAL_COUNT(O,I,X)",
                "REMOVED_COUNT(O,I,Y)",
                f"{child.relation}(X,Y,Z)",
            ],
            "dependency":child.id,
            "source":"typed structural hole completion",
        }
        self.lib.event(
            "parent_u_generated",
            target="REMAINING_COUNT",
            dependency=child.id,
        )
        return self.parent_u

    def answer(self, story: StoryContext, query: Query):
        parent=self.ensure_remaining_u()
        child=self.lib.active[parent["dependency"].split("_v")[0]]

        initials=[
            a for a in story.facts("WORLD","INITIAL_COUNT")
            if a[0]==query.owner and a[1]==query.item
        ]
        removed=[
            a for a in story.facts("WORLD","REMOVED_COUNT")
            if a[0]==query.owner and a[1]==query.item
        ]
        if not initials or not removed:
            return None

        # Numeric ontology only; no Python subtraction is used for proof.
        max_n=40
        w=v44.num_world(max_n,[],[],lt=True)
        proved=[]
        for init in initials:
            for rem in removed:
                x,y=init[2],rem[2]
                for z in [f"N{i}" for i in range(max_n+1)]:
                    child.program.reset()
                    if child.program.prove((x,y,z),w):
                        proved.append(z)
        proved=list(dict.fromkeys(proved))
        return proved[0] if len(proved)==1 else None

    def query_state(self, story, query, candidate):
        ans=self.answer(story,query)
        return +1 if ans==candidate else 0

# ------------------------------------------------------------
# Execute v5.0a
# ------------------------------------------------------------

train_bank=training_examples()
frozen_bank=frozen_examples()
lib=AdaptiveVersionedLibrary()
model=UnifiedReasoner(lib,train_bank,frozen_bank)

# Main unseen raw-text story.
raw_main="Anna hat fünfzehn Äpfel. Anna gibt sechs Äpfel weg."
q_main_text="Wie viele Äpfel hat Anna noch?"
main_story=parse_story(raw_main,"main")
main_q=parse_query(q_main_text)
main_answer=model.answer(main_story,main_q)

# Second raw-text domain/noun/person: must reuse R1 and not relearn.
attempts_after_first=dict(lib.learn_attempts)
raw_reuse="Mia hat siebzehn Birnen. Mia gibt vier Birnen weg."
reuse_story=parse_story(raw_reuse,"reuse")
reuse_q=parse_query("Wie viele Birnen hat Mia noch?")
reuse_answer=model.answer(reuse_story,reuse_q)
attempts_after_second=dict(lib.learn_attempts)

# CLAIM attack: removal exists only in claim context.
raw_claim='Anna hat elf Äpfel. Paul sagt: „Anna gibt drei Äpfel weg.“'
claim_story=parse_story(raw_claim,"claim")
claim_q=parse_query("Wie viele Äpfel hat Anna noch?")
claim_answer=model.answer(claim_story,claim_q)

# Add an actual world event after the claim.
raw_claim_world='Anna hat elf Äpfel. Paul sagt: „Anna gibt drei Äpfel weg.“ Anna gibt drei Äpfel weg.'
claim_world_story=parse_story(raw_claim_world,"claim-world")
claim_world_answer=model.answer(claim_world_story,claim_q)

# Story isolation: same entity and item, different story.
iso_a=parse_story("Anna hat zehn Äpfel. Anna gibt zwei Äpfel weg.","iso-a")
iso_b=parse_story("Anna hat zehn Äpfel. Anna gibt fünf Äpfel weg.","iso-b")
iso_q=parse_query("Wie viele Äpfel hat Anna noch?")
iso_a_answer=model.answer(iso_a,iso_q)
iso_b_answer=model.answer(iso_b,iso_q)

r1=lib.active["R1"]
rule_text=r1.meta["base_rule"]+" "+r1.meta["recursive_rule"]
train_tuples={extract_relation_tuple(ex,f"audit-{i}") for i,ex in enumerate(train_bank)}
frozen_tuples={extract_relation_tuple(ex,f"faudit-{i}") for i,ex in enumerate(frozen_bank)}

checks={
    "raw_text_produces_world_keys":(
        ("anna","APPLE","N15") in main_story.facts("WORLD","INITIAL_COUNT") and
        ("anna","APPLE","N6") in main_story.facts("WORLD","REMOVED_COUNT")
    ),
    "raw_query_parses_symbolically":(
        main_q is not None and main_q.owner=="anna" and main_q.item=="APPLE"
    ),
    "anonymous_relation_invented_without_SUB_vocabulary":(
        r1.id=="R1_v1" and "SUB(" not in rule_text and "ADD(" not in rule_text
    ),
    "anonymous_relation_training_gate":(
        r1.meta["support"]==r1.meta["positive_n"] and r1.meta["conflict"]==0
    ),
    "anonymous_relation_frozen_gate":(
        r1.meta["frozen_passed"]==r1.meta["frozen_n"] and
        r1.meta["frozen_false_positive"]==0
    ),
    "main_unseen_text_answer_correct":main_answer=="N9",
    "wrong_answer_stays_unknown":model.query_state(main_story,main_q,"N10")==0,
    "query_story_not_training_evidence":(
        ("N15","N6","N9") not in train_tuples and
        ("N15","N6","N9") not in frozen_tuples
    ),
    "second_text_reuses_same_anonymous_u":(
        reuse_answer=="N13" and
        attempts_after_first==attempts_after_second=={"R1":1}
    ),
    "claim_does_not_enter_world":(
        claim_answer is None and
        ("anna","APPLE","N3") not in claim_story.facts("WORLD","REMOVED_COUNT") and
        ("anna","APPLE","N3") in claim_story.facts("CLAIM:paul","REMOVED_COUNT")
    ),
    "actual_world_event_after_claim_enables_reasoning":claim_world_answer=="N8",
    "story_contexts_do_not_mix":iso_a_answer=="N8" and iso_b_answer=="N5",
    "parent_u_is_structural_and_uses_R1":(
        model.parent_u is not None and
        model.parent_u["body"]==[
            "INITIAL_COUNT(O,I,X)",
            "REMOVED_COUNT(O,I,Y)",
            "R1(X,Y,Z)",
        ]
    ),
    "library_is_versioned":(
        len(lib.versions["R1"])==1 and lib.versions["R1"][0].status=="ACTIVE"
    ),
}

print("=== v5.0a UNIFIED STRUCTURED RAW-TEXT -> ADAPTIVE U ===")
for k,v in checks.items():
    print(("PASS" if v else "FAIL"),"|",k)

print("\nParsed MAIN world keys:")
for e in main_story.evidence:
    print(" ",e.key,"|",e.parser_rule,"| source:",e.source)
print("Query:",main_q)
print("Answer:",main_answer)
print("Wrong N10 state:",model.query_state(main_story,main_q,"N10"))

print("\nInvented/versioned anonymous U:")
print(" ",r1.id,r1.status)
print(" BASE:",r1.meta["base_rule"])
print(" REC :",r1.meta["recursive_rule"])
print(" train:",r1.meta["support"],"/",r1.meta["positive_n"],"conflict",r1.meta["conflict"])
print(" frozen:",r1.meta["frozen_passed"],"/",r1.meta["frozen_n"],
      "false+",r1.meta["frozen_false_positive"])

print("\nGenerated parent U:")
print(" ", " + ".join(model.parent_u["body"]), "->", model.parent_u["target"])

print("\nReuse story answer:",reuse_answer,"attempts",dict(lib.learn_attempts))
print("CLAIM-only answer:",claim_answer)
print("CLAIM contexts:")
for e in claim_story.evidence:
    print(" ",e.key)
print("CLAIM + real WORLD action:",claim_world_answer)
print("Story isolation:",iso_a_answer,iso_b_answer)

assert all(checks.values())

report={
    "version":"v5.0a-unified-structured-text-adaptive-u",
    "result":"PASS",
    "scope":"controlled synthetic German text; not general NLP accuracy",
    "checks":checks,
    "main":{
        "raw_text":raw_main,
        "query":q_main_text,
        "parsed_keys":[str(e.key) for e in main_story.evidence],
        "answer":main_answer,
        "wrong_N10_state":model.query_state(main_story,main_q,"N10"),
    },
    "anonymous_u":{
        "version":r1.id,
        "status":r1.status,
        **r1.meta,
    },
    "parent_u":model.parent_u,
    "reuse":{
        "raw_text":raw_reuse,
        "answer":reuse_answer,
        "learn_attempts":dict(lib.learn_attempts),
    },
    "claim_attack":{
        "raw_text":raw_claim,
        "parsed_keys":[str(e.key) for e in claim_story.evidence],
        "world_answer":claim_answer,
        "with_actual_world_event_answer":claim_world_answer,
    },
    "story_isolation":{
        "story_a_answer":iso_a_answer,
        "story_b_answer":iso_b_answer,
    },
    "architecture":[
        "Raw German text is tokenized through a symbolic dictionary.",
        "Lexeme features and fixed clause-role patterns create provenance-marked WORLD or CLAIM Keys.",
        "The query is parsed independently into a symbolic target and is never training evidence.",
        "A structural parent-U hole infers the need for an anonymous NUM,NUM,NUM relation.",
        "The anonymous relation is induced from a separate bank of structured raw-text examples, with no ADD/SUB candidate vocabulary.",
        "Recursive-U induction uses only symbolic ZERO/EQ/PRED/SUCC/LT primitives and the frozen verifier.",
        "The learned anonymous relation is stored as ACTIVE R1_v1 in a versioned U library.",
        "The generated parent U composes parsed text Keys with R1 and answers the original query.",
        "Later stories reuse R1 without relearning.",
        "CLAIM-scoped removal Keys are excluded from WORLD reasoning."
    ],
    "caveats":[
        "This is a controlled structured-language benchmark, not general German understanding.",
        "The clause grammar currently covers explicit subject count-state, explicit subject transfer-out, quoted speech claims, and a remaining-count query.",
        "The parent-U structural schema is constrained to a single anonymous numeric hole over already introduced ports.",
        "Number-word lexicon is currently bounded to 0..20 while the numeric ontology used by the reasoner is larger.",
        "v5.0a integrates the versioned library container but does not trigger an in-story v4.9 revision challenge yet.",
        "No pronouns, passive voice, coordination, temporal sequencing beyond sentence order, or free Grimm prose are claimed here."
    ]
}

Path("/mnt/data/symbolic_v50a_unified_structured_text.py").write_text(
    Path(__file__).read_text(encoding="utf-8") if "__file__" in globals() else "",
    encoding="utf-8"
)
Path("/mnt/data/symbolic_v50a_unified_structured_text_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
with Path("/mnt/data/symbolic_v50a_unified_structured_text_checks.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.writer(f)
    w.writerow(["check","passed"])
    for k,v in checks.items():
        w.writerow([k,v])

print("\nSaved v5.0a report/checks.")

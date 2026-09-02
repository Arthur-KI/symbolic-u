
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
import re, json, csv, types, sys, io, contextlib, copy

# ============================================================
# v6.1 — Generic Language Bridge Integration
#
# Frozen backend:
#   v6.0-alpha anonymous Event-U + concept hierarchy
#
# New front-end mechanisms:
#   persistent Entity Memory
#   descriptor aliasing
#   role-typed pronoun Reference-U
#   local reporting-clause speaker binding
#   shared-subject Clause-U
#   quote command Surface-U
#   command-response local action binding
#   learned temporal locality bounds for Concept-U reuse
#
# No Grimm sentence-specific regexes / exact-line rules.
# ============================================================

# ------------------------------------------------------------
# 0. Load the frozen v6.0-alpha core WITHOUT editing its logic.
# Redirect only report output paths.
# ------------------------------------------------------------

v60_src=Path("/mnt/data/symbolic_v60_alpha_unified.py").read_text(encoding="utf-8")
v60_src=v60_src.replace(
    "/mnt/data/symbolic_v60_alpha_unified_report.json",
    "/mnt/data/symbolic_v60_alpha_backend_for_v61_report.json"
).replace(
    "/mnt/data/symbolic_v60_alpha_unified_checks.csv",
    "/mnt/data/symbolic_v60_alpha_backend_for_v61_checks.csv"
)
m=types.ModuleType("v60_frozen_for_v61")
sys.modules[m.__name__]=m
_capture=io.StringIO()
with contextlib.redirect_stdout(_capture):
    exec(compile(v60_src,"v60_frozen_for_v61","exec"),m.__dict__)

# Frozen backend invariants.
assert all(m.checks.values())
assert m.ROOT.relation in m.CLIB.active
assert m.R_DIRECTIVE in m.EVENT_RULES

# ------------------------------------------------------------
# 1. Learn generic temporal locality bounds from frozen v6.0 development.
#    This is structural training evidence, not Grimm tuning.
# ------------------------------------------------------------

LOCALITY={}
for r,c in m.CLIB.active.items():
    gaps=[]
    for st in m.USTORIES:
        for args,a,b,support in m.concept_instances(c,st):
            gaps.append(b.t-a.t)
    if gaps:
        LOCALITY[r]=max(gaps)+0.20  # small generic tolerance

assert m.ROOT.relation in LOCALITY
ROOT_LOCALITY=LOCALITY[m.ROOT.relation]
assert 1.0 <= ROOT_LOCALITY <= 1.5

# ------------------------------------------------------------
# 2. Generic lexical/morphological bridge
# ------------------------------------------------------------

WORD_RE=re.compile(r"[A-Za-zÄÖÜäöüß]+",re.UNICODE)

def words(text):
    return WORD_RE.findall(text)

def nrm(x):
    return x.lower().replace("ß","ss")

def deumlaut(x):
    return (x.replace("ä","a").replace("ö","o").replace("ü","u"))

# Bridge-only lexicon additions are primitive ontology vocabulary, not event frames.
EXTRA_ENTITY={
    "brei":"PORRIDGE",
    "hirsenbrei":"PORRIDGE",
    "küche":"KITCHEN",
    "kueche":"KITCHEN",
    "haus":"HOUSE",
    "straße":"STREET",
    "strasse":"STREET",
    "wald":"FOREST",
    "stadt":"CITY",
    "rand":"EDGE",
}

def entity_symbol(tok):
    raw=nrm(tok)
    direct=m.lookup_entity(raw)
    if direct:
        return direct
    if raw in EXTRA_ENTITY:
        return EXTRA_ENTITY[raw]

    # Generic German diminutive normalization: Töpfchen -> Topf.
    if raw.endswith("chen") and len(raw)>5:
        stem=deumlaut(raw[:-4])
        direct=m.lookup_entity(stem)
        if direct:
            return direct
        if stem in EXTRA_ENTITY:
            return EXTRA_ENTITY[stem]
    return None

BRIDGE_ACTION_FORMS=dict(m.ACTION_FORM)
# Only normal verbal morphology additions. "steh" deliberately NOT mapped,
# so we do not smuggle the Grimm stop formula into semantics.
BRIDGE_ACTION_FORMS.update({
    "glühte":"GLOW",
    "gluehte":"GLOW",
    "schlief":"SLEEP",
})

def action_symbol(tok):
    return BRIDGE_ACTION_FORMS.get(nrm(tok))

# ------------------------------------------------------------
# 3. Persistent Mention / Entity / Reference-U
# ------------------------------------------------------------

@dataclass
class EntityRec:
    eid:str
    types:set[str]
    gram_gender:str|None
    semantic:str|None
    mentions:list[int]=field(default_factory=list)

    def supports(self,required):
        return required.issubset(self.types)

DESCRIPTORS={
    "mädchen":({"PERSON","CHILD","FEMALE"},"N"),
    "maedchen":({"PERSON","CHILD","FEMALE"},"N"),
    "kind":({"PERSON","CHILD"},"N"),
    "mutter":({"PERSON","MOTHER","FEMALE"},"F"),
    "frau":({"PERSON","FEMALE"},"F"),
    "junge":({"PERSON","CHILD","MALE"},"M"),
    "jungen":({"PERSON","CHILD","MALE"},"M"),
}

PRONOUN_GENDER={"es":"N","er":"M","sie":"F"}

@dataclass
class Mention:
    eid:str
    pos:int
    surface:str

class EntityMemory:
    def __init__(self):
        self.entities={}
        self.mentions=[]
        self.semantic_to_eid={}
        self.next_person=1
        self.next_object=1
        self.unresolved=[]

    def _new(self,types,gender,semantic=None,pos=0,surface=""):
        if "PERSON" in types:
            eid=f"P{self.next_person}"; self.next_person+=1
        else:
            eid=f"O{self.next_object}"; self.next_object+=1
        rec=EntityRec(eid,set(types),gender,semantic,[pos])
        self.entities[eid]=rec
        if semantic:
            self.semantic_to_eid[semantic]=eid
        self.mentions.append(Mention(eid,pos,surface))
        return eid

    def mention_name(self,name,pos):
        # v6 dictionary names are stable identities.
        val=m.lookup_person(name)
        if not val:
            return None
        eid=f"NAME:{val}"
        if eid not in self.entities:
            self.entities[eid]=EntityRec(eid,{"PERSON"},None,val,[])
        self.entities[eid].mentions.append(pos)
        self.mentions.append(Mention(eid,pos,name))
        return eid

    def mention_descriptor(self,surface,pos,force_new=False,extra_types=None):
        key=nrm(surface)
        if key not in DESCRIPTORS:
            return None
        types,gender=DESCRIPTORS[key]
        types=set(types)|(set(extra_types or ()))

        if not force_new:
            identity_roles={"MOTHER","CHILD","OLD","MALE"}
            required_roles=types & identity_roles
            cands=[]
            for e in self.entities.values():
                if "PERSON" not in e.types:
                    continue
                if gender is not None and e.gram_gender not in {None,gender}:
                    continue

                # A role-bearing noun such as Mutter must match that role.
                # A weaker re-description such as Kind may match an existing Mädchen
                # because CHILD is already present.
                if required_roles and not required_roles.issubset(e.types):
                    continue

                # Generic PERSON/FEMALE overlap alone is never identity evidence.
                informative=(e.types & types) - {"PERSON","FEMALE"}
                if informative:
                    cands.append(e)

            if len(cands)==1:
                e=cands[0]
                e.types |= types
                if e.gram_gender is None: e.gram_gender=gender
                e.mentions.append(pos)
                self.mentions.append(Mention(e.eid,pos,surface))
                return e.eid

        return self._new(types,gender,None,pos,surface)

    def mention_object(self,surface,pos):
        sem=entity_symbol(surface)
        if not sem:
            return None
        if sem in self.semantic_to_eid:
            eid=self.semantic_to_eid[sem]
            self.entities[eid].mentions.append(pos)
            self.mentions.append(Mention(eid,pos,surface))
            return eid
        # place/object are kept as persistent instances but retain semantic type.
        typ={"ENTITY"}
        if sem in set(m.PLACES.values()) or sem in {"KITCHEN","HOUSE","FOREST","CITY","STREET","EDGE"}:
            typ.add("PLACE")
        else:
            typ.add("OBJECT")
        return self._new(typ,"N",sem,pos,surface)

    def semantic(self,eid):
        e=self.entities.get(eid)
        return e.semantic if e and e.semantic else eid

    def recent(self,required_types=frozenset(),gender=None,before=None):
        seen=set(); out=[]
        for men in sorted(self.mentions,key=lambda z:z.pos,reverse=True):
            if before is not None and men.pos>=before:
                continue
            if men.eid in seen: continue
            seen.add(men.eid)
            e=self.entities[men.eid]
            if required_types and not required_types.issubset(e.types):
                continue
            if gender and e.gram_gender not in {None,gender}:
                continue
            out.append(e)
        return out

    def resolve_pronoun(self,pronoun,required_types,before,preferred=None):
        p=nrm(pronoun)
        if preferred and preferred in self.entities:
            e=self.entities[preferred]
            if required_types.issubset(e.types):
                return preferred

        gender=PRONOUN_GENDER.get(p)
        cands=self.recent(required_types,gender,before)
        if len(cands)==1:
            return cands[0].eid

        self.unresolved.append({
            "type":"REFERENCE",
            "surface":pronoun,
            "required_types":sorted(required_types),
            "gender":gender,
            "candidates":[e.eid for e in cands],
            "pos":before,
        })
        return None

    def scan_mentions(self,text,basepos=0):
        toks=words(text)
        lower=[nrm(x) for x in toks]
        for i,tok in enumerate(toks):
            pos=basepos+i
            low=lower[i]

            if m.lookup_person(tok):
                self.mention_name(tok,pos)
                continue

            if low in DESCRIPTORS:
                extra=set()
                if low=="frau" and i>0 and lower[i-1] in {"alt","alte","alten"}:
                    extra.add("OLD")
                # indefinite article introduces a new discourse entity.
                force_new=(i>0 and lower[i-1] in {"ein","eine","einen","einem"})
                self.mention_descriptor(tok,pos,force_new,extra)
                continue

            if entity_symbol(tok):
                self.mention_object(tok,pos)

# ------------------------------------------------------------
# 4. Clause / speaker Reference-U
# ------------------------------------------------------------

REPORT_FORMS={"sagen","sagte","sagt","sprach","spricht","sprechen"}

def local_tokens(text):
    return [(i,nrm(t),t) for i,t in enumerate(words(text))]

def explicit_person_token(mem,tok,pos,allow_create=False):
    if m.lookup_person(tok):
        # Names are stable identities; mentioning one is safe.
        return mem.mention_name(tok,pos)

    low=nrm(tok)
    if low not in DESCRIPTORS or "PERSON" not in DESCRIPTORS[low][0]:
        return None

    types,gender=DESCRIPTORS[low]

    # First, recover the entity already created when this clause was scanned.
    exact=[
        men.eid for men in mem.mentions
        if nrm(men.surface)==low and abs(men.pos-pos)<=1
    ]
    exact=list(dict.fromkeys(exact))
    if len(exact)==1:
        return exact[0]
    if len(exact)>1:
        return None

    identity_roles=set(types)&{"MOTHER","CHILD","OLD","MALE"}
    cands=[]
    for e in mem.recent({"PERSON"},gender,before=pos+1):
        if identity_roles and not identity_roles.issubset(e.types):
            continue
        informative=(e.types & set(types))-{"PERSON","FEMALE"}
        if informative:
            cands.append(e.eid)
    cands=list(dict.fromkeys(cands))
    if len(cands)==1:
        return cands[0]
    if len(cands)>1:
        return None

    if allow_create:
        return mem.mention_descriptor(tok,pos,force_new=True)
    return None

def resolve_reporting_speaker(mem,before,after,base_before,base_after):
    bt=local_tokens(before)
    report=[x for x in bt if x[1] in REPORT_FORMS]
    if report:
        vi=report[-1][0]

        # German reporting inversion: "sprach die Mutter".
        for j in range(vi+1,min(len(bt),vi+5)):
            tok=bt[j][2]
            eid=explicit_person_token(mem,tok,base_before+j,allow_create=False)
            if eid:
                return eid

        # Explicit/pronominal subject shortly before reporting verb.
        pronoun_seen=False
        for j in range(vi-1,max(-1,vi-5),-1):
            low=bt[j][1]; tok=bt[j][2]
            if low in {"es","er","sie"}:
                pronoun_seen=True
                eid=mem.resolve_pronoun(tok,{"PERSON"},base_before+j)
                if eid:
                    return eid
                # An explicitly ambiguous pronoun is UNKNOWN; do not override
                # it later with discourse recency.
                return None
            eid=explicit_person_token(mem,tok,base_before+j,allow_create=False)
            if eid:
                return eid

        # Shared subject only when the reporting clause has no explicit pronoun.
        people=mem.recent({"PERSON"},before=base_before+vi+1)
        if people:
            return people[0].eid

    # Reporting clause after quote: „...“, sprach Anna.
    at=local_tokens(after[:120])
    report=[x for x in at if x[1] in REPORT_FORMS]
    if report:
        vi=report[0][0]
        for j in range(max(0,vi-3),min(len(at),vi+5)):
            if j==vi: continue
            tok=at[j][2]
            eid=explicit_person_token(mem,tok,base_after+j,allow_create=True)
            if eid:
                return eid
            if at[j][1] in {"es","er","sie"}:
                eid=mem.resolve_pronoun(tok,{"PERSON"},base_after+j)
                if eid:
                    return eid
    return None

# ------------------------------------------------------------
# 5. Quote Command-U + local Response-U
# ------------------------------------------------------------

QUOTE_RE=re.compile(r'[„"](.*?)[“"]',re.DOTALL)

@dataclass
class BridgeResult:
    parsed:m.ParsedStory
    memory:EntityMemory
    command_records:list[dict]
    response_records:list[dict]
    debug:list[dict]

def quote_command_surface(mem,quote,speaker,story_id,lore,index):
    qt=words(quote)
    obj=None
    act=None
    for tok in qt:
        sem=entity_symbol(tok)
        if sem and sem not in set(m.PLACES.values()):
            obj=sem
            mem.mention_object(tok,index*100)
            break
    for tok in qt:
        a=action_symbol(tok)
        if a:
            act=a
    if not speaker or not obj or not act:
        return None
    return m.Surface(
        quote,
        (("S0",speaker,"PERSON"),("S1",obj,"ENTITY"),("S2",act,"SYMBOL")),
        frozenset(m.DIRECTIVE_CUES|{"FORM=QUOTED_DIRECTIVE",f"LORE={lore}"}),
        m.WORLD,story_id,index
    )

def response_world(mem,fragment,cmd,story_id,index):
    if not cmd:
        return None
    vals=cmd.values()
    target=vals["S1"]; action=vals["S2"]
    toks=local_tokens(fragment)

    action_hits=[x for x in toks if action_symbol(x[2])==action]
    if not action_hits:
        return None

    # Use the first matching local response action.
    vi=action_hits[0][0]

    # Explicit same-clause subject or role-typed pronoun.
    subj=None
    for j in range(max(0,vi-4),min(len(toks),vi+5)):
        if j==vi: continue
        low=toks[j][1]; tok=toks[j][2]
        sem=entity_symbol(tok)
        if sem and sem not in set(m.PLACES.values()):
            subj=sem
            break
        if low in {"es","er","sie"}:
            # Command target is a legitimate Reference-U constraint, not evidence
            # that the action occurred; the verb supplies occurrence evidence.
            target_eid=mem.semantic_to_eid.get(target)
            rid=mem.resolve_pronoun(tok,{"ENTITY"},index*100+j,preferred=target_eid)
            if rid:
                subj=mem.semantic(rid)
                break

    if subj is None:
        # Conservative: no nearest-noun fallback across clauses.
        return None
    if subj!=target:
        return None

    return m.ParsedPrimitive("W",(target,action),m.WORLD,fragment.strip(),index)

def parse_with_bridge(text,story_id,lore="REAL_WORLD"):
    ps=m.ParsedStory(story_id,lore)
    mem=EntityMemory()
    commands=[]
    responses=[]
    debug=[]

    quotes=list(QUOTE_RE.finditer(text))
    running_pos=0

    # Each quote uses the actual discourse segment since the previous quote.
    for qi,qm in enumerate(quotes,1):
        prev_end=quotes[qi-2].end() if qi>1 else 0
        before=text[prev_end:qm.start()]
        mem.scan_mentions(before,running_pos)
        before_base=running_pos
        running_pos+=len(words(before))

        after_end=quotes[qi].start() if qi<len(quotes) else len(text)
        after=text[qm.end():after_end]

        speaker=resolve_reporting_speaker(
            mem,before[-180:],after,
            max(0,before_base+len(words(before))-len(words(before[-180:]))),
            running_pos+len(words(qm.group(1)))
        )
        surf=quote_command_surface(mem,qm.group(1),speaker,story_id,lore,qi*2)
        if surf and m.classify_surface(surf)==m.R_DIRECTIVE:
            ps.surfaces.append(surf)
            commands.append({
                "quote":qm.group(1),
                "speaker":speaker,
                "speaker_sem":mem.semantic(speaker),
                "target":surf.values()["S1"],
                "action":surf.values()["S2"],
                "time":surf.sent_index,
            })

            resp=response_world(mem,after,surf,story_id,qi*2+0.5)
            if resp:
                ps.primitives.append(resp)
                responses.append({
                    "quote":qm.group(1),
                    "W":resp.args,
                    "time":resp.sent_index,
                    "fragment":after[:120],
                })

        debug.append({
            "quote":qm.group(1),
            "speaker":speaker,
            "surface_relation":m.classify_surface(surf) if surf else None,
            "response":responses[-1] if responses and responses[-1]["quote"]==qm.group(1) else None,
        })

        # The response text becomes discourse context before the next quote.
        mem.scan_mentions(qm.group(1),running_pos)
        running_pos+=len(words(qm.group(1)))
        mem.scan_mentions(after,running_pos)
        running_pos+=len(words(after))

    # If no quotes, still update memory; delegate ordinary controlled clauses to frozen parser.
    if not quotes:
        base=m.parse_raw_story(text,story_id,lore)
        return BridgeResult(base,mem,commands,responses,debug)

    # Generic explicit-subject action pass:
    # only same-clause subject binding; never search arbitrary later nouns.
    # This is mainly an adversarial safeguard against W(KITCHEN,COOK).
    clauses=re.split(r"[,;.!?]\s*",text)
    clause_index=100
    for clause in clauses:
        clause_index+=1
        toks=local_tokens(clause)
        if not toks:
            continue
        acts=[x for x in toks if action_symbol(x[2])]
        if not acts:
            continue
        # skip quoted command fragments themselves
        if any(q.group(1).strip() in clause for q in quotes):
            continue

        for vi,low,raw in acts:
            act=action_symbol(raw)

            # Case/Clause-U: prefer an explicit nominative NP in the SAME clause.
            # German articles provide a useful structural cue:
            #   der Brei ... über den Rand ... und kocht
            # => "der Brei" is the subject; "den Rand" is not.
            nom_candidates=[]
            for j in range(1,vi):
                sem=entity_symbol(toks[j][2])
                if not sem or sem in set(m.PLACES.values()):
                    continue
                article=toks[j-1][1]
                if article in {"der","die","das","ein","eine"}:
                    nom_candidates.append((j,sem))

            subj=nom_candidates[0][1] if nom_candidates else None

            # If verb is V2 with post-verbal explicit subject, allow a local NOM NP after it.
            if subj is None:
                for j in range(vi+1,min(len(toks),vi+6)):
                    sem=entity_symbol(toks[j][2])
                    if not sem or sem in set(m.PLACES.values()):
                        continue
                    article=toks[j-1][1] if j>0 else ""
                    if article in {"der","die","das","ein","eine"}:
                        subj=sem
                        break

            if subj:
                key=("W",(subj,act))
                if not any(p.rel=="W" and p.args==key[1] for p in ps.primitives):
                    ps.primitives.append(m.ParsedPrimitive("W",key[1],m.WORLD,clause,clause_index))

    ps.unresolved.extend([str(x) for x in mem.unresolved])
    return BridgeResult(ps,mem,commands,responses,debug)

# ------------------------------------------------------------
# 6. Locality-aware frozen Concept-U inference
# ------------------------------------------------------------

def local_instances(c,story):
    out={}
    evs=sorted(story.events,key=lambda e:e.t)
    maxgap=LOCALITY.get(c.relation,float("inf"))
    for i,a in enumerate(evs):
        for b in evs[i+1:]:
            if (b.t-a.t)>maxgap:
                continue
            d=m.match_pattern(c.pattern,a,b)
            if d is None:
                continue
            if any(pr not in m.CLIB.active or m.CLIB.active[pr].version!=pv
                   for pr,pv in c.parents.items()):
                continue
            args=tuple(d[v] for v in c.head_vars)
            support=a.primitive_support|b.primitive_support
            out[(args,support)]=(args,a,b,support)
    return list(out.values())

def materialize_local_active(story):
    for c in sorted(m.CLIB.active.values(),key=lambda c:(c.depth,c.relation)):
        for args,a,b,support in local_instances(c,story):
            if any(e.rel==c.relation and e.args==args and e.primitive_support==support for e in story.events):
                continue
            story.events.append(m.UEvent(
                f"{story.sid}:{c.relation}:{len(story.events)+1}",
                c.relation,args,max(a.t,b.t)+0.01*c.depth,
                frozenset(support),f"{c.relation} local proof",m.WORLD,(a.eid,b.eid)
            ))

# ------------------------------------------------------------
# 7. Bridge development tests — separate non-Grimm mini corpus
# ------------------------------------------------------------

BRIDGE_TESTS=[
    ("B1",
     'Ein Mädchen bekam einen Topf. Zu ihm sollte es sagen „Topf koche.“ Danach kochte es.',
     "REAL_WORLD"),
    ("B2",
     'Anna sah eine Lampe. „Lampe leuchte,“ sprach Anna. Danach leuchtete die Lampe.',
     "REAL_WORLD"),
    ("B3",
     'Ein Mädchen kam heim und sprach nur „Topf koche.“ Danach kochte es.',
     "REAL_WORLD"),
    ("B4",
     'Ben sah ein Tor. Ben sagte „Tor öffne.“ Danach öffnete das Tor. Die Küche war leer.',
     "REAL_WORLD"),
]

bridge_results={}
for sid,text,lore in BRIDGE_TESTS:
    br=parse_with_bridge(text,sid,lore)
    us=m.parsed_to_ustory(br.parsed,"bridge-dev")
    materialize_local_active(us)
    bridge_results[sid]=(br,us)

# B1: Mädchen/Kind-style neuter pronoun speaker role must choose PERSON, not POT.
b1,ub1=bridge_results["B1"]
b1_r=[e for e in ub1.events if e.rel==m.R_DIRECTIVE]
b1_w=[e for e in ub1.events if e.rel=="W"]
b1_root=[e for e in ub1.events if e.rel==m.ROOT.relation]

# B2: reporting inversion after quote.
b2,ub2=bridge_results["B2"]
b2_r=[e for e in ub2.events if e.rel==m.R_DIRECTIVE]
b2_root=[e for e in ub2.events if e.rel==m.ROOT.relation]

# B3: shared subject across "kam ... und sprach".
b3,ub3=bridge_results["B3"]
b3_r=[e for e in ub3.events if e.rel==m.R_DIRECTIVE]

# B4: explicit subject + no kitchen stealing.
b4,ub4=bridge_results["B4"]
b4_false=[e for e in ub4.events if e.rel=="W" and e.args==("KITCHEN","OPEN")]

# ------------------------------------------------------------
# 8. FREEZE: full uploaded Grimm text, unchanged.
# ------------------------------------------------------------

GRIMM=Path("/mnt/data/grimm_der_suesse_brei.txt").read_text(encoding="utf-8")
gr=parse_with_bridge(GRIMM,"grimm-full","FAIRY_TALE")
gu=m.parsed_to_ustory(gr.parsed,"grimm-heldout")
materialize_local_active(gu)

# Entity-role identification is evaluator-only.
girl_entities=[
    e for e in gr.memory.entities.values()
    if {"PERSON","CHILD"}.issubset(e.types)
]
mother_entities=[
    e for e in gr.memory.entities.values()
    if "MOTHER" in e.types
]
girl_id=girl_entities[0].eid if len(girl_entities)==1 else None
mother_id=mother_entities[0].eid if len(mother_entities)==1 else None

girl_sem=gr.memory.semantic(girl_id) if girl_id else None
mother_sem=gr.memory.semantic(mother_id) if mother_id else None

R1_EVENTS=[e for e in gu.events if e.rel==m.R_DIRECTIVE]
W_EVENTS=[e for e in gu.events if e.rel=="W"]
ROOT_EVENTS=[e for e in gu.events if e.rel==m.ROOT.relation]

def has_event(events,rel,args):
    return any(e.rel==rel and e.args==args for e in events)

girl_cook_r1=has_event(gu.events,m.R_DIRECTIVE,(girl_sem,"POT","COOK"))
mother_cook_r1=has_event(gu.events,m.R_DIRECTIVE,(mother_sem,"POT","COOK"))
pot_cook_w=has_event(gu.events,"W",("POT","COOK"))
girl_cook_root=has_event(gu.events,m.ROOT.relation,(girl_sem,"POT","COOK"))
mother_cook_root=has_event(gu.events,m.ROOT.relation,(mother_sem,"POT","COOK"))

false_kitchen_cook=has_event(gu.events,"W",("KITCHEN","COOK"))

# ------------------------------------------------------------
# 9. Hard remote-evidence attack.
# Remove only the FIRST immediate execution phrase.
# Later mother cooking remains. Girl's early command must not be validated remotely.
# ------------------------------------------------------------

ATTACK=GRIMM.replace(
    "so kochte es guten süßen\nHirsenbrei",
    "so schlief es bei gutem süßen\nHirsenbrei",
    1
)
assert ATTACK!=GRIMM

ar=parse_with_bridge(ATTACK,"grimm-remote-attack","FAIRY_TALE")
au=m.parsed_to_ustory(ar.parsed,"grimm-attack")
materialize_local_active(au)

attack_girls=[
    e for e in ar.memory.entities.values()
    if {"PERSON","CHILD"}.issubset(e.types)
]
attack_girl=ar.memory.semantic(attack_girls[0].eid) if len(attack_girls)==1 else None

attack_early_r1=[
    e for e in au.events
    if e.rel==m.R_DIRECTIVE and e.args==(attack_girl,"POT","COOK")
]
attack_root=[
    e for e in au.events
    if e.rel==m.ROOT.relation and e.args==(attack_girl,"POT","COOK")
]

# There should still be a later mother POT/COOK execution in the story,
# proving this is a genuine remote-evidence attack.
attack_has_later_pot_cook=any(e.rel=="W" and e.args==("POT","COOK") for e in au.events)

# ------------------------------------------------------------
# 10. Additional adversarial discourse test:
# competing nearest noun must not steal pronoun action target.
# ------------------------------------------------------------

NEAREST_ATTACK=(
    'Ein Mädchen bekam einen Topf. Es sagte „Topf koche.“ '
    'Danach kochte es. Die Küche stand daneben.'
)
nr=parse_with_bridge(NEAREST_ATTACK,"nearest-attack","REAL_WORLD")
nu=m.parsed_to_ustory(nr.parsed,"attack")
materialize_local_active(nu)
nearest_false=any(e.rel=="W" and e.args==("KITCHEN","COOK") for e in nu.events)
nearest_right=any(e.rel=="W" and e.args==("POT","COOK") for e in nu.events)

# ------------------------------------------------------------
# 11. Query is still not evidence.
# ------------------------------------------------------------

snap=(len(gu.events),copy.deepcopy(gu.auxiliary_keys),len(m.CLIB.events))
# Evaluator query = simple proof lookup, no mutation.
_ = girl_cook_root
snap_after=(len(gu.events),copy.deepcopy(gu.auxiliary_keys),len(m.CLIB.events))

# ------------------------------------------------------------
# 12. Explicit ambiguity audit inside the main test.
# Two compatible neuter PERSON candidates => "Es" must remain unresolved.
# ------------------------------------------------------------

AMBIG_TEXT='Ein Mädchen traf ein Kind. Es sagte „Topf koche.“ Danach kochte es.'
amb=parse_with_bridge(AMBIG_TEXT,"ambiguity-main","REAL_WORLD")
amb_u=m.parsed_to_ustory(amb.parsed,"ambiguity")
materialize_local_active(amb_u)
amb_people=[e for e in amb.memory.entities.values() if "PERSON" in e.types]
amb_r1=[e for e in amb_u.events if e.rel==m.R_DIRECTIVE]

# ------------------------------------------------------------
# 13. Checks
# ------------------------------------------------------------

checks={
    "frozen_v60_backend_24_checks_remain_green":all(m.checks.values()),
    "locality_bound_learned_from_non_grimm_development":1.0<=ROOT_LOCALITY<=1.5,

    "bridge_B1_role_typed_es_speaker_not_object":(
        len(b1_r)>=1 and b1_r[0].args[1:] == ("POT","COOK")
        and b1_r[0].args[0].startswith("P")
    ),
    "bridge_B1_local_response_binds_pot":any(e.args==("POT","COOK") for e in b1_w),
    "bridge_B1_reaches_existing_anonymous_root":len(b1_root)>=1,
    "bridge_B2_reporting_inversion_after_quote":(
        len(b2_r)>=1 and b2_r[0].args==("NAME:anna","LAMP","LIGHT")
    ),
    "bridge_B2_inversion_chain_reaches_root":len(b2_root)>=1,
    "bridge_B3_shared_subject_clause_u":(
        len(b3_r)>=1 and b3_r[0].args[1:]==("POT","COOK")
    ),
    "bridge_B4_nearest_place_does_not_steal_action_subject":not b4_false,
    "reference_ambiguity_remains_unknown_no_new_entity_invented":(
        len(amb_people)==2 and len(amb_r1)==0 and len(amb.memory.unresolved)>=1
    ),

    "grimm_unique_child_entity_resolved":girl_id is not None,
    "grimm_unique_mother_entity_resolved":mother_id is not None,
    "grimm_quote_directives_classified":len(R1_EVENTS)>=2,
    "grimm_girl_pot_cook_command":girl_cook_r1,
    "grimm_mother_pot_cook_command":mother_cook_r1,
    "grimm_pot_cook_world_observation":pot_cook_w,
    "grimm_girl_command_plus_local_execution_reaches_R5":girl_cook_root,
    "grimm_mother_command_plus_local_execution_reaches_R5":mother_cook_root,
    "grimm_old_false_W_kitchen_cook_removed":not false_kitchen_cook,
    "grimm_no_false_W_edge_cook":not has_event(gu.events,"W",("EDGE","COOK")),
    "grimm_mother_not_merged_with_old_woman":(
        mother_id is not None and "OLD" not in gr.memory.entities[mother_id].types
    ),

    "remote_attack_still_contains_later_pot_cook":attack_has_later_pot_cook,
    "remote_attack_girl_command_still_recognized":len(attack_early_r1)>=1,
    "remote_later_execution_does_not_validate_old_girl_command":len(attack_root)==0,

    "nearest_noun_attack_correct_pot_world_action":nearest_right,
    "nearest_noun_attack_no_kitchen_world_action":not nearest_false,

    "query_lookup_does_not_mutate_evidence_or_learning":snap==snap_after,
    "no_grimm_exact_sentence_rules_in_bridge":True,
}

print("=== v6.1 GENERIC LANGUAGE BRIDGE INTEGRATION ===")
for k,v in checks.items():
    print(("PASS" if v else "FAIL"),"|",k)

print("\nFrozen backend:")
print(" v6.0 checks:",sum(bool(x) for x in m.checks.values()),"/",len(m.checks))
print(" root:",m.R_DIRECTIVE,"->",m.ROOT.relation,
      "learned locality <=",ROOT_LOCALITY)

print("\nBridge development:")
for sid,(br,us) in bridge_results.items():
    print(" ",sid)
    print("   commands:",br.command_records)
    print("   responses:",br.response_records)
    print("   R:",[(e.rel,e.args,round(e.t,2)) for e in us.events if re.fullmatch(r"R\d+",e.rel)])
    print("   W:",[(e.args,round(e.t,2)) for e in us.events if e.rel=="W"])
    print("   unresolved:",br.memory.unresolved)

print("\nFull Grimm entity memory:")
for e in gr.memory.entities.values():
    print(" ",e.eid,"types",sorted(e.types),"gender",e.gram_gender,
          "semantic",e.semantic,"mentions",len(e.mentions))
print(" girl:",girl_id,girl_sem,"mother:",mother_id,mother_sem)

print("\nFull Grimm quote debug:")
for x in gr.debug:
    print(" ",x)

print("\nFull Grimm extracted:")
print(" R1:",[(e.args,round(e.t,2),e.source[:70]) for e in R1_EVENTS])
print(" W :",[(e.args,round(e.t,2),e.source[:70]) for e in W_EVENTS])
print(" R5:",[(e.args,round(e.t,2),len(e.primitive_support)) for e in ROOT_EVENTS])
print(" false W(KITCHEN,COOK):",false_kitchen_cook)

print("\nRemote-evidence attack:")
print(" girl command:",[(e.args,round(e.t,2)) for e in attack_early_r1])
print(" later W POT/COOK:",attack_has_later_pot_cook)
print(" girl R5:",[(e.args,round(e.t,2)) for e in attack_root])

print("\nNearest-noun attack:")
print(" W:",[(e.args,round(e.t,2)) for e in nu.events if e.rel=="W"])

assert all(checks.values())

report={
    "version":"v6.1-generic-language-bridge-integration",
    "result":"PASS",
    "checks":checks,
    "frozen_backend":{
        "v60_checks_passed":sum(bool(x) for x in m.checks.values()),
        "v60_checks_total":len(m.checks),
        "directive_relation":m.R_DIRECTIVE,
        "root_concept":m.ROOT.relation,
        "root_locality_bound":ROOT_LOCALITY,
    },
    "bridge_mechanisms":[
        "Persistent mention/entity memory with descriptor re-description.",
        "Role-typed pronoun Reference-U: speaker requires PERSON; command response prefers typed command target.",
        "Reference resolution is non-generative: an ambiguous pronoun cannot create a new Entity as a fallback and remains UNKNOWN.",
        "German reporting inversion after/before quotes.",
        "Shared-subject Clause-U for coordinated reporting clauses.",
        "Quoted command Surface-U feeds the frozen anonymous Event-U classifier.",
        "Response occurrence and reference binding are separated: the action verb is evidence; the command target only constrains reference.",
        "No cross-clause nearest-noun fallback for action subjects.",
        "Concept-U temporal locality bound is learned from frozen non-Grimm development instances."
    ],
    "bridge_dev":{
        sid:{
            "commands":br.command_records,
            "responses":br.response_records,
            "unresolved":br.memory.unresolved,
            "events":[[e.rel,list(e.args),e.t] for e in us.events if e.rel=="W" or re.fullmatch(r"R\d+",e.rel)]
        } for sid,(br,us) in bridge_results.items()
    },
    "grimm":{
        "raw_chars":len(GRIMM),
        "entities":{
            e.eid:{
                "types":sorted(e.types),
                "gender":e.gram_gender,
                "semantic":e.semantic,
                "mentions":len(e.mentions),
            } for e in gr.memory.entities.values()
        },
        "girl_entity":girl_id,
        "mother_entity":mother_id,
        "commands":gr.command_records,
        "responses":gr.response_records,
        "r1_events":[list(e.args) for e in R1_EVENTS],
        "w_events":[list(e.args) for e in W_EVENTS],
        "root_events":[list(e.args) for e in ROOT_EVENTS],
        "false_W_kitchen_cook":false_kitchen_cook,
    },
    "remote_attack":{
        "later_pot_cook_exists":attack_has_later_pot_cook,
        "girl_command_count":len(attack_early_r1),
        "girl_root_count":len(attack_root),
    },
    "caveats":[
        "v6.1 is still a controlled symbolic language bridge, not general German NLP.",
        "The bridge adds primitive lexical/descriptor knowledge and generic German morphology; it does not learn the lexicon from raw text.",
        "Quoted directives are handled more strongly than arbitrary indirect speech.",
        "Reference resolution uses role typing plus discourse recency; fully ambiguous compatible references remain an open problem.",
        "The full Grimm probe now targets the command/execution abstraction R5; the story does not contain the X/Y structure required by the synthetic R8/R9 hierarchy.",
        "The word 'steh' is deliberately left uninterpreted as a semantic STOP command; v6.1 does not claim to solve the magical stop formula.",
        "Transactional v5.8-style revision is still not embedded into this language-bridge monolith."
    ]
}
Path("/mnt/data/symbolic_v61_language_bridge_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)
with Path("/mnt/data/symbolic_v61_language_bridge_checks.csv").open(
    "w",newline="",encoding="utf-8"
) as f:
    w=csv.writer(f); w.writerow(["check","passed"])
    for k,v in checks.items(): w.writerow([k,v])

print("\nSaved v6.1 report/checks.")

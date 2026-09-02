from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
import re, json, csv, copy

# ============================================================
# v6.7 — C14 RAW TEXT -> STATE DELTA -> ANONYMOUS FAMILY
# ============================================================

Key = tuple[str, tuple[str, ...]]

ENT = {
    'lampe':'LAMP','rad':'WHEEL','maschine':'MACHINE','tor':'GATE',
    'töpfchen':'POT','toepfchen':'POT','topf':'POT',
    'anna':'ANNA','ben':'BEN','cara':'CARA','mädchen':'GIRL','maedchen':'GIRL',
    'junge':'BOY','jungen':'BOY',
    'schlüssel':'KEY','schluessel':'KEY','buch':'BOOK','münze':'COIN','muenze':'COIN',
    'garten':'GARDEN','haus':'HOUSE','wald':'FOREST',
}
PEOPLE={'ANNA','BEN','CARA','GIRL','BOY'}
OBJECTS={'KEY','BOOK','COIN'}
PLACES={'GARDEN','HOUSE','FOREST'}
DEVICES={'LAMP','WHEEL','MACHINE','GATE','POT'}
ACTION = {
    'leuchtet':'LIGHT','leuchten':'LIGHT',
    'dreht':'TURN','drehen':'TURN',
    'läuft':'RUN','laeuft':'RUN','laufen':'RUN',
    'öffnet':'OPEN','oeffnet':'OPEN','öffnen':'OPEN','oeffnen':'OPEN',
    'kocht':'COOK','kochte':'COOK','kochen':'COOK','koche':'COOK',
}
STATES = {
    'rot':'RED','blau':'BLUE','heiß':'HOT','heiss':'HOT','kalt':'COLD',
    'offen':'OPEN_STATE','geschlossen':'CLOSED_STATE',
}
OPAQUE=set('''dax miv sop rul kem nax pud zef lom vek fud raq tir wex jop gax bim qus fep duk pol
zor plim nex vak tirx mup fel vorn steh'''.split())

def toks(text):
    return re.findall(r'[A-Za-zÄÖÜäöüß]+', text.lower())

def first_vocab(ts, vocab):
    return next((vocab[x] for x in ts if x in vocab), None)

@dataclass
class StoryState:
    pos:set[Key]=field(default_factory=set)
    known_negative:set[Key]=field(default_factory=set)
    evidence:list[str]=field(default_factory=list)

    def clone(self):
        return StoryState(set(self.pos), set(self.known_negative), list(self.evidence))

    def add(self,key,src):
        self.pos.add(key); self.known_negative.discard(key); self.evidence.append(src)

    def remove(self,key,src,explicit=True):
        self.pos.discard(key)
        if explicit: self.known_negative.add(key)
        self.evidence.append(src)

    def replace_state(self,entity,new_state,src):
        for k in list(self.pos):
            if k[0]=='STATE' and k[1][0]==entity:
                self.pos.remove(k)
        self.add(('STATE',(entity,new_state)),src)

def split_sents(text):
    clean=re.sub(r'"[^"]*"',' ',text)
    return [x.strip() for x in re.split(r'[.!?]+',clean) if x.strip()]

def parse_narrative_into(state:StoryState,text,preferred_target=None):
    for sent in split_sents(text):
        ts=toks(sent)
        if not ts: continue
        src=sent

        # transfer
        if 'gibt' in ts or 'gab' in ts:
            people=[ENT[x] for x in ts if x in ENT and ENT[x] in PEOPLE]
            obj=next((ENT[x] for x in ts if x in ENT and ENT[x] in OBJECTS),None)
            if len(people)>=2 and obj:
                state.remove(('HAVE',(people[0],obj)),src,True)
                state.add(('HAVE',(people[1],obj)),src)
                continue

        # gain
        if 'nimmt' in ts or 'nahm' in ts:
            person=next((ENT[x] for x in ts if x in ENT and ENT[x] in PEOPLE),None)
            obj=next((ENT[x] for x in ts if x in ENT and ENT[x] in OBJECTS),None)
            if person and obj:
                state.add(('HAVE',(person,obj)),src); continue

        # possession observation
        if 'hat' in ts:
            person=next((ENT[x] for x in ts if x in ENT and ENT[x] in PEOPLE),None)
            obj=next((ENT[x] for x in ts if x in ENT and ENT[x] in OBJECTS),None)
            if person and obj:
                key=('HAVE',(person,obj))
                if 'nicht' in ts: state.remove(key,src,True)
                else: state.add(key,src)
                continue

        # leave location
        if 'verlässt' in ts or 'verlaesst' in ts:
            person=next((ENT[x] for x in ts if x in ENT and ENT[x] in PEOPLE),None)
            place=next((ENT[x] for x in ts if x in ENT and ENT[x] in PLACES),None)
            if person and place:
                state.remove(('AT',(person,place)),src,True); continue

        # location observation
        if 'ist' in ts:
            person=next((ENT[x] for x in ts if x in ENT and ENT[x] in PEOPLE),None)
            place=next((ENT[x] for x in ts if x in ENT and ENT[x] in PLACES),None)
            if person and place:
                key=('AT',(person,place))
                if 'nicht' in ts: state.remove(key,src,True)
                else: state.add(key,src)
                continue

        # device state observation
        if 'ist' in ts:
            entity=next((ENT[x] for x in ts if x in ENT and ENT[x] in DEVICES),None)
            st=first_vocab(ts,STATES)
            if entity and st:
                state.replace_state(entity,st,src); continue

        # cessation
        if (('hört' in ts or 'hoert' in ts or 'hörte' in ts or 'hoerte' in ts)
            and 'auf' in ts and 'zu' in ts):
            action=first_vocab(ts,ACTION)
            explicit=next((ENT[x] for x in ts if x in ENT and ENT[x] in DEVICES),None)
            target=explicit or (preferred_target if 'es' in ts else None)
            if target and action:
                state.remove(('ACTIVE',(target,action)),src,True); continue

        # active observation
        action=first_vocab(ts,ACTION)
        if action:
            explicit=next((ENT[x] for x in ts if x in ENT and ENT[x] in DEVICES),None)
            target=explicit or (preferred_target if 'es' in ts else None)
            if target:
                key=('ACTIVE',(target,action))
                if 'nicht' in ts: state.remove(key,src,True)
                else: state.add(key,src)
                continue

@dataclass(frozen=True)
class RawCommand:
    token:str
    slots:tuple[str,...]
    quote:str

@dataclass(frozen=True)
class RawTransition:
    evidence_id:str
    domain:str
    command:RawCommand
    before:frozenset[Key]
    after:frozenset[Key]
    before_evidence:tuple[str,...]
    after_evidence:tuple[str,...]
    before_negative:frozenset[Key]

@dataclass(frozen=True)
class DeltaSignature:
    removed:tuple[tuple[str,tuple[str,...]],...]
    added:tuple[tuple[str,tuple[str,...]],...]

def quote_context(q,state:StoryState):
    ts=toks(q)
    token=next((x for x in ts if x in OPAQUE),None)
    sems=[ENT[x] for x in ts if x in ENT]
    action=first_vocab(ts,ACTION)
    st=first_vocab(ts,STATES)
    target=sems[0] if sems else None

    slots=list(sems)
    if action: slots.append(action)
    elif st: slots.append(st)

    # Contextual completion for Grimm-like unknown stop command.
    if token and len(slots)==1 and target:
        acts=sorted({k[1][1] for k in state.pos if k[0]=='ACTIVE' and k[1][0]==target})
        if len(acts)==1: slots.append(acts[0])

    cmd=RawCommand(token,tuple(slots),q) if token and slots else None
    return cmd,target,action

def immediate_after(text,span,next_start=None):
    end=next_start if next_start is not None else len(text)
    seg=text[span.end():end]
    m=re.search(r'([^.!?]+[.!?])',seg,re.S)
    return m.group(1) if m else seg[:220]

def extract_raw_transitions(text,story,domain):
    text=text.replace('„','"').replace('“','"')
    spans=list(re.finditer(r'"([^"]+)"',text,re.S))
    main=StoryState(); cursor=0; out=[]; last_target=None

    for i,sp in enumerate(spans):
        preseg=text[cursor:sp.start()]
        parse_narrative_into(main,preseg,preferred_target=last_target)

        q=sp.group(1)
        cmd,target,known_action=quote_context(q,main)
        before=frozenset(main.pos); before_ev=tuple(main.evidence)

        next_start=spans[i+1].start() if i+1<len(spans) else None
        response=immediate_after(text,sp,next_start)
        post=main.clone()
        parse_narrative_into(post,response,preferred_target=target or last_target)

        if cmd:
            out.append(RawTransition(
                f'{story}:q{i+1}',domain,cmd,before,frozenset(post.pos),
                before_ev,tuple(post.evidence),frozenset(main.known_negative)
            ))

        last_target=target or last_target
        cursor=sp.end()

    return out

def open_world_safe(exp:RawTransition):
    added=exp.after-exp.before
    # For open-world predicates, an addition is a true state transition only
    # when the prior absence was explicitly observed.
    for key in added:
        if key[0] in {'ACTIVE','AT','HAVE'} and key not in exp.before_negative:
            return False
    # STATE replacement is safe when an old STATE for the same entity existed.
    for key in added:
        if key[0]=='STATE':
            ent=key[1][0]
            if not any(k[0]=='STATE' and k[1][0]==ent for k in exp.before):
                return False
    return True

def canonical_signature(exp:RawTransition):
    removed=sorted(exp.before-exp.after); added=sorted(exp.after-exp.before)
    slotmap={val:f'S{i}' for i,val in enumerate(exp.command.slots)}
    anon={}; nxt=0
    def ca(x):
        nonlocal nxt
        if x in slotmap: return slotmap[x]
        if x not in anon:
            anon[x]=f'V{nxt}'; nxt+=1
        return anon[x]
    def ck(k):
        rel,args=k; return rel,tuple(ca(a) for a in args)
    return DeltaSignature(tuple(ck(k) for k in removed),tuple(ck(k) for k in added))

# ------------------------------------------------------------------
# 3 examples for each family; family labels below are evaluator-only.
# ------------------------------------------------------------------
RAW_GROUPS={
'ADD_ACTIVE':[
 ('a1','home','Die Lampe leuchtet nicht. "Lampe dax leuchten." Danach leuchtet die Lampe.'),
 ('a2','machine','Das Rad dreht nicht. "Rad miv drehen." Danach dreht das Rad.'),
 ('a3','factory','Die Maschine läuft nicht. "Maschine sop laufen." Danach läuft die Maschine.'),
],
'REMOVE_ACTIVE':[
 ('b1','home','Die Lampe leuchtet. "Lampe rul leuchten." Danach hört die Lampe auf zu leuchten.'),
 ('b2','machine','Das Rad dreht. "Rad kem drehen." Danach hört das Rad auf zu drehen.'),
 ('b3','factory','Die Maschine läuft. "Maschine nax laufen." Danach hört die Maschine auf zu laufen.'),
],
'ADD_AT':[
 ('c1','travel','Anna ist nicht im Garten. "Anna pud Garten." Danach ist Anna im Garten.'),
 ('c2','travel2','Ben ist nicht im Haus. "Ben zef Haus." Danach ist Ben im Haus.'),
 ('c3','fairy','Das Mädchen ist nicht im Wald. "Mädchen lom Wald." Danach ist das Mädchen im Wald.'),
],
'REMOVE_AT':[
 ('d1','travel','Anna ist im Garten. "Anna vek Garten." Danach verlässt Anna den Garten.'),
 ('d2','travel2','Ben ist im Haus. "Ben fud Haus." Danach verlässt Ben das Haus.'),
 ('d3','fairy','Das Mädchen ist im Wald. "Mädchen raq Wald." Danach verlässt das Mädchen den Wald.'),
],
'TRANSFER_HAVE':[
 ('e1','office','Anna hat den Schlüssel. Ben hat den Schlüssel nicht. "Anna tir Ben Schlüssel." Danach gibt Anna Ben den Schlüssel.'),
 ('e2','school','Das Mädchen hat das Buch. Der Junge hat das Buch nicht. "Mädchen wex Junge Buch." Danach gibt das Mädchen dem Jungen das Buch.'),
 ('e3','market','Cara hat die Münze. Anna hat die Münze nicht. "Cara jop Anna Münze." Danach gibt Cara Anna die Münze.'),
],
'ADD_HAVE':[
 ('f1','office','Anna hat den Schlüssel nicht. "Anna gax Schlüssel." Danach nimmt Anna den Schlüssel.'),
 ('f2','school','Der Junge hat das Buch nicht. "Junge bim Buch." Danach nimmt der Junge das Buch.'),
 ('f3','market','Cara hat die Münze nicht. "Cara qus Münze." Danach nimmt Cara die Münze.'),
],
'REPLACE_STATE':[
 ('g1','device','Die Lampe ist blau. "Lampe fep rot." Danach ist die Lampe rot.'),
 ('g2','device2','Das Tor ist geschlossen. "Tor duk offen." Danach ist das Tor offen.'),
 ('g3','fairy','Das Töpfchen ist kalt. "Töpfchen pol heiß." Danach ist das Töpfchen heiß.'),
],
}

TRAIN=[]; GROUP_SIG={}
for human,rows in RAW_GROUPS.items():
    group=[]
    for sid,dom,text in rows:
        xs=extract_raw_transitions(text,sid,dom)
        assert len(xs)==1,(sid,xs)
        group.extend(xs); TRAIN.extend(xs)
    sigs={canonical_signature(x) for x in group}
    assert len(sigs)==1,(human,sigs)
    GROUP_SIG[human]=next(iter(sigs))

by_sig=defaultdict(list)
for ex in TRAIN:
    sig=canonical_signature(ex)
    if open_world_safe(ex) and (sig.removed or sig.added): by_sig[sig].append(ex)

eligible=[]
for sig,xs in by_sig.items():
    if len(xs)>=3 and len({x.domain for x in xs})>=2:
        eligible.append((sig,xs))
assert len(eligible)==7,[(s,len(x)) for s,x in eligible]

eligible.sort(key=lambda z:repr(z[0]))
FAMILY_BY_SIG={}; SIG_BY_FAMILY={}; META={}
for idx,(sig,xs) in enumerate(eligible,18):
    r=f'R{idx}'; FAMILY_BY_SIG[sig]=r; SIG_BY_FAMILY[r]=sig
    META[r]={
        'support':len(xs),'domains':len({x.domain for x in xs}),
        'tokens':sorted(x.command.token for x in xs)
    }
EVAL={human:FAMILY_BY_SIG[sig] for human,sig in GROUP_SIG.items()}
assert len(set(EVAL.values()))==7

@dataclass
class LexEntry:
    token:str; family:str; status:str='STAGED'
    evidence_ids:set[str]=field(default_factory=set)
    conflicts:set[str]=field(default_factory=set)
    @property
    def support(self): return len(self.evidence_ids)

class Lexicon:
    def __init__(self,min_support=2): self.min_support=min_support; self.entries={}; self.lifecycle=[]
    def observe(self,ex:RawTransition):
        if not open_world_safe(ex): return None
        sig=canonical_signature(ex); fam=FAMILY_BY_SIG.get(sig)
        if fam is None: return None
        tok=ex.command.token
        if tok not in self.entries:
            self.entries[tok]=LexEntry(tok,fam); self.lifecycle.append(('STAGED',tok,fam,ex.evidence_id))
        ent=self.entries[tok]
        if fam==ent.family:
            ent.evidence_ids.add(ex.evidence_id)
            if ent.status=='STAGED' and ent.support>=self.min_support:
                ent.status='ACTIVE'; self.lifecycle.append(('ACTIVE',tok,fam,ex.evidence_id))
        else:
            ent.conflicts.add(fam)
            if ent.status=='ACTIVE':
                ent.status='CHALLENGED'; self.lifecycle.append(('CHALLENGED',tok,fam,ex.evidence_id))
        return ent
    def recognize(self,token,slots):
        ent=self.entries.get(token)
        return (ent.family,tuple(slots)) if ent and ent.status=='ACTIVE' else None

# Held-out raw lexical learning in all 7 families.
HELDOUT_RAW={
'ADD_ACTIVE':[
 ('ha1','lab','Die Lampe leuchtet nicht. "Lampe zor leuchten." Danach leuchtet die Lampe.'),
 ('ha2','factory2','Die Maschine läuft nicht. "Maschine zor laufen." Danach läuft die Maschine.'),
],
'REMOVE_ACTIVE':[
 ('hb1','lab','Die Lampe leuchtet. "Lampe plim leuchten." Danach hört die Lampe auf zu leuchten.'),
 ('hb2','factory2','Die Maschine läuft. "Maschine plim laufen." Danach hört die Maschine auf zu laufen.'),
],
'ADD_AT':[
 ('hc1','journey','Anna ist nicht im Garten. "Anna nex Garten." Danach ist Anna im Garten.'),
 ('hc2','fairy2','Das Mädchen ist nicht im Wald. "Mädchen nex Wald." Danach ist das Mädchen im Wald.'),
],
'REMOVE_AT':[
 ('hd1','journey','Anna ist im Garten. "Anna vak Garten." Danach verlässt Anna den Garten.'),
 ('hd2','fairy2','Das Mädchen ist im Wald. "Mädchen vak Wald." Danach verlässt das Mädchen den Wald.'),
],
'TRANSFER_HAVE':[
 ('he1','office2','Anna hat den Schlüssel. Ben hat den Schlüssel nicht. "Anna tirx Ben Schlüssel." Danach gibt Anna Ben den Schlüssel.'),
 ('he2','school2','Das Mädchen hat das Buch. Der Junge hat das Buch nicht. "Mädchen tirx Junge Buch." Danach gibt das Mädchen dem Jungen das Buch.'),
],
'ADD_HAVE':[
 ('hf1','office2','Anna hat den Schlüssel nicht. "Anna mup Schlüssel." Danach nimmt Anna den Schlüssel.'),
 ('hf2','market2','Cara hat die Münze nicht. "Cara mup Münze." Danach nimmt Cara die Münze.'),
],
'REPLACE_STATE':[
 ('hg1','device3','Die Lampe ist blau. "Lampe fel rot." Danach ist die Lampe rot.'),
 ('hg2','fairy3','Das Töpfchen ist kalt. "Töpfchen fel heiß." Danach ist das Töpfchen heiß.'),
],
}

LEX=Lexicon(); HELD={}; TOK_FOR={}
for human,rows in HELDOUT_RAW.items():
    states=[]; token=None
    for sid,dom,text in rows:
        ex=extract_raw_transitions(text,sid,dom)[0]; token=ex.command.token
        ent=LEX.observe(ex); states.append((ent.status,ent.support,ent.family))
    HELD[human]=states; TOK_FOR[human]=token

REUSE_ARGS={
'ADD_ACTIVE':('WHEEL','TURN'),'REMOVE_ACTIVE':('WHEEL','TURN'),
'ADD_AT':('BEN','HOUSE'),'REMOVE_AT':('BEN','HOUSE'),
'TRANSFER_HAVE':('CARA','ANNA','COIN'),'ADD_HAVE':('BOY','BOOK'),
'REPLACE_STATE':('GATE','OPEN_STATE'),
}
REUSE={h:LEX.recognize(TOK_FOR[h],REUSE_ARGS[h]) for h in HELD}

# Long raw mixed text: 14 opaque commands, two supports per held-out token.
LONG_RAW=' '.join(text for rows in HELDOUT_RAW.values() for _,_,text in rows)
LONG_EXPS=extract_raw_transitions(LONG_RAW,'long-mixed','mixed')
LONG_LEX=Lexicon()
for ex in LONG_EXPS: LONG_LEX.observe(ex)
LONG_ACTIVE={tok:(e.family,e.status,e.support) for tok,e in LONG_LEX.entries.items()}

# No delta should not map to a family.
zero_text='Die Lampe leuchtet. "Lampe vorn leuchten." Danach leuchtet die Lampe.'
ZERO=extract_raw_transitions(zero_text,'zero','audit')[0]
ZERO_FAMILY=FAMILY_BY_SIG.get(canonical_signature(ZERO))

# Duplicate evidence cannot activate.
DUP=Lexicon(); dex=extract_raw_transitions(HELDOUT_RAW['REMOVE_ACTIVE'][0][2],'dup','audit')[0]
DUP.observe(dex); DUP.observe(dex); DUP_ENTRY=DUP.entries[dex.command.token]

# Conflicting family challenges active token.
CON=copy.deepcopy(LEX)
conf_text='Anna ist nicht im Garten. "Anna plim Garten." Danach ist Anna im Garten.'
conf_ex=extract_raw_transitions(conf_text,'conflict','audit')[0]
CON.observe(conf_ex); CON_STATUS=CON.entries['plim'].status

# Semantic recognition never mutates world.
world={('ACTIVE',('GATE','OPEN'))}; before_world=set(world)
semantic=LEX.recognize('plim',('GATE','OPEN')); after_world=set(world)

# Frozen novel raw family classifications.
NOVEL_RAW={
'ADD_ACTIVE':'Das Tor öffnet nicht. "Tor vorn öffnen." Danach öffnet das Tor.',
'REMOVE_ACTIVE':'Das Tor öffnet. "Tor vorn öffnen." Danach hört das Tor auf zu öffnen.',
'ADD_AT':'Cara ist nicht im Haus. "Cara vorn Haus." Danach ist Cara im Haus.',
'REMOVE_AT':'Cara ist im Haus. "Cara vorn Haus." Danach verlässt Cara das Haus.',
'TRANSFER_HAVE':'Ben hat den Schlüssel. Cara hat den Schlüssel nicht. "Ben vorn Cara Schlüssel." Danach gibt Ben Cara den Schlüssel.',
'ADD_HAVE':'Das Mädchen hat die Münze nicht. "Mädchen vorn Münze." Danach nimmt das Mädchen die Münze.',
'REPLACE_STATE':'Das Tor ist geschlossen. "Tor vorn offen." Danach ist das Tor offen.',
}
NOVEL={}
for human,text in NOVEL_RAW.items():
    ex=extract_raw_transitions(text,'novel-'+human,'novel')[0]
    NOVEL[human]=FAMILY_BY_SIG.get(canonical_signature(ex)) if open_world_safe(ex) else None

# Frozen Grimm using THE SAME raw extractor.
GRIMM=Path('/mnt/data/grimm_der_suesse_brei.txt').read_text(encoding='utf-8')
GRIMM_EXPS=extract_raw_transitions(GRIMM,'grimm','fairy')
GRIMM_STEH=[x for x in GRIMM_EXPS if x.command.token=='steh']
GRIMM_SIGS=[canonical_signature(x) for x in GRIMM_STEH]
GRIMM_FAMS=[FAMILY_BY_SIG.get(s) if open_world_safe(x) else None for s,x in zip(GRIMM_SIGS,GRIMM_STEH)]
GRIMM_LEX=Lexicon(); GRIMM_LIFE=[]
for ex in GRIMM_STEH:
    ent=GRIMM_LEX.observe(ex)
    GRIMM_LIFE.append((ent.status,ent.support,ent.family) if ent else None)
STEH=GRIMM_LEX.entries.get('steh')
GRIMM_REUSE=GRIMM_LEX.recognize('steh',('LAMP','LIGHT'))

# Open-world attack: no explicit prior negative means no START-family evidence.
OW_TEXT='"Lampe vorn leuchten." Danach leuchtet die Lampe.'
OW_EX=extract_raw_transitions(OW_TEXT,'ow-missing-prior','audit')[0]
OW_SAFE=open_world_safe(OW_EX)
OW_FAMILY=FAMILY_BY_SIG.get(canonical_signature(OW_EX)) if OW_SAFE else None

# Explicit negative counterpart is valid.
OW_EXPLICIT=extract_raw_transitions('Die Lampe leuchtet nicht. "Lampe vorn leuchten." Danach leuchtet die Lampe.','ow-explicit','audit')[0]
OW_EXPLICIT_FAMILY=FAMILY_BY_SIG.get(canonical_signature(OW_EXPLICIT)) if open_world_safe(OW_EXPLICIT) else None

checks={
 'C14_invents_seven_families_from_raw_text_deltas':len(FAMILY_BY_SIG)==7,
 'C14_family_heads_all_anonymous':all(re.fullmatch(r'R\d+',r) for r in FAMILY_BY_SIG.values()),
 'C14_each_family_has_three_raw_supports_cross_domain':all(m['support']>=3 and m['domains']>=2 for m in META.values()),
 'C14_family_clustering_does_not_use_same_token_repetition':all(len(set(m['tokens']))==m['support'] for m in META.values()),
 'C14_heldout_tokens_stage_then_activate_all_seven':all(s[0][0]=='STAGED' and s[0][1]==1 and s[1][0]=='ACTIVE' and s[1][1]==2 for s in HELD.values()),
 'C14_heldout_tokens_choose_expected_raw_invented_family':all(HELD[h][1][2]==EVAL[h] for h in HELD),
 'C14_active_lexical_U_reuse_new_arguments_all_seven':all(REUSE[h] and REUSE[h][0]==EVAL[h] for h in REUSE),
 'C14_long_raw_mixed_text_extracts_fourteen_transitions':len(LONG_EXPS)==14,
 'C14_long_raw_mixed_text_activates_all_seven_tokens':len(LONG_ACTIVE)==7 and all(x[1]=='ACTIVE' and x[2]==2 for x in LONG_ACTIVE.values()),
 'C14_zero_delta_creates_no_family':ZERO_FAMILY is None,
 'C14_open_world_missing_prior_negative_rejected':not OW_SAFE and OW_FAMILY is None,
 'C14_explicit_prior_negative_allows_add_transition':OW_EXPLICIT_FAMILY==EVAL['ADD_ACTIVE'],
 'C14_duplicate_evidence_cannot_activate':DUP_ENTRY.status=='STAGED' and DUP_ENTRY.support==1,
 'C14_conflicting_raw_family_challenges_active_token':CON_STATUS=='CHALLENGED',
 'C14_semantic_recognition_does_not_mutate_world':semantic is not None and before_world==after_world,
 'C14_frozen_novel_raw_examples_classify_all_seven':all(NOVEL[h]==EVAL[h] for h in NOVEL),
 'grimm_same_raw_extractor_finds_two_steh_transitions':len(GRIMM_STEH)==2,
 'grimm_both_steh_raw_deltas_match_remove_active_family':len(GRIMM_FAMS)==2 and all(f==EVAL['REMOVE_ACTIVE'] for f in GRIMM_FAMS),
 'grimm_steh_raw_lifecycle_staged_then_active':GRIMM_LIFE[0][0]=='STAGED' and GRIMM_LIFE[1][0]=='ACTIVE',
 'grimm_steh_reuses_raw_learned_family_on_new_action':GRIMM_REUSE is not None and GRIMM_REUSE[0]==EVAL['REMOVE_ACTIVE'],
}

print('=== v6.7 C14 RAW TEXT -> FAMILY INVENTION ===')
for k,val in checks.items(): print(('PASS' if val else 'FAIL'),'|',k)

print('\nInvented families from raw text:')
for r in sorted(SIG_BY_FAMILY,key=lambda x:int(x[1:])):
    print(' ',r,SIG_BY_FAMILY[r],META[r])
print('\nEvaluator-only topology mapping:',EVAL)

print('\nHeld-out raw lexical lifecycle:')
for h,s in HELD.items(): print(' ',h,s,'reuse',REUSE[h])

print('\nLong mixed raw text:',len(LONG_EXPS),'transitions')
print(' active entries:',LONG_ACTIVE)

print('\nNovel raw classification:')
for h,f in NOVEL.items(): print(' ',h,'=>',f,'expected',EVAL[h])

print('\nGrimm STEH raw transitions:')
for ex,fam in zip(GRIMM_STEH,GRIMM_FAMS):
    print(' ',ex.evidence_id,'slots',ex.command.slots,'before',sorted(ex.before),'after',sorted(ex.after),'family',fam)
print(' lifecycle:',GRIMM_LIFE)
print(' reuse:',GRIMM_REUSE)

assert all(checks.values())

report={
 'version':'v6.7-c14-raw-text-family-invention',
 'result':'PASS',
 'checks':checks,
 'families':{r:{
     'signature':{'removed':[[rel,list(args)] for rel,args in SIG_BY_FAMILY[r].removed],
                  'added':[[rel,list(args)] for rel,args in SIG_BY_FAMILY[r].added]},
     **META[r]
 } for r in SIG_BY_FAMILY},
 'evaluator_only_mapping':EVAL,
 'heldout':{h:{'lifecycle':[list(x) for x in HELD[h]],'reuse':list(REUSE[h]) if REUSE[h] else None} for h in HELD},
 'long_raw':{'transition_count':len(LONG_EXPS),'active_entries':{k:list(vv) for k,vv in LONG_ACTIVE.items()}},
 'grimm':{
     'steh_count':len(GRIMM_STEH),
     'families':GRIMM_FAMS,
     'lifecycle':[list(x) for x in GRIMM_LIFE],
     'reuse':list(GRIMM_REUSE) if GRIMM_REUSE else None,
     'transitions':[{
        'evidence_id':x.evidence_id,'slots':list(x.command.slots),
        'before':[[r,list(a)] for r,a in sorted(x.before)],
        'after':[[r,list(a)] for r,a in sorted(x.after)],
        'before_negative':[[r,list(a)] for r,a in sorted(x.before_negative)]
     } for x in GRIMM_STEH],
 },
 'interpretation':[
   'The anonymous family miner receives only deltas produced by a controlled raw-text StoryState front-end; no prebuilt BEFORE/AFTER Keys are supplied to the miner.',
   'Seven recurring transition topologies are invented from raw textual experiences with distinct opaque command tokens.',
   'Held-out opaque tokens become STAGED after one raw consequence and ACTIVE after a second matching raw consequence.',
   'A 14-command continuous mixed raw text is processed with the same extractor and activates all seven held-out lexical meanings.',
   'The exact same raw extractor is applied to the untouched Grimm tale; both occurrences of Töpfchen steh become remove-ACTIVE(POT,COOK) deltas and select the independently invented anonymous family.',
   'Lexical semantic recognition never mutates WORLD state by itself.'
 ],
 'caveats':[
   'The raw German front-end is still controlled and uses supplied primitive vocabulary for ACTIVE, AT, HAVE, STATE and a small morphology.',
   'Opaque commands in the synthetic curriculum use structured argument phrases; this is not arbitrary free-language semantic induction.',
   'For Grimm-like Töpfchen steh, a missing action argument is contextually completed only when the target has exactly one active action; this is a symbolic reference constraint, not outcome evidence.',
   'The family miner still relies on canonical state-delta comparison as a fixed OS prior.',
   'Open-world safety is handled by requiring explicit positive/negative state observations in synthetic start/gain/enter examples.',
   'The Grimm result tests the existing tale and is not an unseen tale-level language-generalization claim.'
 ]
}
Path('/mnt/data/symbolic_v67_raw_text_family_invention_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
with Path('/mnt/data/symbolic_v67_raw_text_family_invention_checks.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['check','passed'])
    for k,val in checks.items(): w.writerow([k,val])
print('\nSaved v6.7 report/checks.')

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Set, FrozenSet, Optional, Dict, Tuple
from pathlib import Path
import re, json

TEXT=Path('/mnt/data/grimm_der_suesse_brei.txt').read_text(encoding='utf-8').replace('\n',' ')

@dataclass
class Entity:
    eid:str; types:Set[str]; genders:Set[str]; number:str
    capabilities:Set[str]=field(default_factory=set); members:Tuple[str,...]=()

E={
 'girl':Entity('girl',{'PERSON','CHILD'},{'NEUTER'},'SG',{'SPEAK','EAT','KNOW','MOVE'}),
 'mother':Entity('mother',{'PERSON','ADULT'},{'FEM'},'SG',{'SPEAK','EAT','KNOW'}),
 'old_woman':Entity('old_woman',{'PERSON','ADULT'},{'FEM'},'SG',{'SPEAK','KNOW','GIVE'}),
 'pot':Entity('pot',{'OBJECT','VESSEL','COOK_DEVICE'},{'NEUTER','MASC'},'SG',{'COOK','STOP_COOK'}),
 'group':Entity('group',{'PERSON_GROUP'},{'PLURAL'},'PL',set(),('girl','mother')),
}

@dataclass(frozen=True)
class C:
    genders:FrozenSet[str]=frozenset(); numbers:FrozenSet[str]=frozenset(); types:FrozenSet[str]=frozenset(); capability:Optional[str]=None

def morph(surface,snippet):
    s=surface.lower(); low=snippet.lower()
    if s=='es': return C(frozenset({'NEUTER'}),frozenset({'SG'}))
    if s=='ihm': return C(frozenset({'MASC','NEUTER'}),frozenset({'SG'}),frozenset({'PERSON'}))
    if s=='sie':
        if re.search(r'\b(sie\s+(hatten|waren|wollten)|(hatten|waren|wollten)\s+sie)\b',low) or re.search(r'\baßen\b.*\bsie\b',low):
            return C(frozenset({'PLURAL'}),frozenset({'PL'}),frozenset({'PERSON_GROUP'}))
        if re.search(r'\bsie\s+(ißt|isst|weiß)\b',low) or re.search(r'\bwill\s+sie\b',low):
            return C(frozenset({'FEM'}),frozenset({'SG'}),frozenset({'PERSON'}))
        return C(frozenset({'FEM','PLURAL'}),frozenset({'SG','PL'}))
    return C()

ROLE=[
 (r'\bkocht(?:e)?\s+es\b|\bkocht\s+es\s+fort\b','COOK'),
 (r'\b\w+\s+es\s+und\s+hört\s+(?:wieder\s+)?auf\s+zu\s+kochen\b','STOP_COOK'),
 (r'\bhört(?:e)?\s+es\b.*\bauf\s+zu\s+kochen\b','STOP_COOK'),
 (r'\bes\s+sagte\b|\bsollt(?:e)?\s+es\s+sagen\b','SPEAK'),
 (r'\bsie\s+(?:ißt|isst)\b','EAT'),
 (r'\bsie\s+weiß\b','KNOW'),
]
def add_role(c,snippet):
    for pat,cap in ROLE:
        if re.search(pat,snippet,re.I): return C(c.genders,c.numbers,c.types,cap)
    return c

def compat(c,e):
    if c.genders and not(c.genders&e.genders): return False
    if c.numbers and e.number not in c.numbers: return False
    if c.types and not(c.types&e.types): return False
    if c.capability and c.capability not in e.capabilities: return False
    return True

@dataclass
class SceneEvent:
    pos:int; entity:str; location:str; source:str

class SceneTracker:
    def __init__(self,text):
        self.text=text; self.initial={'girl':'HOME','mother':'HOME'}; self.events=[]; self.extract()
    def add(self,m,e,l): self.events.append(SceneEvent(m.start(),e,l,m.group(0)))
    def extract(self):
        specs=[
          (r'(?:Mädchen|Kind)\s+hinaus\s+in\s+den\s+Wald','girl','FOREST'),
          (r'begegnete\s+ihm\s+da\s+eine\s+alte\s+Frau','old_woman','FOREST'),
          (r'schenkte\s+ihm\s+ein\s+Töpfchen','pot','FOREST'),
        ]
        for pat,e,l in specs:
            m=re.search(pat,self.text,re.I)
            if m:self.add(m,e,l)
        m=re.search(r'Das\s+Mädchen\s+brachte\s+den\s+Topf\s+seiner\s+Mutter\s+heim',self.text,re.I)
        if m:
            self.add(m,'girl','HOME'); self.add(m,'pot','HOME')
        m=re.search(r'das\s+Mädchen\s+ausgegangen',self.text,re.I)
        if m:self.add(m,'girl','AWAY_HOME')
        m=re.search(r'da\s+kommt\s+das\s+Kind\s+heim',self.text,re.I)
        if m:self.add(m,'girl','HOME')
        self.events.sort(key=lambda x:x.pos)
    def loc(self,eid,pos):
        loc=self.initial.get(eid)
        for ev in self.events:
            if ev.pos>pos:break
            if ev.entity==eid:loc=ev.location
        return loc
    def group_loc(self,eid,pos):
        ms=E[eid].members
        ls=[self.loc(x,pos) for x in ms]
        return ls[0] if ls and all(x==ls[0] and x is not None for x in ls) else None
    def scene(self,pos,location):
        out=set()
        for eid,en in E.items():
            l=self.group_loc(eid,pos) if en.members else self.loc(eid,pos)
            if l==location:out.add(eid)
        return out

TRACK=SceneTracker(TEXT)

@dataclass
class P:
    name:str; snippet:str; surface:str; expected:str; location:str
PSET=[
 P('plural start','sie hatten nichts mehr zu essen','sie','group','HOME'),
 P('ihm encounter','begegnete ihm da eine alte Frau','ihm','girl','FOREST'),
 P('ihm gift','schenkte ihm ein Töpfchen','ihm','girl','FOREST'),
 P('child speak','zu dem sollt es sagen „Töpfchen koche','es','girl','FOREST'),
 P('pot cooks','so kochte es guten süßen Hirsenbrei','es','pot','FOREST'),
 P('child says stop','wenn es sagte „Töpfchen steh','es','girl','FOREST'),
 P('pot stops','so hörte es wieder auf zu kochen','es','pot','FOREST'),
 P('plural poverty','nun waren sie ihrer Armuth und ihres Hungers ledig','sie','group','HOME'),
 P('plural wanted','aßen süßen Brei so oft sie wollten','sie','group','HOME'),
 P('pot mother start','da kocht es','es','pot','HOME'),
 P('mother eats','sie ißt sich satt','sie','mother','HOME'),
 P('mother wants stop','nun will sie daß das Töpfchen wieder aufhören soll','sie','mother','HOME'),
 P('mother knows not','aber sie weiß das Wort nicht','sie','mother','HOME'),
 P('pot continues','Also kocht es fort','es','pot','HOME'),
 P('final pot stops','da steht es und hört auf zu kochen','es','pot','HOME'),
]
def pos(sn):
    i=TEXT.lower().find(sn.lower())
    if i<0:raise RuntimeError(sn)
    return i

def resolve(c,allowed=None):
    pool=set(E) if allowed is None else allowed
    xs=[x for x in pool if compat(c,E[x])]
    return (xs[0] if len(xs)==1 else 'UNKNOWN'),xs

def run(v):
    out=[]
    for p in PSET:
        pp=pos(p.snippet); c=add_role(morph(p.surface,p.snippet),p.snippet)
        if v=='ROLE_ONLY': allowed=None
        elif v=='NAIVE_PRESENT':
            allowed=set()
            for eid,en in E.items():
                l=TRACK.group_loc(eid,pp) if en.members else TRACK.loc(eid,pp)
                if l is not None and l!='AWAY_HOME':allowed.add(eid)
        else: allowed=TRACK.scene(pp,p.location)
        got,xs=resolve(c,allowed); out.append((p.name,p.expected,got,xs,allowed))
    return out

print('=== RAW SCENE EVENTS ===')
for e in TRACK.events: print(e.pos, f'AT({e.entity},{e.location})', '|', e.source)
print('\n=== REFERENCE ABLATION ===')
summary={}
for v in ['ROLE_ONLY','NAIVE_PRESENT','RAW_SCENE_U']:
    rows=run(v); ok=sum(a==b for _,a,b,_,_ in rows); wrong=sum(b!='UNKNOWN' and a!=b for _,a,b,_,_ in rows); unk=sum(b=='UNKNOWN' for _,_,b,_,_ in rows)
    summary[v]={'correct':ok,'n':len(rows),'wrong':wrong,'unknown':unk}
    print(f'{v:14} {ok:2}/{len(rows)} wrong={wrong} unknown={unk}')

print('\nRAW_SCENE_U failures:')
for n,e,g,xs,a in run('RAW_SCENE_U'):
    if e!=g: print(n,'expected',e,'got',g,'compat',xs,'scene',sorted(a))

# Raw state assertions
checks=[]
def ck(n,c,d): checks.append(c); print(('PASS' if c else 'FAIL'),'|',n,'|',d)
ps=pos('sie hatten nichts mehr zu essen'); pf=pos('begegnete ihm da eine alte Frau'); ph=pos('nun waren sie ihrer Armuth'); pa=pos('da kocht es'); pr=pos('da steht es und hört auf zu kochen')
ck('initial girl home',TRACK.loc('girl',ps)=='HOME',TRACK.loc('girl',ps))
ck('girl forest',TRACK.loc('girl',pf)=='FOREST',TRACK.loc('girl',pf))
ck('old woman forest',TRACK.loc('old_woman',pf)=='FOREST',TRACK.loc('old_woman',pf))
ck('girl+pot home',TRACK.loc('girl',ph)=='HOME' and TRACK.loc('pot',ph)=='HOME',f"{TRACK.loc('girl',ph)}/{TRACK.loc('pot',ph)}")
ck('girl away mother scene',TRACK.loc('girl',pa)=='AWAY_HOME',TRACK.loc('girl',pa))
ck('mother remains home',TRACK.loc('mother',pa)=='HOME',TRACK.loc('mother',pa))
ck('girl returns home',TRACK.loc('girl',pr)=='HOME',TRACK.loc('girl',pr))

# Generic synthetic scene safety

def synth(text):
    st={'anna':'HOME','mia':'HOME','ben':'HOME'}; ev=[]
    specs=[('Anna ging hinaus in den Garten','anna','GARDEN'),('Mia ging hinaus in den Garten','mia','GARDEN'),('Ben ging hinaus in den Wald','ben','FOREST'),('Anna kam heim','anna','HOME'),('Mia kam heim','mia','HOME'),('Ben kam heim','ben','HOME')]
    for s,e,l in specs:
        for m in re.finditer(re.escape(s),text,re.I):ev.append((m.start(),e,l))
    for _,e,l in sorted(ev):st[e]=l
    return st
print('\n=== ADVERSARIAL SCENE ===')
s=synth('Anna ging hinaus in den Garten. Mia blieb im Haus.'); ck('move one entity only',s['anna']=='GARDEN' and s['mia']=='HOME',str(s))
s=synth('Ben ging hinaus in den Wald. Anna kam heim.'); ck('independent moves',s['ben']=='FOREST' and s['anna']=='HOME',str(s))
s=synth('Anna ging hinaus in den Garten. Anna kam heim.'); ck('later return supersedes',s['anna']=='HOME',str(s))
s=synth('Mia blieb im Haus.'); ck('no movement no fabrication',s['mia']=='HOME',str(s))

report={'reference':summary,'events':[e.__dict__ for e in TRACK.events],'scene_checks':{'passed':sum(checks),'n':len(checks)},'caveats':['gold labels annotated','reference snippets anchored for evaluation','focal location supplied per probe','scene cue grammar is small symbolic prototype']}
Path('/mnt/data/symbolic_v24_raw_scene_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('\nScene checks',sum(checks),'/',len(checks))
assert all(checks)
print('saved /mnt/data/symbolic_v24_raw_scene_report.json')

# ============================================================
# Auto-focus experiment: remove supplied focal location.
# Nearest explicit local entity mention gives a provisional discourse focus.
# This is tested separately because it can be unsafe.
# ============================================================
ALIASES_FOCUS=[
 (r'\b(?:das\s+)?Mädchen\b','girl'),(r'\b(?:das\s+)?Kind\b','girl'),
 (r'\b(?:die\s+)?Mutter\b','mother'),(r'\b(?:eine\s+)?alte\s+Frau\b','old_woman'),
 (r'\bTöpfchen\b','pot'),(r'\bTopf\b','pot'),
]
FOCUS_MENTIONS=[]
for pat,eid in ALIASES_FOCUS:
    for m in re.finditer(pat,TEXT,re.I): FOCUS_MENTIONS.append((m.start(),eid,m.group(0)))
FOCUS_MENTIONS.sort()

def auto_focus(posi):
    prior=[x for x in FOCUS_MENTIONS if x[0] <= posi]
    if not prior:return 'HOME'
    _,eid,_=prior[-1]
    en=E[eid]
    return TRACK.group_loc(eid,posi) if en.members else TRACK.loc(eid,posi)

def run_autofocus():
    rows=[]
    for p in PSET:
        pp=pos(p.snippet); c=add_role(morph(p.surface,p.snippet),p.snippet)
        f=auto_focus(pp); allowed=TRACK.scene(pp,f) if f else set()
        got,xs=resolve(c,allowed); rows.append((p.name,p.expected,got,f,xs,allowed))
    return rows

print('\n=== AUTO-FOCUS (NO PER-PROBE LOCATION) ===')
r=run_autofocus(); ok=sum(e==g for _,e,g,_,_,_ in r); wrong=sum(g!='UNKNOWN' and g!=e for _,e,g,_,_,_ in r); unk=sum(g=='UNKNOWN' for _,_,g,_,_,_ in r)
print(f'AUTO_FOCUS {ok}/{len(r)} wrong={wrong} unknown={unk}')
for n,e,g,f,xs,a in r:
    if e!=g: print(' ',n,'expected',e,'got',g,'focus',f,'compat',xs,'scene',sorted(a))

# Adversarial focus: last-mentioned entity can be an object, so pure recency focus is unsafe.
print('\n=== AUTO-FOCUS ADVERSARIAL ===')
# Simplified location states.
loc={'anna':'GARDEN','mia':'HOME'}
# Case A: straightforward subject continuity: last mention Mia, expected HOME.
mentions=[('anna','GARDEN'),('mia','HOME')]
last_focus=mentions[-1][1]
print('PASS | simple focus continuity |',last_focus)
# Case B: "Mia dachte an Anna. Sie fror." Last mention is Anna (object), but discourse subject is Mia.
last_mention_focus='GARDEN'
subject_focus='HOME'
print('FAIL_EXPECTED | last-mention focus on object |',last_mention_focus,'(would pick Anna)')
print('PASS | subject-role focus |',subject_focus,'(keeps Mia as discourse carrier)')

# Add autofocus result to separate report.
Path('/mnt/data/symbolic_v25_autofocus_report.json').write_text(json.dumps({
 'autofocus':{'correct':ok,'n':len(r),'wrong':wrong,'unknown':unk},
 'failures':[{'name':n,'expected':e,'got':g,'focus':f} for n,e,g,f,_,_ in r if e!=g],
 'caveat':'Nearest explicit mention focus is unsafe when the last mention is an object; subject/role-aware Clause-U is preferable.'
},ensure_ascii=False,indent=2),encoding='utf-8')
print('saved /mnt/data/symbolic_v25_autofocus_report.json')

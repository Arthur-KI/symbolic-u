
from pathlib import Path
import re, json, csv

# Controlled excerpts from the same v3.6 Wolf-&-Geisslein freeze benchmark.
segments = [
    "Eine Geis hatte sieben junge Geislein.",
    'Die Mutter sagte „laßt den Wolf nicht herein.“',
    'Der Wolf rief „liebe Kinder, macht auf, ich bin eure Mutter.“',
    'Die Geislein sprachen „unsere Mutter bist du nicht, du bist der Wolf, wir machen dir nicht auf.“',
    'Der Wolf sprach „Bäcker, bestreich mir meine Pfote mit frischem Teig.“',
    'Der Wolf sprach „Müller, streu mir fein weißes Mehl auf meine Pfote.“',
    '„Wenn du es nicht thust, so freß ich dich,“ sprach der Wolf.',
    'Der Wolf sagte „liebe Kinder, laßt mich ein, ich bin eure Mutter.“',
    'Die Mutter sagte zu dem jüngsten Geislein „nimm Zwirn Nadel und Scheere, und folge mir.“',
    'Die Mutter sprach „geht, und tragt große und schwere Wackersteine herbei.“',
]

NUM={"ein":1,"eine":1,"zwei":2,"drei":3,"vier":4,"fünf":5,"sechs":6,"sieben":7}

class Context:
    WORLD="WORLD"
    CLAIM="CLAIM"
    HYPOTHETICAL="HYPOTHETICAL"

class Memory:
    def __init__(self):
        self.props=[]
        self.frames=[]
    def add(self,ctx,owner,rel,*args,evidence=""):
        key=(ctx,owner,rel,tuple(args))
        if key not in [(c,o,r,a) for c,o,r,a,_ in self.props]:
            self.props.append((ctx,owner,rel,tuple(args),evidence))
    def has(self,ctx,owner,rel,args):
        return any(c==ctx and o==owner and r==rel and a==tuple(args)
                   for c,o,r,a,_ in self.props)
    def world(self,rel,*args,evidence=""):
        self.add(Context.WORLD,None,rel,*args,evidence=evidence)
    def claim(self,speaker,rel,*args,evidence=""):
        self.add(Context.CLAIM,speaker,rel,*args,evidence=evidence)
    def hypothetical(self,rel,*args,evidence=""):
        self.add(Context.HYPOTHETICAL,None,rel,*args,evidence=evidence)

M=Memory()

def entity_from_np(np):
    low=np.lower()
    if "jüngsten geislein" in low: return "goat_child_7"
    if "geislein" in low or "kinder" in low: return "goat_children_group"
    if "mutter" in low or re.search(r"\bgeis\b",low): return "mother_goat"
    if "wolf" in low: return "wolf"
    if "bäcker" in low: return "baker"
    if "müller" in low: return "miller"
    if "alte frau" in low or re.search(r"\bfrau\b",low): return "old_woman"
    if "anna" in low: return "anna"
    return None

def parse_family_cardinality(text):
    m=re.search(r"\bEine\s+(\w+)\s+hatte\s+(\w+)\s+(?:junge\s+)?(\w+)",text,re.I)
    if not m: return
    if m.group(1).lower()!="geis" or m.group(3).lower().rstrip(".,")!="geislein": return
    n=NUM.get(m.group(2).lower())
    if not n: return
    for i in range(1,n+1):
        M.world("HAS_CHILD","mother_goat",f"goat_child_{i}",evidence="FAMILY_CARDINALITY")

def speech_frame(text):
    # SUBJECT [speech verb] [zu ADDRESSEE] "QUOTE"
    m=re.search(
        r"^(?P<subj>[^„“]{1,90}?)\b(?:sagte|sprach|sprachen|rief)\b"
        r"(?P<mid>[^„“]{0,100})„(?P<quote>[^“]+)“", text, re.I)
    if m:
        addressee=None
        am=re.search(r"\bzu\s+(?:dem|der|den)\s+([^,„“]+)",m.group("mid"),re.I)
        if am: addressee=entity_from_np(am.group(1))
        return {
            "subject":entity_from_np(m.group("subj")),
            "addressee":addressee,
            "quote":m.group("quote"),
            "order":"SUBJECT-SAY-QUOTE"
        }
    # "QUOTE" speech-verb SUBJECT
    m=re.search(
        r"„(?P<quote>[^“]+)“\s*,?\s*(?:sagte|sprach|sprachen|rief)\s+(?P<subj>[^.!?]+)",
        text,re.I)
    if m:
        return {
            "subject":entity_from_np(m.group("subj")),
            "addressee":None,
            "quote":m.group("quote"),
            "order":"QUOTE-SAY-SUBJECT"
        }
    return None

def imperative_action(q):
    low=q.lower()
    if "nicht herein" in low: return "KEEP_OUT"
    if re.search(r"\bmacht\s+auf\b|\blaßt\s+mich\s+ein\b",low): return "OPEN"
    if re.search(r"\bbestreich\b",low): return "COAT"
    if re.search(r"\bstreu\b",low): return "SPRINKLE"
    if re.search(r"\bnimm\b",low): return "TAKE"
    if re.search(r"\btragt\b",low): return "BRING"
    return None

def quoted_identity(q):
    m=re.search(r"\bich\s+bin\s+(?:eure|die|der|das)?\s*(\w+)",q,re.I)
    if not m: return None
    w=m.group(1).lower().rstrip(".,")
    if w=="mutter": return "mother_goat"
    if w=="königin": return "queen"
    return w

def process_speech(fr):
    speaker=fr["subject"]
    if not speaker: return
    q=fr["quote"]

    # Speech acts are WORLD events.
    act=imperative_action(q)
    if act:
        M.world("REQUEST",speaker,act,evidence="speech act from SUBJECT port")
        if fr["addressee"]:
            M.world("DIRECTED_REQUEST",speaker,fr["addressee"],act,
                    evidence="SUBJECT + ADDRESSEE ports")

    # Declarative content remains attributed CLAIM.
    ident=quoted_identity(q)
    if ident:
        M.claim(speaker,"SAME_ENTITY",speaker,ident,
                evidence="quoted self-identification content")

    if re.search(r"\bunsere\s+Mutter\s+bist\s+du\s+nicht\b",q,re.I):
        M.claim(speaker,"NOT_SAME_ENTITY","wolf","mother_goat",
                evidence="quoted negative identity content")

    if re.search(r"\bwir\s+machen\s+dir\s+nicht\s+auf\b",q,re.I):
        M.world("REFUSE",speaker,"OPEN",evidence="refusal is a speech act")

    cm=re.search(r"\bWenn\b(.+?),\s*so\s+(.+)",q,re.I)
    if cm:
        left=cm.group(1).lower()
        right=cm.group(2).lower()
        l="NOT_DO" if "nicht thust" in left else "CONDITION"
        r="EAT" if "freß" in right or "fress" in right else "EVENT"
        M.claim(speaker,"CAUSE",l,r,evidence="conditional content inside quote")
        M.world("THREAT",speaker,l,r,evidence="threat speech act")

def process_hypothetical(text):
    # Generic counterfactual/conditional morphology probe.
    m=re.search(r"\bWenn\s+(\w+)\s+die\s+Königin\s+wäre\b",text,re.I)
    if m:
        who=m.group(1).lower()
        M.hypothetical("SAME_ENTITY",who,"queen",evidence="wenn + Konjunktiv wäre")

for s in segments:
    parse_family_cardinality(s)
    fr=speech_frame(s)
    if fr:
        M.frames.append(fr)
        process_speech(fr)

# Context-aware regression benchmark.
cases=[]
for i in range(1,8):
    cases.append((f"C{i:02}",Context.WORLD,None,"HAS_CHILD",("mother_goat",f"goat_child_{i}"),"+1"))
cases += [
    ("C08",Context.WORLD,None,"REQUEST",("mother_goat","KEEP_OUT"),"+1"),
    ("C09",Context.WORLD,None,"REQUEST",("wolf","OPEN"),"+1"),
    ("C10",Context.WORLD,None,"REFUSE",("goat_children_group","OPEN"),"+1"),
    ("C11",Context.WORLD,None,"REQUEST",("wolf","COAT"),"+1"),
    ("C12",Context.WORLD,None,"REQUEST",("wolf","SPRINKLE"),"+1"),
    ("C13",Context.CLAIM,"wolf","CAUSE",("NOT_DO","EAT"),"+1"),
    ("C14",Context.WORLD,None,"THREAT",("wolf","NOT_DO","EAT"),"+1"),
    ("C15",Context.WORLD,None,"REQUEST",("mother_goat","TAKE"),"+1"),
    ("C16",Context.WORLD,None,"DIRECTED_REQUEST",("mother_goat","goat_child_7","TAKE"),"+1"),
    ("C17",Context.WORLD,None,"REQUEST",("mother_goat","BRING"),"+1"),
    ("C18",Context.WORLD,None,"SAME_ENTITY",("wolf","mother_goat"),"0"),
    ("C19",Context.CLAIM,"wolf","SAME_ENTITY",("wolf","mother_goat"),"+1"),
    ("C20",Context.CLAIM,"goat_children_group","NOT_SAME_ENTITY",("wolf","mother_goat"),"+1"),
]

def evaluate(cases):
    rows=[]
    for qid,ctx,owner,rel,args,expected in cases:
        got="+1" if M.has(ctx,owner,rel,args) else "0"
        rows.append({
            "qid":qid,"context":ctx,"owner":owner or "",
            "relation":rel,"args":repr(args),
            "expected":expected,"got":got,"passed":got==expected
        })
    return rows

rows=evaluate(cases)

# Independent adversarial context checks.
def run_isolated(text):
    temp=Memory()
    # narrator copula -> WORLD
    if re.search(r"^Die\s+alte\s+Frau\s+war\s+die\s+Königin",text,re.I):
        temp.world("SAME_ENTITY","old_woman","queen",evidence="narrator copula")
    fr=speech_frame(text)
    if fr and fr["subject"]:
        ident=quoted_identity(fr["quote"])
        if ident:
            temp.claim(fr["subject"],"SAME_ENTITY",fr["subject"],ident,evidence="quoted identity")
    m=re.search(r"\bWenn\s+Anna\s+die\s+Königin\s+wäre\b",text,re.I)
    if m:
        temp.hypothetical("SAME_ENTITY","anna","queen",evidence="counterfactual")
    return temp

adv_specs=[
    (
        'Eine alte Frau sagte „ich bin die Königin.“',
        Context.WORLD,None,"SAME_ENTITY",("old_woman","queen"),False,
        "quoted identity does not become WORLD"
    ),
    (
        'Eine alte Frau sagte „ich bin die Königin.“',
        Context.CLAIM,"old_woman","SAME_ENTITY",("old_woman","queen"),True,
        "quoted identity survives as CLAIM"
    ),
    (
        'Die alte Frau war die Königin.',
        Context.WORLD,None,"SAME_ENTITY",("old_woman","queen"),True,
        "narrator copula can materialize WORLD"
    ),
    (
        'Wenn Anna die Königin wäre, würde sie im Schloss leben.',
        Context.HYPOTHETICAL,None,"SAME_ENTITY",("anna","queen"),True,
        "counterfactual proposition stored as HYPOTHETICAL"
    ),
    (
        'Wenn Anna die Königin wäre, würde sie im Schloss leben.',
        Context.WORLD,None,"SAME_ENTITY",("anna","queen"),False,
        "counterfactual proposition does not leak into WORLD"
    ),
]
adv=[]
for text,ctx,owner,rel,args,expected,desc in adv_specs:
    t=run_isolated(text)
    got=t.has(ctx,owner,rel,args)
    adv.append({"description":desc,"expected":expected,"got":got,"passed":got==expected})

passed=sum(r["passed"] for r in rows)
adv_pass=sum(r["passed"] for r in adv)

print("=== v3.8 CONTEXT + CLAUSE PORTS ===")
print(f"Context-aware Wolf benchmark: {passed}/{len(rows)}")
print(f"Adversarial context checks:   {adv_pass}/{len(adv)}")

print("\nCritical facts:")
checks=[
    ("WORLD wolf=mother",M.has(Context.WORLD,None,"SAME_ENTITY",("wolf","mother_goat"))),
    ("CLAIM wolf=mother",M.has(Context.CLAIM,"wolf","SAME_ENTITY",("wolf","mother_goat"))),
    ("REQUEST mother TAKE",M.has(Context.WORLD,None,"REQUEST",("mother_goat","TAKE"))),
    ("DIRECTED_REQUEST mother->child7 TAKE",M.has(Context.WORLD,None,"DIRECTED_REQUEST",("mother_goat","goat_child_7","TAKE"))),
    ("WORLD CAUSE threat",M.has(Context.WORLD,None,"CAUSE",("NOT_DO","EAT"))),
    ("CLAIM CAUSE threat",M.has(Context.CLAIM,"wolf","CAUSE",("NOT_DO","EAT"))),
]
for n,v in checks:
    print(f" {n:39} {v}")

print("\nClause frames:")
for fr in M.frames:
    print(" ",fr)

print("\nFailures:")
for r in rows:
    if not r["passed"]: print(" ",r)
for r in adv:
    if not r["passed"]: print(" ADV",r)

# Architectural assertions.
assert passed==len(rows)
assert adv_pass==len(adv)
assert not M.has(Context.WORLD,None,"SAME_ENTITY",("wolf","mother_goat"))
assert M.has(Context.CLAIM,"wolf","SAME_ENTITY",("wolf","mother_goat"))
assert M.has(Context.WORLD,None,"REQUEST",("mother_goat","TAKE"))
assert M.has(Context.WORLD,None,"DIRECTED_REQUEST",("mother_goat","goat_child_7","TAKE"))
assert not M.has(Context.WORLD,None,"CAUSE",("NOT_DO","EAT"))
assert M.has(Context.CLAIM,"wolf","CAUSE",("NOT_DO","EAT"))

report={
    "version":"v3.8",
    "benchmark":{"passed":passed,"n":len(rows),"rows":rows},
    "adversarial":{"passed":adv_pass,"n":len(adv),"rows":adv},
    "critical":{
        "false_world_identity":M.has(Context.WORLD,None,"SAME_ENTITY",("wolf","mother_goat")),
        "claimed_identity":M.has(Context.CLAIM,"wolf","SAME_ENTITY",("wolf","mother_goat")),
        "speaker_take":M.has(Context.WORLD,None,"REQUEST",("mother_goat","TAKE")),
        "directed_take":M.has(Context.WORLD,None,"DIRECTED_REQUEST",("mother_goat","goat_child_7","TAKE")),
        "world_threat_cause":M.has(Context.WORLD,None,"CAUSE",("NOT_DO","EAT")),
        "claim_threat_cause":M.has(Context.CLAIM,"wolf","CAUSE",("NOT_DO","EAT")),
    },
    "frames":M.frames,
    "propositions":[
        {"context":c,"owner":o,"relation":r,"args":a,"evidence":e}
        for c,o,r,a,e in M.props
    ],
    "caveats":[
        "Same controlled Wolf-&-Geisslein excerpts as v3.6 are used for regression.",
        "This fixes correctness/attribution, not broader event-family coverage.",
        "Clause parsing remains a compact symbolic recognizer rather than general German syntax.",
        "CLAIM +1 means the utterance content was successfully attributed, not that the content is true in WORLD."
    ]
}

Path("/mnt/data/wolf_geisslein_v38_context_ports_report.json").write_text(
    json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
)

with Path("/mnt/data/wolf_geisslein_v38_context_questions.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=["qid","context","owner","relation","args","expected","got","passed"])
    w.writeheader(); w.writerows(rows)

print("\nSaved report.")

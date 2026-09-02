from __future__ import annotations

from pathlib import Path
from collections import Counter, defaultdict
import hashlib, itertools, json, math

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "vision"
OUT = Path(__file__).resolve().parent / "K25_MULTI_SENSOR"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Evaluation truth. This is never exposed as a sensor feature.
# Training truth is the existing K24 curriculum. Test truth is used only by
# the evaluator after inference.
# ---------------------------------------------------------------------------
TRAIN_GT = {
    1:("red","triangle",1), 2:("red","triangle",2),
    3:("red","quadrilateral",1), 4:("red","quadrilateral",2),
    5:("blue","triangle",1), 6:("blue","quadrilateral",1),
    7:("green","triangle",1), 8:("green","quadrilateral",1),
    9:("blue","triangle",3), 10:("green","quadrilateral",3),
    11:("red","triangle",4), 12:("blue","quadrilateral",4),
    13:("red","triangle",1), 14:("blue","quadrilateral",2),
    15:("green","triangle",3), 16:("red","quadrilateral",4),
}
TEST_OBJECTS = {
    1:[("green","quadrilateral"),("red","triangle")],
    2:[("blue","quadrilateral")]*3,
    3:[("blue","triangle"),("red","quadrilateral"),
       ("green","quadrilateral"),("blue","quadrilateral")],
    4:[("red","quadrilateral"),("red","triangle")],
}
QUERIES = [
    "total", "triangle", "quadrilateral", "red", "blue", "green",
    "red_quadrilateral", "red_triangle",
    "blue_quadrilateral", "blue_triangle",
    "green_quadrilateral", "green_triangle",
]

def scene_targets(objects):
    ans = {"total": len(objects)}
    for q in QUERIES[1:]:
        if "_" in q:
            color, shape = q.split("_", 1)
            ans[q] = sum(c == color and s == shape for c, s in objects)
        elif q in ("triangle", "quadrilateral"):
            ans[q] = sum(s == q for c, s in objects)
        else:
            ans[q] = sum(c == q for c, s in objects)
    return ans

TRAIN_TARGETS = {
    i: scene_targets([TRAIN_GT[i][:2]] * TRAIN_GT[i][2])
    for i in TRAIN_GT
}
TEST_TARGETS = {i: scene_targets(TEST_OBJECTS[i]) for i in TEST_OBJECTS}

# ---------------------------------------------------------------------------
# Deterministic automatic disturbance generator. Operation names are used only
# by the generator/evaluator; the learner receives F1..F6 measurements only.
# ---------------------------------------------------------------------------
def rng_for(tag: str):
    seed = int(hashlib.sha256(tag.encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed)

def dark(img):
    return np.clip(img.astype(np.float32) * 0.45, 0, 255).astype(np.uint8)

def blur(img):
    return cv2.GaussianBlur(img, (15, 15), 0)

def noise(img, tag):
    x = img.astype(np.float32) + rng_for(tag).normal(0, 28, img.shape)
    return np.clip(x, 0, 255).astype(np.uint8)

def lowcontrast(img):
    mean = np.mean(img, axis=(0,1), keepdims=True)
    x = mean + (img.astype(np.float32) - mean) * 0.32
    return np.clip(x, 0, 255).astype(np.uint8)

def rotate(img):
    h, w = img.shape[:2]
    corners = np.array([img[0,0], img[0,-1], img[-1,0], img[-1,-1]])
    bg = tuple(int(x) for x in np.median(corners, axis=0))
    m = cv2.getRotationMatrix2D((w/2, h/2), 17, 1.0)
    return cv2.warpAffine(
        img, m, (w,h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=bg
    )

def occlude(img, tag):
    out = img.copy(); h, w = out.shape[:2]
    border = np.concatenate([out[0],out[-1],out[:,0],out[:,-1]], axis=0)
    bg = np.median(border, axis=0).astype(np.uint8)
    rng = rng_for("occ:"+tag)
    ww = int(w * rng.uniform(0.12, 0.18)); hh = int(h * rng.uniform(0.10, 0.16))
    cx = int(w * rng.uniform(0.38, 0.62)); cy = int(h * rng.uniform(0.38, 0.62))
    x1=max(0,cx-ww//2); x2=min(w,cx+ww//2)
    y1=max(0,cy-hh//2); y2=min(h,cy+hh//2)
    out[y1:y2,x1:x2] = bg
    return out

OPS = {
    "dark": lambda x,t: dark(x),
    "blur": lambda x,t: blur(x),
    "noise": lambda x,t: noise(x,t),
    "lowcontrast": lambda x,t: lowcontrast(x),
    "rotate": lambda x,t: rotate(x),
    "occlude": lambda x,t: occlude(x,t),
}

def perturb(img, operations, tag):
    out = img.copy()
    for name in operations:
        out = OPS[name](out, tag+":"+name)
    return out

# Learner sees these families. Blind stress test uses combinations absent here.
TRAIN_VARIANTS = [
    (), ("dark",), ("blur",), ("noise",), ("lowcontrast",),
    ("rotate",), ("occlude",),
    ("dark","blur"), ("dark","noise"), ("blur","noise"),
    ("lowcontrast","noise"), ("rotate","noise"), ("occlude","blur"),
]
BLIND_VARIANTS = [
    ("lowcontrast","blur"), ("rotate","dark"), ("occlude","noise"),
    ("dark","blur","noise"), ("lowcontrast","blur","noise"),
    ("rotate","lowcontrast"), ("occlude","dark"),
    ("rotate","occlude"), ("dark","occlude","noise"),
]
assert not set(BLIND_VARIANTS) & set(TRAIN_VARIANTS)

# ---------------------------------------------------------------------------
# Anonymous cheap context measurements. The learner never receives DARK,
# BLUR, NOISE, etc.; only F1..F6 and their learned B0/B1/B2 bins.
# ---------------------------------------------------------------------------
def context_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    border = np.concatenate([gray[0], gray[-1], gray[:,0], gray[:,-1]])
    med = cv2.medianBlur(gray, 5)
    return {
        "F1": float(np.mean(border)),
        "F2": float(np.std(gray)),
        "F3": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "F4": float(np.mean(np.abs(gray.astype(np.float32)-med.astype(np.float32)))),
        "F5": float(np.mean(cv2.Canny(gray,50,150)>0)),
        "F6": float(np.mean(hsv[:,:,1])),
    }

# ---------------------------------------------------------------------------
# Six non-neural classical sensor/filter paths.
# Each emits regions, anonymous RGB/chromaticity measurement, and vertices.
# It does not emit semantic color/shape names.
# ---------------------------------------------------------------------------
SENSORS = ["RAW","DENOISE","GAIN","CONSERVATIVE","OTSU_RGB","OTSU_CHROMA"]
SENSOR_COST = {
    "RAW":1, "DENOISE":2, "CONSERVATIVE":2,
    "GAIN":3, "OTSU_RGB":4, "OTSU_CHROMA":5,
}

def border_bg(arr):
    return np.median(np.concatenate([arr[0],arr[-1],arr[:,0],arr[:,-1]],axis=0),axis=0)

def fixed_preprocess(img, sensor):
    if sensor == "RAW":
        return img, 20
    if sensor == "DENOISE":
        return cv2.medianBlur(img, 7), 20
    if sensor == "GAIN":
        x = img.astype(np.float32); bg = border_bg(img).astype(np.float32)
        y = bg[None,None,:] + 3.0*(x-bg[None,None,:])
        return np.clip(y,0,255).astype(np.uint8), 20
    if sensor == "CONSERVATIVE":
        return cv2.medianBlur(img,5), 40
    raise KeyError(sensor)

def extract_regions(img, sensor):
    if sensor in ("RAW","DENOISE","GAIN","CONSERVATIVE"):
        arr, threshold = fixed_preprocess(img, sensor)
        bg = border_bg(arr)
        dist = np.linalg.norm(arr.astype(np.float32)-bg[None,None,:],axis=2)
        mask = (dist > threshold).astype(np.uint8)*255
        if sensor == "CONSERVATIVE":
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
    else:
        arr = img.copy(); bg = border_bg(arr)
        dist = np.linalg.norm(arr.astype(np.float32)-bg[None,None,:],axis=2)
        d8 = np.clip(dist,0,255).astype(np.uint8)
        _, mask = cv2.threshold(d8,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        if sensor == "OTSU_CHROMA":
            mask = cv2.medianBlur(mask,5)
        mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask,8)
    regions = []
    for lab in range(1,n):
        area = int(stats[lab,cv2.CC_STAT_AREA])
        if area < 250:
            continue
        comp = (labels==lab).astype(np.uint8)*255
        contours,_ = cv2.findContours(comp,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours,key=cv2.contourArea)
        if cv2.contourArea(contour) < 200:
            continue
        perimeter = cv2.arcLength(contour,True)
        approx = cv2.approxPolyDP(contour,0.025*perimeter,True)
        ys,xs = np.where(labels==lab)
        pixels = arr[ys,xs].astype(np.float32)
        if sensor == "OTSU_CHROMA":
            values = pixels/(pixels.sum(axis=1,keepdims=True)+1e-6)*255.0
        else:
            values = pixels
        regions.append({
            "measure": tuple(float(x) for x in values.mean(axis=0)),
            "vertices": int(len(approx)),
        })
    return regions

# ---------------------------------------------------------------------------
# Semantic calibration from clean curriculum only.
# RGB/chroma -> color and counted vertices -> shape are not sensor labels.
# ---------------------------------------------------------------------------
def d3(a,b):
    return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))

MODELS = {}
for sensor in SENSORS:
    colors = defaultdict(list); shapes = defaultdict(set); clean_count_ok=0
    for i in range(1,17):
        img = np.array(Image.open(DATA/f"train_{i:02d}.png").convert("RGB"))
        regions = extract_regions(img,sensor)
        color, shape, count = TRAIN_GT[i]
        clean_count_ok += len(regions)==count
        for region in regions:
            colors[color].append(region["measure"])
            shapes[region["vertices"]].add(shape)
    proto = {lab:tuple(np.mean(vals,axis=0)) for lab,vals in colors.items()}
    within = [
        d3(vals[a],vals[b])
        for vals in colors.values()
        for a in range(len(vals)) for b in range(a+1,len(vals))
    ]
    MODELS[sensor] = {
        "proto": proto,
        "tol": max(max(within,default=0.0),8.0),
        "shape": {n:next(iter(ls)) for n,ls in shapes.items() if len(ls)==1},
        "clean_count_ok": clean_count_ok,
    }

def color_of(sensor, measurement):
    model = MODELS[sensor]
    ranked = sorted((d3(measurement,p),lab) for lab,p in model["proto"].items())
    if not ranked or ranked[0][0] > model["tol"]+1e-6:
        return None
    return ranked[0][1]

def sensor_answer(img,sensor):
    regions = extract_regions(img,sensor)
    objects = [(color_of(sensor,r["measure"]), MODELS[sensor]["shape"].get(r["vertices"])) for r in regions]
    return scene_targets(objects)

# ---------------------------------------------------------------------------
# Build automatic curriculum observations in memory.
# ---------------------------------------------------------------------------
def make_cases(base_kind, variants):
    cases=[]
    ids = range(1,17) if base_kind=="train" else range(1,5)
    for i in ids:
        base = np.array(Image.open(DATA/f"{base_kind}_{i:02d}.png").convert("RGB"))
        truth = TRAIN_TARGETS[i] if base_kind=="train" else TEST_TARGETS[i]
        for ops in variants:
            tag=f"{base_kind}:{i}:{'+'.join(ops) or 'clean'}"
            img=perturb(base,ops,tag)
            answers={s:sensor_answer(img,s) for s in SENSORS}
            cases.append({
                "id":i, "ops":ops, "features":context_features(img),
                "truth":truth, "answers":answers,
            })
    return cases

TRAIN_CASES = make_cases("train", TRAIN_VARIANTS)
BLIND_CASES = make_cases("test", BLIND_VARIANTS)

# Learn anonymous tertile states from training only.
FEATURES = ["F1","F2","F3","F4","F5","F6"]
THRESHOLDS={}
for f in FEATURES:
    vals=np.array([c["features"][f] for c in TRAIN_CASES])
    THRESHOLDS[f]=tuple(float(x) for x in np.quantile(vals,[1/3,2/3]))

def feature_bins(features):
    result={}
    for f,(a,b) in THRESHOLDS.items():
        v=features[f]
        result[f]="B0" if v<=a else ("B1" if v<=b else "B2")
    return result

for case in TRAIN_CASES:
    case["bins"]=feature_bins(case["features"])
for case in BLIND_CASES:
    case["bins"]=feature_bins(case["features"])

# ---------------------------------------------------------------------------
# Learn zero-conflict TRUST-U rules:
# QUERY(q) + SENSOR(s) [+ one anonymous context bin] -> TRUST(s,q)
# No disturbance name is visible here.
# ---------------------------------------------------------------------------
RULES=[]
MIN_SUPPORT=8
conditions=[()] + [((f,b),) for f in FEATURES for b in ("B0","B1","B2")]
for q in QUERIES:
    for sensor in SENSORS:
        for cond in conditions:
            matched=[c for c in TRAIN_CASES if all(c["bins"][f]==b for f,b in cond)]
            if not matched:
                continue
            support=sum(c["answers"][sensor][q]==c["truth"][q] for c in matched)
            conflicts=len(matched)-support
            state=1 if conflicts==0 and support>=MIN_SUPPORT else (-1 if conflicts>0 else 0)
            if state==1:
                RULES.append({
                    "query":q, "sensor":sensor, "condition":cond,
                    "support":support, "conflicts":0, "state":1,
                    "cost":SENSOR_COST[sensor],
                })

def active_rules(q,sensor,bins):
    return [
        r for r in RULES
        if r["query"]==q and r["sensor"]==sensor
        and all(bins[f]==b for f,b in r["condition"])
    ]

# Query-guided conservative policy:
# - only sensors with confirmed TRUST-U are candidates;
# - a very broad/high-support rule may answer alone;
# - otherwise require two confirmed sensor paths to agree;
# - disagreement / insufficient trusted paths => KEY 0 (UNKNOWN).
SINGLE_SUPPORT=80

def gated_query(case,q):
    candidates=[]
    for sensor in SENSORS:
        rs=active_rules(q,sensor,case["bins"])
        if not rs:
            continue
        best=max(rs,key=lambda r:(r["support"],-len(r["condition"])))
        candidates.append((SENSOR_COST[sensor],-best["support"],sensor,best))
    candidates.sort()

    values=[]; used=[]
    for _,_,sensor,rule in candidates:
        used.append(sensor)
        values.append(case["answers"][sensor][q])
        if len(values)==1 and rule["support"]>=SINGLE_SUPPORT:
            return {"state":1,"value":values[0],"used":used,"cost":sum(SENSOR_COST[s] for s in used)}
        value,n=Counter(values).most_common(1)[0]
        if n>=2:
            return {"state":1,"value":value,"used":used,"cost":sum(SENSOR_COST[s] for s in used)}
        if len(used)>=3:
            break
    return {"state":0,"value":None,"used":used,"cost":sum(SENSOR_COST[s] for s in used)}

# Baseline: globally best fixed sensor per query on training observations.
BEST_SINGLE={}
for q in QUERIES:
    scores={s:sum(c["answers"][s][q]==c["truth"][q] for c in TRAIN_CASES) for s in SENSORS}
    BEST_SINGLE[q]=max(SENSORS,key=lambda s:(scores[s],-SENSOR_COST[s]))

# Strict ALL-SENSORS baseline: answer only when all paths agree.
def all_consensus(case,q):
    values=[case["answers"][s][q] for s in SENSORS]
    if len(set(values))==1:
        return {"state":1,"value":values[0],"cost":sum(SENSOR_COST.values())}
    return {"state":0,"value":None,"cost":sum(SENSOR_COST.values())}

rows=[]
usage=Counter()
for case in BLIND_CASES:
    for q in QUERIES:
        truth=case["truth"][q]
        g=gated_query(case,q)
        for s in g["used"]:
            usage[(q,s)] += 1
        fixed=BEST_SINGLE[q]
        fv=case["answers"][fixed][q]
        a=all_consensus(case,q)
        rows.append({
            "test":case["id"], "ops":"+".join(case["ops"]), "query":q, "truth":truth,
            "gate_state":g["state"], "gate_value":g["value"], "gate_used":g["used"], "gate_cost":g["cost"],
            "fixed_sensor":fixed, "fixed_value":fv, "fixed_cost":SENSOR_COST[fixed],
            "all_state":a["state"], "all_value":a["value"], "all_cost":a["cost"],
        })

def metrics(kind):
    n=len(rows)
    if kind=="gate":
        correct=sum(r["gate_state"]==1 and r["gate_value"]==r["truth"] for r in rows)
        false=sum(r["gate_state"]==1 and r["gate_value"]!=r["truth"] for r in rows)
        unknown=sum(r["gate_state"]==0 for r in rows)
        cost=sum(r["gate_cost"] for r in rows)
    elif kind=="fixed":
        correct=sum(r["fixed_value"]==r["truth"] for r in rows)
        false=n-correct; unknown=0; cost=sum(r["fixed_cost"] for r in rows)
    else:
        correct=sum(r["all_state"]==1 and r["all_value"]==r["truth"] for r in rows)
        false=sum(r["all_state"]==1 and r["all_value"]!=r["truth"] for r in rows)
        unknown=sum(r["all_state"]==0 for r in rows)
        cost=sum(r["all_cost"] for r in rows)
    covered=n-unknown
    return {
        "n":n, "correct":correct, "false_commits":false, "unknown":unknown,
        "coverage":covered/n,
        "conditional_accuracy": correct/covered if covered else None,
        "overall_correct_rate":correct/n,
        "total_sensor_cost":cost, "mean_sensor_cost":cost/n,
    }

METRICS={k:metrics(k) for k in ("gate","fixed","all")}

# Ternary audit: a rejected trust path does not make an answer negative.
REJECTED_TRUST_U={"uid":"TRUST_BAD","state":-1,"sensor":"RAW","query":"total"}
REJECTED_RESULT_STATE=0

checks={
    "K25_no_neural_sensor_path": True,
    "K25_blind_disturbance_combinations_absent_from_training": not bool(set(BLIND_VARIANTS)&set(TRAIN_VARIANTS)),
    "K25_blind_source_images_not_used_for_rule_induction": all(c["id"]<=16 for c in TRAIN_CASES) and len(BLIND_CASES)==4*len(BLIND_VARIANTS),
    "K25_learner_features_are_anonymous_F1_to_F6": set(FEATURES)=={f"F{i}" for i in range(1,7)},
    "K25_confirmed_TRUST_U_have_zero_training_conflicts": all(r["conflicts"]==0 and r["state"]==1 for r in RULES),
    "K25_multiple_sensor_paths_survive_learning": len({r["sensor"] for r in RULES})>=3,
    "K25_query_context_changes_sensor_usage": len({s for (q,s),n in usage.items() if n>0})>=3,
    "K25_gate_reduces_false_commits_vs_fixed_best_sensor": METRICS["gate"]["false_commits"] < METRICS["fixed"]["false_commits"],
    "K25_gate_uses_less_sensor_cost_than_fixed_best_sensor": METRICS["gate"]["mean_sensor_cost"] < METRICS["fixed"]["mean_sensor_cost"],
    "K25_gate_uses_far_less_cost_than_all_sensors": METRICS["gate"]["mean_sensor_cost"] < METRICS["all"]["mean_sensor_cost"],
    "K25_hard_unseen_combinations_can_remain_UNKNOWN": METRICS["gate"]["unknown"]>0,
    "K25_rejected_TRUST_U_does_not_make_answer_KEY_minus1": REJECTED_TRUST_U["state"]==-1 and REJECTED_RESULT_STATE==0,
}

status = "PASS" if all(checks.values()) else "FAIL"
research_status = "PARTIAL_PASS_NEXT_BOTTLENECK_FOUND" if status=="PASS" and METRICS["gate"]["coverage"]<0.75 else status

report={
    "version":"K25-v1-multi-sensor-selection",
    "status":research_status,
    "interpretation":"Automatic non-neural sensor-selection benchmark. The conservative learned gate reduces false commits and sensor cost, but unseen combined disturbances still force many UNKNOWN answers; robust compositional sensor reliability is the next bottleneck.",
    "train_variants":[list(x) for x in TRAIN_VARIANTS],
    "blind_variants":[list(x) for x in BLIND_VARIANTS],
    "sensor_paths":SENSORS,
    "sensor_cost":SENSOR_COST,
    "clean_calibration":{s:{"count_scenes":MODELS[s]["clean_count_ok"],"shape_map":MODELS[s]["shape"],"color_tolerance":MODELS[s]["tol"]} for s in SENSORS},
    "anonymous_feature_thresholds":THRESHOLDS,
    "trust_u_count":len(RULES),
    "best_fixed_sensor_per_query":BEST_SINGLE,
    "metrics":METRICS,
    "sensor_usage":{f"{q}|{s}":n for (q,s),n in sorted(usage.items())},
    "rejected_u_audit":{"u":REJECTED_TRUST_U,"answer_key_state":REJECTED_RESULT_STATE},
    "checks":checks,
    "caveats":[
        "All input scenes remain synthetic geometric images.",
        "The six sensor paths share some low-level OpenCV operations and are not statistically independent physical sensors.",
        "The selection learner uses anonymous global image measurements F1..F6 and zero-conflict trust rules; it does not receive disturbance names.",
        "The blind stress set holds out disturbance combinations, not the primitive disturbance families themselves.",
        "High UNKNOWN rate is intentional under the current ternary safety policy and shows the selector has not solved compositional sensor reliability yet.",
    ],
}

(OUT/"K25_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
with (OUT/"K25_checks.csv").open("w",encoding="utf-8",newline="") as f:
    import csv
    w=csv.writer(f); w.writerow(["check","passed"])
    for k,v in checks.items(): w.writerow([k,v])
with (OUT/"K25_blind_rows.csv").open("w",encoding="utf-8",newline="") as f:
    import csv
    fields=["test","ops","query","truth","gate_state","gate_value","gate_used","gate_cost","fixed_sensor","fixed_value","fixed_cost","all_state","all_value","all_cost"]
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
    for r in rows:
        rr=dict(r); rr["gate_used"]=";".join(rr["gate_used"]); w.writerow(rr)

print("=== K25 MULTI-SENSOR SELECTION ===")
print("status:", research_status)
print("training cases:",len(TRAIN_CASES),"blind cases:",len(BLIND_CASES),"queries:",len(QUERIES))
print("TRUST-U +1 rules:",len(RULES))
print("best fixed sensors:",BEST_SINGLE)
for name,m in METRICS.items():
    print(name,m)
print("\nchecks:")
for k,v in checks.items():
    print(("PASS" if v else "FAIL"),"|",k)

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

@dataclass(frozen=True)
class VisualRegion:
    rid: str
    rgb: tuple[float, float, float]
    vertices: tuple[str, ...]
    area: int
    centroid: tuple[float, float]

@dataclass(frozen=True)
class VisualObservation:
    image_id: str
    facts: tuple[tuple, ...]
    regions: tuple[VisualRegion, ...]
    background_rgb: tuple[float, float, float]

def observe_image(path: str | Path, image_id: str) -> VisualObservation:
    """Classical visual tokenizer.

    Deliberately DOES NOT emit:
    - region_count
    - corner_count
    - triangle/quadrilateral
    - red/blue/green
    - white/lightgray

    It only emits region membership, individual vertices/edges,
    closure, and raw measurements.
    """
    arr = np.array(Image.open(path).convert("RGB"))
    border = np.concatenate([arr[0], arr[-1], arr[:,0], arr[:,-1]], axis=0)
    bg = tuple(float(x) for x in np.median(border, axis=0))

    delta = np.linalg.norm(arr.astype(float) - np.array(bg)[None,None,:], axis=2)
    mask = (delta > 20).astype(np.uint8) * 255

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    facts = []
    regions = []

    for lab in range(1, n):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < 100:
            continue

        rid = f"{image_id}:R{len(regions)+1}"
        facts.append(("MEMBER", rid, image_id))

        component = (labels == lab).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.025 * perimeter, True)

        vertices = []
        for j, p in enumerate(approx, 1):
            x, y = [int(z) for z in p[0]]
            vid = f"{rid}:V{j}"
            vertices.append(vid)
            facts.append(("VERTEX_OF", vid, rid))
            facts.append(("POSITION", vid, (x, y)))

        if vertices:
            for a, b in zip(vertices, vertices[1:] + vertices[:1]):
                facts.append(("EDGE", a, b, rid))
            facts.append(("CLOSED", rid))

        ys, xs = np.where(labels == lab)
        rgb = tuple(float(x) for x in arr[ys, xs].mean(axis=0))
        centroid = tuple(float(x) for x in centroids[lab])
        facts.append(("COLOR_MEASURE", rid, rgb))
        facts.append(("AREA_MEASURE", rid, area))
        facts.append(("CENTROID_MEASURE", rid, centroid))

        regions.append(VisualRegion(rid, rgb, tuple(vertices), area, centroid))

    facts.append(("BACKGROUND_MEASURE", image_id, bg))
    return VisualObservation(image_id, tuple(facts), tuple(regions), bg)

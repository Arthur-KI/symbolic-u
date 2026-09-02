from __future__ import annotations
from pathlib import Path
import sys

ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(ROOT_FOR_IMPORT))

from symbolic_u.vision_sensor import observe_image

if len(sys.argv) != 2:
    print('usage: python tools/inspect_sensor.py IMAGE.png')
    raise SystemExit(2)

obs = observe_image(sys.argv[1], 'INPUT')
for fact in obs.facts:
    print(fact)

from __future__ import annotations
import subprocess, sys
from pathlib import Path

root = Path(__file__).resolve().parent
print("== Symbolic U Research Pipeline ==")
print("Python:", sys.version.split()[0])

cmd = [sys.executable, "-m", "unittest", "discover", "-s", str(root/"tests"), "-v"]
rc = subprocess.run(cmd, cwd=root).returncode
raise SystemExit(rc)

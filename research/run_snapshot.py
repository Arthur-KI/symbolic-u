from __future__ import annotations
from pathlib import Path
import tempfile, shutil, subprocess, sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent

if len(sys.argv)<2:
    print("usage: python research/run_snapshot.py <relative-script>")
    raise SystemExit(2)

rel=Path(sys.argv[1])
src=HERE/rel
if not src.exists():
    raise SystemExit(f"not found: {src}")

with tempfile.TemporaryDirectory(prefix="symbolic_u_snapshot_") as td:
    work=Path(td)

    # Flat historical workspace, matching how the old /mnt/data experiments
    # referenced one another by basename.
    for p in HERE.rglob("*"):
        if p.is_file() and p.name not in {"run_snapshot.py","README.md"}:
            dst=work/p.name
            if not dst.exists():
                shutil.copy2(p,dst)

    # Vision data and Grimm source texts can also be addressed historically.
    vision=ROOT/"data"/"vision"
    for p in vision.glob("*.png"):
        shutil.copy2(p,work/p.name)

    target=work/src.name
    code=target.read_text(encoding="utf-8")
    code=code.replace("/mnt/data/", str(work).replace("\\","/") + "/")
    target.write_text(code,encoding="utf-8")

    r=subprocess.run([sys.executable,str(target)],cwd=work)
    raise SystemExit(r.returncode)

"""Triage of the Rumoca 0.10.2 MultiBody blocker
  "DAE structural proof failed: ... external builtin function `Modelica.Math.cos` calls `cos`, which this runtime
   cannot execute"
on RocketLanding.SimpleLander.FusalageInGravity (MSL 3.2.3).

Each attempt works on its OWN scratch copy of the MSL 3.2.3 root (results/scratch/msl323_triage_<attempt>/), copied
from results/scratch/msl323 (itself a patched scratch copy; the OpenModelica originals are never touched), so the
support-matrix run that reads results/scratch/msl323 is not disturbed. Attempts:
  A  Modelica.Math.{sin,...,log10}: replace the 'external "builtin" y = f(u);' body by 'algorithm y := .f(u);'
     (leading-dot global lookup of the builtin).
  B  rewrite the call sites under Modelica/Mechanics/MultiBody from Modelica.Math.f( to the builtin f(.
  C  A and B together.
The model is recompiled + simulated (1 s, dt 0.1) in a child process via scripts/stretch_multibody.py --child.

Outputs: results/live/multibody_cos_triage.json, results/live/multibody_cos_triage.md
Re-run: cd C:/GIT/rumoca-rocket-landing && PYTHONIOENCODING=utf-8 JAX_PLATFORMS=cpu .venv/Scripts/python.exe scripts/stretch_multibody_triage.py
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

FUNCS = ["sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh", "exp", "log", "log10", "sqrt"]
MODEL = "RocketLanding.SimpleLander.FusalageInGravity"
FILE = Path(r"C:\GIT\RocketLanding\SimpleLander\FusalageInGravity.mo")
SRC_ROOT = common.MSL323_ROOT
TIMEOUT = 300.0


def fresh_root(tag: str) -> Path:
    dst = common.ROOT / "results" / "scratch" / f"msl323_triage_{tag}"
    if dst.exists():
        shutil.rmtree(dst)
    # Resources/ (images, data, C sources) is not needed to compile; everything else is copied verbatim.
    shutil.copytree(SRC_ROOT, dst, ignore=shutil.ignore_patterns("Images", "Data", "C-Sources", "Library", "Scripts"))
    return dst


def patch_math_bodies(root: Path) -> dict:
    """Attempt A: external "builtin" -> algorithm with leading-dot builtin call."""
    p = root / "Modelica" / "Math" / "package.mo"
    txt = p.read_text(encoding="utf-8")
    names = "|".join(FUNCS)
    pat = re.compile(r'external\s+"builtin"\s+y\s*=\s*(' + names + r')\(([^)]*)\)\s*;')
    hits = pat.findall(txt)
    new = pat.sub(lambda m: f"algorithm\n  y := .{m.group(1)}({m.group(2)});", txt)
    p.write_text(new, encoding="utf-8")
    return {"file": str(p), "n_replaced": len(hits), "functions": [h[0] for h in hits]}


def patch_call_sites(root: Path) -> dict:
    """Attempt B: Modelica.Math.cos( -> cos( under Modelica/Mechanics/MultiBody."""
    names = "|".join(FUNCS)
    pat = re.compile(r"\bModelica\.Math\.(" + names + r")\(")
    total = 0
    per_file = {}
    for f in (root / "Modelica" / "Mechanics" / "MultiBody").rglob("*.mo"):
        txt = f.read_text(encoding="utf-8")
        new, n = pat.subn(r"\1(", txt)
        if n:
            f.write_text(new, encoding="utf-8")
            per_file[str(f.relative_to(root))] = n
            total += n
    return {"n_replaced": total, "per_file": per_file}


def compile_and_simulate(root: Path, tag: str) -> dict:
    out_json = common.ROOT / "results" / "scratch" / "multibody" / f"triage_{tag}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.unlink(missing_ok=True)
    cmd = [sys.executable, str(Path(__file__).with_name("stretch_multibody.py")), "--child", MODEL, str(FILE), str(root),
           str(out_json)]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", JAX_PLATFORMS="cpu")
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT, env=env, encoding="utf-8",
                              errors="replace")
        rec = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {
            "compiled": False, "simulated": False, "compile_error": "child wrote no record; stderr: " + proc.stderr[-400:]}
        rec["child_returncode"] = proc.returncode
    except subprocess.TimeoutExpired:
        rec = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {"compiled": False, "simulated": False}
        rec["timeout"] = f"TIMEOUT after {TIMEOUT:.0f} s"
    rec["wall_s"] = round(time.perf_counter() - t0, 2)
    return rec


def main() -> int:
    attempts = []
    t_start = time.perf_counter()

    # Baseline on the unpatched scratch root, for the exact diagnostic string.
    base = compile_and_simulate(SRC_ROOT, "baseline")
    attempts.append({"attempt": "0-baseline", "description": "unpatched results/scratch/msl323", "patch": None, "result": base})
    print("baseline:", base.get("compiled"), base.get("simulated"), base.get("simulate_error", base.get("compile_error", ""))[:300])

    for tag, desc, steps in [
        ("A", "Modelica.Math.f: external \"builtin\" body -> algorithm y := .f(u)", ["bodies"]),
        ("B", "call sites under Modelica/Mechanics/MultiBody: Modelica.Math.f( -> f(", ["sites"]),
        ("C", "A + B together", ["bodies", "sites"]),
    ]:
        if time.perf_counter() - t_start > 15 * 60:
            attempts.append({"attempt": tag, "description": desc, "skipped": "15-minute triage budget exhausted"})
            continue
        root = fresh_root(tag)
        patch = {}
        if "bodies" in steps:
            patch["bodies"] = patch_math_bodies(root)
        if "sites" in steps:
            patch["sites"] = patch_call_sites(root)
        res = compile_and_simulate(root, tag)
        attempts.append({"attempt": tag, "description": desc, "root": str(root), "patch": patch, "result": res})
        print(f"attempt {tag}: compiled={res.get('compiled')} simulated={res.get('simulated')} "
              f"{res.get('simulate_error', res.get('compile_error', ''))[:300]}")

    report = {"model": MODEL, "file": str(FILE), "source_root": str(SRC_ROOT), "sim_t_end": 1.0, "sim_dt": 0.1,
              "child_timeout_s": TIMEOUT, "wall_s": round(time.perf_counter() - t_start, 1), "attempts": attempts}
    common.write_json(common.RESULTS / "multibody_cos_triage.json", report)

    lines = ["# Triage: `Modelica.Math.cos` external-builtin blocker (FusalageInGravity, MSL 3.2.3)", "",
             "| attempt | what was changed (scratch copy only) | compiled? | compile s | simulated? | diagnostic (first 400 chars) |",
             "|---|---|---|---|---|---|"]
    for a in attempts:
        r = a.get("result", {})
        diag = a.get("skipped") or r.get("simulate_error") or r.get("compile_error") or r.get("timeout") or "-"
        lines.append(f"| {a['attempt']} | {a['description']} | {'yes' if r.get('compiled') else 'no'} | {r.get('compile_s', '')} | "
                     f"{'yes' if r.get('simulated') else 'no'} | {diag[:400].replace('|', chr(92) + '|')} |")
    (common.RESULTS / "multibody_cos_triage.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("wrote", common.RESULTS / "multibody_cos_triage.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

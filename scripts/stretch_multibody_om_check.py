"""OpenModelica cross-check of the unlocked MultiBody drone model DroneLibrary.Tests.motorTest.

Rumoca 0.10.2 (native simulate, patched MSL 4.0.0 scratch root results/scratch/msl400_triage_A) and OpenModelica
(omc, dassl, tol 1e-6) run the SAME DroneLibrary source for T_END s with dt/interval 0.1 s. The window covers both
stimuli of the test (Step(startTime=1) into the motor current, Step(startTime=2) subtracted again), so the motor
actually moves. Signals are compared on a common grid that avoids the event instants (interpolation only).
Note: omc resolves `loadModel(Modelica, {"4.0.0"})` to whatever compatible MSL it has installed (recorded in the JSON
as om_msl_version), while Rumoca reads the MSL 4.0.0 scratch copy; small differences may also come from this.
Outputs: results/live/multibody_om_motortest_check.json, results/live/multibody_om_motortest_check.png
Re-run: (from the repository root) PYTHONIOENCODING=utf-8 JAX_PLATFORMS=cpu .venv/Scripts/python.exe scripts/stretch_multibody_om_check.py
"""
from __future__ import annotations

import csv
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import common  # noqa: E402

MODEL = "DroneLibrary.Tests.motorTest"
FILE = common.DRONE_LIBRARY / "Tests" / "motorTest.mo"
PKG = common.DRONE_LIBRARY / "package.mo"
ROOT400_A = common.ROOT / "results" / "scratch" / "msl400_triage_A"
T_END, DT = 3.0, 0.1
SIGNALS = ["dCmotor.w", "dCmotor.torque", "feedback.y", "step.y", "step1.y",
           "propellerRev1.rotor3_1.revolute.w", "propellerRev1.rotor3_1.revolute.phi",
           "propellerRev1.blades.bodyShape5.r_0[2]", "propellerRev1.blades.bodyShape5.v_0[2]"]
WORKDIR = common.ROOT / "results" / "scratch" / "om_motorTest"


def run_rumoca() -> dict:
    import rumoca as rum

    t0 = time.perf_counter()
    m = rum.Session(roots=[str(ROOT400_A)]).load(str(FILE), model=MODEL)
    c_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    r = m.simulate(T_END, dt=DT)
    s_s = time.perf_counter() - t1
    names = set(r.names)
    sig = {n: np.asarray(r[n], dtype=float) for n in SIGNALS if n in names}
    return {"compile_s": round(c_s, 2), "simulate_s": round(s_s, 2), "time": np.asarray(r.time, dtype=float),
            "signals": sig, "missing": [n for n in SIGNALS if n not in names], "n_names": len(names),
            "termination": r.termination}


def run_om() -> dict:
    WORKDIR.mkdir(parents=True, exist_ok=True)
    prefix = "om_motorTest3s"
    mos = WORKDIR / f"{prefix}.mos"
    mos.write_text("\n".join([
        'loadModel(Modelica, {"4.0.0"}); getErrorString();',
        'print("MSL_VERSION=" + getVersion(Modelica) + "\\n"); getErrorString();',
        f'loadFile("{PKG.as_posix()}"); getErrorString();',
        f'simulate({MODEL}, stopTime={T_END}, numberOfIntervals={int(round(T_END / DT))}, tolerance=1e-6, method="dassl", '
        f'outputFormat="csv", fileNamePrefix="{prefix}"); print(getErrorString());',
    ]) + "\n", encoding="utf-8")
    t0 = time.perf_counter()
    p = subprocess.run([str(common.OMC), mos.name], cwd=WORKDIR, capture_output=True, text=True, timeout=600,
                       encoding="utf-8", errors="replace")
    wall = time.perf_counter() - t0
    out = {"wall_s": round(wall, 1), "omc_stdout_tail": p.stdout[-1200:], "omc_returncode": p.returncode}
    mv = re.search(r"MSL_VERSION=(.*)", p.stdout)
    out["om_msl_version"] = mv.group(1).strip() if mv else None
    csvp = WORKDIR / f"{prefix}_res.csv"
    if not csvp.exists():
        out["error"] = "omc produced no CSV"
        return out
    with open(csvp, newline="") as fh:
        rows = list(csv.reader(fh))
    hdr = rows[0]
    data = np.array([[float(x) for x in r] for r in rows[1:] if r])
    d = {n: data[:, i] for i, n in enumerate(hdr)}
    out["time"] = d["time"]
    out["signals"] = {n: d[n] for n in SIGNALS if n in d}
    out["missing"] = [n for n in SIGNALS if n not in d]
    out["n_cols"] = len(hdr)
    out["csv"] = str(csvp)
    return out


def main() -> int:
    import matplotlib.pyplot as plt

    rep = {"model": MODEL, "t_end": T_END, "dt": DT, "rumoca_root": str(ROOT400_A)}
    try:
        ru = run_rumoca()
        rep["rumoca"] = {k: v for k, v in ru.items() if k not in ("time", "signals")}
    except Exception as exc:  # noqa: BLE001
        rep["rumoca"] = {"error": f"{type(exc).__name__}: {exc}"[:800]}
        common.write_json(common.RESULTS / "multibody_om_motortest_check.json", rep)
        print("Rumoca FAILED:", rep["rumoca"]["error"][:400])
        return 1
    print("rumoca:", rep["rumoca"], flush=True)
    try:
        om = run_om()
        rep["openmodelica"] = {k: v for k, v in om.items() if k not in ("time", "signals")}
    except Exception as exc:  # noqa: BLE001
        rep["openmodelica"] = {"error": f"{type(exc).__name__}: {exc}"[:800]}
        om = {}
    print("om:", {k: v for k, v in rep["openmodelica"].items() if k != "omc_stdout_tail"}, flush=True)

    # comparison grid: mid-interval points, away from the event instants t = 1, 2 (both tools duplicate those rows)
    grid = np.arange(0.05, T_END, DT)
    comp = {}
    if om.get("signals"):
        for n in SIGNALS:
            if n in ru["signals"] and n in om["signals"]:
                a = common.interp_to(grid, ru["time"], ru["signals"][n])
                b = common.interp_to(grid, om["time"], om["signals"][n])
                scale = max(float(np.max(np.abs(b))), 1e-12)
                comp[n] = {"max_abs_diff": float(np.max(np.abs(a - b))), "max_abs_om": float(np.max(np.abs(b))),
                           "max_rel_diff": float(np.max(np.abs(a - b)) / scale),
                           "rumoca_end": float(ru["signals"][n][-1]), "om_end": float(om["signals"][n][-1]),
                           "rumoca_at_1p5s": float(common.interp_to([1.5], ru["time"], ru["signals"][n])[0]),
                           "om_at_1p5s": float(common.interp_to([1.5], om["time"], om["signals"][n])[0]),
                           "rumoca_max": float(np.max(ru["signals"][n])), "om_max": float(np.max(om["signals"][n]))}
        rep["comparison_grid"] = {"t0": 0.05, "t_end": float(grid[-1]), "step": DT, "n": int(len(grid))}
        rep["comparison"] = comp
        rel = [c["max_rel_diff"] for c in comp.values() if c["max_abs_om"] > 1e-9]
        rep["worst_rel_diff_nonzero_signals"] = float(max(rel)) if rel else None
        rep["pass_rel_1e-3"] = bool(rel and max(rel) < 1e-3)
    rep["rumoca_time"] = [float(x) for x in ru["time"]]
    rep["rumoca_signals_end"] = {n: float(v[-1]) for n, v in ru["signals"].items()}

    common.plot_style()
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    panels = [("dCmotor.w", "motor speed w [rad/s]"), ("dCmotor.torque", "motor torque [N m]"),
              ("propellerRev1.blades.bodyShape5.r_0[2]", "bodyShape5.r_0[2] [m] (unconnected body)")]
    for a, (n, lab) in zip(ax, panels):
        if n in ru["signals"]:
            a.plot(ru["time"], ru["signals"][n], label="Rumoca 0.10.2 (patched MSL 4.0.0)")
        if om.get("signals") and n in om["signals"]:
            a.plot(om["time"], om["signals"][n], "--", label=f"OpenModelica dassl ({om.get('om_msl_version') or 'MSL ?'})")
        a.set_xlabel("time [s]"); a.set_ylabel(lab, fontsize=9); a.legend(fontsize=8)
        if n in comp:
            a.set_title(f"max |diff| = {comp[n]['max_abs_diff']:.2e}", fontsize=9)
    fig.suptitle("DroneLibrary.Tests.motorTest, 3 s (steps at 1 s and 2 s)", fontsize=10)
    fig.tight_layout()
    png = common.RESULTS / "multibody_om_motortest_check.png"
    fig.savefig(png, dpi=130)
    rep["figure"] = str(png)
    common.write_json(common.RESULTS / "multibody_om_motortest_check.json", rep)
    for n, c in comp.items():
        print(f"  {n}: max|diff| {c['max_abs_diff']:.3e}  (|om|max {c['max_abs_om']:.3e}, rel {c['max_rel_diff']:.2e})  "
              f"end rumoca {c['rumoca_end']:.6g} om {c['om_end']:.6g}")
    print("wrote", common.RESULTS / "multibody_om_motortest_check.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

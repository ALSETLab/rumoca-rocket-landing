"""Verify the triage work-around (Modelica.Math.f: external "builtin" -> algorithm y := .f(u)) and extend it to MSL 4.0.0.

1. Free-fall check. FusalageInGravity (MSL 3.2.3 patched scratch root results/scratch/msl323_triage_A, produced by
   scripts/stretch_multibody_triage.py) is simulated natively for 2 s with every solver in SOLVERS and every output
   step in DTS, and the Rumoca trajectory of the body's vertical position/velocity is COMPARED (not re-derived) with the
   closed-form free fall z0 - g t^2/2, v = -g t (rigid BodyCylinder in a World with n = {0,0,-1}, g = g_n = 9.80665 m/s^2).
   The model fixes NO initial condition (r_0(start={0,0,166000}, fixed=false), v_0(fixed=false), a_0(fixed=false)):
   Rumoca starts from the start values (z0 = 166000 m, v0 = 0), which is what the closed form assumes; the JSON records
   the simulated z0 and v0 so this can be seen. PRIMARY is the (solver, dt) pair reported as the checked result; the
   default solver ("auto" = BDF path) is kept as a second row because it carries a dt-independent start-up offset.
   --check-only runs just this stage.
2. A patched MSL 4.0.0 copy (results/scratch/msl400_triage_A) is created with the same patch, and the whole support
   matrix is re-run on the patched roots (scripts/stretch_multibody.py --tag _patchedA).

Outputs: results/live/multibody_freefall_check.json, results/live/multibody_freefall_check.png,
         results/live/multibody_support_matrix_patchedA.{json,md}
Re-run: (from the repository root) PYTHONIOENCODING=utf-8 JAX_PLATFORMS=cpu .venv/Scripts/python.exe scripts/stretch_multibody_verify.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import common  # noqa: E402
import stretch_multibody_triage as triage  # noqa: E402

ROOT323_A = common.ROOT / "results" / "scratch" / "msl323_triage_A"
ROOT400_A = common.ROOT / "results" / "scratch" / "msl400_triage_A"

G_N = 9.80665  # Modelica.Mechanics.MultiBody.World default g = Modelica.Constants.g_n (not 9.81)
T_END = 2.0
SOLVERS = ["rk-like", "auto", "bdf"]
DTS = [0.1, 0.01, 0.001]
PRIMARY = ("rk-like", 0.01)
SECONDARY = ("auto", 0.01)  # the default solver
Z_TOL, V_TOL = 1e-3, 1e-4


def ensure_patched_roots() -> dict:
    info = {}
    if not ROOT323_A.exists():
        shutil.copytree(common.MSL323_ROOT, ROOT323_A,
                        ignore=shutil.ignore_patterns("Images", "Data", "C-Sources", "Library", "Scripts"))
        info["msl323_patch"] = triage.patch_math_bodies(ROOT323_A)
    if not ROOT400_A.exists():
        shutil.copytree(common.MSL400_ROOT, ROOT400_A,
                        ignore=shutil.ignore_patterns("Images", "Data", "C-Sources", "Library", "Scripts"))
        info["msl400_patch"] = triage.patch_math_bodies(ROOT400_A)
    return info


def one_run(m, solver: str, dt: float) -> tuple[dict, dict]:
    """Simulate once; return (metrics, arrays). Only comparison/interpolation happens in Python."""
    import rumoca as rum

    rec = {"solver": solver, "dt": dt, "t_end": T_END}
    t1 = time.perf_counter()
    try:
        r = m.simulate(T_END, dt=dt, config=rum.SimConfig(solver=solver))
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"[:600]
        rec["simulate_s"] = round(time.perf_counter() - t1, 3)
        return rec, {}
    rec["simulate_s"] = round(time.perf_counter() - t1, 3)
    names = list(r.names)
    zname = next(n for n in names if n.endswith("FirstStage.r_0[3]") or n == "FirstStage.r_0[3]")
    vname = next(n for n in names if n.endswith("FirstStage.v_0[3]") or n == "FirstStage.v_0[3]")
    t = np.asarray(r.time, dtype=float)
    z = np.asarray(r[zname], dtype=float)
    vz = np.asarray(r[vname], dtype=float)
    z_ref = z[0] - 0.5 * G_N * t ** 2
    v_ref = -G_N * t
    ze, ve = z - z_ref, vz - v_ref
    rec.update({
        "n_points": int(len(t)), "n_result_names": len(names), "z_name": zname, "v_name": vname,
        "z0_m": float(z[0]), "v0_m_s": float(vz[0]), "termination": r.termination,
        "z_end_rumoca_m": float(z[-1]), "z_end_freefall_m": float(z_ref[-1]),
        "drop_rumoca_m": float(z[0] - z[-1]), "drop_freefall_m": float(0.5 * G_N * t[-1] ** 2),
        "vz_end_rumoca_m_s": float(vz[-1]), "vz_end_freefall_m_s": float(v_ref[-1]),
        "max_abs_err_z_m": float(np.max(np.abs(ze))), "max_abs_err_vz_m_s": float(np.max(np.abs(ve))),
        "z_err_end_m": float(ze[-1]),
        "t_first_z_err_gt_1mm_s": (float(t[np.argmax(np.abs(ze) > 1e-3)]) if np.any(np.abs(ze) > 1e-3) else None),
        "all_finite": bool(np.all(np.isfinite(z)) and np.all(np.isfinite(vz))),
    })
    rec[f"pass_z_err_lt_{Z_TOL:g}_m"] = bool(rec["max_abs_err_z_m"] < Z_TOL)
    rec[f"pass_v_err_lt_{V_TOL:g}_m_s"] = bool(rec["max_abs_err_vz_m_s"] < V_TOL)
    return rec, {"t": t, "z": z, "vz": vz, "z_ref": z_ref, "ze": ze, "ve": ve}


def freefall_check() -> dict:
    import rumoca as rum
    import matplotlib.pyplot as plt

    t0 = time.perf_counter()
    m = rum.Session(roots=[str(ROOT323_A)]).load(str(triage.FILE), model=triage.MODEL)
    compile_s = time.perf_counter() - t0
    runs, arrays = [], {}
    for solver in SOLVERS:
        for dt in DTS:
            rec, arr = one_run(m, solver, dt)
            runs.append(rec)
            arrays[(solver, dt)] = arr
            print({k: rec.get(k) for k in ("solver", "dt", "simulate_s", "error", "n_points", "drop_rumoca_m",
                                           "max_abs_err_z_m", "max_abs_err_vz_m_s", "z_err_end_m")}, flush=True)

    def find(solver, dt):
        return next((r for r in runs if r["solver"] == solver and r["dt"] == dt), {})

    primary, secondary = find(*PRIMARY), find(*SECONDARY)
    bdf_like = [r for r in runs if r["solver"] in ("auto", "bdf") and "error" not in r]
    rk = [r for r in runs if r["solver"] == "rk-like" and "error" not in r]
    out = {
        "model": triage.MODEL, "msl_root": str(ROOT323_A), "compile_s": round(compile_s, 2),
        "t_end": T_END, "solvers": SOLVERS, "dts": DTS, "g_ref_m_s2": G_N,
        "initial_conditions_note": "FusalageInGravity.mo fixes no initial condition (r_0(start={0,0,166000}, fixed=false), "
                                   "v_0(fixed=false), a_0(fixed=false)); Rumoca starts from the start values (z0 = 166000 m, "
                                   "v0 = 0), which is what the closed form z0 - g_n t^2/2, -g_n t assumes. The simulated z0/v0 "
                                   "are recorded per run (z0_m, v0_m_s).",
        "primary": {"solver": PRIMARY[0], "dt": PRIMARY[1], **primary},
        "secondary_default_solver": {"solver": SECONDARY[0], "dt": SECONDARY[1], **secondary},
        "runs": runs,
    }
    if bdf_like:
        errs = [r["max_abs_err_z_m"] for r in bdf_like]
        out["bdf_auto_max_abs_err_z_m_all_dts"] = errs
        out["bdf_auto_offset_spread_m"] = float(max(errs) - min(errs))
        out["bdf_auto_offset_dt_independent"] = bool(max(errs) - min(errs) < 1e-6)
        out["auto_equals_bdf"] = bool(all(
            abs(find("auto", dt).get("max_abs_err_z_m", 0) - find("bdf", dt).get("max_abs_err_z_m", 1)) < 1e-12
            for dt in DTS if "error" not in find("auto", dt) and "error" not in find("bdf", dt)))
    if rk:
        out["rk_like_max_abs_err_z_m_all_dts"] = [r["max_abs_err_z_m"] for r in rk]
        out["rk_like_max_abs_err_z_m_worst"] = float(max(r["max_abs_err_z_m"] for r in rk))
    # legacy flat keys = the PRIMARY run (what the report quotes as the checked result)
    for k in ("dt", "n_points", "n_result_names", "z_name", "v_name", "z0_m", "v0_m_s", "z_end_rumoca_m", "z_end_freefall_m",
              "drop_rumoca_m", "drop_freefall_m", "vz_end_rumoca_m_s", "vz_end_freefall_m_s", "max_abs_err_z_m",
              "max_abs_err_vz_m_s", "termination", "simulate_s", f"pass_z_err_lt_{Z_TOL:g}_m", f"pass_v_err_lt_{V_TOL:g}_m_s"):
        if k in primary:
            out[k] = primary[k]
    out["solver"] = PRIMARY[0]

    common.plot_style()
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    a_p, a_s = arrays.get(PRIMARY, {}), arrays.get(SECONDARY, {})
    if a_p:
        ax[0].plot(a_p["t"], a_p["z"] - a_p["z"][0], label=f"Rumoca {PRIMARY[0]}, dt {PRIMARY[1]}")
        ax[0].plot(a_p["t"], a_p["z_ref"] - a_p["z"][0], "--", label="free fall  -g_n t^2/2")
    ax[0].set_xlabel("time [s]"); ax[0].set_ylabel("z - z0 [m]"); ax[0].legend(fontsize=8)
    ax[0].set_title("FusalageInGravity (MSL 3.2.3 MultiBody, patched Math)", fontsize=9)
    if a_s:
        ax[1].plot(a_s["t"], a_s["ze"] * 1e3, label=f"{SECONDARY[0]} (default, =bdf), dt {SECONDARY[1]}")
    if a_p:
        ax[1].plot(a_p["t"], a_p["ze"] * 1e3, label=f"{PRIMARY[0]}, dt {PRIMARY[1]}")
    ax[1].set_xlabel("time [s]"); ax[1].set_ylabel("z error  Rumoca - closed form [mm]"); ax[1].legend(fontsize=8)
    ax[1].set_title("position error vs solver", fontsize=9)
    for r in runs:
        if "error" in r:
            continue
        ax[2].plot(r["dt"], r["max_abs_err_z_m"], "o" if r["solver"] == "rk-like" else ("s" if r["solver"] == "auto" else "^"),
                   label=r["solver"] if r["dt"] == DTS[0] else None, markersize=8, alpha=0.8)
    ax[2].set_xscale("log"); ax[2].set_yscale("log"); ax[2].set_xlabel("output step dt [s]")
    ax[2].set_ylabel("max |z error| over 2 s [m]"); ax[2].legend(fontsize=8)
    ax[2].set_title("dt sweep: BDF offset is dt-independent", fontsize=9)
    fig.tight_layout()
    png = common.RESULTS / "multibody_freefall_check.png"
    fig.savefig(png, dpi=130)
    out["figure"] = str(png)
    common.write_json(common.RESULTS / "multibody_freefall_check.json", out)
    print("PRIMARY", {k: out.get(k) for k in ("solver", "dt", "drop_rumoca_m", "drop_freefall_m", "max_abs_err_z_m",
                                               "max_abs_err_vz_m_s", "simulate_s")})
    return out


def main() -> int:
    info = ensure_patched_roots()
    print("patched roots:", info)
    check_only = "--check-only" in sys.argv
    try:
        freefall_check()
    except Exception as exc:  # noqa: BLE001
        common.write_json(common.RESULTS / "multibody_freefall_check.json",
                          {"error": f"{type(exc).__name__}: {exc}"[:800]})
        print("freefall check FAILED:", f"{type(exc).__name__}: {exc}"[:400])
    if check_only:
        return 0
    env = dict(os.environ, PYTHONIOENCODING="utf-8", JAX_PLATFORMS="cpu")
    cmd = [sys.executable, str(Path(__file__).with_name("stretch_multibody.py")), "--msl323", str(ROOT323_A),
           "--msl400", str(ROOT400_A), "--tag", "_patchedA", "--timeout", "420"]
    print("running:", " ".join(cmd), flush=True)
    return subprocess.run(cmd, env=env).returncode


if __name__ == "__main__":
    sys.exit(main())

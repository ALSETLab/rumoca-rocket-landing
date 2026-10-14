"""Baseline stream: Rumoca runs the ORIGINAL 2021 RocketLanding.SimpleLander.rocket unchanged
and agrees with OpenModelica.

Steps
 1. rum.validate() the original file, load it through common.load_original_rocket(), print
    summary/structure, time compile + 60 s native simulation (dt 0.05; Boolean algebraics
    EngineOn/Ascending and if-expressions handled natively).
 2. Regenerate the OpenModelica reference with omc (dassl, tol 1e-8, 1200 intervals); fall back to
    results/scratch/om_rocket_baseline_res.csv only if omc fails.
 3. Compare y, v, fuel_mass on a common grid for the solvers auto / rk-like / bdf.
 4. Figure results/live/baseline_original_rocket.png, JSON results/live/baseline_original_rocket.json.

Re-runnable from scratch:
  (from the repository root) PYTHONIOENCODING=utf-8 JAX_PLATFORMS=cpu \
      .venv/Scripts/python.exe scripts/baseline_original_rocket.py
"""
from __future__ import annotations

import csv
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import rumoca as rum

import common

T_END = 60.0
DT = 0.05
OM_INTERVALS = 1200
OM_TOL = 1e-8
SCRATCH_CSV = common.ROOT / "results" / "scratch" / "om_rocket_baseline_res.csv"
OM_WORKDIR = common.RESULTS / "om"
PNG = common.RESULTS / "baseline_original_rocket.png"
JSON = common.RESULTS / "baseline_original_rocket.json"
COMPARE = ["y", "v", "fuel_mass"]
SOLVERS = ["auto", "rk-like", "bdf"]

wall_start = time.perf_counter()
report: dict = {
    "rumoca_version": rum.version(),
    "model": "RocketLanding.SimpleLander.rocket",
    "model_file": str(common.ORIGINAL_ROCKET),
    "t_end_s": T_END,
    "rumoca_output_dt_s": DT,
    "openmodelica": {"method": "dassl", "tolerance": OM_TOL, "intervals": OM_INTERVALS},
}


def dedupe_last(t, *cols):
    """Keep the last sample at each repeated time stamp (event points -> post-event value)."""
    t = np.asarray(t, dtype=float)
    keep = np.ones(len(t), dtype=bool)
    keep[:-1] = np.diff(t) > 0
    return (t[keep],) + tuple(np.asarray(c, dtype=float)[keep] for c in cols)


def stamp_stats(t, near_s=1e-9):
    """Time-stamp statistics: exactly repeated stamps (OpenModelica CSV at events) versus near-duplicate
    pre/post-event sample pairs that are strictly increasing (Rumoca, gaps ~1e-13 s)."""
    t = np.asarray(t, dtype=float)
    d = np.diff(t)
    exact = d == 0
    near = (d > 0) & (d < near_s)
    return {"n_points": int(len(t)),
            "n_exact_repeated_stamps": int(exact.sum()),
            "exact_repeated_stamp_values_s": [float(x) for x in t[1:][exact]],
            "n_near_duplicate_pairs_lt_1e-9s": int(near.sum()),
            "near_duplicate_pairs": [{"t_pre_s": float(t[i]), "gap_s": float(d[i])} for i in np.nonzero(near)[0]],
            "min_positive_dt_s": float(d[d > 0].min()) if np.any(d > 0) else None,
            "strictly_increasing": bool(np.all(d > 0))}


def parse_omc_timings(stdout: str) -> dict:
    """Pull the SimulationResult timing fields (seconds) out of omc's stdout."""
    import re
    out = {}
    for key in ["timeFrontend", "timeBackend", "timeSimCode", "timeTemplates", "timeCompile",
                "timeSimulation", "timeTotal"]:
        mm = re.search(key + r"\s*=\s*([0-9.eE+-]+)", stdout)
        if mm:
            out[key + "_s"] = float(mm.group(1))
    return out


def first_crossing(t, a, level=0.0):
    """First time where a <= level, linearly interpolated between the bracketing samples."""
    idx = np.nonzero(np.asarray(a) <= level)[0]
    if len(idx) == 0:
        return None
    i = int(idx[0])
    if i == 0:
        return float(t[0])
    t0, t1, a0, a1 = t[i - 1], t[i], a[i - 1], a[i]
    if a1 == a0:
        return float(t1)
    return float(t0 + (level - a0) * (t1 - t0) / (a1 - a0))


def apogee(t, y):
    i = int(np.argmax(y))
    return {"y_max_m": float(y[i]), "t_apogee_s": float(t[i])}


# ----------------------------------------------------------------------------------------------
# 1. validate + load + native simulate (three solvers)
# ----------------------------------------------------------------------------------------------
t0 = time.perf_counter()
diags = rum.validate(str(common.ORIGINAL_ROCKET))
validate_s = time.perf_counter() - t0
report["validate_diagnostics"] = [str(d) for d in diags]
print("validate:", [str(d) for d in diags])

t0 = time.perf_counter()
m = common.load_original_rocket()
load_s = time.perf_counter() - t0
print(f"load (parse + compile): {load_s:.3f} s")
summary, structure = m.summary(), m.structure()
print(summary)
print(structure)
report["summary"] = str(summary)
report["structure"] = str(structure)
report["states"] = list(m.states.names)
report["inputs"] = list(m.inputs.names)
report["parameters"] = [{"name": p.name, "value": p.value, "unit": p.unit} for p in m.parameters]
report["wall_s"] = {"load_compile": load_s, "validate": validate_s}

rumoca_runs: dict[str, dict] = {}
for solver in SOLVERS:
    try:
        t0 = time.perf_counter()
        r = m.simulate(T_END, dt=DT, config=rum.SimConfig(solver=solver))
        sim_s = time.perf_counter() - t0
        # second call = warm timing (no compile effects)
        t0 = time.perf_counter()
        m.simulate(T_END, dt=DT, config=rum.SimConfig(solver=solver))
        sim_warm_s = time.perf_counter() - t0
        t = np.asarray(r.time, dtype=float)
        run = {"ok": True, "wall_first_s": sim_s, "wall_warm_s": sim_warm_s, "n_points": int(len(t)),
               "termination": r.termination, "names": list(r.names)}
        try:
            run["metrics"] = str(r.metrics)
        except Exception as e:  # metrics is optional information
            run["metrics"] = f"unavailable: {e!r}"
        data = {n: np.asarray(r[n], dtype=float) for n in COMPARE}
        run["final"] = {n: float(data[n][-1]) for n in COMPARE}
        run["apogee"] = apogee(t, data["y"])
        run["fuel_out_t_s"] = first_crossing(t, data["fuel_mass"], 0.0)
        run["time_stamps"] = stamp_stats(t)
        run["_t"] = t
        run["_data"] = data
        rumoca_runs[solver] = run
        print(f"rumoca solver={solver}: {sim_s:.4f} s (warm {sim_warm_s:.4f} s), {len(t)} points, "
              f"termination={r.termination}, apogee={run['apogee']}, fuel_out={run['fuel_out_t_s']}")
    except Exception as e:
        rumoca_runs[solver] = {"ok": False, "error": repr(e)}
        print(f"rumoca solver={solver}: FAILED {e!r}")

# ----------------------------------------------------------------------------------------------
# 2. OpenModelica reference (regenerate; fall back to the scratch CSV)
# ----------------------------------------------------------------------------------------------
om_source = None
om = None
om_wall = None
try:
    t0 = time.perf_counter()
    om, om_stdout = common.omc_simulate("RocketLanding.SimpleLander.rocket", T_END, OM_INTERVALS,
                                        "om_rocket_baseline", [common.ORIGINAL_REPO / "package.mo"],
                                        workdir=OM_WORKDIR, tolerance=OM_TOL, method="dassl")
    om_wall = time.perf_counter() - t0
    om_source = str(OM_WORKDIR / "om_rocket_baseline_res.csv")
    report["openmodelica"]["stdout_tail"] = om_stdout[-1500:]
    report["openmodelica"]["timings"] = parse_omc_timings(om_stdout)
    print(f"omc regenerated reference in {om_wall:.1f} s -> {om_source}; omc timings {report['openmodelica']['timings']}")
except Exception as e:
    report["openmodelica"]["regenerate_error"] = repr(e)[:3000]
    print(f"omc regeneration FAILED: {e!r}\n-> falling back to {SCRATCH_CSV}")
    with open(SCRATCH_CSV, newline="") as fh:
        rows = list(csv.reader(fh))
    header = rows[0]
    arr = np.array([[float(x) for x in r] for r in rows[1:] if r])
    om = {name: arr[:, i] for i, name in enumerate(header)}
    om_source = str(SCRATCH_CSV)
report["openmodelica"]["source_csv"] = om_source
report["openmodelica"]["wall_s"] = om_wall
report["openmodelica"]["n_points_raw"] = int(len(om["time"]))
report["openmodelica"]["time_stamps"] = stamp_stats(om["time"])
print("OM time stamps:", {k: v for k, v in report["openmodelica"]["time_stamps"].items() if k != "near_duplicate_pairs"})

t_om, y_om, v_om, f_om = dedupe_last(om["time"], om["y"], om["v"], om["fuel_mass"])
om_data = {"y": y_om, "v": v_om, "fuel_mass": f_om}
report["openmodelica"]["final"] = {n: float(om_data[n][-1]) for n in COMPARE}
report["openmodelica"]["apogee"] = apogee(t_om, y_om)
report["openmodelica"]["fuel_out_t_s"] = first_crossing(t_om, f_om, 0.0)
print("OpenModelica apogee:", report["openmodelica"]["apogee"], "fuel_out:", report["openmodelica"]["fuel_out_t_s"])

# ----------------------------------------------------------------------------------------------
# 3. compare on a common grid
# ----------------------------------------------------------------------------------------------
t_grid = np.round(np.arange(0.0, T_END + DT / 2, DT), 10)
om_grid = {n: common.interp_to(t_grid, t_om, om_data[n]) for n in COMPARE}
ranges = {n: float(om_data[n].max() - om_data[n].min()) for n in COMPARE}
report["trajectory_range_om"] = ranges

comparison: dict[str, dict] = {}
for solver, run in rumoca_runs.items():
    if not run["ok"]:
        comparison[solver] = {"ok": False, "error": run["error"]}
        continue
    tr, dr = dedupe_last(run["_t"], *[run["_data"][n] for n in COMPARE])[0], None
    tr_full = dedupe_last(run["_t"], *[run["_data"][n] for n in COMPARE])
    tr = tr_full[0]
    ru_grid = {n: common.interp_to(t_grid, tr, tr_full[1 + i]) for i, n in enumerate(COMPARE)}
    errs = {}
    for n in COMPARE:
        e = ru_grid[n] - om_grid[n]
        i = int(np.argmax(np.abs(e)))
        errs[n] = {"max_abs": float(np.abs(e).max()), "t_at_max_abs_s": float(t_grid[i]),
                   "rel_to_range": float(np.abs(e).max() / ranges[n]),
                   "rms": float(np.sqrt(np.mean(e ** 2)))}
    comp = {"ok": True, "errors": errs,
            "apogee_rumoca": run["apogee"], "apogee_om": report["openmodelica"]["apogee"],
            "apogee_dy_m": run["apogee"]["y_max_m"] - report["openmodelica"]["apogee"]["y_max_m"],
            "apogee_dt_s": run["apogee"]["t_apogee_s"] - report["openmodelica"]["apogee"]["t_apogee_s"],
            "fuel_out_rumoca_s": run["fuel_out_t_s"], "fuel_out_om_s": report["openmodelica"]["fuel_out_t_s"],
            "fuel_out_dt_s": (None if run["fuel_out_t_s"] is None or report["openmodelica"]["fuel_out_t_s"] is None
                              else run["fuel_out_t_s"] - report["openmodelica"]["fuel_out_t_s"]),
            "final_rumoca": run["final"], "final_om": report["openmodelica"]["final"],
            "score_max_rel_to_range": float(max(errs[n]["rel_to_range"] for n in COMPARE))}
    comparison[solver] = comp
    run["_err_grid"] = {n: ru_grid[n] - om_grid[n] for n in COMPARE}
    run["_grid"] = ru_grid
    print(f"[{solver}] " + "  ".join(f"{n}: max|err|={errs[n]['max_abs']:.3e} ({errs[n]['rel_to_range']:.2e} of range)"
                                      for n in COMPARE))

# Cross-check: is the residual OpenModelica's own tolerance error? Re-run omc at tol 1e-10 and
# measure OM(1e-8) vs OM(1e-10) and Rumoca vs OM(1e-10) on the same grid.
tight = {"tolerance": 1e-10}
try:
    t0 = time.perf_counter()
    om2, om2_stdout = common.omc_simulate("RocketLanding.SimpleLander.rocket", T_END, OM_INTERVALS, "om_rocket_tol1e10",
                                          [common.ORIGINAL_REPO / "package.mo"], workdir=OM_WORKDIR, tolerance=1e-10,
                                          method="dassl")
    tight["om_tol1e10_wall_s"] = time.perf_counter() - t0
    tight["om_tol1e10_timings"] = parse_omc_timings(om2_stdout)
    tight["om_tol1e10_time_stamps"] = stamp_stats(om2["time"])
    t2, y2, v2, f2 = dedupe_last(om2["time"], om2["y"], om2["v"], om2["fuel_mass"])
    om2_grid = {"y": common.interp_to(t_grid, t2, y2), "v": common.interp_to(t_grid, t2, v2),
                "fuel_mass": common.interp_to(t_grid, t2, f2)}
    tight["om_tol1e8_vs_om_tol1e10_max_abs"] = {n: float(np.abs(om_grid[n] - om2_grid[n]).max()) for n in COMPARE}
    tight["rumoca_vs_om_tol1e10_max_abs"] = {}
    for solver, run in rumoca_runs.items():
        if run.get("ok") and "_grid" in run:
            tight["rumoca_vs_om_tol1e10_max_abs"][solver] = {
                n: float(np.abs(run["_grid"][n] - om2_grid[n]).max()) for n in COMPARE}
    print("OM tol 1e-8 vs tol 1e-10 max|diff|:", tight["om_tol1e8_vs_om_tol1e10_max_abs"])
    print("Rumoca vs OM tol 1e-10 max|diff|:", tight["rumoca_vs_om_tol1e10_max_abs"])
    # Does tightening Rumoca's own tolerance close the gap? (SimConfig rtol/atol default to None.)
    sweep = []
    for solver in ["rk-like", "bdf"]:
        for rtol, atol in [(1e-6, 1e-8), (1e-8, 1e-10), (1e-10, 1e-12)]:
            try:
                t0 = time.perf_counter()
                rr = m.simulate(T_END, dt=DT, config=rum.SimConfig(solver=solver, rtol=rtol, atol=atol))
                w = time.perf_counter() - t0
                tt = dedupe_last(rr.time, *[rr[n] for n in COMPARE])
                errs_t = {n: float(np.abs(common.interp_to(t_grid, tt[0], tt[1 + i]) - om2_grid[n]).max())
                          for i, n in enumerate(COMPARE)}
                sweep.append({"solver": solver, "rtol": rtol, "atol": atol, "wall_s": w, "n_points": int(len(tt[0])),
                              "max_abs_vs_om_tol1e10": errs_t})
                print(f"  rumoca {solver} rtol={rtol:g} atol={atol:g}: max|err| vs OM(1e-10) {errs_t} ({w:.3f} s)")
            except Exception as e:
                sweep.append({"solver": solver, "rtol": rtol, "atol": atol, "error": repr(e)[:500]})
                print(f"  rumoca {solver} rtol={rtol:g} atol={atol:g}: FAILED {e!r}")
    tight["rumoca_tolerance_sweep"] = sweep
except Exception as e:
    tight["error"] = repr(e)[:2000]
    print("tight-tolerance OM cross-check FAILED:", repr(e))
report["om_tight_tolerance_crosscheck"] = tight

ok_solvers = [s for s, c in comparison.items() if c.get("ok")]
best = min(ok_solvers, key=lambda s: comparison[s]["score_max_rel_to_range"]) if ok_solvers else None
report["comparison"] = comparison

# Which solvers return bitwise-identical trajectories (time vector and all compared states)?
identical_pairs: list[tuple[str, str]] = []
for i, a in enumerate(ok_solvers):
    for b in ok_solvers[i + 1:]:
        ra, rb = rumoca_runs[a], rumoca_runs[b]
        same = np.array_equal(ra["_t"], rb["_t"]) and all(np.array_equal(ra["_data"][n], rb["_data"][n]) for n in COMPARE)
        if same:
            identical_pairs.append((a, b))
report["bitwise_identical_solver_pairs"] = [list(p) for p in identical_pairs]
print("bitwise identical solver pairs:", identical_pairs)
report["best_agreeing_solver"] = best
report["solver_ranking_by_max_rel_error"] = sorted(ok_solvers, key=lambda s: comparison[s]["score_max_rel_to_range"])
print("best agreeing solver:", best, "ranking:", report["solver_ranking_by_max_rel_error"])

# Where does the residual error come from? Both tools agree on the analytic burnout time
# InitialFuelMass/FuelBurnRate = 10 s; the remaining difference is solver tolerance + output sampling.
report["analytic_fuel_out_t_s"] = 100.0 / 10.0
report["notes"] = [
    "Rumoca native simulation with fixed output dt=0.05; OpenModelica dassl with tolerance 1e-8 and 1200 "
    "intervals plus event samples. Both detect the EngineOn (fuel_mass>0) and Ascending (v>=0) events; the "
    "EngineOn event has an analytic solution t=InitialFuelMass/FuelBurnRate=10 s.",
    "Comparison is on the uniform 0.05 s grid by linear interpolation of both results. OpenModelica's CSV "
    f"repeats the time stamp exactly at events ({report['openmodelica']['time_stamps']['n_exact_repeated_stamps']} "
    f"exact repeats at t = {report['openmodelica']['time_stamps']['exact_repeated_stamp_values_s']}); dedupe_last keeps "
    "the post-event sample there. Rumoca's output has NO exactly repeated stamps: its pre/post-event sample pairs are "
    "strictly increasing and ~1e-13 s apart ("
    + "; ".join(f"{s}: {rumoca_runs[s]['time_stamps']['n_exact_repeated_stamps']} exact repeats, "
                f"{rumoca_runs[s]['time_stamps']['n_near_duplicate_pairs_lt_1e-9s']} near-duplicate pairs, "
                f"min dt {rumoca_runs[s]['time_stamps']['min_positive_dt_s']:.2e} s" for s in ok_solvers)
    + "), so dedupe_last removes nothing from Rumoca output and np.interp works on it directly.",
    "rum.validate reports only a naming-convention warning (lower-case model name 'rocket'); no errors.",
]

# ----------------------------------------------------------------------------------------------
# 4. figure
# ----------------------------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

common.plot_style()
fig, axes = plt.subplots(4, 1, figsize=(8.5, 11.5), sharex=True)
labels = {"y": "altitude y [m]", "v": "velocity v [m/s]", "fuel_mass": "fuel mass [kg]"}
main = best or next(iter(ok_solvers), None)
for ax, n in zip(axes[:3], COMPARE):
    ax.plot(t_om, om_data[n], color="#d65a31", lw=3.0, alpha=0.55, label="OpenModelica 1.27 (dassl, tol 1e-8)")
    if main:
        run = rumoca_runs[main]
        ax.plot(run["_t"], run["_data"][n], color="#007c91", lw=1.3, ls="--",
                label=f"Rumoca {rum.version()} native (solver={main}, dt={DT})")
    ax.set_ylabel(labels[n])
    ax.grid(alpha=0.3)
axes[0].legend(loc="upper right", fontsize=9)
fo = report["openmodelica"]["fuel_out_t_s"]
ap = report["openmodelica"]["apogee"]["t_apogee_s"]
for ax in axes[:3]:
    ax.axvline(fo, color="grey", ls=":", lw=1)
    ax.axvline(ap, color="grey", ls=":", lw=1)
axes[0].text(fo - 0.6, -900, "burnout (EngineOn=false)", rotation=90, fontsize=8.5, color="grey", ha="right", va="center")
axes[0].text(ap + 0.6, -900, "apogee (Ascending=false)", rotation=90, fontsize=8.5, color="grey", ha="left", va="center")
axes[0].set_title("Unchanged 2021 RocketLanding.SimpleLander.rocket: Rumoca vs OpenModelica", fontsize=12)

ax = axes[3]
# Line style is chosen from the solver NAME (best-agreeing solver solid, the others dashed/dotted in the
# order plotted). A solver whose trajectories are bitwise identical to one already drawn is not drawn
# again (it would be invisible underneath); the drawn solver's label says so.
STYLE_NAME = {"-": "solid", "--": "dashed", ":": "dotted"}
plot_order = ([main] if main else []) + [s for s in ok_solvers if s != main]
plotted: list[str] = []
error_panel_styles: dict[str, dict] = {}
for solver in plot_order:
    run = rumoca_runs.get(solver)
    if not run or not run.get("ok") or "_err_grid" not in run:
        continue
    twins = [b if a == solver else a for a, b in identical_pairs if solver in (a, b)]
    if any(tw in plotted for tw in twins):
        error_panel_styles[solver] = {"drawn": False, "identical_to": [tw for tw in twins if tw in plotted]}
        continue
    ls = ["-", "--", ":"][len(plotted)]
    tag = solver + (f" (= {', '.join(twins)}, bitwise identical)" if twins else "")
    for n, c in zip(COMPARE, ["#007c91", "#d65a31", "#4c956c"]):
        rel = np.abs(run["_err_grid"][n]) / ranges[n]
        ax.semilogy(t_grid, np.maximum(rel, 1e-16), color=c, ls=ls, lw=1.1, label=f"{n} ({tag})")
    error_panel_styles[solver] = {"drawn": True, "linestyle": STYLE_NAME[ls], "label_tag": tag}
    plotted.append(solver)
ax.set_ylabel("|Rumoca - OM| / range")
ax.set_xlabel("time [s]")
ax.grid(alpha=0.3, which="both")
ax.set_ylim(1e-16, 1e-1)
legend_title = "; ".join(f"{error_panel_styles[s]['linestyle']} = {error_panel_styles[s]['label_tag']}" for s in plotted)
ax.legend(loc="upper right", fontsize=8, ncol=3, title=legend_title, title_fontsize=8)
ax.set_xlim(0, T_END)
report["figure_error_panel_styles"] = error_panel_styles
print("error-panel styles:", error_panel_styles)
fig.tight_layout()
common.RESULTS.mkdir(parents=True, exist_ok=True)
fig.savefig(PNG, dpi=130)
print("figure ->", PNG)

# ----------------------------------------------------------------------------------------------
# JSON (strip private arrays)
# ----------------------------------------------------------------------------------------------
report["rumoca_runs"] = {s: {k: v for k, v in run.items() if not k.startswith("_")} for s, run in rumoca_runs.items()}
report["wall_s"]["total_script"] = time.perf_counter() - wall_start
report["figure"] = str(PNG)
common.write_json(JSON, report)
print("json ->", JSON)
print(f"total wall {report['wall_s']['total_script']:.1f} s")
print("BASELINE DONE")

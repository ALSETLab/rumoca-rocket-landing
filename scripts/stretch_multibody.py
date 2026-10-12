"""Stretch stream "multibody": how far does Rumoca 0.10.2 get with the real 3-D MultiBody models of the
two ALSETLab libraries (RocketLanding, MSL 3.2.3; DroneLibrary, MSL 4.0.0)?

Each model is compiled in a CHILD process (fresh rum.Session(roots=[msl_root]) per model) with a wall-clock
timeout, so a hang or a crash in one model cannot take the matrix down. Stages per model, each try/except:
  load -> summary()/structure() -> to_dict('flat') (size + scalar state count from the variables' dims)
       -> simulate(t_end, dt 0.1) (+ number of time-varying / non-finite signals, a few key signals)
t_end is 1 s except for the models in SIM_T_END_BY_MODEL (motorTest: 3 s, because its stimuli are
Step(startTime=1) and Step(startTime=2) and a 1 s window would end exactly where the first step begins).
Outputs (all under results/live/):
  multibody_support_matrix.json  - every number quoted in the report
  multibody_support_matrix.md    - the table

Re-runnable from scratch:
  cd C:/GIT/rumoca-rocket-landing && PYTHONIOENCODING=utf-8 JAX_PLATFORMS=cpu .venv/Scripts/python.exe scripts/stretch_multibody.py
  Options: --only NAME (substring filter), --timeout SECONDS (default 420 per model), --msl323 DIR / --msl400 DIR
  (alternative roots, used by the verify script to re-run the matrix on the patched scratch copies), --tag SUFFIX,
  --render-only (re-render the .md from the existing .json).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

ROCKET = Path(r"C:\GIT\RocketLanding")
DRONE = Path(r"C:\GIT\Modelica-Drone-3D-FMI\A_Modelica\DroneLibrary")

MODELS = [
    # (fully-qualified class, file, MSL label)
    ("RocketLanding.SimpleLander.FusalageInGravity", ROCKET / "SimpleLander" / "FusalageInGravity.mo", "3.2.3"),
    ("RocketLanding.SimpleLander.FusalageWithForce", ROCKET / "SimpleLander" / "FusalageWithForce.mo", "3.2.3"),
    ("RocketLanding.LanderWithHorizontalVelocity.RetrogradeRocketLanding",
     ROCKET / "LanderWithHorizontalVelocity" / "RetrogradeRocketLanding.mo", "3.2.3"),
    ("DroneLibrary.Tests.motorTest", DRONE / "Tests" / "motorTest.mo", "4.0.0"),
    ("DroneLibrary.Tests.propellerTest", DRONE / "Tests" / "propellerTest.mo", "4.0.0"),
    ("DroneLibrary.Tests.simpleHoverTest", DRONE / "Tests" / "simpleHoverTest.mo", "4.0.0"),
    ("DroneLibrary.Examples.DroneWithIdealPower.TestSystem",
     DRONE / "Examples" / "DroneWithIdealPower" / "TestSystem.mo", "4.0.0"),
]

SIM_T_END = 1.0
SIM_DT = 0.1
# motorTest: Step(startTime=1) and Step(startTime=2) -> 3 s covers both steps and the motor's response.
SIM_T_END_BY_MODEL = {"DroneLibrary.Tests.motorTest": 3.0}
# substrings of result names whose first/last/min/max are recorded (so "what moved?" has an answer on disk)
KEY_SIGNAL_PATTERNS = ["FirstStage.r_0[3]", "FirstStage.v_0[3]", "dCmotor.w", "dCmotor.torque", "feedback.y",
                       "rotor3_1.revolute.w", "blades.bodyShape5.r_0[2]", "droneChassis1.frame_a.r_0[3]",
                       "ramp.y", "PID.u"]


def err_text(exc: BaseException, n: int = 400) -> str:
    return f"{type(exc).__name__}: {str(exc)}"[:n]


def scalar_count(flat_vars: dict, names) -> tuple[int, int]:
    """Expand array-valued variable names by their dims from to_dict('flat')['variables'].
    Returns (scalar components, names without a usable dims record)."""
    total, missing = 0, 0
    for n in names:
        v = flat_vars.get(n)
        if not isinstance(v, dict):
            missing += 1
            continue
        k = 1
        for e in (v.get("dims") or []):
            try:
                k *= int(e)
            except (TypeError, ValueError):
                missing += 1
                k = 0
                break
        total += k
    return total, missing


def probe_one(model: str, file: str, root: str) -> dict:
    """Runs inside the child process. Never raises: every stage is recorded."""
    import numpy as np
    import rumoca as rum

    t_end = SIM_T_END_BY_MODEL.get(model, SIM_T_END)
    rec = {"model": model, "file": file, "msl_root": root, "rumoca": rum.version(),
           "compiled": False, "simulated": False, "sim_t_end_requested": t_end, "sim_dt": SIM_DT}
    t0 = time.perf_counter()
    try:
        session = rum.Session(roots=[root])
        m = session.load(file, model=model)
        rec["compile_s"] = round(time.perf_counter() - t0, 2)
        rec["compiled"] = True
    except Exception as exc:  # noqa: BLE001
        rec["compile_s"] = round(time.perf_counter() - t0, 2)
        rec["compile_error"] = err_text(exc)
        return rec

    try:
        rec["summary"] = str(m.summary())
    except Exception as exc:  # noqa: BLE001
        rec["summary_error"] = err_text(exc)
    try:
        st = m.structure()
        rec["structure"] = str(st)[:600]
        for key in ("n_equations", "n_states", "n_algebraics", "n_parameters", "n_inputs", "n_outputs",
                    "equations", "states", "algebraics", "parameters", "inputs", "outputs"):
            if hasattr(st, key):
                val = getattr(st, key)
                try:
                    rec["structure_" + key] = int(val) if isinstance(val, (int, float)) else len(val)
                except Exception:  # noqa: BLE001
                    rec["structure_" + key] = str(val)[:200]
    except Exception as exc:  # noqa: BLE001
        rec["structure_error"] = err_text(exc)
    try:
        rec["n_states"] = len(m.states.names)          # array-valued NAMES (aliases included), not scalars
        rec["n_algebraics"] = len(m.algebraics.names)
        rec["n_parameters"] = len(m.parameters)
        rec["n_inputs"] = len(m.inputs.names)
        rec["n_outputs"] = len(m.outputs.names)
        rec["state_names"] = list(m.states.names)
    except Exception as exc:  # noqa: BLE001
        rec["counts_error"] = err_text(exc)

    t1 = time.perf_counter()
    try:
        d = m.to_dict("flat")
        txt = json.dumps(d, default=str)
        rec["flat_dict_chars"] = len(txt)
        rec["flat_dict_s"] = round(time.perf_counter() - t1, 2)
        if isinstance(d, dict):
            rec["flat_dict_keys"] = {k: (len(v) if hasattr(v, "__len__") and not isinstance(v, str) else str(v)[:60])
                                     for k, v in list(d.items())[:40]}
            for key in ("equations", "eqs"):
                if key in d and hasattr(d[key], "__len__"):
                    rec["n_equations_flat"] = len(d[key])
            if isinstance(d.get("variables"), dict) and "state_names" in rec:
                rec["n_states_scalar"], rec["n_states_scalar_missing"] = scalar_count(d["variables"], rec["state_names"])
                rec["n_algebraics_scalar"], _ = scalar_count(d["variables"], m.algebraics.names)
                rec["state_dims"] = {n: (d["variables"].get(n) or {}).get("dims") for n in rec["state_names"]}
    except Exception as exc:  # noqa: BLE001
        rec["flat_dict_error"] = err_text(exc)
        rec["flat_dict_s"] = round(time.perf_counter() - t1, 2)

    t2 = time.perf_counter()
    try:
        r = m.simulate(t_end, dt=SIM_DT)
        rec["simulate_s"] = round(time.perf_counter() - t2, 2)
        rec["simulated"] = True
        rec["sim_n_points"] = len(r.time)
        rec["sim_t_end"] = float(r.time[-1])
        rec["sim_termination"] = r.termination
        names = list(r.names)
        rec["sim_n_names"] = len(names)
        n_var, n_bad, key = 0, 0, {}
        for n in names:
            a = np.asarray(r[n], dtype=float)
            if a.size == 0:
                continue
            if not np.all(np.isfinite(a)):
                n_bad += 1
            elif np.ptp(a) != 0.0:
                n_var += 1
            if len(key) < 16 and any(p in n for p in KEY_SIGNAL_PATTERNS):
                key[n] = {"first": float(a[0]), "last": float(a[-1]), "min": float(np.min(a)), "max": float(np.max(a))}
        rec["sim_n_varying"] = n_var
        rec["sim_n_non_finite"] = n_bad
        rec["sim_key_signals"] = key
        if "state_names" in rec:  # cross-check of the dims-based scalar count against the result names
            rec["n_states_scalar_from_results"] = sum(
                sum(1 for n in names if n == s or n.startswith(s + "[")) for s in rec["state_names"])
    except Exception as exc:  # noqa: BLE001
        rec["simulate_s"] = round(time.perf_counter() - t2, 2)
        rec["simulate_error"] = err_text(exc)
    return rec


def run_child(model: str, file: Path, root: Path, timeout: float, out_json: Path) -> dict:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", model, str(file), str(root), str(out_json)]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", JAX_PLATFORMS="cpu")
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env,
                              encoding="utf-8", errors="replace")
        wall = round(time.perf_counter() - t0, 2)
        if out_json.exists():
            rec = json.loads(out_json.read_text(encoding="utf-8"))
        else:
            rec = {"model": model, "file": str(file), "msl_root": str(root), "compiled": False, "simulated": False,
                   "compile_error": f"child exited {proc.returncode} without a record; stderr tail: "
                                    + proc.stderr[-400:]}
        rec["child_returncode"] = proc.returncode
        rec["child_wall_s"] = wall
        if proc.returncode != 0:
            rec["child_stderr_tail"] = proc.stderr[-800:]
        return rec
    except subprocess.TimeoutExpired as exc:
        wall = round(time.perf_counter() - t0, 2)
        rec = {"model": model, "file": str(file), "msl_root": str(root), "compiled": False, "simulated": False,
               "timeout_s": timeout, "child_wall_s": wall,
               "compile_error": f"TIMEOUT after {timeout:.0f} s in the child process (no stage record written)"}
        if out_json.exists():  # the child may have written a partial record before hanging in simulate
            partial = json.loads(out_json.read_text(encoding="utf-8"))
            partial.update({"child_wall_s": wall, "timeout_s": timeout})
            if "simulate_s" not in partial and partial.get("compiled"):
                partial["simulate_error"] = f"TIMEOUT: simulate() still running after {timeout:.0f} s wall"
            partial.pop("compile_error", None) if partial.get("compiled") else None
            rec = partial
        tail = (exc.stderr or b"")
        if isinstance(tail, bytes):
            tail = tail.decode("utf-8", "replace")
        rec["child_stderr_tail"] = tail[-800:]
        return rec


def blocker(rec: dict) -> str:
    if not rec.get("compiled"):
        return "compile: " + rec.get("compile_error", "?")
    if not rec.get("simulated"):
        return "simulate: " + rec.get("simulate_error", "?")
    return "-"


def write_markdown(records: list[dict], path: Path, title: str) -> None:
    lines = [f"# {title}", "",
             f"Rumoca {records[0].get('rumoca', '?')} native pipeline; one fresh `rum.Session(roots=[msl_root])` per model, "
             f"simulate(t_end, dt={SIM_DT}) with t_end = {SIM_T_END} s except motorTest ({SIM_T_END_BY_MODEL['DroneLibrary.Tests.motorTest']} s, "
             "its stimuli are Step(startTime=1) and Step(startTime=2)).", "",
             "Column semantics: `equations` is the SCALARIZED count from `structure()` (StructuralInfo.equations, before matching). "
             "`state vars` is the number of array-valued NAMES in `m.states.names` (alias pairs such as `FirstStage.r_0` and "
             "`FirstStage.body.frame_a.r_0` are both listed); `scalar states` expands those names by their dims from "
             "`to_dict('flat')['variables']` and therefore also counts the aliases (a free rigid body has 13 independent scalar "
             "states; FusalageInGravity lists 19 = 3+3+3+3+3+4 with two alias pairs). `algebraic vars` counts names, not scalars. "
             "`varying` = number of result signals that are not constant over the simulated window.", "",
             "| model | MSL | t_end s | equations (scalar) | state vars (names) | scalar states (dims) | algebraic vars (names) | parameters | compile s | simulate s | compiled? | simulated? | varying/total signals | blocker (first 400 chars) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in records:
        eq = r.get("structure_n_equations", r.get("n_equations_flat", ""))
        var = f"{r['sim_n_varying']}/{r['sim_n_names']}" if r.get("simulated") else ""
        lines.append("| {m} | {msl} | {te} | {eq} | {st} | {sc} | {al} | {pa} | {cs} | {ss} | {c} | {s} | {v} | {b} |".format(
            m=r["model"], msl=r.get("msl", ""), te=r.get("sim_t_end_requested", SIM_T_END), eq=eq,
            st=r.get("n_states", ""), sc=r.get("n_states_scalar", ""), al=r.get("n_algebraics", ""),
            pa=r.get("n_parameters", ""), cs=r.get("compile_s", r.get("child_wall_s", "")),
            ss=r.get("simulate_s", "") if r.get("simulated") else "",
            c="yes" if r.get("compiled") else "no", s="yes" if r.get("simulated") else "no", v=var,
            b=blocker(r).replace("|", "\\|").replace("\n", " ")))
    lines.append("")
    keys = [r for r in records if r.get("sim_key_signals")]
    if keys:
        lines += ["Key signals of the simulated models (first -> last value over the window):", ""]
        for r in keys:
            for n, v in r["sim_key_signals"].items():
                lines.append(f"- {r['model'].split('.')[-1]} `{n}`: {v['first']:.6g} -> {v['last']:.6g} "
                             f"(min {v['min']:.6g}, max {v['max']:.6g})")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", nargs=4, metavar=("MODEL", "FILE", "ROOT", "OUT_JSON"))
    ap.add_argument("--only", default=None)
    ap.add_argument("--timeout", type=float, default=420.0)
    ap.add_argument("--msl323", default=str(common.MSL323_ROOT))
    ap.add_argument("--msl400", default=str(common.MSL400_ROOT))
    ap.add_argument("--tag", default="", help="suffix for the output files (used by the triage script)")
    ap.add_argument("--render-only", action="store_true", help="re-render the .md table from the existing .json")
    args = ap.parse_args()

    if args.render_only:
        report = json.loads((common.RESULTS / f"multibody_support_matrix{args.tag}.json").read_text(encoding="utf-8"))
        write_markdown(report["models"], common.RESULTS / f"multibody_support_matrix{args.tag}.md",
                       "Rumoca 0.10.2 support matrix: ALSETLab MultiBody models" + (f" ({args.tag.strip('_')})" if args.tag else ""))
        return 0

    if args.child:
        model, file, root, out_json = args.child
        rec = probe_one(model, file, root)
        Path(out_json).write_text(json.dumps(rec, indent=2, default=str), encoding="utf-8")
        print(json.dumps({k: rec[k] for k in rec if k in ("model", "compiled", "compile_s", "simulated", "simulate_s")}))
        return 0

    common.RESULTS.mkdir(parents=True, exist_ok=True)
    scratch = common.ROOT / "results" / "scratch" / "multibody"
    scratch.mkdir(parents=True, exist_ok=True)
    records = []
    for model, file, msl in MODELS:
        if args.only and args.only.lower() not in model.lower():
            continue
        root = Path(args.msl323 if msl == "3.2.3" else args.msl400)
        out_json = scratch / (model.replace(".", "_") + args.tag + ".json")
        out_json.unlink(missing_ok=True)
        print(f"== {model}  (MSL {msl}, root {root})", flush=True)
        rec = run_child(model, file, root, args.timeout, out_json)
        rec["msl"] = msl
        records.append(rec)
        print(f"   compiled={rec.get('compiled')} ({rec.get('compile_s', rec.get('child_wall_s'))} s)  "
              f"simulated={rec.get('simulated')}  blocker={blocker(rec)[:160]}", flush=True)

    report = {
        "rumoca_version": records[0].get("rumoca") if records else None,
        "sim_t_end": SIM_T_END, "sim_t_end_by_model": SIM_T_END_BY_MODEL, "sim_dt": SIM_DT,
        "child_timeout_s": args.timeout,
        "msl323_root": args.msl323, "msl400_root": args.msl400,
        "n_models": len(records),
        "n_compiled": sum(1 for r in records if r.get("compiled")),
        "n_simulated": sum(1 for r in records if r.get("simulated")),
        "models": records,
    }
    out = common.RESULTS / f"multibody_support_matrix{args.tag}.json"
    common.write_json(out, report)
    write_markdown(records, common.RESULTS / f"multibody_support_matrix{args.tag}.md",
                   "Rumoca 0.10.2 support matrix: ALSETLab MultiBody models" + (f" ({args.tag.strip('_')})" if args.tag else ""))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

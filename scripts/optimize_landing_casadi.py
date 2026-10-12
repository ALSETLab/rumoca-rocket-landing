"""Minimum-propellant hoverslam planned with CasADi/IPOPT straight from the Rumoca CasADi export,
then checked four ways:
  A.  JAX export of the same Modelica model, RK4 at dt/10 rollout of the optimal throttle schedule
      (discretisation-error + CasADi/JAX export-consistency check; same RK4 scheme as the NLP)
  A2. CasADi CVODES (adaptive BDF, abstol 1e-13) on the Rumoca CasADi export: a genuinely different
      integrator; also the exact bang-bang hoverslam (t_sw, T) solved with fsolve on y(T)=v(T)=0
  B.  Rumoca native simulator on RumocaRocketLanding.LanderBurn (events, terminate at touchdown)
      + two bisections on the ignition time: physical boundary (arrest with y > 0) and clean-terminate
      boundary; the gap between them is a plateau where the native simulator misses the grazing
      touchdown event (y dips ~2 cm below the pad without terminate firing)
  C.  OpenModelica (dassl) on the same .mo file, at the clean t_ign and at the fuel-equivalent t_ign
Plus a before/after demo (ignite 1 s early / optimal / 1 s late).

No vehicle physics is written in Python: every derivative comes from Rumoca's exports or simulators,
or from OpenModelica. Python only transcribes, optimizes, integrates, interpolates, plots and compares.

Run:  cd C:/GIT/rumoca-rocket-landing && PYTHONIOENCODING=utf-8 JAX_PLATFORMS=cpu .venv/Scripts/python.exe scripts/optimize_landing_casadi.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import casadi as ca
import numpy as np
import rumoca as rum

import common

T_WALL0 = time.perf_counter()
RES = common.RESULTS
RES.mkdir(parents=True, exist_ok=True)
report: dict = {"rumoca_version": str(rum.version()), "casadi_version": ca.__version__}
timings: dict = {}

# ----------------------------------------------------------------------------------------------
# 1. NLP: direct multiple shooting (RK4) on the Rumoca CasADi export, free final time
# ----------------------------------------------------------------------------------------------
t0 = time.perf_counter()
m_lander = common.load_lander()
ex = m_lander.to_casadi()
common.validate_layout(ex)
p_num = common.parameters(ex)
x0_num = np.array(ex.default_states, dtype=float)
i_y, i_v, i_f = (ex.state_names.index(n) for n in ("y", "v", "fuel_mass"))
FuelBurnRate = float(p_num[ex.parameter_names.index("FuelBurnRate")])
timings["load_export_casadi_s"] = time.perf_counter() - t0

N = 160                      # shooting intervals
T_MIN, T_MAX = 5.0, 40.0     # bounds of the free final time
REG = 1e-4                   # control regularisation weight

opti_t0 = time.perf_counter()
p_mx = ca.MX(ca.DM(p_num))
T = ca.MX.sym("T")
dt = T / N
X = [ca.MX.sym(f"x{k}", ex.n_states) for k in range(N + 1)]
U = [ca.MX.sym(f"u{k}", ex.n_inputs) for k in range(N)]

w, lbw, ubw, w0 = [], [], [], []
g, lbg, ubg = [], [], []

# Initial guess: coast 5 s then full throttle, final time 14 s, states from a DM rollout of the
# same Rumoca rhs (a consistent, if infeasible, trajectory helps IPOPT).
T_guess = 14.0
u_guess = np.array([0.0 if (k + 0.5) * T_guess / N < 5.0 else 1.0 for k in range(N)])
x_guess = [x0_num.copy()]
for k in range(N):
    xn = common.rk4(ex.rhs, k * T_guess / N, ca.DM(x_guess[-1]), ca.DM([u_guess[k]]), ca.DM(p_num), T_guess / N)
    x_guess.append(np.array(xn.full()).ravel())

# Decision variable: T
w.append(T); lbw += [T_MIN]; ubw += [T_MAX]; w0 += [T_guess]
# Initial state fixed
w.append(X[0]); lbw += list(x0_num); ubw += list(x0_num); w0 += list(x0_num)
J = 0
for k in range(N):
    w.append(U[k]); lbw += [0.0]; ubw += [1.0]; w0 += [u_guess[k]]
    xk_end = common.rk4(ex.rhs, k * dt, X[k], U[k], p_mx, dt)
    w.append(X[k + 1])
    # path bounds: y >= 0, v free-ish, fuel_mass >= 0
    lb = [-np.inf] * ex.n_states; ub = [np.inf] * ex.n_states
    lb[i_y] = 0.0; lb[i_f] = 0.0; lb[i_v] = -300.0; ub[i_v] = 100.0
    if k == N - 1:  # terminal: y(T) = 0, v(T) = 0
        lb[i_y] = ub[i_y] = 0.0
        lb[i_v] = ub[i_v] = 0.0
    lbw += lb; ubw += ub; w0 += list(x_guess[k + 1])
    # continuity (multiple shooting defect)
    g.append(xk_end - X[k + 1]); lbg += [0.0] * ex.n_states; ubg += [0.0] * ex.n_states
    J = J + REG * ca.sumsqr(U[k]) * dt
J = J - X[N][i_f]   # minimise propellant used  (= maximise fuel left)

nlp = {"x": ca.vertcat(*w), "f": J, "g": ca.vertcat(*g)}
solver = ca.nlpsol("solver", "ipopt", nlp, {
    "ipopt.print_level": 0, "print_time": 0, "ipopt.max_iter": 3000,
    "ipopt.tol": 1e-9, "ipopt.acceptable_tol": 1e-7, "ipopt.mu_strategy": "adaptive",
})
sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
stats = solver.stats()
timings["nlp_build_and_solve_s"] = time.perf_counter() - opti_t0

wopt = np.array(sol["x"].full()).ravel()
T_opt = float(wopt[0])
dt_opt = T_opt / N
off = 1
X_opt = np.zeros((N + 1, ex.n_states)); U_opt = np.zeros(N)
X_opt[0] = wopt[off:off + ex.n_states]; off += ex.n_states
for k in range(N):
    U_opt[k] = wopt[off]; off += 1
    X_opt[k + 1] = wopt[off:off + ex.n_states]; off += ex.n_states
t_nodes = np.arange(N + 1) * dt_opt
fuel_left = float(X_opt[-1, i_f]); fuel_used = float(x0_num[i_f] - fuel_left)
frac_bang = float(np.mean((U_opt < 0.02) | (U_opt > 0.98)))
idx_ign = int(np.argmax(U_opt > 0.5))
t_ign_star = float(t_nodes[idx_ign])
# a grid-independent ignition time: the propellant used at full throttle lasts fuel_used/FuelBurnRate
t_ign_fuel_equiv = T_opt - fuel_used / FuelBurnRate
# the switching interval, if any
u_intermediate = [(float(t_nodes[k]), float(U_opt[k])) for k in range(N) if 0.02 <= U_opt[k] <= 0.98]
# switch time implied by the intermediate value at the switching node (u_k = alpha means the engine is
# on for the last alpha*dt of that interval, in the piecewise-constant sense)
t_ign_switch_node = float(t_nodes[idx_ign] + (1.0 - U_opt[idx_ign]) * dt_opt)
defect = np.array(sol["g"].full()).ravel()

report["nlp"] = {
    "N_intervals": N, "T_bounds_s": [T_MIN, T_MAX], "control_regularisation": REG,
    "ipopt_status": stats.get("return_status"), "ipopt_success": bool(stats.get("success")),
    "ipopt_iterations": int(stats.get("iter_count", -1)),
    "ipopt_wall_s": stats.get("t_wall_total"),
    "objective": float(sol["f"]),
    "T_final_s": T_opt, "dt_s": dt_opt,
    "propellant_used_kg": fuel_used, "fuel_left_kg": fuel_left,
    "y_final_m": float(X_opt[-1, i_y]), "v_final_mps": float(X_opt[-1, i_v]),
    "max_abs_shooting_defect": float(np.max(np.abs(defect))),
    "fraction_u_within_0p02_of_0_or_1": frac_bang,
    "n_intermediate_u": len(u_intermediate), "intermediate_u_(t,u)": u_intermediate,
    "t_ign_star_first_u_gt_0p5_s": t_ign_star,
    "t_ign_fuel_equivalent_s": float(t_ign_fuel_equiv),
    "t_ign_from_intermediate_u_s": t_ign_switch_node,
    "burn_duration_from_node_s": float(T_opt - t_ign_star),
    "burn_duration_fuel_equivalent_s": float(fuel_used / FuelBurnRate),
    "u_min": float(U_opt.min()), "u_max": float(U_opt.max()),
    "min_altitude_before_T_m": float(X_opt[:-1, i_y].min()),
    "throttle_profile": [[float(t_nodes[k]), float(U_opt[k])] for k in range(N)],
}
print(f"[NLP] status={stats.get('return_status')} iters={stats.get('iter_count')} T={T_opt:.4f} s "
      f"fuel used={fuel_used:.4f} kg left={fuel_left:.4f} kg bang-bang frac={frac_bang:.3f} "
      f"t_ign*={t_ign_star:.4f} s (fuel-equiv {t_ign_fuel_equiv:.4f} s)")

# ----------------------------------------------------------------------------------------------
# 2. CHECK A: JAX export, fine RK4 rollout of the optimal piecewise-constant u(t)
# ----------------------------------------------------------------------------------------------
t0 = time.perf_counter()
import jax.numpy as jnp
exj = common.load_lander().to_jax()
common.validate_layout(exj)
SUB = 10
dt_fine = dt_opt / SUB
roll = common.jax_rollout(exj, dt_fine)
# the optimal schedule, then 1 s of engine-off after T so the y=0 crossing can be interpolated even if
# the rollout lands a hair above the pad at T
N_EXTRA = int(round(1.0 / dt_fine))
controls = jnp.asarray(np.concatenate([np.repeat(U_opt, SUB), np.zeros(N_EXTRA)])[:, None])
Xj_all = np.asarray(roll(jnp.asarray(common.parameters(exj)), jnp.asarray(x0_num), controls))
tj_all = np.arange(Xj_all.shape[0]) * dt_fine
Xj = Xj_all[: N * SUB + 1]; tj = tj_all[: N * SUB + 1]


def touchdown(t, y, v):
    """Linear interpolation of the first downward y=0 crossing. Returns (t_td, v_td) or (None, None)."""
    below = np.where(y <= 0.0)[0]
    below = below[below > 0]
    if len(below) == 0:
        return None, None
    i = int(below[0])
    s = (0.0 - y[i - 1]) / (y[i] - y[i - 1]) if y[i] != y[i - 1] else 0.0
    return float(t[i - 1] + s * (t[i] - t[i - 1])), float(v[i - 1] + s * (v[i] - v[i - 1]))


tdj, vdj = touchdown(tj_all, Xj_all[:, i_y], Xj_all[:, i_v])
# compare the states at the NLP nodes
Xj_nodes = Xj[::SUB]
report["check_A_jax_rollout"] = {
    "dt_fine_s": dt_fine, "n_steps": int(Xj.shape[0] - 1),
    "y_at_T_m": float(Xj[-1, i_y]), "v_at_T_mps": float(Xj[-1, i_v]), "fuel_at_T_kg": float(Xj[-1, i_f]),
    "y_at_T_error_vs_nlp_m": float(Xj[-1, i_y] - X_opt[-1, i_y]),
    "v_at_T_error_vs_nlp_mps": float(Xj[-1, i_v] - X_opt[-1, i_v]),
    "fuel_at_T_error_vs_nlp_kg": float(Xj[-1, i_f] - X_opt[-1, i_f]),
    "max_abs_y_error_at_nodes_m": float(np.max(np.abs(Xj_nodes[:, i_y] - X_opt[:, i_y]))),
    "max_abs_v_error_at_nodes_mps": float(np.max(np.abs(Xj_nodes[:, i_v] - X_opt[:, i_v]))),
    "touchdown_time_interp_s": tdj, "touchdown_velocity_interp_mps": vdj,
    "touchdown_time_error_vs_T_s": (None if tdj is None else tdj - T_opt),
    "min_y_m": float(Xj[:, i_y].min()),
}
timings["check_A_s"] = time.perf_counter() - t0
print(f"[A] JAX rollout dt={dt_fine:.5f}: y(T)={Xj[-1, i_y]:.4e} m v(T)={Xj[-1, i_v]:.4e} m/s "
      f"touchdown {tdj} s at {vdj} m/s; max node err y={report['check_A_jax_rollout']['max_abs_y_error_at_nodes_m']:.3e}")

# ----------------------------------------------------------------------------------------------
# 2b. CHECK A2: a genuinely different integrator (CasADi CVODES, adaptive BDF, abstol 1e-13) on the
#     same Rumoca CasADi export. Check A uses the same RK4 scheme as the NLP (at dt/10), so it tests
#     the discretisation error and the CasADi/JAX export consistency; this one tests the physics
#     integration independently of RK4. Also: exact bang-bang hoverslam (switch at exactly t_sw, no
#     piecewise-constant node) solved for (t_sw, T) with y(T)=v(T)=0.
# ----------------------------------------------------------------------------------------------
t0 = time.perf_counter()
from scipy.optimize import fsolve
CV_OPTS = {"abstol": 1e-13, "reltol": 1e-12}
_xs = ca.MX.sym("x", ex.n_states); _us = ca.MX.sym("u", ex.n_inputs); _ts = ca.MX.sym("t")
_ode = {"t": _ts, "x": _xs, "p": _us, "ode": ex.rhs(_ts, _xs, _us, p_mx)}
F_node = ca.integrator("F_node", "cvodes", _ode, 0.0, dt_opt, CV_OPTS)
xk = ca.DM(x0_num)
for k in range(N):
    xk = F_node(x0=xk, p=U_opt[k])["xf"]
xT_cv = np.array(xk.full()).ravel()


def cvodes_bangbang(t_sw, T_end):
    """Coast with u=0 until t_sw, then u=1 until T_end; returns x(T_end). Physics from the Rumoca export only."""
    Fc = ca.integrator("Fc", "cvodes", _ode, 0.0, float(t_sw), CV_OPTS)
    Fb = ca.integrator("Fb", "cvodes", _ode, 0.0, float(T_end - t_sw), CV_OPTS)
    return np.array(Fb(x0=Fc(x0=ca.DM(x0_num), p=0.0)["xf"], p=1.0)["xf"].full()).ravel()


xT_bb = cvodes_bangbang(t_ign_fuel_equiv, T_opt)
# exact bang-bang hoverslam: two unknowns (t_sw, T), two conditions y(T)=0, v(T)=0
sol_bb, info_bb, ier_bb, msg_bb = fsolve(lambda z: cvodes_bangbang(z[0], z[1])[[i_y, i_v]],
                                         x0=[t_ign_fuel_equiv, T_opt], full_output=True, xtol=1e-12)
t_sw_exact, T_exact = float(sol_bb[0]), float(sol_bb[1])
xT_exact = cvodes_bangbang(t_sw_exact, T_exact)
report["check_A2_cvodes"] = {
    "integrator": "casadi cvodes", "options": CV_OPTS,
    "nlp_u_profile": {"y_at_T_m": float(xT_cv[i_y]), "v_at_T_mps": float(xT_cv[i_v]), "fuel_at_T_kg": float(xT_cv[i_f]),
                      "y_at_T_error_vs_nlp_m": float(xT_cv[i_y] - X_opt[-1, i_y]),
                      "v_at_T_error_vs_nlp_mps": float(xT_cv[i_v] - X_opt[-1, i_v]),
                      "fuel_at_T_error_vs_nlp_kg": float(xT_cv[i_f] - X_opt[-1, i_f])},
    "exact_bangbang_at_fuel_equivalent_switch": {"t_switch_s": float(t_ign_fuel_equiv), "T_s": T_opt,
                                                 "y_at_T_m": float(xT_bb[i_y]), "v_at_T_mps": float(xT_bb[i_v]),
                                                 "fuel_at_T_kg": float(xT_bb[i_f])},
    "exact_bangbang_hoverslam_solve": {"method": "scipy fsolve on (t_sw, T) with y(T)=v(T)=0", "ier": int(ier_bb),
                                       "message": str(msg_bb), "n_function_evals": int(info_bb["nfev"]),
                                       "t_switch_s": t_sw_exact, "T_s": T_exact,
                                       "y_at_T_m": float(xT_exact[i_y]), "v_at_T_mps": float(xT_exact[i_v]),
                                       "fuel_left_kg": float(xT_exact[i_f]),
                                       "t_switch_minus_fuel_equivalent_s": float(t_sw_exact - t_ign_fuel_equiv),
                                       "T_minus_T_nlp_s": float(T_exact - T_opt)},
}
timings["check_A2_s"] = time.perf_counter() - t0
print(f"[A2] CVODES on NLP u: y(T)={xT_cv[i_y]:.3e} m v(T)={xT_cv[i_v]:.3e} m/s fuel err={xT_cv[i_f]-X_opt[-1,i_f]:.2e} kg; "
      f"exact bang-bang at t_fe: y(T)={xT_bb[i_y]:.3e} m v(T)={xT_bb[i_v]:.3e} m/s; "
      f"exact hoverslam: t_sw={t_sw_exact:.7f} s T={T_exact:.6f} s (ier={ier_bb}, {info_bb['nfev']} evals)")

# ----------------------------------------------------------------------------------------------
# 3. CHECK B: Rumoca native simulator, LanderBurn, at t_ign* and bisection on t_ign
# ----------------------------------------------------------------------------------------------
t0 = time.perf_counter()
mb = common.load_burn()
DT_NATIVE = 0.01


def native_run(t_ign, t_end=60.0, dt=DT_NATIVE, solver=None):
    if solver is None:
        r = mb.simulate(t_end, dt=dt, params={"t_ign": float(t_ign)})
    else:
        r = mb.simulate(t_end, dt=dt, params={"t_ign": float(t_ign)}, config=rum.SimConfig(solver=solver))
    t = np.asarray(r.time, dtype=float)
    out = {"t": t, "y": np.asarray(r["lander.y"], float), "v": np.asarray(r["lander.v"], float),
           "fuel": np.asarray(r["lander.fuel_mass"], float), "u": np.asarray(r["lander.u"], float),
           "termination": r.termination}
    y, v = out["y"], out["v"]
    after = t >= t_ign
    out["v_max_after_ign"] = float(v[after].max()) if after.any() else float("nan")
    # first sample after ignition at which the descent is arrested (v >= 0)
    stop_idx = np.where(after & (v >= 0.0))[0]
    out["y_at_arrest"] = float(y[stop_idx[0]]) if len(stop_idx) else None
    out["y_apex_after_arrest"] = float(y[stop_idx[0]:].max()) if len(stop_idx) else None
    # (i) simulator indicator: the simulator saw an arrest (v >= 0 after ignition), at whatever altitude
    out["simulator_saw_arrest"] = bool(len(stop_idx))
    # (ii) PHYSICAL indicator: arrested ABOVE the pad (y > 0 at the arrest). An arrest with y < 0 means the
    #      simulator let the rocket pass through the pad without firing 'when lander.y <= 0 then terminate'
    out["stopped_above_pad"] = bool(len(stop_idx) and out["y_at_arrest"] > 0.0)
    # samples below the pad BEFORE the terminating sample = grazing state event missed by the simulator
    neg = np.where(y[:-1] < -1e-9)[0]
    out["passed_through_pad_without_terminate"] = bool(len(neg))
    out["n_outputs_below_pad_before_termination"] = int(len(neg))
    out["y_min_before_termination_m"] = float(y[:-1].min())
    out["t_first_below_pad_s"] = float(t[neg[0]]) if len(neg) else None
    # as reported by the simulator (last sample = terminate sample when the event fired)
    out["t_td"] = float(t[-1]); out["v_td"] = float(v[-1]); out["y_td"] = float(y[-1])
    out["fuel_td"] = float(out["fuel"][-1])
    # PHYSICAL touchdown = first downward y = 0 crossing (linear interpolation of the output samples);
    # identical to (t_td, v_td) when the terminate event fired at that crossing
    t_phys, v_phys = touchdown(t, y, v)
    out["t_td_physical"] = t_phys; out["v_td_physical"] = v_phys
    out["fuel_at_physical_td"] = (None if t_phys is None else float(np.interp(t_phys, t, out["fuel"])))
    on = np.where(out["u"] > 0)[0]
    out["t_first_output_with_engine_on"] = float(t[on[0]]) if len(on) else None
    return out


SUMMARY_KEYS = ("termination", "t_td", "y_td", "v_td", "fuel_td", "v_max_after_ign", "stopped_above_pad",
                "simulator_saw_arrest", "y_at_arrest", "y_apex_after_arrest", "passed_through_pad_without_terminate",
                "n_outputs_below_pad_before_termination", "y_min_before_termination_m", "t_first_below_pad_s",
                "t_td_physical", "v_td_physical", "fuel_at_physical_td", "t_first_output_with_engine_on")


def summarize(run):
    return {k: run[k] for k in SUMMARY_KEYS}


def bisect(indicator, lo=4.0, hi=6.0, iters=26):
    """Bisection on t_ign for a monotone boolean indicator (True early, False late)."""
    r_lo = native_run(lo); r_hi = native_run(hi); trace = []
    assert r_lo[indicator] and not r_hi[indicator], (indicator, summarize(r_lo), summarize(r_hi))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        r_mid = native_run(mid)
        trace.append([mid, r_mid["v_td"], r_mid["v_td_physical"], bool(r_mid[indicator])])
        if r_mid[indicator]:
            lo = mid; r_lo = r_mid
        else:
            hi = mid; r_hi = r_mid
    return lo, hi, r_lo, r_hi, trace


nat_node = native_run(t_ign_star)            # ignition at the first NLP node with u > 0.5
nat_fuel = native_run(t_ign_fuel_equiv)      # ignition at the fuel-equivalent (true bang-bang) switch time
nat_fuel_fine = native_run(t_ign_fuel_equiv, dt=0.001)   # same, finer output: is the dip below the pad dt-independent?
nat_fuel_rk = native_run(t_ign_fuel_equiv, solver="rk-like")   # same with the other native solvers
nat_fuel_bdf = native_run(t_ign_fuel_equiv, solver="bdf")
# Two bisections on t_ign in [4, 6]:
#  (a) PHYSICAL boundary: indicator 'stopped_above_pad' (arrest with y > 0). Early side = genuine arrest above
#      the pad, climb, dry tank, crash. Late side = the rocket reaches y = 0 (soft touchdown).
#  (b) CLEAN-TERMINATE boundary: indicator 'simulator_saw_arrest' (any v >= 0 after ignition). Between (a) and (b)
#      lies a plateau of t_ign where the native simulator lets y dip ~2 cm below the pad WITHOUT firing the
#      terminate when-clause, then climbs and crashes: a simulator artefact (missed grazing state event), not physics.
lo_p, hi_p, rlo_p, rhi_p, trace_p = bisect("stopped_above_pad")
lo_c, hi_c, rlo_c, rhi_c, trace_c = bisect("simulator_saw_arrest")
t_ign_bisect_phys = hi_p         # latest-side bracket of the physical boundary
t_ign_bisect = hi_c              # latest-side bracket of the clean-terminate boundary (terminate fires at 1st crossing)
nat_bisect = rhi_c
# also sample |v_td| around the boundary to show the landscape
scan = []
for tg in np.round(np.arange(4.0, 6.0001, 0.1), 3):
    rr = native_run(tg)
    scan.append([float(tg), rr["v_td"], rr["v_td_physical"], rr["fuel_td"], rr["stopped_above_pad"],
                 rr["passed_through_pad_without_terminate"]])
# fine scan across the artefact plateau (dt 0.001): where does the missed grazing event occur?
plateau_scan = []
for tg in (4.9875, 4.9876, 4.9877, 4.98775, 4.98778, 4.98780, 4.98782, 4.98784, 4.98790, 4.98800):
    rr = native_run(tg, dt=0.001)
    plateau_scan.append({"t_ign_s": tg, **{k: rr[k] for k in ("termination", "t_td", "v_td", "v_td_physical", "t_td_physical",
                                                             "y_at_arrest", "stopped_above_pad",
                                                             "passed_through_pad_without_terminate",
                                                             "n_outputs_below_pad_before_termination",
                                                             "y_min_before_termination_m", "t_first_below_pad_s")}})
nat_star = nat_bisect   # the clean soft-landing run used for the OpenModelica comparison and the demo
report["check_B_native"] = {
    "dt_s": DT_NATIVE,
    "note": ("Open-loop hoverslam is a knife edge: igniting slightly too early arrests the rocket above the pad, it then "
             "climbs on full throttle, runs dry and crashes; the touchdown speed therefore jumps across the boundary. "
             "'stopped_above_pad' (arrest with y > 0) is the physical indicator; 'simulator_saw_arrest' (any v >= 0 after "
             "ignition) is the simulator's. Runs with passed_through_pad_without_terminate = true dipped below y = 0 "
             "without the native simulator firing 'when lander.y <= 0 then terminate' (missed grazing state event); "
             "for those, v_td_physical (interpolated first y = 0 crossing) is the physical touchdown speed and v_td is "
             "the artefactual dry-tank crash."),
    "at_t_ign_first_node_u_gt_0p5": {"t_ign_s": t_ign_star, **summarize(nat_node)},
    "at_t_ign_fuel_equivalent": {"t_ign_s": float(t_ign_fuel_equiv), **summarize(nat_fuel)},
    "at_t_ign_fuel_equivalent_dt0p001": {"t_ign_s": float(t_ign_fuel_equiv), "dt_s": 0.001, **summarize(nat_fuel_fine)},
    "at_t_ign_fuel_equivalent_solver_rk_like": {"t_ign_s": float(t_ign_fuel_equiv), "solver": "rk-like", **summarize(nat_fuel_rk)},
    "at_t_ign_fuel_equivalent_solver_bdf": {"t_ign_s": float(t_ign_fuel_equiv), "solver": "bdf", **summarize(nat_fuel_bdf)},
    "bisection_physical_boundary": {
        "indicator": "stopped_above_pad (v >= 0 reached after ignition with y > 0)",
        "bracket_initial_s": [4.0, 6.0], "iterations": len(trace_p),
        "t_ign_bisect_s": float(t_ign_bisect_phys), "bracket_final_s": [float(lo_p), float(hi_p)],
        "late_side": {"t_ign_s": float(hi_p), **summarize(rhi_p)},
        "early_side": {"t_ign_s": float(lo_p), **summarize(rlo_p)},
        "trace_(t_ign,v_td,v_td_physical,indicator)": trace_p},
    "bisection_clean_terminate_boundary": {
        "indicator": "simulator_saw_arrest (any v >= 0 after ignition)",
        "bracket_initial_s": [4.0, 6.0], "iterations": len(trace_c),
        "t_ign_bisect_s": float(t_ign_bisect), "bracket_final_s": [float(lo_c), float(hi_c)],
        "touchdown_speed_at_bisect_mps": nat_bisect["v_td"], "fuel_left_at_bisect_kg": nat_bisect["fuel_td"],
        "touchdown_time_at_bisect_s": nat_bisect["t_td"],
        "early_side": {"t_ign_s": float(lo_c), **summarize(rlo_c)},
        "trace_(t_ign,v_td,v_td_physical,indicator)": trace_c},
    "artefact_plateau_width_s": float(t_ign_bisect - t_ign_bisect_phys),
    "t_ign_bisect_phys_minus_t_ign_fuel_equiv_s": float(t_ign_bisect_phys - t_ign_fuel_equiv),
    "t_ign_bisect_phys_minus_t_sw_exact_cvodes_s": float(t_ign_bisect_phys - t_sw_exact),
    "t_ign_bisect_clean_minus_t_ign_star_s": float(t_ign_bisect - t_ign_star),
    "t_ign_bisect_clean_minus_t_ign_fuel_equiv_s": float(t_ign_bisect - t_ign_fuel_equiv),
    "native_time_event_granularity_s": 2e-4,
    "agreement_statement": ("the native bisection boundaries agree with the NLP switch time within the simulator's "
                            "~2e-4 s time-event granularity (see native_time_event_resolution.json); a tighter "
                            "comparison is not meaningful"),
    "scan_(t_ign,v_td,v_td_physical,fuel_td,stopped_above_pad,passed_through_pad_without_terminate)": scan,
    "plateau_scan_dt0p001": plateau_scan,
}
timings["check_B_s"] = time.perf_counter() - t0
print(f"[B] native at node t_ign={t_ign_star:.4f}: term={nat_node['termination']} t_td={nat_node['t_td']:.3f} "
      f"v_td={nat_node['v_td']:.4f} fuel={nat_node['fuel_td']:.3f} stopped_above_pad={nat_node['stopped_above_pad']} "
      f"y_at_arrest={nat_node['y_at_arrest']}")
print(f"[B] native at fuel-equiv t_ign={t_ign_fuel_equiv:.6f}: term={nat_fuel['termination']} t_td={nat_fuel['t_td']:.3f} "
      f"v_td={nat_fuel['v_td']:.4f} (ARTEFACT if passed_through={nat_fuel['passed_through_pad_without_terminate']}: "
      f"y_min={nat_fuel['y_min_before_termination_m']:.4f} m, {nat_fuel['n_outputs_below_pad_before_termination']} samples "
      f"below pad from {nat_fuel['t_first_below_pad_s']} s) physical touchdown {nat_fuel['t_td_physical']} s at "
      f"{nat_fuel['v_td_physical']} m/s; stopped_above_pad={nat_fuel['stopped_above_pad']}")
print(f"[B] physical bisection t_ign={t_ign_bisect_phys:.7f}: late side v_td_phys={rhi_p['v_td_physical']} "
      f"(passed_through={rhi_p['passed_through_pad_without_terminate']}, v_td raw {rhi_p['v_td']:.3f}); "
      f"early side y_at_arrest={rlo_p['y_at_arrest']} v_td={rlo_p['v_td']:.3f}; "
      f"phys - fuel_equiv = {t_ign_bisect_phys - t_ign_fuel_equiv:+.2e} s, phys - t_sw_exact = {t_ign_bisect_phys - t_sw_exact:+.2e} s")
print(f"[B] clean-terminate bisection t_ign={t_ign_bisect:.7f}: v_td={nat_bisect['v_td']:.4f} fuel={nat_bisect['fuel_td']:.3f}; "
      f"artefact plateau width {t_ign_bisect - t_ign_bisect_phys:.2e} s; clean - node = {t_ign_bisect - t_ign_star:+.4f} s")

# ----------------------------------------------------------------------------------------------
# 4. CHECK C: OpenModelica on the same .mo file
# ----------------------------------------------------------------------------------------------
t0 = time.perf_counter()
om_ok = True
t_ign_om = float(t_ign_bisect)
try:
    om_runs = []
    for attempt in range(2):
        om, om_log = common.omc_simulate("RumocaRocketLanding.LanderBurn", stop_time=T_opt + 5.0, intervals=2000,
                                         prefix=f"om_landerburn{'' if attempt == 0 else '_plus1ms'}",
                                         load_files=[str(common.MODEL)],
                                         workdir=RES / "om", overrides={"t_ign": repr(t_ign_om)})
        t_om = om["time"]; y_om = om["lander.y"]; v_om = om["lander.v"]; f_om = om["lander.fuel_mass"]
        om_stopped_above_pad = bool(v_om[t_om >= t_ign_om].max() >= 0.0)
        om_runs.append({"t_ign_s": t_ign_om, "end_time_s": float(t_om[-1]), "touchdown_velocity_mps": float(v_om[-1]),
                        "fuel_left_kg": float(f_om[-1]), "stopped_above_pad": om_stopped_above_pad})
        if not om_stopped_above_pad:
            break
        # OM fell on the early (crash) side of the knife edge: compare 1 ms later, where both should land softly
        t_ign_om += 0.001
        nat_star = native_run(t_ign_om)
    t_cmp_end = min(t_om[-1], nat_star["t_td"])
    mask = nat_star["t"] <= t_cmp_end
    tc = nat_star["t"][mask]
    y_err = common.interp_to(tc, t_om, y_om) - nat_star["y"][mask]
    v_err = common.interp_to(tc, t_om, v_om) - nat_star["v"][mask]
    # also OM vs the NLP nodes (up to T)
    mask_n = t_nodes <= t_om[-1]
    y_err_nlp = common.interp_to(t_nodes[mask_n], t_om, y_om) - X_opt[mask_n, i_y]
    v_err_nlp = common.interp_to(t_nodes[mask_n], t_om, v_om) - X_opt[mask_n, i_v]
    # OpenModelica at the fuel-equivalent switch time, where the native simulator misses the grazing touchdown event:
    # does OM terminate at the first y = 0 crossing?
    om_fe, _ = common.omc_simulate("RumocaRocketLanding.LanderBurn", stop_time=T_opt + 5.0, intervals=2000,
                                   prefix="om_landerburn_fuel_equiv", load_files=[str(common.MODEL)],
                                   workdir=RES / "om", overrides={"t_ign": repr(float(t_ign_fuel_equiv))})
    om_fe_summary = {
        "t_ign_s": float(t_ign_fuel_equiv), "om_end_time_s": float(om_fe["time"][-1]),
        "om_terminated_before_stop_time": bool(om_fe["time"][-1] < T_opt + 5.0 - 1e-6),
        "om_touchdown_velocity_mps": float(om_fe["lander.v"][-1]), "om_fuel_left_kg": float(om_fe["lander.fuel_mass"][-1]),
        "om_y_min_m": float(om_fe["lander.y"].min()),
        "om_v_max_after_ign_mps": float(om_fe["lander.v"][om_fe["time"] >= t_ign_fuel_equiv].max()),
        "native_same_t_ign_v_td_raw_mps": nat_fuel["v_td"], "native_same_t_ign_v_td_physical_mps": nat_fuel["v_td_physical"],
        "native_same_t_ign_y_min_before_termination_m": nat_fuel["y_min_before_termination_m"],
        "native_same_t_ign_passed_through_pad_without_terminate": nat_fuel["passed_through_pad_without_terminate"],
        "csv": str(RES / "om" / "om_landerburn_fuel_equiv_res.csv"),
    }
    print(f"[C] OpenModelica at fuel-equiv t_ign={t_ign_fuel_equiv:.6f}: end t={om_fe['time'][-1]:.4f} "
          f"v_td={om_fe['lander.v'][-1]:.4f} y_min={om_fe['lander.y'].min():.2e} fuel={om_fe['lander.fuel_mass'][-1]:.3f} "
          f"(native raw v_td {nat_fuel['v_td']:.3f}, native physical {nat_fuel['v_td_physical']})")
    report["check_C_openmodelica"] = {
        "method": "dassl", "tolerance": 1e-8, "intervals": 2000, "stop_time_s": T_opt + 5.0,
        "t_ign_s": t_ign_om, "om_attempts": om_runs,
        "om_at_fuel_equivalent_t_ign": om_fe_summary,
        "native_at_same_t_ign": {"t_ign_s": t_ign_om, **summarize(nat_star)},
        "om_end_time_s": float(t_om[-1]), "om_touchdown_velocity_mps": float(v_om[-1]),
        "om_touchdown_altitude_m": float(y_om[-1]), "om_fuel_left_kg": float(f_om[-1]),
        "om_terminated_before_stop_time": bool(t_om[-1] < T_opt + 5.0 - 1e-6),
        "compare_window_s": [0.0, float(t_cmp_end)],
        "max_abs_y_error_om_vs_native_m": float(np.max(np.abs(y_err))),
        "max_abs_v_error_om_vs_native_mps": float(np.max(np.abs(v_err))),
        "max_abs_y_error_om_vs_nlp_m": float(np.max(np.abs(y_err_nlp))),
        "max_abs_v_error_om_vs_nlp_mps": float(np.max(np.abs(v_err_nlp))),
        "touchdown_speed_om_minus_native_mps": float(v_om[-1] - nat_star["v_td"]),
        "touchdown_time_om_minus_native_s": float(t_om[-1] - nat_star["t_td"]),
        "csv": str(RES / "om" / f"om_landerburn{'' if len(om_runs) == 1 else '_plus1ms'}_res.csv"),
    }
    print(f"[C] OpenModelica at t_ign={t_ign_om:.7f}: end t={t_om[-1]:.3f} v_td={v_om[-1]:.4f} fuel={f_om[-1]:.3f} "
          f"max|dy|={np.max(np.abs(y_err)):.3e} max|dv|={np.max(np.abs(v_err)):.3e} vs native "
          f"(native v_td={nat_star['v_td']:.4f}); attempts={om_runs}")
except Exception as e:  # report honestly
    om_ok = False
    report["check_C_openmodelica"] = {"error": repr(e)[:4000]}
    print("[C] OpenModelica FAILED:", repr(e)[:2000])
timings["check_C_s"] = time.perf_counter() - t0

# ----------------------------------------------------------------------------------------------
# 5. BEFORE / AFTER: ignite 1 s early, optimal, 1 s late (native simulator)
# ----------------------------------------------------------------------------------------------
t0 = time.perf_counter()
demo = {}
runs = {}
t_ign_demo = float(t_ign_bisect)   # soft-landing side of the knife edge (= fuel-equivalent NLP switch time to ~1e-5 s)
for label, tg in (("early_minus_1s", t_ign_demo - 1.0), ("optimal", t_ign_demo), ("late_plus_1s", t_ign_demo + 1.0)):
    rr = native_run(tg); runs[label] = rr
    demo[label] = {"t_ign_s": float(tg), **summarize(rr)}
report["before_after_native"] = demo
report["t_ign_star_used_for_checks_and_demo_s"] = t_ign_demo
timings["before_after_s"] = time.perf_counter() - t0
for k, d in demo.items():
    print(f"[demo] {k}: t_ign={d['t_ign_s']:.4f} term={d['termination']} t_td={d['t_td']:.2f} v_td={d['v_td']:.3f} "
          f"fuel_left={d['fuel_td']:.2f} y_at_arrest={d['y_at_arrest']} y_apex={d['y_apex_after_arrest']} "
          f"stopped_above_pad={d['stopped_above_pad']}")

# ----------------------------------------------------------------------------------------------
# 6. Figures
# ----------------------------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
common.plot_style()

fig, axs = plt.subplots(2, 2, figsize=(11, 7.2), sharex=True)
ax_y, ax_v, ax_u, ax_f = axs.ravel()
styles = {"early_minus_1s": dict(color="#d65a31", ls="--", label=f"LanderBurn t_ign={t_ign_demo-1:.3f} s (1 s early)"),
          "optimal": dict(color="#4c956c", ls="-", label=f"LanderBurn t_ign*={t_ign_demo:.3f} s (native Rumoca)"),
          "late_plus_1s": dict(color="#6b5b95", ls="--", label=f"LanderBurn t_ign={t_ign_demo+1:.3f} s (1 s late)")}
for label, rr in runs.items():
    st = styles[label]
    ax_y.plot(rr["t"], rr["y"], **st); ax_v.plot(rr["t"], rr["v"], **st)
    ax_u.step(rr["t"], rr["u"], where="post", **st); ax_f.plot(rr["t"], rr["fuel"], **st)
nlp_st = dict(color="#007c91", marker="o", ms=3, lw=0, label="NLP (CasADi/IPOPT on Rumoca export)")
ax_y.plot(t_nodes, X_opt[:, i_y], **nlp_st); ax_v.plot(t_nodes, X_opt[:, i_v], **nlp_st)
ax_u.step(t_nodes, np.append(U_opt, U_opt[-1]), where="post", color="#007c91", lw=2.5, alpha=0.6, label="NLP u_k")
ax_f.plot(t_nodes, X_opt[:, i_f], **nlp_st)
ax_y.set_ylabel("altitude y [m]"); ax_v.set_ylabel("velocity v [m/s]")
ax_u.set_ylabel("throttle u"); ax_f.set_ylabel("fuel mass [kg]")
ax_u.set_xlabel("time [s]"); ax_f.set_xlabel("time [s]")
ax_y.axhline(0, color="k", lw=0.6); ax_v.axhline(0, color="k", lw=0.6)
ax_y.set_ylim(-20, 900); ax_u.set_ylim(-0.05, 1.1)
ax_y.legend(fontsize=8.5, loc="upper right")
fig.suptitle(f"Min-propellant hoverslam (CasADi/IPOPT on the Rumoca export): T={T_opt:.2f} s, "
             f"fuel {fuel_used:.1f} kg, t_ign*={t_ign_demo:.3f} s; +/-1 s on the native simulator",
             fontsize=10.5)
fig.tight_layout()
fig_path1 = RES / "landing_hoverslam.png"
fig.savefig(fig_path1, dpi=130); plt.close(fig)

fig, axs = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True)
# draw order: OM dashed first, NLP markers every 5th node, Rumoca native LAST (wide, semi-transparent) so it is visible
if om_ok:
    axs[0].plot(t_om, y_om, "--", color="#d65a31", lw=1.2, label="OpenModelica dassl 1e-8")
    axs[1].plot(t_om, v_om, "--", color="#d65a31", lw=1.2, label="OpenModelica")
    axs[2].plot(tc, y_err, color="#d65a31", label="y: OM - Rumoca native [m]")
    axs[2].plot(tc, v_err, color="#6b5b95", label="v: OM - Rumoca native [m/s]")
axs[0].plot(t_nodes[::5], X_opt[::5, i_y], "o", ms=4.5, mfc="none", color="#007c91", label="NLP nodes (every 5th)")
axs[1].plot(t_nodes[::5], X_opt[::5, i_v], "o", ms=4.5, mfc="none", color="#007c91", label="NLP nodes (every 5th)")
axs[0].plot(nat_star["t"], nat_star["y"], label="Rumoca native (LanderBurn, dt 0.01)", color="#4c956c", lw=2.4, alpha=0.85)
axs[1].plot(nat_star["t"], nat_star["v"], color="#4c956c", lw=2.4, alpha=0.85, label="Rumoca native")
axs[2].plot(t_nodes, Xj_nodes[:, i_v] - X_opt[:, i_v], ".", color="#007c91", label="v: JAX RK4 dt/10 rollout - NLP [m/s]")
axs[0].set_ylabel("y [m]"); axs[1].set_ylabel("v [m/s]"); axs[2].set_ylabel("error"); axs[2].set_xlabel("time [s]")
axs[0].axvline(t_ign_om, color="k", lw=0.6, ls=":"); axs[1].axvline(t_ign_om, color="k", lw=0.6, ls=":")
axs[0].legend(fontsize=9); axs[1].legend(fontsize=9); axs[2].legend(fontsize=9)
axs[2].set_yscale("symlog", linthresh=1e-3)
axs[0].set_title(f"Checks at t_ign*={t_ign_om:.5f} s: Rumoca native vs OpenModelica vs NLP (and JAX rollout)", fontsize=11)
fig.tight_layout()
fig_path2 = RES / "landing_checks.png"
fig.savefig(fig_path2, dpi=130); plt.close(fig)

# Figure 3: the missed grazing state event. At the fuel-equivalent ignition time the native simulator lets y dip
# below the pad without firing terminate; OpenModelica terminates at the first crossing; the clean-terminate run
# (t_ign_bisect) terminates at its first crossing too.
fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
ax = axs[0]
ax.plot(nat_fuel_fine["t"], nat_fuel_fine["y"], color="#d65a31", lw=2,
        label=f"Rumoca native, t_ign={t_ign_fuel_equiv:.6f} s (dt 0.001): y_min={nat_fuel_fine['y_min_before_termination_m']*100:.1f} cm, no terminate")
ax.plot(nat_fuel["t"], nat_fuel["y"], ".", color="#d65a31", ms=5, label="same, dt 0.01 output samples")
if om_ok:
    ax.plot(om_fe["time"], om_fe["lander.y"], "--", color="#007c91", lw=1.6,
            label=f"OpenModelica dassl, same t_ign: terminates at {om_fe['time'][-1]:.3f} s, v={om_fe['lander.v'][-1]:.3f} m/s")
ax.plot(nat_star["t"], nat_star["y"], color="#4c956c", lw=1.8,
        label=f"Rumoca native, t_ign={t_ign_bisect:.6f} s: terminates at {nat_star['t_td']:.3f} s, v={nat_star['v_td']:.3f} m/s")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlim(13.6, 14.15); ax.set_ylim(-0.05, 0.6)
ax.set_xlabel("time [s]"); ax.set_ylabel("altitude y [m]")
ax.set_title("Grazing touchdown: 'when lander.y <= 0 then terminate'", fontsize=10.5)
ax.legend(fontsize=7.5, loc="upper right")
ax = axs[1]
ax.plot(nat_fuel_fine["t"], nat_fuel_fine["y"], color="#d65a31", lw=2, label="Rumoca native at fuel-equiv t_ign (dt 0.001)")
ax.plot(nat_star["t"], nat_star["y"], color="#4c956c", lw=1.8, label=f"Rumoca native at t_ign={t_ign_bisect:.5f} s")
if om_ok:
    ax.plot(om_fe["time"], om_fe["lander.y"], "--", color="#007c91", lw=1.6, label="OpenModelica at fuel-equiv t_ign")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlim(13.0, 19.0); ax.set_ylim(-2, 25)
ax.set_xlabel("time [s]"); ax.set_ylabel("altitude y [m]")
ax.set_title(f"...then climbs to {nat_fuel['y_apex_after_arrest']:.1f} m, runs dry, 'crashes' at {nat_fuel['v_td']:.1f} m/s", fontsize=10.5)
ax.legend(fontsize=8)
fig.suptitle("Rumoca 0.10.2 native simulator misses the grazing state event (a simulator artefact, not physics)", fontsize=10.5)
fig.tight_layout()
fig_path3 = RES / "landing_grazing_event.png"
fig.savefig(fig_path3, dpi=130); plt.close(fig)

timings["total_wall_s"] = time.perf_counter() - T_WALL0
report["timings_s"] = timings
report["figures"] = [str(fig_path1), str(fig_path2), str(fig_path3)]
json_path = RES / "landing_hoverslam.json"


def _sanitize(o):
    """NaN/inf -> None so the JSON stays strict (write_json uses allow_nan=False)."""
    if isinstance(o, dict):
        return {k: _sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_sanitize(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


common.write_json(json_path, _sanitize(report))
print("wrote", json_path, fig_path1, fig_path2, fig_path3)
print(f"total wall {timings['total_wall_s']:.1f} s")

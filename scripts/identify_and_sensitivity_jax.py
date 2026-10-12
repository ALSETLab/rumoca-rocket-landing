"""Differentiate through the Modelica rocket with JAX (Rumoca jax-ode export).

Stream "jax" of the AMC 2026 Rumoca open-challenge entry.

  1. Rumoca compiles RumocaRocketLanding.Lander to a JAX right-hand side; common.jax_rollout
     wraps it in a jit-compiled RK4 scan (dt = 0.02 s). Python only integrates/optimizes/plots.
  2. Drag identification: "flight data" = Rumoca NATIVE simulation of the untouched 2021
     RocketLanding.SimpleLander.rocket (launch from rest, full thrust until fuel-out at 10 s,
     then coast) + Gaussian noise (sigma 2 m on altitude, seed 2026). cd (true 0.2) is fitted
     from cd = 0.6 by L-BFGS-B driven by jax.grad through the whole trajectory. A second fit on
     noiseless data quantifies the model mismatch (sign(v) drag law vs smooth v*sqrt(v^2+eps^2),
     native event solver vs fixed-step RK4). A third fit uses an INDEPENDENT reference: the
     OpenModelica (dassl, tol 1e-8) CSV of the same original model in results/scratch, so the
     identification is not only Rumoca-against-Rumoca.
  3. Landing sensitivity: smooth throttle u(t) = sigmoid((t - t_ign)/tau), tau = 0.05 s, fixed
     16 s horizon from [800 m, -70 m/s, 100 kg] (12 s cuts the landing off, see the code comment).
     The RK4 rollout holds u constant over each step (zero-order hold), so the throttle is SAMPLED
     AT THE STEP MIDPOINT t_k + dt/2: sampling at the left endpoint delivers the thrust dt/2 = 10 ms
     late on average and biases the hoverslam time by exactly that (demonstrated in the script at
     dt = 0.02 and 0.01 s). The touchdown speed is defined differentiably:
     first sample with y <= 0 (stop_gradient on the index) and linear interpolation of v at the
     y = 0 crossing with differentiated weights. The hoverslam ignition time t_ign_opt is found
     by bisection on 4..6 s. Gradients of the touchdown speed w.r.t. t_ign, cd, Fengine and
     EmptyRocket come from jax.grad and are checked against central finite differences.

Run:  cd C:/GIT/rumoca-rocket-landing && PYTHONIOENCODING=utf-8 JAX_PLATFORMS=cpu \
        .venv/Scripts/python.exe scripts/identify_and_sensitivity_jax.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402  (sets JAX_PLATFORMS=cpu, MPLBACKEND=Agg)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rumoca as rum  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

T_WALL0 = time.time()
DT = 0.02
RESULTS = common.RESULTS
RESULTS.mkdir(parents=True, exist_ok=True)
report: dict = {"rumoca_version": str(rum.version()), "dt": DT, "jax_x64": True}

# ----------------------------------------------------------------------------------------------
# 1. Rumoca -> JAX export and jit RK4 rollout
# ----------------------------------------------------------------------------------------------
lander = common.load_lander()
exj = lander.to_jax()
common.validate_layout(exj)
rollout = common.jax_rollout(exj, dt=DT)
PNAMES = list(exj.parameter_names)
P0 = jnp.array(exj.default_parameters, dtype=jnp.float64)
IDX = {n: i for i, n in enumerate(PNAMES)}
report["lander"] = {"state_names": list(exj.state_names), "input_names": list(exj.input_names),
                    "parameter_names": PNAMES, "default_parameters": [float(x) for x in exj.default_parameters],
                    "default_states": [float(x) for x in exj.default_states]}
print("JAX export OK:", PNAMES, report["lander"]["default_parameters"])

# ----------------------------------------------------------------------------------------------
# 2. Drag identification against Rumoca's native simulation of the ORIGINAL 2021 model
# ----------------------------------------------------------------------------------------------
T_ID = 25.0
N_ID = int(round(T_ID / DT))
t_grid = np.arange(N_ID + 1) * DT
orig = common.load_original_rocket()
r = orig.simulate(T_ID, dt=DT)
t_nat = np.asarray(r.time, dtype=float)
y_nat = np.asarray(r["y"], dtype=float)
v_nat = np.asarray(r["v"], dtype=float)
fm_nat = np.asarray(r["fuel_mass"], dtype=float)
y_true = common.interp_to(t_grid, t_nat, y_nat)  # noiseless altitude on the RK4 grid
v_true = common.interp_to(t_grid, t_nat, v_nat)
fm_true = common.interp_to(t_grid, t_nat, fm_nat)
k_burnout = int(np.argmax(fm_nat <= 1e-9))
report["flight_data"] = {
    "source": "Rumoca native simulate(25.0, dt=0.02) of RocketLanding.SimpleLander.rocket (untouched, "
              + str(common.ORIGINAL_ROCKET) + ")",
    "n_native_samples": int(len(t_nat)), "termination": r.termination,
    "burnout_time_native": float(t_nat[k_burnout]) if fm_nat.min() <= 1e-9 else None,
    "apogee_m": float(y_nat.max()), "apogee_time_s": float(t_nat[int(np.argmax(y_nat))]),
    "max_speed_mps": float(v_nat.max()), "y_end_m": float(y_nat[-1]), "v_end_mps": float(v_nat[-1]),
}
print(f"native original rocket: {len(t_nat)} samples, apogee {y_nat.max():.1f} m at {t_nat[y_nat.argmax()]:.2f} s, "
      f"burnout at {report['flight_data']['burnout_time_native']} s, y(25)={y_nat[-1]:.2f} m")

SIGMA_Y = 2.0
rng = np.random.default_rng(2026)
y_meas = y_true + SIGMA_Y * rng.standard_normal(y_true.shape)
report["flight_data"].update({"noise_sigma_y_m": SIGMA_Y, "noise_seed": 2026})

# Known throttle schedule: full thrust while fuel remains (100 kg at 10 kg/s -> 10 s), then coast.
U_ID = (t_grid[:-1] < 10.0 - 1e-12).astype(float)[:, None]
X0_ID = jnp.array([0.0, 0.0, 100.0], dtype=jnp.float64)


def make_loss(y_data):
    y_data = jnp.asarray(y_data)

    def loss(cd):
        p = P0.at[IDX["cd"]].set(cd)
        X = rollout(p, X0_ID, U_ID)
        return jnp.mean((X[:, 0] - y_data) ** 2)
    return jax.jit(jax.value_and_grad(loss))


def fit_cd(y_data, cd0=0.6, label=""):
    vg = make_loss(y_data)
    hist = []

    def fun(z):
        val, g = vg(float(z[0]))
        hist.append((float(z[0]), float(val), float(g)))
        return float(val), np.array([float(g)])

    t0 = time.time()
    res = minimize(fun, x0=np.array([cd0]), jac=True, method="L-BFGS-B", bounds=[(1e-3, 5.0)],
                   options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 200})
    cd_hat = float(res.x[0])
    p = P0.at[IDX["cd"]].set(cd_hat)
    X = np.asarray(rollout(p, X0_ID, U_ID))
    resid = X[:, 0] - np.asarray(y_data)
    out = {"label": label, "cd_start": cd0, "cd_fit": cd_hat, "cd_true": 0.2, "cd_error": cd_hat - 0.2,
           "cd_rel_error_pct": 100 * (cd_hat - 0.2) / 0.2, "final_loss_mean_sq_m2": float(res.fun),
           "residual_rms_m": float(np.sqrt(np.mean(resid ** 2))), "residual_max_abs_m": float(np.abs(resid).max()),
           "n_fun_evals": len(hist), "n_iterations": int(res.nit), "success": bool(res.success),
           "message": str(res.message), "wall_s": time.time() - t0,
           "history": [{"cd": h[0], "loss": h[1], "grad": h[2]} for h in hist]}
    print(f"fit {label}: cd={cd_hat:.6f} (err {cd_hat-0.2:+.2e}), RMS {out['residual_rms_m']:.4f} m, "
          f"{len(hist)} evals, {res.nit} it, {res.message}")
    return out, X


fit_noisy, X_noisy = fit_cd(y_meas, 0.6, "noisy")
fit_clean, X_clean = fit_cd(y_true, 0.6, "noiseless")
# Model-mismatch reference: the JAX Lander at the TRUE cd vs the native original model
X_truecd = np.asarray(rollout(P0, X0_ID, U_ID))
mismatch = {
    "y_rms_m": float(np.sqrt(np.mean((X_truecd[:, 0] - y_true) ** 2))),
    "y_max_abs_m": float(np.abs(X_truecd[:, 0] - y_true).max()),
    "v_rms_mps": float(np.sqrt(np.mean((X_truecd[:, 1] - v_true) ** 2))),
    "fuel_end_jax_kg": float(X_truecd[-1, 2]), "fuel_end_native_kg": float(fm_true[-1]),
    "note": "JAX Lander (smooth drag v*sqrt(v^2+eps^2), eps=0.1, RK4 dt=0.02) at cd=0.2 vs Rumoca native "
            "solve of the original sign(v) drag model with fuel-out event",
}
report["drag_identification"] = {"noisy": fit_noisy, "noiseless": fit_clean, "model_mismatch_at_true_cd": mismatch,
                                 "loss_and_grad_at_start_noisy": {"cd": 0.6, "loss": fit_noisy["history"][0]["loss"],
                                                                  "grad": fit_noisy["history"][0]["grad"]}}
print(f"model mismatch at true cd: y RMS {mismatch['y_rms_m']:.2e} m, max {mismatch['y_max_abs_m']:.2e} m")

# Independent reference: OpenModelica (dassl, tol 1e-8, 0.05 s output grid) CSV of the SAME original
# 2021 model, produced by omc rather than Rumoca. Fitting the Rumoca-JAX Lander to it shows the
# identification is not merely Rumoca-against-Rumoca self-consistency.
OM_CSV = common.ROOT / "results" / "scratch" / "om_rocket_baseline_res.csv"
om = np.loadtxt(OM_CSV, delimiter=",", skiprows=1)
with open(OM_CSV, encoding="utf-8") as fh:
    om_cols = [c.strip().strip('"') for c in fh.readline().strip().split(",")]
t_om, y_om_raw, v_om_raw, fm_om_raw = (om[:, om_cols.index(c)] for c in ["time", "y", "v", "fuel_mass"])
y_om = common.interp_to(t_grid, t_om, y_om_raw)
v_om = common.interp_to(t_grid, t_om, v_om_raw)
fit_om, X_om = fit_cd(y_om, 0.6, "om_csv")
om_ref = {
    "source": str(OM_CSV) + " (OpenModelica omc, dassl, tol 1e-8, 1200 intervals over 60 s; rows 0..25 s used, "
                            "linearly interpolated onto the 0.02 s RK4 grid)",
    "n_rows": int(len(t_om)), "apogee_m": float(y_om_raw.max()), "apogee_time_s": float(t_om[int(np.argmax(y_om_raw))]),
    "burnout_time_s": float(t_om[int(np.argmax(fm_om_raw <= 1e-9))]), "y_25s_m": float(y_om[-1]),
    "om_vs_native_y_rms_m": float(np.sqrt(np.mean((y_om - y_true) ** 2))),
    "om_vs_native_y_max_abs_m": float(np.abs(y_om - y_true).max()),
    "jax_true_cd_vs_om_y_rms_m": float(np.sqrt(np.mean((X_truecd[:, 0] - y_om) ** 2))),
    "jax_true_cd_vs_om_y_max_abs_m": float(np.abs(X_truecd[:, 0] - y_om).max()),
    "jax_true_cd_vs_om_v_rms_mps": float(np.sqrt(np.mean((X_truecd[:, 1] - v_om) ** 2))),
}
report["drag_identification"]["om_csv_independent"] = fit_om
report["drag_identification"]["om_reference"] = om_ref
print(f"OpenModelica reference: apogee {om_ref['apogee_m']:.3f} m at {om_ref['apogee_time_s']:.3f} s, burnout "
      f"{om_ref['burnout_time_s']} s; OM vs native y RMS {om_ref['om_vs_native_y_rms_m']:.2e} m; "
      f"JAX(cd=0.2) vs OM y RMS {om_ref['jax_true_cd_vs_om_y_rms_m']:.2e} m")

# Figure 1 --------------------------------------------------------------------------------------
common.plot_style()
fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
ax[0].plot(t_grid, y_meas, ".", ms=2.5, color="0.55", label="data: native sim + noise (sigma 2 m)")
ax[0].plot(t_grid, y_true, "k--", lw=1, label="native original model (noiseless)")
ax[0].plot(t_grid, X_noisy[:, 0], lw=2, label=f"JAX Lander fit, cd = {fit_noisy['cd_fit']:.4f}")
ax[0].plot(t_grid, np.asarray(rollout(P0.at[IDX['cd']].set(0.6), X0_ID, U_ID))[:, 0], ":", lw=1.5,
           label="JAX Lander, start cd = 0.6")
ax[0].axvline(10.0, color="0.7", lw=0.8)
ax[0].text(10.1, 20, "fuel-out", color="0.4", fontsize=9)
ax[0].set_xlabel("time [s]"); ax[0].set_ylabel("altitude y [m]"); ax[0].legend(fontsize=8, loc="upper left")
ax[0].set_title("Drag identification: data vs fit", fontsize=11)
ax[1].plot(t_grid, X_noisy[:, 0] - y_meas, lw=0.8, label=f"fit - data, RMS {fit_noisy['residual_rms_m']:.2f} m")
ax[1].plot(t_grid, X_clean[:, 0] - y_true, lw=1.2, label=f"noiseless fit - truth, RMS {fit_clean['residual_rms_m']:.1e} m")
ax[1].plot(t_grid, X_om[:, 0] - y_om, lw=1.0, ls="--",
           label=f"fit to OpenModelica CSV (cd = {fit_om['cd_fit']:.5f}) - OM, RMS {fit_om['residual_rms_m']:.1e} m")
ax[1].set_xlabel("time [s]"); ax[1].set_ylabel("altitude residual [m]"); ax[1].legend(fontsize=8)
ax[1].set_title("Residuals", fontsize=11)
h = fit_noisy["history"]
ax[2].semilogy([k for k in range(len(h))], [max(x["loss"], 1e-16) for x in h], "o-", ms=4, label="noisy data")
h2 = fit_clean["history"]
ax[2].semilogy([k for k in range(len(h2))], [max(x["loss"], 1e-16) for x in h2], "s-", ms=4, label="noiseless data")
ax[2].set_xlabel("L-BFGS-B function evaluation"); ax[2].set_ylabel("mean squared altitude residual [m$^2$]")
ax[2].set_title("Loss history (jax.grad through RK4 rollout)", fontsize=11); ax[2].legend(fontsize=8)
ax2b = ax[2].twinx()
ax2b.plot(range(len(h)), [x["cd"] for x in h], "^--", color="0.4", ms=4, lw=0.8)
ax2b.axhline(0.2, color="0.6", lw=0.8, ls=":")
ax2b.set_ylabel("cd iterate", color="0.4")
fig.suptitle("Rumoca jax-ode export of RumocaRocketLanding.Lander: fitting cd to the original 2021 model's flight",
             fontsize=11)
fig.tight_layout()
fig_id = RESULTS / "identify_drag_jax.png"
fig.savefig(fig_id, dpi=130)
plt.close(fig)
print("wrote", fig_id)

# ----------------------------------------------------------------------------------------------
# 3. Sensitivity of the landing (hoverslam) through the differentiable touchdown metric
# ----------------------------------------------------------------------------------------------
# Horizon: the fall from 800 m at 70-90 m/s plus the burn takes 12-13 s, so a 12 s horizon cuts the
# landing off (the bisection then finds the 'touches down before the horizon ends' boundary, not the
# hoverslam). 16 s leaves margin; fuel stays positive until touchdown and the mass never reaches zero.
T_H = 16.0
N_H = int(round(T_H / DT))
TAU = 0.05
# Zero-order hold: common.jax_rollout applies controls[k] over the whole RK4 step [t_k, t_k + dt].
# Sampling a continuous schedule at the LEFT endpoint t_k therefore delivers it dt/2 late on average;
# sampling at the step midpoint t_k + dt/2 removes that bias (second-order accurate hold).
U_OFFSET = DT / 2
t_ctrl = jnp.arange(N_H) * DT
X0_LAND = jnp.array([800.0, -70.0, 100.0], dtype=jnp.float64)
SENS_NAMES = ["t_ign", "cd", "Fengine", "EmptyRocket"]


def throttle(t_ign, offset=U_OFFSET, tc=t_ctrl):
    """Smooth ignition: u(t) = sigmoid((t - t_ign)/tau), tau = 0.05 s (10-90 % rise in 0.22 s),
    sampled at t_k + offset (offset = dt/2 -> midpoint of the hold interval)."""
    return jax.nn.sigmoid((tc + offset - t_ign) / TAU)[:, None]


def params_of(theta):
    _, cd, Fengine, EmptyRocket = theta
    return P0.at[IDX["cd"]].set(cd).at[IDX["Fengine"]].set(Fengine).at[IDX["EmptyRocket"]].set(EmptyRocket)


def landing_rollout(theta):
    return rollout(params_of(theta), X0_LAND, throttle(theta[0]))


def touchdown_metric(X, dt):
    """Differentiable touchdown speed: index of the first y<=0 sample is frozen (stop_gradient),
    the linear-interpolation weight at the y=0 crossing is differentiated."""
    y, v = X[:, 0], X[:, 1]
    below = y <= 0.0
    touched = jnp.any(below)
    k = jax.lax.stop_gradient(jnp.argmax(below))  # first crossing sample (0 if none)
    k = jnp.maximum(k, 1)
    y0, y1 = y[k - 1], y[k]
    v0, v1 = v[k - 1], v[k]
    w = y0 / (y0 - y1)  # fraction of the step at which y = 0
    v_td = v0 + w * (v1 - v0)
    t_td = (k - 1 + w) * dt
    return jnp.where(touched, v_td, jnp.nan), touched, t_td, k


def touchdown(theta):
    return touchdown_metric(landing_rollout(theta), DT)


v_td_fn = jax.jit(lambda th: touchdown(th)[0])
touched_fn = jax.jit(lambda th: touchdown(th)[1])
t_td_fn = jax.jit(lambda th: touchdown(th)[2])
k_td_fn = jax.jit(lambda th: touchdown(th)[3])
grad_fn = jax.jit(jax.grad(lambda th: touchdown(th)[0]))


def make_variant(dt, offset):
    """Touchdown functions for a different RK4 step / throttle sampling offset (ZOH demonstration)."""
    roll = rollout if dt == DT else common.jax_rollout(exj, dt=dt)
    tc = jnp.arange(int(round(T_H / dt))) * dt

    def td(theta):
        return touchdown_metric(roll(params_of(theta), X0_LAND, throttle(theta[0], offset, tc)), dt)
    return jax.jit(lambda th: td(th)[0]), jax.jit(lambda th: td(th)[1])


def bisect_hoverslam(touched_flag, lo=4.0, hi=6.0, iters=60):
    """Smallest t_ign that touches down within the horizon (earlier ignitions stall and climb)."""
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if bool(touched_flag(theta_of(mid))):
            hi = mid
        else:
            lo = mid
    return hi
nominal = {"cd": 0.2, "Fengine": 2500.0, "EmptyRocket": 100.0}


def theta_of(t_ign, **over):
    d = dict(nominal); d.update(over)
    return jnp.array([t_ign, d["cd"], d["Fengine"], d["EmptyRocket"]], dtype=jnp.float64)


# Sweep t_ign over 4..6 s
t_sweep = np.linspace(4.0, 6.0, 201)
v_sweep = np.array([float(v_td_fn(theta_of(t))) for t in t_sweep])
touch_sweep = np.array([bool(touched_fn(theta_of(t))) for t in t_sweep])
# y_min for the non-touching cases (rocket stops above the pad and climbs)
ymin_sweep = np.array([float(jnp.min(landing_rollout(theta_of(t))[:, 0])) for t in t_sweep])
assert touch_sweep[0] is np.False_ or not touch_sweep[0], "t_ign=4 unexpectedly touches down"
assert touch_sweep[-1], "t_ign=6 unexpectedly does not touch down"

# Hoverslam: the smallest t_ign that touches down. |v_td| is minimal there (v_td -> 0-), and
# earlier ignitions never reach the pad in the horizon. Bisection on the touchdown flag.
t_opt = bisect_hoverslam(touched_fn)
v_opt = float(v_td_fn(theta_of(t_opt)))
t_td_opt = float(t_td_fn(theta_of(t_opt)))
k_td_opt = int(k_td_fn(theta_of(t_opt)))  # first sample with y <= 0 (touchdown sample)
X_opt = np.asarray(landing_rollout(theta_of(t_opt)))
assert k_td_opt == int(np.argmax(X_opt[:, 0] <= 0.0))
fuel_td_opt = float(X_opt[k_td_opt, 2])
if t_td_opt > T_H - 1.0:
    raise RuntimeError(f"touchdown at the optimum ({t_td_opt:.3f} s) too close to the horizon {T_H} s")
if fuel_td_opt <= 0:
    raise RuntimeError("fuel exhausted before touchdown at the optimum")
print(f"hoverslam: t_ign_opt = {t_opt:.6f} s, v_td = {v_opt:.4f} m/s, touchdown at t = {t_td_opt:.3f} s "
      f"(sample k = {k_td_opt}, t_k = {k_td_opt*DT:.2f} s), fuel left {fuel_td_opt:.2f} kg")

# ZOH demonstration: the same sigmoid sampled at the LEFT endpoint of each hold interval moves the
# hoverslam EARLIER by dt/2 (the hold delivers thrust late, so the optimiser compensates); the
# midpoint-sampled optimum is insensitive to dt. (dt = 0.01 needs a second jit rollout, ~0.2 s.)
zoh_variants = []
for dt_v, off_v, lab in [(DT, 0.0, "left endpoint"), (DT, DT / 2, "midpoint"),
                         (0.01, 0.0, "left endpoint"), (0.01, 0.005, "midpoint")]:
    v_fn_v, touched_v = make_variant(dt_v, off_v)
    t_opt_v = bisect_hoverslam(touched_v)
    zoh_variants.append({"dt_s": dt_v, "sampling": lab, "u_offset_s": off_v, "t_ign_opt_s": t_opt_v,
                         "shift_vs_main_ms": 1000 * (t_opt_v - t_opt),
                         "v_td_at_main_opt_plus_100ms_mps": float(v_fn_v(theta_of(t_opt + 0.1)))})
    print(f"  ZOH variant dt={dt_v:.3f} {lab:13s}: t_ign_opt = {t_opt_v:.6f} s (shift {1000*(t_opt_v-t_opt):+.3f} ms), "
          f"v_td(t_opt+0.1) = {zoh_variants[-1]['v_td_at_main_opt_plus_100ms_mps']:.3f} m/s")
zoh_shift_left_dt02_ms = zoh_variants[0]["shift_vs_main_ms"]
assert abs(zoh_shift_left_dt02_ms + 1000 * DT / 2) < 0.1, "left-endpoint sampling should lag by dt/2"

# Near the optimum the touchdown speed follows a square-root law, v_td ~ -sqrt(K*(t_ign - t_opt)):
# y(t) is tangent to the pad at the perfect hoverslam, so a late ignition by delta lowers y_min by
# ~|v_ign|*delta and the pad is hit at v = -sqrt(2*a*|v_ign|*delta). The derivative is therefore
# infinite AT the optimum. We verify the law and evaluate gradients at a design point with margin.
deltas = np.array([0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5])
v_delta = np.array([float(v_td_fn(theta_of(t_opt + d))) for d in deltas])
K_fit = float(np.sum(v_delta ** 2 * deltas) / np.sum(deltas ** 2))  # LS fit of v^2 = K*delta
sqrt_law = {"deltas_s": deltas.tolist(), "v_td_mps": v_delta.tolist(), "K_fit_m2_s3": K_fit,
            "v_pred_sqrt_law_mps": (-np.sqrt(K_fit * deltas)).tolist(),
            "max_rel_dev_pct": float(100 * np.max(np.abs((-np.sqrt(K_fit * deltas) - v_delta) / v_delta)))}
v_100ms = float(v_td_fn(theta_of(t_opt + 0.1)))
v_10ms = float(v_td_fn(theta_of(t_opt + 0.01)))
v_1s = float(v_td_fn(theta_of(t_opt + 1.0)))
print(f"sqrt law: K = {K_fit:.1f}; 10 ms late -> {v_10ms:.2f} m/s, 100 ms late -> {v_100ms:.2f} m/s, "
      f"1 s late -> {v_1s:.2f} m/s; max dev {sqrt_law['max_rel_dev_pct']:.1f} %")

# Gradients at the design point t_des = t_opt + 0.1 s (checked vs central finite differences)
DELTA_DES = 0.1
t_des = t_opt + DELTA_DES
th_des = theta_of(t_des)
v_des = float(v_td_fn(th_des))
g_ad = np.asarray(grad_fn(th_des))
# FD steps chosen so that each perturbs v_td by ~0.1-0.2 m/s (v_td is strongly curved near the cliff)
fd_steps = {"t_ign": 2e-3, "cd": 2e-4, "Fengine": 1.0, "EmptyRocket": 0.05}
grads = {}
for i, name in enumerate(SENS_NAMES):
    hstep = fd_steps[name]
    e = jnp.zeros(4).at[i].set(hstep)
    vp = float(v_td_fn(th_des + e)); vm = float(v_td_fn(th_des - e))
    g_fd = (vp - vm) / (2 * hstep)
    grads[name] = {"jax_grad": float(g_ad[i]), "central_fd": float(g_fd), "fd_step": hstep,
                   "rel_diff": float(abs(g_ad[i] - g_fd) / max(abs(g_fd), 1e-300)),
                   "unit": {"t_ign": "(m/s)/s", "cd": "(m/s)/1", "Fengine": "(m/s)/N", "EmptyRocket": "(m/s)/kg"}[name]}
    print(f"  d v_td/d {name:12s}: jax {g_ad[i]: .6e}  FD {g_fd: .6e}  rel diff {grads[name]['rel_diff']:.2e}")

# Same check at a second design point further from the singularity (t_opt + 0.5 s)
th_des2 = theta_of(t_opt + 0.5)
g_ad2 = np.asarray(grad_fn(th_des2))
grads2 = {}
for i, name in enumerate(SENS_NAMES):
    hstep = fd_steps[name]
    e = jnp.zeros(4).at[i].set(hstep)
    g_fd = (float(v_td_fn(th_des2 + e)) - float(v_td_fn(th_des2 - e))) / (2 * hstep)
    grads2[name] = {"jax_grad": float(g_ad2[i]), "central_fd": float(g_fd), "fd_step": hstep,
                    "rel_diff": float(abs(g_ad2[i] - g_fd) / max(abs(g_fd), 1e-300))}

# Also: how the gradient wrt t_ign blows up approaching the optimum (jax.grad at several deltas)
g_blowup = [{"delta_s": float(d), "dv_dtign_jax": float(np.asarray(grad_fn(theta_of(t_opt + d)))[0]),
             "sqrt_law_prediction": float(-0.5 * np.sqrt(K_fit / d))} for d in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]]

# One-sentence translation (secant from the exact hoverslam, and tangent at the design point)
sentence = (f"Igniting 100 ms late costs {abs(v_100ms - v_opt):.1f} m/s of touchdown speed (from {v_opt:.2f} to "
            f"{v_100ms:.2f} m/s), and the first 10 ms alone cost {abs(v_10ms - v_opt):.1f} m/s: the perfect hoverslam "
            f"sits on a square-root cliff (v_td ~ -sqrt({K_fit:.0f}*delta)), so its derivative is infinite at the "
            f"optimum; 100 ms after it the local slope is {grads['t_ign']['jax_grad']:.1f} (m/s)/s, i.e. a further "
            f"10 ms of delay costs {abs(grads['t_ign']['jax_grad'])*0.01:.2f} m/s.")
print(sentence)

report["landing_sensitivity"] = {
    "throttle_schedule": "u(t) = sigmoid((t - t_ign)/tau), tau = 0.05 s, sampled at the RK4 step midpoint t_k + dt/2 "
                         "(zero-order hold over each step); no fuel gate (fuel stays > 0 until touchdown; mass stays > 0 "
                         "over the horizon)",
    "u_sampling_offset_s": U_OFFSET,
    "zoh_control_hold_check": {
        "note": "common.jax_rollout holds controls[k] over [t_k, t_k+dt]; sampling the sigmoid at t_k biases the "
                "hoverslam EARLY by dt/2 (thrust arrives dt/2 late on average); midpoint sampling removes the bias "
                "and is dt-insensitive. v_td at delta = +0.1 s is the same for all variants (the cliff is unaffected).",
        "variants": zoh_variants, "left_endpoint_dt0.02_shift_ms": zoh_shift_left_dt02_ms},
    "tau_s": TAU, "horizon_s": T_H, "x0": [800.0, -70.0, 100.0],
    "touchdown_metric": "first sample with y<=0 (index under stop_gradient), v linearly interpolated at the y=0 crossing "
                        "with differentiated weights; NaN when the rocket never reaches the pad in the horizon",
    "sweep": {"t_ign_s": t_sweep.tolist(), "v_td_mps": [None if np.isnan(x) else float(x) for x in v_sweep],
              "touched": touch_sweep.tolist(), "y_min_m": ymin_sweep.tolist()},
    "t_ign_opt_s": t_opt, "v_td_at_opt_mps": v_opt, "t_touchdown_at_opt_s": t_td_opt,
    "touchdown_sample_index": k_td_opt, "touchdown_sample_time_s": k_td_opt * DT,
    "fuel_left_at_opt_kg": fuel_td_opt,
    "fuel_left_at_opt_note": "fuel at the first sample with y <= 0 (k = argmax(y <= 0)); the sample before it has "
                             + f"{float(X_opt[k_td_opt - 1, 2]):.3f} kg",
    "bisection": {"interval": [4.0, 6.0], "iterations": 60, "criterion": "touchdown flag"},
    "sqrt_law_near_optimum": sqrt_law,
    "late_ignition_cost": {"v_td_plus_10ms_mps": v_10ms, "v_td_plus_100ms_mps": v_100ms, "v_td_plus_1s_mps": v_1s,
                           "cost_10ms_mps": abs(v_10ms - v_opt), "cost_100ms_mps": abs(v_100ms - v_opt),
                           "cost_1s_mps": abs(v_1s - v_opt)},
    "design_point": {"t_ign_s": t_des, "delta_from_opt_s": DELTA_DES, "v_td_mps": v_des,
                     "nominal_parameters": nominal, "gradients": grads},
    "design_point_2": {"t_ign_s": t_opt + 0.5, "delta_from_opt_s": 0.5, "v_td_mps": float(v_td_fn(th_des2)),
                       "gradients": grads2},
    "gradient_blowup_toward_optimum": g_blowup,
    "sentence": sentence,
}

# Cross-check against Rumoca's NATIVE simulation of LanderBurn (step throttle with fuel gate,
# terminate at touchdown): same Modelica source, different compiler target and integrator.
burn = common.load_burn()
native_check = []
for t_ign_chk, v_jax_chk in [(6.0, float(v_td_fn(theta_of(6.0)))), (t_opt, v_opt), (t_des, v_des)]:
    rb = burn.simulate(40.0, dt=DT, params={"t_ign": float(t_ign_chk)})
    tb = np.asarray(rb.time); yb = np.asarray(rb["lander.y"]); vb = np.asarray(rb["lander.v"])
    fb = np.asarray(rb["lander.fuel_mass"])
    native_check.append({"t_ign_s": float(t_ign_chk), "native_termination": rb.termination,
                         "native_t_end_s": float(tb[-1]), "native_y_end_m": float(yb[-1]),
                         "native_v_end_mps": float(vb[-1]), "native_fuel_end_kg": float(fb[-1]),
                         "jax_v_td_mps": v_jax_chk, "jax_t_td_s": float(t_td_fn(theta_of(t_ign_chk))),
                         "abs_diff_mps": abs(float(vb[-1]) - v_jax_chk)})
    print(f"native LanderBurn t_ign={t_ign_chk:.4f}: {rb.termination} at t={tb[-1]:.3f} s, v={vb[-1]:.3f} m/s, "
          f"fuel {fb[-1]:.2f} kg | JAX sigmoid v_td={v_jax_chk:.3f} m/s (diff {abs(float(vb[-1]) - v_jax_chk):.3f})")
# Native hoverslam with the STEP throttle: bisection on "direct touchdown with fuel left" (an early
# ignition stalls above the pad, climbs, empties the tank and crashes; fuel at touchdown is then 0).
def native_direct_touchdown(t_ign_chk):
    rb = burn.simulate(40.0, dt=DT, params={"t_ign": float(t_ign_chk)})
    fb = np.asarray(rb["lander.fuel_mass"]); vb = np.asarray(rb["lander.v"]); tb = np.asarray(rb.time)
    return (rb.termination == "touchdown" and fb[-1] > 1e-6), float(vb[-1]), float(tb[-1]), float(fb[-1])

t0 = time.time()
lo_n, hi_n = 4.0, 6.0
for _ in range(28):
    mid = 0.5 * (lo_n + hi_n)
    if native_direct_touchdown(mid)[0]:
        hi_n = mid
    else:
        lo_n = mid
ok_n, v_n, ttd_n, fuel_n = native_direct_touchdown(hi_n)
shift_ms = 1000 * (hi_n - t_opt)
print(f"native step-throttle hoverslam: t_ign = {hi_n:.6f} s (sigmoid optimum {t_opt:.6f}, shift {shift_ms:+.3f} ms), "
      f"v_td = {v_n:.3f} m/s at t = {ttd_n:.3f} s, fuel {fuel_n:.2f} kg  [{time.time()-t0:.1f} s]")
report["landing_sensitivity"]["native_step_hoverslam"] = {
    "t_ign_opt_native_s": hi_n, "v_td_native_mps": v_n, "t_touchdown_native_s": ttd_n, "fuel_left_native_kg": fuel_n,
    "shift_vs_sigmoid_optimum_ms": shift_ms, "bisection_iterations": 28,
    "note": "step throttle (native LanderBurn, event-located ignition) vs midpoint-sampled sigmoid throttle (JAX RK4): "
            "the hoverslam ignition times agree to this shift. With left-endpoint sampling the JAX optimum was "
            f"{zoh_shift_left_dt02_ms:+.2f} ms off (dt/2 zero-order-hold delay), which on the square-root cliff is "
            "worth ~5 m/s of touchdown speed; the physical step-vs-sigmoid difference is the sub-ms remainder"}
report["landing_sensitivity"]["native_cross_check"] = {
    "note": "Rumoca native simulate of RumocaRocketLanding.LanderBurn (step throttle at t_ign, fuel gate, terminate at "
            "touchdown) vs the JAX RK4 rollout with the midpoint-sampled sigmoid throttle (tau 0.05 s); native v is "
            "the last sample at the terminate event. At the hoverslam itself the step schedule can stall and climb "
            "(one-sided boundary), so only cases away from the cliff are expected to agree closely",
    "cases": native_check}

# Figure 2 --------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
ok = touch_sweep
ax[0].plot(t_sweep[ok], v_sweep[ok], lw=2, label="touchdown speed v$_{td}$(t$_{ign}$), JAX rollout")
ax[0].axvspan(4.0, t_opt, color="0.9", label=f"no touchdown in {T_H:.0f} s (stops above pad, climbs)")
ax[0].plot([t_opt], [v_opt], "o", color="#d65a31", ms=8, zorder=5,
           label=f"hoverslam: t$_{{ign}}$ = {t_opt:.3f} s, v$_{{td}}$ = {v_opt:.2f} m/s")
tt = np.linspace(t_des - 0.25, t_des + 0.25, 2)
ax[0].plot(tt, v_des + grads["t_ign"]["jax_grad"] * (tt - t_des), "-", color="#4c956c", lw=2.5,
           label=f"jax.grad tangent at +100 ms: {grads['t_ign']['jax_grad']:.1f} (m/s)/s")
ax[0].plot([t_des], [v_des], "s", color="#4c956c", ms=7, zorder=5)
ax[0].plot([hi_n], [v_n], "D", color="#6b5b95", ms=7, zorder=5,
           label=f"Rumoca native LanderBurn (step throttle): hoverslam {hi_n:.4f} s ({shift_ms:+.2f} ms), {v_n:.2f} m/s")
ax[0].plot([zoh_variants[0]["t_ign_opt_s"]], [0.0], "x", color="0.3", ms=8, mew=1.5, zorder=5,
           label=f"left-endpoint-sampled throttle: {zoh_variants[0]['t_ign_opt_s']:.4f} s ({zoh_shift_left_dt02_ms:+.1f} ms = -dt/2 ZOH bias)")
ax[0].plot([6.0], [native_check[0]["native_v_end_mps"]], "D", color="#6b5b95", ms=7, zorder=5)
ax[0].plot([t_des], [native_check[2]["native_v_end_mps"]], "D", color="#6b5b95", ms=7, zorder=5)
ax[0].annotate(f"+100 ms: {v_100ms:.1f} m/s", (t_des, v_des), (t_des + 0.25, v_des + 8), fontsize=9,
               arrowprops=dict(arrowstyle="->", color="0.3"))
ax[0].annotate(f"+10 ms: {v_10ms:.1f} m/s", (t_opt + 0.01, v_10ms), (t_opt + 0.3, v_10ms + 4), fontsize=9,
               arrowprops=dict(arrowstyle="->", color="0.3"))
ax[0].set_xlabel("ignition time t$_{ign}$ [s]"); ax[0].set_ylabel("touchdown speed [m/s]")
ax[0].set_title("Hoverslam ignition sweep (sigmoid throttle, tau = 0.05 s, sampled at t$_k$ + dt/2)", fontsize=11)
ax[0].legend(fontsize=7.5, loc="lower left"); ax[0].set_xlim(4.0, 6.0)
dd = np.linspace(0, 0.55, 200)
ax[1].plot(deltas, v_delta, "o", ms=6, label="JAX rollout")
ax[1].plot(dd, -np.sqrt(K_fit * dd), "--", lw=1.5, label=f"square-root law  -sqrt(K delta), K = {K_fit:.0f} m$^2$/s$^3$")
ax[1].plot(dd, v_des + grads["t_ign"]["jax_grad"] * (dd - DELTA_DES), "-", color="#4c956c", lw=1.5,
           label="jax.grad tangent at delta = 0.1 s")
ax[1].set_ylim(min(v_delta) - 5, 2)
ax[1].set_xlabel("ignition delay past the hoverslam, delta = t$_{ign}$ - t$_{ign,opt}$ [s]")
ax[1].set_ylabel("touchdown speed [m/s]")
ax[1].set_title("Square-root cliff: infinite slope at the optimum", fontsize=11); ax[1].legend(fontsize=8)
fig.suptitle("d(v$_{td}$)/d(t$_{ign}$, cd, Fengine, EmptyRocket) by jax.grad through Rumoca's JAX export", fontsize=11)
fig.tight_layout()
fig_sens = RESULTS / "landing_sensitivity_jax.png"
fig.savefig(fig_sens, dpi=130)
plt.close(fig)
print("wrote", fig_sens)

report["wall_s"] = time.time() - T_WALL0
report["figures"] = [str(fig_id), str(fig_sens)]
json_path = RESULTS / "jax_results.json"
common.write_json(json_path, report)
print("wrote", json_path, f"({report['wall_s']:.1f} s)")

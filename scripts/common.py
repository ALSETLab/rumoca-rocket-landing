"""Shared helpers for the Rumoca rocket-landing challenge entry.

All vehicle physics comes from the Modelica source compiled by Rumoca; Python
only integrates, optimizes, plots and compares. The rk4/jax_rollout helpers are
adapted from scripts/common.py of CogniPilot/rumoca_workshop_amc2026 (Apache-2.0).
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import numpy as np
import rumoca as rum

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "RumocaRocketLanding.mo"
ORIGINAL_REPO = Path(r"C:\GIT\RocketLanding")
ORIGINAL_ROCKET = ORIGINAL_REPO / "SimpleLander" / "rocket.mo"
RESULTS = ROOT / "results" / "live"
OMC = Path(r"C:\Program Files\OpenModelica1.27.0-64bit\bin\omc.exe")

OM_LIBS = Path(os.environ["APPDATA"]) / ".openmodelica" / "libraries"
# A single root directory holding Modelica/, ModelicaServices/ and Complex/ side by side
# (Rumoca 0.10.2 resolves cross-package references only inside one root).
MSL323_ROOT = ROOT / "results" / "scratch" / "msl323"
MSL400_ROOT = ROOT / "results" / "scratch" / "msl400"

STATE_NAMES = ["y", "v", "fuel_mass"]


def load_lander():
    """Export-only model (pure explicit ODE with throttle input u)."""
    return rum.Session().load(str(MODEL), model="RumocaRocketLanding.Lander")


def load_burn():
    """Native-simulation model: coast then constant throttle from t_ign; terminates at touchdown."""
    return rum.Session().load(str(MODEL), model="RumocaRocketLanding.LanderBurn")


def load_original_rocket():
    """The untouched 2021 model from the ALSETLab RocketLanding repository."""
    return rum.Session().load(str(ORIGINAL_ROCKET), model="RocketLanding.SimpleLander.rocket")


def validate_layout(export):
    if export.state_names != STATE_NAMES or export.input_names != ["u"]:
        raise RuntimeError(f"Unexpected generated layout: {export.state_names} {export.input_names}")


def parameters(export, **overrides):
    values = np.array(export.default_parameters, dtype=float)
    for name, value in overrides.items():
        values[export.parameter_names.index(name)] = value
    return values


def rk4(rhs, t, x, u, p, dt):
    """Integration only: no handwritten vehicle dynamics in Python."""
    k1 = rhs(t, x, u, p)
    k2 = rhs(t + dt / 2, x + dt * k1 / 2, u, p)
    k3 = rhs(t + dt / 2, x + dt * k2 / 2, u, p)
    k4 = rhs(t + dt, x + dt * k3, u, p)
    return x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def jax_rollout(export, dt):
    """jit-compiled RK4 rollout of the Rumoca JAX export; controls is an (N, n_u) array."""
    import jax
    import jax.numpy as jnp

    def rollout(p, x0, controls):
        def step(x, tu):
            t, u = tu
            y = rk4(export.rhs, t, x, u, p, dt)
            return y, y
        _, ys = jax.lax.scan(step, x0, (jnp.arange(len(controls)) * dt, controls))
        return jnp.concatenate((x0[None, :], ys), axis=0)
    return jax.jit(rollout)


def omc_simulate(model, stop_time, intervals, prefix, load_files, workdir,
                 tolerance=1e-8, method="dassl", overrides=None):
    """Run OpenModelica (omc) on the same Modelica source as an independent reference.

    Returns (dict name -> np.array, omc stdout). `overrides` maps parameter name -> value
    and is passed through the simulation flag -override.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    lines = [f'loadFile("{Path(f).as_posix()}"); getErrorString();' for f in load_files]
    ov = ""
    if overrides:
        ov = ', simflags="-override=' + ",".join(f"{k}={v}" for k, v in overrides.items()) + '"'
    lines.append(
        f'simulate({model}, stopTime={stop_time}, numberOfIntervals={intervals}, tolerance={tolerance}, '
        f'method="{method}", outputFormat="csv", fileNamePrefix="{prefix}"{ov}); print(getErrorString());'
    )
    script = workdir / f"{prefix}.mos"
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out = subprocess.run([str(OMC), script.name], cwd=workdir, capture_output=True, text=True, timeout=900)
    csv_path = workdir / f"{prefix}_res.csv"
    if not csv_path.exists():
        raise RuntimeError(f"omc produced no result:\n{out.stdout}\n{out.stderr}")
    with open(csv_path, newline="") as fh:
        rows = list(csv.reader(fh))
    header = rows[0]
    data = np.array([[float(x) for x in r] for r in rows[1:] if r])
    return {name: data[:, i] for i, name in enumerate(header)}, out.stdout


def interp_to(t_ref, t, y):
    return np.interp(t_ref, np.asarray(t, dtype=float), np.asarray(y, dtype=float))


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=float) + "\n",
                          encoding="utf-8", newline="\n")


def plot_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 130,
                         "axes.prop_cycle": plt.cycler(color=["#007c91", "#d65a31", "#4c956c", "#6b5b95"])})

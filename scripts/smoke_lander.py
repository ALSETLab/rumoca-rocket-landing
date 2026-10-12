"""Smoke test: the package compiles, exports to CasADi and JAX, and the open-loop burn simulates natively."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import rumoca as rum

import common

print("rumoca", rum.version())
print("validate:", rum.validate(str(common.MODEL)))
m = common.load_lander()
print(m.summary())
ex = m.to_casadi()
common.validate_layout(ex)
print("casadi OK", ex.parameter_names, ex.default_parameters, ex.default_states)
exj = m.to_jax()
common.validate_layout(exj)
print("jax OK")
mb = common.load_burn()
print(mb.summary())
print(mb.structure())
for t_ign in [0.0, 2.0, 4.0, 6.0, 8.0]:
    r = mb.simulate(40.0, dt=0.02, params={"t_ign": t_ign})
    t = np.asarray(r.time); y = np.asarray(r["lander.y"]); v = np.asarray(r["lander.v"]); fm = np.asarray(r["lander.fuel_mass"])
    print(f"t_ign={t_ign}: termination={r.termination} t_end={t[-1]:.2f} y_end={y[-1]:.2f} v_end={v[-1]:.2f} "
          f"fuel_end={fm[-1]:.1f} v_min={v.min():.1f} v_max_after_ign={v[t >= t_ign].max():.2f}")
print("names:", r.names)
print("SMOKE PASS")

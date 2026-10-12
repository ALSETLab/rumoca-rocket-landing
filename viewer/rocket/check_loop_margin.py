"""Pick an autopilot ignition altitude that lands softly with both Rumoca solvers in loop mode."""
from pathlib import Path
import numpy as np
import rumoca as rum

here = Path(__file__).resolve().parent
src = (here / "LanderLive.mo").read_text(encoding="utf-8") + "\n" + (here / "LanderBatchTest.mo").read_text(encoding="utf-8")
for y_ign in [402.40, 402.35, 402.30]:
    for solver in ["rk-like", "auto"]:
        m = rum.Session().loads(src, model="LanderBatchTest")
        r = m.simulate(40.0, dt=0.01, params={"lander.y_ign": y_ign}, config=rum.SimConfig(solver=solver))
        landed = np.asarray(r["lander.landed"]); t = np.asarray(r.time)
        ups = np.where(np.diff((landed > 0.5).astype(int)) == 1)[0] + 1
        td = np.asarray(r["lander.touchdown_speed"]); fuel = np.asarray(r["lander.fuel_mass"])
        print(f"y_ign={y_ign} {solver:8s}: " + "; ".join(f"t={t[k]:.2f}s v={td[k]:+.2f} fuel={fuel[k]:.1f}" for k in ups))

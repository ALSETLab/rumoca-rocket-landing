"""Batch check of loop mode in viewer/rocket/LanderLive.mo (autopilot on, loop on), Rumoca native simulator."""
from pathlib import Path
import numpy as np
import rumoca as rum

here = Path(__file__).resolve().parent
src = (here / "LanderLive.mo").read_text(encoding="utf-8") + "\n" + (here / "LanderBatchTest.mo").read_text(encoding="utf-8")
for solver in ["rk-like", "auto"]:
    m = rum.Session().loads(src, model="LanderBatchTest")
    r = m.simulate(60.0, dt=0.01, config=rum.SimConfig(solver=solver))
    t = np.asarray(r.time); landed = np.asarray(r["lander.landed"]); fl = np.asarray(r["lander.flights"])
    td = np.asarray(r["lander.touchdown_speed"]); fuel = np.asarray(r["lander.fuel_mass"]); y = np.asarray(r["lander.y"])
    ups = np.where(np.diff((landed > 0.5).astype(int)) == 1)[0] + 1
    print(f"{solver}: {len(ups)} landings in 60 s, flights counter at end = {fl[-1]:.0f}, y range [{y.min():.3f}, {y.max():.1f}]")
    for k in ups:
        print(f"   landed t={t[k]:6.2f} s  touchdown_speed={td[k]:+.3f} m/s  fuel_left={fuel[k]:.2f} kg  flight={fl[k]:.0f}")

"""Batch check of viewer/rocket/LanderLive.mo with the autopilot on (Rumoca native simulator)."""
from pathlib import Path
import numpy as np
import rumoca as rum

here = Path(__file__).resolve().parent
src = (here / "LanderLive.mo").read_text(encoding="utf-8") + "\n" + (here / "LanderBatchTest.mo").read_text(encoding="utf-8")
for solver in ["rk-like", "auto"]:
    for y_ign in [402.51, 402.45, 402.40]:
        m = rum.Session().loads(src, model="LanderBatchTest")
        r = m.simulate(20.0, dt=0.01, params={"lander.y_ign": y_ign}, config=rum.SimConfig(solver=solver))
        t = np.asarray(r.time); y = np.asarray(r["lander.y"]); landed = np.asarray(r["lander.landed"])
        i = int(np.argmax(landed > 0.5)) if (landed > 0.5).any() else None
        ign = np.asarray(r["lander.ignited"]); j = int(np.argmax(ign > 0.5)) if (ign > 0.5).any() else None
        print(f"{solver:8s} y_ign={y_ign}: ignition t={t[j] if j is not None else None:.3f} s | "
              + (f"landed t={t[i]:.3f} s touchdown_speed={r['lander.touchdown_speed'][-1]:.3f} m/s fuel_left={r['lander.fuel_mass'][-1]:.2f} kg crashed={r['lander.crashed'][-1]:.0f} y_min={y.min():.4f}"
                 if i is not None else f"never landed, y_end={y[-1]:.1f} y_min={y.min():.3f}"))

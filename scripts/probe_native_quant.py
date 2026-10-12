"""Probe: resolution of the native simulator's time event (time >= t_ign) in RumocaRocketLanding.LanderBurn.
Touchdown speed vs t_ign in 2e-5 s steps around the hoverslam boundary, at dt = 0.01 and 0.001."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import common
mb = common.load_burn()
out = {}
for dt in (0.01, 0.001):
    rows = []
    for tg in np.arange(4.98780, 4.98830001, 0.00002):
        r = mb.simulate(60.0, dt=dt, params={"t_ign": float(tg)})
        t = np.asarray(r.time); v = np.asarray(r["lander.v"]); u = np.asarray(r["lander.u"])
        rows.append([round(float(tg), 6), float(v[-1]), float(t[-1])])
        print(f"dt={dt} t_ign={tg:.6f} v_td={v[-1]:+.6f} t_td={t[-1]:.4f}")
    out[str(dt)] = rows
# distinct v_td values -> plateau width
for dt, rows in out.items():
    vals = [r[1] for r in rows]
    print(f"dt={dt}: {len(set(round(v, 9) for v in vals))} distinct v_td among {len(vals)} t_ign values spaced 2e-5 s")
common.write_json(common.RESULTS / "native_time_event_resolution.json", out)

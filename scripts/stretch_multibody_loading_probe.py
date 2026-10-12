"""On-disk record of the two Rumoca 0.10.2 LOADING diagnostics quoted in the multibody report:
 (1) loading the library's package.mo (C:/GIT/RocketLanding/package.mo) with roots=[msl323] fails with
     'unresolved type reference: Modelica.Mechanics.MultiBody.World', whereas loading the single model file with the
     same roots works (that is what the support matrix does);
 (2) adding C:/GIT as a second root is rejected as a whole ('Failed to load source root ... PKG-006 ... missing package.mo'):
     one malformed sibling package poisons an entire root. This probe only READS C:/GIT (takes ~1 min).
Output: results/live/multibody_loading_probe.json
Re-run: cd C:/GIT/rumoca-rocket-landing && PYTHONIOENCODING=utf-8 JAX_PLATFORMS=cpu .venv/Scripts/python.exe scripts/stretch_multibody_loading_probe.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

MODEL = "RocketLanding.SimpleLander.FusalageInGravity"
PKG = Path(r"C:\GIT\RocketLanding\package.mo")
FILE = Path(r"C:\GIT\RocketLanding\SimpleLander\FusalageInGravity.mo")


def main() -> int:
    import rumoca as rum

    msl = str(common.MSL323_ROOT)
    out = {"rumoca": rum.version(), "msl_root": msl, "probes": []}

    t0 = time.perf_counter()
    try:
        m = rum.Session(roots=[msl]).load(str(PKG), model=MODEL)
        res = {"outcome": "loaded", "summary": str(m.summary())}
    except Exception as exc:  # noqa: BLE001
        res = {"outcome": "error", "error": f"{type(exc).__name__}: {exc}"[:800]}
    res.update({"probe": "load library package.mo with roots=[msl323]", "file": str(PKG), "wall_s": round(time.perf_counter() - t0, 2)})
    out["probes"].append(res)
    print(res["probe"], "->", res.get("error", res.get("summary"))[:300], flush=True)

    t0 = time.perf_counter()
    try:
        m = rum.Session(roots=[msl]).load(str(FILE), model=MODEL)
        res = {"outcome": "loaded", "summary": str(m.summary())}
    except Exception as exc:  # noqa: BLE001
        res = {"outcome": "error", "error": f"{type(exc).__name__}: {exc}"[:800]}
    res.update({"probe": "load single model file with roots=[msl323] (control, expected to work)", "file": str(FILE),
                "wall_s": round(time.perf_counter() - t0, 2)})
    out["probes"].append(res)
    print(res["probe"], "->", res.get("error", res.get("summary"))[:300], flush=True)

    t0 = time.perf_counter()
    try:
        m = rum.Session(roots=[msl, r"C:\GIT"]).load(str(FILE), model=MODEL)
        res = {"outcome": "loaded", "summary": str(m.summary())}
    except Exception as exc:  # noqa: BLE001
        res = {"outcome": "error", "error": f"{type(exc).__name__}: {exc}"[:800]}
    res.update({"probe": "roots=[msl323, C:/GIT] (parent folder of the library as a second root)", "file": str(FILE),
                "wall_s": round(time.perf_counter() - t0, 2)})
    out["probes"].append(res)
    print(res["probe"], "->", res.get("error", res.get("summary"))[:300], flush=True)

    common.write_json(common.RESULTS / "multibody_loading_probe.json", out)
    print("wrote", common.RESULTS / "multibody_loading_probe.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

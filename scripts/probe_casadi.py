"""Probe: how does the Rumoca CasADi export behave with MX symbols? (scratch helper for optimize_landing_casadi.py)"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import casadi as ca
import common

t0 = time.time()
m = common.load_lander()
ex = m.to_casadi()
common.validate_layout(ex)
print("load+export", time.time() - t0)
print("params", ex.parameter_names, ex.default_parameters)
print("x0", ex.default_states)
x = ca.MX.sym("x", 3); u = ca.MX.sym("u", 1); p = ca.MX(np.array(ex.default_parameters, dtype=float)); t = ca.MX.sym("t")
try:
    xd = ex.rhs(t, x, u, p)
    print("MX ok", type(xd), xd.shape)
except Exception as e:
    print("MX FAIL", repr(e))
try:
    xd2 = ex.rhs(0.0, ca.DM([800, -70, 100]), ca.DM([1.0]), ca.DM(np.array(ex.default_parameters)))
    print("DM ok", xd2)
except Exception as e:
    print("DM FAIL", repr(e))
try:
    xn = common.rk4(ex.rhs, t, x, u, p, 0.1)
    print("rk4 MX ok", xn.shape)
except Exception as e:
    print("rk4 FAIL", repr(e))
print(dir(ex))

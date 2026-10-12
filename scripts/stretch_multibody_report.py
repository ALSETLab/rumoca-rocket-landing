"""Merge the multibody stream's outputs into one Markdown deliverable results/live/multibody_support_matrix.md
(baseline table + patched-MSL table + cos triage table + free-fall check + OpenModelica cross-check + loading probe).
Re-runnable; reads only results/live/*.json/.md.
Run: cd C:/GIT/rumoca-rocket-landing && PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe scripts/stretch_multibody_report.py
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

live = common.RESULTS
base_md = (live / "multibody_support_matrix.md").read_text(encoding="utf-8")
if "## With the Modelica.Math work-around" in base_md:
    base_md = base_md.split("## With the Modelica.Math work-around")[0].rstrip() + "\n"
parts = [base_md.rstrip(), ""]
p = live / "multibody_support_matrix_patchedA.md"
if p.exists():
    patched = p.read_text(encoding="utf-8").split("\n", 2)[2]  # drop its own title line
    parts += ["## With the Modelica.Math work-around (scratch MSL copies, attempt A: `external \"builtin\"` -> `algorithm y := .f(u)`)", "", patched.rstrip(), ""]
p = live / "multibody_cos_triage.md"
if p.exists():
    parts += ["## Triage of the `Modelica.Math.cos` blocker", "", p.read_text(encoding="utf-8").split("\n", 2)[2].rstrip(), ""]
p = live / "multibody_freefall_check.json"
if p.exists():
    c = json.loads(p.read_text(encoding="utf-8"))
    if "error" in c:
        parts += ["## Free-fall check", "", f"FAILED: {c['error']}", ""]
    else:
        pr, sec = c["primary"], c["secondary_default_solver"]
        parts += ["## Free-fall check of the unlocked FusalageInGravity simulation", "",
                  f"Checked result (solver `{pr['solver']}`, dt {pr['dt']}, {c['t_end']} s, patched MSL 3.2.3 root): `{pr['z_name']}` drops "
                  f"{pr['drop_rumoca_m']:.7f} m vs g_n t^2/2 = {pr['drop_freefall_m']:.7f} m, max |dz| = {pr['max_abs_err_z_m']:.2e} m; "
                  f"`{pr['v_name']}` ends at {pr['vz_end_rumoca_m_s']:.5f} m/s vs -g_n t = {pr['vz_end_freefall_m_s']:.5f} m/s, "
                  f"max |dv| = {pr['max_abs_err_vz_m_s']:.2e} m/s (g_n = {c['g_ref_m_s2']}).", "",
                  f"Default solver (`{sec['solver']}`, = `bdf`, dt {sec['dt']}): max |dz| = {sec['max_abs_err_z_m']:.2e} m, formed within the first "
                  f"{sec.get('t_first_z_err_gt_1mm_s')} s and then constant; identical for dt = {c['dts']} "
                  f"(spread {c.get('bdf_auto_offset_spread_m', float('nan')):.1e} m, dt-independent = {c.get('bdf_auto_offset_dt_independent')}), "
                  f"whereas `rk-like` gives at worst {c.get('rk_like_max_abs_err_z_m_worst', float('nan')):.1e} m over the same dts. "
                  "So dt is only the output grid for the BDF path and the offset is a BDF start-up error, not a model error.", "",
                  f"Initial conditions: {c['initial_conditions_note']}", "",
                  "| solver | dt | points | drop [m] | max abs dz [m] | max abs dv [m/s] | simulate s |", "|---|---|---|---|---|---|---|"]
        for r in c["runs"]:
            if "error" in r:
                parts.append(f"| {r['solver']} | {r['dt']} | - | - | ERROR: {r['error'][:120]} | - | {r['simulate_s']} |")
            else:
                parts.append(f"| {r['solver']} | {r['dt']} | {r['n_points']} | {r['drop_rumoca_m']:.7f} | {r['max_abs_err_z_m']:.2e} | "
                             f"{r['max_abs_err_vz_m_s']:.2e} | {r['simulate_s']} |")
        parts += ["", f"Figure: {c['figure']}", ""]
p = live / "multibody_om_motortest_check.json"
if p.exists():
    c = json.loads(p.read_text(encoding="utf-8"))
    parts += ["## OpenModelica cross-check of DroneLibrary.Tests.motorTest (3 s, both steps)", ""]
    if "comparison" in c:
        om = c.get("openmodelica", {})
        parts += [f"Rumoca (patched MSL 4.0.0 copy; compile {c['rumoca'].get('compile_s')} s, simulate {c['rumoca'].get('simulate_s')} s) vs "
                  f"omc dassl tol 1e-6 (MSL resolved by omc: {om.get('om_msl_version')}; {om.get('wall_s')} s wall), compared on "
                  f"{c['comparison_grid']['n']} mid-interval points avoiding the events at 1 s and 2 s. Worst relative difference over the "
                  f"non-zero signals: {c['worst_rel_diff_nonzero_signals']:.2e} (pass < 1e-3: {c['pass_rel_1e-3']}).", "",
                  "| signal | max abs diff | max abs (OM) | value at 1.5 s Rumoca / OM | end value Rumoca / OM |", "|---|---|---|---|---|"]
        for n, v in c["comparison"].items():
            parts.append(f"| `{n}` | {v['max_abs_diff']:.2e} | {v['max_abs_om']:.4g} | {v['rumoca_at_1p5s']:.6g} / {v['om_at_1p5s']:.6g} | "
                         f"{v['rumoca_end']:.6g} / {v['om_end']:.6g} |")
        parts += ["", f"Figure: {c['figure']}", ""]
    else:
        parts += [f"Not compared: rumoca={c.get('rumoca')} openmodelica={ {k: v for k, v in c.get('openmodelica', {}).items() if k != 'omc_stdout_tail'} }", ""]
p = live / "multibody_loading_probe.json"
if p.exists():
    c = json.loads(p.read_text(encoding="utf-8"))
    parts += ["## Loading probes (exact diagnostics)", ""]
    for pr in c["probes"]:
        parts.append(f"- {pr['probe']} ({pr['wall_s']} s): {pr['outcome']} - `{(pr.get('error') or pr.get('summary', ''))[:400]}`".replace("\n", " "))
    parts.append("")
(live / "multibody_support_matrix.md").write_text("\n".join(parts) + "\n", encoding="utf-8", newline="\n")
print("wrote", live / "multibody_support_matrix.md")

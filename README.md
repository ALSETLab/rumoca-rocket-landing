# Beyond GN&C-Enabled Retrograde Landing: Fuel-Optimal Landing of a First Stage Rocket with Rumoca

Entry for the open challenge of the **Rumoca workshop at the American Modelica Conference 2026**
(CogniPilot, Georgia Tech, 2026-10-12, release `workshop-amc2026-r9`). The entry text and the 45-second
pitch are in `ENTRY.md`.

**Question.** Can Rumoca take a Modelica rocket model that our lab published in 2021, without re-typing any
physics in Python, and (1) reproduce the original simulation against OpenModelica, (2) plan a
minimum-propellant hoverslam landing with CasADi, (3) differentiate the touchdown through JAX, and (4) fly it
in Rumoca's own 3-D viewer? And how far does it get with the real 3-D MultiBody models of the library?

## Background: the 2021 paper and the RocketLanding library

C. Canham, M. Podlaski and L. Vanfretti, "Guidance, Navigation, and Control enabling Retrograde Landing of a
First Stage Rocket," Proceedings of the 14th International Modelica Conference, Linköping, Sweden, 2021,
[doi:10.3384/ecp21181171](https://doi.org/10.3384/ecp21181171). Models:
[ALSETLab/RocketLanding](https://github.com/ALSETLab/RocketLanding).

The paper models the first stage of a SpaceX Falcon 9 right after second-stage separation: a 48 m by 3.7 m
MultiBody cylinder starting at 166 km altitude and 293 m/s horizontal speed, one 914 kN engine,
three grid fins, altitude-dependent drag and a kerosene/oxygen fuel injector. A flight controller of four
parts brings it back to a landing pad: a grid-fin controller, a boost-back burn, a 30 s re-entry burn and a
PD landing controller tuned for touchdown below 2 m/s. The simulated trajectory overlaps SpaceX webcast
telemetry, and the paper shows the landing at Cape Canaveral in 3-D with the DLR Visualization library,
which needs a commercial license. The paper closes by noting the opportunity to optimize these flight paths
for fuel consumption.

The library has three packages: `SimpleLander` (a 1-D vertical lander and early MultiBody fuselages),
`LanderWithHorizontalVelocity` (the full controlled 3-D stage) and `LanderWithVisualization` (the same with
DLR Visualization).

## What Rumoca adds

| Topic | 2021 paper | This entry, with Rumoca 0.10.2 |
|---|---|---|
| Compiler | Dymola | Rumoca compiles the untouched `SimpleLander.rocket` in 0.37 s and matches OpenModelica to 0.35 mm |
| Landing burn | Hand-tuned PD landing controller | Fuel-optimal burn from CasADi on Rumoca's export, checked four ways and in OpenModelica |
| Gradients | None | JAX gradients of the touchdown speed; drag coefficient identified from OpenModelica data |
| 3-D view | DLR Visualization library (commercial license) | Rumoca's free, open-source Three.js viewer, live, with autopilot, manual throttle, Land-it-again and loop |
| Touchdown | Below 2 m/s by tuning | 1.8 m/s from the optimal switch altitude, every flight |
| Full 3-D stage | Runs in Dymola | Does not run in Rumoca 0.10.2 yet; support status and blockers documented below |

**Starting model for this entry.** `RocketLanding.SimpleLander.rocket`: point mass, constant thrust while
the tank has propellant, quadratic drag, 1-D vertical motion. The repository is used untouched.

**Change.** `models/RumocaRocketLanding.mo` re-casts the same equations and parameters for Rumoca's
JAX/CasADi targets: a throttle input `u` replaces the fixed on/off engine, initial conditions move from an
`initial equation` section to `start/fixed` attributes, the forces are inlined (the 0.10.2 `jax-ode` and
`casadi-ode` targets reject algebraic variables and initialization sections), and the drag law
`Fdrag = if Ascending then -cd*qbar*S else cd*qbar*S` is written smooth as `-0.5*rho*S*cd*v*sqrt(v^2+eps^2)`
so it can be differentiated. A second model, `LanderBurn`, flies the lander open loop (coast, then full
throttle from `t_ign` until the tank is empty, terminate at touchdown) for Rumoca's native simulator and for
OpenModelica. `viewer/rocket/LanderLive.mo` adds keyboard throttle and an autopilot for Rumoca's 3-D viewer.

## Results

All numbers below are read from the JSON files in `results/reference/` (copies of `results/live/` at the time
of writing). Every trajectory and derivative comes from Rumoca (`SolveExport.rhs` or the native simulator) or
from OpenModelica; Python only integrates, optimizes, interpolates and plots.

### 1. The untouched 2021 model in Rumoca, against OpenModelica

`scripts/baseline_original_rocket.py`, figure `results/reference/baseline_original_rocket.png`.

| Quantity | Rumoca 0.10.2 (rk-like) | OpenModelica 1.27 (dassl, tol 1e-8) |
|---|---|---|
| Compile (parse to simulate-ready) | 0.37 s | 1.34 s (C compilation) |
| 60 s simulation, output every 0.05 s | 5 ms | 0.12 s |
| Burnout (fuel_mass reaches 0) | 10.000 s | 10.000 s |
| Apogee | 383.802 m at 15.0329 s | 383.802 m at 15.0329 s |
| Max abs difference in altitude | 3.5e-4 m (1.3e-7 of the 2803 m range) | reference |
| Max abs difference in velocity | 5.0e-5 m/s (3.8e-7 of range) | reference |

The remaining 0.35 mm is integration tolerance, not translation: tightening Rumoca's tolerances brings it
within 5e-8 m of a scipy reference at 1e-12, and a Dymola run at 1e-10 agrees too. Rumoca's `auto` and `bdf`
solvers give bitwise-identical results on this model.

### 2. Minimum-propellant hoverslam with CasADi, from Rumoca's export

`scripts/optimize_landing_casadi.py`, figures `results/reference/landing_hoverslam.png` and
`landing_checks.png`, data `landing_hoverslam.json`.

Direct multiple shooting (RK4, 160 intervals, free final time) on `Lander.to_casadi()`, IPOPT: status
`Solve_Succeeded` in 16 iterations, shooting defect 3e-12.

| Quantity | Value |
|---|---|
| Start | 800 m altitude, 70 m/s downward, 100 kg propellant |
| Optimal schedule | coast, then full throttle from 4.988 s (99.4 % of nodes within 0.02 of 0 or 1) |
| Touchdown | 13.922 s, y = 0, v = 0 |
| Propellant used / left | 89.35 kg / 10.66 kg |
| Ignite 1 s early (native simulator) | arrest 98 m above the pad, climb to 119 m, dry tank, impact 43 m/s |
| Ignite 1 s late (native simulator) | impact 44 m/s with 52 kg unused |

Checks on the optimum, all from the same Modelica source:

| Route | Ignition time | Result |
|---|---|---|
| CasADi NLP (fuel-equivalent switch) | 4.987775 s | y(T) = 0, v(T) = 0 by construction |
| CVODES re-integration of the NLP profile (abstol 1e-13) | same profile | y(T) = 9e-8 m, v(T) = 8e-8 m/s |
| Exact bang-bang solved with CVODES + fsolve | 4.987748 s | residuals 4e-14 m, 5e-14 m/s |
| JAX rollout (differentiable, midpoint-sampled throttle) | 4.987888 s | touchdown 0.03 m/s |
| Rumoca native `LanderBurn`, bisection on `t_ign` | 4.987820 s | touchdown 0.79 m/s at first crossing |
| OpenModelica on the same `.mo` at 4.98782 s | same | touchdown 0.43 m/s, 10.99 kg left |

The native and OpenModelica touchdown speeds are not zero because the hoverslam is a knife edge: the
touchdown speed follows a square-root law in the ignition delay, so a 0.1 ms placement difference already
costs a few tenths of a metre per second.

### 3. Differentiating through the model with JAX

`scripts/identify_and_sensitivity_jax.py`, figures `results/reference/identify_drag_jax.png` and
`landing_sensitivity_jax.png`, data `jax_results.json`.

- Drag identification: fitting `cd` by gradient descent through `Lander.to_jax()` recovers 0.200102 from a
  noisy Rumoca flight (sigma 2 m), 0.199999 from a noiseless one, and 0.2000012 from the independent
  OpenModelica trajectory of the original model (residual 1.5 mm RMS). True value 0.2.
- Sensitivities of the touchdown speed by `jax.grad`, checked against central finite differences to 1e-5
  relative: igniting 100 ms after the optimum costs 15.0 m/s of touchdown speed; the touchdown speed follows
  v = -sqrt(K * delay) with K = 2069 m^2/s^3 to within 8 % over 10 ms to 0.5 s.

### 4. Flying it in Rumoca's 3-D viewer

`viewer/rocket/` runs `LanderLive.mo` in the Rumoca CLI's interactive Three.js viewer
(PowerShell, from the repository root: `.\viewer\rocket\run_viewer.ps1`, which opens http://localhost:8080). On-screen buttons (or Space, L, R) toggle
the autopilot and the loop and land it again; W/S fly the throttle by hand. The autopilot is the CasADi hoverslam turned into a trigger: ignite when the
altitude drops through 402.40 m, 11 cm below where the optimal switch time falls on the coast. Rumoca's
native simulator lands it at 1.8 m/s with 12.1 kg left (both `rk-like` and `auto`), and loop mode re-flies it
4 s after each landing; closer to the optimum the default solver misses the grazing touchdown event (see findings). Screenshots:
`results/reference/viewer_burn.jpg` and `viewer_landed.jpg`. Details in `viewer/rocket/README.md`.

### 5. How far Rumoca gets with the 3-D MultiBody models (stretch)

`scripts/stretch_multibody.py`, tables `results/reference/multibody_support_matrix.md`.

With the Modelica Standard Library supplied as a single root directory (`Modelica/`, `ModelicaServices/`,
`Complex/` side by side), Rumoca flattens every MultiBody model tried: the rocket fuselage (889 equations),
`DroneLibrary.Tests.simpleHoverTest` (7613 equations, 90 states) and the complete drone example (8005
equations, 1327 parameters), in 5 to 45 s each. Simulation stops at `Modelica.Math.cos`, an
`external "builtin"` function the runtime cannot execute. Rewriting those functions in a scratch copy of the
MSL as `algorithm y := .cos(u);` unlocks simulation: the fuselage free-falls at g (velocity error 4e-15 m/s
over 2 s), and the drone library's `motorTest` and `simpleHoverTest` simulate. Remaining blockers are listed
in the support matrix (an array-shape proof in `Modelica.Fluid` for the full rocket, an algebraic projection
at an event for `propellerTest`, a Solve-IR tensor fill for the full drone example).

## Rumoca 0.10.2 findings collected on the way

- `Session().load(path, model=...)` needs the fully qualified class name for a file with a `within` clause;
  a fresh `Session` per model avoids "Duplicate class ... non-identical definition".
- `to_jax()` / `to_casadi()` return an explicit `rhs(t, x, u, p)`; the targets reject algebraic variables and
  `initial equation` sections, so export models are written as pure explicit ODEs with `start/fixed`.
- Native `simulate()` handles events, `if`, `when` and `terminate`, but `inputs=` is not available yet
  ("time-varying inputs are a fast-follower"), and a model with an unbound `input` cannot be simulated
  natively. Open-loop schedules therefore live in Modelica (`LanderBurn`). The CLI scenario route does drive
  `input` variables (keyboard, gamepad).
- A grazing state event is missed: with `LanderBurn` at the hoverslam ignition time, `lander.y` dips to
  -1.8 cm for 0.11 s (11 output samples at dt 0.01) without `when lander.y <= 0 then terminate` firing;
  OpenModelica terminates at the first crossing. Figure `results/reference/landing_grazing_event.png`.
- Time events (`time >= t_ign`) are applied on an internal grid of about 2e-4 s: ignition times spaced 2e-5 s
  apart produce only five distinct results (`native_time_event_resolution.json`).
- The Windows release binary does not ship the viewer's browser bundle ("Missing Rumoca viewer shell"); it
  reads `packages/rumoca-web/` from the current folder. The viewer's reset restores the states but not the
  model clock, so time-triggered logic misfires after R.
- Library roots: cross-root references do not resolve, so the MSL must be one root directory containing
  `Modelica/`, `ModelicaServices/` and `Complex/`. OpenModelica's `ModelicaServices` needs two
  OpenModelica-specific calls replaced. `Modelica.Math`'s `external "builtin"` functions block simulation
  (work-around above).

## How to run

Clone with the submodules: `external/RocketLanding` (the original ALSETLab models) and
`external/rumoca-web-assets` ([ALSETLab/rumoca-web-assets](https://github.com/ALSETLab/rumoca-web-assets), the
viewer kit the Rumoca 0.10.2 Windows release binary needs). Then run the setup script once (PowerShell, Python
3.12 on PATH as `py -3.12` or `python`):

```powershell
git clone --recurse-submodules https://github.com/ALSETLab/rumoca-rocket-landing.git
cd rumoca-rocket-landing
.\setup.ps1
```

`setup.ps1` initializes the submodules (if the clone was made without `--recurse-submodules`), creates `.venv`
from `requirements.txt` and downloads the Rumoca CLI into the viewer kit, checking its SHA-256. Then, from the
repository root:

```powershell
$env:PYTHONIOENCODING = "utf-8"; $env:JAX_PLATFORMS = "cpu"
.venv\Scripts\python.exe scripts\smoke_lander.py
.venv\Scripts\python.exe scripts\baseline_original_rocket.py
.venv\Scripts\python.exe scripts\optimize_landing_casadi.py
.venv\Scripts\python.exe scripts\identify_and_sensitivity_jax.py
.\viewer\rocket\run_viewer.ps1
```

The baseline and the OpenModelica cross-checks need `omc.exe` from OpenModelica 1.27 (found through
`OPENMODELICAHOME`, otherwise the default install path in `scripts/common.py`). The MultiBody stretch
(`scripts/stretch_multibody*.py`) also needs the Modelica Standard Library 3.2.3 and 4.0.0 as single root
directories (`results/scratch/msl323`, `results/scratch/msl400`, built from OpenModelica's library folder as
described in the stretch script) and, for the DroneLibrary models, a sibling checkout of
[ALSETLab/rumoca-drone-library](https://github.com/ALSETLab/rumoca-drone-library) with its submodules (or
`DRONELIBRARY_DIR`).
## Credits

License: BSD 3-Clause (see `LICENSE`; third-party portions in `NOTICE`). Starting model and parameters: ALSETLab RocketLanding (see that repository's license). Integration helpers
adapted from `scripts/common.py` of CogniPilot/rumoca_workshop_amc2026 (Apache-2.0). Tooling: Rumoca 0.10.2,
CasADi 3.7.2 with IPOPT, JAX 0.4.38, OpenModelica 1.27.0. Agent: Claude Code in the Claude desktop app, with Claude Fable 5.1
and, after 11:30 EDT, Claude Opus 5.5; wall time is recorded in `ENTRY.md`.

Lessons learned across both Rumoca projects: `LESSONS_LEARNED.md`.

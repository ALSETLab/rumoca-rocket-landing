# Rumoca 0.10.2 support matrix: ALSETLab MultiBody models

Rumoca 0.10.2 native pipeline; one fresh `rum.Session(roots=[msl_root])` per model, simulate(1.0 s, dt=0.1). Equations = scalarized count from `structure()` (StructuralInfo.equations); states/algebraics/parameters from the compiled model.

| model | MSL | equations | states | algebraics | parameters | compile s | compiled? | simulated? | blocker (first 400 chars) |
|---|---|---|---|---|---|---|---|---|---|
| RocketLanding.SimpleLander.FusalageInGravity | 3.2.3 | 889 | 6 | 220 | 126 | 31.24 | yes | no | simulate: SimulationError: DAE structural proof failed: invalid structural IR contract without source span: independent state selection: external builtin function `Modelica.Math.cos` calls `cos`, which this runtime cannot execute |
| RocketLanding.SimpleLander.FusalageWithForce | 3.2.3 | 1538 | 9 | 394 | 158 | 28.52 | yes | no | simulate: SimulationError: DAE structural proof failed: invalid structural IR contract without source span: independent state selection: external builtin function `Modelica.Math.cos` calls `cos`, which this runtime cannot execute |
| RocketLanding.LanderWithHorizontalVelocity.RetrogradeRocketLanding | 3.2.3 |  |  |  |  | 33.69 | no | no | compile: CompileError: todae failed: RocketLanding.LanderWithHorizontalVelocity.RetrogradeRocketLanding failed in ToDae: unsupported Flat semantic owner `function shape proof`: dependent extent is not an exact Integer expression over proven shape axes |
| DroneLibrary.Tests.motorTest | 4.0.0 | 3617 | 31 | 917 | 452 | 11.16 | yes | no | simulate: SimulationError: DAE structural proof failed: checked DAE scalar projection failed: external builtin function `Modelica.Math.cos` calls `cos`, which this runtime cannot execute |
| DroneLibrary.Tests.propellerTest | 4.0.0 | 4256 | 56 | 1110 | 751 | 38.24 | yes | no | simulate: SimulationError: DAE structural proof failed: checked DAE scalar projection failed: external builtin function `Modelica.Math.cos` calls `cos`, which this runtime cannot execute |
| DroneLibrary.Tests.simpleHoverTest | 4.0.0 | 7613 | 90 | 1895 | 1196 | 29.96 | yes | no | simulate: SimulationError: DAE structural proof failed: checked DAE scalar projection failed: external builtin function `Modelica.Math.cos` calls `cos`, which this runtime cannot execute |
| DroneLibrary.Examples.DroneWithIdealPower.TestSystem | 4.0.0 | 8005 | 88 | 2004 | 1327 | 42.23 | yes | no | simulate: SimulationError: DAE structural proof failed: checked DAE scalar projection failed: external builtin function `Modelica.Math.cos` calls `cos`, which this runtime cannot execute |

## With the Modelica.Math work-around (scratch MSL copies, attempt A: `external "builtin"` -> `algorithm y := .f(u)`)

Rumoca 0.10.2 native pipeline; one fresh `rum.Session(roots=[msl_root])` per model, simulate(1.0 s, dt=0.1). Equations = scalarized count from `structure()` (StructuralInfo.equations); states/algebraics/parameters from the compiled model.

| model | MSL | equations | states | algebraics | parameters | compile s | compiled? | simulated? | blocker (first 400 chars) |
|---|---|---|---|---|---|---|---|---|---|
| RocketLanding.SimpleLander.FusalageInGravity | 3.2.3 | 854 | 6 | 220 | 126 | 4.83 | yes | yes | - |
| RocketLanding.SimpleLander.FusalageWithForce | 3.2.3 | 1538 | 9 | 394 | 158 | 6.01 | yes | no | simulate: SimulationError: DAE structural proof failed: invalid structural IR contract without source span: independent state selection: input `throttle` has neither a checked default nor a runtime value |
| RocketLanding.LanderWithHorizontalVelocity.RetrogradeRocketLanding | 3.2.3 |  |  |  |  | 15.13 | no | no | compile: CompileError: todae failed: RocketLanding.LanderWithHorizontalVelocity.RetrogradeRocketLanding failed in ToDae: unsupported Flat semantic owner `function shape proof`: dependent extent is not an exact Integer expression over proven shape axes |
| DroneLibrary.Tests.motorTest | 4.0.0 | 3466 | 31 | 917 | 452 | 57.45 | yes | yes | - |
| DroneLibrary.Tests.propellerTest | 4.0.0 | 4256 | 56 | 1110 | 751 | 13.29 | yes | no | simulate: SimulationError: solve-IR evaluation failed: algebraic projection did not converge at event boundary: worst scaled residual row=2227 target=force.frame_b.R.T[2,3] value=1.431604e0 ratio=8.343642e5 norm=1.721900e0 row_scale=1.715802e0 scaled_tolerance=1.715802e-6 |
| DroneLibrary.Tests.simpleHoverTest | 4.0.0 | 7613 | 90 | 1895 | 1196 | 13.51 | yes | yes | - |
| DroneLibrary.Examples.DroneWithIdealPower.TestSystem | 4.0.0 | 8005 | 88 | 2004 | 1327 | 13.58 | yes | no | simulate: SimulationError: invalid Solve IR contract: GuardedAssignmentProgram node 0 scalar program 0 has invalid register flow: tensor projection op 4 is invalid: tensor fill has an invalid extent or lane count |

## Triage of the `Modelica.Math.cos` blocker

| attempt | what was changed (scratch copy only) | compiled? | compile s | simulated? | diagnostic (first 400 chars) |
|---|---|---|---|---|---|
| 0-baseline | unpatched results/scratch/msl323 | yes | 5.45 | no | SimulationError: DAE structural proof failed: invalid structural IR contract without source span: independent state selection: external builtin function `Modelica.Math.cos` calls `cos`, which this runtime cannot execute |
| A | Modelica.Math.f: external "builtin" body -> algorithm y := .f(u) | yes | 40.0 | yes | - |
| B | call sites under Modelica/Mechanics/MultiBody: Modelica.Math.f( -> f( | yes | 10.76 | no | SimulationError: invalid Solve IR contract: pure-call argument or slot interface is invalid |
| C | A + B together | yes | 8.75 | yes | - |

## Free-fall check of the unlocked FusalageInGravity simulation

Rumoca native simulate(2.0 s, dt=0.01) on the patched MSL 3.2.3 root: `FirstStage.r_0[3]` drops 19.6228 m vs g t^2/2 = 19.6133 m (max |dz| = 9.68e-03 m over 2 s); `FirstStage.v_0[3]` ends at -19.6133 m/s vs -g t = -19.61 m/s (max |dv| = 3.55e-15 m/s). Figure: C:\GIT\rumoca-rocket-landing\results\live\multibody_freefall_check.png


# Rumoca 0.10.2 support matrix: ALSETLab MultiBody models

Rumoca 0.10.2 native pipeline; one fresh `rum.Session(roots=[msl_root])` per model, simulate(t_end, dt=0.1) with t_end = 1.0 s except motorTest (3.0 s, its stimuli are Step(startTime=1) and Step(startTime=2)).

Column semantics: `equations` is the SCALARIZED count from `structure()` (StructuralInfo.equations, before matching). `state vars` is the number of array-valued NAMES in `m.states.names` (alias pairs such as `FirstStage.r_0` and `FirstStage.body.frame_a.r_0` are both listed); `scalar states` expands those names by their dims from `to_dict('flat')['variables']` and therefore also counts the aliases (a free rigid body has 13 independent scalar states; FusalageInGravity lists 19 = 3+3+3+3+3+4 with two alias pairs). `algebraic vars` counts names, not scalars. `varying` = number of result signals that are not constant over the simulated window.

| model | MSL | t_end s | equations (scalar) | state vars (names) | scalar states (dims) | algebraic vars (names) | parameters | compile s | simulate s | compiled? | simulated? | varying/total signals | blocker (first 400 chars) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RocketLanding.SimpleLander.FusalageInGravity | 3.2.3 | 1.0 | 889 | 6 | 19 | 220 | 126 | 26.05 |  | yes | no |  | simulate: SimulationError: DAE structural proof failed: invalid structural IR contract without source span: independent state selection: external builtin function `Modelica.Math.cos` calls `cos`, which this runtime cannot execute |
| RocketLanding.SimpleLander.FusalageWithForce | 3.2.3 | 1.0 | 1538 | 9 | 22 | 394 | 158 | 27.13 |  | yes | no |  | simulate: SimulationError: DAE structural proof failed: invalid structural IR contract without source span: independent state selection: external builtin function `Modelica.Math.cos` calls `cos`, which this runtime cannot execute |
| RocketLanding.LanderWithHorizontalVelocity.RetrogradeRocketLanding | 3.2.3 | 1.0 |  |  |  |  |  | 47.39 |  | no | no |  | compile: CompileError: todae failed: RocketLanding.LanderWithHorizontalVelocity.RetrogradeRocketLanding failed in ToDae: unsupported Flat semantic owner `function shape proof`: dependent extent is not an exact Integer expression over proven shape axes |
| DroneLibrary.Tests.motorTest | 4.0.0 | 3.0 | 3617 | 31 | 77 | 917 | 452 | 11.72 |  | yes | no |  | simulate: SimulationError: DAE structural proof failed: checked DAE scalar projection failed: external builtin function `Modelica.Math.cos` calls `cos`, which this runtime cannot execute |
| DroneLibrary.Tests.propellerTest | 4.0.0 | 1.0 | 4256 | 56 | 153 | 1110 | 751 | 11.5 |  | yes | no |  | simulate: SimulationError: DAE structural proof failed: checked DAE scalar projection failed: external builtin function `Modelica.Math.cos` calls `cos`, which this runtime cannot execute |
| DroneLibrary.Tests.simpleHoverTest | 4.0.0 | 1.0 | 7613 | 90 | 239 | 1895 | 1196 | 31.66 |  | yes | no |  | simulate: SimulationError: DAE structural proof failed: checked DAE scalar projection failed: external builtin function `Modelica.Math.cos` calls `cos`, which this runtime cannot execute |
| DroneLibrary.Examples.DroneWithIdealPower.TestSystem | 4.0.0 | 1.0 | 8005 | 88 | 237 | 2004 | 1327 | 13.03 |  | yes | no |  | simulate: SimulationError: DAE structural proof failed: checked DAE scalar projection failed: external builtin function `Modelica.Math.cos` calls `cos`, which this runtime cannot execute |

## With the Modelica.Math work-around (scratch MSL copies, attempt A: `external "builtin"` -> `algorithm y := .f(u)`)

Rumoca 0.10.2 native pipeline; one fresh `rum.Session(roots=[msl_root])` per model, simulate(t_end, dt=0.1) with t_end = 1.0 s except motorTest (3.0 s, its stimuli are Step(startTime=1) and Step(startTime=2)).

Column semantics: `equations` is the SCALARIZED count from `structure()` (StructuralInfo.equations, before matching). `state vars` is the number of array-valued NAMES in `m.states.names` (alias pairs such as `FirstStage.r_0` and `FirstStage.body.frame_a.r_0` are both listed); `scalar states` expands those names by their dims from `to_dict('flat')['variables']` and therefore also counts the aliases (a free rigid body has 13 independent scalar states; FusalageInGravity lists 19 = 3+3+3+3+3+4 with two alias pairs). `algebraic vars` counts names, not scalars. `varying` = number of result signals that are not constant over the simulated window.

| model | MSL | t_end s | equations (scalar) | state vars (names) | scalar states (dims) | algebraic vars (names) | parameters | compile s | simulate s | compiled? | simulated? | varying/total signals | blocker (first 400 chars) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| RocketLanding.SimpleLander.FusalageInGravity | 3.2.3 | 1.0 | 854 | 6 | 19 | 220 | 126 | 6.17 | 0.72 | yes | yes | 10/893 | - |
| RocketLanding.SimpleLander.FusalageWithForce | 3.2.3 | 1.0 | 1538 | 9 | 22 | 394 | 158 | 6.72 |  | yes | no |  | simulate: SimulationError: DAE structural proof failed: invalid structural IR contract without source span: independent state selection: input `throttle` has neither a checked default nor a runtime value |
| RocketLanding.LanderWithHorizontalVelocity.RetrogradeRocketLanding | 3.2.3 | 1.0 |  |  |  |  |  | 16.56 |  | no | no |  | compile: CompileError: todae failed: RocketLanding.LanderWithHorizontalVelocity.RetrogradeRocketLanding failed in ToDae: unsupported Flat semantic owner `function shape proof`: dependent extent is not an exact Integer expression over proven shape axes |
| DroneLibrary.Tests.motorTest | 4.0.0 | 3.0 | 3466 | 31 | 77 | 917 | 452 | 11.39 | 4.02 | yes | yes | 438/3625 | - |
| DroneLibrary.Tests.propellerTest | 4.0.0 | 1.0 | 4256 | 56 | 153 | 1110 | 751 | 12.91 |  | yes | no |  | simulate: SimulationError: solve-IR evaluation failed: algebraic projection did not converge at event boundary: worst scaled residual row=2227 target=force.frame_b.R.T[2,3] value=1.431604e0 ratio=8.343642e5 norm=1.721900e0 row_scale=1.715802e0 scaled_tolerance=1.715802e-6 |
| DroneLibrary.Tests.simpleHoverTest | 4.0.0 | 1.0 | 7613 | 90 | 239 | 1895 | 1196 | 12.58 | 53.14 | yes | yes | 4459/8305 | - |
| DroneLibrary.Examples.DroneWithIdealPower.TestSystem | 4.0.0 | 1.0 | 8005 | 88 | 237 | 2004 | 1327 | 15.11 |  | yes | no |  | simulate: SimulationError: invalid Solve IR contract: GuardedAssignmentProgram node 0 scalar program 0 has invalid register flow: tensor projection op 4 is invalid: tensor fill has an invalid extent or lane count |

Key signals of the simulated models (first -> last value over the window):

- FusalageInGravity `FirstStage.r_0[3]`: 166000 -> 165995 (min 165995, max 166000)
- FusalageInGravity `FirstStage.v_0[3]`: 0 -> -9.80665 (min -9.80665, max 0)
- motorTest `dCmotor.w`: 0 -> 0.0990093 (min 0, max 10)
- motorTest `dCmotor.torque`: 0 -> 0 (min 0, max 2)
- motorTest `feedback.y`: 0 -> 0 (min 0, max 1)
- motorTest `propellerRev1.dCMotor.dCmotor.w`: 0 -> 0.0990093 (min 0, max 10)
- motorTest `propellerRev1.dCMotor.dCmotor.torque`: 0 -> 0 (min 0, max 2)
- motorTest `propellerRev1.rotor3_1.revolute.w`: 0 -> -1.47336e-05 (min -0.15, max 0)
- motorTest `propellerRev1.blades.bodyShape5.r_0[2]`: 0 -> -44.1299 (min -44.1299, max 0)
- motorTest `propellerRev2.dCMotor.dCmotor.w`: 0 -> 0.0990093 (min 0, max 10)
- motorTest `propellerRev2.dCMotor.dCmotor.torque`: 0 -> 0 (min 0, max 2)
- motorTest `propellerRev2.rotor3_1.revolute.w`: 0 -> -1.47336e-05 (min -0.15, max 0)
- motorTest `propellerRev2.blades.bodyShape5.r_0[2]`: 0 -> -44.1299 (min -44.1299, max 0)
- simpleHoverTest `droneChassis1.frame_a.r_0[3]`: 0 -> -3.34974 (min -3.34974, max 0)
- simpleHoverTest `ramp.y`: 0 -> 5 (min 0, max 5)
- simpleHoverTest `PID.u`: 0 -> 8.34974 (min 0, max 8.34974)
- simpleHoverTest `feedback.y`: 0 -> 8.34974 (min 0, max 8.34974)
- simpleHoverTest `propellerRev.dCMotor.dCmotor.w`: 0 -> 55.1115 (min 0, max 55.3702)
- simpleHoverTest `propellerRev.dCMotor.dCmotor.torque`: 0 -> 60.8038 (min 0, max 61.3902)
- simpleHoverTest `propellerRev.rotor3_1.revolute.w`: 0 -> 4.55108 (min 0, max 4.59266)
- simpleHoverTest `propellerRev.blades.bodyShape5.r_0[2]`: 0 -> 0 (min 0, max 0)
- simpleHoverTest `propellerRev1.dCMotor.dCmotor.w`: 0 -> 55.1115 (min 0, max 55.3702)
- simpleHoverTest `propellerRev1.dCMotor.dCmotor.torque`: 0 -> 60.8038 (min 0, max 61.3902)
- simpleHoverTest `propellerRev1.rotor3_1.revolute.w`: 0 -> -4.55116 (min -4.59277, max 0)
- simpleHoverTest `propellerRev1.blades.bodyShape5.r_0[2]`: 0 -> 0 (min 0, max 0)
- simpleHoverTest `propellerRev3.dCMotor.dCmotor.w`: 0 -> 55.1115 (min 0, max 55.3702)
- simpleHoverTest `propellerRev3.dCMotor.dCmotor.torque`: 0 -> 60.8038 (min 0, max 61.3902)
- simpleHoverTest `propellerRev3.rotor3_1.revolute.w`: 0 -> -4.55116 (min -4.59277, max 0)
- simpleHoverTest `propellerRev3.blades.bodyShape5.r_0[2]`: 0 -> 0 (min 0, max 0)

## Triage of the `Modelica.Math.cos` blocker

| attempt | what was changed (scratch copy only) | compiled? | compile s | simulated? | diagnostic (first 400 chars) |
|---|---|---|---|---|---|
| 0-baseline | unpatched results/scratch/msl323 | yes | 5.45 | no | SimulationError: DAE structural proof failed: invalid structural IR contract without source span: independent state selection: external builtin function `Modelica.Math.cos` calls `cos`, which this runtime cannot execute |
| A | Modelica.Math.f: external "builtin" body -> algorithm y := .f(u) | yes | 40.0 | yes | - |
| B | call sites under Modelica/Mechanics/MultiBody: Modelica.Math.f( -> f( | yes | 10.76 | no | SimulationError: invalid Solve IR contract: pure-call argument or slot interface is invalid |
| C | A + B together | yes | 8.75 | yes | - |

## Free-fall check of the unlocked FusalageInGravity simulation

Checked result (solver `rk-like`, dt 0.01, 2.0 s, patched MSL 3.2.3 root): `FirstStage.r_0[3]` drops 19.6133000 m vs g_n t^2/2 = 19.6133000 m, max |dz| = 5.82e-11 m; `FirstStage.v_0[3]` ends at -19.61330 m/s vs -g_n t = -19.61330 m/s, max |dv| = 1.07e-14 m/s (g_n = 9.80665).

Default solver (`auto`, = `bdf`, dt 0.01): max |dz| = 9.68e-03 m, formed within the first 0.08 s and then constant; identical for dt = [0.1, 0.01, 0.001] (spread 6.4e-08 m, dt-independent = True), whereas `rk-like` gives at worst 5.8e-11 m over the same dts. So dt is only the output grid for the BDF path and the offset is a BDF start-up error, not a model error.

Initial conditions: FusalageInGravity.mo fixes no initial condition (r_0(start={0,0,166000}, fixed=false), v_0(fixed=false), a_0(fixed=false)); Rumoca starts from the start values (z0 = 166000 m, v0 = 0), which is what the closed form z0 - g_n t^2/2, -g_n t assumes. The simulated z0/v0 are recorded per run (z0_m, v0_m_s).

| solver | dt | points | drop [m] | max abs dz [m] | max abs dv [m/s] | simulate s |
|---|---|---|---|---|---|---|
| rk-like | 0.1 | 21 | 19.6133000 | 2.91e-11 | 7.11e-15 | 0.804 |
| rk-like | 0.01 | 201 | 19.6133000 | 5.82e-11 | 1.07e-14 | 0.821 |
| rk-like | 0.001 | 2001 | 19.6133000 | 5.82e-11 | 1.42e-14 | 0.855 |
| auto | 0.1 | 21 | 19.6228387 | 9.68e-03 | 3.55e-15 | 0.802 |
| auto | 0.01 | 201 | 19.6228387 | 9.68e-03 | 3.55e-15 | 0.737 |
| auto | 0.001 | 2001 | 19.6228387 | 9.68e-03 | 3.55e-15 | 0.805 |
| bdf | 0.1 | 21 | 19.6228387 | 9.68e-03 | 3.55e-15 | 0.673 |
| bdf | 0.01 | 201 | 19.6228387 | 9.68e-03 | 3.55e-15 | 0.708 |
| bdf | 0.001 | 2001 | 19.6228387 | 9.68e-03 | 3.55e-15 | 0.796 |

Figure: C:\GIT\rumoca-rocket-landing\results\live\multibody_freefall_check.png

## OpenModelica cross-check of DroneLibrary.Tests.motorTest (3 s, both steps)

Rumoca (patched MSL 4.0.0 copy; compile 13.55 s, simulate 4.22 s) vs omc dassl tol 1e-6 (MSL resolved by omc: 4.1.0; 8.7 s wall), compared on 30 mid-interval points avoiding the events at 1 s and 2 s. Worst relative difference over the non-zero signals: 3.67e-07 (pass < 1e-3: True).

| signal | max abs diff | max abs (OM) | value at 1.5 s Rumoca / OM | end value Rumoca / OM |
|---|---|---|---|---|
| `dCmotor.w` | 3.67e-06 | 10 | 10 / 10 | 0.0990093 / 0.0990112 |
| `dCmotor.torque` | 0.00e+00 | 2 | 2 / 2 | 0 / 0 |
| `feedback.y` | 0.00e+00 | 1 | 1 / 1 | 0 / 0 |
| `step.y` | 0.00e+00 | 1 | 1 / 1 | 1 / 1 |
| `step1.y` | 0.00e+00 | 1 | 0 / 0 | 1 / 1 |
| `propellerRev1.rotor3_1.revolute.w` | 2.82e-08 | 0.15 | -0.15 / -0.15 | -1.47336e-05 / -1.47343e-05 |
| `propellerRev1.rotor3_1.revolute.phi` | 6.08e-10 | 0.15 | -0.07335 / -0.07335 | -0.149985 / -0.149985 |
| `propellerRev1.blades.bodyShape5.r_0[2]` | 4.13e-08 | 42.68 | -11.0325 / -11.0325 | -44.1299 / -44.1299 |
| `propellerRev1.blades.bodyShape5.v_0[2]` | 9.24e-14 | 28.93 | -14.71 / -14.71 | -29.42 / -29.4199 |

Figure: C:\GIT\rumoca-rocket-landing\results\live\multibody_om_motortest_check.png

## Loading probes (exact diagnostics)

- load library package.mo with roots=[msl323] (0.5 s): error - `CompileError: RocketLanding.SimpleLander.FusalageInGravity could not be compiled: unresolved type reference: 'Modelica.Mechanics.MultiBody.World' Related failures (showing 2): - <resolve> [Resolve]: unresolved type reference: 'Modelica.Mechanics.MultiBody.World' - <resolve> [Resolve]: unresolved type reference: 'Modelica.Mechanics.MultiBody.Parts.BodyCylinder'`
- load single model file with roots=[msl323] (control, expected to work) (43.9 s): loaded - `RocketLanding.SimpleLander.FusalageInGravity — 6 states, 220 algebraic, 0 inputs, 0 outputs, 126 parameters`
- roots=[msl323, C:/GIT] (parent folder of the library as a second root) (52.87 s): error - `CompileError: Failed to load source root 'C:\GIT': invalid Modelica package layout under 'C:\GIT': PKG-006 directory 'C:\GIT\MMChvdcLibrary\Dependencies\OpenIPSL\Examples\Microgrids\UniversityCampus\CampusA\GUnitTest' is missing package.mo; PKG-006 directory 'C:\GIT\MMChvdcLibrary\Dependencies\OpenIPSL\Examples\Microgrids\UniversityCampus\CampusB\Data' is missing package.mo; PKG-006 directory 'C:\`


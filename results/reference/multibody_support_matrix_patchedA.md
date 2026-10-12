# Rumoca 0.10.2 support matrix: ALSETLab MultiBody models (patchedA)

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


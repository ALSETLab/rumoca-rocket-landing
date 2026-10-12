# Rumoca 0.10.2 support matrix: ALSETLab MultiBody models (patchedA)

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


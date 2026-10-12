# Challenge entry: a Falcon 9-style landing from the ALSETLab RocketLanding library, through Rumoca

Rumoca workshop, American Modelica Conference 2026 (CogniPilot, Georgia Tech, 2026-10-12). Enter with the
presenters by 12:40; the gallery post below is for
https://github.com/CogniPilot/rumoca_workshop_amc2026/issues/1.

## One-line entry for the presenters

Luigi Vanfretti (ALSETLab, RPI). Question: can Rumoca take our open-source Modelica rocket-landing library
(Modelica Conference 2021) and do what the paper left as future work, a fuel-optimal landing, then fly it in
3-D without the licensed DLR Visualization library? Starting model: `RocketLanding.SimpleLander.rocket`
(Canham, Podlaski, Vanfretti). Tool: Claude Code (Claude Fable 5.1, then Claude Opus 5.5) with Rumoca 0.10.2,
CasADi, JAX, OpenModelica 1.27. Elapsed: 2 h 00 min wall time, 10:05 to 12:05 EDT, debugging included.
Shown live: Rumoca's 3-D viewer at http://localhost:8080 (autopilot landing on a loop, Land-it-again button).

## Gallery post (paste as a comment)

### Falcon 9-style landing burn: the ALSETLab RocketLanding Modelica library meets Rumoca

**Name / question.** Luigi Vanfretti, ALSETLab, Rensselaer Polytechnic Institute. Our 2021 Modelica
Conference paper modeled a Falcon 9 first stage returning from stage separation and closed with a call for
fuel-optimal landing trajectories. Can Rumoca take the open-source models from that paper, plan the
fuel-optimal landing burn, check it against an independent tool, and fly it in 3-D without the licensed DLR
Visualization library the paper used?

**Starting model and authors.** The [ALSETLab/RocketLanding](https://github.com/ALSETLab/RocketLanding)
library from C. Canham, M. Podlaski and L. Vanfretti, "Guidance, Navigation, and Control enabling
Retrograde Landing of a First Stage Rocket," Proceedings of the 14th International Modelica Conference,
Linköping, 2021, [doi:10.3384/ecp21181171](https://doi.org/10.3384/ecp21181171). The library holds a 3-D
MultiBody first stage (48 m by 3.7 m, one 914 kN engine, three grid fins, altitude-dependent drag, a
kerosene/oxygen injector) with a four-part flight controller: grid fins, boost-back, re-entry burn and a PD
landing controller. Started at 166 km and 293 m/s, it matched SpaceX Falcon 9 webcast telemetry. Its 3-D
view used the DLR Visualization library, which needs a commercial license. Here I used the library's
`SimpleLander.rocket`, the vertical landing burn, unchanged for the baseline.

**My change.** A re-cast `Lander` model with a throttle input, start/fixed initial conditions, inlined
forces and a smooth drag law, so Rumoca 0.10.2 can export it to CasADi and JAX. A `LanderBurn` model replays
a coast-then-full-throttle schedule in Rumoca's native simulator and in OpenModelica. A `LanderLive` model
with a Three.js scene flies it in Rumoca's own 3-D viewer: an autopilot, manual throttle, a Land-it-again
button and a loop mode.

**Results, with what is new compared with the 2021 paper.**
- **Same model, new compiler.** Rumoca compiles the untouched 2021 model in 0.37 s and simulates 60 s of
  flight in 5 ms. It matches OpenModelica to 0.35 mm in altitude over 2.8 km, with the same burnout
  (10.000 s) and apogee (383.80 m).
- **The paper's future work, done.** CasADi on Rumoca's export (multiple shooting, IPOPT, 16 iterations)
  finds the minimum-propellant landing from 800 m at 70 m/s down: coast, then full throttle from 4.988 s,
  touchdown at 13.92 s with 10.7 of 100 kg propellant left. One second earlier the stage stops 98 m up,
  climbs, runs dry and hits at 43 m/s. One second later it hits at 44 m/s.
- **Checked four ways.** CasADi, a CVODES re-integration, a differentiable JAX rollout and a bisection with
  Rumoca's native event simulator agree on the ignition time within 0.1 ms. OpenModelica, run on the same
  file at that time, lands at 0.3 to 0.4 m/s.
- **Differentiable.** JAX recovers the drag coefficient (0.2000) from OpenModelica's trajectory and shows
  the knife edge: igniting 100 ms late costs 15 m/s of touchdown speed.
- **3-D without the DLR library.** Rumoca's free, open-source Three.js viewer flies the landing live. The
  autopilot lights the engine at 402.4 m and lands at 1.8 m/s, inside the paper's 2 m/s touchdown
  criterion, every 18 s in loop mode.
- **For the Rumoca team.** The repository lists the issues found, with exact errors. Among them: a missed
  grazing touchdown event, time events on a 0.2 ms grid, and a viewer reset that keeps the clock running.
  The full 3-D MultiBody stage does not run yet. With a `Modelica.Math` work-around the fuselage alone
  free-falls exactly at g, but the complete model stops in the compiler at a `Modelica.Fluid` array-shape
  check.

**Tool and time.** Claude Code (Claude Fable 5.1, then Claude Opus 5.5), Rumoca 0.10.2, CasADi 3.7.2, JAX
0.4.38, OpenModelica 1.27. 2 h 00 min wall time including debugging.

**Links.** Models: https://github.com/ALSETLab/RocketLanding. Paper:
https://doi.org/10.3384/ecp21181171. Rumoca experiment: repository link to follow.

## 45-second pitch

"In 2021 our lab published an open-source Modelica model of a Falcon 9 first stage coming home, with a
flight controller that matched SpaceX telemetry. The paper ended by saying the landing could be made
fuel-optimal. Today Rumoca did that. It compiles our model unchanged in a third of a second and matches
OpenModelica to a third of a millimetre. Its CasADi export gives the fuel-optimal landing burn: coast, full
throttle at 4.988 seconds, 10.7 kg of fuel left. One second early or late, you crash. CasADi, CVODES, JAX
and Rumoca's own simulator agree on that time within a tenth of a millisecond. And instead of the licensed
DLR visualization from the paper, here it is in Rumoca's free 3-D viewer, landing at 1.8 metres per second,
on a loop. Paper and models are linked in the gallery. Built with Claude Code in two hours."

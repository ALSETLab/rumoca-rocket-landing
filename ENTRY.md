# Challenge entry: ALSETLab RocketLanding through Rumoca

Text for the presenters (enter by 12:40) and for the gallery issue
(https://github.com/CogniPilot/rumoca_workshop_amc2026/issues/1). Times are wall time including debugging.

## One-line entry for the presenters

Luigi Vanfretti (ALSETLab, RPI). Question: can Rumoca take our 2021 Modelica rocket-landing model, unchanged,
and plan a minimum-propellant hoverslam with CasADi that OpenModelica confirms? Starting model:
`RocketLanding.SimpleLander.rocket` (Canham, Podlaski, Vanfretti, Modelica 2021). Tool: Claude Code
(Claude Fable 5.1, then Claude Opus 5.5) in the Claude desktop app, with Rumoca 0.10.2, CasADi, JAX,
OpenModelica 1.27. Elapsed: 1 h 00 min to the verified plots (10:05 to 11:05 EDT), 1 h 45 min including
the 3-D viewer (to 11:50 EDT), all debugging included. Shown live: the Rumoca viewer at http://localhost:8080
with the autopilot landing, plus the figures.

## Gallery post

**Name / question.** Luigi Vanfretti, ALSETLab (RPI). Can Rumoca run a Modelica rocket model our lab published
in 2021, without re-typing any physics in Python, and let CasADi plan a Falcon-style landing burn that an
independent tool confirms?

**Starting model and change.** `RocketLanding.SimpleLander.rocket` from ALSETLab/RocketLanding (C. Canham,
M. Podlaski, L. Vanfretti, 14th Modelica Conference 2021), used untouched for the baseline. My change: a
re-cast `RumocaRocketLanding.Lander` with a throttle input, `start/fixed` initial conditions instead of an
`initial equation` section, inlined forces and a smooth drag law, because Rumoca 0.10.2's `jax-ode` and
`casadi-ode` targets reject algebraic variables and initialization sections. A second model, `LanderBurn`,
replays an open-loop coast-then-full-throttle schedule in Rumoca's native simulator and in OpenModelica.
A third, `LanderLive`, flies in Rumoca's own 3-D viewer with keyboard throttle or an autopilot that ignites
near the altitude where the CasADi switch falls (402.40 m); Rumoca's native simulator lands it at 1.8 m/s with
12.1 kg left, and a loop mode re-flies it every 18 s.

**Before/after and checked result.**
- Rumoca compiles the untouched 2021 model in 0.37 s and simulates 60 s of flight in 5 ms. Against
  OpenModelica (dassl, tol 1e-8) the altitude differs by at most 0.35 mm over a 2.8 km range, and both tools
  put burnout at 10.000 s and apogee at 383.80 m, 15.033 s.
- CasADi, fed directly by Rumoca's `to_casadi()` export (multiple shooting, RK4, 160 intervals, free final
  time, IPOPT 16 iterations), finds the minimum-propellant landing from 800 m at 70 m/s down: coast, then full
  throttle from 4.988 s, touchdown at 13.92 s with 10.66 kg of 100 kg propellant left. Igniting 1 s early
  arrests the rocket 98 m above the pad, it climbs, runs dry and hits at 43 m/s; 1 s late hits at 44 m/s.
- The check: the same ignition time falls out of four independent routes within 0.1 ms: the CasADi NLP
  (4.98778 s), a CVODES re-integration (4.98775 s), a differentiable JAX rollout (4.98789 s), and a bisection
  with Rumoca's native event-based simulator (4.98782 s). OpenModelica, run on the same `.mo` at that
  ignition time, lands at 0.3 to 0.4 m/s. JAX also recovers the drag coefficient 0.2000 from the OpenModelica
  trajectory and shows the knife edge: igniting 100 ms late costs 15 m/s of touchdown speed.
- Rumoca 0.10.2 findings from the checks: a grazing state event is missed (the rocket passes 1.8 cm below
  the pad for 0.11 s without the `terminate` when-clause firing, which OpenModelica catches); time events are
  resolved on an internal 0.2 ms grid; the Windows release binary lacks the viewer's browser bundle; and the
  viewer's reset restores the states but not the clock.

**Tool and time.** Claude Code (Claude Fable 5.1, then Claude Opus 5.5 after 11:30 EDT), Rumoca 0.10.2,
CasADi 3.7.2, JAX 0.4.38, OpenModelica 1.27. Wall time 1 h 00 min to the verified figures (10:05 to 11:05
EDT) and 1 h 45 min including the 3-D viewer, all debugging included.

**Links.** Repository folder `C:\GIT\rumoca-rocket-landing` (public link to follow if published).

## 45-second pitch

"Our lab published this rocket-landing Modelica model in 2021. Rumoca compiles it unchanged in a third of a
second and matches OpenModelica to a third of a millimetre. I re-cast it with a throttle input, handed
Rumoca's CasADi export to IPOPT, and it planned the hoverslam: coast, full throttle at 4.988 seconds,
touchdown with 10.7 kg of fuel left. One second early you stall and crash, one second late you crash. The
check: four independent routes, CasADi, CVODES, JAX and Rumoca's own event simulator, agree on the ignition
time within a tenth of a millisecond, and OpenModelica lands the same file at 0.3 m/s. Along the way we caught
two event-handling bugs in Rumoca. And here it is in Rumoca's own 3-D viewer: the autopilot lights the
engine at 402 metres and sets it down at under two metres per second, again and again. Built with Claude Code in under two hours."

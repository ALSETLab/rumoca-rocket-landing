# Presenter cheat sheet: Rumoca workshop challenge, AMC 2026

**Title:** Beyond GN&C-Enabled Retrograde Landing: Fuel-Optimal Landing of a First Stage Rocket with Rumoca

## Timeline

- **By 12:40:** enter in person with a presenter (James Goppert or Micah Condie).
- **12:45:** up to five entries drawn at random, plus a backup and the presentation order.
- **12:50–12:55:** 45-second demos, then a show-of-hands vote. The winner gets a CogniPilot stocking cap.

## What to tell the presenter

1. "Luigi Vanfretti, ALSETLab, RPI."
2. "Question: can Rumoca do what our 2021 Modelica paper left as future work, a fuel-optimal Falcon 9-style
   landing, and fly it in 3-D without the licensed DLR library?"
3. "Starting model: SimpleLander.rocket from our open-source RocketLanding library."
4. "Claude Code, about two hours."
5. "Live in Rumoca's 3-D viewer on my laptop, and it's already in the gallery."

## 45-second pitch (if drawn)

"Question: our 2021 Modelica paper landed a Falcon 9 first stage and named fuel-optimal landing as future
work. Can Rumoca do it? Change: I re-cast our open-source rocket model so Rumoca exports it to CasADi and JAX,
and built a scene for Rumoca's own 3-D viewer instead of the licensed DLR library. Result: CasADi plans the
fuel-optimal burn: full throttle at 4.988 seconds, 10.7 kilograms of fuel left. One second early or late, you
crash. Check: CasADi, CVODES, JAX and Rumoca's own simulator agree on that time within a tenth of a
millisecond, and OpenModelica lands the same file softly. Worked with Claude Code for two hours."

**Demo timing:** click "Land it again" as you start the "Result" sentence. The engine lights about 5 s
later and the rocket lands at about 14 s, inside the 45 seconds.

## Tabs to have open

- Rumoca 3-D viewer: http://localhost:8080
- Optimal-landing plot: `results\reference\landing_hoverslam.png` in this repository
- Gallery post: https://github.com/CogniPilot/rumoca_workshop_amc2026/issues/1#issuecomment-6128299659

If the viewer is not running, start it in PowerShell from the repository root:

```powershell
.\viewer\rocket\run_viewer.ps1
```

## Links

- Rumoca experiment: https://github.com/ALSETLab/rumoca-rocket-landing
- Models: https://github.com/ALSETLab/RocketLanding
- Paper: C. Canham, M. Podlaski, L. Vanfretti, "Guidance, Navigation, and Control enabling Retrograde
  Landing of a First Stage Rocket," 14th International Modelica Conference, 2021,
  https://doi.org/10.3384/ecp21181171

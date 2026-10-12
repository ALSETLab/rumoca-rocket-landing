# Triage: `Modelica.Math.cos` external-builtin blocker (FusalageInGravity, MSL 3.2.3)

| attempt | what was changed (scratch copy only) | compiled? | compile s | simulated? | diagnostic (first 400 chars) |
|---|---|---|---|---|---|
| 0-baseline | unpatched results/scratch/msl323 | yes | 5.45 | no | SimulationError: DAE structural proof failed: invalid structural IR contract without source span: independent state selection: external builtin function `Modelica.Math.cos` calls `cos`, which this runtime cannot execute |
| A | Modelica.Math.f: external "builtin" body -> algorithm y := .f(u) | yes | 40.0 | yes | - |
| B | call sites under Modelica/Mechanics/MultiBody: Modelica.Math.f( -> f( | yes | 10.76 | no | SimulationError: invalid Solve IR contract: pure-call argument or slot interface is invalid |
| C | A + B together | yes | 8.75 | yes | - |

# ALSETLab RocketLanding hoverslam in Rumoca's 3-D viewer

Fly a retrograde landing burn yourself, or let the CasADi-derived autopilot do it, and watch it live in
Rumoca's own interactive Three.js viewer (no DLR Visualization).

The vehicle is the 1-D point-mass lander of ALSETLab's `RocketLanding.SimpleLander.rocket` (Canham, Podlaski,
Vanfretti, Modelica 2021): dry mass 100 kg, 100 kg propellant, 2500 N and 10 kg/s at full throttle, drag
0.5*rho*S*cd*v*|v| with rho = 1.225 kg/m3, S = 1.632 m2, cd = 0.2, g = 9.81 m/s2. It starts 800 m above the
pad falling at 70 m/s.

## Files

| File | What |
|---|---|
| `LanderLive.mo` | Self-contained Modelica model: inputs `throttle_cmd` (0..1) and `autopilot` (0/1); states `y`, `v`, `fuel_mass`; latches `ignited`, `landed`, `touchdown_speed`, `crashed`; viewer outputs `position[3]`, `quat[4]`, `thrust`, `vx`, `vy`, `vz`. |
| `rumoca-scenario.toml` | Rumoca scenario: keyboard input, signal wiring, viewer frame and heads-up display, ports 8080 (HTTP) and 8081 (WebSocket). |
| `rocket_scene.js` | Three.js scene: ground, grid, landing pad, rocket with legs, thrust-scaled exhaust, chase camera. No external assets. |
| `run_viewer.sh` | Starts the viewer from `C:\GIT\rumoca-web-assets` (see below). |
| `LanderBatchTest.mo`, `check_autopilot.py`, `batch_autopilot_result.txt` | Batch check of the autopilot with Rumoca's native simulator. |
| `ignition_sweep.txt` | Earlier sweep with a time-triggered autopilot, showing how knife-edge the hoverslam is. |

## The autopilot

The CasADi minimum-propellant solution (`scripts/optimize_landing_casadi.py`) coasts, then burns at full
throttle from t = 4.98775 s. On the coast that instant is at 402.51 m altitude and 87.1 m/s downward. The
autopilot ignites when the altitude drops through `y_ign = 402.45 m`, 6 cm lower, so the rocket touches down
firmly instead of grazing the pad at zero speed. Triggering on altitude rather than time matters because the
viewer's reset (R) restores the states but not the model clock.

Batch check (`check_autopilot.py`, Rumoca native simulator, dt 0.01 s):

| Ignition altitude | Solver | Touchdown | Speed | Propellant left |
|---|---|---|---|---|
| 402.45 m (default) | rk-like | 13.82 s | 1.35 m/s | 11.72 kg |
| 402.45 m (default) | auto | 13.82 s | 1.31 m/s | 11.68 kg |
| 402.51 m (CasADi optimum) | rk-like | 13.90 s | 0.34 m/s | 10.92 kg |
| 402.51 m (CasADi optimum) | auto | 18.42 s | 18.7 m/s, crash | 0 kg |

The last row is Rumoca 0.10.2's missed grazing state event: the touchdown when-clause does not fire, the
rocket climbs again, runs dry and falls. Screenshots of a viewer run: `results/reference/viewer_burn.jpg`
(burn at 58 m, 34 m/s) and `results/reference/viewer_landed.jpg` (landed, green).

## Run it

The Rumoca 0.10.2 Windows release binary does not include the viewer's browser bundle; without it the page
shows "Missing Rumoca viewer shell". The CLI looks for `packages/rumoca-web/viz/viewer.html` and
`packages/rumoca-web/vendor/three_viewer.js` under the current folder, so the launch script runs it from
`C:\GIT\rumoca-web-assets`, which holds both (see the README there).

```powershell
& C:\GIT\rumoca-rocket-landing\viewer\rocket\run_viewer.ps1
```

From Git Bash: `bash C:/GIT/rumoca-rocket-landing/viewer/rocket/run_viewer.sh`.

Then open <http://localhost:8080>. The viewer runs in real time until you press Q or stop `rumoca.exe`.

## Controls

| Key | Action |
|---|---|
| Space | Autopilot on or off (starts on) |
| W / S | Throttle up or down in manual mode (0.7 per second, clamped 0 to 1) |
| R | Reset to 800 m, 70 m/s down, full tank |
| T | Real-time pacing on or off |
| F | Fullscreen |
| C | Camera |
| Q | Quit |

## Credits

Model and parameters from ALSETLab RocketLanding (C. Canham, M. Podlaski, L. Vanfretti). Viewer and scenario
format from Rumoca 0.10.2 (CogniPilot, Apache-2.0).

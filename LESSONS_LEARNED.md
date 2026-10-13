# Lessons learned: ALSETLab Modelica libraries in Rumoca 0.10.2

Notes from bringing two ALSETLab libraries into Rumoca at the AMC 2026 Rumoca workshop (2026-10-12/13):
the RocketLanding library ([ALSETLab/rumoca-rocket-landing](https://github.com/ALSETLab/rumoca-rocket-landing))
and the DroneLibrary ([ALSETLab/rumoca-drone-library](https://github.com/ALSETLab/rumoca-drone-library)).
Both repositories carry this same file. Versions: Rumoca 0.10.2 (Python wheel and Windows CLI), CasADi 3.7.2,
JAX 0.4.38, OpenModelica 1.27, Windows 11.

## Rumoca's Python API

- **Fully qualified names.** `Session().load(path, model=...)` needs the fully qualified class name for a
  file with a `within` clause, for example `RocketLanding.SimpleLander.rocket`.
- **One session per model.** Reusing a session for a second file gives "Duplicate class ... non-identical
  definition".
- **CasADi and JAX exports need pure explicit ODEs.** `to_casadi()` and `to_jax()` return `rhs(t, x, u, p)`,
  but reject models with algebraic variables or an `initial equation` section. Inline the forces and use
  `start`/`fixed` attributes for initial conditions.
- **Smooth laws for gradients.** Write `if`-based laws smoothly, e.g. drag `v*sqrt(v^2 + eps^2)` instead of
  `if v >= 0 ...`, so CasADi and JAX can differentiate them.
- **No time-varying inputs in batch.** Native `simulate()` has no `inputs=` argument yet, and a model with an
  unbound `input` cannot run in batch. Put open-loop schedules in Modelica (a wrapper model). The CLI
  scenario route does drive inputs live (keyboard, gamepad).
- **Events work natively,** including `if`, `when`, `reinit` and `terminate`, but see the event caveats below.

## Event handling caveats

- **Grazing state events can be missed.** A touchdown at nearly zero speed passed 1.8 cm below the pad for
  0.11 s without the `when y <= 0` clause firing. OpenModelica caught it. Keep a margin, e.g. a firm 1.8 m/s
  touchdown rather than a zero-speed graze.
- **Time events sit on a ~0.2 ms grid.** Ignition times spaced 2e-5 s apart gave only five distinct results.
- **Live viewer and time-based conditions.** In the live viewer, a `when time >= t_land + hold` condition
  never fired, while a state-based `when pad_timer >= hold` did.
- **Solver choice matters.** With an extra state, the default `auto` solver missed a grazing event that
  `rk-like` caught.
- **Same accuracy at default tolerance.** Against OpenModelica the untouched 2021 rocket model matched to
  0.35 mm. The residual was Rumoca's default tolerance, which shrinks with `SimConfig(rtol, atol)`.

## Modelica Standard Library and MultiBody

- **One root for the MSL.** References across separate source roots do not resolve, so the MSL must be one
  folder holding `Modelica/`, `ModelicaServices/` and `Complex/` side by side.
- **OpenModelica's ModelicaServices needs two patches:** `OpenModelica.Internal.Architecture.integerMax()` to
  `2147483647`, and `OpenModelica.Scripting.uriToFilename(uri)` to `uri`.
- **MultiBody needs a `Modelica.Math` work-around.** Every MultiBody model compiled, but none simulated
  ("external builtin function `Modelica.Math.cos` calls `cos`, which this runtime cannot execute") until the
  `external "builtin"` bodies were rewritten as `algorithm y := .cos(u);`. Rewriting the call sites instead
  did not work.
- **Package layout is checked.** The CLI rejects a source root containing any folder without `package.mo`
  (PKG-006); the Python API loads the same model file from inside the tree.
- **Support status.** 3 of 7 MultiBody models from the two libraries simulate. The full rocket stage stops in
  the compiler at a `Modelica.Fluid` array-shape check; details in the rocket repository's
  `results/reference/multibody_support_matrix.md`.
- **Reset and re-initialization fail on MultiBody** (reported as
  [CogniPilot/rumoca#430](https://github.com/CogniPilot/rumoca/issues/430)): the viewer reset ends the drone
  viewer, and a variant with one extra state fails at start-up.
- **Cross-check pays off.** The unmodified DroneLibrary hover test matched OpenModelica to 0.22 mm in altitude
  over 4 s, which showed that its odd behavior (sink, overshoot, divergence) is the test's own PID tuning.

## Rumoca's 3-D viewer on Windows

- **The Windows release binary lacks the browser bundle.** Without it the page says "Missing Rumoca viewer
  shell". The CLI looks for `packages/rumoca-web/viz/viewer.html` and
  `packages/rumoca-web/vendor/three_viewer.js` under the current folder. viewer.html comes from the v0.10.2
  source; three_viewer.js is an esbuild bundle of three 0.164.1 plus `GLTFLoader`, the same entry as
  `packages/rumoca-web/build.mjs`. Run `rumoca sim -c <scenario>` from the folder that holds them.
- **Reset keeps the clock.** The viewer's reset restores the states but not model time, so make autopilots
  and loops state-triggered (an altitude, a timer state), not time-triggered.
- **Buttons from the scene.** The viewer page forwards key presses over its WebSocket, so a scene script can
  add DOM buttons that dispatch synthetic `KeyboardEvent`s (R, L, C, Space).
- **Replay when live fails.** When a model cannot be reset, record one Rumoca run and play it back on a tiny
  clock model; label it as a replay.
- **Large models.** The 7613-equation drone compiles in one to two minutes and runs at 0.7 to 1.1 times real
  time.

## Windows practicalities

- **`localhost` can be slow.** Here `http://localhost:<port>` answered about 2 s late, because Windows tried IPv6
  first and the viewer listens on IPv4 only; `127.0.0.1` answered in milliseconds. A readiness poll with a
  short timeout must use `127.0.0.1`.
- **Launchers.** The `.ps1` launchers start `rumoca.exe` as a child process in the same window, poll
  `127.0.0.1` until the viewer answers, open the browser, then wait. A background job for the poll was
  unreliable.
- **OpenModelica as a reference.** `omc` scripts with `outputFormat="csv"` gave independent references for
  every result here, including MultiBody models.

## Working with a coding agent

These results were produced with Claude Code (Claude Fable 5.1, then Claude Opus 5.5). What helped:

- **Adversarial verification.** Running each result past an independent verifier caught a mislabeled plot
  legend, a 10 ms zero-order-hold bias in a JAX rollout, and a "crash" that was really a missed event.
- **Keep the check independent.** Every number came from Rumoca or OpenModelica, never from hand-written
  physics in Python.
- **Review before publishing.** Check staged files before pushing to a public repository, and remove
  untested claims before filing an issue.
- **Background agents keep their model.** Agents launched under one model stop when that model's quota runs
  out; relaunch them under the new model.

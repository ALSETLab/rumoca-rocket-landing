# Start the Rumoca 3-D viewer for the rocket lander (PowerShell): http://localhost:8080
# Compiles in a second; the autopilot lands the rocket on a loop.
# Uses the viewer kit in the external/rumoca-web-assets submodule (browser bundle the Rumoca 0.10.2 Windows
# release binary lacks; rumoca.exe is downloaded there on first use). The browser opens by itself.
# Stop with Q in the page or Ctrl+C here.
$kit = Join-Path $PSScriptRoot "..\..\external\rumoca-web-assets\start_viewer.ps1"
if (-not (Test-Path $kit)) { throw "Viewer kit missing: run 'git submodule update --init' in the repository root." }
& $kit -Scenario (Join-Path $PSScriptRoot "rumoca-scenario.toml") -Url "http://localhost:8080"

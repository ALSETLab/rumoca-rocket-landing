#!/usr/bin/env bash
# Start the Rumoca 3-D viewer for the rocket lander (Git Bash): http://localhost:8080
# Rumoca 0.10.2 (Windows release binary) looks for its browser bundle (packages/rumoca-web) under the CURRENT
# folder, so run from the external/rumoca-web-assets submodule (fetch bin/rumoca.exe with its get_rumoca.ps1).
here="$(cd "$(dirname "$0")" && pwd)"
kit="$here/../../external/rumoca-web-assets"
cd "$kit" && exec ./bin/rumoca.exe sim -c "$here/rumoca-scenario.toml"

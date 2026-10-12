#!/usr/bin/env bash
# Start the Rumoca 3-D viewer for the rocket lander, then open http://localhost:8080
# The Rumoca 0.10.2 Windows release binary does not ship the browser bundle; it looks for
# packages/rumoca-web/{viz/viewer.html,vendor/three_viewer.js} under the CURRENT folder,
# so run from C:/GIT/rumoca-web-assets (built with esbuild from three@0.164.1, see its README).
cd C:/GIT/rumoca-web-assets && exec "$LOCALAPPDATA/rumoca/bin/rumoca.exe" sim -c C:/GIT/rumoca-rocket-landing/viewer/rocket/rumoca-scenario.toml

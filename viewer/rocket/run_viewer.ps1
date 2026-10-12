# Start the Rumoca 3-D viewer for the rocket lander (PowerShell), then open http://localhost:8080
# The Rumoca 0.10.2 Windows release binary looks for its browser bundle (packages/rumoca-web) under the
# CURRENT folder, so it is started from C:\GIT\rumoca-web-assets. Stop with Q in the page or Ctrl+C here.
$rumoca = "C:\GIT\rumoca-web-assets\bin\rumoca.exe"   # Rumoca 0.10.2 Windows CLI (release asset rumoca-windows-x86_64.exe)
Push-Location C:\GIT\rumoca-web-assets
try {
    Start-Process "http://localhost:8080" | Out-Null
    & $rumoca sim -c C:\GIT\rumoca-rocket-landing\viewer\rocket\rumoca-scenario.toml
}
finally {
    Pop-Location
}

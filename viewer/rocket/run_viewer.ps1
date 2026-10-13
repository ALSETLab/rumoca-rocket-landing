# Start the Rumoca 3-D viewer for the rocket lander (PowerShell), then open http://localhost:8080
# The Rumoca 0.10.2 Windows release binary looks for its browser bundle (packages/rumoca-web) under the
# CURRENT folder, so it is started from C:\GIT\rumoca-web-assets. Stop with Q in the page or Ctrl+C here.
$rumoca = "C:\GIT\rumoca-web-assets\bin\rumoca.exe"   # Rumoca 0.10.2 Windows CLI (release asset rumoca-windows-x86_64.exe)
Push-Location C:\GIT\rumoca-web-assets
try {
    # Open the browser only once the viewer answers (compiling can take a minute or two)
    $null = Start-Job -ScriptBlock {
        param($url)
        for ($i = 0; $i -lt 600; $i++) {
            try { Invoke-WebRequest $url -UseBasicParsing -TimeoutSec 2 | Out-Null; Start-Process $url; break }
            catch { Start-Sleep -Seconds 1 }
        }
    } -ArgumentList "http://localhost:8080"
    & $rumoca sim -c C:\GIT\rumoca-rocket-landing\viewer\rocket\rumoca-scenario.toml
}
finally {
    Pop-Location
}

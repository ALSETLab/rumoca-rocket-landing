# Start the Rumoca 3-D viewer for the rocket lander (PowerShell): http://localhost:8080
# Compiles in a second; the autopilot lands the rocket on a loop.
# Rumoca 0.10.2 (Windows release binary) looks for its browser bundle (packages/rumoca-web) under the CURRENT
# folder, so it is started from C:\GIT\rumoca-web-assets. The browser opens by itself once the viewer answers.
# Stop with Q in the page or Ctrl+C here.
$rumoca   = "C:\GIT\rumoca-web-assets\bin\rumoca.exe"   # release asset rumoca-windows-x86_64.exe
$scenario = "C:\GIT\rumoca-rocket-landing\viewer\rocket\rumoca-scenario.toml"
$url      = "http://localhost:8080"
$probe    = $url -replace "localhost", "127.0.0.1"   # localhost tries IPv6 first and answers ~2 s late

$proc = Start-Process -FilePath $rumoca -ArgumentList @("sim", "-c", $scenario) `
    -WorkingDirectory "C:\GIT\rumoca-web-assets" -NoNewWindow -PassThru
try {
    # Poll until the viewer serves its page (compiling can take a while), then open the browser once.
    while (-not $proc.HasExited) {
        try {
            Invoke-WebRequest $probe -UseBasicParsing -TimeoutSec 2 | Out-Null
            Start-Process $url
            break
        } catch {
            Start-Sleep -Seconds 1
        }
    }
    $proc.WaitForExit()
}
finally {
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
}

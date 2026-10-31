# pusula dead-man check (registered as the scheduled task "pusula-health").
#
# Separate from the nightly task on purpose: when that task does not start at all (its script
# moved, 2026-10-29), nothing inside it can report it. Measures, alerts, changes nothing.
param(
    [string]$EnvFile = $(if ($env:PUSULA_ENV_FILE) { $env:PUSULA_ENV_FILE } else { Join-Path (Split-Path -Parent $PSScriptRoot) ".env" }),
    [string]$Uv = "uv"
)
$repo = Split-Path -Parent $PSScriptRoot
$pusulaHome = if ($env:PUSULA_HOME) { $env:PUSULA_HOME } else { Join-Path $env:USERPROFILE ".pusula" }
$logDir = Join-Path $pusulaHome "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$env:PUSULA_ENV_FILE = $EnvFile
$env:PYTHONUTF8 = "1"
$p = Start-Process -FilePath $Uv -ArgumentList @("run", "--no-sync", "pusula", "health", "--notify") `
    -WorkingDirectory $repo -NoNewWindow -Wait -PassThru `
    -RedirectStandardOutput (Join-Path $logDir "health.out.log") `
    -RedirectStandardError (Join-Path $logDir "health.err.log")
exit $p.ExitCode

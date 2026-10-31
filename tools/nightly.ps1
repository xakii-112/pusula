# pusula nightly forward run (registered as the scheduled task "pusula-forward").
#
# Updates bars and implied vols, advances every forward book by the newly completed sessions
# and writes the dashboard snapshot and daily report. Idempotent: a session already processed
# is never processed again, so a missed or repeated run is harmless. Then reviews the forward
# records and runs the dead-man check, which reports the run's own exit code: a failed night
# reaches the operator instead of a log nobody reads (PUSULA_TELEGRAM_* in the dotenv file).
#
# Credentials are read at run time from the dotenv file named here (never copied).
param(
    [string]$EnvFile = $(if ($env:PUSULA_ENV_FILE) { $env:PUSULA_ENV_FILE } else { Join-Path (Split-Path -Parent $PSScriptRoot) ".env" }),
    [string]$Uv = "uv",
    # "all" or "family/book" to mirror in the paper account; empty: nothing is sent
    [string]$Mirror = "",
    # with -Mirror all: dollars each book gets, instead of an equal share of the account
    [string]$CapitalPerBook = ""
)
$repo = Split-Path -Parent $PSScriptRoot
$pusulaHome = if ($env:PUSULA_HOME) { $env:PUSULA_HOME } else { Join-Path $env:USERPROFILE ".pusula" }
$logDir = Join-Path $pusulaHome "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmm"
$out = Join-Path $logDir "nightly-$stamp.out.log"
$err = Join-Path $logDir "nightly-$stamp.err.log"
$env:PUSULA_ENV_FILE = $EnvFile
$env:PYTHONWARNINGS = "ignore::UserWarning"
$env:PYTHONUTF8 = "1"
# Start-Process keeps stderr out of PowerShell's error stream (in 5.1 a redirected native
# stderr line becomes a terminating error under -ErrorAction Stop).
$p = Start-Process -FilePath $Uv -ArgumentList @("run", "--no-sync", "pusula", "forward", "--family", "all") `
    -WorkingDirectory $repo -RedirectStandardOutput $out -RedirectStandardError $err `
    -NoNewWindow -Wait -PassThru
$code = $p.ExitCode
function Invoke-Pusula([string[]]$CommandArgs) {
    $what = $CommandArgs[0]
    $q = Start-Process -FilePath $Uv -ArgumentList (@("run", "--no-sync", "pusula") + $CommandArgs) `
        -WorkingDirectory $repo -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput (Join-Path $logDir "nightly-$stamp.$what.out.log") `
        -RedirectStandardError (Join-Path $logDir "nightly-$stamp.$what.err.log")
    return $q.ExitCode
}
$null = Invoke-Pusula @("review", "--family", "all")
if ($Mirror -eq "all" -and $code -eq 0) {
    $sendArgs = @("execute", "--all", "--send")
    if ($CapitalPerBook) { $sendArgs += @("--capital-per-book", $CapitalPerBook) }
    $null = Invoke-Pusula $sendArgs
} elseif ($Mirror -and $code -eq 0) {
    $family, $book = $Mirror -split "/", 2
    $null = Invoke-Pusula @("execute", "--family", $family, "--book", $book, "--send")
}
$health = Invoke-Pusula @("health", "--notify", "--digest", "--forward-exit", "$code", "--err-log", "`"$err`"")
if ($code -ne 0) { exit $code }
exit $health

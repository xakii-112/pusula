# Registers the two scheduled tasks from this checkout's own location (an operator action).
# Safe to run again: it replaces the tasks, so run it after moving the repository; a task keeps
# the path it was registered with and fails silently once the script is gone (2026-10-29).
#
#   pusula-forward  01:00 Tuesday to Saturday: bars, books, review, health
#   pusula-health   12:00 every day: reports what the nightly run did or failed to do
#
# -Mirror all (or family/book) makes the nightly run send the orders to the paper account too;
# -CapitalPerBook 100000 gives each book that much instead of an equal share of the account.
param([string]$Mirror = "", [string]$CapitalPerBook = "")
$ErrorActionPreference = "Stop"

function Register-PusulaTask(
    [string]$Name, [string]$Script, $Trigger, [string]$Description, [int]$Hours, [string]$Extra = ""
) {
    $arg = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$Script`" $Extra"
    $action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable `
        -ExecutionTimeLimit (New-TimeSpan -Hours $Hours) -MultipleInstances IgnoreNew `
        -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    Register-ScheduledTask -TaskName $Name -Action $action -Trigger $Trigger -Settings $settings `
        -Description $Description -Force | Out-Null
    Write-Output "registered $Name -> $Script"
}

$nightly = New-ScheduledTaskTrigger -Weekly -At "01:00" `
    -DaysOfWeek Tuesday, Wednesday, Thursday, Friday, Saturday
$extra = if ($Mirror) { "-Mirror $Mirror" } else { "" }
if ($CapitalPerBook) { $extra += " -CapitalPerBook $CapitalPerBook" }
Register-PusulaTask "pusula-forward" (Join-Path $PSScriptRoot "nightly.ps1") $nightly `
    "pusula shadow books" 3 $extra
$noon = New-ScheduledTaskTrigger -Daily -At "12:00"
Register-PusulaTask "pusula-health" (Join-Path $PSScriptRoot "health.ps1") $noon `
    "pusula dead-man check" 1

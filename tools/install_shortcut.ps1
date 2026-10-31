# Puts a "Pusula" shortcut on the desktop that opens the program in its own window, without a
# console. It runs the project's own interpreter, so it follows `uv sync`. Run it again after moving
# the project; to remove the program's entry, delete the shortcut.
param([string]$Name = "Pusula")
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$pythonw = Join-Path $repo ".venv\Scripts\pythonw.exe"
if (-not (Test-Path -LiteralPath $pythonw)) { throw "no $pythonw : run ``uv sync`` first" }
$icon = Join-Path $repo "src\pusula\web\pusula.ico"
$desktop = [Environment]::GetFolderPath("Desktop")
$path = Join-Path $desktop "$Name.lnk"
$link = (New-Object -ComObject WScript.Shell).CreateShortcut($path)
$link.TargetPath = $pythonw
$link.Arguments = "-m pusula.cli app"
$link.WorkingDirectory = $repo
$link.IconLocation = $icon
$link.Description = "Pusula: forward books, research ledger and system health (read-only)"
$link.Save()
Write-Output "wrote $path"

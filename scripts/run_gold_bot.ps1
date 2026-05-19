$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BotDir = Split-Path -Parent $ScriptDir
$BotFile = Join-Path $BotDir "gold_bot.py"
$LogDir = Join-Path $BotDir "logs"
$LogFile = Join-Path $LogDir "gold_control.log"
$RuntimeLog = Join-Path $LogDir "gold_runtime.log"

if (!(Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

Set-Location $BotDir
Add-Content -Path $LogFile -Value ""
Add-Content -Path $LogFile -Value "============================================================"
Add-Content -Path $LogFile -Value "[$(Get-Date)] Gold bot PowerShell launcher started"
Add-Content -Path $LogFile -Value "Bot file: $BotFile"
Add-Content -Path $LogFile -Value "Runtime log: $RuntimeLog"
Write-Host "Starting gold bot..."
Write-Host "Control log: $LogFile"
Write-Host "Runtime log: $RuntimeLog"
python -u $BotFile 2>&1 | Tee-Object -FilePath $RuntimeLog -Append
$ExitCode = if ($LASTEXITCODE -ne $null) { $LASTEXITCODE } else { 0 }
Add-Content -Path $LogFile -Value "[$(Get-Date)] gold_bot.py exited with code $ExitCode"
exit $ExitCode

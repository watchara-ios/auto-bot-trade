$ErrorActionPreference = "Stop"

$BotDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BotFile = Join-Path $BotDir "forex_bot.py"
$LogDir = Join-Path $BotDir "logs"
$LogFile = Join-Path $LogDir "forex_cron.log"

if (!(Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

Set-Location $BotDir
Write-Host "Starting forex bot..."
Write-Host "Log: $LogFile"
python -u $BotFile 2>&1 | Tee-Object -FilePath $LogFile -Append

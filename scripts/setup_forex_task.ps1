$ErrorActionPreference = "Stop"

$TaskName = "Forex Bot"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BotDir = Split-Path -Parent $ScriptDir
$BotFile = Join-Path $BotDir "forex_bot.py"
$LogDir = Join-Path $BotDir "logs"
$LogFile = Join-Path $LogDir "forex_cron.log"

if (!(Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

$Python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $Python) {
    throw "Python was not found in PATH. Install Python or edit this script and set `$Python to python.exe."
}

$ActionCommand = "cd /d `"$BotDir`" && `"$Python`" -u `"$BotFile`" >> `"$LogFile`" 2>&1"
$Action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c $ActionCommand"
$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At 14:00
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description "Runs forex_bot.py on working days at 14:00" `
    -Force | Out-Null

Write-Host "Scheduled task created: $TaskName" -ForegroundColor Green
Write-Host "Runs Monday-Friday at 14:00"
Write-Host "Bot: $BotFile"
Write-Host "Log: $LogFile"

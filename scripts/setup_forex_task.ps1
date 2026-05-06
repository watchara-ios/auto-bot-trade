$ErrorActionPreference = "Stop"

$TaskName = "Forex Bot"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BotDir = Split-Path -Parent $ScriptDir
$BotFile = Join-Path $BotDir "forex_bot.py"
$RunBat = Join-Path $ScriptDir "run_forex_bot.bat"
$LogDir = Join-Path $BotDir "logs"
$LogFile = Join-Path $LogDir "forex_control.log"
$RuntimeLog = Join-Path $LogDir "forex_runtime.log"

if (!(Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

if (!(Test-Path $RunBat)) {
    throw "Runner not found: $RunBat"
}

$Action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c `"$RunBat`""
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
Write-Host "Control log: $LogFile"
Write-Host "Runtime log: $RuntimeLog"

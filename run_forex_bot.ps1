$ErrorActionPreference = "Stop"

$BotDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BotFile = Join-Path $BotDir "forex_bot.py"

Set-Location $BotDir
python $BotFile

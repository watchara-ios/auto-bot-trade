@echo off
setlocal EnableExtensions

set "BOT_DIR=%~dp0.."
for %%I in ("%BOT_DIR%") do set "BOT_DIR=%%~fI"
set "LOG_DIR=%BOT_DIR%\logs"
set "LOG_FILE=%LOG_DIR%\forex_cron.log"
set "TASK_NAME=Forex Bot"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

>> "%LOG_FILE%" echo.
>> "%LOG_FILE%" echo ============================================================
>> "%LOG_FILE%" echo [%date% %time%] Forex bot stop requested

schtasks /End /TN "%TASK_NAME%" >> "%LOG_FILE%" 2>&1

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$procs = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*forex_bot.py*' }; " ^
  "if (-not $procs) { Write-Host 'No forex_bot.py process found'; exit 0 }; " ^
  "$procs | ForEach-Object { Write-Host ('Stopping PID {0}: {1}' -f $_.ProcessId, $_.CommandLine); Stop-Process -Id $_.ProcessId -Force }" ^
  >> "%LOG_FILE%" 2>&1

set "EXIT_CODE=%ERRORLEVEL%"
>> "%LOG_FILE%" echo [%date% %time%] Forex bot stop finished code %EXIT_CODE%
exit /b %EXIT_CODE%

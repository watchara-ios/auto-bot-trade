@echo off
setlocal EnableExtensions

set "TASK_NAME=Forex Bot"
for %%I in ("%~dp0..") do set "BOT_DIR=%%~fI"
set "BOT_FILE=%BOT_DIR%\forex_bot.py"
set "RUN_BAT=%BOT_DIR%\scripts\run_forex_bot.bat"
set "LOG_DIR=%BOT_DIR%\logs"
set "LOG_FILE=%LOG_DIR%\forex_cron.log"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

if not exist "%RUN_BAT%" (
    echo Runner not found: %RUN_BAT%
    pause
    exit /b 1
)

echo [%date% %time%] Installing scheduled task...>> "%LOG_FILE%"
echo Runner: %RUN_BAT%>> "%LOG_FILE%"

schtasks /Create /F ^
    /TN "%TASK_NAME%" ^
    /SC WEEKLY ^
    /D MON,TUE,WED,THU,FRI ^
    /ST 14:00 ^
    /TR "\"%RUN_BAT%\"" ^
    /RL LIMITED

if errorlevel 1 (
    echo Failed to create scheduled task.
    echo [%date% %time%] ERROR: Failed to create scheduled task.>> "%LOG_FILE%"
    pause
    exit /b 1
)

echo Scheduled task created: %TASK_NAME%
echo Runs Monday-Friday at 14:00
echo Runner: %RUN_BAT%
echo Log file: %LOG_FILE%
echo.
echo Test now:
echo   schtasks /Run /TN "%TASK_NAME%"
echo.
echo Watch log:
echo   powershell -Command "Get-Content -Wait '%LOG_FILE%'"
echo [%date% %time%] Scheduled task created: %TASK_NAME%>> "%LOG_FILE%"
pause

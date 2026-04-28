@echo off
setlocal

set "TASK_NAME=Forex Bot"
for %%I in ("%~dp0..") do set "BOT_DIR=%%~fI\"
set "BOT_FILE=%BOT_DIR%forex_bot.py"
set "LOG_DIR=%BOT_DIR%logs"
set "LOG_FILE=%LOG_DIR%\forex_cron.log"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found in PATH.
    echo Install Python or edit this file and replace python with the full python.exe path.
    pause
    exit /b 1
)

schtasks /Create /F ^
    /TN "%TASK_NAME%" ^
    /SC WEEKLY ^
    /D MON,TUE,WED,THU,FRI ^
    /ST 14:00 ^
    /TR "cmd /c cd /d \"%BOT_DIR%\" && python \"%BOT_FILE%\" >> \"%LOG_FILE%\" 2>&1"

if errorlevel 1 (
    echo Failed to create scheduled task.
    pause
    exit /b 1
)

echo Scheduled task created: %TASK_NAME%
echo Runs Monday-Friday at 14:00
echo Log file: %LOG_FILE%
pause

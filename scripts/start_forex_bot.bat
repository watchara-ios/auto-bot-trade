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
>> "%LOG_FILE%" echo [%date% %time%] Forex bot scheduled start requested

schtasks /Run /TN "%TASK_NAME%" >> "%LOG_FILE%" 2>&1
set "EXIT_CODE=%ERRORLEVEL%"

>> "%LOG_FILE%" echo [%date% %time%] Forex bot scheduled start finished code %EXIT_CODE%
exit /b %EXIT_CODE%

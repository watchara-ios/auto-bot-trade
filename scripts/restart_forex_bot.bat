@echo off
setlocal EnableExtensions

set "BOT_DIR=%~dp0.."
for %%I in ("%BOT_DIR%") do set "BOT_DIR=%%~fI"
set "LOG_DIR=%BOT_DIR%\logs"
set "LOG_FILE=%LOG_DIR%\forex_control.log"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

>> "%LOG_FILE%" echo.
>> "%LOG_FILE%" echo ============================================================
>> "%LOG_FILE%" echo [%date% %time%] Forex bot restart requested

call "%~dp0stop_forex_bot.bat"
timeout /t 3 /nobreak >nul
call "%~dp0start_forex_bot.bat"

set "EXIT_CODE=%ERRORLEVEL%"
>> "%LOG_FILE%" echo [%date% %time%] Forex bot restart finished code %EXIT_CODE%
exit /b %EXIT_CODE%

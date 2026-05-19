@echo off
setlocal EnableExtensions

set "BOT_DIR=%~dp0.."
for %%I in ("%BOT_DIR%") do set "BOT_DIR=%%~fI"
set "BOT_FILE=%BOT_DIR%\gold_bot.py"
set "LOG_DIR=%BOT_DIR%\logs"
set "LOG_FILE=%LOG_DIR%\gold_control.log"
set "RUNTIME_LOG=%LOG_DIR%\gold_runtime.log"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

>> "%LOG_FILE%" echo.
>> "%LOG_FILE%" echo ============================================================
>> "%LOG_FILE%" echo [%date% %time%] Gold bot launcher started
>> "%LOG_FILE%" echo BOT_DIR=%BOT_DIR%
>> "%LOG_FILE%" echo BOT_FILE=%BOT_FILE%
>> "%LOG_FILE%" echo USER=%USERNAME%
>> "%LOG_FILE%" echo COMPUTER=%COMPUTERNAME%
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
chcp 65001 >nul 2>nul
>> "%LOG_FILE%" echo PYTHONUTF8=%PYTHONUTF8%
>> "%LOG_FILE%" echo PYTHONIOENCODING=%PYTHONIOENCODING%

cd /d "%BOT_DIR%"
if errorlevel 1 (
    >> "%LOG_FILE%" echo [%date% %time%] ERROR: cannot cd to "%BOT_DIR%"
    exit /b 1
)

set "PYTHON_CMD="
where py >nul 2>nul
if not errorlevel 1 set "PYTHON_CMD=py -3"

if "%PYTHON_CMD%"=="" (
    where python >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=python"
)

if "%PYTHON_CMD%"=="" (
    >> "%LOG_FILE%" echo [%date% %time%] ERROR: Python was not found in PATH.
    exit /b 1
)

>> "%LOG_FILE%" echo [%date% %time%] Python command: %PYTHON_CMD%
>> "%LOG_FILE%" echo [%date% %time%] Starting gold_bot.py
>> "%LOG_FILE%" echo [%date% %time%] Runtime log: %RUNTIME_LOG%

%PYTHON_CMD% -u "%BOT_FILE%" >> "%RUNTIME_LOG%" 2>&1
set "EXIT_CODE=%ERRORLEVEL%"

>> "%LOG_FILE%" echo [%date% %time%] gold_bot.py exited with code %EXIT_CODE%
exit /b %EXIT_CODE%

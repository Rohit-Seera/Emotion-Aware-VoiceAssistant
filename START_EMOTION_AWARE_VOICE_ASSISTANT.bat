@echo off
setlocal
cd /d "%~dp0"

title Emotion Aware Voice Assistant
echo ===============================================
echo   Emotion Aware Voice Assistant
echo ===============================================

REM Reuse this project's server if it is already responding.
curl.exe -fsS --max-time 2 "http://127.0.0.1:8000/openapi.json" 2>nul | findstr /c:"Emotion Aware Voice Assistant" >nul
if not errorlevel 1 goto :open_browser

if not exist ".venv\Scripts\python.exe" (
  echo Creating the Python environment...
  py -3.13 -m venv .venv
  if errorlevel 1 goto :python_error
)

echo Checking and installing project dependencies...
where uv >nul 2>nul
if not errorlevel 1 (
  uv sync --extra dev --locked
  if errorlevel 1 goto :dependency_error
) else (
  .venv\Scripts\python.exe -m pip install -U pip
  .venv\Scripts\python.exe -m pip install -e ".[dev]"
  if errorlevel 1 goto :dependency_error
)

if not exist "logs" mkdir "logs"
echo Starting the backend. First startup can take about a minute...
start "Emotion Aware Voice Assistant server" /min "%~dp0.venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --log-level info

set /a attempts=0
:wait_for_server
curl.exe -fsS --max-time 2 "http://127.0.0.1:8000/openapi.json" 2>nul | findstr /c:"Emotion Aware Voice Assistant" >nul
if not errorlevel 1 goto :open_browser
set /a attempts+=1
if %attempts% GEQ 90 goto :server_error
timeout /t 2 /nobreak >nul
goto :wait_for_server

:open_browser
echo Opening the voice assistant...
start "" "http://127.0.0.1:8000/api/v1/demo?visual=particles"
exit /b 0

:python_error
echo Python 3.13 could not create the virtual environment.
echo Install Python 3.13, then double-click this file again.
pause
exit /b 1

:dependency_error
echo Dependency setup failed. Check your internet connection and try again.
pause
exit /b 1

:server_error
echo The backend did not become ready within 3 minutes.
echo Check the minimized server window for its error message.
echo The launch was stopped without opening a broken browser tab.
pause
exit /b 1

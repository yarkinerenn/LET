@echo off
setlocal

rem Determine the absolute path to the repository root (where this script lives)
set "SCRIPT_DIR=%~dp0"

echo Starting frontend (npm start)...
pushd "%SCRIPT_DIR%explainable-nlp"
start "" cmd /k "npm start"
popd

echo Starting backend (python app.py)...
pushd "%SCRIPT_DIR%backend"
python app.py
set "BACKEND_EXIT=%ERRORLEVEL%"
popd

echo Backend exited with code %BACKEND_EXIT%.
echo If the frontend window is still running, close it manually.

endlocal & exit /b %BACKEND_EXIT%


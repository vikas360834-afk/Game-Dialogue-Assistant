@echo off
setlocal
cd /d "%~dp0"
echo ==========================================
echo   Game Dialogue Assistant - Web Launcher
echo ==========================================
if not exist ".venv\Scripts\python.exe" (
    echo Creating Python virtual environment...
    py -3 -m venv .venv
    if errorlevel 1 goto error
)
call ".venv\Scripts\activate.bat"
echo Installing/checking dependencies...
python -m pip install --upgrade pip
if errorlevel 1 goto error
python -m pip install -r requirements.txt
if errorlevel 1 goto error
echo Starting Streamlit...
python -m streamlit run app/web.py
if errorlevel 1 goto error
goto end
:error
echo.
echo Startup failed. Check the error above and confirm Python 3 is installed.
pause
:end
endlocal

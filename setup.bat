@echo off
REM Setup script for Spend Data Management Platform (Windows .bat)

REM Ensure UTF-8 output for emojis and nicer text (requires Windows 10/11 cmd)
chcp 65001 >nul 2>&1

echo 🚀 Setting up Spend Data Management Platform...

REM If a local virtualenv exists, activate it so subsequent commands run inside it
if exist ".venv\Scripts\activate.bat" (
    echo Activating local virtual environment...
    call ".venv\Scripts\activate.bat"
)
REM If no .venv exists, try to create one
if not exist ".venv\Scripts\activate.bat" (
    echo No .venv found — creating virtual environment...
    python -m venv .venv
    if %ERRORLEVEL% NEQ 0 (
        echo Failed to create .venv. Please create a virtual environment manually and re-run.
        exit /b 1
    )
    REM Ensure pip is available in the created venv; use ensurepip then upgrade pip/setuptools/wheel
    echo Ensuring pip is available in the virtual environment...
    call ".venv\Scripts\python.exe" -m ensurepip --upgrade >nul 2>&1
    call ".venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel >nul 2>&1
    echo Activating created virtual environment...
    call ".venv\Scripts\activate.bat"
)

REM Resolve DB path from config and show pre-flight confirmation
for /f "delims=" %%p in ('python - <<"PY"
import sys
sys.path.insert(0, '.')
from src.config import config
print(config.database.path)
PY') do set DB_PATH=%%p

echo Resolved database path: %DB_PATH%
if "%SKIP_CONFIRM%"=="" (
    set /p response=This will initialize or overwrite data at %DB_PATH%. Continue? [y/N] 
    if /I not "%response%"=="y" if /I not "%response%"=="yes" (
        echo Aborted by user. No changes made.
        exit /b 0
    )
)
REM Check if uv is available
where uv >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ❌ "uv" is not found on PATH.
    echo Please install "uv" first. On Windows you can use PowerShell to run the installer:
    echo.
    echo   powershell -Command "iwr -useb https://astral.sh/uv/install.ps1 | iex"
    echo.
    echo If that does not apply, see https://astral.sh/uv for platform-specific instructions.
    exit /b 1
)

echo ✅ uv is available

echo 📦 Installing dependencies...
uv sync
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: "uv sync" failed. Please check the output above.
)

echo 🗄️ Initializing database...
uv run python scripts/init_db.py
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: database initialization script failed. Check the output.
)

echo 📊 Loading sample data...
uv run python scripts/load_sample_data.py
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: sample data load failed. Check the output.
)

echo.
echo 🎉 Setup completed (or attempted). If there were warnings above, address them and re-run this script.
echo.
echo To start the application:
echo   uv run streamlit run src/app.py
echo.
echo Then open your browser and go to: http://localhost:8501
echo.
echo Demo Credentials:
echo   Admin: admin / admin1234
echo   Spend Manager: manager / manager1234
echo   Data Analyst: analyst / analyst1234
echo.
pause

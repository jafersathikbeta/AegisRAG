@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ============================================
echo  AegisRAG — Build Windows Setup.exe
echo ============================================
echo.

where python >nul 2>&1
if errorlevel 1 (
  echo ERROR: python not found on PATH.
  exit /b 1
)

echo [1/3] Installing PyInstaller + pywin32 + winshell...
python -m pip install --upgrade pip pyinstaller pywin32 winshell
if errorlevel 1 exit /b 1

if not exist "installer\icon.ico" (
  echo WARNING: installer\icon.ico missing — Setup.exe will use default icon.
)

echo.
echo [2/3] Compiling installer\setup_ui.py --^> dist\Setup\Setup.exe ...
echo.

set "ROOT=%cd%"
python -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --windowed ^
  --onedir ^
  --name "Setup" ^
  --distpath "%ROOT%\dist" ^
  --workpath "%ROOT%\build\pyinstaller" ^
  --specpath "%ROOT%\build" ^
  --icon "%ROOT%\installer\icon.ico" ^
  --hidden-import win32com.client ^
  --hidden-import winshell ^
  --collect-submodules win32com ^
  --add-data "%ROOT%\installer\icon.ico;." ^
  --add-data "%ROOT%\installer\download_models.py;." ^
  "%ROOT%\installer\setup_ui.py"

if errorlevel 1 (
  echo.
  echo BUILD FAILED.
  exit /b 1
)

echo.
echo [3/3] Done.
echo.
echo Final binary:
echo   %cd%\dist\Setup\Setup.exe
echo.
echo IMPORTANT: Copy the entire dist\Setup\ folder into the AegisRAG project
echo root (next to requirements.txt and run.py), then run Setup.exe.
echo The wizard creates venv, installs deps, and adds a Desktop shortcut.
echo.
pause
endlocal

@echo off
rem Double-click to start Satellite Space Flight (Windows).
rem The first start installs what it needs; see start.py. You can also drop a
rem scenario .json file onto this file to open it.
title Satellite Space Flight
cd /d "%~dp0"

rem Prefer a Python that already has numpy and pygame-ce, then any Python 3.10+
rem (start.py then sets one up). The py launcher goes before "python", which
rem may be the Microsoft Store placeholder.
set "READY=import sys, numpy, pygame; sys.exit(sys.version_info < (3, 10) or not getattr(pygame, 'IS_CE', 0))"
set "RECENT=import sys; sys.exit(sys.version_info < (3, 10))"
py -3 -c "%READY%" >nul 2>nul && (set "PY=py -3" & goto run)
python -c "%READY%" >nul 2>nul && (set "PY=python" & goto run)
python3 -c "%READY%" >nul 2>nul && (set "PY=python3" & goto run)
py -3 -c "%RECENT%" >nul 2>nul && (set "PY=py -3" & goto run)
python -c "%RECENT%" >nul 2>nul && (set "PY=python" & goto run)
python3 -c "%RECENT%" >nul 2>nul && (set "PY=python3" & goto run)

echo.
echo  Satellite Space Flight needs Python 3.10 or newer, and none was found.
echo.
echo   1. Install Python from https://www.python.org/downloads/
echo      (on the installer's first screen, tick "Add python.exe to PATH")
echo   2. Double-click this file again.
echo.
pause
exit /b 1

:run
echo  This window closes by itself when you quit the simulator.
echo.
%PY% start.py %*
if errorlevel 1 (
    echo.
    pause
)

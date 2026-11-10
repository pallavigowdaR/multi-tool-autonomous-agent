@echo off
title Multi-Tool Autonomous AI Agent Console
cd /d "%~dp0"
echo ===================================================
echo Starting Multi-Tool Autonomous AI Agent...
echo ===================================================
echo.
.\venv\Scripts\python.exe agent.py
echo.
pause


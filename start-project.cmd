@echo off
setlocal
cd /d "%~dp0" || exit /b 1
chcp 65001 >nul
title QQ Project Supervisor
if not exist ".venv\Scripts\python.exe" (
  echo Python virtual environment is missing. See README.md.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m qq_ollama_bot.supervisor %*
if errorlevel 1 pause

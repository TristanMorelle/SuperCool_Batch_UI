@echo off
cd /d "%~dp0"
call .\venv\Scripts\activate
python ui_select.py
pause
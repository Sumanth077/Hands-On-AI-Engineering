@echo off
cd /d "%~dp0"
echo Starting Streamlit Form Filling App...
call .venv\Scripts\activate.bat
streamlit run app.py
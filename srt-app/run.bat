@echo off
REM Launch the Sinhala Subtitle Translator locally.
REM Optionally point at a specific model folder:
REM   set SINHALA_MODEL_DIR=D:\Personal_Projects\Subtitle\models\nllb-sinhala-v4
cd /d "%~dp0"
python app.py
pause

@echo off
echo ====================================================
echo Handwritten Digit Recognition - Windows Setup
echo ====================================================

echo.
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.7+ from https://python.org
    pause
    exit /b 1
)

echo Python found!
echo.

echo Installing requirements...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements
    pause
    exit /b 1
)

echo.
echo Creating directories...
if not exist "models" mkdir models
if not exist "data" mkdir data

echo.
echo Testing installation...
python test_installation.py

echo.
echo ====================================================
echo Setup completed!
echo ====================================================
echo.
echo To train a model: python train_model.py
echo To run the app:   python main.py
echo.
pause

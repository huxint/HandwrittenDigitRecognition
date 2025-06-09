@echo off
echo Starting Handwritten Digit Recognition App...
echo.

REM Check if model exists
if not exist "models\digit_model.pth" (
    echo No trained model found!
    echo.
    echo Would you like to train a model first? This will take 5-10 minutes.
    set /p choice="Train model now? (y/n): "
    if /i "%choice%"=="y" (
        echo Training model...
        python train_model.py
        if errorlevel 1 (
            echo Training failed!
            pause
            exit /b 1
        )
    )
)

echo Launching GUI application...
python main.py

pause

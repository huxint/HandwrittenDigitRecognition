@echo off
echo ====================================================
echo Uploading Handwritten Digit Recognition to GitHub
echo ====================================================

echo.
echo Step 1: Adding remote repository...
git remote add origin https://github.com/huxint/HandwrittenDigitRecognition.git

echo.
echo Step 2: Pushing to GitHub...
git push -u origin main

echo.
echo ====================================================
echo Upload completed!
echo ====================================================
echo.
echo Your repository should now be available at:
echo https://github.com/huxint/HandwrittenDigitRecognition
echo.
pause

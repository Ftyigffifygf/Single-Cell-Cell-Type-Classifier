@echo off
echo ========================================
echo  Single-Cell Cell-Type Classifier
echo  Starting System...
echo ========================================

echo.
echo [1/3] Starting API Backend Server...
start "API Backend" python serve_api_simple.py

echo [2/3] Waiting for backend to initialize...
timeout /t 3 /nobreak > nul

echo [3/3] Opening Frontend Interface...
start "" "simple_frontend.html"

echo.
echo ========================================
echo  System Started Successfully!
echo ========================================
echo  Backend API: http://localhost:8000
echo  Frontend UI: Opened in browser
echo ========================================
echo.
echo Instructions:
echo 1. Use sample data buttons for quick testing
echo 2. Enter gene names and expression levels
echo 3. Click "Predict Cell Type" to get results
echo.
echo Press any key to stop the API server...
pause > nul

echo.
echo Stopping API server...
taskkill /f /fi "WINDOWTITLE eq API Backend" > nul 2>&1
echo System stopped.
pause

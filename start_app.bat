@echo off
echo ========================================
echo  Single-Cell Cell-Type Classifier
echo  Starting Backend + Frontend Servers
echo ========================================

echo.
echo Starting API Backend Server (Port 8000)...
start "API Backend" python serve_api_simple.py

echo Waiting for backend to initialize...
timeout /t 3 /nobreak > nul

echo.
echo Starting Frontend Server (Port 3000)...
start "Frontend" python serve_frontend.py

echo.
echo ========================================
echo  Servers Started Successfully!
echo ========================================
echo  Backend API: http://localhost:8000
echo  Frontend UI: http://localhost:3000
echo ========================================
echo.
echo Press any key to stop all servers...
pause > nul

echo.
echo Stopping servers...
taskkill /f /im python.exe > nul 2>&1
echo Servers stopped.
pause

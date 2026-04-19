@echo off
REM Quick start script for Lizard Lens development (Windows)

echo 🦎 Lizard Lens - Development Setup
echo ================================================
echo.

REM Check Python
echo Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.11+
    exit /b 1
)
echo ✅ Python found
echo.

REM Check Node
echo Checking Node.js...
node --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Node.js is not installed. Please install Node.js 18+
    exit /b 1
)
echo ✅ Node.js found
echo.

REM Setup Backend
echo Setting up backend...
cd backend

if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing Python dependencies...
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo ✅ Backend setup complete
echo.

REM Check models
echo Checking for trained models...
if not exist "..\Spring_2025\runs\detect\train_yolov8n_v2\weights\best.pt" (
    echo ⚠️ Warning: YOLO model not found at expected path
    echo Expected: Spring_2025\runs\detect\train_yolov8n_v2\weights\best.pt
) else (
    echo ✅ YOLO model found
)
echo.

REM Setup Frontend
echo Setting up frontend...
cd ..\frontend

if not exist "node_modules" (
    echo Installing Node dependencies...
    call npm install
) else (
    echo Node modules already installed
)

echo ✅ Frontend setup complete
echo.

cd ..

echo ✅ Setup Complete!
echo.
echo To start the application:
echo.
echo Terminal 1 (Backend):
echo   cd backend
echo   venv\Scripts\activate
echo   python -m app.main
echo.
echo Terminal 2 (Frontend):
echo   cd frontend
echo   npm run dev
echo.
echo Then open: http://localhost:5173
echo.
echo API Documentation: http://localhost:8000/api/docs
echo.

pause




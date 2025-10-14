#!/bin/bash
# Quick start script for Florida Anole Classifier development

set -e

echo "🦎 Florida Anole Classifier - Development Setup"
echo "================================================"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Conda
echo -e "${BLUE}Checking Conda...${NC}"
if ! command_exists conda; then
    echo -e "${RED}❌ Conda is not installed${NC}"
    echo "Please install Anaconda or Miniconda:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
echo -e "${GREEN}✅ Conda found: $(conda --version)${NC}"

# Check Node
echo -e "${BLUE}Checking Node.js...${NC}"
if ! command_exists node; then
    echo -e "${RED}❌ Node.js is not installed. Please install Node.js 18+${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Node.js found: $(node --version)${NC}"

# Setup Backend
echo ""
echo -e "${BLUE}Setting up backend...${NC}"
cd backend

# Check if conda environment exists
if ! conda env list | grep -q "anole-classifier"; then
    echo "Creating conda environment from environment.yml..."
    conda env create -f environment.yml
else
    echo "Conda environment 'anole-classifier' already exists"
    echo "Updating dependencies from environment.yml..."
    conda env update -f environment.yml --prune -q
fi

echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate anole-classifier

echo -e "${GREEN}✅ Backend setup complete${NC}"

# Check if models exist
echo ""
echo -e "${BLUE}Checking for trained models...${NC}"
if [ ! -f "../Spring_2025/runs/detect/train_yolov8n_v2/weights/best.pt" ]; then
    echo -e "${RED}⚠️  Warning: YOLO model not found at expected path${NC}"
    echo "Expected: Spring_2025/runs/detect/train_yolov8n_v2/weights/best.pt"
    echo "Please ensure your trained models are available."
else
    echo -e "${GREEN}✅ YOLO model found${NC}"
fi

# Setup Frontend
echo ""
echo -e "${BLUE}Setting up frontend...${NC}"
cd ../frontend

if [ ! -d "node_modules" ]; then
    echo "Installing Node dependencies..."
    npm install
else
    echo "Node modules already installed"
fi

echo -e "${GREEN}✅ Frontend setup complete${NC}"

# Summary
cd ..
echo ""
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo ""
echo "To start the application:"
echo ""
echo "Terminal 1 (Backend):"
echo -e "  ${BLUE}cd backend${NC}"
echo -e "  ${BLUE}conda activate anole-classifier${NC}"
echo -e "  ${BLUE}python -m app.main${NC}"
echo ""
echo "Terminal 2 (Frontend):"
echo -e "  ${BLUE}cd frontend${NC}"
echo -e "  ${BLUE}npm run dev${NC}"
echo ""
echo "Then open: http://localhost:5173"
echo ""
echo "API Documentation: http://localhost:8000/api/docs"
echo ""




#!/bin/bash

# Frontend Setup Script for ONNX Support
# This script sets up the frontend to use ONNX models by default

echo "================================================"
echo "Frontend ONNX Setup"
echo "================================================"

# Check if we're in the frontend directory
if [ ! -f "package.json" ]; then
    echo "Error: Not in frontend directory"
    echo "Please run: cd frontend && ./setup_frontend.sh"
    exit 1
fi

# Step 1: Install dependencies
echo ""
echo "Step 1: Installing dependencies..."
npm install

if [ $? -ne 0 ]; then
    echo "✗ npm install failed"
    exit 1
fi
echo "✓ Dependencies installed"

# Step 2: Copy WASM files
echo ""
echo "Step 2: Copying ONNX Runtime WASM files..."

if [ ! -d "node_modules/onnxruntime-web/dist" ]; then
    echo "✗ onnxruntime-web not found in node_modules"
    echo "  Make sure package.json includes: \"onnxruntime-web\": \"^1.23.0\""
    exit 1
fi

# Create public directory if it doesn't exist
mkdir -p public

# Copy WASM files
cp node_modules/onnxruntime-web/dist/*.wasm public/ 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✓ WASM files copied to public/"
    ls -lh public/*.wasm
else
    echo "✗ Failed to copy WASM files"
    exit 1
fi

# Step 3: Check ONNX models
echo ""
echo "Step 3: Checking ONNX models..."

if [ -f "public/models/yolo_best.onnx" ]; then
    size=$(du -h public/models/yolo_best.onnx | cut -f1)
    echo "✓ YOLO model found ($size)"
else
    echo "⚠ YOLO model not found at public/models/yolo_best.onnx"
    echo "  Run: python ../export_yolo_to_onnx.py"
fi

if [ -f "public/models/swin_model.onnx" ]; then
    size=$(du -h public/models/swin_model.onnx | cut -f1)
    echo "✓ Swin model found ($size)"
else
    echo "⚠ Swin model not found at public/models/swin_model.onnx"
    echo "  Run: python ../export_swin_to_onnx.py"
fi

# Step 4: Create .env file if it doesn't exist
echo ""
echo "Step 4: Checking environment configuration..."

if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    cat > .env << EOF
# Backend API URL
VITE_API_URL=http://localhost:8000

# ONNX Model URLs (relative to public/)
VITE_YOLO_MODEL_URL=/models/yolo_best.onnx
VITE_SWIN_MODEL_URL=/models/swin_model.onnx
EOF
    echo "✓ .env file created"
else
    echo "✓ .env file exists"
fi

# Summary
echo ""
echo "================================================"
echo "Setup Summary"
echo "================================================"
echo ""
echo "Frontend Configuration:"
echo "  ✓ Dependencies installed"
echo "  ✓ WASM files in place"
echo "  ✓ Default mode: ONNX (client-side)"
echo "  ✓ Fallback: Backend API"
echo ""

if [ -f "public/models/yolo_best.onnx" ] && [ -f "public/models/swin_model.onnx" ]; then
    echo "✅ All ONNX models ready!"
    echo ""
    echo "Start the dev server:"
    echo "  npm run dev"
    echo ""
    echo "Frontend will:"
    echo "  1. Load ONNX models on startup"
    echo "  2. Use client-side inference by default"
    echo "  3. Fall back to backend API if ONNX fails"
else
    echo "⚠️  ONNX models missing!"
    echo ""
    echo "Export the models first:"
    echo "  cd .."
    echo "  python export_yolo_to_onnx.py"
    echo "  python export_swin_to_onnx.py"
    echo ""
    echo "Then start the dev server:"
    echo "  cd frontend"
    echo "  npm run dev"
fi

echo ""
echo "================================================"


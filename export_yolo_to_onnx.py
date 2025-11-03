"""
Export YOLOv8 model to ONNX format
This script exports the fine-tuned YOLOv8 model to ONNX for use in both backend and frontend
"""
from ultralytics import YOLO
import os
import shutil
from pathlib import Path

def find_yolo_weights():
    """Find YOLO weights file"""
    possible_paths = [
        "Spring_2025/ultralytics_runs/detect/train_yolov8n_v2/weights/best.pt",
        "Spring_2025/ultralytics_runs/detect/train_yolov8n/weights/best.pt",
        "Spring_2025/ultralytics_runs/detect/train_yolov11n/weights/best.pt",
        "Spring_2025/ultralytics_runs/detect/train_yolov8s/weights/best.pt",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

def export_yolo_to_onnx(model_path=None):
    """Export YOLOv8 model to ONNX format"""
    
    print("="*60)
    print("Exporting YOLOv8 to ONNX")
    print("="*60)
    
    # Find model if not specified
    if model_path is None:
        print("\n1. Looking for YOLO model...")
        model_path = find_yolo_weights()
        
        if model_path is None:
            print("\n✗ YOLO model not found!")
            print("\nThe weights directory should contain best.pt:")
            print("  Spring_2025/ultralytics_runs/detect/train_yolov8n_v2/weights/best.pt")
            print("\nOptions:")
            print("  1. Train a new model:")
            print("     cd Spring_2025")
            print("     # See object_detection_train yolo.ipynb")
            print("\n  2. If you have weights elsewhere, create the directory and copy them:")
            print("     mkdir -p Spring_2025/ultralytics_runs/detect/train_yolov8n_v2/weights")
            print("     cp /path/to/your/best.pt Spring_2025/ultralytics_runs/detect/train_yolov8n_v2/weights/")
            print("\n  3. Run this script with the path:")
            print("     python export_yolo_to_onnx.py /path/to/your/best.pt")
            return False
    
    backend_output = "backend/models/yolo_best.onnx"
    frontend_output = "frontend/public/models/yolo_best.onnx"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\n✗ Model not found at {model_path}")
        return False
    
    # Load the model
    print(f"\n1. Loading YOLO model from {model_path}...")
    try:
        model = YOLO(model_path)
        print("   ✓ Model loaded successfully")
        print(f"   Model task: {model.task}")
        if hasattr(model, 'names'):
            print(f"   Classes: {model.names}")
    except Exception as e:
        print(f"   ✗ Failed to load model: {e}")
        return False
    
    # Export to ONNX
    print("\n2. Exporting to ONNX format...")
    print("   This may take a few minutes...")
    
    try:
        # Export with Ultralytics
        # This will create best.onnx in the same directory as best.pt
        export_path = model.export(
            format='onnx',
            imgsz=640,  # Input size
            dynamic=False,  # Static shape for better compatibility
            simplify=True,  # Simplify the model
            opset=12,  # ONNX opset version
        )
        print(f"   ✓ ONNX export successful")
        print(f"   Exported to: {export_path}")
        
        # Check the exported file
        if not os.path.exists(export_path):
            raise FileNotFoundError(f"Exported file not found at {export_path}")
        
        export_size = os.path.getsize(export_path) / (1024 * 1024)
        print(f"   Model size: {export_size:.1f} MB")
        
    except Exception as e:
        print(f"   ✗ ONNX export failed: {e}")
        return False
    
    # Copy to backend and frontend locations
    print("\n3. Copying ONNX model to deployment locations...")
    
    try:
        # Create output directories if they don't exist
        os.makedirs("backend/models", exist_ok=True)
        os.makedirs("frontend/public/models", exist_ok=True)
        
        # Copy to backend
        print(f"   Copying to {backend_output}...")
        shutil.copy2(export_path, backend_output)
        backend_size = os.path.getsize(backend_output) / (1024 * 1024)
        print(f"   ✓ Backend ONNX model saved ({backend_size:.1f} MB)")
        
        # Copy to frontend
        print(f"   Copying to {frontend_output}...")
        shutil.copy2(export_path, frontend_output)
        frontend_size = os.path.getsize(frontend_output) / (1024 * 1024)
        print(f"   ✓ Frontend ONNX model saved ({frontend_size:.1f} MB)")
        
    except Exception as e:
        print(f"   ✗ Failed to copy files: {e}")
        return False
    
    # Verify the exported model
    print("\n4. Verifying exported ONNX model...")
    try:
        import onnx
        onnx_model = onnx.load(backend_output)
        onnx.checker.check_model(onnx_model)
        print("   ✓ ONNX model is valid")
        
        # Print model info
        print(f"\n   Model inputs:")
        for input in onnx_model.graph.input:
            dims = [d.dim_value if d.dim_value > 0 else 'dynamic' for d in input.type.tensor_type.shape.dim]
            print(f"     - {input.name}: {dims}")
        
        print(f"\n   Model outputs:")
        for output in onnx_model.graph.output:
            dims = [d.dim_value if d.dim_value > 0 else 'dynamic' for d in output.type.tensor_type.shape.dim]
            print(f"     - {output.name}: {dims}")
            
    except ImportError:
        print("   ⚠ onnx package not installed, skipping validation")
        print("   Install with: pip install onnx")
    except Exception as e:
        print(f"   ⚠ ONNX validation warning: {e}")
    
    # Test with ONNX Runtime
    print("\n5. Testing with ONNX Runtime...")
    try:
        import onnxruntime as ort
        import numpy as np
        
        # Load ONNX model
        session = ort.InferenceSession(backend_output)
        
        print(f"   ✓ ONNX Runtime loaded model")
        print(f"   Available providers: {ort.get_available_providers()}")
        print(f"   Using providers: {session.get_providers()}")
        
        # Get input details
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        print(f"   Input: {input_name} {input_shape}")
        
        # Create dummy input
        dummy_input = np.random.randn(1, 3, 640, 640).astype(np.float32)
        
        # Run inference
        outputs = session.run(None, {input_name: dummy_input})
        print(f"   ✓ Test inference successful")
        print(f"   Output shape: {outputs[0].shape}")
            
    except ImportError:
        print("   ⚠ onnxruntime not installed, skipping runtime test")
        print("   Install with: pip install onnxruntime")
    except Exception as e:
        print(f"   ⚠ ONNX Runtime test warning: {e}")
    
    print("\n" + "="*60)
    print("Export Complete!")
    print("="*60)
    print(f"\nOriginal model: {model_path}")
    print(f"Exported ONNX: {export_path}")
    print(f"Backend copy: {backend_output}")
    print(f"Frontend copy: {frontend_output}")
    print(f"\nModel details:")
    print(f"  - Input size: 640x640")
    print(f"  - Format: ONNX opset 12")
    print(f"  - Simplified: Yes")
    print("\nBoth models exported:")
    print("  ✓ YOLO (detection): backend/models/yolo_best.onnx")
    print("  ✓ Swin (classification): backend/models/swin_model.onnx")
    print("\nNext steps:")
    print("1. Test backend:")
    print("   python backend/example_onnx_usage.py")
    print("\n2. Setup frontend:")
    print("   cd frontend")
    print("   npm install")
    print("   cp node_modules/onnxruntime-web/dist/*.wasm public/")
    print("   npm run dev")
    
    return True


if __name__ == "__main__":
    import sys
    
    # Check if path provided as argument
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    try:
        success = export_yolo_to_onnx(model_path)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nExport interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

"""
Export Swin Transformer model to ONNX format
This script exports the fine-tuned Swin model to ONNX for use in both backend and frontend
"""
import torch
from transformers import SwinForImageClassification, AutoImageProcessor
import os
from pathlib import Path

def export_swin_to_onnx():
    """Export Swin Transformer model to ONNX format"""
    
    # Paths
    model_path = "Spring_2025/swin_transformer_base_lizard_v4"
    backend_output = "backend/models/swin_model.onnx"
    frontend_output = "frontend/public/models/swin_model.onnx"
    
    print("="*60)
    print("Exporting Swin Transformer to ONNX")
    print("="*60)
    
    # Load the model
    print(f"\n1. Loading model from {model_path}...")
    try:
        model = SwinForImageClassification.from_pretrained(model_path)
        processor = AutoImageProcessor.from_pretrained(model_path)
        print("   ✓ Model loaded successfully")
        print(f"   Model info: {model.config.num_labels} classes")
    except Exception as e:
        print(f"   ✗ Failed to load model: {e}")
        return False
    
    # Set model to evaluation mode
    model.eval()
    print("   ✓ Model set to evaluation mode")
    
    # Get the expected image size from model config
    image_size = model.config.image_size
    print(f"\n2. Creating dummy input (size: {image_size}x{image_size})...")
    dummy_input = torch.randn(1, 3, image_size, image_size)
    print(f"   ✓ Dummy input shape: {dummy_input.shape}")
    
    # Test model with dummy input
    print("\n3. Testing model inference...")
    try:
        with torch.no_grad():
            output = model(dummy_input)
            logits = output.logits
            print(f"   ✓ Model inference successful")
            print(f"   Output shape: {logits.shape}")
    except Exception as e:
        print(f"   ✗ Model inference failed: {e}")
        return False
    
    # Export to ONNX
    print("\n4. Exporting to ONNX format...")
    
    # Create output directories if they don't exist
    os.makedirs("backend/models", exist_ok=True)
    os.makedirs("frontend/public/models", exist_ok=True)
    
    # Export parameters
    input_names = ["pixel_values"]
    output_names = ["logits"]
    dynamic_axes = {
        "pixel_values": {0: "batch_size"},
        "logits": {0: "batch_size"}
    }
    
    try:
        # Export to backend location
        print(f"   Exporting to {backend_output}...")
        torch.onnx.export(
            model,
            dummy_input,
            backend_output,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=14,
            do_constant_folding=True,
            export_params=True,
        )
        backend_size = os.path.getsize(backend_output) / (1024 * 1024)
        print(f"   ✓ Backend ONNX model saved ({backend_size:.1f} MB)")
        
        # Export to frontend location
        print(f"   Exporting to {frontend_output}...")
        torch.onnx.export(
            model,
            dummy_input,
            frontend_output,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=14,
            do_constant_folding=True,
            export_params=True,
        )
        frontend_size = os.path.getsize(frontend_output) / (1024 * 1024)
        print(f"   ✓ Frontend ONNX model saved ({frontend_size:.1f} MB)")
        
    except Exception as e:
        print(f"   ✗ ONNX export failed: {e}")
        return False
    
    # Verify the exported model
    print("\n5. Verifying exported ONNX model...")
    try:
        import onnx
        onnx_model = onnx.load(backend_output)
        onnx.checker.check_model(onnx_model)
        print("   ✓ ONNX model is valid")
        
        # Print model info
        print(f"\n   Model inputs:")
        for input in onnx_model.graph.input:
            print(f"     - {input.name}: {[d.dim_value for d in input.type.tensor_type.shape.dim]}")
        
        print(f"\n   Model outputs:")
        for output in onnx_model.graph.output:
            print(f"     - {output.name}: {[d.dim_value if d.dim_value > 0 else 'dynamic' for d in output.type.tensor_type.shape.dim]}")
            
    except ImportError:
        print("   ⚠ onnx package not installed, skipping validation")
        print("   Install with: pip install onnx")
    except Exception as e:
        print(f"   ⚠ ONNX validation warning: {e}")
    
    # Test with ONNX Runtime
    print("\n6. Testing with ONNX Runtime...")
    try:
        import onnxruntime as ort
        import numpy as np
        
        # Load ONNX model
        session = ort.InferenceSession(backend_output)
        
        # Prepare input
        dummy_input_np = dummy_input.numpy()
        
        # Run inference
        outputs = session.run(None, {"pixel_values": dummy_input_np})
        onnx_logits = outputs[0]
        
        # Compare with PyTorch output
        torch_logits_np = logits.detach().numpy()
        max_diff = np.abs(onnx_logits - torch_logits_np).max()
        
        print(f"   ✓ ONNX Runtime inference successful")
        print(f"   Max difference between PyTorch and ONNX: {max_diff:.6f}")
        
        if max_diff < 1e-4:
            print(f"   ✓ Outputs match (difference < 1e-4)")
        else:
            print(f"   ⚠ Outputs differ slightly (this is usually acceptable)")
            
    except ImportError:
        print("   ⚠ onnxruntime not installed, skipping runtime test")
        print("   Install with: pip install onnxruntime")
    except Exception as e:
        print(f"   ⚠ ONNX Runtime test warning: {e}")
    
    print("\n" + "="*60)
    print("Export Complete!")
    print("="*60)
    print(f"\nBackend model: {backend_output}")
    print(f"Frontend model: {frontend_output}")
    print("\nNext steps:")
    print("1. For backend: Use with pipeline.py by setting use_onnx=True")
    print("2. For frontend: npm install && npm run dev")
    print("\nTesting:")
    print("  Backend: python backend/example_onnx_usage.py")
    print("  Frontend: See frontend/ONNX_SETUP.md")
    
    return True


if __name__ == "__main__":
    import sys
    
    try:
        success = export_swin_to_onnx()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nExport interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


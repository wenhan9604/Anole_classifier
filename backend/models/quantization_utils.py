"""
ONNX Model Quantization Utilities

This module provides utilities for quantizing ONNX models to reduce size and improve inference speed.
Supports dynamic int8 quantization (both QInt8 and QUInt8 types).

Functions:
    - quantize_model(): Quantize ONNX models using dynamic int8 quantization
    - validate_quantized_model(): Verify quantized models are valid and loadable
    - get_model_info(): Extract metadata from ONNX models
"""

import os
import logging
from typing import Dict, Any, Union

try:
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType
    import onnxruntime as ort
except ImportError as e:
    raise ImportError(
        f"Required packages not found: {e}. "
        "Install with: pip install onnx onnxruntime"
    )


logger = logging.getLogger(__name__)


# ONNX data type code to string name mapping
DTYPE_MAP = {
    1: "float32",
    2: "uint8",
    3: "int8",
    4: "uint16",
    5: "int16",
    6: "int32",
    7: "int64",
    8: "string",
    9: "bool",
    10: "float16",
    11: "float64",
    12: "uint32",
    13: "uint64",
}


def quantize_model(
    model_path: str,
    output_path: str,
    quantization_type: QuantType = QuantType.QUInt8,
) -> Dict[str, Union[str, float]]:
    """
    Quantize an ONNX model using dynamic int8 quantization.
    
    This function reduces model size and can improve inference speed by converting
    floating-point weights to 8-bit integer representations dynamically.
    
    Args:
        model_path: Path to the input ONNX model file
        output_path: Path where the quantized model will be saved
        quantization_type: Type of quantization - QuantType.QInt8 (signed) or QuantType.QUInt8 (unsigned).
                          Default is QuantType.QUInt8.
    
    Returns:
        Dictionary containing:
            - "original_size_mb": Original model size in MB
            - "quantized_size_mb": Quantized model size in MB
            - "size_reduction_percent": Size reduction percentage
            - "quantization_type": Type of quantization applied
            - "output_path": Path to the quantized model
    
    Raises:
        FileNotFoundError: If the input model file doesn't exist
        RuntimeError: If quantization fails
    
    Examples:
        >>> result = quantize_model(
        ...     "backend/models/swin_model.onnx",
        ...     "backend/models/swin_model_quantized.onnx",
        ...     quantization_type=QuantType.QInt8
        ... )
        >>> print(f"Reduced size from {result['original_size_mb']:.1f}MB "
        ...       f"to {result['quantized_size_mb']:.1f}MB")
    """
    # Validate input
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Use the provided quantization type directly
    q_type = quantization_type
    quantization_type_str = "qint8" if q_type == QuantType.QInt8 else "quint8"
    
    # Get original model size
    original_size = os.path.getsize(model_path)
    original_size_mb = original_size / (1024 * 1024)
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        logger.info(f"Quantizing model: {model_path}")
        logger.info(f"Quantization type: {quantization_type_str}")
        
        # Perform quantization using ONNX Runtime
        # Note: quantize_dynamic doesn't take optimize_model parameter
        quantize_dynamic(
            model_input=model_path,
            model_output=output_path,
            per_channel=False,
            reduce_range=False,
            weight_type=q_type,
        )
        
        # Verify output file exists
        if not os.path.exists(output_path):
            raise RuntimeError(
                f"Quantization completed but output file not found: {output_path}"
            )
        
        # Get quantized model size
        quantized_size = os.path.getsize(output_path)
        quantized_size_mb = quantized_size / (1024 * 1024)
        
        # Calculate reduction
        size_reduction_bytes = original_size - quantized_size
        size_reduction_percent = (size_reduction_bytes / original_size) * 100
        
        logger.info(
            f"Quantization successful: {original_size_mb:.1f}MB → {quantized_size_mb:.1f}MB "
            f"({size_reduction_percent:.1f}% reduction)"
        )
        
        return {
            "original_size_mb": round(original_size_mb, 2),
            "quantized_size_mb": round(quantized_size_mb, 2),
            "size_reduction_percent": round(size_reduction_percent, 2),
            "quantization_type": quantization_type_str,
            "output_path": output_path,
        }
    
    except Exception as e:
        raise RuntimeError(f"Quantization failed: {str(e)}") from e


def validate_quantized_model(model_path: str) -> bool:
    """
    Validate that a quantized ONNX model is valid and loadable.
    
    Performs two types of validation:
    1. ONNX format validation using onnx.checker
    2. ONNX Runtime session loading to verify runtime compatibility
    
    Args:
        model_path: Path to the quantized ONNX model file to validate
    
    Returns:
        True if the model is valid (ONNX format valid AND ONNX Runtime can load it), False otherwise.
    
    Raises:
        FileNotFoundError: If the model file doesn't exist
    
    Examples:
        >>> is_valid = validate_quantized_model("backend/models/swin_model_quantized.onnx")
        >>> if is_valid:
        ...     print("Model is valid and ready for use")
        >>> else:
        ...     print("Model validation failed")
    """
    validation_errors = []
    
    # Check file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    onnx_format_valid = False
    runtime_loadable = False
    
    # Step 1: ONNX format validation
    try:
        logger.info(f"Validating ONNX format: {model_path}")
        onnx_model = onnx.load(model_path)
        onnx.checker.check_model(onnx_model)
        onnx_format_valid = True
        logger.info("✓ ONNX format is valid")
    except Exception as e:
        error_msg = f"ONNX format validation failed: {str(e)}"
        logger.warning(error_msg)
        validation_errors.append(error_msg)
    
    # Step 2: ONNX Runtime loading
    try:
        logger.info(f"Testing ONNX Runtime loading: {model_path}")
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        runtime_loadable = True
        logger.info("✓ ONNX Runtime loaded successfully")
        
        # Log session details
        providers = session.get_providers()
        logger.info(f"  Execution providers: {providers}")
        
    except Exception as e:
        error_msg = f"ONNX Runtime loading failed: {str(e)}"
        logger.warning(error_msg)
        validation_errors.append(error_msg)
    
    all_valid = onnx_format_valid and runtime_loadable
    
    return all_valid


def get_model_info(model_path: str) -> Dict[str, Any]:
    """
    Extract metadata from an ONNX model.
    
    Retrieves information about model inputs, outputs, and file size.
    Useful for understanding model specifications before inference.
    
    Args:
        model_path: Path to the ONNX model file
    
    Returns:
        Dictionary containing:
            - "model_path": Path to the model file
            - "size_mb": Model file size in MB
            - "inputs": List of input specifications, each with:
                - "name": Input name
                - "shape": Input shape (may contain dynamic dimensions)
                - "dtype": Data type name
            - "outputs": List of output specifications, each with:
                - "name": Output name
                - "shape": Output shape (may contain dynamic dimensions)
                - "dtype": Data type name
    
    Raises:
        FileNotFoundError: If the model file doesn't exist
        RuntimeError: If model info extraction fails
    
    Examples:
        >>> info = get_model_info("backend/models/swin_model_quantized.onnx")
        >>> print(f"Model size: {info['size_mb']:.1f} MB")
        >>> for input_spec in info['inputs']:
        ...     print(f"Input: {input_spec['name']} {input_spec['shape']}")
        >>> for output_spec in info['outputs']:
        ...     print(f"Output: {output_spec['name']} {output_spec['shape']}")
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        # Load model
        logger.info(f"Loading model info from: {model_path}")
        onnx_model = onnx.load(model_path)
        
        # Get file size
        size_bytes = os.path.getsize(model_path)
        size_mb = size_bytes / (1024 * 1024)
        
        # Helper function to get dtype name from ONNX type code
        def get_dtype_name(dtype_code):
            """Map ONNX data type code to string name."""
            return DTYPE_MAP.get(dtype_code, f"type_{dtype_code}")
        
        # Extract inputs
        inputs = []
        for input_node in onnx_model.graph.input:
            shape = []
            if hasattr(input_node.type, 'tensor_type'):
                for dim in input_node.type.tensor_type.shape.dim:
                    if dim.dim_value > 0:
                        shape.append(dim.dim_value)
                    else:
                        shape.append("dynamic")
            
            # Get data type
            dtype_name = "unknown"
            if hasattr(input_node.type, 'tensor_type'):
                dtype_code = input_node.type.tensor_type.elem_type
                dtype_name = get_dtype_name(dtype_code)
            
            inputs.append({
                "name": input_node.name,
                "shape": shape,
                "dtype": dtype_name,
            })
        
        # Extract outputs
        outputs = []
        for output_node in onnx_model.graph.output:
            shape = []
            if hasattr(output_node.type, 'tensor_type'):
                for dim in output_node.type.tensor_type.shape.dim:
                    if dim.dim_value > 0:
                        shape.append(dim.dim_value)
                    else:
                        shape.append("dynamic")
            
            # Get data type
            dtype_name = "unknown"
            if hasattr(output_node.type, 'tensor_type'):
                dtype_code = output_node.type.tensor_type.elem_type
                dtype_name = get_dtype_name(dtype_code)
            
            outputs.append({
                "name": output_node.name,
                "shape": shape,
                "dtype": dtype_name,
            })
        
        logger.info(f"✓ Model info extracted successfully")
        logger.info(f"  Size: {size_mb:.2f} MB")
        logger.info(f"  Inputs: {len(inputs)}")
        logger.info(f"  Outputs: {len(outputs)}")
        
        return {
            "model_path": model_path,
            "size_mb": round(size_mb, 2),
            "inputs": inputs,
            "outputs": outputs,
        }
    
    except Exception as e:
        raise RuntimeError(f"Failed to extract model info: {str(e)}") from e

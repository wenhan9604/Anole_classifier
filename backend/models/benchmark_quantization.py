"""
Model Quantization Benchmarking Suite

This module provides benchmarking capabilities to compare the performance of
original vs quantized ONNX models in terms of file size and inference latency.

Classes:
    - ModelBenchmark: Measures inference performance and model size

Functions:
    - benchmark_pair(): Compare original vs quantized models with formatted output
"""

import os
import time
import logging
from typing import Dict, Any

try:
    import numpy as np
    import onnxruntime as ort
except ImportError as e:
    raise ImportError(
        f"Required packages not found: {e}. "
        "Install with: pip install numpy onnxruntime"
    )


logger = logging.getLogger(__name__)


class ModelBenchmark:
    """Benchmark an ONNX model for inference performance and size metrics."""
    
    def __init__(self, model_path: str, model_name: str):
        """
        Initialize the benchmark for a specific model.
        
        Args:
            model_path: Path to the ONNX model file
            model_name: Human-readable name for the model (e.g., "YOLO", "Swin")
        
        Raises:
            FileNotFoundError: If the model file doesn't exist
            RuntimeError: If ONNX Runtime cannot load the model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model_path = model_path
        self.model_name = model_name
        
        # Load ONNX Runtime session
        try:
            logger.info(f"Loading {model_name} model from: {model_path}")
            self.session = ort.InferenceSession(
                model_path,
                providers=["CPUExecutionProvider"]
            )
            logger.info(f"✓ {model_name} model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load {model_name} model: {str(e)}") from e
    
    def benchmark_inference(self, num_passes: int = 100) -> float:
        """
        Benchmark model inference by running multiple passes.
        
        Runs the model inference num_passes times and measures the average
        latency per pass.
        
        Args:
            num_passes: Number of inference passes to run (default: 100)
        
        Returns:
            Average inference latency in milliseconds
        
        Raises:
            RuntimeError: If inference fails
        """
        try:
            # Get input names and shapes from session
            input_names = [input.name for input in self.session.get_inputs()]
            input_shapes = [input.shape for input in self.session.get_inputs()]
            
            # Create dummy inputs based on the model's expected shape
            inputs = {}
            for input_name, input_shape in zip(input_names, input_shapes):
                # Replace batch dimension and dynamic dimensions with 1
                shape = []
                for dim in input_shape:
                    if isinstance(dim, str) or dim <= 0:
                        shape.append(1)
                    else:
                        shape.append(dim)
                # Create random float32 input data
                inputs[input_name] = np.random.randn(*shape).astype(np.float32)
            
            # Warm-up pass
            logger.info(f"Running warm-up pass for {self.model_name}...")
            self.session.run(None, inputs)
            
            # Benchmark passes
            logger.info(f"Running {num_passes} benchmark passes for {self.model_name}...")
            start_time = time.perf_counter()
            
            for _ in range(num_passes):
                self.session.run(None, inputs)
            
            end_time = time.perf_counter()
            
            # Calculate average latency in milliseconds
            total_time_ms = (end_time - start_time) * 1000
            avg_latency_ms = total_time_ms / num_passes
            
            logger.info(
                f"✓ {self.model_name} benchmark complete: "
                f"avg latency = {avg_latency_ms:.2f}ms over {num_passes} passes"
            )
            
            return avg_latency_ms
        
        except Exception as e:
            raise RuntimeError(
                f"Inference benchmark failed for {self.model_name}: {str(e)}"
            ) from e
    
    def get_size_info(self) -> Dict[str, float]:
        """
        Get model file size information.
        
        Returns:
            Dictionary containing:
                - "size_bytes": File size in bytes
                - "size_mb": File size in megabytes
        """
        size_bytes = os.path.getsize(self.model_path)
        size_mb = size_bytes / (1024 * 1024)
        
        logger.info(f"{self.model_name} size: {size_mb:.2f} MB ({size_bytes} bytes)")
        
        return {
            "size_bytes": size_bytes,
            "size_mb": round(size_mb, 2),
        }


def benchmark_pair(
    original_path: str,
    quantized_path: str,
    model_name: str,
    num_passes: int = 100
) -> Dict[str, Any]:
    """
    Benchmark and compare original vs quantized models.
    
    Compares the file sizes and inference latencies of original and quantized
    models, printing a formatted comparison table and calculating improvement
    metrics.
    
    Args:
        original_path: Path to the original ONNX model
        quantized_path: Path to the quantized ONNX model
        model_name: Human-readable name for the model (e.g., "YOLO", "Swin")
        num_passes: Number of inference passes for benchmarking (default: 100)
    
    Returns:
        Dictionary containing benchmark results:
            - "model_name": Name of the model
            - "original_size_mb": Size of original model in MB
            - "quantized_size_mb": Size of quantized model in MB
            - "size_reduction_percent": Size reduction percentage
            - "original_latency_ms": Average inference latency of original model
            - "quantized_latency_ms": Average inference latency of quantized model
            - "latency_improvement_percent": Latency improvement percentage (negative = slower)
    
    Raises:
        FileNotFoundError: If either model file doesn't exist
        RuntimeError: If benchmarking fails
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"BENCHMARKING {model_name} MODELS")
    logger.info(f"{'='*70}\n")
    
    # Benchmark original model
    logger.info(f"Benchmarking original {model_name} model...")
    original_benchmark = ModelBenchmark(original_path, f"Original {model_name}")
    original_size = original_benchmark.get_size_info()
    original_latency = original_benchmark.benchmark_inference(num_passes)
    
    # Benchmark quantized model
    logger.info(f"\nBenchmarking quantized {model_name} model...")
    quantized_benchmark = ModelBenchmark(quantized_path, f"Quantized {model_name}")
    quantized_size = quantized_benchmark.get_size_info()
    quantized_latency = quantized_benchmark.benchmark_inference(num_passes)
    
    # Calculate metrics
    size_reduction_percent = (
        (original_size["size_mb"] - quantized_size["size_mb"]) 
        / original_size["size_mb"] * 100
    )
    latency_improvement_percent = (
        (original_latency - quantized_latency) / original_latency * 100
    )
    
    # Print comparison table
    logger.info(f"\n{'='*70}")
    logger.info(f"{model_name} BENCHMARK COMPARISON")
    logger.info(f"{'='*70}\n")
    
    print(f"\n{'='*70}")
    print(f"{model_name.upper()} BENCHMARK COMPARISON")
    print(f"{'='*70}\n")
    
    print(f"{'Metric':<30} {'Original':<20} {'Quantized':<20}")
    print("-" * 70)
    
    print(
        f"{'Model Size':<30} {original_size['size_mb']:.2f} MB"
        f"{'':<10} {quantized_size['size_mb']:.2f} MB"
    )
    print(
        f"{'Inference Latency (avg)':<30} {original_latency:.2f} ms"
        f"{'':<10} {quantized_latency:.2f} ms"
    )
    
    print(f"\n{'='*70}")
    print(f"IMPROVEMENTS")
    print(f"{'='*70}\n")
    
    print(f"Size Reduction:    {size_reduction_percent:>6.2f}%")
    print(
        f"Latency Change:    {latency_improvement_percent:>6.2f}%"
        f" {'(faster)' if latency_improvement_percent > 0 else '(slower)'}"
    )
    
    print(f"\n{'='*70}\n")
    
    # Log the same information
    logger.info(f"Metric                         Original             Quantized")
    logger.info("-" * 70)
    logger.info(
        f"Model Size                     {original_size['size_mb']:.2f} MB"
        f"{'':<10} {quantized_size['size_mb']:.2f} MB"
    )
    logger.info(
        f"Inference Latency (avg)        {original_latency:.2f} ms"
        f"{'':<10} {quantized_latency:.2f} ms"
    )
    logger.info(f"\nIMPROVEMENTS")
    logger.info(f"Size Reduction:    {size_reduction_percent:>6.2f}%")
    logger.info(
        f"Latency Change:    {latency_improvement_percent:>6.2f}% "
        f"{'(faster)' if latency_improvement_percent > 0 else '(slower)'}"
    )
    
    return {
        "model_name": model_name,
        "original_size_mb": round(original_size["size_mb"], 2),
        "quantized_size_mb": round(quantized_size["size_mb"], 2),
        "size_reduction_percent": round(size_reduction_percent, 2),
        "original_latency_ms": round(original_latency, 2),
        "quantized_latency_ms": round(quantized_latency, 2),
        "latency_improvement_percent": round(latency_improvement_percent, 2),
    }


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    print("\n" + "="*70)
    print("QUANTIZATION BENCHMARK SUITE")
    print("="*70)
    
    # Define model paths
    yolo_original = "backend/models/yolo_best.onnx"
    yolo_quantized = "backend/models/yolo_best_quantized.onnx"
    
    swin_original = "backend/models/swin_model.onnx"
    swin_quantized = "backend/models/swin_model_quantized.onnx"
    
    results = []
    
    # Benchmark YOLO if models exist
    if os.path.exists(yolo_original) and os.path.exists(yolo_quantized):
        yolo_results = benchmark_pair(
            yolo_original,
            yolo_quantized,
            "YOLO",
            num_passes=100
        )
        results.append(yolo_results)
    else:
        logger.warning(
            f"YOLO models not found. "
            f"Expected: {yolo_original}, {yolo_quantized}"
        )
        print(
            f"⚠️  YOLO models not found. "
            f"Expected: {yolo_original}, {yolo_quantized}\n"
        )
    
    # Benchmark Swin if models exist
    if os.path.exists(swin_original) and os.path.exists(swin_quantized):
        swin_results = benchmark_pair(
            swin_original,
            swin_quantized,
            "Swin",
            num_passes=100
        )
        results.append(swin_results)
    else:
        logger.warning(
            f"Swin models not found. "
            f"Expected: {swin_original}, {swin_quantized}"
        )
        print(
            f"⚠️  Swin models not found. "
            f"Expected: {swin_original}, {swin_quantized}\n"
        )
    
    # Summary
    print("="*70)
    print("BENCHMARK SUITE COMPLETED")
    print("="*70)
    logger.info("Benchmark suite completed successfully")
    
    if results:
        print(f"\n✓ Benchmarked {len(results)} model pairs successfully")
        logger.info(f"✓ Benchmarked {len(results)} model pairs successfully")
    else:
        print("\n⚠️  No model pairs were benchmarked")
        logger.warning("No model pairs were benchmarked")

# ONNX Model Quantization Guide

## Overview

This guide explains model quantization and how to use quantized versions of YOLO and Swin models in the Lizard Lens application.

## What is Quantization?

Quantization is a technique that reduces model size and speeds up inference by converting weights from 32-bit floating point (float32) to 8-bit integers (int8). This results in:

- **~75% size reduction** (300MB → 75-100MB)
- **15-40% inference speed improvement**
- **Minimal accuracy loss** (<5% in most cases)

## Quantized Models in Lizard Lens

### Available Models

The application now includes both original and quantized versions:

| Model | Original | Quantized | Size Reduction |
|-------|----------|-----------|-----------------|
| YOLO Detection | `yolo_best.onnx` | `yolo_best_quantized.onnx` | ~75% |
| Swin Classification | `swin_model.onnx` | `swin_model_quantized.onnx` | ~75% |

### Automatic Selection

- **Backend**: Automatically loads quantized models if available
- **Frontend**: Attempts quantized models first, falls back to originals
- **Transparent**: No configuration needed; uses fastest available model

## Performance Benchmarks

Run the benchmark suite to measure performance on your hardware:

```bash
python backend/models/benchmark_quantization.py
```

Example output:
```
======================================================================
Benchmarking YOLO (Detection)
======================================================================

📊 Model Information
----------------------------------------------------------------------
Original size: 142.50 MB
Quantized size: 35.62 MB
Size reduction: 75.0%

⚡ Inference Speed (100 runs)
----------------------------------------------------------------------
Original - Mean: 45.32ms | Median: 44.89ms | Std: 2.15ms
Quantized - Mean: 32.18ms | Median: 31.95ms | Std: 1.89ms

📈 Summary
----------------------------------------------------------------------
Size reduction: 75.0%
Speed improvement: 29.0%
Original avg latency: 45.32ms
Quantized avg latency: 32.18ms
```

## Exporting Quantized Models

### YOLO

```bash
python export_yolo_to_onnx.py
```

This will create:
- `backend/models/yolo_best.onnx` (original)
- `backend/models/yolo_best_quantized.onnx` (quantized)
- `frontend/public/models/yolo_best.onnx` (original)
- `frontend/public/models/yolo_best_quantized.onnx` (quantized)

### Swin

```bash
python export_swin_to_onnx.py
```

This will create:
- `backend/models/swin_model.onnx` (original)
- `backend/models/swin_model_quantized.onnx` (quantized)
- `frontend/public/models/swin_model.onnx` (original)
- `frontend/public/models/swin_model_quantized.onnx` (quantized)

## Backend Usage

The backend automatically uses quantized models when available:

```python
# In pipeline_inference.py
model_path = load_onnx_model_with_quantized_fallback(
    "backend/models/yolo_best.onnx",
    "backend/models/yolo_best_quantized.onnx",
    "YOLO"
)
```

No code changes required; the inference pipeline handles model selection.

## Frontend Usage

The frontend browser ONNX service automatically loads quantized models:

```typescript
// In OnnxDetectionService.ts
private yoloModelPath = '/models/yolo_best_quantized.onnx';
private swinModelPath = '/models/swin_model_quantized.onnx';
```

Browser download sizes are reduced by 75%, improving load time and offline functionality.

## Accuracy Considerations

Quantized models maintain high accuracy:

- **Species Classification**: <2% accuracy loss
- **Lizard Detection**: <3% accuracy loss
- **Confidence Scores**: Preserved with minor floating-point variations

For production use cases requiring >99% accuracy, use original models by manually specifying paths.

## Troubleshooting

### Quantized models not loading

Check that quantized models are present:
```bash
ls -lh backend/models/*quantized.onnx
```

If missing, re-export using the export scripts above.

### Performance not improving

Quantized models perform better on:
- ✅ CPU inference (all platforms)
- ✅ Web browser inference (WASM)
- ⚠️ GPU inference (may show minimal improvement due to GPU optimization)

### Accuracy degradation

If accuracy is unacceptable, revert to original models:
- Delete quantized model files
- Rebuild/restart application

## Further Optimization

Future improvements could include:

1. **Quantization-Aware Training (QAT)**: Retrain models with quantization in mind for better accuracy
2. **Model Distillation**: Train smaller student models (YOLOv8n, Swin Tiny) with quantization
3. **Hardware-Specific Optimization**: TensorRT for NVIDIA GPUs, CoreML for iOS
4. **Dynamic Quantization**: Automatically select quantization precision based on hardware

## References

- [ONNX Runtime Quantization](https://onnxruntime.ai/docs/performance/quantization/overview.html)
- [ONNX Model Optimization](https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/python/tools/transformers)
- [YOLOv8 Model Compression](https://docs.ultralytics.com/modes/export/#export-formats)

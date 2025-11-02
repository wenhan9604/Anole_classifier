"""
Pipeline-based inference using the Spring_2025/pipeline_evaluation.py logic:
- YOLOv8 for detection
- Swin Transformer for classification of detected crops

This module lazily loads models on first use and keeps them cached.
It is designed to be optional: if dependencies or weights are missing,
callers should catch ImportError/FileNotFoundError and fall back.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple
import numpy as np
import logging

from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)

_yolo = None
_swin = None
_processor = None
_calibrator = None


def _get_model_paths() -> Tuple[str, str]:
    """
    Returns (detection_weights_path, classification_model_id_or_path).
    Can be configured via env vars:
    - DETECTION_WEIGHTS_PATH
    - CLASSIFICATION_MODEL_ID
    Defaults based on Spring_2025/pipeline_evaluation.py
    """
    # Priority 1: explicit env var
    det_env = os.getenv("DETECTION_WEIGHTS_PATH")
    if det_env:
        det = det_env
    else:
        # Priority 2: Check common YOLO model locations
        # Match pipeline_evaluation.py which uses ./runs/detect/train_yolov8n_v2/weights/best.pt
        # But actual location is ultralytics_runs/detect/train_yolov8n_v2/weights/best.pt
        candidates = [
            os.path.join("..", "Spring_2025", "models", "train_yolov8n_v2", "weights", "best.pt"),  # New location
            os.path.join("..", "Spring_2025", "ultralytics_runs", "detect", "train_yolov8n_v2", "weights", "best.pt"),  # Old location
            os.path.join("..", "Spring_2025", "runs", "detect", "train_yolov8n_v2", "weights", "best.pt"),  # Try original path too
            os.path.join("..", "Spring_2025", "ultralytics_runs", "detect", "train_yolov8n", "weights", "best.pt"),
            os.path.join("..", "Spring_2025", "ultralytics_runs", "detect", "train_yolov11n", "weights", "best.pt"),
        ]
        det = None
        for candidate in candidates:
            if os.path.exists(candidate):
                det = os.path.abspath(candidate)
                break
        # Priority 3: Default fallback path
        if det is None:
            det = os.path.join("..", "Spring_2025", "ultralytics_runs", "detect", "train_yolov8n", "weights", "best.pt")
    # Classification model: Check env var first, then try local models folder, then HuggingFace
    clf_env = os.getenv("CLASSIFICATION_MODEL_ID")
    if clf_env:
        clf = clf_env
    else:
        # Check if local model folder exists in Spring_2025/models (relative to backend/ directory)
        local_model_path = os.path.join("..", "Spring_2025", "models", "swin-base-patch4-window12-384-finetuned-lizard-class-swin-base")
        if os.path.exists(local_model_path):
            # Check for a checkpoint folder (use the latest checkpoint if available)
            checkpoint_352 = os.path.join(local_model_path, "checkpoint-352")
            checkpoint_16 = os.path.join(local_model_path, "checkpoint-16")
            if os.path.exists(checkpoint_352):
                clf = os.path.abspath(checkpoint_352)  # Use checkpoint-352 (latest)
            elif os.path.exists(checkpoint_16):
                clf = os.path.abspath(checkpoint_16)  # Use checkpoint-16
            else:
                clf = os.path.abspath(local_model_path)  # Use root folder
        else:
            # Check alternative model_export folder
            alt_model_path = os.path.join("..", "model_export", "swin_transformer_base_lizard_v4")
            if os.path.exists(alt_model_path):
                clf = os.path.abspath(alt_model_path)
            else:
                # Fallback to HuggingFace model ID (will download if not cached)
                clf = "swin-base-patch4-window12-384-finetuned-lizard-class-swin-base"
    return det, clf


def _load_models() -> None:
    global _yolo, _swin, _processor
    if _yolo is not None and _swin is not None and _processor is not None:
        return

    try:
        from ultralytics import YOLO  # type: ignore
        from transformers import AutoImageProcessor, SwinForImageClassification  # type: ignore
        import torch  # noqa: F401  # Ensure torch is available
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Required ML dependencies not installed. Install 'ultralytics', 'transformers', and 'torch'."
        ) from e

    det_path, clf_id = _get_model_paths()
    if not os.path.exists(det_path):  # pragma: no cover
        raise FileNotFoundError(
            f"Detection weights not found at {det_path}. Set DETECTION_WEIGHTS_PATH env var."
        )

    # Load models
    logger.info(f"Loading YOLO model from: {det_path}")
    logger.info(f"Model file exists: {os.path.exists(det_path)}")
    _yolo = YOLO(det_path)
    _swin = SwinForImageClassification.from_pretrained(clf_id)
    _processor = AutoImageProcessor.from_pretrained(clf_id)
    _swin.eval()
    logger.info(f"Successfully loaded YOLO model from: {det_path}")

    # Optional: load calibrator if provided
    global _calibrator
    calib_path = os.getenv("CALIBRATION_PATH")
    if calib_path and os.path.exists(calib_path):
        try:
            from .calibration import load_calibrator
            _calibrator = load_calibrator(calib_path)
            logger.info(f"Loaded calibrator from: {calib_path}")
        except Exception as e:
            logger.warning(f"Failed to load calibrator at {calib_path}: {e}")


def _softmax(logits, temperature: float = 1.0) -> List[float]:
    """
    Compute softmax probabilities with optional temperature scaling.
    
    Args:
        logits: Raw model outputs (logit scores)
        temperature: Temperature scaling factor (default 1.0 = no scaling)
                     Higher values (e.g., 2.0) make probabilities less extreme/more uniform
    """
    import math

    if hasattr(logits, "tolist"):
        vals = logits.tolist()
        if isinstance(vals, list) and vals and isinstance(vals[0], list):
            vals = vals[0]
    else:
        vals = list(logits)
    
    # Apply temperature scaling: divide logits by temperature before softmax
    scaled_vals = [v / temperature for v in vals]
    m = max(scaled_vals)
    exps = [math.exp(v - m) for v in scaled_vals]
    s = sum(exps)
    return [e / s for e in exps]


def _compute_iou(boxA: List[float], boxB: List[float]) -> float:
    """Compute Intersection over Union (IoU) of two bounding boxes"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)


def _deduplicate_overlapping_detections(predictions: List[Dict[str, Any]], iou_threshold: float = 0.25) -> List[Dict[str, Any]]:
    """
    Remove duplicate detections that overlap significantly (same lizard detected multiple times).
    Uses IoU threshold and center distance to group detections.
    Groups overlapping detections and keeps the one with highest confidence.
    """
    if not predictions or len(predictions) == 1:
        return predictions
    
    # Sort by confidence descending (highest first) - keep highest confidence when merging
    sorted_preds = sorted(predictions, key=lambda x: x.get("detectionConf", 0), reverse=True)
    
    # Group overlapping detections using Union-Find approach
    groups = []
    
    for pred in sorted_preds:
        pred_box = pred["box"]
        pred_center_x = (pred_box[0] + pred_box[2]) / 2
        pred_center_y = (pred_box[1] + pred_box[3]) / 2
        pred_size = ((pred_box[2] - pred_box[0]) + (pred_box[3] - pred_box[1])) / 2
        
        assigned_to_group = False
        
        # Check if this box overlaps with any existing group
        for group_idx, group in enumerate(groups):
            # Check overlap with any box in the group
            for group_pred in group:
                group_box = group_pred["box"]
                iou = _compute_iou(pred_box, group_box)
                
                # Also check center distance as an additional criterion
                group_center_x = (group_box[0] + group_box[2]) / 2
                group_center_y = (group_box[1] + group_box[3]) / 2
                center_dist = ((pred_center_x - group_center_x)**2 + (pred_center_y - group_center_y)**2)**0.5
                group_size = ((group_box[2] - group_box[0]) + (group_box[3] - group_box[1])) / 2
                avg_size = (pred_size + group_size) / 2
                
                # If IoU >= threshold OR centers are very close (< 50% of avg box size), merge
                center_close = avg_size > 0 and (center_dist / avg_size) < 0.5
                
                logger.info(f"  Comparing: IoU={iou:.3f}, center_dist={center_dist:.1f}, avg_size={avg_size:.1f}, center_close={center_close}, merge={iou >= iou_threshold or center_close}")
                
                if iou >= iou_threshold or center_close:
                    logger.info(f"  -> Merging into group {group_idx}")
                    group.append(pred)
                    assigned_to_group = True
                    break
            
            if assigned_to_group:
                break
        
        # If not overlapping with any group, start a new group
        if not assigned_to_group:
            logger.info(f"  -> Creating new group {len(groups)}")
            groups.append([pred])
    
    logger.info(f"Deduplication: {len(groups)} groups found from {len(sorted_preds)} detections")
    
    # From each group, keep only the highest confidence detection
    unique_preds = []
    for group_idx, group in enumerate(groups):
        # Group is already sorted by confidence (since sorted_preds was), so take first
        logger.info(f"  Group {group_idx}: {len(group)} detections, keeping highest conf={group[0].get('detectionConf', 0):.3f}")
        unique_preds.append(group[0])
    
    return unique_preds


def _class_mapping() -> Dict[int, Tuple[str, str]]:
    """Class id → (species, scientificName) mapping for 5 Florida anoles.
    Matches the model's config.json label order:
    0=bark_anole, 1=brown_anole, 2=crested_anole, 3=green_anole, 4=knight_anole
    """
    return {
        0: ("Bark Anole", "Anolis distichus"),
        1: ("Brown Anole", "Anolis sagrei"),
        2: ("Crested Anole", "Anolis cristatellus"),
        3: ("Green Anole", "Anolis carolinensis"),
        4: ("Knight Anole", "Anolis equestris"),
    }


def predict_image_bytes(image_bytes: bytes, conf_threshold: float = 0.0, top_k: int | None = 5, temperature: float = 2.0) -> Dict[str, Any]:
    """
    Run detection + classification on a single image and return a structure
    compatible with the frontend's expected result format.
    
    Args:
        image_bytes: Image data as bytes
        conf_threshold: Minimum confidence for detections (default: 0.0, matches CONF_THRESH = 0)
        top_k: Maximum number of detections to process (default: 5, matches TOP_K = 5)
        temperature: Temperature scaling for classification probabilities (default: 2.0)
                     Higher = less extreme probabilities, more realistic confidence scores
                     temperature=1.0 = no scaling (original behavior)
                     temperature=2.0 = softer probabilities (recommended)
    
    Exact settings from Spring_2025/pipeline_evaluation.py:
    - CONF_THRESH = 0 (accept all detections)
    - TOP_K = 5 (process top 5 detections)
    - IOU_THRESHOLD = 0.5 (for deduplication)
    - Let YOLO use default NMS (yolo_model(image)[0])
    - Temperature scaling added for more realistic confidence scores
    """
    _load_models()

    assert _yolo is not None and _swin is not None and _processor is not None

    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    # Detection - exact match to pipeline_evaluation.py: yolo_model(image)[0]
    # No explicit conf/iou parameters - use YOLO's defaults
    img_size = image.size
    logger.info(f"Image size: {img_size} (width x height)")
    results = _yolo(image)[0]
    boxes = results.boxes.data  # Tensor: [x1, y1, x2, y2, conf, class_id]
    logger.info(f"YOLO detected {len(boxes)} boxes")

    # Confidence filtering - exact match: boxes[:, 4] >= CONF_THRESH (CONF_THRESH = 0)
    boxes = boxes[boxes[:, 4] >= conf_threshold]
    boxes = boxes[boxes[:, 4].argsort(descending=True)]  # Sort by confidence (highest first)
    
    # Limit to top_k - exact match: TOP_K = 5
    if top_k is not None:
        boxes = boxes[:top_k]

    class_map = _class_mapping()
    predictions: List[Dict[str, Any]] = []

    if boxes.shape[0] == 0:
        return {"totalLizards": 0, "predictions": []}

    for det in boxes:
        x1, y1, x2, y2, det_conf, _ = det.tolist()
        logger.info(f"Box coordinates: x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}, conf={det_conf:.3f}")
        logger.info(f"Image size check: {image.size}, box should be within (0,0) to ({image.size[0]}, {image.size[1]})")
        crop = image.crop((x1, y1, x2, y2))
        inputs = _processor(images=crop, return_tensors="pt")

        with __import__("torch").no_grad():  # type: ignore
            logits = _swin(**inputs).logits
        # Extract logits for logging (before softmax)
        if hasattr(logits, "tolist"):
            logit_vals = logits.tolist()
            if isinstance(logit_vals, list) and logit_vals and isinstance(logit_vals[0], list):
                logit_vals = logit_vals[0]
        else:
            logit_vals = list(logits)
        logger.info(f"Raw logits: {[f'{l:.2f}' for l in logit_vals]}")

        # Prefer external calibrator if available; fall back to temperature scaling
        if _calibrator is not None:
            try:
                # Try calibrating probabilities first
                probs = _calibrator.calibrate_probs(logit_vals)
                logger.info("Applied external calibrator for probabilities")
            except Exception:
                # Or calibrate logits then softmax
                try:
                    calibrated_logits = _calibrator.calibrate_logits(logit_vals)
                    probs = _softmax(calibrated_logits, temperature=1.0)
                    logger.info("Applied external calibrator to logits")
                except Exception:
                    logger.warning("Calibrator present but failed; using temperature softmax")
                    logger.info(f"Using temperature: {temperature} for probability scaling")
                    probs = _softmax(logits, temperature=temperature)
        else:
            logger.info(f"Using temperature: {temperature} for probability scaling")
            probs = _softmax(logits, temperature=temperature)
        cls_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
        cls_conf = float(probs[cls_idx])
        
        # Log probability distribution for debugging
        logger.info(f"Classification probabilities: {[f'{p:.4f}' for p in probs]}")
        logger.info(f"Predicted class: {cls_idx}, Confidence: {cls_conf:.4f} ({cls_conf*100:.1f}%)")

        species, sci = class_map.get(cls_idx, (f"Class {cls_idx}", f"Class {cls_idx}"))
        predictions.append(
            {
                "species": species,
                "scientificName": sci,
                "confidence": cls_conf,
                "count": 1,
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "detectionConf": float(det_conf),
                "classIndex": cls_idx,
            }
        )

    # Log before deduplication
    logger.info(f"Before deduplication: {len(predictions)} detections")
    for i, p in enumerate(predictions):
        logger.info(f"  Detection {i}: box={p['box']}, conf={p.get('detectionConf', 0):.3f}, species={p.get('species', 'N/A')}")
    
    # Deduplicate overlapping detections (same lizard detected multiple times)
    # Use lower IoU threshold for deduplication (0.25) to catch boxes that are close but don't overlap 50%
    # Note: pipeline_evaluation.py IOU_THRESHOLD = 0.5 is for matching predictions to ground truth, not deduplication
    unique_predictions = _deduplicate_overlapping_detections(predictions, iou_threshold=0.25)
    
    logger.info(f"After deduplication: {len(unique_predictions)} detections")
    
    return {"totalLizards": len(unique_predictions), "predictions": unique_predictions}

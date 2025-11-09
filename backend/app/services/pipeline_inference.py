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

from io import BytesIO
from PIL import Image

_yolo = None
_swin = None
_processor = None


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
        # Priority 2: absolute path based on user's specified location
        abs_candidate = \
            "/Users/niralverma/Anole_classifier/Spring_2025/runs/detect/train_yolov8n_v2/weights/best.pt"
        if os.path.exists(abs_candidate):
            det = abs_candidate
        else:
            # Priority 3: repo-relative default mirroring Spring_2025 script
            det = os.path.join(
                "Spring_2025", "runs", "detect", "train_yolov8n_v2", "weights", "best.pt"
            )
    clf = os.getenv(
        "CLASSIFICATION_MODEL_ID",
        "swin-base-patch4-window12-384-finetuned-lizard-class-swin-base",
    )
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
    _yolo = YOLO(det_path)
    _swin = SwinForImageClassification.from_pretrained(clf_id)
    _processor = AutoImageProcessor.from_pretrained(clf_id)
    _swin.eval()


def _get_yolo_infer_kwargs() -> Dict[str, Any]:
    """Build YOLO inference kwargs with configurable NMS IoU.

    Env vars:
    - YOLO_IOU: float (default 0.85) — higher keeps more overlapping boxes
    - YOLO_CONF: float (optional) — detector confidence threshold
    - YOLO_MAX_DET: int (optional) — maximum detections
    """
    # IoU default tuned to keep nearby boxes
    iou_str = os.getenv("YOLO_IOU", "0.85")
    try:
        iou = float(iou_str)
    except Exception:
        iou = 0.85

    kwargs: Dict[str, Any] = {"iou": iou}

    conf_str = os.getenv("YOLO_CONF")
    if conf_str:
        try:
            kwargs["conf"] = float(conf_str)
        except Exception:
            pass

    max_det_str = os.getenv("YOLO_MAX_DET")
    if max_det_str:
        try:
            kwargs["max_det"] = int(max_det_str)
        except Exception:
            pass

    return kwargs


def _softmax(logits) -> List[float]:
    import math

    if hasattr(logits, "tolist"):
        vals = logits.tolist()
        if isinstance(vals, list) and vals and isinstance(vals[0], list):
            vals = vals[0]
    else:
        vals = list(logits)
    m = max(vals)
    exps = [math.exp(v - m) for v in vals]
    s = sum(exps)
    return [e / s for e in exps]


def _class_mapping() -> Dict[int, Tuple[str, str]]:
    """Default class id → (species, scientificName) mapping for 5 Florida anoles.
    Adjust via code or future env if needed.
    """
    return {
        0: ("Green Anole", "Anolis carolinensis"),
        1: ("Brown Anole", "Anolis sagrei"),
        2: ("Crested Anole", "Anolis cristatellus"),
        3: ("Knight Anole", "Anolis equestris"),
        4: ("Bark Anole", "Anolis distichus"),
    }


def predict_image_bytes(image_bytes: bytes, conf_threshold: float = 0.0, top_k: int | None = 5) -> Dict[str, Any]:
    """
    Run detection + classification on a single image and return a structure
    compatible with the frontend's expected result format.
    """
    _load_models()

    assert _yolo is not None and _swin is not None and _processor is not None

    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    # Detection (configurable NMS IoU to keep nearby boxes)
    yolo_kwargs = _get_yolo_infer_kwargs()
    results = _yolo(image, **yolo_kwargs)[0]
    boxes = results.boxes.data  # Tensor: [x1, y1, x2, y2, conf, class_id]

    # Confidence filtering
    boxes = boxes[boxes[:, 4] >= conf_threshold]
    boxes = boxes[boxes[:, 4].argsort(descending=True)]
    if top_k is not None:
        boxes = boxes[:top_k]

    class_map = _class_mapping()
    predictions: List[Dict[str, Any]] = []

    if boxes.shape[0] == 0:
        return {"totalLizards": 0, "predictions": []}

    for det in boxes:
        x1, y1, x2, y2, det_conf, _ = det.tolist()
        crop = image.crop((x1, y1, x2, y2))
        inputs = _processor(images=crop, return_tensors="pt")

        with __import__("torch").no_grad():  # type: ignore
            logits = _swin(**inputs).logits
        probs = _softmax(logits)
        cls_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
        cls_conf = float(probs[cls_idx])

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

    return {"totalLizards": len(predictions), "predictions": predictions}

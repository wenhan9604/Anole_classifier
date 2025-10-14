"""
3-Stage Anole Classification Pipeline
Stage 1: YOLOv8 Detection
Stage 2: Image Cropping
Stage 3: Swin Transformer Classification
"""
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import AutoImageProcessor, SwinForImageClassification
import logging
from typing import List, Dict, Any
import time
import numpy as np

logger = logging.getLogger(__name__)

# Species mapping (class_id -> species info)
SPECIES_MAP = {
    0: {"name": "Bark Anole", "scientific": "Anolis distichus"},
    1: {"name": "Brown Anole", "scientific": "Anolis sagrei"},
    2: {"name": "Crested Anole", "scientific": "Anolis cristatellus"},
    3: {"name": "Green Anole", "scientific": "Anolis carolinensis"},
    4: {"name": "Knight Anole", "scientific": "Anolis equestris"},
}

class AnolePipeline:
    """3-Stage pipeline for anole detection and classification"""
    
    def __init__(
        self,
        yolo_model_path: str,
        swin_model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        device: str = None
    ):
        """
        Initialize the pipeline with models
        
        Args:
            yolo_model_path: Path to fine-tuned YOLOv8 weights
            swin_model_path: Path to fine-tuned Swin Transformer
            conf_threshold: Confidence threshold for detection
            iou_threshold: IoU threshold for NMS
            device: Device to run models on ('cuda', 'cpu', or None for auto)
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Load models
        self._load_models(yolo_model_path, swin_model_path)
        
    def _load_models(self, yolo_path: str, swin_path: str):
        """Load YOLOv8 and Swin Transformer models"""
        try:
            # Load YOLO detection model
            logger.info(f"Loading YOLO model from {yolo_path}")
            self.yolo_model = YOLO(yolo_path)
            
            # Load Swin Transformer classification model
            logger.info(f"Loading Swin Transformer from {swin_path}")
            self.swin_model = SwinForImageClassification.from_pretrained(swin_path)
            self.processor = AutoImageProcessor.from_pretrained(swin_path)
            
            # Move to device and set to eval mode
            self.swin_model.to(self.device)
            self.swin_model.eval()
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run full 3-stage pipeline on an image
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with detection results
        """
        start_time = time.time()
        
        # Stage 1: Detect lizards with YOLO
        detections = self._detect_lizards(image)
        
        if len(detections) == 0:
            return {
                "totalLizards": 0,
                "predictions": [],
                "processingTime": time.time() - start_time
            }
        
        # Stage 2 & 3: Crop and classify each detection
        predictions = []
        for det in detections:
            x1, y1, x2, y2, conf = det
            
            # Crop the detected region
            cropped_image = image.crop((x1, y1, x2, y2))
            
            # Classify the cropped image
            species_id, species_conf = self._classify_species(cropped_image)
            
            # Get species information
            species_info = SPECIES_MAP.get(species_id, {
                "name": f"Unknown (Class {species_id})",
                "scientific": "Unknown"
            })
            
            predictions.append({
                "species": species_info["name"],
                "scientificName": species_info["scientific"],
                "confidence": float(species_conf),
                "count": 1,
                "boundingBox": [float(x1), float(y1), float(x2), float(y2)],
                "detectionConfidence": float(conf)
            })
        
        processing_time = time.time() - start_time
        
        return {
            "totalLizards": len(predictions),
            "predictions": predictions,
            "processingTime": processing_time
        }
    
    def _detect_lizards(self, image: Image.Image) -> List:
        """
        Stage 1: Detect lizards using YOLOv8
        
        Args:
            image: PIL Image
            
        Returns:
            List of detections [x1, y1, x2, y2, confidence]
        """
        results = self.yolo_model(image, conf=self.conf_threshold, iou=self.iou_threshold)[0]
        
        detections = []
        if results.boxes is not None:
            boxes = results.boxes.data.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2, conf, cls = box
                detections.append([x1, y1, x2, y2, conf])
        
        logger.info(f"Detected {len(detections)} lizard(s)")
        return detections
    
    def _classify_species(self, cropped_image: Image.Image) -> tuple:
        """
        Stage 3: Classify species using Swin Transformer
        
        Args:
            cropped_image: PIL Image of cropped lizard
            
        Returns:
            Tuple of (class_id, confidence)
        """
        # Preprocess image
        inputs = self.processor(images=cropped_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.swin_model(**inputs)
            logits = outputs.logits
            
            # Get predicted class and confidence
            probs = torch.nn.functional.softmax(logits, dim=1)
            confidence, predicted_class = torch.max(probs, dim=1)
            
        return predicted_class.item(), confidence.item()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            "device": self.device,
            "yolo_loaded": self.yolo_model is not None,
            "swin_loaded": self.swin_model is not None,
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }


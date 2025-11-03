"""
3-Stage Anole Classification Pipeline
Stage 1: YOLOv8 Detection (supports both PyTorch and ONNX)
Stage 2: Image Cropping
Stage 3: Swin Transformer Classification (supports both PyTorch and ONNX)
"""
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import AutoImageProcessor, SwinForImageClassification
import logging
from typing import List, Dict, Any, Optional, Tuple
import time
import numpy as np

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logging.warning("onnxruntime not available. ONNX support will be disabled.")

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
        device: str = None,
        use_onnx: bool = False
    ):
        """
        Initialize the pipeline with models
        
        Args:
            yolo_model_path: Path to fine-tuned YOLOv8 weights (.pt or .onnx)
            swin_model_path: Path to fine-tuned Swin Transformer (directory or .onnx)
            conf_threshold: Confidence threshold for detection
            iou_threshold: IoU threshold for NMS
            device: Device to run models on ('cuda', 'cpu', or None for auto)
            use_onnx: Whether to use ONNX models instead of PyTorch
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.use_onnx = use_onnx
        self.yolo_model = None
        self.yolo_session = None
        self.swin_model = None
        self.swin_session = None
        self.processor = None
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        logger.info(f"Using ONNX: {self.use_onnx}")
        
        # Load models
        self._load_models(yolo_model_path, swin_model_path)
        
    def _load_models(self, yolo_path: str, swin_path: str):
        """Load YOLOv8 and Swin Transformer models (PyTorch or ONNX)"""
        try:
            if self.use_onnx:
                if not ONNX_AVAILABLE:
                    raise RuntimeError("ONNX support requested but onnxruntime is not installed")
                
                # Load YOLO ONNX model
                logger.info(f"Loading YOLO ONNX model from {yolo_path}")
                providers = self._get_onnx_providers()
                self.yolo_session = ort.InferenceSession(yolo_path, providers=providers)
                logger.info(f"YOLO ONNX model loaded with providers: {self.yolo_session.get_providers()}")
                
                # Load Swin ONNX model
                logger.info(f"Loading Swin ONNX model from {swin_path}")
                self.swin_session = ort.InferenceSession(swin_path, providers=providers)
                logger.info(f"Swin ONNX model loaded with providers: {self.swin_session.get_providers()}")
                
            else:
                # Load PyTorch YOLO detection model
                logger.info(f"Loading YOLO model from {yolo_path}")
                self.yolo_model = YOLO(yolo_path)
                
                # Load PyTorch Swin Transformer classification model
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
    
    def _get_onnx_providers(self) -> List[str]:
        """Get ONNX execution providers based on device"""
        if self.device == "cuda" and torch.cuda.is_available():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        return providers
    
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
        Stage 1: Detect lizards using YOLOv8 (PyTorch or ONNX)
        
        Args:
            image: PIL Image
            
        Returns:
            List of detections [x1, y1, x2, y2, confidence]
        """
        if self.use_onnx:
            detections = self._detect_lizards_onnx(image)
        else:
            results = self.yolo_model(image, conf=self.conf_threshold, iou=self.iou_threshold)[0]
            
            detections = []
            if results.boxes is not None:
                boxes = results.boxes.data.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2, conf, cls = box
                    detections.append([x1, y1, x2, y2, conf])
        
        logger.info(f"Detected {len(detections)} lizard(s)")
        return detections
    
    def _detect_lizards_onnx(self, image: Image.Image) -> List:
        """
        Stage 1: Detect lizards using YOLO ONNX model
        
        Args:
            image: PIL Image
            
        Returns:
            List of detections [x1, y1, x2, y2, confidence]
        """
        # Preprocess image with letterbox
        input_tensor, orig_w, orig_h, pad_x, pad_y, scale = self._preprocess_yolo_image(image)
        
        # Run inference
        input_name = self.yolo_session.get_inputs()[0].name
        outputs = self.yolo_session.run(None, {input_name: input_tensor})
        
        # Check if single-class model and adjust threshold
        output_shape = outputs[0].shape
        is_single_class = output_shape[1] == 5 if output_shape[1] < output_shape[2] else output_shape[2] == 5
        
        # Use higher threshold for single-class models to reduce false positives
        original_threshold = self.conf_threshold
        if is_single_class and self.conf_threshold < 0.5:
            logger.info("Single-class YOLO model detected, raising threshold to 0.5")
            self.conf_threshold = 0.5
        
        # Process outputs
        detections = self._process_yolo_output(
            outputs[0], orig_w, orig_h, pad_x, pad_y, scale
        )
        
        # Restore original threshold
        self.conf_threshold = original_threshold
        
        return detections
    
    def _preprocess_yolo_image(self, image: Image.Image, input_size: int = 640) -> Tuple:
        """
        Preprocess image for YOLO ONNX model with letterbox
        Similar to frontend OnnxDetectionService.preprocessImage
        
        Returns:
            Tuple of (input_tensor, original_width, original_height, pad_x, pad_y, scale)
        """
        orig_w, orig_h = image.size
        
        # Calculate scale and padding (letterboxing)
        scale = min(input_size / orig_w, input_size / orig_h)
        scaled_w = int(orig_w * scale)
        scaled_h = int(orig_h * scale)
        pad_x = (input_size - scaled_w) / 2
        pad_y = (input_size - scaled_h) / 2
        
        # Create letterboxed image
        letterbox_img = Image.new('RGB', (input_size, input_size), (114, 114, 114))
        resized_img = image.resize((scaled_w, scaled_h), Image.BILINEAR)
        letterbox_img.paste(resized_img, (int(pad_x), int(pad_y)))
        
        # Convert to numpy array and normalize
        img_array = np.array(letterbox_img).astype(np.float32) / 255.0
        
        # Convert from HWC to CHW format
        img_array = img_array.transpose(2, 0, 1)
        
        # Add batch dimension
        input_tensor = np.expand_dims(img_array, axis=0)
        
        logger.debug(f"Letterbox preprocessing: orig={orig_w}x{orig_h}, scale={scale:.3f}, pad=({pad_x:.1f}, {pad_y:.1f})")
        
        return input_tensor, orig_w, orig_h, pad_x, pad_y, scale
    
    def _process_yolo_output(
        self, output: np.ndarray, orig_w: int, orig_h: int, 
        pad_x: float, pad_y: float, scale: float
    ) -> List:
        """
        Process YOLO ONNX output and apply NMS
        Similar to frontend OnnxDetectionService.processOutput
        """
        # Handle different output formats [batch, detections, values] or [batch, values, detections]
        if output.shape[1] < output.shape[2]:
            # Transposed format: [1, values, detections]
            output = output.transpose(0, 2, 1)
        
        detections = []
        boxes = []
        scores = []
        
        # Process each detection
        num_values = output.shape[2]
        is_single_class = num_values == 5  # Single-class model: [x, y, w, h, objectness]
        
        for det in output[0]:
            # YOLO format: [x_center, y_center, width, height, objectness, class_scores...]
            x_center, y_center, width, height = det[:4]
            objectness = self._sigmoid(det[4])
            
            if is_single_class:
                # Single-class model: use objectness directly
                max_score = objectness
            else:
                # Multi-class model: get class scores (apply sigmoid and multiply by objectness)
                class_scores = self._sigmoid(det[5:])
                final_scores = objectness * class_scores
                max_score = np.max(final_scores) if len(final_scores) > 0 else objectness
            
            # Filter by confidence threshold
            if max_score > self.conf_threshold:
                boxes.append([x_center, y_center, width, height])
                scores.append(max_score)
        
        # Apply NMS
        if len(boxes) > 0:
            boxes_array = np.array(boxes)
            scores_array = np.array(scores)
            selected_indices = self._non_max_suppression(boxes_array, scores_array, self.iou_threshold)
            
            for idx in selected_indices:
                box = boxes[idx]
                x_center, y_center, w, h = box
                
                # Convert from center format to corner format
                x1_640 = x_center - w / 2
                y1_640 = y_center - h / 2
                x2_640 = x_center + w / 2
                y2_640 = y_center + h / 2
                
                # Remove letterbox padding and scale back to original size
                x1 = (x1_640 - pad_x) / scale
                y1 = (y1_640 - pad_y) / scale
                x2 = (x2_640 - pad_x) / scale
                y2 = (y2_640 - pad_y) / scale
                
                detections.append([x1, y1, x2, y2, scores[idx]])
        
        return detections
    
    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def _non_max_suppression(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """
        Non-Maximum Suppression
        boxes: array of shape (N, 4) in format [x_center, y_center, width, height]
        scores: array of shape (N,)
        """
        # Convert to corner format for IoU calculation
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        
        areas = boxes[:, 2] * boxes[:, 3]
        order = scores.argsort()[::-1]
        
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(int(i))
            
            if len(order) == 1:
                break
            
            # Calculate IoU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            intersection = w * h
            
            union = areas[i] + areas[order[1:]] - intersection
            iou = intersection / union
            
            # Keep boxes with IoU less than threshold
            indices = np.where(iou <= iou_threshold)[0]
            order = order[indices + 1]
        
        return keep
    
    def _classify_species(self, cropped_image: Image.Image) -> tuple:
        """
        Stage 3: Classify species using Swin Transformer (PyTorch or ONNX)
        
        Args:
            cropped_image: PIL Image of cropped lizard
            
        Returns:
            Tuple of (class_id, confidence)
        """
        if self.use_onnx:
            return self._classify_species_onnx(cropped_image)
        else:
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
    
    def _classify_species_onnx(self, cropped_image: Image.Image) -> tuple:
        """
        Stage 3: Classify species using Swin ONNX model
        
        Args:
            cropped_image: PIL Image of cropped lizard
            
        Returns:
            Tuple of (class_id, confidence)
        """
        # Preprocess image for Swin (224x224, normalized)
        input_tensor = self._preprocess_swin_image(cropped_image)
        
        # Run inference
        input_name = self.swin_session.get_inputs()[0].name
        outputs = self.swin_session.run(None, {input_name: input_tensor})
        
        # Process output (logits)
        logits = outputs[0][0]  # Remove batch dimension
        
        # Apply softmax to get probabilities
        exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
        probs = exp_logits / np.sum(exp_logits)
        
        # Get predicted class and confidence
        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])
        
        return predicted_class, confidence
    
    def _preprocess_swin_image(self, image: Image.Image, input_size: int = 384) -> np.ndarray:
        """
        Preprocess image for Swin Transformer ONNX model
        Note: This model uses 384x384 (swin-base-patch4-window12-384)
        
        Returns:
            Preprocessed image tensor
        """
        # Resize to 384x384
        img_resized = image.resize((input_size, input_size), Image.BILINEAR)
        
        # Convert to numpy array and normalize (ImageNet normalization)
        img_array = np.array(img_resized).astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_array = (img_array - mean) / std
        
        # Convert from HWC to CHW format
        img_array = img_array.transpose(2, 0, 1)
        
        # Add batch dimension
        input_tensor = np.expand_dims(img_array, axis=0)
        
        return input_tensor
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        info = {
            "device": self.device,
            "use_onnx": self.use_onnx,
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }
        
        if self.use_onnx:
            info.update({
                "yolo_loaded": self.yolo_session is not None,
                "swin_loaded": self.swin_session is not None,
                "yolo_providers": self.yolo_session.get_providers() if self.yolo_session else None,
                "swin_providers": self.swin_session.get_providers() if self.swin_session else None
            })
        else:
            info.update({
                "yolo_loaded": self.yolo_model is not None,
                "swin_loaded": self.swin_model is not None
            })
        
        return info


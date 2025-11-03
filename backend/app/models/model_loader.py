"""
Singleton model loader for lazy loading and caching models
Ensures models are loaded only once and shared across requests
"""
import logging
from pathlib import Path
from typing import Optional
from app.models.pipeline import AnolePipeline

logger = logging.getLogger(__name__)

class ModelLoader:
    """Singleton class for managing model lifecycle"""
    
    _instance: Optional['ModelLoader'] = None
    _pipeline: Optional[AnolePipeline] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_pipeline(
        self,
        yolo_model_path: str = "../Spring_2025/yolov8x/weights/best.pt",
        swin_model_path: str = "swin-base-patch4-window12-384-finetuned-lizard-class-swin-base",  # Using base model for now - update with your fine-tuned model
        force_reload: bool = False,
        use_onnx: bool = False
    ) -> AnolePipeline:
        """
        Get or initialize the pipeline
        
        Args:
            yolo_model_path: Path to YOLO weights (.pt or .onnx)
            swin_model_path: Path to Swin Transformer (directory or .onnx)
            force_reload: Force reload models even if cached
            use_onnx: Whether to use ONNX models instead of PyTorch
            
        Returns:
            AnolePipeline instance
        """
        if self._pipeline is None or force_reload:
            logger.info("Initializing ML pipeline...")
            logger.info(f"Using ONNX: {use_onnx}")
            
            # Validate paths
            yolo_path = Path(yolo_model_path)
            if not yolo_path.exists():
                raise FileNotFoundError(f"YOLO model not found at {yolo_model_path}")
            
            # Initialize pipeline
            self._pipeline = AnolePipeline(
                yolo_model_path=str(yolo_path),
                swin_model_path=swin_model_path,
                conf_threshold=0.25,
                iou_threshold=0.5,
                use_onnx=use_onnx
            )
            
            logger.info("Pipeline initialized successfully")
            logger.info(f"Model info: {self._pipeline.get_model_info()}")
        
        return self._pipeline
    
    def is_loaded(self) -> bool:
        """Check if pipeline is loaded"""
        return self._pipeline is not None
    
    def unload(self):
        """Unload models from memory"""
        if self._pipeline is not None:
            logger.info("Unloading models...")
            self._pipeline = None
            logger.info("Models unloaded")

# Global instance
model_loader = ModelLoader()


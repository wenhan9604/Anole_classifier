#!/usr/bin/env python3
"""
YOLO Detection Pipeline for Anole Classification
1. Load trained YOLO model
2. Run detection on images
3. Crop detected lizards
4. Organize for classification training
"""

import os
import cv2
import numpy as np
from pathlib import Path
import shutil
import logging
from PIL import Image
import random
from ultralytics import YOLO
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class YOLODetectionPipeline:
    def __init__(self, yolo_model_path, conf_threshold=0.5, iou_threshold=0.45):
        """
        Initialize the YOLO detection pipeline
        
        Args:
            yolo_model_path: Path to trained YOLO model weights
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Load YOLO model
        logger.info(f"Loading YOLO model from: {yolo_model_path}")
        self.model = YOLO(yolo_model_path)
        logger.info("YOLO model loaded successfully!")
    
    def detect_and_crop_image(self, image_path, output_dir, image_id):
        """
        Detect lizards in an image and crop them
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save cropped images
            image_id: Unique identifier for this image
            
        Returns:
            List of cropped image paths
        """
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Could not load image: {image_path}")
                return []
            
            # Run YOLO detection
            results = self.model(image, conf=self.conf_threshold, iou=self.iou_threshold)
            
            cropped_paths = []
            detection_count = 0
            
            # Process detections
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for i, box in enumerate(boxes):
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        
                        # Convert to integers
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Ensure coordinates are within image bounds
                        h, w = image.shape[:2]
                        x1 = max(0, min(x1, w))
                        y1 = max(0, min(y1, h))
                        x2 = max(0, min(x2, w))
                        y2 = max(0, min(y2, h))
                        
                        # Skip if box is too small
                        if (x2 - x1) < 10 or (y2 - y1) < 10:
                            continue
                        
                        # Crop the image
                        cropped = image[y1:y2, x1:x2]
                        
                        # Save cropped image
                        cropped_filename = f"{image_id}_detection_{detection_count}_conf_{conf:.3f}.jpg"
                        cropped_path = output_dir / cropped_filename
                        
                        cv2.imwrite(str(cropped_path), cropped)
                        cropped_paths.append(cropped_path)
                        detection_count += 1
            
            return cropped_paths
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return []
    
    def process_dataset(self, input_dir, output_dir, max_images_per_class=None, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Process entire dataset through YOLO detection and cropping
        
        Args:
            input_dir: Directory containing taxon folders with images
            output_dir: Directory to save organized cropped images
            max_images_per_class: Maximum images to process per class (None for all)
            train_ratio: Proportion for training split
            val_ratio: Proportion for validation split
            test_ratio: Proportion for test split
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")
        
        # Create output directory structure
        splits = ['train', 'validation', 'test']
        for split in splits:
            (output_path / split).mkdir(parents=True, exist_ok=True)
        
        # Get taxon folders
        taxon_folders = [f for f in input_path.iterdir() if f.is_dir() and f.name.startswith('taxon_')]
        
        if not taxon_folders:
            raise ValueError(f"No taxon folders found in {input_dir}")
        
        logger.info(f"Found taxon folders: {[f.name for f in taxon_folders]}")
        
        total_detections = 0
        stats = {'train': 0, 'validation': 0, 'test': 0}
        
        for taxon_folder in taxon_folders:
            taxon_name = taxon_folder.name
            logger.info(f"Processing {taxon_name}...")
            
            # Get all images in this taxon folder
            image_files = list(taxon_folder.glob("*.jpg")) + list(taxon_folder.glob("*.JPG")) + list(taxon_folder.glob("*.jpeg"))
            
            if max_images_per_class:
                image_files = image_files[:max_images_per_class]
            
            if not image_files:
                logger.warning(f"No images found in {taxon_folder}")
                continue
            
            logger.info(f"Processing {len(image_files)} images from {taxon_name}")
            
            # Shuffle images for random split
            random.shuffle(image_files)
            
            # Calculate split indices
            total_count = len(image_files)
            train_count = int(total_count * train_ratio)
            val_count = int(total_count * val_ratio)
            
            # Split images
            train_images = image_files[:train_count]
            val_images = image_files[train_count:train_count + val_count]
            test_images = image_files[train_count + val_count:]
            
            logger.info(f"Split {taxon_name}: Train={len(train_images)}, Val={len(val_images)}, Test={len(test_images)}")
            
            # Process each split
            for split_name, split_images in [('train', train_images), ('validation', val_images), ('test', test_images)]:
                split_output_dir = output_path / split_name / taxon_name
                split_output_dir.mkdir(parents=True, exist_ok=True)
                
                for i, img_path in enumerate(split_images):
                    image_id = f"{taxon_name}_{img_path.stem}_{i}"
                    
                    # Detect and crop
                    cropped_paths = self.detect_and_crop_image(img_path, split_output_dir, image_id)
                    
                    # Update statistics
                    stats[split_name] += len(cropped_paths)
                    total_detections += len(cropped_paths)
                    
                    if (i + 1) % 100 == 0:
                        logger.info(f"Processed {i + 1}/{len(split_images)} images in {split_name}/{taxon_name}")
        
        # Print summary
        logger.info("=" * 50)
        logger.info("YOLO DETECTION PIPELINE COMPLETE")
        logger.info("=" * 50)
        logger.info(f"Total detections: {total_detections}")
        logger.info(f"Train detections: {stats['train']}")
        logger.info(f"Validation detections: {stats['validation']}")
        logger.info(f"Test detections: {stats['test']}")
        logger.info(f"Output directory: {output_path}")
        
        return output_path

def main():
    # Set random seed for reproducible splits
    random.seed(42)
    
    # Configuration
    yolo_model_path = "Spring_2025/ultralytics_runs/detect/train_yolov8n_v2/weights/best.pt"
    input_dir = "Dataset/downloaded_images_36488_36391_36455"
    output_dir = "Dataset/yolo_detected_anole_classification"
    
    # Check if YOLO model exists
    if not os.path.exists(yolo_model_path):
        logger.error(f"YOLO model not found: {yolo_model_path}")
        logger.error("Please ensure you have trained a YOLO model first.")
        return
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        logger.error(f"Input directory not found: {input_dir}")
        logger.error("Please run the image download script first.")
        return
    
    # Initialize pipeline
    pipeline = YOLODetectionPipeline(
        yolo_model_path=yolo_model_path,
        conf_threshold=0.5,  # Adjust based on your needs
        iou_threshold=0.45
    )
    
    # Process dataset
    try:
        output_path = pipeline.process_dataset(
            input_dir=input_dir,
            output_dir=output_dir,
            max_images_per_class=1000,  # Limit for faster processing
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15
        )
        
        logger.info(f"Pipeline completed successfully!")
        logger.info(f"Detected and cropped images saved to: {output_path}")
        logger.info("You can now use this dataset for classification training!")
        
    except Exception as e:
        logger.error(f"Error in pipeline: {e}")
        raise

if __name__ == "__main__":
    main()

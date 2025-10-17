#!/usr/bin/env python3
"""
Script to reorganize downloaded anole images into proper train/validation/test splits
for HuggingFace ImageFolder format
"""

import os
import shutil
import random
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_dataset_structure(source_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Reorganize images from taxon-based folders to train/val/test splits
    
    Args:
        source_dir: Path to downloaded_images_36488_36391_36455 folder
        output_dir: Path where new dataset structure will be created
        train_ratio: Proportion of images for training
        val_ratio: Proportion of images for validation  
        test_ratio: Proportion of images for testing
    """
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")
    
    # Create output directory structure
    splits = ['train', 'validation', 'test']
    for split in splits:
        (output_path / split).mkdir(parents=True, exist_ok=True)
    
    # Get all taxon folders
    taxon_folders = [f for f in source_path.iterdir() if f.is_dir() and f.name.startswith('taxon_')]
    
    if not taxon_folders:
        raise ValueError(f"No taxon folders found in {source_dir}")
    
    logger.info(f"Found taxon folders: {[f.name for f in taxon_folders]}")
    
    total_images = 0
    stats = {'train': 0, 'validation': 0, 'test': 0}
    
    for taxon_folder in taxon_folders:
        taxon_name = taxon_folder.name
        logger.info(f"Processing {taxon_name}...")
        
        # Get all images in this taxon folder
        image_files = list(taxon_folder.glob("*.jpg")) + list(taxon_folder.glob("*.JPG")) + list(taxon_folder.glob("*.jpeg"))
        
        if not image_files:
            logger.warning(f"No images found in {taxon_folder}")
            continue
            
        logger.info(f"Found {len(image_files)} images in {taxon_name}")
        
        # Shuffle images for random split
        random.shuffle(image_files)
        
        # Calculate split indices
        total_count = len(image_files)
        train_count = int(total_count * train_ratio)
        val_count = int(total_count * val_ratio)
        # test_count gets the remainder
        
        # Split images
        train_images = image_files[:train_count]
        val_images = image_files[train_count:train_count + val_count]
        test_images = image_files[train_count + val_count:]
        
        logger.info(f"Split {taxon_name}: Train={len(train_images)}, Val={len(val_images)}, Test={len(test_images)}")
        
        # Create taxon subdirectories in each split
        for split in splits:
            (output_path / split / taxon_name).mkdir(parents=True, exist_ok=True)
        
        # Copy images to appropriate directories
        def copy_images(image_list, split_name):
            for img_path in image_list:
                dest_path = output_path / split_name / taxon_name / img_path.name
                shutil.copy2(img_path, dest_path)
                stats[split_name] += 1
        
        copy_images(train_images, 'train')
        copy_images(val_images, 'validation')
        copy_images(test_images, 'test')
        
        total_images += total_count
    
    # Print summary
    logger.info("=" * 50)
    logger.info("DATASET REORGANIZATION COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Total images processed: {total_images}")
    logger.info(f"Train images: {stats['train']}")
    logger.info(f"Validation images: {stats['validation']}")
    logger.info(f"Test images: {stats['test']}")
    logger.info(f"Output directory: {output_path}")
    
    # Verify the structure
    verify_structure(output_path, taxon_folders)
    
    return output_path

def verify_structure(output_path, taxon_folders):
    """Verify that the new dataset structure is correct"""
    logger.info("Verifying dataset structure...")
    
    splits = ['train', 'validation', 'test']
    taxon_names = [f.name for f in taxon_folders]
    
    for split in splits:
        split_path = output_path / split
        if not split_path.exists():
            logger.error(f"Missing split directory: {split}")
            continue
            
        for taxon_name in taxon_names:
            taxon_path = split_path / taxon_name
            if not taxon_path.exists():
                logger.error(f"Missing taxon directory: {split}/{taxon_name}")
                continue
                
            image_count = len(list(taxon_path.glob("*.jpg"))) + len(list(taxon_path.glob("*.JPG")))
            if image_count == 0:
                logger.warning(f"No images in {split}/{taxon_name}")
            else:
                logger.info(f"{split}/{taxon_name}: {image_count} images")
    
    logger.info("Structure verification complete!")

def main():
    # Set random seed for reproducible splits
    random.seed(42)
    
    # Define paths
    source_dir = "Dataset/downloaded_images_36488_36391_36455"
    output_dir = "Dataset/anole_classification"
    
    # Check if source directory exists
    if not os.path.exists(source_dir):
        logger.error(f"Source directory not found: {source_dir}")
        logger.error("Please run the download script first to download images.")
        return
    
    # Create the new dataset structure
    try:
        output_path = create_dataset_structure(
            source_dir=source_dir,
            output_dir=output_dir,
            train_ratio=0.7,    # 70% for training
            val_ratio=0.15,     # 15% for validation
            test_ratio=0.15     # 15% for testing
        )
        
        logger.info(f"Dataset successfully reorganized to: {output_path}")
        logger.info("You can now use this with the HuggingFace training script!")
        
    except Exception as e:
        logger.error(f"Error reorganizing dataset: {e}")
        raise

if __name__ == "__main__":
    main()

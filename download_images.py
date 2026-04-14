#!/usr/bin/env python3
"""
Script to download JPG images from iNaturalist CSV file
Downloads images to a organized folder structure
"""

import csv
import os
import requests
import time
from pathlib import Path
from urllib.parse import urlparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Thread-safe counter for downloads
download_count = 0
download_lock = threading.Lock()

def download_image(url, output_path, max_retries=3):
    """
    Download a single image from URL to output path
    Returns True if successful, False otherwise
    """
    global download_count
    
    for attempt in range(max_retries):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=30, stream=True)
            response.raise_for_status()
            
            # Create directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write file
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            with download_lock:
                download_count += 1
                if download_count % 100 == 0:
                    logger.info(f"Downloaded {download_count} images...")
            
            return True
            
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)  # Wait before retry
    
    logger.error(f"Failed to download {url} after {max_retries} attempts")
    return False

def get_filename_from_url(url, record_id):
    """
    Generate a filename using the record ID
    """
    # Always use the record ID as filename
    return f"{record_id}.jpg"

def download_images_from_csv(csv_path, output_dir, max_workers=10):
    """
    Download all images from CSV file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for each taxon
    taxon_dirs = {
        '36391': output_path / 'taxon_36391',
        '36488': output_path / 'taxon_36488', 
        '36455': output_path / 'taxon_36455'
    }
    
    for taxon_dir in taxon_dirs.values():
        taxon_dir.mkdir(parents=True, exist_ok=True)
    
    # Read CSV and collect download tasks
    download_tasks = []
    total_images = 0
    
    logger.info(f"Reading CSV file: {csv_path}")
    
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            image_url = row['image_url'].strip()
            record_id = row['id'].strip()
            taxon_id = row['taxon_id'].strip()
            
            if image_url and image_url.startswith('http'):
                # Get filename
                filename = get_filename_from_url(image_url, record_id)
                
                # Determine output directory based on taxon
                if taxon_id in taxon_dirs:
                    output_file = taxon_dirs[taxon_id] / filename
                else:
                    output_file = output_path / 'other' / filename
                
                # Only add if file doesn't already exist
                if not output_file.exists():
                    download_tasks.append((image_url, output_file, record_id))
                else:
                    logger.debug(f"File already exists: {output_file}")
                
                total_images += 1
    
    logger.info(f"Found {total_images} images total, {len(download_tasks)} new downloads needed")
    
    if not download_tasks:
        logger.info("All images already downloaded!")
        return
    
    # Download images with threading
    successful_downloads = 0
    failed_downloads = 0
    
    logger.info(f"Starting downloads with {max_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_task = {
            executor.submit(download_image, url, path): (url, path, record_id)
            for url, path, record_id in download_tasks
        }
        
        # Process completed downloads
        for future in as_completed(future_to_task):
            url, path, record_id = future_to_task[future]
            try:
                success = future.result()
                if success:
                    successful_downloads += 1
                else:
                    failed_downloads += 1
            except Exception as e:
                logger.error(f"Download task failed for {record_id}: {e}")
                failed_downloads += 1
    
    logger.info(f"Download complete!")
    logger.info(f"Successful downloads: {successful_downloads}")
    logger.info(f"Failed downloads: {failed_downloads}")
    logger.info(f"Images saved to: {output_path}")

def main():
    csv_file = "Dataset/36488&36391&36455.csv"
    output_directory = "Dataset/downloaded_images_36488_36391_36455"
    
    if not os.path.exists(csv_file):
        logger.error(f"CSV file not found: {csv_file}")
        return
    
    logger.info(f"Starting image download from {csv_file}")
    logger.info(f"Output directory: {output_directory}")
    
    start_time = time.time()
    download_images_from_csv(csv_file, output_directory, max_workers=10)
    end_time = time.time()
    
    logger.info(f"Total time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()

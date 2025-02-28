import os
import shutil
from pathlib import Path

def copy_image_and_text_files(src_folder, dest_folder, n=200):
    # Ensure source and destination folders exist
    src_img_path = Path(str(src_folder + "/images"))
    src_txt_path = Path(str(src_folder + "/labels"))
    dest_img_path = Path(str(dest_folder + "/images"))
    dest_txt_path = Path(str(dest_folder + "/labels"))
    
    if not src_img_path.is_dir():
        print(f"Source folder '{src_folder}' does not exist.")
        return
    
    # Create destination folder if it doesn't exist
    dest_img_path.mkdir(parents=True, exist_ok=True)
    dest_txt_path.mkdir(parents=True, exist_ok=True)  
    
    # Get all files sorted by modification time (most recent first)
    images = sorted(src_img_path.glob('*'), key=lambda img_file: img_file.name)
    
    # Filter to ensure we only copy files, not directories
    images = [img_file for img_file in images if img_file.is_file()]
    
    # Copy the last 'n' files
    for image in images[-n:]:
        img_name = image.stem

        #Move image
        shutil.move(str(image), str(dest_img_path / image.name))
        print(f"Copied image: {image} -> {dest_img_path}")

        #Move corresponding label text file
        txt_file_name = img_name + ".txt"
        corresponding_txt_file_path = str(src_txt_path/txt_file_name)
        shutil.move(str(corresponding_txt_file_path), str(dest_txt_path/txt_file_name))
        print(f"Copied txt file: {corresponding_txt_file_path} -> {dest_txt_path}")


    print(f"Successfully copied {len(images[-n:])} files.")

# Example usage
source_folder = "../dataset/YOLO_training/YOLO_training_test/train"
destination_folder = "../dataset/YOLO_training/YOLO_training_test/test"
copy_image_and_text_files(source_folder, destination_folder)

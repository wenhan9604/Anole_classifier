import os

folder_path = "Dataset/YOLO_training/florida_10000_cleaned/crestedanole_2000/train/crested_new_images"
image_extensions = ('.jpg', '.jpeg', '.png')  # Add more if needed

images = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]

print(f"Total images: {len(images)}")

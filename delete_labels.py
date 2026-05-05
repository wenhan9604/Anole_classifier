import os

# Paths to your folders
images_folder = "Dataset/YOLO_training/florida_10000_cleaned/crestedanole_2000/train/crested_unclear_second_half"
labels_folder = "Dataset/YOLO_training/florida_10000_cleaned/crestedanole_2000/train/labels"

# List of rejected image filenames (you can build this dynamically too)
rejected_images = [f for f in os.listdir(images_folder) if f.endswith(".jpg")]

deleted = 0
for image_file in rejected_images:
    label_file = image_file.replace(".jpg", ".txt")
    label_path = os.path.join(labels_folder, label_file)

    # Delete label file if it exists
    if os.path.exists(label_path):
        os.remove(label_path)
        deleted += 1

print(f"✅ Deleted {deleted} matching label files.")
remaining_labels = len([f for f in os.listdir(labels_folder) if f.endswith(".txt")])
print(f"📊 Total remaining label files: {remaining_labels}")

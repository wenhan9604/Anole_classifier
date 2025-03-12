import cv2 
from pathlib import Path
import numpy as np

def crop_image(img_path, coord):
    "Will crop image based on bounding box coordinate. Coordinate will be given in YOLO format (x_center, y_center, width, height)"

    input = cv2.imread(img_path)

    print("input shape: ", input.shape)


    cv2.imshow('image', input) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 

    x_center, y_center, width, height = coord

    row_start = np.uint16(y_center - (height // 2))
    row_end = np.uint16(y_center + (height // 2))

    col_start = np.uint16(x_center - (width // 2))
    col_end = np.uint16(x_center + (width // 2))

    cropped_segment = input[row_start:row_end, col_start:col_end]

    print("Cropped shape: ", cropped_segment.shape)

    cv2.imshow('cropped', cropped_segment) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 

    return cropped_segment

def get_coord(labels_file_path):
    #Get coordinate from YOLOv8 txt format. YOLOv8 txt format are in normalized coordinate (0 to 1). 
    #Images are all resized to 640x640 from roboflow output.
    #Returns pixel coordinate
    
    with open(labels_file_path, "r") as file:
        data = file.read().strip().split()  # Read, strip, and split by spaces

    #Normalized coord (0 to 1)
    class_ID, x_center, y_center, width, height = map(float, data)

    #Convert to pixel coordinate
    x_center = np.floor(x_center * 640)
    y_center = np.floor(y_center * 640)
    width = np.floor(width * 640)
    height = np.floor(height * 640)

    # Print values to verify
    print(x_center, y_center, width, height)

    return (x_center, y_center, width, height)

def resize_img(img, size):
    """
    Args:
        img(numpy array)
        size(int,int) = (width, height)
    """

    resized_img = cv2.resize(img, size)

    return resized_img

def crop_resize_img_folder(src_folder_dir, dest_folder_dir):
    """
    Will crop and resize images from source folder and store in destination folder
    """

    if not src_folder_dir.is_dir():
        print(f"Source folder '{src_folder_dir}' does not exist.")
        return

    # Create destination folder if it doesn't exist
    dest_folder_dir.mkdir(parents=True, exist_ok=True)

    # Get all files sorted by modification time (most recent first)
    images = sorted(src_folder_dir.glob('*'), key=lambda img_file: img_file.name)
    
    # Filter to ensure we only copy files, not directories
    images = [img_file for img_file in images if img_file.is_file()]

    count = 0
    for image in images:

        img_name = "36391_5983992_jpg.rf.820094a9f215cce647b3d64d5bc0f5dc"

        img_path = "C:/Projects/OMSCS/Lizard_Classification/Anole_classifier/Dataset/YOLO_training/lizard_10000/train/images/" + img_name + ".jpg"
        labels_file_path = "C:/Projects/OMSCS/Lizard_Classification/Anole_classifier/Dataset/YOLO_training/lizard_10000/train/labels/" + img_name + ".txt"

        dest_img_name = img_name + "_cropped"
        dest_path = dest_folder_dir + dest_img_name + ".jpg"

        coord = get_coord(labels_file_path)
        cropped_image = crop_image(img_path, coord)
        cv2.imwrite(dest_path, cropped_image)

        count += 1
        print(f"Image count: {count}")

    print("Successfully Saved ")







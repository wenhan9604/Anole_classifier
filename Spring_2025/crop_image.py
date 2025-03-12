import cv2 
import numpy as np
from pathlib import Path

def crop_image(img_path, coord):
    "Will crop image based on bounding box coordinate. Coordinate will be given in YOLO format (x_center, y_center, width, height)"

    input = cv2.imread(img_path)

    # print("input shape: ", input.shape)

    # cv2.imshow('image', input) 
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows() 

    x_center, y_center, width, height = coord

    row_start = np.uint16(y_center - (height // 2))
    row_end = np.uint16(y_center + (height // 2))

    col_start = np.uint16(x_center - (width // 2))
    col_end = np.uint16(x_center + (width // 2))

    cropped_segment = input[row_start:row_end, col_start:col_end]

    # print("Cropped shape: ", cropped_segment.shape)

    # cv2.imshow('cropped', cropped_segment) 
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows() 

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
    # print(x_center, y_center, width, height)

    return (x_center, y_center, width, height)

def crop_resize_img_folder(src_folder_path, dest_folder_path, resize_value):
    """
    Will crop and resize images from source folder and store in destination folder

    Args:
        resize_value(int, int) : The size of the final resized images (width, height)
    """

    src_img_folder = Path(str(src_folder_path + "/images"))
    src_txt_folder = Path(str(src_folder_path + "/labels"))
    dest_folder = Path(str(dest_folder_path))

    if not src_img_folder.is_dir():
        print(f"Source folder '{src_folder_path}' does not exist.")
        return

    # Create destination folder if it doesn't exist
    dest_folder.mkdir(parents=True, exist_ok=True)

    # Get all files sorted by modification time (most recent first)
    images = sorted(src_img_folder.glob('*'), key=lambda img_file: img_file.name)
    
    # Filter to ensure we only copy files, not directories
    images = [img_file for img_file in images if img_file.is_file()]

    count = 0
    for image in images:

        img_name = image.stem

        img_text_path = img_name + ".txt"
        labels_file_path = str(src_txt_folder / img_text_path)

        #Core functions: Crop and Resize images
        coord = get_coord(labels_file_path)
        cropped_image = crop_image(image, coord)
        resized_img = cv2.resize(cropped_image, resize_value)

        dest_img_path = img_name + "_cropped.jpg"
        dest_path = str(dest_folder / dest_img_path)

        cv2.imwrite(dest_path, resized_img)

        count += 1
        print(f"Image count: {count}")

    print("Images Successfully cropped, resized and saved")

def unit_test_save_image():

    img_path = r"C:\Projects\OMSCS\Lizard_Classification\Anole_classifier\Dataset\YOLO_training\barkanole_2000\barkanole_2000\train\images\36455_116141_jpg.rf.2ca8cfe1b397edda02a70e1a51d6ae98.jpg"

    labels_file_path = r"C:\Projects\OMSCS\Lizard_Classification\Anole_classifier\Dataset\YOLO_training\barkanole_2000\barkanole_2000\train\labels\36455_116141_jpg.rf.2ca8cfe1b397edda02a70e1a51d6ae98.txt"

    resize_value = (320, 320)

    #Core functions: Crop and Resize images
    coord = get_coord(labels_file_path)
    cropped_image = crop_image(img_path, coord)
    resized_img = cv2.resize(cropped_image, resize_value)

    dest_img_name = "test_cropped"
    dest_path = dest_img_name + ".jpg"

    cv2.imwrite(dest_path, resized_img)

    print("End of test")

src_folder_path = "C:/Projects/OMSCS/Lizard_Classification/Anole_classifier/Dataset/YOLO_training/original/barkanole_2000/train"
dest_folder_path = "C:/Projects/OMSCS/Lizard_Classification/Anole_classifier/Dataset/YOLO_training/barkanole_2000_cropped"
resize_value = (320, 320)

crop_resize_img_folder(src_folder_path, dest_folder_path, resize_value)

# unit_test_save_image()






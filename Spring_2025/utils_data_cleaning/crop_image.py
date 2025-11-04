import cv2 
import numpy as np
from pathlib import Path
import math
import math

def crop_image_center(img_path, coord):
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

def crop_image_tl_br(img_path, coord):
    """
    coord is a tuple that consist of (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    """

    input = cv2.imread(img_path)

    # print("coord")
    # print(coord)

    tl_x, tl_y, br_x, br_y = coord

    tl_x = np.uint16(math.ceil(tl_x))
    tl_y = np.uint16(math.ceil(tl_y))
    br_x = np.uint16(br_x)
    br_y = np.uint16(br_y)

    # print("tl_x")
    # print(tl_x)

    cropped_segment = input[tl_y:br_y, tl_x:br_x]

    # print("Cropped shape: ", cropped_segment.shape)

    # cv2.imshow('cropped', cropped_segment) 
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows() 

    return cropped_segment

def get_coord(labels_file_path, img_width_height, is_coord_normalized = True):
    #Get coordinate from YOLOv8 txt format. YOLOv8 txt format are in normalized coordinate (0 to 1). 
    #Returns pixel coordinate

    instances_coord = []

    img_width = img_width_height[0]
    img_height = img_width_height[1]
    
    with open(labels_file_path, "r") as file:
        for line in file:
            data = line.strip().split()  # Read, strip, and split by spaces

            #Normalized coord (0 to 1)
            class_ID, x_center, y_center, width, height = map(float, data)

            if (is_coord_normalized):

                #Convert to pixel coordinate
                x_center = np.floor(x_center * img_width)
                y_center = np.floor(y_center * img_height)
                width = np.floor(width * img_width)
                height = np.floor(height * img_height)

            instances_coord.append((x_center, y_center, width, height))

    # Print values to verify
    # print(x_center, y_center, width, height)

    return instances_coord

def crop_resize_img_folder(src_folder_path, dest_folder_path, resize_value, coord_type="xywh_center"):
    """
    Will crop and resize images from source folder and store in destination folder

    Args:
        src_folder_path: Source folder must contain images and labels sub folder
        resize_value(int, int) : The size of the final resized images (width, height)
        coord_type : 
            "xyxy" represents topleft and bot right coord. 
            "xywh_center" represents xy coordinate of bounding box's center 
            "xywhn_center" represents xy coordinate of bounding box's center, normalized

    """

    src_img_folder = Path(src_folder_path + "/images")
    src_txt_folder = Path(src_folder_path + "/labels")
    dest_folder = Path(dest_folder_path)

    if not src_img_folder.is_dir():
        print(f"Source folder '{src_folder_path}' does not exist.")
        return
    
    coord_types_list = ["xyxy", "xywh_center", "xywhn_center"]

    if not coord_type in coord_types_list:
        print(f"coord_type is not compatible:{coord_type}")
        return

    # Create destination folder if it doesn't exist
    dest_folder.mkdir(parents=True, exist_ok=True)

    # Get all files sorted by modification time (most recent first)
    images = sorted(src_img_folder.glob('*'), key=lambda img_file: img_file.name)
    
    # Filter to ensure we only copy files, not directories
    images = [img_file for img_file in images if img_file.is_file()]

    processed_image_count = 0
    output_image_count = 0
    for img_file_path in images:

        img_name = img_file_path.stem

        image = cv2.imread(img_file_path)
        img_width_height = (image.shape[1], image.shape[0]) #(x, y) or (width, height)

        img_text_path = img_name + ".txt"
        labels_file_path = src_txt_folder / img_text_path

        #Core functions: Crop and Resize images
        if(coord_type == "xywhn_center"):
            instances_coord = get_coord(labels_file_path, img_width_height, True)
        else:
            instances_coord = get_coord(labels_file_path, img_width_height, False)

        instance_count = 0
        for coord in instances_coord:

            if(coord_type == "xyxy"):
                cropped_image = crop_image_tl_br(img_file_path, coord)

            elif (coord_type == "xywh_center" or coord_type == "xywhn_center"):
                cropped_image = crop_image_center(img_file_path, coord)

            resized_img = cv2.resize(cropped_image, resize_value)

            dest_img_path = img_name + "_cropped_" + str(instance_count) + ".jpg"
            dest_path = dest_folder / dest_img_path

            cv2.imwrite(dest_path, resized_img)
            instance_count += 1
            output_image_count += 1

        processed_image_count += 1

    print(f"Input image count: {processed_image_count}. Individual instance image count: {output_image_count}")
    print(f"Images successfully cropped, resized and saved to {dest_folder}")

def unit_test_save_image():

    img_path = r"C:\Projects\OMSCS\Lizard_Classification\Anole_classifier\Dataset\YOLO_training\barkanole_2000\barkanole_2000\train\images\36455_116141_jpg.rf.2ca8cfe1b397edda02a70e1a51d6ae98.jpg"

    labels_file_path = r"C:\Projects\OMSCS\Lizard_Classification\Anole_classifier\Dataset\YOLO_training\barkanole_2000\barkanole_2000\train\labels\36455_116141_jpg.rf.2ca8cfe1b397edda02a70e1a51d6ae98.txt"

    resize_value = (320, 320)

    #Core functions: Crop and Resize images
    coord = get_coord(labels_file_path)
    cropped_image = crop_image_center(img_path, coord)
    resized_img = cv2.resize(cropped_image, resize_value)

    dest_img_name = "test_cropped"
    dest_path = dest_img_name + ".jpg"

    cv2.imwrite(dest_path, resized_img)

    print("End of test")

def unit_test_save_image_instances():

    img_path = r"C:\Projects\OMSCS\Lizard_Classification\Anole_classifier\Dataset\YOLO_training\original\barkanole_2000\train\images\36455_101026830_jpg.rf.e839b9ff35e4f042bccaf77bc1381ff2.jpg"

    labels_file_path = r"C:\Projects\OMSCS\Lizard_Classification\Anole_classifier\Dataset\YOLO_training\original\barkanole_2000\train\labels\36455_101026830_jpg.rf.e839b9ff35e4f042bccaf77bc1381ff2.txt"

    resize_value = (320, 320)

    #Core functions: Crop and Resize images
    instances_coord = get_coord(labels_file_path)

    dest_folder_path = "C:/Projects/OMSCS/Lizard_Classification/Anole_classifier/Dataset/YOLO_training"

    count = 0
    for coord in instances_coord:
        
        cropped_image = crop_image_center(img_path, coord)
        resized_img = cv2.resize(cropped_image, resize_value)

        dest_img_path = "test_cropped_" + str(count) + ".jpg"
        dest_path = dest_folder_path + "/" + dest_img_path

        cv2.imwrite(dest_path, resized_img)

        count += 1
        print(f"Image count: {count}")

    print("End of test")

if (__name__ == "__main__"):

    # src_folder_path = "C:/Projects/OMSCS/Lizard_Classification/Anole_classifier/Dataset/YOLO_training/original/barkanole_2000/train"
    # dest_folder_path = "C:/Projects/OMSCS/Lizard_Classification/Anole_classifier/Dataset/YOLO_training/barkanole_2000_cropped"
    # resize_value = (320, 320)

    # crop_resize_img_folder(src_folder_path, dest_folder_path, resize_value)

    # unit_test_save_image()

    # unit_test_save_image_instances()

    # src_folder_path = "../Dataset/YOLO_training/dataset_v3/original_cleaned/florida_10000_cleaned_revised/cropped_lizard_10000_v3/bark_anole"
    # dest_folder_path = "../Dataset/YOLO_training/dataset_v3/original_cleaned/florida_10000_cleaned_verified/bark_anole_2000_verified/ichha_new_annotated_cropped"
    # resize_value = (384, 384)
    # crop_resize_img_folder(src_folder_path, dest_folder_path, resize_value, coord_type="xywhn_center")

    # species_name = "knight_anole"
    species_names = ["bark_anole", "brown_anole", "crested_anole", "green_anole", "knight_anole"]
    resize_value = (384, 384) #Following swin transformer format 

    source_parent_folder = "../Dataset/YOLO_training/dataset_v4/cropped_lizard_10000_v4/split"
    destination_parent_folder = "../Dataset/YOLO_training/dataset_v4/cropped_lizard_10000_v4"

    sub_directories = ['test', 'valid', 'train']

    for species_name in species_names:
        for sub_folder in sub_directories:

            source_folder_path = f"{source_parent_folder}/{species_name}/{sub_folder}"
            destination_folder_path = f"{destination_parent_folder}/{sub_folder}/{species_name}"

            crop_resize_img_folder(source_folder_path, destination_folder_path, resize_value, coord_type="xywhn_center")
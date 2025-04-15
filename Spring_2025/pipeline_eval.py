from ultralytics import YOLO 
import os
from IPython.display import display, Image 
from IPython import display 
import ultralytics
from pathlib import Path
from crop_image import crop_resize_img_folder

ODmodel = YOLO("./ultralytics_runs/detect/train_yolov8n_v2/weights/best.pt")
#ODmodel = YOLO("./runs/detect/train_yolov8s/weights/best.pt")

dict_anole = {0: "bark_anole",
              1: "brown_anole",
              2: "crested_anole",
              3: "green_anole",
              4: "knight_anole"}

def OD_inference():

    test_folder_path = "../Dataset/YOLO_training/florida_five_anole_10000/test"
    dest_folder_path = "../Dataset/YOLO_training/inference/run1/lizard_detection"
    log_folder_path = "../Dataset/YOLO_training/inference/run1/lizard_detection_log"

    missed_detection = []

    #Get images and text folder 
    test_img_folder = Path(str(test_folder_path + "/images"))
    test_txt_folder = Path(str(test_folder_path + "/labels"))

    for key, value in dict_anole.items():
        dest_img_folder = Path(str(dest_folder_path + "/" + value + "/images"))
        dest_txt_folder = Path(str(dest_folder_path + "/" + value + "/labels"))

        # Create destination folder if it doesn't exist
        dest_img_folder.mkdir(parents=True, exist_ok=True)
        dest_txt_folder.mkdir(parents=True, exist_ok=True)

    images = sorted(test_img_folder.glob('*'), key=lambda img_file: img_file.name)
    images = [img_file for img_file in images if img_file.is_file()]

    for image in images:

        img_name = image.stem

        # 1.Get ground truth label of image
        label_file_name = img_name + ".txt"
        label_file_path = str(test_txt_folder / label_file_name)

        with open(label_file_path, "r") as file:
            for line in file:
                data = line.strip().split() 

                class_ID, x_center, y_center, width, height = map(float, data)

        # Determine destination folder 

        class_label = dict_anole[class_ID]

        print(f"class label: {class_label}")

        dest_img_folder = Path(str(dest_folder_path + "/" + class_label + "/images"))
        dest_txt_folder = Path(str(dest_folder_path + "/" + class_label + "/labels"))

        #Get inference on image 
        img_file_name = img_name + ".jpg"
        img_file_path = str(test_img_folder / img_file_name)

        results = ODmodel(img_file_path)

        # 1.Save image with bounding box 
        dest_img_file_path = str(dest_img_folder / img_file_name)
        result = results[0]
        result.save(filename=dest_img_file_path)

        # 2.Get new bounding box coordinate
        xyxy = result.boxes.xyxy
        
        if(xyxy.nelement() == 0):
            # log_file_path = Path(str(log_folder_path + "/log_missed_detection.txt"))
            # try:
            #     with open(log_file_path, "x") as file:
            #         file.write(f"{img_file_name} \n")
            # except FileExistsError:
            #     print(f"Error: {log_file_path} unable to be appended")
                    
            missed_detection.append(img_file_name)
            
            continue
            
        
        coord = tuple(xyxy[0].tolist())
        
        # 4.Write label text file using ground truth ID & coordinate

        dest_label_file_name = img_name + ".txt"
        dest_label_file_path = str(dest_txt_folder / dest_label_file_name)

        try:
            with open(dest_label_file_path, 'x') as file:  # 'x' mode creates only if file doesn't exist
                file.write(f"{class_ID} {coord[0]} {coord[1]} {coord[2]} {coord[3]}\n")
        except FileExistsError:
            print(f"Error: {dest_label_file_path} already exists!")

    print(f"Missed Detections: {missed_detection.count}\n")

    for item in missed_detection:
        print(f"{item}\n")
    

def crop_image():
    src_folder_path = "../Dataset/YOLO_training/inference/run1/lizard_detection"
    dest_folder_path = "../Dataset/YOLO_training/inference/run1/cropped_image"
    resize = (384, 384)

    for key, value in dict_anole.items():
        source_target = src_folder_path + "/" + value
        dest_target = str(dest_folder_path + "/" + value)

    crop_resize_img_folder(source_target, dest_target, resize, coord_type="xyxy")


OD_inference()
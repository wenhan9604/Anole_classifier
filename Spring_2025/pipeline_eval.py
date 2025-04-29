from ultralytics import YOLO 
import os
from IPython.display import display, Image 
from IPython import display 
import ultralytics
from pathlib import Path
from crop_image import crop_resize_img_folder

# ODmodel = YOLO("./ultralytics_runs/detect/train_yolov8n_v2/weights/best.pt")
#ODmodel = YOLO("./runs/detect/train_yolov8s/weights/best.pt")

def OD_inference(yolo_model_file_path,
                test_folder_path = "../Dataset/YOLO_training/florida_five_anole_10000/test",
                dest_folder_path = "../Dataset/YOLO_training/inference/run1/lizard_detection",
                log_folder_path = "../Dataset/YOLO_training/inference/run1/lizard_detection_log"):

    test_folder_path = Path(test_folder_path)
    dest_folder_path = Path(dest_folder_path)
    log_folder_path = Path(log_folder_path)

    ODmodel = YOLO(yolo_model_file_path)

    dict_anole = {0: "bark_anole",
                1: "brown_anole",
                2: "crested_anole",
                3: "green_anole",
                4: "knight_anole"}

    #Create log folder if it doesnt exist
    log_folder_path.mkdir(parents=True, exist_ok=True)
    log_file_path = Path(log_folder_path) / "log_missed_detection.txt" 

    missed_detection_count = {"bark_anole": 0,
                "brown_anole" : 0,
                "crested_anole" : 0,
                "green_anole" : 0,
                "knight_anole" : 0}

    #Setup: Create destination images and text folder 
    test_img_folder = test_folder_path / "images"
    test_txt_folder = test_folder_path / "labels"

    for key, value in dict_anole.items():
        dest_img_folder = dest_folder_path / value / "images"
        dest_txt_folder = dest_folder_path / value / "labels"

        # Create destination folder if it doesn't exist
        dest_img_folder.mkdir(parents=True, exist_ok=True)
        dest_txt_folder.mkdir(parents=True, exist_ok=True)

    # Iterate through each image: 
    # 1. Get ground truth label (lizard class) of image
    # 2. Determine desination folder (To sort images and text into each destination folder later)
    # 3. Run prediction on image 
    #   3.1 Save image with bounding box
    #   3.2 Get predicted bounding box coordinate
    #   3.3 Save predicted text file using ground truth label (lizard class) & predicted coordinate

    images = sorted(test_img_folder.glob('*'), key=lambda img_file: img_file.name)
    images = [img_file for img_file in images if img_file.is_file()]

    for image in images:

        img_name = image.stem

        # 1.Get ground truth label (lizard class) of image
        label_file_path = test_txt_folder / f"{img_name}.txt"

        with open(label_file_path, "r") as file:
            for line in file:
                data = line.strip().split() 

                class_ID, x_center, y_center, width, height = map(float, data)

        # 2. Determine destination folder 

        class_label = dict_anole[class_ID]

        print(f"class label: {class_label}")

        dest_img_folder = dest_folder_path / class_label / "images"
        dest_txt_folder = dest_folder_path / class_label / "labels"

        # 3.Get inference on image 
        img_file_name = f"{img_name}.jpg"
        img_file_path = str(test_img_folder / img_file_name) 

        results = ODmodel(img_file_path)
        result = results[0]

        xyxy = result.boxes.xyxy
        bb_pred_conf = result.boxes.conf
        
        # Check for missed detections
        if(xyxy.nelement() == 0):
            try:
                with open(log_file_path, "w") as file:
                    file.write(f"{class_label} {img_file_name} \n")
            except FileExistsError:
                print(f"Error: {log_file_path} unable to be appended")
                    
            # missed_detection.append(img_file_name)
            missed_detection_count[class_label] = missed_detection_count[class_label] + 1

            print(f"missed detection bb confidence: {bb_pred_conf}\n")

            continue

        # 3.1 Save image with bounding box 
        dest_img_file_path = str(dest_img_folder / img_file_name)
        result.save(filename=dest_img_file_path)

        # 3.2 Get predicted bounding box coordinate
        coord = tuple(xyxy[0].tolist())
        
        # 3.3 Save predicted text file using ground truth label (lizard class) & predicted coordinate

        dest_label_file_name = img_name + ".txt"
        dest_label_file_path = str(dest_txt_folder / dest_label_file_name)

        try:
            with open(dest_label_file_path, 'x') as file:  # 'x' mode creates only if file doesn't exist
                file.write(f"{class_ID} {coord[0]} {coord[1]} {coord[2]} {coord[3]}\n")
        except FileExistsError:
            print(f"Error: {dest_label_file_path} already exists!")


    print(f"Missed Detections: \n {missed_detection_count}\n")     

    #Save missed detections into log file
    try:
        with open(log_file_path, "a") as file:
            file.write("\nSummary of missed detections:\n")

            for key, value in missed_detection_count.items():
                file.write(f"{key} : {value} \n")

    except FileExistsError:
        print(f"Error: {log_file_path} unable to be appended")

def crop_image():

    dict_anole = {0: "bark_anole",
                1: "brown_anole",
                2: "crested_anole",
                3: "green_anole",
                4: "knight_anole"}

    src_folder_path = "../Dataset/YOLO_training/inference/run1/lizard_detection"
    dest_folder_path = "../Dataset/YOLO_training/inference/run1/cropped_image"
    resize = (384, 384)

    for key, value in dict_anole.items():
        source_target = src_folder_path + "/" + value
        dest_target = str(dest_folder_path + "/" + value)

    crop_resize_img_folder(source_target, dest_target, resize, coord_type="xyxy")

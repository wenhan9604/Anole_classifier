import os
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import AutoImageProcessor, SwinForImageClassification
from sklearn.metrics import classification_report
import pandas as pd
from datetime import datetime
from pathlib import Path
import numpy as np
import math
import cv2 

# --- Load models ---
YOLO_MODEL_FILE_PATH = "./runs/detect/train22_yolov8x_dataset_v4/weights/best.pt"
SWIN_MODEL_FILE_PATH = "swin-base-patch4-window12-384-finetuned-lizard-v3-swin-base"

# --- YOLO model config --- 
NMS_IOU_THRESHOLD = 0.25
CONF_THRESH = 0.5
TOP_K = 5           # Max number of boxes to classify (set to None for no limit)

# --- Evaluation Config ---
DEST_FOLDER_PATH = "./inference"
INPUT_IMAGE_FOLDER = "../Dataset/yolo_training/florida_five_anole_10000_v4/test/images"
INPUT_LABEL_FOLDER = "../Dataset/yolo_training/florida_five_anole_10000_v4/test/labels"
MISSED_CLASS_ID = 5  # Custom label for missed detections
EVAL_IOU_THRESHOLD = 0.2

ID_TO_NAME = {0: "bark", 
                1: "brown",
                2: "crested",
                3: "green",
                4: "knight"}

#Helper function

def clamp_coords(x1, y1, x2, y2, img_w, img_h):

    x1 = int(round(x1))
    y1 = int(round(y1))
    x2 = int(round(x2))
    y2 = int(round(y2))

    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))

    return x1, y1, x2, y2

# --- Helper: Convert YOLO to [x1, y1, x2, y2] ---
def yolo_to_xyxy(yolo_box, img_w, img_h):

    cls, xc, yc, w, h = yolo_box #yolo_box format (normalized) = (class_id, x_center, y_center, width_boundingbox, height_boundingbox)

    xc, yc, w, h = xc * img_w, yc * img_h, w * img_w, h * img_h

    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2

    x1, y1, x2, y2 = clamp_coords(x1, y1, x2, y2, img_w, img_h)

    return [x1, y1, x2, y2]

# --- IoU computation ---
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]) #X coord of top left corner of intersection
    yA = max(boxA[1], boxB[1]) #Y coord of top left corner of intersection
    xB = min(boxA[2], boxB[2]) #X coord of bottom right corner of intersection
    yB = min(boxA[3], boxB[3]) #Y coord of bottom right corner of intersection
    interArea = max(0, xB - xA) * max(0, yB - yA) #Edge case of 0 intersection area 
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]) # (x2 - x1) * (y2 - y1)
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]) 
    return interArea / float(boxAArea + boxBArea - interArea)

def annotate_and_save_image(image_rgb, gt_boxes, gt_labels, pred_boxes, pred_labels, pred_conf, img_name = "annotated", target_dir = "."):
# ============================================================
#  Draw GT + prediction boxes and save annotated image
# ============================================================

    # Work in BGR for OpenCV drawing
    annotated = image_rgb.copy()

    # Draw GT boxes (green)
    for box, cls_id in zip(gt_boxes.tolist(), gt_labels.tolist()):
        gx1, gy1, gx2, gy2 = map(int, box)
        cv2.rectangle(annotated, (gx1, gy1), (gx2, gy2), (255, 0, 0), 2)

        if ID_TO_NAME is not None:
            label_text = f"GT: {ID_TO_NAME.get(cls_id, cls_id)}"
        else:
            label_text = f"GT: {cls_id}"

        cv2.putText(
            annotated,
            label_text,
            (gx1, max(0, gy1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

    # Draw prediction boxes (red)
    for (px1, py1, px2, py2), cls_id, conf in zip(pred_boxes, pred_labels, pred_conf):
        cv2.rectangle(annotated, (px1, py1), (px2, py2), (0, 0, 255), 2)

        if ID_TO_NAME is not None:
            label_text = f"Pred: {ID_TO_NAME.get(cls_id, cls_id)} Det Conf: {conf:.3f}"
        else:
            label_text = f"Pred: {cls_id} Det Conf: {conf:.3f}"

        cv2.putText(
            annotated,
            label_text,
            (px1, max(0, py2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    # Save annotated image
    out_name = Path(img_name).stem + "_annotated.jpg"
    save_path = target_dir / out_name
    cv2.imwrite(str(save_path), annotated)
    print(f"Annotated image saved to: {save_path}")

def main_function():

    print(f"--- EVALUATION PIPELINE ---")
    print(f"YOLO Model Config: NMS_IOU_THRESHOLD: {NMS_IOU_THRESHOLD} \n CONF_THRESH: {CONF_THRESH} \n MAX_DETECTION: {TOP_K} \n")

    print(f"EVAL Config: EVAL_IOU_THRESHOLD: {EVAL_IOU_THRESHOLD} \n")

    print(f"--- LOADING MODELS ---")
    print(f"YOLO_MODEL_FILE_PATH: {YOLO_MODEL_FILE_PATH} \n SWIN_MODEL_FILE_PATH; {SWIN_MODEL_FILE_PATH} \n")

    # --- Load models ---
    yolo_model = YOLO(YOLO_MODEL_FILE_PATH)
    swin_model = SwinForImageClassification.from_pretrained(SWIN_MODEL_FILE_PATH)
    processor = AutoImageProcessor.from_pretrained(SWIN_MODEL_FILE_PATH)
    swin_model.eval()

    print(f"--- COMPLETED LOADING MODELS ---")

    #Create a unique output directory based on the time of each run
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_root_dir = Path(DEST_FOLDER_PATH) / f"run_{run_id}" 
    dest_root_dir.mkdir(parents=True, exist_ok=True)
    results_path = dest_root_dir / "eval_results.csv"

    annotated_img_dir = dest_root_dir / "annotated_images"
    annotated_img_dir.mkdir(parents=True, exist_ok=True)

    missed_detection_img_dir = dest_root_dir / "missed_detections"
    missed_detection_img_dir.mkdir(parents=True, exist_ok=True)

    false_positive_img_dir = dest_root_dir / "false_positives"
    false_positive_img_dir.mkdir(parents=True, exist_ok=True)

    mis_class_img_dir = dest_root_dir / "mis_classification"
    mis_class_img_dir.mkdir(parents=True, exist_ok=True)


    print(f"--- Saving results and debug images to {dest_root_dir} ---\n")

    # --- Storage for final eval ---
    y_true_all = []
    y_pred_all = []

    if not os.path.isdir(INPUT_IMAGE_FOLDER) or not os.path.isdir(INPUT_LABEL_FOLDER):
        
        print(f"Application unable to find image/label folder! image_folder: {INPUT_IMAGE_FOLDER}")
        exit()

    # --- Dataset loop ---
    for img_name in os.listdir(INPUT_IMAGE_FOLDER):
        if not img_name.endswith((".jpg", ".png")):
            continue

        image_path = os.path.join(INPUT_IMAGE_FOLDER, img_name)
        label_path = os.path.join(INPUT_LABEL_FOLDER, os.path.splitext(img_name)[0] + ".txt")
        if not os.path.exists(label_path):
            continue

        # print(f"Loaded image file: {image_path} \n loaded label file: {label_path}")

        image = cv2.imread(image_path)
        img_h, img_w, _ = image.shape

        # --- Load GT ---
        # Loads each instance of target found in image 
        gt_boxes, gt_labels = [], []
        with open(label_path, "r") as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                cls_id = int(parts[0])
                box = yolo_to_xyxy(parts, img_w, img_h)
                gt_boxes.append(box)
                gt_labels.append(cls_id)

                print(f"\n\nLabel Path: {label_path}. Parts: {parts}")
                print(f"yolo converted box: {box}")


        gt_boxes = torch.tensor(gt_boxes)
        gt_labels = torch.tensor(gt_labels)

        # --- Detection + Cropping + Classification ---
        results = yolo_model(
            image,
            conf=CONF_THRESH,   # confidence threshold
            iou=NMS_IOU_THRESHOLD,     # IoU threshold for NMS
            max_det=TOP_K, # max detections per image
        )[0]

        boxes = results.boxes.data  # Tensor: [x1, y1, x2, y2, conf, class_id]

        # print(f"Raw Detection result: {boxes}")

        # --- Confidence filtering ---
        boxes = boxes[boxes[:, 4] >= CONF_THRESH]
        boxes = boxes[boxes[:, 4].argsort(descending=True)]

        # --- Limit to top K ---
        if TOP_K is not None:
            boxes = boxes[:TOP_K]

        pred_boxes, pred_labels, pred_conf = [], [], []

        for det in boxes:
            x1, y1, x2, y2, conf, _ = det.tolist()

            print(f"Individual Detection result: {det.tolist()}")

            x1, y1, x2, y2 = clamp_coords(x1, y1, x2, y2, img_w, img_h)

            crop = image[y1:y2, x1:x2]

            print(f"Cropped image dimensions: {crop.shape}")

            inputs = processor(images=crop, return_tensors="pt")
            with torch.no_grad():
                logits = swin_model(**inputs).logits
                swin_class = logits.argmax(dim=1).item()

            print(f"Prediction bb: {x1} {y1} {x2} {y2}, Conf: {conf:.3f}, Swin Class: {swin_class}")

            pred_boxes.append([x1, y1, x2, y2])
            pred_conf.append(conf)
            pred_labels.append(swin_class)

        # Annotate image with ground truth and pred labels and save image

        annotate_and_save_image(image, gt_boxes, gt_labels, pred_boxes, pred_labels, pred_conf, img_name, annotated_img_dir)

        # --- Match predictions to GT using IoU ---
        # This block contains logic that matches each prediction to each ground truth (gt) label
        # Matches exactly one gt bounding box to one prediction bounding box
        # Match condition: Based on IoU between pred and gt bounding box. IoU > threshold
        # Note: Rmb that these indexes are only for 1 image 
        matched_gt = set() 
        matched_pred = set()
        for pred_idx, (pb, pl) in enumerate(zip(pred_boxes, pred_labels)):
            best_iou = 0
            best_idx = -1

            for idx, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                if idx in matched_gt:
                    continue

                iou = compute_iou(pb, gb.tolist()) 
                if iou >= EVAL_IOU_THRESHOLD and iou > best_iou: 
                    best_iou = iou
                    best_idx = idx

                print(f"--- Matching each prediction to best ground truth bbox by IoU ---")

            #Append the GT label and predicted label to lists used later for classification metrics.
            if best_idx >= 0:
                y_true_all.append(gt_labels[best_idx].item())
                y_pred_all.append(pl)
                matched_gt.add(best_idx)
                matched_pred.add(pred_idx)

                print(f"Best match found for Pred: {pl} -> GT: {gl} IOU: {best_iou}")

                if(pl != gl):
                    print(f"Mis-classification: Prediction label doesnt match with GT label")
                    annotate_and_save_image(image, gt_boxes, gt_labels, pred_boxes, pred_labels, pred_conf, img_name, mis_class_img_dir)
            else:
                print(f"No best match found for Pred: {pl}")
        
        # --- Handle missed detections (False Negatives) ---
        # For gt labels that are not matched, will be deemed as missed detection
        for idx, gt_label in enumerate(gt_labels):
            if idx not in matched_gt:
                y_true_all.append(gt_label.item())   # Ground truth exists
                y_pred_all.append(MISSED_CLASS_ID)   # But no prediction found

                annotate_and_save_image(image, gt_boxes, gt_labels, pred_boxes, pred_labels, pred_conf, img_name, missed_detection_img_dir)

        # --- Handle false positives (extra predictions) ---
        for pred_idx, pl in enumerate(pred_labels):
            if pred_idx not in matched_pred:
                y_true_all.append(MISSED_CLASS_ID)  # No GT object
                y_pred_all.append(pl)               # Model predicted a class

                annotate_and_save_image(image, gt_boxes, gt_labels, pred_boxes, pred_labels, pred_conf, img_name, false_positive_img_dir)

    #Save results into .csv file
    df = pd.DataFrame({
        "y_true": y_true_all,
        "y_pred": y_pred_all
    })
    df.to_csv(results_path, index=False)
    print(f"\nSaved evaluation results to {results_path}")

    # --- Compute Precision, Recall, F1 ---
    ALL_LABELS = list(ID_TO_NAME.keys()) + [MISSED_CLASS_ID]
    TARGET_NAMES = list(ID_TO_NAME.values()) + ["Background"]

    print(classification_report(
        y_true_all,
        y_pred_all,
        labels=ALL_LABELS,
        target_names=TARGET_NAMES,
        zero_division=0
    ))


if __name__ == "__main__":

    main_function()



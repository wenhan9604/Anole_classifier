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
yolo_model = YOLO("./runs/detect/yolov8x_data_v4/weights/best.pt")
swin_model = SwinForImageClassification.from_pretrained("swin-base-patch4-window12-384-finetuned-lizard-v3-swin-base")
processor = AutoImageProcessor.from_pretrained("swin-base-patch4-window12-384-finetuned-lizard-v3-swin-base")
swin_model.eval()

print(f"Models loaded!")

# --- Config ---
dest_folder_path = "./inference"
image_folder = "../Dataset/yolo_training/florida_five_anole_10000_v4/test/images"
label_folder = "../Dataset/yolo_training/florida_five_anole_10000_v4/test/labels"
MISSED_CLASS_ID = 5  # Custom label for missed detections
NUM_CLASSES = 5
IOU_THRESHOLD = 0.15
CONF_THRESH = 0.3
TOP_K = 5           # Max number of boxes to classify (set to None for no limit)

#Create a unique output directory based on the time of each run
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
dest_root_dir = Path(dest_folder_path) / f"run_{run_id}" 
dest_root_dir.mkdir(parents=True, exist_ok=True)
results_path = dest_root_dir / "eval_results.csv"

# --- Helper: Convert YOLO to [x1, y1, x2, y2] ---
def yolo_to_xyxy(yolo_box, img_w, img_h):
    cls, xc, yc, w, h = yolo_box #yolo_box format (normalized) = (class_id, x_center, y_center, width_boundingbox, height_boundingbox)
    xc, yc, w, h = xc * img_w, yc * img_h, w * img_w, h * img_h
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
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
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

# --- Storage for final eval ---
y_true_all = []
y_pred_all = []

if not os.path.isdir(image_folder) or not os.path.isdir(label_folder):
    
    print(f"Application unable to find image/label folder! image_folder: {image_folder}")
    exit()

# --- Dataset loop ---
for img_name in os.listdir(image_folder):
    if not img_name.endswith((".jpg", ".png")):
        continue

    image_path = os.path.join(image_folder, img_name)
    label_path = os.path.join(label_folder, os.path.splitext(img_name)[0] + ".txt")
    if not os.path.exists(label_path):
        continue

    # print(f"Loaded image file: {image_path} \n loaded label file: {label_path}")

    # image = Image.open(image_path).convert("RGB")
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_w, img_h, _ = image.shape

    # print(f"Image size: {image.shape}")

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

    gt_boxes = torch.tensor(gt_boxes)
    gt_labels = torch.tensor(gt_labels)

    # --- Detection + Cropping + Classification ---
    results = yolo_model(image)[0]
    boxes = results.boxes.data  # Tensor: [x1, y1, x2, y2, conf, class_id]

    # print(f"Raw Detection result: {boxes}")

    # --- Confidence filtering ---
    boxes = boxes[boxes[:, 4] >= CONF_THRESH]
    boxes = boxes[boxes[:, 4].argsort(descending=True)]

    # --- Limit to top K ---
    if TOP_K is not None:
        boxes = boxes[:TOP_K]

    pred_boxes, pred_labels = [], []

    for det in boxes:
        x1, y1, x2, y2, conf, _ = det.tolist()

        print(f"Individual Detection result: {det.tolist()}")

        # RT-DETR returns BB coords that are beyond the image coord. Need to set it to be 0 < coord < 1.0
        # x1 = max(0, x1)
        # y1 = max(0, y1)
        # x2 = min(1.0, x2)
        # y2 = min(1.0, y2)

        #Absolute coord because RT-DETR returns in normalized value
        # x1 = x1 * img_w
        # y1 = y1 * img_h
        # x2 = x2 * img_w
        # y2 = y2 * img_h

        x1 = np.uint16(math.ceil(x1))
        y1 = np.uint16(math.ceil(y1))
        x2 = np.uint16(x2)
        y2 = np.uint16(y2)

        # print("tl_x")
        # print(tl_x)

        crop = image[y1:y2, x1:x2]

        # crop = image.crop((x1, y1, x2, y2))

        print(f"Cropped image dimensions: {crop.shape}")

        inputs = processor(images=crop, return_tensors="pt")
        with torch.no_grad():
            logits = swin_model(**inputs).logits
            swin_class = logits.argmax(dim=1).item()

        pred_boxes.append([x1, y1, x2, y2])
        pred_labels.append(swin_class)

    # --- Match predictions to GT using IoU ---
    # This block contains logic that matches each prediction to each ground truth (gt) label
    # Matches exactly one gt bounding box to one prediction bounding box
    # Match condition: Based on IoU between pred and gt bounding box. IoU > threshold
    # Note: Rmb that these indexes are only for 1 image 
    matched_gt = set() 
    for pb, pl in zip(pred_boxes, pred_labels):
        best_iou = 0
        best_idx = -1
        for idx, gb in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            iou = compute_iou(pb, gb.tolist()) 
            if iou >= IOU_THRESHOLD and iou > best_iou: 
                best_iou = iou
                best_idx = idx

        #Append the GT label and predicted label to lists used later for classification metrics.
        if best_idx >= 0:
            y_true_all.append(gt_labels[best_idx].item())
            y_pred_all.append(pl)
            matched_gt.add(best_idx)
    
    # --- Handle missed detections (False Negatives) ---
    # For gt labels that are not matched, will be deemed as missed detection
    for idx, gt_label in enumerate(gt_labels):
        if idx not in matched_gt:
            y_true_all.append(gt_label.item())   # Ground truth exists
            y_pred_all.append(MISSED_CLASS_ID)   # But no prediction found


#Save results into .csv file
df = pd.DataFrame({
    "y_true": y_true_all,
    "y_pred": y_pred_all
})
df.to_csv(results_path, index=False)
print(f"Saved evaluation results to {results_path}")

# --- Compute Precision, Recall, F1 ---
print(f"\nClassification Report (IoU ≥ {IOU_THRESHOLD} matched):")
print(classification_report(
    y_true_all,
    y_pred_all,
    labels=list(range(NUM_CLASSES)),  # exclude class 6
    target_names=[f"Class {i}" for i in range(NUM_CLASSES)],
    zero_division=0
))

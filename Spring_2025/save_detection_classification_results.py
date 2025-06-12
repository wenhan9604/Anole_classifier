import os
import torch
import shutil
import pandas as pd
from PIL import Image
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from ultralytics import YOLO
from transformers import AutoImageProcessor, SwinForImageClassification
from sklearn.metrics import classification_report

# --- CONFIG ---
NUM_CLASSES = 5
MISSED_CLASS_ID = 5
CONF_THRESH = 0.0
IOU_THRESHOLD = 0.5
TOP_K = 5

class_names = [f"Class {i}" for i in range(NUM_CLASSES)]

image_folder = "Dataset/YOLO_training/florida_five_anole_10000/test/images"
label_folder = "Dataset/YOLO_training/florida_five_anole_10000/test/labels"
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
base_output = Path(f"Spring_2025/eval_outputs/run_{run_id}")
base_output.mkdir(parents=True, exist_ok=True)

folders = [
    "detection_correct", "detection_incorrect",
    "classification_correct", "classification_incorrect",
    "detection_incorrect_visual", "classification_incorrect_visual"
]
for folder in folders:
    (base_output / folder).mkdir(parents=True, exist_ok=True)

results_path = base_output / "eval_results.csv"
det_fail_path = base_output / "failed_detections.csv"
cls_fail_path = base_output / "failed_classifications.csv"

# --- LOAD MODELS ---
yolo_model = YOLO("Spring_2025/models/train_yolov8n_v2/weights/best.pt")
swin_model = SwinForImageClassification.from_pretrained(
    "Spring_2025/models/swin-base-patch4-window12-384-finetuned-lizard-class-swin-base/checkpoint-352",
    local_files_only=True
)
processor = AutoImageProcessor.from_pretrained(
    "Spring_2025/models/swin-base-patch4-window12-384-finetuned-lizard-class-swin-base/checkpoint-352",
    local_files_only=True
)
swin_model.eval()

# --- UTILS ---
def yolo_to_xyxy(yolo_box, img_w, img_h):
    cls, xc, yc, w, h = yolo_box
    xc, yc, w, h = xc * img_w, yc * img_h, w * img_w, h * img_h
    return [xc - w/2, yc - h/2, xc + w/2, yc + h/2]

def compute_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

# --- EVAL LOOP ---
y_true_all, y_pred_all = [], []
failed_detections, failed_classifications = [], []

for img_name in os.listdir(image_folder):
    if not img_name.endswith(".jpg"):
        continue

    img_path = os.path.join(image_folder, img_name)
    label_path = os.path.join(label_folder, img_name.replace(".jpg", ".txt"))
    if not os.path.exists(label_path):
        continue

    image = Image.open(img_path).convert("RGB")
    img_w, img_h = image.size

    # Load GT boxes
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

    # --- Detection ---
    results = yolo_model(img_path)[0]
    if results.boxes is None or len(results.boxes) == 0:
        print(f"[FAIL] No detection for {img_name}")
        shutil.copy(img_path, base_output / "detection_incorrect" / img_name)
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.set_title(f"No Detection | True: {gt_labels.tolist()}")
        ax.axis("off")
        fig.savefig(base_output / "detection_incorrect_visual" / img_name)
        plt.close(fig)
        for label in gt_labels.tolist():
            y_true_all.append(label)
            y_pred_all.append(MISSED_CLASS_ID)
        failed_detections.append({
            "image": img_name,
            "reason": "no detection",
            "gt_count": len(gt_labels)
        })
        continue

    if len(results.boxes) > 1:
        print(f"[FAIL] Multiple detections for {img_name}")
        shutil.copy(img_path, base_output / "detection_incorrect" / img_name)
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.set_title(f"Multiple Detections | True: {gt_labels.tolist()}")
        ax.axis("off")
        fig.savefig(base_output / "detection_incorrect_visual" / img_name)
        plt.close(fig)
        for label in gt_labels.tolist():
            y_true_all.append(label)
            y_pred_all.append(MISSED_CLASS_ID)
        failed_detections.append({
            "image": img_name,
            "reason": "multiple detections",
            "gt_count": len(gt_labels)
        })
        continue

    shutil.copy(img_path, base_output / "detection_correct" / img_name)

    # --- Classification ---
    pred_boxes = results.boxes.xyxy.cpu().numpy()
    pred_box = pred_boxes[0]
    x1, y1, x2, y2 = map(int, pred_box)
    cropped = image.crop((x1, y1, x2, y2))

    inputs = processor(images=cropped, return_tensors="pt")
    with torch.no_grad():
        logits = swin_model(**inputs).logits
        pred_class = logits.argmax().item()

    best_iou = 0
    best_idx = -1
    for idx, gt_box in enumerate(gt_boxes):
        iou = compute_iou(pred_box, gt_box.tolist())
        if iou >= IOU_THRESHOLD and iou > best_iou:
            best_iou = iou
            best_idx = idx

    if best_idx >= 0:
        true_class = gt_labels[best_idx].item()
        y_true_all.append(true_class)
        y_pred_all.append(pred_class)
        if pred_class != true_class:
            cropped.save(base_output / "classification_incorrect" / f"{img_name}")
            fig, ax = plt.subplots()
            ax.imshow(image)
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                       edgecolor='red', linewidth=2, fill=False))
            ax.text(x1, y1 - 10,
                    f"Pred: {class_names[pred_class]} | True: {class_names[true_class]}",
                    color='white', fontsize=12,
                    bbox=dict(facecolor='red', alpha=0.8))
            ax.axis("off")
            fig.savefig(base_output / "classification_incorrect_visual" / f"{img_name}")
            plt.close(fig)
            failed_classifications.append({
                "image": img_name,
                "predicted": class_names[pred_class],
                "true": class_names[true_class]
            })
        else:
            cropped.save(base_output / "classification_correct" / f"{img_name}")
    else:
        for label in gt_labels.tolist():
            y_true_all.append(label)
            y_pred_all.append(MISSED_CLASS_ID)
        failed_detections.append({
            "image": img_name,
            "reason": "low IoU",
            "gt_count": len(gt_labels)
        })

# --- Save logs ---
pd.DataFrame({
    "y_true": y_true_all,
    "y_pred": y_pred_all
}).to_csv(results_path, index=False)

pd.DataFrame(failed_detections).to_csv(det_fail_path, index=False)
pd.DataFrame(failed_classifications).to_csv(cls_fail_path, index=False)

print(f"\nSaved results to {results_path}")
print(f"Saved detection failures to {det_fail_path}")
print(f"Saved classification failures to {cls_fail_path}")
print("\nClassification Report:")
print(classification_report(
    y_true_all, y_pred_all,
    labels=list(range(NUM_CLASSES)),
    target_names=class_names,
    zero_division=0
))

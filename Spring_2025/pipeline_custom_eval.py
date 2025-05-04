import os
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import AutoImageProcessor, SwinForImageClassification
from sklearn.metrics import classification_report

# --- Load models ---
yolo_model = YOLO("./runs/detect/train_yolov8n_v2/weights/best.pt")
swin_model = SwinForImageClassification.from_pretrained("swin-base-patch4-window12-384-finetuned-lizard-class-swin-base")
processor = AutoImageProcessor.from_pretrained("swin-base-patch4-window12-384-finetuned-lizard-class-swin-base")
swin_model.eval()

# --- Config ---
image_folder = "../Dataset/yolo_training/florida_five_anole_10000/test/images"
label_folder = "../Dataset/yolo_training/florida_five_anole_10000/test/labels"
MISSED_CLASS_ID = 5  # Custom label for missed detections
NUM_CLASSES = 5
IOU_THRESHOLD = 0.5

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

# --- Dataset loop ---
for img_name in os.listdir(image_folder):
    if not img_name.endswith((".jpg", ".png")):
        continue

    image_path = os.path.join(image_folder, img_name)
    label_path = os.path.join(label_folder, os.path.splitext(img_name)[0] + ".txt")
    if not os.path.exists(label_path):
        continue

    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size

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

    # --- Detection + Classification ---
    results = yolo_model(image)[0] #Getting 1 result with highest confidence only?
    pred_boxes, pred_labels = [], []

    for det in results.boxes.data:
        x1, y1, x2, y2, conf, _ = det.tolist()
        crop = image.crop((x1, y1, x2, y2))
        inputs = processor(images=crop, return_tensors="pt")
        with torch.no_grad():
            logits = swin_model(**inputs).logits
            swin_class = logits.argmax(dim=1).item()

        pred_boxes.append([x1, y1, x2, y2])
        pred_labels.append(swin_class)

    # --- Match predictions to GT using IoU ---
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

        if best_idx >= 0:
            y_true_all.append(gt_labels[best_idx].item())
            y_pred_all.append(pl)
            matched_gt.add(best_idx)
    
    # --- Handle missed detections (False Negatives) ---
    for idx, gt_label in enumerate(gt_labels):
        if idx not in matched_gt:
            y_true_all.append(gt_label.item())   # Ground truth exists
            y_pred_all.append(MISSED_CLASS_ID)   # But no prediction found


# --- Compute Precision, Recall, F1 ---
print("\nClassification Report (IoU ≥ 0.5 matched):")
print(classification_report(y_true_all, y_pred_all, labels=list(range(NUM_CLASSES)) + [MISSED_CLASS_ID], digits=4))

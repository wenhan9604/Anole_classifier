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
import matplotlib.pyplot as plt
from collections import defaultdict

# --- Evaluation Config ---
DEST_FOLDER_PATH = "./inference"
INPUT_IMAGE_FOLDER = "../Dataset/yolo_training/florida_five_anole_10000_v4/test/images"
INPUT_LABEL_FOLDER = "../Dataset/yolo_training/florida_five_anole_10000_v4/test/labels"
MISSED_CLASS_ID = 5  # Custom label for missed detections

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

def voc_ap(rec, prec):
    """
    Compute AP using the VOC 2010+ style:
    integration over the precision–recall curve.
    rec, prec are numpy arrays.
    """
    # Append sentinel values at both ends
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    # Make precision monotonically decreasing
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    # Compute area under curve
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap

def compute_pr_for_class(cls_id, iou_thr, gt_by_cls, pred_by_cls):
    """
    Compute precision-recall arrays for a single class at a given IoU threshold.
    Returns: rec, prec (both numpy arrays of same length)
    """
    gts = gt_by_cls.get(cls_id, [])
    preds = pred_by_cls.get(cls_id, [])

    npos = len(gts)
    if npos == 0 or len(preds) == 0:
        # No GTs or no predictions for this class => undefined PR
        return np.array([0.0]), np.array([0.0])

    # Map image_id -> list of GT indices for this class
    gt_img_map = {}
    for i, g in enumerate(gts):
        img_id = g["image_id"]
        gt_img_map.setdefault(img_id, [])
        gt_img_map[img_id].append(i)

    # Track which GT boxes are already matched
    gt_used = [False] * len(gts)

    # Sort predictions by confidence descending
    preds_sorted = sorted(preds, key=lambda x: x["conf"], reverse=True)

    tp = np.zeros(len(preds_sorted))
    fp = np.zeros(len(preds_sorted))

    for i, p in enumerate(preds_sorted):
        img_id = p["image_id"]
        box_p = p["box"]

        if img_id not in gt_img_map:
            fp[i] = 1
            continue

        best_iou = 0.0
        best_gt_idx = -1
        for gt_idx in gt_img_map[img_id]:
            if gt_used[gt_idx]:
                continue
            box_g = gts[gt_idx]["box"]
            iou = compute_iou(box_p, box_g)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_iou >= iou_thr and best_gt_idx >= 0:
            tp[i] = 1
            gt_used[best_gt_idx] = True
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)

    eps = 1e-8
    rec = tp_cum / (npos + eps)
    prec = tp_cum / (tp_cum + fp_cum + eps)

    return rec, prec


def annotate_and_save_image(image_rgb, gt_boxes, gt_labels, pred_boxes, pred_labels, pred_conf, img_name = "annotated", target_dir = "."):
# ============================================================
#  Draw GT + prediction boxes and save annotated image
# ============================================================

    # Work in BGR for OpenCV drawing
    annotated = image_rgb.copy()
    annotated = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR) #cv2.imwrite() requires BGR

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

def save_image(img_rgb, img_name = "annotated", target_dir = "."):

    # Work in BGR for OpenCV drawing
    img_copy = img_rgb.copy()
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR) #cv2.imwrite() requires BGR

    out_name = Path(img_name).stem + "_annotated.jpg"
    save_path = target_dir / out_name
    cv2.imwrite(str(save_path), img_copy)
    print(f"Cropped image saved to: {save_path}")

# eval_iou_thresh - Affects f1-score, precision, recall and confusion matrix. Standardize at 0.5, aligns with PASCAL VOC
def evaluate_performance(yolo_model_file_path=None, swin_model_file_path=None, classification_conf_thresh=0.5, mAP_conf_thresh = 0.001, nms_iou_thresh=0.7, eval_iou_thresh = 0.5, top_k=5):
    """
    Parameters:
        classification_conf_thresh (float): Filters predictions lower than this threshold. Affects calculation for confusion matrix; f1-score, precision, recall  
        mAP_conf_thresh (float): Filters prediction lower than this threshold. Affects calculation for mAP. Use 0.001 
        nms_iou_thresh (float): Controls how aggressively overlapping predictions are suppressed. Rec. range: 0.5–0.7
        eval_iou_thresh (float): Filters for pred and GT bbox overlap. Affects f1-score, precision, recall and confusion matrix. Set at 0.5, aligns with PASCAL VOC 
        top_k (int): Defines the number of prediction from detection model for 1 image.

    """

    if (yolo_model_file_path is None or not os.path.exists(yolo_model_file_path)):
        print(f"yolo model file path cannot be found. yolo_model_file_path: {yolo_model_file_path}")
        return

    print(f"--- EVALUATION PIPELINE ---")
    print(f"YOLO Model Config: NMS_IOU_THRESHOLD: {nms_iou_thresh} \n CONF_THRESH: {classification_conf_thresh} \n MAX_DETECTION: {top_k} \n")

    print(f"EVAL Config: EVAL_IOU_THRESHOLD: {eval_iou_thresh} \n")

    print(f"--- LOADING MODELS ---")
    print(f"YOLO_MODEL_FILE_PATH: {yolo_model_file_path} \n")

    if (swin_model_file_path):
        print(f"SWIN_MODEL_FILE_PATH; {swin_model_file_path} \n")


    # --- Load models ---
    yolo_model = YOLO(yolo_model_file_path)

    if (swin_model_file_path):
        swin_model = SwinForImageClassification.from_pretrained(swin_model_file_path)
        processor = AutoImageProcessor.from_pretrained(swin_model_file_path)
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

    if (swin_model_file_path):
        cropped_img_dir = dest_root_dir / "cropped_img"
        cropped_img_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Saving results and debug images to {dest_root_dir} ---\n")

    # --- Storage for final eval ---
    y_true_all = []
    y_pred_all = []

    # --- Storage for detection mAP computation ---
    # Group all GTs and predictions by class across the whole dataset.
    gt_by_cls = defaultdict(list)    # cls_id -> list of {"image_id", "box"}
    pred_by_cls = defaultdict(list)  # cls_id -> list of {"image_id", "box", "conf"}

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

        print(f"\n\n\n --- Analyzing Image --- \nLoaded image file: {image_path} \nLoaded label file: {label_path}")

        image_BGR = cv2.imread(image_path)
        image_RGB = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
        img_h, img_w, _ = image_RGB.shape

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

                print(f"\nLabel File content (GT): {parts}")
                print(f"GT Box (yolo to xyxy): {box}")

                # For mAP: store GT per class
                gt_by_cls[cls_id].append({
                    "image_id": img_name,
                    "box": np.array(box, dtype=float)
                })

        gt_boxes = torch.tensor(gt_boxes)
        gt_labels = torch.tensor(gt_labels)

        print(f"\n--- Pipeline inference ---")

        # --- Detection + Cropping + Classification ---
        results = yolo_model.predict(
            image_RGB,
            conf=classification_conf_thresh,   # confidence threshold
            iou=nms_iou_thresh,     # IoU threshold for NMS
            max_det=top_k, # max detections per image
            agnostic_nms=True
        )[0]

        boxes = results.boxes.data  # Tensor: [x1, y1, x2, y2, conf, class_id]

        print(f"Raw Detection result: {boxes}")

        # --- Confidence filtering ---
        boxes = boxes[boxes[:, 4] >= classification_conf_thresh]
        boxes = boxes[boxes[:, 4].argsort(descending=True)]

        # --- Limit to top K ---
        if top_k is not None:
            boxes = boxes[:top_k]

        pred_boxes, pred_labels, pred_conf = [], [], []

        for det in boxes:

            pred_label = None
            x1, y1, x2, y2, conf, yolo_pred_label = det.tolist()

            print(f"Individual Detection result: {det.tolist()}")

            x1, y1, x2, y2 = clamp_coords(x1, y1, x2, y2, img_w, img_h)

            if (swin_model_file_path):

                crop = image_RGB[y1:y2, x1:x2]

                save_image(crop, img_name, cropped_img_dir)

                inputs = processor(images=crop, return_tensors="pt")
                with torch.no_grad():
                    logits = swin_model(**inputs).logits
                    pred_label = logits.argmax(dim=1).item()

            else:
                pred_label = yolo_pred_label

            print(f"Prediction bbox: {x1} {y1} {x2} {y2}, Conf: {conf:.3f}, Pred_label: {ID_TO_NAME[pred_label]}")

            pred_boxes.append([x1, y1, x2, y2])
            pred_conf.append(conf)
            pred_labels.append(pred_label)

            # For mAP: store prediction per class
            pred_by_cls[int(pred_label)].append({
                "image_id": img_name,
                "box": np.array([x1, y1, x2, y2], dtype=float),
                "conf": float(conf)
            })

        # Annotate image with ground truth and pred labels and save image

        annotate_and_save_image(image_RGB, gt_boxes, gt_labels, pred_boxes, pred_labels, pred_conf, img_name, annotated_img_dir)

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
                if iou >= eval_iou_thresh and iou > best_iou: 
                    best_iou = iou
                    best_idx = idx

                print(f"\n--- Matching each prediction to best ground truth bbox by IoU ---")

            #Append the GT label and predicted label to lists used later for classification metrics.
            if best_idx >= 0:
                y_true_all.append(gt_labels[best_idx].item())
                y_pred_all.append(pl)
                matched_gt.add(best_idx)
                matched_pred.add(pred_idx)

                print(f"Best match found for Pred: {ID_TO_NAME[int(pl)]} -> GT: {ID_TO_NAME[int(gl)]} IOU: {best_iou}")

                if(pl != gl):
                    print(f"Mis-classification: Prediction label doesnt match with GT label")
                    annotate_and_save_image(image_RGB, gt_boxes, gt_labels, pred_boxes, pred_labels, pred_conf, img_name, mis_class_img_dir)
            else:
                print(f"No best match found for Pred: {ID_TO_NAME[int(pl)]}")
        
        # --- Handle missed detections (False Negatives) ---
        # For gt labels that are not matched, will be deemed as missed detection
        for idx, gt_label in enumerate(gt_labels):
            if idx not in matched_gt:
                y_true_all.append(gt_label.item())   # Ground truth exists
                y_pred_all.append(MISSED_CLASS_ID)   # But no prediction found

                annotate_and_save_image(image_RGB, gt_boxes, gt_labels, pred_boxes, pred_labels, pred_conf, img_name, missed_detection_img_dir)

        # --- Handle false positives (extra predictions) ---
        for pred_idx, pl in enumerate(pred_labels):
            if pred_idx not in matched_pred:
                y_true_all.append(MISSED_CLASS_ID)  # No GT object
                y_pred_all.append(pl)               # Model predicted a class

                annotate_and_save_image(image_RGB, gt_boxes, gt_labels, pred_boxes, pred_labels, pred_conf, img_name, false_positive_img_dir)

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

    # ----------------------------------------------------
    #  Detection mAP evaluation (COCO-style IoU thresholds)
    # ----------------------------------------------------
    print("\n--- Detection mAP Evaluation ---")

    # IoU thresholds for COCO-style eval: 0.50:0.95 with step 0.05
    iou_thresholds = [round(t, 2) for t in np.arange(0.50, 0.96, 0.05)]

    aps_by_thr = {thr: {} for thr in iou_thresholds}
    mAP_by_thr = {}

    # Loop over IoU thresholds
    for thr in iou_thresholds:
        print(f"\nEvaluating AP at IoU = {thr:.2f}")
        for cls_id, cls_name in ID_TO_NAME.items():
            gts = gt_by_cls.get(cls_id, [])
            preds = pred_by_cls.get(cls_id, [])

            npos = len(gts)
            if npos == 0:
                print(f"  Class '{cls_name}' (id={cls_id}): no GT boxes, skipping AP.")
                aps_by_thr[thr][cls_id] = np.nan
                continue

            # Map: image_id -> list of GT indices for this class
            gt_img_map = {}
            for i, g in enumerate(gts):
                img_id = g["image_id"]
                gt_img_map.setdefault(img_id, [])
                gt_img_map[img_id].append(i)

            # Track which GT boxes are already matched
            gt_used = [False] * len(gts)

            # Sort predictions by confidence (descending)
            preds_sorted = sorted(preds, key=lambda x: x["conf"], reverse=True)

            tp = np.zeros(len(preds_sorted))
            fp = np.zeros(len(preds_sorted))

            for i, p in enumerate(preds_sorted):
                img_id = p["image_id"]
                box_p = p["box"]

                if img_id not in gt_img_map:
                    # No GT of this class in this image
                    fp[i] = 1
                    continue

                best_iou = 0.0
                best_gt_idx = -1
                for gt_idx in gt_img_map[img_id]:
                    if gt_used[gt_idx]:
                        continue
                    box_g = gts[gt_idx]["box"]
                    iou = compute_iou(box_p, box_g)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= thr and best_gt_idx >= 0:
                    # True positive
                    tp[i] = 1
                    gt_used[best_gt_idx] = True
                else:
                    # False positive
                    fp[i] = 1

            # Cumulative sums
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(fp)

            eps = 1e-8
            rec = tp_cum / (npos + eps)
            prec = tp_cum / (tp_cum + fp_cum + eps)

            ap = voc_ap(rec, prec)
            aps_by_thr[thr][cls_id] = ap
            print(f"  AP for class '{cls_name}' (id={cls_id}) at IoU={thr:.2f}: {ap:.4f}")

        # mAP at this IoU = mean over valid classes
        ap_values = [v for v in aps_by_thr[thr].values() if not np.isnan(v)]
        if len(ap_values) > 0:
            mAP_by_thr[thr] = float(np.mean(ap_values))
        else:
            mAP_by_thr[thr] = 0.0

    # Extract key metrics
    mAP_50 = mAP_by_thr.get(0.50, 0.0)
    mAP_75 = mAP_by_thr.get(0.75, 0.0)
    mAP_50_95 = float(np.mean(list(mAP_by_thr.values()))) if len(mAP_by_thr) > 0 else 0.0

    print("\n--- Summary mAP ---")
    print(f"mAP@0.50      : {mAP_50:.4f}")
    print(f"mAP@0.75      : {mAP_75:.4f}")
    print(f"mAP@0.50:0.95 : {mAP_50_95:.4f}")

    # ----------------------------------------------------
    #  Precision-Recall Curves at a chosen IoU threshold
    # ----------------------------------------------------
    pr_iou = 0.50   # or 0.75, or EVAL_IOU_THRESHOLD, up to you

    print(f"\n--- Plotting Precision-Recall Curves at IoU = {pr_iou:.2f} ---")

    plt.figure(figsize=(7, 6))

    for cls_id, cls_name in ID_TO_NAME.items():
        rec, prec = compute_pr_for_class(cls_id, pr_iou, gt_by_cls, pred_by_cls)

        # Some classes may have trivial curves; skip if no useful data
        if len(rec) <= 1 and len(prec) <= 1:
            print(f"  Class '{cls_name}' has insufficient data for PR curve, skipping.")
            continue

        plt.plot(rec, prec, label=f"{cls_name} (id={cls_id})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curves (IoU = {pr_iou:.2f})")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True)
    plt.legend()

    pr_save_path = dest_root_dir / f"pr_curves_iou_{int(pr_iou*100):02d}.png"
    plt.tight_layout()
    plt.savefig(pr_save_path, dpi=200)
    plt.close()

    print(f"Saved Precision-Recall curves to: {pr_save_path}")




if __name__ == "__main__":

    evaluate_performance(yolo_model_file_path="./runs/detect/train22_yolov8x_dataset_v4/weights/best.pt",
    swin_model_file_path="swin-base-patch4-window12-384-finetuned-lizard-v3-swin-base", nms_iou_thresh=0.25,classification_conf_thresh=0.5, top_k=5)



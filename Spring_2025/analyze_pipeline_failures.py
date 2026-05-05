import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
import cv2
from transformers import AutoImageProcessor, SwinForImageClassification
from ultralytics import YOLO


# --- Defaults & Constants ---
NUM_CLASSES = 5
MISSED_CLASS_ID = NUM_CLASSES  # Used when detection fails
CLASS_NAMES = [
    "Bark Anole",
    "Brown Anole",
    "Crested Anole",
    "Green Anole",
    "Knight Anole",
]


class SwinClassificationWrapper(torch.nn.Module):
    """Wrap Swin model so GradCAM receives logits tensor."""

    def __init__(self, model: SwinForImageClassification):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits


# --- Geometry helpers ---
def yolo_to_xyxy(yolo_box: Sequence[float], img_w: int, img_h: int) -> List[float]:
    """Convert YOLO normalized box to [x1, y1, x2, y2] absolute coordinates."""
    _, xc, yc, w, h = yolo_box
    xc, yc, w, h = xc * img_w, yc * img_h, w * img_w, h * img_h
    return [xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2]


def compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_w = max(0.0, x_b - x_a)
    inter_h = max(0.0, y_b - y_a)
    inter_area = inter_w * inter_h
    if inter_area == 0.0:
        return 0.0

    area_a = max(0.0, (box_a[2] - box_a[0])) * max(0.0, (box_a[3] - box_a[1]))
    area_b = max(0.0, (box_b[2] - box_b[0])) * max(0.0, (box_b[3] - box_b[1]))
    if area_a <= 0 or area_b <= 0:
        return 0.0
    return inter_area / (area_a + area_b - inter_area)


def ensure_rgb(img: Image.Image) -> Image.Image:
    return img if img.mode == "RGB" else img.convert("RGB")


def draw_box(ax, box: Sequence[float], color: str, label: Optional[str] = None, linewidth: int = 2) -> None:
    x1, y1, x2, y2 = box
    rect = plt.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        fill=False,
        color=color,
        linewidth=linewidth,
    )
    ax.add_patch(rect)
    if label:
        ax.text(
            x1,
            max(y1 - 6, 2),
            label,
            fontsize=10,
            color="white",
            bbox=dict(facecolor=color, alpha=0.7, pad=2),
        )


def save_detection_visual(
    image: Image.Image,
    gt_boxes: Sequence[Sequence[float]],
    unmatched_gt_indices: Sequence[int],
    pred_boxes: Sequence[Sequence[float]],
    pred_confs: Sequence[float],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots()
    ax.imshow(image)
    unmatched_set = set(unmatched_gt_indices)
    for idx, box in enumerate(gt_boxes):
        color = "yellow" if idx in unmatched_set else "lime"
        suffix = " (missed)" if idx in unmatched_set else ""
        draw_box(ax, box, color=color, label=f"GT #{idx}{suffix}")

    for idx, (box, conf) in enumerate(zip(pred_boxes, pred_confs)):
        draw_box(ax, box, color="red", label=f"Pred #{idx} | {conf:.2f}")

    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_classification_visuals(
    image: Image.Image,
    gt_boxes: Sequence[Sequence[float]],
    gt_labels: Sequence[int],
    pred_boxes: Sequence[Sequence[float]],
    pred_classes: Sequence[int],
    matched_pairs: Sequence[Tuple[int, int]],
    image_out_path: Path,
    highlight_pair: Optional[Tuple[int, int]] = None,
) -> None:
    fig, ax = plt.subplots()
    ax.imshow(image)

    # Draw GT boxes
    for gt_idx, gt_box in enumerate(gt_boxes):
        gt_name = CLASS_NAMES[gt_labels[gt_idx]] if gt_labels[gt_idx] < len(CLASS_NAMES) else f"Class {gt_labels[gt_idx]}"
        draw_box(ax, gt_box, color="lime", label=f"GT {gt_idx}: {gt_name}")

    # Draw matched predicted boxes
    for pred_idx, gt_idx in matched_pairs:
        pred_box = pred_boxes[pred_idx]
        pred_label = pred_classes[pred_idx]
        pred_name = CLASS_NAMES[pred_label] if pred_label < len(CLASS_NAMES) else f"Class {pred_label}"
        gt_label = gt_labels[gt_idx]
        gt_name = CLASS_NAMES[gt_label] if gt_label < len(CLASS_NAMES) else f"Class {gt_label}"
        color = "red" if highlight_pair and highlight_pair[0] == pred_idx and highlight_pair[1] == gt_idx else "orange"
        draw_box(
            ax,
            pred_box,
            color=color,
            label=f"Pred #{pred_idx}: {pred_name}\nGT #{gt_idx}: {gt_name}",
        )

    ax.axis("off")
    fig.tight_layout()
    fig.savefig(image_out_path)
    plt.close(fig)


def swin_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    """Reshape Swin token output (B, N, C) to (B, C, H, W) for GradCAM."""
    if tensor.ndim != 3:
        raise ValueError(f"Expected tensor with 3 dims (B, N, C), got {tensor.shape}")
    batch, num_tokens, channels = tensor.shape
    side = int(math.sqrt(num_tokens))
    if side * side != num_tokens:
        raise ValueError(f"Token count {num_tokens} is not a perfect square, cannot reshape to grid.")
    reshaped = tensor.view(batch, side, side, channels).permute(0, 3, 1, 2).contiguous()
    return reshaped


def generate_gradcam_overlay(
    gradcam_model: torch.nn.Module,
    target_layer: torch.nn.Module,
    processor: AutoImageProcessor,
    crop: Image.Image,
    target_class: int,
    use_cuda: bool,
) -> Image.Image:
    crop = ensure_rgb(crop)
    inputs = processor(images=crop, return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    if use_cuda:
        pixel_values = pixel_values.to("cuda")

    from pytorch_grad_cam import GradCAM  # type: ignore[import-not-found]
    from pytorch_grad_cam.utils.image import show_cam_on_image  # type: ignore[import-not-found]
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget  # type: ignore[import-not-found]

    with GradCAM(
        model=gradcam_model,
        target_layers=[target_layer],
        reshape_transform=swin_reshape_transform,
    ) as cam:
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = cam(input_tensor=pixel_values, targets=targets)[0]

    crop_np = np.array(crop).astype(np.float32) / 255.0
    cam_resized = cv2.resize(grayscale_cam, (crop_np.shape[1], crop_np.shape[0]))
    heatmap = show_cam_on_image(crop_np, cam_resized, use_rgb=True)
    return Image.fromarray(heatmap)


# --- Main evaluation ---
def evaluate(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir)
    if not image_dir.is_absolute():
        image_dir = (repo_root / image_dir).resolve()
    if not label_dir.is_absolute():
        label_dir = (repo_root / label_dir).resolve()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (repo_root / output_dir).resolve()
    output_dir = output_dir / f"run_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    detection_fail_vis_dir = output_dir / "detection_failures_visual"
    detection_fail_vis_dir.mkdir(exist_ok=True)

    classification_crop_dir = output_dir / "classification_failures_crops"
    classification_crop_dir.mkdir(exist_ok=True)

    classification_heatmap_dir = output_dir / "classification_failures_heatmaps"
    classification_heatmap_dir.mkdir(exist_ok=True)

    classification_original_dir = output_dir / "classification_failures_original"
    classification_original_dir.mkdir(exist_ok=True)

    eval_results_path = output_dir / "eval_results.csv"
    det_failures_path = output_dir / "failed_detections.csv"
    det_outputs_path = output_dir / "detection_outputs.csv"
    cls_failures_path = output_dir / "failed_classifications.csv"

    # Models
    yolo_model = YOLO(args.yolo_weights)
    swin_model = SwinForImageClassification.from_pretrained(
        args.swin_model_dir,
        local_files_only=True,
    )
    processor = AutoImageProcessor.from_pretrained(
        args.swin_model_dir,
        local_files_only=True,
    )

    swin_model.eval()

    use_cuda = torch.cuda.is_available() and args.use_cuda
    if use_cuda:
        swin_model.to("cuda")

    gradcam_model = SwinClassificationWrapper(swin_model)
    if use_cuda:
        gradcam_model = gradcam_model.cuda()

    # Use the downsampled features before pooling (better spatial support)
    if args.gradcam_stage_index < 0 or args.gradcam_stage_index >= len(gradcam_model.model.swin.encoder.layers):
        raise ValueError(
            f"gradcam_stage_index {args.gradcam_stage_index} is out of range "
            f"(0 – {len(gradcam_model.model.swin.encoder.layers) - 1})."
        )
    target_stage = gradcam_model.model.swin.encoder.layers[args.gradcam_stage_index]
    # Optionally adjust target layer to use the attention output for more focused CAMs
    if hasattr(target_stage.blocks[-1], "layernorm_before"):
        target_layer = target_stage.blocks[-1].layernorm_before
    else:
        # Fallback in case architecture differs
        target_layer = target_stage.blocks[-1]

    y_true_all: List[int] = []
    y_pred_all: List[int] = []

    detection_failures: List[Dict] = []
    detection_outputs: List[Dict] = []
    classification_failures: List[Dict] = []

    image_names = [n for n in sorted(os.listdir(image_dir)) if n.lower().endswith((".jpg", ".png", ".jpeg"))]
    for image_name in image_names:
        image_path = image_dir / image_name
        label_path = label_dir / f"{Path(image_name).stem}.txt"
        if not label_path.exists():
            continue

        image = ensure_rgb(Image.open(image_path))
        img_w, img_h = image.size

        # Load ground-truth boxes/labels
        gt_boxes: List[List[float]] = []
        gt_labels: List[int] = []
        with open(label_path, "r") as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                if len(parts) != 5:
                    continue
                cls_id = int(parts[0])
                gt_boxes.append(yolo_to_xyxy(parts, img_w, img_h))
                gt_labels.append(cls_id)

        gt_boxes_tensor = torch.tensor(gt_boxes) if gt_boxes else torch.zeros((0, 4))

        if args.species_id_override is not None:
            gt_species_labels: List[int] = [args.species_id_override] * len(gt_labels)
        else:
            gt_species_labels = list(gt_labels)

        # Run detector
        det_results = yolo_model(image_path)[0]
        det_boxes = det_results.boxes
        if det_boxes is None or det_boxes.data.numel() == 0:
            det_data = torch.zeros((0, 6))
        else:
            det_data = det_boxes.data.cpu()

        if det_data.shape[0] > 0:
            # Filter by confidence threshold
            conf_mask = det_data[:, 4] >= args.conf_threshold
            det_data = det_data[conf_mask]

        if det_data.shape[0] > 0:
            det_data = det_data[det_data[:, 4].argsort(descending=True)]
            if args.max_detections is not None:
                det_data = det_data[: args.max_detections]

        pred_boxes: List[List[float]] = det_data[:, :4].tolist() if det_data.shape[0] > 0 else []
        pred_confs: List[float] = det_data[:, 4].tolist() if det_data.shape[0] > 0 else []
        pred_det_classes: List[int] = det_data[:, 5].long().tolist() if det_data.shape[0] > 0 else []

        # Record detection outputs for debugging
        for idx, (box, conf, det_cls) in enumerate(zip(pred_boxes, pred_confs, pred_det_classes)):
            best_iou = 0.0
            best_gt = None
            for gt_idx, gt_box in enumerate(gt_boxes):
                iou = compute_iou(box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt_idx
            detection_outputs.append(
                {
                    "image": image_name,
                    "det_index": idx,
                    "x1": box[0],
                    "y1": box[1],
                    "x2": box[2],
                    "y2": box[3],
                    "confidence": conf,
                    "detected_class": det_cls,
                    "matched_gt_index": best_gt,
                    "matched_iou": best_iou,
                }
            )

        matched_gt_indices: Dict[int, int] = {}
        matched_pairs: List[Tuple[int, int]] = []

        # Classify each detection
        pred_class_labels: List[int] = []
        pred_logits: List[torch.Tensor] = []
        pred_probs: List[List[float]] = []

        for box in pred_boxes:
            x1, y1, x2, y2 = map(int, box)
            crop = image.crop((x1, y1, x2, y2))
            inputs = processor(images=crop, return_tensors="pt")
            pixel_values = inputs["pixel_values"]
            if use_cuda:
                pixel_values = pixel_values.to("cuda")
            with torch.no_grad():
                logits = swin_model(pixel_values=pixel_values).logits.cpu()
            probs = torch.softmax(logits, dim=1)
            pred_class = int(probs.argmax(dim=1).item())

            pred_class_labels.append(pred_class)
            pred_logits.append(logits.squeeze(0))
            pred_probs.append(probs.squeeze(0).tolist())

        # Greedy matching
        for pred_idx, (box, pred_class) in enumerate(zip(pred_boxes, pred_class_labels)):
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_idx in matched_gt_indices.values():
                    continue
                iou = compute_iou(box, gt_box)
                if iou >= args.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_gt_idx >= 0:
                matched_gt_indices[pred_idx] = best_gt_idx
                matched_pairs.append((pred_idx, best_gt_idx))
                true_label = gt_species_labels[best_gt_idx]
                y_true_all.append(true_label)
                y_pred_all.append(pred_class)

                if pred_class != true_label:
                    stem = Path(image_name).stem
                    crop = image.crop(tuple(map(int, pred_boxes[pred_idx])))

                    crop_path = classification_crop_dir / f"{stem}_pred{pred_class}_true{true_label}.jpg"
                    crop.save(crop_path)

                    heatmap = generate_gradcam_overlay(
                        gradcam_model=gradcam_model,
                        target_layer=target_layer,
                        processor=processor,
                        crop=crop,
                        target_class=pred_class,
                        use_cuda=use_cuda,
                    )
                    heatmap_path = classification_heatmap_dir / f"{stem}_pred{pred_class}_true{true_label}.jpg"
                    heatmap.save(heatmap_path)

                    original_vis_path = classification_original_dir / f"{stem}_pred{pred_class}_true{true_label}.jpg"
                    save_classification_visuals(
                        image=image,
                        gt_boxes=gt_boxes,
                        gt_labels=gt_species_labels,
                        pred_boxes=pred_boxes,
                        pred_classes=pred_class_labels,
                        matched_pairs=matched_pairs,
                        highlight_pair=(pred_idx, best_gt_idx),
                        image_out_path=original_vis_path,
                    )

                    classification_failures.append(
                        {
                            "image": image_name,
                            "error_stage": "classification",
                            "predicted_index": pred_class,
                            "predicted_label": CLASS_NAMES[pred_class] if pred_class < len(CLASS_NAMES) else f"Class {pred_class}",
                            "true_index": true_label,
                            "true_label": CLASS_NAMES[true_label] if true_label < len(CLASS_NAMES) else f"Class {true_label}",
                            "confidence": max(pred_probs[pred_idx]),
                            "probabilities": json.dumps(pred_probs[pred_idx]),
                        }
                    )
            else:
                # Unmatched prediction (false positive)
                pass

        # Handle unmatched ground truths (missed detections)
        unmatched_gt = set(range(len(gt_boxes))) - set(matched_gt_indices.values())
        if unmatched_gt:
            stem = Path(image_name).stem
            det_vis_path = detection_fail_vis_dir / f"{stem}.jpg"
            save_detection_visual(
                image=image,
                gt_boxes=gt_boxes,
                unmatched_gt_indices=list(unmatched_gt),
                pred_boxes=pred_boxes,
                pred_confs=pred_confs,
                out_path=det_vis_path,
            )

        for gt_idx in unmatched_gt:
            gt_label = gt_species_labels[gt_idx]
            gt_box = gt_boxes[gt_idx]
            ious = [compute_iou(gt_box, box) for box in pred_boxes]
            best_iou = max(ious) if ious else 0.0
            best_conf = pred_confs[int(np.argmax(ious))] if ious else None
            detection_failures.append(
                {
                    "image": image_name,
                    "error_stage": "detection",
                    "gt_index": gt_idx,
                    "gt_label": CLASS_NAMES[gt_label] if gt_label < len(CLASS_NAMES) else f"Class {gt_label}",
                    "failure_type": "no_predictions" if not pred_boxes else "low_iou",
                    "best_iou": best_iou,
                    "best_confidence": best_conf,
                    "predictions_available": len(pred_boxes),
                    "detection_outputs": json.dumps(
                        [
                            {
                                "box": box,
                                "confidence": conf,
                                "detected_class": det_cls,
                            }
                            for box, conf, det_cls in zip(pred_boxes, pred_confs, pred_det_classes)
                        ]
                    ),
                }
            )
            y_true_all.append(gt_label)
            y_pred_all.append(MISSED_CLASS_ID)

    if args.species_id_override is not None and classification_failures:
        for entry in classification_failures:
            entry.setdefault("note", "")
            entry["note"] += f"species_id_override={args.species_id_override}"

    # Persist logs
    pd.DataFrame({"y_true": y_true_all, "y_pred": y_pred_all}).to_csv(eval_results_path, index=False)
    pd.DataFrame(detection_failures).to_csv(det_failures_path, index=False)
    pd.DataFrame(detection_outputs).to_csv(det_outputs_path, index=False)
    pd.DataFrame(classification_failures).to_csv(cls_failures_path, index=False)

    print(f"Saved evaluation results to {eval_results_path}")
    print(f"Saved detection failures to {det_failures_path}")
    print(f"Saved detection outputs to {det_outputs_path}")
    print(f"Saved classification failures to {cls_failures_path}")
    print(f"Detection failure visuals → {detection_fail_vis_dir}")
    print(f"Classification failure crops → {classification_crop_dir}")
    print(f"Classification failure heatmaps → {classification_heatmap_dir}")
    print(f"Classification failure originals → {classification_original_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze pipeline failures for detection and classification.")
    parser.add_argument(
        "--image-dir",
        default="Dataset/YOLO_training/florida_five_anole_v4/test/images",
        help="Directory containing test images.",
    )
    parser.add_argument(
        "--label-dir",
        default="Dataset/YOLO_training/florida_five_anole_v4/test/labels",
        help="Directory containing YOLO-format label files.",
    )
    parser.add_argument(
        "--output-dir",
        default="Spring_2025/eval_outputs",
        help="Directory to store evaluation outputs.",
    )
    parser.add_argument(
        "--yolo-weights",
        default="Spring_2025/models/train22_yolov8x_dataset_v4/weights/best.pt",
        help="Path to YOLOv8x weights.",
    )
    parser.add_argument(
        "--swin-model-dir",
        default="Spring_2025/models/swin_transformer_base_lizard_v4",
        help="Directory containing the fine-tuned Swin Transformer checkpoint.",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.3,
        help="Confidence threshold for detection filtering.",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.2,
        help="IoU threshold used for matching detections with ground truth.",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=5,
        help="Maximum number of detections per image to evaluate.",
    )
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        help="If set and CUDA is available, run GradCAM inference on GPU.",
    )
    parser.add_argument(
        "--gradcam-stage-index",
        type=int,
        default=1,
        help="Swin encoder stage index to probe for GradCAM (0-based). Layer 1 often yields the best localization.",
    )
    parser.add_argument(
        "--species-id-override",
        type=int,
        default=None,
        help="Force all ground-truth species labels to this class id (useful when YOLO labels are generic).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)


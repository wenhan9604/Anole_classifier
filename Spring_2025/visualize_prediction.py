import os
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import AutoImageProcessor, SwinForImageClassification
import matplotlib.pyplot as plt

# --- Settings ---
class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
image_folder = "Dataset/YOLO_training/florida_five_anole_10000/test/sample"
label_folder = "Dataset/YOLO_training/florida_five_anole_10000/test/sample_labels"

sample_imgs = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]

# --- Load models ---
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

# --- Loop through sample images ---
for img_name in sample_imgs[:7]:
    img_path = os.path.join(image_folder, img_name)
    label_path = os.path.join(label_folder, img_name.replace(".jpg", ".txt"))

    # Get true label from .txt file (YOLO format: class x_center y_center width height)
    try:
        with open(label_path, "r") as f:
            true_class = int(f.readline().strip().split()[0])
            true_label = class_names[true_class]
    except:
        true_label = "Unknown"

    image = Image.open(img_path).convert("RGB")
    results = yolo_model(img_path)

    if not results or results[0].boxes is None:
        print(f"No detections for {img_name}")
        continue

    for box in results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box)
        cropped = image.crop((x1, y1, x2, y2))

        inputs = processor(images=cropped, return_tensors="pt")
        with torch.no_grad():
            logits = swin_model(**inputs).logits
        pred_class = torch.argmax(logits, dim=1).item()
        pred_label = class_names[pred_class]

        # --- Plot with both predicted and true label ---
        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        ax = plt.gca()
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   edgecolor='red', linewidth=2, fill=False))
        ax.text(x1, y1 - 10,
                f"Pred: {pred_label} | True: {true_label}",
                color='white', fontsize=12,
                bbox=dict(facecolor='red', alpha=0.8))
        plt.title(f"Detection & Classification: {img_name}")
        plt.axis('off')
        plt.show()

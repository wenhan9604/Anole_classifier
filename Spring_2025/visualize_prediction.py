import os
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import AutoImageProcessor, SwinForImageClassification
import matplotlib.pyplot as plt

# --- Settings ---
class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']
image_folder = "Dataset/YOLO_training/florida_five_anole_10000/test/no_detection_image"
label_folder = "Dataset/YOLO_training/florida_five_anole_10000/test/no_detection_label"

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
for img_name in sample_imgs[:6]:
    img_path = os.path.join(image_folder, img_name)
    label_path = os.path.join(label_folder,
                              img_name.replace(".jpg", ".txt"))

    # --- ground-truth class (optional) ---
    try:
        with open(label_path) as f:
            all_lines = f.readlines()
            if len(all_lines) > 1:
                print(f"[Warning] Multiple GT labels for {img_name}; using first only")
            true_class = int(all_lines[0].strip().split()[0])
            true_label = class_names[true_class]
    except Exception:
        true_label, true_class = "Unknown", None

    # --- detection ---
    image   = Image.open(img_path).convert("RGB")
    results = yolo_model(img_path)
    if (not results or results[0].boxes is None
            or len(results[0].boxes) == 0):
        print(f"[Detection FAIL] {img_name}")
        # visualise fail -- save or show
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image)
        ax.set_title(f"{img_name}\nNo Detection | True: {true_label}", fontsize=12)

        ax.axis("off")
        plt.show()         # or fig.savefig(...)
        plt.close(fig)
        continue           # <-- guarantees nothing below runs

    # --- detection success ---
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    print(f"{img_name} | Num boxes: {len(boxes)}")

    # pick **one** box or loop over all
    for i, (box, conf) in enumerate(zip(boxes, confs)):
        # optional confidence filter
        if conf < 0.25:
            continue

        x1, y1, x2, y2 = map(int, box)
        cropped = image.crop((x1, y1, x2, y2))

        with torch.no_grad():
            inputs = processor(images=cropped, return_tensors="pt")
            logits = swin_model(**inputs).logits
        pred_class = logits.argmax().item()
        pred_label = class_names[pred_class]

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(image)
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   edgecolor="red", linewidth=2, fill=False))
        ax.text(x1, y1 - 10,
                f"Pred: {pred_label} | True: {true_label}",
                color="white", fontsize=10,
                bbox=dict(facecolor="red", alpha=0.8))
        
        plt.tight_layout()

        ax.axis("off")
        plt.show()        
        plt.close(fig)
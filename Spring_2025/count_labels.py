import os
from collections import Counter

label_folder = "Dataset/YOLO_training/florida_five_anole_10000/train/labels"
class_counts = Counter()

for filename in os.listdir(label_folder):
    if filename.endswith(".txt"):
        with open(os.path.join(label_folder, filename), "r") as f:
            for line in f:
                class_id = int(line.strip().split()[0])
                class_counts[class_id] += 1

# Display result
for cls_id, count in sorted(class_counts.items()):
    print(f"Class {cls_id}: {count} objects")

total = sum(class_counts.values())
print(f"\nTotal labeled objects: {total}")

import subprocess
import sys
import shutil

def main():
    # Ensure 'yolo' CLI is available
    
    yolo_path = "/home/hice1/wchia7/.local/bin/yolo"
    training_dataset_path = "/home/hice1/wchia7/scratch/Anole_classifier/Dataset/yolo_training/lizard_10000_v4/data.yaml"
    model_name = "yolov8x.pt"

    cmd = [
        "torchrun",
        "--nproc_per_node=2",
        yolo_path,
        "detect", "train",
        f"data={training_dataset_path}",
        "imgsz=640",
        "epochs=100",
        "batch=48",
        f"model={model_name}",
        "device=0,1",
        "name=yolov8x"
    ]

    print("Launching command:\n", " ".join(cmd))
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("Training failed with code", result.returncode)
    else:
        print("Training finished successfully.")

if __name__ == "__main__":
    main()



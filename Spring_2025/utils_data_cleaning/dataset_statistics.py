"""
Count instances and images per class across all YOLO-format annotation .txt files.

Usage:
    python count_instances.py <annotations_folder> --yaml data.yaml [--output dataset_summary.txt]

Each annotation file: <TaxonID>_<ImageID>.txt
Each line in a file:   <class_id> <cx> <cy> <w> <h>
"""

import os
import argparse
from collections import Counter, defaultdict

import yaml


# Taxon ID -> class ID mapping
TAXON_TO_CLASS = {
    "36391": 4,   # knight_anole
    "36488": 2,   # crested_anole
    "36455": 0,   # bark_anole
    "36514": 3,   # green_anole
    "116461": 1,  # brown_anole
}


def load_class_names(yaml_path: str) -> dict:
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("names", {})


def count_instances_and_files(folder: str):
    instance_counts = Counter()
    file_counts = Counter()  # number of .txt files per class (by taxon ID)

    for fname in os.listdir(folder):
        if not fname.endswith(".txt"):
            continue
        filepath = os.path.join(folder, fname)

        # Extract taxon ID from filename: <TaxonID>_<ImageID>.txt
        stem = fname.rsplit(".", 1)[0]
        parts = stem.split("_", 1)
        taxon_id = parts[0] if len(parts) == 2 else None

        if taxon_id and taxon_id in TAXON_TO_CLASS:
            class_from_file = TAXON_TO_CLASS[taxon_id]
            file_counts[class_from_file] += 1

        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                class_id = int(line.split()[0])
                instance_counts[class_id] += 1

    return instance_counts, file_counts


def write_summary(instance_counts: Counter, file_counts: Counter,
                  class_names: dict, output_path: str):
    total_instances = sum(instance_counts.values())
    total_files = sum(file_counts.values())
    all_classes = sorted(set(instance_counts.keys()) | set(file_counts.keys()))

    with open(output_path, "w") as f:
        f.write(f"{'Class ID':<10} {'Name':<20} {'Images':>8} {'Instances':>10} {'Inst %':>8}\n")
        f.write("-" * 58 + "\n")
        for cid in all_classes:
            name = class_names.get(cid, f"unknown_{cid}")
            imgs = file_counts.get(cid, 0)
            insts = instance_counts.get(cid, 0)
            pct = insts / total_instances * 100 if total_instances else 0
            f.write(f"{cid:<10} {name:<20} {imgs:>8} {insts:>10} {pct:>7.1f}%\n")
        f.write("-" * 58 + "\n")
        f.write(f"{'Total':<30} {total_files:>8} {total_instances:>10}\n")

    print(f"Summary written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count YOLO annotation instances and images per class.")
    parser.add_argument("folder", help="Path to folder containing .txt annotation files")
    parser.add_argument("--yaml", required=True, help="Path to data.yaml with class name mapping")
    parser.add_argument("--output", default="dataset_summary.txt", help="Output summary file path")
    args = parser.parse_args()

    class_names = load_class_names(args.yaml)
    instance_counts, file_counts = count_instances_and_files(args.folder)

    if not instance_counts and not file_counts:
        print(f"No annotations found in {args.folder}")
    else:
        write_summary(instance_counts, file_counts, class_names, args.output)
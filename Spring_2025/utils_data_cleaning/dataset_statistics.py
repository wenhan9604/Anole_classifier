"""
Count instances and images per class across all YOLO-format annotation .txt files,
and report images containing more than one class label.

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


def parse_annotations(folder: str):
    """Parse all annotation files and return per-file class counts."""
    instance_counts = Counter()
    file_counts = Counter()
    # file_class_counts[filename] = Counter of {class_id: count}
    file_class_counts = {}

    for fname in sorted(os.listdir(folder)):
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

        per_file = Counter()
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                class_id = int(line.split()[0])
                per_file[class_id] += 1
                instance_counts[class_id] += 1

        if per_file:
            file_class_counts[fname] = per_file

    return instance_counts, file_counts, file_class_counts


def get_multi_class_images(file_class_counts: dict):
    """Find images containing more than one class label."""
    multi = {}
    for fname, counts in file_class_counts.items():
        if len(counts) > 1:
            multi[fname] = counts
    return multi


def write_summary(instance_counts, file_counts, file_class_counts,
                  class_names, output_path):
    total_instances = sum(instance_counts.values())
    total_files = sum(file_counts.values())
    all_classes = sorted(set(instance_counts.keys()) | set(file_counts.keys()))

    multi_class_images = get_multi_class_images(file_class_counts)

    # Group multi-class images by their primary class (from taxon ID)
    multi_by_class = defaultdict(list)
    for fname, counts in multi_class_images.items():
        stem = fname.rsplit(".", 1)[0]
        parts = stem.split("_", 1)
        taxon_id = parts[0] if len(parts) == 2 else None
        primary_class = TAXON_TO_CLASS.get(taxon_id) if taxon_id else None
        if primary_class is not None:
            multi_by_class[primary_class].append((fname, counts))

    with open(output_path, "w") as f:
        # ── Section 1: Overall Summary ──
        f.write("=" * 60 + "\n")
        f.write("DATASET SUMMARY\n")
        f.write("=" * 60 + "\n\n")

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

        # ── Section 2: Multi-Class Images ──
        f.write("\n\n")
        f.write("=" * 60 + "\n")
        f.write("IMAGES WITH MULTIPLE CLASS LABELS\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Total multi-class images: {len(multi_class_images)}\n\n")

        for cid in all_classes:
            name = class_names.get(cid, f"unknown_{cid}")
            images = multi_by_class.get(cid, [])
            f.write("-" * 58 + "\n")
            f.write(f"Class {cid} ({name}): {len(images)} multi-class image(s)\n")
            f.write("-" * 58 + "\n")

            if not images:
                f.write("  (none)\n\n")
                continue

            for fname, counts in sorted(images):
                # Build string showing counts of each class in this file
                class_breakdown = ", ".join(
                    f"{class_names.get(c, f'class_{c}')}: {n}"
                    for c, n in sorted(counts.items())
                )
                f.write(f"  {fname:<35} [{class_breakdown}]\n")
            f.write("\n")

    print(f"Summary written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count YOLO annotation instances/images per class and report multi-class images."
    )
    parser.add_argument("folder", help="Path to folder containing .txt annotation files")
    parser.add_argument("--yaml", required=True, help="Path to data.yaml with class name mapping")
    parser.add_argument("--output", default="dataset_summary.txt", help="Output summary file path")
    args = parser.parse_args()

    class_names = load_class_names(args.yaml)
    instance_counts, file_counts, file_class_counts = parse_annotations(args.folder)

    if not instance_counts and not file_counts:
        print(f"No annotations found in {args.folder}")
    else:
        write_summary(instance_counts, file_counts, file_class_counts,
                      class_names, args.output)
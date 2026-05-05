#!/usr/bin/env python3
"""
Generate Grad-CAM heatmaps for all 4 Swin encoder layers (stages 0–3) for a cropped image.
Outputs a composite figure (2x2) and optionally individual layer images.

Usage:
    python3 generate_gradcam_all_layers.py --image path/to/cropped.jpg
    python3 generate_gradcam_all_layers.py --image-dir Dataset/cropped_img --output-dir path/to/out
"""

import argparse
import math
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, SwinForImageClassification

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except ModuleNotFoundError:
    print(
        "Missing package: pytorch_grad_cam. Install it in your current environment with:\n"
        "  pip install grad-cam\n"
        "Then run this script again."
    )
    raise SystemExit(1)

CLASS_NAMES = [
    "Bark Anole",
    "Brown Anole",
    "Crested Anole",
    "Green Anole",
    "Knight Anole",
]

NUM_STAGES = 4  # Swin encoder has 4 stages (0–3)


class SwinClassificationWrapper(torch.nn.Module):
    """Wrap Swin model so GradCAM receives logits tensor."""

    def __init__(self, model: SwinForImageClassification):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits


def swin_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    """Reshape Swin token output (B, N, C) to (B, C, H, W) for GradCAM."""
    if tensor.ndim != 3:
        raise ValueError(f"Expected tensor with 3 dims (B, N, C), got {tensor.shape}")
    batch, num_tokens, channels = tensor.shape
    side = int(math.sqrt(num_tokens))
    if side * side != num_tokens:
        raise ValueError(
            f"Token count {num_tokens} is not a perfect square, cannot reshape to grid."
        )
    reshaped = tensor.view(batch, side, side, channels).permute(0, 3, 1, 2).contiguous()
    return reshaped


def generate_gradcam_for_stage(
    model: SwinClassificationWrapper,
    processor: AutoImageProcessor,
    image: Image.Image,
    target_class: int,
    stage_index: int,
    pixel_values: torch.Tensor,
    use_cuda: bool,
) -> np.ndarray:
    """Generate Grad-CAM heatmap for one encoder stage. pixel_values already on correct device."""
    target_stage = model.model.swin.encoder.layers[stage_index]
    target_layer = target_stage.blocks[-1].layernorm_before

    with GradCAM(
        model=model,
        target_layers=[target_layer],
        reshape_transform=swin_reshape_transform,
    ) as cam:
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = cam(input_tensor=pixel_values, targets=targets)[0]

    img_np = np.array(image).astype(np.float32) / 255.0
    cam_resized = cv2.resize(grayscale_cam, (img_np.shape[1], img_np.shape[0]))
    heatmap = show_cam_on_image(img_np, cam_resized, use_rgb=True)
    return heatmap


def process_one_image(
    image_path: Path,
    output_dir: Path,
    model,
    wrapper: SwinClassificationWrapper,
    processor: AutoImageProcessor,
    target_class: Optional[int],
    use_cuda: bool,
    save_individual: bool,
) -> None:
    """Load one image, generate all 4 layer heatmaps, save composite (and optionally per-stage)."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]
    if use_cuda:
        pixel_values = pixel_values.cuda()

    with torch.no_grad():
        logits = model(pixel_values=pixel_values).logits
        probs = torch.softmax(logits, dim=-1)
        predicted_class = logits.argmax(dim=1).item()
        confidence = probs[0, predicted_class].item()

    target_class = target_class if target_class is not None else predicted_class

    heatmaps = []
    for stage_index in range(NUM_STAGES):
        heatmap = generate_gradcam_for_stage(
            model=wrapper,
            processor=processor,
            image=image,
            target_class=target_class,
            stage_index=stage_index,
            pixel_values=pixel_values,
            use_cuda=use_cuda,
        )
        heatmaps.append(heatmap)
        if save_individual:
            out_path = output_dir / f"{image_path.stem}_stage{stage_index}.jpg"
            Image.fromarray(heatmap).save(out_path)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for i, (ax, heatmap) in enumerate(zip(axes.flatten(), heatmaps)):
        ax.imshow(heatmap)
        ax.set_title(f"Stage {i}", fontsize=12)
        ax.axis("off")
    fig.suptitle(
        f"Grad-CAM: {image_path.name} — Class {target_class} ({CLASS_NAMES[target_class]})",
        fontsize=11,
    )
    plt.tight_layout()
    composite_path = output_dir / f"{image_path.stem}_gradcam_all_layers.jpg"
    fig.savefig(composite_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM for all 4 Swin layers on cropped image(s)"
    )
    parser.add_argument("--image", type=str, default=None, help="Path to a single cropped image")
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Path to directory of cropped images (processes all jpg/png)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for composites (default: gradcam_all_layers or next to --image)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="Spring_2025/models/swin_transformer_base_lizard_v4",
        help="Path to Swin model directory",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=None,
        help="Class ID to visualize (0–4). If not set, uses predicted class.",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Disable CUDA",
    )
    parser.add_argument(
        "--save-individual",
        action="store_true",
        help="Also save one image per layer (stage0.jpg, stage1.jpg, ...)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images that already have a composite in output-dir",
    )

    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("Provide either --image or --image-dir")
    if args.image and args.image_dir:
        parser.error("Provide only one of --image or --image-dir")

    repo_root = Path(__file__).resolve().parents[1]

    if args.image_dir:
        image_dir = Path(args.image_dir)
        if not image_dir.is_absolute():
            image_dir = (repo_root / image_dir).resolve()
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Image dir not found: {image_dir}")
        image_paths = sorted(
            image_dir / f
            for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        )
        default_output = repo_root / "Spring_2025" / "gradcam_all_layers_cropped"
    else:
        image_path = Path(args.image)
        if not image_path.is_absolute():
            image_path = (repo_root / image_path).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image_paths = [image_path]
        default_output = image_path.parent / "gradcam_all_layers"

    output_dir = Path(args.output_dir) if args.output_dir else default_output
    if not output_dir.is_absolute():
        output_dir = (repo_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = (repo_root / model_dir).resolve()
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    use_cuda = torch.cuda.is_available() and not args.no_cuda

    print(f"Loading model from {model_dir}...")
    model = SwinForImageClassification.from_pretrained(
        str(model_dir),
        local_files_only=True,
    )
    processor = AutoImageProcessor.from_pretrained(
        str(model_dir),
        local_files_only=True,
    )
    model.eval()
    if use_cuda:
        model = model.cuda()
        print("Using CUDA")
    else:
        print("Using CPU")

    wrapper = SwinClassificationWrapper(model)
    if use_cuda:
        wrapper = wrapper.cuda()

    if args.skip_existing:
        to_process = [
            p for p in image_paths
            if not (output_dir / f"{p.stem}_gradcam_all_layers.jpg").exists()
        ]
        print(f"Skipping {len(image_paths) - len(to_process)} existing, processing {len(to_process)} image(s) -> {output_dir}")
    else:
        to_process = image_paths
        print(f"Processing {len(to_process)} image(s) -> {output_dir}")

    for idx, image_path in enumerate(to_process):
        try:
            print(f"  [{idx+1}/{len(to_process)}] {image_path.name}")
            process_one_image(
                image_path=image_path,
                output_dir=output_dir,
                model=model,
                wrapper=wrapper,
                processor=processor,
                target_class=args.class_id,
                use_cuda=use_cuda,
                save_individual=args.save_individual,
            )
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    print("Done.")


if __name__ == "__main__":
    main()

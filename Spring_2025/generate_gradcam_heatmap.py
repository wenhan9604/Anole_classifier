#!/usr/bin/env python3
"""
Standalone script to generate Grad-CAM heatmaps for Swin Transformer model.

Usage:
    python3 generate_gradcam_heatmap.py --image path/to/image.jpg
    python3 generate_gradcam_heatmap.py --image path/to/image.jpg --class-id 2
    python3 generate_gradcam_heatmap.py --image-dir path/to/images --output-dir path/to/output
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
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from transformers import AutoImageProcessor, SwinForImageClassification

# Class names matching the model's id2label
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


def generate_gradcam_heatmap(
    model: SwinForImageClassification,
    processor: AutoImageProcessor,
    image: Image.Image,
    target_class: Optional[int] = None,
    stage_index: int = 2,
    use_cuda: bool = True,
) -> tuple[np.ndarray, int, float]:
    """
    Generate Grad-CAM heatmap for an image.

    Args:
        model: Swin Transformer model
        processor: Image processor
        image: PIL Image (will be resized to model input size)
        target_class: Class ID to visualize (None = use predicted class)
        stage_index: Which Swin encoder stage to use (0-3, default=2)
        use_cuda: Whether to use GPU

    Returns:
        Tuple of (heatmap_image, predicted_class_id, confidence)
    """
    # Ensure RGB
    image = image.convert("RGB")

    # Process image
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"]

    # Get prediction
    with torch.no_grad():
        if use_cuda:
            pixel_values = pixel_values.to("cuda")
        logits = model(pixel_values=pixel_values).logits
        probs = torch.softmax(logits, dim=-1)
        predicted_class = logits.argmax(dim=1).item()
        confidence = probs[0, predicted_class].item()

    # Use predicted class if not specified
    if target_class is None:
        target_class = predicted_class

    # Wrap model for Grad-CAM
    gradcam_model = SwinClassificationWrapper(model)
    if use_cuda:
        gradcam_model = gradcam_model.cuda()

    # Get target layer
    if stage_index < 0 or stage_index >= len(gradcam_model.model.swin.encoder.layers):
        raise ValueError(
            f"stage_index {stage_index} is out of range "
            f"(0-{len(gradcam_model.model.swin.encoder.layers) - 1})"
        )
    target_stage = gradcam_model.model.swin.encoder.layers[stage_index]
    target_layer = target_stage.blocks[-1].layernorm_before

    # Generate Grad-CAM
    with GradCAM(
        model=gradcam_model,
        target_layers=[target_layer],
        reshape_transform=swin_reshape_transform,
    ) as cam:
        targets = [ClassifierOutputTarget(target_class)]
        grayscale_cam = cam(input_tensor=pixel_values, targets=targets)[0]

    # Convert image to numpy for overlay
    img_np = np.array(image).astype(np.float32) / 255.0

    # Resize CAM to match image size
    cam_resized = cv2.resize(grayscale_cam, (img_np.shape[1], img_np.shape[0]))

    # Create heatmap overlay
    heatmap = show_cam_on_image(img_np, cam_resized, use_rgb=True)

    return heatmap, predicted_class, confidence


def main():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM heatmaps for Swin Transformer model"
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to a single image file",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        help="Path to directory containing images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="gradcam_outputs",
        help="Output directory for heatmaps (default: gradcam_outputs)",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="Spring_2025/models/swin_transformer_base_lizard_v4/checkpoint-352",
        help="Path to Swin model directory (default: checkpoint-352)",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=None,
        help="Class ID to visualize (0-4). If not specified, uses predicted class.",
    )
    parser.add_argument(
        "--stage-index",
        type=int,
        default=2,
        help="Swin encoder stage index for Grad-CAM (0-3, default=2)",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Disable CUDA even if available",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.image and not args.image_dir:
        parser.error("Either --image or --image-dir must be specified")

    # Setup paths
    repo_root = Path(__file__).resolve().parents[1]
    model_dir = Path(args.model_dir)
    if not model_dir.is_absolute():
        model_dir = (repo_root / model_dir).resolve()

    if not model_dir.exists():
        raise ValueError(f"Model directory not found: {model_dir}")

    # Load model
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

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    if use_cuda:
        model = model.cuda()
        print("Using CUDA")
    else:
        print("Using CPU")

    # Setup output directory
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (repo_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect image paths
    image_paths = []
    if args.image:
        image_paths.append(Path(args.image))
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        if not image_dir.is_absolute():
            image_dir = (repo_root / image_dir).resolve()
        image_paths = [
            image_dir / f
            for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]

    print(f"Processing {len(image_paths)} image(s)...")

    # Process each image
    for img_path in image_paths:
        try:
            print(f"\nProcessing: {img_path.name}")

            # Load image
            image = Image.open(img_path).convert("RGB")

            # Generate heatmap
            heatmap, pred_class, confidence = generate_gradcam_heatmap(
                model=model,
                processor=processor,
                image=image,
                target_class=args.class_id,
                stage_index=args.stage_index,
                use_cuda=use_cuda,
            )

            # Save heatmap
            output_path = output_dir / f"{img_path.stem}_gradcam.jpg"
            Image.fromarray(heatmap).save(output_path)
            print(f"  Saved: {output_path}")
            print(
                f"  Predicted: {CLASS_NAMES[pred_class]} (class {pred_class}) "
                f"with {confidence*100:.1f}% confidence"
            )
            if args.class_id is not None and args.class_id != pred_class:
                print(f"  Heatmap shows: {CLASS_NAMES[args.class_id]} (class {args.class_id})")

        except Exception as e:
            print(f"  Error processing {img_path.name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nDone! Heatmaps saved to: {output_dir}")


if __name__ == "__main__":
    main()


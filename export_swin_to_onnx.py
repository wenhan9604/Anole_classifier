"""
Export Swin Transformer (Hugging Face checkpoint dir, e.g. model.safetensors) to ONNX.

Loads weights via ``from_pretrained(local_dir)`` — Safetensors or PyTorch bin is picked up
automatically. Writes ``swin_model.onnx`` for backend ONNX and/or frontend static hosting.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import torch
from transformers import SwinForImageClassification


def _default_model_candidates(repo_root: Path) -> list[Path]:
    rels = [
        Path("Spring_2025/models/swin_transformer_base_lizard_v4"),
        Path("Spring_2025/swin_transformer_base_lizard_v4"),
        Path("model_export/swin_transformer_base_lizard_v4"),
    ]
    return [repo_root / r for r in rels]


def resolve_model_dir(repo_root: Path, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_absolute():
            p = repo_root / p
        if not p.is_dir():
            raise FileNotFoundError(f"Model directory not found: {p}")
        if not (p / "config.json").is_file():
            raise FileNotFoundError(f"Not a Hugging Face model folder (missing config.json): {p}")
        return p.resolve()

    env = os.getenv("CLASSIFICATION_MODEL_ID")
    if env and Path(env).is_dir():
        p = Path(env).expanduser()
        if not p.is_absolute():
            p = repo_root / p
        if (p / "config.json").is_file():
            return p.resolve()

    for c in _default_model_candidates(repo_root):
        if c.is_dir() and (c / "config.json").is_file():
            return c.resolve()

    raise FileNotFoundError(
        "No Swin checkpoint directory found. Pass --model DIR (must contain config.json "
        "and model weights such as model.safetensors), or set CLASSIFICATION_MODEL_ID."
    )


def export_swin_to_onnx(
    model_dir: Path,
    primary_out: Path,
    copy_to: Path | None,
    opset: int = 14,
) -> bool:
    print("=" * 60)
    print("Exporting Swin Transformer to ONNX")
    print("=" * 60)
    print(f"\nCheckpoint: {model_dir}")

    print("\n1. Loading model (Safetensors / bin via Hugging Face)...")
    try:
        model = SwinForImageClassification.from_pretrained(str(model_dir))
        print("   ✓ Loaded")
        print(f"   Classes: {model.config.num_labels}, image_size: {model.config.image_size}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False

    model.eval()
    image_size = int(model.config.image_size)
    dummy_input = torch.randn(1, 3, image_size, image_size)

    print("\n2. Sanity-check forward...")
    try:
        with torch.no_grad():
            logits = model(dummy_input).logits
        print(f"   ✓ logits shape: {tuple(logits.shape)}")
    except Exception as e:
        print(f"   ✗ Forward failed: {e}")
        return False

    primary_out.parent.mkdir(parents=True, exist_ok=True)
    input_names = ["pixel_values"]
    output_names = ["logits"]
    dynamic_axes = {"pixel_values": {0: "batch_size"}, "logits": {0: "batch_size"}}

    print(f"\n3. torch.onnx.export → {primary_out} (opset {opset})...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(primary_out),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            export_params=True,
        )
        mb = primary_out.stat().st_size / (1024 * 1024)
        print(f"   ✓ Saved ({mb:.1f} MB)")
    except Exception as e:
        print(f"   ✗ Export failed: {e}")
        return False

    if copy_to is not None:
        copy_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(primary_out, copy_to)
        print(f"\n4. Copied → {copy_to}")

    print("\n5. Optional checks...")
    try:
        import onnx

        onnx.checker.check_model(onnx.load(str(primary_out)))
        print("   ✓ onnx.checker OK")
    except ImportError:
        print("   ⚠ pip install onnx for structural validation")
    except Exception as e:
        print(f"   ⚠ onnx checker: {e}")

    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(primary_out), providers=["CPUExecutionProvider"])
        got = sess.run(None, {"pixel_values": dummy_input.numpy()})[0]
        diff = float(np.abs(got - logits.numpy()).max())
        print(f"   ✓ onnxruntime max |Δ| vs torch: {diff:.6f}")
    except ImportError:
        print("   ⚠ pip install onnxruntime for runtime check")
    except Exception as e:
        print(f"   ⚠ onnxruntime: {e}")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)
    print(f"Primary: {primary_out}")
    if copy_to:
        print(f"Copy:    {copy_to}")
    print("\nNext: copy frontend/public/models/swin_model.onnx to dist after build, or rsync to server.")
    return True


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    os.chdir(repo_root)

    p = argparse.ArgumentParser(description="Export Swin HF folder (safetensors) to ONNX.")
    p.add_argument(
        "--model",
        default=None,
        help="Path to HF model dir (config.json + model.safetensors). Default: env CLASSIFICATION_MODEL_ID or standard repo locations.",
    )
    p.add_argument(
        "--backend-out",
        default="backend/models/swin_model.onnx",
        help="Output ONNX path (default: backend/models/swin_model.onnx)",
    )
    p.add_argument(
        "--frontend-out",
        default="frontend/public/models/swin_model.onnx",
        help="Second copy for Vite public/ (default). Use empty string to skip.",
    )
    p.add_argument("--opset", type=int, default=14)
    args = p.parse_args()

    try:
        model_dir = resolve_model_dir(repo_root, args.model)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    primary = (repo_root / args.backend_out).resolve()
    copy_to = (repo_root / args.frontend_out).resolve() if args.frontend_out else None

    ok = export_swin_to_onnx(model_dir, primary, copy_to, opset=args.opset)
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        raise SystemExit(130)

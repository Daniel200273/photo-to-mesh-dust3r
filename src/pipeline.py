import os
import sys
import torch
import argparse
import numpy as np
import open3d as o3d
from pathlib import Path

# Add DUSt3R to path
sys.path.append(str(Path(__file__).parent.parent / "dust3r"))

from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic"}


def normalize_images(raw_data_dir: Path):
    """Rename images to sequential numbers for consistent ordering."""
    print(f"--- Normalizing images in {raw_data_dir} ---")
    images = sorted([f for f in raw_data_dir.iterdir() if f.suffix.lower() in VALID_EXTENSIONS])
    for i, img_path in enumerate(images, start=1):
        new_name = f"{i:05d}{img_path.suffix.lower()}"
        new_path = raw_data_dir / new_name
        if img_path != new_path:
            img_path.rename(new_path)
    print(f"Normalized {len(images)} images successfully.\n")


def has_images(directory: Path) -> bool:
    """Check if a directory exists and contains valid image files."""
    if not directory.exists():
        return False
    return any(f.suffix.lower() in VALID_EXTENSIONS for f in directory.iterdir() if f.is_file())


def load_model():
    """Detect hardware and load the DUSt3R model (done once, reused across passes)."""
    print("--- Hardware Detection & Loading Model ---")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 NVIDIA GPU detected. Using CUDA acceleration.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🍏 Apple Silicon detected. Using MPS acceleration.")
    else:
        device = torch.device("cpu")
        print("🐢 No dedicated GPU found. Falling back to CPU.")
        print("   Note: This will be slow, but it will work on integrated graphics!")

    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([argparse.Namespace])

    model_path = Path(__file__).parent.parent / "dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    
    # Load local model file if it exists, otherwise try HuggingFace
    if model_path.exists():
        print(f"Loading model from local cache: {model_path}")
        model = AsymmetricCroCo3DStereo.from_pretrained(str(model_path), local_files_only=True).to(device)
    else:
        print(f"Model not found locally at {model_path}")
        print("Downloading from HuggingFace (this may take a while)...")
        model = AsymmetricCroCo3DStereo.from_pretrained("naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt").to(device)
    
    model.eval()

    return model, device


def run_dust3r(raw_data_dir: Path, output_dir: Path, model, device, output_name: str = "reconstruction.ply"):
    """
    Run DUSt3R on a set of images and export a coloured point cloud.

    Args:
        raw_data_dir:  folder containing input images
        output_dir:    folder for the output .ply
        model:         pre-loaded DUSt3R model
        device:        torch device
        output_name:   filename for the output point cloud
    """
    print(f"\n--- Extracting 3D Point Maps from {raw_data_dir.name} ---")
    processing_size = 512

    image_paths = sorted([str(p) for p in raw_data_dir.glob("*") if p.is_file()])
    images = load_images(image_paths, size=processing_size)
    pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)

    with torch.no_grad():
        output = inference(pairs, model, device, batch_size=1)

    print("--- Global Alignment ---")
    aligner = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    aligner.compute_global_alignment(init="mst", niter=100, schedule="linear", lr=0.01)

    print("--- Exporting Point Cloud ---")
    output_dir.mkdir(parents=True, exist_ok=True)
    ply_path = output_dir / output_name

    pts3d = aligner.get_pts3d()
    masks = aligner.get_masks()
    imgs = aligner.imgs

    all_pts = []
    all_colors = []

    for i in range(len(imgs)):
        p = pts3d[i].detach().cpu().numpy() if hasattr(pts3d[i], 'detach') else pts3d[i]
        c = imgs[i].detach().cpu().numpy() if hasattr(imgs[i], 'detach') else imgs[i]
        m = masks[i].detach().cpu().numpy() if hasattr(masks[i], 'detach') else masks[i]

        all_pts.append(p[m])
        all_colors.append(c[m])

    all_pts = np.concatenate(all_pts, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)

    o3d.io.write_point_cloud(str(ply_path), pcd)
    print(f"🎉 Point Cloud saved to {ply_path}\n")


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.resolve()
    raw_dir    = project_root / "data" / "raw_data"
    processed_dir  = project_root / "data" / "processed_data"

    if not has_images(raw_dir):
        print(f"Error: No images found in {raw_dir}")
        exit(1)

    model, device = load_model()

    # ── Run DUSt3R on upright scan ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("  3D Reconstruction — (raw_data)")
    print("=" * 60)
    normalize_images(raw_dir)
    run_dust3r(raw_dir, processed_dir, model, device, output_name="reconstruction.ply")

    print("\n✅ Done! Run mesh_reconstruction.py to generate the final mesh:")
    print("   python src/mesh_reconstruction.py")
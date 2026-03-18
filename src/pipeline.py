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

def normalize_images(raw_data_dir: Path):
    print(f"--- Step 1: Normalizing images in {raw_data_dir} ---")
    valid_extensions = {".jpg", ".jpeg", ".png", ".heic"}
    images = sorted([f for f in raw_data_dir.iterdir() if f.suffix.lower() in valid_extensions])
    
    for i, img_path in enumerate(images, start=1):
        new_name = f"{i:05d}{img_path.suffix.lower()}"
        new_path = raw_data_dir / new_name
        if img_path != new_path:
            img_path.rename(new_path)
    print(f"Normalized {len(images)} images successfully.\n")

def run_dust3r(raw_data_dir: Path, output_dir: Path):
    print("--- Step 2: Hardware Detection & Loading Model ---")
    
    # --- UNIVERSAL HARDWARE CHECK ---
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

    # PyTorch 2.6 Fix
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([argparse.Namespace])

    model_path = "dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    
    # Map location handles loading models trained on GPU onto a CPU machine
    model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(device)
    model.eval()

    print("--- Step 3: Extracting 3D Point Maps ---")
    # --- MEMORY MANAGEMENT FOR LOW-END PCS ---
    # If your teammates hit Out Of Memory (OOM) errors, tell them to change 512 to 256.
    processing_size = 512 
    
    image_paths = sorted([str(p) for p in raw_data_dir.glob("*") if p.is_file()])
    images = load_images(image_paths, size=processing_size)
    pairs = make_pairs(images, scene_graph="complete", prefilter=None, symmetrize=True)
    
    with torch.no_grad():
        # batch_size=1 ensures we don't overwhelm integrated graphics VRAM/RAM
        output = inference(pairs, model, device, batch_size=1)
    
    print("--- Step 4: Global Alignment ---")
    aligner = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    aligner.compute_global_alignment(init="mst", niter=100, schedule="linear", lr=0.01)

    print("--- Step 5: Exporting Point Cloud ---")
    output_dir.mkdir(parents=True, exist_ok=True)
    ply_path = output_dir / "reconstruction.ply"
    
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
    print(f"🎉 Success! 3D Point Cloud saved to {ply_path}")

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.resolve()
    raw_dir = project_root / "data" / "raw_data"
    processed_dir = project_root / "data" / "processed_data"
    
    if not raw_dir.exists():
        print(f"Error: Raw data directory not found at {raw_dir}")
        exit(1)
        
    normalize_images(raw_dir)
    run_dust3r(raw_dir, processed_dir)
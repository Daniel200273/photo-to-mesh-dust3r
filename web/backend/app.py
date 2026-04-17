"""
FastAPI backend for FewShot-NeRF web interface
Handles image upload, 3D reconstruction, and stylization
"""

import os
import sys
import shutil
import json
import uuid
from pathlib import Path
from typing import List
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
from PIL import Image

# Add parent directories to path to import pipeline and stylize
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "dust3r"))

from pipeline import load_model, run_dust3r, normalize_images, has_images
from stylize import apply_low_poly, apply_voxel, apply_soft_voxel, apply_hologram
import open3d as o3d

# ─────────────────────────────── SETUP ───────────────────────────────
app = FastAPI(title="FewShot-NeRF API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
WEB_DIR = Path(__file__).parent.parent
UPLOADS_DIR = WEB_DIR / "uploads"
RESULTS_DIR = WEB_DIR / "results"
UPLOADS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Global model (loaded once, reused)
MODEL = None
DEVICE = None

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic"}

EFFECTS = {
    "low_poly": "Low Poly",
    "voxel": "Voxel (Minecraft)",
    "soft_voxel": "Soft Voxel",
    "hologram": "Hologram Neon"
}

# ─────────────────────────────── HELPER FUNCTIONS ───────────────────
def initialize_model():
    """Load model on first use"""
    global MODEL, DEVICE
    if MODEL is None:
        print("Initializing DUSt3R model...")
        MODEL, DEVICE = load_model()
        print(f"Model loaded on device: {DEVICE}")


def validate_images(files: List[UploadFile]) -> bool:
    """Validate that files are images"""
    if len(files) < 3:
        return False
    for file in files:
        if file.content_type not in ["image/jpeg", "image/png", "image/heic"]:
            return False
    return True


def save_uploaded_files(files: List[UploadFile], job_dir: Path) -> bool:
    """Save uploaded files to job directory"""
    for i, file in enumerate(files, 1):
        try:
            # Get file extension
            ext = Path(file.filename).suffix.lower()
            save_path = job_dir / f"{i:05d}{ext}"
            
            with open(save_path, "wb") as f:
                content = file.file.read()
                f.write(content)
        except Exception as e:
            print(f"Error saving file: {e}")
            return False
    return True


def apply_effect(mesh_path: Path, effect: str) -> Path:
    """Apply stylization effect to mesh"""
    print(f"Loading mesh: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    
    if effect == "low_poly":
        result_mesh = apply_low_poly(mesh)
    elif effect == "voxel":
        result_mesh = apply_voxel(mesh)
    elif effect == "soft_voxel":
        result_mesh = apply_soft_voxel(mesh)
    elif effect == "hologram":
        result_mesh = apply_hologram(mesh)
    else:
        raise ValueError(f"Unknown effect: {effect}")
    
    # Save result
    output_path = mesh_path.parent / f"styled_{effect}_{mesh_path.name}"
    o3d.io.write_triangle_mesh(str(output_path), result_mesh)
    print(f"Saved styled mesh: {output_path}")
    
    return output_path


def mesh_to_json(mesh_path: Path) -> dict:
    """Convert mesh to JSON format for Three.js"""
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    mesh.compute_vertex_normals()
    
    vertices = np.asarray(mesh.vertices).flatten().tolist()
    triangles = np.asarray(mesh.triangles).flatten().tolist()
    normals = np.asarray(mesh.vertex_normals).flatten().tolist()
    
    # Colors
    if mesh.has_vertex_colors():
        colors = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8).flatten().tolist()
    else:
        colors = [200, 200, 200] * len(mesh.vertices)
    
    return {
        "vertices": vertices,
        "triangles": triangles,
        "normals": normals,
        "colors": colors,
        "vertex_count": len(mesh.vertices),
        "triangle_count": len(mesh.triangles)
    }


# ─────────────────────────────── API ENDPOINTS ───────────────────────
@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    initialize_model()


@app.get("/api/health")
async def health():
    """Health check"""
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "device": str(DEVICE) if DEVICE else None
    }


@app.get("/api/effects")
async def get_effects():
    """Get available effects"""
    return {"effects": EFFECTS}


@app.post("/api/process")
async def process_images(files: List[UploadFile] = File(...), effect: str = "low_poly"):
    """
    Upload images, reconstruct 3D model, and apply effect
    
    Flow:
    1. Validate images
    2. Save to job directory
    3. Run DUSt3R reconstruction (slowest step, ~5-10 min on CPU)
    4. Apply stylization effect
    5. Convert mesh to JSON
    6. Return download link + preview data
    """
    try:
        # Validate
        if not validate_images(files):
            raise HTTPException(
                status_code=400,
                detail="Need at least 3 images (JPEG/PNG/HEIC)"
            )
        
        if effect not in EFFECTS:
            raise HTTPException(status_code=400, detail=f"Invalid effect: {effect}")
        
        # Create job directory
        job_id = str(uuid.uuid4())[:8]
        job_dir = UPLOADS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = datetime.now()
        print(f"\n{'='*70}")
        print(f"[JOB {job_id}] START: Processing {len(files)} images with '{effect}' effect")
        print(f"[JOB {job_id}] Device: {DEVICE} (CPU will be slow, ~5-10 min)")
        print(f"{'='*70}\n")
        
        # Save files
        print(f"[JOB {job_id}] 📥 Step 1/5: Saving images...")
        if not save_uploaded_files(files, job_dir):
            raise HTTPException(status_code=400, detail="Error saving images")
        print(f"[JOB {job_id}]     ✓ Saved {len(files)} images\n")
        
        # Normalize image names
        normalize_images(job_dir)
        
        # Run reconstruction
        print(f"[JOB {job_id}] 🧠 Step 2/5: Running DUSt3R reconstruction...")
        print(f"[JOB {job_id}]     ⏳ This is the slowest step, please wait...\n")
        output_dir = RESULTS_DIR / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        run_dust3r(job_dir, output_dir, MODEL, DEVICE, output_name="reconstruction.ply")
        
        mesh_path = output_dir / "reconstruction.ply"
        if not mesh_path.exists():
            raise HTTPException(status_code=500, detail="Reconstruction failed")
        print(f"[JOB {job_id}]     ✓ Reconstruction complete\n")
        
        # Apply effect
        print(f"[JOB {job_id}] 🎨 Step 3/5: Applying '{effect}' effect...")
        styled_mesh_path = apply_effect(mesh_path, effect)
        print(f"[JOB {job_id}]     ✓ Effect applied\n")
        
        # Convert to JSON for preview
        print(f"[JOB {job_id}] 📊 Step 4/5: Converting mesh to 3D format...")
        mesh_json = mesh_to_json(styled_mesh_path)
        print(f"[JOB {job_id}]     ✓ Converted ({mesh_json['vertex_count']:,} vertices, {mesh_json['triangle_count']:,} triangles)\n")
        
        # Save metadata
        print(f"[JOB {job_id}] 💾 Step 5/5: Saving metadata...")
        metadata = {
            "job_id": job_id,
            "timestamp": datetime.now().isoformat(),
            "effect": effect,
            "file_count": len(files),
            "mesh_file": styled_mesh_path.name,
            "processing_time": str(datetime.now() - start_time)
        }
        
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[JOB {job_id}]     ✓ Saved\n")
        
        elapsed = datetime.now() - start_time
        print(f"{'='*70}")
        print(f"[JOB {job_id}] ✅ COMPLETE in {elapsed}")
        print(f"{'='*70}\n")
        
        return JSONResponse({
            "success": True,
            "job_id": job_id,
            "mesh_data": mesh_json,
            "download_url": f"/api/download/{job_id}",
            "effect": effect,
            "processing_time": str(elapsed)
        })
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/download/{job_id}")
async def download_mesh(job_id: str):
    """Download the processed mesh file"""
    try:
        job_results = RESULTS_DIR / job_id
        metadata_path = job_results / "metadata.json"
        
        if not metadata_path.exists():
            raise HTTPException(status_code=404, detail="Job not found")
        
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        mesh_file = job_results / metadata["mesh_file"]
        
        if not mesh_file.exists():
            raise HTTPException(status_code=404, detail="Mesh file not found")
        
        return FileResponse(
            mesh_file,
            filename=f"model_{metadata['effect']}.ply",
            media_type="application/octet-stream"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get job status and metadata"""
    try:
        metadata_path = RESULTS_DIR / job_id / "metadata.json"
        
        if not metadata_path.exists():
            raise HTTPException(status_code=404, detail="Job not found")
        
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        return metadata
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)

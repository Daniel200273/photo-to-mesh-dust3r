# Photo-to-Mesh Reconstruction with Dust3r

Generates textured 3D meshes from 10–15 photos using [DUSt3R](https://github.com/naver/dust3r. Runs on Mac, Windows, and Linux with auto GPU/CPU detection.

> **✨ Quick Start:** Want to get started immediately? Run `python src/studio.py` after completing the setup for an interactive CLI that guides you through the entire pipeline.

---

## 🛠️ Setup

```bash
conda create --name cv-project python=3.10 -y
conda activate cv-project
pip install -r requirements.txt
```

Set up DUSt3R (if not already present):

```bash
git clone --recursive https://github.com/naver/dust3r.git
mkdir -p dust3r/checkpoints
curl -L -o dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
  https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
```

---

## 📋 Manual Pipeline

```bash
# 1. Place 10–15 photos in data/raw_data/

# 2. Reconstruct 3D point cloud
python src/pipeline.py

# 3. Build and clean mesh with pedestal
python src/mesh_reconstruction.py

# 4. View result
python src/visualize_mesh.py

# 5. Optionally stylize
python src/stylize.py --input data/processed_data/mesh/object_mesh.ply \
  --filter ff7 --param 800 --pedestal data/processed_data/mesh/pedestal.ply \
  --output data/stylized/ff7.ply

# 6. View stylized mesh
python src/visualize_mesh_browser.py --path data/stylized/ff7.ply
```

---

## Interactive CLI

```bash
python src/studio.py
```

This guided workflow:

- Automatically runs reconstruction and mesh building
- Lets you preview results interactively
- Offers stylization filters with live previews
- Supports both desktop (Open3D) and browser (Plotly) viewers
- Auto-switches between filters without re-running previous steps

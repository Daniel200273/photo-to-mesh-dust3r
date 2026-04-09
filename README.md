# Few-Shot 3D Reconstruction Pipeline

This pipeline uses [DUSt3R](https://github.com/naver/dust3r) (Dense and Unconstrained Stereo 3D Reconstruction) to generate 3D point clouds from sparse, uncalibrated image sets (e.g., 10–15 images) and reconstruct textured 3D meshes — no COLMAP or camera calibration required. It runs on Windows, Mac, and Linux, automatically falling back to CPU if no dedicated GPU is available.

## 🛠️ Setup

### 1. Create your Python environment

```bash
conda create --name cv-project python=3.10 -y
conda activate cv-project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Set up DUSt3R

If the `dust3r/` directory is not present in the project root:

```bash
git clone --recursive https://github.com/naver/dust3r.git

mkdir -p dust3r/checkpoints
curl -L -o dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
  https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
```

---

## 🚀 Usage

### Step 1 — Generate a Point Cloud

Place your photos (10–15 images) in `data/raw_data/`, then run:

```bash
python src/pipeline.py
```

This outputs `data/processed_data/reconstruction.ply`. Visualize it with:

```bash
python src/visualize_points.py
```

### Step 2 — Build a 3D Mesh

#### Single-scan mode (default)

```bash
python src/mesh_reconstruction.py
```

This takes the point cloud from Step 1 and runs: noise removal → plane segmentation → object isolation → Poisson surface reconstruction → hole filling → vertex color transfer.

The final mesh is saved to `data/processed_data/mesh/final_mesh.ply`.

You can also specify custom paths:

```bash
python src/mesh_reconstruction.py --input path/to/cloud.ply --output path/to/output/
```

#### Two-pass mode (flipped scan)

For objects with a non-flat bottom, take two sets of photos — one upright (Scan A) and one flipped/on its side (Scan B). Run `pipeline.py` on each set to produce two `.ply` files, then:

```bash
python src/mesh_reconstruction.py --scan_a scan_a.ply --scan_b scan_b.ply --output mesh/
```

This cleans each scan independently, aligns them using FPFH + ICP registration, merges, and generates a single watertight mesh.

### Step 3 — Visualize the Mesh

```bash
python src/visualize_mesh.py
```

Or point to a specific file:

```bash
python src/visualize_mesh.py --path data/processed_data/mesh/mesh_trimmed.ply
```

**Controls:** Left-click = rotate, Right-click = pan, Scroll = zoom, Q = close.

---

## 📁 Project Structure

```
FewShot-NeRF/
├── src/
│   ├── pipeline.py             # DUSt3R point cloud generation
│   ├── mesh_reconstruction.py  # Cleaning, registration & meshing
│   ├── visualize_points.py     # Point cloud viewer
│   └── visualize_mesh.py       # Triangle mesh viewer
├── data/
│   ├── raw_data/               # Input photos
│   └── processed_data/         # Point clouds & meshes
├── dust3r/                     # DUSt3R (cloned submodule)
└── requirements.txt
```

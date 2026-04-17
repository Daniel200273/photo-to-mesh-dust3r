# Few-Shot 3D Reconstruction Pipeline

Uses [DUSt3R](https://github.com/naver/dust3r) to generate textured 3D meshes from a small set of photos (10–15 images) — no COLMAP or camera calibration required. Runs on Mac, Windows, and Linux; falls back to CPU if no GPU is available.

---

## 🛠️ Setup

```bash
conda create --name cv-project python=3.10 -y
conda activate cv-project
pip install -r requirements.txt
```

**Set up DUSt3R** (if `dust3r/` is not already present):

```bash
git clone --recursive https://github.com/naver/dust3r.git

mkdir -p dust3r/checkpoints
curl -L -o dust3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth \
  https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
```

---

## 🚀 Pipeline

### Step 1 — Generate Point Cloud

Place 10–15 photos of your object in `data/raw_data/`, then run:

```bash
python src/pipeline.py
```

Output: `data/processed_data/reconstruction.ply`

```bash
# Optional: preview the point cloud
python src/visualize_points.py
```

---

### Step 2 — Build Mesh

```bash
python src/mesh_reconstruction.py
```

Runs: noise removal → plane segmentation → object isolation → Poisson reconstruction → hole filling → color transfer → auto-generated pedestal.

Two output files are saved to `data/processed_data/mesh/`:
| File | Contents |
|---|---|
| `object_mesh.ply` | Object only (use this for stylization) |
| `pedestal.ply` | Base disc only |
| `final_mesh.ply` | Object + pedestal combined |

Custom paths:
```bash
python src/mesh_reconstruction.py --input path/to/cloud.ply --output path/to/output/
```

```bash
# Preview the result
python src/visualize_mesh.py --path data/processed_data/mesh/final_mesh.ply
```

---

### Step 3 — Stylize (Optional)

All filters take `--input` (the mesh) and `--output` (a `.ply` file path).  
Use `--pedestal` to re-attach the base disc **unstyled** after the filter is applied.

#### FF7 — PS1 / Final Fantasy 7 style

Flat-shaded faces, angular geometry, posterized colours. Closest to the original PS1 aesthetic.

```bash
python src/stylize.py \
  --input    data/processed_data/mesh/object_mesh.ply \
  --filter   ff7 \
  --param    800 \
  --pedestal data/processed_data/mesh/pedestal.ply \
  --output   data/stylized/ff7.ply
```

`--param`: triangle budget (default `800`, range `300`–`2000`).

#### Low-Poly

Smooth, faceted simplification via voxel clustering.

```bash
python src/stylize.py \
  --input    data/processed_data/mesh/object_mesh.ply \
  --filter   low_poly \
  --param    1500 \
  --pedestal data/processed_data/mesh/pedestal.ply \
  --output   data/stylized/low_poly.ply
```

`--param`: target triangle count (default `1500`).

#### Voxel — Minecraft style

Rebuilds the mesh as coloured cubes.

```bash
python src/stylize.py \
  --input    data/processed_data/mesh/object_mesh.ply \
  --filter   voxel \
  --param    0.015 \
  --pedestal data/processed_data/mesh/pedestal.ply \
  --output   data/stylized/voxel.ply
```

`--param`: voxel size in metres (default `0.01`).

#### Soft Voxel

Like voxel but uses overlapping spheres for a rounder look.

```bash
python src/stylize.py \
  --input    data/processed_data/mesh/object_mesh.ply \
  --filter   soft_voxel \
  --param    0.02 \
  --pedestal data/processed_data/mesh/pedestal.ply \
  --output   data/stylized/soft_voxel.ply
```

#### Hologram — neon wireframe

Produces two files: a blue body (`_body.ply`) and a neon edge wireframe (`_edges.ply`).

```bash
python src/stylize.py \
  --input    data/processed_data/mesh/object_mesh.ply \
  --filter   hologram \
  --color    0.035,0.714,0.902 \
  --pedestal data/processed_data/mesh/pedestal.ply \
  --output   data/stylized/hologram.ply
```

`--color`: edge colour as `R,G,B` in 0–1 range (default: bright cyan).

---

### Step 4 — View Results

```bash
# Desktop viewer (Open3D)
python src/visualize_mesh.py --path data/stylized/ff7.ply

# Browser viewer (Plotly — Wayland-compatible)
python src/visualize_mesh_browser.py --path data/stylized/ff7.ply
```

For hologram files, pass the `_body.ply` path — edges are loaded automatically:
```bash
python src/visualize_mesh.py --path data/stylized/hologram_body.ply --hologram
```

---

## 📁 Project Structure

```
FewShot-NeRF/
├── src/
│   ├── pipeline.py               # DUSt3R point cloud generation
│   ├── mesh_reconstruction.py    # Cleaning, meshing & pedestal
│   ├── stylize.py                # Artistic filters (ff7, low-poly, voxel, hologram)
│   ├── visualize_points.py       # Point cloud viewer
│   ├── visualize_mesh.py         # Desktop 3D viewer (Open3D)
│   └── visualize_mesh_browser.py # Browser 3D viewer (Plotly)
├── data/
│   ├── raw_data/                 # Input photos go here
│   ├── processed_data/
│   │   ├── reconstruction.ply    # Point cloud (pipeline.py output)
│   │   └── mesh/                 # object_mesh.ply, pedestal.ply, final_mesh.ply
│   └── stylized/                 # Stylized mesh outputs
├── dust3r/                       # DUSt3R submodule
├── requirements.txt
└── README.md
```

---

## ⚙️ Configuration

Key parameters can be changed at the top of each script:

| Script | Parameter | Description |
|---|---|---|
| `mesh_reconstruction.py` | `POISSON_DEPTH` | Mesh detail level (default `9`) |
| `mesh_reconstruction.py` | `PEDESTAL_PADDING` | Pedestal size relative to object footprint |
| `stylize.py` | `FF7_TARGET_TRIANGLES` | Default FF7 triangle budget |
| `stylize.py` | `FF7_COLOR_LEVELS` | Colour quantization steps (default `16`) |
| `stylize.py` | `HOLOGRAM_NEON_COLOR` | Default hologram edge colour |

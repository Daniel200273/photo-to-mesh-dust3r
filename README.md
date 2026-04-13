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

### Step 3 — Stylization (Optional)

Apply artistic filters to transform the reconstructed mesh:

#### Low-Poly Stylization

Creates a low-polygon version with smooth surfaces. Uses isotropic remeshing by applying controlled subdivision and clustering.

```bash

# On existing mesh
python src/stylize.py --input data/processed_data/mesh/final_mesh.ply --filter low_poly --param 1500
```

**Parameters:**
- `--param`: Target triangle count (default: `1500`). Lower = fewer triangles, blockier appearance. Range: 500–5000

#### Voxel Stylization

Creates blocky, Minecraft-style geometry by clustering vertices into a grid. Preserves color from the original mesh.

```bash

python src/stylize.py --input data/processed_data/mesh/final_mesh.ply --filter voxel --param 0.015
```

**Parameters:**
- `--param`: Voxel size in mesh units (default: `0.01`). Smaller = finer detail. Suggested range: 0.005–0.05 (depends on mesh scale)

#### Soft Voxel Stylization

Like voxel but interpolates overlapping spheres for a smoother, rounded appearance while maintaining block-like structure.

```bash

python src/stylize.py --input data/processed_data/mesh/final_mesh.ply --filter soft_voxel --param 0.02
```

**Parameters:**
- `--param`: Voxel size in mesh units (default: `0.01`). Controls the softness radius. Suggested range: 0.008–0.06

#### Hologram Stylization

Creates a neon wireframe effect with a transparent body. Perfect for sci-fi, technical, or artistic presentations. Generates TWO files: a low-poly transparent body (`_body.ply`) and cyan neon edges (`_edges.ply`).

```bash

python src/stylize.py --input data/processed_data/mesh/final_mesh.ply --filter hologram --color 1,0,1
```

**Parameters:**
- `--color`: RGB neon color for edges (0–1 range). Examples:
  - `0.035,0.714,0.902` — Bright cyan (default)
  - `1,0,1` — Magenta
  - `0,1,1` — Full cyan
  - `1,1,0` — Yellow
  - `1,0.5,0` — Orange

**Output files:**
- `final_mesh_hologram_body.ply` — Semi-transparent colored body
- `final_mesh_hologram_edges.ply` — Neon wireframe edges

---

### Step 4 — View & Compare

#### Desktop Viewer (Open3D)

```bash
python src/visualize_mesh.py --path data/processed_data/mesh/final_mesh.ply

# For hologram effect (auto-loads edges)
python src/visualize_mesh.py --path data/processed_data/mesh/final_mesh_hologram_body.ply --hologram
```

**Controls:** Left-click = rotate | Right-click = pan | Scroll = zoom | Q = close

#### Browser Viewer (Plotly — Wayland-compatible)

```bash
python src/visualize_mesh_browser.py --path data/processed_data/mesh/final_mesh.ply --hologram
```

Opens an interactive 3D view in your browser. Supports full rotation, zoom, color customization. Perfect for presenting results or working on Wayland systems (GNOME, KDE Plasma 6).

```

---

## 📁 Project Structure

```
FewShot-NeRF/
├── src/
│   ├── pipeline.py             # DUSt3R point cloud generation
│   ├── mesh_reconstruction.py  # Cleaning, registration & meshing
│   ├── stylize.py              # Artistic filters (low-poly, voxel, hologram)
│   ├── visualize_points.py     # Point cloud viewer
│   ├── visualize_mesh.py       # Desktop 3D viewer (Open3D)
│   └── visualize_mesh_browser.py # Browser 3D viewer (Plotly)
├── data/
│   ├── raw_data/               # Input photos (organize by scan)
│   └── processed_data/
│       ├── reconstruction.ply  # Point cloud output
│       └── mesh/               # Mesh, derivatives, & styled outputs
├── dust3r/                     # DUSt3R submodule (cloned separately)
├── requirements.txt
└── README.md
```

---

## ⚙️ Advanced Configuration

### Stylization Parameters

Edit `src/stylize.py` to customize:
- `LOW_POLY_TARGET_TRIANGLES` — default target for low-poly
- `LOW_POLY_VOXEL_DIVISOR` — internal voxel grid subdivision
- `HOLOGRAM_NEON_COLOR` — default edge color `[R, G, B]` (0–1 range)

"""
mesh_reconstruction.py — 3D Reconstruction Pipeline

Workflow:
  Point cloud → clean → mesh → auto-pedestal

Architecture:
  clean_cloud()       — SOR + RANSAC plane removal + DBSCAN object isolation
  generate_mesh()     — Poisson reconstruction + density/distance trim + hole fill + color transfer
  generate_pedestal() — auto-sized flat disc placed at the object's base
  run_pipeline()      — orchestrates the full workflow

Usage:
  python mesh_reconstruction.py --input cloud.ply --output mesh/
"""

import argparse
import numpy as np
import open3d as o3d
import trimesh
from pathlib import Path


# ═══════════════════════════ PARAMETERS ══════════════════════════════════════

# Cleanser — Statistical Outlier Removal
MICRO_VOXEL_SIZE  = 0.0     # 0 = disabled. Used to be 0.0005
SOR_NB_NEIGHBORS  = 20
SOR_STD_RATIO     = 2.05

# Cleanser — RANSAC Plane Segmentation
PLANE_DISTANCE_THRESHOLD = 0.005
PLANE_RANSAC_N           = 3
PLANE_NUM_ITERATIONS     = 1000
PLANE_CROP_MARGIN        = 0.01

# Cleanser — DBSCAN Clustering
CLUSTER_EPS        = 0.05
CLUSTER_MIN_POINTS = 10
CLUSTER_VOXEL_SIZE = 0.005

# Mesher — Poisson Surface Reconstruction
POISSON_DEPTH          = 8
POISSON_DENSITY_QUANT  = 0.05

# Mesher — Point Cloud Spatial Smoothing
CLOUD_SMOOTH_ITERS     = 2
CLOUD_SMOOTH_K         = 30
CLOUD_SMOOTH_SIGMA     = 0.002

# Mesher — Post-clean
MIN_COMPONENT_RATIO    = 0.01

# Mesher — Smoothing
RECON_SMOOTH_ITERS     = 3

# Mesher — Color Transfer
COLOR_KNN = 1

# Pedestal
PEDESTAL_PADDING       = 1.15   # pedestal radius = bounding radius × this factor
PEDESTAL_THICKNESS     = 0.005  # thin slab height
PEDESTAL_RESOLUTION    = 128    # angular subdivisions for the disc
PEDESTAL_COLOR_SAMPLES = 64     # bottom-edge vertices to sample for colour matching


# ═══════════════════════════ MODULE 1: CLEANSER ══════════════════════════════

def clean_cloud(
    pcd: o3d.geometry.PointCloud,
    label: str = "scan",
    output_dir: Path | None = None,
) -> tuple[o3d.geometry.PointCloud, tuple[float, float, float, float] | None]:
    """
    Full cleansing pipeline for a single raw point cloud.

    Steps:
      1. Statistical Outlier Removal
      2. RANSAC Plane Segmentation (signed-distance crop with auto-flip)
      3. DBSCAN Clustering (OOM-safe via voxel proxy)

    Returns: cleaned point cloud containing only the target object.
    """
    save_dir = None
    if output_dir is not None:
        save_dir = output_dir / label
        save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  CLEANSER — {label}")
    print(f"{'='*60}")

    # ── Step 0: Micro-Voxel Downsampling (Smoothing) ─────────────────────
    if MICRO_VOXEL_SIZE > 0:
        print(f"\n  [{label}] Micro-Voxel Downsampling (Smoothing)")
        before = len(pcd.points)
        pcd = pcd.voxel_down_sample(voxel_size=MICRO_VOXEL_SIZE)
        after = len(pcd.points)
        print(f"    Merged points: {before:,} → {after:,} ({100*(before-after)/before:.1f}%)")

    # ── Step 1: Statistical Outlier Removal ──────────────────────────────
    print(f"\n  [{label}] Statistical Outlier Removal")
    before = len(pcd.points)
    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=SOR_NB_NEIGHBORS, std_ratio=SOR_STD_RATIO,
    )
    after = len(pcd.points)
    print(f"    Removed {before - after:,} noise points ({100*(before-after)/before:.1f}%)")

    if save_dir:
        o3d.io.write_point_cloud(str(save_dir / "01_denoised.ply"), pcd)

    # ── Step 2: RANSAC Plane Segmentation ────────────────────────────────
    print(f"\n  [{label}] RANSAC Plane Segmentation")
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=PLANE_DISTANCE_THRESHOLD,
        ransac_n=PLANE_RANSAC_N,
        num_iterations=PLANE_NUM_ITERATIONS,
    )
    a, b, c, d = plane_model
    print(f"    Plane: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")
    print(f"    Plane inliers: {len(inliers):,}")

    pts = np.asarray(pcd.points)
    normal = np.array([a, b, c])
    signed_dist = pts @ normal + d

    if (signed_dist > 0).sum() < (signed_dist < 0).sum():
        signed_dist = -signed_dist
        print("    (flipped normal toward object)")

    keep_mask = signed_dist > PLANE_CROP_MARGIN
    pcd = pcd.select_by_index(np.where(keep_mask)[0])
    print(f"    Remaining: {len(pcd.points):,} points")

    if save_dir:
        o3d.io.write_point_cloud(str(save_dir / "02_no_plane.ply"), pcd)

    # ── Step 3: DBSCAN Clustering ────────────────────────────────────────
    print(f"\n  [{label}] DBSCAN Clustering")
    n_full = len(pcd.points)

    pcd_down = pcd.voxel_down_sample(voxel_size=CLUSTER_VOXEL_SIZE)
    n_down = len(pcd_down.points)
    print(f"    Proxy: {n_full:,} → {n_down:,} points")

    labels = np.asarray(
        pcd_down.cluster_dbscan(
            eps=CLUSTER_EPS, min_points=CLUSTER_MIN_POINTS, print_progress=True,
        )
    )
    num_clusters = labels.max() + 1
    print(f"    Found {num_clusters} cluster(s), {(labels == -1).sum():,} noise")

    if num_clusters == 0:
        print("    ⚠️  No clusters — returning full cloud")
        return pcd, plane_model

    largest = np.argmax(np.bincount(labels[labels >= 0]))
    pcd_down_main = pcd_down.select_by_index(np.where(labels == largest)[0])

    down_tree = o3d.geometry.KDTreeFlann(pcd_down_main)
    full_pts = np.asarray(pcd.points)
    keep_mask = np.zeros(n_full, dtype=bool)
    radius_sq = (CLUSTER_VOXEL_SIZE * 2.0) ** 2

    for i in range(n_full):
        _, _, dist = down_tree.search_knn_vector_3d(full_pts[i], 1)
        if dist[0] < radius_sq:
            keep_mask[i] = True

    pcd = pcd.select_by_index(np.where(keep_mask)[0])
    print(f"    Isolated object: {keep_mask.sum():,} / {n_full:,} points")

    if save_dir:
        o3d.io.write_point_cloud(str(save_dir / "03_object.ply"), pcd)

    return pcd, plane_model


# ═══════════════════════════ MODULE 2: MESHER ════════════════════════════════

def generate_mesh(
    pcd: o3d.geometry.PointCloud,
    colored_pcd: o3d.geometry.PointCloud | None = None,
    output_dir: Path | None = None,
) -> o3d.geometry.TriangleMesh:
    """
    Full meshing pipeline: Poisson reconstruction → density/distance trim →
    hole fill → vertex color transfer.

    Args:
      pcd:          point cloud to mesh (will have normals estimated)
      colored_pcd:  source cloud for color transfer (defaults to pcd)
      output_dir:   save intermediates if provided
    """
    if colored_pcd is None:
        colored_pcd = pcd

    print(f"\n{'='*60}")
    print(f"  MESHER — Surface Reconstruction")
    print(f"{'='*60}")

    # ── Point Cloud Spatial Smoothing (KD-Tree Clustering) ───────────────
    if CLOUD_SMOOTH_ITERS > 0:
        from scipy.spatial import cKDTree
        print(f"\n  Smoothing point cloud directly (KD-Tree, {CLOUD_SMOOTH_ITERS} passes) …")
        pts = np.asarray(pcd.points).copy()
        inv_2sig2 = 1.0 / (2.0 * (CLOUD_SMOOTH_SIGMA ** 2))
        
        for it in range(CLOUD_SMOOTH_ITERS):
            tree = cKDTree(pts)
            # Find K neighbors (clusters) around each point
            dists, indices = tree.query(pts, k=CLOUD_SMOOTH_K, workers=-1)
            # Gaussian weighted average
            weights = np.exp(-(dists ** 2) * inv_2sig2)
            weights /= weights.sum(axis=1, keepdims=True)
            pts = (weights[:, :, None] * pts[indices]).sum(axis=1)
            
        pcd.points = o3d.utility.Vector3dVector(pts)
        print(f"    Point cloud geometry seamlessly melted.")

    # ── Adaptive normal estimation ───────────────────────────────────────
    print("\n  Estimating normals (density-adaptive radius) …")
    nn_dists = np.asarray(pcd.compute_nearest_neighbor_distance())
    avg_nn = np.mean(nn_dists)
    normal_radius = max(avg_nn * 4.0, 0.01)
    print(f"    Avg NN distance: {avg_nn:.5f}  →  normal radius: {normal_radius:.5f}")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=normal_radius, max_nn=50,
        ),
    )
    pcd.orient_normals_consistent_tangent_plane(k=30)

    # ── Poisson Reconstruction ───────────────────────────────────────────
    print(f"  Poisson reconstruction (depth={POISSON_DEPTH}) …")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=POISSON_DEPTH,
    )
    print(f"    Raw: {len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris")

    # ── Density-based trimming ───────────────────────────────────────────
    print(f"  Density-based trimming …")
    density_arr = np.asarray(densities)
    density_threshold = np.quantile(density_arr, POISSON_DENSITY_QUANT)
    density_mask = density_arr < density_threshold
    mesh.remove_vertices_by_mask(density_mask)
    print(f"    Removed {density_mask.sum():,} low-density vertices")

    # ── Connected Component Trimming ─────────────────────────────────────
    print(f"  Connected component trimming (keeping ONLY largest component) …")
    triangle_clusters, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    
    if len(cluster_n_triangles) > 0:
        largest_cluster_idx = np.argmax(cluster_n_triangles)
        
        # Mask to keep ONLY the triangles belonging to the largest cluster
        triangles_to_remove = triangle_clusters != largest_cluster_idx
        
        mesh.remove_triangles_by_mask(triangles_to_remove)
        mesh.remove_unreferenced_vertices()
        components_removed = len(cluster_n_triangles) - 1
        print(f"    Trimmed: {components_removed} disconnected phantom/internal shells removed")
        
    print(f"    Result:  {len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris")

    if output_dir:
        o3d.io.write_triangle_mesh(str(output_dir / "mesh_trimmed.ply"), mesh)

    # ── Hole Filling ─────────────────────────────────────────────────────
    print("\n  Filling holes …")
    vertices_before = np.asarray(mesh.vertices)
    triangles_before = np.asarray(mesh.triangles)
    has_colors = len(mesh.vertex_colors) > 0
    vc_before = np.asarray(mesh.vertex_colors) if has_colors else None

    tm = trimesh.Trimesh(
        vertices=vertices_before, faces=triangles_before, process=False,
    )
    print(f"    Watertight before: {tm.is_watertight}")
    trimesh.repair.fill_holes(tm)
    print(f"    Watertight after:  {tm.is_watertight}")
    print(f"    Triangles: {len(triangles_before):,} → {len(tm.faces):,}")

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(tm.vertices)
    mesh.triangles = o3d.utility.Vector3iVector(tm.faces)

    if has_colors:
        n_old, n_new = len(vertices_before), len(tm.vertices)
        if n_new > n_old:
            avg = vc_before.mean(axis=0)
            all_colors = np.vstack([vc_before, np.tile(avg, (n_new - n_old, 1))])
        else:
            all_colors = vc_before[:n_new]
        mesh.vertex_colors = o3d.utility.Vector3dVector(all_colors)

    if output_dir:
        o3d.io.write_triangle_mesh(str(output_dir / "mesh_filled.ply"), mesh)

    # ── Surface Smoothing ────────────────────────────────────────────────
    if RECON_SMOOTH_ITERS > 0:
        print(f"\n  Smoothing mesh ({RECON_SMOOTH_ITERS} Taubin passes) …")
        mesh = mesh.filter_smooth_taubin(number_of_iterations=RECON_SMOOTH_ITERS)
        mesh.compute_vertex_normals()

    # ── Vertex Color Transfer ────────────────────────────────────────────
    print("\n  Transferring vertex colors …")
    pcd_tree = o3d.geometry.KDTreeFlann(colored_pcd)
    mesh_verts = np.asarray(mesh.vertices)
    src_colors = np.asarray(colored_pcd.colors)
    colors = np.zeros_like(mesh_verts)

    for i, v in enumerate(mesh_verts):
        _, idx, _ = pcd_tree.search_knn_vector_3d(v, COLOR_KNN)
        colors[i] = src_colors[idx[0]]

    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh.compute_vertex_normals()
    print(f"    Coloured {len(mesh_verts):,} vertices ✅")

    return mesh


# ═══════════════════════════ MODULE 3: PEDESTAL ══════════════════════════════

def generate_pedestal(
    mesh: o3d.geometry.TriangleMesh,
    plane_model: tuple | None = None,
    output_dir: Path | None = None,
) -> o3d.geometry.TriangleMesh:
    """
    Generate a flat circular pedestal and attach it below the object.

    If a RANSAC plane_model (a, b, c, d) is provided the disc is oriented
    perpendicular to that plane's normal — which is the actual support
    surface direction regardless of how the scan is rotated in space.
    Falls back to Y-up if no plane model is given.

    Returns: combined mesh (object + pedestal).
    """
    print(f"\n{'='*60}")
    print(f"  PEDESTAL — Auto-Generated Base")
    print(f"{'='*60}")

    verts  = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors) if len(mesh.vertex_colors) > 0 else None

    # ── Determine the "up" direction (plane normal) ────────────────────
    if plane_model is not None:
        a, b, c, d = plane_model
        up = np.array([a, b, c], dtype=float)
        up /= np.linalg.norm(up)
        # The object is on the positive-signed-distance side of the plane.
        # Ensure up points toward the object (positive mean projection).
        if (verts @ up + d).mean() < 0:
            up = -up
        print(f"  Using RANSAC plane normal as up-axis: "
              f"({up[0]:.3f}, {up[1]:.3f}, {up[2]:.3f})")
    else:
        up = np.array([0.0, 1.0, 0.0])
        print("  No plane model — falling back to Y-up axis")

    # ── Build an orthonormal basis for the disc plane ─────────────────
    # tangent1 and tangent2 span the plane; up is the disc normal.
    ref = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    tangent1 = np.cross(up, ref);  tangent1 /= np.linalg.norm(tangent1)
    tangent2 = np.cross(up, tangent1)

    # ── Find the "bottom" of the object along the up axis ─────────────
    # Project every vertex onto the up axis; minimum = base of object.
    proj = verts @ up                    # scalar projection for each vertex
    base_proj = proj.min()               # lowest point along up
    obj_height = proj.max() - proj.min()

    # ── Compute pedestal centre (footprint centroid at object base) ────
    bottom_mask = proj <= base_proj + obj_height * 0.05   # bottom 5%
    centre_3d = verts[bottom_mask].mean(axis=0) if bottom_mask.sum() > 0 \
                else verts.mean(axis=0)
    # Snap centre onto the base plane
    centre_3d = centre_3d + (base_proj - centre_3d @ up) * up

    # ── Compute pedestal radius from footprint ─────────────────────────
    # Footprint = each vertex projected onto the disc plane (subtract up component)
    up_scalars = verts @ up
    footprint  = verts - up_scalars[:, np.newaxis] * up   # in-plane coords
    centre_fp  = centre_3d - (centre_3d @ up) * up
    dists = np.linalg.norm(footprint - centre_fp, axis=1)
    pedestal_radius = dists.max() * PEDESTAL_PADDING

    print(f"  Base centre (3D): ({centre_3d[0]:.4f}, {centre_3d[1]:.4f}, {centre_3d[2]:.4f})")
    print(f"  Pedestal radius:  {pedestal_radius:.4f}")

    # ── Sample bottom-edge colour ──────────────────────────────────────
    if colors is not None and len(colors) > 0:
        bottom_colors = colors[bottom_mask]
        pedestal_color = bottom_colors.mean(axis=0) if len(bottom_colors) > 0 \
                         else colors.mean(axis=0)
        print(f"  Pedestal colour: RGB({pedestal_color[0]:.3f}, "
              f"{pedestal_color[1]:.3f}, {pedestal_color[2]:.3f})  "
              f"(sampled from {bottom_mask.sum():,} bottom-edge verts)")
    else:
        pedestal_color = np.array([0.4, 0.4, 0.4])
        print("  Pedestal colour: neutral grey (no vertex colours on mesh)")

    # ── Build disc vertices oriented along the plane normal ───────────
    n       = PEDESTAL_RESOLUTION
    h       = PEDESTAL_THICKNESS
    angles  = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cos_a   = np.cos(angles)
    sin_a   = np.sin(angles)

    # Top centre (touching object base) and bottom centre (slab below)
    top_centre    = centre_3d
    bottom_centre = centre_3d - up * h

    # Ring vertices: sweep around the tangent plane
    top_ring    = top_centre    + pedestal_radius * (cos_a[:, None] * tangent1 + sin_a[:, None] * tangent2)
    bottom_ring = bottom_centre + pedestal_radius * (cos_a[:, None] * tangent1 + sin_a[:, None] * tangent2)

    # Vertex layout: [top_centre(0), bottom_centre(1), top_ring(2..n+1), bottom_ring(n+2..2n+1)]
    disc_verts = np.vstack([top_centre, bottom_centre, top_ring, bottom_ring])
    tc_idx   = 0
    bc_idx   = 1
    tr_start = 2
    br_start = 2 + n

    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces.append([tc_idx, tr_start + j, tr_start + i])   # top fan
        faces.append([bc_idx, br_start + i, br_start + j])   # bottom fan
        ti, tj = tr_start + i, tr_start + j
        bi, bj = br_start + i, br_start + j
        faces.append([ti, tj, bj])                            # side wall
        faces.append([ti, bj, bi])

    disc_faces  = np.array(faces, dtype=np.int32)
    disc_colors = np.tile(pedestal_color, (len(disc_verts), 1))

    # ── Build Open3D mesh ─────────────────────────────────────────────
    pedestal_mesh = o3d.geometry.TriangleMesh()
    pedestal_mesh.vertices      = o3d.utility.Vector3dVector(disc_verts)
    pedestal_mesh.triangles     = o3d.utility.Vector3iVector(disc_faces)
    pedestal_mesh.vertex_colors = o3d.utility.Vector3dVector(disc_colors)
    pedestal_mesh.compute_vertex_normals()
    print(f"  Pedestal: {len(disc_verts):,} verts, {len(disc_faces):,} tris")

    if output_dir:
        o3d.io.write_triangle_mesh(str(output_dir / "pedestal.ply"), pedestal_mesh)

    combined = mesh + pedestal_mesh
    combined.compute_vertex_normals()
    print(f"  Combined: {len(combined.vertices):,} verts, {len(combined.triangles):,} tris ✅")
    return combined


# ═══════════════════════════ PIPELINE ORCHESTRATOR ═══════════════════════════

def run_pipeline(input_ply: Path, output_dir: Path):
    """Single-scan pipeline: clean → mesh → pedestal."""
    output_dir.mkdir(parents=True, exist_ok=True)

    pcd = o3d.io.read_point_cloud(str(input_ply))
    if pcd.is_empty():
        raise RuntimeError(f"Empty point cloud: {input_ply}")
    print(f"📂 Loaded {len(pcd.points):,} points from {input_ply}")

    pcd_clean, plane_model = clean_cloud(pcd, label="scan", output_dir=output_dir)
    mesh = generate_mesh(pcd_clean, colored_pcd=pcd_clean, output_dir=output_dir)

    # Save object-only mesh — used by stylize.py so the pedestal stays unstyled
    object_path = output_dir / "object_mesh.ply"
    o3d.io.write_triangle_mesh(str(object_path), mesh)
    print(f"  Object mesh saved to {object_path}")

    mesh = generate_pedestal(mesh, plane_model=plane_model, output_dir=output_dir)

    final_path = output_dir / "final_mesh.ply"
    o3d.io.write_triangle_mesh(str(final_path), mesh)
    print(f"\n🎉 Done! Final mesh saved to {final_path}")
    print(f"   To stylize without affecting the pedestal, use object_mesh.ply as input")
    return mesh


# ═══════════════════════════ CLI ═════════════════════════════════════════════

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.resolve()
    default_input  = project_root / "data" / "processed_data" / "reconstruction.ply"
    default_output = project_root / "data" / "processed_data" / "mesh"

    parser = argparse.ArgumentParser(
        description="3D Mesh Reconstruction with auto-pedestal",
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Path to input .ply point cloud (default: data/processed_data/reconstruction.ply)",
    )
    parser.add_argument(
        "--output", type=str, default=str(default_output),
        help="Output directory for intermediates and final mesh",
    )

    args = parser.parse_args()
    out = Path(args.output)
    input_path = Path(args.input) if args.input else default_input
    run_pipeline(input_path, out)

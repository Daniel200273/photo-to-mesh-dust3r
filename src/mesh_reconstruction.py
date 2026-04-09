"""
mesh_reconstruction.py — Modular 3D Reconstruction Pipeline

Supports two workflows:
  1. Single-scan:  one point cloud → cleaned mesh
  2. Two-pass:     two scans (top + flipped bottom) → registered, merged, meshed

Architecture:
  clean_cloud()      — SOR + RANSAC plane removal + DBSCAN object isolation
  register_clouds()  — FPFH global registration + Point-to-Plane ICP refinement
  generate_mesh()    — Poisson reconstruction + distance trim + hole fill + color transfer

Usage:
  python mesh_reconstruction.py --input cloud.ply --output mesh/
  python mesh_reconstruction.py --scan_a top.ply --scan_b bottom.ply --output mesh/
"""

import argparse
import numpy as np
import open3d as o3d
import trimesh
from pathlib import Path
import copy


# ═══════════════════════════ PARAMETERS ══════════════════════════════════════

# Cleanser — Statistical Outlier Removal
SOR_NB_NEIGHBORS  = 20
SOR_STD_RATIO     = 2.0

# Cleanser — RANSAC Plane Segmentation
PLANE_DISTANCE_THRESHOLD = 0.005
PLANE_RANSAC_N           = 3
PLANE_NUM_ITERATIONS     = 1000
PLANE_CROP_MARGIN        = 0.01

# Cleanser — DBSCAN Clustering
CLUSTER_EPS        = 0.05
CLUSTER_MIN_POINTS = 10
CLUSTER_VOXEL_SIZE = 0.005

# Registration — Feature Extraction
REG_VOXEL_SIZE   = 0.005
REG_FPFH_RADIUS  = 0.025
REG_FPFH_MAX_NN  = 100

# Registration — Global (FPFH + RANSAC)
REG_RANSAC_DISTANCE        = 0.015
REG_RANSAC_MAX_ITERATIONS  = 4_000_000
REG_RANSAC_CONFIDENCE      = 0.999

# Registration — Local (ICP)
REG_ICP_DISTANCE  = 0.01
REG_ICP_MAX_ITER  = 200

# Mesher — Poisson Surface Reconstruction
POISSON_DEPTH          = 9
POISSON_TRIM_DISTANCE  = 0.02

# Mesher — Color Transfer
COLOR_KNN = 1


# ═══════════════════════════ MODULE 1: CLEANSER ══════════════════════════════

def clean_cloud(
    pcd: o3d.geometry.PointCloud,
    label: str = "scan",
    output_dir: Path | None = None,
) -> o3d.geometry.PointCloud:
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
        return pcd

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

    return pcd


# ═══════════════════════════ MODULE 2: STITCHER ══════════════════════════════

def _compute_fpfh(pcd: o3d.geometry.PointCloud, voxel_size: float):
    """Downsample, estimate normals, and compute FPFH features."""
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2, max_nn=30,
        )
    )
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=REG_FPFH_RADIUS, max_nn=REG_FPFH_MAX_NN,
        ),
    )
    return pcd_down, fpfh


def register_clouds(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    output_dir: Path | None = None,
) -> tuple[np.ndarray, o3d.geometry.PointCloud]:
    """
    Align source (Scan B / bottom) onto target (Scan A / top).

    1. FPFH feature extraction on both clouds
    2. Global registration via RANSAC on FPFH correspondences
    3. Local refinement via Point-to-Plane ICP

    Returns:
      transformation: 4×4 matrix that brings source into target's frame
      source_aligned: transformed source cloud
    """
    print(f"\n{'='*60}")
    print(f"  STITCHER — Global + Local Registration")
    print(f"{'='*60}")

    # ── FPFH Features ────────────────────────────────────────────────────
    print("\n  Computing FPFH features …")
    source_down, source_fpfh = _compute_fpfh(source, REG_VOXEL_SIZE)
    target_down, target_fpfh = _compute_fpfh(target, REG_VOXEL_SIZE)
    print(f"    Source: {len(source_down.points):,} keypoints")
    print(f"    Target: {len(target_down.points):,} keypoints")

    # ── Global Registration (RANSAC) ─────────────────────────────────────
    print("\n  Global Registration (FPFH + RANSAC) …")
    checkers = [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(REG_RANSAC_DISTANCE),
    ]
    global_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down,
        source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=REG_RANSAC_DISTANCE,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=checkers,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            REG_RANSAC_MAX_ITERATIONS, REG_RANSAC_CONFIDENCE,
        ),
    )
    print(f"    Fitness:  {global_result.fitness:.4f}")
    print(f"    RMSE:     {global_result.inlier_rmse:.6f}")

    # ── Local Refinement (Point-to-Plane ICP) ────────────────────────────
    print("\n  Local Refinement (Point-to-Plane ICP) …")
    source_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=REG_VOXEL_SIZE * 2, max_nn=30,
        )
    )
    target_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=REG_VOXEL_SIZE * 2, max_nn=30,
        )
    )

    icp_result = o3d.pipelines.registration.registration_icp(
        source_down, target_down,
        REG_ICP_DISTANCE,
        init=global_result.transformation,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=REG_ICP_MAX_ITER),
    )
    print(f"    Fitness:  {icp_result.fitness:.4f}")
    print(f"    RMSE:     {icp_result.inlier_rmse:.6f}")

    transformation = icp_result.transformation
    source_aligned = copy.deepcopy(source)
    source_aligned.transform(transformation)

    if output_dir is not None:
        o3d.io.write_point_cloud(str(output_dir / "scan_b_aligned.ply"), source_aligned)
        np.save(str(output_dir / "registration_transform.npy"), transformation)

    return transformation, source_aligned


# ═══════════════════════════ MODULE 3: MESHER ════════════════════════════════

def generate_mesh(
    pcd: o3d.geometry.PointCloud,
    colored_pcd: o3d.geometry.PointCloud | None = None,
    output_dir: Path | None = None,
) -> o3d.geometry.TriangleMesh:
    """
    Full meshing pipeline: Poisson reconstruction → distance trim →
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

    # ── Normals ──────────────────────────────────────────────────────────
    print("\n  Estimating normals …")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30),
    )
    pcd.orient_normals_consistent_tangent_plane(k=15)

    # ── Poisson Reconstruction ───────────────────────────────────────────
    print(f"  Poisson reconstruction (depth={POISSON_DEPTH}) …")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=POISSON_DEPTH,
    )
    print(f"    Raw: {len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris")

    # ── Distance-based Trimming ──────────────────────────────────────────
    print(f"  Trimming (max dist={POISSON_TRIM_DISTANCE}) …")
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    verts = np.asarray(mesh.vertices)
    remove = np.zeros(len(verts), dtype=bool)
    sq_thresh = POISSON_TRIM_DISTANCE ** 2

    for i, v in enumerate(verts):
        _, _, dists = pcd_tree.search_knn_vector_3d(v, 1)
        if dists[0] > sq_thresh:
            remove[i] = True

    mesh.remove_vertices_by_mask(remove)
    print(f"    Trimmed: {remove.sum():,} vertices removed")
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


# ═══════════════════════════ PIPELINE ORCHESTRATORS ══════════════════════════

def run_single_scan(input_ply: Path, output_dir: Path):
    """Single-scan pipeline: clean → mesh."""
    output_dir.mkdir(parents=True, exist_ok=True)

    pcd = o3d.io.read_point_cloud(str(input_ply))
    if pcd.is_empty():
        raise RuntimeError(f"Empty point cloud: {input_ply}")
    print(f"📂 Loaded {len(pcd.points):,} points from {input_ply}")

    pcd_clean = clean_cloud(pcd, label="scan", output_dir=output_dir)
    mesh = generate_mesh(pcd_clean, colored_pcd=pcd_clean, output_dir=output_dir)

    final_path = output_dir / "final_mesh.ply"
    o3d.io.write_triangle_mesh(str(final_path), mesh)
    print(f"\n🎉 Done! Final mesh saved to {final_path}")
    return mesh


def run_two_pass(scan_a_ply: Path, scan_b_ply: Path, output_dir: Path):
    """
    Two-pass pipeline:
      1. Clean each scan independently
      2. Register Scan B onto Scan A
      3. Merge, downsample, and mesh
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    pcd_a = o3d.io.read_point_cloud(str(scan_a_ply))
    pcd_b = o3d.io.read_point_cloud(str(scan_b_ply))
    if pcd_a.is_empty() or pcd_b.is_empty():
        raise RuntimeError("One or both scan point clouds are empty.")
    print(f"📂 Scan A: {len(pcd_a.points):,} points from {scan_a_ply}")
    print(f"📂 Scan B: {len(pcd_b.points):,} points from {scan_b_ply}")

    # ── Clean each scan ──────────────────────────────────────────────────
    clean_a = clean_cloud(pcd_a, label="scan_a", output_dir=output_dir)
    clean_b = clean_cloud(pcd_b, label="scan_b", output_dir=output_dir)

    # ── Register Scan B onto Scan A ──────────────────────────────────────
    _, aligned_b = register_clouds(
        source=clean_b, target=clean_a, output_dir=output_dir,
    )

    # ── Merge and unify density ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  FUSION — Merge & Downsample")
    print(f"{'='*60}")

    merged = clean_a + aligned_b
    print(f"  Merged: {len(merged.points):,} points")

    merged = merged.voxel_down_sample(voxel_size=REG_VOXEL_SIZE)
    print(f"  After voxel unification: {len(merged.points):,} points")

    o3d.io.write_point_cloud(str(output_dir / "merged_cloud.ply"), merged)

    # ── Generate final mesh ──────────────────────────────────────────────
    mesh = generate_mesh(merged, colored_pcd=merged, output_dir=output_dir)

    final_path = output_dir / "final_mesh.ply"
    o3d.io.write_triangle_mesh(str(final_path), mesh)
    print(f"\n🎉 Done! Final two-pass mesh saved to {final_path}")
    return mesh


# ═══════════════════════════ CLI ═════════════════════════════════════════════

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.resolve()
    default_input  = project_root / "data" / "processed_data" / "reconstruction.ply"
    default_output = project_root / "data" / "processed_data" / "mesh"

    parser = argparse.ArgumentParser(
        description="3D Mesh Reconstruction — single-scan or two-pass flipped-scan",
    )

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--input", type=str, default=None,
        help="Single-scan mode: path to one .ply point cloud",
    )
    group.add_argument(
        "--scan_a", type=str, default=None,
        help="Two-pass mode: path to Scan A (upright) .ply",
    )

    parser.add_argument(
        "--scan_b", type=str, default=None,
        help="Two-pass mode: path to Scan B (flipped) .ply",
    )
    parser.add_argument(
        "--output", type=str, default=str(default_output),
        help="Output directory for intermediates and final mesh",
    )

    args = parser.parse_args()
    out = Path(args.output)

    if args.scan_a and args.scan_b:
        run_two_pass(Path(args.scan_a), Path(args.scan_b), out)
    elif args.scan_a and not args.scan_b:
        parser.error("--scan_a requires --scan_b for two-pass mode")
    else:
        input_path = Path(args.input) if args.input else default_input
        run_single_scan(input_path, out)

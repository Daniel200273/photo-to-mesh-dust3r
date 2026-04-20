"""
stylize.py — 3D Structural and Visual Filters

Filters:
  - low_poly:    Simplify mesh using voxel clustering
  - voxel:       Convert mesh to a voxel grid (Minecraft style)
  - soft_voxel:  Downsample to a voxel grid with spheres (smooth voxel look)
  - hologram:    Extract edges and apply neon wireframe aesthetic
  - ff7:         PS1-era faceted look (aggressive decimation + flat shading)
  - material:    Replace vertex colours with physical textures

Usage:
  python stylize.py --input mesh.ply --filter low_poly       --param 1500 --output styled.ply
    python stylize.py --input mesh.ply --filter low_poly       --param 1500 --close --output closed_lowpoly.ply
  python stylize.py --input mesh.ply --filter material       --texture path/to.jpg --output styled.ply
  python stylize.py --input mesh.ply --filter hologram       --color 0.035,0.714,0.902 --output holo.ply
"""

import open3d as o3d
import numpy as np
import sys
from pathlib import Path

# ═══════════════════════════ PARAMETERS ══════════════════════════════════════

# LOW-POLY Filter
LOW_POLY_TARGET_TRIANGLES = 1500           # Target triangle count after decimation
LOW_POLY_SMOOTHING_ITERS  = 3              # Taubin smoothing iterations
LOW_POLY_SUBDIVIDE_ITERS  = 2              # Midpoint subdivision iterations
LOW_POLY_VOXEL_DIVISOR    = 20             # Voxel size = bbox_size / divisor

# VOXEL Filter
VOXEL_SIZE_DEFAULT        = 0.01           # Default voxel cube size
VOXEL_SAMPLE_DENSITY      = 2.0            # Multiplier for point sampling
VOXEL_MAX_CUBES           = 200000         # Safety cap to avoid native Open3D crashes
VOXEL_MAX_TRIANGLES       = 3000000        # Safe write threshold for voxel meshes

# SOFT-VOXEL Filter
SOFT_VOXEL_SIZE_DEFAULT   = 0.01           # Default soft voxel size
SOFT_VOXEL_SPHERE_RADIUS_MULT = 0.6       # Sphere radius = voxel_size * this
SOFT_VOXEL_MAX_NODES      = 150000         # Safety cap for sphere node count

# HOLOGRAM Filter
HOLOGRAM_TARGET_TRIANGLES = 2000           # Target tris for low-poly base
HOLOGRAM_NEON_COLOR       = [0.035, 0.714, 0.902]  # #09b6e6 bright cyan
HOLOGRAM_BODY_COLOR       = [0.259, 0.510, 0.961]  # #4287f5 blue
HOLOGRAM_HOLE_CLOSE_MAX_TRIANGLES = 250000  # Auto-repair body holes below this density

# FF7 Filter
FF7_TARGET_TRIANGLES      = 800            # Very low poly (PS1 character range)
FF7_COLOR_LEVELS          = 16             # Colour quantization steps per channel
                                           # (PS1 = 32 levels / 5-bit, 16 = more stylised)
# ═══════════════════════════ FILTERS ═════════════════════════════════════════

def decouple_geometry(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """
    Shatters a mesh so every triangle has 3 mathematically unique, unshared vertices.
    This strictly enforces Flat Shading by preventing the rendering engine from 
    averaging vertex normals across sharp geometric boundaries (which causes 'missing faces' artifacts).
    """
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)
    
    new_vertices = vertices[triangles].reshape(-1, 3)
    new_triangles = np.arange(len(new_vertices)).reshape(-1, 3)
    
    flat_mesh = o3d.geometry.TriangleMesh()
    flat_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    flat_mesh.triangles = o3d.utility.Vector3iVector(new_triangles)
    
    if len(mesh.vertex_colors) > 0:
        colors = np.asarray(mesh.vertex_colors)
        new_colors = colors[triangles].reshape(-1, 3)
        flat_mesh.vertex_colors = o3d.utility.Vector3dVector(new_colors)
        
    flat_mesh.compute_triangle_normals()
    return flat_mesh


def close_mesh_holes(
    mesh: o3d.geometry.TriangleMesh,
    label: str = "mesh",
) -> o3d.geometry.TriangleMesh:
    """Attempt to close open boundary loops while preserving vertex colors."""
    import trimesh

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()

    verts = np.asarray(mesh.vertices)
    tris = np.asarray(mesh.triangles)
    if len(verts) == 0 or len(tris) == 0:
        return mesh

    has_colors = len(mesh.vertex_colors) > 0
    src_colors = np.asarray(mesh.vertex_colors) if has_colors else None

    tm = trimesh.Trimesh(vertices=verts, faces=tris, process=False)
    was_watertight = bool(tm.is_watertight)
    faces_before = len(tm.faces)
    trimesh.repair.fill_holes(tm)

    # fill_holes is conservative; for large open bottoms, try a simple stitch fallback.
    if not tm.is_watertight:
        try:
            stitched = trimesh.repair.stitch(tm, insert_vertices=True)
            stitched = np.asarray(stitched, dtype=np.int64)
            if stitched.size > 0:
                stitched = stitched.reshape(-1, 3)
                tm.faces = np.vstack([tm.faces, stitched])
                trimesh.repair.fill_holes(tm)
        except Exception:
            pass

    is_watertight = bool(tm.is_watertight)
    faces_after = len(tm.faces)

    print(f"  → Closing holes ({label}): watertight {was_watertight} -> {is_watertight}, "
          f"faces {faces_before:,} -> {faces_after:,}")

    closed = o3d.geometry.TriangleMesh()
    closed.vertices = o3d.utility.Vector3dVector(tm.vertices)
    closed.triangles = o3d.utility.Vector3iVector(tm.faces)

    if has_colors and src_colors is not None and len(src_colors) > 0:
        new_verts = np.asarray(tm.vertices)
        if len(new_verts) == len(verts):
            new_colors = src_colors
        else:
            try:
                from scipy.spatial import cKDTree
                tree = cKDTree(verts)
                _, idx = tree.query(new_verts, workers=-1)
                new_colors = src_colors[idx]
            except Exception:
                avg = src_colors.mean(axis=0)
                new_colors = np.tile(avg, (len(new_verts), 1))
        closed.vertex_colors = o3d.utility.Vector3dVector(new_colors)

    closed.compute_vertex_normals()
    closed.compute_triangle_normals()
    return closed


def _pedestal_alignment_frame(
    pedestal_path: str | None,
    object_center: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Build world<->local rotation matrices from pedestal plane orientation."""
    if not pedestal_path:
        return None

    ped_path = Path(pedestal_path)
    if not ped_path.exists():
        return None

    pedestal = o3d.io.read_triangle_mesh(str(ped_path))
    if pedestal.is_empty():
        return None

    verts = np.asarray(pedestal.vertices)
    if len(verts) == 0:
        return None

    centered = verts - verts.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    up = vh[-1]
    up_norm = np.linalg.norm(up)
    if up_norm < 1e-12:
        return None
    up = up / up_norm

    if object_center is not None:
        ped_center = verts.mean(axis=0)
        if np.dot(object_center - ped_center, up) < 0:
            up = -up

    ref = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    tangent1 = np.cross(up, ref)
    t1_norm = np.linalg.norm(tangent1)
    if t1_norm < 1e-12:
        return None
    tangent1 = tangent1 / t1_norm
    tangent2 = np.cross(up, tangent1)
    t2_norm = np.linalg.norm(tangent2)
    if t2_norm < 1e-12:
        return None
    tangent2 = tangent2 / t2_norm

    # Rows are local basis vectors in world coordinates.
    world_to_local = np.vstack([tangent1, up, tangent2])
    local_to_world = world_to_local.T
    return world_to_local, local_to_world


def _prepare_mesh_for_pedestal_aligned_voxelization(
    mesh: o3d.geometry.TriangleMesh,
    pedestal_path: str | None,
) -> tuple[o3d.geometry.TriangleMesh, np.ndarray | None]:
    """Rotate mesh into pedestal-aligned local frame for voxel-style filters."""
    mesh_work = o3d.geometry.TriangleMesh(mesh)
    local_to_world = None

    frame = _pedestal_alignment_frame(
        pedestal_path=pedestal_path,
        object_center=np.asarray(mesh_work.vertices).mean(axis=0) if len(mesh_work.vertices) > 0 else None,
    )
    if frame is not None:
        world_to_local, local_to_world = frame
        mesh_work.rotate(world_to_local, center=(0, 0, 0))
        print("  → Aligning voxel axes to pedestal plane...")

    return mesh_work, local_to_world


def apply_low_poly(
    mesh: o3d.geometry.TriangleMesh,
    target_triangles: int = LOW_POLY_TARGET_TRIANGLES,
    force_close: bool = False,
) -> o3d.geometry.TriangleMesh:
    """
    Simplify mesh to uniform low-poly style.

    Steps:
      1. Clean degenerate/duplicate geometry
      2. Taubin smoothing (volume-preserving noise removal)
      3. Midpoint subdivision (uniform triangles)
            4. Optional voxel pre-pass (performance)
            5. Quadric decimation to requested target triangles
            6. Final cleanup
    """
    if target_triangles <= 0:
        raise ValueError("low_poly target_triangles must be > 0")

    print(f"Applying low-poly filter (target: {target_triangles} tris)...")

    # Clean
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()

    # Taubin smoothing
    print(f"  → Taubin smoothing ({LOW_POLY_SMOOTHING_ITERS} iterations)...")
    mesh = mesh.filter_smooth_taubin(number_of_iterations=LOW_POLY_SMOOTHING_ITERS)

    # Subdivide for uniform triangles
    print(f"  → Subdividing ({LOW_POLY_SUBDIVIDE_ITERS} iterations)...")
    mesh = mesh.subdivide_midpoint(number_of_iterations=LOW_POLY_SUBDIVIDE_ITERS)

    # Optional voxel pre-pass: speed up very dense meshes before exact decimation.
    # If the pre-pass drops below the requested target, we skip it and decimate from
    # the subdivided mesh so the user-provided target remains authoritative.
    mesh_simplified = mesh
    tri_count = len(mesh.triangles)
    if tri_count > target_triangles * 4:
        bbox = mesh.get_axis_aligned_bounding_box()
        size = np.linalg.norm(bbox.get_extent())
        voxel_size = size / (np.sqrt(float(target_triangles)) * 1.5)

        print(f"  → Voxel pre-pass (size: {voxel_size:.4f})...")
        mesh_candidate = mesh.simplify_vertex_clustering(
            voxel_size=voxel_size,
            contraction=o3d.geometry.SimplificationContraction.Average,
        )
        mesh_candidate.remove_degenerate_triangles()
        mesh_candidate.remove_unreferenced_vertices()

        if len(mesh_candidate.triangles) >= target_triangles:
            mesh_simplified = mesh_candidate
            print(f"    after pre-pass: {len(mesh_simplified.triangles):,} tris")
        else:
            print(
                f"    pre-pass undershot target ({len(mesh_candidate.triangles):,} < {target_triangles:,}); "
                "using direct decimation"
            )

    if len(mesh_simplified.triangles) > target_triangles:
        print(f"  → Quadric decimation to {target_triangles} triangles...")
        mesh_simplified = mesh_simplified.simplify_quadric_decimation(
            target_number_of_triangles=target_triangles
        )
    else:
        print(
            f"  → Input already at or below target ({len(mesh_simplified.triangles):,} <= {target_triangles:,}); "
            "skipping decimation"
        )

    # Final cleanup
    mesh_simplified.remove_degenerate_triangles()
    mesh_simplified.remove_unreferenced_vertices()

    if force_close:
        mesh_simplified = close_mesh_holes(mesh_simplified, label="low_poly")
    
    # Decouple the geometry to permanently lock-in crisp flat shading without PBR artifacts
    print("  → Decoupling facets for True Flat Shading...")
    mesh_simplified = decouple_geometry(mesh_simplified)

    print(f"  ✓ Result: {len(mesh_simplified.vertices):,} verts, {len(mesh_simplified.triangles):,} tris")
    return mesh_simplified


def apply_voxel(
    mesh: o3d.geometry.TriangleMesh,
    voxel_size: float = VOXEL_SIZE_DEFAULT,
    pedestal_path: str | None = None,
) -> o3d.geometry.TriangleMesh:
    """
    Convert mesh into Minecraft-style voxel blocks.

    Process:
      1. Sample mesh surface points uniformly
      2. Create voxel grid from point cloud
      3. Reconstruct as individual cubes with original colors
    """
    print(f"Applying voxel filter (size: {voxel_size})...")

    mesh_work, local_to_world = _prepare_mesh_for_pedestal_aligned_voxelization(mesh, pedestal_path)

    # Sample mesh surface
    num_samples = max(int(len(mesh_work.vertices) * VOXEL_SAMPLE_DENSITY), 200000)
    print(f"  → Sampling {num_samples:,} points...")
    pcd = mesh_work.sample_points_uniformly(number_of_points=num_samples)

    # Create voxel grid
    print(f"  → Creating voxel grid...")
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    voxels = voxel_grid.get_voxels()
    print(f"  → Reconstructing {len(voxels)} voxel cubes...")

    if len(voxels) > VOXEL_MAX_CUBES:
        raise RuntimeError(
            f"Voxel grid too dense: {len(voxels):,} cubes exceeds safety limit {VOXEL_MAX_CUBES:,}. "
            "Increase voxel size (recommended >= 0.003)."
        )

    # Reconstruct mesh from voxels
    voxel_mesh = o3d.geometry.TriangleMesh()
    for v in voxels:
        center = voxel_grid.get_voxel_center_coordinate(v.grid_index)
        cube = o3d.geometry.TriangleMesh.create_box(
            width=voxel_size, height=voxel_size, depth=voxel_size
        )
        cube.translate(center - np.array([voxel_size/2, voxel_size/2, voxel_size/2]))
        cube.paint_uniform_color(v.color)
        voxel_mesh += cube

    if len(voxels) == 0:
        print("  ⚠ Warning: No voxels generated, returning original mesh")
        return mesh

    if local_to_world is not None:
        voxel_mesh.rotate(local_to_world, center=(0, 0, 0))

    voxel_mesh.compute_vertex_normals()
    print(f"  ✓ Result: {len(voxel_mesh.vertices):,} verts, {len(voxel_mesh.triangles):,} tris")
    return voxel_mesh


def apply_soft_voxel(
    mesh: o3d.geometry.TriangleMesh,
    voxel_size: float = SOFT_VOXEL_SIZE_DEFAULT,
    pedestal_path: str | None = None,
) -> o3d.geometry.TriangleMesh:
    """
    Create smooth voxel effect using overlapping spheres.

    Process:
      1. Sample mesh to get colors
      2. Voxel downsample the point cloud
      3. Place small spheres at each voxel center
    """
    print(f"Applying soft-voxel filter (size: {voxel_size})...")

    mesh_work, local_to_world = _prepare_mesh_for_pedestal_aligned_voxelization(mesh, pedestal_path)

    # Sample and downsample
    pcd = mesh_work.sample_points_uniformly(number_of_points=len(mesh_work.vertices) * 2)
    print(f"  → Downsampling to voxel grid...")
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)

    points = np.asarray(pcd_down.points)
    colors = np.asarray(pcd_down.colors)
    sphere_radius = voxel_size * SOFT_VOXEL_SPHERE_RADIUS_MULT

    if len(points) > SOFT_VOXEL_MAX_NODES:
        raise RuntimeError(
            f"Soft-voxel grid too dense: {len(points):,} nodes exceeds safety limit {SOFT_VOXEL_MAX_NODES:,}. "
            "Increase voxel size (recommended >= 0.003)."
        )

    print(f"  → Creating {len(points)} sphere nodes...")
    final_mesh = o3d.geometry.TriangleMesh()

    for i, (pt, col) in enumerate(zip(points, colors)):
        node = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius, resolution=5)
        node.translate(pt)
        node.paint_uniform_color(col)
        final_mesh += node

    if local_to_world is not None:
        final_mesh.rotate(local_to_world, center=(0, 0, 0))

    final_mesh.compute_vertex_normals()
    print(f"  ✓ Result: {len(final_mesh.vertices):,} verts, {len(final_mesh.triangles):,} tris")
    return final_mesh


def _should_preserve_hologram_geometry(mesh: o3d.geometry.TriangleMesh) -> bool:
    """Heuristic: keep geometry when mesh is already low-poly or voxel-like."""
    tri_count = len(mesh.triangles)
    if tri_count == 0:
        return False

    # Low-poly outputs are already in the target style.
    if tri_count <= int(HOLOGRAM_TARGET_TRIANGLES * 1.5):
        return True

    # Reuse robust voxel-like detector (orientation-invariant).
    return _is_voxel_like_mesh(mesh)


def _prepare_hologram_base_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Clean and repair boundaries so hologram body does not look full of holes."""
    prepared = o3d.geometry.TriangleMesh(mesh)
    prepared.remove_degenerate_triangles()
    prepared.remove_duplicated_vertices()
    prepared.remove_unreferenced_vertices()

    tri_count = len(prepared.triangles)
    if tri_count == 0:
        return prepared

    watertight = False
    try:
        watertight = bool(prepared.is_watertight())
    except Exception:
        watertight = False

    if not watertight:
        if tri_count <= HOLOGRAM_HOLE_CLOSE_MAX_TRIANGLES:
            print("  → Repairing open boundaries for hologram body...")
            try:
                # Hologram body is repainted, so skip expensive color transfer work.
                repair_in = o3d.geometry.TriangleMesh(prepared)
                repair_in.vertex_colors = o3d.utility.Vector3dVector()
                prepared = close_mesh_holes(repair_in, label="hologram")
            except Exception as e:
                print(f"  ⚠ Hologram hole repair failed, proceeding without repair: {e}")
        else:
            print(
                f"  → Skipping hole repair on very dense mesh ({tri_count:,} tris > "
                f"{HOLOGRAM_HOLE_CLOSE_MAX_TRIANGLES:,})"
            )

    # Try to orient winding consistently to reduce backface-culling artifacts.
    try:
        prepared.orient_triangles()
    except Exception:
        pass

    return prepared


def apply_hologram(
    mesh: o3d.geometry.TriangleMesh,
    neon_color: list = HOLOGRAM_NEON_COLOR,
    simplify_first: bool = True
) -> dict:
    """
    Create hologram effect: body tint + neon wireframe edges.
    For already low-poly / voxel-like meshes, geometry is preserved.

    Returns dict with 'body' (mesh) and 'edges' (LineSet) for composite rendering.
    """
    print("Applying hologram effect...")

    neon = np.asarray(neon_color, dtype=float).reshape(-1)
    if len(neon) != 3:
        neon = np.asarray(HOLOGRAM_NEON_COLOR, dtype=float)
    neon = np.clip(neon[:3], 0.0, 1.0)

    # Simplify to low-poly for detailed inputs only.
    preserve_geometry = simplify_first and _should_preserve_hologram_geometry(mesh)
    if simplify_first and not preserve_geometry:
        print("  → Simplifying mesh to low-poly...")
        mesh_base = apply_low_poly(
            mesh,
            target_triangles=HOLOGRAM_TARGET_TRIANGLES,
            force_close=True,
        )
    elif simplify_first and preserve_geometry:
        print("  → Preserving source geometry (voxel/low-poly detected)...")
        mesh_base = mesh
    else:
        mesh_base = mesh

    mesh_base = _prepare_hologram_base_mesh(mesh_base)

    # Create body
    body_color = np.clip(0.15 + 0.55 * neon, 0.0, 1.0)
    print(f"  → Creating hologram body tint RGB({body_color[0]:.3f}, {body_color[1]:.3f}, {body_color[2]:.3f})...")
    mesh_body = o3d.geometry.TriangleMesh(mesh_base)
    mesh_body.vertex_colors = o3d.utility.Vector3dVector()
    mesh_body.paint_uniform_color(body_color.tolist())
    mesh_body.compute_vertex_normals()

    # Extract edges
    print("  → Extracting wireframe edges...")
    triangles = np.asarray(mesh_base.triangles)
    edge_set = set()

    for tri in triangles:
        for j in range(3):
            a, b = sorted((tri[j], tri[(j + 1) % 3]))
            edge_set.add((a, b))

    all_edges = [[a, b] for a, b in edge_set]

    # Create LineSet
    edges = o3d.geometry.LineSet()
    edges.points = mesh_base.vertices
    edges.lines = o3d.utility.Vector2iVector(all_edges)
    edges.paint_uniform_color(neon.tolist())

    print(f"  ✓ Hologram: {len(mesh_body.vertices):,} verts, {len(all_edges)} edges")

    return {
        'body': mesh_body,
        'edges': edges,
        'neon_color': neon.tolist(),
        'body_color': body_color.tolist(),
    }


def apply_ff7(
    mesh: o3d.geometry.TriangleMesh,
    target_triangles: int = FF7_TARGET_TRIANGLES,
    color_levels: int = FF7_COLOR_LEVELS,
    force_close: bool = False,
) -> o3d.geometry.TriangleMesh:
    """
    Final Fantasy 7 (PS1 era) aesthetic.

    Key properties:
      1. Aggressive decimation — chunky, angular, faceted silhouette
      2. Flat shading: vertices are exploded (one unique set per triangle)
         so zero colour blending occurs across face boundaries
      3. Posterized colour palette: each face's average colour is quantized
         to `color_levels` steps per channel (PS1 was 32=5-bit; 16 is more stylised)

    Args:
      target_triangles: polygon budget (default 800, range 300–2000)
      color_levels:     quantization steps per RGB channel (default 16)
    """
    print(f"Applying FF7 filter "
          f"(target: {target_triangles} tris, {color_levels} colour levels) …")

    # ── 1. Decimate ─────────────────────────────────────────────────────
    # Open3D's simplify_quadric_decimation has a hard topological floor
    # and silently stops far above the requested target on complex meshes.
    # Fix: two-stage decimation —
    #   Stage A: voxel clustering (unconstrained, always reaches any target)
    #            to bring the mesh to ≈ 4× the target triangle count.
    #   Stage B: quadric decimation (shape-quality refinement) to hit the
    #            exact target.  At 4× input size it reliably converges.
    #
    # Colors are stripped before decimation (quadric decimation in Open3D
    # misbehaves with per-vertex colour) and re-transferred afterward via
    # nearest-neighbour lookup on the original point cloud.
    has_colors = len(mesh.vertex_colors) > 0
    if has_colors:
        orig_pts  = np.asarray(mesh.vertices).copy()
        orig_cols = np.asarray(mesh.vertex_colors).copy()

    # Lightweight clean (skip remove_non_manifold_edges — O(n²), hangs on
    # large meshes; both decimation methods tolerate non-manifold input).
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()

    # Strip colours so decimation operates on geometry only
    mesh.vertex_colors = o3d.utility.Vector3dVector()

    n_tris_input = len(mesh.triangles)
    print(f"  → Decimating to {target_triangles} triangles …")

    # Stage A — voxel clustering: drives the mesh near the target
    if n_tris_input > target_triangles * 2:
        # Pick a voxel size that should land at ~4× target (generous margin
        # so Stage B always has room to reduce further).
        bbox      = mesh.get_axis_aligned_bounding_box()
        extent    = np.linalg.norm(bbox.get_extent())
        voxel_size = extent / (np.sqrt(target_triangles) * 1.5)
        mesh = mesh.simplify_vertex_clustering(
            voxel_size=voxel_size,
            contraction=o3d.geometry.SimplificationContraction.Average,
        )
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
        print(f"    after voxel pass : {len(mesh.triangles):,} tris")

    # Stage B — quadric decimation: shape-quality fine pass
    if len(mesh.triangles) > target_triangles:
        mesh = mesh.simplify_quadric_decimation(
            target_number_of_triangles=target_triangles
        )
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()

    if force_close:
        mesh = close_mesh_holes(mesh, label="ff7")

    print(f"    ✓ {len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris")

    # Re-transfer colours from original mesh via nearest-neighbour lookup
    if has_colors:
        from scipy.spatial import cKDTree
        tree = cKDTree(orig_pts)
        _, idx = tree.query(np.asarray(mesh.vertices), workers=-1)
        mesh.vertex_colors = o3d.utility.Vector3dVector(orig_cols[idx])

    # ── 2. Flat shading via vertex explosion ───────────────────────────
    # In a normal mesh, adjacent faces share vertices, so vertex colours are
    # interpolated across the boundary.  By giving every triangle its own
    # unique three vertices, there is no shared vertex → no interpolation →
    # true per-face flat colour.
    print("  → Exploding vertices for flat shading …")
    verts      = np.asarray(mesh.vertices)
    tris       = np.asarray(mesh.triangles)
    has_colors = len(mesh.vertex_colors) > 0
    src_colors = np.asarray(mesh.vertex_colors) if has_colors else None

    # New vertex array: each row of tris becomes 3 consecutive vertices
    new_verts = verts[tris.reshape(-1)]                        # (n_tris×3, 3)
    new_tris  = np.arange(len(new_verts)).reshape(-1, 3)       # (n_tris, 3)

    # ── 3. Per-face colour posterization ──────────────────────────────
    if has_colors:
        print(f"  → Posterizing colours to {color_levels} levels per channel …")
        face_colors = src_colors[tris]               # (n_tris, 3 verts, 3 RGB)
        avg_colors  = face_colors.mean(axis=1)       # (n_tris, 3) — one colour/face
        # Quantize: snap each channel to nearest step
        avg_colors  = np.floor(avg_colors * color_levels) / color_levels
        avg_colors  = np.clip(avg_colors, 0.0, 1.0)
        new_colors  = np.repeat(avg_colors, 3, axis=0)
    else:
        new_colors = None

    flat_mesh = o3d.geometry.TriangleMesh()
    flat_mesh.vertices  = o3d.utility.Vector3dVector(new_verts)
    flat_mesh.triangles = o3d.utility.Vector3iVector(new_tris)
    if new_colors is not None:
        flat_mesh.vertex_colors = o3d.utility.Vector3dVector(new_colors)
    flat_mesh.compute_triangle_normals()

    print(f"  ✓ Result: {len(flat_mesh.vertices):,} verts, "
          f"{len(flat_mesh.triangles):,} tris  (FF7 style)")
    return flat_mesh


def apply_smooth(
    mesh: o3d.geometry.TriangleMesh,
    iterations: int = 5,
    sigma: float = 0.0,
) -> o3d.geometry.TriangleMesh:
    """
    Gaussian-weighted spatial smoothing for both Geometry and Color.

    Args:
      iterations: number of Gaussian-Laplacian passes.
      sigma:      Gaussian radius in mesh units (0 = auto).
    """
    from scipy.spatial import cKDTree

    print(f"Applying Gaussian-Laplacian smooth filter "
          f"({iterations} passes, sigma={'auto' if sigma <= 0 else f'{sigma:.4f}'}) …")

    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()

    import trimesh
    print("  → Filling holes …")
    verts_before = np.asarray(mesh.vertices)
    has_colors = len(mesh.vertex_colors) > 0
    colors_before = np.asarray(mesh.vertex_colors) if has_colors else None
    
    tm = trimesh.Trimesh(
        vertices=verts_before, 
        faces=np.asarray(mesh.triangles), 
        process=False
    )
    trimesh.repair.fill_holes(tm)
    
    mesh.triangles = o3d.utility.Vector3iVector(tm.faces)
    verts = np.asarray(tm.vertices).copy()
    
    if has_colors:
        n_added = len(verts) - len(verts_before)
        if n_added > 0:
            print(f"    Filled holes: added {n_added} new vertices")
            added_colors = np.full((n_added, 3), 0.5)
            colors = np.vstack([colors_before, added_colors]).copy()
        else:
            colors = colors_before.copy()
    else:
        colors = None

    orig_verts = verts.copy()

    if sigma <= 0:
        sample_pts = verts[np.random.choice(len(verts), min(len(verts), 10000), replace=False)]
        tree_sample = cKDTree(sample_pts)
        d, _ = tree_sample.query(sample_pts, k=2, workers=-1)
        mean_edge = d[:, 1].mean()
        sigma = mean_edge * 2.0
        print(f"  → Auto sigma = {sigma:.5f}  (from sample mean NN dist={mean_edge:.5f})")

    inv_2sig2 = 1.0 / (2.0 * sigma * sigma)

    print(f"  → Building KD-Tree for spatial neighbors …")
    tree = cKDTree(verts)
    dists, indices = tree.query(verts, k=10, workers=-1)

    weights = np.exp(-(dists**2) * inv_2sig2)
    weights /= weights.sum(axis=1, keepdims=True)

    print("  → Blurring geometry and colors …")
    for it in range(iterations):
        verts = (weights[:, :, None] * verts[indices]).sum(axis=1)
        if has_colors:
            colors = (weights[:, :, None] * colors[indices]).sum(axis=1)

        print(f"    iteration {it + 1}/{iterations} done")

    mu = 0.1 * iterations
    displacement = verts - orig_verts
    verts = verts - mu * displacement
    print(f"  → Shrinkage correction applied to geometry (μ={mu:.2f})")

    mesh.vertices = o3d.utility.Vector3dVector(verts)
    if has_colors:
        # Clip colors safely to [0, 1] range after blending
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))

    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    print(f"  ✓ Result: {len(mesh.vertices):,} verts (Geometry + Color smoothed)")
    return mesh


def _sample_texture(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorized texture sampler taking UVs in [0, 1] and handling wrapping."""
    H, W = img.shape[:2]
    u = u % 1.0
    v = v % 1.0
    x = (u * (W - 1)).astype(int)
    y = ((1.0 - v) * (H - 1)).astype(int)
    
    sampled = img[y, x] / 255.0
    if sampled.ndim == 1:
        sampled = np.stack([sampled]*3, axis=-1)
    elif sampled.shape[-1] == 4:
        sampled = sampled[:, :3]
    return sampled


def _is_voxel_like_mesh(mesh: o3d.geometry.TriangleMesh) -> bool:
    """Heuristic detector for blocky voxel meshes (orientation-invariant)."""
    if len(mesh.triangles) == 0:
        return False

    mesh.compute_triangle_normals()
    tri_normals = np.asarray(mesh.triangle_normals)
    if len(tri_normals) == 0:
        return False

    # Subsample for speed on very large meshes.
    if len(tri_normals) > 200000:
        step = max(1, len(tri_normals) // 200000)
        tri_normals = tri_normals[::step]

    norms = np.linalg.norm(tri_normals, axis=1, keepdims=True)
    tri_normals = tri_normals / np.maximum(norms, 1e-12)

    # Orientation-invariant signature: opposite normals collapse via abs().
    abs_normals = np.abs(tri_normals)
    quantized = np.round(abs_normals, 3)
    unique_dirs = len(np.unique(quantized, axis=0))

    # Voxel meshes (even rotated) generally have very few dominant face directions.
    if unique_dirs <= 20:
        return True

    # Fallback for axis-aligned voxel meshes.
    axis_aligned_ratio = float((np.max(abs_normals, axis=1) > 0.995).mean())
    return axis_aligned_ratio > 0.9

def apply_material(
    mesh: o3d.geometry.TriangleMesh,
    texture_path: str = None,
    shininess: float = 48.0,
    scale: float = 1.0,
    preserve_geometry: bool = False,
) -> o3d.geometry.TriangleMesh:
    """
    Replace vertex colours with a material texture driven by 3-D vertex positions.
    UV mapping issue is circumvented using Image-based Triplanar Mapping.
    """
    print(f"Applying material filter…")


    original_verts = len(mesh.vertices)
    voxel_like = _is_voxel_like_mesh(mesh)
    preserve = preserve_geometry or voxel_like
    if preserve_geometry:
        print("  → Preserve-geometry mode enabled: skipping smoothing/subdivision")
    
    if original_verts > 150000 and not preserve:
        # High resolution organic NeRF meshes contain millions of microscopic surface bumps.
        # Under intense PBR lighting, these bumps act like randomized mirrors, causing extreme flickering.
        print("  → Ironing out high-res micro-surface noise to prevent lighting alias flickering...")
        mesh = mesh.filter_smooth_taubin(number_of_iterations=5)
    elif original_verts > 150000 and preserve:
        print("  → Voxel-like mesh detected: skipping smoothing to preserve block structure")
        
    if texture_path and Path(texture_path).exists():
        # High-res textures require high vertex density because we bake into vertex_colors.
        # If the mesh is low-poly, it physically lacks the "pixels" to show the image details!
        # subdivide_midpoint adds vertices but strictly preserves flat, blocky geometry!
        TARGET_MIN_VERTS = 150000
        if len(mesh.vertices) < TARGET_MIN_VERTS and not preserve:
            print(f"  → Low vertex count detected. Subdividing geometry to increase texture resolution...")
            while len(mesh.vertices) < TARGET_MIN_VERTS:
                mesh = mesh.subdivide_midpoint(number_of_iterations=1)
            print(f"  → Geometry boosted to {len(mesh.vertices)} vertices for crisp textures!")
        elif preserve:
            print("  → Voxel-like mesh detected: skipping auto-subdivision")

    pts = np.asarray(mesh.vertices)
    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)

    # Normalize vertex coordinates to [0, 1]³ so that texture scaling is independent
    # of the mesh's absolute physical size (prevents extreme zoom-in blurring on tiny meshes)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    rng_size = np.maximum(hi - lo, 1e-8)
    pts_n = (pts - lo) / rng_size

    # ── 1. Calculate Base Colors (Albedo) ──
    if texture_path and Path(texture_path).exists():
        # IMAGE-BASED TRIPLANAR MAPPING
        print(f"  → Loading seamless texture: {Path(texture_path).name}")
        try:
            # We use Open3D's image IO
            img_o3d = o3d.io.read_image(texture_path)
            img_arr = np.asarray(img_o3d)
            if img_arr.size == 0:
                raise ValueError("Image empty")
                
            # Drastically bump sharpness (exponent 16) to prevent textures from blending/blurring 
            # into a muddy mess on diagonal surfaces.
            weights = np.abs(normals) ** 16.0
            weights /= np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-8)
            
            # Use normalized coordinates so scale=1.0 means exactly 1 repetition across the object
            uv_x = pts_n[:, [2, 1]] * scale
            uv_y = pts_n[:, [0, 2]] * scale
            uv_z = pts_n[:, [0, 1]] * scale
            
            col_x = _sample_texture(img_arr, uv_x[:, 0], uv_x[:, 1])
            col_y = _sample_texture(img_arr, uv_y[:, 0], uv_y[:, 1])
            col_z = _sample_texture(img_arr, uv_z[:, 0], uv_z[:, 1])
            
            base_col = col_x * weights[:, [0]] + col_y * weights[:, [1]] + col_z * weights[:, [2]]
        except Exception as e:
            print(f"  ⚠ Failed to load texture {texture_path}. Falling back to grey. Error: {e}")
            base_col = np.full((len(pts), 3), 0.5)
    else:
        # Safe fallback: keep existing mesh colors when available, otherwise neutral albedo.
        if len(mesh.vertex_colors) > 0:
            print("  → No texture provided; reusing existing vertex colors as albedo")
            base_col = np.asarray(mesh.vertex_colors)
        else:
            print("  → No texture provided; using neutral grey albedo")
            base_col = np.full((len(pts), 3), 0.55)

    # ── 2. Real-Time Lighting Handoff ──
    # We DO NOT bake diffuse/specular lighting into the vertex colors! 
    # Baking view-dependent lighting (like specular gloss or cast shadows) paints them permanently onto the 
    # geometry. When you rotate the object, the highlights rotate with it like a painted white sticker.
    # Instead, we provide the pure Albedo (texture color) and allow the interactive 3D engine to render real-time lighting!

    if base_col.ndim == 1:
        base_col = np.tile(base_col, (len(pts), 1))
        
    colors = np.clip(base_col, 0.0, 1.0)
    
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh.compute_vertex_normals()

    print(f"  ✓ Result: {len(mesh.vertices):,} verts — material applied (shininess: {shininess})")
    return mesh




# ═══════════════════════════ CLI ═════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Apply stylization filters to 3D meshes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
filter / --param pairs
  low_poly   --param <int>    target triangle count          (default 1500)
  voxel      --param <float>  voxel cube size                (default 0.01)
  soft_voxel --param <float>  sphere / voxel size            (default 0.01)
  ff7        --param <int>    target triangle count          (default 800)
  hologram   --color R,G,B   neon edge colour override
    --close                    force-close open boundary holes (low_poly/ff7)
""")
    parser.add_argument("--input",    type=str, required=True,
                        help="Input mesh file (.ply/.obj)")
    parser.add_argument("--filter",   type=str, default="low_poly",
                        choices=["low_poly", "voxel", "soft_voxel", "hologram",
                                 "ff7", "material"],
                        help="Filter to apply (see epilog for --param meaning)")
    parser.add_argument("--output",  type=str, required=True,
                        help="Output path for styled mesh(es)")
    parser.add_argument("--param",   type=str, default=None,
                        help="Filter parameter — see epilog for each filter's meaning")
    parser.add_argument("--color",   type=str, default=None,
                        help="Hologram edge colour as CSV: R,G,B (e.g. 0.035,0.714,0.902)")
    parser.add_argument("--pedestal", type=str, default=None,
                        help="Optional pedestal mesh path used as orientation reference for voxel filters")
    parser.add_argument("--close", action="store_true",
                        help="Force close open boundaries for low_poly and ff7 filters")
    parser.add_argument("--texture", type=str, default=None,
                        help="Path to seamless image texture for the material filter")
    parser.add_argument("--shininess", type=float, default=48.0,
                        help="Specular shininess for material lighting")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Texture scale multiplier for triplanar mapping")
    parser.add_argument("--preserve-geometry", action="store_true",
                        help="For material filter: skip smoothing/subdivision to preserve original geometry")

    args = parser.parse_args()

    # Load mesh
    print(f"Loading mesh from {args.input}...")
    mesh = o3d.io.read_triangle_mesh(args.input)
    if mesh.is_empty():
        print("Error: Mesh is empty")
        exit(1)

    # ── Apply filter ──────────────────────────────────────────────────────
    if args.filter == "low_poly":
        target = int(args.param) if args.param else LOW_POLY_TARGET_TRIANGLES
        styled = apply_low_poly(mesh, target, force_close=args.close)

    elif args.filter == "voxel":
        vsize  = float(args.param) if args.param else VOXEL_SIZE_DEFAULT
        styled = apply_voxel(mesh, vsize, pedestal_path=args.pedestal)

    elif args.filter == "soft_voxel":
        vsize  = float(args.param) if args.param else SOFT_VOXEL_SIZE_DEFAULT
        styled = apply_soft_voxel(mesh, vsize, pedestal_path=args.pedestal)

    elif args.filter == "hologram":
        color = HOLOGRAM_NEON_COLOR
        if args.color:
            try:
                parsed = [float(x) for x in args.color.split(",")]
                if len(parsed) == 3:
                    color = parsed
                else:
                    print(f"Warning: Expected 3 RGB values, got {len(parsed)}; using default {color}")
            except ValueError:
                print(f"Warning: Invalid color format, using default {color}")
        styled = apply_hologram(mesh, color)

    elif args.filter == "ff7":
        target = int(args.param) if args.param else FF7_TARGET_TRIANGLES
        styled = apply_ff7(
            mesh,
            target_triangles=target,
            color_levels=FF7_COLOR_LEVELS,
            force_close=args.close,
        )

    elif args.filter == "material":
        styled = apply_material(
            mesh,
            texture_path=args.texture,
            shininess=args.shininess,
            scale=args.scale,
            preserve_geometry=args.preserve_geometry,
        )

    # Note: Pedestal attachment logic completely removed from stylize.py
    # to prevent chained filters from permanently destroying it. 
    # visualization handles it cleanly!

    # ── Save output ───────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(styled, dict) and 'body' in styled:
        # Hologram: save both body and edges
        body_path  = output_path.parent / f"{output_path.stem}_body.ply"
        edges_path = output_path.parent / f"{output_path.stem}_edges.ply"
        o3d.io.write_triangle_mesh(str(body_path),  styled['body'])
        o3d.io.write_line_set(str(edges_path),      styled['edges'])
        print(f"✓ Hologram body saved to:  {body_path}")
        print(f"✓ Hologram edges saved to: {edges_path}")
    else:
        if args.filter in {"voxel", "soft_voxel"} and len(styled.triangles) > VOXEL_MAX_TRIANGLES:
            print(
                f"Error: mesh too dense to write safely ({len(styled.triangles):,} triangles). "
                "Increase voxel size (recommended >= 0.003)."
            )
            sys.exit(1)
        o3d.io.write_triangle_mesh(str(output_path), styled)
        print(f"✓ Styled mesh saved to: {output_path}")

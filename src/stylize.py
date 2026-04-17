"""
stylize.py — 3D Structural and Visual Filters

Filters:
  - low_poly:    Simplify mesh using voxel clustering
  - voxel:       Convert mesh to a voxel grid (Minecraft style)
  - soft_voxel:  Downsample to a voxel grid with spheres (smooth voxel look)
  - hologram:    Extract edges and apply neon wireframe aesthetic
  - ff7:         PS1-era faceted look (aggressive decimation + flat shading)
  - smooth:      Taubin surface denoising (fixes over-sharpened reconstructions)
  - material:    Replace vertex colours with procedural textures
                 presets: wood | stone | marble | clay | metal
  - outline:     Cel-shade ink outline (darkens silhouette edges)

Usage:
  python stylize.py --input mesh.ply --filter low_poly       --param 1500 --output styled.ply
  python stylize.py --input mesh.ply --filter smooth         --param 15   --output smooth.ply
  python stylize.py --input mesh.ply --filter material       --param marble --output marble.ply
  python stylize.py --input mesh.ply --filter outline        --param 25   --output cel.ply
  python stylize.py --input mesh.ply --filter hologram       --color 0.035,0.714,0.902 --output holo.ply
"""

import open3d as o3d
import numpy as np
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

# SOFT-VOXEL Filter
SOFT_VOXEL_SIZE_DEFAULT   = 0.01           # Default soft voxel size
SOFT_VOXEL_SPHERE_RADIUS_MULT = 0.6       # Sphere radius = voxel_size * this

# HOLOGRAM Filter
HOLOGRAM_TARGET_TRIANGLES = 2000           # Target tris for low-poly base
HOLOGRAM_NEON_COLOR       = [0.035, 0.714, 0.902]  # #09b6e6 bright cyan
HOLOGRAM_BODY_COLOR       = [0.259, 0.510, 0.961]  # #4287f5 blue

# FF7 Filter
FF7_TARGET_TRIANGLES      = 800            # Very low poly (PS1 character range)
FF7_COLOR_LEVELS          = 16             # Colour quantization steps per channel
                                           # (PS1 = 32 levels / 5-bit, 16 = more stylised)

# SMOOTH Filter
SMOOTH_ITERATIONS         = 10             # Taubin smoothing iterations (range 5–50)

# MATERIAL Filter
MATERIAL_DEFAULT_PRESET   = "wood"         # wood | stone | marble | clay | metal

# OUTLINE Filter
OUTLINE_SOFTNESS_DEG      = 25.0           # Width of the dark silhouette band (degrees)


# ═══════════════════════════ NOISE HELPERS ════════════════════════════════════

def _value_noise_3d(pts: np.ndarray, scale: float = 1.0, seed: int = 0) -> np.ndarray:
    """
    Fast deterministic trilinear value noise for an (N, 3) array of 3-D points.
    Returns values in [0, 1].  Pure NumPy — no extra dependencies.
    """
    rng   = np.random.default_rng(seed)
    GRID  = 64
    table = rng.random((GRID, GRID, GRID))

    p  = pts * scale
    i0 = np.floor(p[:, 0]).astype(int) % GRID
    j0 = np.floor(p[:, 1]).astype(int) % GRID
    k0 = np.floor(p[:, 2]).astype(int) % GRID
    i1 = (i0 + 1) % GRID
    j1 = (j0 + 1) % GRID
    k1 = (k0 + 1) % GRID

    # Fractional part + smooth-step
    fx = p[:, 0] - np.floor(p[:, 0]);  ux = fx * fx * (3 - 2 * fx)
    fy = p[:, 1] - np.floor(p[:, 1]);  uy = fy * fy * (3 - 2 * fy)
    fz = p[:, 2] - np.floor(p[:, 2]);  uz = fz * fz * (3 - 2 * fz)

    # Trilinear interpolation
    return (table[i0, j0, k0] * (1-ux)*(1-uy)*(1-uz) +
            table[i1, j0, k0] *    ux *(1-uy)*(1-uz) +
            table[i0, j1, k0] * (1-ux)*   uy *(1-uz) +
            table[i1, j1, k0] *    ux *   uy *(1-uz) +
            table[i0, j0, k1] * (1-ux)*(1-uy)*   uz  +
            table[i1, j0, k1] *    ux *(1-uy)*   uz  +
            table[i0, j1, k1] * (1-ux)*   uy *   uz  +
            table[i1, j1, k1] *    ux *   uy *   uz)


def _fbm(pts: np.ndarray, octaves: int = 4, scale: float = 1.0,
         lacunarity: float = 2.0, gain: float = 0.5, seed: int = 0) -> np.ndarray:
    """
    Fractal Brownian Motion: sum of value-noise octaves at increasing frequencies.
    Returns values in [0, 1].
    """
    total, amplitude, frequency, norm = np.zeros(len(pts)), 1.0, scale, 0.0
    for i in range(octaves):
        total     += amplitude * _value_noise_3d(pts, scale=frequency, seed=seed + i)
        norm      += amplitude
        amplitude *= gain
        frequency *= lacunarity
    return total / norm


# ═══════════════════════════ FILTERS ═════════════════════════════════════════

def apply_low_poly(mesh: o3d.geometry.TriangleMesh, target_triangles: int = LOW_POLY_TARGET_TRIANGLES) -> o3d.geometry.TriangleMesh:
    """
    Simplify mesh to uniform low-poly style.

    Steps:
      1. Clean degenerate/duplicate geometry
      2. Taubin smoothing (volume-preserving noise removal)
      3. Midpoint subdivision (uniform triangles)
      4. Voxel clustering decimation
      5. Final cleanup
    """
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

    # Scale-aware voxel clustering
    bbox = mesh.get_axis_aligned_bounding_box()
    size = np.linalg.norm(bbox.get_extent())
    voxel_size = size / LOW_POLY_VOXEL_DIVISOR

    print(f"  → Voxel clustering (size: {voxel_size:.4f})...")
    mesh_simplified = mesh.simplify_vertex_clustering(
        voxel_size=voxel_size,
        contraction=o3d.geometry.SimplificationContraction.Average
    )

    # Final cleanup
    mesh_simplified.remove_degenerate_triangles()
    mesh_simplified.remove_unreferenced_vertices()
    mesh_simplified.compute_triangle_normals()

    print(f"  ✓ Result: {len(mesh_simplified.vertices):,} verts, {len(mesh_simplified.triangles):,} tris")
    return mesh_simplified


def apply_voxel(mesh: o3d.geometry.TriangleMesh, voxel_size: float = VOXEL_SIZE_DEFAULT) -> o3d.geometry.TriangleMesh:
    """
    Convert mesh into Minecraft-style voxel blocks.

    Process:
      1. Sample mesh surface points uniformly
      2. Create voxel grid from point cloud
      3. Reconstruct as individual cubes with original colors
    """
    print(f"Applying voxel filter (size: {voxel_size})...")

    # Sample mesh surface
    num_samples = max(int(len(mesh.vertices) * VOXEL_SAMPLE_DENSITY), 200000)
    print(f"  → Sampling {num_samples:,} points...")
    pcd = mesh.sample_points_uniformly(number_of_points=num_samples)

    # Create voxel grid
    print(f"  → Creating voxel grid...")
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    voxels = voxel_grid.get_voxels()
    print(f"  → Reconstructing {len(voxels)} voxel cubes...")

    # Reconstruct mesh from voxels
    cubes = []
    for v in voxels:
        center = voxel_grid.get_voxel_center_coordinate(v.grid_index)
        cube = o3d.geometry.TriangleMesh.create_box(
            width=voxel_size, height=voxel_size, depth=voxel_size
        )
        cube.translate(center - np.array([voxel_size/2, voxel_size/2, voxel_size/2]))
        cube.paint_uniform_color(v.color)
        cubes.append(cube)

    if not cubes:
        print("  ⚠ Warning: No voxels generated, returning original mesh")
        return mesh

    voxel_mesh = cubes[0]
    for cube in cubes[1:]:
        voxel_mesh += cube

    voxel_mesh.compute_vertex_normals()
    print(f"  ✓ Result: {len(voxel_mesh.vertices):,} verts, {len(voxel_mesh.triangles):,} tris")
    return voxel_mesh


def apply_soft_voxel(mesh: o3d.geometry.TriangleMesh, voxel_size: float = SOFT_VOXEL_SIZE_DEFAULT) -> o3d.geometry.TriangleMesh:
    """
    Create smooth voxel effect using overlapping spheres.

    Process:
      1. Sample mesh to get colors
      2. Voxel downsample the point cloud
      3. Place small spheres at each voxel center
    """
    print(f"Applying soft-voxel filter (size: {voxel_size})...")

    # Sample and downsample
    pcd = mesh.sample_points_uniformly(number_of_points=len(mesh.vertices) * 2)
    print(f"  → Downsampling to voxel grid...")
    pcd_down = pcd.voxel_down_sample(voxel_size=voxel_size)

    points = np.asarray(pcd_down.points)
    colors = np.asarray(pcd_down.colors)
    sphere_radius = voxel_size * SOFT_VOXEL_SPHERE_RADIUS_MULT

    print(f"  → Creating {len(points)} sphere nodes...")
    final_mesh = o3d.geometry.TriangleMesh()

    for i, (pt, col) in enumerate(zip(points, colors)):
        node = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius, resolution=5)
        node.translate(pt)
        node.paint_uniform_color(col)
        final_mesh += node

    final_mesh.compute_vertex_normals()
    print(f"  ✓ Result: {len(final_mesh.vertices):,} verts, {len(final_mesh.triangles):,} tris")
    return final_mesh


def apply_hologram(
    mesh: o3d.geometry.TriangleMesh,
    neon_color: list = HOLOGRAM_NEON_COLOR,
    simplify_first: bool = True
) -> dict:
    """
    Create hologram effect: low-poly mesh + neon wireframe edges.

    Returns dict with 'body' (mesh) and 'edges' (LineSet) for composite rendering.
    """
    print("Applying hologram effect...")

    # Simplify to low-poly
    if simplify_first:
        print("  → Simplifying mesh to low-poly...")
        mesh_base = apply_low_poly(mesh, target_triangles=HOLOGRAM_TARGET_TRIANGLES)
    else:
        mesh_base = mesh

    # Create body
    print("  → Creating transparent blue body...")
    mesh_body = o3d.geometry.TriangleMesh(mesh_base)
    mesh_body.vertex_colors = o3d.utility.Vector3dVector()
    mesh_body.paint_uniform_color(HOLOGRAM_BODY_COLOR)
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
    edges.paint_uniform_color(neon_color)

    print(f"  ✓ Hologram: {len(mesh_body.vertices):,} verts, {len(all_edges)} edges")

    return {
        'body': mesh_body,
        'edges': edges,
        'neon_color': neon_color
    }


def apply_ff7(
    mesh: o3d.geometry.TriangleMesh,
    target_triangles: int = FF7_TARGET_TRIANGLES,
    color_levels: int = FF7_COLOR_LEVELS,
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
        # Assign the same flat colour to all 3 vertices of every face
        new_colors  = np.repeat(avg_colors, 3, axis=0)  # (n_tris×3, 3)
    else:
        new_colors = None

    # ── Assemble result ──────────────────────────────────────────────────
    flat_mesh = o3d.geometry.TriangleMesh()
    flat_mesh.vertices  = o3d.utility.Vector3dVector(new_verts)
    flat_mesh.triangles = o3d.utility.Vector3iVector(new_tris)
    if new_colors is not None:
        flat_mesh.vertex_colors = o3d.utility.Vector3dVector(new_colors)
    # Use triangle normals (flat), not smooth vertex normals
    flat_mesh.compute_triangle_normals()

    print(f"  ✓ Result: {len(flat_mesh.vertices):,} verts, "
          f"{len(flat_mesh.triangles):,} tris  (FF7 style)")
    return flat_mesh


def apply_smooth(
    mesh: o3d.geometry.TriangleMesh,
    iterations: int = SMOOTH_ITERATIONS,
) -> o3d.geometry.TriangleMesh:
    """
    Denoise / polish the mesh surface using Taubin smoothing.

    Taubin smoothing alternates a positive Laplacian pass (λ) and a negative
    pass (μ) so the two steps roughly cancel volume shrinkage — unlike plain
    Laplacian which collapses the mesh over many iterations.

    This is ideal for fixing over-sharpened, speckled surfaces that come out
    of Screened Poisson reconstruction.

    Args:
      iterations: number of Taubin iterations (default 10, range 5–50).
                  More iterations → smoother surface but less fine detail.
    """
    print(f"Applying smooth filter ({iterations} Taubin iterations) …")

    # Lightweight pre-clean
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_unreferenced_vertices()

    print("  → Running Taubin smoothing …")
    mesh = mesh.filter_smooth_taubin(number_of_iterations=iterations)

    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()

    print(f"  ✓ Result: {len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris  (smoothed)")
    return mesh


def apply_material(
    mesh: o3d.geometry.TriangleMesh,
    preset: str = MATERIAL_DEFAULT_PRESET,
) -> o3d.geometry.TriangleMesh:
    """
    Replace vertex colours with a procedural material texture driven by 3-D
    vertex positions.  No UV unwrapping needed — entirely geometry-based.

    The existing scan colours are replaced (this is a stylization filter).
    For best results, run `smooth` first to remove reconstruction noise.

    Presets
    -------
    wood   — concentric cylindrical rings + fractal grain noise
    stone  — cool-grey layered value noise (multi-scale rock face)
    marble — sinusoidal colour veins driven by turbulence noise
    clay   — warm terracotta with height-based ambient-occlusion darkening
    metal  — anisotropic Blinn-Phong shading using surface normals (steel)

    Args:
      preset: one of 'wood', 'stone', 'marble', 'clay', 'metal'
    """
    VALID = {"wood", "stone", "marble", "clay", "metal"}
    if preset not in VALID:
        raise ValueError(f"Unknown material preset '{preset}'. Choose from: {VALID}")

    print(f"Applying material filter (preset: {preset}) …")

    pts = np.asarray(mesh.vertices)

    # Normalize coordinates to [0, 1]³ for scale-independent noise
    lo  = pts.min(axis=0)
    hi  = pts.max(axis=0)
    rng_size = hi - lo
    rng_size[rng_size < 1e-8] = 1.0          # avoid divide-by-zero on flat dims
    pts_n = (pts - lo) / rng_size            # (N, 3) in [0, 1]³

    if preset == "wood":
        # ── Wood ──────────────────────────────────────────────────────
        # Concentric cylindrical rings around the Y axis, perturbed by
        # fractal grain noise so the rings look organic.
        cx, cz  = 0.5, 0.5
        r       = np.sqrt((pts_n[:, 0] - cx)**2 + (pts_n[:, 2] - cz)**2)
        grain   = _fbm(pts_n, octaves=5, scale=6.0, seed=42)
        rings   = np.sin(r * 22.0 * np.pi + grain * 3.5) * 0.5 + 0.5  # [0, 1]

        dark_wood  = np.array([0.33, 0.16, 0.04])
        light_wood = np.array([0.76, 0.52, 0.22])
        colors = dark_wood + rings[:, None] * (light_wood - dark_wood)

    elif preset == "stone":
        # ── Stone ─────────────────────────────────────────────────────
        # Cool grey multi-scale value noise — looks like a weathered rock
        # face or natural stone surface.
        n      = _fbm(pts_n, octaves=6, scale=5.0, gain=0.55, seed=7)
        dark   = np.array([0.32, 0.31, 0.33])
        light  = np.array([0.76, 0.75, 0.72])
        colors = dark + n[:, None] * (light - dark)

    elif preset == "marble":
        # ── Marble ────────────────────────────────────────────────────
        # Sinusoidal veins in one axis perturbed by turbulence noise —
        # the classic procedural marble pattern.
        turb   = _fbm(pts_n, octaves=6, scale=4.5, gain=0.6, seed=13)
        veins  = np.sin(pts_n[:, 0] * 9.0 + turb * 7.0) * 0.5 + 0.5  # [0, 1]

        white  = np.array([0.95, 0.93, 0.91])
        vein_c = np.array([0.22, 0.20, 0.22])
        colors = white + veins[:, None] * (vein_c - white)

    elif preset == "clay":
        # ── Clay ──────────────────────────────────────────────────────
        # Warm terracotta tint with height-based pseudo-AO: vertices
        # lower on the Y axis are slightly darker (like dried, unglazed clay).
        height = pts_n[:, 1]                                         # Y = up
        noise  = _fbm(pts_n, octaves=3, scale=4.0, seed=99) * 0.12
        t      = np.clip(height + noise, 0.0, 1.0)

        shadow = np.array([0.50, 0.24, 0.12])
        light_c= np.array([0.82, 0.46, 0.28])
        colors = shadow + t[:, None] * (light_c - shadow)

    elif preset == "metal":
        # ── Metal ─────────────────────────────────────────────────────
        # Anisotropic Blinn-Phong shading using the mesh's vertex normals.
        # Gives the appearance of brushed steel under a fixed studio light.
        mesh.compute_vertex_normals()
        normals = np.asarray(mesh.vertex_normals)

        light_dir = np.array([0.6, 0.8, 0.3]);  light_dir /= np.linalg.norm(light_dir)
        view_dir  = np.array([0.0, 0.0, 1.0])
        half      = light_dir + view_dir;        half /= np.linalg.norm(half)

        diffuse   = np.clip(normals @ light_dir, 0.0, 1.0)           # (N,)
        specular  = np.clip(normals @ half,       0.0, 1.0) ** 48    # (N,)
        # Add a secondary fill light from the opposite side for depth
        fill_dir  = np.array([-0.4, 0.5, -0.6]); fill_dir /= np.linalg.norm(fill_dir)
        fill      = np.clip(normals @ fill_dir, 0.0, 1.0) * 0.25

        base_col  = np.array([0.52, 0.55, 0.60])
        hi_col    = np.array([0.96, 0.97, 1.00])
        colors = (base_col * (diffuse[:, None] * 0.65 + fill[:, None])
                  + hi_col * specular[:, None] * 0.85)
        colors = np.clip(colors, 0.0, 1.0)

    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    mesh.compute_vertex_normals()

    print(f"  ✓ Result: {len(mesh.vertices):,} verts — '{preset}' material applied")
    return mesh


def apply_outline(
    mesh: o3d.geometry.TriangleMesh,
    softness_deg: float = OUTLINE_SOFTNESS_DEG,
) -> o3d.geometry.TriangleMesh:
    """
    Cel-shading ink outline: darken vertices whose normals are nearly
    perpendicular to the camera (i.e. silhouette edges).

    The technique is purely colour-based — no extra geometry is added.
    Works best on a mesh that already has vertex colours (scan colours or
    a material preset applied first).

    Args:
      softness_deg: angular width of the dark silhouette band in degrees
                    (default 25, range 10–45).  Larger → wider dark band.
    """
    print(f"Applying outline filter (softness: {softness_deg}°) …")

    # Ensure normals are up to date
    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)            # (N, 3)

    # Existing colours — fall back to white if mesh has none
    if len(mesh.vertex_colors) > 0:
        colors = np.asarray(mesh.vertex_colors).copy()
    else:
        colors = np.ones((len(normals), 3))

    # View direction: camera looks along -Z in Open3D's default coordinate
    # frame, so we test against +Z as the "facing" direction.
    view = np.array([0.0, 0.0, 1.0])

    # |normal · view| = cos(angle from silhouette plane)
    # = 0 at silhouette edge, = 1 at face-on centre
    cos_a = np.abs(normals @ view)                       # (N,) in [0, 1]

    # Map to a smooth dark-band factor:
    # threshold = sin(softness_deg) ≈ cos(90° - softness_deg)
    threshold = np.sin(np.radians(softness_deg))
    factor    = np.clip(cos_a / max(threshold, 1e-6), 0.0, 1.0)  # 0 = dark, 1 = lit

    # Smooth-step for a softer transition
    factor = factor * factor * (3.0 - 2.0 * factor)

    # Apply: silhouette vertices go black, interior vertices stay at original colour
    darkened = colors * factor[:, None]

    mesh.vertex_colors = o3d.utility.Vector3dVector(darkened)
    mesh.compute_vertex_normals()

    outline_pct = (factor < 0.5).mean() * 100
    print(f"  ✓ Outline applied — {outline_pct:.1f}% of vertices darkened as silhouette")
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
  smooth     --param <int>    Taubin iterations (5–50)       (default 10)
  material   --param <str>    wood|stone|marble|clay|metal   (default wood)
  outline    --param <float>  silhouette band degrees (10–45)(default 25)
  hologram   --color R,G,B   neon edge colour override
""")
    parser.add_argument("--input",    type=str, required=True,
                        help="Input mesh file (.ply/.obj)")
    parser.add_argument("--filter",   type=str, default="low_poly",
                        choices=["low_poly", "voxel", "soft_voxel", "hologram",
                                 "ff7", "smooth", "material", "outline"],
                        help="Filter to apply (see epilog for --param meaning)")
    parser.add_argument("--output",  type=str, required=True,
                        help="Output path for styled mesh(es)")
    parser.add_argument("--param",   type=str, default=None,
                        help="Filter parameter — see epilog for each filter's meaning")
    parser.add_argument("--color",   type=str, default=None,
                        help="Hologram edge colour as CSV: R,G,B (e.g. 0.035,0.714,0.902)")
    parser.add_argument("--pedestal", type=str, default=None,
                        help="Path to pedestal .ply to re-attach after styling (stays unstyled)")

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
        styled = apply_low_poly(mesh, target)

    elif args.filter == "voxel":
        vsize  = float(args.param) if args.param else VOXEL_SIZE_DEFAULT
        styled = apply_voxel(mesh, vsize)

    elif args.filter == "soft_voxel":
        vsize  = float(args.param) if args.param else SOFT_VOXEL_SIZE_DEFAULT
        styled = apply_soft_voxel(mesh, vsize)

    elif args.filter == "hologram":
        color = HOLOGRAM_NEON_COLOR
        if args.color:
            try:
                color = [float(x) for x in args.color.split(",")]
            except ValueError:
                print(f"Warning: Invalid color format, using default {color}")
        styled = apply_hologram(mesh, color)

    elif args.filter == "ff7":
        target = int(args.param) if args.param else FF7_TARGET_TRIANGLES
        styled = apply_ff7(mesh, target_triangles=target, color_levels=FF7_COLOR_LEVELS)

    elif args.filter == "smooth":
        iters  = int(args.param) if args.param else SMOOTH_ITERATIONS
        styled = apply_smooth(mesh, iterations=iters)

    elif args.filter == "material":
        preset = args.param if args.param else MATERIAL_DEFAULT_PRESET
        styled = apply_material(mesh, preset=preset)

    elif args.filter == "outline":
        softness = float(args.param) if args.param else OUTLINE_SOFTNESS_DEG
        styled   = apply_outline(mesh, softness_deg=softness)

    # ── Re-attach pedestal (unstyled) ────────────────────────────────────
    if args.pedestal:
        print(f"Loading pedestal from {args.pedestal} …")
        pedestal = o3d.io.read_triangle_mesh(args.pedestal)
        if pedestal.is_empty():
            print("  ⚠ Warning: pedestal file is empty, skipping")
        else:
            if isinstance(styled, dict) and 'body' in styled:
                styled['body'] = styled['body'] + pedestal
                styled['body'].compute_vertex_normals()
            else:
                styled = styled + pedestal
                styled.compute_vertex_normals()
            print("  ✓ Pedestal re-attached (unstyled)")

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
        o3d.io.write_triangle_mesh(str(output_path), styled)
        print(f"✓ Styled mesh saved to: {output_path}")

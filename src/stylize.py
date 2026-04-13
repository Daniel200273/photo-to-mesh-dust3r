"""
stylize.py — 3D Structural and Visual Filters

Filters:
  - low_poly:    Simplify mesh using isotropic remeshing
  - voxel:       Convert mesh to a voxel grid (Minecraft style)
  - soft_voxel:  Downsample to a voxel grid with spheres (smooth voxel look)
  - hologram:    Extract edges and apply neon wireframe aesthetic

Architecture:
  apply_low_poly()    — Isotropic remeshing
  apply_voxel()       — Voxel grid reconstruction with color preservation
  apply_soft_voxel()  — Voxel sampling with sphere nodes
  apply_hologram()    — Low-poly base + sharp edge extraction

Usage:
  python stylize.py --input mesh.ply --filter low_poly --output styled.ply
  python stylize.py --input mesh.ply --filter hologram --color 0.035,0.714,0.902 --output styled.ply
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

# ═══════════════════════════ CLI ═════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply stylization filters to 3D meshes")
    parser.add_argument("--input", type=str, required=True, help="Input mesh file (.ply/.obj)")
    parser.add_argument("--filter", type=str, choices=["low_poly", "voxel", "soft_voxel", "hologram"], 
                        default="low_poly", help="Filter to apply")
    parser.add_argument("--output", type=str, required=True, help="Output path for styled mesh(es)")
    parser.add_argument("--param", type=str, default=None, 
                        help="Filter parameter (tris for low_poly, size for voxel/soft_voxel)")
    parser.add_argument("--color", type=str, default=None,
                        help="Color for hologram edges as CSV: R,G,B (e.g., 0.035,0.714,0.902)")
    
    args = parser.parse_args()
    
    # Load mesh
    print(f"Loading mesh from {args.input}...")
    mesh = o3d.io.read_triangle_mesh(args.input)
    if mesh.is_empty():
        print("Error: Mesh is empty")
        exit(1)
    
    # Apply filter
    if args.filter == "low_poly":
        target = int(args.param) if args.param else LOW_POLY_TARGET_TRIANGLES
        styled = apply_low_poly(mesh, target)
        
    elif args.filter == "voxel":
        vsize = float(args.param) if args.param else VOXEL_SIZE_DEFAULT
        styled = apply_voxel(mesh, vsize)
        
    elif args.filter == "soft_voxel":
        vsize = float(args.param) if args.param else SOFT_VOXEL_SIZE_DEFAULT
        styled = apply_soft_voxel(mesh, vsize)
        
    elif args.filter == "hologram":
        color = HOLOGRAM_NEON_COLOR
        if args.color:
            try:
                color = [float(x) for x in args.color.split(",")]
            except:
                print(f"Warning: Invalid color format, using default {color}")
        styled = apply_hologram(mesh, color)
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(styled, dict) and 'body' in styled:
        # Hologram: save both body and edges
        body_path = output_path.parent / f"{output_path.stem}_body.ply"
        edges_path = output_path.parent / f"{output_path.stem}_edges.ply"
        
        o3d.io.write_triangle_mesh(str(body_path), styled['body'])
        o3d.io.write_line_set(str(edges_path), styled['edges'])
        
        print(f"✓ Hologram body saved to: {body_path}")
        print(f"✓ Hologram edges saved to: {edges_path}")
    else:
        # Standard mesh output
        o3d.io.write_triangle_mesh(str(output_path), styled)
        print(f"✓ Styled mesh saved to: {output_path}")

"""
visualize_mesh.py — View triangle meshes (.ply / .obj) in an interactive 3D window.

Usage:
  python visualize_mesh.py                                  # opens final_mesh.ply
  python visualize_mesh.py --path data/processed_data/mesh/step4_mesh_grey.ply
"""

import argparse
import open3d as o3d
from pathlib import Path


def visualize_mesh(mesh_path: Path):
    """Load and display a triangle mesh with lighting and normals."""
    print(f"--- Loading mesh from {mesh_path} ---")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))

    if mesh.is_empty():
        print("Error: Mesh is empty or invalid.")
        return

    # Compute normals so the mesh is properly lit
    mesh.compute_vertex_normals()

    has_colors = len(mesh.vertex_colors) > 0
    print(f"Mesh loaded: {len(mesh.vertices):,} vertices, "
          f"{len(mesh.triangles):,} triangles, "
          f"colors={'yes' if has_colors else 'no (grey)'}")

    print("\nControls:")
    print(" - Mouse Left : Rotate")
    print(" - Mouse Right: Pan")
    print(" - Scroll     : Zoom")
    print(" - Q / Esc    : Close viewer")

    o3d.visualization.draw_geometries(
        [mesh],
        window_name="Mesh Viewer",
        width=1024,
        height=768,
        mesh_show_back_face=True,
    )


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.resolve()
    default_path = project_root / "data" / "processed_data" / "mesh" / "final_mesh.ply"

    parser = argparse.ArgumentParser(description="Visualize a 3D triangle mesh.")
    parser.add_argument("--path", type=str, default=str(default_path),
                        help="Path to the mesh file (.ply or .obj)")
    args = parser.parse_args()

    mesh_file = Path(args.path)
    if not mesh_file.exists():
        print(f"💀 Error: File not found at {mesh_file}")
        print("Run mesh_reconstruction.py first to generate the mesh.")
    else:
        visualize_mesh(mesh_file)

"""
visualize_mesh_browser.py — View triangle meshes in a web browser using Plotly.
Useful for Wayland/WSL users where Open3D's GUI might fail.
"""

import argparse
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
from pathlib import Path

def visualize_mesh_plotly(mesh_path: Path):
    print(f"--- Loading data from {mesh_path} ---")
    
    # Check extension
    is_ply = mesh_path.suffix.lower() == ".ply"
    
    # Try different loading strategies
    line_set = None
    mesh = None

    # Try BOTH independently
    temp_ls = o3d.io.read_line_set(str(mesh_path))
    if temp_ls.has_lines():
        line_set = temp_ls
        print("Loaded LineSet.")

    temp_mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if temp_mesh.has_triangles():
        mesh = temp_mesh
        print("Loaded TriangleMesh.")

    # 🆕 AUTO-DETECT companion edges file for holograms
    # If this is a *_body.ply, look for *_edges.ply
    if mesh_path.stem.endswith("_body"):
        companion_edges = mesh_path.parent / f"{mesh_path.stem.replace('_body', '')}_edges{mesh_path.suffix}"
        if companion_edges.exists():
            print(f"Found companion edges file: {companion_edges.name}")
            temp_edges = o3d.io.read_line_set(str(companion_edges))
            if temp_edges.has_lines():
                line_set = temp_edges
                print(f"  Loaded {len(line_set.lines)} hologram edges")

    fig = go.Figure()

    # Check if this is a hologram (_body file)
    is_hologram = mesh_path.stem.endswith("_body")

    # --- Render mesh (base body) ---
    if mesh and mesh.has_triangles():
        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)

        print(f"Mesh: {len(verts)} verts, {len(faces)} tris")

        # For holograms: medium opacity (0.15), otherwise higher opacity
        opacity = 0.15 if is_hologram else 1

        # For holograms: use uniform blue color, otherwise use vertex colors
        if is_hologram:
            # Force uniform blue for hologram (#4287f5)
            fig.add_trace(go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color='rgb(66,135,245)',  # #4287f5
                opacity=opacity,
                name='Mesh'
            ))
        else:
            # Use actual vertex colors from mesh
            if mesh.has_vertex_colors():
                colors = np.asarray(mesh.vertex_colors)
                vertex_colors = ['rgb({},{},{})'.format(int(c[0]*255), int(c[1]*255), int(c[2]*255)) for c in colors]
                fig.add_trace(go.Mesh3d(
                    x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                    vertexcolor=vertex_colors,
                    opacity=opacity,
                    name='Mesh'
                ))
            else:
                # Default grey if no colors
                fig.add_trace(go.Mesh3d(
                    x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                    color='rgb(200,200,200)',
                    opacity=opacity,
                    name='Mesh'
                ))

    # --- Render edges (glow lines) ---
    if line_set and line_set.has_lines():
        pts = np.asarray(line_set.points)
        lines = np.asarray(line_set.lines)

        x, y, z = [], [], []
        for line in lines:
            x.extend([pts[line[0]][0], pts[line[1]][0], None])
            y.extend([pts[line[0]][1], pts[line[1]][1], None])
            z.extend([pts[line[0]][2], pts[line[1]][2], None])

        line_color = 'rgb(0,255,120)'

        if line_set.has_colors():
            c = np.asarray(line_set.colors)[0]
            line_color = f'rgb({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)})'

        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(color=line_color, width=4),
            name='Hologram Edges'
        ))

    # Save to HTML
    html_path = mesh_path.with_suffix(".html")
    # If path already has .ply, it becomes .html. 
    # To avoid overwriting raw reconstruction, we can prepend a prefix
    html_output = mesh_path.parent / f"reconstruction_{mesh_path.stem}_viewer.html"
    
    fig.write_html(str(html_output))
    print(f"\n✓ Interactive mesh viewer saved to: {html_output}")
    print("Open this file in your browser to view the 3D model.")

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.resolve()
    default_path = project_root / "data" / "processed_data" / "mesh" / "final_mesh.ply"

    parser = argparse.ArgumentParser(description="Visualize a 3D mesh in the browser.")
    parser.add_argument("--path", type=str, default=str(default_path), help="Path to the mesh file (.ply/.obj)")
    parser.add_argument("--hologram", action="store_true", help="Enable hologram effect (transparent body + edges)")
    args = parser.parse_args()

    mesh_file = Path(args.path)
    if not mesh_file.exists():
        print(f"💀 Error: File not found at {mesh_file}")
    else:
        # Auto-detect hologram if file ends with _body
        is_hologram = args.hologram or mesh_file.stem.endswith("_body")
        # For now, always render holograms with the special mode
        visualize_mesh_plotly(mesh_file)

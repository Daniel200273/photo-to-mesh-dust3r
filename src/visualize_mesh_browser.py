"""
visualize_mesh_browser.py — View triangle meshes in a web browser using Plotly.
This browser viewer mirrors the local Open3D viewer behavior as closely as possible.
"""

import argparse
import webbrowser
from pathlib import Path

import numpy as np
import open3d as o3d
import plotly.graph_objects as go


def _decouple_triangles(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Convert to triangle soup so sharp edges keep independent normals."""
    triangles = np.asarray(mesh.triangles)
    vertices = np.asarray(mesh.vertices)

    soup_vertices = vertices[triangles].reshape(-1, 3)
    soup_triangles = np.arange(len(soup_vertices), dtype=np.int32).reshape(-1, 3)

    soup = o3d.geometry.TriangleMesh()
    soup.vertices = o3d.utility.Vector3dVector(soup_vertices)
    soup.triangles = o3d.utility.Vector3iVector(soup_triangles)

    if len(mesh.vertex_colors) > 0:
        colors = np.asarray(mesh.vertex_colors)
        soup.vertex_colors = o3d.utility.Vector3dVector(colors[triangles].reshape(-1, 3))

    return soup


def _sanitize_for_render(mesh: o3d.geometry.TriangleMesh) -> None:
    """Remove obvious geometry hazards that can cause unstable shading."""
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()


def _estimate_thin_axis(vertices: np.ndarray) -> np.ndarray:
    """Estimate a mesh normal from the thinnest principal direction."""
    if len(vertices) == 0:
        return np.array([0.0, 1.0, 0.0])

    centered = vertices - vertices.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[-1]
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.array([0.0, 1.0, 0.0])
    return axis / norm


def _align_pedestal_to_object(
    object_mesh: o3d.geometry.TriangleMesh,
    pedestal_mesh: o3d.geometry.TriangleMesh,
    embed: float,
    separation: float,
) -> np.ndarray:
    """Align pedestal top to object bottom along support normal."""
    ped_vertices = np.asarray(pedestal_mesh.vertices)
    obj_vertices = np.asarray(object_mesh.vertices)

    if len(ped_vertices) == 0 or len(obj_vertices) == 0:
        return np.array([0.0, 1.0, 0.0])

    normal = _estimate_thin_axis(ped_vertices)
    ped_center = ped_vertices.mean(axis=0)
    obj_center = obj_vertices.mean(axis=0)

    if np.dot(obj_center - ped_center, normal) < 0:
        normal = -normal

    obj_bottom = float((obj_vertices @ normal).min())
    ped_top = float((ped_vertices @ normal).max())
    desired_top = obj_bottom + max(embed, 0.0) - max(separation, 0.0)

    delta = desired_top - ped_top
    pedestal_mesh.translate((normal * delta).tolist())
    return normal


def _apply_global_view_transform(
    mesh: o3d.geometry.TriangleMesh,
    pedestal: o3d.geometry.TriangleMesh | None,
    edges: o3d.geometry.LineSet | None,
    target_diag: float,
) -> None:
    """Match local viewer orientation and scale for stable framing."""
    R = o3d.geometry.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    bbox_diag = float(np.linalg.norm(mesh.get_axis_aligned_bounding_box().get_extent()))

    scale_factor = 1.0
    if bbox_diag > 0 and bbox_diag < target_diag:
        scale_factor = target_diag / bbox_diag

    mesh.rotate(R, center=(0, 0, 0))
    if scale_factor != 1.0:
        mesh.scale(scale_factor, center=(0, 0, 0))

    if pedestal is not None:
        pedestal.rotate(R, center=(0, 0, 0))
        if scale_factor != 1.0:
            pedestal.scale(scale_factor, center=(0, 0, 0))

    if edges is not None and len(edges.points) > 0:
        pts = np.asarray(edges.points)
        pts = pts @ R.T
        if scale_factor != 1.0:
            pts = pts * scale_factor
        edges.points = o3d.utility.Vector3dVector(pts)


def _to_css_rgb(color01: np.ndarray | list[float] | tuple[float, float, float]) -> str:
    c = np.clip(np.asarray(color01, dtype=float).reshape(-1)[:3], 0.0, 1.0)
    return f"rgb({int(c[0] * 255)}, {int(c[1] * 255)}, {int(c[2] * 255)})"


def visualize_mesh_plotly(
    mesh_path: Path,
    hologram: bool = False,
    pedestal_path: str = None,
    bg_color: str = "studio_cool",
    unlit: bool = False,
    **kwargs,
):
    print(f"--- Loading mesh from {mesh_path} ---")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))

    if mesh.is_empty() or not mesh.has_triangles():
        print("Error: Mesh is empty or invalid.")
        return

    stabilize = kwargs.get("stabilize", True)
    if stabilize:
        _sanitize_for_render(mesh)
    else:
        mesh.compute_vertex_normals()

    is_hologram = hologram or mesh_path.stem.endswith("_body")
    edges = None
    pedestal = None

    if is_hologram:
        companion_edges = mesh_path.parent / f"{mesh_path.stem.replace('_body', '')}_edges{mesh_path.suffix}"
        if companion_edges.exists():
            print(f"Loading companion edges: {companion_edges.name}")
            temp_edges = o3d.io.read_line_set(str(companion_edges))
            if temp_edges.has_lines():
                edges = temp_edges
                print(f"Loaded {len(edges.lines)} hologram edges")

    if pedestal_path and Path(pedestal_path).exists():
        print(f"Loading unstyled pedestal: {Path(pedestal_path).name}")
        pedestal = o3d.io.read_triangle_mesh(pedestal_path)
        if pedestal.is_empty() or not pedestal.has_triangles():
            pedestal = None
        else:
            pedestal = _decouple_triangles(pedestal)

            pedestal_offset = float(kwargs.get("pedestal_offset", 0.0))
            pedestal_embed = float(kwargs.get("pedestal_embed", -1.0))
            if pedestal_embed < 0:
                obj_diag = float(np.linalg.norm(mesh.get_axis_aligned_bounding_box().get_extent()))
                pedestal_embed = max(obj_diag * 0.01, 0.001)

            _align_pedestal_to_object(
                object_mesh=mesh,
                pedestal_mesh=pedestal,
                embed=pedestal_embed,
                separation=pedestal_offset,
            )

            if stabilize:
                _sanitize_for_render(pedestal)
            else:
                pedestal.compute_vertex_normals()

            if unlit:
                normals = np.asarray(pedestal.vertex_normals)
                light_dir = np.array([0.5, 0.8, -0.4])
                light_dir /= np.linalg.norm(light_dir)
                diffuse = np.clip(normals @ light_dir, 0.0, 1.0)
                ped_color = np.array([0.75, 0.75, 0.75])
                colors = ped_color * (diffuse[:, None] * 0.6 + 0.4)
                pedestal.vertex_colors = o3d.utility.Vector3dVector(colors)

    _apply_global_view_transform(
        mesh=mesh,
        pedestal=pedestal,
        edges=edges,
        target_diag=float(kwargs.get("target_diag", 0.3)),
    )

    bg_map = {
        "studio_cool": (0.20, 0.23, 0.27, 1.0),
        "neutral_slate": (0.24, 0.25, 0.27, 1.0),
        "warm_stone": (0.27, 0.26, 0.23, 1.0),
        "soft_fog": (0.32, 0.34, 0.36, 1.0),
        "blue_mist": (0.16, 0.20, 0.24, 1.0),
    }
    bg_color_rgba = bg_map.get(bg_color.lower(), bg_map["studio_cool"])
    bg_css = _to_css_rgb(bg_color_rgba[:3])

    fig = go.Figure()

    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    print(
        f"Mesh loaded: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles, "
        f"colors={'yes' if mesh.has_vertex_colors() else 'no (grey)'}"
    )

    if unlit:
        body_lighting = dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0)
    else:
        shininess = kwargs.get("shininess")
        specular = 0.12 if shininess is None else float(np.clip(float(shininess) / 100.0, 0.02, 0.9))
        body_lighting = dict(ambient=0.45, diffuse=0.65, specular=specular, roughness=0.95, fresnel=0.03)

    body_opacity = 0.92 if is_hologram else 1.0

    if mesh.has_vertex_colors() and len(mesh.vertex_colors) == len(mesh.vertices):
        colors = np.asarray(mesh.vertex_colors)
        vertex_colors = [_to_css_rgb(c) for c in colors]
        fig.add_trace(
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                vertexcolor=vertex_colors,
                opacity=body_opacity,
                lighting=body_lighting,
                flatshading=False,
                name="Object",
            )
        )
    else:
        fallback_color = [0.259, 0.510, 0.961] if is_hologram else [0.78, 0.78, 0.78]
        fig.add_trace(
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color=_to_css_rgb(fallback_color),
                opacity=body_opacity,
                lighting=body_lighting,
                flatshading=False,
                name="Object",
            )
        )

    if pedestal is not None and pedestal.has_triangles():
        p_verts = np.asarray(pedestal.vertices)
        p_faces = np.asarray(pedestal.triangles)

        pedestal_lit = bool(kwargs.get("pedestal_lit", True))
        if pedestal_lit and not unlit:
            p_lighting = dict(ambient=0.6, diffuse=0.5, specular=0.04, roughness=1.0, fresnel=0.0)
        else:
            p_lighting = dict(ambient=1.0, diffuse=0.0, specular=0.0, roughness=1.0, fresnel=0.0)

        if pedestal.has_vertex_colors() and len(pedestal.vertex_colors) == len(pedestal.vertices):
            p_colors = np.asarray(pedestal.vertex_colors)
            p_vertexcolor = [_to_css_rgb(c) for c in p_colors]
            fig.add_trace(
                go.Mesh3d(
                    x=p_verts[:, 0],
                    y=p_verts[:, 1],
                    z=p_verts[:, 2],
                    i=p_faces[:, 0],
                    j=p_faces[:, 1],
                    k=p_faces[:, 2],
                    vertexcolor=p_vertexcolor,
                    opacity=1.0,
                    lighting=p_lighting,
                    flatshading=True,
                    name="Pedestal",
                )
            )
        else:
            base = [0.40, 0.40, 0.40] if pedestal_lit and not unlit else [0.42, 0.42, 0.42]
            fig.add_trace(
                go.Mesh3d(
                    x=p_verts[:, 0],
                    y=p_verts[:, 1],
                    z=p_verts[:, 2],
                    i=p_faces[:, 0],
                    j=p_faces[:, 1],
                    k=p_faces[:, 2],
                    color=_to_css_rgb(base),
                    opacity=1.0,
                    lighting=p_lighting,
                    flatshading=True,
                    name="Pedestal",
                )
            )

    if edges is not None and edges.has_lines():
        pts = np.asarray(edges.points)
        lines = np.asarray(edges.lines)

        x = []
        y = []
        z = []
        for line in lines:
            x.extend([pts[line[0]][0], pts[line[1]][0], None])
            y.extend([pts[line[0]][1], pts[line[1]][1], None])
            z.extend([pts[line[0]][2], pts[line[1]][2], None])

        line_color = "rgb(9, 182, 230)"
        if edges.has_colors() and len(edges.colors) > 0:
            line_color = _to_css_rgb(np.asarray(edges.colors)[0])

        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines",
                line=dict(color=line_color, width=5),
                name="Hologram Edges",
            )
        )

    fig.update_layout(
        title="FewShot-NeRF Studio",
        scene=dict(
            bgcolor=bg_css,
            xaxis=dict(visible=False, showbackground=False),
            yaxis=dict(visible=False, showbackground=False),
            zaxis=dict(visible=False, showbackground=False),
            aspectmode="data",
            camera=dict(
                up=dict(x=0.0, y=1.0, z=0.0),
                eye=dict(x=1.55, y=1.15, z=1.55),
            ),
        ),
        paper_bgcolor=bg_css,
        margin=dict(l=0, r=0, t=42, b=0),
        showlegend=is_hologram,
    )

    html_output = mesh_path.parent / f"reconstruction_{mesh_path.stem}_viewer.html"
    fig.write_html(str(html_output), include_plotlyjs="cdn")
    print(f"\nInteractive mesh viewer saved to: {html_output}")

    if kwargs.get("open_browser", True):
        webbrowser.open(html_output.resolve().as_uri())
        print("Opened in your default browser.")


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.resolve()
    default_path = project_root / "data" / "processed_data" / "mesh" / "final_mesh.ply"

    parser = argparse.ArgumentParser(description="Visualize a 3D triangle mesh in the browser.")
    parser.add_argument("--path", type=str, default=str(default_path), help="Path to the mesh file (.ply or .obj)")
    parser.add_argument("--hologram", action="store_true", help="Enable hologram effect (load companion edges)")
    parser.add_argument("--pedestal", type=str, default=None, help="Path to pedestal mesh to overlay")
    parser.add_argument(
        "--bg",
        type=str,
        default="studio_cool",
        choices=["studio_cool", "neutral_slate", "warm_stone", "soft_fog", "blue_mist"],
        help="Background preset for object presentation",
    )
    parser.add_argument("--unlit", action="store_true", help="Use unlit appearance (hologram/material compatibility)")
    parser.add_argument("--shininess", type=float, default=None, help="Material shininess hint")
    parser.add_argument("--target-diag", type=float, default=0.3, help="Minimum scene diagonal used for auto scaling")
    parser.add_argument("--pedestal-offset", type=float, default=0.0, help="Gap between pedestal and object along support normal")
    parser.add_argument("--pedestal-embed", type=float, default=-1.0, help="How much pedestal penetrates object base (negative = auto)")
    parser.add_argument(
        "--pedestal-lit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render pedestal with lit shading (default: lit; use --no-pedestal-lit to disable)",
    )
    parser.add_argument("--no-stabilize", action="store_true", help="Disable mesh cleanup and shading safeguards")
    parser.add_argument(
        "--open",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Open generated HTML in your default browser",
    )

    args = parser.parse_args()

    mesh_file = Path(args.path)
    if not mesh_file.exists():
        print(f"Error: File not found at {mesh_file}")
        print("Run mesh_reconstruction.py first to generate the mesh.")
    else:
        visualize_mesh_plotly(
            mesh_file,
            hologram=args.hologram,
            pedestal_path=args.pedestal,
            bg_color=args.bg,
            unlit=args.unlit,
            shininess=args.shininess,
            target_diag=args.target_diag,
            pedestal_offset=args.pedestal_offset,
            pedestal_embed=args.pedestal_embed,
            pedestal_lit=args.pedestal_lit,
            stabilize=not args.no_stabilize,
            open_browser=args.open,
        )

"""
visualize_mesh.py — View triangle meshes (.ply / .obj) in an interactive 3D window.

Usage:
  python visualize_mesh.py                                  # opens final_mesh.ply
  python visualize_mesh.py --path data/processed_data/mesh/final_mesh.ply
  python visualize_mesh.py --path data/processed_data/mesh/final_holo_body.ply --hologram
"""

import argparse
import open3d as o3d
import numpy as np
from pathlib import Path


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
    """Align pedestal top to object bottom along support normal.

    Positive embed pushes pedestal slightly into object to hide open bottoms.
    Positive separation adds a visible gap.
    """
    ped_vertices = np.asarray(pedestal_mesh.vertices)
    obj_vertices = np.asarray(object_mesh.vertices)

    if len(ped_vertices) == 0 or len(obj_vertices) == 0:
        return np.array([0.0, 1.0, 0.0])

    normal = _estimate_thin_axis(ped_vertices)
    ped_center = ped_vertices.mean(axis=0)
    obj_center = obj_vertices.mean(axis=0)

    # Ensure normal points from pedestal toward object.
    if np.dot(obj_center - ped_center, normal) < 0:
        normal = -normal

    obj_bottom = float((obj_vertices @ normal).min())
    ped_top = float((ped_vertices @ normal).max())
    desired_top = obj_bottom + max(embed, 0.0) - max(separation, 0.0)

    delta = desired_top - ped_top
    pedestal_mesh.translate((normal * delta).tolist())
    return normal


def visualize_mesh(mesh_path: Path, hologram: bool = False, pedestal_path: str = None, bg_color: str = "studio_cool", unlit: bool = False, **kwargs):
    """Load and display a triangle mesh with lighting and normals."""
    print(f"--- Loading mesh from {mesh_path} ---")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))

    if mesh.is_empty():
        print("Error: Mesh is empty or invalid.")
        return

    stabilize = kwargs.get("stabilize", True)
    if stabilize:
        _sanitize_for_render(mesh)

    geometries = [mesh]

    # For hologram: try to load companion edges file
    if hologram:
        print("Hologram mode detected")
        companion_edges = mesh_path.parent / f"{mesh_path.stem.replace('_body', '')}_edges{mesh_path.suffix}"
        if companion_edges.exists():
            print(f"Loading companion edges: {companion_edges.name}")
            edges = o3d.io.read_line_set(str(companion_edges))
            if edges.has_lines():
                geometries.append(edges)
                print(f"Loaded {len(edges.lines)} edges")

    # For pedestal: overlay the unstyled base
    if pedestal_path and Path(pedestal_path).exists():
        print(f"Loading unstyled pedestal: {Path(pedestal_path).name}")
        pedestal = o3d.io.read_triangle_mesh(pedestal_path)
        if not pedestal.is_empty():
            # Use triangle soup so top rim never gets averaged lighting artifacts.
            pedestal = _decouple_triangles(pedestal)

            # Keep physical contact with object base and lightly embed to hide bottom holes.
            pedestal_offset = float(kwargs.get("pedestal_offset", 0.0))
            pedestal_embed = float(kwargs.get("pedestal_embed", -1.0))
            if pedestal_embed < 0:
                # Auto embed based on object scale.
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
                # Bake basic diffuse lighting into pedestal so it renders beautifully without Open3D's global lights
                normals = np.asarray(pedestal.vertex_normals)
                light_dir = np.array([0.5, 0.8, -0.4])
                light_dir /= np.linalg.norm(light_dir)
                diffuse = np.clip(normals @ light_dir, 0.0, 1.0)
                ped_color = np.array([0.75, 0.75, 0.75])
                colors = ped_color * (diffuse[:, None] * 0.6 + 0.4)
                pedestal.vertex_colors = o3d.utility.Vector3dVector(colors)
                
            geometries.append(pedestal)

    # Compute normals so the mesh is properly lit
    if not stabilize:
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

    # 1. Background colors (curated for product/object presentation)
    bg_map = {
        "studio_cool": (0.20, 0.23, 0.27, 1.0),     # Best all-around object presentation
        "neutral_slate": (0.24, 0.25, 0.27, 1.0),   # Balanced neutral contrast
        "warm_stone": (0.27, 0.26, 0.23, 1.0),      # Complements cool-colored assets
        "soft_fog": (0.32, 0.34, 0.36, 1.0),        # Gentle, low-fatigue backdrop
        "blue_mist": (0.16, 0.20, 0.24, 1.0),       # Great for hologram/cyan accents
        "white": (1.0, 1.0, 1.0, 1.0),              # Clean publication/presentation background
    }
    bg_color_rgba = bg_map.get(bg_color.lower(), bg_map["studio_cool"])

    # 2. Fix Upside-Down & Near-Clip issues
    # Rotate 180 degrees around X-axis for reconstruction coordinates.
    # Scale only if the scene is very small to preserve depth precision.
    R = o3d.geometry.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    target_diag = float(kwargs.get("target_diag", 0.3))
    bbox_diag = float(np.linalg.norm(mesh.get_axis_aligned_bounding_box().get_extent()))
    scale_factor = 1.0
    if bbox_diag > 0 and bbox_diag < target_diag:
        scale_factor = target_diag / bbox_diag

    for geom in geometries:
        geom.rotate(R, center=(0, 0, 0))
        if scale_factor != 1.0:
            geom.scale(scale_factor, center=(0, 0, 0))

    # ── Modern PBR & Standard Visualization (Unified) ──
    import open3d.visualization.rendering as rendering
    
    draw_objects = []
    
    # ── 1. Global Linearization (sRGB -> Linear) ──
    # The modern Filament PBR engine computes lighting in physical Linear Space.
    # However, raw Open3D meshes and our stylized textures natively store data in sRGB Space!
    # If we pass sRGB directly, the engine double-gamma corrects the pixels, violently shifting 
    # mid-tones to white and causing a crippling "washed-out pastel" blowout bug on *all* meshes!
    # We dynamically de-gamma the arrays on the fly right before drawing!
    # Apply color linearization only to the primary object mesh.
    if len(mesh.vertex_colors) > 0:
        srgb_colors = np.asarray(mesh.vertex_colors)
        linear_colors = np.power(srgb_colors, 2.2)
        mesh.vertex_colors = o3d.utility.Vector3dVector(linear_colors)
            
    # ── 2. Base Object Material ──
    mat = rendering.MaterialRecord()
    if unlit:
        mat.shader = "defaultUnlit"
    else:
        mat.shader = "defaultLit"
        
        # Open3D's Filament engine defaults to an incredibly bright physical sun and HDR environment map 
        # tuned for architectural scenes. We aggressively throttle the PBR base color to act as a 
        # negative exposure compensation (-1.5 EV), preventing the intense lights from artificially 
        # washing out the rich linear textures!
        mat.base_color = [0.35, 0.35, 0.35, 1.0]

        # A rougher baseline reduces specular shimmer on high-frequency meshes.
        mat.base_roughness = float(kwargs.get("roughness", 0.95))
        mat.base_reflectance = 0.08
        mat.base_clearcoat = 0.0
        
        # Dial in specular highlights
        if kwargs.get("shininess") is not None:
            m_shininess = kwargs["shininess"]
            roughness = np.clip(1.0 - (m_shininess / 100.0), 0.05, 1.0)
            mat.base_roughness = roughness

    draw_objects.append({'name': 'object', 'geometry': geometries[0], 'material': mat})

    # 2. Secondary Geometries (Hologram Edges / Pedestal)
    for i, geom in enumerate(geometries[1:], 1):
        g_mat = rendering.MaterialRecord()
        geom_name = f"geometry_{i}"
        
        if isinstance(geom, o3d.geometry.LineSet):
            g_mat.shader = "unlitLine"
            g_mat.line_width = 3.0
            if len(geom.colors) > 0:
                c = np.asarray(geom.colors)[0]
                g_mat.base_color = [float(c[0]), float(c[1]), float(c[2]), 1.0]
            geom_name = "hologram_edges"
        else:
            # Pedestal: default unlit to avoid grazing-angle shadow bands.
            pedestal_lit = bool(kwargs.get("pedestal_lit", True))
            if pedestal_lit and not unlit:
                g_mat.shader = "defaultLit"
                g_mat.base_color = [0.4, 0.4, 0.4, 1.0]
                g_mat.base_roughness = 0.99
                g_mat.base_metallic = 0.0
                g_mat.base_reflectance = 0.04
            else:
                g_mat.shader = "defaultUnlit"
                g_mat.base_color = [0.42, 0.42, 0.42, 1.0]
            geom_name = "pedestal"
            
        draw_objects.append({'name': geom_name, 'geometry': geom, 'material': g_mat})

    # Render unified scene
    draw_kwargs = {
        "bg_color": bg_color_rgba,
        "title": "FewShot-NeRF Studio",
        "show_ui": False,
        "show_skybox": False,
    }

    if not unlit:
        draw_kwargs["sun_intensity"] = float(kwargs.get("sun_intensity", 20000.0))
        draw_kwargs["ibl_intensity"] = float(kwargs.get("ibl_intensity", 12000.0))

    try:
        o3d.visualization.draw(draw_objects, **draw_kwargs)
    except TypeError:
        # Compatibility path for Open3D builds without sun/IBL keyword support.
        draw_kwargs.pop("sun_intensity", None)
        draw_kwargs.pop("ibl_intensity", None)
        o3d.visualization.draw(draw_objects, **draw_kwargs)


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.resolve()
    default_path = project_root / "data" / "processed_data" / "mesh" / "final_mesh.ply"

    parser = argparse.ArgumentParser(description="Visualize a 3D triangle mesh.")
    parser.add_argument("--path", type=str, default=str(default_path),
                        help="Path to the mesh file (.ply or .obj)")
    parser.add_argument("--hologram", action="store_true", help="Enable hologram effect (load companion edges)")
    parser.add_argument("--pedestal", type=str, default=None, help="Path to pedestal mesh to overlay")
    parser.add_argument(
        "--bg",
        type=str,
        default="studio_cool",
        choices=["studio_cool", "neutral_slate", "warm_stone", "soft_fog", "blue_mist", "white"],
        help="Background preset for object presentation",
    )
    parser.add_argument("--unlit", action="store_true", help="Disable Open3D default glossy lighting (used for pre-baked materials)")
    parser.add_argument("--shininess", type=float, default=None, help="PBR shininess constraint for materials")
    parser.add_argument("--roughness", type=float, default=0.95, help="Base PBR roughness (higher = less flicker)")
    parser.add_argument("--target-diag", type=float, default=0.3, help="Minimum scene diagonal used for auto scaling")
    parser.add_argument("--pedestal-offset", type=float, default=0.0, help="Gap between pedestal and object along support normal")
    parser.add_argument("--pedestal-embed", type=float, default=-1.0, help="How much pedestal penetrates object base (negative = auto)")
    parser.add_argument(
        "--pedestal-lit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render pedestal with lit shader (default: lit; use --no-pedestal-lit to disable)",
    )
    parser.add_argument("--sun-intensity", type=float, default=20000.0, help="Directional light intensity")
    parser.add_argument("--ibl-intensity", type=float, default=12000.0, help="Image-based light intensity")
    parser.add_argument("--no-stabilize", action="store_true", help="Disable mesh cleanup and flicker safeguards")
    args = parser.parse_args()

    mesh_file = Path(args.path)
    if not mesh_file.exists():
        print(f"💀 Error: File not found at {mesh_file}")
        print("Run mesh_reconstruction.py first to generate the mesh.")
    else:
        visualize_mesh(
            mesh_file,
            hologram=args.hologram,
            pedestal_path=args.pedestal,
            bg_color=args.bg,
            unlit=args.unlit,
            shininess=args.shininess,
            roughness=args.roughness,
            target_diag=args.target_diag,
            pedestal_offset=args.pedestal_offset,
            pedestal_embed=args.pedestal_embed,
            pedestal_lit=args.pedestal_lit,
            sun_intensity=args.sun_intensity,
            ibl_intensity=args.ibl_intensity,
            stabilize=not args.no_stabilize,
        )

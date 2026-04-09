import open3d as o3d
import argparse
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

def visualize_ply(ply_path):
    print(f"--- Loading point cloud from {ply_path} ---")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    
    if pcd.is_empty():
        print("Error: Point cloud is empty or invalid.")
        return

    print("Opening 3D viewer with plotly...")
    
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    
    # Debug info
    print(f"Points shape: {points.shape}")
    print(f"Points range: X=[{points[:, 0].min():.2f}, {points[:, 0].max():.2f}], "
          f"Y=[{points[:, 1].min():.2f}, {points[:, 1].max():.2f}], "
          f"Z=[{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    print(f"Has colors: {colors is not None}")
    
    # Remove any NaN or Inf points
    valid_mask = np.isfinite(points).all(axis=1)
    points = points[valid_mask]
    if colors is not None:
        colors = colors[valid_mask]
    
    print(f"Valid points after filtering: {len(points)}")
    
    # Aggressively downsample for browser performance
    if len(points) > 100000:
        downsample_ratio = max(1, len(points) // 75000)
        points = points[::downsample_ratio]
        if colors is not None:
            colors = colors[::downsample_ratio]
        print(f"Downsampled to {len(points)} points (keeping every {downsample_ratio}th point)")
    
    # Use actual RGB colors if available, otherwise use Z-depth
    if colors is not None and colors.shape[1] == 3:
        # Colors are in [0, 1], convert to plotly format
        color_list = ['rgb({},{},{})'.format(int(r*255), int(g*255), int(b*255)) for r, g, b in colors]
    else:
        color_list = points[:, 2]  # Use Z-depth for coloring
    
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(
            size=1.5,
            color=color_list,
            colorscale='Viridis' if isinstance(color_list, np.ndarray) else None,
            opacity=0.6,
            showscale=True
        )
    )])
    
    fig.update_layout(
        title='DUSt3R Point Cloud Viewer',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.2, y=1.2, z=1.2)
            )
        ),
        width=1024,
        height=768,
        showlegend=False
    )
    
    print("Controls:")
    print(" - Click and drag: Rotate")
    print(" - Scroll: Zoom")
    print(" - Right-click and drag: Pan")
    
    # Save HTML file instead of trying to open in browser
    html_path = str(ply_path).replace(".ply", "_viewer.html")
    fig.write_html(html_path)
    print(f"\n✓ Interactive viewer saved to {html_path}")
    print(f"Open this file in your web browser to view the point cloud.")

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.resolve()
    default_ply = project_root / "data" / "processed_data" / "reconstruction_no_plane.ply"
    
    parser = argparse.ArgumentParser(description="Visualize a 3D PLY file.")
    parser.add_argument("--path", type=str, default=str(default_ply), help="Path to the .ply file")
    args = parser.parse_args()
    
    ply_file = Path(args.path)
    if not ply_file.exists():
        print(f"💀 Error: File not found at {ply_file}")
        print("Please ensure the DUSt3R pipeline finished successfully.")
    else:
        visualize_ply(ply_file)
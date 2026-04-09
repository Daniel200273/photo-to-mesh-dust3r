import open3d as o3d
import argparse
from pathlib import Path

def visualize_ply(ply_path):
    print(f"--- Loading point cloud from {ply_path} ---")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    
    if pcd.is_empty():
        print("Error: Point cloud is empty or invalid.")
        return

    print("Point cloud loaded successfully. Opening viewer...")
    print("Controls:")
    print(" - Mouse Left: Rotate")
    print(" - Mouse Right: Pan")
    print(" - Scroll Wheel: Zoom")
    print(" - Q / Esc: Close viewer")
    
    # Renders the point cloud in an interactive 3D window
    o3d.visualization.draw_geometries(
        [pcd], 
        window_name="DUSt3R Point Cloud Viewer", 
        width=1024, 
        height=768,
        point_show_normal=False
    )

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.resolve()
    default_ply = project_root / "data" / "processed_data" / "reconstruction.ply"
    
    parser = argparse.ArgumentParser(description="Visualize a 3D PLY file.")
    parser.add_argument("--path", type=str, default=str(default_ply), help="Path to the .ply file")
    args = parser.parse_args()
    
    ply_file = Path(args.path)
    if not ply_file.exists():
        print(f"💀 Error: File not found at {ply_file}")
        print("Please ensure the DUSt3R pipeline finished successfully.")
    else:
        visualize_ply(ply_file)
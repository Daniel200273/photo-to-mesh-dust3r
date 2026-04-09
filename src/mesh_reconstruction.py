import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


def remove_statistical_noise(
	pcd: o3d.geometry.PointCloud,
	nb_neighbors: int = 20,
	std_ratio: float = 2.0,
) -> o3d.geometry.PointCloud:
	"""Remove isolated points using statistical outlier removal.

	This is the first cleanup pass for Phase 2. It removes points whose average
	distance to their neighbors is too far from the global distribution.
	"""
	if pcd.is_empty():
		return pcd

	cleaned_pcd, _ = pcd.remove_statistical_outlier(
		nb_neighbors=nb_neighbors,
		std_ratio=std_ratio,
	)
	return cleaned_pcd

def segment_plane(
    pcd: o3d.geometry.PointCloud,
    distance_threshold: float = 0.0015,
    ransac_n: int = 3,
    num_iterations: int = 1000,
) -> tuple:
    if pcd.is_empty():
        return None, None, None
    
    # 1. Run RANSAC (This returns indices, which is memory-cheap)
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    
    # 2. Use the 'invert' logic but only for the object you actually need.
    # select_by_index is faster than np.delete for O3D objects because 
    # it handles the colors and normals automatically.
    outliers_cloud = pcd.select_by_index(inliers, invert=True)
    
    # 3. Optional: If you REALLY don't need the plane cloud, 
    # don't even create it. Just return the model and the apple.
    return outliers_cloud, plane_model

def finalize_apple(apple_pcd):
    """
    Removes leftover plane bits by keeping only the largest 
    connected component (the apple).
    """
    # 1. Cluster points based on physical connectivity
    # 'cluster_indices' is just a list of integers, very low memory
    # 'eps' is the distance search radius - 0.02 (2cm) is usually safe
    labels = np.array(apple_pcd.cluster_dbscan(eps=0.01, min_points=10, print_progress=False))

    if len(labels) == 0:
        return apple_pcd

    # 2. Find the ID of the largest cluster
    # We ignore label -1 which is 'noise'
    candidate_labels = labels[labels >= 0]
    if candidate_labels.size == 0:
        return apple_pcd
        
    largest_cluster_idx = np.bincount(candidate_labels).argmax()

    # 3. Filter the cloud to keep ONLY that cluster
    apple_only = apple_pcd.select_by_index(np.where(labels == largest_cluster_idx)[0])
    
    return apple_only

def cluster_dbscan(
	pcd: o3d.geometry.PointCloud,
	eps: float = 0.05,
	min_points: int = 10,
) -> tuple:
	"""Cluster point cloud using DBSCAN algorithm.
	
	Args:
		pcd: Input point cloud
		eps: Clustering radius (max distance between points in a cluster)
		min_points: Minimum number of points to form a cluster
	
	Returns:
		tuple: (labels_array, number_of_clusters)
		- labels_array: Cluster ID for each point (-1 = noise/unclustered)
		- number_of_clusters: Total number of clusters found
	"""
	if pcd.is_empty():
		return None, 0
	
	labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
	n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
	
	return labels, n_clusters


def isolate_main_object(
	input_path: Path,
	output_path: Path,
	eps: float = 0.05,
	min_points: int = 10,
	voxel_size: float = 0.002,
) -> o3d.geometry.PointCloud:
	"""Load a point cloud, cluster using DBSCAN, keep the largest cluster.
	
	This isolates the main object from background walls, noise, and other clusters.
	"""
	print(f"--- Loading point cloud from {input_path} ---")
	pcd = o3d.io.read_point_cloud(str(input_path))
	
	if pcd.is_empty():
		raise ValueError(f"Point cloud is empty or invalid: {input_path}")
	
	before_count = len(pcd.points)
	print(f"Input points: {before_count}")
	
	# Downsample first for computational efficiency
	print(f"Downsampling with voxel size {voxel_size}...")
	pcd_downsampled = pcd.voxel_down_sample(voxel_size=voxel_size)
	downsampled_count = len(pcd_downsampled.points)
	print(f"Downsampled to {downsampled_count} points")
	
	print(f"Running DBSCAN clustering (eps={eps}, min_points={min_points})...")
	
	labels, n_clusters = cluster_dbscan(pcd_downsampled, eps=eps, min_points=min_points)
	
	print(f"Found {n_clusters} clusters")
	
	# Count points in each cluster
	unique_labels, counts = np.unique(labels, return_counts=True)
	
	# Print cluster statistics
	for label, count in zip(unique_labels, counts):
		if label == -1:
			print(f"  - Noise points: {count}")
		else:
			print(f"  - Cluster {label}: {count} points")
	
	# Find the largest cluster (main object)
	# Filter out noise (-1 label)
	cluster_ids = unique_labels[unique_labels != -1]
	if len(cluster_ids) == 0:
		print("Warning: No valid clusters found. Keeping all non-noise points.")
		largest_cluster_label = -1
		largest_points = np.sum(counts[unique_labels != -1])
	else:
		cluster_sizes = counts[unique_labels != -1]
		largest_cluster_idx = np.argmax(cluster_sizes)
		largest_cluster_label = cluster_ids[largest_cluster_idx]
		largest_points = cluster_sizes[largest_cluster_idx]
	
	print(f"Largest cluster: ID={largest_cluster_label}, points={largest_points}")
	
	# Extract only the largest cluster from downsampled cloud
	if largest_cluster_label == -1:
		# Keep all non-noise points
		mask = labels != -1
	else:
		mask = labels == largest_cluster_label
	
	main_object_downsampled = pcd_downsampled.select_by_index(np.where(mask)[0])
	
	# Now filter the original (non-downsampled) point cloud based on the largest cluster's bounding box
	print("Using convex hull of largest cluster to filter original point cloud...")
	
	if len(main_object_downsampled.points) > 3:
		# Create a convex hull of the largest cluster
		hull, _ = main_object_downsampled.compute_convex_hull()
		
		# Use the downsampled cluster's bounding box to filter the original cloud
		main_object_pcd = pcd.crop(main_object_downsampled.get_axis_aligned_bounding_box())
	else:
		main_object_pcd = pcd
	
	after_count = len(main_object_pcd.points)
	removed_count = before_count - after_count
	print(f"Kept points from original cloud: {after_count}")
	print(f"Removed points: {removed_count}")
	
	output_path.parent.mkdir(parents=True, exist_ok=True)
	o3d.io.write_point_cloud(str(output_path), main_object_pcd)
	print(f"Saved isolated object to {output_path}")
	
	return main_object_pcd


def clean_point_cloud(
	input_path: Path,
	output_path: Path,
	nb_neighbors: int = 20,
	std_ratio: float = 2.0,
) -> o3d.geometry.PointCloud:
	"""Load a point cloud, remove statistical noise, and save the result."""
	print(f"--- Loading point cloud from {input_path} ---")
	pcd = o3d.io.read_point_cloud(str(input_path))

	if pcd.is_empty():
		raise ValueError(f"Point cloud is empty or invalid: {input_path}")

	before_count = len(pcd.points)
	print(f"Input points: {before_count}")
	print(
		f"Removing statistical outliers (nb_neighbors={nb_neighbors}, std_ratio={std_ratio})..."
	)

	cleaned_pcd = remove_statistical_noise(
		pcd,
		nb_neighbors=nb_neighbors,
		std_ratio=std_ratio,
	)

	after_count = len(cleaned_pcd.points)
	removed_count = before_count - after_count
	print(f"Kept points: {after_count}")
	print(f"Removed points: {removed_count}")

	output_path.parent.mkdir(parents=True, exist_ok=True)
	o3d.io.write_point_cloud(str(output_path), cleaned_pcd)
	print(f"Saved cleaned point cloud to {output_path}")

	return cleaned_pcd


def main() -> None:
	project_root = Path(__file__).parent.parent.resolve()
	default_input = project_root / "data" / "processed_data" / "reconstruction.ply"
	default_output = project_root / "data" / "processed_data" / "reconstruction_clean.ply"
	default_plane_removed = project_root / "data" / "processed_data" / "reconstruction_no_plane.ply"

	parser = argparse.ArgumentParser(description="Phase 2 point cloud cleanup and isolation.")
	parser.add_argument("--input", type=str, default=str(default_input))
	parser.add_argument("--output", type=str, default=str(default_output))
	args = parser.parse_args()

	# Task 2.1: Statistical noise removal
	cleaned_pcd = clean_point_cloud(
		Path(args.input),
		Path(args.output),
		nb_neighbors=20,
		std_ratio=2.0,
	)
	
	# Task 2.2: Plane removal on the cleaned result
	print(f"--- Removing plane from cleaned point cloud ---")
	outliers_cloud, plane_model = segment_plane(
		cleaned_pcd,
		distance_threshold=0.0015
	)

	final_object = finalize_apple(outliers_cloud)
	
	if outliers_cloud:
		plane_removed_path = default_plane_removed
		plane_removed_path.parent.mkdir(parents=True, exist_ok=True)
		o3d.io.write_point_cloud(str(plane_removed_path), final_object)
		print(f"Saved plane-removed point cloud to {plane_removed_path}")
	else:
		print("No plane found or error in segmentation.")


	# Task 2.3: DBSCAN clustering (currently disabled as requested)
	# isolate_main_object(...)


if __name__ == "__main__":
	main()

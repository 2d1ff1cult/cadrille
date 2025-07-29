# mesh_to_pc.py
# used to learn a bit more about trimesh library and how to process meshes
import numpy as np
from argparse import ArgumentParser
import trimesh
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from pytorch3d.ops import sample_farthest_points
import torch

# basic function to get information about a mesh
def get_metrics(mesh):
    volume = mesh.volume
    surface_area = mesh.area
    is_watertight = mesh.is_watertight
    print(f"Volume: {volume}")
    print(f"Surface Area: {surface_area}")
    print(f"Is Watertight: {is_watertight}")

# might be useless, but keeping it for reference
# def convert_to_point_cloud(mesh, num_points=1024):
#     """
#     Convert a mesh to a point cloud by sampling points uniformly on its surface.
#     """
#     if not isinstance(mesh, trimesh.Trimesh):
#         raise ValueError("Input must be a trimesh object")
    
#     points = mesh.sample(num_points)
#     centroid = points.mean(axis=0)
#     points -= centroid  # Center the point cloud
#     max_dist = np.max(np.linalg.norm(points, axis=1))
#     if max_dist > 0:
#         points /= max_dist  # Normalize the point cloud
#    
#     return points.astype(np.float32).shape

# below is taken from CAD Recode
def process_mesh_to_point_cloud(mesh, n_points=33, n_pre_points=8192):
    """
    Convert a mesh to a point cloud by sampling points uniformly on its surface.
    mesh: trimesh.Trimesh object
    n_points: Number of points to sample in the final point cloud
    n_pre_points: Number of points to sample in the preliminary point cloud
    """
    vertices, _ = trimesh.sample.sample_surface(mesh, n_pre_points)
    _, ids = sample_farthest_points(torch.tensor(vertices).unsqueeze(0), K=n_points)
    ids = ids[0].numpy()
    return np.asarray(vertices[ids])
# end cad_recode stuff

# here, trying to save pcs to files
def create_and_save_point_cloud_file(point_cloud_data):
    # Create an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_data)

    # Save the point cloud to a PLY file
    # output_ply_file = "test.ply"
    # o3d.io.write_point_cloud(output_ply_file, pcd)

    # print(f"Point cloud saved to {output_ply_file}")

    # Optional: Visualize the point cloud (requires a display)
    # o3d.visualization.draw_geometries([pcd])
    # gui.Application.instance.initialize()

    vis = o3d.visualization.O3DVisualizer("Point Cloud Viewer", 1024, 768)

    # Add the point cloud geometry to the scene
    vis.add_geometry("point_cloud", pcd)

    # Show 3D label at the center of the cloud
    point_count = len(point_cloud_data)
    stats_text = f"Points: {point_count:,}"
    center = pcd.get_center()
    vis.add_3d_label(center, stats_text)

    # Add window and run (this blocks)
    gui.Application.instance.add_window(vis)
    gui.Application.instance.run()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--mesh-path', type=str, required=True,
                        help='Path to the mesh file')
    args = parser.parse_args()
    # mesh = trimesh.load_mesh(args.mesh_path, force='mesh')
    # get_metrics(mesh)
    # print(convert_to_point_cloud(mesh))

    # stuff from cadrecode
    # this is to compare methods used from CADRecode vs. what i read from trimesh documentation
    gt_mesh = trimesh.load_mesh(args.mesh_path, force='mesh')
    gt_mesh.apply_translation(-(gt_mesh.bounds[0] + gt_mesh.bounds[1]) / 2.0)
    gt_mesh.apply_scale(2.0 / max(gt_mesh.extents))
    # np.random.seed(0)
    point_cloud = process_mesh_to_point_cloud(gt_mesh)
    # print(f"CADRecode point_cloud: {point_cloud}")
    create_and_save_point_cloud_file(point_cloud)
    print(f"CADRecode point_cloud shape: {point_cloud.shape}")

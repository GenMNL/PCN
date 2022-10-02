import open3d as o3d
import os

idx = 3
path = os.path.join('bridge', 'fine', str(idx) + ".ply")
# path = os.path.join("..", "..", "mesurement", "library", "20220822_bridge01_comp.pcd")

pc = o3d.io.read_point_cloud(path)
o3d.visualization.draw_geometries([pc])

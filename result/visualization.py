import open3d as o3d
import os

idx = 3
path = os.path.join('bridge', 'fine', str(idx) + ".ply")

pc = o3d.io.read_point_cloud(path)
o3d.visualization.draw_geometries([pc])

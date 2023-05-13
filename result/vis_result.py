import open3d as o3d
import os

if __name__ == "__main__":

    type_result = input("select type :")
    num = input("select num :")

    path = os.path.join(".", "bridge", type_result, num+".ply")

    pc = o3d.io.read_point_cloud(path)

    o3d.visualization.draw_geometries([pc])

import numpy as np
import argparse
from vedo import Mesh, show
from os.path import basename


def calc_area_3d(verts):
    """
    计算 3D 三角形的面积
    参数:
        verts: (3, 3) 的顶点坐标数组
    返回:
        float: 三角形的面积
    """
    a = verts[1] - verts[0]
    b = verts[2] - verts[0]
    cross_prod = np.cross(a, b)
    area = 0.5 * np.linalg.norm(cross_prod)
    return area


def calc_area_2d(verts):
    """
    计算 2D 三角形的面积
    参数:
        verts: (3, 2) 的顶点坐标数组
    返回:
        float: 三角形的面积
    """
    a = verts[0]
    b = verts[1]
    c = verts[2]
    area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - 
                     (c[0] - a[0]) * (b[1] - a[1]))
    return area


def visualize_3d_2d_area_ratio(npz_fp):
    data = np.load(npz_fp)
    V = data['V']       # 顶点位置 (N, 3)
    F = data['F']       # 面片索引 (M, 3)
    UV = data['UV']     # 对应 UV 坐标 (N, 2)
    
    mesh = Mesh([UV, F])
    area_ratio = np.zeros((F.shape[0],))

    for i in range(F.shape[0]):
        f = F[i]
        area_3d = calc_area_3d(V[f])
        area_2d = calc_area_2d(UV[f])
        if area_2d > 0:
            area_ratio[i] = np.sqrt(area_3d / area_2d)
        else:
            area_ratio[i] = 0.0  # 避免除以零


    mesh.celldata["area_ratio"] = area_ratio
    mesh.cmap("viridis", 
              area_ratio,
              name="3d/2d area ratio",
              vmin=np.percentile(area_ratio, 5),
              vmax=np.percentile(area_ratio, 95),
              logscale=False
              ).add_scalarbar("3d/2d area ratio")

    # mesh.cellcolors = area_ratio

    show(mesh, axes=1, viewup='y', interactive=True, title=basename(npz_fp)).close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize npz file UV area distortion")
    parser.add_argument("npz_fp", help=".npz file path.")
    args = parser.parse_args()

    visualize_3d_2d_area_ratio(args.npz_fp)

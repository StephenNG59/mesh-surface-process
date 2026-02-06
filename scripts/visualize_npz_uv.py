import numpy as np
import argparse
from vedo import Mesh, show, dataurl
import igl


def visualize_uv_by_texture(npz_fp):
    data = np.load(npz_fp)
    V = data['V']
    F = data['F']
    UV = data['UV']
    labels = data['labels']
    mesh = Mesh([V, F])
    mesh.texture(dataurl+'textures/'+'bricks.jpg', UV)  # bricks.jpg / earth1.jpg
    show(mesh, axes=1, viewup='z', interactive=True).close()


def visualize_vertex_normal_by_color(npz_fp):
    data = np.load(npz_fp)
    F = data['F']
    UV = data['UV']
    VN = data['N']
    mesh = Mesh([UV, F])

    VN = (VN - VN.min(axis=0)) / (VN.max(axis=0) - VN.min(axis=0))
    mesh.pointcolors = VN * 255

    show(mesh, axes=1, viewup='y', interactive=True).close()


def visualize_face_normal_by_color(npz_fp):
    data = np.load(npz_fp)
    V = data['V']
    F = data['F']
    UV = data['UV']
    mesh = Mesh([UV, F])

    # face normals
    FN = igl.per_face_normals(V, F, np.array([1.,1.,1.]))
    FN = (FN - FN.min(axis=0)) / (FN.max(axis=0) - FN.min(axis=0))
    mesh.cellcolors = FN * 255

    show(mesh, axes=1, viewup='y', interactive=True).close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize npz file uv")
    parser.add_argument("npz_fp", help=".npz file path.")
    args = parser.parse_args()

    # visualize_uv_by_texture(args.npz_fp)
    # visualize_face_normal_by_color(args.npz_fp)
    visualize_vertex_normal_by_color(args.npz_fp)


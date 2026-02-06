import numpy as np
from vedo import Mesh, show, dataurl
import colorcet
import json

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize .obj .json file")
    parser.add_argument("mesh_fp", help=".obj file path.")
    parser.add_argument("anno_fp", help=".json file path.")
    args = parser.parse_args()

    mesh = Mesh(args.mesh_fp)

    with open(args.anno_fp) as f:
        data = json.load(f)
    labels = data['labels']

    scals = labels #UV[:, 0]

    mycmap = colorcet.bmy
    alphas = np.linspace(0.9, 0.6, num=len(mycmap))
    mesh.cmap(mycmap, scals, alpha=alphas).add_scalarbar()

    show(mesh, axes=1, viewup='z', interactive=True).close()
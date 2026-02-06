import numpy as np
from vedo import Mesh, show, dataurl
import colorcet

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize npz file")
    parser.add_argument("npz_fp", help=".npz file path.")
    args = parser.parse_args()

    data = np.load(args.npz_fp)
    V = data['V']
    F = data['F']
    UV = data['UV']
    # labels = data['labels']
    labels = data['instances']
    mesh = Mesh([V, F])


    scals = labels #UV[:, 0]

    mycmap = colorcet.bmy
    alphas = np.linspace(0.9, 0.6, num=len(mycmap))
    mesh.cmap(mycmap, scals, alpha=alphas).add_scalarbar()

    # mesh.texture(dataurl+'textures/'+'bricks.jpg', UV)  # bricks.jpg / earth1.jpg

    show(mesh, axes=1, viewup='z', interactive=True).close()
import numpy as np
from vedo import Mesh, show, dataurl
import colorcet
import re

def read_m_file(filename):
    V = []
    F = []
    labels = []
    instances = []
    id_to_index = {}

    label_pattern = re.compile(r'label=(\d+)')
    instance_pattern = re.compile(r'instance=(\d+)')

    with open(filename, 'r') as f:
        vertex_count = 0
        for line in f:
            if line.startswith('Vertex'):
                parts = line.split()
                vid = int(parts[1])
                x, y, z = map(float, parts[2:5])
                label_match = label_pattern.search(line)
                instance_match = instance_pattern.search(line)
                label = int(label_match.group(1)) if label_match else -1
                instance = int(instance_match.group(1)) if instance_match else -1

                V.append([x, y, z])
                labels.append(label)
                instances.append(instance)
                id_to_index[vid] = vertex_count
                vertex_count += 1

            elif line.startswith('Face'):
                parts = line.split()
                fid = int(parts[1])
                v1, v2, v3 = map(int, parts[2:5])
                # 使用映射转换成实际索引
                F.append([
                    id_to_index[v1],
                    id_to_index[v2],
                    id_to_index[v3]
                ])

    return (
        np.array(V),
        np.array(F),
        np.array(labels),
        np.array(instances)
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize .m file")
    parser.add_argument("m_fp", help=".m file path.")
    args = parser.parse_args()

    V, F, labels, instances = read_m_file(args.m_fp)


    mesh = Mesh([V, F])
    scals = labels #UV[:, 0]

    mycmap = colorcet.bmy
    alphas = np.linspace(0.9, 0.6, num=len(mycmap))
    mesh.cmap(mycmap, scals, alpha=alphas).add_scalarbar()

    show(mesh, axes=1, viewup='z', interactive=True).close()

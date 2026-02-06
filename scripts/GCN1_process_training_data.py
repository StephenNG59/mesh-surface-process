import numpy as np
import os
import torch
from torch_geometric.data import Data
from glob import glob
import igl
from tqdm import tqdm


dirs = {
    'obj': 'single_tooth_meshes/upper/test(16cls)/obj',  # sample对应的mesh(V, F)
    'npy': 'single_tooth_meshes/upper/test(16cls)/npy',  # 记录了每个顶点的attr(normals,k1,k2,k3)
    'pred_mask': 'single_tooth_meshes/upper/test(16cls)/pred_mask',
    'gt_mask': 'single_tooth_meshes/upper/test(16cls)/gt_mask',
}
dataset_output = "single_tooth_meshes/upper/test(16cls)/single_tooth_meshes(test).pt"


def build_edge_index_from_faces(faces):
    edge_set = set()
    for tri in faces:
        i, j, k = tri
        edge_set.update([(i, j), (j, i), (j, k), (k, j), (k, i), (i, k)])
    edge_index = np.array(list(edge_set)).T
    return edge_index  # shape: [2, num_edges]

def process_sample(name, dirs):
    # print(name)
    # 加载数据
    V, F = igl.read_triangle_mesh(os.path.join(dirs['obj'], name + '.obj'))  # V: [N, 3], F: [M, 3]
    attr = np.load(os.path.join(dirs['npy'], name + '.npy'))  # [N, 6]
    with open(os.path.join(dirs['pred_mask'], name + '.txt'), 'r') as f:
        pred_mask = f.readline().strip().split()
        pred_mask = np.array([int(d) for d in pred_mask])    # [N]
    with open(os.path.join(dirs['gt_mask'], name + '.txt'), 'r') as f:
        gt_mask = f.readline().strip().split()
        gt_mask = np.array([int(d) for d in gt_mask])        # [N]

    assert attr.shape[0] == pred_mask.shape[0] == gt_mask.shape[0] == V.shape[0]

    # 找出 pred_mask ≠ 0 区域中对应的 gt_label
    foreground_idx = np.where(pred_mask != 0)[0]
    foreground_gt = gt_mask[foreground_idx]
    if len(foreground_gt) == 0:
        print(f"[Warning] No foreground in {name}, skip.")
        return None

    values, counts = np.unique(foreground_gt, return_counts=True)
    main_label = values[np.argmax(counts)]  # 最常见的 gt label

    # 屏蔽非当前牙齿的部分
    filtered_gt = np.where(gt_mask == main_label, gt_mask, 0)

    # 映射为 1~8（gt 是 11~48）
    final_label = filtered_gt % 10

    # 把 pred_mask 和 final_label 转化成 0 or 
    pred_mask_bin = (pred_mask != 0).astype(np.float32)  # shape [N], values ∈ {0.0, 1.0}
    final_label_bin = (final_label != 0).astype(np.int32)  # shape [N], values ∈ {0, 1}

    # 添加 pred_mask 为第 7 维 feature
    x = np.concatenate([V, attr, pred_mask_bin[:, None]], axis=1)  # [N, 7]

    # 构建 edge_index
    edge_index = build_edge_index_from_faces(F)  # shape: [2, num_edges]

    # 构建 PyG Data 对象
    data = Data(
        x=torch.tensor(x, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        y=torch.tensor(final_label_bin, dtype=torch.long),
    )

    return data


if __name__ == "__main__":

    all_names = [os.path.splitext(os.path.basename(p))[0] for p in glob(f"{dirs['obj']}/*.obj")]
    dataset = []
    print(all_names)

    for name in tqdm(all_names[:]):
        data = process_sample(name, dirs)
        if data is not None:
            dataset.append(data)
        
    torch.save(dataset, dataset_output)

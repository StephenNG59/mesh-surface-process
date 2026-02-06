# 由TeethGNN得到的predictions(10000 faces labels)，映射回原先的顶点labels
# 需要.h5（包含10000faces的数据）和.npz（包含原始mesh的数据）

# 读取数据：
# 从 .h5 获取 fea → face centers (10000, 3)；
# 从 .txt 获取每个 face 的预测标签；
# 从 .npz 获取顶点 V 和三角面 F。

# KNN 查询：
# 使用每个 vertex 的位置与 face center 计算距离；
# 找到距离最近的 k 个 faces，统计这些 faces 的预测标签；
# 将众数作为该 vertex 的预测标签（多数投票，若平局则可设为 0）。

# 输出：
# 将所有 vertex 的预测标签写入新的 .txt 文件。

import os
import numpy as np
import h5py
from scipy.spatial import cKDTree
from collections import Counter

npz_dir = R"E:\wyc\Data\teeth3ds+\new\upper_preprocessed\npz"
h5_dir = R"E:\wyc\Data\teethgnn_data\2_features_o3d_simplified_10000"
pred_dir = R"E:\wyc\Projects\Detect&Teethgnn\teethgnn\runs\MICCAI\plain_gcn_batch\Log_2025-06-17_14-52-44\results\predictions"
output_dir = R"E:\wyc\Projects\Detect&Teethgnn\teethgnn\runs\MICCAI\plain_gcn_batch\Log_2025-06-17_14-52-44\results\predictions_map2origin"


def infer_vertex_labels(txt_dir, npz_dir, h5_dir, output_dir, k=3):
    os.makedirs(output_dir, exist_ok=True)
    
    for txt_file in os.listdir(txt_dir):
        if not txt_file.endswith('.txt'):
            continue
        
        base_name = os.path.splitext(txt_file)[0]
        npz_path = os.path.join(npz_dir, base_name + '.npz')
        h5_path = os.path.join(h5_dir, base_name + '.h5')
        txt_path = os.path.join(txt_dir, txt_file)
        
        if not os.path.exists(npz_path) or not os.path.exists(h5_path):
            print(f"Missing npz or h5 for {base_name}, skipping.")
            continue

        # 1. 读取 face 预测标签
        with open(txt_path, 'r') as f:
            face_pred_labels = np.array([int(line.strip()) for line in f])

        # 2. 读取 fea
        with h5py.File(h5_path, 'r') as f:
            face_centers = f['fea'][:, :3]  # 取前三维作为中心坐标

        # 3. 读取 mesh 数据
        data = np.load(npz_path)
        vertices = data['V']
        
        # 4. KNN 查询每个 vertex 最近的 face centers
        tree = cKDTree(face_centers)
        dists, idxs = tree.query(vertices, k=min(k, face_centers.shape[0]))

        # 保证 idxs 是二维
        if len(idxs.shape) == 1:
            idxs = idxs[:, np.newaxis]

        # 5. 投票确定 vertex label
        vertex_labels = []
        for indices in idxs:
            labels = face_pred_labels[indices]
            count = Counter(labels)
            common = count.most_common()
            if len(common) > 1 and common[0][1] == common[1][1]:
                vertex_labels.append(0)  # 平票
            else:
                vertex_labels.append(common[0][0])
        
        # 6. 写入 vertex-level label 文件
        output_path = os.path.join(output_dir, base_name + ".txt")
        with open(output_path, 'w') as f_out:
            f_out.write("\n".join(str(l) for l in vertex_labels))

        print(f"Processed: {base_name}")


if __name__ == "__main__":
    infer_vertex_labels(
        txt_dir=pred_dir,
        npz_dir=npz_dir,
        h5_dir=h5_dir,
        output_dir=output_dir,
        k=5  # 可调整
    )
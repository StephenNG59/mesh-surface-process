import os
import numpy as np
from tqdm import tqdm
from scipy.spatial import cKDTree

# 路径配置
base_dir = R"E:\wyc\Data\teeth3ds+\new\single_tooth_meshes\upper\test(16cls)"
results_dir = os.path.join(base_dir, "preds(stage2)", "results")
names_txt = os.path.join(base_dir, "3977_names.txt")
obj_dir = os.path.join(base_dir, "obj")
npz_dir = R"E:\wyc\Data\teeth3ds+\new\upper_preprocessed\npz(ARAP_UV)"
yolo_dir = R"E:\wyc\Data\teeth3ds+\YOLO训练结果(upper)\MICCAI_16cls\predict\labels" # R"E:\wyc\Data\teeth3ds+\YOLO训练结果\MICCAI_16cls_lower_yolo11n\predict\labels"
output_dir = os.path.join(base_dir, "final_prediction_output")
os.makedirs(output_dir, exist_ok=True)

# 加载所有名字
with open(names_txt, 'r') as f:
    all_names = eval(f.read())  # assume it's a Python list format
print("all_names cnt: ", all_names.__len__())

# 分组 names -> casename: [(instance_id, global_index, name)]
casename_dict = {}
for idx, name in enumerate(all_names):
    if name.count('_') < 2:
        continue
    casename = '_'.join(name.split('_')[:2])
    instance_id = int(name.split('_')[-1])
    casename_dict.setdefault(casename, []).append((instance_id, idx, name))

# 处理每个 casename
case_i = 0
print(len(casename_dict.items()))
for casename, instances in casename_dict.items():
    case_i += 1
    # if case_i >= 52:
    #     continue

    # 加载 npz 顶点信息
    npz_path = os.path.join(npz_dir, casename + ".npz")
    if not os.path.exists(npz_path):
        continue
    npz_data = np.load(npz_path)
    full_vertices = npz_data['V']  # shape: (N, 3)
    n_vertices = full_vertices.shape[0]
    print(f"=== [{case_i}] Processing {casename}... (Total NV = {n_vertices})")

    # 初始化结果数组
    result_array = np.zeros((n_vertices, 3), dtype=np.float32)  # class, instance, confidence
    result_array[:, 0] = -1  # no class => -1

    # 加载 YOLO 标签
    yolo_path = os.path.join(yolo_dir, casename + ".txt")
    if not os.path.exists(yolo_path):
        continue
    with open(yolo_path, 'r') as f:
        yolo_lines = f.readlines()

    # 按 instance_id 排序
    instances_sorted = sorted(instances, key=lambda x: x[0])  # (instance_id, global_idx, name)

    # 建立整个 mesh 的 kdtree
    kdtree = cKDTree(full_vertices)

    for instance_id, global_idx, name in instances_sorted:
        pred_path = os.path.join(results_dir, f"{global_idx}.txt")
        obj_path = os.path.join(obj_dir, f"{name}.obj")
        if not os.path.exists(pred_path) or not os.path.exists(obj_path):
            continue

        # 读取预测二值标签（0/1）
        with open(pred_path, 'r') as f:
            pred_labels = np.array([int(line.strip()) for line in f.readlines()], dtype=np.uint8)

        # 读取 obj 顶点
        obj_vertices = []
        with open(obj_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    obj_vertices.append(list(map(float, line.strip().split()[1:])))
        obj_vertices = np.array(obj_vertices, dtype=np.float32)

        print(f'  ({case_i}-{instance_id})nv={len(pred_labels)} for {name}:{global_idx}.txt', end='')
        
        if len(pred_labels) != len(obj_vertices):
            print(f"Length mismatch in {name}: labels {len(pred_labels)} vs vertices {len(obj_vertices)}")
            continue

        # 从 YOLO 文件中读取该 instance 的 class 和 confidence
        if instance_id - 1 >= len(yolo_lines):
            continue
        yolo_parts = yolo_lines[instance_id - 1].strip().split()
        class_id = int(float(yolo_parts[0]))
        confidence = float(yolo_parts[-1])

        # 匹配并更新每个前景点
        match_count = 0
        update_count = 0
        for i, (v, pred) in enumerate(zip(obj_vertices, pred_labels)):
            if pred == 1:
                # =============================================
                # === New version ===
                dist, idx = kdtree.query(v, k=1)
                if dist > 1e-5:
                    continue  # no match
                match_count += 1
                if result_array[idx, 0] == 0 or result_array[idx, 2] < confidence:
                    result_array[idx] = [class_id, instance_id, confidence]
                    update_count += 1

                # =============================================
                # === Old version ===
                # # 找匹配的 index in full_vertices
                # dists = np.linalg.norm(full_vertices - v, axis=1)
                # idx_match = np.where(dists < 1e-5)[0]
                # if len(idx_match) == 0:
                #     continue
                # match_count += len(idx_match)
                # # if len(idx_match) > 1:
                # #     idx = idx_match[0]
                # for idx in idx_match:
                #     # 若尚未赋值，或当前 confidence 更大，则赋值
                #     if result_array[idx, 0] == 0 or result_array[idx, 2] < confidence:
                #         result_array[idx] = [class_id, instance_id, confidence]
                #         update_count += 1

        print(f" | updated/total match: {update_count}/{match_count}({update_count / match_count * 100:.3f}%)")

    # 保存最终结果
    out_path = os.path.join(output_dir, casename + ".npy")
    np.save(out_path, result_array)

# 结合YOLO的segment预测结果以及UNet的instance mask预测结果，得到最终结果
# 记录到.npy中，(nV, 3)数组，每个顶点对应了(class, instance, confidence)三个数字
# 需要的原材料：
# 1. YOLO segment result: class_id, confidence
# 2. 从YOLO到UNet instance提取过程的uv bboxes
# 3. UNet对每个instance的预测结果mask
# 4. 记录了原始mesh的UV信息的.npz文件
# 关键：
# * UNet的predicted mask一个图像生成一个contour（如果有多个，取面积最大的）
# * cv2生成的contour的y轴和我的记录是反转的
# * 一个mesh的所有instances的masks组合起来，如有重叠，取confidence较高的为准
# * 保存的结果为.npy，形状为[n_vertices, 3]，三维：class_id，instance_id，confidence

import os
import cv2
import numpy as np
from shapely.geometry import Point, Polygon
import trimesh
import matplotlib.pyplot as plt
import re
from tqdm import tqdm

paths = {
    "segment_txt": R"E:\wyc\Data\teeth3ds+\YOLO训练结果\MICCAI_ARAP_normal_8cls_100ep\labels\test", #  ./sample_name.txt
    "uv_bbox": "640NKM/uv",                      #  ./sample_name_1.uv.txt
    "local_masks": "640NKM/pred_masks(stage2)/640px,bnd_size20,bnd_wt10,non_bnd_wt1,bs4,s1.0,cls1,cha7",          #  ./sample_name_1.png
    "mesh_npz": "upper_preprocessed/npz",         #  ./sample_name.npz
    "final_output": "640NKM/final_predictions/640px,bnd_size20,bnd_wt10,non_bnd_wt1,bs4,s1.0,cls1,cha7"
}

# updated_cases = {
#     'DG27PDD4_upper', '014RZJT4_upper', 'U9UBDO05_upper', '4OTQF1P6_upper', 'APLDXK7R_upper', 'X67M31XQ_upper', '0140XF2Z_upper', 'BIIEY91S_upper', 'SJDH33M1_upper', 'TZ2JPAJT_upper', '5WCMFAT0_upper', '0154T9CN_upper', '01KXSPAA_upper', '017APYC4_upper', '01422MSK_upper', '0OTKQ5J9_upper', 'PM5K088N_upper', '01F6WV5D_upper', 'Q21I52KT_upper', '2DZ821ZJ_upper', '01KK2DER_upper', 'FWTHWWD3_upper', 'TEPBA32B_upper'
# }


def parse_yolo_segment_line(line):
    """解析YOLO的segment格式行"""
    parts = line.strip().split()
    class_id = int(parts[0])
    confidence = float(parts[-1])
    coords = list(map(float, parts[1:-1]))
    points = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
    return class_id, points, confidence

def read_contour_from_mask(mask_path):
    """读取局部mask图像并提取最大contour"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    contours, _ = cv2.findContours((mask == 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # 选最大面积的
    max_contour = max(contours, key=cv2.contourArea)

    # 把 contour 点坐标归一化
    max_contour = max_contour[:, 0, :].astype(np.float64) # 去除嵌套结构 shape: (N, 2)
    max_contour[:, 0] /= float(mask.shape[0])
    max_contour[:, 1] /= float(mask.shape[1])
    # cv2 提取出来的contour坐标y轴和我的y轴方向相反，需要反转y轴
    max_contour[:, 1] = 1 - max_contour[:, 1]

    return max_contour 

def transform_local_to_global(contour, bbox):
    """将局部图像坐标的contour转换回全局UV坐标系（归一化 0~1）"""
    x_min, y_min, x_max, y_max = bbox
    w, h = x_max - x_min, y_max - y_min
    contour_uv = contour.astype(np.float32)
    contour_uv[:, 0] = contour_uv[:, 0] * w + x_min
    contour_uv[:, 1] = contour_uv[:, 1] * h + y_min
    # contour_uv[:, 1] = (1 - contour_uv[:, 1]) * h + y_min
    return contour_uv

def point_in_polygon(point_uv, polygon):
    """判断一个UV点是否在多边形内"""
    return Polygon(polygon).contains(Point(point_uv))

# 可视化 contour 在全图 UV 空间中的位置
def visualize_contour_uv(contour_uv, instance_id, class_id=None, sample_name=None):
    plt.figure(figsize=(5, 5))
    plt.plot(contour_uv[:, 0], contour_uv[:, 1], '-o', linewidth=2, markersize=2)
    # plt.gca().invert_yaxis()  # UV图像Y轴方向与图像一致
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(f"Instance {instance_id}" + (f" | Class {class_id}" if class_id is not None else "") +
              (f" | Sample {sample_name}" if sample_name else ""))
    plt.xlabel("U")
    plt.ylabel("V")
    plt.grid(True)
    plt.show()

def process_sample(sample_name, paths):
    # === 读取数据 ===
    with open(os.path.join(paths['segment_txt'], f"{sample_name}.txt"), 'r') as f:
        segment_lines = f.readlines()

    uv_files = sorted([f for f in os.listdir(paths['uv_bbox']) if f.startswith(f"{sample_name}_") and f.endswith(".uv.txt")])
    npz_data = np.load(os.path.join(paths['mesh_npz'], f"{sample_name}.npz"))
    UV = npz_data['UV']  # shape: (num_vertices, 2)

    # Normalize UV to [0,1]
    UV = (UV - UV.min(axis=0)) / (UV.max(axis=0) - UV.min(axis=0) + 1e-8)

    predictions = [None for _ in range(len(UV))]  # 每个顶点的(class, instance_id, confidence)

    for uv_file in tqdm(uv_files):
        match = re.search(r'_(\d+)\.uv\.txt$', uv_file)  # *_{1}.uv.txt 找出{1}这个数字
        if match:
            instance_id = int(match.group(1))
        else:
            continue
        uv_path = os.path.join(paths['uv_bbox'], uv_file)
        png_path = os.path.join(paths['local_masks'], f"{sample_name}_{instance_id}.png")
        # print(f"instance_id:{instance_id}")

        # Step 1: 获取bbox
        with open(uv_path, 'r') as f:
            bbox = list(map(float, f.read().strip().split()))

        # Step 2: 获取mask contour
        contour = read_contour_from_mask(png_path)
        if contour is None or len(contour) < 3:
            continue  # 忽略空mask或不成多边形的
        # visualize_contour_uv(contour, instance_id, class_id=None, sample_name=sample_name)

        # Step 3: 转换contour到全图UV坐标系
        contour_uv = transform_local_to_global(contour, bbox)
        # visualize_contour_uv(contour_uv, instance_id, class_id=None, sample_name=sample_name)

        # Step 4: 获取对应的YOLO预测
        if instance_id - 1 >= len(segment_lines):
            continue
        class_id, _, confidence = parse_yolo_segment_line(segment_lines[instance_id - 1])

        polygon = Polygon(contour_uv)

        # Step 5: 判断哪些mesh UV点落入polygon
        for v_idx, uv in enumerate(UV):
            if not (bbox[0] <= uv[0] <= bbox[2] and bbox[1] <= uv[1] <= bbox[3]):
                continue  # 不在bbox中
            if polygon.contains(Point(uv)):
                if predictions[v_idx] is None or confidence > predictions[v_idx][2]:
                    predictions[v_idx] = (class_id, instance_id, confidence)

    return predictions  # 返回每个顶点的预测信息（class, instance, conf）

def visualize_mesh_with_labels(npz_path, prediction_path, save_path=None):
    data = np.load(npz_path)
    V = data['V']  # shape: (N, 3)
    F = data['F']  # shape: (M, 3)

    preds = np.load(prediction_path)  # shape: (N, 3)
    class_ids = preds[:, 0].astype(int)

    # 为每个顶点分配颜色：用不同class对应不同颜色；未预测的(-1)设为灰色
    num_classes = class_ids[class_ids >= 0].max() + 1 if np.any(class_ids >= 0) else 1
    color_map = plt.get_cmap('tab20', num_classes)

    colors = np.zeros((len(V), 4))  # RGBA
    for i, cid in enumerate(class_ids):
        if cid == -1:
            colors[i] = [0.5, 0.5, 0.5, 1.0]  # 未分类灰色
        else:
            colors[i] = color_map(cid)

    mesh = trimesh.Trimesh(vertices=V, faces=F, vertex_colors=(colors[:, :3] * 255).astype(np.uint8))

    mesh.show()  # 用trimesh的默认viewer显示

    if save_path:
        mesh.export(save_path)
        print(f"Mesh saved to {save_path}")


if __name__ == "__main__":

    segment_files = [f for f in os.listdir(paths['segment_txt']) if f.endswith('.txt')]
    sample_names = [os.path.splitext(f)[0] for f in segment_files]

    for i, sample_name in enumerate(sample_names[:]):
        # if sample_name not in updated_cases:
        #     continue
        print(f"[{i}] Processing {sample_name}...")
        # 输出格式：vertex_predictions[idx] = (class_id, instance_id, confidence)
        # 如果某个顶点没有匹配任何instance，值为 None
        vertex_predictions = process_sample(sample_name, paths)

        # 示例保存预测结果为数组
        output_array = np.full((len(vertex_predictions), 3), -1.0)
        for i, pred in enumerate(vertex_predictions):
            if pred:
                output_array[i] = pred

        os.makedirs(paths["final_output"], exist_ok=True)
        np.save(os.path.join(paths["final_output"], f"{sample_name}.npy"), output_array)

        # visualize_mesh_with_labels(
        #     npz_path=os.path.join(paths['mesh_npz'], f"{sample_name}.npz"),
        #     prediction_path=os.path.join(paths['final_output'], f"{sample_name}.npy"),
        #     save_path=os.path.join(paths['final_output'], f"{sample_name}.ply")  # 可选，保存为PLY
        #     )



# 从 .npz 中获取 UV 矩阵
# 从 .txt 中获取 mask
# 对 mesh 每个顶点，根据 UV 对应 mask 中的一个像素，获得 label
# 把基于 mesh 的 predictions 记录在新 .txt 中
# ！注意！get_label_for_vertex(vertex, masks) 0,1,...,7 变为 0,1,...,8 了

import numpy as np
from shapely.geometry import Point, Polygon
import os
from tqdm import tqdm
import json

label_mapping_type = 1
npz_dir = R'E:\wyc\Data\teeth3ds+\new\lower_preprocessed\npz(arap)'
txt_dir = R'E:\wyc\Data\teeth3ds+\YOLO训练结果\MICCAI_16cls_lower_yolo11n\predict\labels'
output_dir = R'E:\wyc\Data\teeth3ds+\YOLO训练结果\MICCAI_16cls_lower_yolo11n\predict_labels(per_vertex)'

def map_single_label(origin_label):
    if label_mapping_type == 1:
        return origin_label + 1
    elif label_mapping_type == 2:
        return origin_label % 10

def load_UVs_from_npz(npz_path):
    data = np.load(npz_path)
    uv = np.array(data['UV'])
    min_uv = uv.min(axis=0)
    max_uv = uv.max(axis=0)
    normalized_uv = (uv - min_uv) / (max_uv - min_uv)
    return normalized_uv

def load_mask_from_txt(txt_path):
    masks = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            label = int(parts[0])
            label = map_single_label(label)
            points = [(float(parts[i]), 1 - float(parts[i+1])) for i in range(1, len(parts)-1, 2)]  # NOTE! 这里需要反转y轴!!!
            confidence = float(parts[-1])
            masks.append((label, Polygon(points), confidence))
    return masks

def get_label_for_vertex(vertex, masks):
    """
    空出的0号label，表示Background
    """
    point = Point(vertex)
    for label, polygon, confidence in masks:
        if polygon.contains(point):
            return label, confidence
    return 0, None  # If the vertex is not inside any polygon

def process_directory(npz_dir, txt_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for txt_file in tqdm(os.listdir(txt_dir)[:], desc="Processing files"):
        txt_path = os.path.join(txt_dir, txt_file)
        npz_path = os.path.join(npz_dir, txt_file.replace('.txt', '.npz'))

        vertices = load_UVs_from_npz(npz_path)

        # Load the corresponding mask annotations
        masks = load_mask_from_txt(txt_path)

        # Prepare the output file
        output_file_path = os.path.join(output_dir, txt_file)
        labels = []
        for vertex in vertices:
            label, confidence = get_label_for_vertex(vertex, masks)
            labels.append(str(label))

        with open(output_file_path, "w") as f:
            f.write("\n".join(labels))


if __name__ == "__main__":
    process_directory(npz_dir, txt_dir, output_dir)

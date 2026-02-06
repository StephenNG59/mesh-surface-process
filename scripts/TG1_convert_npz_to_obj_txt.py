# 遍历指定目录中的所有 .npz 文件;
# 读取每个文件中的 ["V"], ["F"], ["labels"];
# 生成 .obj 文件并保存到指定目录;
# 计算每个面（face）的标签并保存为 .txt 文件.
# 后续可以使用teethgnn中的1_create_features.py等进行处理.

import os
import numpy as np
from tqdm import tqdm


input_dir = R"E:\wyc\Data\teeth3ds+\new\upper_preprocessed\npz"
obj_output_dir = R"E:\wyc\Data\teethgnn_data\obj"
label_output_dir = R"E:\wyc\Data\teethgnn_data\txt"


def process_npz_files(input_dir, obj_output_dir, label_output_dir):
    os.makedirs(obj_output_dir, exist_ok=True)
    os.makedirs(label_output_dir, exist_ok=True)

    for file_name in tqdm(os.listdir(input_dir)):
        if file_name.endswith(".npz"):
            file_path = os.path.join(input_dir, file_name)
            data = np.load(file_path)

            V = data["V"]  # 顶点
            F = data["F"]  # 面
            vertex_labels = data["labels"]  # 每个顶点的标签

            # 生成 .obj 文件
            obj_lines = []
            for v in V:
                obj_lines.append(f"v {v[0]} {v[1]} {v[2]}")
            for f in F:
                obj_lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}")  # OBJ 是从1开始编号的

            obj_filename = os.path.splitext(file_name)[0] + ".obj"
            with open(os.path.join(obj_output_dir, obj_filename), "w") as f_obj:
                f_obj.write("\n".join(obj_lines))

            # 生成面标签
            face_labels = []
            for face in F:
                face_vertex_labels = vertex_labels[face]
                values, counts = np.unique(face_vertex_labels, return_counts=True)
                if len(counts) == 1 or np.max(counts) == 2: 
                    label = values[np.argmax(counts)]
                else:
                    label = 0  # 平局时为0
                face_labels.append(str(label))

            txt_filename = os.path.splitext(file_name)[0] + ".txt"
            with open(os.path.join(label_output_dir, txt_filename), "w") as f_txt:
                f_txt.write("\n".join(face_labels))


if __name__ == "__main__":
    process_npz_files(input_dir, obj_output_dir, label_output_dir)

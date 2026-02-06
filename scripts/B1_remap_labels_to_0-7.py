import os
from tqdm import tqdm

labels_dir = "./output/1280normals/labels"


def remap_yolo_segmentation_labels(file_path):
    def map_label(label):
        label = int(label)
        if 11 <= label <= 18:
            return str(label - 11)
        elif 21 <= label <= 28:
            return str(label - 21)
        elif 31 <= label <= 38:
            return str(label - 31)
        elif 41 <= label <= 48:
            return str(label - 41)
        else:
            return str(label)

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        parts[0] = map_label(parts[0])
        new_lines.append(' '.join(parts) + '\n')

    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)


if __name__ == "__main__":
    for filename in tqdm(os.listdir(labels_dir)):
        filepath = os.path.join(labels_dir, filename)
        remap_yolo_segmentation_labels(filepath)

"""
1.读取图片和对应的 GT、预测记录；
2.将 YOLO 格式的 detect 数据（归一化的边界点）解析成实际坐标；
3.可视化每个 sample，每张图分成 n 个子图，显示：
- gt（带 bbox + label）；
- 每个预测（带 bbox + label + confidence，按是否预测正确控制透明度）；
4.使用指定调色板（palette）着色。
"""

imgs_dir = R"E:\wyc\Data\teeth3ds+\new\upper_preprocessed\output_whole_mesh_16cls_det\images\test"
gt_dir = R"E:\wyc\Data\teeth3ds+\new\upper_preprocessed\output_whole_mesh_16cls_det\labels\test"
predicts_dirs = [
    R"E:\wyc\Data\teeth3ds+\YOLO训练结果\RTDETR\pretrain\predict\labels",
    R"E:\wyc\Data\teeth3ds+\YOLO训练结果\RTDETR\retrain_nq300\predict\labels",
    R"E:\wyc\Data\teeth3ds+\YOLO训练结果\RTDETR\retrain_nq16\predict\labels",
]
predicts_names = [
    "pretrain",
    "retrain np=300",
    "retrain np=16",
]
output_dir = "visualizations/RTDETR"

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from tqdm import tqdm


def map_class(cls):
    return cls + 1

def load_yolo_det(file_path, is_prediction=False):
    """
    YOLO detect txt: class cx cy w h [conf]
    均为归一化坐标，conf 仅预测有
    """
    instances = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = map_class(int(parts[0]))
            cx, cy, bw, bh = map(float, parts[1:5])
            confidence = float(parts[5]) if (is_prediction and len(parts) >= 6) else None
            instances.append({
                'class': cls,
                'bbox': np.array([cx, cy, bw, bh], dtype=np.float32),  # 归一化
                'confidence': confidence
            })
    return instances

def denormalize_bbox(bbox, img_width, img_height):
    cx, cy, bw, bh = bbox
    return np.array([cx * img_width, cy * img_height, bw * img_width, bh * img_height], dtype=np.float32)

def compute_iou_bbox(b1, b2):
    # b*: [cx, cy, w, h] in pixels
    x1_1, y1_1 = b1[0] - b1[2] / 2, b1[1] - b1[3] / 2
    x2_1, y2_1 = b1[0] + b1[2] / 2, b1[1] + b1[3] / 2
    x1_2, y1_2 = b2[0] - b2[2] / 2, b2[1] - b2[3] / 2
    x2_2, y2_2 = b2[0] + b2[2] / 2, b2[1] + b2[3] / 2

    inter_w = max(0.0, min(x2_1, x2_2) - max(x1_1, x1_2))
    inter_h = max(0.0, min(y2_1, y2_2) - max(y1_1, y1_2))
    inter = inter_w * inter_h
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def match_prediction_to_gt(pred_instance, gt_instances, iou_thresh=0.5):
    same_class_gts = [gt for gt in gt_instances if gt['class'] == pred_instance['class']]
    best_iou = 0
    best_gt = None

    for gt in gt_instances:
        iou = compute_iou_bbox(pred_instance['abs_bbox'], gt['abs_bbox'])
        if iou >= iou_thresh:
            gt['matched'] = True
            if gt['class'] == pred_instance['class']:
                return True
        
    return False


def visualize_sample(img_path, gt_path, pred_paths, palette, iou_thresh=0.5):
    image = cv2.imread(img_path)[..., ::-1]
    h, w = image.shape[:2]

    gt_instances = load_yolo_det(gt_path, is_prediction=False)
    pred_groups = [load_yolo_det(pred_path, is_prediction=True) for pred_path in pred_paths]

    dpi = 100
    fig_w = (w * (1 + len(pred_groups))) / dpi
    fig_h = h / dpi
    fig, axs = plt.subplots(1, 1 + len(pred_groups), figsize=(fig_w, fig_h), dpi=dpi)
    if len(pred_groups) == 1:
        axs = [axs[0], axs[1]]  # 转为列表形式

    # --- Ground Truth ---
    axs[0].imshow(image, alpha=1.0)
    # axs[0].set_xlim([0, w])
    # axs[0].set_ylim([h, 0])
    axs[0].set_title("GT")
    for inst in gt_instances:
        inst['abs_bbox'] = denormalize_bbox(inst['bbox'], w, h)
        cls = inst['class']
        cx, cy, bw, bh = inst['abs_bbox']
        x0, y0 = cx - bw / 2, cy - bh / 2
        axs[0].add_patch(Rectangle((x0, y0), bw, bh, facecolor=palette[cls], alpha=0.5))
        axs[0].text(cx, cy, f'{cls}', fontsize=6, color='white', bbox=dict(facecolor='black', alpha=0.5))

    # --- Predictions ---
    for i, pred_instances in enumerate(pred_groups):
        # clear gt_instances status
        for inst in gt_instances:
            inst['matched'] = False

        axs[i + 1].imshow(image, alpha=0.2)
        # axs[i + 1].set_xlim([0, w])
        # axs[i + 1].set_ylim([h, 0])
        axs[i + 1].set_title(predicts_names[i])
        for inst in pred_instances:
            inst['abs_bbox'] = denormalize_bbox(inst['bbox'], w, h)
            cls = inst['class']
            conf = inst['confidence']
            correct = match_prediction_to_gt(inst, gt_instances, iou_thresh=iou_thresh)

            cx, cy, bw, bh = inst['abs_bbox']
            x0, y0 = cx - bw / 2, cy - bh / 2
            alpha = 0.2 if correct else 1.0
            axs[i + 1].add_patch(
                Rectangle((x0, y0), bw, bh,
                        facecolor=palette[cls],
                        edgecolor=None if correct else 'r',
                        linewidth=3, alpha=alpha)
            )
            axs[i + 1].text(cx, cy, f'{cls} ({conf:.2f})', fontsize=6, color='white', bbox=dict(facecolor='black', alpha=0.3, pad=1, boxstyle='round,pad=0.2'))
        
        for gt in gt_instances:
            if not gt['matched']:
                cls_miss = gt['class']
                cx, cy, bw, bh = gt['abs_bbox']
                x0, y0 = cx - bw / 2, cy - bh / 2
                axs[i + 1].add_patch(
                    Rectangle((x0, y0), bw, bh,
                            facecolor=palette[cls_miss],
                            edgecolor='black', linewidth=9, alpha=1.0)
                )

    for ax in axs:
        ax.axis('off')

    output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(img_path))[0] + ".png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sample_name = os.path.splitext(os.path.basename(img_path))[0]
    fig.suptitle(sample_name, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':

    PALETTE = [
        [174, 199, 232],
        [152, 223, 138],
        [31, 119, 180],
        [255, 187, 120],
        [188, 189, 34],
        [140, 86, 75],
        [255, 152, 150],
        [197, 176, 213],
        [214, 39, 40],
        [148, 103, 189],
        [196, 156, 148],
        [23, 190, 207],
        [247, 182, 210],
        [219, 219, 141],
        [255, 127, 14],
        [158, 218, 229],
        [44, 160, 44],
    ]
    palette = np.asarray(PALETTE, dtype=float)
    palette /= 255.0

    for fname in tqdm(os.listdir(imgs_dir)[:]):
        if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        img_path = os.path.join(imgs_dir, fname)
        stem = os.path.splitext(fname)[0]
        gt_path = os.path.join(gt_dir, stem + '.txt')
        pred_paths = [os.path.join(p_dir, stem + '.txt') for p_dir in predicts_dirs]
        if not os.path.exists(gt_path) or not all(os.path.exists(p) for p in pred_paths):
            continue
        visualize_sample(img_path, gt_path, pred_paths, palette)

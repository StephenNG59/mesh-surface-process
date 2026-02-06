"""
1.读取图片和对应的 GT、预测记录；
2.将 YOLO 格式的 segmentation 数据（归一化的边界点）解析成实际坐标；
3.可视化每个 sample，每张图分成 n 个子图，显示：
- gt（带 mask + label）；
- 每个预测（带 mask + label + confidence，按是否预测正确控制透明度）；
4.使用指定调色板（palette）着色。
"""

imgs_dir = R"E:\wyc\Data\teeth3ds+\new\lower_preprocessed\output_whole_mesh_16classes\images\test"
gt_dir = R"E:\wyc\Data\teeth3ds+\new\lower_preprocessed\output_whole_mesh_16classes\labels\test"
predicts_dirs = [
    R"E:\wyc\Data\teeth3ds+\YOLO训练结果\MICCAI_16cls_lower_yolo11n\predict\labels",
    R"E:\wyc\Data\teeth3ds+\YOLO训练结果\MICCAI_16cls_lower_yolo11n_lossandassign\predict(agnostic)\labels",
    R"E:\wyc\Data\teeth3ds+\YOLO训练结果\MICCAI_16cls_lower_yolo11n_lossandassign\predict\labels",
    # R"E:\wyc\Data\teeth3ds+\YOLO训练结果\only_1st_detect_head\predict\labels",
    # R"E:\wyc\Data\teeth3ds+\YOLO训练结果\only_2nd_detect_head\predict\labels",
    # R"E:\wyc\Data\teeth3ds+\YOLO训练结果\only_3rd_detect_head\predict\labels"
]
predicts_names = [
    "w/o loss&assign",
    "w/ loss&assign(agnostic)",
    "w/ loss&assign",
    # "1st head",
    # "2nd head",
    # "3rd head"
]
output_dir = R"E:\wyc\Data\teeth3ds+\YOLO训练结果\visualizations\loss&assign"  # "visualizations/16cls"

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from tqdm import tqdm


def map_class(cls):
    return cls + 1

def load_yolo_seg(file_path, is_prediction=False):
    instances = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 7:  # 至少需要1类+3点（x,y）对
                continue
            cls = map_class(int(parts[0]))
            coords = np.array(parts[1:-1], dtype=np.float32) if is_prediction else np.array(parts[1:], dtype=np.float32)
            points = coords.reshape(-1, 2)
            confidence = float(parts[-1]) if is_prediction else None
            instances.append({
                'class': cls,
                'points': points,
                'confidence': confidence
            })
    return instances

def denormalize_points(points, img_width, img_height):
    return np.stack([
        points[:, 0] * img_width,
        points[:, 1] * img_height
    ], axis=-1)

def compute_iou(poly1_pts, poly2_pts, img_size=(1024, 1024), sample_stride=2):
    h, w = img_size
    mask1 = np.zeros((h, w), dtype=np.uint8)
    mask2 = np.zeros((h, w), dtype=np.uint8)

    def sparse_poly(pts, stride):
        if len(pts) <= stride:
            return pts
        return pts[::stride]

    poly1 = np.round(sparse_poly(poly1_pts, sample_stride)).astype(np.int32)
    poly2 = np.round(sparse_poly(poly2_pts, sample_stride)).astype(np.int32)

    if len(poly1) >= 3:
        cv2.fillPoly(mask1, [poly1], 1)
    if len(poly2) >= 3:
        cv2.fillPoly(mask2, [poly2], 1)

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    # print(f"iou={intersection}/{union} = {intersection/union}")
    return intersection / union if union > 0 else 0.0

def match_prediction_to_gt(pred_instance, gt_instances, iou_thresh=0.5, img_size=(1024, 1024)):
    same_class_gts = [gt for gt in gt_instances if gt['class'] == pred_instance['class']]
    best_iou = 0
    best_gt = None

    for gt in gt_instances:
        iou = compute_iou(pred_instance['abs_points'], gt['abs_points'], img_size)
        if iou >= iou_thresh:
            gt['matched'] = True
            if gt['class'] == pred_instance['class']:
                return True
        
    return False

    for gt in same_class_gts:
        iou = compute_iou(pred_instance['abs_points'], gt['abs_points'], img_size)
        if iou > best_iou:
            best_iou = iou
            best_gt = gt
    
    if best_iou >= iou_thresh and best_gt is not None:
        best_gt['matched'] = True
        return True
    return False

def visualize_sample(img_path, gt_path, pred_paths, palette, iou_thresh=0.5):
    image = cv2.imread(img_path)[..., ::-1]
    h, w = image.shape[:2]

    gt_instances = load_yolo_seg(gt_path, is_prediction=False)
    pred_groups = [load_yolo_seg(pred_path, is_prediction=True) for pred_path in pred_paths]

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
        inst['abs_points'] = denormalize_points(inst['points'], w, h)
        cls = inst['class']
        # poly = Polygon(inst['abs_points'], True)
        axs[0].add_patch(Polygon(inst['abs_points'], closed=True, facecolor=palette[cls], alpha=0.5))
        text_pos = inst['abs_points'].mean(axis=0)
        axs[0].text(text_pos[0], text_pos[1], f'{cls}', fontsize=6, color='white', bbox=dict(facecolor='black', alpha=0.5))

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
            inst['abs_points'] = denormalize_points(inst['points'], w, h)
            cls = inst['class']
            conf = inst['confidence']
            correct = match_prediction_to_gt(inst, gt_instances, iou_thresh=iou_thresh, img_size=(h, w))

            # poly = Polygon(inst['abs_points'], True)
            alpha = 0.2 if correct else 1.0
            axs[i + 1].add_patch(Polygon(inst['abs_points'], closed=True, facecolor=palette[cls], edgecolor=None if correct else 'r', alpha=alpha, linewidth=3))
            label_text = f'{cls} ({conf:.2f})'
            text_pos = inst['abs_points'].mean(axis=0)
            axs[i + 1].text(text_pos[0], text_pos[1], label_text, fontsize=6, color='white', bbox=dict(facecolor='black', alpha=0.3, pad=1, boxstyle='round,pad=0.2'))
        
        for gt in gt_instances:
            if not gt['matched']:
                miss_cls = gt['class']
                axs[i+1].add_patch(Polygon(gt['abs_points'], closed=True, facecolor=palette[miss_cls], edgecolor='black', alpha=1.0, linewidth=9))

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

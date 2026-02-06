# 1. 在.h5文件中记录的GT labels
# 2. 在.txt文件中记录的pred labels
# 计算1、2的metrics

import os
import numpy as np
import h5py
import json
from collections import defaultdict
from tqdm import tqdm
import glob

single_cls = False
num_classes = 9
# pred_dir = R"E:\wyc\Projects\Detect&Teethgnn\teethgnn\runs\MICCAI\plain_gcn_batch\Log_2025-06-17_14-52-44\results\predictions"
# gt_dir = R"E:\wyc\Data\teethgnn_data\2_features_o3d_simplified_10000"
# pred_dir = R"E:\wyc\Projects\Detect&Teethgnn\teethgnn\runs\MICCAI\plain_gcn_batch\Log_2025-06-17_14-52-44\results\predictions_map2origin"
gt_dir = R"E:\wyc\Data\teeth3ds+\new\upper_preprocessed\npz"
pred_dir = R"E:\wyc\Data\teeth3ds+\YOLO训练结果\MICCAI_ARAP_normal_8cls_100ep\pred_labels"

# =================================================================
# ============ metrics code in MMDetection3D ======================
def map_gt_labels(gt_labels):
    # 创建一个映射字典
    label_mapping = {}

    # 将11~18映射到1~8
    for i in range(11, 19):
        label_mapping[i] = i - 10

    # 将21~28映射到1~8
    for i in range(21, 29):
        label_mapping[i] = i - 20

    # 将0映射为0
    label_mapping[0] = 0

    # 对每个GT标签应用映射
    mapped_labels = [label_mapping.get(label, label) for label in gt_labels]
    
    return np.array(mapped_labels)

def read_pred_labels(txt_file):
    with open(txt_file) as f:
        lines = f.readlines()
    data = np.array([line.strip() for line in lines])
    return data

def read_gt_labels(gt_file):
    ext = os.path.splitext(gt_file)[1]
    if ext == '.h5':
        with h5py.File(gt_file, 'r') as f:
            data = f['labels'][:]
    elif ext == '.npz':
        data = np.load(gt_file)['labels']
        data = map_gt_labels(data)
    return data

def fast_hist(preds, labels, num_classes):
    """Compute the confusion matrix for every batch.

    Args:
        preds (np.ndarray):  Prediction labels of points with shape of
        (num_points, ).
        labels (np.ndarray): Ground truth labels of points with shape of
        (num_points, ).
        num_classes (int): number of classes

    Returns:
        np.ndarray: Calculated confusion matrix.
    """
    k = (labels >= 0) & (labels < num_classes)
    bin_count = np.bincount(
        num_classes * labels[k].astype(int) + preds[k],
        minlength=num_classes**2)
    return bin_count[:num_classes**2].reshape(num_classes, num_classes)

def per_class_iou(hist):
    """Compute the per class iou.

    Args:
        hist(np.ndarray):  Overall confusion martix
        (num_classes, num_classes ).

    Returns:
        np.ndarray: Calculated per class iou
    """

    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def get_acc(hist):
    """Compute the overall accuracy.

    Args:
        hist(np.ndarray):  Overall confusion martix
        (num_classes, num_classes ).

    Returns:
        float: Calculated overall acc
    """

    return np.diag(hist).sum() / hist.sum()

def get_acc_cls(hist):
    """Compute the class average accuracy.

    Args:
        hist(np.ndarray):  Overall confusion martix
        (num_classes, num_classes ).

    Returns:
        float: Calculated class average acc
    """

    return np.nanmean(np.diag(hist) / hist.sum(axis=1))

def seg_eval(gt_labels, seg_preds, case_names):
    """Semantic Segmentation  Evaluation.

    Evaluate the result of the Semantic Segmentation.

    Args:
        gt_labels (list[torch.Tensor]): Ground truth labels.
        seg_preds  (list[torch.Tensor]): Predictions.
        label2cat (dict): Map from label to category name.
        ignore_index (int): Index that will be ignored in evaluation.
        logger (logging.Logger | str, optional): The way to print the mAP
            summary. See `mmdet.utils.print_log()` for details. Default: None.

    Returns:
        dict[str, float]: Dict of results.
    """
    assert len(seg_preds) == len(gt_labels)

    hist_list = []
    for i in tqdm(range(len(gt_labels)), desc="Calculating metrics"):
        gt_seg = gt_labels[i].astype(np.int64)
        pred_seg = seg_preds[i].astype(np.int64)

        # calculate one instance result
        hist = fast_hist(pred_seg, gt_seg, num_classes)
        hist_list.append(hist)
        
        iou = per_class_iou(hist)
        miou = np.nanmean(iou)
        acc = get_acc(hist)
        if miou < 0.1:
            print(f"==={case_names[i]}")
            print("iou:" + ",".join([f"{i:.3f}" for i in iou]))
            print(f"miou:{miou:.4f} | acc:{acc:.4f}")

    iou = per_class_iou(sum(hist_list))
    miou = np.nanmean(iou)
    acc = get_acc(sum(hist_list))
    acc_cls = get_acc_cls(sum(hist_list))

    # used_classes = np.r_[0:2, 3:8]
    # sum_hist_list_partial = sum(hist_list)[np.ix_(used_classes, used_classes)]
    # iou_partial = per_class_iou(sum_hist_list_partial)
    # miou_partial = np.nanmean(iou_partial)
    # acc_partial = get_acc(sum_hist_list_partial)

    print("iou:" + ",".join([f"{i:.4f}" for i in iou]))
    print(f"miou:{miou:.4f}")
    print(f"acc:{acc:.4f}")
    print(f"acc_cls:{acc_cls:.4f}")
    # print(f"miou_without_278:{miou_partial:.4f}")
    # print(f"acc_without_278:{acc_partial:.4f}")

def main():
    gt_labels = []
    pred_labels = []
    case_names = []

    # load predicted and gt labels
    for pred_file in tqdm(os.listdir(pred_dir)[:], desc="Loading predictions and gts"):
        if pred_file.endswith('.txt'):
            # load predict label
            pred_filepath = os.path.join(pred_dir, pred_file)
            pred = read_pred_labels(pred_filepath)

            # load gt label
            case_name = os.path.splitext(os.path.basename(pred_file))[0]  # case_name: abcdefgh
            gt_files = glob.glob(os.path.join(gt_dir, case_name + '.*'))  # extension may be: '.h5' or '.npz'
            if not gt_files:
                raise FileNotFoundError(f"No GT file found for case: {case_name}")
            elif len(gt_files) > 1:
                raise ValueError(f"Multiple GT files found for case: {case_name}: {gt_files}")
            gt_filepath = gt_files[0]
            gt = read_gt_labels(gt_filepath)

            # pad zeros to same size
            if len(pred) < len(gt):
                pred = np.pad(pred, (0, len(gt) - len(pred)), mode='constant', constant_values=0)

            # 如果只考虑单一类别的预测
            if single_cls:
                pred = np.where(pred >= 1, 1, 0)
                gt = np.where(gt >= 1, 1, 0)
                num_classes = 2

            pred_labels.append(pred)
            case_names.append(case_name)
            gt_labels.append(gt)
    
    seg_eval(gt_labels, pred_labels, case_names)


if __name__ == "__main__":
    main()

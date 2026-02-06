import os
import math
import numpy as np
import json
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score
import scipy.spatial.distance as compute_dist_matrix
from scipy.optimize import linear_sum_assignment

single_cls = False # True, False
num_classes = 9 # 17, 9, 2
# pred_dir = R"E:\wyc\Data\teeth3ds+\new\640NKM\test_set\final_predictions\640px,bnd_size20,bnd_wt10,non_bnd_wt1,bs4,s1.0,cls1,cha7"

pred_dir = R"E:\wyc\Data\teeth3ds+\new\single_tooth_meshes\upper\test(8cls)\final_prediction_output(teegraph_true)"
# pred_dir = R"E:\wyc\Data\teeth3ds+\new\single_tooth_meshes\upper\test(16cls)\final_prediction_output"
gt_dir = R"E:\wyc\Data\teeth3ds+\new\upper_preprocessed\npz(ARAP_UV)"
# pred_dir = R"E:\wyc\Data\teeth3ds+\new\single_tooth_meshes\lower\test\final_prediction_output(teegraph_lower)"
# gt_dir = R"E:\wyc\Data\teeth3ds+\new\lower_preprocessed\npz(arap)"

# =================================================================
# ============ metrics code in MMDetection3D ======================
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
# ==================================================================

def read_gt_labels_V(npz_file):
    data = np.load(npz_file)
    return data['labels'], data['V']

def map_pred_labels(pred_labels):
    return pred_labels + 1   # -1 => 0, [0~7] => [1~8]

def map_gt_labels(gt_labels):
    # 创建一个映射字典
    label_mapping = {}

    if num_classes == 9 or num_classes == 2:
        # 将11~18映射到1~8
        for i in range(11, 19):
            label_mapping[i] = i - 10
        # 将21~28映射到1~8
        for i in range(21, 29):
            label_mapping[i] = i - 20
        for i in range(31, 39):
            label_mapping[i] = i - 30
        for i in range(41, 49):
            label_mapping[i] = i - 40

    elif num_classes == 17:
        for i in range(11, 19):
            label_mapping[i] = i - 10
        # 将21~28映射到9~16
        for i in range(21, 29):
            label_mapping[i] = i - 12
        for i in range(31, 39):
            label_mapping[i] = i - 30
        # 将41~48映射到9~16
        for i in range(41, 49):
            label_mapping[i] = i - 32

    # 将0映射为0
    label_mapping[0] = 0

    # 对每个GT标签应用映射
    mapped_labels = [label_mapping.get(label, label) for label in gt_labels]
    
    return np.array(mapped_labels)

def read_pred_labels_instances_confidence(npy_file):
    # labels, instances, confidences
    data = np.load(npy_file)
    return data[:, 0], data[:, 1], data[:, 2]

def seg_eval(gt_labels_list, seg_preds_list, case_names):
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
    assert len(seg_preds_list) == len(gt_labels_list)

    hist_list = []
    gts_np = []
    preds_np = []
    for i in tqdm(range(len(gt_labels_list)), desc="Calculating metrics"):
        gt_seg = gt_labels_list[i].astype(np.int64)
        pred_seg = seg_preds_list[i].astype(np.int64)
        gts_np.extend(gt_seg)
        preds_np.extend(pred_seg)

        # calculate one instance result
        hist = fast_hist(pred_seg, gt_seg, num_classes)
        hist_list.append(hist)
        
        iou = per_class_iou(hist)
        miou = np.nanmean(iou)
        acc = get_acc(hist)
        if miou < 0.1:
            print(f"==={case_names[i]}")
            print("iou:" + ",".join([f"{i:.4f}" for i in iou]))
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

    print(classification_report(gts_np, preds_np, digits=4))

# ==================================================================
# =============== metrics in MICCAI Challenges =====================
# https://github.com/abenhamadou/3DTeethSeg_MICCAI_Challenges/blob/main/evaluation/evaluation.py
def calc_gt_instances(gt_labels):
    """
    输入:
        gt_labels: numpy 数组，形状为 (n,)，元素在 0 或者 [11, 48] 之间，代表每个顶点的标签
    输出:
        gt_instances: numpy 数组，形状为 (n,)，元素在 [0, num_instances]，
        其中 label 为 0 的对应 instance 也为 0，其余从 1 开始编号
    """
    gt_instances = np.zeros_like(gt_labels)
    valid_mask = gt_labels != 0
    valid_labels = np.unique(gt_labels[valid_mask])

    label_to_instance = {label: i+1 for i, label in enumerate(valid_labels)}
    gt_instances[valid_mask] = np.vectorize(label_to_instance.get)(gt_labels[valid_mask])

    return gt_instances

def compute_tooth_size(points, centroid):
    size = np.sqrt(np.sum((centroid - points) ** 2, axis=0))
    return size

def centroids_pred_to_gt_attribution(gt_instance_label_dict, pred_instance_label_dict):
    """把gt和pred的instances根据centroids来进行一一配对"""

    gt_centroids_list = []
    for k, v in gt_instance_label_dict.items():
        gt_centroids_list.append((v["centroid"]))

    pred_centroids_list = []
    for k, v in pred_instance_label_dict.items():
        pred_centroids_list.append((v["centroid"]))
    
    M = compute_dist_matrix.cdist(gt_centroids_list, pred_centroids_list)
    row_index, col_index = linear_sum_assignment(M)

    matching_dict = {
        list(gt_instance_label_dict.keys())[i]: list(pred_instance_label_dict.keys())[j] 
        for i, j in zip(row_index, col_index)}
    
    return matching_dict

def calculate_jaw_TLA(gt_instance_label_dict, pred_instance_label_dict, matching_dict):

    """
    Teeth localization accuracy (TLA): mean of normalized Euclidean distance between ground truth (GT) teeth centroids and the closest localized teeth
    centroid. Each computed Euclidean distance is normalized by the size of the corresponding GT tooth.
    In case of no centroid (e.g. algorithm crashes or missing output for a given scan) a nominal penalty of 5 per GT
    tooth will be given. This corresponds to a distance 5 times the actual GT tooth size. As the number of teeth per
    patient may be variable, here the mean is computed over all gathered GT Teeth in the two testing sets.
    Parameters
    ----------
    matching_dict
    gt_instance_label_dict
    pred_instance_label_dict

    Returns
    -------
    """
    TLA = 0
    for inst, info in gt_instance_label_dict.items():
        if inst in matching_dict.keys():

            TLA += np.linalg.norm((gt_instance_label_dict[inst]['centroid'] - pred_instance_label_dict[matching_dict
            [inst]]['centroid']) / gt_instance_label_dict[inst]['tooth_size'])
        else:
            TLA += 5 # * np.linalg.norm(gt_instance_label_dict[inst]['tooth_size'])

    return TLA/len(gt_instance_label_dict.keys())

def calculate_jaw_TSA(gt_instances, pred_instances):
    """
    Teeth segmentation accuracy (TSA): is computed as the average F1-score over all instances of teeth point clouds.
    The F1-score of each tooth instance is measured as: F1=2*(precision * recall)/(precision+recall)

    Returns F1-score per jaw
    -------

    """
    gt_instances[gt_instances != 0] = 1
    pred_instances[pred_instances != 0] = 1
    return f1_score(gt_instances, pred_instances, average='micro')

def calculate_jaw_TIR(gt_instance_label_dict, pred_instance_label_dict, matching_dict, threshold=0.5):
    """
    Teeth identification rate (TIR): is computed as the percentage of true identification cases relatively to all GT
    teeth in the two testing sets. A true identification is considered when for a given GT Tooth,
    the closest detected tooth centroid : is localized at a distance under half of the GT tooth size, and is
    attributed the same label as the GT tooth
    Returns
    -------

    """
    tir = 0
    # print(f"matching_dict len={len(matching_dict)}")
    for gt_inst, pred_inst in matching_dict.items():
        dist = np.linalg.norm((gt_instance_label_dict[gt_inst]["centroid"]-pred_instance_label_dict[pred_inst]["centroid"])
                         /gt_instance_label_dict[gt_inst]['tooth_size'])
        if dist < threshold and gt_instance_label_dict[gt_inst]["label"]==pred_instance_label_dict[pred_inst]["label"]:
            tir += 1
            # wyc test
            # print(f"gt_instance_label_dict[{gt_inst}]['label']={gt_instance_label_dict[gt_inst]['label']}, pred_instance_label_dict[{pred_inst}]['label']={pred_instance_label_dict[pred_inst]['label']}, dist={dist}")
    return tir/len(matching_dict)

def MICCAImetrics_eval(gt_labels_list, gt_instances_list, pred_labels_list, pred_instances_list, V_list, case_names):
    assert len(gt_labels_list) == len(pred_labels_list) == len(case_names)
    tla_list = []
    tsa_list = []
    tir_list = []

    for i in tqdm(range(len(gt_labels_list)), desc="Calculating MICCAI metrics"):
        gt_labels = gt_labels_list[i]
        gt_instances = gt_instances_list[i]
        u_instances = np.unique(gt_instances)
        u_instances = u_instances[u_instances != 0]

        pred_labels = pred_labels_list[i]
        pred_instances = pred_instances_list[i]
        u_pred_instances = np.unique(pred_instances)
        # delete 0 instance
        u_pred_instances = u_pred_instances[u_pred_instances != 0]
        pred_instance_label_dict = {}

        # === Predicted ===
        # check if one instance match exactly one label else this instance(label) will be attributed to gingiva 0
        for pred_inst in u_pred_instances:
            pred_label_inst = pred_labels[pred_instances == pred_inst]
            nb_predicted_labels_per_inst = np.unique(pred_label_inst)
            if len(nb_predicted_labels_per_inst) == 1:
                # compute predicted tooth center
                V = V_list[i]
                pred_verts = V[pred_instances == pred_inst]
                pred_center = np.mean(pred_verts, axis=0)
                pred_instance_label_dict[str(pred_inst)] = {
                    "label": pred_label_inst[0], 
                    "centroid": pred_center
                }
            
            else:
                pred_labels[pred_instances == pred_inst] = 0
                pred_instances[pred_instances == pred_inst] = 0
        
        # === GT ===
        gt_instance_label_dict = {}
        for gt_inst in u_instances:
            gt_lbl = gt_labels[gt_instances == gt_inst]
            label = np.unique(gt_lbl)

            assert len(label) == 1

            # compute gt tooth center and size
            gt_verts = V[gt_instances == gt_inst]
            gt_center = np.mean(gt_verts, axis=0)
            tooth_size = compute_tooth_size(gt_verts, gt_center)
            gt_instance_label_dict[str(gt_inst)] = {
                "label": label[0], 
                "centroid": gt_center, 
                "tooth_size": tooth_size
            }
        
        # Matching predicted and GT
        matching_dict = centroids_pred_to_gt_attribution(gt_instance_label_dict, pred_instance_label_dict)

        # == TLA
        tla = calculate_jaw_TLA(gt_instance_label_dict, pred_instance_label_dict, matching_dict)
        tla_list.append(math.exp(-tla))

        # == TSA
        tsa = calculate_jaw_TSA(gt_instances, pred_instances)
        tsa_list.append(tsa)

        # == TIR
        tir = calculate_jaw_TIR(gt_instance_label_dict, pred_instance_label_dict, matching_dict)
        tir_list.append(tir)

    print("TLA : {} +- {}".format(np.mean(tla_list), np.std(tla_list)))
    print("TSA : {} +- {}".format(np.mean(tsa_list), np.std(tsa_list)))
    print("TIR : {} +- {}".format(np.mean(tir_list), np.std(tir_list)))



if __name__ == "__main__":
    gt_labels_list = []
    pred_labels_list = []
    gt_instances_list = []
    pred_instances_list = []
    case_names = []
    V_list = []

    # load predicted and gt labels
    for pred_file in tqdm(os.listdir(pred_dir)[:], desc="Loading predictions and gts"):
        if pred_file.endswith('.npy'):
            # load predict label
            pred_filepath = os.path.join(pred_dir, pred_file)
            pred_labels, pred_instances, _ = read_pred_labels_instances_confidence(pred_filepath)
            pred_labels = map_pred_labels(pred_labels)
            if pred_instances.min() == -1:  # if gingiva instance==-1
                pred_instances += 1         # gingiva: instance==0

            # load gt label
            case_name = os.path.splitext(os.path.basename(pred_file))[0]  # case_name: abcdefgh
            gt_filepath = os.path.join(gt_dir, case_name + '.npz')
            gt_labels, V = read_gt_labels_V(gt_filepath)
            gt_instances = calc_gt_instances(gt_labels)
            gt_labels = map_gt_labels(gt_labels)

            # 如果只考虑单一类别的预测
            if single_cls:
                pred_labels = np.where(pred_labels >= 1, 1, 0)
                gt_labels = np.where(gt_labels >= 1, 1, 0)
                num_classes = 2

            gt_labels_list.append(gt_labels)
            pred_labels_list.append(pred_labels)
            gt_instances_list.append(gt_instances)
            pred_instances_list.append(pred_instances)
            case_names.append(case_name)
            V_list.append(V)
    
    seg_eval(gt_labels_list, pred_labels_list, case_names)
    MICCAImetrics_eval(gt_labels_list, gt_instances_list, pred_labels_list, pred_instances_list, V_list, case_names)

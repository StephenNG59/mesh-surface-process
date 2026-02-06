import os
import cv2
import numpy as np
import shutil
import argparse
from tqdm import tqdm

# labels range:
# - upper: list(range(11, 19)) + list(range(21, 29))
# - lower: list(range(31, 39)) + list(range(41, 49))
tooth_labels_range = list(range(11, 19)) + list(range(21, 29))

# example usage: 
#  python .\A5+_YOLOPoseAnnGenerator.py .\upper_preprocessed\imgs_whole_mesh\ .\upper_preprocessed\output_whole_mesh_pose

def findSrcPairedSamples(src_dir):
    img_names = set()
    mask_names = set()

    for file_name in os.listdir(src_dir):
        if file_name.endswith('.mask.png'):
            mask_names.add(file_name[:-9])
        elif file_name.endswith('.png'):
            img_names.add(file_name[:-4])
    
    paired_names = img_names & mask_names
    return paired_names

def get_contours_from_mask(mask_img):
    # 获取物体的轮廓，对于每个物体类别（排除背景），提取其轮廓
    unique_labels = np.unique(mask_img)
    unique_labels = unique_labels[(unique_labels != 0)]  # 排除背景0
    unique_labels = unique_labels[(unique_labels != 255)]  # 排除背景255
    contours = []
    
    for label in unique_labels:
        # 为每个物体类别创建一个二进制mask
        binary_mask = np.uint8(mask_img == label)
        # 查找轮廓
        cnts, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 如果一个label有多个contours，多半是有问题的，暂时只采用面积最大的那个
        if len(cnts) > 1:
            print(f"Warning: {len(cnts)} contours found for label {label}, using the largest one.")
            cnts = sorted(cnts, key=lambda x: cv2.contourArea(x), reverse=True)[:1]
        
        for cnt in cnts:
            contours.append((label, cnt.reshape(-1, 2)))  # 保存物体标签和对应的轮廓
    
    return contours

def process_image_and_mask(image_path, mask_path, label_dir, image_dir):
    # 读取图片和mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    image_height, image_width = image.shape[:2]
    
    # 获取所有物体的轮廓
    contours = get_contours_from_mask(mask)
    if len(contours) == 0:
        print(f"None contours found in {mask_path}, ignoring this one.")
        return
    
    # 创建目标路径
    image_name = os.path.basename(image_path)
    label_name = image_name.replace('.png', '.txt')
    
    # 保存标签文件
    label_file_path = os.path.join(label_dir, label_name)
    with open(label_file_path, 'w') as label_file:
        keypoints = []
        class_id = 0  # 假设唯一类别为“牙齿”，class_index 为 0

        label_to_centroid = {label: polygon.mean(axis=0) for label, polygon in contours}

        for tooth_label in tooth_labels_range:
            if tooth_label in label_to_centroid:
                cx, cy = label_to_centroid[tooth_label]
                visibility = 2  # 可见
            else:
                cx, cy, visibility = 0, 0, 0  # 不存在
            keypoints.extend([cx / image_width, cy / image_height, visibility])

        # 计算整体 bbox（包围所有牙齿轮廓）
        all_points = np.concatenate([polygon for _, polygon in contours], axis=0)
        x_min, y_min = np.min(all_points, axis=0)
        x_max, y_max = np.max(all_points, axis=0)
        cx = (x_min + x_max) / 2 / image_width
        cy = (y_min + y_max) / 2 / image_height
        w = (x_max - x_min) / image_width
        h = (y_max - y_min) / image_height

        keypoint_str = ' '.join([f"{kp:.6f}" for kp in keypoints])
        label_file.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {keypoint_str}\n")
    
    # 复制图像到新目录
    image_save_path = os.path.join(image_dir, image_name)
    shutil.copy2(image_path, image_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate YOLO-pose anno .txt files according to image and mask.")
    parser.add_argument("src_dir", help="Src dir containing image and mask.")
    parser.add_argument("dest_dir", help="Dest dir to output annnotation files.")
    args = parser.parse_args()

    # Create dest dir
    labels_dir = os.path.join(args.dest_dir, "labels")
    images_dir = os.path.join(args.dest_dir, "images")
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # Find src samples
    sample_names = list(findSrcPairedSamples(args.src_dir))

    for sample in tqdm(sample_names[:]):
        # print(f"==Processing {sample}..")
        process_image_and_mask(
            image_path=os.path.join(args.src_dir, sample+".png"),
            mask_path=os.path.join(args.src_dir, sample+".mask.png"),
            label_dir=labels_dir,
            image_dir=images_dir
        )

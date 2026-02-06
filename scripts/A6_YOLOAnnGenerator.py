import os
import cv2
import numpy as np
import shutil
import argparse
from tqdm import tqdm

# === 在 src_dir 中，对所有 [*.png, *.mask.png] 对，
# === 生成 YOLO 分割格式的注释 .txt 文件
# === 保存在 dest_dir 中
feat_ext = ".png"
# test_split = "../testing_upper.txt"


def findSrcPairedSamples(src_dir):
    img_names = set()
    mask_names = set()

    for file_name in os.listdir(src_dir):
        if file_name.endswith('.mask.png'):
            mask_names.add(file_name[:-9])
        elif file_name.endswith(feat_ext):
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

def convert_polygon_to_yolo_format(image_width, image_height, polygon):
    # 将每个轮廓转化为YOLO的Segmentation格式

    # 将多边形的坐标归一化到图像尺寸
    normalized_polygon = []
    for point in polygon:
        x_norm = point[0] / image_width
        y_norm = point[1] / image_height
        normalized_polygon.append(f"{x_norm:.6f} {y_norm:.6f}")
    
    return " ".join(normalized_polygon)

def process_image_and_mask(image_path, mask_path, label_dir, image_dir):
    # 读取图片和mask
    image = cv2.imread(image_path) if feat_ext == ".png" else np.load(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    image_height, image_width = image.shape[:2]
    
    # 获取所有物体的轮廓
    contours = get_contours_from_mask(mask)
    
    # 创建目标路径
    image_name = os.path.basename(image_path)
    label_name = image_name.replace(feat_ext, '.txt')
    
    # 保存标签文件
    label_file_path = os.path.join(label_dir, label_name)
    with open(label_file_path, 'w') as label_file:
        for label, polygon in contours:
            # 转换为YOLO格式
            yolo_format = convert_polygon_to_yolo_format(image_width, image_height, polygon)
            # 写入标签
            label_file.write(f"{label} {yolo_format}\n")
    
    # 复制图像到新目录
    image_save_path = os.path.join(image_dir, image_name)
    shutil.copy2(image_path, image_save_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate YOLO-seg anno .txt files according to image and mask.")
    parser.add_argument("src_dir", help="Src dir containing image and mask.")
    parser.add_argument("dest_dir", help="Dest dir to output annnotation files.")
    parser.add_argument("--npy", action='store_true', help="Use .npy file to store features instead of .png.")
    # parser.add_argument("--test_split", help="Only generate test set.")
    # parser.add_argument("")
    args = parser.parse_args()

    if args.npy:
        feat_ext = ".npy"

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
            image_path=os.path.join(args.src_dir, sample+feat_ext),
            mask_path=os.path.join(args.src_dir, sample+".mask.png"),
            label_dir=labels_dir,
            image_dir=images_dir
        )



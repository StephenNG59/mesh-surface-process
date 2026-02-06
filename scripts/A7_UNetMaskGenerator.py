import numpy as np
from PIL import Image
import os
from glob import glob
from tqdm import tqdm

input_mask_dir = "./imgs"
output_npy_dir = "./output/masks"


def mask_to_npy(png_path, npy_path):
    # Step 1: 读取单通道PNG图像
    img = Image.open(png_path)  # .convert('L')  # 'L'模式会将图像转为灰度图（单通道）
    img_array = np.array(img)

    # Step 2: 将值为 0 或 255 的像素设为 0（非实例）
    # 其他值将作为实例标签，保留原始值（通常是从1开始的标签）
    mask = np.where((img_array == 0) | (img_array == 255), 0, img_array)

    # Step 3: 保存为.npy文件
    np.save(npy_path, mask)

def mask_to_clean_mask(input_path, output_path):
    img = Image.open(input_path)
    img_array = np.array(img)
    
    # 把mask中255改为0
    img_array[img_array == 255] = 0
    img_array[img_array != 0] = 1

    clean_img = Image.fromarray(img_array)

    clean_img.save(output_path)


def main():
    os.makedirs(output_npy_dir, exist_ok=True)
    
    mask_fps = glob(f"{input_mask_dir}/*.mask.png")

    for mask_fp in tqdm(mask_fps[100:200]):
        dest_fn = os.path.basename(mask_fp)#.replace(".mask.png", ".npy")
        dest_fp = f"{output_npy_dir}/{dest_fn}"
        # png_to_npy(mask_fp, dest_fp)
        mask_to_clean_mask(mask_fp, dest_fp)


if __name__ == "__main__":
    main()

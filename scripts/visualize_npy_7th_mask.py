# .npy中第7维的mask(通常由YOLO预测)，保存为图片

import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# 输入和输出目录（你可以自行修改）
input_dir = R'E:\wyc\Data\teeth3ds+\new\640NKM\7d_imgs'       # 这里填你的 .npy 文件目录
output_dir = R'E:\wyc\Data\teeth3ds+\new\640NKM\pred_masks(stage1)'  # 这里填输出图像目录


if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)

    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith('.npy'):
            filepath = os.path.join(input_dir, filename)
            array = np.load(filepath)
            
            # 提取 mask
            mask = array[:, :, 6]
            
            # 确保是二值图像（例如：值为0或1）
            mask_img = (mask > 0).astype(np.uint8) * 255  # 转为 0 和 255

            # 保存为 PNG 图片
            img = Image.fromarray(mask_img)
            out_path = os.path.join(output_dir, filename.replace('.npy', '.png'))
            img.save(out_path)

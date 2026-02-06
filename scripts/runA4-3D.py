import subprocess
import glob
import os
from tqdm import tqdm

program_path = "./A5_SingleToothExtractor.exe"
npz_dir = "./upper_preprocessed/npz(ARAP_UV)" # "./lower_preprocessed/npz(arap)/" 
output_dir = "./single_tooth_meshes/upper/test(16cls)"
clamp_lower = "0.005"
clamp_upper = "0.995"
log_level = "warning" # trace, debug, info, warning, error
expand_scale = "1.1"
split_txt = "../testing_upper.txt" #"../testing_upper.txt" # None, "../testing_upper.txt", "../training_upper.txt"
pred_labels_dir =  R"E:\wyc\Data\teeth3ds+\YOLO训练结果(upper)\MICCAI_16cls\predict\labels" # R"E:\wyc\Data\teeth3ds+\YOLO训练结果\MICCAI_16cls_lower_yolo11n\predict(train)\labels"
remove_ratio = "0" # 0.8 for train, 0 for test


files = glob.glob(f"{npz_dir}/*.npz")
if not files:
    print("No .npz files found!")
    exit(1)

if split_txt:
    # If only test split
    with open(split_txt) as f:
        split_samples = f.readlines()
    split_samples = [sam.strip() for sam in split_samples]
    files = [file for file in files if os.path.splitext(os.path.basename(file))[0] in split_samples]


for npz in tqdm(files[:]):
    sample_name = os.path.splitext(os.path.basename(npz))[0]
    pred_label_fp = os.path.join(pred_labels_dir, sample_name + '.txt')

    cmd = [program_path, 
           "-i", npz, 
           "-o", output_dir, 
           "--lower", clamp_lower, 
           "--upper", clamp_upper,
           "--remove-ratio", remove_ratio,
           "--expand-scale", expand_scale,
           "-l", log_level,
           "--pred-result", pred_label_fp]
    # print(" ".join(cmd))
    subprocess.run(cmd)

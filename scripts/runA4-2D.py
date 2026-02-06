import subprocess
import glob
import os

program_path = "./A4_ImageExtractor.exe"
npz_dir = "./lower_preprocessed/npz(arap)/" # "./upper_preprocessed/npz/" 
output_dir = "./imgs(K3)"
attribute = "K3"  # N,K1,K2,K3,NK,NKM
clamp_lower = "0.005"
clamp_upper = "0.995"
width = "640"
height = "640"
log_level = "info" # debug, warning, error, trace
per_instance = False # False, True
remove_ratio = "0" # 0.8, 0
random_jitter = False
# 
with_pred_masks = False # False, True
expand_scale = "1.1"
split_txt = "../testing_lower.txt" # None, "../testing_upper.txt", "../training_upper.txt"
pred_labels_dir = R"E:\wyc\Data\teeth3ds+\YOLO训练结果\MICCAI_ARAP_normal_8cls_100ep\labels\test"


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


for npz in files[:]:
    sample_name = os.path.splitext(os.path.basename(npz))[0]
    pred_label_fp = os.path.join(pred_labels_dir, sample_name + '.txt')

    cmd = [program_path, 
           "-i", npz, 
           "-o", output_dir, 
           "-a", attribute,
           "--lower", clamp_lower, 
           "--upper", clamp_upper,
           "-w", width,
           "-h", height,
           "--remove-ratio", remove_ratio,
           "--expand-scale", expand_scale,
           "-l", log_level]
    if per_instance:
        cmd.append("--instance")
    if random_jitter:
        cmd.append("--random-jitter")
    if with_pred_masks:
        cmd.append("--pred-result")
        cmd.append(pred_label_fp)
    # print(" ".join(cmd))
    subprocess.run(cmd)

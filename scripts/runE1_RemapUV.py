import os
import subprocess

program_path = R"E:\wyc\Projects\mesh-surface-process\build\bin\Release\E1_RemapUV.exe"
origin_npz_dir = "./upper_preprocessed/npz(ARAP_UV)"
new_npz_dir = "./upper_preprocessed/npz(Eigen_UV)"
regularize_coeff = "0"

issue_filenames = {'F892DPWZ_upper.npz', '725VOCK0_upper.npz', 'I13W39XQ_upper.npz', 'C4EMFL0D_upper.npz', 'S307GDC6_upper.npz', 'XTF24UY3_upper.npz', '87N5YSES_upper.npz', '4G9LHQ2X_upper.npz', 'ZCZHZ260_upper.npz', 'QOCAWJXM_upper.npz', '01909P9K_upper.npz', 'IFB97TD3_upper.npz', 'mccarthy_upper.npz', '8WZSZBYG_upper.npz', '01MGY4X8_upper.npz', '0U1LI1CB_upper.npz', 'GRG0112S_upper.npz', '8T3LV6TZ_upper.npz', 'KBGVMAF7_upper.npz', 'O9U99070_upper.npz'}

os.makedirs(new_npz_dir, exist_ok=True)

for i, filename in enumerate(os.listdir(origin_npz_dir)):
    if not filename.endswith('.npz'):
        continue
    if not filename in issue_filenames:
        continue

    print(f"=== [{i}] Processing {filename}")

    origin_npz_path = os.path.join(origin_npz_dir, filename)
    new_npz_path = os.path.join(new_npz_dir, filename)

    cmd = [
        program_path,
        "-i", origin_npz_path,
        "-o", new_npz_path,
        "-c", regularize_coeff,
        "-l", "trace"
    ]
    subprocess.run(cmd)


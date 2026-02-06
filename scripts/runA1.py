import os
import subprocess


program_path = "./A1_MeshCleaner.exe"
subdir = "./lower"  # "./upper", "./lower"

# issues_vdegree_0 = {'0U1LI1CB', '79R5A68H', '8WZSZBYG', 'AJP25A9G', 'F892DPWZ', 'I13W39XQ', 'LSMGKLAH'}

for case_name in os.listdir(subdir)[:]:
    # if case_name not in issues_vdegree_0:
    #     continue
    # if case_name not in {'01346914'}:
    #     continue

    case_path = os.path.join(subdir, case_name)
    if os.path.isdir(case_path):
        obj_file = None
        json_file = None
        for file in os.listdir(case_path):
            if file.endswith(".obj"):
                obj_file = os.path.join(case_path, file)
            elif file.endswith(".json"):
                json_file = os.path.join(case_path, file)

        if obj_file and json_file:
            cmd = [program_path, "-m", obj_file, "-a", json_file, "-o", "./temp", "-l", "warning"]
            print(" ".join(cmd))
            subprocess.run(cmd)
        else:
            print(f"Skipped {case_name}, missing .obj or .json")

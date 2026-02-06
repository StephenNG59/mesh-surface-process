import subprocess
import glob
import os

program = "" # "python"
script = "./A3_MeshMapper.exe" # "./A3_MeshMapper.py"

issues = {'015WW8D6_lower', '01AS003D_lower', '01M0RWA6_lower', '0EAKT1CU_lower', '0KIUZ4G5_lower', '270KABNB_lower', '4FEQPMVC_lower', '9U19TREK_lower', 'B7JSFJPG_lower', 'CNUR69O9_lower', 'CXAJM3O9_lower', 'ITTMBOHC_lower', 'IUIE4BYI_lower', 'J8YGBFK2_lower', 'KSHNN3DV_lower', 'LNQ2C7W2_lower', 'N8SOKA9R_lower', 'RMZC48A0_lower', 'SGIFTXFD_lower', 'XD63NNJX_lower', 'YCM36SR6_lower', }

files = glob.glob("./lower_temp/*.A2.m")
if not files:
    print("No .A2.m files found in ./temp/")
    exit(1)

for mesh_file in files[:]:
    if os.path.basename(mesh_file)[:-5] not in issues:
        continue
    cmd = [script, "-m", mesh_file, "-o", "./npz"]
    print(" ".join(cmd))
    subprocess.run(cmd)
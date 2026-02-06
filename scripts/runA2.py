import subprocess
import glob
import os

program_path = "./A2_MinCycles.exe"

# issues = {'KC1RZ7D9_upper', 'DVVHA02D_upper', 'T9NVJ8ZL_upper', 'O9U99070_upper', '8T3LV6TZ_upper', 'LSMGKLAH_upper', '0U1LI1CB_upper', 'AKHIE0CJ_upper', 'AJP25A9G_upper', 'XTF24UY3_upper', '8C8LPIIY_upper', 'F892DPWZ_upper', 'GRG0112S_upper', 'VS5EECWH_upper', '8WZSZBYG_upper', 'I13W39XQ_upper', '79R5A68H_upper'}
# issues_degree = {'0U1LI1CB_upper', '79R5A68H_upper', '8WZSZBYG_upper', 'AJP25A9G_upper', 'F892DPWZ_upper', 'I13W39XQ_upper', 'LSMGKLAH_upper'}
# issues = {'0U1LI1CB_upper', '8C8LPIIY_upper', '8T3LV6TZ_upper', '8WZSZBYG_upper', 'AKHIE0CJ_upper', 'DVVHA02D_upper', 'F892DPWZ_upper', 'GRG0112S_upper', 'I13W39XQ_upper', 'KC1RZ7D9_upper', 'O9U99070_upper', 'T9NVJ8ZL_upper', 'VS5EECWH_upper', 'XTF24UY3_upper'}
issues = {'015WW8D6_lower', '01AS003D_lower', '01M0RWA6_lower', '0EAKT1CU_lower', '0KIUZ4G5_lower', '270KABNB_lower', '4FEQPMVC_lower', '9U19TREK_lower', 'B7JSFJPG_lower', 'CNUR69O9_lower', 'CXAJM3O9_lower', 'ITTMBOHC_lower', 'IUIE4BYI_lower', 'J8YGBFK2_lower', 'KSHNN3DV_lower', 'LNQ2C7W2_lower', 'N8SOKA9R_lower', 'RMZC48A0_lower', 'SGIFTXFD_lower', 'XD63NNJX_lower', 'YCM36SR6_lower', }

files = glob.glob("./lower_temp/*.A1.m")
if not files:
    print("No .A1.m files found in ./temp/")
    exit(1)

for mesh_file in files[:]:
    if os.path.split(mesh_file)[1][:-5] not in issues:
        continue
    cmd = [program_path, mesh_file, "-output_file", mesh_file.replace(".A1.m", ".A2.m")]
    print("==========\n", " ".join(cmd))
    subprocess.run(cmd)

# [:400] 1 missing file
# [400:600] 10 missing files
# [600:700] 3 missing files
# [700:800] 1 missing files
# [800:] 2 missing files


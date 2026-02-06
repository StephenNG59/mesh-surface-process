import subprocess

program_file = "./A6_YOLOAnnGenerator.py"
src_dir = "./imgs" #"./upper_preprocessed/imgs_whole_mesh"
dest_dir = "./lower_preprocessed/output_whole_mesh_16classes" #"./upper_preprocessed/output_whole_mesh_16classes"
use_npy = False

cmd = ["python", program_file, src_dir, dest_dir]
if use_npy:
    cmd.append("--npy")
subprocess.run(cmd)

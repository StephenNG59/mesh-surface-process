import shutil
import os
from tqdm import tqdm

train_txt = "../training_upper.txt"
test_txt = "../testing_upper.txt"
dirs_to_split = [
    "./upper_preprocessed/output_whole_mesh_16classes/images", 
    "./upper_preprocessed/output_whole_mesh_16classes/labels", 
    # "./output/masks"
    ]


def read(txt):
    with open(txt) as f:
        data = f.readlines()
    if not data:
        print(f"No data in {txt}!")
    else:
        return [d.strip() for d in data]


def main():
    train_samples = read(train_txt)
    test_samples = read(test_txt)

    for dir_to_split in dirs_to_split:
        train_path = os.path.join(dir_to_split, "train")
        test_path = os.path.join(dir_to_split, "test")
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)

        for filename in tqdm(os.listdir(dir_to_split), desc=f"splitting {dir_to_split}"):
            if any(filename.startswith(sample) for sample in train_samples):
                source_path = os.path.join(dir_to_split, filename)
                target_path = os.path.join(train_path, filename)
                shutil.move(source_path, target_path)
            elif any(filename.startswith(sample) for sample in test_samples):
                source_path = os.path.join(dir_to_split, filename)
                target_path = os.path.join(test_path, filename)
                shutil.move(source_path, target_path)


if __name__ == "__main__":
    main()
# split_dataset.py
import os
import shutil
import random

# Root directory of the dataset
dataset_dir = "path/to/your/dataset"  # Change this to your dataset path
# Target classes
classes = ["cloudy", "foggy", "rainy", "shine", "sunrise"]
# Output directory
output_dir = "./weather_split"

# Split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1


def make_dirs():
    """Create output directories for train/val/test splits"""
    for split in ["train", "val", "test"]:
        for cls in classes:
            path = os.path.join(output_dir, split, cls)
            os.makedirs(path, exist_ok=True)


def split_dataset():
    """Split dataset into train/val/test sets and copy files"""
    total_train = 0
    total_val = 0
    total_test = 0

    for cls in classes:
        class_dir = os.path.join(dataset_dir, cls)
        images = [f for f in os.listdir(class_dir) if os.path.isfile(os.path.join(class_dir, f))]

        random.shuffle(images)
        total = len(images)

        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        train_files = images[:train_end]
        val_files = images[train_end:val_end]
        test_files = images[val_end:]

        print(f"{cls}: train {len(train_files)}, val {len(val_files)}, test {len(test_files)}")

        total_train += len(train_files)
        total_val += len(val_files)
        total_test += len(test_files)

        for split, files in zip(["train", "val", "test"], [train_files, val_files, test_files]):
            for f in files:
                src = os.path.join(class_dir, f)
                dst = os.path.join(output_dir, split, cls, f)
                shutil.copy(src, dst)

    return total_train, total_val, total_test


# Fix random seed for reproducibility
random.seed(114514)
make_dirs()
train_count, val_count, test_count = split_dataset()
print("------------------------")
print("Total train:", train_count)
print("Total val:", val_count)
print("Total test:", test_count)
print("------------------------")
print("Dataset splitting completed.")
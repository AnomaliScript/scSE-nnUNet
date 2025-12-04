import os
import random
import shutil

# Set random seed for reproducibility
random.seed(2025)

# Define paths
images_tr = "scSE_at_home/nnUNet_raw/imagesTr"
images_ts = "scSE_at_home/nnUNet_raw/imagesTs"

# Get all files from imagesTr
all_files = os.listdir(images_tr)
print(f"Total files in imagesTr: {len(all_files)}")

# Randomly select 43 files
selected_files = random.sample(all_files, 43)
print(f"\nSelected {len(selected_files)} files to move:")

# Move the files
for filename in selected_files:
    src = os.path.join(images_tr, filename)
    dst = os.path.join(images_ts, filename)
    shutil.move(src, dst)
    print(f"Moved: {filename}")

print(f"\nCompleted! Moved {len(selected_files)} files from imagesTr to imagesTs")
print(f"Remaining in imagesTr: {len(os.listdir(images_tr))}")
print(f"Total in imagesTs: {len(os.listdir(images_ts))}")

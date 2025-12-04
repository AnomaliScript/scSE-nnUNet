"""
Create custom train/validation split for nnU-Net Dataset001_Cervical
Forces 43 out of 383 cases to be used for validation
"""
import json
import random
from pathlib import Path

# Paths
raw_folder = Path(r"C:\\Users\\anoma\\Downloads\\scse-nnUNet\\scSE_at_home\\nnUNet_raw\\imagesTr")

# Get all case identifiers from imagesTr
image_files = sorted(raw_folder.glob("*.nii.gz"))
case_ids = [f.name.replace("_0000.nii.gz", "") for f in image_files]

print(f"Found {len(case_ids)} cases in imagesTr")

# Randomly sample 43 for validation
random.seed(2025)  # For reproducibility
val_cases = random.sample(case_ids, 43)
train_cases = [c for c in case_ids if c not in val_cases]

print(f"\nTrain: {len(train_cases)} cases")
print(f"Val:   {len(val_cases)} cases")

# Create splits_final.json format
# nnU-Net expects a list with one dict per fold
# For a single train/val split, we use fold 0
splits = [
    {
        "train": train_cases,
        "val": val_cases
    }
]

# Save splits file
output_file = Path(r"C:\\Users\\anoma\\Downloads\\scse-nnUNet\\scSE_at_home") / "splits_final.json"
with open(output_file, 'w') as f:
    json.dump(splits, f, indent=4)

print(f"\n✅ Splits file created: {output_file}")
print(f"\nFirst 5 validation cases:")
for case in val_cases[:5]:
    print(f"  - {case}")

print(f"\nTo use this split, train with fold 0:")
print(f"  nnUNetv2_train 001 3d_fullres 0 -tr nnUNetTrainerCervicalAttentionResEnc")

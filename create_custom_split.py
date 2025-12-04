"""
Create custom train/validation split for nnU-Net Dataset001_Cervical
Just prints the JSON - you can redirect to a file

Usage:
  python create_custom_split.py > splits_final.json
  python create_custom_split.py "path/to/imagesTr" > splits_final.json
"""
import json
import random
from pathlib import Path
import sys

# Get raw data folder from command line argument or use default
if len(sys.argv) > 1:
    raw_folder = Path(sys.argv[1])
else:
    raw_folder = Path(r"C:\Users\anoma\Downloads\scse-nnUNet\scSE_at_home\nnUNet_raw\Dataset001_Cervical\imagesTr")

# Get all case identifiers from imagesTr
image_files = sorted(raw_folder.glob("*.nii.gz"))
case_ids = [f.name.replace("_0000.nii.gz", "") for f in image_files]

# Randomly sample 43 for validation
random.seed(2025)  # For reproducibility
val_cases = random.sample(case_ids, 43)
train_cases = [c for c in case_ids if c not in val_cases]

# Create splits_final.json format
splits = [
    {
        "train": train_cases,
        "val": val_cases
    }
]

# Print JSON to stdout
print(json.dumps(splits, indent=4))

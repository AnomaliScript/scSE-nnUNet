"""
Extract YOLO slices from preprocessed nnU-Net data
Uses the same normalization as generate_yolo_attention_mask() for consistency
"""

import numpy as np
from pathlib import Path
import cv2
from tqdm import tqdm

def normalize_for_yolo(volume):
    """Same normalization as scse_modules.py line 74"""
    return ((volume - volume.min()) / (volume.max() - volume.min() + 1e-8) * 255).astype(np.uint8)

def mask_to_yolo_bbox(mask_2d, class_id, img_width=640, img_height=640):
    """Convert 2D segmentation mask to YOLO bounding box"""
    coords = np.argwhere(mask_2d > 0)

    if len(coords) == 0:
        return None

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    width = x_max - x_min
    height = y_max - y_min

    if width < 20 or height < 20:
        return None

    x_center = ((x_min + x_max) / 2) / img_width
    y_center = ((y_min + y_max) / 2) / img_height
    box_width = width / img_width
    box_height = height / img_height

    return f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"

# Paths
PREPROCESSED_DIR = Path(r"C:\Users\anoma\Downloads\scse-nnUNet\yolo_preprocessing\nnUNet_preprocessed\Dataset001_Cervical\nnUNetPlans_3d_fullres")
TRAIN_IDS_FILE = Path(r"C:\Users\anoma\Downloads\scse-nnUNet\yolo_preprocessing\train_ids.txt")
VAL_IDS_FILE = Path(r"C:\Users\anoma\Downloads\scse-nnUNet\yolo_preprocessing\val_ids.txt")

TRAIN_IMAGES_OUT = Path(r"C:\Users\anoma\Downloads\yolo-4-scSE\images_preprocessed\train")
TRAIN_LABELS_OUT = Path(r"C:\Users\anoma\Downloads\yolo-4-scSE\labels_preprocessed\train")
VAL_IMAGES_OUT = Path(r"C:\Users\anoma\Downloads\yolo-4-scSE\images_preprocessed\val")
VAL_LABELS_OUT = Path(r"C:\Users\anoma\Downloads\yolo-4-scSE\labels_preprocessed\val")

# Create output directories
for d in [TRAIN_IMAGES_OUT, TRAIN_LABELS_OUT, VAL_IMAGES_OUT, VAL_LABELS_OUT]:
    d.mkdir(parents=True, exist_ok=True)

# Load train/val split
with open(TRAIN_IDS_FILE) as f:
    train_ids = [line.strip() for line in f if line.strip()]

with open(VAL_IDS_FILE) as f:
    val_ids = [line.strip() for line in f if line.strip()]

print(f"Train cases: {len(train_ids)}")
print(f"Val cases: {len(val_ids)}")

def process_case(case_id, images_out_dir, labels_out_dir):
    """Extract slices from one preprocessed case"""
    npz_file = PREPROCESSED_DIR / f"{case_id}.npz"

    if not npz_file.exists():
        print(f"WARNING: {npz_file} not found")
        return 0

    # Load preprocessed data
    data = np.load(npz_file)
    volume = data['data']  # Shape: (C, D, H, W) or (D, H, W)
    seg = data['seg']      # Shape: (D, H, W)

    # Handle channel dimension
    if volume.ndim == 4:
        volume = volume[0]  # Take first channel

    D, H, W = volume.shape

    # Normalize using same method as generate_yolo_attention_mask
    volume_normalized = normalize_for_yolo(volume)

    slice_count = 0
    num_slices_per_orientation = 200

    # Extract SAGITTAL slices
    sagittal_indices = np.linspace(
        int(D * 0.2),
        int(D * 0.8),
        num_slices_per_orientation,
        dtype=int
    )

    for idx in sagittal_indices:
        # Image slice
        img_slice = volume_normalized[idx, :, :]
        img_slice = cv2.resize(img_slice, (640, 640))

        # Label slice
        label_slice = seg[idx, :, :]
        label_slice = cv2.resize(label_slice, (640, 640), interpolation=cv2.INTER_NEAREST)

        # Convert to YOLO bboxes
        yolo_lines = []
        for vert_label in range(1, 8):  # C1-C7
            mask = (label_slice == vert_label)
            bbox_line = mask_to_yolo_bbox(mask, vert_label - 1, 640, 640)
            if bbox_line:
                yolo_lines.append(bbox_line)

        # Save if we have bboxes
        if yolo_lines:
            slice_name = f"{case_id}_sag_{idx:03d}"
            cv2.imwrite(str(images_out_dir / f"{slice_name}.jpg"), img_slice)
            with open(labels_out_dir / f"{slice_name}.txt", 'w') as f:
                f.write('\n'.join(yolo_lines))
            slice_count += 1

    # Extract CORONAL slices
    coronal_indices = np.linspace(
        int(H * 0.2),
        int(H * 0.8),
        num_slices_per_orientation,
        dtype=int
    )

    for idx in coronal_indices:
        # Image slice
        img_slice = volume_normalized[:, idx, :]
        img_slice = cv2.resize(img_slice, (640, 640))

        # Label slice
        label_slice = seg[:, idx, :]
        label_slice = cv2.resize(label_slice, (640, 640), interpolation=cv2.INTER_NEAREST)

        # Convert to YOLO bboxes
        yolo_lines = []
        for vert_label in range(1, 8):  # C1-C7
            mask = (label_slice == vert_label)
            bbox_line = mask_to_yolo_bbox(mask, vert_label - 1, 640, 640)
            if bbox_line:
                yolo_lines.append(bbox_line)

        # Save if we have bboxes
        if yolo_lines:
            slice_name = f"{case_id}_cor_{idx:03d}"
            cv2.imwrite(str(images_out_dir / f"{slice_name}.jpg"), img_slice)
            with open(labels_out_dir / f"{slice_name}.txt", 'w') as f:
                f.write('\n'.join(yolo_lines))
            slice_count += 1

    return slice_count

# Process training cases
print("\n" + "="*70)
print("PROCESSING TRAINING CASES")
print("="*70)
train_slices = 0
for case_id in tqdm(train_ids, desc="Train"):
    slices = process_case(case_id, TRAIN_IMAGES_OUT, TRAIN_LABELS_OUT)
    train_slices += slices

# Process validation cases
print("\n" + "="*70)
print("PROCESSING VALIDATION CASES")
print("="*70)
val_slices = 0
for case_id in tqdm(val_ids, desc="Val"):
    slices = process_case(case_id, VAL_IMAGES_OUT, VAL_LABELS_OUT)
    val_slices += slices

print("\n" + "="*70)
print("COMPLETE")
print("="*70)
print(f"Train slices: {train_slices}")
print(f"Val slices: {val_slices}")
print(f"Total: {train_slices + val_slices}")
print("\nOutput saved to:")
print(f"  {TRAIN_IMAGES_OUT}")
print(f"  {VAL_IMAGES_OUT}")
print("="*70)

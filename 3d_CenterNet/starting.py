from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    Spacingd, ScaleIntensityRanged, 
    Lambdad, ToTensord
)
import numpy as np
from scipy.ndimage import gaussian_filter
from pathlib import Path

# 1. CREATE FILE PATH LIST
def prepare_data_dicts(image_dir, label_dir):
    """Create list of dicts with file paths"""
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    
    data_dicts = []
    for img_path in sorted(image_dir.glob("*_ct.nii.gz")):
        patient_id = img_path.name.replace("_ct.nii.gz", "")
        label_path = label_dir / f"{patient_id}_seg.nii.gz"
        
        if label_path.exists():
            data_dicts.append({
                "image": str(img_path),
                "label": str(label_path),
                "patient_id": patient_id
            })
    
    return data_dicts

# 2. SEGMENTATION → HEATMAP CONVERSION
def seg_to_heatmap(data):
    """Convert segmentation mask to CenterNet heatmap"""
    seg = data["label"][0]  # Remove channel dimension
    
    # Create heatmap
    heatmap = np.zeros_like(seg, dtype=np.float32)
    box_map = np.zeros((*seg.shape, 3), dtype=np.float32)  # w,h,d for each voxel
    
    # Process each vertebra
    for vert_id in np.unique(seg)[1:]:  # Skip background (0)
        mask = (seg == vert_id)
        coords = np.argwhere(mask)
        
        if len(coords) > 0:
            # Get center and size
            min_coords = coords.min(axis=0)
            max_coords = coords.max(axis=0)
            center = (min_coords + max_coords) // 2
            size = max_coords - min_coords + 1
            
            # Place Gaussian at center
            heatmap[tuple(center)] = 1.0
            
            # Store box size at center location
            box_map[tuple(center)] = size
    
    # Gaussian blur the heatmap
    heatmap = gaussian_filter(heatmap, sigma=2.0)
    heatmap = heatmap / (heatmap.max() + 1e-8)  # Normalize to [0,1]
    
    # Add to data dict
    data["heatmap"] = heatmap[None, ...]  # Add channel back
    data["box_regression"] = box_map.transpose(3, 0, 1, 2)  # Channels first
    
    # Remove original label if not needed
    del data["label"]
    
    return data

# 3. CREATE TRANSFORMS
train_transforms = Compose([
    # Load files
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    
    # Standardize spacing
    Spacingd(
        keys=["image", "label"], 
        pixdim=(1.0, 1.0, 2.0), 
        mode=("bilinear", "nearest")
    ),
    
    # Intensity normalization (bone window)
    ScaleIntensityRanged(
        keys=["image"], 
        a_min=-200, a_max=400, 
        b_min=0, b_max=1, 
        clip=True
    ),
    
    # CONVERT SEGMENTATION → HEATMAP
    Lambdad(keys=["image", "label"], func=seg_to_heatmap),
    
    # Convert to tensors
    ToTensord(keys=["image", "heatmap", "box_regression"])
])

# 4. CREATE DATASET & DATALOADER
# Prepare data
train_files = prepare_data_dicts("data/images", "data/labels")
val_files = prepare_data_dicts("data/val_images", "data/val_labels")

# Create datasets
train_dataset = Dataset(data=train_files, transform=train_transforms)
val_dataset = Dataset(data=val_files, transform=val_transforms)  # Use val_transforms

# Create dataloaders
train_loader = DataLoader(
    train_dataset,
    batch_size=2,      # Adjust based on GPU memory
    shuffle=True,
    num_workers=4,     # Parallel loading
    pin_memory=True    # Faster GPU transfer
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,      # Usually 1 for validation
    shuffle=False,
    num_workers=2
)

# 5. USE IN TRAINING LOOP
for batch_data in train_loader:
    images = batch_data["image"]           # [B, 1, H, W, D]
    heatmaps = batch_data["heatmap"]       # [B, 1, H, W, D]
    box_regression = batch_data["box_regression"]  # [B, 3, H, W, D]
    patient_ids = batch_data["patient_id"]  # List of IDs
    
    # Forward pass
    pred_heatmap, pred_boxes = model(images)
    
    # Calculate losses
    heatmap_loss = focal_loss(pred_heatmap, heatmaps)
    box_loss = l1_loss(pred_boxes, box_regression, mask=heatmaps>0.1)
    
    # ... rest of training
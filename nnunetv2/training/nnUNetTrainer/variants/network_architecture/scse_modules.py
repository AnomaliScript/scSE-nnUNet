"""
Detector-Guided Cervical Attention Modules

Integrates vertebra detector outputs (bounding boxes) with attention mechanisms.
Instead of learning which vertebra is where, we USE the detector's predictions
to apply appropriate attention in the right spatial regions.

Author: Brandon's Cervical Spine Project
Date: 2025-11-19
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
from pathlib import Path
import cv2
from typing import Dict, List, Tuple


# ============================================================================
# YOLO MASK GENERATION UTILITIES
# ============================================================================

def generate_yolo_attention_mask(
    volume_3d: np.ndarray,
    yolo_model_path: str = 'runs/vertebra_detector_114/weights/best.pt',
    conf_threshold: float = 0.25
) -> np.ndarray:
    """
    Generate 3D attention mask from preprocessed volume using YOLO detector.

    This function:
    1. Takes a 3D volume (preprocessed by nnUNet)
    2. Runs YOLO detection slice-by-slice
    3. Aggregates 2D detections into 3D bounding boxes
    4. Creates a 3D mask with values:
       - 0: background (no vertebra)
       - 1: C1 regions (spatial attention)
       - 2: C2 regions (enhanced scSE)
       - 3: C3-C7 regions (standard scSE)

    Args:
        volume_3d: Preprocessed 3D volume, shape (C, D, H, W) or (D, H, W)
        yolo_model_path: Path to trained YOLO weights
        conf_threshold: Detection confidence threshold

    Returns:
        attention_mask: 3D array same spatial shape as volume, dtype uint8
    """
    from ultralytics import YOLO

    # Handle channel dimension
    if volume_3d.ndim == 4:
        # (C, D, H, W) - take first channel
        volume = volume_3d[0]
    else:
        # (D, H, W)
        volume = volume_3d

    D, H, W = volume.shape

    # Normalize to 0-255 for YOLO
    volume_normalized = ((volume - volume.min()) / (volume.max() - volume.min() + 1e-8) * 255).astype(np.uint8)

    # Load YOLO model
    model = YOLO(yolo_model_path)

    # Store 2D detections per slice
    slice_detections = {}

    print(f"  Running YOLO on {D} slices...")
    for slice_idx in range(D):
        slice_2d = volume_normalized[slice_idx, :, :]

        # Convert to RGB (YOLO expects 3 channels)
        slice_rgb = cv2.cvtColor(slice_2d, cv2.COLOR_GRAY2RGB)

        # Run YOLO inference
        results = model(slice_rgb, conf=conf_threshold, verbose=False)

        # Parse detections for this slice
        detections = []
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    detection = {
                        'class_name': result.names[int(box.cls[0])],
                        'bbox': box.xyxy[0].cpu().numpy(),  # [x1, y1, x2, y2]
                        'confidence': float(box.conf[0].cpu().numpy())
                    }
                    detections.append(detection)

        if detections:
            slice_detections[slice_idx] = detections

    print(f"  Found vertebrae in {len(slice_detections)} slices")

    # Aggregate 2D detections into 3D bounding boxes
    vertebrae_3d = aggregate_detections_to_3d(slice_detections, (D, H, W))

    # Create 3D attention mask
    attention_mask = create_attention_mask_from_vertebrae(vertebrae_3d, (D, H, W))

    return attention_mask


def aggregate_detections_to_3d(
    slice_detections: Dict[int, List[Dict]],
    volume_shape: Tuple[int, int, int]
) -> Dict[str, Dict]:
    """
    Aggregate 2D slice detections into 3D vertebra bounding boxes.

    For each vertebra (C1-C7):
    - Find all slices where it appears
    - Take median bbox across those slices
    - Create 3D bbox: [x_min, x_max, y_min, y_max, z_min, z_max]

    Args:
        slice_detections: Dict mapping slice_idx -> list of detections
        volume_shape: (D, H, W)

    Returns:
        vertebrae_3d: Dict mapping vertebra name -> {'bbox': [...], 'slices': [...]}
    """
    D, H, W = volume_shape

    # Group detections by vertebra name
    vertebra_groups = {}

    for slice_idx, detections in slice_detections.items():
        for det in detections:
            vert_name = det['class_name']  # e.g., 'C1', 'C2', etc.

            if vert_name not in vertebra_groups:
                vertebra_groups[vert_name] = []

            vertebra_groups[vert_name].append({
                'slice': slice_idx,
                'bbox_2d': det['bbox'],  # [x1, y1, x2, y2]
                'confidence': det['confidence']
            })

    # For each vertebra, create 3D bbox
    vertebrae_3d = {}

    for vert_name, detections in vertebra_groups.items():
        slices = [d['slice'] for d in detections]
        bboxes_2d = np.array([d['bbox_2d'] for d in detections])  # (N, 4)

        # Take median bbox (more robust than mean)
        median_bbox_2d = np.median(bboxes_2d, axis=0)
        x1, y1, x2, y2 = median_bbox_2d

        # Z extent is min/max slices
        z_min = min(slices)
        z_max = max(slices) + 1  # +1 for inclusive upper bound

        # Clamp to volume bounds
        x1 = int(max(0, x1))
        x2 = int(min(W, x2))
        y1 = int(max(0, y1))
        y2 = int(min(H, y2))
        z_min = int(max(0, z_min))
        z_max = int(min(D, z_max))

        vertebrae_3d[vert_name] = {
            'bbox': [x1, x2, y1, y2, z_min, z_max],
            'slices': slices,
            'num_detections': len(detections)
        }

    return vertebrae_3d


def create_attention_mask_from_vertebrae(
    vertebrae_3d: Dict[str, Dict],
    volume_shape: Tuple[int, int, int]
) -> np.ndarray:
    """
    Create 3D attention mask from vertebra bounding boxes.

    Mask values:
    - 0: background
    - 1: C1 regions (spatial attention)
    - 2: C2 regions (enhanced scSE)
    - 3: C3-C7 regions (standard scSE)

    Args:
        vertebrae_3d: Dict mapping vertebra name -> {'bbox': [x1, x2, y1, y2, z1, z2]}
        volume_shape: (D, H, W)

    Returns:
        attention_mask: 3D array, dtype uint8
    """
    D, H, W = volume_shape
    attention_mask = np.zeros((D, H, W), dtype=np.uint8)

    for vert_name, vert_data in vertebrae_3d.items():
        x1, x2, y1, y2, z1, z2 = vert_data['bbox']

        # Determine attention group
        if vert_name == 'C1':
            mask_value = 1
        elif vert_name == 'C2':
            mask_value = 2
        else:  # C3, C4, C5, C6, C7
            mask_value = 3

        # Fill this 3D region with the appropriate mask value
        attention_mask[z1:z2, y1:y2, x1:x2] = mask_value

    # Count coverage
    c1_voxels = np.sum(attention_mask == 1)
    c2_voxels = np.sum(attention_mask == 2)
    c3_c7_voxels = np.sum(attention_mask == 3)
    total_voxels = D * H * W

    print(f"  Attention mask coverage:")
    print(f"    C1: {c1_voxels / total_voxels * 100:.1f}%")
    print(f"    C2: {c2_voxels / total_voxels * 100:.1f}%")
    print(f"    C3-C7: {c3_c7_voxels / total_voxels * 100:.1f}%")
    print(f"    Background: {(total_voxels - c1_voxels - c2_voxels - c3_c7_voxels) / total_voxels * 100:.1f}%")

    return attention_mask


# ============================================================================
# ATTENTION MODULES
# ============================================================================

class ChannelSELayer3D(nn.Module):    
    def __init__(self, num_channels, reduction_ratio=2):
        super(ChannelSELayer3D, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_tensor):
        batch_size, num_channels, D, H, W = input_tensor.size()
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))
        output_tensor = input_tensor * fc_out_2.view(batch_size, num_channels, 1, 1, 1)
        return output_tensor


class SpatialSELayer3D(nn.Module):
    def __init__(self, num_channels):
        super(SpatialSELayer3D, self).__init__()
        self.conv = nn.Conv3d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_tensor):
        squeeze_tensor = self.sigmoid(self.conv(input_tensor))
        output_tensor = input_tensor * squeeze_tensor
        return output_tensor


class ChannelSpatialSELayer3D(nn.Module):    
    def __init__(self, num_channels, reduction_ratio=2):
        super(ChannelSpatialSELayer3D, self).__init__()
        self.cSE = ChannelSELayer3D(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer3D(num_channels)
    
    def forward(self, input_tensor):
        cSE_out = self.cSE(input_tensor)
        sSE_out = self.sSE(input_tensor)
        output_tensor = cSE_out + sSE_out
        return output_tensor


class DetectorGuidedCervicalAttention3D(nn.Module):
    """
    Detector-Guided Cervical Level-Aware Attention
    
    Uses vertebra detector output (bounding boxes + labels) to spatially route attention.
    
    Architecture:
    - C1 regions: Spatial attention only (ring structure, no vertebral body)
    - C2 regions: Enhanced scSE (odontoid process needs strong discrimination)  
    - C3-C7 regions: Standard scSE (similar anatomy)
    
    NO learned classifier - uses geometric lookup from detector bounding boxes.
    """
    
    def __init__(self, num_channels, reduction_ratio=2, detections_dir=None):
        """
        Args:
            num_channels: Number of feature channels
            reduction_ratio: Channel reduction for scSE bottleneck
            detections_dir: Path to directory containing detection JSON files
                           Format: {case_id}.json
        """
        super(DetectorGuidedCervicalAttention3D, self).__init__()
        
        self.detections_dir = Path(detections_dir) if detections_dir else None
        self.current_detections = None  # Will be set per case during forward
        
        # C1-specific pathway: Spatial attention only
        self.c1_attention = SpatialSELayer3D(num_channels)
        
        # C2-specific pathway: Enhanced scSE (lower reduction ratio = more capacity)
        self.c2_attention = ChannelSpatialSELayer3D(
            num_channels, 
            reduction_ratio=max(1, reduction_ratio // 2)
        )
        
        # C3-C7 pathway: Standard scSE
        self.c3_c7_attention = ChannelSpatialSELayer3D(num_channels, reduction_ratio)
    
    def load_detections(self, case_id):
        """
        Load detector output for a specific case.
        
        Args:
            case_id: Case identifier (e.g., 'RSNA_010')
        
        Returns:
            dict: Detection results with vertebra locations
        """
        if self.detections_dir is None:
            return None
        
        detection_file = self.detections_dir / f"{case_id}.json"
        
        if not detection_file.exists():
            print(f"⚠️  Warning: No detection file found for {case_id}")
            return None
        
        with open(detection_file, 'r') as f:
            detections = json.load(f)
        
        return detections
    
    def create_spatial_attention_map(self, detections, feature_shape, device):
        """
        Create 3D spatial map indicating which attention type to apply where.
        
        Args:
            detections: Detection results from detector
            feature_shape: (D, H, W) of feature map
            device: torch device
        
        Returns:
            attention_routing: (3, D, H, W) tensor
                - Channel 0: C1 attention regions
                - Channel 1: C2 attention regions  
                - Channel 2: C3-C7 attention regions
        """
        D, H, W = feature_shape
        attention_routing = torch.zeros(3, D, H, W, device=device, dtype=torch.float32)
        
        if detections is None or 'vertebrae_detected' not in detections:
            # No detections available - apply C3-C7 attention everywhere (fallback)
            attention_routing[2, :, :, :] = 1.0
            return attention_routing
        
        vertebrae_detected = detections['vertebrae_detected']
        
        # For each detected vertebra, mark its region with appropriate attention type
        for vert_name, vert_data in vertebrae_detected.items():
            if not vert_data.get('present', False):
                continue
            
            bbox = vert_data.get('bbox')
            if bbox is None or len(bbox) != 6:
                continue
            
            x_min, x_max, y_min, y_max, z_min, z_max = bbox
            
            # Clamp to feature map bounds
            x_min, x_max = max(0, x_min), min(W, x_max)
            y_min, y_max = max(0, y_min), min(H, y_max)
            z_min, z_max = max(0, z_min), min(D, z_max)
            
            # Determine attention group
            if vert_name == 'C1':
                group_idx = 0  # Spatial attention only
            elif vert_name == 'C2':
                group_idx = 1  # Enhanced scSE
            else:  # C3, C4, C5, C6, C7
                group_idx = 2  # Standard scSE
            
            # Mark this 3D region for this attention type
            # Note: Coordinates might need adjustment based on downsampling factor
            attention_routing[group_idx, z_min:z_max, y_min:y_max, x_min:x_max] = 1.0
        
        # Normalize: ensure each voxel has total weight = 1
        # (handles overlapping regions)
        total_weight = attention_routing.sum(dim=0, keepdim=True)
        total_weight = torch.clamp(total_weight, min=1e-6)  # Avoid division by zero
        attention_routing = attention_routing / total_weight
        
        return attention_routing
    
    def create_routing_from_attention_mask(self, attention_mask, feature_shape, device):
        """
        Convert YOLO attention mask to routing map for feature maps.

        Args:
            attention_mask: (D_full, H_full, W_full) array with values 0,1,2,3
            feature_shape: (D_feat, H_feat, W_feat) target feature map size
            device: torch device

        Returns:
            attention_routing: (3, D_feat, H_feat, W_feat) tensor
        """
        D_feat, H_feat, W_feat = feature_shape

        # Convert numpy to torch and move to device
        if isinstance(attention_mask, np.ndarray):
            attention_mask = torch.from_numpy(attention_mask).float().to(device)

        # Resize attention mask to match feature map resolution
        # attention_mask shape: (D, H, W) -> (1, 1, D, H, W) for interpolation
        mask_resized = F.interpolate(
            attention_mask.unsqueeze(0).unsqueeze(0),
            size=(D_feat, H_feat, W_feat),
            mode='nearest'
        ).squeeze(0).squeeze(0)  # Back to (D_feat, H_feat, W_feat)

        # Create 3-channel routing map
        attention_routing = torch.zeros(3, D_feat, H_feat, W_feat, device=device, dtype=torch.float32)

        # Channel 0: C1 regions (where mask == 1)
        attention_routing[0] = (mask_resized == 1).float()

        # Channel 1: C2 regions (where mask == 2)
        attention_routing[1] = (mask_resized == 2).float()

        # Channel 2: C3-C7 regions (where mask == 3)
        attention_routing[2] = (mask_resized == 3).float()

        # Normalize: ensure each voxel has total weight = 1
        total_weight = attention_routing.sum(dim=0, keepdim=True)
        total_weight = torch.clamp(total_weight, min=1e-6)
        attention_routing = attention_routing / total_weight

        return attention_routing

    def forward(self, x, attention_mask=None, case_id=None, detections=None):
        """
        Apply detector-guided attention.

        Args:
            x: Input features (B, C, D, H, W)
            attention_mask: YOLO attention mask (B, D_full, H_full, W_full) or (D_full, H_full, W_full)
            case_id: Case identifier for loading detections (optional, legacy)
            detections: Pre-loaded detection dict (optional, legacy)

        Returns:
            attended: Attention-weighted features (B, C, D, H, W)
            attention_routing: Spatial routing map (B, 3, D, H, W) for visualization
        """
        batch_size, num_channels, D, H, W = x.size()

        # Priority 1: Use attention_mask if provided
        if attention_mask is not None:
            # Handle batch dimension
            if attention_mask.ndim == 3:
                # Single mask for whole batch
                attention_routing = self.create_routing_from_attention_mask(
                    attention_mask, (D, H, W), x.device
                )
                attention_routing = attention_routing.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)
            elif attention_mask.ndim == 4:
                # Separate mask per batch item
                routing_list = []
                for b in range(batch_size):
                    routing_b = self.create_routing_from_attention_mask(
                        attention_mask[b], (D, H, W), x.device
                    )
                    routing_list.append(routing_b)
                attention_routing = torch.stack(routing_list, dim=0)
            else:
                raise ValueError(f"attention_mask must be 3D or 4D, got shape {attention_mask.shape}")

        # Priority 2: Use legacy detection dict (for backwards compatibility)
        elif detections is not None or case_id is not None:
            if detections is None and case_id is not None:
                detections = self.load_detections(case_id)
            attention_routing = self.create_spatial_attention_map(detections, (D, H, W), x.device)
            attention_routing = attention_routing.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)

        # Fallback: Apply C3-C7 attention everywhere
        else:
            attention_routing = torch.zeros(batch_size, 3, D, H, W, device=x.device, dtype=torch.float32)
            attention_routing[:, 2, :, :, :] = 1.0  # All C3-C7

        # Apply all three attention pathways
        c1_out = self.c1_attention(x)
        c2_out = self.c2_attention(x)
        c3_c7_out = self.c3_c7_attention(x)

        # Spatially blend outputs based on routing map
        attended = (
            attention_routing[:, 0:1, :, :, :] * c1_out +
            attention_routing[:, 1:2, :, :, :] * c2_out +
            attention_routing[:, 2:3, :, :, :] * c3_c7_out
        )

        return attended, attention_routing


class SimpleCervicalAttention3D(nn.Module):
    """
    Baseline: Standard scSE applied uniformly (no detector guidance)
    """
    
    def __init__(self, num_channels, reduction_ratio=2):
        super(SimpleCervicalAttention3D, self).__init__()
        self.attention = ChannelSpatialSELayer3D(num_channels, reduction_ratio)
    
    def forward(self, x, case_id=None, detections=None):
        attended = self.attention(x)
        return attended, None


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Testing Detector-Guided Cervical Attention Module")
    print("=" * 70)
    
    # Simulate feature map from encoder
    batch_size = 2
    num_channels = 64
    D, H, W = 32, 64, 64
    
    x = torch.randn(batch_size, num_channels, D, H, W)
    
    print(f"\nInput features: {x.shape}")
    
    # Example detection result (what detector outputs)
    example_detections = {
        "case_id": "RSNA_010",
        "vertebrae_detected": {
            "C1": {
                "present": True,
                "center": [32, 32, 5],
                "bbox": [27, 37, 27, 37, 2, 8],
                "confidence": 0.92
            },
            "C2": {
                "present": True,
                "center": [32, 32, 12],
                "bbox": [27, 37, 27, 37, 9, 15],
                "confidence": 0.95
            },
            "C5": {
                "present": True,
                "center": [32, 32, 22],
                "bbox": [27, 37, 27, 37, 19, 25],
                "confidence": 0.88
            }
        }
    }
    
    # Initialize module
    attention_module = DetectorGuidedCervicalAttention3D(
        num_channels=num_channels,
        reduction_ratio=2,
        detections_dir=None  # Would be set to actual directory in training
    )
    
    print("\n" + "-" * 70)
    print("Running forward pass with detector guidance...")
    
    # Forward pass
    attended, routing_map = attention_module(x, detections=example_detections)
    
    print(f"\nOutput attended features: {attended.shape}")
    print(f"Attention routing map: {routing_map.shape}")
    
    # Analyze routing
    print("\n" + "-" * 70)
    print("Attention Routing Analysis:")
    c1_coverage = (routing_map[:, 0] > 0).float().mean().item() * 100
    c2_coverage = (routing_map[:, 1] > 0).float().mean().item() * 100
    c3_c7_coverage = (routing_map[:, 2] > 0).float().mean().item() * 100
    
    print(f"  C1 attention applied to: {c1_coverage:.1f}% of volume")
    print(f"  C2 attention applied to: {c2_coverage:.1f}% of volume")
    print(f"  C3-C7 attention applied to: {c3_c7_coverage:.1f}% of volume")
    
    print("\n✓ Module test complete!")
    print("=" * 70)
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
# CONFIGURATION SWITCHES
# ============================================================================

# Enable/disable Faster R-CNN for detector-guided attention
# Set to True to use Faster R-CNN detections for spatial attention routing
# Set to False to use fallback uniform attention (C3-C7 attention everywhere)
USE_FASTER_RCNN_ATTENTION = False  # <-- EDIT THIS LINE TO ENABLE/DISABLE

# Path to pretrained Faster R-CNN weights (if USE_FASTER_RCNN_ATTENTION=True)
FASTER_RCNN_WEIGHTS_PATH = "nnunetv2/training/pretrained_models/faster_rcnn_vertebra.pth"

# Enable/disable scSE attention on skip connections
# Set to True to apply attention to encoder skip connections before concatenation
# Set to False to leave skip connections unchanged (current behavior)
USE_SKIP_CONNECTION_ATTENTION = True  # <-- Edit this line for skip connection attention

# Enable/disable scSE attention on bottleneck (deepest layer)
# Set to True to apply attention to bottleneck features
# Set to False to skip bottleneck attention
USE_BOTTLENECK_ATTENTION = True  # <-- Edit this line for bottleneck attention

# Enable/disable scSE attention on decoder blocks
# Set to True to apply attention after each decoder convolution block
# Set to False to skip decoder attention
USE_DECODER_ATTENTION = True  # <-- Edit this line for decoder attention

# Enable/disable scSE attention on encoder blocks
# Set to True to apply attention after each encoder conv block (before pooling)
# Set to False to leave encoder unchanged
USE_ENCODER_ATTENTION = False  # <-- Edit this line for encoder attention


# ============================================================================
# 3D FASTER R-CNN FOR VERTEBRA DETECTION
# ============================================================================

class RegionProposalNetwork3D(nn.Module):
    """
    3D Region Proposal Network (RPN)
    Generates region proposals from 3D feature maps for vertebra detection.
    """
    def __init__(self, in_channels, num_anchors=9, anchor_stride=2):
        """
        Args:
            in_channels: Number of input feature channels
            num_anchors: Number of anchor boxes per spatial location
            anchor_stride: Stride for anchor generation (downsampling factor)
        """
        super(RegionProposalNetwork3D, self).__init__()

        self.num_anchors = num_anchors
        self.anchor_stride = anchor_stride

        # Shared 3D convolution for feature extraction
        self.conv = nn.Conv3d(in_channels, 512, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # Classification branch: objectness score (vertebra vs background)
        self.cls_logits = nn.Conv3d(512, num_anchors * 2, kernel_size=1)

        # Regression branch: bbox refinement (6 coords for 3D: x, y, z, d, h, w)
        self.bbox_pred = nn.Conv3d(512, num_anchors * 6, kernel_size=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, features):
        """
        Args:
            features: (B, C, D, H, W) feature map from backbone

        Returns:
            objectness: (B, num_anchors*2, D', H', W') objectness scores
            bbox_deltas: (B, num_anchors*6, D', H', W') bbox refinements
        """
        # Shared feature extraction
        x = self.relu(self.conv(features))

        # Objectness classification
        objectness = self.cls_logits(x)

        # Bounding box regression
        bbox_deltas = self.bbox_pred(x)

        return objectness, bbox_deltas


class RoIAlign3D(nn.Module):
    """
    3D RoI Align for extracting fixed-size features from proposals.
    Uses trilinear interpolation for smooth gradients.
    """
    def __init__(self, output_size=(7, 7, 7), spatial_scale=1.0):
        """
        Args:
            output_size: Target size for output features (D, H, W)
            spatial_scale: Scale factor between feature map and input volume
        """
        super(RoIAlign3D, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, features, proposals):
        """
        Args:
            features: (B, C, D, H, W) feature map
            proposals: List of tensors, each (N, 6) [x1, y1, z1, x2, y2, z2]

        Returns:
            roi_features: (total_proposals, C, D_out, H_out, W_out)
        """
        B, C, D, H, W = features.shape
        D_out, H_out, W_out = self.output_size

        roi_features_list = []

        for batch_idx, batch_proposals in enumerate(proposals):
            if batch_proposals.shape[0] == 0:
                continue

            # Scale proposals to feature map coordinates
            scaled_proposals = batch_proposals * self.spatial_scale

            for proposal in scaled_proposals:
                x1, y1, z1, x2, y2, z2 = proposal

                # Create sampling grid for this RoI
                # Generate normalized coordinates (-1 to 1) for grid_sample
                z_coords = torch.linspace(z1, z2, D_out, device=features.device)
                y_coords = torch.linspace(y1, y2, H_out, device=features.device)
                x_coords = torch.linspace(x1, x2, W_out, device=features.device)

                # Normalize to [-1, 1] for grid_sample
                z_coords_norm = 2.0 * z_coords / (D - 1) - 1.0
                y_coords_norm = 2.0 * y_coords / (H - 1) - 1.0
                x_coords_norm = 2.0 * x_coords / (W - 1) - 1.0

                # Create meshgrid
                grid_z, grid_y, grid_x = torch.meshgrid(z_coords_norm, y_coords_norm, x_coords_norm, indexing='ij')
                grid = torch.stack([grid_x, grid_y, grid_z], dim=-1).unsqueeze(0)  # (1, D_out, H_out, W_out, 3)

                # Sample features using trilinear interpolation
                roi_feature = F.grid_sample(
                    features[batch_idx:batch_idx+1],
                    grid,
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=True
                )  # (1, C, D_out, H_out, W_out)

                roi_features_list.append(roi_feature.squeeze(0))

        if len(roi_features_list) == 0:
            # No proposals, return empty tensor
            return torch.zeros(0, C, D_out, H_out, W_out, device=features.device)

        return torch.stack(roi_features_list, dim=0)


class FasterRCNNHead3D(nn.Module):
    """
    3D Faster R-CNN detection head.
    Classifies proposals and refines bounding boxes.
    """
    def __init__(self, in_channels, num_classes=8, roi_size=(7, 7, 7)):
        """
        Args:
            in_channels: Number of RoI feature channels
            num_classes: Number of vertebra classes (C1-C7 + background)
            roi_size: Size of RoI features (D, H, W)
        """
        super(FasterRCNNHead3D, self).__init__()

        self.num_classes = num_classes

        # Feature compression
        roi_flatten_size = in_channels * roi_size[0] * roi_size[1] * roi_size[2]
        self.fc1 = nn.Linear(roi_flatten_size, 1024)
        self.fc2 = nn.Linear(1024, 1024)

        # Classification head
        self.cls_score = nn.Linear(1024, num_classes)

        # Box regression head (6 values per class: dx, dy, dz, dw, dh, dd)
        self.bbox_pred = nn.Linear(1024, num_classes * 6)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.fc1.weight, std=0.01)
        nn.init.normal_(self.fc2.weight, std=0.01)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)

        for m in [self.fc1, self.fc2, self.cls_score, self.bbox_pred]:
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, roi_features):
        """
        Args:
            roi_features: (N, C, D, H, W) RoI features

        Returns:
            class_logits: (N, num_classes) classification scores
            bbox_deltas: (N, num_classes * 6) bbox refinements
        """
        # Flatten RoI features
        x = roi_features.flatten(start_dim=1)

        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)

        # Classification and regression
        class_logits = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return class_logits, bbox_deltas


class FasterRCNN3D(nn.Module):
    """
    Complete 3D Faster R-CNN for cervical vertebra detection.

    Architecture:
    1. Backbone: Shares features with main segmentation network
    2. RPN: Proposes candidate vertebra regions
    3. RoI Align: Extracts fixed-size features from proposals
    4. Detection Head: Classifies vertebrae (C1-C7) and refines boxes
    """
    def __init__(
        self,
        in_channels=32,
        num_classes=8,  # C1-C7 + background
        rpn_num_anchors=9,
        roi_size=(7, 7, 7),
        nms_threshold=0.5,
        score_threshold=0.25
    ):
        super(FasterRCNN3D, self).__init__()

        self.num_classes = num_classes
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold

        # Region Proposal Network
        self.rpn = RegionProposalNetwork3D(in_channels, num_anchors=rpn_num_anchors)

        # RoI Align
        self.roi_align = RoIAlign3D(output_size=roi_size, spatial_scale=1.0)

        # Detection head
        self.detection_head = FasterRCNNHead3D(in_channels, num_classes, roi_size)

        # Anchor generation parameters
        self.anchor_sizes = [(16, 16, 16), (32, 32, 32), (64, 64, 64)]  # Small, medium, large vertebrae
        self.aspect_ratios = [(1.0, 1.0, 1.0), (1.0, 1.0, 1.5), (1.0, 1.0, 2.0)]  # Vertebrae are elongated in Z

    def generate_anchors(self, feature_shape, device):
        """
        Generate 3D anchor boxes on feature map.

        Args:
            feature_shape: (D, H, W) shape of feature map
            device: torch device

        Returns:
            anchors: (N, 6) tensor [x1, y1, z1, x2, y2, z2]
        """
        D, H, W = feature_shape
        anchors = []

        # Generate anchors at each spatial location
        for z in range(D):
            for y in range(H):
                for x in range(W):
                    cx, cy, cz = x, y, z

                    # Generate anchors with different sizes and aspect ratios
                    for size in self.anchor_sizes:
                        for ratio in self.aspect_ratios:
                            w = size[0] * ratio[0]
                            h = size[1] * ratio[1]
                            d = size[2] * ratio[2]

                            x1 = cx - w / 2
                            y1 = cy - h / 2
                            z1 = cz - d / 2
                            x2 = cx + w / 2
                            y2 = cy + h / 2
                            z2 = cz + d / 2

                            anchors.append([x1, y1, z1, x2, y2, z2])

        return torch.tensor(anchors, device=device, dtype=torch.float32)

    def forward(self, features, targets=None):
        """
        Args:
            features: (B, C, D, H, W) feature map from encoder
            targets: Optional training targets

        Returns:
            detections: List of dicts with keys 'boxes', 'labels', 'scores'
        """
        B, C, D, H, W = features.shape

        # 1. Region Proposal Network
        objectness, bbox_deltas = self.rpn(features)

        # 2. Generate anchors
        anchors = self.generate_anchors((D, H, W), features.device)

        # 3. Generate proposals from RPN output
        proposals = self._generate_proposals(objectness, bbox_deltas, anchors, (D, H, W))

        # 4. RoI Align
        roi_features = self.roi_align(features, proposals)

        # 5. Detection head
        if roi_features.shape[0] > 0:
            class_logits, bbox_deltas_refined = self.detection_head(roi_features)

            # 6. Post-processing: NMS and score filtering
            detections = self._postprocess_detections(
                class_logits, bbox_deltas_refined, proposals, (D, H, W)
            )
        else:
            # No proposals
            detections = [{'boxes': torch.zeros(0, 6, device=features.device),
                          'labels': torch.zeros(0, dtype=torch.long, device=features.device),
                          'scores': torch.zeros(0, device=features.device)}
                         for _ in range(B)]

        return detections

    def _generate_proposals(self, objectness, bbox_deltas, anchors, feature_shape):
        """Generate proposals from RPN outputs (simplified for inference)."""
        # This is a simplified implementation
        # In practice, you'd implement proper proposal generation with NMS
        B = objectness.shape[0]
        proposals = []

        for b in range(B):
            # Take top-k objectness scores
            obj_scores = objectness[b].flatten()
            top_k = min(100, obj_scores.shape[0])
            top_indices = torch.topk(obj_scores, top_k).indices

            # Select corresponding anchors
            batch_proposals = anchors[top_indices % anchors.shape[0]]
            proposals.append(batch_proposals)

        return proposals

    def _postprocess_detections(self, class_logits, bbox_deltas, proposals, feature_shape):
        """Apply NMS and score thresholding (simplified)."""
        # Simplified post-processing
        # In practice, implement proper NMS and bbox refinement
        scores = F.softmax(class_logits, dim=1)
        max_scores, labels = scores[:, 1:].max(dim=1)  # Exclude background
        labels = labels + 1  # Offset for background class

        # Filter by score threshold
        keep = max_scores > self.score_threshold

        detections = []
        # Simplified: return one detection dict
        # In practice, split by batch
        detections.append({
            'boxes': torch.cat([p for p in proposals], dim=0)[keep],
            'labels': labels[keep],
            'scores': max_scores[keep]
        })

        return detections


def generate_faster_rcnn_attention_mask(
    volume_3d: np.ndarray,
    faster_rcnn_model: FasterRCNN3D,
    features: torch.Tensor,
    conf_threshold: float = 0.25,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Generate confidence-weighted 3D attention mask using 3D Faster R-CNN.

    Args:
        volume_3d: Preprocessed 3D volume, shape (C, D, H, W) or (D, H, W)
        faster_rcnn_model: Trained 3D Faster R-CNN model
        features: Feature map from encoder (B, C, D, H, W)
        conf_threshold: Detection confidence threshold
        device: Device for inference

    Returns:
        attention_mask: 4D array shape (4, D, H, W), dtype float32
    """
    # Handle channel dimension
    if volume_3d.ndim == 4:
        volume = volume_3d[0]
    else:
        volume = volume_3d

    D, H, W = volume.shape

    # Run 3D Faster R-CNN
    faster_rcnn_model.eval()
    with torch.no_grad():
        detections = faster_rcnn_model(features)

    # Convert detections to vertebrae_3d format
    vertebrae_3d = {}
    vertebra_names = ['background', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']

    for detection in detections:
        boxes = detection['boxes'].cpu().numpy()
        labels = detection['labels'].cpu().numpy()
        scores = detection['scores'].cpu().numpy()

        for box, label, score in zip(boxes, labels, scores):
            if score < conf_threshold:
                continue

            vert_name = vertebra_names[label]
            if vert_name == 'background':
                continue

            x1, y1, z1, x2, y2, z2 = box

            vertebrae_3d[vert_name] = {
                'bbox': [int(x1), int(x2), int(y1), int(y2), int(z1), int(z2)],
                'confidence': float(score)
            }

    # Create attention mask
    attention_mask = create_attention_mask_from_vertebrae(vertebrae_3d, (D, H, W))

    return attention_mask


def create_attention_mask_from_vertebrae(
    vertebrae_3d: Dict[str, Dict],
    volume_shape: Tuple[int, int, int]
) -> np.ndarray:
    """
    Create confidence-weighted 3D attention mask from vertebra bounding boxes.

    Returns 4-channel mask:
    - Channel 0: C1 regions (confidence-weighted)
    - Channel 1: C2 regions (confidence-weighted)
    - Channel 2: C3-C7 regions (confidence-weighted)
    - Channel 3: Background (1.0 where no vertebra detected)

    Args:
        vertebrae_3d: Dict mapping vertebra name -> {'bbox': [...], 'confidence': float}
        volume_shape: (D, H, W)

    Returns:
        attention_mask: 4D array (4, D, H, W), dtype float32
    """
    D, H, W = volume_shape
    attention_mask = np.zeros((4, D, H, W), dtype=np.float32)

    for vert_name, vert_data in vertebrae_3d.items():
        x1, x2, y1, y2, z1, z2 = vert_data['bbox']
        confidence = vert_data['confidence']

        # Clamp to volume bounds
        x1 = int(max(0, min(W, x1)))
        x2 = int(max(0, min(W, x2)))
        y1 = int(max(0, min(H, y1)))
        y2 = int(max(0, min(H, y2)))
        z1 = int(max(0, min(D, z1)))
        z2 = int(max(0, min(D, z2)))

        # Determine attention channel
        if vert_name == 'C1':
            channel_idx = 0
        elif vert_name == 'C2':
            channel_idx = 1
        else:  # C3, C4, C5, C6, C7
            channel_idx = 2

        # Fill with confidence-weighted value
        attention_mask[channel_idx, z1:z2, y1:y2, x1:x2] = confidence

    # Background channel
    vertebra_coverage = attention_mask[0:3].sum(axis=0)
    attention_mask[3] = (vertebra_coverage == 0).astype(np.float32)

    # Normalize so weights sum to 1
    total_weight = attention_mask.sum(axis=0, keepdims=True)
    total_weight = np.maximum(total_weight, 1e-6)
    attention_mask = attention_mask / total_weight

    # Statistics
    c1_coverage = (attention_mask[0] > 0).sum() / (D * H * W) * 100
    c2_coverage = (attention_mask[1] > 0).sum() / (D * H * W) * 100
    c3_c7_coverage = (attention_mask[2] > 0).sum() / (D * H * W) * 100
    bg_coverage = (attention_mask[3] > 0).sum() / (D * H * W) * 100

    print(f"  Attention mask coverage (Faster R-CNN):")
    print(f"    C1: {c1_coverage:.1f}%")
    print(f"    C2: {c2_coverage:.1f}%")
    print(f"    C3-C7: {c3_c7_coverage:.1f}%")
    print(f"    Background: {bg_coverage:.1f}%")

    return attention_mask


# ============================================================================
# ATTENTION MODULES
# ============================================================================

class ChannelSELayer3D(nn.Module):    
    def __init__(self, num_channels, reduction_ratio=2):
        super(ChannelSELayer3D, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio # // is integer division (floor operator)
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
            attention_mask: (4, D_full, H_full, W_full) confidence-weighted mask
                           OR (D_full, H_full, W_full) legacy discrete mask
            feature_shape: (D_feat, H_feat, W_feat) target feature map size
            device: torch device

        Returns:
            attention_routing: (3, D_feat, H_feat, W_feat) tensor with confidence weights
        """
        D_feat, H_feat, W_feat = feature_shape

        # Convert numpy to torch and move to device
        if isinstance(attention_mask, np.ndarray):
            attention_mask = torch.from_numpy(attention_mask).float().to(device)

        # Handle both new (4-channel) and legacy (discrete) formats
        if attention_mask.ndim == 4 and attention_mask.shape[0] == 4:
            # New format: (4, D, H, W) confidence-weighted channels
            # Resize to feature map resolution
            mask_resized = F.interpolate(
                attention_mask.unsqueeze(0),  # (1, 4, D, H, W)
                size=(D_feat, H_feat, W_feat),
                mode='trilinear',  # Use trilinear for smooth confidence interpolation
                align_corners=False
            ).squeeze(0)  # Back to (4, D_feat, H_feat, W_feat)

            # Extract first 3 channels (C1, C2, C3-C7), ignoring background channel
            attention_routing = mask_resized[0:3]

        else:
            # Legacy format: (D, H, W) discrete values 0,1,2,3
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

            # Handle background regions (mask == 0): apply C3-C7 attention as default
            total_weight = attention_routing.sum(dim=0, keepdim=True)
            background_mask = (total_weight == 0).float()  # Where no vertebra detected
            attention_routing[2] += background_mask.squeeze(0)  # Assign background to C3-C7

            # Normalize: ensure each voxel has total weight = 1
            total_weight = attention_routing.sum(dim=0, keepdim=True)
            total_weight = torch.clamp(total_weight, min=1e-6)
            attention_routing = attention_routing / total_weight

        return attention_routing

    def forward(self, x, attention_mask=None, case_id=None, detections=None, faster_rcnn_model=None):
        """
        Apply detector-guided or uniform scSE attention based on USE_FASTER_RCNN_ATTENTION switch.

        Args:
            x: Input features (B, C, D, H, W)
            attention_mask: Pre-computed attention mask (optional)
            case_id: Case identifier (optional)
            detections: Detection dict (optional)
            faster_rcnn_model: Trained FasterRCNN3D model (optional)

        Returns:
            attended: Attention-weighted features (B, C, D, H, W)
            attention_routing: Routing map if detector used, else None
        """
        # Check global switch
        if not USE_FASTER_RCNN_ATTENTION:
            # Fallback: uniform C3-C7 scSE attention everywhere
            attended = self.c3_c7_attention(x)
            return attended, None

        # Faster R-CNN enabled: use detector-guided attention
        if faster_rcnn_model is not None:
            # Generate detections from current features
            with torch.no_grad():
                detections_list = faster_rcnn_model(x)

            # Create spatial attention routing map
            B, C, D, H, W = x.shape
            attention_routing = torch.zeros(B, 3, D, H, W, device=x.device, dtype=torch.float32)

            # Convert detections to routing map
            vertebra_names = ['background', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']

            for batch_idx, detection in enumerate(detections_list):
                boxes = detection['boxes']
                labels = detection['labels']
                scores = detection['scores']

                for box, label, score in zip(boxes, labels, scores):
                    if label == 0:  # Skip background
                        continue

                    vert_name = vertebra_names[label]
                    x1, y1, z1, x2, y2, z2 = box

                    # Clamp to feature map bounds
                    x1, x2 = int(max(0, x1)), int(min(W, x2))
                    y1, y2 = int(max(0, y1)), int(min(H, y2))
                    z1, z2 = int(max(0, z1)), int(min(D, z2))

                    # Determine attention channel
                    if vert_name == 'C1':
                        channel_idx = 0
                    elif vert_name == 'C2':
                        channel_idx = 1
                    else:  # C3-C7
                        channel_idx = 2

                    # Mark this region (confidence-weighted)
                    attention_routing[batch_idx, channel_idx, z1:z2, y1:y2, x1:x2] = score

            # Normalize routing (ensure sum to 1)
            total_weight = attention_routing.sum(dim=1, keepdim=True)
            total_weight = torch.clamp(total_weight, min=1e-6)
            attention_routing = attention_routing / total_weight

            # Apply vertebra-specific attention
            c1_attended = self.c1_attention(x) * attention_routing[:, 0:1]
            c2_attended = self.c2_attention(x) * attention_routing[:, 1:2]
            c3_c7_attended = self.c3_c7_attention(x) * attention_routing[:, 2:3]

            attended = c1_attended + c2_attended + c3_c7_attended
            return attended, attention_routing

        else:
            # No detector model provided, fall back
            attended = self.c3_c7_attention(x)
            return attended, None


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
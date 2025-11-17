"""
3D scSE Attention Modules for Cervical Spine Segmentation

Based on:
- Roy et al., "Concurrent Spatial and Channel Squeeze & Excitation in 
  Fully Convolutional Networks", MICCAI 2018
- Original 2D implementation: https://github.com/ai-med/squeeze_and_excitation

Extensions:
- Adapted to 3D for volumetric medical imaging
- Added cervical-level-aware soft weighting system for anatomically-informed attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelSELayer3D(nn.Module):
    """
    3D Channel Squeeze-and-Excitation Layer
    
    Adaptively recalibrates channel-wise feature responses by explicitly 
    modeling interdependencies between channels.
    
    Based on: Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
    """
    
    def __init__(self, num_channels, reduction_ratio=2):
        """
        Args:
            num_channels (int): Number of input channels
            reduction_ratio (int): Channel reduction ratio for bottleneck
        """
        super(ChannelSELayer3D, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_tensor):
        """
        Args:
            input_tensor: (batch_size, num_channels, D, H, W)
        Returns:
            output_tensor: (batch_size, num_channels, D, H, W)
        """
        batch_size, num_channels, D, H, W = input_tensor.size()
        
        # Squeeze: Global average pooling
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)
        
        # Excitation: FC -> ReLU -> FC -> Sigmoid
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))
        
        # Scale channels
        output_tensor = input_tensor * fc_out_2.view(batch_size, num_channels, 1, 1, 1)
        return output_tensor


class SpatialSELayer3D(nn.Module):
    """
    3D Spatial Squeeze-and-Excitation Layer
    
    Adaptively recalibrates spatial responses by modeling spatial 
    interdependencies.
    
    Based on: Roy et al., "Concurrent Spatial and Channel Squeeze & Excitation 
    in Fully Convolutional Networks", MICCAI 2018
    """
    
    def __init__(self, num_channels):
        """
        Args:
            num_channels (int): Number of input channels
        """
        super(SpatialSELayer3D, self).__init__()
        self.conv = nn.Conv3d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_tensor):
        """
        Args:
            input_tensor: (batch_size, num_channels, D, H, W)
        Returns:
            output_tensor: (batch_size, num_channels, D, H, W)
        """
        # Squeeze: Channel-wise convolution
        squeeze_tensor = self.sigmoid(self.conv(input_tensor))
        
        # Excitation: Scale spatial locations
        output_tensor = input_tensor * squeeze_tensor
        return output_tensor


class ChannelSpatialSELayer3D(nn.Module):
    """
    3D Concurrent Spatial and Channel Squeeze & Excitation (scSE)
    
    Combines channel and spatial recalibration through element-wise addition.
    
    Based on: Roy et al., "Concurrent Spatial and Channel Squeeze & Excitation 
    in Fully Convolutional Networks", MICCAI 2018, arXiv:1803.02579
    
    Note: Original paper uses addition (not max) for combining cSE and sSE outputs.
    """
    
    def __init__(self, num_channels, reduction_ratio=2):
        """
        Args:
            num_channels (int): Number of input channels
            reduction_ratio (int): Channel reduction ratio for cSE bottleneck
        """
        super(ChannelSpatialSELayer3D, self).__init__()
        self.cSE = ChannelSELayer3D(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer3D(num_channels)
    
    def forward(self, input_tensor):
        """
        Args:
            input_tensor: (batch_size, num_channels, D, H, W)
        Returns:
            output_tensor: (batch_size, num_channels, D, H, W)
        """
        # Concurrent channel and spatial attention (combined via addition)
        cSE_out = self.cSE(input_tensor)
        sSE_out = self.sSE(input_tensor)
        output_tensor = cSE_out + sSE_out  # Element-wise addition (per paper)
        return output_tensor


class CervicalLevelAwareAttention3D(nn.Module):
    """
    Cervical Level-Aware Attention with Soft Weighting
    
    Novel contribution: Applies anatomically-informed attention pathways based on 
    cervical vertebra characteristics:
    - C1 (atlas): Spatial attention only - no vertebral body, unique ring structure
    - C2 (axis): Enhanced scSE - odontoid process (dens) requires strong feature discrimination
    - C3-C7: Standard scSE - similar rectangular vertebral body anatomy
    
    The network learns to identify which cervical level(s) are present in the features
    and applies a soft-weighted combination of appropriate attention mechanisms.
    
    Soft weighting allows:
    1. Graceful handling of classifier uncertainty
    2. Mixed regions (multiple vertebrae in receptive field)
    3. Gradient flow through all pathways during training
    """
    
    def __init__(self, num_channels, reduction_ratio=2):
        """
        Args:
            num_channels (int): Number of input channels
            reduction_ratio (int): Channel reduction ratio for scSE
        """
        super(CervicalLevelAwareAttention3D, self).__init__()
        
        # Learnable vertebra-level classifier
        # Predicts probability distribution over cervical vertebra groups
        self.level_classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # Global context
            nn.Flatten(),
            nn.Linear(num_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # 3 groups: C1, C2, C3-C7
            nn.Softmax(dim=1)  # Soft probabilities (sum to 1)
        )
        
        # C1-specific pathway: Spatial attention only
        # Rationale: C1 (atlas) lacks a vertebral body; spatial structure is key
        self.c1_attention = SpatialSELayer3D(num_channels)
        
        # C2-specific pathway: Enhanced scSE (lower reduction ratio = more capacity)
        # Rationale: C2 (axis) has unique odontoid process requiring strong discrimination
        self.c2_attention = ChannelSpatialSELayer3D(
            num_channels, 
            reduction_ratio=max(1, reduction_ratio // 2)  # Enhanced capacity
        )
        
        # C3-C7 pathway: Standard scSE
        # Rationale: C3-C7 have similar anatomy; standard attention suffices
        self.c3_c7_attention = ChannelSpatialSELayer3D(num_channels, reduction_ratio)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, num_channels, D, H, W) - Feature map from encoder
        
        Returns:
            attended: (batch_size, num_channels, D, H, W) - Attended features
            level_probs: (batch_size, 3) - Predicted vertebra level probabilities [P(C1), P(C2), P(C3-C7)]
        """
        # Step 1: Predict vertebra level probabilities
        level_probs = self.level_classifier(x)  # (batch_size, 3)
        
        # Step 2: Apply all three attention pathways in parallel
        c1_out = self.c1_attention(x)
        c2_out = self.c2_attention(x)
        c3_c7_out = self.c3_c7_attention(x)
        
        # Step 3: Soft-weighted combination based on predicted levels
        # Reshape weights for broadcasting: (B, 3) -> (B, 3, 1, 1, 1)
        weights = level_probs.view(-1, 3, 1, 1, 1)
        
        # Blend outputs: weighted sum across pathways
        attended = (
            weights[:, 0:1] * c1_out +      # C1 contribution
            weights[:, 1:2] * c2_out +      # C2 contribution
            weights[:, 2:3] * c3_c7_out     # C3-C7 contribution
        )
        
        return attended, level_probs


class SimpleCervicalAttention3D(nn.Module):
    """
    Simplified Cervical Attention (Baseline)
    
    Applies standard scSE attention uniformly to all cervical vertebrae.
    Serves as baseline for comparison against level-aware version.
    """
    
    def __init__(self, num_channels, reduction_ratio=2):
        """
        Args:
            num_channels (int): Number of input channels
            reduction_ratio (int): Channel reduction ratio for scSE
        """
        super(SimpleCervicalAttention3D, self).__init__()
        self.attention = ChannelSpatialSELayer3D(num_channels, reduction_ratio)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, num_channels, D, H, W)
        
        Returns:
            attended: (batch_size, num_channels, D, H, W)
            level_probs: None (for interface compatibility)
        """
        attended = self.attention(x)
        return attended, None  # Return None for level_probs (which reverts this to standard scSE)
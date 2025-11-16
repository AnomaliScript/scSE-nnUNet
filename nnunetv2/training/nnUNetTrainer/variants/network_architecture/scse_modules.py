"""
Concurrent Spatial and Channel Squeeze & Excitation for nnUNetv2
Based on Roy et al. MICCAI 2018

Integration: Skip connections (best performance per paper)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelSELayer3D(nn.Module):
    """Channel Squeeze and Excitation for 3D"""
    def __init__(self, num_channels, reduction_ratio=2):
        super().__init__()
        num_channels_reduced = max(num_channels // reduction_ratio, 1)
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c = x.size(0), x.size(1)
        squeeze = F.adaptive_avg_pool3d(x, 1).view(b, c)
        excitation = self.sigmoid(self.fc2(self.relu(self.fc1(squeeze))))
        return x * excitation.view(b, c, 1, 1, 1)


class SpatialSELayer3D(nn.Module):
    """Spatial Squeeze and Excitation for 3D"""
    def __init__(self, num_channels):
        super().__init__()
        self.conv = nn.Conv3d(num_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.sigmoid(self.conv(x))
        return x * attention


class ChannelSpatialSELayer3D(nn.Module):
    """
    scSE: Concurrent Channel and Spatial SE
    This is what you'll use in skip connections
    """
    def __init__(self, num_channels, reduction_ratio=2):
        super().__init__()
        self.cSE = ChannelSELayer3D(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer3D(num_channels)

    def forward(self, x):
        return torch.max(self.cSE(x), self.sSE(x))


class CervicalLevelAwareAttention3D(nn.Module):
    """
    YOUR NOVEL CONTRIBUTION
    Level-specific attention for C1, C2, C3-C7
    """
    def __init__(self, num_channels, reduction_ratio=2):
        super().__init__()
        self.c1_attention = SpatialSELayer3D(num_channels)
        self.c2_attention = ChannelSpatialSELayer3D(num_channels, reduction_ratio)
        self.c3_7_attention = ChannelSpatialSELayer3D(num_channels, reduction_ratio)
        
        self.level_predictor = nn.Sequential(
            nn.Conv3d(num_channels, num_channels // 4, kernel_size=1),
            nn.InstanceNorm3d(num_channels // 4),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(num_channels // 4, 3, kernel_size=1),
        )
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        level_logits = self.level_predictor(x)
        level_probs = self.softmax(level_logits)
        
        c1_out = self.c1_attention(x) * level_probs[:, 0:1]
        c2_out = self.c2_attention(x) * level_probs[:, 1:2]
        c3_7_out = self.c3_7_attention(x) * level_probs[:, 2:3]
        
        return c1_out + c2_out + c3_7_out, level_probs
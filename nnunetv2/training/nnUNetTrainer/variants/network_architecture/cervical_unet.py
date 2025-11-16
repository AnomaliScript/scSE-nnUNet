"""
Custom U-Net with scSE attention in skip connections
For cervical spine segmentation
"""

from dynamic_network_architectures.architectures.unet import PlainConvUNet, UNetDecoder
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from torch import nn
import numpy as np
from typing import Union, Type, List, Tuple
import torch

# Import our scSE modules
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from scse_modules import ChannelSpatialSELayer3D, CervicalLevelAwareAttention3D


class PlainConvUNetWithAttention(PlainConvUNet): # <- Inheriting from nnUNet's PlainConvUNet
    """
    PlainConvUNet with scSE attention in skip connections
    
    This modifies the standard nnUNet to add attention gates
    in the skip connections (encoder to decoder)
    """
    
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[nn.Module],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[nn.Module]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False,
                 use_cervical_attention: bool = False  # NEW PARAMETER
                 ):
        
        # Call parent constructor
        super().__init__(
            input_channels, n_stages, features_per_stage, conv_op, kernel_sizes,
            strides, n_conv_per_stage, num_classes, n_conv_per_stage_decoder,
            conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs,
            nonlin, nonlin_kwargs, deep_supervision, nonlin_first
        )
        
        # Add attention modules to skip connections
        self.use_cervical_attention = use_cervical_attention
        
        if use_cervical_attention:
            # Create attention modules for each skip connection
            self.attention_modules = nn.ModuleList()
            
            # features_per_stage can be int or list
            if isinstance(features_per_stage, int):
                features = [features_per_stage * (2 ** i) for i in range(n_stages)]
            else:
                features = list(features_per_stage)
            
            # Add attention for each encoder stage (except last, no skip there)
            for i in range(n_stages - 1):
                num_channels = features[i]
                # Use cervical-specific attention
                attention = CervicalLevelAwareAttention3D(num_channels, reduction_ratio=2)
                self.attention_modules.append(attention)
            
            print(f"✅ Added {len(self.attention_modules)} cervical attention modules to skip connections")
    
    def forward(self, x):
        """
        Modified forward pass with attention in skip connections
        """
        # Encoder forward pass
        skips = []
        for s in self.encoder.stages:
            x = s(x)
            skips.append(x)
        
        # Apply attention to skip connections (except last one - bottleneck)
        if self.use_cervical_attention and hasattr(self, 'attention_modules'):
            for i in range(len(self.attention_modules)):
                # Apply cervical attention to this skip
                skips[i], _ = self.attention_modules[i](skips[i])
        
        # Decoder forward pass with attended skips
        return self.decoder(skips)
# Custom trainer that uses cervical attention U-Net

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
import torch
from torch import nn
from typing import Union, Tuple, List
import sys
from pathlib import Path

# Import our custom U-Net
sys.path.append(str(Path(__file__).parent))
from cervical_residual_unet import ResidualEncoderUNetWithAttention


class nnUNetTrainerCervicalAttentionResEnc(nnUNetTrainer):
    # Trainer using cervical level-aware attention in skip connections (versus happening in encoding or decoding)
    
    @staticmethod
    def build_network_architecture(plans_manager: PlansManager,
                                     dataset_json: dict,
                                     configuration_manager: ConfigurationManager,
                                     num_input_channels: int,
                                     enable_deep_supervision: bool = True) -> nn.Module:

        # Get architecture parameters from plans
        num_stages = len(configuration_manager.conv_kernel_sizes)
        
        # Determine conv op (2D or 3D)
        dim = len(configuration_manager.patch_size)
        conv_op = nn.Conv2d if dim == 2 else nn.Conv3d
        
        # Get norm op
        norm_op = nn.InstanceNorm2d if dim == 2 else nn.InstanceNorm3d
        
        # Build network with attention
        model = ResidualEncoderUNetWithAttention(
            input_channels=num_input_channels,
            n_stages=num_stages,
            features_per_stage=[min(configuration_manager.UNet_base_num_features * 2 ** i,
                                   configuration_manager.unet_max_num_features)
                               for i in range(num_stages)],
            conv_op=conv_op,
            kernel_sizes=configuration_manager.conv_kernel_sizes,
            strides=configuration_manager.pool_op_kernel_sizes,
            n_conv_per_stage=configuration_manager.n_conv_per_stage_encoder,
            num_classes=dataset_json["labels"].__len__(),
            n_conv_per_stage_decoder=configuration_manager.n_conv_per_stage_decoder,
            conv_bias=True,
            norm_op=norm_op,
            norm_op_kwargs={'eps': 1e-5, 'affine': True},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True},
            deep_supervision=enable_deep_supervision,
            use_cervical_attention=True  # ENABLE ATTENTION
        )
        
        print("Built cervical attention U-Net!")
        return model
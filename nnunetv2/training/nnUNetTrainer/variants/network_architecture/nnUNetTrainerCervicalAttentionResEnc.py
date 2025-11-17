# Trainer for ResidualEncoderUNet with Cervical Attention

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
import torch
from torch import nn
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
import sys
from pathlib import Path

# Import attention modules
sys.path.append(str(Path(__file__).parent))
from scse_modules import CervicalLevelAwareAttention3D


class nnUNetTrainerCervicalAttentionResEnc(nnUNetTrainer):
    """
    Trainer using ResidualEncoderUNet + Cervical Attention
    """
    
    def __init__(self, plans: dict, configuration: str, fold: int, 
                 dataset_json: dict, unpack_dataset: bool = True, 
                 device: torch.device = torch.device('cuda')):
        
        super().__init__(plans, configuration, fold, dataset_json, 
                        unpack_dataset, device)
        
        print("ResEncL + Cervical Attention Trainer Initialized")
    
    def build_network_architecture(self, 
                                   architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: tuple,
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """
        Override to add cervical attention to ResidualEncoderUNet
        
        This signature matches the parent class exactly!
        """
        
        # First, build the base ResidualEncoderUNet using parent's method
        network = super().build_network_architecture(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            enable_deep_supervision
        )
        
        # Now add attention modules to the encoder outputs (skip connections)
        print("Adding cervical attention to skip connections...")
        
        # Get number of stages and features from the network
        n_stages = len(network.encoder.stages)
        
        # Create attention modules
        attention_modules = nn.ModuleList()
        
        for i in range(n_stages - 1):  # No skip from bottleneck
            # Get number of channels at this stage
            # This depends on how features_per_stage was configured
            try:
                # Try to get from the encoder stage output channels
                stage = network.encoder.stages[i]
                # The last block in the stage should have the output channels
                if hasattr(stage, 'blocks'):
                    num_channels = stage.blocks[-1].conv.out_channels
                elif hasattr(stage, 'output_channels'):
                    num_channels = stage.output_channels
                else:
                    # Fallback: estimate from arch_init_kwargs
                    num_channels = arch_init_kwargs['features_per_stage'][i]
                
                attention = CervicalLevelAwareAttention3D(num_channels, reduction_ratio=2)
                attention_modules.append(attention)
                print(f"  Added attention to stage {i} ({num_channels} channels)")
                
            except Exception as e:
                print(f"  Warning: Could not add attention to stage {i}: {e}")
                attention_modules.append(None)
        
        # Store attention modules in the network
        network.attention_modules = attention_modules
        
        # Wrap the decoder's forward pass to apply attention
        original_decoder_forward = network.decoder.forward
        
        def decoder_forward_with_attention(skips):
            # Apply attention to skips before passing to decoder
            attended_skips = []
            for i, skip in enumerate(skips):
                if i < len(network.attention_modules) and network.attention_modules[i] is not None:
                    attended_skip, _ = network.attention_modules[i](skip)
                    attended_skips.append(attended_skip)
                else:
                    attended_skips.append(skip)
            
            # Call original decoder with attended skips
            return original_decoder_forward(attended_skips)
        
        # Replace decoder forward
        network.decoder.forward = decoder_forward_with_attention
        
        print(f"ResEncL + Cervical Attention network built successfully!")
        
        return network
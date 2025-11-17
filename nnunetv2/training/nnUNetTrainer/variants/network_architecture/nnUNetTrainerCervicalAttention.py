# Custom trainer that uses cervical attention U-Net

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from torch import nn
from typing import Union, List, Tuple
import sys
from pathlib import Path

# Import attention modules ONLY
sys.path.append(str(Path(__file__).parent))
from scse_modules import CervicalLevelAwareAttention3D


class nnUNetTrainerCervicalAttention(nnUNetTrainer):
    """
    Trainer using PlainConvUNet + Cervical Attention
    """
    
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """
        Build network with cervical attention added to skip connections
        """
        # Build the base network from plans
        network = get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision
        )
        
        print("🎯 Adding cervical attention to skip connections...")
        
        # Add attention modules
        n_stages = len(network.encoder.stages)
        attention_modules = nn.ModuleList()
        
        features_per_stage = arch_init_kwargs.get('features_per_stage', [])
        
        for i in range(n_stages - 1):
            try:
                num_channels = features_per_stage[i] if i < len(features_per_stage) else 32 * (2 ** i)
                attention = CervicalLevelAwareAttention3D(num_channels, reduction_ratio=2)
                attention_modules.append(attention)
                print(f"  ✅ Stage {i}: {num_channels} channels")
            except Exception as e:
                print(f"  ⚠️ Stage {i}: Failed - {e}")
                attention_modules.append(None)
        
        network.attention_modules = attention_modules
        
        # Wrap encoder forward to apply attention
        def encoder_forward_with_attention(x):
            skips = []
            for s in network.encoder.stages:
                x = s(x)
                skips.append(x)
            
            attended_skips = []
            for i, skip in enumerate(skips):
                if i < len(network.attention_modules) and network.attention_modules[i] is not None:
                    attended_skip, _ = network.attention_modules[i](skip)
                    attended_skips.append(attended_skip)
                else:
                    attended_skips.append(skip)
            
            return attended_skips
        
        network.encoder.forward = encoder_forward_with_attention
        
        print("PlainConv + Cervical Attention network built!")
        
        return network

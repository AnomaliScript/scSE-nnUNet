from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from torch import nn
from typing import Union, List, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from scse_modules import CervicalLevelAwareAttention3D


class nnUNetTrainerCervicalAttention(nnUNetTrainer):
    """
    Trainer using PlainConvUNet + Cervical Attention in skip connections
    """
    
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """
        Build network with cervical attention in skip connections
        """
        # Build base network
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
        
        # Create attention modules for each skip connection
        n_stages = len(network.encoder.stages)
        features_per_stage = arch_init_kwargs.get('features_per_stage', [])
        
        attention_modules = nn.ModuleList()
        for i in range(n_stages - 1):  # No skip from bottleneck
            try:
                num_channels = features_per_stage[i] if i < len(features_per_stage) else 32 * (2 ** i)
                attention = CervicalLevelAwareAttention3D(num_channels, reduction_ratio=2)
                attention_modules.append(attention)
                print(f"  ✅ Stage {i}: {num_channels} channels")
            except Exception as e:
                print(f"  ⚠️ Stage {i}: Failed - {e}")
                attention_modules.append(None)
        
        # Store attention modules
        network.attention_modules = attention_modules
        
        # CORRECTED: Wrap DECODER forward, not encoder!
        original_decoder_forward = network.decoder.forward
        
        def decoder_forward_with_attention(skips):
            """
            Apply attention to skips before decoder processes them
            skips: List of encoder outputs
            """
            attended_skips = []
            for i, skip in enumerate(skips[:-1]):  # All except bottleneck
                if i < len(network.attention_modules) and network.attention_modules[i] is not None:
                    attended_skip, _ = network.attention_modules[i](skip)
                    attended_skips.append(attended_skip)
                else:
                    attended_skips.append(skip)
            
            # Add bottleneck (last skip) without attention
            attended_skips.append(skips[-1])
            
            # Call original decoder with attended skips
            return original_decoder_forward(attended_skips)
        
        # Replace decoder forward
        network.decoder.forward = decoder_forward_with_attention
        
        print("✅ PlainConv + Cervical Attention network built!")
        
        return network
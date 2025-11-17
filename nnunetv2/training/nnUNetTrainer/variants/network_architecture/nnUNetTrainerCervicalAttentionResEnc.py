"""
nnU-Net Trainer with Cervical Level-Aware Attention

Custom trainer that integrates anatomically-informed attention mechanisms into 
nnU-Net's ResidualEncoderUNet architecture for cervical spine segmentation.

Key modifications:
1. Adds CervicalLevelAwareAttention3D modules to skip connections
2. Applies attention to encoder outputs before passing to decoder
3. Maintains compatibility with nnU-Net's training pipeline

Author: Brandon K (@AnomaliScript)
Date: 2025
"""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from torch import nn
from typing import Union, List, Tuple
import sys
from pathlib import Path

# Import custom attention modules
sys.path.append(str(Path(__file__).parent))
from scse_modules import CervicalLevelAwareAttention3D


class nnUNetTrainerCervicalAttentionResEnc(nnUNetTrainer):
    """
    nnU-Net Trainer with Cervical Level-Aware Attention for ResidualEncoderUNet
    
    Extends the base nnUNetTrainer to integrate anatomically-informed attention
    mechanisms into the network architecture's skip connections.
    """
    
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """
        Build network architecture with integrated cervical attention.
        
        Overrides the base class method to add attention modules to skip connections
        while maintaining full compatibility with nnU-Net's planning and configuration.
        
        Args:
            architecture_class_name: Fully qualified class name (e.g., "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet")
            arch_init_kwargs: Architecture initialization arguments from plans
            arch_init_kwargs_req_import: Keys in arch_init_kwargs that need import resolution
            num_input_channels: Number of input modalities
            num_output_channels: Number of segmentation classes
            enable_deep_supervision: Whether to use deep supervision
        
        Returns:
            Modified network with attention modules integrated into skip connections
        """
        # Step 1: Build base network using nnU-Net's standard method
        network = get_network_from_plans(
            architecture_class_name,
            arch_init_kwargs,
            arch_init_kwargs_req_import,
            num_input_channels,
            num_output_channels,
            allow_init=True,
            deep_supervision=enable_deep_supervision
        )
        
        print("🎯 Building ResidualEncoderUNet with Cervical Level-Aware Attention...")
        
        # Step 2: Create attention modules for each encoder stage
        n_stages = len(network.encoder.stages)
        features_per_stage = arch_init_kwargs.get('features_per_stage', [])
        
        attention_modules = nn.ModuleList()
        
        # Add attention for each skip connection (all stages except bottleneck)
        for i in range(n_stages - 1):
            try:
                # Get number of channels at this stage
                if i < len(features_per_stage):
                    num_channels = features_per_stage[i]
                else:
                    # Fallback: assume doubling pattern
                    num_channels = 32 * (2 ** i)
                
                # Create cervical-aware attention module
                attention = CervicalLevelAwareAttention3D(
                    num_channels=num_channels,
                    reduction_ratio=2
                )
                attention_modules.append(attention)
                
                print(f"  ✅ Stage {i}: Added attention module ({num_channels} channels)")
                
            except Exception as e:
                print(f"  ⚠️ Stage {i}: Failed to add attention - {e}")
                attention_modules.append(None)
        
        # Step 3: Attach attention modules to network
        network.attention_modules = attention_modules
        
        # Step 4: Modify decoder to apply attention to skip connections
        original_decoder_forward = network.decoder.forward
        
        def decoder_forward_with_attention(skips):
            """
            Modified decoder forward pass that applies attention to skip connections.
            
            Args:
                skips: List of encoder outputs [stage0, stage1, ..., bottleneck]
            
            Returns:
                Decoder output (segmentation logits)
            """
            attended_skips = []
            
            # Apply attention to all skips except bottleneck
            for i, skip in enumerate(skips[:-1]):
                if i < len(network.attention_modules) and network.attention_modules[i] is not None:
                    # Apply cervical-aware attention
                    attended_skip, level_probs = network.attention_modules[i](skip)
                    attended_skips.append(attended_skip)
                else:
                    # No attention for this stage (fallback)
                    attended_skips.append(skip)
            
            # Add bottleneck without attention
            attended_skips.append(skips[-1])
            
            # Pass attended skips to original decoder
            return original_decoder_forward(attended_skips)
        
        # Replace decoder's forward method
        network.decoder.forward = decoder_forward_with_attention
        
        print("✅ Cervical Level-Aware Attention integration complete!")
        print(f"   Total attention modules: {len(attention_modules)}")
        print(f"   Architecture: {architecture_class_name.split('.')[-1]}")
        
        return network
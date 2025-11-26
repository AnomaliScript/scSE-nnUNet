"""
 nnU-Net Trainer with Cervical Level-Aware Attention

Custom trainer that integrates anatomically-informed attention mechanisms into
nnU-Net's ResidualEncoderUNet architecture for cervical spine segmentation.

Key modifications:
1. Adds DetectorGuidedCervicalAttention3D modules to skip connections
2. Applies YOLO-guided attention to encoder outputs before passing to decoder
3. Maintains compatibility with nnU-Net's training pipeline

Author: Brandon K (@AnomaliScript)
Date: 2025
"""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetWithYOLOAttention
from torch import nn
from typing import Union, List, Tuple
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
from batchgenerators.utilities.file_and_folder_operations import join, isfile

# Import custom attention modules
sys.path.append(str(Path(__file__).parent))
from scse_modules import DetectorGuidedCervicalAttention3D, generate_yolo_attention_mask


class nnUNetTrainerCervicalAttentionResEnc(nnUNetTrainer):
    """
    nnU-Net Trainer with Cervical Level-Aware Attention for ResidualEncoderUNet

    Extends the base nnUNetTrainer to integrate anatomically-informed attention
    mechanisms into the network architecture's skip connections.

    Automatically pre-generates YOLO attention masks before training starts.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.yolo_model_path = 'runs/vertebra_detector_114/weights/best.pt'

    def initialize(self):
        """Override to pre-generate YOLO masks before training"""
        # Call parent initialization
        super().initialize()

        # Pre-generate YOLO attention masks
        self.pregenerate_yolo_masks()

    def pregenerate_yolo_masks(self):
        """
        Pre-generate YOLO attention masks for all cases in the dataset.

        This runs before training starts, generating masks for all cases
        that don't already have them.
        """
        print("\n" + "=" * 70)
        print("CHECKING FOR YOLO ATTENTION MASKS")
        print("=" * 70)

        # Get preprocessed data folder
        preprocessed_folder = self.preprocessed_dataset_folder

        # Get all case identifiers
        from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetNumpy
        identifiers = nnUNetDatasetNumpy.get_identifiers(preprocessed_folder)

        # Check which cases are missing YOLO masks
        missing_masks = []
        for identifier in identifiers:
            yolo_file = join(preprocessed_folder, identifier + '_yolo_attention.npz')
            if not isfile(yolo_file):
                missing_masks.append(identifier)

        if not missing_masks:
            print(f"✅ All {len(identifiers)} cases already have YOLO masks!")
            print("=" * 70 + "\n")
            return

        print(f"Found {len(missing_masks)}/{len(identifiers)} cases without YOLO masks")
        print(f"Generating masks now (using {self.yolo_model_path})...")
        print("This is a ONE-TIME process. Future training runs will be fast.\n")

        # Generate masks for missing cases
        for identifier in tqdm(missing_masks, desc="Generating YOLO masks"):
            try:
                # Load preprocessed volume
                data_file = join(preprocessed_folder, identifier + '.npz')
                data = np.load(data_file)['data']

                # Generate YOLO attention mask
                attention_mask = generate_yolo_attention_mask(
                    data,
                    yolo_model_path=self.yolo_model_path,
                    conf_threshold=0.25
                )

                # Save mask
                yolo_file = join(preprocessed_folder, identifier + '_yolo_attention.npz')
                np.savez_compressed(yolo_file, attention=attention_mask)

            except Exception as e:
                print(f"\n⚠️  Error generating mask for {identifier}: {e}")
                continue

        print(f"\n✅ All YOLO masks generated and saved!")
        print("=" * 70 + "\n")

    def get_tr_and_val_datasets(self):
        """Override to use custom dataset with YOLO attention masks"""
        # Get parent's datasets
        tr_keys, val_keys = self.do_split()

        # Create datasets with YOLO attention support
        dataset_tr = nnUNetDatasetWithYOLOAttention(
            self.preprocessed_dataset_folder,
            tr_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage
        )

        dataset_val = nnUNetDatasetWithYOLOAttention(
            self.preprocessed_dataset_folder,
            val_keys,
            folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage
        )

        return dataset_tr, dataset_val

    def train_step(self, batch: dict) -> dict:
        """
        Override training step to inject YOLO attention masks into network.

        The dataloader returns 5 items: (data, seg, seg_prev, properties, yolo_attention)
        We need to set the yolo_attention on the network before forward pass.
        """
        # YOLO attention mask should be in batch if our custom dataset is working
        yolo_attention = batch.get('yolo_attention', None)

        # Set attention mask on network (will be read by decoder wrapper)
        if yolo_attention is not None:
            self.network.current_attention_mask = yolo_attention
        else:
            self.network.current_attention_mask = None

        # Call parent's train_step with modified batch
        return super().train_step(batch)

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

                # Create YOLO-guided attention module
                attention = DetectorGuidedCervicalAttention3D(
                    num_channels=num_channels,
                    reduction_ratio=2
                )
                attention_modules.append(attention)

                print(f"  ✅ Stage {i}: Added attention module ({num_channels} channels)")

            except Exception as e:
                print(f"  ⚠️ Stage {i}: Failed to add attention - {e}")
                attention_modules.append(None)

        # Step 3: Attach attention modules and mask placeholder to network
        network.attention_modules = attention_modules
        network.current_attention_mask = None  # Will be set during forward pass

        # Step 4: Modify decoder to apply attention to skip connections
        original_decoder_forward = network.decoder.forward

        def decoder_forward_with_attention(skips):
            """
            Modified decoder forward pass that applies YOLO-guided attention to skip connections.

            Args:
                skips: List of encoder outputs [stage0, stage1, ..., bottleneck]

            Returns:
                Decoder output (segmentation logits)
            """
            # Get attention mask from network attribute (set during forward pass)
            attention_mask = getattr(network, 'current_attention_mask', None)

            attended_skips = []

            # Apply attention to all skips except bottleneck
            for i, skip in enumerate(skips[:-1]):
                if i < len(network.attention_modules) and network.attention_modules[i] is not None:
                    # Apply YOLO-guided attention with mask
                    attended_skip, _ = network.attention_modules[i](skip, attention_mask=attention_mask)
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

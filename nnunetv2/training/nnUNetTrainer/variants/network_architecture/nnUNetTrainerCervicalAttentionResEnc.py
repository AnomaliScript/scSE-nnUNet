"""
nnU-Net Trainer with Cervical Level-Aware Attention (Bottleneck + Decoder)

Custom trainer that integrates anatomically-informed attention mechanisms into
nnU-Net's ResidualEncoderUNet architecture for cervical spine segmentation.

Key modifications:
1. Adds DetectorGuidedCervicalAttention3D to bottleneck (deepest encoder layer)
2. Adds DetectorGuidedCervicalAttention3D to all decoder blocks (boundary refinement)
3. Skip connections remain UNCHANGED for direct gradient flow
4. Class-weighted loss: Gradient weighting for C3-C7 (1.5x for C3-C5, 2.5x for C6-C7)
5. Maintains full compatibility with nnU-Net's training pipeline

NOTE: YOLO vertebra detector integration is currently disabled (Plan B).
      Testing scSE attention alone first.

Author: Brandon K (@AnomaliScript)
Date: 2025
"""

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
# from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetWithYOLOAttention  # YOLO: Plan B
# from nnunetv2.training.dataloading.data_loader import nnUNetDataLoaderWithYOLO  # YOLO: Plan B
from torch import nn
from typing import Union, List, Tuple
import sys
from pathlib import Path
import numpy as np
# from tqdm import tqdm  # YOLO: Plan B
# from batchgenerators.utilities.file_and_folder_operations import join, isfile  # YOLO: Plan B

# Import custom attention modules
sys.path.append(str(Path(__file__).parent))
from scse_modules import DetectorGuidedCervicalAttention3D  # , generate_yolo_attention_mask  # YOLO: Plan B


class nnUNetTrainerCervicalAttentionResEnc(nnUNetTrainer):
    """
    nnU-Net Trainer with Cervical Level-Aware Attention for ResidualEncoderUNet

    Extends the base nnUNetTrainer to integrate scSE attention mechanisms
    into the bottleneck and decoder blocks.

    YOLO integration is currently disabled (testing scSE attention alone).
    """

    def initialize(self):
        # YOLO: Plan B (commented out)
        # nnunetv2_root = Path(__file__).resolve().parents[4]
        # self.yolo_model_path = str(nnunetv2_root / 'yolo_models' / 'vertebra_detector_114.pt')

        # Reduce patch size BEFORE parent initialization builds the network
        if not self.was_initialized:
            # Get original patch size from configuration
            original_patch_size = self.configuration_manager.configuration['patch_size']

            # OPTION 1: Current 96³ (AS-IS)
            reduced_patch_size = [96, 96, 96]

            # OPTION 2: Hard-set to 64³
            # reduced_patch_size = [64, 64, 64]

            # # OPTION 3: Try 128³
            # reduced_patch_size = [128, 128, 128]

            # # OPTION 4: Anisotropic 160×96×96 (follows spine anatomy)
            # reduced_patch_size = [160, 96, 96]

            print(f"   Patch size: {original_patch_size} -> {reduced_patch_size}")

        # well well well you can also edit batch size here too
        self.configuration_manager.configuration['batch_size'] = 22

        # Call parent initialization
        super().initialize()

        # YOLO: Plan B (commented out)
        # self.pregenerate_yolo_masks()

    # def pregenerate_yolo_masks(self):
    #     """
    #     Pre-generate YOLO attention masks for all cases in the dataset.

    #     This runs before training starts, generating masks for all cases
    #     that don't already have them.
    #     """
    #     print("\n" + "=" * 70)
    #     print("CHECKING FOR YOLO ATTENTION MASKS")
    #     print("=" * 70)

    #     # Get preprocessed data folder
    #     preprocessed_folder = self.preprocessed_dataset_folder

    #     # Create yolo_bbox as sibling to nnUNetPlans_3d_fullres
    #     yolo_folder = join(self.preprocessed_dataset_folder_base, 'yolo_bbox')
    #     from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
    #     maybe_mkdir_p(yolo_folder)

    #     # Get all case identifiers
    #     from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetNumpy
    #     identifiers = nnUNetDatasetNumpy.get_identifiers(preprocessed_folder)

    #     # Check which cases are missing YOLO masks
    #     missing_masks = []
    #     for identifier in identifiers:
    #         yolo_file = join(yolo_folder, identifier + '_yolo_attention.npz')
    #         if not isfile(yolo_file):
    #             missing_masks.append(identifier)

    #     if not missing_masks:
    #         print(f"✅ All {len(identifiers)} cases already have YOLO masks!")
    #         print(f"📁 Stored in: {yolo_folder}")
    #         print("=" * 70 + "\n")
    #         return

    #     print(f"Found {len(missing_masks)}/{len(identifiers)} cases without YOLO masks")
    #     print(f"Generating masks now (using {self.yolo_model_path})...")
    #     print(f"📁 Saving to: {yolo_folder}")
    #     print("This is a ONE-TIME process. Future training runs will be fast.\n")

    #     # Generate masks for missing cases
    #     for identifier in tqdm(missing_masks, desc="Generating YOLO masks"):
    #         try:
    #             # Load preprocessed volume
    #             data_file = join(preprocessed_folder, identifier + '.npz')
    #             data = np.load(data_file)['data']

    #             # Generate YOLO attention mask
    #             attention_mask = generate_yolo_attention_mask(
    #                 data,
    #                 yolo_model_path=self.yolo_model_path,
    #                 conf_threshold=0.25
    #             )

    #             # Save mask in yolo_bbox subfolder
    #             yolo_file = join(yolo_folder, identifier + '_yolo_attention.npz')
    #             np.savez_compressed(yolo_file, attention=attention_mask)

    #         except Exception as e:
    #             print(f"\n⚠️  Error generating mask for {identifier}: {e}")
    #             continue

    #     print(f"\n✅ All YOLO masks generated and saved!")
    #     print(f"📁 Location: {yolo_folder}")
    #     print("=" * 70 + "\n")

    # YOLO: Plan B (commented out)
    # def get_tr_and_val_datasets(self):
    #     """Use standard nnU-Net dataset (YOLO attention disabled for now)"""
    #     # TODO: Fix nnUNetDatasetWithYOLOAttention to work with .b2nd format
    #     # For now, use standard dataset without YOLO attention
    #     print("\n⚠️ YOLO attention temporarily disabled - using standard nnU-Net dataset\n")
    #     return super().get_tr_and_val_datasets()

    # YOLO: Plan B (commented out)
    # def get_dataloaders(self):
    #     """Use standard nnU-Net dataloader (YOLO dataloader disabled for now)"""
    #     return super().get_dataloaders()

    def _build_loss(self):
        """
        Override to add class-weighted loss that penalizes C6/C7 errors more.

        Class weights:
        - Background (0): 1.0 (standard)
        - C1-C5 (1-5): 1.0 (standard)
        - C6 (6): 2.0 (2x penalty for errors)
        - C7 (7): 2.0 (2x penalty for errors)

        This helps the model focus on the most challenging vertebrae (C6/C7)
        which are often harder to segment due to proximity and anatomical variation.
        """
        from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
        from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
        from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
        import torch

        # Create class weights: prioritize lower cervical vertebrae (clinically critical)
        # num_classes = 8 (background + C1-C7)
        num_classes = len(self.label_manager.foreground_labels) + 1  # +1 for background

        class_weights = torch.ones(num_classes, dtype=torch.float32, device=self.device)
        # C1-C2: Standard weight (easier anatomy - atlas/axis)
        # C3-C5: Moderate weight (mid-cervical - important but less challenging)
        class_weights[3] = 1.5  # C3: 1.5x weight
        class_weights[4] = 1.5  # C4: 1.5x weight
        class_weights[5] = 1.5  # C5: 1.5x weight
        # C6-C7: High weight (hardest to segment, most clinically critical)
        class_weights[6] = 2.5  # C6: 2.5x weight (proximity to shoulders)
        class_weights[7] = 2.5  # C7: 2.5x weight (most challenging)

        print(f"\n🎯 Class-weighted loss configured:")
        print(f"   C1-C2: 1.0x weight (standard - distinct anatomy)")
        print(f"   C3-C5: 1.5x weight (moderate - mid-cervical)")
        print(f"   C6-C7: 2.5x weight (high - challenging + critical)")
        print(f"   Device: {self.device}")

        # Build DC + CE loss with class weights for CE component
        loss = DC_and_CE_loss(
            {'batch_dice': self.configuration_manager.batch_dice,
             'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp},
            {'weight': class_weights},  # Add class weights to CE loss
            weight_ce=1,
            weight_dice=1,
            ignore_label=self.label_manager.ignore_label,
            dice_class=MemoryEfficientSoftDiceLoss
        )

        if self._do_i_compile():
            loss.dc = torch.compile(loss.dc)

        # Deep supervision weights
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            if self.is_ddp and self.batch_size == 1:
                weights[-1] = 0
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)

        return loss

    # YOLO: Plan B (commented out)
    # def train_step(self, batch: dict) -> dict:
    #     """
    #     Override training step to inject YOLO attention masks into network.
    #
    #     The dataloader returns 5 items: (data, seg, seg_prev, properties, yolo_attention)
    #     We need to set the yolo_attention on the network before forward pass.
    #     """
    #     # YOLO attention mask should be in batch if our custom dataset is working
    #     yolo_attention = batch.get('yolo_attention', None)
    #
    #     # Set attention mask on network (will be read by decoder wrapper)
    #     if yolo_attention is not None:
    #         self.network.current_attention_mask = yolo_attention
    #     else:
    #         self.network.current_attention_mask = None
    #
    #     # Call parent's train_step with modified batch
    #     return super().train_step(batch)

    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """
        Build network architecture with cervical attention in BOTTLENECK + DECODER.

        Applies anatomically-informed scSE attention at:
        1. Bottleneck (deepest encoder layer) - refine most abstract features
        2. Decoder blocks (after upsampling + concatenation) - boundary refinement

        Skip connections are left UNCHANGED for direct gradient flow.

        Args:
            architecture_class_name: Fully qualified class name (e.g., "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet")
            arch_init_kwargs: Architecture initialization arguments from plans
            arch_init_kwargs_req_import: Keys in arch_init_kwargs that need import resolution
            num_input_channels: Number of input modalities
            num_output_channels: Number of segmentation classes
            enable_deep_supervision: Whether to use deep supervision

        Returns:
            Modified network with bottleneck + decoder attention
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

        print("🎯 Building ResidualEncoderUNet with Bottleneck + Decoder Cervical Attention...")

        # Step 2: Create attention modules for BOTTLENECK and DECODER blocks
        n_stages = len(network.encoder.stages)
        n_decoder_stages = len(network.decoder.stages)
        features_per_stage = arch_init_kwargs.get('features_per_stage', [])

        # Bottleneck attention (deepest encoder layer)
        print("\n📌 BOTTLENECK ATTENTION:")
        try:
            # Bottleneck is the last encoder stage
            bottleneck_channels = features_per_stage[-1] if len(features_per_stage) > 0 else 320
            bottleneck_attention = DetectorGuidedCervicalAttention3D(
                num_channels=bottleneck_channels,
                reduction_ratio=2
            )
            print(f"  ✅ Bottleneck: {bottleneck_channels} channels")
        except Exception as e:
            print(f"  ⚠️ Bottleneck: Failed - {e}")
            bottleneck_attention = None

        # Decoder block attention (after concatenation + conv)
        decoder_attention_modules = nn.ModuleList()

        print("\n📌 DECODER BLOCK ATTENTION:")
        for s in range(n_decoder_stages):
            try:
                decoder_stage = network.decoder.stages[s]

                # Get output channels from the last conv in this decoder stage
                if hasattr(decoder_stage, 'blocks') and len(decoder_stage.blocks) > 0:
                    last_block = decoder_stage.blocks[-1]
                    if hasattr(last_block, 'conv') and hasattr(last_block.conv, 'out_channels'):
                        num_channels = last_block.conv.out_channels
                    else:
                        # Fallback: use features_per_stage in reverse
                        encoder_stage_idx = n_stages - 2 - s
                        num_channels = features_per_stage[encoder_stage_idx] if encoder_stage_idx >= 0 else 32
                else:
                    encoder_stage_idx = n_stages - 2 - s
                    num_channels = features_per_stage[encoder_stage_idx] if encoder_stage_idx >= 0 else 32

                attention = DetectorGuidedCervicalAttention3D(
                    num_channels=num_channels,
                    reduction_ratio=2
                )
                decoder_attention_modules.append(attention)
                print(f"  ✅ Decoder {s}: {num_channels} channels")

            except Exception as e:
                print(f"  ⚠️ Decoder {s}: Failed - {e}")
                decoder_attention_modules.append(None)

        # Step 3: Attach attention modules and mask placeholder to network
        network.bottleneck_attention = bottleneck_attention
        network.decoder_attention_modules = decoder_attention_modules
        network.current_attention_mask = None

        # Step 4: Replace decoder forward to inject attention
        # We need to reimplement the decoder logic (not call original) to inject
        # attention after each decoder block convolution
        import torch

        def decoder_forward_with_bottleneck_decoder_attention(skips):
            """
            Modified decoder with bottleneck + decoder attention:
            1. Apply attention to bottleneck (deepest features)
            2. Apply attention to each decoder block (boundary refinement)
            3. Skip connections pass through UNCHANGED

            Args:
                skips: List of encoder outputs [stage0, stage1, ..., bottleneck]

            Returns:
                Decoder output (segmentation logits)
            """
            attention_mask = getattr(network, 'current_attention_mask', None)

            # PHASE 1: Apply attention to BOTTLENECK only
            bottleneck = skips[-1]
            if network.bottleneck_attention is not None:
                bottleneck, _ = network.bottleneck_attention(bottleneck, attention_mask=attention_mask)

            # Skip connections pass through unchanged
            modified_skips = skips[:-1] + [bottleneck]

            # PHASE 2: Decoder forward with attention on decoder blocks
            lres_input = modified_skips[-1]
            seg_outputs = []

            for s in range(len(network.decoder.stages)):
                # Upsample
                x = network.decoder.transpconvs[s](lres_input)

                # Concatenate with skip (unmodified)
                x = torch.cat((x, modified_skips[-(s+2)]), 1)

                # Decoder convolution block
                x = network.decoder.stages[s](x)

                # APPLY DECODER ATTENTION
                if s < len(network.decoder_attention_modules) and network.decoder_attention_modules[s] is not None:
                    x, _ = network.decoder_attention_modules[s](x, attention_mask=attention_mask)

                # Deep supervision
                if network.decoder.deep_supervision:
                    seg_outputs.append(network.decoder.seg_layers[s](x))
                elif s == (len(network.decoder.stages) - 1):
                    seg_outputs.append(network.decoder.seg_layers[-1](x))

                lres_input = x

            # Invert seg outputs (largest first)
            seg_outputs = seg_outputs[::-1]

            if not network.decoder.deep_supervision:
                return seg_outputs[0]
            else:
                return seg_outputs

        # Replace decoder's forward method
        network.decoder.forward = decoder_forward_with_bottleneck_decoder_attention

        print("\n✅ Bottleneck + Decoder Cervical Attention integration complete!")
        print(f"   Bottleneck attention: {'✓' if bottleneck_attention else '✗'}")
        print(f"   Decoder attention modules: {len(decoder_attention_modules)}")
        print(f"   Skip connections: UNCHANGED (direct gradient flow)")
        print(f"   Architecture: {architecture_class_name.split('.')[-1]}")

        return network

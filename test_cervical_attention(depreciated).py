#!/usr/bin/env python
"""
Quick diagnostic script to test cervical attention implementation.

Usage:
    python test_cervical_attention.py <dataset_id> <preprocessed_case_name>

Example:
    python test_cervical_attention.py 001 RSNA_001
"""

import sys
import numpy as np
from pathlib import Path
from batchgenerators.utilities.file_and_folder_operations import join

def test_yolo_mask_generation(dataset_id, case_name):
    """Test YOLO mask generation on a single case"""
    print("\n" + "="*70)
    print("TEST 1: YOLO Mask Generation")
    print("="*70)

    # Import the function
    from nnunetv2.training.nnUNetTrainer.variants.network_architecture.scse_modules import generate_yolo_attention_mask

    # Paths
    preprocessed_dir = f"nnUNet_preprocessed/Dataset{dataset_id}_*/nnUNetPlans_3d_fullres"
    import glob
    preprocessed_dir = glob.glob(preprocessed_dir)[0]
    data_file = join(preprocessed_dir, f"{case_name}.npz")
    yolo_model = "nnunetv2/yolo_models/vertebra_detector_114.pt"

    print(f"Loading: {data_file}")
    data = np.load(data_file)['data']
    print(f"Data shape: {data.shape}")

    print(f"\nGenerating YOLO mask using: {yolo_model}")
    mask = generate_yolo_attention_mask(data, yolo_model_path=yolo_model)

    # Verify
    print(f"\n✓ Mask shape: {mask.shape} (expected: (4, D, H, W))")
    print(f"✓ Mask dtype: {mask.dtype} (expected: float32)")

    # Check normalization
    voxel_sums = mask.sum(axis=0)
    print(f"✓ Sum per voxel - mean: {voxel_sums.mean():.6f}, std: {voxel_sums.std():.6f} (expected: ~1.0)")

    # Channel statistics
    print(f"\n  Channel 0 (C1):     coverage={100*(mask[0]>0).mean():.1f}%, max_conf={mask[0].max():.3f}")
    print(f"  Channel 1 (C2):     coverage={100*(mask[1]>0).mean():.1f}%, max_conf={mask[1].max():.3f}")
    print(f"  Channel 2 (C3-C7):  coverage={100*(mask[2]>0).mean():.1f}%, max_conf={mask[2].max():.3f}")
    print(f"  Channel 3 (BG):     coverage={100*(mask[3]>0).mean():.1f}%")

    return mask


def test_dataset_loading(dataset_id, case_name):
    """Test dataset loading with YOLO masks"""
    print("\n" + "="*70)
    print("TEST 2: Dataset Loading")
    print("="*70)

    from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetWithYOLOAttention
    import glob

    preprocessed_dir = f"nnUNet_preprocessed/Dataset{dataset_id}_*/nnUNetPlans_3d_fullres"
    preprocessed_dir = glob.glob(preprocessed_dir)[0]

    dataset = nnUNetDatasetWithYOLOAttention(preprocessed_dir, [case_name])

    print(f"Loading case: {case_name}")
    case_data = dataset.load_case(case_name)

    if len(case_data) == 5:
        data, seg, seg_prev, props, yolo = case_data
        print(f"✓ Dataset returns 5 items (includes YOLO mask)")
        print(f"✓ Data shape: {data.shape}")
        print(f"✓ Seg shape: {seg.shape}")
        print(f"✓ YOLO shape: {yolo.shape}")
        print(f"✓ YOLO dtype: {yolo.dtype}")
        return True
    else:
        print(f"✗ Dataset returns {len(case_data)} items (expected 5)")
        return False


def test_attention_module():
    """Test attention module forward pass"""
    print("\n" + "="*70)
    print("TEST 3: Attention Module Forward Pass")
    print("="*70)

    import torch
    from nnunetv2.training.nnUNetTrainer.variants.network_architecture.scse_modules import DetectorGuidedCervicalAttention3D

    # Create dummy inputs
    batch_size, channels, D, H, W = 2, 32, 32, 64, 64
    x = torch.randn(batch_size, channels, D, H, W).cuda()

    # Create dummy 4-channel mask
    mask = np.random.rand(4, D, H, W).astype(np.float32)
    mask = mask / mask.sum(axis=0, keepdims=True)  # Normalize

    print(f"Input shape: {x.shape}")
    print(f"Mask shape: {mask.shape}")

    # Create attention module
    attention_module = DetectorGuidedCervicalAttention3D(num_channels=channels).cuda()

    # Forward pass
    with torch.no_grad():
        output, routing = attention_module(x, attention_mask=mask)

    print(f"✓ Output shape: {output.shape} (expected: {x.shape})")
    print(f"✓ Routing shape: {routing.shape} (expected: ({batch_size}, 3, {D}, {H}, {W}))")
    print(f"✓ Forward pass successful!")

    return True


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    dataset_id = sys.argv[1]
    case_name = sys.argv[2]

    try:
        # Run tests
        mask = test_yolo_mask_generation(dataset_id, case_name)
        test_dataset_loading(dataset_id, case_name)
        test_attention_module()

        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\nYou can now run full training with:")
        print(f"  nnUNetv2_train {dataset_id} 3d_fullres 0 -tr nnUNetTrainerCervicalAttentionResEnc")

    except Exception as e:
        print("\n" + "="*70)
        print("❌ TEST FAILED")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

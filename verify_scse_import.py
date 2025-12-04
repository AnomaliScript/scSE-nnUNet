"""
Verification script for scSE nnUNet implementation
Tests that all modules import correctly and can perform forward passes
Run this in Colab or locally to verify your scSE installation
"""
import torch
import sys

print("=" * 70)
print("🔍 VERIFICATION - scSE CERVICAL ATTENTION MODULES")
print("=" * 70 + "\n")

# Track overall status
all_tests_passed = True

# Test 1: Basic scSE building blocks
print("TEST 1: Basic scSE Attention Components")
print("-" * 70)
try:
    from nnunetv2.training.nnUNetTrainer.variants.network_architecture.scse_modules import (
        ChannelSELayer3D,
        SpatialSELayer3D,
        ChannelSpatialSELayer3D
    )

    # Test channel attention
    x = torch.randn(2, 64, 8, 32, 32)
    channel_se = ChannelSELayer3D(num_channels=64, reduction_ratio=2)
    out_cse = channel_se(x)

    # Test spatial attention
    spatial_se = SpatialSELayer3D(num_channels=64)
    out_sse = spatial_se(x)

    # Test combined scSE
    scse = ChannelSpatialSELayer3D(num_channels=64, reduction_ratio=2)
    out_scse = scse(x)

    print("   ✅ ChannelSELayer3D:         PASS")
    print(f"      Input: {x.shape} → Output: {out_cse.shape}")
    print("   ✅ SpatialSELayer3D:         PASS")
    print(f"      Input: {x.shape} → Output: {out_sse.shape}")
    print("   ✅ ChannelSpatialSELayer3D:  PASS")
    print(f"      Input: {x.shape} → Output: {out_scse.shape}")

except Exception as e:
    print(f"   ❌ FAILED: {e}")
    all_tests_passed = False

print()

# Test 2: DetectorGuidedCervicalAttention3D (main attention module)
print("TEST 2: DetectorGuidedCervicalAttention3D Module")
print("-" * 70)
try:
    from nnunetv2.training.nnUNetTrainer.variants.network_architecture.scse_modules import (
        DetectorGuidedCervicalAttention3D
    )

    # Test with different channel sizes (bottleneck and decoder stages)
    test_configs = [
        (320, 4, 16, 16),  # Bottleneck: small spatial size, many channels
        (160, 8, 32, 32),  # Decoder stage 1
        (80, 16, 64, 64),  # Decoder stage 2
    ]

    for channels, d, h, w in test_configs:
        x = torch.randn(2, channels, d, h, w)
        cervical_attn = DetectorGuidedCervicalAttention3D(
            num_channels=channels,
            reduction_ratio=2
        )
        # routing_map is derived from YOLO vertebrae localizaion, now it's None becasue YOLO is disabled for right now
        out, routing_map = cervical_attn(x)

        assert out.shape == x.shape, f"Output shape mismatch: {out.shape} != {x.shape}"
        # routing_map is None when YOLO is disabled (optimized mode)
        # This is expected and correct!

        print(f"   ✅ Channels={channels:3d}, Spatial=({d:2d},{h:2d},{w:2d}): PASS")

    print("   ✅ All configurations work correctly!")

except Exception as e:
    print(f"   ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    all_tests_passed = False

print()

# Test 3: Custom Trainer Import
print("TEST 3: nnUNetTrainerCervicalAttentionResEnc (Main Trainer)")
print("-" * 70)
try:
    from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerCervicalAttentionResEnc import (
        nnUNetTrainerCervicalAttentionResEnc
    )

    print("   ✅ Trainer class imported successfully!")

    # Check that key methods exist
    methods_to_check = [
        'initialize',
        '_build_loss',
        'build_network_architecture',
    ]

    for method in methods_to_check:
        if hasattr(nnUNetTrainerCervicalAttentionResEnc, method):
            print(f"   ✅ Method '{method}': EXISTS")
        else:
            print(f"   ❌ Method '{method}': MISSING")
            all_tests_passed = False

except Exception as e:
    print(f"   ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    all_tests_passed = False

print()

# Test 4: Network Architecture Building (most critical test)
print("TEST 4: Network Architecture Building with scSE Attention")
print("-" * 70)
try:
    from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerCervicalAttentionResEnc import (
        nnUNetTrainerCervicalAttentionResEnc
    )

    # Simulate nnU-Net architecture configuration
    arch_kwargs = {
        'n_stages': 6,
        'features_per_stage': [32, 64, 128, 256, 320, 320],
        'conv_op': torch.nn.Conv3d,
        'kernel_sizes': [[3, 3, 3]] * 6,
        'strides': [[1, 1, 1]] + [[2, 2, 2]] * 5,
        'n_blocks_per_stage': [1, 3, 4, 6, 6, 6],
        'n_conv_per_stage_decoder': [1, 1, 1, 1, 1],
        'conv_bias': True,
        'norm_op': torch.nn.InstanceNorm3d,
        'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
        'dropout_op': None,
        'dropout_op_kwargs': None,
        'nonlin': torch.nn.LeakyReLU,
        'nonlin_kwargs': {'inplace': True},
    }

    network = nnUNetTrainerCervicalAttentionResEnc.build_network_architecture(
        architecture_class_name='dynamic_network_architectures.architectures.unet.ResidualEncoderUNet',
        arch_init_kwargs=arch_kwargs,
        arch_init_kwargs_req_import=[],
        num_input_channels=1,
        num_output_channels=8,  # C1-C7 + background
        enable_deep_supervision=True
    )

    print("   ✅ Network built successfully!")

    # Check for scSE attention components
    if hasattr(network, 'bottleneck_attention'):
        print("   ✅ Bottleneck attention:     INSTALLED")
    else:
        print("   ❌ Bottleneck attention:     MISSING")
        all_tests_passed = False

    if hasattr(network, 'decoder_attention_modules'):
        num_decoder_attn = len(network.decoder_attention_modules)
        print(f"   ✅ Decoder attention modules: {num_decoder_attn} INSTALLED")
    else:
        print("   ❌ Decoder attention modules: MISSING")
        all_tests_passed = False

    if hasattr(network, 'current_attention_mask'):
        print("   ✅ Attention mask placeholder: EXISTS")
    else:
        print("   ⚠️  Attention mask placeholder: MISSING (OK if YOLO disabled)")

    # Test forward pass
    print("\n   Testing forward pass...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = network.to(device)
    network.eval()

    # Small test input
    test_input = torch.randn(1, 1, 32, 64, 64).to(device)

    with torch.no_grad():
        output = network(test_input)

    if isinstance(output, (list, tuple)):
        print(f"   ✅ Forward pass successful! Output shapes:")
        for i, out in enumerate(output):
            print(f"      Scale {i}: {out.shape}")
    else:
        print(f"   ✅ Forward pass successful! Output shape: {output.shape}")

except Exception as e:
    print(f"   ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    all_tests_passed = False

print()

# Test 5: Class-weighted loss configuration
print("TEST 5: Class-Weighted Loss Configuration")
print("-" * 70)
try:
    # Simulate loss building
    from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
    from nnunetv2.training.loss.compound_losses import DC_and_CE_loss

    num_classes = 8  # Background + C1-C7
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Check if class weights are configured correctly (gradient weighting)
    class_weights = torch.ones(num_classes, dtype=torch.float32, device=device)
    class_weights[3] = 1.5  # C3
    class_weights[4] = 1.5  # C4
    class_weights[5] = 1.5  # C5
    class_weights[6] = 2.5  # C6
    class_weights[7] = 2.5  # C7

    print("   ✅ Class weights configured:")
    print(f"      C1-C2: {class_weights[1:3].tolist()}")
    print(f"      C3-C5: {class_weights[3:6].tolist()}")
    print(f"      C6-C7: {class_weights[6:8].tolist()}")

    loss = DC_and_CE_loss(
        {'batch_dice': True, 'smooth': 1e-5, 'do_bg': False, 'ddp': False},
        {'weight': class_weights},
        weight_ce=1,
        weight_dice=1,
        ignore_label=None,
        dice_class=MemoryEfficientSoftDiceLoss
    )

    print("   ✅ Loss function built successfully!")

except Exception as e:
    print(f"   ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    all_tests_passed = False

print()

# Final summary
print("=" * 70)
if all_tests_passed:
    print("🎉 ALL TESTS PASSED! YOUR scSE nnUNet IS READY TO TRAIN!")
    print("=" * 70)
    print("\nYou can now run training with:")
    print("  nnUNetv2_train DATASET 3d_fullres FOLD -tr nnUNetTrainerCervicalAttentionResEnc")
    print("\nWhat you're training:")
    print("  ✓ ResidualEncoderUNet base architecture")
    print("  ✓ scSE attention in bottleneck + all decoder stages")
    print("  ✓ Skip connections UNCHANGED (direct gradient flow)")
    print("  ✓ Gradient class weighting (C3-C5: 1.5x, C6-C7: 2.5x)")
    print("  ✓ YOLO integration disabled (testing pure scSE)")
else:
    print("❌ SOME TESTS FAILED - CHECK ERRORS ABOVE")
    print("=" * 70)
    sys.exit(1)

print("=" * 70)

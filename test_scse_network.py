"""
Test script to verify scSE nnUNet architecture loads correctly
"""
import os
import sys
from pathlib import Path

# Set environment variables
base_path = Path(r"C:\Users\anoma\Downloads\scse-nnUNet\trial_preprocessing")
os.environ['nnUNet_raw'] = str(base_path / 'nnUNet_raw')
os.environ['nnUNet_preprocessed'] = str(base_path / 'nnUNet_preprocessed')
os.environ['nnUNet_results'] = str(base_path / 'nnUNet_results')

print("=" * 70)
print("Testing scSE nnUNet Architecture")
print("=" * 70)
print(f"nnUNet_raw:          {os.environ['nnUNet_raw']}")
print(f"nnUNet_preprocessed: {os.environ['nnUNet_preprocessed']}")
print(f"nnUNet_results:      {os.environ['nnUNet_results']}")
print("=" * 70)

# Create results directory if needed
results_path = Path(os.environ['nnUNet_results'])
results_path.mkdir(parents=True, exist_ok=True)

try:
    # Import the custom trainer
    print("\n1. Importing custom trainer...")
    from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerCervicalAttentionResEnc import nnUNetTrainerCervicalAttentionResEnc
    print("   ✓ nnUNetTrainerCervicalAttentionResEnc imported successfully")

    # Import required modules
    print("\n2. Importing nnUNet modules...")
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.label_handling.label_handling import LabelManager
    print("   ✓ nnUNet modules imported successfully")

    # Load dataset information
    print("\n3. Loading dataset configuration...")
    dataset_id = 1
    preprocessed_folder = Path(os.environ['nnUNet_preprocessed']) / f"Dataset{dataset_id:03d}_Cervical"

    if not preprocessed_folder.exists():
        print(f"   ✗ Preprocessed folder not found: {preprocessed_folder}")
        sys.exit(1)

    plans_file = preprocessed_folder / "nnUNetPlans.json"
    if not plans_file.exists():
        print(f"   ✗ Plans file not found: {plans_file}")
        sys.exit(1)

    print(f"   ✓ Dataset folder found: {preprocessed_folder}")
    print(f"   ✓ Plans file found: {plans_file}")

    # Load plans
    print("\n4. Loading nnUNet plans...")
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    plans_manager = PlansManager(plans_file)

    print(f"   ✓ Plans loaded")
    print(f"   - Configurations: {plans_manager.available_configurations}")

    # Get 3d_fullres configuration
    if '3d_fullres' not in plans_manager.available_configurations:
        print("   ✗ 3d_fullres configuration not found in plans")
        sys.exit(1)

    config_manager = plans_manager.get_configuration('3d_fullres')
    print(f"   - Patch size: {config_manager.patch_size}")
    print(f"   - Batch size: {config_manager.batch_size}")

    # Test network architecture building
    print("\n5. Testing network architecture initialization...")

    # Build network using the custom trainer's method
    print("\n   Building network with Bottleneck + Decoder scSE attention...")
    network = nnUNetTrainerCervicalAttentionResEnc.build_network_architecture(
        architecture_class_name=config_manager.unet_class_name,
        arch_init_kwargs=config_manager.network_arch_init_kwargs,
        arch_init_kwargs_req_import=config_manager.network_arch_init_kwargs_req_import,
        num_input_channels=config_manager.n_input_channels,
        num_output_channels=plans_manager.label_manager.num_segmentation_heads,
        enable_deep_supervision=True
    )

    print("\n   ✓ Network initialized successfully!")

    # Print network summary
    print("\n6. Network Summary:")
    print(f"   - Has bottleneck attention: {hasattr(network, 'bottleneck_attention')}")
    print(f"   - Has decoder attention modules: {hasattr(network, 'decoder_attention_modules')}")
    if hasattr(network, 'decoder_attention_modules'):
        print(f"   - Number of decoder attention modules: {len(network.decoder_attention_modules)}")

    # Test forward pass with dummy data
    print("\n7. Testing forward pass with dummy data...")
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   - Using device: {device}")

    network = network.to(device)
    network.eval()

    # Create dummy input
    batch_size = 1
    dummy_input = torch.randn(
        batch_size,
        config_manager.n_input_channels,
        *config_manager.patch_size
    ).to(device)

    print(f"   - Input shape: {dummy_input.shape}")

    # Forward pass
    with torch.no_grad():
        output = network(dummy_input)

    if isinstance(output, (list, tuple)):
        print(f"   - Output shapes (deep supervision): {[o.shape for o in output]}")
    else:
        print(f"   - Output shape: {output.shape}")

    print("\n   ✓ Forward pass successful!")

    print("\n" + "=" * 70)
    print("✓ All tests passed! scSE nnUNet is ready for training.")
    print("=" * 70)
    print("\nTo start training, run:")
    print("nnUNetv2_train 001 3d_fullres 0 -tr nnUNetTrainerCervicalAttentionResEnc")
    print("=" * 70)

except Exception as e:
    print(f"\n✗ Error occurred: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

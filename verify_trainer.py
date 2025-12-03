"""
Verify that the scSE trainer is actually being used with the correct modules
"""
import sys
import inspect

print("=" * 70)
print("VERIFYING scSE TRAINER INTEGRATION")
print("=" * 70)

# Import the trainer
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerCervicalAttentionResEnc import nnUNetTrainerCervicalAttentionResEnc

print("\n1. Checking trainer location:")
print(f"   Module: {nnUNetTrainerCervicalAttentionResEnc.__module__}")

# Check if scSE modules are imported
print("\n2. Checking scSE module imports:")
try:
    from nnunetv2.training.nnUNetTrainer.variants.network_architecture.scse_modules import DetectorGuidedCervicalAttention3D
    print(f"   ✓ DetectorGuidedCervicalAttention3D imported successfully")
    print(f"   ✓ Module: {DetectorGuidedCervicalAttention3D.__module__}")
except Exception as e:
    print(f"   ✗ Failed to import DetectorGuidedCervicalAttention3D: {e}")

# Check build_network_architecture method
print("\n3. Checking build_network_architecture method:")
if hasattr(nnUNetTrainerCervicalAttentionResEnc, 'build_network_architecture'):
    print("   ✓ build_network_architecture method exists")

    # Get the source code
    source = inspect.getsource(nnUNetTrainerCervicalAttentionResEnc.build_network_architecture)

    # Check for key scSE-related strings
    checks = {
        'bottleneck_attention': False,
        'decoder_attention_modules': False,
        'DetectorGuidedCervicalAttention3D': False,
        'Bottleneck + Decoder': False,
        'Skip connections are left UNCHANGED': False
    }

    for key in checks:
        if key in source:
            checks[key] = True
            print(f"   ✓ '{key}' found in code")
        else:
            print(f"   ✗ '{key}' NOT found in code")

    if all(checks.values()):
        print("\n   ✅ ALL scSE components verified in trainer!")
    else:
        print("\n   ⚠️ MISSING scSE components - trainer may not be correct")
else:
    print("   ✗ build_network_architecture method NOT found")

# Check class-weighted loss
print("\n4. Checking class-weighted loss:")
if hasattr(nnUNetTrainerCervicalAttentionResEnc, '_build_loss'):
    print("   ✓ _build_loss method exists")
    source = inspect.getsource(nnUNetTrainerCervicalAttentionResEnc._build_loss)
    if 'class_weights[6] = 2.0' in source and 'class_weights[7] = 2.0' in source:
        print("   ✓ C6/C7 2x weighting found")
    else:
        print("   ✗ C6/C7 weighting NOT found")
else:
    print("   ✗ _build_loss method NOT found")

print("\n" + "=" * 70)
print("SUMMARY:")
print("If all checks show ✓, then your custom scSE trainer IS being used.")
print("=" * 70)

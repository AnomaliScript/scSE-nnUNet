#!/usr/bin/env python3
"""
Quick test to verify the field import fix works
"""

print("Testing surgical_marching_cubes imports...")

try:
    # Test that all the reorganized classes can be imported
    from surgical_marching_cubes import (
        extract_surface_mesh,
        create_pyvista_mesh,
        AnatomicalRegion,
        SurgicalPhase,
        AnatomicalCriticalityScore,
        ANATOMICAL_CRITICALITY_PROFILES,
        AdaptiveResolutionConfig
    )
    print("SUCCESS: All imports working correctly!")
    print("  - extract_surface_mesh")
    print("  - create_pyvista_mesh")
    print("  - AnatomicalRegion")
    print("  - SurgicalPhase")
    print("  - AnatomicalCriticalityScore")
    print("  - ANATOMICAL_CRITICALITY_PROFILES")
    print("  - AdaptiveResolutionConfig")

    # Test that the field import fix allows AnatomicalCriticalityScore to be instantiated
    print("\nTesting AnatomicalCriticalityScore instantiation...")
    score = AnatomicalCriticalityScore(
        structural_importance=0.8,
        vascular_risk=0.6,
        neural_risk=0.7
    )
    print(f"SUCCESS: Created AnatomicalCriticalityScore with default field")
    print(f"  surgical_phase_weight: {score.surgical_phase_weight}")

    # Test the ANATOMICAL_CRITICALITY_PROFILES dictionary
    print("\nTesting ANATOMICAL_CRITICALITY_PROFILES...")
    pedicle_profile = ANATOMICAL_CRITICALITY_PROFILES[AnatomicalRegion.PEDICLE]
    importance = pedicle_profile.compute_total_importance(SurgicalPhase.PLANNING)
    print(f"SUCCESS: Pedicle importance (PLANNING phase): {importance:.3f}")

    # Test AdaptiveResolutionConfig
    print("\nTesting AdaptiveResolutionConfig...")
    config = AdaptiveResolutionConfig()
    print(f"SUCCESS: Created AdaptiveResolutionConfig")
    print(f"  Max octree depth: {config.max_octree_depth}")

    print("\n" + "="*60)
    print("ALL TESTS PASSED")
    print("="*60)
    print("\nThe 'field' import fix is working correctly!")
    print("Code reorganization complete:")
    print("  - All shared classes centralized in surgical_marching_cubes.py")
    print("  - Classes properly use 'field' for default factories")
    print("  - No import errors")

except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

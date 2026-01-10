# Version 1: Binary Segmentation (Worst-Case Scenario)

## Overview
This version is designed for segmentation models that only output **binary labels**:
- **Label 0**: Background (air, soft tissue, everything else)
- **Label 1**: All vertebrae lumped together as a single bone mass

## When to Use This Version
Use this version if your segmentation model:
- Only distinguishes between "bone" and "not bone"
- Cannot differentiate individual vertebrae (C1, C2, C3, etc.)
- Outputs only 2 unique values when you run `np.unique(segmentation)`

**Quick test**:
```python
import numpy as np
unique_labels = np.unique(your_segmentation)
print(unique_labels)
# If you see: [0 1] → Use this version (v1)
# If you see: [0 1 2 3 4 5 6 7] → Use v2_per_vertebra instead
```

## Capabilities

### What Works:
✓ **Adaptive octree subdivision** - Based on uncertainty and surgical importance
✓ **Confidence-weighted vertex interpolation** - Accurate surface placement
✓ **Surface smoothing** - With feature preservation
✓ **Mesh quality validation** - Surgical quality standards
✓ **Real-time mesh updates** - Adapts during surgery
✓ **Multi-format export** - STL, OBJ, PLY, JSON

### What Doesn't Work:
❌ **Vertebra-specific detail enhancement** - Cannot distinguish C1 from C7
❌ **Per-vertebra adaptive resolution** - Uniform resolution across all bone
❌ **Target vertebra selection** - No way to prioritize specific levels

## Usage

### Basic Usage:
```python
from colab.v1_binary_segmentation.marching_cubes_core_data import (
    extract_surface_mesh,
    create_pyvista_mesh
)

# Load your CT volume
volume = np.load('ct_volume.npy')

# Extract mesh
vertices, faces = extract_surface_mesh(
    volume,
    threshold=0.5,
    spacing=(0.5, 0.5, 0.5)  # your voxel spacing in mm
)

# Visualize
mesh = create_pyvista_mesh(vertices, faces)
mesh.plot()
```

### Advanced Usage with Uncertainty Maps:
```python
from colab.v1_binary_segmentation.marching_cubes_core_data import (
    extract_surface_mesh,
    SurgicalPhase
)

# Extract mesh with uncertainty-aware subdivision
vertices, faces = extract_surface_mesh(
    volume,
    threshold=0.5,
    spacing=(0.5, 0.5, 0.5),
    phase=SurgicalPhase.SCREW_TRAJECTORY,
    uncertainty_map=model_uncertainty,  # from your model output
    importance_map=trajectory_importance  # custom importance weights
)
```

## Limitations

### Uniform Resolution
Since the segmentation doesn't distinguish between vertebrae, the adaptive resolution system treats all bone uniformly:
- Cannot increase detail at C5 pedicles (screw target)
- Cannot reduce detail at C1/C7 (less critical)
- All vertebrae get the same mesh density

### No Anatomical Context
The algorithm cannot:
- Identify which vertebra a surface point belongs to
- Prioritize critical anatomical landmarks
- Provide vertebra-specific warnings or metrics

## Upgrade Path

If you later obtain a segmentation model that can differentiate C1-C7 vertebrae:
1. Switch to `colab/v2_per_vertebra/` version
2. Update your imports:
   ```python
   from colab.v2_per_vertebra.marching_cubes_core_data import ...
   ```
3. Enjoy vertebra-specific adaptive resolution!

## Files Included
- `marching_cubes_tables.py` - Lookup tables (dependency)
- `marching_cubes_core_data.py` - Foundation with API functions
- `h_arch_adapt_grid_system.py` - Octree system (simplified)
- `mesh_gen_surface_extract.py` - Mesh generation
- `marching_cubes_integration.py` - Integration & export
- `__init__.py` - Package init

## Technical Notes

### AnatomicalRegion Enum
Only two regions are active:
```python
class AnatomicalRegion(Enum):
    BACKGROUND = 0
    VERTEBRAL_BODY = 1
    # All other regions commented out
```

### Importance Profile
Uniform importance for all bone:
```python
ANATOMICAL_CRITICALITY_PROFILES = {
    AnatomicalRegion.BACKGROUND: 0.0,
    AnatomicalRegion.VERTEBRAL_BODY: 0.6  # moderate importance
}
```

### Critical Neighbor Detection
Simplified to just detect bone/background boundaries:
```python
def _has_critical_neighbors(...):
    unique_labels = set(...)
    return len(unique_labels) > 1  # Any boundary
```

## Performance
Expected performance on typical cervical spine CT (128x128x64):
- Mesh extraction: ~0.5-2 seconds
- Vertex count: ~10,000-50,000 (depends on octree depth)
- Triangle count: ~20,000-100,000

## Support
See [VERSION_GUIDE.md](../VERSION_GUIDE.md) for guidance on choosing between v1 and v2.

See [INTEGRATION_NOTES.md](../INTEGRATION_NOTES.md) for integration details.

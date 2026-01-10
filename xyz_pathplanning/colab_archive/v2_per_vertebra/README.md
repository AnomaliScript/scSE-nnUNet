# Version 2: Per-Vertebra Segmentation (C1-C7)

## Overview
This version is designed for segmentation models that output **individual cervical vertebrae labels**:
- **Label 0**: Background
- **Label 1**: C1 vertebra (Atlas)
- **Label 2**: C2 vertebra (Axis)
- **Label 3**: C3 vertebra
- **Label 4**: C4 vertebra (common screw target)
- **Label 5**: C5 vertebra (MOST COMMON screw target - HIGHEST detail)
- **Label 6**: C6 vertebra (common screw target)
- **Label 7**: C7 vertebra

## When to Use This Version
Use this version if your segmentation model:
- Can differentiate individual cervical vertebrae (C1 through C7)
- Outputs 8 unique values when you run `np.unique(segmentation)`
- Provides per-vertebra labels from your nnUNet or other segmentation model

**Quick test**:
```python
import numpy as np
unique_labels = np.unique(your_segmentation)
print(unique_labels)
# If you see: [0 1 2 3 4 5 6 7] → Use this version (v2)
# If you see: [0 1] → Use v1_binary_segmentation instead
```

## Key Feature: Vertebra-Specific Adaptive Resolution

This is the **main advantage** of this version over v1. The marching cubes algorithm automatically allocates different mesh detail levels based on surgical importance:

### Adaptive Detail Allocation:
- **C5 (Label 5)**: HIGHEST detail - importance weight = 1.0
  - Most common screw placement target in cervical spine surgery
  - Maximum mesh resolution, finest surface detail

- **C4 & C6 (Labels 4, 6)**: HIGH detail - importance weight = 0.9
  - Common screw placement targets
  - High mesh resolution

- **C1, C2, C3, C7 (Labels 1, 2, 3, 7)**: MEDIUM detail - importance weight = 0.5
  - Adjacent vertebrae, less commonly targeted
  - Standard mesh resolution

- **Background (Label 0)**: MINIMAL detail - importance weight = 0.0
  - Low resolution for efficiency

### Surgical Phase Awareness:
The importance weights further increase during critical surgical phases:

**C5 Vertebra Example**:
- **Planning Phase**: importance × 1.5 = 1.5 (even higher detail)
- **Screw Trajectory Phase**: importance × 2.0 = 2.0 (maximum detail)
- **Screw Placement Phase**: importance × 2.5 = 2.5 (extreme detail for real-time)

## Capabilities

### What Works:
✓ **Per-vertebra adaptive resolution** - C5 gets 2-3× more mesh detail than C1/C7
✓ **Target vertebra selection** - Prioritize C4, C5, or C6 for surgery
✓ **Surgical corridor analysis** - Track which vertebrae the trajectory passes through
✓ **Vertebra-specific metrics** - Separate vertex counts for target vs adjacent vertebrae
✓ **Adaptive octree subdivision** - Based on uncertainty, importance, and phase
✓ **Confidence-weighted vertex interpolation** - Accurate surface placement
✓ **Surface smoothing** - With feature preservation
✓ **Mesh quality validation** - Surgical quality standards
✓ **Real-time mesh updates** - Adapts during surgery
✓ **Multi-format export** - STL, OBJ, PLY, JSON

### What Doesn't Work (Still):
❌ **Pedicle-specific detail** - Would need pedicle segmentation within each vertebra
❌ **Spinal canal warnings** - Would need separate canal segmentation
❌ **Nerve/artery avoidance** - Would need vascular/neural structure segmentation

To enable these, you'd need a segmentation model with even more anatomical detail (pedicles, canal, arteries, nerves as separate labels).

## Usage

### Basic Usage:
```python
from colab.v2_per_vertebra.marching_cubes_core_data import (
    extract_surface_mesh,
    create_pyvista_mesh,
    SurgicalPhase
)

# Load your CT volume and per-vertebra segmentation
volume = np.load('ct_volume.npy')
segmentation = np.load('segmentation_c1_c7.npy')  # Labels 0-7

# Extract mesh with automatic vertebra-specific resolution
vertices, faces = extract_surface_mesh(
    volume,
    threshold=0.5,
    spacing=(0.5, 0.5, 0.5),
    phase=SurgicalPhase.PLANNING  # Adjusts importance weights
)

# Visualize
mesh = create_pyvista_mesh(vertices, faces)
mesh.plot()
```

### Advanced Usage - Targeting C5 for Surgery:
```python
from colab.v2_per_vertebra.marching_cubes_core_data import (
    extract_surface_mesh,
    SurgicalPhase,
    AnatomicalRegion
)
from colab.v2_per_vertebra.h_arch_adapt_grid_system import (
    SurgicalCorridorOptimizedOctree,
    SurgicalCorridorAnalyzer
)
from colab.v2_per_vertebra.mesh_gen_surface_extract import (
    AdaptiveMarchingCubesEngine
)

# Define surgical trajectory targeting C5 pedicle screw
trajectory = np.array([
    [36.0, 30.0, 12.0],  # Entry point (posterior, C5 level)
    [38.0, 26.0, 16.0],  # Through C5 vertebral body
    [40.0, 24.0, 20.0]   # Target (anterior C5)
])

# Build octree with trajectory awareness
octree = SurgicalCorridorOptimizedOctree(
    volume, segmentation, importance_map, uncertainty_map,
    spacing=(0.5, 0.5, 0.5),
    max_depth=7,
    phase=SurgicalPhase.SCREW_TRAJECTORY,
    trajectory_points=trajectory,
    trajectory_radius=3.0  # mm - boost resolution within 3mm of trajectory
)
octree.build()

# Extract mesh with maximum detail along trajectory and at C5
engine = AdaptiveMarchingCubesEngine(octree)
engine.extract_surface()

# Analyze corridor safety
analyzer = SurgicalCorridorAnalyzer(octree)
analysis = analyzer.analyze_corridor()

print(f"Corridor passes through:")
for vertebra, metrics in analysis['critical_structure_proximity'].items():
    print(f"  {vertebra}: {metrics['count']} nodes, "
          f"confidence={metrics['avg_confidence']:.2f}")
```

### Importance Map Customization:
```python
# Create custom importance map (override defaults)
importance_map = np.zeros_like(segmentation, dtype=np.float32)

# Set custom weights for each vertebra
importance_map[segmentation == AnatomicalRegion.C1_VERTEBRA.value] = 0.3
importance_map[segmentation == AnatomicalRegion.C2_VERTEBRA.value] = 0.3
importance_map[segmentation == AnatomicalRegion.C3_VERTEBRA.value] = 0.4
importance_map[segmentation == AnatomicalRegion.C4_VERTEBRA.value] = 0.8  # Target
importance_map[segmentation == AnatomicalRegion.C5_VERTEBRA.value] = 1.0  # Primary target
importance_map[segmentation == AnatomicalRegion.C6_VERTEBRA.value] = 0.8  # Target
importance_map[segmentation == AnatomicalRegion.C7_VERTEBRA.value] = 0.4

# Use in mesh extraction
vertices, faces = extract_surface_mesh(
    volume,
    threshold=0.5,
    spacing=(0.5, 0.5, 0.5),
    importance_map=importance_map  # Custom weighting
)
```

## Technical Details

### AnatomicalRegion Enum:
```python
class AnatomicalRegion(Enum):
    BACKGROUND = 0
    C1_VERTEBRA = 1  # Atlas
    C2_VERTEBRA = 2  # Axis
    C3_VERTEBRA = 3
    C4_VERTEBRA = 4  # Common screw target
    C5_VERTEBRA = 5  # MOST COMMON target - highest detail
    C6_VERTEBRA = 6  # Common screw target
    C7_VERTEBRA = 7  # Transitional vertebra
```

### Importance Profiles:
```python
ANATOMICAL_CRITICALITY_PROFILES = {
    AnatomicalRegion.C1_VERTEBRA: (0.5 base, phase multipliers: planning=0.8),
    AnatomicalRegion.C2_VERTEBRA: (0.5 base, phase multipliers: planning=0.8),
    AnatomicalRegion.C3_VERTEBRA: (0.5 base, phase multipliers: planning=0.8),

    AnatomicalRegion.C4_VERTEBRA: (0.9 base, phase multipliers up to 2.0),
    AnatomicalRegion.C5_VERTEBRA: (1.0 base, phase multipliers up to 2.5),  # HIGHEST
    AnatomicalRegion.C6_VERTEBRA: (0.9 base, phase multipliers up to 2.0),

    AnatomicalRegion.C7_VERTEBRA: (0.5 base, phase multipliers: planning=0.8),
}
```

### Critical Neighbor Detection:
Octree subdivides more when boundaries are adjacent to **target vertebrae (C4-C6)**:
```python
def _has_critical_neighbors(...):
    target_vertebrae = {C4_VERTEBRA, C5_VERTEBRA, C6_VERTEBRA}
    return bool(target_vertebrae & unique_labels_in_region)
```

### Metrics Output:
```python
metrics = engine.get_quality_metrics()
print(f"Target vertebrae (C4-C6) vertices: {metrics.pedicle_vertex_count}")
print(f"Adjacent vertebrae (C1-C3,C7) vertices: {metrics.critical_structure_vertex_count}")
# C5 will typically have 2-3× more vertices than C1
```

## Comparison: v1 vs v2

| Feature | v1 (Binary) | v2 (Per-Vertebra) |
|---------|-------------|-------------------|
| Labels | 0=background, 1=all bone | 0=background, 1-7=C1-C7 |
| Adaptive resolution | Uniform across all bone | **Per-vertebra (C5 gets 2-3× detail)** |
| Target selection | No | **Yes - can prioritize C4/C5/C6** |
| Corridor analysis | Basic (bone vs background) | **Detailed (which vertebrae)** |
| Surgical planning | Limited | **Advanced (vertebra-aware)** |
| Mesh efficiency | Lower (wastes detail on non-critical areas) | **Higher (concentrates detail where needed)** |

## Performance
Expected performance on typical cervical spine CT (128×128×64):
- Mesh extraction: ~1-3 seconds (varies by octree depth)
- Vertex count: ~20,000-80,000 (C5 gets more vertices)
- Triangle count: ~40,000-160,000
- C5 mesh density: ~2-3× higher than C1/C7

## Files Included
- `marching_cubes_tables.py` - Lookup tables (dependency)
- `marching_cubes_core_data.py` - Foundation with C1-C7 enums
- `h_arch_adapt_grid_system.py` - Octree with vertebra-specific logic
- `mesh_gen_surface_extract.py` - Mesh generation with per-vertebra metrics
- `marching_cubes_integration.py` - Integration & export
- `__init__.py` - Package init

## Upgrade from v1
If you're currently using v1 (binary segmentation) and obtain a per-vertebra segmentation model:

1. **Update segmentation model** to output C1-C7 labels (0-7)
2. **Switch imports**:
   ```python
   # Old:
   from colab.v1_binary_segmentation.marching_cubes_core_data import ...

   # New:
   from colab.v2_per_vertebra.marching_cubes_core_data import ...
   ```
3. **Enjoy per-vertebra adaptive resolution!** No other code changes needed.

## Support
See [VERSION_GUIDE.md](../VERSION_GUIDE.md) for guidance on choosing between v1 and v2.

See [INTEGRATION_NOTES.md](../INTEGRATION_NOTES.md) for integration details.

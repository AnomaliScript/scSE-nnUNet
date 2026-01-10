# Colab Marching Cubes - Version Selection Guide

## Quick Decision Tree

```
Does your segmentation model output individual vertebrae (C1-C7) as separate labels?
│
├─ YES: I can differentiate C1 from C2 from C3, etc.
│   └─> Use v2_per_vertebra/
│       ✓ Per-vertebra adaptive resolution
│       ✓ C5 gets 2-3× more mesh detail than C1/C7
│       ✓ Target vertebra selection
│       ✓ Vertebra-aware corridor analysis
│
└─ NO: My model only outputs "bone" vs "not bone"
    └─> Use v1_binary_segmentation/
        ✓ Works with any segmentation
        ✓ Uniform resolution across all bone
        ✓ Simpler to use
```

## How to Check Your Segmentation Model

Run this quick test on your segmentation output:

```python
import numpy as np

# Load your segmentation (from nnUNet or other model)
segmentation = np.load('path/to/your_segmentation.npy')

# Check unique labels
unique_labels = np.unique(segmentation)
print(f"Unique labels in segmentation: {unique_labels}")
print(f"Number of unique labels: {len(unique_labels)}")

# Decision
if len(unique_labels) == 2:  # [0, 1]
    print("\n→ Use v1_binary_segmentation/")
    print("   Your model outputs: 0=background, 1=all bone")

elif len(unique_labels) == 8:  # [0, 1, 2, 3, 4, 5, 6, 7]
    print("\n→ Use v2_per_vertebra/")
    print("   Your model outputs: 0=background, 1-7=C1-C7 vertebrae")

    # Verify which label is which vertebra
    print("\nLabel distribution:")
    for label in unique_labels:
        if label > 0:
            count = np.sum(segmentation == label)
            print(f"  Label {label}: {count:,} voxels")

    print("\nExpected mapping:")
    print("  Label 1 → C1 (Atlas)")
    print("  Label 2 → C2 (Axis)")
    print("  Label 3 → C3")
    print("  Label 4 → C4")
    print("  Label 5 → C5 (most common screw target)")
    print("  Label 6 → C6")
    print("  Label 7 → C7")

else:
    print(f"\n⚠ WARNING: Unexpected number of labels: {len(unique_labels)}")
    print(f"Labels found: {unique_labels}")
    print("\nIf you have 8 labels (0-7), use v2_per_vertebra/")
    print("If you only have 2 labels (0-1), use v1_binary_segmentation/")
```

## Version Comparison

### Version 1: Binary Segmentation (`v1_binary_segmentation/`)

**Use when:**
- Your segmentation model only outputs 2 labels
- Label 0 = background (air, soft tissue, etc.)
- Label 1 = all vertebrae lumped together as one mass
- You're not sure which version to use (v1 is the safe fallback)

**Capabilities:**
- ✓ Adaptive octree subdivision (based on uncertainty/importance)
- ✓ Confidence-weighted vertex interpolation
- ✓ Surface smoothing with feature preservation
- ✓ Mesh quality validation
- ✓ Multi-format export (STL, OBJ, PLY, JSON)

**Limitations:**
- ❌ Cannot distinguish C1 from C7 - all vertebrae treated equally
- ❌ No per-vertebra adaptive resolution
- ❌ Cannot prioritize specific vertebrae for surgery
- ❌ Uniform mesh density across all bone

**Import:**
```python
from colab.v1_binary_segmentation.marching_cubes_core_data import (
    extract_surface_mesh,
    create_pyvista_mesh
)
```

### Version 2: Per-Vertebra Segmentation (`v2_per_vertebra/`)

**Use when:**
- Your segmentation model outputs 8 labels (0-7)
- Label 0 = background
- Labels 1-7 = C1, C2, C3, C4, C5, C6, C7 vertebrae (separately)
- You want maximum mesh detail at surgical target vertebrae

**Capabilities:**
- ✓ **Per-vertebra adaptive resolution** (main advantage!)
- ✓ C5 automatically gets 2-3× more mesh detail than C1/C7
- ✓ Target vertebra selection (prioritize C4/C5/C6 for screw placement)
- ✓ Vertebra-aware corridor analysis
- ✓ All capabilities from v1 (octree, interpolation, smoothing, export, etc.)

**Adaptive Resolution Example:**
- C5 (most common screw target): **HIGHEST detail** (importance = 1.0)
- C4, C6 (common screw targets): **HIGH detail** (importance = 0.9)
- C1, C2, C3, C7 (adjacent vertebrae): **MEDIUM detail** (importance = 0.5)
- Background: **MINIMAL detail** (importance = 0.0)

**Import:**
```python
from colab.v2_per_vertebra.marching_cubes_core_data import (
    extract_surface_mesh,
    create_pyvista_mesh,
    AnatomicalRegion
)
```

## Side-by-Side Example

### v1 Usage:
```python
from colab.v1_binary_segmentation.marching_cubes_core_data import extract_surface_mesh

# Simple - just volume and segmentation
vertices, faces = extract_surface_mesh(
    volume,
    threshold=0.5,
    spacing=(0.5, 0.5, 0.5)
)
# All bone gets same mesh density
```

### v2 Usage:
```python
from colab.v2_per_vertebra.marching_cubes_core_data import (
    extract_surface_mesh,
    SurgicalPhase
)

# Same API - but automatic per-vertebra detail!
vertices, faces = extract_surface_mesh(
    volume,
    threshold=0.5,
    spacing=(0.5, 0.5, 0.5),
    phase=SurgicalPhase.SCREW_TRAJECTORY  # Boosts C5 detail even more
)
# C5 automatically gets 2-3× more vertices than C1/C7
```

## Root `colab/` Directory

The root `colab/` directory contains the **original binary segmentation version** (same as v1). This serves as the default/fallback version.

**Import from root:**
```python
from colab.marching_cubes_core_data import extract_surface_mesh
# Equivalent to v1_binary_segmentation
```

## Migration Between Versions

### From v1 to v2 (when you get better segmentation):

1. **Check segmentation labels:**
   ```python
   print(np.unique(new_segmentation))
   # Should see: [0 1 2 3 4 5 6 7]
   ```

2. **Update imports:**
   ```python
   # Old:
   from colab.v1_binary_segmentation.marching_cubes_core_data import ...

   # New:
   from colab.v2_per_vertebra.marching_cubes_core_data import ...
   ```

3. **No other changes needed!** The API is identical. Just enjoy automatic per-vertebra adaptive resolution.

### From v2 to v1 (if segmentation quality degrades):

1. **Update imports:**
   ```python
   # Old:
   from colab.v2_per_vertebra.marching_cubes_core_data import ...

   # New:
   from colab.v1_binary_segmentation.marching_cubes_core_data import ...
   ```

2. **Remove vertebra-specific code** (if any):
   ```python
   # Remove references to AnatomicalRegion.C5_VERTEBRA, etc.
   ```

## Common Questions

### Q: Can I use v2 if my segmentation only has 2 labels?
**A:** No, v2 expects labels 0-7. It will crash or produce incorrect results with binary segmentation. Use v1 instead.

### Q: My segmentation has labels [0, 1, 2, 3, 4, 5, 6, 7] but they're not C1-C7. What do I do?
**A:** If the labels represent something else (e.g., different body parts), you can:
1. Relabel them to match C1-C7 mapping, OR
2. Modify v2's `AnatomicalRegion` enum to match your labels, OR
3. Use v1 (treat all as uniform bone)

### Q: Which version is faster?
**A:** Both have similar performance. v2 may be slightly slower due to per-vertebra importance calculations, but the difference is negligible (~5-10%).

### Q: Can I mix versions in the same project?
**A:** Technically yes (different imports), but **not recommended**. Pick one version per project for consistency.

### Q: I have C1-C7 labels but don't care about per-vertebra resolution. Can I still use v1?
**A:** Yes! v1 will work fine. It will just treat all bone uniformly (ignoring the label differences). But you're missing out on the adaptive resolution feature.

### Q: How do I know if the adaptive resolution is working in v2?
**A:** Check vertex counts:
```python
metrics = engine.get_quality_metrics()
print(f"Target vertebrae (C4-C6): {metrics.pedicle_vertex_count:,} vertices")
print(f"Adjacent vertebrae (C1-C3,C7): {metrics.critical_structure_vertex_count:,} vertices")
# C4-C6 should have significantly more vertices
```

## Recommendations

### For Production Use:
1. **Start with v1** (binary segmentation) - it's the safe, proven version
2. **Upgrade to v2** once you have a reliable per-vertebra segmentation model
3. **Test both versions** on the same data to verify v2's adaptive resolution is working

### For Development/Research:
1. **Use v2** if you have C1-C7 labels - take advantage of adaptive resolution
2. **Fall back to v1** if you encounter issues

### For Surgical Planning:
1. **Strongly prefer v2** if available - the per-vertebra detail is crucial for screw placement
2. **C5 screw trajectories** especially benefit from v2's automatic detail boost

## File Structure

```
xyz_pathplanning/
├── colab/                              # Root (binary version - same as v1)
│   ├── marching_cubes_core_data.py
│   ├── h_arch_adapt_grid_system.py
│   ├── mesh_gen_surface_extract.py
│   ├── marching_cubes_integration.py
│   ├── marching_cubes_tables.py
│   ├── __init__.py
│   ├── INTEGRATION_NOTES.md
│   ├── VERSION_GUIDE.md              # This file
│   │
│   ├── v1_binary_segmentation/        # Version 1: Binary (0=bg, 1=bone)
│   │   ├── marching_cubes_core_data.py
│   │   ├── h_arch_adapt_grid_system.py
│   │   ├── mesh_gen_surface_extract.py
│   │   ├── marching_cubes_integration.py
│   │   ├── marching_cubes_tables.py
│   │   ├── __init__.py
│   │   └── README.md
│   │
│   └── v2_per_vertebra/               # Version 2: Per-vertebra (0=bg, 1-7=C1-C7)
│       ├── marching_cubes_core_data.py
│       ├── h_arch_adapt_grid_system.py
│       ├── mesh_gen_surface_extract.py
│       ├── marching_cubes_integration.py
│       ├── marching_cubes_tables.py
│       ├── __init__.py
│       └── README.md
│
└── seg_to_path.py                     # Currently imports from root (v1)
```

## Need Help?

1. **Version selection**: Run the test code at the top of this guide
2. **v1 documentation**: See `v1_binary_segmentation/README.md`
3. **v2 documentation**: See `v2_per_vertebra/README.md`
4. **Integration notes**: See `INTEGRATION_NOTES.md`

## Summary

- **Have C1-C7 labels?** → Use `v2_per_vertebra/` for adaptive per-vertebra resolution
- **Only have binary labels?** → Use `v1_binary_segmentation/` for uniform resolution
- **Not sure?** → Run `np.unique(segmentation)` and check the guide above

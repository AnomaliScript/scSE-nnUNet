# Code Organization Guide

## Overview

The surgical marching cubes codebase has been reorganized for better maintainability and logical separation of concerns. Classes and functions are now in sensible locations and properly referenced across files.

## File Structure

```
xyz_pathplanning/
├── surgical_marching_cubes.py    # Core module (3400+ lines)
│   ├── Shared classes & enums
│   ├── Marching cubes implementation
│   └── Advanced surgical features
│
├── seg_to_path.py                # Path planning dashboard
│   ├── FMM path finding
│   ├── Interactive UI
│   └── Uses surgical_marching_cubes for 3D rendering
│
├── sancturary_spacings.py        # Dataset preprocessing
│   ├── Voxel spacing normalization
│   └── Imports classes from surgical_marching_cubes
│
└── test_integration.py           # Integration tests
```

## Module Responsibilities

### surgical_marching_cubes.py (Central Module)

**Purpose**: Core marching cubes implementation with surgical enhancements

**Exports** (via `__all__`):
```python
# Core Functions (used by seg_to_path.py)
extract_surface_mesh()           # Main marching cubes
create_pyvista_mesh()            # PyVista conversion
extract_multi_label_mesh()       # Multi-label extraction

# Shared Classes (used by multiple modules)
AnatomicalRegion                 # Enum for anatomy labels
SurgicalPhase                    # Enum for surgical phases
AnatomicalCriticalityScore       # Multi-factor scoring
ANATOMICAL_CRITICALITY_PROFILES  # Pre-defined profiles
AdaptiveResolutionConfig         # Resolution configuration

# Advanced Features (optional)
CompleteSurgicalReconstructionSystem
SurgicalCorridorOptimizedOctree
AdaptiveMarchingCubesEngine
```

**Why here?**
- Central location for all surgical visualization code
- Shared by both seg_to_path.py and sancturary_spacings.py
- Single source of truth for anatomical definitions
- Easier to maintain and update

### seg_to_path.py (Path Planning Dashboard)

**Purpose**: Interactive surgical path planning interface

**Dependencies**:
```python
from surgical_marching_cubes import (
    extract_surface_mesh,
    create_pyvista_mesh
)
```

**Key Features**:
- Fast Marching Method (FMM) path finding
- Interactive 2D/3D visualization
- Safety metrics calculation
- Uses marching cubes for volume rendering

**Why separate?**
- Different purpose (path planning vs surface extraction)
- User-facing interactive tool
- Imports from surgical_marching_cubes as needed

### sancturary_spacings.py (Dataset Preprocessing)

**Purpose**: Normalize voxel spacing for datasets

**Dependencies**:
```python
from surgical_marching_cubes import (
    SurgicalPhase,
    AnatomicalRegion,
    AnatomicalCriticalityScore,
    ANATOMICAL_CRITICALITY_PROFILES,
    AdaptiveResolutionConfig,
    MarchingCubesLookupTables
)
```

**Key Features**:
- Selective dimension-wise resampling
- Creates "sanctuary" of ≤1mm spacing
- Uses shared anatomical classifications

**Why imports?**
- Needs anatomical definitions for processing
- Avoids code duplication
- Maintains consistency across modules

**Note**: Original class definitions (lines 575+) are wrapped in `if False:` for documentation purposes only.

## Import Chain

```
sancturary_spacings.py  ──┐
                          ├──> surgical_marching_cubes.py
seg_to_path.py          ──┘      (central module)
```

**One-way dependencies**: Both files import from `surgical_marching_cubes.py`, but not vice versa.

## Class Locations (Quick Reference)

| Class/Function | Location | Used By |
|----------------|----------|---------|
| `AnatomicalRegion` | surgical_marching_cubes.py | seg_to_path, sancturary_spacings |
| `SurgicalPhase` | surgical_marching_cubes.py | seg_to_path, sancturary_spacings |
| `AnatomicalCriticalityScore` | surgical_marching_cubes.py | sancturary_spacings, advanced features |
| `ANATOMICAL_CRITICALITY_PROFILES` | surgical_marching_cubes.py | sancturary_spacings, advanced features |
| `AdaptiveResolutionConfig` | surgical_marching_cubes.py | sancturary_spacings, advanced features |
| `MarchingCubesLookupTables` | surgical_marching_cubes.py | sancturary_spacings (reference) |
| `extract_surface_mesh()` | surgical_marching_cubes.py | seg_to_path |
| `create_pyvista_mesh()` | surgical_marching_cubes.py | seg_to_path |
| `InteractivePathPlanner` | seg_to_path.py | seg_to_path (main) |
| `create_sanctuary_for_dataset()` | sancturary_spacings.py | sancturary_spacings (main) |

## Design Principles

### 1. Single Source of Truth
✅ Anatomical definitions in one place (`surgical_marching_cubes.py`)
✅ No duplicate class definitions
✅ Imports ensure consistency

### 2. Clear Module Boundaries
✅ `surgical_marching_cubes.py` = Core visualization
✅ `seg_to_path.py` = Path planning application
✅ `sancturary_spacings.py` = Dataset preprocessing

### 3. Minimal Coupling
✅ One-way import dependencies
✅ `surgical_marching_cubes.py` doesn't import from other modules
✅ Clear public API via `__all__`

### 4. Backwards Compatibility
✅ Fallback definitions in `sancturary_spacings.py` if import fails
✅ Original code preserved in `if False:` blocks for reference

## Migration Guide

### Before (Duplicated Classes)
```python
# In sancturary_spacings.py
class SurgicalPhase(Enum):
    PLANNING = "planning"
    # ... (lines 565-575)

class AnatomicalRegion(Enum):
    BACKGROUND = 0
    # ... (lines 578-593)

# In surgical_marching_cubes.py
class AnatomicalRegion(Enum):
    BACKGROUND = 0
    # ... (duplicate definition)
```

### After (Centralized)
```python
# In surgical_marching_cubes.py (lines 49-178)
class AnatomicalRegion(Enum):
    """Cervical spine anatomical regions with surgical criticality scores"""
    BACKGROUND = 0
    VERTEBRAL_BODY = 1
    # ...

class SurgicalPhase(Enum):
    """Surgical phase states..."""
    PLANNING = "planning"
    # ...

class AnatomicalCriticalityScore:
    # ...

ANATOMICAL_CRITICALITY_PROFILES = { ... }

class AdaptiveResolutionConfig:
    # ...

# In sancturary_spacings.py (lines 29-38)
from surgical_marching_cubes import (
    SurgicalPhase,
    AnatomicalRegion,
    AnatomicalCriticalityScore,
    ANATOMICAL_CRITICALITY_PROFILES,
    AdaptiveResolutionConfig
)

# In seg_to_path.py (lines 30-32)
from surgical_marching_cubes import (
    extract_surface_mesh,
    create_pyvista_mesh
)
```

## Benefits

### For Development
- ✅ Easier to find where classes are defined
- ✅ Single place to update anatomical definitions
- ✅ Clear dependencies between modules
- ✅ Reduced code duplication

### For Maintenance
- ✅ Changes propagate automatically via imports
- ✅ No need to sync duplicate definitions
- ✅ Easier to add new anatomical regions
- ✅ Clear module responsibilities

### For Testing
- ✅ Can test `surgical_marching_cubes.py` independently
- ✅ Can mock imports in other modules
- ✅ Integration test covers import chain
- ✅ Each module has clear test boundaries

## Common Tasks

### Adding a New Anatomical Region

**Edit**: `surgical_marching_cubes.py`

```python
# 1. Add to enum (line ~60)
class AnatomicalRegion(Enum):
    # ... existing ...
    NEW_STRUCTURE = 10  # Add new value

# 2. Add to profiles (line ~109)
ANATOMICAL_CRITICALITY_PROFILES = {
    # ... existing ...
    AnatomicalRegion.NEW_STRUCTURE: AnatomicalCriticalityScore(
        0.8, 0.5, 0.6,  # structural, vascular, neural
        {SurgicalPhase.PLANNING: 1.0}
    ),
}
```

**Result**: Automatically available in both `seg_to_path.py` and `sancturary_spacings.py`

### Adding a New Surgical Phase

**Edit**: `surgical_marching_cubes.py`

```python
# Add to enum (line ~63)
class SurgicalPhase(Enum):
    # ... existing ...
    NEW_PHASE = "new_phase"
```

**Result**: Available everywhere via import

### Using Advanced Features

```python
# In your script
from surgical_marching_cubes import (
    CompleteSurgicalReconstructionSystem,
    SurgicalCorridorOptimizedOctree,
    AdaptiveMarchingCubesEngine
)

# These are exported in __all__ so they're part of the public API
system = CompleteSurgicalReconstructionSystem(...)
```

## Troubleshooting

### Import Error: "surgical_marching_cubes module not found"

**Cause**: Module not in Python path

**Solution**:
1. Ensure `surgical_marching_cubes.py` is in the same directory
2. Or add to `PYTHONPATH`:
   ```bash
   export PYTHONPATH="/path/to/xyz_pathplanning:$PYTHONPATH"
   ```

### "SurgicalPhase is not defined"

**Cause**: Not imported from `surgical_marching_cubes`

**Solution**: Add import at top of file:
```python
from surgical_marching_cubes import SurgicalPhase
```

### "Using local definitions" warning in sancturary_spacings.py

**Cause**: `surgical_marching_cubes.py` not found

**Effect**: Falls back to minimal local definitions (works but loses advanced features)

**Solution**: Ensure `surgical_marching_cubes.py` is accessible

## Future Improvements

Potential further organization:

```python
# Could split into:
surgical_mc/
├── __init__.py           # Exports public API
├── core.py               # extract_surface_mesh, etc.
├── anatomical.py         # AnatomicalRegion, SurgicalPhase
├── octree.py            # Octree-related classes
└── advanced.py          # Advanced reconstruction systems
```

But current single-file organization is fine for now given:
- Clear `__all__` export list
- Good internal section comments
- Manageable file size (~3400 lines)
- Everything logically related to marching cubes

## Summary

**Key Changes**:
1. ✅ All shared classes moved to `surgical_marching_cubes.py`
2. ✅ Other modules import from central location
3. ✅ No duplicate definitions
4. ✅ Clear `__all__` export list
5. ✅ Backwards compatibility maintained

**Result**: Clean, maintainable code organization with sensible module boundaries.

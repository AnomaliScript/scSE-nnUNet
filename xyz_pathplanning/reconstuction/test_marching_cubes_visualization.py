"""
Test marching cubes visualization on CTS1K_007 segmentation

This script will:
1. Load the CT volume and segmentation
2. Extract surface mesh using v2 (per-vertebra adaptive resolution)
3. Visualize with PyVista
4. Export to STL for external viewing
"""

import numpy as np
import nibabel as nib
import pyvista as pv
import sys
from pathlib import Path

# Add parent directory to path to import from colab folder
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from v2_per_vertebra (per-vertebra adaptive resolution)
from colab.v2_per_vertebra.marching_cubes_core_data import (
    extract_surface_mesh,
    create_pyvista_mesh,
    SurgicalPhase,
    AnatomicalRegion
)

print("="*70)
print("MARCHING CUBES VISUALIZATION TEST")
print("="*70)

# File paths
volume_path = "C:/Users/anoma/Downloads/spine-segmentation-data-cleaning/CTSpine1K/clean_volumes/CTS1K_007_0000.nii.gz"
seg_path = "C:/Users/anoma/Downloads/spine-segmentation-data-cleaning/CTSpine1K/clean_labels/CTS1K_007.nii.gz"

print(f"\nLoading data...")
print(f"  Volume: {volume_path}")
print(f"  Segmentation: {seg_path}")

# Load volume
volume_nii = nib.load(volume_path)
volume = volume_nii.get_fdata()
spacing = volume_nii.header.get_zooms()  # Get voxel spacing in mm

# Load segmentation
seg_nii = nib.load(seg_path)
segmentation = seg_nii.get_fdata().astype(np.int32)

print(f"\nData loaded:")
print(f"  Volume shape: {volume.shape}")
print(f"  Segmentation shape: {segmentation.shape}")
print(f"  Voxel spacing: {spacing} mm")
print(f"  Volume range: [{volume.min():.1f}, {volume.max():.1f}] HU")

# Check unique labels
unique_labels = np.unique(segmentation)
print(f"\nSegmentation labels found: {unique_labels}")
print(f"  --> Using v2_per_vertebra (per-vertebra adaptive resolution)")

# Count voxels per vertebra
print(f"\nLabel distribution:")
for label in unique_labels:
    if label > 0:
        count = np.sum(segmentation == label)
        vertebra = AnatomicalRegion(int(label))
        print(f"  {vertebra.name}: {count:,} voxels")

# Extract mesh using marching cubes with per-vertebra adaptive resolution
print(f"\n" + "="*70)
print("EXTRACTING SURFACE MESH")
print("="*70)
print(f"Using surgical phase: PLANNING")
print(f"This will automatically give C5 more detail than C1/C7")

# Normalize volume to [0, 1] for marching cubes
volume_normalized = (volume - volume.min()) / (volume.max() - volume.min())

print(f"\nNormalized volume range: [{volume_normalized.min():.3f}, {volume_normalized.max():.3f}]")

vertices, faces = extract_surface_mesh(
    volume_normalized,  # Use normalized volume!
    threshold=0.3,  # Threshold in normalized space (bone is higher intensity)
    spacing=spacing,
    segmentation=segmentation,  # Pass the C1-C7 labels!
    phase=SurgicalPhase.PLANNING  # Use planning phase
)

print(f"\nMesh extracted:")
print(f"  Vertices: {len(vertices):,}")
print(f"  Faces: {len(faces):,}")

# Create PyVista mesh
print(f"\nCreating PyVista mesh...")
mesh = create_pyvista_mesh(vertices, faces)

# Export to STL
output_stl = "c:/Users/anoma/Downloads/scse-nnUNet/xyz_pathplanning/reconstuction/CTS1K_007_mesh.stl"
print(f"\nExporting to STL: {output_stl}")
mesh.save(output_stl)
print(f"  ✓ Saved to {output_stl}")

# Compute mesh statistics
print(f"\n" + "="*70)
print("MESH STATISTICS")
print("="*70)
print(f"  Total vertices: {mesh.n_points:,}")
print(f"  Total faces: {mesh.n_faces:,}")
print(f"  Bounds (mm):")
print(f"    X: [{mesh.bounds[0]:.1f}, {mesh.bounds[1]:.1f}]")
print(f"    Y: [{mesh.bounds[2]:.1f}, {mesh.bounds[3]:.1f}]")
print(f"    Z: [{mesh.bounds[4]:.1f}, {mesh.bounds[5]:.1f}]")

# Visualize
print(f"\n" + "="*70)
print("LAUNCHING INTERACTIVE VISUALIZATION")
print("="*70)
print("Controls:")
print("  - Left click + drag: Rotate")
print("  - Right click + drag: Pan")
print("  - Scroll: Zoom")
print("  - 'q': Quit")
print("  - 's': Screenshot")
print("="*70)

# Create plotter with better settings
plotter = pv.Plotter()
plotter.add_mesh(
    mesh,
    color='lightblue',
    show_edges=False,
    smooth_shading=True,
    opacity=1.0
)
plotter.add_axes()
plotter.show_grid()
plotter.add_text(
    "CTS1K_007 - Cervical Spine Mesh\n(v2: Per-Vertebra Adaptive Resolution)",
    position='upper_left',
    font_size=10
)

# Show the mesh
plotter.show()

print(f"\n" + "="*70)
print("VISUALIZATION COMPLETE")
print("="*70)
print(f"Mesh saved to: {output_stl}")
print("You can open this file in any 3D viewer (MeshLab, Blender, etc.)")

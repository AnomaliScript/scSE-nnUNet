
import numpy as np
import nibabel as nib
import pyvista as pv
from skimage.measure import marching_cubes
import os

print("="*70)
print("SIMPLE MESH VISUALIZATION REPRODUCTION")
print("="*70)

# File paths
volume_path = "C:/Users/anoma/Downloads/spine-segmentation-data-cleaning/CTSpine1K/clean_volumes/CTS1K_007_0000.nii.gz"
seg_path = "C:/Users/anoma/Downloads/spine-segmentation-data-cleaning/CTSpine1K/clean_labels/CTS1K_007.nii.gz"

if not os.path.exists(volume_path):
    print(f"Volume not found: {volume_path}")
    exit(1)

print(f"\nLoading data...")
volume_nii = nib.load(volume_path)
volume = volume_nii.get_fdata()
spacing = volume_nii.header.get_zooms()

seg_nii = nib.load(seg_path)
segmentation = seg_nii.get_fdata()

# Create binary mask (all vertebrae combined)
bone_mask = (segmentation > 0).astype(float)

print(f"Running marching cubes...")
vertices, faces, normals, values = marching_cubes(
    bone_mask,
    level=0.5,
    spacing=spacing
)

print(f"\nMesh extracted:")
print(f"  Vertices: {len(vertices):,}")
print(f"  Faces: {len(faces):,}")

# Create PyVista mesh
faces_pv = np.hstack([[3] + list(face) for face in faces])
mesh = pv.PolyData(vertices, faces_pv)

print(f"\nOriginal mesh:")
print(f"  Vertices: {mesh.n_points:,}")
print(f"  Faces: {mesh.n_cells:,}")

# Apply Catmull-Clark subdivision for smoother mesh
print(f"\nApplying Catmull-Clark subdivision...")
n_subdivisions = 2
try:
    mesh_smooth = mesh.subdivide(n_subdivisions, subfilter='catmull-clark')
except Exception as e:
    print(f"Error during subdivision: {e}")
    # Try loop subdivision as fallback
    print("Trying loop subdivision instead...")
    mesh_smooth = mesh.subdivide(n_subdivisions, subfilter='loop')


print(f"\nSubdivided mesh:")
print(f"  Vertices: {mesh_smooth.n_points:,}")
print(f"  Faces: {mesh_smooth.n_cells:,}")

# Export
output_stl_smooth = "c:/Users/anoma/Downloads/scse-nnUNet/xyz_pathplanning/reconstuction/CTS1K_007_smooth_repro.stl"
mesh_smooth.save(output_stl_smooth)
print(f"Saved to {output_stl_smooth}")

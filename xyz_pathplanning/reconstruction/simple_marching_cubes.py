"""
Simple mesh visualization using scikit-image marching cubes
No complicated octrees, just works.
"""

import numpy as np
import nibabel as nib
import pyvista as pv
from skimage.measure import marching_cubes
import scipy.ndimage
import vtk
import time

print("="*70)
print("SIMPLE MESH VISUALIZATION")
print("="*70)

# Start total timer
total_start_time = time.time()

# Configuration
DO_RESAMPLE = True              # Resample to 1.0mm isotropic
MESH_EXTRACTION = 'mc' # Options: 'mc', 'fe'
MESH_SMOOTHING = 'catmull_clark' # Options: 'windowed_sinc', 'laplacian', 'bilateral', 'taubin', 'catmull_clark', 'none'
SUBDIVISION_LEVEL = 3           # Recommended: 2 or 3. 4 is very heavy.

# File paths
volume_path = r"C:\Users\anoma\Downloads\spine-segmentation-data-cleaning\VerSe_clean_v3\clean_volumes\VerSe_086_0000.nii.gz"
seg_path = r"C:\Users\anoma\Downloads\spine-segmentation-data-cleaning\VerSe_clean_v3\clean_labels\VerSe_086.nii.gz"

def resample_volume(volume, spacing, new_spacing=[1.0, 1.0, 1.0]):
    """Resample volume to isotropic spacing using spline interpolation."""
    print(f"Resampling volume from {spacing} to {new_spacing}...")
    resize_factor = spacing / np.array(new_spacing)
    new_shape = np.round(volume.shape * resize_factor)
    real_resize_factor = new_shape / volume.shape
    
    # Order 1 (linear) is faster and usually sufficient for binary masks, 
    # but for CT data Order 3 (cubic) is better.
    resampled_vol = scipy.ndimage.zoom(volume, real_resize_factor, order=3, prefilter=False)
    
    print(f"  Old shape: {volume.shape}")
    print(f"  New shape: {resampled_vol.shape}")
    return resampled_vol, new_spacing


def smooth_windowed_sinc(mesh, n_iter=20, pass_band=0.1, boundary_smoothing=True, feature_smoothing=True, feature_angle=60.0, edge_angle=15.0, non_manifold_smoothing=True):
    """Apply vtkWindowedSincPolyDataFilter to a PyVista mesh."""
    print("  Using vtkWindowedSincPolyDataFilter...")
    alg = vtk.vtkWindowedSincPolyDataFilter()
    alg.SetInputData(mesh)
    alg.SetNumberOfIterations(n_iter)
    alg.SetPassBand(pass_band)
    alg.SetBoundarySmoothing(boundary_smoothing)
    alg.SetFeatureEdgeSmoothing(feature_smoothing)
    alg.SetFeatureAngle(feature_angle)
    alg.SetEdgeAngle(edge_angle)
    alg.SetNonManifoldSmoothing(non_manifold_smoothing)
    alg.Update()
    return pv.wrap(alg.GetOutput())

def apply_catmull_clark(mesh, iterations=1):
    """Apply Catmull-Clark subdivision using the user's module (requires OpenMesh)."""
    print("  Applying Catmull-Clark Smoothing...")
    try:
        import openmesh
        import sys
        from pathlib import Path
        # Add reconstuction directory to path to import catmull_clark
        sys.path.insert(0, str(Path(__file__).parent))
        import catmull_clark
    except ImportError as e:
        print(f"  [WARNING] Could not import openmesh or catmull_clark: {e}")
        print("  Falling back to built-in Loop Subdivision (approximate equivalent for triangles).")
        return mesh.subdivide(iterations, subfilter='loop')

    # Convert PyVista to OpenMesh
    print("  Converting to OpenMesh format...")
    om_mesh = openmesh.PolyMesh()
    # Add vertices
    # openmesh requires a list of points
    om_mesh.add_vertices(mesh.points)
    
    # Add faces
    # mesh.faces is flat: [n, v1, v2, v3, n, v1, v2, v3...]
    i = 0
    faces = mesh.faces
    n_faces = mesh.n_cells
    
    # This loop is slow in Python. For 500k faces it might take a while.
    # But it's necessary to use the user's code.
    while i < len(faces):
        n = faces[i]
        face_indices = faces[i+1 : i+1+n]
        v_handles = [om_mesh.vertex_handle(idx) for idx in face_indices]
        om_mesh.add_face(v_handles)
        i += n + 1

    # Apply Catmull-Clark
    print(f"  Running Catmull-Clark ({iterations} iterations)...")
    try:
        om_mesh = catmull_clark.catmull_clark_iter(om_mesh, iterations)
    except Exception as e:
        print(f"  [ERROR] Catmull-Clark failed: {e}")
        return mesh

    # Convert back to PyVista
    print("  Converting back to PyVista...")
    new_points = om_mesh.points()
    new_faces = []
    for f in om_mesh.faces():
        v_handles = list(om_mesh.fv(f))
        new_faces.append(len(v_handles))
        new_faces.extend([vh.idx() for vh in v_handles])
    
    # Create PyVista mesh
    # Note: CC produces quads (mostly), so we handle generic polygons
    return pv.PolyData(new_points, new_faces)

print(f"\nLoading data...")
print(f"  Volume: {volume_path}")
print(f"  Segmentation: {seg_path}")

# Load data
volume_nii = nib.load(volume_path)
volume = volume_nii.get_fdata()
spacing = np.array(volume_nii.header.get_zooms())

seg_nii = nib.load(seg_path)
segmentation = seg_nii.get_fdata()

print(f"\nData loaded:")
print(f"  Volume shape: {volume.shape}")
print(f"  Voxel spacing: {spacing} mm")

# 1. Resample (Isotropic)
if DO_RESAMPLE:
    step1_start = time.time()
    print(f"\n[STEP 1] Resampling to Isotropic...")
    # Resample volume
    volume, spacing = resample_volume(volume, spacing)
    # Resample segmentation (Nearest Neighbor for labels -> order=0)
    resize_factor = spacing / np.array([1.0, 1.0, 1.0]) # Re-calc factor based on new spacing (which is 1.0)
    # Actually we need to use the original spacing to calculate factor relative to current volume state
    # But since we overwrote 'volume' and 'spacing', let's just reload/recalc or be careful.
    # Simplified: Just resample segmentation using the same target spacing logic
    # We need the original spacing again for the segmentation calculation
    orig_spacing = np.array(volume_nii.header.get_zooms())
    resize_factor = orig_spacing / np.array([1.0, 1.0, 1.0])
    new_shape = np.round(segmentation.shape * resize_factor)
    real_resize_factor = new_shape / segmentation.shape
    segmentation = scipy.ndimage.zoom(segmentation, real_resize_factor, order=0, prefilter=False)
    step1_time = time.time() - step1_start
    print(f"  ⏱ Time: {step1_time:.2f}s")

# 2. Mesh Extraction
step2_start = time.time()
print(f"\n[STEP 2] Extracting Mesh ({MESH_EXTRACTION.replace('_', ' ').title()})...")

# Create binary mask (all vertebrae combined)
bone_mask = (segmentation > 0).astype(float)

if MESH_EXTRACTION == 'mc':
    # Use scikit-image marching cubes
    vertices, faces, normals, values = marching_cubes(
        bone_mask,
        level=0.5,
        spacing=spacing
    )
    # Create PyVista mesh
    faces_pv = np.hstack([[3] + list(face) for face in faces])
    mesh = pv.PolyData(vertices, faces_pv)

elif MESH_EXTRACTION == 'fe':
    # Use VTK Flying Edges (via PyVista)
    # Convert to PyVista ImageData
    # Note: dimensions define the grid points, so for a volume of shape (x, y, z),
    # we need dimensions = (x, y, z) which creates x*y*z points
    grid = pv.ImageData()
    grid.dimensions = bone_mask.shape
    grid.spacing = spacing
    grid.point_data['values'] = bone_mask.flatten(order='F')

    # Extract surface using Flying Edges (contour)
    mesh = grid.contour([0.5], scalars='values', method='flying_edges')

else:
    raise ValueError(f"Unknown mesh extraction method: {MESH_EXTRACTION}")

step2_time = time.time() - step2_start
print(f"  Vertices: {mesh.n_points:,}")
print(f"  Faces: {mesh.n_cells:,}")
print(f"  ⏱ Time: {step2_time:.2f}s")

# 3. Mesh Smoothing
if MESH_SMOOTHING != 'none':
    step3_start = time.time()

if MESH_SMOOTHING == 'windowed_sinc':
    print(f"\n[STEP 3] Mesh Smoothing (Windowed Sinc)...")
    # Generate feature edges first if we want to preserve them?
    # Windowed Sinc has 'boundary_smoothing' but not explicit feature edge protection like Bilateral.
    # But it's very good.
    mesh = smooth_windowed_sinc(mesh, n_iter=20, pass_band=0.1, boundary_smoothing=False)
elif MESH_SMOOTHING == 'bilateral':
    print(f"\n[STEP 3] Mesh Smoothing (Bilateral - WindowedSinc with Feature Edges)...")
    # "Bilateral" for meshes often means preserving sharp features while smoothing flat areas.
    # VTK's WindowedSinc with FeatureEdgeSmoothingOn is the standard way to do this.
    # We use a tighter pass_band for less shrinkage and enable feature smoothing.
    mesh = smooth_windowed_sinc(
        mesh,
        n_iter=20,
        pass_band=0.01,
        boundary_smoothing=True,
        feature_smoothing=True,
        feature_angle=60.0, # Angle to detect sharp edges (degrees)
        edge_angle=15.0,    # Angle to detect edges (degrees)
        non_manifold_smoothing=True
    )
elif MESH_SMOOTHING == 'taubin':
    print(f"\n[STEP 3] Mesh Smoothing (Taubin - Non-shrinking)...")
    # Taubin smoothing prevents the "shrinkage" effect of Laplacian smoothing
    mesh = mesh.smooth_taubin(n_iter=50, pass_band=0.1)
elif MESH_SMOOTHING == 'catmull_clark':
    print(f"\n[STEP 3] Mesh Smoothing (Catmull-Clark)...")
    # Use the custom module
    # Note: CC is a subdivision scheme, so it increases resolution.
    # We use 1 iteration here as a "smoothing" step.
    mesh = apply_catmull_clark(mesh, iterations=1)
elif MESH_SMOOTHING == 'laplacian':
    print(f"\n[STEP 3] Mesh Smoothing (Laplacian)...")
    mesh = mesh.smooth(n_iter=50)

if MESH_SMOOTHING != 'none':
    step3_time = time.time() - step3_start
    print(f"  ⏱ Time: {step3_time:.2f}s")

# 4. Subdivision
step4_start = time.time()
print(f"\n[STEP 4] Loop Subdivision (Level {SUBDIVISION_LEVEL})...")
mesh_smooth = mesh.subdivide(SUBDIVISION_LEVEL, subfilter='loop')
step4_time = time.time() - step4_start

print(f"\nFinal Mesh:")
print(f"  Vertices: {mesh_smooth.n_points:,}")
print(f"  Faces: {mesh_smooth.n_cells:,}")
print(f"  ⏱ Time: {step4_time:.2f}s")

# Export
save_start = time.time()
output_stl = f"c:/Users/anoma/Downloads/scse-nnUNet/xyz_pathplanning/reconstruction/VerSe_086_{MESH_EXTRACTION}_{MESH_SMOOTHING}.stl"
print(f"\nExporting to {output_stl}...")
mesh_smooth.save(output_stl)
save_time = time.time() - save_start
print(f"  ⏱ Save time: {save_time:.2f}s")

# Total time
total_time = time.time() - total_start_time
print(f"\n{'='*70}")
print(f"TOTAL PIPELINE TIME: {total_time:.2f}s ({total_time/60:.2f} minutes)")
print(f"{'='*70}")

# Visualize
print(f"\n" + "="*70)
print("LAUNCHING VISUALIZATION")
print("="*70)
plotter = pv.Plotter()
plotter.add_mesh(mesh_smooth, color='lightblue', smooth_shading=True)
plotter.add_text(f"Advanced Pipeline\nResample: {DO_RESAMPLE}\nExtraction: {MESH_EXTRACTION}\nMesh Smooth: {MESH_SMOOTHING}\nSubdiv: {SUBDIVISION_LEVEL}\nTotal Time: {total_time:.1f}s", font_size=10)
plotter.add_axes()

# Set isometric view
# Options: 'xy', 'xz', 'yz', 'yx', 'zx', 'zy', 'iso'
plotter.camera_position = 'iso'  # Isometric view
# OR set custom camera position for better angle:
# plotter.camera_position = [(1, 1, 1), (0, 0, 0), (0, 0, 1)]  # (position, focal_point, view_up)

plotter.show()


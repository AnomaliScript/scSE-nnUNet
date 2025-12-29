#!/usr/bin/env python3
"""
Cervical Spine Surgical Path Planning System
Uses Fast Marching Method (FMM) for optimal path finding

This is a reference implementation for educational purposes.
Modify and adapt as needed for your project.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import nibabel as nib
from scipy.ndimage import distance_transform_edt
try:
    import skfmm
except ImportError:
    print("ERROR: skfmm not installed. Run: pip install scikit-fmm")
    exit(1)

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    print("WARNING: PyVista not installed. 3D visualization disabled.")
    print("Install with: pip install pyvista")
    PYVISTA_AVAILABLE = False

# Import surgical marching cubes module for volume rendering
try:
    from surgical_marching_cubes import extract_surface_mesh, create_pyvista_mesh
    MARCHING_CUBES_AVAILABLE = True
except ImportError:
    print("WARNING: surgical_marching_cubes module not found.")
    print("Volume rendering will use fallback scatter plot visualization.")
    MARCHING_CUBES_AVAILABLE = False
    
# ============================================================================
# CONFIGURATION
# ============================================================================

SEGMENTATION_FILES = [
    r"C:\\Users\\anoma\\Downloads\\spine-segmentation-data-cleaning\\CTSpine1K\\clean_labels\\CTS1K_007.nii.gz"
]

# Safety margin in voxels (conservative estimate for unseen vessels/nerves)
SAFETY_MARGIN_MM = 5.0

# Use dummy data for testing (set to False when you have real data)
USE_DUMMY_DATA = False

# Enable 3D visualization with PyVista (opens after path is computed)
ENABLE_3D_VISUALIZATION = True


# ============================================================================
# CORE PATH PLANNING FUNCTIONS
# ============================================================================

def create_dummy_cervical_segmentation(shape=(256, 256, 100)):
    """
    Generate synthetic cervical vertebrae segmentation for testing
    Creates realistic-looking vertebral bodies C1-C7
    """
    print("Generating dummy cervical vertebrae segmentation...")
    seg = np.zeros(shape, dtype=np.uint8)
    
    # Create 7 vertebrae (C1-C7) along z-axis
    center_x, center_y = shape[0] // 2, shape[1] // 2
    
    for i in range(7):
        z_center = 15 + i * 12  # Space vertebrae along z-axis
        
        # Create vertebral body (roughly rectangular with canal)
        for z in range(z_center - 4, z_center + 4):
            if z < 0 or z >= shape[2]:
                continue
                
            # Vertebral body (outer rectangle)
            seg[center_x-20:center_x+20, center_y-25:center_y+25, z] = i + 1
            
            # Spinal canal (hollow center)
            seg[center_x-8:center_x+8, center_y-8:center_y+8, z] = 0
            
            # Add some lateral processes
            seg[center_x-28:center_x-20, center_y-10:center_y+10, z] = i + 1
            seg[center_x+20:center_x+28, center_y-10:center_y+10, z] = i + 1
    
    print(f"Created dummy segmentation: {shape}, 7 vertebrae")
    return seg


def load_segmentation(filepath=None):
    """
    Load vertebrae segmentation from NIfTI file or generate dummy data
    
    Args:
        filepath: Path to .nii or .nii.gz file
        
    Returns:
        segmentation: 3D numpy array
        affine: Affine transformation matrix (for real data)
    """
    if USE_DUMMY_DATA or filepath is None:
        seg = create_dummy_cervical_segmentation()
        affine = np.eye(4)  # Identity matrix for dummy data
        return seg, affine
    
    try:
        print(f"Loading segmentation from: {filepath}")
        nifti_img = nib.load(filepath)
        segmentation = nifti_img.get_fdata()
        affine = nifti_img.affine
        print(f"Loaded: shape={segmentation.shape}, dtype={segmentation.dtype}")
        return segmentation, affine
    except FileNotFoundError:
        print(f"WARNING: File not found: {filepath}")
        print("Using dummy data instead...")
        seg = create_dummy_cervical_segmentation()
        affine = np.eye(4)
        return seg, affine


def compute_distance_transform(segmentation):
    # Calculate distance from each voxel to nearest bone
    # Outputs distance_map: 3D array with distances in voxels
 
    print("Computing distance transform from bone surfaces...")
    
    # Create binary mask (any non-zero value = bone)
    bone_mask = (segmentation > 0).astype(np.uint8)
    
    # Calculate distance to nearest bone voxel
    distance_map = distance_transform_edt(bone_mask == 0)
    
    print(f"Distance range: {distance_map.min():.2f} to {distance_map.max():.2f} voxels")
    return distance_map


def create_speed_map(distance_map, safety_margin=5.0):
    # Convert distance map to speed map for FMM
    # Speed = how fast the wavefront can travel (high speed = safe)
    # distance_map: Distance to nearest bone in voxels (3D pixels)
    # safety_margin: Minimum safe distance in voxels
    # Output: speed_map: Values > 0 (higher = safer/faster travel)
    
    print(f"Creating speed map with {safety_margin} voxel safety margin...")
    
    # Speed increases with distance from bone
    # Add safety margin to avoid division by zero and ensure safe paths
    speed_map = distance_map + safety_margin
    
    # Normalize to reasonable range (0.1 to 1.0)
    speed_map = speed_map / speed_map.max()
    speed_map = np.clip(speed_map, 0.1, 1.0)  # Minimum speed to avoid zero
    
    return speed_map


def plan_path_fmm(speed_map, start_point, end_point):
    """
    Use Fast Marching Method to find optimal path
    
    Args:
        speed_map: 3D array of travel speeds (higher = better)
        start_point: (x, y, z) tuple for entry point
        end_point: (x, y, z) tuple for target point
        
    Returns:
        path: Nx3 array of (x,y,z) coordinates
        travel_time: Total "travel time" (lower = better path)
    """
    print(f"Planning path from {start_point} to {end_point}...")
    
    # Create phi: distance field (negative at start, positive elsewhere)
    phi = np.ones_like(speed_map)
    phi[start_point] = -1  # Start point marked as negative
    
    # Run Fast Marching to compute travel time from start to all points
    try:
        travel_time = skfmm.travel_time(phi, speed_map)
    except Exception as e:
        print(f"ERROR in FMM: {e}")
        return None, None
    
    # Trace path from end back to start using gradient descent
    path = [end_point]
    current = np.array(end_point, dtype=float)
    
    max_iterations = 10000
    step_size = 0.5
    
    for iteration in range(max_iterations):
        # Calculate gradient of travel time (points toward start)
        grad = np.array(np.gradient(travel_time))
        
        # Get gradient at current position (with bounds checking)
        pos = tuple(np.clip(current.astype(int), 0, 
                           [s-1 for s in speed_map.shape]))
        gradient_vec = np.array([grad[i][pos] for i in range(3)])
        
        # Move opposite to gradient (downhill toward start)
        if np.linalg.norm(gradient_vec) < 1e-6:
            break
            
        current = current - step_size * gradient_vec / np.linalg.norm(gradient_vec)
        
        # Check bounds
        if np.any(current < 0) or np.any(current >= speed_map.shape):
            break
            
        path.append(tuple(current.astype(int)))
        
        # Check if we reached start
        if np.linalg.norm(current - np.array(start_point)) < 2.0:
            break
    
    path = np.array(path)
    total_time = travel_time[end_point]
    
    print(f"Path found: {len(path)} points, travel time: {total_time:.2f}")
    return path, total_time


def calculate_safety_metrics(path, distance_map):
    """
    Calculate safety metrics for the planned path
    
    Args:
        path: Nx3 array of path coordinates
        distance_map: Distance to nearest bone
        
    Returns:
        metrics: Dictionary with safety statistics
    """
    # Get distance values along path
    distances = []
    for point in path:
        p = tuple(point.astype(int))
        if all(0 <= p[i] < distance_map.shape[i] for i in range(3)):
            distances.append(distance_map[p])
    
    distances = np.array(distances)
    
    metrics = {
        'min_clearance': float(np.min(distances)),
        'max_clearance': float(np.max(distances)),
        'avg_clearance': float(np.mean(distances)),
        'path_length': len(path),
        'safe': np.min(distances) >= 3.0  # 3 voxel minimum
    }
    
    return metrics


# ============================================================================
# 3D VISUALIZATION WITH PYVISTA
# ============================================================================

def get_spacing_from_affine(affine):
    """
    Extract voxel spacing from NIfTI affine matrix

    Args:
        affine: 4x4 affine transformation matrix

    Returns:
        Tuple of (x_spacing, y_spacing, z_spacing) in mm
    """
    spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))
    return tuple(spacing)


def create_vertebrae_mesh(segmentation, threshold=0.5, spacing=(1.0, 1.0, 1.0)):
    """
    Convert segmentation volume to 3D surface mesh using marching cubes

    Args:
        segmentation: 3D numpy array
        threshold: Value to use for surface extraction
        spacing: Voxel spacing in (x, y, z) mm

    Returns:
        PyVista mesh object
    """
    if not PYVISTA_AVAILABLE:
        return None

    # Use surgical marching cubes if available (better quality)
    if MARCHING_CUBES_AVAILABLE:
        print("Using surgical marching cubes for surface extraction...")
        vertices, faces = extract_surface_mesh(segmentation, threshold=threshold, spacing=spacing)

        if len(vertices) > 0:
            mesh = create_pyvista_mesh(vertices, faces)
            return mesh
        else:
            print("WARNING: Marching cubes failed, trying PyVista fallback...")

    # Fallback to PyVista's built-in contour method
    try:
        grid = pv.wrap(segmentation)
        mesh = grid.contour([threshold], scalars=segmentation.ravel(order='F'))
        return mesh
    except Exception as e:
        print(f"ERROR creating mesh: {e}")
        return None


def visualize_path_3d(segmentation, path, start_point, end_point, metrics, spacing=(1.0, 1.0, 1.0)):
    """
    Create interactive 3D visualization of vertebrae and surgical path

    Args:
        segmentation: 3D vertebrae segmentation
        path: Nx3 array of path coordinates
        start_point: (x,y,z) start coordinates
        end_point: (x,y,z) end coordinates
        metrics: Dictionary of safety metrics
        spacing: Voxel spacing in (x, y, z) mm
    """
    if not PYVISTA_AVAILABLE:
        print("PyVista not available. Skipping 3D visualization.")
        return

    if not ENABLE_3D_VISUALIZATION:
        return

    print("\nOpening 3D visualization...")

    # Create plotter
    plotter = pv.Plotter()
    plotter.set_background('black')

    # Add vertebrae mesh with proper spacing
    print("Creating vertebrae mesh...")
    vertebrae_mesh = create_vertebrae_mesh(segmentation, threshold=0.5, spacing=spacing)
    
    if vertebrae_mesh is not None and vertebrae_mesh.n_points > 0:
        plotter.add_mesh(
            vertebrae_mesh,
            color='lightgray',
            opacity=0.4,
            smooth_shading=True,
            label='Vertebrae'
        )
    
    # Add path as tube
    if path is not None and len(path) > 1:
        print("Adding path...")
        path_polydata = pv.PolyData(path)
        
        # Create line connecting path points
        lines = np.full((len(path)-1, 3), 2, dtype=np.int_)
        lines[:, 1] = np.arange(len(path)-1)
        lines[:, 2] = np.arange(1, len(path))
        path_polydata.lines = lines
        
        # Add as tube for better visibility
        tube = path_polydata.tube(radius=0.8)
        plotter.add_mesh(
            tube,
            color='blue',
            label='Surgical Path'
        )
    
    # Add start point
    if start_point is not None:
        start_sphere = pv.Sphere(radius=2.0, center=start_point)
        plotter.add_mesh(
            start_sphere,
            color='red',
            label='Start Point'
        )
        plotter.add_point_labels(
            [start_point],
            ['START'],
            font_size=20,
            text_color='red',
            point_color='red',
            point_size=10
        )
    
    # Add end point
    if end_point is not None:
        end_sphere = pv.Sphere(radius=2.0, center=end_point)
        plotter.add_mesh(
            end_sphere,
            color='green',
            label='Target Point'
        )
        plotter.add_point_labels(
            [end_point],
            ['TARGET'],
            font_size=20,
            text_color='green',
            point_color='green',
            point_size=10
        )
    
    # Add text with metrics
    if metrics:
        status = "SAFE" if metrics['safe'] else "WARNING"
        metrics_text = (
            f"Path Planning Results\n"
            f"Status: {status}\n"
            f"Min Clearance: {metrics['min_clearance']:.2f} voxels\n"
            f"Avg Clearance: {metrics['avg_clearance']:.2f} voxels\n"
            f"Path Length: {metrics['path_length']} points"
        )
        plotter.add_text(
            metrics_text,
            position='upper_right',
            font_size=10,
            color='white'
        )
    
    # Add title
    plotter.add_text(
        "Cervical Spine Surgical Path Planning - 3D View",
        position='upper_left',
        font_size=14,
        color='white',
        font='arial'
    )
    
    # Add legend
    plotter.add_legend(size=(0.2, 0.2), loc='lower_left')
    
    # Add axes
    plotter.add_axes()
    
    # Show orientation widget
    plotter.add_camera_orientation_widget()
    
    # Set camera position for good initial view
    plotter.camera_position = 'xy'
    plotter.camera.zoom(1.2)
    
    # Show the plot
    print("3D visualization window opened. Rotate with mouse, zoom with scroll.")
    print("Close the window to continue...")
    plotter.show()


# ============================================================================
# INTERACTIVE VISUALIZATION
# ============================================================================

class InteractivePathPlanner:
    """
    Interactive matplotlib interface for path planning
    Click to select start and end points, see path drawn in real-time
    """

    def __init__(self, segmentation, distance_map, speed_map, affine=None):
        self.segmentation = segmentation
        self.distance_map = distance_map
        self.speed_map = speed_map
        self.affine = affine if affine is not None else np.eye(4)

        # Extract voxel spacing for accurate rendering
        self.spacing = get_spacing_from_affine(self.affine)

        # Selected points
        self.start_point = None
        self.end_point = None

        # Computed path and metrics (for 3D visualization)
        self.computed_path = None
        self.computed_metrics = None

        # Current slice to display
        self.current_slice = segmentation.shape[2] // 2

        # View toggle state ('axial' or 'sagittal')
        self.current_view = 'axial'

        # Real-time 3D view (PyVista plotter embedded)
        self.pyvista_plotter = None
        self.vertebrae_mesh = None

        # 3D crosshair system for point selection
        self.crosshair_x = segmentation.shape[0] // 2
        self.crosshair_y = segmentation.shape[1] // 2
        self.crosshair_z = segmentation.shape[2] // 2
        self.crosshair_mode = False  # Toggle 3D crosshair point selection

        # Zoom control
        self.zoom_level = 1.0  # 1.0 = normal, >1.0 = zoomed in

        # Create figure with 1x2 layout (single 2D view + 3D view)
        self.fig = plt.figure(figsize=(18, 9))
        self.fig.patch.set_facecolor('black')
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1, 1])

        self.ax_2d = self.fig.add_subplot(gs[0, 0])  # Single toggleable 2D view
        self.ax_3d = self.fig.add_subplot(gs[0, 1], projection='3d')  # 3D view

        self.fig.suptitle('Cervical Spine Surgical Path Planning - Interactive Interface',
                         fontsize=14, fontweight='bold', color='white')

        self.setup_display()
        
    def setup_display(self):
        """Initialize the visualization"""

        # Pre-compute vertebrae mesh for 3D view (only once)
        if MARCHING_CUBES_AVAILABLE or PYVISTA_AVAILABLE:
            print("Pre-computing vertebrae mesh for 3D view...")
            print(f"  Using voxel spacing: {self.spacing} mm")
            self.vertebrae_mesh = create_vertebrae_mesh(
                self.segmentation,
                threshold=0.5,
                spacing=self.spacing
            )

        self.update_display()

        # Connect events
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
    def update_display(self):
        """Redraw all views"""

        # 2D VIEW (single toggleable view)
        self.ax_2d.clear()
        self.ax_2d.set_facecolor('black')

        if self.current_view == 'axial':
            axial_slice = self.segmentation[:, :, self.current_slice]
            self.ax_2d.imshow(axial_slice.T, cmap='gray', origin='lower')
            mode_text = " [CROSSHAIR MODE]" if self.crosshair_mode else ""
            self.ax_2d.set_title(
                f'AXIAL View (Slice {self.current_slice}/{self.segmentation.shape[2]-1}){mode_text}\n'
                'Click: points | ↑↓: slice | V: toggle view | C: crosshair mode',
                fontsize=10, fontweight='bold', color='white'
            )

            # Draw crosshair if in crosshair mode
            if self.crosshair_mode:
                self.ax_2d.axhline(y=self.crosshair_y, color='cyan', linestyle='--', linewidth=1, alpha=0.7)
                self.ax_2d.axvline(x=self.crosshair_x, color='cyan', linestyle='--', linewidth=1, alpha=0.7)
                self.ax_2d.plot(self.crosshair_x, self.crosshair_y, 'c+', markersize=20, markeredgewidth=2)

            # Draw points if selected
            if self.start_point:
                circle = Circle((self.start_point[0], self.start_point[1]),
                              3, color='red', linewidth=2, fill=False)
                self.ax_2d.add_patch(circle)
                self.ax_2d.plot(self.start_point[0], self.start_point[1],
                              'r*', markersize=15)

            if self.end_point:
                circle = Circle((self.end_point[0], self.end_point[1]),
                              3, color='lime', linewidth=2, fill=False)
                self.ax_2d.add_patch(circle)
                self.ax_2d.plot(self.end_point[0], self.end_point[1],
                              'g*', markersize=15)

        elif self.current_view == 'sagittal':
            mid_x = self.segmentation.shape[0] // 2
            sagittal_slice = self.segmentation[mid_x, :, :]
            self.ax_2d.imshow(sagittal_slice.T, cmap='gray', origin='lower')
            mode_text = " [CROSSHAIR MODE]" if self.crosshair_mode else ""
            self.ax_2d.set_title(
                f'SAGITTAL View (X={mid_x}){mode_text}\n'
                'Click: points | V: toggle view | C: crosshair mode',
                fontsize=10, fontweight='bold', color='white'
            )

            # Draw crosshair if in crosshair mode
            if self.crosshair_mode:
                self.ax_2d.axhline(y=self.crosshair_z, color='cyan', linestyle='--', linewidth=1, alpha=0.7)
                self.ax_2d.axvline(x=self.crosshair_y, color='cyan', linestyle='--', linewidth=1, alpha=0.7)
                self.ax_2d.plot(self.crosshair_y, self.crosshair_z, 'c+', markersize=20, markeredgewidth=2)

            # Draw points if selected (project to sagittal)
            if self.start_point:
                self.ax_2d.plot(self.start_point[1], self.start_point[2],
                              'r*', markersize=15)

            if self.end_point:
                self.ax_2d.plot(self.end_point[1], self.end_point[2],
                              'g*', markersize=15)

        # 3D VIEW
        self.update_3d_view()

        plt.draw()
        
    def update_3d_view(self):
        """Update the 3D matplotlib view"""
        self.ax_3d.clear()

        # Set black background
        self.ax_3d.set_facecolor('black')
        self.ax_3d.xaxis.pane.fill = False
        self.ax_3d.yaxis.pane.fill = False
        self.ax_3d.zaxis.pane.fill = False
        self.ax_3d.xaxis.pane.set_edgecolor('gray')
        self.ax_3d.yaxis.pane.set_edgecolor('gray')
        self.ax_3d.zaxis.pane.set_edgecolor('gray')
        self.ax_3d.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)

        if self.vertebrae_mesh is None:
            # Fallback: voxel visualization using scatter plot
            indices = np.argwhere(self.segmentation > 0)
            if len(indices) > 0:
                # Downsample for performance
                step = max(1, len(indices) // 5000)
                indices = indices[::step]
                self.ax_3d.scatter(indices[:, 0], indices[:, 1], indices[:, 2],
                                  c='white', alpha=0.5, s=1, marker='.')
        else:
            # Render mesh as smooth 3D surface using marching cubes triangles
            vertices = self.vertebrae_mesh.points

            if len(vertices) > 0:
                # PyVista faces format: [n, v1, v2, v3, n, v1, v2, v3, ...]
                # Need to extract triangles properly
                faces_raw = self.vertebrae_mesh.faces

                # Parse faces: iterate through the array
                triangles_list = []
                i = 0
                while i < len(faces_raw):
                    n_verts = faces_raw[i]
                    if n_verts == 3:  # Triangle
                        triangle_indices = faces_raw[i+1:i+4]
                        triangles_list.append(vertices[triangle_indices])
                        i += 4
                    else:
                        # Skip non-triangular faces
                        i += n_verts + 1

                if len(triangles_list) > 0:
                    # Downsample for performance in matplotlib
                    face_step = max(1, len(triangles_list) // 3000)
                    triangles_sampled = triangles_list[::face_step]

                    # Plot triangular mesh surface with smooth shading
                    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

                    # Create mesh collection with smooth appearance
                    mesh_collection = Poly3DCollection(triangles_sampled,
                                                       alpha=0.3,
                                                       facecolor='lightgray',
                                                       edgecolor='darkgray',
                                                       linewidths=0.2)
                    self.ax_3d.add_collection3d(mesh_collection)

        # Draw 3D crosshair axes if in crosshair mode
        if self.crosshair_mode:
            x_range = [0, self.segmentation.shape[0]-1]
            y_range = [0, self.segmentation.shape[1]-1]
            z_range = [0, self.segmentation.shape[2]-1]

            # X-axis crosshair (red line through crosshair position)
            self.ax_3d.plot([x_range[0], x_range[1]], [self.crosshair_y, self.crosshair_y],
                           [self.crosshair_z, self.crosshair_z], 'r-', linewidth=2, alpha=0.7, label='X-axis')

            # Y-axis crosshair (green line)
            self.ax_3d.plot([self.crosshair_x, self.crosshair_x], [y_range[0], y_range[1]],
                           [self.crosshair_z, self.crosshair_z], 'g-', linewidth=2, alpha=0.7, label='Y-axis')

            # Z-axis crosshair (blue line)
            self.ax_3d.plot([self.crosshair_x, self.crosshair_x], [self.crosshair_y, self.crosshair_y],
                           [z_range[0], z_range[1]], 'b-', linewidth=2, alpha=0.7, label='Z-axis')

            # Crosshair center point
            self.ax_3d.scatter(self.crosshair_x, self.crosshair_y, self.crosshair_z,
                             c='cyan', s=150, marker='o', edgecolors='white', linewidths=2)

        # Draw start and end points
        if self.start_point:
            self.ax_3d.scatter(*self.start_point, c='red', s=200, marker='*',
                             edgecolors='white', linewidths=1, label='Start', zorder=10)

        if self.end_point:
            self.ax_3d.scatter(*self.end_point, c='lime', s=200, marker='*',
                             edgecolors='white', linewidths=1, label='Target', zorder=10)

        # Draw path if computed
        if self.computed_path is not None and len(self.computed_path) > 0:
            path = self.computed_path
            self.ax_3d.plot(path[:, 0], path[:, 1], path[:, 2],
                           'cyan', linewidth=3, label='Path', zorder=5)

        # Style axes
        self.ax_3d.set_xlabel('X', color='white', fontweight='bold')
        self.ax_3d.set_ylabel('Y', color='white', fontweight='bold')
        self.ax_3d.set_zlabel('Z', color='white', fontweight='bold')
        self.ax_3d.tick_params(colors='white', labelsize=8)
        zoom_text = f" [Zoom: {self.zoom_level:.1f}x]" if self.zoom_level != 1.0 else ""
        self.ax_3d.set_title(f'3D View (Real-time){zoom_text} - Drag to rotate',
                            fontsize=10, fontweight='bold', color='white')

        if self.start_point or self.end_point or self.computed_path is not None or self.crosshair_mode:
            legend = self.ax_3d.legend(loc='upper left', fontsize=8, facecolor='black', edgecolor='white')
            for text in legend.get_texts():
                text.set_color('white')

        # Set aspect ratio with zoom control
        max_range = np.array([
            self.segmentation.shape[0],
            self.segmentation.shape[1],
            self.segmentation.shape[2]
        ]).max() / 2.0

        # Apply zoom (smaller range = more zoomed in)
        zoomed_range = max_range / self.zoom_level

        mid_x = self.segmentation.shape[0] // 2
        mid_y = self.segmentation.shape[1] // 2
        mid_z = self.segmentation.shape[2] // 2

        self.ax_3d.set_xlim(mid_x - zoomed_range, mid_x + zoomed_range)
        self.ax_3d.set_ylim(mid_y - zoomed_range, mid_y + zoomed_range)
        self.ax_3d.set_zlim(mid_z - zoomed_range, mid_z + zoomed_range)

    def on_key(self, event):
        """Handle keyboard input for slice navigation, view toggle, crosshair control, and zoom"""

        # Zoom controls (+ and - keys)
        if event.key == '+' or event.key == '=':
            self.zoom_level = min(self.zoom_level * 1.2, 10.0)  # Max 10x zoom
            print(f"Zoom: {self.zoom_level:.1f}x")
            self.update_display()
            return
        elif event.key == '-' or event.key == '_':
            self.zoom_level = max(self.zoom_level / 1.2, 0.5)  # Min 0.5x zoom (zoom out)
            print(f"Zoom: {self.zoom_level:.1f}x")
            self.update_display()
            return
        elif event.key == '0':
            self.zoom_level = 1.0  # Reset zoom
            print("Zoom reset to 1.0x")
            self.update_display()
            return

        # Crosshair mode toggle
        if event.key.lower() == 'c':
            self.crosshair_mode = not self.crosshair_mode
            status = "ENABLED" if self.crosshair_mode else "DISABLED"
            print(f"Crosshair mode {status}")
            if self.crosshair_mode:
                print("  Use arrow keys: ↑↓ (X), ←→ (Y), W/S (Z)")
                print("  Press ENTER to set point at crosshair position")
            self.update_display()
            return

        # View toggle
        if event.key.lower() == 'v':
            if self.current_view == 'axial':
                self.current_view = 'sagittal'
                print("Switched to SAGITTAL view")
            else:
                self.current_view = 'axial'
                print("Switched to AXIAL view")
            self.update_display()
            return

        # Crosshair movement (if crosshair mode is active)
        if self.crosshair_mode:
            if event.key == 'up':
                self.crosshair_x = min(self.crosshair_x + 1, self.segmentation.shape[0] - 1)
                self.update_display()
            elif event.key == 'down':
                self.crosshair_x = max(self.crosshair_x - 1, 0)
                self.update_display()
            elif event.key == 'left':
                self.crosshair_y = max(self.crosshair_y - 1, 0)
                self.update_display()
            elif event.key == 'right':
                self.crosshair_y = min(self.crosshair_y + 1, self.segmentation.shape[1] - 1)
                self.update_display()
            elif event.key.lower() == 'w':
                self.crosshair_z = min(self.crosshair_z + 1, self.segmentation.shape[2] - 1)
                self.update_display()
            elif event.key.lower() == 's':
                self.crosshair_z = max(self.crosshair_z - 1, 0)
                self.update_display()
            elif event.key == 'enter':
                # Set point at current crosshair position
                point = (self.crosshair_x, self.crosshair_y, self.crosshair_z)
                if self.start_point is None:
                    self.start_point = point
                    print(f"Start point set at crosshair: {point}")
                elif self.end_point is None:
                    self.end_point = point
                    print(f"End point set at crosshair: {point}")
                    self.compute_and_display_path()
                else:
                    # Reset
                    print("Resetting points...")
                    self.start_point = None
                    self.end_point = None
                    self.computed_path = None
                    self.computed_metrics = None
                self.update_display()

        # Slice navigation (only in axial view when NOT in crosshair mode)
        elif self.current_view == 'axial':
            if event.key == 'up':
                self.current_slice = min(self.current_slice + 1, self.segmentation.shape[2] - 1)
                self.update_display()
            elif event.key == 'down':
                self.current_slice = max(self.current_slice - 1, 0)
                self.update_display()
            
    def on_click(self, event):
        """Handle mouse clicks to select start/end points"""

        # Only process clicks on 2D view (ignore clicks in crosshair mode)
        if event.inaxes != self.ax_2d or self.crosshair_mode:
            return

        if event.xdata is None or event.ydata is None:
            return

        # Parse coordinates based on current view
        if self.current_view == 'axial':
            x, y = int(event.xdata), int(event.ydata)
            z = self.current_slice

            # Check bounds
            if not (0 <= x < self.segmentation.shape[0] and
                    0 <= y < self.segmentation.shape[1]):
                return

        elif self.current_view == 'sagittal':
            # In sagittal view: x-axis is Y, y-axis is Z
            mid_x = self.segmentation.shape[0] // 2
            x = mid_x
            y = int(event.xdata)
            z = int(event.ydata)

            # Check bounds
            if not (0 <= y < self.segmentation.shape[1] and
                    0 <= z < self.segmentation.shape[2]):
                return

        # Set start point (first click)
        if self.start_point is None:
            self.start_point = (x, y, z)
            print(f"Start point set: {self.start_point}")
            self.update_display()

        # Set end point and compute path (second click)
        elif self.end_point is None:
            self.end_point = (x, y, z)
            print(f"End point set: {self.end_point}")
            self.compute_and_display_path()

        # Reset (third click)
        else:
            print("Resetting points...")
            self.start_point = None
            self.end_point = None
            self.computed_path = None
            self.computed_metrics = None
            self.update_display()
            
    def compute_and_display_path(self):
        """Run FMM and visualize the result"""

        print("\n" + "="*60)
        print("COMPUTING PATH...")
        print("="*60)

        # Compute path
        path, travel_time = plan_path_fmm(
            self.speed_map,
            self.start_point,
            self.end_point
        )

        if path is None:
            print("ERROR: Path planning failed!")
            return

        # Calculate metrics
        metrics = calculate_safety_metrics(path, self.distance_map)

        # Store path and metrics (including travel time)
        self.computed_path = path
        self.computed_metrics = metrics
        self.computed_metrics['travel_time'] = travel_time

        print("\n" + "="*60)
        print("RESULTS:")
        print("="*60)
        print(f"Status: {'SAFE ✓' if metrics['safe'] else 'WARNING ⚠'}")
        print(f"Travel Time: {travel_time:.2f}")
        for key, value in metrics.items():
            print(f"{key:20s}: {value}")
        print("="*60 + "\n")

        # Redraw all views including the 3D view with the new path
        self.update_display()
        
    def _get_recommendation(self, metrics):
        """Generate clinical recommendation based on metrics"""
        if metrics['min_clearance'] < 3.0:
            return "⚠ Path too close to bone\nConsider alternative approach"
        elif metrics['min_clearance'] < 5.0:
            return "⚠ Acceptable but use caution\nConsider imaging guidance"
        else:
            return "✓ Path is safe\nGood surgical corridor"
    
    def show(self):
        """Display the interactive interface"""
        print("\n" + "="*70)
        print("INTERACTIVE PATH PLANNER - 3D CROSSHAIR INTERFACE")
        print("="*70)
        print("Layout:")
        print("  • LEFT:  2D View (toggleable: Axial ↔ Sagittal)")
        print("  • RIGHT: Real-time 3D View with crosshair system")
        print()
        print("Two Point Selection Modes:")
        print()
        print("1. CLICK MODE (default)")
        print("   - Click on 2D view to set points directly")
        print("   - First click: START point (red)")
        print("   - Second click: END point (green)")
        print("   - Third click: Reset")
        print()
        print("2. CROSSHAIR MODE (press C to enable)")
        print("   - Use keyboard to position 3D crosshair")
        print("   - Arrow keys ↑↓: Move X-axis (red)")
        print("   - Arrow keys ←→: Move Y-axis (green)")
        print("   - W/S keys:      Move Z-axis (blue)")
        print("   - ENTER:         Set point at crosshair position")
        print()
        print("General Controls:")
        print("  C          - Toggle Crosshair mode ON/OFF")
        print("  V          - Toggle view (Axial ↔ Sagittal)")
        print("  ↑↓         - Change slice (axial) OR move X-axis (crosshair)")
        print("  +/-        - Zoom in/out 3D view")
        print("  0          - Reset zoom to 1.0x")
        print("  Mouse drag - Rotate 3D view")
        print()
        if PYVISTA_AVAILABLE and ENABLE_3D_VISUALIZATION:
            print("After planning:")
            print("  Close window to open enhanced PyVista 3D visualization")
        print("="*70 + "\n")

        plt.tight_layout()
        plt.show()

        # After matplotlib closes, show enhanced 3D visualization if path was computed
        if (PYVISTA_AVAILABLE and ENABLE_3D_VISUALIZATION and
            self.computed_path is not None):
            print("\nMatplotlib closed. Opening enhanced 3D visualization...")
            visualize_path_3d(
                self.segmentation,
                self.computed_path,
                self.start_point,
                self.end_point,
                self.computed_metrics,
                spacing=self.spacing
            )


# ============================================================================
# BATCH PROCESSING (for multiple files)
# ============================================================================

def process_single_file(filepath):
    """
    Process a single segmentation file
    Can be called in a loop for batch processing
    """
    print(f"\nProcessing: {filepath}")

    # Load data
    segmentation, affine = load_segmentation(filepath)

    # Compute distance and speed maps
    distance_map = compute_distance_transform(segmentation)
    speed_map = create_speed_map(distance_map, safety_margin=SAFETY_MARGIN_MM)

    # Launch interactive planner with affine for accurate spacing
    planner = InteractivePathPlanner(segmentation, distance_map, speed_map, affine=affine)
    planner.show()


def process_multiple_files(filepaths):
    """
    Process multiple segmentation files sequentially
    """
    print(f"\nBatch processing {len(filepaths)} files...")
    
    for i, filepath in enumerate(filepaths, 1):
        print(f"\n{'='*60}")
        print(f"File {i}/{len(filepaths)}")
        print(f"{'='*60}")
        process_single_file(filepath)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point"""
    
    print("\n" + "="*60)
    print("CERVICAL SPINE SURGICAL PATH PLANNING SYSTEM")
    print("Fast Marching Method (FMM)")
    print("="*60 + "\n")
    
    # Process files based on configuration
    if len(SEGMENTATION_FILES) == 1:
        process_single_file(SEGMENTATION_FILES[0])
    else:
        process_multiple_files(SEGMENTATION_FILES)


if __name__ == "__main__":
    main()
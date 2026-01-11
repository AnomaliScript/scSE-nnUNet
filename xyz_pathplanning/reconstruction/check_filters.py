import vtk
import importlib

def check_vtk_class(name):
    try:
        cls = getattr(vtk, name)
        print(f"Found {name}")
        return True
    except AttributeError:
        print(f"{name} not found in vtk module")
        return False

print("Checking for Bilateral filters in VTK...")
check_vtk_class("vtkBilateralImageFilter") # We know this exists
check_vtk_class("vtkBilateralNormalFilter") # Maybe?
check_vtk_class("vtkMeshDenoising")
check_vtk_class("vtkDenoiseMesh")

print("\nChecking for other libraries...")
try:
    import trimesh
    print("trimesh is installed")
except ImportError:
    print("trimesh is NOT installed")

try:
    import open3d
    print("open3d is installed")
except ImportError:
    print("open3d is NOT installed")

try:
    import pymeshlab
    print("pymeshlab is installed")
except ImportError:
    print("pymeshlab is NOT installed")

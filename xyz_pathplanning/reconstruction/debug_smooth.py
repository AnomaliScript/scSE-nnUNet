import pyvista as pv
import vtk
print(f"PyVista Version: {pv.__version__}")
print(f"VTK Version: {vtk.VTK_VERSION}")

mesh = pv.Sphere()
methods = [m for m in dir(mesh) if 'smooth' in m.lower()]
print("\nSmoothing methods available in pv.PolyData:")
for m in methods:
    print(f" - {m}")

print("\nChecking if we can use vtkWindowedSincPolyDataFilter directly:")
try:
    f = vtk.vtkWindowedSincPolyDataFilter()
    print("vtkWindowedSincPolyDataFilter is available in vtk module.")
except Exception as e:
    print(f"Error: {e}")

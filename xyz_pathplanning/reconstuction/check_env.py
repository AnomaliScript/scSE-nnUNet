
import pyvista
import vtk
print(f"PyVista version: {pyvista.__version__}")
print(f"VTK version: {vtk.vtkVersion.GetVTKVersion()}")
filters = [x for x in dir(vtk) if 'Subdivision' in x]
print(f"Available subdivision filters: {filters}")


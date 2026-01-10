
import vtk
import importlib
import pkgutil

print("Searching for vtkCatmullClarkSubdivisionFilter...")

# Check if we can import it from vtkmodules.vtkFiltersModeling
try:
    from vtkmodules.vtkFiltersModeling import vtkCatmullClarkSubdivisionFilter
    print("Found in vtkmodules.vtkFiltersModeling")
except ImportError:
    print("Not found in vtkmodules.vtkFiltersModeling")

# Check if we can import it from vtk.vtkFiltersModeling
try:
    import vtk.vtkFiltersModeling
    if hasattr(vtk.vtkFiltersModeling, 'vtkCatmullClarkSubdivisionFilter'):
        print("Found in vtk.vtkFiltersModeling")
    else:
        print("Not found in vtk.vtkFiltersModeling")
except ImportError:
    print("Could not import vtk.vtkFiltersModeling")

# List all modules in vtkmodules
import vtkmodules
print(f"\nvtkmodules path: {vtkmodules.__path__}")

# Try to find where Loop is to see if Catmull is nearby
try:
    from vtkmodules.vtkFiltersModeling import vtkLoopSubdivisionFilter
    print("vtkLoopSubdivisionFilter is in vtkFiltersModeling")
except ImportError:
    print("vtkLoopSubdivisionFilter is NOT in vtkFiltersModeling")


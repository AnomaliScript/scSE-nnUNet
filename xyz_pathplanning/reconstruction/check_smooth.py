
import vtk
filters = [x for x in dir(vtk) if 'Smooth' in x or 'Sinc' in x]
print(f"Available smoothing filters: {filters}")

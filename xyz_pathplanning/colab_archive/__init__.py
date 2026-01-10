"""
Colab Marching Cubes Implementation
====================================

This package contains a 4-part advanced marching cubes implementation
originally developed in a Jupyter notebook for surgical navigation.

Parts:
- marching_cubes_core_data: Foundation layer with lookup tables and data structures
- h_arch_adapt_grid_system: Hierarchical adaptive octree system
- mesh_gen_surface_extract: Mesh generation and surface extraction
- marching_cubes_integration: Complete integration and export pipeline

For simple usage, import from marching_cubes_core_data:
    from colab.marching_cubes_core_data import extract_surface_mesh, create_pyvista_mesh
"""

__version__ = "1.0"
__all__ = [
    "marching_cubes_core_data",
    "h_arch_adapt_grid_system",
    "mesh_gen_surface_extract",
    "marching_cubes_integration"
]

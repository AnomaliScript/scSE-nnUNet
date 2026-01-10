import numpy as np
from scipy.sparse import *
from scipy.sparse.linalg import eigs, spsolve
import openmesh

def calculate_face_centroid(mesh, face):
  points = mesh.points()
  mean_vertex = np.array([0, 0, 0], dtype=float)
  vertex_count = 0.0

  # Iterator of face vertices
  for v in mesh.fv(face):
    mean_vertex += points[v.idx()]
    vertex_count += 1.0

  mean_vertex /= vertex_count

  return mean_vertex

def calculate_edge_point(mesh, edge, new_mesh):
  points = mesh.points()
  new_mesh_points = new_mesh.points()
  
  he = mesh.halfedge_handle(edge, 0)
  mean_vertex = np.array([0, 0, 0], dtype=float)
  vertex_count = 0.0

  # Detect if the edge is not a boundary
  if not mesh.is_boundary(edge):
    # Get adjacent faces
    f1 = mesh.face_handle(he)
    f2 = mesh.face_handle(mesh.opposite_halfedge_handle(he))

    # Sum the face centroids
    mean_vertex += new_mesh_points[f1.idx()] + new_mesh_points[f2.idx()]
    vertex_count += 2.0

  # Get edge vertices
  v1 = mesh.from_vertex_handle(he)
  v2 = mesh.to_vertex_handle(he)
  
  # Sum the edge vertices
  mean_vertex += points[v1.idx()] + points[v2.idx()]
  vertex_count += 2.0

  # Average
  mean_vertex /= vertex_count

  return mean_vertex

def calculate_vertex_point(mesh, vertex, new_mesh):
  points = mesh.points()
  new_mesh_points = new_mesh.points()
  
  v_new = np.array([0, 0, 0], dtype=float)
  v_old = points[vertex.idx()]
  edges_count = 0.0
  faces_count = 0.0

  if not mesh.is_boundary(vertex):
    edge_vertex = np.array([0, 0, 0], dtype=float)
    # Edge circulator
    for eh in mesh.ve(vertex):
      index = eh.idx()
      edge_point = new_mesh_points[mesh.n_faces() + index]
      edge_vertex += (edge_point - v_old)
      edges_count += 1.0

    face_vertex = np.array([0, 0, 0], dtype=float)
    # Face iterator
    for f in mesh.vf(vertex):
      index = f.idx()
      face_centroid = new_mesh_points[index]
      face_vertex += (face_centroid - v_old)
      faces_count += 1.0

    edge_vertex /= np.power(edges_count, 2)
    face_vertex /= np.power(edges_count, 2)

    v_new = v_old + edge_vertex + face_vertex

  else:
    edge_vertex = np.array([0, 0, 0], dtype=float)
    # Boundary edge circulator
    for eh in mesh.ve(vertex):
      if mesh.is_boundary(eh):
        index = eh.idx()
        edge_point = new_mesh_points[mesh.n_faces() + index]
        edge_vertex += (edge_point - v_old)
        edges_count += 1.0

    edge_vertex /= np.power(edges_count, 2)
    v_new = v_old + edge_vertex

  return v_new

def catmull_clark(mesh: openmesh.PolyMesh):
  # In each iteration:
    # Construct face vertices: centroids
    # Construct edge vertices: average of edge vertices
    # Update original vertex positions

  # Create a new mesh
  # Indices:
    # [0, ..., n_faces() - 1]: face centroids
    # [n_faces(), ..., n_faces() + n_edges() - 1]: edge points
    # [n_faces() + n_edges(), ..., n_faces() + n_edges() + n_vertices() - 1]: original vertices
  new_mesh = openmesh.PolyMesh()
  n_faces = mesh.n_faces()
  n_edges = mesh.n_edges()

  # Face iterator
  for f in mesh.faces():
    # Add vertex to new mesh
    _ = new_mesh.add_vertex(calculate_face_centroid(mesh, f))

  # Edge iterator
  for e in mesh.edges():
    # Add vertex to new mesh
    _ = new_mesh.add_vertex(calculate_edge_point(mesh, e, new_mesh))

  # Vertex iterator
  for v in mesh.vertices():
    # Add vertex to new mesh
    _ = new_mesh.add_vertex(calculate_vertex_point(mesh, v, new_mesh))

  for f in mesh.faces():
    face_centroid = new_mesh.vertex_handle(f.idx())
    
    vertices = [vh for vh in mesh.fv(f)]
    edges = [eh for eh in mesh.fe(f)]

    # [WARNING] this is supposed to work mostly for quads, some weird stuff happens with triangles and borders sometimes
    # EXAMPLE model airplane_0627 gets some weird holes which messes up the mesh for the following iterations
    # but we'll let it slide for now using the range with len(edges) instead of 4
    for i in range(len(edges)):
      corner = new_mesh.vertex_handle(
        n_faces + n_edges + vertices[i].idx()
      )
      edge_point_prev = new_mesh.vertex_handle(
        n_faces + edges[i].idx()
      )
      edge_point_next = new_mesh.vertex_handle(
        n_faces + edges[(i + 1) % len(edges)].idx()
      )

      _ = new_mesh.add_face([corner, edge_point_prev, face_centroid, edge_point_next])

  return new_mesh

def catmull_clark_iter(mesh: openmesh.PolyMesh, iterations: int):
  new_mesh = mesh
  for _ in range(iterations):
    new_mesh = catmull_clark(new_mesh)

  return new_mesh
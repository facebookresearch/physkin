# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import warp as wp

# Configure warp
wp.config.quiet = True
wp.init()


def face_areas(v: torch.Tensor, f: torch.Tensor):
    e0, e1 = compute_triangle_edge_vectors(v, f)
    N = torch.linalg.cross(e1, e0)
    areas = 0.5 * torch.linalg.vector_norm(N, dim=-1)
    return areas


def edge_lengths(pos, faces, face_edges, n_edges):
    """Compute edge lengths from a custom face topology tied to an edge topology using face_edges"""
    face_pos = pos[faces]
    e0 = torch.linalg.vector_norm(face_pos[:, 0] - face_pos[:, 1], dim=-1)
    e1 = torch.linalg.vector_norm(face_pos[:, 1] - face_pos[:, 2], dim=-1)
    e2 = torch.linalg.vector_norm(face_pos[:, 2] - face_pos[:, 0], dim=-1)
    el = torch.zeros(n_edges, dtype=pos.dtype, device=pos.device)
    el = el.index_add(0, face_edges[:, 0], e0)
    el = el.index_add(0, face_edges[:, 1], e1)
    el = el.index_add(0, face_edges[:, 2], e2)

    # Ensures boundary edges are weighted as 1 and interior are divided by 2
    count = torch.zeros(n_edges, dtype=pos.dtype, device=pos.device)
    ones = torch.ones(len(faces), dtype=pos.dtype, device=pos.device)
    count = count.index_add(0, face_edges[:, 0], ones)
    count = count.index_add(0, face_edges[:, 1], ones)
    count = count.index_add(0, face_edges[:, 2], ones)

    return el / count


def get_boundary_vertices(faces):
    """
    Find boundary vertices in a triangle mesh.

    Parameters:
    - faces: numpy array of shape (n_faces, 3) containing triangle indices

    Returns:
    - boundary_vertices: numpy array containing indices of boundary vertices
    """
    # Create all edges (vertex pairs)
    edges = np.vstack(
        [
            np.column_stack([faces[:, 0], faces[:, 1]]),
            np.column_stack([faces[:, 1], faces[:, 2]]),
            np.column_stack([faces[:, 2], faces[:, 0]]),
        ]
    )

    # Sort vertex indices for each edge
    sorted_edges = np.sort(edges, axis=1)

    # Find unique edges and their counts
    unique_edges, edge_counts = np.unique(sorted_edges, axis=0, return_counts=True)

    # Boundary edges appear only once
    boundary_edges = unique_edges[edge_counts == 1]

    # Get unique vertices from boundary edges
    boundary_vertices = np.unique(boundary_edges)

    return boundary_vertices


def compute_local_bases(vertices, faces):
    """
    Compute local orthonormal bases for each vertex in a mesh.

    Parameters:
    - vertices: tensor of shape (n_vertices, 3) containing vertex positions
    - faces: tensor of shape (n_faces, 3) containing face indices

    Returns:
    - bases: tensor of shape (n_vertices, 3, 3) where each 3x3 matrix is an orthonormal basis
             with the first column being the vertex normal
    """
    # Convert inputs to PyTorch tensors if they aren't already
    if not isinstance(vertices, torch.Tensor):
        vertices = torch.tensor(vertices, dtype=torch.float32)
    if not isinstance(faces, torch.Tensor):
        faces = torch.tensor(faces, dtype=torch.int64)

    # Compute vertex normals using the vertex_normals function
    normals = vertex_normals(vertices, faces)

    # Create orthonormal bases for each vertex
    n_vertices = vertices.shape[0]
    bases = torch.zeros(
        (n_vertices, 3, 3), dtype=vertices.dtype, device=vertices.device
    )

    # Build vertex neighbor lists
    vertex_neighbors = [[] for _ in range(n_vertices)]
    for f in faces:
        for i in range(3):
            v1 = f[i].item()
            v2 = f[(i + 1) % 3].item()
            if v2 not in vertex_neighbors[v1]:
                vertex_neighbors[v1].append(v2)
            if v1 not in vertex_neighbors[v2]:
                vertex_neighbors[v2].append(v1)

    for i in range(n_vertices):
        # First basis vector is the normal
        normal = normals[i]
        normal = normal / torch.linalg.norm(normal)

        # Use edge to a neighbor as tangent direction if available
        tangent = None
        if vertex_neighbors[i]:
            # Use the first neighbor to define tangent direction
            neighbor_idx = vertex_neighbors[i][0]
            edge = vertices[neighbor_idx] - vertices[i]
            # Project edge to be perpendicular to normal
            tangent = edge - torch.dot(edge, normal) * normal
            if torch.linalg.norm(tangent) > 1e-6:
                tangent = tangent / torch.linalg.norm(tangent)
            else:
                tangent = None

        # If no valid tangent from topology, find a vector not parallel to normal
        if tangent is None:
            if abs(normal[0]) < 0.9:
                tangent = torch.tensor(
                    [1.0, 0.0, 0.0], dtype=vertices.dtype, device=vertices.device
                )
            else:
                tangent = torch.tensor(
                    [0.0, 1.0, 0.0], dtype=vertices.dtype, device=vertices.device
                )
            # Make it perpendicular to the normal
            tangent = tangent - torch.dot(tangent, normal) * normal
            tangent = tangent / torch.linalg.norm(tangent)

        # Third basis vector from cross product
        bitangent = torch.linalg.cross(normal, tangent)

        # Store the basis
        bases[i, :, 0] = normal
        bases[i, :, 1] = tangent
        bases[i, :, 2] = bitangent

    return bases


def edge_areas_from_face_areas(f: torch.Tensor, fa: torch.Tensor):
    # Compute edge areas from explicit face areas.
    # This way you can use face areas from a different rest topology
    e, inverse = compute_edge_topology(f.numpy(), return_inverse=True)
    inverse = torch.tensor(inverse)
    ea = torch.zeros((len(e),), dtype=fa.dtype, device=fa.device)
    ea = ea.index_add(0, inverse[:, 0], fa, alpha=1 / 3)
    ea = ea.index_add(0, inverse[:, 1], fa, alpha=1 / 3)
    ea = ea.index_add(0, inverse[:, 2], fa, alpha=1 / 3)
    return ea


def face_normals(v, f, normalize=True):
    e0, e1 = compute_triangle_edge_vectors(v, f)
    N = torch.linalg.cross(e1, e0)
    if normalize:
        N = torch.nn.functional.normalize(N, dim=-1)
    return N


def vertex_normals(v: torch.Tensor, f: torch.Tensor, normalize=True):
    VN = torch.zeros_like(v)
    FN = face_normals(v, f, normalize=True)
    VN = VN.index_add(-2, f[:, 0], FN)
    VN = VN.index_add(-2, f[:, 1], FN)
    VN = VN.index_add(-2, f[:, 2], FN)
    if normalize:
        VN = torch.nn.functional.normalize(VN, dim=-1)
    return VN


def face_normals_and_areas(v: torch.Tensor, f: torch.Tensor, normalize=True):
    e0, e1 = compute_triangle_edge_vectors(v, f)
    N = torch.linalg.cross(e1, e0)
    areas = 0.5 * torch.linalg.vector_norm(N, dim=-1)
    if normalize:
        N = torch.nn.functional.normalize(N, dim=-1)
    return N, areas


def compute_triangle_edge_vectors(v: torch.Tensor, f: torch.Tensor):
    """
    Compute two tensors E0 and E1 corresponding to the two edge vectors of each triangle
    Arguments:
        v has shape [... x] V x 3
        f has shape F x 3
    Returns:
        E0, E1 have shape [... x] F x 3
    """
    v0 = v[..., f[:, 0], :]
    v1 = v[..., f[:, 1], :]
    v2 = v[..., f[:, 2], :]
    return v0 - v2, v1 - v2


def compute_rest_shape_inv(pos, faces, DXinv=None):
    """
    Compute the rest shape inverse matrix for each face.
    This function supports batching in multiple dimensions.
    pos has shape [... x] V x 3
    """
    if DXinv is None:
        # Copies batch dimensions from pos
        DXinv = torch.zeros(
            pos.shape[:-2] + (faces.shape[0], 2, 2), dtype=torch.float32
        )

    E0, E1 = compute_triangle_edge_vectors(pos, faces)

    # Compute change of basis for the given triangles in canonical representation.

    E0_u = torch.linalg.vector_norm(E0, dim=-1)
    DXinv[..., 0, 0] = E0_u
    # D[:,1,0] = E0_v = 0

    E0_hat = torch.div(E0, E0_u.unsqueeze(-1))
    E1_u = torch.linalg.vecdot(E0_hat, E1)
    DXinv[..., 0, 1] = E1_u
    DXinv[..., 1, 1] = torch.linalg.vector_norm(
        E1 - torch.mul(E0_hat, E1_u.unsqueeze(-1)), dim=-1
    )
    torch.linalg.inv(DXinv, out=DXinv)
    return DXinv


def compute_shape(pos, faces):
    """
    Computes a [N x] F x 3 x 2 tensor of shape matrices for each face.
    Assumes `pos` has shape V x 3
    Here N is the optional batch size
    """
    return torch.stack(compute_triangle_edge_vectors(pos, faces), dim=-1)


def compute_edge_topology(faces, return_inverse=False):
    """
    Construct a set of edge indices for a triangle mesh.
    Parameters:
    faces (numpy array): A list of triangles, where each triangle is represented by three vertex indices.
    return_inverse (bool): if True, return indices into the unique array
    Returns:
    edges (numpy array): A list of unique edges in the triangle mesh.
    indices (numpy array): Indices into the unique array for each face, edges are ordered as 0-1, 1-2, and 2-0. Only if return_inverse is true
    """
    # Extract vertex indices of all triangles
    v1 = faces[..., 0]
    v2 = faces[..., 1]
    v3 = faces[..., 2]
    # Create arrays of edges
    e1 = np.stack((v1, v2), axis=-1)
    e2 = np.stack((v2, v3), axis=-1)
    e3 = np.stack((v3, v1), axis=-1)
    # Stack all edges into one array
    edges = np.concatenate((e1, e2, e3))
    # Sort vertex indices of each edge to ensure consistency
    edges = np.sort(edges, axis=-1)
    # Remove duplicate edges
    if return_inverse:
        edges, inverse = np.unique(edges, axis=-2, return_inverse=return_inverse)
        return edges, inverse.reshape(3, len(faces)).T  # num_faces x 3
    else:
        return np.unique(edges, axis=-2)


def compute_interior_edge_topology(faces):
    """
    Returns a list of edges that connect two faces
    (i.e., all the edges except borders)
    """
    nFaces = len(faces)
    edges = np.zeros((nFaces, 3, 2), dtype=np.int32)
    edges[:, 0, 0] = faces[:, 0]
    edges[:, 0, 1] = faces[:, 1]
    edges[:, 1, 0] = faces[:, 1]
    edges[:, 1, 1] = faces[:, 2]
    edges[:, 2, 0] = faces[:, 2]
    edges[:, 2, 1] = faces[:, 0]
    edges_sorted_indices = np.argsort(edges, axis=-1)
    edges = np.take_along_axis(edges, edges_sorted_indices, axis=-1).reshape(
        nFaces * 3, 2
    )
    indices = np.lexsort((edges[:, 1], edges[:, 0]))
    edges_sorted = edges[indices]
    u, ui, uc = np.unique(edges_sorted, return_index=True, return_counts=True, axis=0)
    mask = uc > 1  # edges with more than one neighbour
    fi = np.arange(nFaces * 3) // 3

    num_shared_edges = int(np.sum(mask))
    ef = np.zeros((num_shared_edges, 2), dtype=np.int32)
    ef[:, 0] = fi[indices][ui[mask]]
    ef[:, 1] = fi[indices][ui[mask] + 1]

    # Also build correspondences to the vertex within face
    f_edges = np.zeros((nFaces, 3, 2), dtype=np.int32)
    f_edges[:, 0, 1] = 1
    f_edges[:, 1, 0] = 1
    f_edges[:, 1, 1] = 2
    f_edges[:, 2, 0] = 2
    f_edges = np.take_along_axis(f_edges, edges_sorted_indices, axis=-1).reshape(
        nFaces * 3, 2
    )

    efv = np.zeros((num_shared_edges, 2, 2), dtype=np.int32)  # edge x face x vertex
    efv[:, 0, :] = f_edges[indices][ui[mask]]
    efv[:, 1, :] = f_edges[indices][ui[mask] + 1]

    ev = u[mask]
    return ev, ef, efv


# Defines dihedral topology. This excludes any boundary edges
class DihedralTopo(nn.Module):
    def __init__(self, f):
        super().__init__()
        if type(f) is torch.Tensor:
            f = f.numpy()
        ev, ef, efv = compute_interior_edge_topology(f)
        # #e by 2, list of edges described as pair of vertices.
        self.register_buffer("verts", torch.from_numpy(ev))
        # #e by w, list storing edge-triangle relation, uses -1 to indicate boundaries.
        self.register_buffer("faces", torch.from_numpy(ef))
        self.register_buffer("face_verts", torch.from_numpy(efv))

        # TODO: Refactor to not use for loops here
        comp_verts = torch.zeros_like(self.verts)
        for ei, (fs, vs) in enumerate(zip(self.faces, self.verts)):
            for side in range(2):
                for i in f[fs[side]]:
                    if i not in vs:
                        comp_verts[ei][side] = i
                        break
        self.register_buffer("comp_verts", comp_verts)


def compute_dihedral_areas(area: torch.Tensor, dihedralTopo: DihedralTopo):
    a0 = area[..., dihedralTopo.faces[:, 0]]
    a1 = area[..., dihedralTopo.faces[:, 1]]
    return a0, a1


def compute_dihedral_rest_shape(
    rest_pos: torch.Tensor, rest_areas: torch.Tensor, dihedralTopo: DihedralTopo
):
    rda0, rda1 = compute_dihedral_areas(rest_areas, dihedralTopo)
    rest_pos = rest_pos.reshape(rest_pos.shape[:-2] + (rest_pos.shape[-2] // 3, 3, 3))
    assert rest_pos.shape[-3] == rest_areas.shape[-1]
    # edge on the side of face 0
    x00 = rest_pos[..., dihedralTopo.faces[:, 0], dihedralTopo.face_verts[:, 0, 0], :]
    x01 = rest_pos[..., dihedralTopo.faces[:, 0], dihedralTopo.face_verts[:, 0, 1], :]
    e0_norm = torch.linalg.vector_norm(x01 - x00, dim=-1)
    # edge on the side of face 1
    x10 = rest_pos[..., dihedralTopo.faces[:, 1], dihedralTopo.face_verts[:, 1, 0], :]
    x11 = rest_pos[..., dihedralTopo.faces[:, 1], dihedralTopo.face_verts[:, 1, 1], :]
    e1_norm = torch.linalg.vector_norm(x11 - x10, dim=-1)
    e_norm = 0.5 * (e0_norm + e1_norm)
    return torch.ones_like(e_norm)  # 3.0 * torch.square(e_norm) / (rda0 + rda1)


def compute_centroids(vertices, faces):
    C = vertices[..., faces[:, 0], :] / 3.0
    C += vertices[..., faces[:, 1], :] / 3.0
    C += vertices[..., faces[:, 2], :] / 3.0
    return C


def uniform_laplacian(faces, n_verts):
    adjacency = np.zeros((n_verts, n_verts), dtype=bool)

    np_faces = faces.cpu().numpy()

    adjacency[np_faces[:, 0], np_faces[:, 1]] = True
    adjacency[np_faces[:, 1], np_faces[:, 0]] = True
    adjacency[np_faces[:, 1], np_faces[:, 2]] = True
    adjacency[np_faces[:, 2], np_faces[:, 1]] = True
    adjacency[np_faces[:, 2], np_faces[:, 0]] = True
    adjacency[np_faces[:, 0], np_faces[:, 2]] = True

    # Compute uniform Laplacian
    degree = np.sum(adjacency, axis=1)
    lap = sp.sparse.diags(degree) - sp.sparse.csr_matrix(adjacency)
    L = (
        torch.sparse_coo_tensor(lap.nonzero(), lap.data, lap.shape, device=faces.device)
        .to_sparse_csr()
        .float()
    )
    return L

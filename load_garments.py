# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
from glob import glob

import lbs
import numpy as np
import potpourri3d as pp3d
import pymeshlab as pml
import torch
import torch_geometric
import trimesh
import weight_transfer
from garment import Garment
from scipy import spatial
from scipy.spatial import KDTree
from training import dataio


def compute_geodesic_for_vertex(vidx, solver):
    """Compute geodesic distances and nearest neighbors for a single vertex"""
    gdist = solver.compute_distance(vidx)
    geo_nn = gdist.argsort()
    return gdist, geo_nn


def bones_placement_fps(verts, faces, bone_indices_sampled=None, bone_lods=None):
    print("Bone Sampling: Farthest Geodesic")
    if bone_indices_sampled is None:
        bone_indices_sampled = []
    if bone_lods is None:
        bone_lods = []
    init_length = len(bone_indices_sampled)
    if init_length == 0:
        bone_indices_sampled.append(int(verts[:, 1].argmin()))
    solver = pp3d.MeshHeatMethodDistanceSolver(verts, faces)
    # Initialize distances to infinity
    n_points = len(verts)
    min_distances = np.ones(n_points) * np.inf
    # Update distances from initial points
    for idx in bone_indices_sampled:
        distances = solver.compute_distance(idx)
        min_distances = np.minimum(min_distances, distances)
    placements = {}
    for lod_i in bone_lods:
        # Select remaining points
        terminating_length = init_length + lod_i
        while len(bone_indices_sampled) < terminating_length:
            # Set distances of already selected points to -1
            min_distances[bone_indices_sampled] = -1
            # Find the point with maximum distance
            max_idx = np.argmax(min_distances)
            bone_indices_sampled.append(max_idx)
            # Update distances with the new point
            distances = solver.compute_distance(max_idx)
            min_distances = np.minimum(min_distances, distances)
        placements[f"{lod_i}"] = copy.deepcopy(np.array(bone_indices_sampled))
    print("Sampling Done!!!")
    return placements


def filter_top_k(W, k):
    idx = np.argpartition(W, -k, axis=0)[-k:]
    # Create a mask of zeros
    mask = np.zeros_like(W, dtype=bool)
    # Set True for the k largest elements per column
    np.put_along_axis(mask, idx, True, axis=0)
    # Zero out all but the k largest per column
    return W * mask, idx, mask


def get_boundary_verts(meshset, filter_top_by_y=False):
    ms = meshset
    _all_verts = ms.current_mesh().vertex_matrix()  # noqa: F841
    ms.compute_selection_from_mesh_border()
    ms.apply_selection_inverse(invfaces=True, invverts=True)
    ms.meshing_remove_selected_vertices()
    boundary_verts = ms.current_mesh().vertex_matrix()
    if filter_top_by_y:
        ymax = np.max(boundary_verts[:, 1])
        cond = np.abs(boundary_verts[:, 1] - ymax) < 5.0
        boundary_verts = boundary_verts[cond, :]
    return boundary_verts


def is_active(name):
    # return name.startswith("skirt")
    return name.startswith("garment") or name.startswith("pinned")


def is_pinned(name):
    return name.startswith("pinned")


def is_unpinned(name):
    return name.startswith("garment")


def _load_mesh_with_normals(
    mesh_path, invert_face_normals, invert_vertex_normals, poly_to_tri=False
):
    """Load a mesh file, normalize normals, and optionally invert them."""
    ms = pml.MeshSet()
    ms.load_new_mesh(mesh_path)
    if poly_to_tri:
        ms.meshing_poly_to_tri()
    ms.apply_normal_normalization_per_face()
    ms.apply_normal_normalization_per_vertex()
    if invert_face_normals:
        ms.meshing_invert_face_orientation()
    if invert_vertex_normals:
        ms.compute_normal_by_function_per_vertex(
            x="-nx", y="-ny", z="-nz", onselected=False
        )
    return ms


def _init_body_avatar(cfg, scene):
    """Initialize body mesh from a custom avatar OBJ."""
    scene["body_name"] = cfg.body.name
    _reference_body_name = cfg.body.custom_obj_path.split("/")[-1].split(".")[0]  # noqa: F841
    ms_io_b = _load_mesh_with_normals(
        cfg.body.custom_obj_path,
        cfg.body.invert_face_normals,
        cfg.body.invert_vertex_normals,
    )
    custom_body_mesh = ms_io_b.current_mesh()
    custom_body_verts = custom_body_mesh.vertex_matrix()
    custom_body_faces = custom_body_mesh.face_matrix()
    nn_blended_body_wts, _ = weight_transfer.copy_weights_via_knn(
        sourceV=scene["bodyPos"],
        sourceW=scene["bodyBoneWeights"].T,
        targetV=custom_body_verts,
        nn=16,
    )
    custom_body_bone_weights = nn_blended_body_wts.T
    scene["bodyPos"] = custom_body_verts
    scene["bodyFaces"] = custom_body_faces
    scene["bodyBoneWeights"] = custom_body_bone_weights
    body_verts_tree = KDTree(scene["bodyPos"])
    rigid_labels = ["_lhand", "_rhand", "_lfoot", "_rfoot"]
    for idx in range(len(rigid_labels)):
        label = rigid_labels[idx]
        ref_idx = cfg.body.rigid_verts_parent_idx[idx]
        sub_verts = trimesh.load(
            cfg.body.custom_obj_path[:-4] + f"{label}.obj"
        ).vertices
        dist, nn_indices = body_verts_tree.query(sub_verts, k=1)
        ref_wt = custom_body_bone_weights[:, ref_idx].reshape(-1, 1)
        ref_wts = np.repeat(ref_wt, nn_indices.shape[0], axis=-1)
        scene["bodyBoneWeights"][:, nn_indices.flatten()] = ref_wts


def _init_body_3dbicar(cfg, scene):
    """Initialize body mesh from a 3DBiCar dataset."""
    scene["body_name"] = cfg.body.name
    body_info_rootdir = cfg.body.verts_data_path
    _reference_body_name = cfg.body.custom_obj_path.split("/")[-1].split(".")[0]  # noqa: F841
    ms_io_b = _load_mesh_with_normals(
        cfg.body.custom_obj_path,
        cfg.body.invert_face_normals,
        cfg.body.invert_vertex_normals,
        poly_to_tri=True,
    )
    custom_body_mesh = ms_io_b.current_mesh()
    custom_body_verts = custom_body_mesh.vertex_matrix()
    custom_body_faces = custom_body_mesh.face_matrix()
    custom_body_bone_weights = np.load(body_info_rootdir + "/shape/weight_matrix.npy")
    custom_body_bone_weights = custom_body_bone_weights.T
    scene["bodyPos"] = custom_body_verts
    scene["bodyFaces"] = custom_body_faces
    scene["bodyBoneWeights"] = custom_body_bone_weights
    body_verts_tree = KDTree(scene["bodyPos"])
    ignore_labels = ["_tail"]
    ignore_verts_indices = []
    for idx in range(len(ignore_labels)):
        label = ignore_labels[idx]
        sub_verts = trimesh.load(
            cfg.body.custom_obj_path[:-4] + f"{label}.obj"
        ).vertices
        dist, nn_indices = body_verts_tree.query(sub_verts, k=1)
        ignore_verts_indices.append(nn_indices.flatten())
    scene["ignore_verts_indices"] = np.concatenate(ignore_verts_indices, axis=-1)
    # todo: the ordering of joint names might be incorrect
    joints_names = np.load(
        body_info_rootdir + "/shape/pose_order.npy", allow_pickle=True
    )
    scene["boneNames"] = joints_names
    scene["boneParentIndices"] = np.array([-1] * len(scene["boneNames"]))
    scene["bonePos"] = np.array([0.0, 0.0, 0.0] * len(scene["boneNames"])).reshape(
        -1, 3
    )
    scene["boneTransform"] = np.array([np.eye(3)] * len(scene["boneNames"])).reshape(
        -1, 3, 3
    )
    scene["activeBones"] = np.array([False] * len(scene["boneNames"]))
    affine_state = np.array([np.eye(4)] * len(scene["boneNames"])).reshape(-1, 4, 4)
    affine_state[:, :3, :3] = scene["boneTransform"]
    affine_state[:, :3, 3] = scene["bonePos"]
    scene["affine_state"] = affine_state


def _load_garment_mesh(cfg, custom_garment_dirpath, scene):
    """Load the garment mesh, compute normals, and populate scene."""
    reference_garment_mesh_path = glob(custom_garment_dirpath + "/*_sim.obj")[0]
    reference_garment_name = reference_garment_mesh_path[
        len(custom_garment_dirpath) + 1 : -len("_sim.obj")
    ]
    ms_io_c = _load_mesh_with_normals(
        reference_garment_mesh_path,
        cfg.garment.invert_face_normals,
        cfg.garment.invert_vertex_normals,
    )
    reference_garment_mesh = ms_io_c.current_mesh()
    reference_verts = reference_garment_mesh.vertex_matrix()
    reference_faces = reference_garment_mesh.face_matrix()
    reference_vertex_normals = reference_garment_mesh.vertex_normal_matrix()
    reference_vertex_normals = reference_vertex_normals / np.linalg.norm(
        reference_vertex_normals, axis=-1, keepdims=True
    )
    print(f"# Vertices:{reference_verts.shape[0]} | # Faces:{reference_faces.shape[0]}")
    scene["initPos"] = reference_verts
    scene["faces"] = reference_faces
    tex_coords = np.zeros(
        (3 * len(reference_faces), 3)
    )  # ( 3 x number of faces, 2) ---> appending zeros in the third column to make it 3D --> (3 x number of faces, 3) (used for rest positions calculation)
    tex_coords[:, :2] = reference_garment_mesh.wedge_tex_coord_matrix()
    scene["texCoords"] = tex_coords
    scene["faceGarmentName"] = np.array(
        [reference_garment_name] * scene["faces"].shape[0]
    )
    return (
        reference_garment_mesh,
        reference_garment_name,
        reference_verts,
        reference_faces,
        reference_vertex_normals,
    )


def _load_panels_and_seams(cfg, custom_garment_dirpath, scene, reference_verts):
    """Load panel mesh, compute seam info, and optionally set rest positions."""
    panels_mesh_path = glob(custom_garment_dirpath + "/*_boxmesh.obj")[0]
    ms_io_panels = pml.MeshSet()
    ms_io_panels.load_new_mesh(panels_mesh_path)
    panels_mesh = ms_io_panels.current_mesh()
    panels_verts = panels_mesh.vertex_matrix()
    panels_faces = panels_mesh.face_matrix()
    # estimate stitching vertices
    stitch_seam_indices_path = glob(custom_garment_dirpath + "/*_stitch.npy")[0]
    seam_verts_indices = np.load(stitch_seam_indices_path)
    # Find faces that contain any of the seam vertices
    face_contains_seam_vertex = np.any(
        np.isin(panels_faces, seam_verts_indices), axis=1
    )
    if cfg.train.panels_as_rest:
        print("Using panels as rest positions...")
        rest_pos = panels_verts[panels_faces]
        rest_pos[face_contains_seam_vertex] = reference_verts[
            panels_faces[face_contains_seam_vertex]
        ]
        rest_pos = rest_pos.reshape(-1, 3)
        scene["restPos"] = rest_pos
        _ = trimesh.PointCloud(rest_pos).export(f"{custom_garment_dirpath}/rest.obj")
    else:
        rest_pos = None
    return panels_verts, panels_faces, seam_verts_indices, rest_pos


def _prepare_remesh_data(cfg, scene, reference_garment_mesh_path):
    """Load remesh data if configured for inference remeshing."""
    if not (
        cfg.train.max_num_garments == 1 and cfg.infer.run_inference and cfg.infer.remesh
    ):
        return None, None, None, None, None
    ms_io_remesh = pml.MeshSet()
    remesh_path = f"{reference_garment_mesh_path[:-4]}_{cfg.infer.remesh_tag}.obj"
    ms_io_remesh.load_new_mesh(remesh_path)
    remesh_mesh = ms_io_remesh.current_mesh()
    remesh_verts = copy.deepcopy(remesh_mesh.vertex_matrix())
    remesh_faces = copy.deepcopy(remesh_mesh.face_matrix())
    remesh_uv_w = copy.deepcopy(remesh_mesh.wedge_tex_coord_matrix())
    original_mesh = trimesh.Trimesh(scene["initPos"], scene["faces"])
    proximesh = trimesh.proximity.ProximityQuery(original_mesh)
    surface_pts, _, triangles_indices = proximesh.on_surface(remesh_verts)
    vertices_indices = original_mesh.faces[triangles_indices]
    triangles_coords = original_mesh.vertices[vertices_indices]
    # barycentric coefficients
    barycentric_wts = trimesh.triangles.points_to_barycentric(
        triangles=triangles_coords, points=surface_pts
    )
    barycentric_verts_idx = vertices_indices
    ms_io_remesh.clear()
    return (
        remesh_verts,
        remesh_faces,
        remesh_uv_w,
        barycentric_wts,
        barycentric_verts_idx,
    )


def _estimate_pinning_verts(cfg, scene):
    """Estimate which garment vertices should be pinned."""
    verts_pinning_mask = np.array([False] * len(scene["initPos"]))
    if not cfg.train.estimate_pinning_bones:
        return verts_pinning_mask
    print("Estimating pinning vertices...")
    mesh_set = pml.MeshSet()
    mesh_set.add_mesh(pml.Mesh(scene["initPos"], scene["faces"]), "cloth")
    boundary_verts = get_boundary_verts(mesh_set)
    ymax = np.max(boundary_verts[:, 1])
    cond = np.abs(boundary_verts[:, 1] - ymax) < 5.0
    pinning_verts = boundary_verts[cond, :]
    if cfg.train.max_num_pin_verts > 0:
        pinning_verts = pinning_verts[: cfg.train.max_num_pin_verts]
    tree = spatial.KDTree(np.array(scene["initPos"]))
    dist, idx = tree.query(pinning_verts, k=1, p=2)
    verts_pinning_mask[idx] = True
    return verts_pinning_mask


def _estimate_pinning_bones(
    cfg,
    scene,
    verts_pinning_mask,
    max_num_random_bones,
    reference_pinned_bones,
    custom_garment_dirpath,
):
    """Estimate pinning bone positions via farthest-point geodesic sampling."""
    random_pinned_bones_indices = []
    if not cfg.train.estimate_pinning_bones:
        return random_pinned_bones_indices, reference_pinned_bones
    print("Estimating pinning bones...")
    num_random_pinned_bones = (
        (2 * int(np.sqrt(max_num_random_bones)))
        if cfg.train.max_num_pin_bones == -1
        else cfg.train.max_num_pin_bones
    )
    random_pinned_bones_indices.append(int(scene["initPos"][:, 1].argmax()))
    terminating_length = num_random_pinned_bones
    solver = pp3d.MeshHeatMethodDistanceSolver(scene["initPos"], scene["faces"])
    min_distances = solver.compute_distance(random_pinned_bones_indices[0])
    min_distances[~verts_pinning_mask] = 0.0
    # Select remaining points
    while len(random_pinned_bones_indices) < terminating_length:
        # Set distances of already selected points to -1
        min_distances[random_pinned_bones_indices] = -1
        # Find the point with maximum distance
        max_idx = np.argmax(min_distances)
        random_pinned_bones_indices.append(max_idx)
        # Update distances with the new point
        distances = solver.compute_distance(max_idx)
        distances[~verts_pinning_mask] = 0.0
        min_distances = np.minimum(min_distances, distances)
    random_pinned_bones_indices = np.array(random_pinned_bones_indices)
    if len(reference_pinned_bones) == 0:
        reference_pinned_bones = random_pinned_bones_indices
    trimesh.PointCloud(scene["initPos"][random_pinned_bones_indices]).export(
        f"{custom_garment_dirpath}/random_pinned_bones.obj"
    )
    return random_pinned_bones_indices, reference_pinned_bones


def _compute_body_pinning_verts(cfg, scene, random_pinned_bones_indices):
    """Compute the body vertices to which pinned bones attach."""
    if cfg.train.body_pinning_info:
        body_predefined_pinning_verts = trimesh.load(
            cfg.body.custom_obj_path[:-4] + "_waist_verts.obj"
        ).vertices
        if cfg.garment.class_type == "upper":
            body_predefined_pinning_verts = trimesh.load(
                cfg.body.custom_obj_path[:-4] + "_neckline_verts.obj"
            ).vertices
    else:
        _body_tree = spatial.KDTree(np.array(scene["bodyPos"]))
        _, _nn_idx = _body_tree.query(
            scene["initPos"][random_pinned_bones_indices], k=1
        )
        body_predefined_pinning_verts = scene["bodyPos"][_nn_idx]
    return body_predefined_pinning_verts


def _add_garment_bones_to_scene(cfg, scene, num_pinned_bones, max_num_random_bones):
    """Add new garment bone entries to the scene arrays."""
    num_new_garment_bones = scene["initPos"].shape[0]
    if cfg.train.random_bones_placement:
        num_new_garment_bones = max_num_random_bones + num_pinned_bones
    scene["boneNames"] = np.append(
        scene["boneNames"],
        np.array(
            [f"garment_{i}" for i in range(num_new_garment_bones - num_pinned_bones)]
        ),
    )
    scene["boneNames"] = np.append(
        scene["boneNames"], np.array([f"pinned_{i}" for i in range(num_pinned_bones)])
    )
    scene["bonePos"] = np.append(
        scene["bonePos"], np.zeros((num_new_garment_bones, 3)), axis=0
    )
    scene["bodyBoneWeights"] = np.append(
        scene["bodyBoneWeights"],
        np.zeros((num_new_garment_bones, scene["bodyBoneWeights"].shape[1])),
        axis=0,
    )
    scene["boneTransform"] = np.append(
        scene["boneTransform"], np.array([np.eye(3)] * num_new_garment_bones), axis=0
    )
    scene["boneParentIndices"] = np.append(
        scene["boneParentIndices"], np.array([-1] * num_new_garment_bones), axis=0
    )
    scene["activeBones"] = np.append(
        scene["activeBones"], np.array([True] * num_new_garment_bones), axis=0
    )
    scene["clothBoneWeights"] = np.zeros(
        (scene["boneNames"].shape[0], scene["initPos"].shape[0])
    )


def _compute_bone_masks(scene):
    """Compute active, pinned, and unpinned bone masks."""
    active_bones = np.array([is_active(bone) for bone in scene["boneNames"]])
    pinned_bones = np.array(
        [is_pinned(bone) for bone in scene["boneNames"][active_bones]]
    )
    unpinned_active_bones = np.array([is_unpinned(bone) for bone in scene["boneNames"]])
    n_active_bones = sum(active_bones)
    bones_pinning_mask = np.array([False] * n_active_bones)
    bones_pinning_mask[pinned_bones] = True
    active_indices = np.where(active_bones)[0]
    pinned_bone_indices = active_indices[bones_pinning_mask]
    return (
        active_bones,
        pinned_bones,
        unpinned_active_bones,
        n_active_bones,
        bones_pinning_mask,
        pinned_bone_indices,
    )


def _place_custom_garment_bones(
    cfg,
    scene,
    active_bones,
    unpinned_active_bones,
    pinned_bone_indices,
    reference_pinned_bones,
    num_pinned_bones,
    bones_lod_levels,
    max_num_random_bones,
    panels_verts,
    body_predefined_pinning_verts,
    custom_garment_dirpath,
):
    """Place garment bones via FPS or all-vertex strategy and compute body NN indices."""
    scene["boneTransform"][active_bones] = np.array([np.eye(3)] * sum(active_bones))
    tree = KDTree(scene["initPos"])
    dist, nn_indices = tree.query(scene["bonePos"][active_bones], k=1)
    scene["bonePos"][active_bones] = scene["initPos"][nn_indices]

    vertex_bone_selection_mask = np.array([False] * len(scene["initPos"]))
    sampled_panel_points = None
    bones_lod_sampled = None

    if cfg.train.random_bones_placement:
        print("Randomly placing garment bones...")
        bone_indices_init = reference_pinned_bones.tolist()
        bones_lod_sampled = bones_placement_fps(
            scene["initPos"],
            scene["faces"],
            bone_indices_sampled=bone_indices_init.copy(),
            bone_lods=bones_lod_levels,
        )
        np.save(f"{custom_garment_dirpath}/bones_lod_sampled.npy", bones_lod_sampled)
        bone_indices_sampled = bones_lod_sampled[f"{max_num_random_bones}"]
        vertex_bone_selection_mask[bone_indices_sampled] = True
        # removing pinned bones from the sampled list (adding them to scene separately later)
        bone_indices_sampled = bone_indices_sampled[num_pinned_bones:]
        sampled_points = scene["initPos"][bone_indices_sampled]
        if cfg.train.panels_as_rest:
            sampled_panel_points = panels_verts[bone_indices_sampled]
        # update bone positions
        scene["bonePos"][unpinned_active_bones] = sampled_points
    else:
        # use all vertices as garment bones
        scene["bonePos"][active_bones] = scene["initPos"]

    # find nearest body vertex for each pinned bone
    pinned_bones_bodyverts_idx = np.array([-1])
    if len(reference_pinned_bones) > 0:
        pinned_bone_pos = scene["initPos"][reference_pinned_bones]
        scene["bonePos"][pinned_bone_indices] = pinned_bone_pos
        # estimate pinning bone positions-indices on body
        bodytree = spatial.KDTree(scene["bodyPos"])
        _, predefined_pin_idx = bodytree.query(body_predefined_pinning_verts)
        predefined_tree = spatial.KDTree(body_predefined_pinning_verts)
        _, pin_idx = predefined_tree.query(pinned_bone_pos)
        pinned_bones_bodyverts_idx = predefined_pin_idx[pin_idx]
        np.save(
            f"{custom_garment_dirpath}/pinned_bones_bodyverts_idx.npy",
            pinned_bones_bodyverts_idx,
        )
        pinned_bones_bodyverts_idx = torch.Tensor(pinned_bones_bodyverts_idx).long()
        trimesh.PointCloud(scene["bodyPos"][pinned_bones_bodyverts_idx]).export(
            f"{custom_garment_dirpath}/body_pinned_bones.obj"
        )

    # find nearest body vertex for each garment vertex and bone
    bodytree = spatial.KDTree(scene["bodyPos"])
    _, garment_verts_body_nnidx = bodytree.query(scene["initPos"], k=1, p=2)
    _, garment_bones_body_nnidx = bodytree.query(
        scene["bonePos"][active_bones], k=1, p=2
    )

    return (
        vertex_bone_selection_mask,
        sampled_panel_points,
        bones_lod_sampled,
        pinned_bones_bodyverts_idx,
        garment_verts_body_nnidx,
        garment_bones_body_nnidx,
    )


def _compute_geodesic_skin_weights(cfg, scene, active_bones, custom_garment_dirpath):
    """Compute geodesic-based skin weights and transfer to scene."""
    weight_mask = np.array([True] * len(scene["boneNames"]))
    cloth_joint_positions = scene["bonePos"][active_bones]
    cloth_verts_weights, _ = weight_transfer.transfer_weights_via_rbf(
        sourceV=scene["bodyPos"],
        sourceW=scene["bodyBoneWeights"],
        targetV=scene["initPos"],
        source_mask=weight_mask,
    )
    _, nn_bone_indices = weight_transfer.copy_weights_via_knn(
        sourceV=scene["initPos"],
        sourceW=cloth_verts_weights,
        targetV=cloth_joint_positions,
        nn=1,
    )
    cloth_bones_weights, _ = weight_transfer.copy_weights_via_knn(
        sourceV=scene["bodyPos"],
        sourceW=scene["bodyBoneWeights"].T,
        targetV=cloth_joint_positions,
        nn=1,
    )
    solver = pp3d.MeshHeatMethodDistanceSolver(scene["initPos"], scene["faces"])
    vertex_gdist_cache_path = os.path.join(
        custom_garment_dirpath, "vertex_gdist_cached.npy"
    )
    if os.path.exists(vertex_gdist_cache_path):
        print("Using cached geodesic distances...")
        vertex_gdistances = np.load(vertex_gdist_cache_path)
    else:
        print("Calculating geodesic distances...")
        vertex_gdistances_list = []
        for idx in range(len(scene["initPos"])):
            gdist = solver.compute_distance(idx)
            vertex_gdistances_list.append(gdist)
        vertex_gdistances = np.array(vertex_gdistances_list)
        np.save(vertex_gdist_cache_path, vertex_gdistances)
    # make vertex_gdistances a symmetric matrix
    vertex_gdistances = (vertex_gdistances + vertex_gdistances.T) / 2
    # compute geodesic weights
    geodesic_distances = vertex_gdistances[nn_bone_indices]
    spread = cfg.train.skinwt_spread
    sigma = np.sqrt(scene["initPos"].shape[0] / len(geodesic_distances))
    geodesic_wts = 1.0 / np.exp(geodesic_distances / (spread * sigma))
    geodesic_wts = np.nan_to_num(geodesic_wts, 0.0)
    geodesic_wts, filter_idx, filter_mask = filter_top_k(
        geodesic_wts, k=cfg.train.skinwt_influence_k
    )
    scene["clothBoneWeights"][~active_bones, :] = 0.0
    scene["clothBoneWeights"][active_bones] = geodesic_wts
    scene["clothBoneWeights"] = scene["clothBoneWeights"] / scene[
        "clothBoneWeights"
    ].sum(axis=0, keepdims=True)
    scene["clothBoneWeights"] = np.nan_to_num(scene["clothBoneWeights"], 0.0)
    return vertex_gdistances, geodesic_distances, nn_bone_indices, cloth_bones_weights


def _apply_hard_attach_and_normalize(
    cfg, scene, active_bones, verts_pinning_mask, nn_bone_indices
):
    """Hard-attach sampled vertices to bones and normalize weights."""
    if cfg.garment.hard_attach_bones:
        pinned_verts_wts = scene["clothBoneWeights"][:, verts_pinning_mask]
        scene["clothBoneWeights"][:, nn_bone_indices] = 0.0
        scene["clothBoneWeights"][active_bones, nn_bone_indices] = 1.0
        # restore pinned vertices weights (for smooth pinning)
        scene["clothBoneWeights"][:, verts_pinning_mask] = copy.deepcopy(
            pinned_verts_wts
        )
    # normalize skinning weights to sum to 1.0
    scene["clothBoneWeights"] = scene["clothBoneWeights"] / scene[
        "clothBoneWeights"
    ].sum(axis=0, keepdims=True)
    scene["clothBoneWeights"] = np.nan_to_num(scene["clothBoneWeights"], 0.0)


def _visualize_skinning(
    cfg,
    scene,
    active_bones,
    geodesic_distances,
    bones_lod_levels,
    num_pinned_bones,
    panels_verts,
    sampled_panel_points,
    custom_garment_dirpath,
):
    """Export per-LOD skinning visualization meshes."""
    np.random.seed(10000)
    skinning_vis_path = custom_garment_dirpath + "/skinning_visdir"
    os.makedirs(skinning_vis_path, exist_ok=True)
    colorwts = scene["clothBoneWeights"][active_bones]
    spread = cfg.train.skinwt_spread
    sigma = np.sqrt(scene["initPos"].shape[0] / len(geodesic_distances))
    geodesic_wts = np.log(1.0 / (colorwts + 1e-6)) * (spread * sigma)
    random_bone_colors = np.random.rand(geodesic_wts.shape[0], 3) * 255
    for lod_i in bones_lod_levels:
        _visualize_skinning_lod(
            cfg,
            scene,
            active_bones,
            lod_i,
            num_pinned_bones,
            spread,
            geodesic_wts,
            random_bone_colors,
            panels_verts,
            sampled_panel_points,
            skinning_vis_path,
        )


def _visualize_skinning_lod(
    cfg,
    scene,
    active_bones,
    lod_i,
    num_pinned_bones,
    spread,
    geodesic_wts,
    random_bone_colors,
    panels_verts,
    sampled_panel_points,
    skinning_vis_path,
):
    """Export skinning visualization for a single LOD level."""
    curr_bones_num = num_pinned_bones + lod_i
    sigma_i = np.sqrt(scene["initPos"].shape[0] / curr_bones_num)
    colorwts_i = geodesic_wts[:lod_i, :]
    colorwts_i = 1.0 / np.exp(colorwts_i / (spread * sigma_i))
    colorwts_i = colorwts_i / colorwts_i.sum(axis=0, keepdims=True)
    vertex_colors = (colorwts_i.T @ random_bone_colors[:lod_i]).astype(np.uint8)
    rgba_colors = np.ones((vertex_colors.shape[0], 4), dtype=np.uint8) * 255
    rgba_colors[:, :3] = vertex_colors
    trimesh.Trimesh(scene["initPos"], scene["faces"], vertex_colors=rgba_colors).export(
        f"{skinning_vis_path}/{lod_i}_skinned_cloth.obj"
    )
    rgba_bone_colors = np.ones((colorwts_i.shape[0], 4), dtype=np.uint8) * 255
    rgba_bone_colors[:, :3] = random_bone_colors[:lod_i].astype(np.uint8)
    trimesh.PointCloud(
        scene["bonePos"][active_bones][:lod_i], colors=rgba_bone_colors
    ).export(f"{skinning_vis_path}/{lod_i}_skinned_bones.obj")
    if cfg.train.panels_as_rest:
        trimesh.Trimesh(panels_verts, scene["faces"], vertex_colors=rgba_colors).export(
            f"{skinning_vis_path}/{lod_i}_skinned_cloth_panels.obj"
        )
        trimesh.PointCloud(
            sampled_panel_points[:lod_i], colors=rgba_bone_colors
        ).export(f"{skinning_vis_path}/{lod_i}_skinned_bones_panels.obj")


def _build_graph_data(
    cfg,
    scene,
    reference_verts,
    reference_faces,
    reference_vertex_normals,
    reference_garment_mesh,
    vertex_bone_selection_mask,
    verts_pinning_mask,
    vertex_gdistances,
    remesh_verts,
    remesh_faces,
    remesh_uv_w,
    barycentric_wts,
    barycentric_verts_idx,
    bones_lod_sampled,
):
    """Build the torch_geometric graph data structure."""
    print("Preparing graph data...")
    reference_mesh = trimesh.Trimesh(reference_verts, reference_faces)
    graph_data = torch_geometric.utils.from_trimesh(reference_mesh)
    GenerateMeshNormals = torch_geometric.transforms.GenerateMeshNormals()
    graph_data = GenerateMeshNormals(graph_data)

    # remeshed data
    graph_data.remesh_verts = remesh_verts
    graph_data.remesh_faces = remesh_faces
    graph_data.remesh_uv_w = remesh_uv_w
    graph_data.barycentric_wts = barycentric_wts
    graph_data.barycentric_verts_idx = barycentric_verts_idx

    # areas
    _face_areas = pp3d.face_areas(reference_verts, reference_faces)  # noqa: F841
    verts_areas = pp3d.vertex_areas(reference_verts, reference_faces)

    faces_to_edges = torch_geometric.transforms.FaceToEdge()
    graph_data = faces_to_edges(graph_data)
    graph_data.faces = torch.Tensor(reference_faces)
    # connect bones to bones
    bone_nodes_idx = np.argwhere(vertex_bone_selection_mask)
    bone_edges_idx = np.array(
        [
            [bone_nodes_idx[i], bone_nodes_idx[j]]
            for i in range(len(bone_nodes_idx))
            for j in range(i + 1, len(bone_nodes_idx))
        ]
    )
    bone_edges_idx = torch.tensor(bone_edges_idx.reshape(2, -1)).long()
    graph_data.edge_index = torch.cat([graph_data.edge_index, bone_edges_idx], dim=-1)

    pinning_labels = torch.Tensor((verts_pinning_mask).astype("float").reshape(-1, 1))
    bone_labels = torch.Tensor(
        (vertex_bone_selection_mask).astype("float").reshape(-1, 1)
    )
    graph_data.x = torch.cat(
        [graph_data.pos.clone(), pinning_labels, bone_labels], axis=-1
    )
    nodes_geodesic_wts = 1 / np.exp(vertex_gdistances)
    edge_wts = nodes_geodesic_wts[graph_data.edge_index[0], graph_data.edge_index[1]]
    graph_data.edge_attr = torch.Tensor(edge_wts)

    graph_data.bones_lod_sampled = bones_lod_sampled
    graph_data.verts = torch.Tensor(reference_verts)
    graph_data.faces = torch.Tensor(reference_faces)
    graph_data.vertex_normals = torch.Tensor(reference_vertex_normals)
    graph_data.tex_coords = torch.Tensor(
        reference_garment_mesh.wedge_tex_coord_matrix()
    )
    graph_data.verts_areas = torch.Tensor(verts_areas)
    graph_data.nodes_geodesic_wts = nodes_geodesic_wts

    return graph_data, verts_areas, nodes_geodesic_wts


def _apply_graph_pe_and_ppf(cfg, graph_data, nodes_geodesic_wts, verts_areas):
    """Apply positional encodings and point-pair features for multi-garment mode."""
    if cfg.hypermodulator.graph_pe_type == "random_walk":
        print("Random Walk...")
        RandomWalkPE = torch_geometric.transforms.AddRandomWalkPE(
            walk_length=cfg.garment_embedder_config.pos_enc_dim, attr_name="pe"
        )
        graph_data = RandomWalkPE(
            graph_data.to("cuda:0")
        ).cpu()  # random walk for sparse tensor is only supported on cuda
    elif cfg.hypermodulator.graph_pe_type == "laplacian_eigenvecs":
        print("Computing Laplacian Eigenvectors...")
        LaplacianPE = torch_geometric.transforms.AddLaplacianEigenvectorPE(
            k=cfg.garment_embedder_config.pos_enc_dim, attr_name="pe"
        )
        graph_data = LaplacianPE(graph_data)

    # compute PPF features
    PPF = torch_geometric.transforms.PointPairFeatures(cat=True)
    graph_data = PPF(graph_data)
    graph_data.pos = None
    # scatter edge ppf features to nodes
    node_ppf = torch.zeros(
        graph_data.num_nodes,
        graph_data.edge_attr.size(1),
        device=graph_data.edge_attr.device,
    )
    node_ppf = node_ppf.scatter_add_(
        0,
        graph_data.edge_index[1].unsqueeze(-1).expand_as(graph_data.edge_attr),
        graph_data.edge_attr,
    )
    counts = torch.bincount(
        graph_data.edge_index[1], minlength=graph_data.num_nodes
    ).unsqueeze(-1)
    node_ppf = node_ppf / counts.clamp(min=1)
    # add node features
    graph_data.node_ppf = node_ppf
    graph_data.verts_areas = torch.Tensor(verts_areas)
    nodes_geodesic_wts = filter_top_k(nodes_geodesic_wts, k=32)[0]
    graph_data.verts_geowts = torch.Tensor(nodes_geodesic_wts)
    return graph_data


def _populate_graph_metadata(
    graph_data,
    scene,
    active_bones,
    pinned_bones,
    unpinned_active_bones,
    verts_pinning_mask,
    bones_pinning_mask,
    random_pinned_bones_indices,
    pinned_bones_bodyverts_idx,
    garment_bones_body_nnidx,
    garment_verts_body_nnidx,
    vertex_bone_selection_mask,
    cloth_bones_weights,
    cloth_joint_positions,
    panels_verts,
    panels_faces,
    seam_verts_indices,
    rest_pos,
    cfg,
):
    """Attach all metadata tensors to the graph data object."""
    graph_data.batch = torch.zeros(len(graph_data.x), dtype=torch.int64)
    graph_data.active_bones = torch.Tensor(active_bones)
    graph_data.verts_pinning_mask = torch.Tensor(verts_pinning_mask)
    graph_data.bones_pinning_mask = torch.Tensor(bones_pinning_mask)
    graph_data.pinned_bones = (
        torch.Tensor(pinned_bones) if len(pinned_bones) > 0 else torch.Tensor([-1])
    )
    graph_data.random_pinned_bones_indices = (
        torch.Tensor(random_pinned_bones_indices)
        if len(random_pinned_bones_indices) > 0
        else torch.Tensor([-1])
    )
    graph_data.pinned_bones_bodyverts_idx = (
        torch.Tensor(pinned_bones_bodyverts_idx)
        if len(pinned_bones_bodyverts_idx) > 0
        else torch.Tensor([-1])
    )
    graph_data.garment_bones_body_nnidx = torch.Tensor(garment_bones_body_nnidx)
    graph_data.garment_verts_body_nnidx = torch.Tensor(garment_verts_body_nnidx)
    graph_data.unpinned_active_bones = torch.Tensor(unpinned_active_bones)
    graph_data.cloth_joint_positions = torch.Tensor(cloth_joint_positions)
    graph_data.vertex_bone_selection_mask = torch.Tensor(vertex_bone_selection_mask)
    graph_data.cloth_bones_weights = torch.Tensor(cloth_bones_weights).T
    graph_data.verts_skinning_weights = torch.Tensor(
        scene["clothBoneWeights"][active_bones]
    ).T
    graph_data.cloth_bones_rest_positions = torch.Tensor(scene["bonePos"][active_bones])
    graph_data.all_bones_rest_positions = torch.Tensor(scene["bonePos"])
    graph_data.all_bones_rest_transforms = torch.Tensor(scene["boneTransform"])
    graph_data.body_verts_weights = torch.Tensor(scene["bodyBoneWeights"])
    graph_data.panels_verts = torch.Tensor(panels_verts)
    graph_data.panels_faces = torch.Tensor(panels_faces)
    graph_data.seam_verts_indices = torch.Tensor(seam_verts_indices)
    if cfg.train.panels_as_rest:
        graph_data.rest_pos = torch.Tensor(rest_pos)


def load_garments_meshes(
    cfg,
    garment_paths,
    garment_idx,
    global_normalization=False,
    global_y_translation=0.0,
    global_scale=1.0,
):
    custom_garment_dirpath = garment_paths[garment_idx]

    print(f"Processing garment : {garment_idx}", flush=True)

    # initialize config
    config = dataio.load_config(cfg.train.config_path)

    # initialize scene
    scene = dataio.load_scene(cfg.train.scene_path)
    scene = dict(scene)  # Allow modifications for scene

    # initialize from reference body mesh
    if cfg.body.name == "avatar":
        _init_body_avatar(cfg, scene)
    elif cfg.body.name == "3DBiCar":
        _init_body_3dbicar(cfg, scene)

    # GARMENT MESH
    (
        reference_garment_mesh,
        reference_garment_name,
        reference_verts,
        reference_faces,
        reference_vertex_normals,
    ) = _load_garment_mesh(cfg, custom_garment_dirpath, scene)
    reference_pinned_bones = np.array(list(cfg.garment.pinned_verts_indices))

    # load the panel mesh
    panels_verts, panels_faces, seam_verts_indices, rest_pos = _load_panels_and_seams(
        cfg, custom_garment_dirpath, scene, reference_verts
    )

    # remeshing for inference
    reference_garment_mesh_path = glob(custom_garment_dirpath + "/*_sim.obj")[0]
    remesh_verts, remesh_faces, remesh_uv_w, barycentric_wts, barycentric_verts_idx = (
        _prepare_remesh_data(cfg, scene, reference_garment_mesh_path)
    )

    # number of bones
    bones_lod_levels = cfg.train.bones_lod_levels
    max_num_random_bones = max(bones_lod_levels)

    # estimate pinning vertices
    verts_pinning_mask = _estimate_pinning_verts(cfg, scene)

    # estimate pinning bones
    random_pinned_bones_indices, reference_pinned_bones = _estimate_pinning_bones(
        cfg,
        scene,
        verts_pinning_mask,
        max_num_random_bones,
        reference_pinned_bones,
        custom_garment_dirpath,
    )

    # pin to body
    body_predefined_pinning_verts = _compute_body_pinning_verts(
        cfg, scene, random_pinned_bones_indices
    )

    # add new bones
    num_pinned_bones = len(reference_pinned_bones)
    _add_garment_bones_to_scene(cfg, scene, num_pinned_bones, max_num_random_bones)

    # compute bone masks
    (
        active_bones,
        pinned_bones,
        unpinned_active_bones,
        n_active_bones,
        bones_pinning_mask,
        pinned_bone_indices,
    ) = _compute_bone_masks(scene)

    vertex_bone_selection_mask = np.array([False] * len(scene["initPos"]))
    np.save(
        f"{custom_garment_dirpath}/vertex_bone_selection_mask.npy",
        vertex_bone_selection_mask,
    )

    sampled_panel_points = None
    bones_lod_sampled = None
    pinned_bones_bodyverts_idx = np.array([-1])
    garment_verts_body_nnidx = None
    garment_bones_body_nnidx = None

    if cfg.train.custom_garment_bones:
        (
            vertex_bone_selection_mask,
            sampled_panel_points,
            bones_lod_sampled,
            pinned_bones_bodyverts_idx,
            garment_verts_body_nnidx,
            garment_bones_body_nnidx,
        ) = _place_custom_garment_bones(
            cfg,
            scene,
            active_bones,
            unpinned_active_bones,
            pinned_bone_indices,
            reference_pinned_bones,
            num_pinned_bones,
            bones_lod_levels,
            max_num_random_bones,
            panels_verts,
            body_predefined_pinning_verts,
            custom_garment_dirpath,
        )

    joint_positions = scene["bonePos"]
    cloth_joint_positions = joint_positions[active_bones]

    vertex_gdistances = None
    geodesic_distances = None
    nn_bone_indices = None
    cloth_bones_weights = None

    if cfg.garment.skin_weight_transfer_method == "geodesic":
        vertex_gdistances, geodesic_distances, nn_bone_indices, cloth_bones_weights = (
            _compute_geodesic_skin_weights(
                cfg, scene, active_bones, custom_garment_dirpath
            )
        )

    # hard attach sampled/selected vertices to corresponding bones
    _apply_hard_attach_and_normalize(
        cfg, scene, active_bones, verts_pinning_mask, nn_bone_indices
    )

    # skinning visualization (requires geodesic weight transfer)
    if cfg.garment.skin_weight_transfer_method == "geodesic":
        _visualize_skinning(
            cfg,
            scene,
            active_bones,
            geodesic_distances,
            bones_lod_levels,
            num_pinned_bones,
            panels_verts,
            sampled_panel_points,
            custom_garment_dirpath,
        )

    # prepare graph data
    graph_data, verts_areas, nodes_geodesic_wts = _build_graph_data(
        cfg,
        scene,
        reference_verts,
        reference_faces,
        reference_vertex_normals,
        reference_garment_mesh,
        vertex_bone_selection_mask,
        verts_pinning_mask,
        vertex_gdistances,
        remesh_verts,
        remesh_faces,
        remesh_uv_w,
        barycentric_wts,
        barycentric_verts_idx,
        bones_lod_sampled,
    )

    if cfg.train.max_num_garments > 1:
        graph_data = _apply_graph_pe_and_ppf(
            cfg, graph_data, nodes_geodesic_wts, verts_areas
        )

    _populate_graph_metadata(
        graph_data,
        scene,
        active_bones,
        pinned_bones,
        unpinned_active_bones,
        verts_pinning_mask,
        bones_pinning_mask,
        random_pinned_bones_indices,
        pinned_bones_bodyverts_idx,
        garment_bones_body_nnidx,
        garment_verts_body_nnidx,
        vertex_bone_selection_mask,
        cloth_bones_weights,
        cloth_joint_positions,
        panels_verts,
        panels_faces,
        seam_verts_indices,
        rest_pos,
        cfg,
    )

    print("Creating garment object...")
    # initialize garment object
    garment_metadata = Garment(cfg, scene)
    garment_metadata.graph_data = graph_data
    garment_metadata.dirpath = custom_garment_dirpath

    garment_metadata.lbs = lbs.LBS.from_scene_and_config(scene, config)
    garment_metadata.type = cfg.garment.class_type

    print("Done!!!")

    return garment_metadata, reference_garment_name

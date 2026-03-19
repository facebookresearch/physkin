# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import bvh
import geo
import igl
import numpy as np
import torch
import torch.nn as nn
import utils


def validateTopo(f):
    assert torch.all(f[:, 0] != f[:, 1])
    assert torch.all(f[:, 1] != f[:, 2])
    assert torch.all(f[:, 0] != f[:, 2])


class SceneConfig(nn.Module):
    """
    Configuration for the scene including cloth and collision body, without any physics or rigging information.

    Expected keys for the mesh and config dictionaries:
    Assume
    V - number of cloth vertices
    F - number of cloth faces
    Vb - number of body vertices
    Fb - number of body triangles
    Vr - number of cloth vertices at rest. Note these are expected to correspond to 0-Vr vertex indices.
    mesh = {
        "initPos", Vx3 matrix
        "restPos", Vrx3 matrix
        "faces", Fx3 matrix
        "fixed", V vector
        "bodyPos", Vb x 3 matrix
        "bodyFaces", Fb x 3 matrix
    }
    """

    def __init__(self, data, dtype=torch.float32, options=None):
        super(SceneConfig, self).__init__()
        if options is None:
            options = {}

        # Remember the inputs
        self.scene = data

        # Initial positions
        initPos = torch.tensor(data["initPos"], dtype=dtype)
        self.register_buffer("initPos", initPos)
        self.numPos = len(self.initPos)
        initPosAffine = torch.concatenate(
            (self.initPos, torch.ones((len(self.initPos), 1), dtype=dtype)),
            dim=1,
        )
        self.register_buffer("initPosAffine", initPosAffine)

        if "domainPos" in data:
            domainPos = torch.tensor(data["domainPos"], dtype=dtype)
            self.register_buffer("domainPos", domainPos)
        else:
            self.register_buffer("domainPos", initPos)

        # Faces (triangle vertex indices)
        faces = torch.tensor(data["faces"], dtype=torch.int32)
        self.register_buffer("faces", faces)
        validateTopo(faces)

        # Edges (used by spring constraints)
        edges, face_edges = geo.compute_edge_topology(faces, return_inverse=True)
        face_edges = torch.tensor(face_edges, dtype=torch.int32)
        edges = torch.tensor(edges, dtype=torch.int32)
        self.register_buffer("edges", edges)

        # Detect boundary vertices. Useful for things like smoothing
        bdry = torch.zeros(len(initPos), dtype=torch.bool)
        bdry[geo.get_boundary_vertices(faces)] = True
        self.register_buffer("bdry", bdry)

        # Laplacian matrix
        lap = igl.cotmatrix(initPos.numpy(), faces.numpy())  # type: ignore
        if np.isnan(lap.data).any():
            print("Warning, cotmatrix from igl has nans, so using uniform laplacian.")
            lap = geo.uniform_laplacian(faces, len(initPos)).to(dtype)
        else:
            lap = (
                torch.sparse_coo_tensor(lap.nonzero(), lap.data, lap.shape)
                .to_sparse_csr()
                .to(dtype)
            )
        self.register_buffer("laplacian", lap)

        # Vertex Local bases
        R = geo.compute_local_bases(initPos.numpy(), faces.numpy())
        Rinv = np.linalg.inv(R)
        Rinv_isnan = np.isnan(Rinv.sum(axis=(1, 2)))
        Rinv_stable = torch.tensor(
            np.where(
                Rinv_isnan.reshape((len(Rinv_isnan), 1, 1)),
                np.stack([np.eye(3, dtype=np.float32)] * len(R)),
                Rinv,
            ),
            dtype=torch.float32,
        )
        R_stable = torch.tensor(
            np.where(
                Rinv_isnan.reshape((len(Rinv_isnan), 1, 1)),
                np.stack([np.eye(3, dtype=np.float32)] * len(R)),
                R,
            ),
            dtype=torch.float32,
        )
        R_stable_affine = torch.concatenate(
            (
                R_stable,
                torch.zeros((len(R), 1, 3), dtype=torch.float32),
            ),
            dim=1,
        )
        self.register_buffer("vertex_local_bases_affine", R_stable_affine)
        L_init_pos = lap @ initPos
        Rinv_L_init_pos = torch.matmul(Rinv_stable, L_init_pos.unsqueeze(-1))
        self.register_buffer("delta_mush_target", Rinv_L_init_pos)

        # Fixed vertices
        if "fixed" in data:
            fixed = torch.tensor(data["fixed"], dtype=torch.bool)
        else:
            fixed = torch.zeros(self.initPos.shape[0], dtype=torch.bool)
        self.register_buffer("fixed", fixed)
        self.numFixed = sum(fixed)

        assert len(self.fixed) == self.initPos.shape[0]

        # Body vertex positions
        if "bodyPos" in data:
            bodyPos = torch.tensor(data["bodyPos"], dtype=dtype)
            self.register_buffer("bodyPos", bodyPos)
            self.numBodyPos = len(self.bodyPos)
            bodyPosAffine = torch.concatenate(
                (self.bodyPos, torch.ones((len(self.bodyPos), 1), dtype=dtype)),
                dim=1,
            )
            self.register_buffer("bodyPosAffine", bodyPosAffine)

            # Body face triangle vertex indices
            bodyFaces = torch.tensor(data["bodyFaces"], dtype=torch.int32)
            self.register_buffer("bodyFaces", bodyFaces)

        # rest vertex positions
        has_explicit_rest_pos = "restPos" in data
        if has_explicit_rest_pos:
            print()
            print("Using explicit rest positions...")
            print()
            self.restPos = torch.tensor(data["restPos"], dtype=dtype)
        else:
            # use initPos as restPos
            self.restPos = self.initPos[self.faces.flatten(), :]

        # rest triangle faces
        restFaces = torch.arange(0, self.restPos.shape[0]).view(-1, 3)
        if len(self.restPos) != 0:
            restShapeInv = geo.compute_rest_shape_inv(self.restPos, restFaces)
            restAreas = geo.face_areas(self.restPos, restFaces)

            # Edge rest areas are associated with original topology, not rest topology
            restEdgeAreas = geo.edge_areas_from_face_areas(faces, restAreas)
        else:
            restShapeInv = torch.zeros(restFaces.shape[0], 2, 2, dtype=dtype)
            restAreas = torch.zeros(restFaces.shape[0], dtype=dtype)
            restEdgeAreas = torch.zeros(edges.shape[0], dtype=dtype)
        self.register_buffer("restShapeInv", restShapeInv)
        self.register_buffer("restAreas", restAreas)
        self.register_buffer("restEdgeAreas", restEdgeAreas)

        rest_edge_lengths = geo.edge_lengths(
            self.restPos, restFaces, face_edges, len(edges)
        )
        self.register_buffer("restEdgeLengths", rest_edge_lengths)

        if "texCoords" in data:
            texCoords = data["texCoords"]
            assert texCoords.shape[0] == self.restPos.shape[0]
            self.register_buffer(
                "texCoords",
                torch.tensor(
                    texCoords.reshape(len(self.faces), 3, -1)[..., 0:2], dtype=dtype
                ),
            )
        else:
            self.texCoords = None

        if (
            "disable_dihedral_topo" not in options
            or not options["disable_dihedral_topo"]
        ):
            self.dihedralTopo = geo.DihedralTopo(self.faces)
            self.register_buffer(
                "dihedralRestShape",
                geo.compute_dihedral_rest_shape(
                    self.restPos, self.restAreas, self.dihedralTopo
                ),
            )
        if "disable_cloth_bvh" not in options or not options["disable_cloth_bvh"]:
            self.clothBvh = bvh.BVH(self.initPos.to(torch.float32), self.faces)

    @property
    def has_body(self) -> bool:
        return hasattr(self, "bodyPos")


class ParamConfig(nn.Module):
    """
    Configuration parameters for the physics simulator without any rigging information.
    """

    def __init__(self, config, dtype=torch.float32):
        super(ParamConfig, self).__init__()

        self.config = config

        # Gravity
        g = self.config["gravity"]
        if isinstance(g, list) and len(g) == 3:
            gravity = torch.tensor(g, dtype=dtype)
        else:
            gravity = -torch.tensor([0.0, g, 0.0], dtype=dtype)
        self.register_buffer("gravity", gravity)

        # Register parameters with nn Module
        if isinstance(self.config.get("clothMaterial"), str):
            clothMaterialDict = utils.material_dict(self.config["clothMaterials"])
            assert self.config["clothMaterial"] in clothMaterialDict, (
                "Expected one of clothMaterial to be the name of one of the given materials, but none were found"
            )
            self.material = clothMaterialDict[self.config["clothMaterial"]]
        else:
            # Just pick the first one
            assert len(self.config["clothMaterials"]) > 0
            self.material = self.config["clothMaterials"][0]


class RigConfig(nn.Module):
    """
    Rig configuration.

    Expected keys for the data and config dictionaries:
    Assume
    V - number of cloth vertices
    F - number of cloth faces
    Vb - number of body vertices
    Fb - number of body triangles
    Vs - number of skeleton (bone) vertices
    Vsb - number of bones per body vertex
    Vsc - number of bones per cloth vertex
    Vr - number of cloth vertices at rest. Note these are expected to correspond to 0-Vr vertex indices.
    data = {
        "bodyBoneWeights", Vb x Vsb matrix
        "bodyBoneIndices", Vb x Vsb matrix (optional (if None, assume Vsb == Vs))
        "clothBoneWeights", V x Vsc matrix
        "clothBoneIndices", V x Vsc matrix (optional (if None, assume Vsc == Vs))
        "bonePos", Vs x 3 matrix
    }
    """

    def __init__(self, data, config):
        super(RigConfig, self).__init__()

        if "affine_state" in data:
            affine_state = torch.tensor(data["affine_state"], dtype=torch.float32)
        elif "bonePos" in data:
            affine_state = (
                torch.eye(4).unsqueeze(0).expand(len(data["bonePos"]), -1, -1).clone()
            )
        else:
            raise Exception("No joint position data")

        # add identity transforms for custom bones
        num_new_bones = np.abs(affine_state.shape[0] - len(data["bonePos"]))
        identity_transforms = torch.tensor(
            np.array([torch.eye(4)] * num_new_bones, dtype=np.float32)
        )
        affine_state = torch.cat((affine_state, identity_transforms), dim=0)

        # Check for custom bone positions and transforms
        if "bonePos" in data:
            affine_state[:, 0:3, 3] = torch.tensor(data["bonePos"], dtype=torch.float32)

        if "boneTransform" in data:
            affine_state[:, 0:3, 0:3] = torch.tensor(
                data["boneTransform"], dtype=torch.float32
            )

        if "domainBonePos" in data:
            domainBonePos = torch.tensor(data["domainBonePos"], dtype=torch.float32)
            self.register_buffer("domainBonePos", domainBonePos)
        else:
            self.register_buffer("domainBonePos", affine_state[:, 0:3, 3])

        self.register_buffer("boneAffineTransform", affine_state)
        boneAffineTransformInv = torch.linalg.inv(affine_state)
        self.register_buffer("boneAffineTransformInv", boneAffineTransformInv)
        self.numBones = len(self.boneAffineTransformInv)

        if "activeBones" in data:
            self.register_buffer(
                "activeBones", torch.tensor(data["activeBones"], dtype=torch.bool)
            )
        else:
            self.register_buffer(
                "activeBones", torch.ones(self.numBones, dtype=torch.bool)
            )

        # Cloth mesh bone weights
        clothBoneWeights = torch.tensor(data["clothBoneWeights"], dtype=torch.float32)
        if "clothBoneIndices" in data:
            clothBoneIndices = torch.tensor(data["clothBoneIndices"], dtype=torch.int64)
            clothBoneSparseWeights = clothBoneWeights
            clothBoneWeights = torch.zeros(
                (len(clothBoneSparseWeights), self.numBones), dtype=torch.float32
            )
            clothBoneWeights.scatter_(1, clothBoneIndices, clothBoneSparseWeights)
        assert clothBoneWeights.shape[0] == self.numBones
        self.register_buffer("clothBoneWeights", clothBoneWeights)

        # Body mesh bone weights
        bodyBoneWeights = torch.tensor(data["bodyBoneWeights"], dtype=torch.float32)
        if "bodyBoneIndices" in data:
            bodyBoneIndices = torch.tensor(data["bodyBoneIndices"], dtype=torch.int64)
            bodyBoneSparseWeights = bodyBoneWeights
            bodyBoneWeights = torch.zeros(
                (len(bodyBoneSparseWeights), self.numBones), dtype=torch.float32
            )
            bodyBoneWeights.scatter_(1, bodyBoneIndices, bodyBoneSparseWeights)
        assert bodyBoneWeights.shape[0] == self.numBones
        self.register_buffer("bodyBoneWeights", bodyBoneWeights)

    def __str__(self) -> str:
        # Explicitly override the str representation to hide skeleton_parameter_transform details
        return "RigConfig()"

    def __repr__(self) -> str:
        return self.__str__()


class Config(nn.Module):
    """
    Configuration for the physics simulator.

    This includes scene, param and rig configs.

    Expected keys for the mesh and config dictionaries:
    Assume
    V - number of cloth vertices
    F - number of cloth faces
    Vb - number of body vertices
    Fb - number of body triangles
    Vs - number of skeleton (bone) vertices
    Vsb - number of bones per body vertex
    Vsc - number of bones per cloth vertex
    Vr - number of cloth vertices at rest. Note these are expected to correspond to 0-Vr vertex indices.
    mesh = {
        "initPos", Vx3 matrix
        "restPos", Vrx3 matrix
        "faces", Fx3 matrix
        "fixed", V vector
        "bodyPos", Vb x 3 matrix
        "bodyFaces", Fb x 3 matrix
        "bodyBoneWeights", Vb x Vsb matrix
        "bodyBoneIndices", Vb x Vsb matrix (optional (if None, assume Vsb == Vs))
        "clothBoneWeights", V x Vsc x V matrix
        "clothBoneIndices", V x Vsc matrix (optional (if None, assume Vsc == Vs))
        "bonePos", Vs x 3 matrix
    }
    """

    def __init__(self, data: dict, config: dict):
        super(Config, self).__init__()

        self._raw_scene = data
        self._raw_config = config

        # Remember the inputs
        self.scene = SceneConfig(data, options=config)
        self.params = ParamConfig(config)
        self.rig = RigConfig(data, config)

        self.register_buffer(
            "vertex_mass", vertex_mass_from_config(self.scene, self.params)
        )


def compute_vertex_mass(
    num_vertices: int, faces: torch.Tensor, areas: torch.Tensor, density: float
):
    """
    Computes the mass of each vertex according to triangle areas and fabric density
    """
    triangle_masses = density * areas
    vertex_masses = torch.zeros(num_vertices, dtype=torch.float32, device=faces.device)

    vertex_masses[faces[:, 0]] += triangle_masses / 3.0
    vertex_masses[faces[:, 1]] += triangle_masses / 3.0
    vertex_masses[faces[:, 2]] += triangle_masses / 3.0
    return vertex_masses


def vertex_mass_from_config(scene: SceneConfig, params: ParamConfig):
    """Convenience routine for creating vertex mass array from a given config"""
    vertex_mass = compute_vertex_mass(
        scene.numPos,
        scene.faces,
        scene.restAreas,
        params.material["density"],
    )
    assert torch.any(vertex_mass > 0.0)
    return vertex_mass

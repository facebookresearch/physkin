# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time

import geo
import lbs
import torch
from bvh import BVH
from energy import dihedral_ortho, gravity, triangle_ortho_stvk_membrane


def physics_loss_from_state(
    lbs: lbs.LBS,
    contact_params,
    states: torch.Tensor,
    vertex_positions: torch.Tensor,
    body_vertex_positions: torch.Tensor | None = None,
    timings: dict | None = None,
    cloth_weights: torch.Tensor | None = None,
    only_collision: bool = False,
    pinning_mask: torch.Tensor | None = None,
    pinning_strength: float = 1.0,
):
    """
    Computes the physics loss from given (batched) poses.
    This uses a contact heuristic that assumes that cloth is penetration free with respect to the body at the zero pose.
    """
    begin_t = time.time()
    body_collision_energy = torch.zeros_like(states[..., 0])
    if lbs.config.params.material["bodyContactPenalty"] > 0.0:
        faces_c = lbs.config.scene.faces.detach()
        faces_b = lbs.config.scene.bodyFaces.detach()
        thickness = lbs.config.params.material["bodyOffsetThickness"]
        max_interp_tol = contact_params["body_contact_max_interpenetration_tolerance"]
        prox_pad = contact_params["body_contact_proximity_padding"]

        face_nml_b = geo.face_normals(body_vertex_positions, faces_b)
        nml_c = geo.vertex_normals(vertex_positions.detach(), faces_c)

        batch_size = states.shape[0]
        faces_b_cloned = torch.stack(
            [faces_b.clone() + (i * (1 + faces_b.max())) for i in range(batch_size)]
        )
        faces_c_cloned = torch.stack(
            [faces_c.clone() + (i * (1 + faces_c.max())) for i in range(batch_size)]
        )

        pos_b_cloned = body_vertex_positions.clone()
        pos_c_cloned = vertex_positions.clone()

        max_z_offset = torch.max(pos_b_cloned[:, :, 2])
        padding = max_z_offset * 2.0
        placement_offsets = torch.Tensor([padding * i for i in range(batch_size)]).to(
            pos_c_cloned.device
        )
        offsets_b = placement_offsets.repeat(pos_b_cloned.shape[1], 1).T
        offsets_c = placement_offsets.repeat(pos_c_cloned.shape[1], 1).T
        pos_b_cloned[:, :, 2] += offsets_b
        pos_c_cloned[:, :, 2] += offsets_c

        faces_b_cloned = faces_b_cloned.reshape(-1, 3)
        faces_c_cloned = faces_c_cloned.reshape(-1, 3)
        pos_b_cloned = pos_b_cloned.reshape(-1, 3)
        pos_c_cloned = pos_c_cloned.reshape(-1, 3)
        face_nml_b = face_nml_b.reshape(-1, 3)
        nml_c = nml_c.reshape(-1, 3)

        bvh = BVH(pos_b_cloned, faces_b_cloned)
        signed_dist = bvh.orientation_aware_signed_distance(
            pos_c_cloned, nml_c, pos_b_cloned, face_nml_b, max_interp_tol + prox_pad
        )
        signed_dist = signed_dist.reshape(batch_size, -1)

        interp = torch.relu(thickness - signed_dist)
        valid_mask = interp < (max_interp_tol + thickness)
        valid_interp = interp * valid_mask.float()
        interpenetration_penalty = torch.sum(valid_interp**2, dim=-1)

        body_collision_energy = (
            lbs.config.params.material["bodyContactPenalty"] * interpenetration_penalty
        )

    body_contact_t = time.time()

    if timings is not None:
        timings["body_contact"] += body_contact_t - begin_t

    pos = vertex_positions

    cfg = lbs.config
    begin_t = time.time()
    if cfg.params.material["membraneType"] == "stvk":
        triangle_energy, faces_energy = triangle_ortho_stvk_membrane.compute_energy(
            pos,
            cfg.scene.faces,
            cfg.scene.restShapeInv,
            cfg.scene.restAreas,
            cfg.params.material,
        )
    else:
        triangle_energy = torch.zeros(
            pos.shape[:-2], dtype=torch.float32, device=pos.device
        )
    triangle_t = time.time()

    bendStiffness = dihedral_ortho.compute_bend_stiffness(
        cfg.params.material, cfg.scene.texCoords, cfg.scene.dihedralTopo
    )

    if cfg.params.material["bendingType"] == "dihedral":
        dihedral_energy = dihedral_ortho.compute_energy(
            pos,
            bendStiffness,
            cfg.scene.dihedralRestShape,
            cfg.scene.faces,
            cfg.scene.dihedralTopo,
        )
    else:
        dihedral_energy = torch.zeros(
            pos.shape[:-2], dtype=torch.float32, device=pos.device
        )
    dihedral_t = time.time()

    gravity_energy = gravity.compute_energy(pos, cfg.vertex_mass, cfg.params.gravity)

    if pinning_mask is not None:
        faces_energy_with_pinning = (faces_energy * pinning_mask * pinning_strength) + (
            faces_energy * (1 - pinning_mask)
        )
        triangle_energy = 0.5 * torch.sum(faces_energy_with_pinning, dim=-1)

    gravity_t = time.time()
    if timings is not None:
        timings["triangle"] += triangle_t - begin_t
        timings["dihedral"] += dihedral_t - triangle_t
        timings["gravity"] += gravity_t - dihedral_t

    def reduce(energy):
        return torch.abs(energy.mean())

    assert triangle_energy.shape == dihedral_energy.shape
    assert triangle_energy.shape == gravity_energy.shape
    losses = {
        "triangle": (
            reduce(triangle_energy)
            if not only_collision
            else 0.0 * reduce(triangle_energy)
        ),
        "dihedral": (
            reduce(dihedral_energy)
            if not only_collision
            else 0.0 * reduce(dihedral_energy)
        ),
        "gravity": (
            reduce(gravity_energy)
            if not only_collision
            else 0.0 * reduce(gravity_energy)
        ),
        "body_collision": reduce(body_collision_energy),
    }

    total_loss = sum(losses.values())

    return total_loss, {key: value.detach() for key, value in losses.items()}

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Tuple

import geo
import torch
from energy.dihedral import compute_common, NORMALIZE_EPS, threshold
from torch import autograd


def compute_bend_stiffness(
    material: dict, uv: torch.Tensor | None, dihedralTopo: geo.DihedralTopo
):
    if uv is None:
        return material["bendStiffnessT"]

    bendingCoeffT = material["bendStiffnessT"]
    bendingCoeffP = material["bendStiffnessP"]

    def uv_blend(duv: torch.Tensor) -> torch.Tensor:
        duv0_sq = torch.square(duv[:, 0])
        duv1_sq = torch.square(duv[:, 1])
        denom = threshold(duv0_sq + duv1_sq, NORMALIZE_EPS)
        return (duv0_sq * bendingCoeffP + duv1_sq * bendingCoeffT) / denom

    duv_f0 = (
        uv[dihedralTopo.faces[:, 0], dihedralTopo.face_verts[:, 0, 0]]
        - uv[dihedralTopo.faces[:, 0], dihedralTopo.face_verts[:, 0, 1]]
    )
    duv_f1 = (
        uv[dihedralTopo.faces[:, 1], dihedralTopo.face_verts[:, 1, 0]]
        - uv[dihedralTopo.faces[:, 1], dihedralTopo.face_verts[:, 1, 1]]
    )
    return 0.5 * (uv_blend(duv_f0) + uv_blend(duv_f1))


def compute_energy(
    x: torch.Tensor,
    bendStiffness: torch.Tensor,
    rest_dihedral_shape: torch.Tensor,
    f: torch.Tensor,
    dihedralTopo: geo.DihedralTopo,
):
    """
    Computes the damped dihedral elastic bend energy
    Args:
        x: vertex positions
        f: per-face vertex indices
        uv: per-face per-vertex uv coords, |F| x 3 x 2 tensor
        dihedralTopo: dihedral topology
        material: dictionary containing the key "bendStiffnessT"
    """
    energy, _ = DihedralOrtho.apply(
        x, bendStiffness, rest_dihedral_shape, f, dihedralTopo
    )

    return energy


class DihedralOrtho(autograd.Function):
    @staticmethod
    def forward(
        x: torch.Tensor,
        bendStiffness: torch.Tensor,
        rest_dihedral_shape: torch.Tensor,
        f: torch.Tensor,
        dihedralTopo: geo.DihedralTopo,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x0, _x1, an0, an1, e_unit, el, a0, a1 = compute_common(x, f, dihedralTopo)
        cx1 = x[..., dihedralTopo.comp_verts[:, 1], :]
        e01 = x0 - cx1
        # Compute dihedral angle between faces
        cos = torch.linalg.vecdot(an0, an1)
        sin = torch.linalg.vecdot(an0, e01) * el

        # theta = torch.atan2(sin, cos)  # Can use this with the other sin version
        theta = torch.atan2(sin, cos)
        return (
            0.5
            * torch.sum(
                rest_dihedral_shape * bendStiffness * torch.square(theta), dim=-1
            ),
            theta,
        )

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        x, stiffness, rest_shape, f, topo = inputs
        _, theta = outputs

        ctx.save_for_backward(x, stiffness, theta)

        # Dont need to propagate grads for these
        ctx.f = f
        ctx.rest_shape = rest_shape
        ctx.topo = topo

        # Optimize gradient initialization when not needed
        ctx.set_materialize_grads(False)

    @staticmethod
    def backward(ctx, grad_output, grad_theta):
        # Skip if input is None (needed when set_materialize_grad is set to False)
        if grad_output is None:
            return None, None, None, None, None
        grad_x = grad_stiffness = grad_rest_shape = grad_f = grad_topo = None

        x, stiffness, theta = ctx.saved_tensors

        if ctx.needs_input_grad[0]:
            # TODO: optimize this function to reuse elements from forward call
            x0, _x1, an0, an1, e_unit, el, a0, a1 = compute_common(x, ctx.f, ctx.topo)

            cx0 = x[..., ctx.topo.comp_verts[:, 0], :]
            cx1 = x[..., ctx.topo.comp_verts[:, 1], :]
            e00 = x0 - cx0
            e01 = x0 - cx1

            n0a0 = an0 / (4.0 * threshold(a0**2, NORMALIZE_EPS).unsqueeze(-1))
            n1a1 = an1 / (4.0 * threshold(a1**2, NORMALIZE_EPS).unsqueeze(-1))
            el = el.unsqueeze(-1)

            mult = (stiffness * theta * ctx.rest_shape).unsqueeze(-1)

            gradC_comp_verts0 = mult * el * n0a0
            gradC_comp_verts1 = mult * el * n1a1
            gradC_verts1 = mult * (
                torch.linalg.vecdot(e00, e_unit).unsqueeze(-1) * n0a0
                + torch.linalg.vecdot(e01, e_unit).unsqueeze(-1) * n1a1
            )
            gradC_verts0 = -gradC_comp_verts0 - gradC_comp_verts1 - gradC_verts1

            # Angle gradient
            grad = torch.zeros_like(x, device=x.device)
            grad = grad.index_add(-2, ctx.topo.comp_verts[:, 0], gradC_comp_verts0)
            grad = grad.index_add(-2, ctx.topo.comp_verts[:, 1], gradC_comp_verts1)
            grad = grad.index_add(-2, ctx.topo.verts[:, 0], gradC_verts0)
            grad = grad.index_add(-2, ctx.topo.verts[:, 1], gradC_verts1)

            grad_x = -grad * grad_output.unsqueeze(-1).unsqueeze(-1)
        if ctx.needs_input_grad[1]:
            grad_stiffness = 0.5 * torch.square(theta) * ctx.rest_shape * grad_output

        return (grad_x, grad_stiffness, grad_rest_shape, grad_f, grad_topo)


class TestDihedralOrtho(unittest.TestCase):
    def test_simple(self):
        v = torch.tensor(
            [[-0.5, 0.0, -0.5], [0.35, -0.1, -0.35], [-0.5, 0.0, 0.5], [0.5, 0.0, 0.5]],
            dtype=torch.float64,
        )
        f = torch.tensor([[0, 1, 3], [0, 3, 2]])
        rest_pos = v[f].reshape((-1, 3))
        uv = torch.tensor(
            [
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]],
                [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            ],
            dtype=torch.float64,
        )

        rest_areas = geo.face_areas(
            rest_pos, torch.reshape(torch.arange(0, rest_pos.shape[0]), (-1, 3))
        )
        topo = geo.DihedralTopo(f)
        bendStiffness = compute_bend_stiffness(
            {"bendStiffnessT": 1.0, "bendStiffnessP": 0.5}, uv, topo
        )
        rest_shape = geo.compute_dihedral_rest_shape(rest_pos, rest_areas, topo)

        v.requires_grad_()
        bendStiffness.requires_grad_()

        # Energy grad
        torch.autograd.gradcheck(
            lambda x, b: torch.log(compute_energy(x, b, rest_shape, f, topo)),
            (v, bendStiffness),
            eps=1e-5,
            atol=1e-4,
        )

    def test_sphere_drape(self):
        import trimesh

        mesh = trimesh.load("./assets/sphere_drape.obj")
        v = torch.tensor(mesh.vertices, dtype=torch.float64)
        f = torch.tensor(mesh.faces, dtype=torch.int32)
        rest_pos = v[f].reshape((-1, 3))
        texCoords = torch.tensor(mesh.visual.uv, dtype=torch.float64)
        uv = texCoords[f].reshape(len(f), 3, -1)
        rest_areas = geo.face_areas(
            rest_pos, torch.reshape(torch.arange(0, rest_pos.shape[0]), (-1, 3))
        )
        topo = geo.DihedralTopo(f)
        bendStiffness = compute_bend_stiffness(
            {"bendStiffnessT": 2, "bendStiffnessP": 4}, uv, topo
        )
        rest_shape = geo.compute_dihedral_rest_shape(rest_pos, rest_areas, topo)

        v.requires_grad_()
        bendStiffness.requires_grad_()

        # Energy grad
        torch.autograd.gradcheck(
            lambda x, b: torch.log(compute_energy(x, b, rest_shape, f, topo)),
            (v, bendStiffness),
            eps=1e-5,
            atol=1e-4,
        )


if __name__ == "__main__":
    unittest.main()

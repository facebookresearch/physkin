# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import geo
import torch


def stvkStiffness(material):
    EP = material["youngsModulusP"]
    ET = material["youngsModulusT"]
    nu = material["poissonRatioTP"]
    mu = material["shearModulusTP"]

    denom = ET - nu * nu * EP
    common = ET / denom
    CT = ET * common
    CP = EP * common
    CTP = nu * CP
    return (CT, CP, CTP, mu)


def compute_energy(pos, faces, restShapeInv, restAreas, material):
    """
    Computes damped stretch energy of the cloth for the vertex positions v
    Material model: orthotropic Saint-Venant–Kirchhoff (StVK)
    """

    Dx = geo.compute_shape(pos, faces)

    F = torch.einsum("...fab,fbc->...fac", Dx, restShapeInv.to(Dx.dtype))

    Id = torch.eye(2, device=pos.device).expand(
        pos.shape[:-2] + (faces.shape[0], -1, -1)
    )
    E = 0.5 * (torch.matmul(torch.transpose(F, -2, -1), F) - Id)  # Green strain
    CT, CP, CTP, mu = stvkStiffness(material)

    faceEnergy = restAreas * (
        CT * E[..., 0, 0] ** 2
        + 2 * CTP * E[..., 0, 0] * E[..., 1, 1]
        + CP * E[..., 1, 1] ** 2
        + 4 * mu * E[..., 0, 1] ** 2
    )
    return 0.5 * torch.sum(faceEnergy, dim=-1), faceEnergy

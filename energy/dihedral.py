# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import geo
import torch

NORMALIZE_EPS = 1e-7


def threshold(x, tol):
    return torch.nn.functional.threshold(x, tol, tol)


def compute_common(x: torch.Tensor, f: torch.Tensor, dihedralTopo: geo.DihedralTopo):
    # Compute face normals
    fn, area = geo.face_normals_and_areas(x, f, normalize=False)

    an0 = fn[..., dihedralTopo.faces[:, 0], :]
    an1 = fn[..., dihedralTopo.faces[:, 1], :]

    # Compute edge length
    x0 = x[..., dihedralTopo.verts[:, 0], :]
    x1 = x[..., dihedralTopo.verts[:, 1], :]
    e = x1 - x0

    e_norm = torch.linalg.vector_norm(e, dim=-1)
    el = threshold(e_norm, NORMALIZE_EPS)
    e_unit = e / el.unsqueeze(-1)

    # Compute area
    a0, a1 = geo.compute_dihedral_areas(area, dihedralTopo)

    return x0, x1, an0, an1, e_unit, el, a0, a1

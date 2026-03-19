# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


def normalize_vector(v, return_mag=False):
    v_mag = torch.norm(v, p=2, dim=2, keepdim=True)
    v_normalized = torch.nn.functional.normalize(v, p=2, dim=2)

    if return_mag:
        return v_normalized, v_mag.squeeze(-1)
    else:
        return v_normalized


def cross_product(u, v):
    # u, v: (num_bones, batch, 3)
    i = u[..., 1] * v[..., 2] - u[..., 2] * v[..., 1]
    j = u[..., 2] * v[..., 0] - u[..., 0] * v[..., 2]
    k = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    out = torch.stack((i, j, k), dim=-1)  # (num_bones, batch, 3)
    return out


def compute_rotation_matrix_from_ortho6d(ortho6d):
    # ortho6d: (num_bones, batch, 6)
    x_raw = ortho6d[..., 0:3]  # (num_bones, batch, 3)
    y_raw = ortho6d[..., 3:6]  # (num_bones, batch, 3)
    x = normalize_vector(x_raw)  # (num_bones, batch, 3)
    z = cross_product(x, y_raw)  # (num_bones, batch, 3)
    z = normalize_vector(z)  # (num_bones, batch, 3)
    y = cross_product(z, x)  # (num_bones, batch, 3)
    x = x.unsqueeze(-1)  # (num_bones, batch, 3, 1)
    y = y.unsqueeze(-1)
    z = z.unsqueeze(-1)
    matrix3x3 = torch.cat((x, y, z), dim=3)  # (num_bones, batch, 3, 3)
    return matrix3x3


def compute_vectors_from_rotation_matrix(matrix):
    rotation_matrix = matrix[..., :3, :3]  # (..., 3, 3)
    translation_vector = matrix[..., :3, 3]  # (..., 3)
    ortho6d = rotation_matrix[..., :2].mT.reshape(
        *rotation_matrix.shape[:-2], 6
    )  # (..., 6)
    return ortho6d, translation_vector

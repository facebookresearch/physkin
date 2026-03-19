# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.backends import opt_einsum

opt_einsum.enabled = False


def LBS_positions_batched(input, rest, weights, bone_affine_transform_inv):
    """
    Non-jagged batched version of linear blend skinning using einsum for faster execution.

    Args:
        input:      Tensor of shape (B, M, 4, 4) - bone transforms
        rest:       Tensor of shape (B, N, 4) - homogenized positions
        weights:    Tensor of shape (B, M, N) - skinning weights
        bone_affine_transform_inv: Tensor of shape (B, M, 4, 4) - inverse of bind pose transforms
    Returns:
        results:    Tensor of shape (B, N, 3) - blended positions per batch
    """
    # Transform rest positions by base inverse transforms
    # (B, M, 4, 4) @ (B, N, 4) -> (B, M, N, 4)
    rest_local = torch.einsum("bmij,bnj->bmni", bone_affine_transform_inv, rest)

    # Apply skinning transforms
    # (B, M, 4, 4) @ (B, M, N, 4) -> (B, M, N, 4)
    transformed = torch.einsum("bmij,bmnj->bmni", input, rest_local)

    # Apply skinning weights and sum over bones dimension (M)
    # (B, M, N) * (B, M, N, 4) -> (B, N, 4)
    weighted_sum = torch.einsum("bmn,bmni->bni", weights, transformed)

    # Extract the 3D positions (discard homogeneous coordinate)
    return weighted_sum[:, :, :3]  # (B, N, 3)


def LBS_transforms_batched(input, rest, weights, bone_affine_transform_inv):
    """
    Non-jagged batched version of linear blend skinning using einsum for faster execution.

    Args:
        input:      Tensor of shape (B, M, 4, 4) - bone transforms
        rest:       Tensor of shape (B, N, 4, 4) - rest transforms
        weights:    Tensor of shape (B, M, N) - skinning weights
        bone_affine_transform_inv: Tensor of shape (B, M, 4, 4) - inverse of bind pose transforms
    Returns:
        results:    Tensor of shape (B, N, 4, 4) - blended transforms per batch
    """
    B, M, N = weights.shape

    # Reshape and broadcast tensors for parallel processing across all bones

    # Reshape rest transforms for broadcasting: (B, N, 4, 4) -> (B, 1, N, 4, 4)
    rest_expanded = rest.unsqueeze(1)  # (B, 1, N, 4, 4)

    # Reshape inverse bind pose for broadcasting: (B, M, 4, 4) -> (B, M, 1, 4, 4)
    inv_bind_expanded = bone_affine_transform_inv.unsqueeze(2)  # (B, M, 1, 4, 4)

    local_transforms = torch.matmul(
        inv_bind_expanded.expand(B, M, N, 4, 4),  # (B, M, N, 4, 4)
        rest_expanded.expand(B, M, N, 4, 4),  # (B, M, N, 4, 4)
    )

    # Reshape bone transforms for broadcasting: (B, M, 4, 4) -> (B, M, 1, 4, 4)
    bone_transforms_expanded = input.unsqueeze(2)  # (B, M, 1, 4, 4)

    # Apply bone transforms to local transforms for all bones and vertices at once
    # (B, M, 1, 4, 4) @ (B, M, N, 4, 4) -> (B, M, N, 4, 4)
    transformed = torch.matmul(
        bone_transforms_expanded,
        local_transforms,  # (B, M, N, 4, 4)  # (B, M, N, 4, 4)
    )

    # Apply skinning weights to transformed matrices
    # (B, M, N) -> (B, M, N, 1, 1)
    weights_expanded = weights.unsqueeze(-1).unsqueeze(-1)  # (B, M, N, 1, 1)

    # Apply weights: (B, M, N, 1, 1) * (B, M, N, 4, 4) -> (B, M, N, 4, 4)
    weighted_transforms = weights_expanded * transformed

    # Sum over bones dimension (M) to get final blended transforms: (B, M, N, 4, 4) -> (B, N, 4, 4)
    result = weighted_transforms.sum(dim=1)

    return result

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy

import numpy as np


def gaussian_rbf(distances, min_dist):
    """
    Compute weights using a Gaussian RBF.
    Returns:
         [num_vert_garment, num_vert_body]
    """
    # Compute the weights using the Gaussian RBF formula
    sigma = min_dist + 1e-7
    k = 0.25
    weights = np.exp(
        -((distances - np.expand_dims(min_dist, axis=1)) ** 2)
        / np.expand_dims(k * sigma**2, axis=1)
    )
    # Normalize the weights
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    return weights


def transfer_weights_via_rbf(sourceV, sourceW, targetV, source_mask=None):
    sourceV_copy = copy.deepcopy(sourceV)
    sourceW_copy = copy.deepcopy(sourceW)
    targetV_copy = copy.deepcopy(targetV)
    # distance
    S2T_distance = np.sqrt(
        np.sum(
            (sourceV_copy[:, np.newaxis, :] - targetV_copy[np.newaxis, :, :]) ** 2,
            axis=2,
        )
    )
    min_dist = np.min(S2T_distance, axis=1)
    dist_weights = gaussian_rbf(S2T_distance, min_dist)
    if source_mask is not None:
        source_mask = ~source_mask
        sourceW_copy[source_mask] = (
            np.ones((source_mask.sum(), sourceV_copy.shape[0])) * 1e-7
        )
    w_skinning = np.dot(sourceW_copy, dist_weights)
    w_skinning_normalized = w_skinning / np.sum(w_skinning, axis=0, keepdims=True)
    # normalize distance weights to return them for later use
    dist_weights_normalized = dist_weights / np.sum(dist_weights, axis=1, keepdims=True)
    return w_skinning_normalized.T, dist_weights_normalized


def copy_weights_via_knn(sourceV, sourceW, targetV, nn=1):
    sourceV_copy = copy.deepcopy(sourceV)
    sourceW_copy = copy.deepcopy(sourceW)
    targetV_copy = copy.deepcopy(targetV)
    from sklearn.neighbors import NearestNeighbors

    # Fit Nearest Neighbors model
    nbrs = NearestNeighbors(n_neighbors=nn, algorithm="auto").fit(sourceV_copy)
    distances, indices = nbrs.kneighbors(targetV_copy)
    if nn == 1:
        # copy weights from nearest neighbors
        copied_weights = sourceW_copy[indices.flatten()]
        return copied_weights, indices.flatten()
    else:
        # blend weights from nearest neighbors
        blend_wts = 1.0 / ((distances) ** 2 + 1e-7)
        blend_wts = np.expand_dims(blend_wts, axis=-1)
        copied_weights = sourceW_copy[indices] * blend_wts
        copied_weights = np.sum(copied_weights, axis=1) / np.sum(blend_wts, axis=1)
        return copied_weights, indices

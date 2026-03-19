# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from config import Config


class LBS(nn.Module):
    """LBS helper functions"""

    def __init__(self, config: Config):
        super(LBS, self).__init__()
        self.config = config

    @classmethod
    def from_scene_and_config(cls, data: dict, config: dict):
        return cls(Config(data, config))

    def input_to_affine(self, input):
        if type(input) is dict and "transform" in input and "pos" in input:
            pos = input["pos"]
            transform = input["transform"]
            assert len(pos) == len(transform)
            assert len(pos.shape) >= 2
            assert len(transform.shape) >= 3
            assert pos.shape[-1] == 3
            assert transform.shape[-1] == 3 and transform.shape[-2] == 3
            affine = torch.zeros(pos.shape[:-1] + (4, 4), dtype=torch.float32)
            affine[..., 0:3, 0:3] = torch.tensor(transform, dtype=torch.float32)
            affine[..., 0:3, 3] = torch.tensor(pos, dtype=torch.float32)
            affine[..., 3, 3] = 1.0
        elif (
            type(input) is torch.Tensor
            and len(input.shape) == 3
            and input.shape[1] == 4
            and input.shape[2] == 4
        ):
            affine = input
        elif type(input) is torch.Tensor and input.shape[-1] == 3 * 4 * len(
            self.config.rig.boneAffineTransform
        ):
            n_bones = len(self.config.rig.boneAffineTransform)
            affine = torch.zeros(
                input.shape[:-1] + (n_bones, 4, 4),
                dtype=torch.float32,
                device=input.device,
            )
            affine[..., 0:3, 0:4] = input.view(-1, n_bones, 3, 4)
            affine[..., 3, 3] = 1.0
        else:
            if type(input) is dict:
                raise Exception(f"Invalid input: {input.keys()}")
            elif type(input) is torch.Tensor:
                raise Exception(f"Invalid input shape: {input.shape}")
            else:
                raise Exception("Invalid input")
        return affine

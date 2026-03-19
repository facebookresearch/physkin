# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torch import autograd


def compute_energy(
    x: torch.Tensor, mass: torch.Tensor, g: torch.Tensor
) -> torch.Tensor:
    return Gravity.apply(x, mass.to(x.dtype), g.to(x.dtype))


class Gravity(autograd.Function):
    @staticmethod
    def forward(x: torch.Tensor, mass: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...bc,c,b->...", x, -g, mass)

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        _, mass, g = inputs

        # Dont need to propagate grads for these
        ctx.mass = mass
        ctx.g = g

        # Optimize gradient initialization when not needed
        ctx.set_materialize_grads(False)

    @staticmethod
    def backward(ctx, grad_output):
        # Skip if input is None (needed when set_materialize_grad is set to False)
        if grad_output is None:
            return None, None, None
        grad_x = grad_mass = grad_g = None

        if ctx.needs_input_grad[0]:
            grad_x = torch.einsum("...,c,b->...bc", grad_output, -ctx.g, ctx.mass)

        return (grad_x, grad_mass, grad_g)


class TestGravity(unittest.TestCase):
    def test_batched(self):
        v1 = (
            torch.tensor(
                [
                    [0.5, -0.5, 0.5],
                    [-0.5, -0.5, 0.5],
                    [0.5, 0.5, 0.5],
                    [-0.5, 0.5, 0.5],
                    [-0.5, -0.5, -0.5],
                    [0.5, -0.5, -0.5],
                    [-0.5, 0.5, -0.5],
                    [0.5, 0.5, -0.5],
                ],
                dtype=torch.float64,
            )
            + 0.1
        )
        v2 = v1 + 1.0
        v = torch.stack((v1, v2))
        mass = torch.ones(len(v1), dtype=torch.float64)
        g = torch.tensor([0.0, -981.0, 0.0], dtype=torch.float64)
        v.requires_grad_()
        energy_autograd = torch.log(compute_energy(v, mass, g))
        self.assertTrue(abs(energy_autograd[0] - 6.6654) < 1e-3)
        self.assertTrue(abs(energy_autograd[1] - 9.0633) < 1e-3)
        # check gradients
        torch.autograd.gradcheck(
            lambda x: torch.log(compute_energy(x, mass, g)), v, eps=1e-5, atol=1e-4
        )


if __name__ == "__main__":
    unittest.main()

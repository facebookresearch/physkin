# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn


class Conv1dMLP(nn.Module):
    """
    MLP implementation using Conv1d layers with kernel_size=1 instead of Linear layers.
    This is functionally equivalent to the original MLP but can be more efficient in some cases.
    """

    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        softmax=True,
        channel_transpose=True,
    ):
        super().__init__()
        # Input layer: Linear(in_features, hidden_features) -> Conv1d(in_features, hidden_features, kernel_size=1)
        self.input_layer = nn.Conv1d(
            in_features, hidden_features, kernel_size=1, bias=True
        )
        self.channel_transpose = channel_transpose

        # Hidden layers
        hidden = []
        for _i in range(hidden_layers):
            hidden.append(nn.ReLU())
            hidden.append(
                nn.Conv1d(hidden_features, hidden_features, kernel_size=1, bias=True)
            )
        hidden.append(nn.ReLU())
        self.hidden = nn.Sequential(*hidden)

        # Output layer: Linear(hidden_features, out_features) -> Conv1d(hidden_features, out_features, kernel_size=1)
        self.output_layer = nn.Conv1d(
            hidden_features, out_features, kernel_size=1, bias=True
        )

        # Optional softmax
        if softmax:
            self.softmax = nn.Softmax(
                dim=1
            )  # Apply softmax along the channel dimension
        else:
            self.softmax = None

    def forward(self, coords):
        # Reshape input: [batch_size, in_features] -> [batch_size, in_features, 1]
        if coords.dim() == 2:
            coords = coords.unsqueeze(-1)

        if self.channel_transpose:
            coords = coords.transpose(-1, -2)

        # Forward pass through the network
        l1 = self.input_layer(coords)
        l2 = self.hidden(l1)
        output = self.output_layer(l2)

        # Reshape output: [batch_size, out_features, 1] -> [batch_size, out_features]
        output = output.squeeze(-1)

        # Apply softmax if needed
        if self.softmax is not None:
            output = self.softmax(output)

        if self.channel_transpose:
            output = output.transpose(1, 2)

        return output

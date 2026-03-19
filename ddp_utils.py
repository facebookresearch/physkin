# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import os
from collections import OrderedDict

import torch


def convert_state_dict(state_dict, ddp_enabled):
    if ddp_enabled:
        return state_dict
    else:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[name] = v
        return new_state_dict


def setup_ddp():
    rank = int(os.environ["RANK"])
    device_id = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    timeout_value = datetime.timedelta(hours=12)
    torch.distributed.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, timeout=timeout_value
    )
    torch.cuda.set_device(device_id)
    return device_id, rank, world_size


def get_ddp_model(model, device_id, gradient_as_bucket_view=False, static_graph=False):
    model = model.to(device_id)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[device_id],
        find_unused_parameters=True,
        gradient_as_bucket_view=gradient_as_bucket_view,
        static_graph=static_graph,
    )
    return model

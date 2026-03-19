# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from torchtnt.framework.callback import Callback
from torchtnt.framework.state import State
from torchtnt.framework.unit import TTrainUnit
from torchtnt.utils.distributed import get_global_rank
from torchtnt.utils.tqdm import close_progress_bar
from tqdm.auto import tqdm


class TQDMProgressBarDDP(Callback):
    def __init__(self, num_steps_per_epoch, refresh_rate: int = 1):
        self._refresh_rate = refresh_rate
        self._num_steps_per_epoch = num_steps_per_epoch
        self._train_progress_bar: Optional[tqdm] = None

    def on_train_start(self, state: State, unit: TTrainUnit) -> None:
        if get_global_rank() == 0:
            epochs = state.train_state.max_epochs
            total_steps = epochs * self._num_steps_per_epoch
            self._train_progress_bar = tqdm(
                desc="Train",
                total=total_steps,
                initial=unit.train_progress.num_steps_completed,
                bar_format="{desc}: {percentage:3.1f}%|{bar}| {n:.0f}/{total_fmt} [{elapsed}<{remaining}]",
            )

    def on_train_step_end(self, state: State, unit: TTrainUnit) -> None:
        pbar = self._train_progress_bar
        if pbar is not None:
            if unit.train_progress.num_steps_completed % self._refresh_rate == 0:
                pbar.update(self._refresh_rate)

    def on_train_end(self, state: State, unit: TTrainUnit) -> None:
        pbar = self._train_progress_bar
        if pbar is not None:
            close_progress_bar(
                pbar,
                unit.train_progress.num_steps_completed,
                self._refresh_rate,
            )

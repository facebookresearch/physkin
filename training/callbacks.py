# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from typing import Any, Dict, Iterable, Optional

import torch
import torch.distributed as dist
from torchtnt.framework.callbacks.base_checkpointer import BaseCheckpointer
from torchtnt.framework.callbacks.checkpointer_types import KnobOptions, RestoreOptions
from torchtnt.framework.state import State
from torchtnt.framework.unit import AppStateMixin, TTrainData
from torchtnt.utils import loggers
from torchtnt.utils.checkpoint import BestCheckpointConfig


class Checkpoint(BaseCheckpointer):
    """
    A callback which creates a checkpoint for the model state_dict using torch.save.
    """

    metadata_fname: str = ".checkpoint_metadata"

    def __init__(
        self,
        dirpath: str,
        *,
        save_every_n_train_steps: Optional[int] = None,
        save_every_n_epochs: Optional[int] = None,
        save_every_n_eval_epochs: Optional[int] = None,
        keep_last_n_checkpoints: Optional[int] = None,
        best_checkpoint_config: Optional[BestCheckpointConfig] = None,
        process_group: Optional[dist.ProcessGroup] = None,
    ) -> None:
        super().__init__(
            dirpath=dirpath,
            save_every_n_train_steps=save_every_n_train_steps,
            save_every_n_epochs=save_every_n_epochs,
            save_every_n_eval_epochs=save_every_n_eval_epochs,
            keep_last_n_checkpoints=keep_last_n_checkpoints,
            best_checkpoint_config=best_checkpoint_config,
            process_group=process_group,
        )

    @staticmethod
    def restore_if(
        unit: AppStateMixin,
        continue_from_last_path: Optional[str] = None,
        init_model_path: Optional[str] = None,
    ) -> None:
        if init_model_path is not None:
            Checkpoint.restore(
                init_model_path,
                unit,
                restore_options=RestoreOptions(
                    restore_optimizers=False,
                    restore_lr_schedulers=False,
                    restore_eval_progress=False,
                    restore_train_progress=False,
                    strict=True,
                ),
            )
        elif continue_from_last_path is not None:
            Checkpoint.restore_from_latest(
                continue_from_last_path,
                unit,
            )

    def save_checkpoint(self, unit: AppStateMixin, checkpoint_path: str) -> bool:
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        # Write metadata file to describe the module structure for reference
        metadata_path = os.path.join(checkpoint_path, self.metadata_fname)
        with open(metadata_path, "w") as f:
            f.write(str(unit.tracked_modules()))

        success = True
        for name, module in unit.tracked_modules().items():
            file_path = os.path.join(checkpoint_path, f"{name}.pth")
            if os.path.exists(file_path):
                success = False
            torch.save(module.state_dict(), file_path)

        for name, progress in unit.tracked_progress().items():
            file_path = os.path.join(checkpoint_path, f"{name}.json")
            if os.path.exists(file_path):
                success = False
            with open(file_path, "w") as f:
                json.dump(progress.state_dict(), f)

        return success

    def _checkpoint_impl(
        self,
        state: State,
        unit: AppStateMixin,
        *,
        checkpoint_path: str,
        hook: str,
    ) -> bool:
        """
        Checkpoint the current state of the application.
        """
        if hook not in [
            "on_train_step_end",
            "on_train_epoch_end",
            "on_train_end",
            "on_eval_epoch_end",
        ]:
            raise RuntimeError(f"Unexpected hook encountered '{hook}'")
        return self.save_checkpoint(unit, checkpoint_path)

    @staticmethod
    def restore(
        path: str,
        unit: AppStateMixin,
        *,
        train_dataloader: Optional[Iterable[TTrainData]] = None,
        process_group: Optional[dist.ProcessGroup] = None,
        restore_options: Optional[RestoreOptions] = None,
        storage_options: Optional[Dict[str, Any]] = None,
        knob_options: Optional[KnobOptions] = None,
        strict: bool = True,
    ) -> None:
        """Restore checkpoint"""
        metadata_path = os.path.join(path, ".checkpoint_metadata")
        with open(metadata_path, "r") as f:
            metadata = f.read()
            if metadata != str(unit.tracked_modules()):
                raise RuntimeError(
                    f"Cannot restore from checkpoint since modules don't match:\ncur:\n{str(unit.tracked_modules())}\nfound:\n{metadata}\n"
                )

        for name, module in unit.tracked_modules().items():
            module.load_state_dict(
                torch.load(
                    os.path.join(path, f"{name}.pth"),
                    map_location="cpu",
                    weights_only=True,
                ),
                strict=strict,
            )

        for name, progress in unit.tracked_progress().items():
            file_path = os.path.join(path, f"{name}.json")
            if not os.path.exists(file_path):
                raise RuntimeError(
                    f"Cannot restore progress from checkpoint, missing file: {file_path}"
                )
            with open(file_path, "r") as f:
                progress.load_state_dict(json.load(f))


# Override the original tensorboard logger with one that puts hparams alongside all other logs without an additional subdir.
class TensorBoardLogger(loggers.TensorBoardLogger):
    def __init__(self, path: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(path, *args, **kwargs)

    def log_hparams(
        self, hparams: Dict[str, loggers.Scalar], metrics: Dict[str, loggers.Scalar]
    ) -> None:
        """Add hyperparameter data to TensorBoard.

        Args:
            hparams (dict): dictionary of hyperparameter names and corresponding values
            metrics (dict): dictionary of name of metric and corresponding values
        """

        if self._writer:
            self._writer.add_hparams(hparams, metrics, run_name=".")

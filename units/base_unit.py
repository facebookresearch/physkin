# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import trimesh
from line_profiler import profile
from models.mlp import PositionalEncoding
from torchtnt.framework.auto_unit import TrainStepResults
from torchtnt.framework.state import State
from torchtnt.framework.unit import TrainUnit
from torchtnt.utils import TLRScheduler


class BasePhysRigUnit(TrainUnit[dict]):
    def _init_common(
        self,
        module,
        pose_modulator,
        lbs,
        body_bones_weights,
        default_body_rest_verts,
        prediction_type,
        vertex_offset_type,
        model_input_type,
        log_every_n_steps,
        tb_logger,
        device,
        rank,
        debug_path,
        ckpt_path,
        hparams,
        custom_cloth_bones,
        dataset_type,
        dataset,
        input_batch_size,
        checkpoint_every_n_epochs,
        optimize_skinning_weights,
        optimize_skinning_weights_start_epoch,
        use_vertex_model,
        use_graph_transformer,
        joint_training,
        bone_encoder,
        hypermodulator,
        vertex_model_start_epoch,
        material_change,
        material_change_start_epoch,
        debug_train_data,
        ddp_enabled,
        skinwt_spread,
        bones_lod_levels,
        custom_collate_fn,
        save_meshes_info,
        predict_shape_deltas,
        gradient_clipping,
        hypermodulator_start_epoch,
        freeze_garment_encoder,
        bone_module_inp_type,
        drape_mode,
    ):
        super().__init__()
        self.device = device
        self.rank = rank
        self.ddp_enabled = ddp_enabled
        self.input_batch_size = input_batch_size
        self.module = module.to(self.device)
        self.pose_modulator = pose_modulator.to(self.device)
        self.lbs = lbs.to(self.device)
        self.skeleton = self.lbs.config.rig.rig.skeleton.to(self.device)
        self.full_joint_state = self.skeleton.skeleton_states_to_joints(
            self.lbs.config.rig.rig.skeleton_state.squeeze().to(self.device)
        )
        self.custom_collate_fn = custom_collate_fn
        self.dataset_type = dataset_type
        self.dataset = dataset
        self.custom_cloth_bones = custom_cloth_bones
        self.prediction_type = prediction_type
        self.vertex_offset_type = vertex_offset_type
        self.tb_logger = tb_logger
        self.log_every_n_steps = log_every_n_steps
        self.debug_train_data = debug_train_data
        self.debug_path = debug_path
        self.ckpt_path = ckpt_path
        self.checkpoint_every_n_epochs = checkpoint_every_n_epochs
        self.lap_loss_wt = 0.0
        self.optimize_skinning_weights = optimize_skinning_weights
        self.optimize_skinning_weights_start_epoch = (
            optimize_skinning_weights_start_epoch
        )
        self.vertex_model_start_epoch = vertex_model_start_epoch
        self.use_vertex_model = use_vertex_model
        self.skinwt_spread = skinwt_spread
        self.save_meshes_info = save_meshes_info
        self.model_input_type = model_input_type
        self.bone_module_inp_type = bone_module_inp_type
        self.default_body_rest_verts = default_body_rest_verts.unsqueeze(0).to(
            self.device
        )
        self.joint_training = joint_training
        self.bones_lod_levels = bones_lod_levels
        self.material_change = material_change
        self.material_change_start_epoch = material_change_start_epoch
        self.hypermodulator_start_epoch = hypermodulator_start_epoch
        self.freeze_garment_encoder = freeze_garment_encoder

        self.predict_shape_deltas = predict_shape_deltas > 0
        self.shape_specific_verts_skinwts = predict_shape_deltas >= 1
        self.shape_specific_verts_deltas = predict_shape_deltas == 2
        print(
            f"Deltas: {self.predict_shape_deltas} ---> skinwts: {self.shape_specific_verts_skinwts} | rest_verts: {self.shape_specific_verts_deltas}"
        )

        self.drape_mode = drape_mode
        self.hard_pinning_verts = False

        if bone_encoder is not None:
            self.bone_encoder = bone_encoder.to(self.device)

        self.hypermodulator = hypermodulator.to(self.device)
        self.use_graph_transformer = use_graph_transformer

        # positional encoding
        self.fourier_enc = PositionalEncoding()

        # Default hyperparameters
        self.hparams = {"lr": 1e-4, "rig_phys_loss_weight": 1.0}
        if hparams is not None:
            self.hparams |= hparams
        if self.tb_logger is not None and hparams is not None:
            self.tb_logger.log_hparams(hparams, {})

        # Debug dict
        self.debug_dict = {}

        self.enable_gradient_clipping = gradient_clipping
        self.run_eval = True
        self.export_merged_mesh_only = True

        opt, scheduler = self.configure_optimizers_and_lr_scheduler(module)
        self.optimizer = opt
        self.lr_scheduler = scheduler
        self.device = "cpu" if device is None else device

    def print_module_grad(self, module, grad_input, grad_output):
        module_grad_info = f"""
        ------ RANK_{self.rank} -----
        Module: {module}")
        grad_input: {grad_input}
        grad_output: {grad_output}
        """
        print(module_grad_info)

    def check_grad_sync(self, model, print_first_k=0):
        """
        checks if gradients are in sync or not
        (we can't simply use backward hook since it is called before gradient synchronization)
        """
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Clone the gradient tensor
                grad = param.grad.detach().clone()
                # All-reduce to sum gradients across all ranks
                torch.distributed.all_reduce(grad, op=torch.distributed.ReduceOp.SUM)
                # Compute average
                grad /= torch.distributed.get_world_size()
                # Compare local grad to average
                diff = (param.grad - grad).abs().max().item()
                if torch.distributed.get_rank() == 0:
                    print(f"[Grad Sync Check] {name}: max abs diff = {round(diff, 7)}")
                if print_first_k > 0:
                    # gradient comparison across ranks
                    grad_to_compare = (
                        param.grad.detach().cpu().flatten()[:print_first_k]
                    )
                    print(
                        f"Rank {torch.distributed.get_rank()} | World-Size {torch.distributed.get_world_size()} | {name} grad (first 5): {grad_to_compare}"
                    )
            torch.distributed.barrier()

    def on_train_end(self, state: State):
        print(
            f" CUDA:{self.device} memory allocated: ",
            torch.cuda.max_memory_allocated(self.device) / 1024**3,
            " GB",
        )

    def on_train_epoch_end(self, state: State):
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def configure_optimizers_and_lr_scheduler(
        self, module: nn.Module
    ) -> Tuple[torch.optim.Optimizer, Optional[TLRScheduler]]:  # type: ignore
        self.parameters_opt = [
            *module.parameters(),
            *self.pose_modulator.parameters(),
            *self.hypermodulator.parameters(),
        ]
        optimizer = torch.optim.AdamW(
            params=self.parameters_opt,
            lr=self.hparams["lr"],
            eps=1e-8,
            betas=(0.9, 0.999),
            weight_decay=0.01,  # Prevents parameter explosion
            amsgrad=True,  # Can set to True for more stability
            foreach=True,
        )
        return optimizer, None

    @profile
    def train_step(self, state: State, data: torch.Tensor) -> None:
        # hook for subclass-specific pre-train logic (e.g. inference check)
        self._pre_train_step(state, data)

        start_time = time.time()

        # forward pass
        loss, outputs = self.compute_loss(state, data)

        # backward pass
        loss.backward()

        # gradient clipping
        if self.enable_gradient_clipping:
            torch.nn.utils.clip_grad_norm_(self.parameters_opt, max_norm=1.0)

        # gradient sync check
        if self.ddp_enabled and self.debug_grad_sync:
            torch.distributed.barrier()
            module_to_check = self.hypermodulator.module.mesh_embedder
            self.check_grad_sync(module_to_check)

        # optimizer update
        self.optimizer.step()
        self.optimizer.zero_grad()

        end_time = time.time()

        # debug train info
        if self.rank == 0:
            step_end_msg = f"""
            EPOCH: {self.train_progress.num_epochs_completed} | STEP: {self.train_progress.num_steps_completed} | TIME (seconds): {round(end_time - start_time, 3)} | LOSS : {loss.item()}
            """
            print(step_end_msg)

        # end current train step
        results = TrainStepResults(loss, None, outputs)
        step_count = self.train_progress.num_steps_completed
        self.on_train_step_end(state, data, step_count, results)

    def _pre_train_step(self, state, data):
        """Override in subclass for pre-train-step logic."""
        pass

    def _save_eval_meshes(self, state, prefix, sample_info, save_dir):
        """Save evaluation meshes for a single sample.

        Args:
            state: training state
            prefix: 'train', 'val', or 'infer_train'/'infer_test' etc.
            sample_info: dict with 'sample_data', 'sample_idx', 'garment_idx', 'shape_idx', 'pose_idx'
            save_dir: directory to save meshes to
        """
        sample_data = sample_info["sample_data"]
        sample_idx = sample_info["sample_idx"]
        sample_garment_idx = sample_info["garment_idx"]
        sample_shape_idx = sample_info["shape_idx"]
        sample_pose_idx = sample_info["pose_idx"]
        no_prediction = sample_info.get("no_prediction", False)

        with torch.no_grad():
            if no_prediction:
                sample_outputs = self.compute_loss(
                    state, sample_data, eval_step=True, no_prediction=True
                )
            else:
                sample_outputs = self.compute_loss(state, sample_data, eval_step=True)

        verts_c = sample_outputs["pos_c"].squeeze(0).cpu().numpy()
        verts_b = sample_outputs["pos_b"].squeeze(0).cpu().numpy()
        garment_scene_dict = sample_outputs["garment_lbs"].config.scene.scene
        faces_c = garment_scene_dict["faces"]
        faces_b = garment_scene_dict["bodyFaces"]
        colors_c = np.array([[0, 150, 150, 255]] * len(verts_c))
        colors_b = np.array([[120, 120, 120, 255]] * len(verts_b))
        mesh_c = trimesh.Trimesh(verts_c, faces_c, vertex_colors=colors_c)
        mesh_b = trimesh.Trimesh(verts_b, faces_b, vertex_colors=colors_b)
        mesh_merged = trimesh.util.concatenate([mesh_c, mesh_b])
        os.makedirs(save_dir, exist_ok=True)
        if self.export_merged_mesh_only is False:
            mesh_c.export(
                f"{save_dir}/{prefix}_cloth_{sample_idx}_garment{sample_garment_idx}_shape{sample_shape_idx}_pose{sample_pose_idx}.obj"
            )
            mesh_b.export(
                f"{save_dir}/{prefix}_body_{sample_idx}_garment{sample_garment_idx}_shape{sample_shape_idx}_pose{sample_pose_idx}.obj"
            )
        mesh_merged.export(
            f"{save_dir}/{prefix}_merged_{sample_idx}_garment{sample_garment_idx}_shape{sample_shape_idx}_pose{sample_pose_idx}.obj"
        )

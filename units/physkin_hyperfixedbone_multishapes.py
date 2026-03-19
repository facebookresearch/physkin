# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gc
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from base_unit import BasePhysRigUnit
from lbs import LBS
from line_profiler import profile
from loss.physics import physics_loss_from_state
from rotation_utils import (
    compute_rotation_matrix_from_ortho6d,
    compute_vectors_from_rotation_matrix,
)
from torchtnt.framework.auto_unit import TrainStepResults
from torchtnt.framework.state import State
from training.callbacks import TensorBoardLogger
from unit_utils.lbs_batched import LBS_positions_batched, LBS_transforms_batched


class PhysRigUnit(BasePhysRigUnit):
    def __init__(
        self,
        module: nn.Module,
        pose_modulator: nn.Module,
        lbs: LBS,
        body_bones_weights: torch.Tensor,
        default_body_rest_verts: torch.Tensor,
        prediction_type="delta_posed_ortho6d",
        vertex_offset_type="post_lbs",
        model_input_type="pose_vector",
        log_every_n_steps: int = 10,
        tb_logger: Optional[TensorBoardLogger] = None,
        device: Optional[torch.device] = None,
        rank: int | None = None,
        debug_path: str | None = None,
        ckpt_path: str | None = None,
        hparams: dict | None = None,
        custom_cloth_bones=True,
        dataset_type="shapes",
        dataset: Optional[torch.utils.data.Dataset] = None,
        input_batch_size=1,
        vertex_model=None,
        checkpoint_every_n_epochs=1,
        run_inference: bool = False,
        optimize_skinning_weights=False,
        optimize_skinning_weights_start_epoch=None,
        use_vertex_model=False,
        use_graph_transformer=True,
        joint_training=False,
        bone_encoder=None,
        hypermodulator=None,
        vertex_model_start_epoch=None,
        material_change=False,
        material_change_start_epoch=None,
        debug_train_data=False,
        ddp_enabled=False,
        skinwt_spread=1.0,
        bones_lod_levels: None = None,
        custom_collate_fn=None,
        save_meshes_info=None,
        predict_shape_deltas=0,
        gradient_clipping=False,
        hypermodulator_start_epoch=0,
        freeze_garment_encoder=False,
        bone_module_inp_type="transformations",
        drape_mode=None,
        add_noise=False,
        train_stage="stage_1",
        pose_interpolation_start=0,
        pose_interpolation_interval=1,
        cfg=None,
    ):
        if optimize_skinning_weights_start_epoch is None:
            optimize_skinning_weights_start_epoch = np.iinfo(np.int64).max
        if vertex_model_start_epoch is None:
            vertex_model_start_epoch = np.iinfo(np.int64).max
        if material_change_start_epoch is None:
            material_change_start_epoch = np.iinfo(np.int64).max
        self._init_common(
            module=module,
            pose_modulator=pose_modulator,
            lbs=lbs,
            body_bones_weights=body_bones_weights,
            default_body_rest_verts=default_body_rest_verts,
            prediction_type=prediction_type,
            vertex_offset_type=vertex_offset_type,
            model_input_type=model_input_type,
            log_every_n_steps=log_every_n_steps,
            tb_logger=tb_logger,
            device=device,
            rank=rank,
            debug_path=debug_path,
            ckpt_path=ckpt_path,
            hparams=hparams,
            custom_cloth_bones=custom_cloth_bones,
            dataset_type=dataset_type,
            dataset=dataset,
            input_batch_size=input_batch_size,
            checkpoint_every_n_epochs=checkpoint_every_n_epochs,
            optimize_skinning_weights=optimize_skinning_weights,
            optimize_skinning_weights_start_epoch=optimize_skinning_weights_start_epoch,
            use_vertex_model=use_vertex_model,
            use_graph_transformer=use_graph_transformer,
            joint_training=joint_training,
            bone_encoder=bone_encoder,
            hypermodulator=hypermodulator,
            vertex_model_start_epoch=vertex_model_start_epoch,
            material_change=material_change,
            material_change_start_epoch=material_change_start_epoch,
            debug_train_data=debug_train_data,
            ddp_enabled=ddp_enabled,
            skinwt_spread=skinwt_spread,
            bones_lod_levels=bones_lod_levels,
            custom_collate_fn=custom_collate_fn,
            save_meshes_info=save_meshes_info,
            predict_shape_deltas=predict_shape_deltas,
            gradient_clipping=gradient_clipping,
            hypermodulator_start_epoch=hypermodulator_start_epoch,
            freeze_garment_encoder=freeze_garment_encoder,
            bone_module_inp_type=bone_module_inp_type,
            drape_mode=drape_mode,
        )

        self.cfg = cfg
        self.run_inference = run_inference
        self.add_noise = add_noise

        self.pose_interpolation_start = pose_interpolation_start
        self.pose_interpolation_interval = pose_interpolation_interval
        self.alpha_pose = 0.0
        self.rest_pose_params = torch.zeros(1, 120, device=self.device)
        self.rest_affine_state = self.lbs.input_to_affine(self.rest_pose_params)

        self.hypermodulator.train_stage = train_stage
        self.debug_grad_sync = False

        ## hooks for gradient check
        if self.debug_grad_sync:
            if self.ddp_enabled:
                hooked_module = self.hypermodulator.module
            else:
                hooked_module = self.hypermodulator
            hooked_module.garment_embedder.register_full_backward_hook(
                self.print_module_grad
            )

        self.dataset.fixed_garment_index = np.random.randint(
            0, self.dataset.num_train_garment_samples
        )

    def _pre_train_step(self, state, data):
        """Run inference if required before training step."""
        if self.run_inference:
            outputs = {}
            results = TrainStepResults(0.0, None, outputs)
            step_count = self.train_progress.num_steps_completed
            self.on_train_step_end(state, data, step_count, results)

    @profile
    def on_train_step_end(  # noqa: C901
        self,
        state: State,
        data,
        step: int,
        results: TrainStepResults,
    ) -> None:
        if self.rank == 0 or self.ddp_enabled is False:
            # logging
            loss, outputs = results.loss, results.outputs
            if (step + 1) % self.log_every_n_steps == 0 and self.tb_logger is not None:
                self.tb_logger.log_dict(
                    {f"train/{k}": v for k, v in outputs["losses"].items()}, step
                )
                self.tb_logger.log("train/total_train_loss", loss, step)

        # run evaluation
        if self.run_eval and (
            self.train_progress.num_epochs_completed == 0
            or (
                (self.train_progress.num_epochs_completed + 1)
                % self.checkpoint_every_n_epochs
            )
            == 0
        ):
            # enable deterministic mode
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)

            # switch to eval mode
            self.hypermodulator.eval()
            self.pose_modulator.eval()
            self.module.eval()

            if self.run_inference:
                print("----Inference (Train Samples)----")
                max_save_samples_num = len(
                    self.save_meshes_info["infer_train"]["garment_indices"]
                )
                for sample_idx in range(max_save_samples_num):
                    sample_pose_idx = self.save_meshes_info["infer_train"][
                        "pose_indices"
                    ][sample_idx]
                    sample_shape_idx = self.save_meshes_info["infer_train"][
                        "shape_indices"
                    ][sample_idx]
                    sample_garment_idx = self.save_meshes_info["infer_train"][
                        "garment_indices"
                    ][sample_idx]
                    print(
                        f"Pose: {sample_pose_idx} | Shape: {sample_shape_idx} | Garment: {sample_garment_idx}"
                    )
                    self.dataset.fixed_garment_index = sample_garment_idx
                    sample = self.dataset.__getitem__(
                        idx=sample_pose_idx,
                        shape_idx=sample_shape_idx,
                        garment_idx=None,
                        window_start_idx="last",
                        mode="eval",
                    )
                    sample_data = self.custom_collate_fn([sample])
                    save_dir = os.path.join(self.ckpt_path, "meshes", "inference_train")
                    self._save_eval_meshes(
                        state,
                        "infer",
                        {
                            "sample_data": sample_data,
                            "sample_idx": sample_idx,
                            "garment_idx": sample_garment_idx,
                            "shape_idx": sample_shape_idx,
                            "pose_idx": sample_pose_idx,
                        },
                        save_dir,
                    )
                # test samples
                print("----Inference (Test Samples)----")
                max_save_samples_num = len(
                    self.save_meshes_info["infer_test"]["garment_indices"]
                )
                for sample_idx in range(max_save_samples_num):
                    sample_pose_idx = self.save_meshes_info["infer_test"][
                        "pose_indices"
                    ][sample_idx]
                    sample_shape_idx = self.save_meshes_info["infer_test"][
                        "shape_indices"
                    ][sample_idx]
                    sample_garment_idx = self.save_meshes_info["infer_test"][
                        "garment_indices"
                    ][sample_idx]
                    print(
                        f"Pose: {sample_pose_idx} | Shape: {sample_shape_idx} | Garment: {sample_garment_idx}"
                    )
                    self.dataset.fixed_garment_index = sample_garment_idx
                    sample = self.dataset.__getitem__(
                        idx=sample_pose_idx,
                        shape_idx=sample_shape_idx,
                        garment_idx=None,
                        window_start_idx="last",
                        mode="eval",
                    )
                    sample_data = self.custom_collate_fn([sample])
                    save_dir = os.path.join(self.ckpt_path, "meshes", "inference_test")
                    self._save_eval_meshes(
                        state,
                        "infer",
                        {
                            "sample_data": sample_data,
                            "sample_idx": sample_idx,
                            "garment_idx": sample_garment_idx,
                            "shape_idx": sample_shape_idx,
                            "pose_idx": sample_pose_idx,
                        },
                        save_dir,
                    )
                print("Inference Done! Exiting...")
                exit()

            ### save train meshes
            print("Saving Train Samples...")
            max_save_samples_num = len(
                self.save_meshes_info["train"]["garment_indices"]
            )
            sample_idx = self.rank % max_save_samples_num
            sample_pose_idx = self.save_meshes_info["train"]["pose_indices"][sample_idx]
            sample_shape_idx = self.save_meshes_info["train"]["shape_indices"][
                sample_idx
            ]
            sample_garment_idx = self.save_meshes_info["train"]["garment_indices"][
                sample_idx
            ]
            self.dataset.fixed_garment_index = (
                sample_garment_idx  # assign garment index before sampling
            )
            sample = self.dataset.__getitem__(
                idx=sample_pose_idx,
                shape_idx=sample_shape_idx,
                garment_idx=None,
                window_start_idx="last",
                mode="eval",
            )
            sample_data = self.custom_collate_fn([sample])
            is_epoch_zero = self.train_progress.num_epochs_completed == 0
            save_dir = os.path.join(
                self.ckpt_path,
                "meshes",
                f"epoch_{self.train_progress.num_epochs_completed}",
            )
            self._save_eval_meshes(
                state,
                "train",
                {
                    "sample_data": sample_data,
                    "sample_idx": sample_idx,
                    "garment_idx": sample_garment_idx,
                    "shape_idx": sample_shape_idx,
                    "pose_idx": sample_pose_idx,
                    "no_prediction": is_epoch_zero,
                },
                save_dir,
            )
            self.dataset.fixed_garment_index = -1  # reset

            ### save test meshes
            print("Saving Test Samples...")
            max_save_samples_num = len(self.save_meshes_info["test"]["garment_indices"])
            sample_idx = self.rank % max_save_samples_num
            sample_pose_idx = self.save_meshes_info["test"]["pose_indices"][sample_idx]
            sample_shape_idx = self.save_meshes_info["test"]["shape_indices"][
                sample_idx
            ]
            sample_garment_idx = self.save_meshes_info["test"]["garment_indices"][
                sample_idx
            ]
            self.dataset.fixed_garment_index = (
                sample_garment_idx  # assign garment index before sampling
            )
            sample = self.dataset.__getitem__(
                idx=sample_pose_idx,
                shape_idx=sample_shape_idx,
                garment_idx=None,
                window_start_idx="last",
                mode="eval",
            )
            sample_data = self.custom_collate_fn([sample])
            self._save_eval_meshes(
                state,
                "val",
                {
                    "sample_data": sample_data,
                    "sample_idx": sample_idx,
                    "garment_idx": sample_garment_idx,
                    "shape_idx": sample_shape_idx,
                    "pose_idx": sample_pose_idx,
                    "no_prediction": is_epoch_zero,
                },
                save_dir,
            )
            self.dataset.fixed_garment_index = -1  # reset

            # switch back to train mode
            self.hypermodulator.train()
            self.pose_modulator.train()
            self.module.train()

            # disable deterministic mode
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.use_deterministic_algorithms(False)

        if self.debug_train_data and not self.ddp_enabled:
            debug_data = {}
            garment_lbs = outputs["garment_lbs"]
            garment_scene = garment_lbs.config.scene.scene
            active_garment_bones = garment_scene["activeBones"]
            debug_data["cloth_pos"] = outputs["pos_c"].cpu().detach()
            debug_data["body_pos"] = outputs["pos_b"].cpu().detach()
            debug_data["affine_states"] = (
                outputs["affine_states"][:, active_garment_bones, :, :].cpu().detach()
            )
            debug_data["cloth_faces"] = garment_scene["faces"]
            debug_data["body_faces"] = garment_scene["bodyFaces"]
            debug_data["cloth_uv"] = garment_scene["texCoords"]
            self.debug_dict[f"{step}"] = debug_data

        # clean-up
        gc.collect()
        torch.cuda.empty_cache()

    def _get_num_bone_net_layers(self):
        """Count the number of bone network layers, handling DDP wrapping."""
        if self.ddp_enabled:
            module = self.module.module
        else:
            module = self.module
        if "ModulatedMLP" in str(module.__class__):
            return len(
                [layer for layer in module.hidden if isinstance(layer, nn.Linear)]
            )
        else:  # siren
            return len(module.layers)

    def _select_garment_id(self):
        """Select garment ID, randomizing if not fixed."""
        garment_id = self.dataset.fixed_garment_index
        if garment_id == -1:
            torch.manual_seed(
                np.random.randint(100000) + self.rank
            )  # ensures different random numbers across DDP devices
            garment_id = torch.randint(
                self.dataset.num_train_garment_samples, (1,)
            ).item()
        return garment_id

    def _load_garment_data(self, garment_id):
        """Load garment graph data and set instance attributes for the selected garment."""
        self.garment_lbs = self.dataset.garment_lbs[garment_id].to(self.device)
        garment_graph_data = self.dataset.garment_graph_data[garment_id].to(self.device)
        self.active_bones = (
            garment_graph_data.active_bones.bool().unsqueeze(0).to(self.device)
        )
        self.num_active_bones = int(self.active_bones.sum())
        self.all_bones_rest_positions = (
            garment_graph_data.all_bones_rest_positions.unsqueeze(0).to(self.device)
        )
        self.cloth_bones_rest_positions = (
            self.all_bones_rest_positions[self.active_bones]
            .unsqueeze(0)
            .to(self.device)
        )
        self.all_bones_rest_transforms_3x3 = (
            garment_graph_data.all_bones_rest_transforms.unsqueeze(0).to(self.device)
        )
        self.all_bones_weights = garment_graph_data.cloth_bones_weights.unsqueeze(0).to(
            self.device
        )
        self.cloth_verts_skinning_weights = (
            garment_graph_data.verts_skinning_weights.T.unsqueeze(0).to(self.device)
        )
        self.verts_pinning_mask = garment_graph_data.verts_pinning_mask.unsqueeze(0).to(
            self.device
        )
        self.bones_pinning_mask = garment_graph_data.bones_pinning_mask.unsqueeze(0).to(
            self.device
        )
        self.pinned_bones_bodyverts_idx = (
            garment_graph_data.pinned_bones_bodyverts_idx.unsqueeze(0).to(self.device)
        )
        self.cloth_verts_bone_selection_mask = (
            garment_graph_data.vertex_bone_selection_mask.unsqueeze(0).to(self.device)
        )
        self.garment_verts_xyz = (
            garment_graph_data.verts[:, :3].unsqueeze(0).to(self.device)
        )
        self.garment_verts_to_body = (
            garment_graph_data.garment_verts_body_nnidx.long()
            .unsqueeze(0)
            .to(self.device)
        )
        self.garment_bones_to_body = (
            garment_graph_data.garment_bones_body_nnidx.long()
            .unsqueeze(0)
            .to(self.device)
        )
        self.cloth_vertex_bone_selection_mask = (
            garment_graph_data.vertex_bone_selection_mask.unsqueeze(0).to(self.device)
        )
        self.body_verts_weights = garment_graph_data.body_verts_weights.unsqueeze(0).to(
            self.device
        )
        return garment_graph_data

    def _prepare_affine_state(
        self,
        batched_motion_sequences,
        batched_motion_affine_states,
        batch_size,
        motion_window_size,
    ):
        """Prepare the motion affine state, appending identity transforms for cloth bones."""
        # Transform to full affine state
        if batched_motion_sequences.sum() == -1 * batch_size:
            print("No motion data available!")
        else:
            motion_identity_transforms = (
                torch.eye(4)
                .repeat(batch_size, motion_window_size, self.num_active_bones, 1, 1)
                .reshape(batch_size, motion_window_size, -1, 4, 4)
                .to(self.device)
            )
            batched_motion_affine_states = torch.cat(
                [batched_motion_affine_states, motion_identity_transforms], dim=2
            )

        # motion state
        affine_state = batched_motion_affine_states.squeeze(
            1
        )  # removing window dimension
        return affine_state, batched_motion_affine_states

    def _apply_pose_interpolation(self, poses, affine_state, batch_size):
        """Apply pose interpolation from rest pose when alpha_pose < 1.0."""
        if self.alpha_pose >= 1.0:
            return poses, affine_state

        # interpolation scheduling
        if (
            self.train_progress.num_epochs_completed % self.pose_interpolation_interval
            == 0
        ) and (
            self.train_progress.num_epochs_completed >= self.pose_interpolation_start
        ):
            # switch to 'stage_2'
            self.hypermodulator.train_stage = "stage_2"
            self.alpha_pose += 0.1
            print(f"Changing pose interpolation factor to --- {self.alpha_pose}")
        # interpolate
        rest_poses = self.rest_pose_params.repeat(batch_size, 1)
        poses = self.alpha_pose * poses + (1 - self.alpha_pose) * rest_poses
        spt_affine_state = self.lbs.input_to_affine(poses)
        num_spt_bones_ = spt_affine_state.shape[1]
        affine_state[:, :num_spt_bones_, :, :] = spt_affine_state
        return poses, affine_state

    def _compute_body_deformations(
        self,
        batched_body_verts_rest_positions,
        batched_motion_affine_states,
        cloth_bones_rest_transforms,
        batch_size,
    ):
        """Compute deformed body vertices and cloth bone transforms."""
        # skinning weights & transforms
        all_bones_rest_transforms = (
            torch.eye(4)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch_size, self.all_bones_rest_transforms_3x3.shape[1], 1, 1)
        )
        all_bones_rest_transforms[:, :, :3, :3] = self.all_bones_rest_transforms_3x3
        all_bones_rest_transforms[:, :, :3, 3] = self.all_bones_rest_positions
        all_bones_rest_transforms = all_bones_rest_transforms.to(self.device)
        all_bones_rest_transforms_inv = torch.linalg.inv(all_bones_rest_transforms)

        # estimate the deformed body vertices
        body_vertices_h = torch.cat(
            [
                batched_body_verts_rest_positions,
                torch.ones_like(batched_body_verts_rest_positions[..., :1]),
            ],
            dim=-1,
        )
        body_vertices_deformed = LBS_positions_batched(
            batched_motion_affine_states.squeeze(1),
            body_vertices_h,
            self.body_verts_weights,
            all_bones_rest_transforms_inv,
        )

        # garment bones skinning
        deformed_cloth_bones_transforms = LBS_transforms_batched(
            batched_motion_affine_states.squeeze(1),
            cloth_bones_rest_transforms.repeat(batch_size, 1, 1, 1),
            self.all_bones_weights.repeat(batch_size, 1, 1),
            all_bones_rest_transforms_inv,
        )
        deformed_cloth_state = deformed_cloth_bones_transforms[:, :, :3, :]
        if self.drape_mode == "lbs":
            _batch_indices = torch.arange(
                batch_size, device=deformed_cloth_state.device
            ).unsqueeze(1)
            _selected_vertices = body_vertices_deformed[
                _batch_indices, self.garment_bones_to_body.repeat(batch_size, 1)
            ]
            deformed_cloth_state[:, :, :3, 3] = _selected_vertices

        return body_vertices_deformed, deformed_cloth_state

    def _compute_rest_cloth_state_and_skinned(
        self,
        batched_body_verts_rest_positions,
        batched_garment_rest_positions,
        deformed_cloth_state,
        verts_skinning_wts,
        cloth_bones_rest_transforms_inv,
        batch_size,
    ):
        """Compute rest cloth state and skinned vertex positions based only on shape."""
        # skinned vertices based only on shape
        _batch_indices = torch.arange(
            batch_size, device=deformed_cloth_state.device
        ).unsqueeze(1)
        _selected_rest_vertices = batched_body_verts_rest_positions[
            _batch_indices, self.garment_bones_to_body.repeat(batch_size, 1)
        ]
        rest_cloth_state = torch.eye(4, device=self.device).repeat(
            batch_size, self.num_active_bones, 1, 1
        )
        rest_cloth_state[:, :, :3, 3] = _selected_rest_vertices
        rest_cloth_state = rest_cloth_state.detach()
        garment_rest_positions_h = torch.cat(
            [
                batched_garment_rest_positions,
                torch.ones_like(batched_garment_rest_positions[..., :1]),
            ],
            dim=-1,
        )
        pos_rest_skinned = LBS_positions_batched(
            rest_cloth_state,
            garment_rest_positions_h,
            verts_skinning_wts,
            cloth_bones_rest_transforms_inv,
        )
        return rest_cloth_state, pos_rest_skinned

    def _compute_bone_features(
        self,
        rest_cloth_state,
        deformed_cloth_state,
        bones_hyperencodings,
    ):
        """Compute bone network input features based on bone_module_inp_type."""
        rest_cloth_bones = compute_vectors_from_rotation_matrix(rest_cloth_state)
        deformed_cloth_bones = compute_vectors_from_rotation_matrix(
            deformed_cloth_state
        )
        bones_rest_9d = torch.cat(rest_cloth_bones, dim=-1)
        bones_deformed_9d = torch.cat(deformed_cloth_bones, dim=-1)
        if self.bone_module_inp_type == "transformations":
            return torch.cat((bones_rest_9d, bones_deformed_9d), dim=-1)
        elif self.bone_module_inp_type == "hypertransformations":
            return torch.cat(
                (bones_rest_9d, bones_deformed_9d, bones_hyperencodings), dim=-1
            )
        elif self.bone_module_inp_type == "hyperencodings":
            return bones_hyperencodings

    def _predict_deformations_stage2(
        self,
        batched_pattern_embeddings,
        batched_shape_latents,
        garment_graph_data,
        pos_rest_skinned,
        rest_cloth_state,
        deformed_cloth_state,
        poses,
        num_bone_net_layers,
        batch_size,
        cfg,
    ):
        """Stage 2 prediction: compute modulations and bone deformations."""
        # predict modulations
        (
            verts_hyperdeltas,
            bones_hyperdeltas,
            bones_hyperencodings,
            hypermodulations,
        ) = self.hypermodulator(
            batched_pattern_embeddings,
            batched_shape_latents,
            garment_graph_data,
            drape_cond_feat=pos_rest_skinned,
        )

        # bone network input features
        bones_feats_inp = self._compute_bone_features(
            rest_cloth_state,
            deformed_cloth_state,
            bones_hyperencodings,
        )
        bones_feats_inp = bones_feats_inp.view(batch_size, -1)

        hyper_mods, hyper_shifts = (
            hypermodulations[:, :, : hypermodulations.shape[2] // 2],
            hypermodulations[:, :, hypermodulations.shape[2] // 2 :],
        )

        if cfg.train.canonical_pose_only:
            # predict bone deformations
            predicted_deformations = self.module(bones_feats_inp).view(
                batch_size, -1, 9
            )
        else:
            # predict pose modulations
            pose_modulations = self.pose_modulator(poses)
            pose_modulations = pose_modulations.view(
                batch_size, num_bone_net_layers, -1
            )
            pose_mods, pose_shifts = (
                pose_modulations[:, :, : pose_modulations.shape[2] // 2],
                pose_modulations[:, :, pose_modulations.shape[2] // 2 :],
            )
            # predict bone deformations
            mods = hyper_mods * pose_mods
            shifts = hyper_shifts + pose_shifts
            predicted_deformations = self.module(bones_feats_inp, mods, shifts).view(
                batch_size, -1, 9
            )

        return verts_hyperdeltas, bones_hyperdeltas, predicted_deformations

    def _predict_deformations(
        self,
        batched_pattern_embeddings,
        batched_shape_latents,
        garment_graph_data,
        pos_rest_skinned,
        rest_cloth_state,
        deformed_cloth_state,
        poses,
        num_bone_net_layers,
        batch_size,
        cfg,
    ):
        """Run hypermodulator and bone network to predict deformations."""
        predicted_deformations = None

        if self.hypermodulator.train_stage == "stage_1":
            # predict shape deltas
            verts_hyperdeltas, bones_hyperdeltas, bones_hyperencodings = (
                self.hypermodulator(
                    batched_pattern_embeddings,
                    batched_shape_latents,
                    garment_graph_data,
                    drape_cond_feat=pos_rest_skinned,
                )
            )
        else:
            verts_hyperdeltas, bones_hyperdeltas, predicted_deformations = (
                self._predict_deformations_stage2(
                    batched_pattern_embeddings,
                    batched_shape_latents,
                    garment_graph_data,
                    pos_rest_skinned,
                    rest_cloth_state,
                    deformed_cloth_state,
                    poses,
                    num_bone_net_layers,
                    batch_size,
                    cfg,
                )
            )

        return verts_hyperdeltas, bones_hyperdeltas, predicted_deformations

    def _compute_pinned_bones(
        self,
        bones_pinning_mask,
        body_vertices_deformed,
        batch_size,
        motion_window_size,
    ):
        """Compute pinned bone positions and pinning index metadata."""
        # Vectorized batch indexing for pinned bones
        pinning_mask = 1 - bones_pinning_mask.unsqueeze(-1)
        pinning_index_start = int(
            pinning_mask.sum(1).max().item()
        )  # just a hack, assumes pinned bones are placed in the beginning, will be replaced with a more flexible indices estimation later
        pinned_bones_bodyverts_idx = self.pinned_bones_bodyverts_idx.repeat(
            batch_size, 1
        )
        pinned_bones_bodyverts = None
        if bones_pinning_mask.sum() > 0:
            num_pins = pinned_bones_bodyverts_idx.shape[1]
            batch_indices = (
                torch.arange(motion_window_size, device=body_vertices_deformed.device)
                .unsqueeze(1)
                .expand(-1, num_pins)
            )
            pinned_bones_bodyverts = body_vertices_deformed[
                batch_indices, pinned_bones_bodyverts_idx
            ]
        return pinning_index_start, pinned_bones_bodyverts

    def _apply_shape_deltas(
        self,
        verts_hyperdeltas,
        bones_hyperdeltas,
        verts_pinning_mask,
        predicted_deformations,
        batched_garment_rest_positions,
        verts_skinning_wts,
        no_prediction,
    ):
        """Apply shape-specific deltas to rest positions and skinning weights."""
        verts_xyz_deltas = None
        skinwts_delta = None

        # predicted shape specific deltas
        if self.predict_shape_deltas:
            shape_specific_deltas = verts_hyperdeltas * (
                1 - verts_pinning_mask
            ).unsqueeze(-1)  # handle pinning
            verts_xyz_deltas = shape_specific_deltas[:, :, :3]
            skinwts_delta = shape_specific_deltas[:, :, 3:]
            _cloth_bones_shape_deltas = bones_hyperdeltas[:, :, :3]  # noqa: F841

        # only for debugging / visualizing skinning intializations / triggered only on epoch-0
        if no_prediction:
            if predicted_deformations is not None:
                predicted_deformations = predicted_deformations * 0.0
            if self.predict_shape_deltas:
                verts_xyz_deltas = verts_xyz_deltas * 0.0
                skinwts_delta = skinwts_delta * 0.0

        # rest vertices
        pos_rest_xyz = batched_garment_rest_positions

        if self.predict_shape_deltas and self.shape_specific_verts_deltas:
            # #  # update rest vertices
            pos_rest_xyz = pos_rest_xyz + verts_xyz_deltas.clone()

        # verts skinning weights
        if self.predict_shape_deltas and self.shape_specific_verts_skinwts:
            eps = 1e-10
            skinwts_delta = skinwts_delta.transpose(1, 2)
            log_skinwts = torch.log(verts_skinning_wts + eps)
            log_skinwts = log_skinwts + skinwts_delta
            verts_skinning_wts = torch.softmax(log_skinwts, dim=1)

        return pos_rest_xyz, verts_skinning_wts, predicted_deformations

    def _apply_bone_deltas(
        self,
        predicted_deformations,
        deformed_cloth_state,
        affine_state,
        bones_pinning_mask,
        pinning_index_start,
        pinned_bones_bodyverts,
    ):
        """Apply bone delta predictions and pinning to the affine state."""
        # add bone deltas
        pred = predicted_deformations if predicted_deformations is not None else None
        if self.prediction_type == "delta_posed_ortho6d":
            r6d, t = compute_vectors_from_rotation_matrix(deformed_cloth_state)
            # add predicted deltas
            if pred is not None:
                pred_r6d = pred[:, :, :6]
                pred_t = pred[:, :, 6:]
                r6d = r6d + pred_r6d
                t = t + pred_t
            ## pinning to body (pinned bones are the last indices)
            if bones_pinning_mask.sum() > 0:
                t[:, pinning_index_start:, :] = pinned_bones_bodyverts
            pred_affine_states = torch.zeros(
                (r6d.shape[0], r6d.shape[1], 3, 4), device=self.device
            )
            pred_affine_states[:, :, :3, :3] = compute_rotation_matrix_from_ortho6d(r6d)
            pred_affine_states[:, :, :3, 3] = t
            affine_state[:, self.active_bones.squeeze(0), 0:3, :] = pred_affine_states
        return affine_state

    def _apply_vertex_pinning(
        self,
        pos,
        verts_pinning_mask,
        body_vertices_deformed,
        batch_size,
    ):
        """Pin garment vertices to the body where indicated by the pinning mask."""
        if self.hard_pinning_verts:
            g_to_b = self.garment_verts_to_body.repeat(batch_size, 1)
            _batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(1)
            pos_on_body = body_vertices_deformed[_batch_indices, g_to_b]
            pinmask = verts_pinning_mask.unsqueeze(-1)
            pos = (1 - pinmask) * pos + pinmask * pos_on_body
        return pos

    def _build_loss_outputs(
        self,
        affine_states,
        pos,
        body_vertices_deformed,
        verts_skinning_wts,
        eval_step,
    ):
        """Compute physics loss and build the output dictionary."""
        # calculate physics-based losses
        phys_loss, phys_losses = physics_loss_from_state(
            states=affine_states,
            vertex_positions=pos,
            body_vertex_positions=body_vertices_deformed,
            lbs=self.garment_lbs,
            contact_params=self.hparams,
            timings=None,
            cloth_weights=verts_skinning_wts,
            only_collision=False,
        )

        losses = {
            "phys": phys_loss,
        }
        loss_weights = {
            "phys": self.hparams["rig_phys_loss_weight"],
        }

        # IMPORTANT: This only works because losses and loss_weights have the same order.
        total_loss = sum(
            [a * b for a, b in zip(losses.values(), loss_weights.values())]
        )

        phys_losses_debug = {f"phys/{k}": v.detach() for k, v in phys_losses.items()}
        losses_debug = {
            key: value.detach() for key, value in losses.items()
        } | phys_losses_debug

        outputs = {"losses": losses_debug}

        if eval_step or self.debug_train_data:
            outputs["garment_lbs"] = self.garment_lbs
            outputs["pos_c"] = pos.detach()
            outputs["pos_b"] = body_vertices_deformed.detach()
            outputs["affine_states"] = affine_states.detach()

            if eval_step:
                return None, outputs

        return total_loss, outputs

    @profile
    def compute_loss(
        self,
        state: State,
        data: torch.Tensor,
        eval_step=False,
        no_prediction: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        cfg = self.cfg

        # load current batch data
        batched_body_verts_rest_positions = data[0].to(self.device)
        batched_shape_latents = data[1].to(self.device)
        batched_pattern_embeddings = data[2].to(self.device)
        _batched_shape_indices = data[3].to(self.device)  # noqa: F841
        _batched_garment_indices = data[4].to(self.device)  # noqa: F841
        batched_motion_sequences = data[5].to(self.device)
        batched_motion_affine_states = data[6].to(self.device)

        # batch info
        batch_size = len(batched_motion_sequences)
        motion_window_size = self.dataset.motion_window_size
        assert motion_window_size == 1, "Motion Window Size > 1 is not supported!"

        # pose vector
        poses = batched_motion_sequences.view(batch_size, -1)

        # for reshaping modulations
        num_bone_net_layers = self._get_num_bone_net_layers()

        # random (single) garment data per batch
        garment_id = self._select_garment_id()
        garment_graph_data = self._load_garment_data(garment_id)

        # cloth bones rest info
        cloth_bones_rest_transforms = torch.eye(4, device=self.device).repeat(
            1, self.num_active_bones, 1, 1
        )
        cloth_bones_rest_transforms[:, :, :3, 3] = self.cloth_bones_rest_positions
        cloth_bones_rest_transforms_inv = torch.linalg.inv(cloth_bones_rest_transforms)

        affine_state, batched_motion_affine_states = self._prepare_affine_state(
            batched_motion_sequences,
            batched_motion_affine_states,
            batch_size,
            motion_window_size,
        )

        # interpolation from rest pose
        poses, affine_state = self._apply_pose_interpolation(
            poses,
            affine_state,
            batch_size,
        )

        # BODY & GARMENT-BONES DEFORMATIONS

        # garment bone-verts skinning weights
        verts_skinning_wts = self.cloth_verts_skinning_weights.clone()

        # garment rest vertices
        batched_garment_rest_positions = self.garment_verts_xyz.repeat(batch_size, 1, 1)

        body_vertices_deformed, deformed_cloth_state = self._compute_body_deformations(
            batched_body_verts_rest_positions,
            batched_motion_affine_states,
            cloth_bones_rest_transforms,
            batch_size,
        )

        rest_cloth_state, pos_rest_skinned = self._compute_rest_cloth_state_and_skinned(
            batched_body_verts_rest_positions,
            batched_garment_rest_positions,
            deformed_cloth_state,
            verts_skinning_wts,
            cloth_bones_rest_transforms_inv,
            batch_size,
        )

        verts_hyperdeltas, bones_hyperdeltas, predicted_deformations = (
            self._predict_deformations(
                batched_pattern_embeddings,
                batched_shape_latents,
                garment_graph_data,
                pos_rest_skinned,
                rest_cloth_state,
                deformed_cloth_state,
                poses,
                num_bone_net_layers,
                batch_size,
                cfg,
            )
        )

        # pinning masks
        verts_pinning_mask = self.verts_pinning_mask
        bones_pinning_mask = self.bones_pinning_mask

        pinning_index_start, pinned_bones_bodyverts = self._compute_pinned_bones(
            bones_pinning_mask,
            body_vertices_deformed,
            batch_size,
            motion_window_size,
        )

        # apply shape deltas, no_prediction zeroing, and skinning weight updates
        pos_rest_xyz, verts_skinning_wts, predicted_deformations = (
            self._apply_shape_deltas(
                verts_hyperdeltas,
                bones_hyperdeltas,
                verts_pinning_mask,
                predicted_deformations,
                batched_garment_rest_positions,
                verts_skinning_wts,
                no_prediction,
            )
        )

        # skinned vertices before adding prediction
        pos_rest = torch.cat(
            [pos_rest_xyz, torch.ones_like(pos_rest_xyz[..., :1])], dim=-1
        )
        _pos_skinned = LBS_positions_batched(  # noqa: F841
            deformed_cloth_state,
            pos_rest,
            verts_skinning_wts,
            cloth_bones_rest_transforms_inv,
        )

        # add bone deltas
        affine_state = self._apply_bone_deltas(
            predicted_deformations,
            deformed_cloth_state,
            affine_state,
            bones_pinning_mask,
            pinning_index_start,
            pinned_bones_bodyverts,
        )

        # final affine state
        affine_states = affine_state

        # skinned vertices post prediction
        pos = LBS_positions_batched(
            affine_states[:, self.active_bones.squeeze(0), :, :],
            pos_rest,
            verts_skinning_wts,
            cloth_bones_rest_transforms_inv,
        )

        # handle pinning
        pos = self._apply_vertex_pinning(
            pos,
            verts_pinning_mask,
            body_vertices_deformed,
            batch_size,
        )

        # calculate losses and build outputs
        total_loss, outputs = self._build_loss_outputs(
            affine_states,
            pos,
            body_vertices_deformed,
            verts_skinning_wts,
            eval_step,
        )

        if total_loss is None:
            # eval_step early return
            return outputs

        return total_loss, outputs

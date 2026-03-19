# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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

        rest_pose_params = torch.zeros(1, 120, device=self.device)
        self.rest_affine_state = self.lbs.input_to_affine(rest_pose_params)

        self.debug_grad_sync = True

        ## hooks for gradient check
        if self.debug_grad_sync:
            if self.ddp_enabled:
                hooked_module = self.hypermodulator.module
            else:
                hooked_module = self.hypermodulator
            hooked_module.mesh_embedder.register_full_backward_hook(
                self.print_module_grad
            )
            for modulation_head in hooked_module.modulation_heads:
                modulation_head.register_full_backward_hook(self.print_module_grad)

    @profile
    def on_train_step_end(
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
        if (
            self.run_eval
            and self.train_progress.num_epochs_completed
            % self.checkpoint_every_n_epochs
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
            sample = self.dataset.__getitem__(
                idx=sample_pose_idx,
                shape_idx=sample_shape_idx,
                garment_idx=sample_garment_idx,
                window_start_idx="last",
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
            sample = self.dataset.__getitem__(
                idx=sample_pose_idx,
                shape_idx=sample_shape_idx,
                garment_idx=sample_garment_idx,
                window_start_idx="last",
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
            garment_lbs = outputs["garment_lbs"].to("cpu")
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

    def _get_num_bone_net_layers(self):
        """Return the number of bone network layers for reshaping modulations."""
        if self.ddp_enabled:
            module_inner = self.module.module
        else:
            module_inner = self.module
        if "ModulatedMLP" in str(module_inner.__class__):
            return len(
                [layer for layer in module_inner.hidden if isinstance(layer, nn.Linear)]
            )
        else:  # siren
            return len(module_inner.layers)

    def _build_graph_batch_data(
        self, graph_data, batch_size, batched_body_verts_positions
    ):
        """Extract per-sample graph data into lists for batch processing."""
        graph_active_bones = []
        graph_all_bones_rest_positions = []
        graph_cloth_bones_rest_positions = []
        graph_cloth_verts_rest_positions = []
        graph_all_bones_rest_transforms_3x3 = []
        graph_all_bones_weights = []
        graph_cloth_verts_skinning_weights = []
        graph_body_verts_weights = []
        graph_verts_pinning_mask = []
        graph_bones_pinning_mask = []
        graph_pinned_bones_bodyverts_idx = []
        graph_verts_bone_selection_mask = []
        graph_verts_xyz = []
        graph_garment_verts_to_body = []
        graph_garment_bones_to_body = []
        graph_garment_bones_onbody_positions = []
        graph_vertex_bone_selection_mask = []
        for batch_idx in range(batch_size):
            cloth_bones_mask = graph_data[batch_idx].active_bones.bool()
            graph_active_bones.append(cloth_bones_mask)
            graph_all_bones_rest_positions.append(
                graph_data[batch_idx].all_bones_rest_positions
            )
            graph_all_bones_rest_transforms_3x3.append(
                graph_data[batch_idx].all_bones_rest_transforms
            )
            graph_all_bones_weights.append(graph_data[batch_idx].cloth_bones_weights)
            graph_cloth_bones_rest_positions.append(
                graph_data[batch_idx].all_bones_rest_positions[cloth_bones_mask]
            )
            graph_cloth_verts_skinning_weights.append(
                graph_data[batch_idx].verts_skinning_weights
            )
            graph_body_verts_weights.append(graph_data[batch_idx].body_verts_weights)
            graph_verts_pinning_mask.append(graph_data[batch_idx].verts_pinning_mask)
            graph_bones_pinning_mask.append(graph_data[batch_idx].bones_pinning_mask)
            graph_pinned_bones_bodyverts_idx.append(
                graph_data[batch_idx].pinned_bones_bodyverts_idx
            )
            graph_verts_bone_selection_mask.append(
                graph_data[batch_idx].vertex_bone_selection_mask
            )
            graph_verts_xyz.append(graph_data[batch_idx].x[:, :3])
            # placing garment bones on the body
            garment2body_bonesidx = graph_data[
                batch_idx
            ].garment_bones_body_nnidx.long()
            cloth_bones_nnbodypos = batched_body_verts_positions[batch_idx][
                garment2body_bonesidx
            ]
            graph_garment_bones_to_body.append(garment2body_bonesidx)
            graph_garment_bones_onbody_positions.append(cloth_bones_nnbodypos)
            # update rest positions of garment vertices
            garment2body_vertsidx = graph_data[
                batch_idx
            ].garment_verts_body_nnidx.long()
            graph_garment_verts_to_body.append(garment2body_vertsidx)
            cloth_verts_nnbodypos = batched_body_verts_positions[batch_idx][
                garment2body_vertsidx
            ]
            graph_cloth_verts_rest_positions.append(cloth_verts_nnbodypos)
            graph_vertex_bone_selection_mask.append(
                graph_data[batch_idx].vertex_bone_selection_mask
            )

        return {
            "graph_active_bones": graph_active_bones,
            "graph_all_bones_rest_positions": graph_all_bones_rest_positions,
            "graph_cloth_bones_rest_positions": graph_cloth_bones_rest_positions,
            "graph_cloth_verts_rest_positions": graph_cloth_verts_rest_positions,
            "graph_all_bones_rest_transforms_3x3": graph_all_bones_rest_transforms_3x3,
            "graph_all_bones_weights": graph_all_bones_weights,
            "graph_cloth_verts_skinning_weights": graph_cloth_verts_skinning_weights,
            "graph_body_verts_weights": graph_body_verts_weights,
            "graph_verts_pinning_mask": graph_verts_pinning_mask,
            "graph_bones_pinning_mask": graph_bones_pinning_mask,
            "graph_pinned_bones_bodyverts_idx": graph_pinned_bones_bodyverts_idx,
            "graph_verts_bone_selection_mask": graph_verts_bone_selection_mask,
            "graph_verts_xyz": graph_verts_xyz,
            "graph_garment_verts_to_body": graph_garment_verts_to_body,
            "graph_garment_bones_to_body": graph_garment_bones_to_body,
            "graph_garment_bones_onbody_positions": graph_garment_bones_onbody_positions,
            "graph_vertex_bone_selection_mask": graph_vertex_bone_selection_mask,
        }

    def _compute_bones_rest_and_deformations(
        self,
        graph_batch_data,
        batch_size,
        motion_window_size,
        graph_data,
        batched_body_verts_positions,
        batched_motion_affine_states,
        batched_motion_sequences,
    ):
        """Compute bone rest transforms, body deformations, and garment bone skinning."""
        graph_cloth_bones_rest_positions = graph_batch_data[
            "graph_cloth_bones_rest_positions"
        ]
        graph_all_bones_rest_positions = graph_batch_data[
            "graph_all_bones_rest_positions"
        ]
        graph_all_bones_rest_transforms_3x3 = graph_batch_data[
            "graph_all_bones_rest_transforms_3x3"
        ]
        graph_all_bones_weights = graph_batch_data["graph_all_bones_weights"]
        graph_body_verts_weights = graph_batch_data["graph_body_verts_weights"]
        graph_garment_bones_to_body = graph_batch_data["graph_garment_bones_to_body"]
        graph_garment_bones_onbody_positions = graph_batch_data[
            "graph_garment_bones_onbody_positions"
        ]
        graph_active_bones = graph_batch_data["graph_active_bones"]

        # bones rest info
        batched_cloth_bones_rest_positions = torch.stack(
            graph_cloth_bones_rest_positions
        )
        batched_cloth_bones_rest_transforms = torch.eye(4).repeat(
            batched_cloth_bones_rest_positions.shape[0],
            batched_cloth_bones_rest_positions.shape[1],
            1,
            1,
        )
        batched_cloth_bones_rest_transforms = batched_cloth_bones_rest_transforms.to(
            self.device
        )
        batched_cloth_bones_rest_transforms[:, :, :3, 3] = (
            batched_cloth_bones_rest_positions
        )
        batched_cloth_bones_rest_transforms_inv = torch.linalg.inv(
            batched_cloth_bones_rest_transforms
        )
        batched_garment_bones_to_body = torch.stack(graph_garment_bones_to_body)
        _batched_garment_bones_onbody_positions = torch.stack(  # noqa: F841
            graph_garment_bones_onbody_positions
        )

        # Transform to full affine state
        if batched_motion_sequences.sum() == -1 * batch_size:
            print("No motion data available!")
        else:
            motion_identity_transforms = (
                torch.eye(4)
                .repeat(
                    batch_size,
                    motion_window_size,
                    int(graph_data[0].active_bones.sum()),
                    1,
                    1,
                )
                .reshape(batch_size, motion_window_size, -1, 4, 4)
                .to(self.device)
            )
            batched_motion_affine_states = torch.cat(
                [batched_motion_affine_states, motion_identity_transforms], dim=2
            )

        # BODY & GARMENT-BONES DEFORMATIONS

        # garment bones only
        active_garment_bones = graph_active_bones[
            0
        ].bool()  # since bones are fixed across the batch as of now

        # skinning weights & transforms
        all_bones_rest_positions = torch.stack(graph_all_bones_rest_positions)
        all_bones_rest_transforms_3x3 = torch.stack(graph_all_bones_rest_transforms_3x3)
        all_bones_rest_transforms = (
            torch.eye(4)
            .unsqueeze(0)
            .unsqueeze(0)
            .repeat(batch_size, all_bones_rest_transforms_3x3.shape[1], 1, 1)
        )
        all_bones_rest_transforms[:, :, :3, :3] = all_bones_rest_transforms_3x3
        all_bones_rest_transforms[:, :, :3, 3] = all_bones_rest_positions
        all_bones_rest_transforms = all_bones_rest_transforms.to(self.device)
        all_bones_rest_transforms_inv = torch.linalg.inv(all_bones_rest_transforms)
        all_bones_weights = torch.stack(graph_all_bones_weights)

        # estimate the deformed body vertices
        body_verts_weights = torch.stack(graph_body_verts_weights)
        body_vertices_h = torch.cat(
            [
                batched_body_verts_positions,
                torch.ones_like(batched_body_verts_positions[..., :1]),
            ],
            dim=-1,
        )
        batched_body_vertices_deformed = LBS_positions_batched(
            batched_motion_affine_states.squeeze(1),
            body_vertices_h,
            body_verts_weights,
            all_bones_rest_transforms_inv,
        )

        # garment bones skinning
        deformed_cloth_bones_transforms = LBS_transforms_batched(
            batched_motion_affine_states.squeeze(1),
            batched_cloth_bones_rest_transforms,
            all_bones_weights,
            all_bones_rest_transforms_inv,
        )
        deformed_cloth_state = deformed_cloth_bones_transforms[:, :, :3, :]
        if self.drape_mode == "lbs":
            _batch_indices = torch.arange(
                batch_size, device=deformed_cloth_state.device
            ).unsqueeze(1)
            _selected_vertices = batched_body_vertices_deformed[
                _batch_indices, batched_garment_bones_to_body
            ]
            deformed_cloth_state[:, :, :3, 3] = _selected_vertices

        return (
            batched_cloth_bones_rest_transforms_inv,
            batched_motion_affine_states,
            active_garment_bones,
            batched_body_vertices_deformed,
            deformed_cloth_state,
        )

    def _compute_garment_rest_positions(
        self,
        batch_size,
        graph_cloth_verts_rest_positions,
        graph_verts_xyz,
    ):
        """Compute rest positions for garment vertices per sample."""
        batched_pos_rest_xyz = []
        for batch_idx in range(batch_size):
            verts_xyz = graph_verts_xyz[batch_idx].unsqueeze(0)
            if self.drape_mode == "nn":
                pos_rest_xyz = graph_cloth_verts_rest_positions[batch_idx].unsqueeze(0)
            else:
                pos_rest_xyz = verts_xyz.clone()
            batched_pos_rest_xyz.append(pos_rest_xyz.squeeze(0))
        return batched_pos_rest_xyz

    def _predict_pose_dependent_deformations(
        self,
        batch_size,
        num_bone_net_layers,
        batched_motion_sequences,
        hypermodulations,
        mesh_verts_feats,
        mesh_emb,
        mesh_verts_delta,
        bones_hyperfeats,
        deformed_cloth_state,
    ):
        """Predict pose-dependent bone deformations using modulated networks."""
        poses = batched_motion_sequences.view(batch_size, -1)
        predicted_deformations = None
        if self.train_progress.num_epochs_completed >= self.hypermodulator_start_epoch:
            if self.use_graph_transformer and self.freeze_garment_encoder:
                mesh_verts_feats = mesh_verts_feats.detach()
                mesh_emb = mesh_emb.detach()
                mesh_verts_delta = mesh_verts_delta.detach()
            # predict modulations
            pose_modulations = self.pose_modulator(poses, hypermodulations)
            pose_modulations = pose_modulations.view(
                batch_size, num_bone_net_layers, -1
            )
            # bone network input features
            if self.bone_module_inp_type == "transformations":
                deformed_cloth_bones = compute_vectors_from_rotation_matrix(
                    deformed_cloth_state
                )
                bones_feats_inp = torch.cat(deformed_cloth_bones, -1).view(
                    batch_size, -1
                )
            elif self.bone_module_inp_type == "hyperfeatures":
                bones_feats_inp = bones_hyperfeats.view(batch_size, -1)
            # predict bone deformations
            predicted_deformations = self.module(
                bones_feats_inp, pose_modulations
            ).view(batch_size, -1, 9)

        return predicted_deformations, mesh_verts_delta

    def _apply_shape_deltas_to_skinning_weights(
        self,
        verts_skinning_wts,
        mesh_verts_delta,
        graph_batch_mask,
        verts_pinning_mask,
        graph_vertex_bone_selection_mask_item,
        no_prediction,
    ):
        """Apply predicted shape-specific deltas to skinning weights."""
        skinwts_delta = None
        verts_xyz_deltas = None
        if self.use_graph_transformer and self.predict_shape_deltas:
            shape_specific_deltas = mesh_verts_delta[graph_batch_mask].unsqueeze(0)
            shape_specific_deltas = shape_specific_deltas * (
                1 - verts_pinning_mask
            ).unsqueeze(-1)  # handle pinning
            skinwts_delta = shape_specific_deltas[:, :, 3:]
            verts_xyz_deltas = shape_specific_deltas[:, :, :3]
            verts_xyz_deltas = verts_xyz_deltas * (
                1 - graph_vertex_bone_selection_mask_item
            ).unsqueeze(0).unsqueeze(-1)  # handle attachment

        if no_prediction:
            if self.use_graph_transformer and self.predict_shape_deltas:
                verts_xyz_deltas = verts_xyz_deltas * 0.0
                skinwts_delta = skinwts_delta * 0.0

        # verts skinning weights
        if (
            self.use_graph_transformer
            and self.predict_shape_deltas
            and self.shape_specific_verts_skinwts
        ):
            eps = 1e-10
            skinwts_delta = skinwts_delta.transpose(1, 2)
            log_skinwts = torch.log(verts_skinning_wts + eps)
            log_skinwts = log_skinwts + skinwts_delta
            verts_skinning_wts = torch.softmax(log_skinwts, dim=1)
        # if self.predict_shape_deltas and self.shape_specific_verts_deltas:
        #     pos_rest_xyz = pos_rest_xyz + verts_xyz_deltas # update rest vertices based on shape

        return verts_skinning_wts

    def _apply_prediction_to_affine_state(
        self,
        pred,
        deformed_state,
        affine_state,
        active_garment_bones,
        bones_pinning_mask,
        first_pinning_indices,
        pinned_bones_bodyverts,
    ):
        """Apply predicted deformations to the affine state using delta_posed_ortho6d."""
        if self.prediction_type == "delta_posed_ortho6d":
            r6d, t = compute_vectors_from_rotation_matrix(deformed_state)
            # add predicted deltas
            if pred is not None:
                pred_r6d = pred[:, :, :6]
                pred_t = pred[:, :, 6:]
                r6d = r6d + pred_r6d
                t = t + pred_t
            ## pinning to body (pinned bones are the last indices)
            if bones_pinning_mask.sum() > 0:
                t[:, first_pinning_indices:, :] = pinned_bones_bodyverts
            pred_affine_states = torch.zeros(
                (r6d.shape[0], r6d.shape[1], 3, 4), device=self.device
            )
            pred_affine_states[:, :, :3, :3] = compute_rotation_matrix_from_ortho6d(r6d)
            pred_affine_states[:, :, :3, 3] = t
            affine_state[:, active_garment_bones, 0:3, :] = pred_affine_states
            affine_states = affine_state

        return affine_states

    def _unnormalize_positions(self, pos, body_vertices_deformed):
        """Un-normalize garment and body vertex positions if global normalization is enabled."""
        if self.dataset.global_normalization:
            _shift = torch.Tensor([0, self.dataset.global_y_translation, 0]).to(
                self.device
            )
            _scale = self.dataset.global_scale
            # un-normalize garment vertices
            pos = pos * _scale
            pos = pos + _shift
            # un-normalize body vertices
            body_vertices_deformed = body_vertices_deformed * _scale
            body_vertices_deformed = body_vertices_deformed + _shift
        return pos, body_vertices_deformed

    def _compute_per_sample_loss(
        self,
        batch_idx,
        graph_batch_labels,
        predicted_deformations,
        batched_body_vertices_deformed,
        batched_garment_rest_positions,
        batched_garment_lbs,
        batched_motion_affine_states,
        deformed_cloth_state,
        batched_cloth_bones_rest_transforms_inv,
        graph_cloth_verts_skinning_weights,
        graph_verts_pinning_mask,
        graph_bones_pinning_mask,
        graph_pinned_bones_bodyverts_idx,
        graph_vertex_bone_selection_mask,
        graph_garment_verts_to_body,
        active_garment_bones,
        motion_window_size,
        mesh_verts_delta,
        no_prediction,
    ):
        """Compute the physics-based loss for a single sample in the batch."""
        graph_batch_mask = graph_batch_labels == batch_idx

        pred = (
            predicted_deformations[batch_idx].unsqueeze(0)
            if predicted_deformations is not None
            else None
        )
        if no_prediction and predicted_deformations is not None:
            pred = pred * 0.0
        body_vertices_deformed = batched_body_vertices_deformed[batch_idx].unsqueeze(0)
        pos_rest_xyz = batched_garment_rest_positions[batch_idx].unsqueeze(0)

        # lbs object
        garment_lbs = batched_garment_lbs[batch_idx].to(self.device)

        # garment bones metadata
        affine_state = batched_motion_affine_states[batch_idx]
        deformed_state = deformed_cloth_state[batch_idx].unsqueeze(0)
        cloth_bones_restinvT = batched_cloth_bones_rest_transforms_inv[
            batch_idx
        ].unsqueeze(0)

        # garment bone-verts skinning weights
        cloth_verts_skinning_weights = graph_cloth_verts_skinning_weights[
            batch_idx
        ].T.unsqueeze(0)
        verts_skinning_wts = cloth_verts_skinning_weights.clone().detach()

        # pinning masks
        verts_pinning_mask = graph_verts_pinning_mask[batch_idx].unsqueeze(0)
        bones_pinning_mask = graph_bones_pinning_mask[batch_idx].unsqueeze(0)
        pinning_mask = 1 - bones_pinning_mask.unsqueeze(-1)
        first_pinning_indices = int(
            pinning_mask.sum(1).max().item()
        )  # just a hack, assumes pinned bones are placed in the beginning, will be replaced with a more flexible indices estimation later
        pinned_bones_bodyverts_idx = graph_pinned_bones_bodyverts_idx[
            batch_idx
        ].unsqueeze(0)

        # Vectorized batch indexing for pinned bones
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

        # apply shape-specific deltas to skinning weights
        verts_skinning_wts = self._apply_shape_deltas_to_skinning_weights(
            verts_skinning_wts,
            mesh_verts_delta,
            graph_batch_mask,
            verts_pinning_mask,
            graph_vertex_bone_selection_mask[batch_idx],
            no_prediction,
        )

        # skinned vertices before adding prediction
        pos_rest = torch.cat(
            [pos_rest_xyz, torch.ones_like(pos_rest_xyz[..., :1])], dim=-1
        )
        pos_skinned = LBS_positions_batched(  # noqa: F841
            deformed_state, pos_rest, verts_skinning_wts, cloth_bones_restinvT
        )
        pos_skinned = pos_skinned.detach()  # no gradients required for initial skinning

        # apply prediction to affine state
        affine_states = self._apply_prediction_to_affine_state(
            pred,
            deformed_state,
            affine_state,
            active_garment_bones,
            bones_pinning_mask,
            first_pinning_indices,
            pinned_bones_bodyverts,
        )

        # skinned vertices post prediction
        pos = LBS_positions_batched(
            affine_states[:, active_garment_bones, :, :],
            pos_rest,
            verts_skinning_wts,
            cloth_bones_restinvT,
        )

        # handle pinning
        if self.hard_pinning_verts:
            g_to_b = graph_garment_verts_to_body[batch_idx].long()
            pos_on_body = batched_body_vertices_deformed[batch_idx][g_to_b].unsqueeze(0)
            pinmask = verts_pinning_mask.unsqueeze(-1)
            pos = (1 - pinmask) * pos + pinmask * pos_on_body

        # un-normalize positions if needed
        pos, body_vertices_deformed = self._unnormalize_positions(
            pos, body_vertices_deformed
        )

        # calculate physics-based losses
        phys_loss, phys_losses = physics_loss_from_state(
            states=affine_states,
            vertex_positions=pos,
            body_vertex_positions=body_vertices_deformed,
            lbs=garment_lbs,
            contact_params=self.hparams,
            timings=None,
            cloth_weights=verts_skinning_wts,
            only_collision=False,
        )

        return (
            phys_loss,
            phys_losses,
            garment_lbs,
            pos,
            body_vertices_deformed,
            affine_states,
        )

    @profile
    def compute_loss(
        self,
        state: State,
        data: torch.Tensor,
        eval_step=False,
        no_prediction: bool = False,
    ) -> Tuple[torch.Tensor, dict]:
        # load current batch data
        batched_body_verts_positions = data[0].to(self.device)
        batched_shape_latents = data[1].to(self.device)
        batched_pattern_embeddings = data[2].to(self.device)
        graph_data = data[3].to(self.device)
        _batched_shaped_indices = data[4].to(self.device)  # noqa: F841
        batched_garment_indices = data[5].to(self.device)
        batched_motion_sequences = data[6].to(self.device)
        batched_motion_affine_states = data[7].to(self.device)

        # load from dataset
        batched_garment_lbs = np.array(self.dataset.garment_lbs)[
            batched_garment_indices.cpu().numpy()
        ]

        # batch info
        batch_size = len(batched_motion_sequences)
        motion_window_size = self.dataset.motion_window_size
        assert motion_window_size == 1, "Motion Window Size > 1 is not supported!"

        # for reshaping modulations
        num_bone_net_layers = self._get_num_bone_net_layers()

        # batched graph data
        graph_batch_data = self._build_graph_batch_data(
            graph_data, batch_size, batched_body_verts_positions
        )

        # bones rest info, body deformations, garment bone skinning
        (
            batched_cloth_bones_rest_transforms_inv,
            batched_motion_affine_states,
            active_garment_bones,
            batched_body_vertices_deformed,
            deformed_cloth_state,
        ) = self._compute_bones_rest_and_deformations(
            graph_batch_data,
            batch_size,
            motion_window_size,
            graph_data,
            batched_body_verts_positions,
            batched_motion_affine_states,
            batched_motion_sequences,
        )

        """============================== ITERATE OVER GARMENTS ======================================="""

        # GARMENT REST POSE COMPUTATION
        batched_garment_rest_positions = self._compute_garment_rest_positions(
            batch_size,
            graph_batch_data["graph_cloth_verts_rest_positions"],
            graph_batch_data["graph_verts_xyz"],
        )

        """=================================== PARALLELiZED ============================================"""

        # predict hyper-modulations
        (
            hypermodulations,
            mesh_verts_feats,
            mesh_emb,
            mesh_verts_delta,
            bones_hyperfeats,
        ) = self.hypermodulator(
            batched_pattern_embeddings,
            batched_shape_latents,
            graph_data,
            drape_cond_feat=batched_garment_rest_positions,
        )

        # POSE-DEPENDENT GARMENT-BONES DEFORMATIONS
        predicted_deformations, mesh_verts_delta = (
            self._predict_pose_dependent_deformations(
                batch_size,
                num_bone_net_layers,
                batched_motion_sequences,
                hypermodulations,
                mesh_verts_feats,
                mesh_emb,
                mesh_verts_delta,
                bones_hyperfeats,
                deformed_cloth_state,
            )
        )

        """============================== ITERATE OVER GARMENTS ======================================="""

        # SKIN GARMENT TO THE POSED SPACE & COMPUTE LOSS (PER SAMPLE)
        graph_batch_labels = graph_data.batch
        phys_loss_batched = 0
        _regularization = 0  # noqa: F841
        for batch_idx in range(batch_size):
            (
                phys_loss,
                phys_losses,
                garment_lbs,
                pos,
                body_vertices_deformed,
                affine_states,
            ) = self._compute_per_sample_loss(
                batch_idx=batch_idx,
                graph_batch_labels=graph_batch_labels,
                predicted_deformations=predicted_deformations,
                batched_body_vertices_deformed=batched_body_vertices_deformed,
                batched_garment_rest_positions=batched_garment_rest_positions,
                batched_garment_lbs=batched_garment_lbs,
                batched_motion_affine_states=batched_motion_affine_states,
                deformed_cloth_state=deformed_cloth_state,
                batched_cloth_bones_rest_transforms_inv=batched_cloth_bones_rest_transforms_inv,
                graph_cloth_verts_skinning_weights=graph_batch_data[
                    "graph_cloth_verts_skinning_weights"
                ],
                graph_verts_pinning_mask=graph_batch_data["graph_verts_pinning_mask"],
                graph_bones_pinning_mask=graph_batch_data["graph_bones_pinning_mask"],
                graph_pinned_bones_bodyverts_idx=graph_batch_data[
                    "graph_pinned_bones_bodyverts_idx"
                ],
                graph_vertex_bone_selection_mask=graph_batch_data[
                    "graph_vertex_bone_selection_mask"
                ],
                graph_garment_verts_to_body=graph_batch_data[
                    "graph_garment_verts_to_body"
                ],
                active_garment_bones=active_garment_bones,
                motion_window_size=motion_window_size,
                mesh_verts_delta=mesh_verts_delta,
                no_prediction=no_prediction,
            )
            phys_loss_batched += phys_loss

        # combined loss
        phys_loss_total = phys_loss_batched / batch_size

        losses = {
            "phys": phys_loss_total,
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
            outputs["garment_lbs"] = garment_lbs
            outputs["pos_c"] = pos.detach()
            outputs["pos_b"] = body_vertices_deformed.detach()
            outputs["affine_states"] = affine_states.detach()

            if eval_step:
                return outputs

        return total_loss, outputs

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil

import natsort


def material_dict(materials):
    """
    Converts a list of materials, each with an optional "name" key into a dictionary keyed by "name".
    If no "name" key exists, then the key is the index of the material in the input list.
    """
    return {
        (item["name"] if "name" in item else i): item
        for i, item in enumerate(materials)
    }


def build_output_path(
    base_path,
    experiment_name: str,
    stage_name: str | None = None,
    should_continue: bool = False,
    force_overwrite: bool = False,
    mode: str = "train",
):
    output_path = os.path.join(base_path, experiment_name)
    if stage_name is not None:
        output_path = os.path.join(output_path, stage_name)
    if os.path.exists(output_path) and mode == "train":
        if not should_continue and force_overwrite:
            shutil.rmtree(output_path)
            os.makedirs(output_path)
        elif should_continue and force_overwrite:
            raise RuntimeError(
                "ERROR: Instructions unclear, found existing experiment, should we continue or overwrite existing, please choose one or the other."
            )
        elif not should_continue and not force_overwrite:
            raise RuntimeError(
                "ERROR: Refusing to overwrite existing experiment, please choose another name"
            )
    else:
        if should_continue:
            print(
                "WARNING: No existing experiment found to continue from, starting a new one."
            )
    return output_path


def build_experiment_path(cfg):
    model_prefix = cfg.model.type
    model_name = f"{model_prefix}_{cfg.model.hidden_features}x{cfg.model.hidden_layers}"
    strategy = "_BONES_"
    if cfg.train.use_vertex_model:
        strategy += "_VERTEXNET_"
    if cfg.train.optimize_skinning_weights:
        strategy += "_SKINWOPT_"
    parent_path = f"{cfg.train.output_path}/{strategy}/{cfg.train.prediction_type}/{cfg.train.vertex_offset_type}/training_snapshots/physkin/{model_name}/"
    return parent_path


def build_hparams(cfg, physkin_lbs):
    return {
        # Loss weights
        "rig_phys_loss_weight": cfg.train.loss_weight.phys,
        "delta_regularizer": cfg.train.loss_weight.delta_regularizer,
        # Other params
        "body_contact_max_interpenetration_tolerance": cfg.train.body_contact.max_interpenetration_tolerance,
        "body_contact_proximity_padding": cfg.train.body_contact.proximity_padding,
        "lr": cfg.train.learning_rate,
        "model_type": cfg.model.type,
        "hidden_layers": cfg.model.hidden_layers,
        "hidden_features": cfg.model.hidden_features,
        "motion_batch_size": cfg.train.motion_batch_size,
        "seq_dimension": False,
    } | physkin_lbs.config.params.material


def resolve_latest_checkpoint(parent_path, prefix):
    timestamp_dirs = natsort.natsorted(os.listdir(parent_path))
    timestamp_dirs = [ts for ts in timestamp_dirs if ts.startswith(prefix)]
    timestamp = timestamp_dirs[-1]
    output_path = f"{parent_path}/{timestamp}"
    epoch_dirs = natsort.natsorted(os.listdir(output_path))
    epoch_dirs = [ep for ep in epoch_dirs if ep.startswith("epoch_")]
    resume_step = epoch_dirs[-1]
    return timestamp, resume_step

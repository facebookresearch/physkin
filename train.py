# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import logging
import os
import sys

import hydra
import lbs
import numpy as np
import torch
import torch.nn as nn
from models import mlp
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchtnt.framework import train
from torchtnt.utils import init_from_env, seed
from training.progress_bar import TQDMProgressBarDDP

_project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_project_root, "third_party", "diffusion-net", "src"))
import shutil
from glob import glob

import diffusion_net
import natsort
import torch.distributed
import utils
from ddp_utils import convert_state_dict, get_ddp_model, setup_ddp
from line_profiler import profile
from load_garments import load_garments_meshes
from torch_geometric.data import Batch, Data
from tqdm import tqdm
from training import dataio
from training.callbacks import Checkpoint, TensorBoardLogger

# from units import physkin_hyperfixedbone_multigarments_DDP as main_unit
from units import physkin_hyperfixedbone_multishapes as main_unit


def arr_str(arr):
    return np.array2string(np.array(arr), separator=", ", max_line_width=10000)


def custom_collate_fn(batch):
    if len(batch) == 0:
        return batch
    collated_batch = []
    sample_size = len(batch[0])
    for i in range(sample_size):
        elements = [sample[i] for sample in batch]
        first_element = elements[0]
        if isinstance(first_element, torch.Tensor):
            collated_batch.append(torch.stack(elements))
        elif isinstance(first_element, Data):
            collated_batch.append(Batch.from_data_list(elements))
        elif isinstance(first_element, int):
            collated_batch.append(torch.tensor(elements))
    return collated_batch


def prepare_motion_dataloader(dataset, batch_size: int, device: torch.device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn,
    )


def _create_bone_model(cfg, n_active_bones, prediction_dim_dict):
    """Create the bone network model based on config type."""
    if cfg.model.type == "mlp":
        return mlp.ModulatedMLP(
            in_features=n_active_bones * cfg.model.inp_dim,
            hidden_features=cfg.model.hidden_features,
            hidden_layers=cfg.model.hidden_layers,
            out_features=n_active_bones
            * prediction_dim_dict[cfg.train.prediction_type],
            hidden_activation=nn.SiLU(),
        )
    elif cfg.model.type == "siren":
        print("Using SIREN for Bone Network...")
        return mlp.SirenNet(
            dim_in=n_active_bones * cfg.model.inp_dim,
            dim_hidden=cfg.model.hidden_features,
            num_layers=cfg.model.hidden_layers,
            dim_out=n_active_bones * prediction_dim_dict[cfg.train.prediction_type],
            # hidden_activation=nn.SiLU(),
            final_activation=nn.Identity(),
            w0_initial=1.0,
        )
    return None


def _create_vertex_model(cfg, scene, pose_dim):
    """Create the per-vertex network model based on config type."""
    if cfg.vertex_model.type == "mlp":
        input_dim_dict = {"pose_vector": pose_dim}
        print("Using MLP for Vertex Network...")
        return mlp.MLP(
            input_dim_dict[cfg.vertex_model.input_type],
            cfg.vertex_model.hidden_features,
            cfg.vertex_model.hidden_layers,
            scene["initPos"].shape[0] * 3,
            softmax=False,
        )
    elif cfg.vertex_model.type == "siren":
        L_freq = 6
        print("Using SIREN for Vertex Network...")
        return mlp.SirenNetPerPoint(
            dim_in=3 + (3 * 2 * L_freq),
            dim_hidden=cfg.vertex_model.hidden_features,
            num_layers=cfg.vertex_model.hidden_layers,
            dim_out=3,
            final_activation=nn.Identity(),
            w0_initial=1.0,
        )
    elif cfg.vertex_model.type == "diffusion_net":
        print("Using DiffusionNet for Vertex Network...")
        in_feat_dim = 3
        out_feat_dim = 128
        vertex_tensor = torch.tensor(
            scene["initPos"], dtype=torch.float32, device="cpu"
        )
        _faces_tensor = torch.tensor(  # noqa: F841
            scene["faces"], dtype=torch.int32, device="cpu"
        )
        _vertex_tensor_normalized = diffusion_net.geometry.normalize_positions(  # noqa: F841
            vertex_tensor
        )
        return diffusion_net.layers.DiffusionNet(
            C_in=in_feat_dim,
            C_out=out_feat_dim,
            C_width=cfg.vertex_model.hidden_features,  # internal size of the diffusion net. 32 -- 512 is a reasonable range
            N_block=cfg.vertex_model.hidden_layers,
            last_activation=None,  # apply a last softmax to outputs
            outputs_at="global_mean",
            diffusion_method="spectral",
        )
    return None


def _create_pose_modulator(cfg):
    """Create the pose modulator based on config type."""
    if cfg.pose_modulator.type == "siren":
        return mlp.SirenNet(
            dim_in=cfg.pose_modulator.input_dim,
            dim_hidden=cfg.pose_modulator.hidden_features,
            num_layers=cfg.pose_modulator.hidden_layers,
            dim_out=cfg.model.hidden_layers * cfg.model.hidden_features,
            # hidden_activation=nn.SiLU(),
            final_activation=nn.Identity(),
            w0_initial=1.0,
        )
    if cfg.pose_modulator.type == "mlp":
        return mlp.ModulatedMLP(
            in_features=cfg.pose_modulator.input_dim,
            hidden_features=cfg.pose_modulator.hidden_features,
            hidden_layers=cfg.pose_modulator.hidden_layers,
            out_features=2 * cfg.model.hidden_layers * cfg.model.hidden_features,
            # out_features =  n_active_bones * prediction_dim_dict[cfg.train.prediction_type],
            hidden_activation=nn.SiLU(),
        )
    return None


def _build_hypermod_config(cfg, n_active_bones, all_pose_dataset):
    """Build the hypermodulator configuration dictionary."""
    shape_embedder_config = cfg.shape_embedder_config
    garment_embedder_config = cfg.garment_embedder_config
    shape_embedder_config["max_token_len"] = all_pose_dataset.body_token_len
    shape_embedder_config["in_dim"] = all_pose_dataset.train_body_latents.shape[-1]
    # garments
    garment_embedder_config["max_token_len"] = all_pose_dataset.garment_token_len
    _g_in_dim_ = (
        3 + 1 + 1 + 4 + 3 + 1
    )  # (xyz + pin_label + bone_label + ppf + normal + area )
    if "laplacian" in cfg.garment_embedder_config.pos_enc:
        _g_in_dim_ += all_pose_dataset.garment_graph_data[0].pe.shape[-1]
    garment_embedder_config["in_dim"] = 64
    # per bone
    bone_feature_extractor_config = cfg.bone_feature_extractor_config
    bone_feature_extractor_config["in_dim"] = garment_embedder_config["hidden_dim"]
    bone_feature_extractor_config["hidden_dim"] = garment_embedder_config["hidden_dim"]
    # per vertex
    vertex_feature_extractor_config = cfg.vertex_feature_extractor_config
    vertex_feature_extractor_config["in_dim"] = bone_feature_extractor_config[
        "hidden_dim"
    ]
    vertex_feature_extractor_config["hidden_dim"] = (
        n_active_bones + 3
    )  # (skinwts-delta + xyz-delta)

    return {
        "shape_embedder_config": shape_embedder_config,
        "garment_embedder_config": garment_embedder_config,
        "bone_feature_extractor_config": bone_feature_extractor_config,
        "vertex_feature_extractor_config": vertex_feature_extractor_config,
        "hidden_dim": cfg.hypermodulator.shared_hidden_features,
        "hidden_layers": cfg.hypermodulator.shared_hidden_layers,
        "use_seperate_modheads": cfg.hypermodulator.use_seperate_modheads,
        "use_bone_encodings": cfg.model.input_type == "hypertransformations",
        "modheads_hidden_layers": cfg.hypermodulator.modheads_hidden_layers,
        "bonefeats_hidden_layers": cfg.hypermodulator.bonefeats_hidden_layers,
        "vertsfeats_hidden_layers": cfg.hypermodulator.vertsfeats_hidden_layers,
        "max_num_faces": all_pose_dataset.garment_token_len,
        "out_dim": (
            2 * cfg.pose_modulator.hidden_features,
            cfg.pose_modulator.hidden_layers,
        ),  # FiLM (Mods + Shifts)
        "layer_norm": False,
        "residual_blocks": cfg.hypermodulator.residual_blocks,
        "drape_cond_dim": cfg.hypermodulator.drape_cond_dim,
        "drape_cond_type": cfg.hypermodulator.drape_cond_type,
        "drape_mode": cfg.hypermodulator.drape_mode,
        "graph_use_panels_xyz": cfg.hypermodulator.graph_use_panels_xyz,
        "max_num_bones": n_active_bones,
        "bones_feats_out_dim": cfg.model.inp_dim,
        "use_pattern_latents": cfg.hypermodulator.use_pattern_embeddings,
        "pattern_latent_dim": (6, 256),
        "pattern_proj_dim": cfg.hypermodulator.pattern_proj_dim,
        "shape_proj_dim": cfg.hypermodulator.shape_proj_dim,
        "graph_hidden_dim": cfg.hypermodulator.graph_transformer_hidden_dim,
        "graph_hidden_layers": cfg.hypermodulator.graph_transformer_hidden_layers,
        "graph_pe_dim": cfg.hypermodulator.graph_transformer_pe_dim,
        "graph_attn_type": cfg.hypermodulator.graph_attn_type,
        "graph_attn_redraw_interval": cfg.hypermodulator.graph_attn_redraw_interval,
        "graph_pool_k": cfg.hypermodulator.graph_pool_k,
        "mesh_proj_dim": cfg.hypermodulator.mesh_proj_dim,
        "use_graph_transformer": cfg.hypermodulator.use_graph_transformer,
    }


def _load_checkpoint(
    cfg, output_path, ddp_enabled, model, pose_modulator, hypermodulator
):
    """Load model weights from checkpoint."""
    pose_modulator.load_state_dict(
        convert_state_dict(
            torch.load(
                f"{output_path}/{cfg.train.resume_step}/pose_modulator.pth",
                map_location="cpu",
                weights_only=True,
            ),
            ddp_enabled,
        )
    )
    hypermodulator.load_state_dict(
        convert_state_dict(
            torch.load(
                f"{output_path}/{cfg.train.resume_step}/hypermodulator.pth",
                map_location="cpu",
                weights_only=True,
            ),
            ddp_enabled,
        )
    )
    model.load_state_dict(
        convert_state_dict(
            torch.load(
                f"{output_path}/{cfg.train.resume_step}/module.pth",
                map_location="cpu",
                weights_only=True,
            ),
            ddp_enabled,
        )
    )
    if cfg.train.use_vertex_model:
        # vertex_model must exist in caller scope when use_vertex_model is True
        return True
    return False


def _load_vertex_model_checkpoint(cfg, output_path, ddp_enabled, vertex_model):
    """Load vertex model weights from checkpoint."""
    vertex_model.load_state_dict(
        convert_state_dict(
            torch.load(
                f"{output_path}/{cfg.train.resume_step}/vertex_model.pth",
                map_location="cpu",
                weights_only=True,
            ),
            ddp_enabled,
        )
    )


def _build_save_meshes_info_inference(cfg, output_path, all_pose_dataset):
    """Build save_meshes_info dict for inference mode."""
    save_meshes_info = np.load(
        f"{output_path}/eval_meshes_info.npy", allow_pickle=True
    ).item()
    save_meshes_info["infer_train"] = save_meshes_info["train"]
    save_meshes_info["infer_test"] = save_meshes_info["test"]
    _min_ = all_pose_dataset.num_train_garment_samples
    _max_ = all_pose_dataset.num_garment_samples
    save_meshes_info["infer_test"]["garment_indices"] = np.random.choice(
        np.arange(_min_, _max_),
        len(save_meshes_info["infer_test"]["shape_indices"]),
        replace=True,
    )
    return save_meshes_info


def _build_save_meshes_info_training(cfg, world_size, all_pose_dataset):
    """Build save_meshes_info dict for training mode."""
    save_meshes_info = {}
    num_pose_samples = len(all_pose_dataset.all_motion_data)
    num_shape_samples = len(all_pose_dataset.train_body_verts)
    num_garment_samples = all_pose_dataset.num_garment_samples
    num_train_pose_samples = all_pose_dataset.motion_train_samplenum
    num_train_shape_samples = all_pose_dataset.shape_train_samplenum
    num_train_garment_samples = all_pose_dataset.num_train_garment_samples
    total_save_indices = max(
        world_size,
        cfg.train.num_train_eval_samples,
        cfg.train.num_test_eval_samples,
    )
    repeat_train_pose_samples = total_save_indices > num_train_pose_samples
    repeat_test_pose_samples = total_save_indices > (
        num_pose_samples - num_train_pose_samples
    )
    repeat_train_shape_samples = total_save_indices > num_train_shape_samples
    repeat_test_shape_samples = total_save_indices > (
        num_shape_samples - num_train_shape_samples
    )
    repeat_train_garment_samples = total_save_indices > num_train_garment_samples
    repeat_test_garment_samples = total_save_indices > (
        num_garment_samples - num_train_garment_samples
    )
    save_train_pose_indices = np.random.choice(
        np.arange(0, num_train_pose_samples),
        total_save_indices,
        replace=repeat_train_pose_samples,
    )
    save_test_pose_indices = np.random.choice(
        np.arange(num_train_pose_samples, num_pose_samples),
        total_save_indices,
        replace=repeat_test_pose_samples,
    )
    save_train_shape_indices = np.random.choice(
        np.arange(0, num_train_shape_samples),
        total_save_indices,
        replace=repeat_train_shape_samples,
    )
    save_test_shape_indices = np.random.choice(
        np.arange(num_train_shape_samples, num_shape_samples),
        total_save_indices,
        replace=repeat_test_shape_samples,
    )
    save_train_garment_indices = np.random.choice(
        np.arange(0, num_train_garment_samples),
        total_save_indices,
        replace=repeat_train_garment_samples,
    )
    save_test_garment_indices = np.random.choice(
        np.arange(num_train_garment_samples, num_garment_samples),
        total_save_indices,
        replace=repeat_test_garment_samples,
    )
    save_meshes_info["train"] = {}
    save_meshes_info["train"]["pose_indices"] = save_train_pose_indices
    save_meshes_info["train"]["shape_indices"] = save_train_shape_indices
    save_meshes_info["train"]["garment_indices"] = save_train_garment_indices
    save_meshes_info["test"] = {}
    save_meshes_info["test"]["pose_indices"] = save_test_pose_indices
    save_meshes_info["test"]["shape_indices"] = save_test_shape_indices
    save_meshes_info["test"]["garment_indices"] = save_test_garment_indices
    save_meshes_info["infer_train"] = {}
    save_meshes_info["infer_test"] = {}
    return save_meshes_info


def _save_visualization_data(cfg, unit, num_new_garment_bones):
    """Save visualization data from training debug dict."""
    all_pos_b = [
        unit.debug_dict[f"{key}"]["body_pos"].numpy() for key in unit.debug_dict.keys()
    ]
    all_pos_c = [
        unit.debug_dict[f"{key}"]["cloth_pos"].numpy() for key in unit.debug_dict.keys()
    ]
    all_pos_bones_c = [
        unit.debug_dict[f"{key}"]["affine_states"].numpy()[:, -num_new_garment_bones:][
            :, :, :3, 3
        ]
        for key in unit.debug_dict.keys()
    ]
    all_faces_b = [
        unit.debug_dict[f"{key}"]["body_faces"] for key in unit.debug_dict.keys()
    ]
    all_faces_c = [
        unit.debug_dict[f"{key}"]["cloth_faces"] for key in unit.debug_dict.keys()
    ]
    all_uv_c = [unit.debug_dict[f"{key}"]["cloth_uv"] for key in unit.debug_dict.keys()]
    video_data_path = cfg.train.video_data_path
    os.makedirs(video_data_path, exist_ok=True)
    data_dict = {}
    data_dict["all_pos_bones_c"] = all_pos_bones_c
    data_dict["all_faces_b"] = all_faces_b
    data_dict["all_faces_c"] = all_faces_c
    data_dict["all_pos_b"] = all_pos_b
    data_dict["all_pos_c"] = all_pos_c
    data_dict["all_uv_w_c"] = all_uv_c
    data_dict["uv_img"] = None
    np.savez(f"{video_data_path}//data_dict_training.npz", data_dict)
    print(
        f"Saved visualization data to {video_data_path}//data_dict_multigarments_training.npz"
    )


def _place_models_on_device(
    ddp_enabled, device_id, model, pose_modulator, hypermodulator
):
    """Place models on the appropriate device, wrapping with DDP if enabled."""
    if ddp_enabled:
        model = get_ddp_model(model, device_id)
        pose_modulator = get_ddp_model(pose_modulator, device_id)
        hypermodulator = get_ddp_model(hypermodulator, device_id)
    else:
        model = model.to(device_id)
        pose_modulator = pose_modulator.to(device_id)
        hypermodulator = hypermodulator.to(device_id)
    return model, pose_modulator, hypermodulator


def _run_training(
    unit,
    train_motion_dataloader,
    cfg,
    rank,
    ddp_enabled,
    ckpt_callback,
    output_path,
    num_new_garment_bones,
):
    """Run the training loop with appropriate callbacks based on rank and DDP settings."""
    if rank == 0 or not ddp_enabled:
        logging.info(
            f" ------ [ Callbacks and exports enabled on Global Device :{rank} only !!! ] ----"
        )

        tqdm_callback = TQDMProgressBarDDP(
            num_steps_per_epoch=len(train_motion_dataloader), refresh_rate=10
        )

        # save scene configuration
        # physkin_lbs.export(custom_garment_dirpath)

        msg_start = f"""
        :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                            Started training for RANK:{rank}
        :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        """
        print(msg_start)
        logging.info(msg_start)
        if cfg.infer.run_inference:
            train(
                unit,
                train_dataloader=train_motion_dataloader,
                callbacks=None,
                max_epochs=cfg.train.epochs,
            )
        else:
            # include callbacks
            train(
                unit,
                train_dataloader=train_motion_dataloader,
                callbacks=[tqdm_callback, ckpt_callback],
                max_epochs=cfg.train.epochs,
            )
        msg_end = f"""
        :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                            Completed training for RANK:{rank}
        :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        """
        print(msg_end)
        logging.info(msg_end)

        # save optimized skinning weights after training for inference
        # torch.save(unit.cloth_skinning_weights, os.path.join(custom_garment_dirpath, "cloth_skinning_weights_optimized.pt"))
        if cfg.train.visualize_training:
            _save_visualization_data(cfg, unit, num_new_garment_bones)

    else:
        msg_start = f"""
        :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                            Started training for RANK:{rank}
        :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        """
        print(msg_start)
        logging.info(msg_start)

        # don't include callbacks
        train(
            unit,
            train_dataloader=train_motion_dataloader,
            max_epochs=cfg.train.epochs,
        )
        msg_end = f"""
        :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                            Completed training for RANK:{rank}
        :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        """
        print(msg_end)
        logging.info(msg_end)


def prepare_and_train_unit(
    ddp_enabled,
    device_id,
    rank,
    world_size,
    ckpt_callback,
    scene,
    config,
    cfg,
    num_new_garment_bones,
    bones_lod_levels,
    output_path,
    all_pose_dataset,
):
    # create separate log file for each device
    log_filepath = os.path.join(output_path, f"log_ddp_{rank}.txt")
    logging.basicConfig(
        filename=log_filepath,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.info(f"--------- Process started: rank={rank}, pid={os.getpid()} ---------")

    # define checkpoinitng callback before intializing process group
    ckpt_callback = Checkpoint(
        dirpath=output_path,
        save_every_n_epochs=cfg.train.checkpoint_every_n_epochs,
    )

    # create LBS object
    physkin_lbs = lbs.LBS.from_scene_and_config(scene, config)

    n_active_bones = num_new_garment_bones

    hparams = utils.build_hparams(cfg, physkin_lbs)

    # define input dimensions for different prediction and conditioning types
    prediction_dim_dict = {
        "delta_posed_ortho6d": 9,
        "delta_unposed_ortho6d": 9,
        "delta_quaternion": 7,
    }
    pose_dim = physkin_lbs.config.rig.nActivePoseParams

    # define bone network
    model = _create_bone_model(cfg, n_active_bones, prediction_dim_dict)

    if cfg.train.use_vertex_model:
        # define per-vertex network
        vertex_model = _create_vertex_model(cfg, scene, pose_dim)

    # pose modulator
    pose_modulator = _create_pose_modulator(cfg)

    # define fixed garment idx
    if cfg.train.garment_sampling_type == "rank_based":
        all_pose_dataset.fixed_garment_index = rank

    # define token lengths for embedders
    hypermod_config = _build_hypermod_config(cfg, n_active_bones, all_pose_dataset)

    hypermodulator = mlp.HyperModulator(
        hypermod_config=hypermod_config,
        max_num_bones=n_active_bones,
        predict_deltas=cfg.hypermodulator.predict_shape_deltas != 0,
    )

    # load from checkpoint
    if cfg.train.continue_from_last_checkpoint or cfg.infer.run_inference:
        needs_vertex_load = _load_checkpoint(
            cfg, output_path, ddp_enabled, model, pose_modulator, hypermodulator
        )
        if needs_vertex_load:
            _load_vertex_model_checkpoint(cfg, output_path, ddp_enabled, vertex_model)

    model, pose_modulator, hypermodulator = _place_models_on_device(
        ddp_enabled, device_id, model, pose_modulator, hypermodulator
    )

    if cfg.infer.run_inference:
        save_meshes_info = _build_save_meshes_info_inference(
            cfg, output_path, all_pose_dataset
        )
    else:
        save_meshes_info = _build_save_meshes_info_training(
            cfg, world_size, all_pose_dataset
        )

    if rank == 0:
        np.save(f"{output_path}/eval_meshes_info.npy", save_meshes_info)
        save_info_msg = f"""
        ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ------------------------------------------------------------------------------------------------------------- EVAL SAVE INFO -------------------------------------------------------------------------------------------------------------
        ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


        TRAIN --> # poses : {arr_str(save_meshes_info["train"]["pose_indices"])}
        TRAIN --> # shapes : {arr_str(save_meshes_info["train"]["shape_indices"])}
        TRAIN --> # garments : {arr_str(save_meshes_info["train"]["garment_indices"])}

        ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        TEST --> # poses : {arr_str(save_meshes_info["test"]["pose_indices"])}
        TEST --> # shapes : {arr_str(save_meshes_info["test"]["shape_indices"])}
        TEST --> # garments : {arr_str(save_meshes_info["test"]["garment_indices"])}

        ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
        """
        print(save_info_msg, flush=True)

    tb_logger = TensorBoardLogger(output_path)

    unit = main_unit.PhysRigUnit(
        train_stage=cfg.train.stage,
        run_inference=cfg.infer.run_inference,
        module=model,
        pose_modulator=pose_modulator,
        hypermodulator=hypermodulator,
        device=device_id,
        rank=rank,
        lbs=physkin_lbs,
        bones_lod_levels=bones_lod_levels,
        body_bones_weights=torch.Tensor(scene["bodyBoneWeights"]),
        default_body_rest_verts=torch.Tensor(scene["bodyPos"]),
        optimize_skinning_weights_start_epoch=cfg.train.optimize_skinning_weights_start_epoch,
        vertex_model_start_epoch=cfg.train.vertex_model_start_epoch,
        prediction_type=cfg.train.prediction_type,
        vertex_offset_type=cfg.train.vertex_offset_type,
        model_input_type=cfg.model.input_type,
        log_every_n_steps=cfg.train.log_every_n_steps,
        tb_logger=tb_logger,
        debug_path=cfg.train.debug_path,
        ckpt_path=output_path,
        checkpoint_every_n_epochs=cfg.train.checkpoint_every_n_epochs,
        hparams=hparams,
        dataset_type=cfg.train.dataset_type,
        dataset=all_pose_dataset,
        input_batch_size=cfg.train.motion_batch_size,
        joint_training=cfg.train.joint_training,
        use_vertex_model=cfg.train.use_vertex_model,
        optimize_skinning_weights=cfg.train.optimize_skinning_weights,
        material_change=cfg.train.material_change,
        material_change_start_epoch=cfg.train.material_change_start_epoch,
        skinwt_spread=cfg.train.skinwt_spread,
        debug_train_data=cfg.train.visualize_training,
        custom_collate_fn=custom_collate_fn,
        ddp_enabled=ddp_enabled,
        save_meshes_info=save_meshes_info,
        predict_shape_deltas=cfg.hypermodulator.predict_shape_deltas,
        gradient_clipping=cfg.train.gradient_clipping,
        hypermodulator_start_epoch=cfg.train.hypermodulator_start_epoch,
        freeze_garment_encoder=cfg.train.freeze_garment_encoder,
        bone_module_inp_type=cfg.model.input_type,
        drape_mode=cfg.hypermodulator.drape_mode,
        use_graph_transformer=cfg.hypermodulator.use_graph_transformer,
        add_noise=cfg.train.add_noise,
        pose_interpolation_start=cfg.train.pose_interpolation_start,
        pose_interpolation_interval=cfg.train.pose_interpolation_interval,
        cfg=cfg,
    )

    # prepare dataloader
    if not ddp_enabled:
        train_motion_dataloader = prepare_motion_dataloader(
            all_pose_dataset, cfg.train.motion_batch_size, device=init_from_env()
        )
    else:
        sampler = DistributedSampler(
            all_pose_dataset, num_replicas=world_size, rank=rank
        )
        train_motion_dataloader = DataLoader(
            all_pose_dataset,
            batch_size=cfg.train.motion_batch_size,
            sampler=sampler,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            collate_fn=custom_collate_fn,
        )

    _run_training(
        unit,
        train_motion_dataloader,
        cfg,
        rank,
        ddp_enabled,
        ckpt_callback,
        output_path,
        num_new_garment_bones,
    )


def _resolve_output_path(cfg, parent_path, timstamp_fromenv, rank, ddp_enabled):
    """Resolve the output path based on inference/training mode and rank."""
    output_path = None
    if not (cfg.infer.run_inference):
        if rank == 0:
            output_path = utils.build_output_path(
                parent_path,
                (
                    cfg.train.timestamp
                    if cfg.train.continue_from_last_checkpoint
                    else timstamp_fromenv
                ),
                should_continue=cfg.train.continue_from_last_checkpoint,
                force_overwrite=cfg.train.force_overwrite,
            )
    elif cfg.train.timestamp == "latest":
        cfg.train.timestamp, cfg.train.resume_step = utils.resolve_latest_checkpoint(
            parent_path, cfg.train.timestamp_prefix
        )
        output_path = f"{parent_path}/{cfg.train.timestamp}"

    if ddp_enabled:
        # Wrap in a list for broadcast_object_list
        shared_output_path_list = [output_path]
        torch.distributed.broadcast_object_list(shared_output_path_list, src=0)
        # shared data
        output_path = shared_output_path_list[0]

    return output_path


def _prepare_output_directory(cfg, output_path, rank, ddp_enabled):
    """Create output directories and copy config files."""
    if rank == 0 or not ddp_enabled:
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(cfg.train.debug_path, exist_ok=True)
        # config path
        src_config_dir = os.path.join(cfg.DATA_ROOT, "config")
        # copy config files to the output directory
        shutil.copyfile(
            os.path.join(src_config_dir, "defaults.yaml"),
            os.path.join(output_path, "defaults.yaml"),
        )
        shutil.copyfile(
            os.path.join(src_config_dir, "physkin_hyperbone.yaml"),
            os.path.join(output_path, "expconfig.yaml"),
        )


def _compute_num_garment_bones(cfg, bones_lod_levels):
    """Compute the total number of new garment bones including pinned bones."""
    max_num_random_bones = max(bones_lod_levels)
    num_random_pinned_bones = 0
    if cfg.train.estimate_pinning_bones:
        num_random_pinned_bones = (
            (2 * int(np.sqrt(max_num_random_bones)))
            if cfg.train.max_num_pin_bones == -1
            else cfg.train.max_num_pin_bones
        )
    return max_num_random_bones + num_random_pinned_bones


def _resolve_garment_paths(cfg):
    """Resolve and filter garment paths based on config."""
    max_num_garments = cfg.train.max_num_garments
    garment_root_dir = cfg.garment.root_dir
    all_garment_paths = natsort.natsorted(glob(os.path.join(garment_root_dir, "*")))

    if max_num_garments == 1:
        all_garment_paths = [
            all_garment_paths[cfg.garment.fixed_garment_idx]
        ] * 2  # repeating for validation
        max_num_garments = 2
    elif max_num_garments > 1:
        all_garment_paths = all_garment_paths[:max_num_garments]

    return all_garment_paths, max_num_garments


def _load_and_cache_garments_metadata(
    cfg, all_garment_paths, max_num_garments, cached_garments_metadata_path
):
    """Load garment meshes, optionally attach pattern embeddings, and cache to disk."""
    pattern_embeddings = None
    if cfg.hypermodulator.use_pattern_embeddings:
        pattern_embeddings = torch.load(cfg.garment.pattern_embeddings_path)

    # whether to use the cached metadata or not
    if not cfg.garment.use_cached_metadata or not os.path.exists(
        cached_garments_metadata_path
    ):
        GARMENTS_METADATA = []
        # iterate over garments
        for garment_idx in tqdm(range(len(all_garment_paths))):
            garment_metadata, reference_garment_name = load_garments_meshes(
                cfg,
                all_garment_paths,
                garment_idx,
                global_normalization=cfg.body.global_normalization,
                global_y_translation=cfg.body.global_y_translation,
                global_scale=cfg.body.global_scale,
            )
            if garment_metadata == -1:
                print(f"Too many vertices: {all_garment_paths[garment_idx]}")
                shutil.rmtree(all_garment_paths[garment_idx])
                continue
            if pattern_embeddings is not None:
                garment_metadata.pattern_embeddings = pattern_embeddings[
                    reference_garment_name
                ]
                if max_num_garments == 2:
                    print(
                        f"Loading pattern embeddings for --> {reference_garment_name}"
                    )  # confirming both train and val have same garment when training on a single garment
            GARMENTS_METADATA.append(garment_metadata)

        # save garments metadata to disk
        print()
        print(f"Saving garments metadata to disk --> {cached_garments_metadata_path}")
        print()
        torch.save(GARMENTS_METADATA, cached_garments_metadata_path)


def _create_dataset(cfg, physkin_lbs, garments_metadata, val_garments_metadata):
    """Create the training dataset based on dataset type."""
    if cfg.train.dataset_type == "shapes":
        return dataio.ShapePoseAnimDataset(
            body_name=cfg.body.name,
            pose_dataset_path=cfg.train.poseanim_dataset_path,
            motion_sequence_dataset_path=(
                cfg.train.motion_sequences_dataset_path
                if not cfg.train.load_random_poses
                else None
            ),
            shape_dataset_path=cfg.body.verts_data_path,
            shape_latent_type=cfg.hypermodulator.shape_latent_type,
            activePoseParamMask=physkin_lbs.config.rig.activePoseParamMask.to("cpu"),
            lbs=physkin_lbs,
            spt=True,
            canonical_pose_only=cfg.train.canonical_pose_only,
            fixed_root_rotation=cfg.train.fixed_root_rotation,
            fixed_root_translation=cfg.train.fixed_root_translation,
            shape_train_samplenum=cfg.train.first_k_shapes,
            motion_train_samplenum=cfg.train.max_train_samples,
            load_garments=True,
            garments_metadata=garments_metadata,
            num_test_garment_samples=len(val_garments_metadata),
            global_normalization=cfg.body.global_normalization,
            global_y_translation=cfg.body.global_y_translation,
            global_scale=cfg.body.global_scale,
            garment_specific_pose_masking=cfg.train.garment_specific_pose_masking,
            garment_sampling_type=cfg.train.garment_sampling_type,
            motion_window_size=cfg.train.motion_window_size,
            load_random_poses=cfg.train.load_random_poses,
            alpha_canonical_pose=cfg.train.alpha_canonical_pose,
            train_sampling_type=cfg.train.train_sampling_type,
        )
    return None


###### MAIN FUNCTION #######


@hydra.main(
    version_base=None,
    config_path="./config",
    config_name="physkin_hyperbone_multigarments",
)
@profile
def main(cfg: DictConfig) -> None:
    # seed for redproducibility
    base_seed = 42
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed_all(base_seed)

    # memory allocation for deterministic algorithms (mainly for evaluation)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    ddp_enabled = cfg.train.ddp and (not cfg.infer.run_inference)

    # DDP info
    if ddp_enabled:
        ddp_info_msg = f"""
        =============== DDP Environment Variables ===============
        MASTER_ADDR:  {os.environ.get("MASTER_ADDR")}
        MASTER_PORT:  {os.environ.get("MASTER_PORT")}
        WORLD_SIZE:   {os.environ.get("WORLD_SIZE")}
        RANK:         {os.environ.get("RANK")}
        LOCAL_RANK:   {os.environ.get("LOCAL_RANK")}
        ========================================================="""
        print(ddp_info_msg, flush=True)

    if not ddp_enabled:
        # timestamp
        os.environ["EXP_TIMESTAMP"] = datetime.datetime.now().strftime(
            "%Y-%d-%m-%H-%M-%S"
        )

    # timestamp from environment [ make sure to run the following command via shell ---- export EXP_TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S") ----  ]
    timstamp_fromenv = os.environ["EXP_TIMESTAMP"]

    # define paths
    parent_path = utils.build_experiment_path(cfg)

    # define callback (should be defined before init process group call, not sure why)
    ckpt_callback = Checkpoint(
        dirpath=os.path.join(parent_path, timstamp_fromenv),
        save_every_n_epochs=cfg.train.checkpoint_every_n_epochs,
    )

    # intialize process group
    device_id = 0
    rank = 0
    world_size = 1
    if ddp_enabled:
        device_id, rank, world_size = setup_ddp()

    # build output directory
    output_path = _resolve_output_path(
        cfg, parent_path, timstamp_fromenv, rank, ddp_enabled
    )

    print(f"OUTPUT DIR: {output_path}")

    # prepare output directory and config files
    _prepare_output_directory(cfg, output_path, rank, ddp_enabled)

    # initialize config
    seed(cfg.seed)
    config = dataio.load_config(cfg.train.config_path)

    # initialize scene
    scene = dataio.load_scene(cfg.train.scene_path)
    scene = dict(scene)  # Allow modifications for scene

    # different lod-levels of garment bones
    bones_lod_levels = cfg.train.bones_lod_levels

    # number of garment bones
    num_new_garment_bones = _compute_num_garment_bones(cfg, bones_lod_levels)

    # create lbs object
    physkin_lbs = lbs.LBS.from_scene_and_config(scene, config)

    # LOAD GARMENT MESHES
    all_garment_paths, max_num_garments = _resolve_garment_paths(cfg)

    # load garments
    cached_garments_metadata_path = cfg.garment.cached_metadata_path

    if rank == 0 or not ddp_enabled:
        _load_and_cache_garments_metadata(
            cfg, all_garment_paths, max_num_garments, cached_garments_metadata_path
        )

    if ddp_enabled:
        # wait for the main process to save data
        torch.distributed.barrier(device_ids=[device_id])

        # Wrap in a list for broadcast_object_list
        shared_num_new_garment_bones_list = [num_new_garment_bones]
        torch.distributed.broadcast_object_list(
            shared_num_new_garment_bones_list, src=0
        )

        # shared data
        num_new_garment_bones = shared_num_new_garment_bones_list[0]

    # load garments metadata from disk
    print()
    print(
        f"RANK : {rank} | Loading garments metadata from --> {cached_garments_metadata_path}"
    )
    print()
    GARMENTS_METADATA = torch.load(cached_garments_metadata_path, weights_only=False)
    print()
    print(f"RANK : {rank} | Garments Loaded !!!")
    print()

    # train-val split for garments
    GARMENTS_METADATA = GARMENTS_METADATA[:max_num_garments]
    split_length = int(len(GARMENTS_METADATA) * cfg.train.train_split)
    assert split_length < len(GARMENTS_METADATA)
    train_garments_metadata = GARMENTS_METADATA[:split_length]
    val_garments_metadata = GARMENTS_METADATA[
        split_length : split_length + cfg.train.num_test_eval_samples
    ]
    garments_metadata = train_garments_metadata + val_garments_metadata

    # PREPARE DATASET
    all_pose_dataset = _create_dataset(
        cfg, physkin_lbs, garments_metadata, val_garments_metadata
    )

    print()
    print(f"RANK : {rank} | Dataset Prepared !!!")
    print()

    # set up unit and start training
    prepare_and_train_unit(
        ddp_enabled,
        device_id,
        rank,
        world_size,
        ckpt_callback,
        scene,
        config,
        cfg,
        num_new_garment_bones,
        bones_lod_levels,
        output_path,
        all_pose_dataset,
    )

    # cleanup
    if ddp_enabled:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

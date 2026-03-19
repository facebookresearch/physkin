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
from models.conv1d_mlp import Conv1dMLP
from omegaconf import DictConfig
from torchtnt.utils import init_from_env, seed

_project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_project_root, "third_party", "diffusion-net", "src"))
from glob import glob

import cv2
import diffusion_net
import natsort
import potpourri3d as pp3d
import pymeshlab as pml
import torch.distributed
import torch.multiprocessing as mp
import torch_geometric
import trimesh
import utils
import weight_transfer
from ddp_utils import convert_state_dict
from garment import Garment
from load_garments import (
    bones_placement_fps,
    get_boundary_verts,
    is_active,
    is_pinned,
    is_unpinned,
)
from scipy import spatial
from scipy.spatial import KDTree
from torch.profiler import profile as torch_profile, ProfilerActivity, record_function
from tqdm import tqdm
from training import dataio
from training.callbacks import Checkpoint, TensorBoardLogger
from units import physkin_hyperfixedbone_multigarments_DDP as main_unit


def setup_ddp(rank, world_size):
    print()
    print(f"!!! Preparing device CUDA:{rank} for DDP !!!")
    print()
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "19728"
    os.environ["WORLD_SIZE"] = f"{world_size}"
    timeout_value = datetime.timedelta(hours=12)
    torch.distributed.init_process_group(
        "nccl", rank=rank, world_size=world_size, timeout=timeout_value
    )
    torch.cuda.set_device(rank)


def get_ddp_model(model, rank):
    from ddp_utils import get_ddp_model as _get_ddp_model

    return _get_ddp_model(model, rank)


def prepare_and_train_unit(  # noqa: C901
    device,
    ddp_enabled,
    world_size,
    scene,
    config,
    cfg,
    num_new_garment_bones,
    bones_lod_levels,
    output_path,
    epoch_dir,
    all_pose_dataset,
):
    # create separate log file for each device
    log_filepath = os.path.join(output_path, f"log_inference_{device}.txt")
    logging.basicConfig(
        filename=log_filepath,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logging.info(
        f"--------- Process started: rank={device}, pid={os.getpid()} ---------"
    )

    # define checkpoinitng callback before intializing process group
    _ckpt_callback = Checkpoint(  # noqa: F841
        dirpath=output_path,
        save_every_n_epochs=cfg.train.checkpoint_every_n_epochs,
    )

    if ddp_enabled:
        # intialize process group
        setup_ddp(device, world_size)

    # create LBS object
    physkin_lbs = lbs.LBS.from_scene_and_config(scene, config)

    n_active_bones = num_new_garment_bones

    hparams = utils.build_hparams(cfg, physkin_lbs)

    # define input dimensions for different conditioning types
    pose_dim = physkin_lbs.config.rig.nActivePoseParams
    shape_dim = cfg.shape_hyper_net.input_dim
    input_dim = {
        "pose_vector": pose_dim,
        "shape_vector_enc": shape_dim,
        "shape_vector_dec": shape_dim,
        "pose_shape_enc": pose_dim + shape_dim,
        "pose_shape_dec": pose_dim + shape_dim,
    }
    prediction_dim = {
        "delta_posed_ortho6d": 9,
        "delta_unposed_ortho6d": 9,
        "delta_quaternion": 7,
    }
    L_freq = 6

    # define bone network
    model = None
    if cfg.model.type == "mlp":
        print("Using MLP for Bone Network...")
        model = Conv1dMLP(
            in_features=cfg.bone_encoder.out_dim + 3,
            hidden_features=cfg.model.hidden_features,
            hidden_layers=cfg.model.hidden_layers,
            out_features=9,
            softmax=False,
            channel_transpose=True,
        )
    elif cfg.model.type == "siren":
        print("Using SIREN for Bone Network...")
        model = mlp.SirenNet(
            dim_in=n_active_bones * 3,
            dim_hidden=cfg.model.hidden_features,
            num_layers=cfg.model.hidden_layers,
            dim_out=n_active_bones * prediction_dim[cfg.train.prediction_type],
            hidden_activation=nn.ReLU(),
            final_activation=nn.Identity(),
            w0_initial=1.0,
        )

    # define per-vertex network
    vertex_model = None
    _geometry_operators = None  # noqa: F841
    _static_vertex_features = None  # noqa: F841
    if cfg.vertex_model.type == "mlp":
        print("Using MLP for Vertex Network...")
        vertex_model = mlp.MLP(
            input_dim[cfg.train.input_type],
            cfg.vertex_model.hidden_features,
            cfg.vertex_model.hidden_layers,
            scene["initPos"].shape[0] * 3,
            softmax=False,
        )
    elif cfg.vertex_model.type == "siren":
        print("Using SIREN for Vertex Network...")
        vertex_model = mlp.SirenNetPerPoint(
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
        faces_tensor = torch.tensor(scene["faces"], dtype=torch.int32, device="cpu")
        vertex_tensor_normalized = diffusion_net.geometry.normalize_positions(
            vertex_tensor
        )
        _geometry_operators = diffusion_net.geometry.get_operators(  # noqa: F841
            verts=vertex_tensor_normalized, faces=faces_tensor, k_eig=1024
        )
        _static_vertex_features = vertex_tensor_normalized.to(device)  # noqa: F841
        vertex_model = diffusion_net.layers.DiffusionNet(
            C_in=in_feat_dim,
            C_out=out_feat_dim,
            C_width=cfg.vertex_model.hidden_features,  # internal size of the diffusion net. 32 -- 512 is a reasonable range
            N_block=cfg.vertex_model.hidden_layers,
            last_activation=None,  # apply a last softmax to outputs
            outputs_at="global_mean",
            diffusion_method="spectral",
        )

    # define hyper networks
    pose_hyper_net = mlp.SirenNet(
        dim_in=input_dim[cfg.pose_hyper_net.input_type],
        dim_hidden=cfg.pose_hyper_net.hidden_features,
        num_layers=cfg.pose_hyper_net.hidden_layers,
        dim_out=cfg.model.hidden_layers * cfg.model.hidden_features,
        final_activation=nn.Identity(),
        w0_initial=1.0,
    )
    shape_hyper_net = mlp.ShapeMLP(
        input_dim[cfg.shape_hyper_net.input_type],
        cfg.shape_hyper_net.hidden_features,
        cfg.shape_hyper_net.hidden_layers,
        cfg.pose_hyper_net.hidden_layers
        * cfg.pose_hyper_net.hidden_features,  # to predict the weights of the bone network
        input_channel_dim=cfg.shape_hyper_net.input_channels,
        softmax=False,
        batch_norm=True,
        garment_embedding_dim=cfg.garment_encoder.out_dim,
    )

    garment_encoder = mlp.GarmentMLP(
        cfg.garment_encoder.input_dim,
        cfg.garment_encoder.hidden_features,
        cfg.garment_encoder.hidden_layers,
        # cfg.pose_hyper_net.hidden_layers * cfg.pose_hyper_net.hidden_features,
        cfg.garment_encoder.out_dim,
        input_channel_dim=cfg.garment_encoder.input_channels,
        softmax=False,
        batch_norm=True,
        use_graph_conv=True,
        graph_in_feature_dim=4
        + num_new_garment_bones,  # XYZ + Pinning Label + Skin Weights
        use_pattern_embeddings=cfg.garment_encoder.use_pattern_embeddings,
    )

    # load from checkpoint
    if cfg.train.continue_from_last_checkpoint:
        shape_hyper_net.load_state_dict(
            torch.load(
                f"{output_path}/{cfg.train.resume_step}/shape_hyper_net.pth",
                map_location="cpu",
                weights_only=True,
            )
        )
        pose_hyper_net.load_state_dict(
            torch.load(
                f"{output_path}/{cfg.train.resume_step}/pose_hyper_net.pth",
                map_location="cpu",
                weights_only=True,
            )
        )
        garment_encoder.load_state_dict(
            torch.load(
                f"{output_path}/{cfg.train.resume_step}/garment_encoder.pth",
                map_location="cpu",
                weights_only=True,
            )
        )
        model.load_state_dict(
            torch.load(
                f"{output_path}/{cfg.train.resume_step}/module.pth",
                map_location="cpu",
                weights_only=True,
            )
        )
        vertex_model.load_state_dict(
            torch.load(
                f"{output_path}/{cfg.train.resume_step}/vertex_model.pth",
                map_location="cpu",
                weights_only=True,
            )
        )

    if ddp_enabled:
        model = get_ddp_model(model, device)
        vertex_model = get_ddp_model(vertex_model, device)
        pose_hyper_net = get_ddp_model(pose_hyper_net, device)
        shape_hyper_net = get_ddp_model(shape_hyper_net, device)
        garment_encoder = get_ddp_model(garment_encoder, device)
    else:
        model = model.to(device)
        vertex_model = vertex_model.to(device)
        pose_hyper_net = pose_hyper_net.to(device)
        shape_hyper_net = shape_hyper_net.to(device)
        garment_encoder = garment_encoder.to(device)

        # load from checkpoint
    garment_encoder_ckpt_dict = torch.load(
        f"{output_path}/{epoch_dir}/garment_encoder.pth",
        map_location="cpu",
        weights_only=True,
    )
    # load from checkpoint
    pose_hyper_net_ckpt_dict = torch.load(
        f"{output_path}/{epoch_dir}/pose_hyper_net.pth",
        map_location="cpu",
        weights_only=True,
    )
    shape_hyper_net_ckpt_dict = torch.load(
        f"{output_path}/{epoch_dir}/shape_hyper_net.pth",
        map_location="cpu",
        weights_only=True,
    )
    model_ckpt_dict = torch.load(
        f"{output_path}/{epoch_dir}/module.pth",
        map_location="cpu",
        weights_only=True,
    )
    vertex_model_ckpt_dict = torch.load(
        f"{output_path}/{epoch_dir}/vertex_model.pth",
        map_location="cpu",
        weights_only=True,
    )

    if not ddp_enabled:
        garment_encoder_ckpt_dict = convert_state_dict(
            garment_encoder_ckpt_dict, ddp_enabled
        )
        pose_hyper_net_ckpt_dict = convert_state_dict(
            pose_hyper_net_ckpt_dict, ddp_enabled
        )
        shape_hyper_net_ckpt_dict = convert_state_dict(
            shape_hyper_net_ckpt_dict, ddp_enabled
        )
        model_ckpt_dict = convert_state_dict(model_ckpt_dict, ddp_enabled)
        vertex_model_ckpt_dict = convert_state_dict(vertex_model_ckpt_dict, ddp_enabled)

    garment_encoder.load_state_dict(garment_encoder_ckpt_dict)
    pose_hyper_net.load_state_dict(pose_hyper_net_ckpt_dict)
    shape_hyper_net.load_state_dict(shape_hyper_net_ckpt_dict)
    model.load_state_dict(model_ckpt_dict)
    vertex_model.load_state_dict(vertex_model_ckpt_dict)

    tb_logger = TensorBoardLogger(output_path)

    unit = main_unit.PhysRigUnit(
        module=model,
        pose_hyper_net=pose_hyper_net,
        shape_hyper_net=shape_hyper_net,
        garment_encoder=garment_encoder,
        # geometry_operators=geometry_operators,
        # static_vertex_features=static_vertex_features,
        device=device,
        lbs=physkin_lbs,
        # active_bones=torch.tensor(active_bones),
        # vertex_bone_selection_mask=torch.tensor(vertex_bone_selection_mask),
        bones_lod_levels=bones_lod_levels,
        # bones_lod_sampled=bones_lod_sampled,
        # bones_rest_transforms=bone_rest_transforms,
        body_bones_weights=torch.Tensor(scene["bodyBoneWeights"]),
        default_body_rest_verts=torch.Tensor(scene["bodyPos"]),
        # vertex_normals=torch.Tensor(reference_vertex_normals),
        # garment_faces=torch.Tensor(scene['faces']),
        vertex_model=vertex_model,
        optimize_skinning_weights_start_epoch=cfg.train.optimize_skinning_weights_start_epoch,
        vertex_model_start_epoch=cfg.train.vertex_model_start_epoch,
        prediction_type=cfg.train.prediction_type,
        vertex_offset_type=cfg.train.vertex_offset_type,
        model_input_type=cfg.train.input_type,
        log_every_n_steps=cfg.train.log_every_n_steps,
        tb_logger=tb_logger,
        debug_path=cfg.train.debug_path,
        ckpt_path=output_path,
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
        ddp_enabled=ddp_enabled,
    )

    if device == 0:
        logging.info(f" ------ [ Inference on CUDA:{device} only !!! ] ----")

        msg_start = f"""
        :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                            Started inference on CUDA:{device}
        :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        """
        print(msg_start)
        logging.info(msg_start)

        # pytorch profiler
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities += [ProfilerActivity.CUDA]

        shape_indices = [5, 6, 18]

        # INFERENCE
        for garment_idx in tqdm(range(len(all_pose_dataset.garments_metadata))):
            for shape_idx in shape_indices:
                if cfg.train.dataset_type == "shapes":
                    skinning_output = unit.inference(
                        all_pose_dataset,
                        only_skinning=True,
                        gt_given=False,
                        use_vertex_model=False,
                        fixed_shape_index=shape_idx,
                        garment_index=garment_idx,
                    )
                    bones_output = unit.inference(
                        all_pose_dataset,
                        only_skinning=False,
                        gt_given=False,
                        use_vertex_model=False,
                        fixed_shape_index=shape_idx,
                        garment_index=garment_idx,
                    )
                    with torch_profile(
                        activities=activities, record_shapes=True
                    ) as prof:
                        with record_function("full_model_inference"):
                            bones_vertex_output = unit.inference(
                                all_pose_dataset,
                                only_skinning=False,
                                gt_given=False,
                                use_vertex_model=True,
                                fixed_shape_index=shape_idx,
                                garment_index=garment_idx,
                            )

                msg_end = f"""
                :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                                    Completed inference on CUDA:{device}
                :::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                """
                print(msg_end)
                logging.info(msg_end)

                if cfg.infer.visualize_inference:
                    garment_scene = skinning_output["garment_scene"]
                    garment_dirpath = skinning_output["garment_dirpath"]
                    uv_pattern_img = cv2.imread(
                        glob(garment_dirpath + "/*_texture.png")[0]
                    )

                    all_pos_b = skinning_output["pos_b"].cpu().detach().numpy()
                    all_pos_c = (
                        torch.Tensor(bones_vertex_output["pos"]).to("cpu").numpy()
                    )
                    _all_pos_c_gt = (  # noqa: F841
                        torch.Tensor(bones_vertex_output["gt_pos"]).to("cpu").numpy()
                    )
                    all_pos_bones_c = (
                        torch.Tensor(bones_vertex_output["affine_states_cloth"])
                        .to("cpu")
                        .numpy()[:, :, :3, 3]
                    )
                    all_pos_c_skinning_only = (
                        torch.Tensor(skinning_output["pos"]).to("cpu").numpy()
                    )
                    all_pos_bones_c_skinning_only = (
                        torch.Tensor(skinning_output["affine_states_cloth"])
                        .to("cpu")
                        .numpy()[:, :, :3, 3]
                    )
                    all_pos_c_low_freq = (
                        torch.Tensor(bones_output["pos"]).to("cpu").numpy()
                    )

                    # for headless rendering
                    video_data_path = cfg.train.video_data_path
                    os.makedirs(video_data_path, exist_ok=True)

                    data_dict = {}
                    data_dict["faces_b"] = garment_scene["bodyFaces"]
                    data_dict["faces_c"] = garment_scene["faces"]
                    data_dict["verts_b"] = all_pos_b
                    data_dict["verts_c_SkinningInit"] = all_pos_c_skinning_only
                    data_dict["verts_c_BoneNet"] = all_pos_c_low_freq
                    data_dict["verts_c_VertexNet"] = all_pos_c
                    data_dict["uv_w_c"] = garment_scene["texCoords"][:, :2]
                    data_dict["cloth_bones_positions_final"] = all_pos_bones_c
                    data_dict["cloth_bones_positions_SkinningInit"] = (
                        all_pos_bones_c_skinning_only
                    )
                    data_dict["uv_image"] = uv_pattern_img
                    data_dict["garment_dirpath"] = garment_dirpath

                    save_path = f"{video_data_path}//data_dict_shape{shape_idx}_garment{garment_idx}.npz"
                    np.savez(save_path, **data_dict)
                    print(f"Saved visualization data to ---> {save_path}")

                    # export experiment name for rendering
                    with open("temp_env_variables.sh", "w") as f:
                        f.write(f'export INFER_EXP_NAME="{cfg.EXP_NAME}"\n')

    if cfg.infer.profiling:
        print(prof.key_averages().table(sort_by="cpu_time_total"))


###### HELPER FUNCTIONS FOR MAIN #######


def _resolve_checkpoint_paths(cfg):
    """Resolve checkpoint directory paths and return (output_path, epoch_dir)."""
    training_snapshots_dir = utils.build_experiment_path(cfg)
    timestamp_list = natsort.natsorted(os.listdir(training_snapshots_dir))
    timestamp_list = [
        ts for ts in timestamp_list if ts.startswith(cfg.train.timestamp_prefix)
    ]
    assert len(timestamp_list) > 0
    timestamp_dir = cfg.train.timestamp
    if cfg.train.timestamp == "latest":
        timestamp_dir = timestamp_list[-1]
        cfg.train.timestamp = timestamp_dir
    epoch_list = natsort.natsorted(
        os.listdir(os.path.join(training_snapshots_dir, timestamp_dir))
    )
    epoch_list = [ep for ep in epoch_list if ep.startswith("epoch_")]
    assert len(epoch_list) > 0
    epoch_dir = cfg.train.resume_step
    if cfg.train.resume_step == "latest":
        epoch_dir = epoch_list[-1]
        cfg.train.resume_step = epoch_dir
    output_path = utils.build_output_path(
        training_snapshots_dir,
        timestamp_dir,
        should_continue=cfg.train.continue_from_last_checkpoint,
        force_overwrite=cfg.train.force_overwrite,
        mode="test",
    )
    print()
    print(f"Loading checkpoints from --> {output_path} || {epoch_dir}")
    print()
    return output_path, epoch_dir


def _get_garment_paths(cfg, output_path):
    """Load garment paths filtered by validation indices and max count."""
    _train_garment_indices = np.load(  # noqa: F841
        os.path.join(output_path, "train_garment_indices.npy")
    )
    val_garment_indices = np.load(os.path.join(output_path, "val_garment_indices.npy"))

    garment_root_dir = cfg.garment.root_dir
    garment_paths = natsort.natsorted(glob(os.path.join(garment_root_dir, "*")))
    garment_paths = [garment_paths[i] for i in val_garment_indices]

    max_num_garments = cfg.infer.max_num_garments
    if max_num_garments > 0:
        garment_paths = garment_paths[:max_num_garments]
    return garment_paths


def _load_body_mesh(cfg, scene):
    """Load and process the reference body mesh, updating scene in place.

    Returns body_waist_verts.
    """
    _reference_body_name = cfg.body.custom_obj_path.split("/")[-1].split(".")[0]  # noqa: F841
    ms_io_b = pml.MeshSet()
    ms_io_b.load_new_mesh(cfg.body.custom_obj_path)
    ms_io_b.apply_normal_normalization_per_face()
    ms_io_b.apply_normal_normalization_per_vertex()
    if cfg.body.invert_face_normals:
        ms_io_b.meshing_invert_face_orientation()
    if cfg.body.invert_vertex_normals:
        ms_io_b.compute_normal_by_function_per_vertex(
            x="-nx", y="-ny", z="-nz", onselected=False
        )
    custom_body_mesh = ms_io_b.current_mesh()
    custom_body_verts = custom_body_mesh.vertex_matrix()
    custom_body_faces = custom_body_mesh.face_matrix()
    nn_blended_body_wts, _ = weight_transfer.copy_weights_via_knn(
        sourceV=scene["bodyPos"],
        sourceW=scene["bodyBoneWeights"].T,
        targetV=custom_body_verts,
        nn=16,
    )
    custom_body_bone_weights = nn_blended_body_wts.T
    scene["bodyPos"] = custom_body_verts
    scene["bodyFaces"] = custom_body_faces
    scene["bodyBoneWeights"] = custom_body_bone_weights
    body_verts_tree = KDTree(scene["bodyPos"])
    rigid_labels = ["_lhand", "_rhand", "_lfoot", "_rfoot"]
    for idx in range(len(rigid_labels)):
        label = rigid_labels[idx]
        ref_idx = cfg.body.rigid_verts_parent_idx[idx]
        sub_verts = trimesh.load(
            cfg.body.custom_obj_path[:-4] + f"{label}.obj"
        ).vertices
        dist, nn_indices = body_verts_tree.query(sub_verts, k=1)
        ref_wt = custom_body_bone_weights[:, ref_idx].reshape(-1, 1)
        ref_wts = np.repeat(ref_wt, nn_indices.shape[0], axis=-1)
        scene["bodyBoneWeights"][:, nn_indices.flatten()] = ref_wts
    body_waist_verts = trimesh.load(
        cfg.body.custom_obj_path[:-4] + "_waist_verts.obj"
    ).vertices
    return body_waist_verts


def _load_garment_mesh(cfg, scene, custom_garment_dirpath):
    """Load and process the reference garment mesh, updating scene in place.

    Returns (reference_garment_mesh, reference_verts, reference_faces,
             reference_vertex_normals, reference_garment_name, reference_pinned_bones).
    """
    reference_garment_mesh_path = custom_garment_dirpath + ""
    if cfg.train.panels_as_rest:
        # reference_garment_mesh_path = custom_garment_dirpath + '/Configured_design_3D/Configured_design_3D_sim.obj'
        reference_garment_mesh_path = glob(custom_garment_dirpath + "/*_sim.obj")[0]
    reference_garment_name = reference_garment_mesh_path[
        len(custom_garment_dirpath) + 1 : -len("_sim.obj")
    ]
    reference_pinned_bones = np.array(list(cfg.garment.pinned_verts_indices))
    ms_io_c = pml.MeshSet()
    ms_io_c.load_new_mesh(reference_garment_mesh_path)
    _geometric_measures = ms_io_c.get_geometric_measures()  # noqa: F841
    ms_io_c.apply_normal_normalization_per_face()
    ms_io_c.apply_normal_normalization_per_vertex()
    if cfg.garment.invert_face_normals:
        ms_io_c.meshing_invert_face_orientation()
    if cfg.garment.invert_vertex_normals:
        ms_io_c.compute_normal_by_function_per_vertex(
            x="-nx", y="-ny", z="-nz", onselected=False
        )
    reference_garment_mesh = ms_io_c.current_mesh()
    reference_verts = reference_garment_mesh.vertex_matrix()
    reference_faces = reference_garment_mesh.face_matrix()
    reference_vertex_normals = reference_garment_mesh.vertex_normal_matrix()
    scene["initPos"] = reference_verts
    _num_garment_verts = scene["initPos"].shape[0]  # noqa: F841
    scene["faces"] = reference_faces
    tex_coords = np.zeros(
        (3 * len(reference_faces), 3)
    )  # ( 3 x number of faces, 2) ---> appending zeros in the third column to make it 3D --> (3 x number of faces, 3) (used for rest positions calculation)
    tex_coords[:, :2] = reference_garment_mesh.wedge_tex_coord_matrix()
    scene["texCoords"] = tex_coords
    return (
        reference_garment_mesh,
        reference_verts,
        reference_faces,
        reference_vertex_normals,
        reference_garment_name,
        reference_pinned_bones,
    )


def _load_panels_as_rest(scene, custom_garment_dirpath, reference_verts):
    """Load panel meshes and compute rest positions for the garment.

    Returns (panels_verts, panels_faces, seam_verts_indices, rest_pos).
    """
    print("Using panels as rest positions...")
    # panels_mesh_path = custom_garment_dirpath + '/Configured_design_3D/Configured_design_3D_boxmesh.obj'
    panels_mesh_path = glob(custom_garment_dirpath + "/*_boxmesh.obj")[0]
    # load the panels mesh
    ms_io_panels = pml.MeshSet()
    ms_io_panels.load_new_mesh(panels_mesh_path)
    panels_mesh = ms_io_panels.current_mesh()
    panels_verts = panels_mesh.vertex_matrix()
    panels_faces = panels_mesh.face_matrix()
    # estimate stitching vertices
    # stitch_seam_indices_path = custom_garment_dirpath + '/Configured_design_3D/Configured_design_3D_sim_stitch.npy'
    stitch_seam_indices_path = glob(custom_garment_dirpath + "/*_stitch.npy")[0]
    seam_verts_indices = np.load(stitch_seam_indices_path)
    # Find faces that contain any of the seam vertices
    face_contains_seam_vertex = np.any(
        np.isin(panels_faces, seam_verts_indices), axis=1
    )
    rest_pos = panels_verts[panels_faces]
    rest_pos[face_contains_seam_vertex] = reference_verts[
        panels_faces[face_contains_seam_vertex]
    ]
    rest_pos = rest_pos.reshape(-1, 3)
    scene["restPos"] = rest_pos
    _ = trimesh.PointCloud(rest_pos).export(f"{custom_garment_dirpath}/rest.obj")
    return panels_verts, panels_faces, seam_verts_indices, rest_pos


def _estimate_pinning(
    cfg, scene, max_num_random_bones, reference_pinned_bones, custom_garment_dirpath
):
    """Estimate pinning vertices and bones.

    Returns (verts_pinning_mask, random_pinned_bones_indices, reference_pinned_bones).
    """
    # estimate pinning vertices
    verts_pinning_mask = np.array([False] * len(scene["initPos"]))
    if cfg.train.estimate_pinning_vertices:
        print("Estimating pinning vertices...")
        mesh_set = pml.MeshSet()
        mesh_set.add_mesh(pml.Mesh(scene["initPos"], scene["faces"]), "cloth")
        boundary_verts = get_boundary_verts(mesh_set, filter_top_by_y=True)
        tree = spatial.KDTree(np.array(scene["initPos"]))
        dist, idx = tree.query(boundary_verts, k=1, p=2)
        verts_pinning_mask[idx] = True

    # estimate pinning bones
    random_pinned_bones_indices = []
    if cfg.train.estimate_pinning_bones:
        print("Estimating pinning bones...")
        num_random_pinned_bones = int(np.sqrt(max_num_random_bones))
        random_pinned_bones_indices.append(int(scene["initPos"][:, 1].argmax()))
        terminating_length = num_random_pinned_bones
        solver = pp3d.MeshHeatMethodDistanceSolver(scene["initPos"], scene["faces"])
        _n_points = len(scene["initPos"])  # noqa: F841
        min_distances = solver.compute_distance(random_pinned_bones_indices[0])
        min_distances[~verts_pinning_mask] = 0.0
        # Select remaining points
        while len(random_pinned_bones_indices) < terminating_length:
            # Set distances of already selected points to -1
            min_distances[random_pinned_bones_indices] = -1
            # Find the point with maximum distance
            max_idx = np.argmax(min_distances)
            random_pinned_bones_indices.append(max_idx)
            # Update distances with the new point
            distances = solver.compute_distance(max_idx)
            distances[~verts_pinning_mask] = 0.0
            min_distances = np.minimum(min_distances, distances)
        random_pinned_bones_indices = np.array(random_pinned_bones_indices)
        if len(reference_pinned_bones) == 0:
            reference_pinned_bones = random_pinned_bones_indices
        trimesh.PointCloud(scene["initPos"][random_pinned_bones_indices]).export(
            f"{custom_garment_dirpath}/random_pinned_bones.obj"
        )
    return verts_pinning_mask, random_pinned_bones_indices, reference_pinned_bones


def _add_garment_bones(
    cfg, scene, reference_pinned_bones, reference_garment_name, max_num_random_bones
):
    """Add garment and pinned bones to the scene.

    Returns (num_pinned_bones, num_new_garment_bones, active_bones, pinned_bones,
             unpinned_active_bones, n_active_bones, bones_pinning_mask,
             pinned_bone_indices, vertex_bone_selection_mask).
    """
    num_pinned_bones = len(reference_pinned_bones)
    num_new_garment_bones = scene["initPos"].shape[
        0
    ]  # by default all garment vertices are bones
    if cfg.train.random_bones_placement:
        num_new_garment_bones = max_num_random_bones + num_pinned_bones
    scene["boneNames"] = np.append(
        scene["boneNames"],
        np.array(
            [f"garment_{i}" for i in range(num_new_garment_bones - num_pinned_bones)]
        ),
    )
    scene["boneNames"] = np.append(
        scene["boneNames"],
        np.array([f"pinned_{i}" for i in range(num_pinned_bones)]),
    )
    scene["faceGarmentName"] = np.array(
        [reference_garment_name] * scene["faces"].shape[0]
    )
    scene["bonePos"] = np.append(
        scene["bonePos"], np.zeros((num_new_garment_bones, 3)), axis=0
    )
    scene["bodyBoneWeights"] = np.append(
        scene["bodyBoneWeights"],
        np.zeros((num_new_garment_bones, scene["bodyBoneWeights"].shape[1])),
        axis=0,
    )
    scene["boneTransform"] = np.append(
        scene["boneTransform"],
        np.array([np.eye(3)] * num_new_garment_bones),
        axis=0,
    )
    scene["boneParentIndices"] = np.append(
        scene["boneParentIndices"], np.array([-1] * num_new_garment_bones), axis=0
    )
    scene["activeBones"] = np.append(
        scene["activeBones"], np.array([True] * num_new_garment_bones), axis=0
    )
    scene["clothBoneWeights"] = np.zeros(
        (scene["boneNames"].shape[0], scene["initPos"].shape[0])
    )

    active_bones = np.array([is_active(bone) for bone in scene["boneNames"]])
    pinned_bones = np.array(
        [is_pinned(bone) for bone in scene["boneNames"][active_bones]]
    )
    unpinned_active_bones = np.array([is_unpinned(bone) for bone in scene["boneNames"]])
    n_active_bones = sum(active_bones)

    bones_pinning_mask = np.array([False] * n_active_bones)
    bones_pinning_mask[pinned_bones] = True
    active_indices = np.where(active_bones)[0]
    pinned_bone_indices = active_indices[bones_pinning_mask]
    vertex_bone_selection_mask = np.array([False] * len(scene["initPos"]))

    return (
        num_pinned_bones,
        num_new_garment_bones,
        active_bones,
        pinned_bones,
        unpinned_active_bones,
        n_active_bones,
        bones_pinning_mask,
        pinned_bone_indices,
        vertex_bone_selection_mask,
    )


def _place_bones_and_compute_pinning(
    cfg,
    scene,
    reference_pinned_bones,
    bones_lod_levels,
    max_num_random_bones,
    num_pinned_bones,
    active_bones,
    n_active_bones,
    unpinned_active_bones,
    pinned_bone_indices,
    vertex_bone_selection_mask,
    custom_garment_dirpath,
    body_waist_verts,
    panels_verts,
):
    """Place custom garment bones and compute pinned bone body indices.

    Returns (vertex_bone_selection_mask, bones_lod_sampled, sampled_panel_points,
             pinned_bones_bodyverts_idx).
    """
    bones_lod_sampled = None
    sampled_panel_points = None
    pinned_bones_bodyverts_idx = np.array([-1])

    scene["boneTransform"][active_bones] = np.array([np.eye(3)] * n_active_bones)
    tree = KDTree(scene["initPos"])
    dist, nn_indices = tree.query(scene["bonePos"][active_bones], k=1)
    scene["bonePos"][active_bones] = scene["initPos"][nn_indices]

    if cfg.train.random_bones_placement:
        print("Randomly placing garment bones...")
        bone_indices_init = reference_pinned_bones.tolist()
        bones_lod_sampled = bones_placement_fps(
            scene["initPos"],
            scene["faces"],
            bone_indices_sampled=bone_indices_init.copy(),
            bone_lods=bones_lod_levels,
        )
        np.save(f"{custom_garment_dirpath}/bones_lod_sampled.npy", bones_lod_sampled)
        bone_indices_sampled = bones_lod_sampled[f"{max_num_random_bones}"]
        vertex_bone_selection_mask[bone_indices_sampled] = True
        bone_indices_sampled = bone_indices_sampled[
            num_pinned_bones:
        ]  # removing pinned bones from the sampled list (adding them separately below)
        sampled_points = scene["initPos"][bone_indices_sampled]
        if cfg.train.panels_as_rest:
            sampled_panel_points = panels_verts[bone_indices_sampled]
        # update bone positions
        scene["bonePos"][unpinned_active_bones] = sampled_points

    # use all vertices as garment bones
    else:
        scene["bonePos"][active_bones] = scene["initPos"]

    # add pinned bones
    if len(reference_pinned_bones) > 0:
        pinned_bone_pos = scene["initPos"][reference_pinned_bones]
        scene["bonePos"][pinned_bone_indices] = pinned_bone_pos
        # estimate pinning bone positions-indices on body
        bodytree = spatial.KDTree(scene["bodyPos"])
        _, waist_idx = bodytree.query(body_waist_verts)
        waisttree = spatial.KDTree(body_waist_verts)
        _, pin_idx = waisttree.query(pinned_bone_pos)
        pinned_bones_bodyverts_idx = waist_idx[pin_idx]
        np.save(
            f"{custom_garment_dirpath}/pinned_bones_bodyverts_idx.npy",
            pinned_bones_bodyverts_idx,
        )
        pinned_bones_bodyverts_idx = torch.Tensor(pinned_bones_bodyverts_idx).long()
        trimesh.PointCloud(scene["bodyPos"][pinned_bones_bodyverts_idx]).export(
            f"{custom_garment_dirpath}/body_pinned_bones.obj"
        )

    return (
        vertex_bone_selection_mask,
        bones_lod_sampled,
        sampled_panel_points,
        pinned_bones_bodyverts_idx,
    )


def _compute_geodesic_skin_weights(
    cfg, scene, active_bones, cloth_joint_positions, custom_garment_dirpath
):
    """Compute geodesic-based skin weights and update scene cloth bone weights.

    Returns (cloth_bones_weights, geodesic_distances).
    """
    # weight_mask = np.array([ (x.startswith('legLower') or x.startswith('legUpper') or x.startswith('knee') or x.startswith('pelvis')) for x in scene['boneNames'] ])
    weight_mask = np.array([True] * len(scene["boneNames"]))
    cloth_verts_weights, _ = weight_transfer.transfer_weights_via_rbf(
        sourceV=scene["bodyPos"],
        sourceW=scene["bodyBoneWeights"],
        targetV=scene["initPos"],
        source_mask=weight_mask,
    )
    _, indices = weight_transfer.copy_weights_via_knn(
        sourceV=scene["initPos"],
        sourceW=cloth_verts_weights,
        targetV=cloth_joint_positions,
        nn=1,
    )
    cloth_bones_weights, _ = weight_transfer.copy_weights_via_knn(
        sourceV=scene["bodyPos"],
        sourceW=scene["bodyBoneWeights"].T,
        targetV=cloth_joint_positions,
        nn=1,
    )
    solver = pp3d.MeshHeatMethodDistanceSolver(scene["initPos"], scene["faces"])
    vertex_gdistances = []
    vertex_gdist_cache_path = os.path.join(
        custom_garment_dirpath, "vertex_gdist_cached.npy"
    )
    if os.path.exists(vertex_gdist_cache_path):
        print("Using cached geodesic distances...")
        vertex_gdistances = np.load(vertex_gdist_cache_path)
    else:
        print("Calculating geodesic distances...")
        for idx in tqdm(range(len(scene["initPos"]))):
            gdist = solver.compute_distance(idx)
            vertex_gdistances.append(gdist)
        vertex_gdistances = np.array(vertex_gdistances)
        np.save(vertex_gdist_cache_path, vertex_gdistances)
    # vertex_gdistances = vertex_gdistances / vertex_gdistances.sum(axis=0, keepdims=True)
    # vertex_gdistances = (vertex_gdistances - vertex_gdistances.min()) / (vertex_gdistances.max() - vertex_gdistances.min())
    geodesic_distances = vertex_gdistances[indices]
    # geodesic_wts = geodesic_distances.max() - geodesic_distances
    # geodesic_wts = 1.0 / ((geodesic_distances)**2 + 1e-7)
    spread = cfg.train.skinwt_spread
    sigma = np.sqrt(scene["initPos"].shape[0] / len(geodesic_distances))
    geodesic_wts = 1.0 / np.exp(geodesic_distances / (spread * sigma))
    # geodesic_wts = np.nan_to_num(geodesic_wts, 0.0)
    # geodesic_wts = geodesic_wts / geodesic_wts.sum(axis=0, keepdims=True)
    scene["clothBoneWeights"][~active_bones, :] = 0.0
    scene["clothBoneWeights"][active_bones] = geodesic_wts
    scene["clothBoneWeights"] = scene["clothBoneWeights"] / scene[
        "clothBoneWeights"
    ].sum(axis=0, keepdims=True)
    scene["clothBoneWeights"] = np.nan_to_num(scene["clothBoneWeights"], 0.0)
    print("Done!!!")
    return cloth_bones_weights, geodesic_distances


def _correct_skin_weights(scene, active_bones, verts_pinning_mask, pinned_bone_indices):
    """Apply skin weight corrections for pinned bones and vertices."""
    ## SKIN WEIGHTS CORRECTION (not required)
    # assign low weights for pinned bones, but only for non-pinned vertices
    pinned_verts_wts = scene["clothBoneWeights"][:, verts_pinning_mask]
    scene["clothBoneWeights"][pinned_bone_indices] = 1e-17
    scene["clothBoneWeights"][~active_bones, :] = 0.0
    scene["clothBoneWeights"][:, verts_pinning_mask] = (
        pinned_verts_wts  # copy back weights for pinned vertices
    )
    scene["clothBoneWeights"] = scene["clothBoneWeights"] / scene[
        "clothBoneWeights"
    ].sum(axis=0, keepdims=True)
    scene["clothBoneWeights"] = np.nan_to_num(scene["clothBoneWeights"], 0.0)


def _visualize_skinning(
    cfg,
    scene,
    active_bones,
    geodesic_distances,
    bones_lod_levels,
    num_pinned_bones,
    custom_garment_dirpath,
    panels_verts,
    sampled_panel_points,
):
    """Generate skinning visualizations for each LOD level."""
    # skinning visualization
    np.random.seed(10000)
    skinning_vis_path = custom_garment_dirpath + "/skinning_visdir"
    os.makedirs(skinning_vis_path, exist_ok=True)
    colorwts = scene["clothBoneWeights"][active_bones]
    spread = cfg.train.skinwt_spread
    sigma = np.sqrt(scene["initPos"].shape[0] / len(geodesic_distances))
    geodesic_wts = np.log(1.0 / (colorwts + 1e-6)) * (spread * sigma)
    random_bone_colors = np.random.rand(geodesic_wts.shape[0], 3) * 255
    for lod_i in bones_lod_levels:
        curr_bones_num = num_pinned_bones + lod_i
        sigma_i = np.sqrt(scene["initPos"].shape[0] / curr_bones_num)
        colorwts_i = geodesic_wts[:lod_i, :]
        colorwts_i = 1.0 / np.exp(colorwts_i / (spread * sigma_i))
        colorwts_i = colorwts_i / colorwts_i.sum(axis=0, keepdims=True)
        vertex_colors = (colorwts_i.T @ random_bone_colors[:lod_i]).astype(np.uint8)
        rgba_colors = np.ones((vertex_colors.shape[0], 4), dtype=np.uint8) * 255
        rgba_colors[:, :3] = vertex_colors
        trimesh.Trimesh(
            scene["initPos"], scene["faces"], vertex_colors=rgba_colors
        ).export(f"{skinning_vis_path}/{lod_i}_skinned_cloth.obj")
        rgba_bone_colors = np.ones((colorwts_i.shape[0], 4), dtype=np.uint8) * 255
        rgba_bone_colors[:, :3] = random_bone_colors[:lod_i].astype(np.uint8)
        trimesh.PointCloud(
            scene["bonePos"][active_bones][:lod_i], colors=rgba_bone_colors
        ).export(f"{skinning_vis_path}/{lod_i}_skinned_bones.obj")
        if cfg.train.panels_as_rest:
            trimesh.Trimesh(
                panels_verts, scene["faces"], vertex_colors=rgba_colors
            ).export(f"{skinning_vis_path}/{lod_i}_skinned_cloth_panels.obj")
            trimesh.PointCloud(sampled_panel_points, colors=rgba_bone_colors).export(
                f"{skinning_vis_path}/{lod_i}_skinned_bones_panels.obj"
            )


def _sample_garment_points(cfg, panels_verts, panels_faces):
    """Sample garment points using Poisson disk sampling."""
    # sample garment points
    mset = pml.MeshSet()
    sample_mesh = pml.Mesh(panels_verts, panels_faces)
    mset.add_mesh(sample_mesh, "cloth to sample")
    mset.generate_sampling_poisson_disk(
        samplenum=cfg.garment_encoder.num_samples + 500,
        radius=pml.PercentageValue(0.0),
        montecarlorate=20,
        savemontecarlo=False,
        approximategeodesicdistance=False,
        subsample=False,
        refineflag=False,
        refinemesh=0,
        bestsampleflag=True,
        bestsamplepool=True,
        exactnumflag=True,
        exactnumtolerance=0.0,
        radiusvariance=1,
    )
    sampled_points = mset.current_mesh().vertex_matrix()[
        : cfg.garment_encoder.num_samples
    ]
    return sampled_points


def _prepare_graph_data(
    reference_verts, reference_faces, verts_pinning_mask, scene, active_bones
):
    """Build graph data with edge connectivity, pinning labels, and skin weights."""
    # prepare graph data
    mesh_data = torch_geometric.utils.from_trimesh(
        trimesh.Trimesh(reference_verts, reference_faces)
    )
    faces_to_edges = torch_geometric.transforms.FaceToEdge()
    graph_data = faces_to_edges(mesh_data)
    pinning_labels = torch.Tensor(
        (1 - verts_pinning_mask).astype("float").reshape(-1, 1)
    )
    skin_wts = torch.Tensor(scene["clothBoneWeights"][active_bones]).transpose(0, 1)
    graph_data.x = torch.cat([graph_data.pos, pinning_labels, skin_wts], axis=-1)
    graph_data.pos = None
    return graph_data


def _build_garment_metadata(
    cfg,
    scene,
    config,
    custom_garment_dirpath,
    reference_verts,
    reference_faces,
    reference_vertex_normals,
    reference_garment_mesh,
    reference_garment_name,
    panels_verts,
    panels_faces,
    seam_verts_indices,
    rest_pos,
    verts_pinning_mask,
    bones_pinning_mask,
    random_pinned_bones_indices,
    pinned_bones_bodyverts_idx,
    pinned_bones,
    active_bones,
    unpinned_active_bones,
    cloth_joint_positions,
    vertex_bone_selection_mask,
    bones_lod_sampled,
    cloth_bones_weights,
    pattern_embeddings,
    sampled_points,
    graph_data,
):
    """Create and populate a Garment metadata object."""
    # initialize garment object
    garment_metadata = Garment(cfg, scene)
    garment_metadata.dirpath = custom_garment_dirpath
    garment_metadata.lbs = lbs.LBS.from_scene_and_config(scene, config)
    garment_metadata.type = "lower"
    garment_metadata.verts = reference_verts
    garment_metadata.faces = reference_faces
    garment_metadata.vertex_normals = reference_vertex_normals
    garment_metadata.tex_coords = reference_garment_mesh.wedge_tex_coord_matrix()
    garment_metadata.panels_verts = panels_verts
    garment_metadata.panels_faces = panels_faces
    garment_metadata.seam_verts_indices = seam_verts_indices
    garment_metadata.rest_pos = rest_pos
    garment_metadata.verts_pinning_mask = verts_pinning_mask
    garment_metadata.bones_pinning_mask = bones_pinning_mask
    garment_metadata.random_pinned_bones_indices = (
        random_pinned_bones_indices
        if len(random_pinned_bones_indices) > 0
        else np.array([-1])
    )
    garment_metadata.pinned_bones_bodyverts_idx = (
        pinned_bones_bodyverts_idx
        if len(pinned_bones_bodyverts_idx) > 0
        else np.array([-1])
    )
    garment_metadata.pinned_bones = (
        pinned_bones if len(pinned_bones) > 0 else np.array([-1])
    )
    garment_metadata.active_bones = active_bones
    garment_metadata.unpinned_active_bones = unpinned_active_bones
    garment_metadata.cloth_joint_positions = cloth_joint_positions
    garment_metadata.vertex_bone_selection_mask = vertex_bone_selection_mask
    garment_metadata.bones_lod_sampled = bones_lod_sampled
    garment_metadata.cloth_bones_weights = cloth_bones_weights
    garment_metadata.pattern_embeddings = pattern_embeddings[reference_garment_name]
    garment_metadata.sampled_points = sampled_points
    garment_metadata.graph_data = graph_data
    return garment_metadata


def _process_single_garment(
    cfg, scene, config, custom_garment_dirpath, device, pattern_embeddings
):
    """Process a single garment: load mesh, compute bones, weights, and metadata.

    Returns (garment_metadata, num_new_garment_bones, bones_lod_levels).
    """
    # load body mesh into scene
    body_waist_verts = _load_body_mesh(cfg, scene)

    # load garment mesh
    (
        reference_garment_mesh,
        reference_verts,
        reference_faces,
        reference_vertex_normals,
        reference_garment_name,
        reference_pinned_bones,
    ) = _load_garment_mesh(cfg, scene, custom_garment_dirpath)

    # load panels as rest positions
    panels_verts = None
    panels_faces = None
    seam_verts_indices = None
    rest_pos = None
    if cfg.train.panels_as_rest:
        panels_verts, panels_faces, seam_verts_indices, rest_pos = _load_panels_as_rest(
            scene, custom_garment_dirpath, reference_verts
        )

    # number of bones
    bones_lod_levels = cfg.train.bones_lod_levels
    max_num_random_bones = max(bones_lod_levels)

    # estimate pinning
    verts_pinning_mask, random_pinned_bones_indices, reference_pinned_bones = (
        _estimate_pinning(
            cfg,
            scene,
            max_num_random_bones,
            reference_pinned_bones,
            custom_garment_dirpath,
        )
    )

    # add garment bones
    (
        num_pinned_bones,
        num_new_garment_bones,
        active_bones,
        pinned_bones,
        unpinned_active_bones,
        n_active_bones,
        bones_pinning_mask,
        pinned_bone_indices,
        vertex_bone_selection_mask,
    ) = _add_garment_bones(
        cfg,
        scene,
        reference_pinned_bones,
        reference_garment_name,
        max_num_random_bones,
    )

    np.save(
        f"{custom_garment_dirpath}/vertex_bone_selection_mask.npy",
        vertex_bone_selection_mask,
    )

    # place bones and compute pinning
    bones_lod_sampled = None
    sampled_panel_points = None
    pinned_bones_bodyverts_idx = np.array([-1])
    if cfg.train.custom_garment_bones:
        (
            vertex_bone_selection_mask,
            bones_lod_sampled,
            sampled_panel_points,
            pinned_bones_bodyverts_idx,
        ) = _place_bones_and_compute_pinning(
            cfg,
            scene,
            reference_pinned_bones,
            bones_lod_levels,
            max_num_random_bones,
            num_pinned_bones,
            active_bones,
            n_active_bones,
            unpinned_active_bones,
            pinned_bone_indices,
            vertex_bone_selection_mask,
            custom_garment_dirpath,
            body_waist_verts,
            panels_verts,
        )

    joint_positions = scene["bonePos"]
    cloth_joint_positions = joint_positions[active_bones]

    # compute geodesic skin weights
    cloth_bones_weights = None
    geodesic_distances = None
    if cfg.garment.skin_weight_transfer_method == "geodesic":
        cloth_bones_weights, geodesic_distances = _compute_geodesic_skin_weights(
            cfg,
            scene,
            active_bones,
            cloth_joint_positions,
            custom_garment_dirpath,
        )

    cloth_bones_weights = torch.tensor(
        cloth_bones_weights.T, dtype=torch.float32, device=device
    )

    # correct skin weights
    _correct_skin_weights(scene, active_bones, verts_pinning_mask, pinned_bone_indices)

    # visualize skinning
    _visualize_skinning(
        cfg,
        scene,
        active_bones,
        geodesic_distances,
        bones_lod_levels,
        num_pinned_bones,
        custom_garment_dirpath,
        panels_verts,
        sampled_panel_points,
    )

    # sample garment points
    sampled_points = _sample_garment_points(cfg, panels_verts, panels_faces)

    # prepare graph data
    graph_data = _prepare_graph_data(
        reference_verts,
        reference_faces,
        verts_pinning_mask,
        scene,
        active_bones,
    )

    # build garment metadata
    garment_metadata = _build_garment_metadata(
        cfg,
        scene,
        config,
        custom_garment_dirpath,
        reference_verts,
        reference_faces,
        reference_vertex_normals,
        reference_garment_mesh,
        reference_garment_name,
        panels_verts,
        panels_faces,
        seam_verts_indices,
        rest_pos,
        verts_pinning_mask,
        bones_pinning_mask,
        random_pinned_bones_indices,
        pinned_bones_bodyverts_idx,
        pinned_bones,
        active_bones,
        unpinned_active_bones,
        cloth_joint_positions,
        vertex_bone_selection_mask,
        bones_lod_sampled,
        cloth_bones_weights,
        pattern_embeddings,
        sampled_points,
        graph_data,
    )

    return garment_metadata, num_new_garment_bones, bones_lod_levels


def _build_pose_dataset(cfg, physkin_lbs, GARMENTS_METADATA):
    """Build and configure the pose dataset based on dataset type and masking options."""
    all_pose_dataset = None
    num_samples = cfg.train.max_train_samples

    if cfg.train.dataset_type == "animation":
        all_pose_dataset = dataio.PoseAnimDataset(
            pose_dataset_path=cfg.train.poseanim_dataset_path,
            activePoseParamMask=physkin_lbs.config.rig.activePoseParamMask.to("cpu"),
            lbs=physkin_lbs,
            device_for_intersections="cpu",
            invert_transform=True,  # required for current animation sequence example
            spt=True,
            fixed_root_translation=True,
            fixed_root_rotation=False,
        )
        all_pose_dataset.all_data = all_pose_dataset.all_data[:num_samples, :]

    elif cfg.train.dataset_type == "shapes":
        shape_latent_type = "dec" if "dec" in cfg.train.input_type else "enc"
        all_pose_dataset = dataio.ShapePoseAnimDataset(
            pose_dataset_path=cfg.train.poseanim_dataset_path,
            shape_dataset_path=cfg.body.verts_data_path,
            shape_latent_type=shape_latent_type,
            activePoseParamMask=physkin_lbs.config.rig.activePoseParamMask.to("cpu"),
            lbs=physkin_lbs,
            spt=True,
            fixed_root_translation=True,
            fixed_root_rotation=False,
            shape_train_samplenum=cfg.train.first_k_shapes,
            global_normalization=False,
            load_garments=True,
            garments_metadata=GARMENTS_METADATA,
        )

    # sample_idx = 1
    # all_pose_dataset.all_data = all_pose_dataset.all_data[sample_idx:sample_idx+1,:]

    sample_start_idx = 20
    sample_end_idx = 30
    sample_indices_list = list(range(sample_start_idx, sample_end_idx))
    all_pose_dataset.all_data = all_pose_dataset.all_data[sample_indices_list]

    # num_samples = 30
    # all_pose_dataset.all_data = all_pose_dataset.all_data[:num_samples,:]

    _apply_pose_masking(cfg, physkin_lbs, all_pose_dataset)

    if cfg.train.canonical_pose_only:
        all_pose_dataset.all_data *= 0.0

    return all_pose_dataset


def _apply_pose_masking(cfg, physkin_lbs, all_pose_dataset):
    """Apply garment-specific pose masking to the dataset."""
    if not cfg.train.garment_specific_pose_masking:
        return

    if cfg.garment.class_type == "upper":
        # masking everything except the upper body
        def joint_to_mask(name):
            flag = (
                ("elbow" in name)
                or ("wrist" in name)
                or ("forearm" in name)
                or ("shoulder" in name)
                or ("clavicle" in name)
                or ("spine" in name)
            )
            return flag

        joint_names = (
            physkin_lbs.config.rig.rig.skeleton_parameter_transform.transform_names
        )
        joint_pose_mask = np.array([joint_to_mask(name) for name in joint_names])
        all_pose_dataset.all_data[:, ~joint_pose_mask] = 0.0

    elif cfg.garment.class_type == "lower":
        lower_mask = torch.load("lower_garment_mask.pt").cpu()
        lower_mask_val = torch.load("lower_garment_mask_val.pt").cpu()
        all_pose_dataset.all_data[:, lower_mask] = lower_mask_val


def _launch_inference(
    cfg,
    scene,
    config,
    num_new_garment_bones,
    bones_lod_levels,
    output_path,
    epoch_dir,
    all_pose_dataset,
):
    """Launch inference with or without DDP."""
    infer_with_ddp = False
    # set up unit and start training
    world_size = torch.cuda.device_count()
    print("No. of available devices:", world_size)
    if infer_with_ddp:
        ddp_args = (
            cfg.train.ddp,
            world_size,
            scene,
            config,
            cfg,
            num_new_garment_bones,
            bones_lod_levels,
            output_path,
            epoch_dir,
            all_pose_dataset,
        )
        mp.spawn(prepare_and_train_unit, args=ddp_args, nprocs=world_size)

    else:
        prepare_and_train_unit(
            0,
            False,
            world_size,
            scene,
            config,
            cfg,
            num_new_garment_bones,
            bones_lod_levels,
            output_path,
            epoch_dir,
            all_pose_dataset,
        )


###### MAIN FUNCTION #######


@hydra.main(version_base=None, config_path="./config", config_name="expconfig")
def main(cfg: DictConfig) -> None:
    """
    Script to optimize weight fields for a character rig using physics losses
    """

    seed(cfg.seed)

    if cfg.train.device is not None:
        device = torch.device(cfg.train.device)
    else:
        device = init_from_env()

    # setup checkpoint loading path
    output_path, epoch_dir = _resolve_checkpoint_paths(cfg)

    # LOAD GARMENT PATTERN EMBEDDINGS
    pattern_emb_path = cfg.garment.pattern_embeddings_path
    pattern_embeddings = torch.load(pattern_emb_path)

    # get garment paths
    garment_paths = _get_garment_paths(cfg, output_path)

    GARMENTS_METADATA = []
    num_new_garment_bones = None
    bones_lod_levels = None
    scene = None
    config = None

    for custom_garment_dirpath in tqdm(garment_paths):
        # initialize config
        config = dataio.load_config(cfg.train.config_path)

        # initialize scene
        scene = dataio.load_scene(cfg.train.scene_path)
        scene = dict(scene)  # Allow modifications for scene

        garment_metadata, num_new_garment_bones, bones_lod_levels = (
            _process_single_garment(
                cfg,
                scene,
                config,
                custom_garment_dirpath,
                device,
                pattern_embeddings,
            )
        )
        GARMENTS_METADATA.append(garment_metadata)

    print()
    print("Garments Loaded !!!")
    print()

    physkin_lbs = lbs.LBS.from_scene_and_config(scene, config)

    all_pose_dataset = _build_pose_dataset(cfg, physkin_lbs, GARMENTS_METADATA)

    print()
    print("Dataset Prepared !!!")
    print()

    _launch_inference(
        cfg,
        scene,
        config,
        num_new_garment_bones,
        bones_lod_levels,
        output_path,
        epoch_dir,
        all_pose_dataset,
    )


if __name__ == "__main__":
    main()

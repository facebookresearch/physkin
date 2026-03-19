# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import re
from typing import Optional

import bvh
import numpy as np
import pymeshlab
import torch
import torch_geometric
import trimesh
from scipy import spatial
from torch.utils.data import Dataset
from tqdm.auto import tqdm


def load_scene(scene_path):
    return np.load(scene_path)


def load_config(config_path):
    with open(config_path, "r") as f:
        data = re.sub(r"//.*\n", "", f.read())
        return json.loads(data)


class PoseAnimDataset(Dataset):
    """
    Dataset containing a sampling of all poses from the motion dataset as a single 2D matrix.
    If rig is given, then poses will be applied to the rig 1 by 1 and any intersecting poses will be culled.
    """

    def __init__(
        self,
        pose_dataset_path: str = "",
        lbs=None,
        activePoseParamMask: Optional[torch.BoolTensor] = None,
        rig=None,
        device_for_intersections: str = "cpu",
        max_poses: int | None = None,
        spt: bool = False,
        fixed_root_translation: bool = False,
        fixed_root_rotation: bool = False,
    ):
        super().__init__()
        self.lbs = lbs
        self.poseanim_data = np.load(pose_dataset_path, allow_pickle=True)
        if spt:
            self.all_data = torch.tensor(self.poseanim_data, dtype=torch.float32)
        else:
            raise ValueError("Non-SPT bone-based data format is not supported.")

        # Filter out intersections if rig is given
        if rig is not None:
            self.rig = rig
            self.f = self.rig.base_mesh.faces.to(device_for_intersections)
            self.device_for_intersections = device_for_intersections
            self.skin = self.rig.skin
            self.spt = self.rig.skeleton_parameter_transform
            self.verts = self.rig.base_mesh.vertices
            self.bind_state = self.spt.skeleton.skeleton_state_from_joints(
                torch.zeros((self.spt.skeleton.n_joints, 7), dtype=torch.float32)
            )
            self.all_data = self.filter_out_intersecting(self.all_data)
        # Mask out active pose params if mask is given
        if activePoseParamMask is not None:
            self.all_data = self.all_data[..., activePoseParamMask]
        if max_poses is not None:
            self.all_data = self.all_data[:max_poses, ...]
        if fixed_root_translation:
            self.all_data[:, 0:3] = 0.0
        if fixed_root_rotation:
            self.all_data[:, 3:6] = 0.0

    def is_intersecting(self, pose):
        joint_state = self.spt.joint_state_from_pose(pose)
        skeleton_state = self.spt.skeleton.skeleton_state_from_joints(joint_state)
        verts = self.skin(self.bind_state, self.verts, skeleton_state).to(
            self.device_for_intersections
        )
        meshbvh = bvh.BVH(verts, self.f)
        result = meshbvh.triangle_intersections(verts, self.f)
        return torch.sum(result) != 0

    def filter_out_intersecting(self, poses):
        return torch.stack([pose for pose in poses if self.is_intersecting(pose)])

    def __len__(self):
        return self.all_data.shape[0]

    def __getitem__(self, idx):
        return self.all_data[idx]


class ShapePoseAnimDataset(Dataset):
    def __init__(
        self,
        pose_dataset_path: str = "",
        motion_sequence_dataset_path: str = None,
        shape_dataset_path: str = None,
        shape_latent_type: str = "enc",
        shape_latent_return_mean: bool = False,
        lbs=None,
        activePoseParamMask: Optional[torch.BoolTensor] = None,
        rig=None,
        device_for_intersections: str = "cpu",
        spt: bool = False,
        fixed_root_translation: bool = False,
        fixed_root_rotation: bool = False,
        shape_train_samplenum: int = -1,
        motion_train_samplenum: int = -1,
        global_normalization: bool = False,
        global_y_translation: float = 0.0,
        global_scale: float = 1.0,
        load_garments: bool = False,
        garments_metadata: None = None,
        num_test_garment_samples: int = 0,
        canonical_pose_only=False,
        garment_specific_pose_masking=False,
        train_sampling_type="predefined",
        garment_sampling_type="random",
        motion_window_size=1,
        fixed_seq_idx=None,
        fixed_window_idx=None,
        load_random_poses=False,
        alpha_canonical_pose=0.0,
        body_name="shapes",
    ):
        super().__init__()
        self._init_basic_attrs(
            lbs,
            body_name,
            global_normalization,
            global_scale,
            global_y_translation,
            load_garments,
            garments_metadata,
            pose_dataset_path,
            shape_train_samplenum,
            motion_train_samplenum,
            shape_latent_type,
            train_sampling_type,
            garment_sampling_type,
            motion_window_size,
            fixed_seq_idx,
            fixed_window_idx,
        )

        # Load body shape data (sets self.train_body_verts, self.train_body_latents)
        if self.body_name == "avatar":
            reference_faces = self._load_avatar_shape_data(
                shape_dataset_path,
                shape_latent_return_mean,
            )
        elif self.body_name == "3DBiCar":
            reference_faces = self._load_3dbicar_shape_data(
                shape_dataset_path,
                canonical_pose_only,
            )

        # Common normalization and latent sampling
        self._compute_shape_latents(reference_faces)

        # Load body pose data (sets self.all_motion_data, self.all_motion_affine_states)
        if self.body_name == "avatar":
            self._load_avatar_pose_data(
                spt,
                rig,
                device_for_intersections,
                activePoseParamMask,
                fixed_root_translation,
                fixed_root_rotation,
                canonical_pose_only,
                alpha_canonical_pose,
                garment_specific_pose_masking,
                load_random_poses,
                motion_sequence_dataset_path,
            )
        elif self.body_name == "3DBiCar":
            self._load_3dbicar_pose_data()

        # Load garment data
        if self.load_garments:
            self._load_garment_data(num_test_garment_samples)

        print(
            f"Dataset loaded --> motion_seq: {len(self.all_motion_data)} | body_verts: {self.train_body_verts.shape} | body_latents: {self.train_body_latents.shape} | garments: {self.num_garment_samples}"
        )

        # Predefined sampling indices
        if self.train_sampling_type == "predefined":
            self._init_predefined_sampling()

    def _init_basic_attrs(
        self,
        lbs,
        body_name,
        global_normalization,
        global_scale,
        global_y_translation,
        load_garments,
        garments_metadata,
        pose_dataset_path,
        shape_train_samplenum,
        motion_train_samplenum,
        shape_latent_type,
        train_sampling_type,
        garment_sampling_type,
        motion_window_size,
        fixed_seq_idx,
        fixed_window_idx,
    ):
        """Initialize basic scalar/config attributes."""
        self.lbs = lbs
        self.body_name = body_name
        self.global_normalization = global_normalization
        self.global_scale = global_scale
        self.global_y_translation = global_y_translation
        self.load_garments = load_garments
        self.garments_metadata = garments_metadata
        self.fixed_garment_index = -1
        self.poseanim_data = np.load(pose_dataset_path, allow_pickle=True)
        self.shape_train_samplenum = shape_train_samplenum
        self.motion_train_samplenum = motion_train_samplenum
        self.shape_latent_type = shape_latent_type
        self.train_sampling_type = train_sampling_type
        self.garment_sampling_type = garment_sampling_type
        self.motion_window_size = motion_window_size
        self.fixed_seq_idx = fixed_seq_idx
        self.fixed_window_idx = fixed_window_idx
        assert self.motion_window_size > 0, "Motion window size must be greater than 0"
        self.body_token_len = 4096
        self.garment_token_len = -1

    def _load_avatar_shape_data(self, shape_dataset_path, shape_latent_return_mean):
        """Load avatar body shape data. Returns reference_faces for latent sampling."""
        # reference mesh
        ms_io = pymeshlab.MeshSet()
        ms_io.load_new_mesh(f"{shape_dataset_path}/average_mesh/average_tri.obj")
        ms_io.meshing_poly_to_tri()
        reference_mesh = ms_io.current_mesh()
        reference_verts = reference_mesh.vertex_matrix().copy()  # noqa: F841
        reference_faces = reference_mesh.face_matrix().copy()
        ms_io.clear()

        self.shape_data = np.load(
            f"{shape_dataset_path}/avatars_random_latent_data.npz",
            allow_pickle=True,
        )

        if self.global_normalization:
            self._init_global_normalization_from_shape_data()

        if str(self.shape_data).startswith("NpzFile"):
            self.shape_data = dict(self.shape_data)
            self.train_body_verts = torch.tensor(
                self.shape_data["verts"], dtype=torch.float32
            )
            if shape_latent_return_mean:
                print(f"Loading MEAN {self.shape_latent_type} latents...")
                self.train_body_latents = (
                    torch.tensor(
                        self.shape_data[self.shape_latent_type], dtype=torch.float32
                    )
                    .squeeze(1)
                    .mean(-1)
                )
            else:
                print(f"Loading {self.shape_latent_type} latents...")
                self.train_body_latents = self.normalize_along_y(
                    self.train_body_verts.clone(), forced=True, debug=True
                )

        return reference_faces

    def _init_global_normalization_from_shape_data(self):
        """Compute global scale and y-translation from shape data if defaults."""
        if self.global_scale == 1.0:
            # uniform scaling along y-axis
            self.global_scale = (
                self.shape_data["verts"][:, :, 1].max()
                - self.shape_data["verts"][:, :, 1].min()
            )
        if self.global_y_translation == 0.0:
            self.global_y_translation = self.shape_data["verts"][:, :, 1].min()
        global_normalization_info = f"""
                =====================================
                =====================================
                GLOBAL NORMALIZATION INFO (BODY)
                Y-SHIFT --> {self.global_y_translation}
                SCALE --> {self.global_scale}
                =====================================
                =====================================
                """
        print(global_normalization_info)

    def _load_3dbicar_shape_data(self, shape_dataset_path, canonical_pose_only):
        """Load 3DBiCar body shape data. Returns reference_faces for latent sampling."""
        #  load cached data
        self.body_data = dict(np.load(f"{shape_dataset_path}/body_data.npz"))
        if canonical_pose_only:
            all_body_verts = self.body_data["tpose_verts"]
        else:
            all_body_verts = self.body_data["posed_verts"]

        all_body_verts = torch.tensor(all_body_verts, dtype=torch.float32)
        # reference mesh
        ms_io = pymeshlab.MeshSet()
        ms_io.load_new_mesh(f"{shape_dataset_path}/rabit_SMPL_A_scaled.obj")
        ms_io.meshing_poly_to_tri()
        reference_mesh = ms_io.current_mesh()
        reference_verts = reference_mesh.vertex_matrix().copy()
        reference_faces = reference_mesh.face_matrix().copy()
        ms_io.clear()
        # mean mesh
        mean_mesh_path = f"{shape_dataset_path}/mean.obj"
        ms_io.load_new_mesh(mean_mesh_path)
        ms_io.meshing_poly_to_tri()
        mean_mesh = ms_io.current_mesh()
        mean_verts = mean_mesh.vertex_matrix().copy()
        ms_io.clear()
        # scale correction
        _new_scale_ = reference_verts[:, 1].max() - reference_verts[:, 1].min()
        _orig_y_min_ = mean_verts[:, 1].min()
        _orig_scale_ = mean_verts[:, 1].max() - mean_verts[:, 1].min()
        all_body_verts[:, :, 1] = all_body_verts[:, :, 1] - _orig_y_min_
        all_body_verts = all_body_verts / _orig_scale_
        all_body_verts = all_body_verts * _new_scale_
        # scale adjustment for physics losses
        _scale_factor_ = 0.8
        _shift_factor_ = torch.tensor([0.0, 60.0, 0.0], dtype=torch.float32)
        all_body_verts_adjusted = all_body_verts * _scale_factor_ + _shift_factor_
        self.train_body_verts = all_body_verts_adjusted
        self.train_body_latents = self.normalize_along_y(
            self.train_body_verts.clone(), forced=True, debug=True
        )
        # lowres mesh for latent sampling
        ms_io.load_new_mesh(f"{shape_dataset_path}/rabit_SMPL_A_scaled_decimated.obj")
        lowres_reference_mesh = ms_io.current_mesh()
        lowres_verts = lowres_reference_mesh.vertex_matrix().copy()
        ms_io.clear()
        # sub-sample latents
        ref_tree = spatial.cKDTree(reference_verts)
        _, lowres_nn_idx = ref_tree.query(lowres_verts)
        # sampled
        self.train_body_latents = self.train_body_latents[:, lowres_nn_idx, :]

        return reference_faces

    def _compute_shape_latents(self, reference_faces):
        """Normalize body verts and compute point-sampled shape latents."""
        # COMMON NORMALIZATION (generally ignored)
        self.train_body_verts = self.normalize_along_y(self.train_body_verts)

        # COMMON SAMPLING
        generate_normals = torch_geometric.transforms.GenerateMeshNormals()
        generate_uniform_points = torch_geometric.transforms.SamplePoints(
            num=10000, include_normals=True
        )
        shape_latents = []
        for shape_idx in tqdm(range(len(self.train_body_verts))):
            body_verts = self.train_body_verts[shape_idx].numpy()
            body_mesh = trimesh.Trimesh(body_verts, reference_faces)
            body_graph = torch_geometric.utils.from_trimesh(body_mesh)
            body_graph = generate_normals(body_graph)
            body_graph = generate_uniform_points(body_graph)
            point_features = torch.cat([body_graph.pos, body_graph.normal], dim=-1)
            shape_latents.append(point_features)

        self.train_body_latents = torch.stack(shape_latents, dim=0)

    @staticmethod
    def _filter_pose_params(
        pose_data,
        activePoseParamMask,
        fixed_root_translation,
        fixed_root_rotation,
        canonical_pose_only,
    ):
        """Apply pose parameter filtering (mask, root fix, canonical zero-out)."""
        if activePoseParamMask is not None:
            pose_data = pose_data[..., activePoseParamMask]
        if fixed_root_translation:
            pose_data[:, 0:3] = 0.0
        if fixed_root_rotation:
            pose_data[:, 3:6] = 0.0
        if canonical_pose_only:
            pose_data *= 0.0
        return pose_data

    def _apply_garment_pose_masking(self, pose_data, include_clavicle_spine=False):
        """Zero out upper-body joints and optionally set shoulder_ry to t-pose.

        Args:
            pose_data: Pose tensor to mask in-place.
            include_clavicle_spine: If True, also mask clavicle and spine joints.
                Used for motion sequences; the static pose version excludes them.
        """

        def joint_to_mask(name):
            upper_body_flag = (
                ("elbow" in name)
                or ("wrist" in name)
                or ("forearm" in name)
                or ("shoulder" in name)
            )
            if include_clavicle_spine:
                upper_body_flag = upper_body_flag or (
                    ("clavicle" in name) or ("spine" in name)
                )
            return upper_body_flag

        joint_names = (
            self.lbs.config.rig.rig.skeleton_parameter_transform.transform_names
        )
        joint_pose_mask = np.array([joint_to_mask(name) for name in joint_names])
        pose_data[:, joint_pose_mask] = 0.0

        if not include_clavicle_spine:
            # For static poses, also set shoulder_ry to t-pose
            shoulder_mask = np.array(["shoulder_ry" in name for name in joint_names])
            pose_data[:, shoulder_mask] = 1.0

    def _load_avatar_pose_data(
        self,
        spt,
        rig,
        device_for_intersections,
        activePoseParamMask,
        fixed_root_translation,
        fixed_root_rotation,
        canonical_pose_only,
        alpha_canonical_pose,
        garment_specific_pose_masking,
        load_random_poses,
        motion_sequence_dataset_path,
    ):
        """Load avatar pose data, apply filtering, compute affine states, load motions."""
        if spt:
            self.all_data = torch.tensor(self.poseanim_data, dtype=torch.float32)
        else:
            raise ValueError("Non-SPT bone-based data format is not supported.")

        # Filter out intersections if rig is given
        if rig is not None:
            self._init_rig_for_intersection(rig, device_for_intersections)
            self.all_data = self.filter_out_intersecting(self.all_data)

        # filter pose parameters (pass fixed_root_rotation=False here because the
        # avatar static-pose path originally only zeroed index 3, not 3:6)
        self.all_data = self._filter_pose_params(
            self.all_data,
            activePoseParamMask,
            fixed_root_translation,
            False,
            canonical_pose_only,
        )
        if fixed_root_rotation:
            self.all_data[:, 3] = 0.0
        if alpha_canonical_pose > 0.0:
            canonical_pose = torch.zeros_like(self.all_data)
            self.all_data = (
                self.all_data * (1.0 - alpha_canonical_pose)
                + canonical_pose * alpha_canonical_pose
            )
        if garment_specific_pose_masking:
            self._apply_garment_pose_masking(
                self.all_data, include_clavicle_spine=False
            )

        # precompute affine states
        affine_states = self.lbs.input_to_affine(self.all_data)
        self.all_affine_states = self.normalize_along_y(affine_states)

        # normalizing pose data after computing affine states
        if self.global_normalization:
            _MIN_ = self.all_data.min()
            _MAX_ = self.all_data.max()
            self.all_data = (self.all_data - _MIN_) / (_MAX_ - _MIN_)

        # motion sequences
        self._load_avatar_motion_sequences(
            load_random_poses,
            motion_sequence_dataset_path,
            activePoseParamMask,
            fixed_root_translation,
            fixed_root_rotation,
            canonical_pose_only,
            garment_specific_pose_masking,
        )

    def _init_rig_for_intersection(self, rig, device_for_intersections):
        """Set up rig-related attributes for intersection filtering."""
        self.rig = rig
        self.f = self.rig.base_mesh.faces.to(device_for_intersections)
        self.device_for_intersections = device_for_intersections
        self.skin = self.rig.skin
        self.spt = self.rig.skeleton_parameter_transform
        self.verts = self.rig.base_mesh.vertices
        self.bind_state = self.spt.skeleton.skeleton_state_from_joints(
            torch.zeros((self.spt.skeleton.n_joints, 7), dtype=torch.float32)
        )

    def _load_avatar_motion_sequences(
        self,
        load_random_poses,
        motion_sequence_dataset_path,
        activePoseParamMask,
        fixed_root_translation,
        fixed_root_rotation,
        canonical_pose_only,
        garment_specific_pose_masking,
    ):
        """Load motion sequences for avatar pose data."""
        if load_random_poses:
            self.motion_window_size = 1
            self.all_motion_data = self.all_data.unsqueeze(1)
            self.all_motion_affine_states = self.all_affine_states.unsqueeze(1)
            return

        print("Loading animation data...")
        self.all_motion_data = []
        self.all_motion_affine_states = []
        if motion_sequence_dataset_path is None:
            return

        motion_sequences = np.load(motion_sequence_dataset_path, allow_pickle=True)
        motion_poses = motion_sequences["poses"]
        frame_offsets = motion_sequences["frame_offsets"]
        if self.motion_train_samplenum > 0:
            frame_offsets = frame_offsets[: self.motion_train_samplenum + 1]
        for i in tqdm(range(len(frame_offsets) - 1)):
            start = frame_offsets[i]
            end = frame_offsets[i + 1]
            motion_data = torch.tensor(motion_poses[start:end], dtype=torch.float32)

            # filter pose parameters
            motion_data = self._filter_pose_params(
                motion_data,
                activePoseParamMask,
                fixed_root_translation,
                fixed_root_rotation,
                canonical_pose_only,
            )
            if garment_specific_pose_masking:
                self._apply_garment_pose_masking(
                    motion_data, include_clavicle_spine=True
                )

            # precompute motion affine states
            motion_affine_states = self.lbs.input_to_affine(motion_data)

            self.all_motion_data.append(motion_data)
            self.all_motion_affine_states.append(motion_affine_states)

    def _load_3dbicar_pose_data(self):
        """Load 3DBiCar pose data (identity poses and affine states)."""
        total_poses = 2048
        pose_vector_dim = 120
        _scene_ = self.garments_metadata[0].lbs.config.scene.scene  # noqa: F841
        _num_skeleton_joints_ = 24
        self.all_motion_data = torch.zeros(1, 1, pose_vector_dim).repeat(
            total_poses, 1, 1
        )
        self.all_motion_affine_states = (
            torch.eye(4)
            .repeat(_num_skeleton_joints_, 1, 1)
            .repeat(total_poses, 1, 1, 1)
            .unsqueeze(1)
        )

    def _normalize_garment_graph_attr(self, graph_data, attr_name):
        """Normalize a garment graph attribute along Y if it exists and is not None."""
        if (
            hasattr(graph_data, attr_name)
            and getattr(graph_data, attr_name) is not None
        ):
            setattr(
                graph_data,
                attr_name,
                self.normalize_along_y(getattr(graph_data, attr_name)),
            )

    def _load_garment_data(self, num_test_garment_samples):
        """Load and normalize garment metadata."""
        self.num_garment_samples = len(self.garments_metadata)
        self.num_test_garment_samples = num_test_garment_samples
        self.num_train_garment_samples = (
            self.num_garment_samples - self.num_test_garment_samples
        )
        self.pattern_embeddings = []
        self.garment_lbs = []
        self.garment_graph_data = []
        for gm in self.garments_metadata:
            if gm.pattern_embeddings is not None:
                self.pattern_embeddings.append(torch.Tensor(gm.pattern_embeddings))
            else:
                self.pattern_embeddings.append(torch.tensor([-1]))
            gm.graph_data.x[:, :3] = self.normalize_along_y(gm.graph_data.x[:, :3])
            gm.graph_data.verts = self.normalize_along_y(gm.graph_data.verts)
            self._normalize_garment_graph_attr(gm.graph_data, "cloth_joint_positions")
            self._normalize_garment_graph_attr(
                gm.graph_data, "cloth_bones_rest_positions"
            )
            self._normalize_garment_graph_attr(
                gm.graph_data, "all_bones_rest_positions"
            )
            self._normalize_garment_graph_attr(gm.graph_data, "panels_verts")
            self.garment_graph_data.append(gm.graph_data)
            self.garment_lbs.append(gm.lbs)
            num_faces = gm.graph_data.faces.shape[0]
            if num_faces > self.garment_token_len:
                self.garment_token_len = num_faces

    def _init_predefined_sampling(self):
        """Pre-generate random indices for predefined sampling strategy."""
        sampling_size = len(self.all_motion_data)
        self.sampled_shape_indices = np.random.choice(
            self.shape_train_samplenum, sampling_size, replace=True
        )
        self.sampled_garment_indices = np.random.choice(
            self.num_train_garment_samples, sampling_size, replace=True
        )
        if self.motion_window_size == 1:
            self.sampled_window_indices = np.zeros(sampling_size, dtype=int)
        else:
            self.sampled_window_indices = np.array(
                [
                    np.random.randint(
                        self.all_motion_data[i].shape[0] - self.motion_window_size
                    )
                    for i in range(sampling_size)
                ]
            )

    def normalize_along_y(self, data, forced=False, debug=False):
        if debug:
            debug_msg = f"""
            ======================================================================
            BEFORE --> {data.shape} | MIN: {data.min()} | MAX: {data.max()}
            """
        if self.global_normalization or forced:
            if isinstance(data, torch.Tensor):
                if data.dim() == 4 or data.dim() == 3:
                    if data.shape[-2:] == (4, 4) or data.shape[-2:] == (3, 4):
                        data_xyz = data[..., :3, 3]
                        data_xyz[..., 1] = data_xyz[..., 1] - self.global_y_translation
                        data_xyz /= self.global_scale
                        data[..., :3, 3] = data_xyz
                    elif data.shape[-1] == 4 or data.shape[-1] == 3:
                        data_xyz = data[..., :3]
                        data_xyz[..., 1] = data_xyz[..., 1] - self.global_y_translation
                        data_xyz /= self.global_scale
                        data[..., :3] = data_xyz
                elif data.dim() == 2:
                    if data.shape[-1] == 4 or data.shape[-1] == 3:
                        data_xyz = data[:, :3]
                        data_xyz[..., 1] = data_xyz[..., 1] - self.global_y_translation
                        data_xyz /= self.global_scale
                        data[:, :3] = data_xyz
        if debug:
            debug_msg += f"""
            NORMALIZE WITH --> SCALE: {self.global_scale} | Y-TRANSLATION: {self.global_y_translation}
            AFTER --> {data.shape} | MIN: {data.min()} | MAX: {data.max()}
            ======================================================================
            """
            print(debug_msg, flush=True)
        return data

    def is_intersecting(self, pose):
        joint_state = self.spt.joint_state_from_pose(pose)
        skeleton_state = self.spt.skeleton.skeleton_state_from_joints(joint_state)
        verts = self.skin(self.bind_state, self.verts, skeleton_state).to(
            self.device_for_intersections
        )
        meshbvh = bvh.BVH(verts, self.f)
        result = meshbvh.triangle_intersections(verts, self.f)
        return torch.sum(result) != 0

    def filter_out_intersecting(self, poses):
        return torch.stack([pose for pose in poses if self.is_intersecting(pose)])

    def __len__(self):
        return self.motion_train_samplenum

    def _resolve_motion_sample(self, idx, start_idx):
        """Resolve motion sequence index, window start, and return motion samples.

        Returns:
            Tuple of (motion_sample, motion_affine_states_sample, motion_idx, start_idx).
            motion_idx may be None if no motion data is available.
        """
        motion_sample = torch.tensor([-1])
        motion_affine_states_sample = None
        motion_idx = None

        if len(self.all_motion_data) == 0:
            return motion_sample, motion_affine_states_sample, motion_idx, start_idx

        # MOTION SEQUENCE & WINDOW INDEX
        motion_idx = idx if self.fixed_seq_idx is None else self.fixed_seq_idx
        if motion_idx is None:
            # random motion sequence
            motion_idx = np.random.randint(len(self.all_motion_data))
        elif start_idx == "predefined" and self.motion_window_size == 1:
            start_idx = self.sampled_window_indices[idx]
        elif start_idx == "first" or self.motion_window_size == 1:
            start_idx = 0
        elif start_idx == "last":
            start_idx = (
                self.all_motion_data[motion_idx].shape[0] - self.motion_window_size
            )
        elif start_idx == "random":
            # random window
            start_idx = np.random.randint(
                self.all_motion_data[motion_idx].shape[0] - self.motion_window_size
            )
        end_idx = start_idx + self.motion_window_size
        motion_sample = self.all_motion_data[motion_idx][start_idx:end_idx]
        motion_affine_states_sample = self.all_motion_affine_states[motion_idx][
            start_idx:end_idx
        ]
        return motion_sample, motion_affine_states_sample, motion_idx, start_idx

    def _resolve_shape_idx(self, idx, shape_idx):
        """Resolve the shape index for the given sample."""
        if shape_idx is not None:
            return shape_idx
        if self.train_sampling_type == "predefined":
            return self.sampled_shape_indices[idx]
        return np.random.randint(self.shape_train_samplenum)

    def _resolve_garment_idx(self, idx, shape_idx, garment_idx, mode):
        """Resolve the garment index for the given sample."""
        if garment_idx is not None:
            return garment_idx
        garment_idx = self.fixed_garment_index
        if mode == "train" and garment_idx == -1:
            if self.train_sampling_type == "predefined":
                garment_idx = self.sampled_garment_indices[idx]
            elif self.garment_sampling_type == "paired":
                garment_idx = shape_idx % self.num_train_garment_samples
            elif self.garment_sampling_type == "random":
                garment_idx = np.random.randint(self.num_train_garment_samples)
            else:
                print(
                    f"ERROR: '{self.garment_sampling_type}' is not a valid garment sampling type!"
                )
        return garment_idx

    def __getitem__(
        self, idx, shape_idx=None, garment_idx=None, window_start_idx=None, mode="train"
    ):
        if mode == "train":
            start_idx = (
                self.train_sampling_type
                if self.fixed_window_idx is None
                else self.fixed_window_idx
            )
        else:
            start_idx = (
                window_start_idx
                if self.fixed_window_idx is None
                else self.fixed_window_idx
            )

        motion_sample, motion_affine_states_sample, motion_idx, start_idx = (
            self._resolve_motion_sample(idx, start_idx)
        )
        shape_idx = self._resolve_shape_idx(idx, shape_idx)
        garment_idx = self._resolve_garment_idx(idx, shape_idx, garment_idx, mode)

        if not self.load_garments:
            return (
                self.all_data[idx],
                self.train_body_verts[shape_idx],
                self.train_body_latents[shape_idx],
            )

        # sample body shape latent
        body_latent = self.train_body_latents[shape_idx]
        rnd_idx = torch.randperm(len(body_latent))[: self.body_token_len]
        body_latent = body_latent[rnd_idx]

        # sample garment pattern embedding
        if self.pattern_embeddings[0].dim() > 1:
            pattern_embedding = self.pattern_embeddings[garment_idx]
        else:
            pattern_embedding = torch.tensor([-1])

        if not mode == "train":
            print(
                "motion_idx:",
                motion_idx,
                "window_start_idx:",
                start_idx,
                "shape_idx:",
                shape_idx,
                "garment_idx:",
                garment_idx,
            )

        return (
            self.train_body_verts[shape_idx],
            body_latent,
            pattern_embedding,
            int(shape_idx),
            int(garment_idx),
            motion_sample,
            motion_affine_states_sample,
        )

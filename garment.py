# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


class Garment:
    def __init__(self, cfg, scene):
        self.cfg = cfg
        self.scene = scene
        self.lbs = None
        self.dirpath = None
        self.type = None
        self.verts = None
        self.faces = None
        self.vertex_normals = None
        self.face_normals = None
        self.tex_coords = None
        self.panels_verts = None
        self.panels_faces = None
        self.seam_verts_indices = None
        self.rest_pos = None
        self.verts_pinning_mask = None
        self.bones_pinning_mask = None
        self.random_pinned_bones_indices = None
        self.active_bones = None
        self.pinned_bones = None
        self.unpinned_active_bones = None
        self.cloth_joint_positions = None
        self.pinned_bones_bodyverts_idx = None
        self.vertex_bone_selection_mask = None
        self.bones_lod_sampled = None
        self.cloth_bones_weights = None
        self.pattern_embeddings = None
        self.sampled_points = None
        self.graph_data = None

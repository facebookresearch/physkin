# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Tuple

import torch
import warp as wp
from torch import autograd


def compute_bounds(vf: torch.Tensor):
    vf = vf.to(torch.float32)
    lowers, _ = torch.min(vf, dim=-2)
    uppers, _ = torch.max(vf, dim=-2)
    device_lowers = wp.from_torch(lowers, dtype=wp.vec3f, requires_grad=False)
    device_uppers = wp.from_torch(uppers, dtype=wp.vec3f, requires_grad=False)
    return device_lowers, device_uppers


class BVH:
    def __init__(self, verts: torch.Tensor, faces: torch.Tensor):
        super().__init__()
        """Constructs a BVH for the triangles in a given mesh"""
        self.verts = verts
        self.faces = faces.int()
        self.wp_faces = wp.from_torch(self.faces, dtype=wp.vec3i, requires_grad=False)
        self.bvh = None
        self.init_bvh()

    def init_bvh(self):
        self.bvh = wp.Bvh(*compute_bounds(self.verts[self.faces]))

    def triangle_intersections(self, pos: torch.Tensor, tri: torch.Tensor):
        """
        Check if the given triangles intersect the mesh.
        The given mesh can be the same as the one used to initialize this bvh.

        A better test will be to test for segment intersections, but this is easier to implement.
        """
        assert self.bvh is not None

        verts = wp.from_torch(self.verts, dtype=wp.vec3f, requires_grad=False)
        q_faces = wp.from_torch(tri.int(), dtype=wp.vec3i, requires_grad=False)
        q_verts = wp.from_torch(pos, dtype=wp.vec3f, requires_grad=False)
        assert verts.device == self.wp_faces.device
        assert q_verts.device == q_faces.device
        assert q_verts.device == verts.device
        assert self.bvh.device == q_verts.device
        n = len(tri)
        hits = wp.zeros(
            (n,),
            dtype=wp.int32,  # type: ignore
            requires_grad=False,
            device=self.wp_faces.device,
        )
        wp.launch(
            kernel=bvh_query_tri,
            dim=(int(n),),
            inputs=[self.bvh.id, verts, self.wp_faces, q_verts, q_faces],
            outputs=[hits],
            device=self.wp_faces.device,
        )
        return wp.to_torch(hits)

    def orientation_aware_signed_distance(
        self,
        q_pos: torch.Tensor,
        q_nml: torch.Tensor,
        t_pos: torch.Tensor,
        t_face_nml: torch.Tensor,
        max_dist: float,
    ) -> torch.Tensor:
        """Once differentiable orientation aware signed distance"""
        assert self.bvh is not None

        val, _grad = BVHOrientationAwareSignedDistance.apply(
            q_pos.view(-1, 3),
            q_nml.view(-1, 3),
            t_pos.view(-1, 3),
            t_face_nml,  # Should not be batched
            self.faces,
            max_dist,
            self.bvh.id,
        )  # type: ignore

        return val


# See https://stackoverflow.com/questions/2924795/fastest-way-to-compute-point-to-triangle-distance-in-3d for reference (this is a very common code)
@wp.func
def closest_point_triangle(
    p: wp.vec3f, a: wp.vec3f, b: wp.vec3f, c: wp.vec3f
):  # -> Tuple[wp.vec3f, wp.vec3f, wp.bool]:
    """
    Returns the position of the closest point on the corresponding triangle,
    its uv coordinates in the triangle
    whether the closest point is in the interior of the triangle respectively.
    """
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = wp.dot(ab, ap)
    d2 = wp.dot(ac, ap)
    if d1 <= float(0.0) and d2 <= float(0.0):
        return a, wp.vec3f(1.0, 0.0, 0.0), 1  # 1

    bp = p - b
    d3 = wp.dot(ab, bp)
    d4 = wp.dot(ac, bp)
    if d3 >= float(0.0) and d4 <= d3:
        return b, wp.vec3f(0.0, 1.0, 0.0), 2  # 2

    cp = p - c
    d5 = wp.dot(ab, cp)
    d6 = wp.dot(ac, cp)
    if d6 >= float(0.0) and d5 <= d6:
        return c, wp.vec3f(0.0, 0.0, 1.0), 3  # 3

    vc = d1 * d4 - d3 * d2
    if vc <= float(0.0) and d1 >= float(0.0) and d3 <= float(0.0):
        v = d1 / (d1 - d3)
        return a + v * ab, wp.vec3f(1.0 - v, v, 0.0), 4  # 4

    vb = d5 * d2 - d1 * d6
    if vb <= float(0.0) and d2 >= float(0.0) and d6 <= float(0.0):
        v = d2 / (d2 - d6)
        return a + v * ac, wp.vec3f(1.0 - v, 0.0, v), 5  # 5

    va = d3 * d6 - d5 * d4
    if va <= float(0.0) and (d4 - d3) >= float(0.0) and (d5 - d6) >= float(0.0):
        v = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + v * (c - b), wp.vec3f(0.0, 1.0 - v, v), 6  # 6

    denom = float(1.0) / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return a + v * ab + w * ac, wp.vec3f(1.0 - v - w, v, w), 0  # 0


@wp.kernel
def bvh_orientation_aware_signed_distance(
    bvh_id: wp.uint64,
    # Query positions and normals
    q_pos: wp.array(dtype=wp.vec3f),  # type: ignore
    q_nml: wp.array(dtype=wp.vec3f),  # type: ignore
    # Positions and normals indexed by faces stored in the bvh (same as q_pos, q_nml for self-contact)
    bvh_pos: wp.array(dtype=wp.vec3f),  # type: ignore
    bvh_nml: wp.array(dtype=wp.vec3f),  # type: ignore
    bvh_tri: wp.array(dtype=wp.vec3i),  # type: ignore
    max_dist: float,
    # (output) orientation aware signed distance
    signed_distance: wp.array(dtype=wp.float32),  # type: ignore
    signed_distance_grad: wp.array(dtype=wp.vec3f),  # type: ignore
):
    """Compute signed distance on the given query points but ignoring all distances where the q_nml makes a > 90 degree angle with the the bvh_nml"""
    tid = wp.tid()

    q = q_pos[tid]

    padv = wp.vec3f(max_dist, max_dist, max_dist)
    query = wp.bvh_query_aabb(bvh_id, q - padv, q + padv)
    bound_idx = wp.int32(0)
    min_d2 = max_dist * max_dist
    closest_idx = wp.int32(-1)
    min_diff = padv
    while wp.bvh_query_next(query, bound_idx):
        f = bvh_tri[bound_idx]
        p0 = bvh_pos[f[0]]
        p1 = bvh_pos[f[1]]
        p2 = bvh_pos[f[2]]

        # Found a face whose aabb intersects the current point
        p, bary_, type_ = closest_point_triangle(q, p0, p1, p2)
        diff = q - p
        d2 = wp.length_sq(diff)

        if d2 < min_d2:
            min_d2 = d2
            closest_idx = bound_idx
            min_diff = diff

    if closest_idx != -1:
        f_nml = bvh_nml[closest_idx]
        valid = wp.dot(q_nml[tid], f_nml) > float(0.0)
        if valid:
            sd = wp.dot(min_diff, f_nml)
            signed_distance[tid] = sd
            signed_distance_grad[tid] = f_nml


class BVHOrientationAwareSignedDistance(autograd.Function):
    @staticmethod
    def forward(
        q_pos: torch.Tensor,
        q_nml: torch.Tensor,
        bvh_pos: torch.Tensor,
        bvh_face_nml: torch.Tensor,
        bvh_faces: torch.Tensor,
        max_dist: float,
        bvh_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Should require grad if backtracking is used. In this case we compute gradient explicitly
        wp_q_pos = wp.from_torch(q_pos, dtype=wp.vec3f, requires_grad=False)
        wp_q_nml = wp.from_torch(q_nml, dtype=wp.vec3f, requires_grad=False)
        wp_bvh_pos = wp.from_torch(bvh_pos, dtype=wp.vec3f)
        wp_bvh_face_nml = wp.from_torch(bvh_face_nml, dtype=wp.vec3f)
        wp_bvh_faces = wp.from_torch(bvh_faces, dtype=wp.vec3i)
        n = len(wp_q_pos)

        wp_oasd = wp.full(
            shape=(n,),
            value=max_dist,  # type: ignore
            dtype=wp.float32,  # type: ignore
            device=wp_q_pos.device,
        )
        wp_oasd_grad = wp.zeros(
            shape=(n,),
            dtype=wp.vec3f,  # type: ignore
            device=wp_q_pos.device,
        )

        wp.launch(
            kernel=bvh_orientation_aware_signed_distance,
            dim=n,
            inputs=[
                bvh_id,
                wp_q_pos,
                wp_q_nml,
                wp_bvh_pos,
                wp_bvh_face_nml,
                wp_bvh_faces,
                max_dist,
            ],
            outputs=[wp_oasd, wp_oasd_grad],
            device=wp_q_pos.device,
        )

        return wp.to_torch(wp_oasd), wp.to_torch(wp_oasd_grad)

    @staticmethod
    def backward(ctx, adj_oasd, adj_oasd_grad):
        # Skip if input is None (needed when set_materialize_grad is set to False)
        if adj_oasd is None:
            return None, None, None, None, None
        (oasd_grad,) = ctx.saved_tensors
        return (
            oasd_grad * adj_oasd.unsqueeze(-1),
            None,
            None,
            None,
            None,
            None,
            None,
        )

    @staticmethod
    def setup_context(ctx, inputs, output):
        (
            ctx.q_pos,
            ctx.q_nml,
            ctx.bvh_pos,
            ctx.bvh_face_nml,
            ctx.bvh_faces,
            ctx.max_dist,
            ctx.bvh,
        ) = inputs

        ctx.oasd, oasd_grad = output

        ctx.save_for_backward(oasd_grad)

        # Optimize gradient initialization when not needed
        ctx.set_materialize_grads(False)


@wp.kernel
def bvh_query_tri(
    bvh_id: wp.uint64,
    verts1: wp.array(dtype=wp.vec3f),  # type: ignore
    faces1: wp.array(dtype=wp.vec3i),  # type: ignore
    verts2: wp.array(dtype=wp.vec3f),  # type: ignore
    faces2: wp.array(dtype=wp.vec3i),  # type: ignore
    hits: wp.array(dtype=wp.int32),  # type: ignore
):
    tid = wp.tid()

    f2 = faces2[tid]
    v20 = verts2[f2[0]]
    v21 = verts2[f2[1]]
    v22 = verts2[f2[2]]
    upper = wp.max(wp.max(v20, v21), v22)
    lower = wp.min(wp.min(v20, v21), v22)

    query = wp.bvh_query_aabb(bvh_id, lower, upper)
    bound_idx = wp.int32(0)

    while wp.bvh_query_next(query, bound_idx):
        # The ray intersects the volume with index bound_idx
        f1 = faces1[bound_idx]
        if (
            f1[0] == f2[0]
            or f1[0] == f2[1]
            or f1[0] == f2[2]
            or f1[1] == f2[0]
            or f1[1] == f2[1]
            or f1[1] == f2[2]
            or f1[2] == f2[0]
            or f1[2] == f2[1]
            or f1[2] == f2[2]
        ):
            continue
        v10 = verts1[f1[0]]
        v11 = verts1[f1[1]]
        v12 = verts1[f1[2]]

        res = wp.intersect_tri_tri(v10, v11, v12, v20, v21, v22)
        if res > int(0):
            hits[tid] = wp.int32(1)
            break


class TestTriangleIntersect(unittest.TestCase):
    def test_tri_intersect(self):
        # Test if two triangles intersect
        device = torch.device("cuda:0")
        faces = torch.tensor([[0, 1, 2], [2, 1, 3], [4, 5, 6]], device=device)
        verts = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
                [0.2, 0.5, 0.5],
                [-0.2, 0.6, 0.6],
                [-0.2, 0.7, 0.7],
            ],
            device=device,
        )

        bvh = BVH(verts, faces[0:2])
        result = bvh.triangle_intersections(verts.clone(), faces[2:3].clone())
        self.assertTrue(result[0])
        result = bvh.triangle_intersections(verts.clone(), faces[1:2].clone())
        self.assertFalse(result[0])


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main()

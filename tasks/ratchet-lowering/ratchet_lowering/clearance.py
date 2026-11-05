"""Measured robot/moving-part clearance and grip diagnostics; no physics changes.

Conservative world-axis bounding boxes of individual collision shapes prevent
supporting a different part from counting as release, without filling the empty
space between a rotor's separated handles, wheel and weight.
Grip diagnostics sample robot collider vertices, not rendered silhouettes.
"""
from itertools import product
from pathlib import Path

import numpy as np

from . import design as d


def bounds_gap(a_lo, a_hi, b_lo, b_hi):
    separation = np.maximum(np.maximum(a_lo[:, None] - b_hi, b_lo - a_hi[:, None]), 0.0)
    return float(np.linalg.norm(separation, axis=-1).min())


class Clearance:
    def __init__(self):
        from pxr import Usd, UsdGeom, UsdPhysics
        from yam_arena.analysis import link_hulls
        generated = Path(__file__).resolve().parents[1] / 'assets' / 'generated'

        self.links = {name: [h.vertices for h in hulls]
                      for name, hulls in link_hulls().items() if hulls}
        self.parts = {}
        for name in ('rotor', 'handle', 'pawl'):
            stage = Usd.Stage.Open(str(generated / f'{name}.usda'))
            root = stage.GetDefaultPrim()
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            corners = []
            for prim in stage.Traverse():
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    box = cache.ComputeRelativeBound(prim, root).ComputeAlignedRange()
                    lo, hi = np.array(box.GetMin()), np.array(box.GetMax())
                    corners.append(np.array(list(product(*zip(lo, hi))), float))
            if not corners:
                raise ValueError(f'No collision geometry for {name}')
            self.parts[name] = corners

    def measure(self, scene, poses, arms):
        robot, labels = [], []
        for arm in arms:
            obj = scene[arm]
            measured = obj.data.body_pose_w.torch[0].cpu().numpy()
            for name, pose in zip(obj.body_names, measured, strict=True):
                if name in self.links:
                    for i, vertices in enumerate(self.links[name]):
                        robot.append(vertices @ d.quat_to_matrix(pose[3:]).T + pose[:3])
                        labels.append(f'{arm}/{name}/{i}')
        parts, part_labels = [], []
        for name, shapes in self.parts.items():
            for i, v in enumerate(shapes):
                parts.append(v @ d.quat_to_matrix(poses[name][3:]).T + poses[name][:3])
                part_labels.append(f'{name}/{i}')
        if not robot:
            raise ValueError('No measured robot collision geometry')
        alo, ahi = np.array([v.min(0) for v in robot]), np.array([v.max(0) for v in robot])
        blo, bhi = np.array([v.min(0) for v in parts]), np.array([v.max(0) for v in parts])
        gaps = np.linalg.norm(np.maximum(np.maximum(alo[:, None] - bhi, blo - ahi[:, None]), 0), axis=-1)
        ai, bi = np.unravel_index(gaps.argmin(), gaps.shape)
        gap = float(gaps[ai, bi])
        self.closest_pair = (labels[ai], part_labels[bi])
        points = np.concatenate(robot)
        # Signed distances of collider vertices into actual grip solids. Negative
        # values establish penetration; a positive sample alone cannot exclude it.
        p = poses['pawl']; local = (points - p[:3]) @ d.quat_to_matrix(p[3:])
        tab = d.TAB_END - d.PAWL_PIVOT
        q = np.abs(local - np.array([tab[0], 0, tab[1]])) - np.array(d.TAB) / 2
        tab_sdf = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0)
        p = poses['handle']; local = (points - p[:3]) @ d.quat_to_matrix(p[3:])
        q = np.c_[np.linalg.norm(local[:, [0, 2]], axis=1) - d.HANDLE_R,
                  np.abs(local[:, 1]) - d.HANDLE_L / 2]
        sleeve_sdf = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0)
        return gap, float(tab_sdf.min()), float(sleeve_sdf.min())

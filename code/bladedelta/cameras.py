"""Camera rig shared by the Blender generator and the differentiable renderer.

Matches the earlier project rig: seven cameras on a ring of radius 2.3 around
the normalized blade, lens 52 mm on a 36 mm sensor, 512 x 512 pixels.
"""
from __future__ import annotations

import numpy as np

RES = 512
FOCAL_PX = RES * 52. / 36.
ANGLES = {f'v{i:03d}': a for i, a in enumerate((-60, -30, 0, 30, 60, 120, 180))}
FIT_VIEWS = ('v004', 'v005')
BASE_DIRECTION = np.array([0., -1., .1]) / np.linalg.norm([0., -1., .1])
DISTANCE = 2.3


def rot_z(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def rot_axis(axis, deg):
    axis = np.asarray(axis, float); axis = axis / np.linalg.norm(axis)
    a = np.radians(deg); K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def intrinsics():
    return np.array([[FOCAL_PX, 0, RES / 2], [0, FOCAL_PX, RES / 2], [0, 0, 1.]])


def camera(view_id: str, yaw=0., pitch=0., roll=0., shift=(0., 0., 0.)):
    """Return dict(K, c2w_blender, w2c_cv) for a nominal or perturbed view.

    yaw rotates the camera position about the blade axis (degrees); pitch and
    roll rotate the camera about its own right and viewing axes; shift moves
    the camera centre in world units. Nominal = all zero.
    """
    pos = rot_z(ANGLES[view_id] + yaw) @ BASE_DIRECTION * DISTANCE
    f = -pos / np.linalg.norm(pos)
    right = np.cross(f, [0, 0, 1.]); right /= np.linalg.norm(right)
    up = np.cross(right, f)
    R = np.stack([right, up, -f], axis=1)  # Blender camera axes as columns
    R = rot_axis(right, pitch) @ R
    R = rot_axis(R[:, 2], roll) @ R
    c2w = np.eye(4); c2w[:3, :3] = R; c2w[:3, 3] = pos + np.asarray(shift, float)
    w2c_cv = np.diag([1., -1., -1., 1.]) @ np.linalg.inv(c2w)
    return dict(view_id=view_id, K=intrinsics(), c2w_blender=c2w, w2c_cv=w2c_cv)

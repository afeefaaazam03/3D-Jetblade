"""Differentiable renderer: nvdiffrast rasterization + neural shader.

The shader maps per-pixel (world normal, world position, view direction,
albedo multiplier, light gains) to the 8-bit display colour Blender/Cycles
produces for the fixed inspection scene. It is trained only on exposed shapes.
Smooth (vertex-normal) shading matches the Blender scene.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nvdiffrast.torch as dr

from .cameras import RES

_CTX = {}


def ctx():
    if 'gl' not in _CTX:
        _CTX['gl'] = dr.RasterizeCudaContext()
    return _CTX['gl']


def so3_exp(w: torch.Tensor) -> torch.Tensor:
    """Rodrigues; w (..., 3) radians -> (..., 3, 3)."""
    th = w.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    k = w / th
    K = torch.zeros(*w.shape[:-1], 3, 3, device=w.device, dtype=w.dtype)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K = K - K.transpose(-1, -2)
    th = th[..., None]
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(K)
    return I + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


def clip_space(v: torch.Tensor, K: torch.Tensor, w2c: torch.Tensor, near=.1, far=10.):
    """World vertices (N,3) -> clip coords (B,N,4) for B OpenCV cameras."""
    X = torch.einsum('bij,nj->bni', w2c[:, :3, :3], v) + w2c[:, None, :3, 3]
    Z = X[..., 2:3]
    u = K[:, None, 0, 0, None] * X[..., :1] + K[:, None, 0, 2, None] * Z
    vv = K[:, None, 1, 1, None] * X[..., 1:2] + K[:, None, 1, 2, None] * Z
    x_c = 2 * u / RES - Z
    y_c = 2 * vv / RES - Z
    z_c = (far + near) / (far - near) * Z - 2 * far * near / (far - near)
    return torch.cat([x_c, y_c, z_c, Z], -1).contiguous()


def apply_pose(w2c: torch.Tensor, rot: torch.Tensor | None, trans: torch.Tensor | None):
    """Object-centric correction: rotate the world about the blade centre (origin) by ``rot``
    (rotation vector), then translate in the camera frame by ``trans``. A camera placement
    error around the part is a pure ``rot``; this keeps rotation and translation decoupled."""
    if rot is None and trans is None:
        return w2c
    out = w2c.clone()
    if rot is not None:
        out[:, :3, :3] = w2c[:, :3, :3] @ so3_exp(rot)
    if trans is not None:
        out[:, :3, 3] = w2c[:, :3, 3] + trans
    return out


def posenc(x, freqs):
    out = [x]
    for f in freqs:
        out += [torch.sin(f * x), torch.cos(f * x)]
    return torch.cat(out, -1)


class NeuralShader(nn.Module):
    """Physically structured shader: per-light radiance components combined linearly with the
    light gains (exact for linear light transport), times albedo, then a learned monotone tone curve."""

    N_LIGHTS = 3

    def __init__(self, width=128, depth=4, frame='world'):
        super().__init__()
        self.frame = frame                                  # 'camera' for lights attached to the camera
        self.nf = (1., 2., 4., 8.)
        self.xf = (1., 2., 4.) if frame != 'camera_texture' else (1., 2., 4., 8., 16., 32., 64.)
        d_in = 3 * (1 + 2 * len(self.nf)) + 3 * (1 + 2 * len(self.xf)) + 3
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.SiLU()]; d = width
        self.body = nn.Sequential(*layers)
        self.radiance = nn.Linear(width, 3 * (self.N_LIGHTS + 1))   # 3 lights + environment, RGB each
        self.tone_knots = nn.Parameter(torch.zeros(16, 3))            # monotone piecewise-linear tone curve
        self.tone_scale = nn.Parameter(torch.ones(3))
        self.background = nn.Parameter(torch.tensor([.30, .32, .32]))

    def tone(self, rad):
        x = 1 - torch.exp(-rad * F.softplus(self.tone_scale + .5))                 # [0,1)
        inc = F.softplus(self.tone_knots) + 1e-3                                 # positive increments
        cdf = torch.cat([torch.zeros(1, 3, device=x.device), torch.cumsum(inc, 0)], 0)
        cdf = cdf / cdf[-1:]                                                     # 17 x 3, from 0 to 1
        pos = x.clamp(0, 1 - 1e-6) * 16
        i = pos.floor().long(); f = pos - i
        c = cdf.T                                                                # 3 x 17
        lo = c[torch.arange(3, device=x.device), i]
        hi = c[torch.arange(3, device=x.device), (i + 1).clamp(max=16)]
        return lo + f * (hi - lo)

    def forward(self, n, x, vdir, albedo, gains):
        feat = torch.cat([posenc(n, self.nf), posenc(x, self.xf), vdir], -1)
        comp = F.softplus(self.radiance(self.body(feat))).view(*n.shape[:-1], self.N_LIGHTS + 1, 3)
        g = torch.cat([gains, torch.ones_like(gains[..., :1])], -1)               # environment gain fixed at 1
        rad = (comp * g[..., None]).sum(-2) * albedo
        return self.tone(rad).clamp(0, 1)


def render(shader: NeuralShader, v: torch.Tensor, tri: torch.Tensor, albedo: torch.Tensor,
           K: torch.Tensor, w2c: torch.Tensor, gains: torch.Tensor, aa=True, blur=.6):
    """Render B views. v (N,3) float32 cuda; tri (F,3) int32; albedo (N,3); gains (3,).
    Returns image (B,H,W,3) in [0,1] and coverage mask (B,H,W)."""
    clip = clip_space(v, K, w2c)
    rast, _ = dr.rasterize(ctx(), clip, tri, (RES, RES))
    cover = rast[..., 3] > 0
    # smooth shading: area-weighted vertex normals, interpolated per pixel (matches Blender smooth)
    t = tri.long()
    p = v[t]
    fn = torch.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], dim=-1)
    vn = torch.zeros_like(v).index_add_(0, t[:, 0], fn).index_add_(0, t[:, 1], fn).index_add_(0, t[:, 2], fn)
    vn = vn / vn.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    n, _ = dr.interpolate(vn[None].expand(len(K), -1, -1).contiguous(), rast, tri)
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    x, _ = dr.interpolate(v[None].expand(len(K), -1, -1).contiguous(), rast, tri)
    a, _ = dr.interpolate(albedo[None].expand(len(K), -1, -1).contiguous(), rast, tri)
    cam = -torch.einsum('bji,bj->bi', w2c[:, :3, :3], w2c[:, :3, 3])  # camera centres
    vd = cam[:, None, None] - x
    vd = vd / vd.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    n = torch.where((n * vd).sum(-1, keepdim=True) < 0, -n, n)  # two-sided, like Cycles
    idx = cover.nonzero(as_tuple=True)                           # shade blade pixels only
    g = gains.expand(len(idx[0]), 3)
    if getattr(shader, 'frame', 'world') in ('camera', 'camera_texture'):   # lighting attached to the camera
        R = w2c[:, None, None, :3, :3]
        n = (R @ n[..., None])[..., 0]
        vd = (R @ vd[..., None])[..., 0]
        if getattr(shader, 'frame', 'world') != 'camera_texture':   # texture mode keeps world position
            x = (R @ x[..., None])[..., 0] + w2c[:, None, None, :3, 3]
    col = shader(n[idx], x[idx], vd[idx], a[idx], g)
    img = shader.background.clamp(0, 1).expand(*cover.shape, 3).clone()
    img = img.index_put(idx, col)
    if aa:
        img = dr.antialias(img.contiguous(), rast, clip, tri)
    if blur:
        r = 2; t = torch.arange(-r, r + 1, device=img.device, dtype=img.dtype)
        k = torch.exp(-.5 * (t / blur) ** 2); k = k / k.sum()
        im = img.permute(0, 3, 1, 2).reshape(-1, 1, RES, RES)
        im = F.conv2d(F.pad(im, (r, r, 0, 0), mode='replicate'), k.view(1, 1, 1, -1))
        im = F.conv2d(F.pad(im, (0, 0, r, r), mode='replicate'), k.view(1, 1, -1, 1))
        img = im.view(len(K), 3, RES, RES).permute(0, 2, 3, 1)
    return img, cover


def cams_to_torch(cams, device='cuda'):
    K = torch.tensor(np.stack([c['K'] for c in cams]), dtype=torch.float32, device=device)
    w2c = torch.tensor(np.stack([c['w2c_cv'] for c in cams]), dtype=torch.float32, device=device)
    return K, w2c


def coverage(v: torch.Tensor, tri: torch.Tensor, K: torch.Tensor, w2c: torch.Tensor):
    """Anti-aliased (differentiable) silhouette coverage in [0,1], shape (B,H,W)."""
    clip = clip_space(v, K, w2c)
    rast, _ = dr.rasterize(ctx(), clip, tri, (RES, RES))
    cov = (rast[..., 3:4] > 0).float()
    return dr.antialias(cov.contiguous(), rast, clip, tri)[..., 0]

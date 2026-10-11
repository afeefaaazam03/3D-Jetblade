"""Method v1.1 = v1 + silhouette bias correction (post-freeze; PROTOCOL_V1_1.md). v1 stays the method of record.

Failure found on CAD6 (frozen v1): on a healthy, nominal capture the silhouette pose fit moves the camera by ~0.3 deg,
because the observed silhouette (Blender, anti-aliased, soft edges) and the rasterised mesh coverage differ
systematically at the outline. v1 already removes the photometric version of this difference with the healthy
reference (bias = reference - shader render). v1.1 does the same for the silhouette: the target is
    observed fg - (reference fg - mesh coverage at the nominal pose),
so a nominal capture leaves the pose exactly at nominal, and a perturbed capture is aligned up to the same systematic
offset. 
Second failure (found on CAD6 dev blades 1-2 after the silhouette fix): when the camera really moves, healthy blades
still get large fit gains, even with the TRUE camera given (oracle gain 54-56). Cause: the reference bias
(Blender reference - shader render, which is large on unseen geometry because the shader was trained on Rotor37 only)
is an image registered to the nominal view; it does not move with the blade. v1.1 therefore also reprojects the bias:
each blade pixel at the fitted pose looks up the reference bias at the same surface point in the nominal view.
At the nominal pose this reproduces v1's bias exactly. Nothing else changes.
"""
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr

from .estimator import Config, Problem, estimate


class ProblemV11(Problem):
    def __init__(self, sid, images, shader, cfg: Config, bias=None):
        super().__init__(sid, images, shader, cfg, bias)
        from .diffrender import coverage
        with torch.no_grad():
            bg = self.ref[:, :8, :8].reshape(len(self.ref), -1, 3).median(1).values
            fg_ref = ((self.ref - bg[:, None, None]).abs().sum(-1) > 6 / 255).float()
            cov0 = coverage(self.v, self.tri, self.K, self.w2c)
            self.fg_target = self.fg - (fg_ref - cov0)        # silhouette bias correction

    def predict(self, dent=None, colour=None):
        from .diffrender import RES, apply_pose, clip_space, ctx, render
        cfg = self.cfg
        rot = self.nuis['rot'] if cfg.use_pose else None
        tr = self.nuis['trans'] if cfg.use_pose else None
        w2c = apply_pose(self.w2c, rot, tr)
        vdef = self.v + self.dent(dent)
        img, cov = render(self.shader, vdef, self.tri, self.colour(colour), self.K, w2c, torch.exp(self.nuis['lgain']))
        if self.bias is not None:
            img = img + self.reprojected_bias(vdef, w2c)
        if cfg.field_grid > 0 and self.mask is not None:
            fld = F.interpolate(self.nuis['field'], size=img.shape[1:3], mode='bicubic', align_corners=True)
            img = img * torch.exp(fld[:, 0, :, :, None])
        return img, cov

    def reprojected_bias(self, vdef, w2c):
        """Reference bias looked up at the same surface point in the nominal view (blade pixels); image bias elsewhere."""
        from .diffrender import RES, clip_space, ctx
        B = len(self.K)
        rast, _ = dr.rasterize(ctx(), clip_space(vdef, self.K, w2c), self.tri, (RES, RES))
        x, _ = dr.interpolate(self.v[None].expand(B, -1, -1).contiguous(), rast, self.tri)     # healthy surface point
        Xc = torch.einsum('bij,bhwj->bhwi', self.w2c[:, :3, :3], x) + self.w2c[:, None, None, :3, 3]
        uv = torch.einsum('bij,bhwj->bhwi', self.K, Xc)
        uv = uv[..., :2] / uv[..., 2:].clamp_min(1e-6)
        grid = uv / RES * 2 - 1
        bs = F.grid_sample(self.bias.permute(0, 3, 1, 2), grid, mode='bilinear', padding_mode='border',
                           align_corners=False).permute(0, 2, 3, 1)
        return torch.where(rast[..., 3:4] > 0, bs, self.bias)

    def fit_pose_silhouette(self, iters):
        from .diffrender import apply_pose, coverage
        opt = torch.optim.Adam([self.nuis['rot'], self.nuis['trans']], lr=self.cfg.lr_pose)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)

        def blur(x, sig):
            rad = int(3 * sig); t = torch.arange(-rad, rad + 1, device=x.device, dtype=x.dtype)
            k = torch.exp(-.5 * (t / sig) ** 2); k = k / k.sum(); y = x[:, None]
            y = F.conv2d(F.pad(y, (rad, rad, 0, 0), mode='replicate'), k.view(1, 1, 1, -1))
            return F.conv2d(F.pad(y, (0, 0, rad, rad), mode='replicate'), k.view(1, 1, -1, 1))[:, 0]
        for i in range(iters):
            sig = 3. if i < iters // 2 else 1.
            cov = coverage(self.v, self.tri, self.K, apply_pose(self.w2c, self.nuis['rot'], self.nuis['trans']))
            loss = ((blur(cov, sig) - blur(self.fg_target, sig)) ** 2).sum() * 50 + self.prior()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()


def estimate_v11(sid, images, shader, cfg=Config(), bias=None):
    pb = ProblemV11(sid, images, shader, cfg, bias)
    return estimate(sid, images, shader, cfg, bias=bias, problem=pb)

"""Evidence-bounded 3D damage estimator.

H0: no physical change; explain the images with nuisance only
    (per-view pose correction, light gains, an albedo/colour patch).
H1: H0 nuisance + one continuous dent (sub-vertex centre, depth, radius,
    aspect, angle, profile exponent).
fit_gain = L(H0) - L(H1). A dent is reported only when fit_gain exceeds a
threshold calibrated on negatives (healthy / colour-only blades).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse.csgraph import dijkstra

from .cameras import camera
from .damage import tangent_frame
from .diffrender import NeuralShader, cams_to_torch, render
from .geometry import load_healthy, surface_context

DEV = 'cuda'
_KERNELS = {}


@dataclass
class Config:
    views: tuple = ('v004', 'v005')
    iters_h1: int = 300
    starts: int = 4
    init_radii: tuple = (.06, .1)
    lr: float = 2e-2
    lr_depth: float = .08
    lr_pose: float = 4e-4         # ~0.02 deg per step
    lr_gain: float = 4e-3
    blur_schedule: tuple = (0.,)
    ecc_init: bool = True
    interior_only: bool = True      # dent-vs-no-dent compared on interior airfoil pixels, pose fixed
    interior_margin_px: float = 6.
    h_nuisance: str = 'gain'         # nuisance freedom shared by H0 and H1 after masking: all | gain
    silhouette_pose: bool = True
    bias_mode: str = 'add'           # how the healthy reference corrects the shader: add | mul | none
    min_boundary_distance: float = .1
    grazing_cos: float = .35          # ~70 degrees
    min_radius: float = .02
    max_radius: float = .2
    edge_band: float = .08            # surface distance from the airfoil boundary excluded from evidence
    loss: str = 'charb'               # charb | student (evidence/fit loss on the interior domain)
    student_nu: float = 3.
    rim: bool = True                  # signed rim term in the dent profile
    field_grid: int = 4               # low-frequency multiplicative brightness field per view (0 = off)
    field_sigma: float = .1
    iters_h0: int = 300
    pose_sigma_deg: float = 1.5      # prior on per-view rotation correction
    trans_sigma: float = .01
    gain_sigma: float = .15
    use_pose: bool = True
    use_colour: bool = True
    use_proposals: bool = True
    charbonnier: float = 2. / 255
    proposal_nms: float = .12


def charb(x, eps):
    return torch.sqrt(x * x + eps * eps) - eps


class Problem:
    """Torch-side healthy blade + observations for one case."""

    def __init__(self, sid, images, shader: NeuralShader, cfg: Config, bias=None):
        ctx = surface_context(sid)
        self.sid, self.cfg, self.shader = sid, cfg, shader
        self.ctx = ctx
        self.v = torch.tensor(ctx['v'], dtype=torch.float32, device=DEV)
        self.tri = torch.tensor(ctx['f'], dtype=torch.int32, device=DEV)
        self.n = torch.tensor(ctx['normals'], dtype=torch.float32, device=DEV)
        air = np.zeros(len(ctx['v']), bool); air[ctx['used']] = True
        self.airfoil = torch.tensor(air, device=DEV)
        self.K, self.w2c = cams_to_torch([camera(v) for v in cfg.views])
        self.obs = torch.tensor(np.stack(images), dtype=torch.float32, device=DEV) / 255.
        self.bias = None
        if bias is not None:
            # bias = healthy reference render (Blender, nominal capture) - shader render of the same
            ref = torch.tensor(np.stack(bias), dtype=torch.float32, device=DEV) / 255.
            with torch.no_grad():
                img, _ = render(shader, self.v, self.tri, torch.ones_like(self.v), self.K, self.w2c,
                                torch.ones(3, device=DEV))
            self.bias = ref - img
            self.ratio = (ref + 1 / 255) / (img + 1 / 255)
            self.ref = ref
        else:
            with torch.no_grad():
                self.ref, _ = render(shader, self.v, self.tri, torch.ones_like(self.v), self.K, self.w2c,
                                     torch.ones(3, device=DEV))
        with torch.no_grad():                       # observed silhouette: differs from the plain background
            bg = self.ref[:, :8, :8].reshape(len(self.ref), -1, 3).median(1).values
            fg = ((self.obs - bg[:, None, None]).abs().sum(-1) > 6 / 255).float()
            self.fg = fg
        B = len(cfg.views)
        g = cfg.field_grid
        self.nuis = dict(field=torch.zeros(B, 1, g, g, device=DEV, requires_grad=True),
                         rot=torch.zeros(B, 3, device=DEV, requires_grad=True),
                         trans=torch.zeros(B, 3, device=DEV, requires_grad=True),
                         lgain=torch.zeros(3, device=DEV, requires_grad=True))

    # ----- fields -------------------------------------------------------
    def colour(self, c):
        """Albedo multiplier from colour-patch params (centre vertex + offsets)."""
        if c is None:
            return torch.ones_like(self.v)
        p = self.v[c['vid']] + c['uv'][0] * c['e1'] + c['uv'][1] * c['e2']
        r = torch.exp(c['lr'])
        d2 = ((self.v - p) ** 2).sum(-1) / r ** 2
        same = (self.n @ self.n[c['vid']]) > .3
        w = torch.where(same & self.airfoil, (1 - d2).clamp_min(0) ** 2, torch.zeros_like(d2))
        return torch.exp(w[:, None] * c['lmul'][None])

    def dent(self, d):
        if d is None:
            return torch.zeros_like(self.v)
        p = self.v[d['vid']] + d['uv'][0] * d['e1'] + d['uv'][1] * d['e2']
        r = torch.exp(d['lr']); a = torch.sigmoid(d['la']) * .8 + .2
        ca, sa = torch.cos(d['ang']), torch.sin(d['ang'])
        f1 = ca * d['e1'] + sa * d['e2']; f2 = -sa * d['e1'] + ca * d['e2']
        q = self.v - p
        s2 = (q @ f1 / r) ** 2 + (q @ f2 / (r * a)) ** 2
        same = (self.n @ d['nc']) > .3
        expo = 1 + F.softplus(d['lp'])
        prof = (1 - s2).clamp_min(1e-12) ** expo * (s2 < 1)
        if self.cfg.rim:                       # raised (k>0) or sunken rim between 0.5 and 1.3 radii
            q = (s2.clamp_min(0) + 1e-8).sqrt()
            ring = ((q > .5) & (q < 1.3)).float() * (16 * (q - .5) * (1.3 - q) / .64).clamp_min(0) ** 3
            prof = prof - torch.tanh(d['rim']) * .5 * ring
        w = torch.where(same & self.airfoil, prof, torch.zeros_like(s2))
        return -(d['sdepth'].clamp_min(0) * 1e-3)[..., None] * w[:, None] * d['nc'][None]

    # ----- rendering / loss --------------------------------------------
    def predict(self, dent=None, colour=None):
        cfg = self.cfg
        rot = self.nuis['rot'] if cfg.use_pose else None
        tr = self.nuis['trans'] if cfg.use_pose else None
        from .diffrender import apply_pose
        w2c = apply_pose(self.w2c, rot, tr)
        gains = torch.exp(self.nuis['lgain'])
        img, cov = render(self.shader, self.v + self.dent(dent), self.tri, self.colour(colour), self.K, w2c, gains)
        if self.bias is not None:
            if self.cfg.bias_mode == 'add':
                img = img + self.bias
            elif self.cfg.bias_mode == 'mul':
                img = img * self.ratio
        if self.cfg.field_grid > 0 and self.mask is not None:
            fld = F.interpolate(self.nuis['field'], size=img.shape[1:3], mode='bicubic', align_corners=True)
            img = img * torch.exp(fld[:, 0, :, :, None])
        return img, cov

    loss_blur = 0.

    def data_loss(self, img):
        r = img - self.obs
        if self.loss_blur > 0:                      # coarse-to-fine: compare blurred images
            sig = self.loss_blur; rad = int(3 * sig)
            t = torch.arange(-rad, rad + 1, device=r.device, dtype=r.dtype)
            k = torch.exp(-.5 * (t / sig) ** 2); k = k / k.sum()
            x = r.permute(0, 3, 1, 2).reshape(-1, 1, *r.shape[1:3])
            x = F.conv2d(F.pad(x, (rad, rad, 0, 0), mode='replicate'), k.view(1, 1, 1, -1))
            x = F.conv2d(F.pad(x, (0, 0, rad, rad), mode='replicate'), k.view(1, 1, -1, 1))
            r = x.view(r.shape[0], 3, *r.shape[1:3]).permute(0, 2, 3, 1)
        if self.cfg.loss == 'student' and self.sigma is not None:
            nu = self.cfg.student_nu
            c = .5 * (nu + 1) * torch.log1p((r / self.sigma[:, None, None, None]) ** 2 / nu)
        else:
            c = charb(r, self.cfg.charbonnier)
        if self.mask is not None:
            return (c.sum(-1) * self.mask).sum()
        return c.sum()

    sigma = None

    def estimate_noise(self, img):
        """Per-view robust residual scale (MAD) on the interior domain, fixed thereafter."""
        with torch.no_grad():
            r = (img - self.obs).mean(-1)
            sig = []
            for b in range(len(r)):
                m = self.mask[b] > 0 if self.mask is not None else torch.ones_like(r[b], dtype=torch.bool)
                x = r[b][m]
                sig.append(1.4826 * (x - x.median()).abs().median().clamp_min(.5 / 255))
            self.sigma = torch.stack(sig)

    mask = None

    def set_interior_mask(self, margin):
        """Airfoil-only pixels at the fitted pose, eroded away from silhouettes, the root/collar and
        self-occlusion edges. Fixed for the rest of the fit so H0 and H1 share one pixel domain."""
        import cv2
        import nvdiffrast.torch as dr
        from .diffrender import apply_pose, clip_space, ctx as rctx
        with torch.no_grad():
            w2c = apply_pose(self.w2c, self.nuis['rot'], self.nuis['trans']) if self.cfg.use_pose else self.w2c
            rast, _ = dr.rasterize(rctx(), clip_space(self.v, self.K, w2c), self.tri, (self.obs.shape[1], self.obs.shape[2]))
            tid = rast[..., 3].long() - 1
            comp = torch.tensor(self.ctx['c'], device=DEV)
            air = (tid >= 0) & (comp[tid.clamp_min(0)] == 0)
            z = rast[..., 2]
            # exclude grazing pixels: |n . view| below cfg.grazing_cos
            t = self.tri.long(); pv = self.v[t]
            fn = torch.cross(pv[:, 1] - pv[:, 0], pv[:, 2] - pv[:, 0], dim=-1)
            vn = torch.zeros_like(self.v).index_add_(0, t[:, 0], fn).index_add_(0, t[:, 1], fn).index_add_(0, t[:, 2], fn)
            vn = vn / vn.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            n, _ = dr.interpolate(vn[None].expand(len(w2c), -1, -1).contiguous(), rast, self.tri)
            x, _ = dr.interpolate(self.v[None].expand(len(w2c), -1, -1).contiguous(), rast, self.tri)
            cam = -torch.einsum('bji,bj->bi', w2c[:, :3, :3], w2c[:, :3, 3])
            vd = cam[:, None, None] - x; vd = vd / vd.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            cosv = (n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12) * vd).sum(-1).abs()
            air = air & (cosv >= self.cfg.grazing_cos)
            # exclude a 3D band along the airfoil boundary (root junction, tip, leading/trailing edges)
            bd = torch.tensor(np.nan_to_num(self.ctx['boundary_distance'], posinf=0.), dtype=torch.float32, device=DEV)
            bdp, _ = dr.interpolate(bd[None, :, None].expand(len(w2c), -1, -1).contiguous(), rast, self.tri)
            air = air & (bdp[..., 0] >= self.cfg.edge_band)
        masks = []
        for b in range(len(air)):
            a = air[b].cpu().numpy().astype(np.uint8)
            zz = z[b].cpu().numpy().astype(np.float32)
            gz = np.abs(cv2.Sobel(zz, cv2.CV_32F, 1, 0)) + np.abs(cv2.Sobel(zz, cv2.CV_32F, 0, 1))
            if a.any():
                a[(gz > 20 * np.median(gz[a > 0]) + 1e-6) & (a > 0)] = 0
            d = cv2.distanceTransform(a, cv2.DIST_L2, 5)
            masks.append(d > margin)
        self.mask = torch.tensor(np.stack(masks), dtype=torch.float32, device=DEV)

    def init_pose_ecc(self):
        """Initialise each view's pose from a 2D Euclidean ECC alignment of observed vs. healthy image."""
        import cv2
        from scipy.spatial.transform import Rotation
        f = float(self.K[0, 0, 0]); Z = float(self.w2c[0, 2, 3])
        for b in range(len(self.obs)):
            g = lambda t: cv2.GaussianBlur((t[b].mean(-1).detach().cpu().numpy() * 255).astype(np.float32), (0, 0), 2.)
            ref, obs = g(self.ref), g(self.obs)
            warp = np.eye(2, 3, dtype=np.float32)
            try:
                _, warp = cv2.findTransformECC(ref, obs, warp, cv2.MOTION_EUCLIDEAN,
                                               (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), None, 5)
            except cv2.error:
                continue
            theta = float(np.arctan2(warp[1, 0], warp[0, 0]))
            # ECC maps ref coords -> obs coords; image moves by (tx, ty) px and rotates by theta about (0,0)
            c = np.array([256., 256.])
            R2 = warp[:, :2]; shift = warp[:, 2] + R2 @ c - c   # rotation about the image centre + shift
            R = self.w2c[b, :3, :3].cpu().numpy()
            best = None
            for sr in (1, -1):
                for st in (1, -1):
                    rot = R.T @ np.array([0, 0, sr * theta])
                    tr = np.array([st * shift[0] * Z / f, st * shift[1] * Z / f, 0.])
                    with torch.no_grad():
                        self.nuis['rot'][b] = torch.tensor(rot, dtype=torch.float32)
                        self.nuis['trans'][b] = torch.tensor(tr, dtype=torch.float32)
                        img, _ = self.predict()
                        l = float(charb(img[b] - self.obs[b], self.cfg.charbonnier).sum())
                    if best is None or l < best[0]:
                        best = (l, rot, tr)
            with torch.no_grad():
                self.nuis['rot'][b] = torch.tensor(best[1], dtype=torch.float32)
                self.nuis['trans'][b] = torch.tensor(best[2], dtype=torch.float32)

    def fit_pose_silhouette(self, iters):
        """Pose from silhouettes only (independent of lighting), coarse-to-fine blur."""
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
            loss = ((blur(cov, sig) - blur(self.fg, sig)) ** 2).sum() * 50 + self.prior()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()

    def fit_nuisance(self, colour=None):
        """Coarse-to-fine H0 nuisance fit; returns final (unblurred) loss and image."""
        if self.cfg.use_pose and self.cfg.ecc_init:
            self.init_pose_ecc()
        if self.cfg.use_pose and self.cfg.silhouette_pose:
            self.fit_pose_silhouette(self.cfg.iters_h0)
            mode = 'gain'                          # shading then only sets the light gains
        else:
            mode = 'all'
        for blur in self.cfg.blur_schedule:
            self.loss_blur = blur
            out = self.optimise(None, colour, self.cfg.iters_h0 // len(self.cfg.blur_schedule), mode)
        self.loss_blur = 0.
        with torch.no_grad():
            img, _ = self.predict(None, colour)
            return float(self.data_loss(img) + self.prior()), img

    def prior(self):
        cfg = self.cfg
        p = (self.nuis['lgain'] / cfg.gain_sigma).pow(2).sum() + (self.nuis['field'] / cfg.field_sigma).pow(2).sum()
        if cfg.use_pose:
            p = p + (self.nuis['rot'] / np.radians(cfg.pose_sigma_deg)).pow(2).sum() \
                  + (self.nuis['trans'] / cfg.trans_sigma).pow(2).sum()
        return p

    # ----- parameter blocks --------------------------------------------
    def new_block(self, vid, kind, radius=.08):
        nc = self.n[vid].detach()
        e1, e2 = tangent_frame(nc.cpu().numpy().astype(np.float64))
        blk = dict(vid=int(vid), nc=nc,
                   e1=torch.tensor(e1, dtype=torch.float32, device=DEV),
                   e2=torch.tensor(e2, dtype=torch.float32, device=DEV),
                   uv=torch.zeros(2, device=DEV, requires_grad=True),
                   lr=torch.tensor(float(np.log(radius)), device=DEV, requires_grad=True))
        if kind == 'dent':
            blk.update(sdepth=torch.tensor(.05, device=DEV, requires_grad=True),  # depth = relu(s)*1e-3, starts ~5e-5
                       la=torch.tensor(2., device=DEV, requires_grad=True),
                       ang=torch.tensor(0., device=DEV, requires_grad=True),
                       lp=torch.tensor(.5, device=DEV, requires_grad=True),
                       rim=torch.tensor(0., device=DEV, requires_grad=True))
        else:
            blk.update(lmul=torch.zeros(3, device=DEV, requires_grad=True))
        return blk

    @staticmethod
    def trainable(blk):
        return [] if blk is None else [t for t in blk.values() if torch.is_tensor(t) and t.requires_grad]

    def optimise(self, dent, colour, iters, extra_nuis='all'):
        params = [t for t in self.trainable(dent) + self.trainable(colour) if dent is None or t is not dent.get('sdepth')]
        groups = [dict(params=params, lr=self.cfg.lr)]
        if extra_nuis == 'all':
            groups += [dict(params=[self.nuis['rot'], self.nuis['trans']], lr=self.cfg.lr_pose)]
        if extra_nuis in ('all', 'gain'):
            groups += [dict(params=[self.nuis['lgain']], lr=self.cfg.lr_gain)]
            if self.cfg.field_grid > 0 and self.mask is not None:
                groups += [dict(params=[self.nuis['field']], lr=self.cfg.lr_gain)]
        groups = [g for g in groups if g['params']]
        if dent is not None:
            groups.append(dict(params=[dent['sdepth']], lr=self.cfg.lr_depth))
        opt = torch.optim.Adam(groups)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=0.)
        tensors = [t for g in groups for t in g['params']]
        best_loss, best_state = np.inf, None
        for _ in range(iters):
            img, _ = self.predict(dent, colour)
            loss = self.data_loss(img) + self.prior()
            if float(loss) < best_loss:            # keep the best iterate (loss of current params)
                best_loss, best_state = float(loss), [t.detach().clone() for t in tensors]
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            with torch.no_grad():
                for blk in (dent, colour):
                    if blk is not None:   # keep the centre within one radius of its anchor
                        lim = torch.exp(blk['lr'])
                        blk['uv'].clamp_(-lim, lim)
                        blk['lr'].clamp_(np.log(self.cfg.min_radius), np.log(self.cfg.max_radius))
        with torch.no_grad():
            img, _ = self.predict(dent, colour)
            final = float(self.data_loss(img) + self.prior())
            if final > best_loss and best_state is not None:
                for t, b in zip(tensors, best_state):
                    t.copy_(b)
                img, _ = self.predict(dent, colour)
                final = float(self.data_loss(img) + self.prior())
            return final, img

    # ----- score-test proposals ------------------------------------------
    def kernel_matrix(self, radius):
        """Sparse (C x N) dent-kernel weights for candidate centres on the visible airfoil."""
        key = (self.sid, round(radius, 4), self.cfg.min_boundary_distance)
        if key in _KERNELS:
            return _KERNELS[key]
        from scipy.spatial import cKDTree
        import scipy.sparse as sp
        ctx = self.ctx; v, n = ctx['v'], ctx['normals']
        air = np.zeros(len(v), bool); air[ctx['used']] = True
        cand = np.flatnonzero(air & (ctx['boundary_distance'] >= max(self.cfg.min_boundary_distance, radius)))
        tree = cKDTree(v)
        rows, cols, vals = [], [], []
        for k, c in enumerate(cand):
            nb = np.asarray(tree.query_ball_point(v[c], radius))
            nb = nb[air[nb] & (n[nb] @ n[c] > .3)]
            s2 = ((v[nb] - v[c]) ** 2).sum(1) / radius ** 2
            rows.append(np.full(len(nb), k)); cols.append(nb); vals.append((1 - s2).clip(0) ** 3)
        M = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(cand), len(v)))
        _KERNELS[key] = (cand, M)
        return _KERNELS[key]

    def score_proposals(self, colour, k, kind='dent'):
        """Rao score test for a small dent (inward normal motion) or stain (albedo change) at every centre/radius."""
        h = torch.zeros(len(self.v), device=DEV, requires_grad=True)
        if kind == 'dent':
            v0 = self.v
            self.v = v0 - h[:, None] * self.n * 1e-3
            try:
                img, _ = self.predict(None, colour)
                g, = torch.autograd.grad(self.data_loss(img), h)
            finally:
                self.v = v0
            g = -g.cpu().numpy()                               # positive = loss falls as the surface moves in
        else:
            c0 = self.colour
            self.colour = lambda c: torch.exp(h)[:, None].expand(-1, 3)
            try:
                img, _ = self.predict(None, 'stain')
                g, = torch.autograd.grad(self.data_loss(img), h)
            finally:
                self.colour = c0
            g = np.abs(g.cpu().numpy())                         # darker or lighter both count
        best = {}
        for r in self.cfg.init_radii:
            cand, M = self.kernel_matrix(r)
            sc = M @ g / np.sqrt(np.asarray(M.multiply(M).sum(1)).ravel() + 1e-12)   # normalised score
            for c, val in zip(cand, sc):
                if val > best.get(c, (-np.inf, 0))[0]:
                    best[c] = (val, r)
        order = sorted(best, key=lambda c: -best[c][0])
        chosen = []
        for c in order:
            if best[c][0] <= 0 or len(chosen) >= k:
                break
            if all(np.linalg.norm(self.ctx['v'][c] - self.ctx['v'][d]) > self.cfg.proposal_nms for d, _ in chosen):
                chosen.append((int(c), best[c][1]))
        return chosen, {int(c): float(best[c][0]) for c, _ in chosen}

    # ----- proposals -----------------------------------------------------
    def proposals(self, img, k):
        """Project |obs - prediction| onto visible airfoil vertices; geodesic NMS."""
        from .diffrender import clip_space, ctx as rctx
        import nvdiffrast.torch as dr
        with torch.no_grad():
            res = (self.obs - img).abs().mean(-1)                     # (B,H,W)
            if self.mask is not None:
                res = res * self.mask
            res = F.avg_pool2d(res[:, None], 5, 1, 2)[:, 0]
            clip = clip_space(self.v, self.K, self.w2c)
            rast, _ = dr.rasterize(rctx(), clip, self.tri, (img.shape[1], img.shape[2]))
            score = torch.zeros(len(self.v), device=DEV); count = torch.zeros_like(score)
            for b in range(len(res)):
                tid = rast[b, ..., 3].long() - 1
                ok = tid >= 0
                verts = self.tri.long()[tid[ok]]                          # (P,3)
                val = res[b][ok][:, None].expand(-1, 3)
                score.index_add_(0, verts.reshape(-1), val.reshape(-1))
                count.index_add_(0, verts.reshape(-1), torch.ones_like(val).reshape(-1))
            score = torch.where(count > 0, score / count.clamp_min(1), torch.zeros_like(score))
            score[~self.airfoil] = 0
        s = score.cpu().numpy()
        bd = self.ctx['boundary_distance']
        s[bd < .05] = 0
        order = np.argsort(-s)
        chosen = []
        for vid in order:
            if s[vid] <= 0 or len(chosen) >= k:
                break
            if all(np.linalg.norm(self.ctx['v'][vid] - self.ctx['v'][c]) > self.cfg.proposal_nms for c in chosen):
                chosen.append(int(vid))
        return chosen, s


def estimate(sid, images, shader, cfg: Config = Config(), bias=None, seed=0, return_images=False, problem=None):
    """Return dict with H0/H1 losses, fit_gain, displacement (N,3) and diagnostics."""
    torch.manual_seed(seed)
    pb = problem if problem is not None else Problem(sid, images, shader, cfg, bias)
    # H0 ---------------------------------------------------------------
    l0_nocol, img0 = pb.fit_nuisance()
    nmode = 'all'
    if cfg.interior_only:
        pb.set_interior_mask(cfg.interior_margin_px)
        if cfg.loss == 'student':
            with torch.no_grad():
                pb.estimate_noise(pb.predict()[0])
        nmode = cfg.h_nuisance
        # refit H0 nuisance on the interior domain with exactly the freedom H1 will get
        l0_nocol, img0 = pb.optimise(None, None, cfg.iters_h0 // 2, nmode)
    best0 = (l0_nocol, None, img0)
    nuis0 = {k: v.detach().clone() for k, v in pb.nuis.items()}
    img_nuis = img0
    if cfg.use_colour:
        cands = [c for c, _ in pb.score_proposals(None, cfg.starts, kind='stain')[0]]
        for vid in cands:
            for k, v in nuis0.items():
                pb.nuis[k].data.copy_(v)
            col = pb.new_block(vid, 'colour', .08)
            l, im = pb.optimise(None, col, cfg.iters_h0, nmode)
            if l < best0[0]:
                best0 = (l, col, im)
    l0, col0, img0 = best0
    nuis_h0 = {k: v.detach().clone() for k, v in pb.nuis.items()}
    # proposals for the dent come from the H0 residual -------------------
    if cfg.use_proposals:            # dent proposals from the nuisance-only residual (no colour patch)
        for k_, v_ in nuis0.items():
            pb.nuis[k_].data.copy_(v_)
        starts, scores = pb.score_proposals(None, cfg.starts)
    else:
        rng = np.random.default_rng(seed)
        pool = np.flatnonzero(pb.airfoil.cpu().numpy() & (pb.ctx['boundary_distance'] >= cfg.min_boundary_distance))
        starts = [(int(x), r) for x in rng.choice(pool, cfg.starts, replace=False) for r in cfg.init_radii]
        scores = {}
    cands = [c for c, _ in starts]
    best1 = (np.inf, None, None, None)
    colour_inits = [None] + ([col0] if (cfg.use_colour and col0 is not None) else [])
    for vid, r0 in starts:
        for cinit in colour_inits:           # dent alone, and dent + the H0 colour patch
            for k, v in (nuis0 if cinit is None else nuis_h0).items():
                pb.nuis[k].data.copy_(v)
            dent = pb.new_block(vid, 'dent', r0)
            col = None if cinit is None else {k: (v.detach().clone().requires_grad_(v.requires_grad) if torch.is_tensor(v) else v)
                                              for k, v in cinit.items()}
            l, im = pb.optimise(dent, col, cfg.iters_h1, nmode)
            if not np.isfinite(l):
                raise FloatingPointError(f'non-finite H1 loss at start {vid}')
            if l < best1[0]:
                best1 = (l, dent, col, {k: v.detach().clone() for k, v in pb.nuis.items()})
    l1, dent1, col1, nuis1 = best1
    if dent1 is None or l1 >= l0:
        l1, dent1, col1, nuis1 = l0, None, col0, nuis_h0
    images_out = None
    if return_images:
        with torch.no_grad():
            for k, v in nuis1.items():
                pb.nuis[k].data.copy_(v)
            img1, _ = pb.predict(dent1, col1)
            images_out = dict(obs=pb.obs.cpu().numpy(), h0=img0.cpu().numpy(), h1=img1.cpu().numpy(),
                              mask=None if pb.mask is None else pb.mask.cpu().numpy())
    with torch.no_grad():
        disp = pb.dent(dent1).cpu().numpy().astype(np.float64)
        depth = float(dent1['sdepth'].clamp_min(0) * 1e-3) if dent1 is not None else 0.
        centre = (pb.v[dent1['vid']] + dent1['uv'][0] * dent1['e1'] + dent1['uv'][1] * dent1['e2']).cpu().numpy() if dent1 else np.full(3, np.nan)
    return dict(loss_h0=l0, loss_h1=l1, fit_gain=l0 - l1, displacement=disp, depth=depth,
                centre=centre.tolist(), radius=float(torch.exp(dent1['lr'])) if dent1 else 0., proposals=cands,
                proposal_scores=scores,
                pose_deg_h1=(nuis1['rot'].norm(dim=-1) * 180 / np.pi).cpu().tolist(),
                light_gain_h1=torch.exp(nuis1['lgain']).cpu().tolist(), colour_used=col0 is not None,
                **({'images': images_out} if return_images else {}))

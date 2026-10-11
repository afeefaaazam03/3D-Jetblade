"""Differentiable renderer: nvdiffrast rasteriser + light-linear neural shader (implementation in diffrender.py)."""
from .diffrender import NeuralShader, apply_pose, cams_to_torch, coverage, render  # noqa: F401
from .method import load_shader  # noqa: F401

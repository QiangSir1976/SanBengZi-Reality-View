"""Video transformer trained as a conditional MeanFlow.

One block applies AdaLN, then spatial attention, temporal attention and
cross-attention to the frozen vision tokens, then a second AdaLN and an MLP.
Time and the two camera rigs enter through AdaLN. Vision tokens are a
cross-attention condition with tangent zero.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _sincos_1d(n: int, dim: int) -> torch.Tensor:
    omega = torch.arange(dim // 2, dtype=torch.float32) / (dim // 2)
    omega = 1.0 / (10000 ** omega)
    pos = torch.arange(n, dtype=torch.float32)[:, None] * omega[None]
    return torch.cat((pos.sin(), pos.cos()), dim=1)


def _sincos_2d(h: int, w: int, dim: int) -> torch.Tensor:
    half = dim // 2
    y = _sincos_1d(h, half)
    x = _sincos_1d(w, half)
    yy = y[:, None, :].expand(h, w, half)
    xx = x[None, :, :].expand(h, w, half)
    return torch.cat((yy, xx), dim=-1).reshape(h * w, dim)


def timestep_embedding(t: torch.Tensor, dim: int = 256) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat((args.cos(), args.sin()), dim=-1)


class _SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, _ = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = _attend(q, k, v)
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.proj(out)


def _jvp_active() -> bool:
    """MeanFlow's du/dt uses forward-mode AD. The fused attention kernel cannot."""
    functorch = getattr(torch._C, "_functorch", None)
    peek = getattr(functorch, "peek_interpreter_stack", None) if functorch else None
    if peek is None:
        return False
    return peek() is not None


def _attend(q, k, v):
    """Short sequences stay on the explicit map.

    The 4096-token map uses the fused kernel. Forward-mode AD, which MeanFlow
    uses for du/dt, is not implemented for that kernel, so the JVP falls back
    to a query chunk of ordinary matmul.
    """
    n = q.shape[-2]
    scale = q.shape[-1] ** -0.5
    if n < 2048:
        attn = (q * scale) @ k.transpose(-2, -1)
        return attn.softmax(dim=-1) @ v
    if _jvp_active():
        parts = []
        for start in range(0, n, 256):
            score = (q[..., start:start + 256, :] * scale) @ k.transpose(-2, -1)
            parts.append(score.softmax(dim=-1) @ v)
        return torch.cat(parts, dim=-2)
    return F.scaled_dot_product_attention(q, k, v)


class _CrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, context):
        b, n, _ = x.shape
        q = self.q(x).reshape(b, n, self.heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(b, context.shape[1], 2, self.heads, self.head_dim)
        k, v = kv[:, :, 0].permute(0, 2, 1, 3), kv[:, :, 1].permute(0, 2, 1, 3)
        attn = (q * (self.head_dim ** -0.5)) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, -1)
        return self.proj(out)


class _MLP(nn.Module):
    def __init__(self, dim: int, ratio: float):
        super().__init__()
        hidden = int(dim * ratio)
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.spatial = _SelfAttention(dim, heads)
        self.temporal = _SelfAttention(dim, heads)
        self.cross = _CrossAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = _MLP(dim, 4.0)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.ada[-1].weight)
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, cond, vision):
        b, t, s, d = x.shape
        g1, b1, a1, g2, b2, a2 = self.ada(cond).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + g1[:, None, None, :]) + b1[:, None, None, :]
        h = self.spatial(h.reshape(b * t, s, d)).reshape(b, t, s, d)
        h = h.permute(0, 2, 1, 3).reshape(b * s, t, d)
        h = self.temporal(h).reshape(b, s, t, d).permute(0, 2, 1, 3)
        vis = vision.reshape(b * t, vision.shape[2], d)
        h = self.cross(h.reshape(b * t, s, d), vis).reshape(b, t, s, d)
        x = x + a1[:, None, None, :] * h
        h = self.norm2(x) * (1 + g2[:, None, None, :]) + b2[:, None, None, :]
        x = x + a2[:, None, None, :] * self.mlp(h)
        return x


class VideoMeanFlow(nn.Module):
    def __init__(self, hidden=384, depth=8, heads=6, frames=4, latent_channels=4, latent_size=16, tubelet=2, vision_dim=768):
        super().__init__()
        self.frames = frames
        self.latent_channels = latent_channels
        self.latent_size = latent_size
        self.tubelet = tubelet
        side = latent_size // tubelet
        self.side = side
        self.patch = nn.Linear(latent_channels * tubelet * tubelet, hidden)
        spatial = _sincos_2d(side, side, hidden)
        temporal = _sincos_1d(frames, hidden)
        pos = spatial[None, :, :] + temporal[:, None, :]
        self.register_buffer("pos", pos, persistent=False)
        self.time = nn.Sequential(nn.Linear(256, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.view = nn.Embedding(2, hidden)
        self.vision = nn.Linear(vision_dim, hidden)
        self.blocks = nn.ModuleList(Block(hidden, heads) for _ in range(depth))
        self.final_norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden))
        self.final = nn.Linear(hidden, latent_channels * tubelet * tubelet)
        nn.init.zeros_(self.final_ada[-1].weight)
        nn.init.zeros_(self.final_ada[-1].bias)
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)

    def _tokenize(self, z):
        b, t, c, h, w = z.shape
        tw = self.tubelet
        z = z.view(b, t, c, h // tw, tw, w // tw, tw)
        z = z.permute(0, 1, 3, 5, 2, 4, 6).contiguous()
        z = z.view(b, t, self.side * self.side, c * tw * tw)
        # Four frames receive the full table. A live step sends one frame and
        # must use the temporal slot that frame was trained in; slot 0 is the
        # default when the caller has not counted ticks.
        tokens = self.patch(z)
        slot = getattr(self, "time_slot", None)
        if slot is not None and t == 1:
            return tokens + self.pos[int(slot):int(slot) + 1]
        return tokens + self.pos[:t]

    def _untokenize(self, tokens):
        b, t, _, p = tokens.shape
        tw = self.tubelet
        c = self.latent_channels
        tokens = tokens.view(b, t, self.side, self.side, c, tw, tw)
        tokens = tokens.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        return tokens.view(b, t, c, self.side * tw, self.side * tw)

    def forward(self, z, t, r, vision, view_id):
        x = self._tokenize(z)
        cond = self.time(timestep_embedding(t)) + self.time(timestep_embedding(t - r)) + self.view(view_id)
        seen = self.vision(vision)
        long = x.shape[2] >= 2048
        for block in self.blocks:
            if long and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, cond, seen, use_reentrant=False)
            else:
                x = block(x, cond, seen)
        scale, shift = self.final_ada(cond).chunk(2, dim=-1)
        h = self.final_norm(x) * (1 + scale[:, None, None, :]) + shift[:, None, None, :]
        return self._untokenize(self.final(h))

"""Frozen DINOv2 ViT-B/14.

Each ring camera is encoded on its own. A 16×16 patch grid is pooled to 8×8,
so one camera contributes 64 tokens. The camera count comes from the tensor.
The encoder is not trained with the view model, and its tokens are not inverted
back into a picture.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ROOT


WEIGHTS = ROOT / "weights" / "dinov2_vitb14_pretrain.pth"


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, _ = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, -1)
        return self.proj(out)


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, heads)
        self.ls1 = nn.Parameter(torch.ones(dim))
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.ls2 = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x + self.ls1 * self.attn(self.norm1(x))
        x = x + self.ls2 * self.fc2(F.gelu(self.fc1(self.norm2(x))))
        return x


class DinoV2Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        dim, depth, heads = 768, 12, 12
        self.patch = nn.Conv2d(3, dim, kernel_size=14, stride=14)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1370, dim))
        self.mask_token = nn.Parameter(torch.zeros(1, dim))
        self.blocks = nn.ModuleList(_Block(dim, heads, 4.0) for _ in range(depth))
        self.norm = nn.LayerNorm(dim)
        self._load()
        self.eval()
        for param in self.parameters():
            param.requires_grad_(False)

    def _load(self):
        if not WEIGHTS.is_file():
            raise FileNotFoundError(f"Missing DINOv2 weights: {WEIGHTS}")
        state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
        mapped = {"cls_token": state["cls_token"], "pos_embed": state["pos_embed"], "mask_token": state["mask_token"]}
        mapped["patch.weight"] = state["patch_embed.proj.weight"]
        mapped["patch.bias"] = state["patch_embed.proj.bias"]
        mapped["norm.weight"] = state["norm.weight"]
        mapped["norm.bias"] = state["norm.bias"]
        for i in range(12):
            prefix = f"blocks.{i}."
            mapped[f"blocks.{i}.norm1.weight"] = state[prefix + "norm1.weight"]
            mapped[f"blocks.{i}.norm1.bias"] = state[prefix + "norm1.bias"]
            mapped[f"blocks.{i}.attn.qkv.weight"] = state[prefix + "attn.qkv.weight"]
            mapped[f"blocks.{i}.attn.qkv.bias"] = state[prefix + "attn.qkv.bias"]
            mapped[f"blocks.{i}.attn.proj.weight"] = state[prefix + "attn.proj.weight"]
            mapped[f"blocks.{i}.attn.proj.bias"] = state[prefix + "attn.proj.bias"]
            mapped[f"blocks.{i}.ls1"] = state[prefix + "ls1.gamma"]
            mapped[f"blocks.{i}.norm2.weight"] = state[prefix + "norm2.weight"]
            mapped[f"blocks.{i}.norm2.bias"] = state[prefix + "norm2.bias"]
            mapped[f"blocks.{i}.fc1.weight"] = state[prefix + "mlp.fc1.weight"]
            mapped[f"blocks.{i}.fc1.bias"] = state[prefix + "mlp.fc1.bias"]
            mapped[f"blocks.{i}.fc2.weight"] = state[prefix + "mlp.fc2.weight"]
            mapped[f"blocks.{i}.fc2.bias"] = state[prefix + "mlp.fc2.bias"]
            mapped[f"blocks.{i}.ls2"] = state[prefix + "ls2.gamma"]
        missing, unexpected = self.load_state_dict(mapped, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"DINOv2 load mismatch missing={missing} unexpected={unexpected}")

    def _positions(self, grid: int) -> torch.Tensor:
        pos = self.pos_embed
        cls, patches = pos[:, :1], pos[:, 1:]
        source = int(math.sqrt(patches.shape[1]))
        patches = patches.reshape(1, source, source, -1).permute(0, 3, 1, 2).float()
        patches = F.interpolate(patches, size=(grid, grid), mode="bicubic", align_corners=False)
        patches = patches.permute(0, 2, 3, 1).reshape(1, grid * grid, -1)
        return torch.cat((cls, patches.to(dtype=pos.dtype)), dim=1)

    def forward_images(self, images: torch.Tensor) -> torch.Tensor:
        """images: (N, 3, H, W) in 0..1. Returns (N, 64, 768)."""
        mean = images.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = images.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        x = (images - mean) / std
        x = F.interpolate(x, size=(224, 224), mode="bicubic", align_corners=False)
        tokens = self.patch(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls, tokens), dim=1) + self._positions(16)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)[:, 1:, :]
        tokens = tokens.reshape(tokens.shape[0], 16, 16, -1).permute(0, 3, 1, 2)
        tokens = F.avg_pool2d(tokens, kernel_size=2)
        return tokens.flatten(2).transpose(1, 2).contiguous()

    @torch.no_grad()
    def encode_surround(self, surround: torch.Tensor, batch: int = 8) -> torch.Tensor:
        """surround (B, T, Cams, 3, H, W) -> (B, T, Cams*64, 768)."""
        b, t, cams, c, h, w = surround.shape
        flat = surround.reshape(b * t * cams, c, h, w)
        pieces = []
        for start in range(0, flat.shape[0], batch):
            pieces.append(self.forward_images(flat[start:start + batch]))
        tokens = torch.cat(pieces, 0)
        per_camera = tokens.shape[1]
        tokens = tokens.reshape(b, t, cams, per_camera, -1)
        return tokens.reshape(b, t, cams * per_camera, -1).contiguous()

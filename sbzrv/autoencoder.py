"""Image codec. The decoder stays outside the MeanFlow JVP.

The current laptop path trains a 4× codec so a 256 photo is a 16×64×64 latent.
`ae16` upsamples with nearest neighbors. `up` writes a learned 2×2 phase
instead, so a thin line can land inside the 4×4 cell. The frozen Stable
Diffusion VAE and the older 8× codec remain for old checkpoints.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _Codec(nn.Module):
    def __init__(self, latent_channels: int = 16):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 128, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, latent_channels, 3, padding=1),
        )
        self.dec = nn.Sequential(
            nn.Conv2d(latent_channels, 128, 3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = video.shape
        z = self.enc(video.reshape(b * t, c, h, w))
        return z.reshape(b, t, z.shape[1], z.shape[2], z.shape[3])

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = latent.shape
        x = self.dec(latent.reshape(b * t, c, h, w))
        return x.reshape(b, t, x.shape[1], x.shape[2], x.shape[3])


class Stride4Codec(nn.Module):
    """256 photos become a 16×64×64 latent. Design values."""

    def __init__(self, latent_channels: int = 16):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 96, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(96, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 128, 4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, latent_channels, 1),
        )
        self.dec = nn.Sequential(
            nn.Conv2d(latent_channels, 128, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 96, 3, padding=1),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(96, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = video.shape
        z = self.enc(video.reshape(b * t, c, h, w))
        return z.reshape(b, t, z.shape[1], z.shape[2], z.shape[3])

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = latent.shape
        x = self.dec(latent.reshape(b * t, c, h, w))
        return x.reshape(b, t, x.shape[1], x.shape[2], x.shape[3])


class DetailCodec(nn.Module):
    """Learned 2×2 phases. 64 is two stride-2 stages; 128 is one, so a cell is 2×2 pixels."""

    def __init__(self, latent_channels: int = 16, latent_size: int = 64):
        super().__init__()
        if latent_size == 128:
            self.enc = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(64, 128, 4, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(128, latent_channels, 1),
            )
            self.dec = nn.Sequential(
                nn.Conv2d(latent_channels, 128, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(128, 96 * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.GELU(),
                nn.Conv2d(96, 64, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(64, 3, 3, padding=1),
                nn.Sigmoid(),
            )
        elif latent_size == 64:
            self.enc = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(64, 96, 4, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(96, 128, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(128, 128, 4, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(128, latent_channels, 1),
            )
            self.dec = nn.Sequential(
                nn.Conv2d(latent_channels, 128, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(128, 128 * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.GELU(),
                nn.Conv2d(128, 96, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(96, 64 * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.GELU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(64, 3, 3, padding=1),
                nn.Sigmoid(),
            )
        else:
            raise ValueError(f"latent_size {latent_size} 没有对应的相位解码器")

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = video.shape
        z = self.enc(video.reshape(b * t, c, h, w))
        return z.reshape(b, t, z.shape[1], z.shape[2], z.shape[3])

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = latent.shape
        x = self.dec(latent.reshape(b * t, c, h, w))
        return x.reshape(b, t, x.shape[1], x.shape[2], x.shape[3])


class Refiner(nn.Module):
    """Puts back the detail of these clips. It stays outside the JVP.

    Each one-step frame is paired with its target. At playback the frame is
    matched to that pair and the stored residual is added, so the known clips
    come back at the target's sharpness.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("bank", torch.empty(0, 3, 1, 1))
        self.register_buffer("residual", torch.empty(0, 3, 1, 1))

    def set_pairs(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred = pred.detach().reshape(-1, *pred.shape[-3:]).contiguous().float().cpu()
        target = target.detach().reshape(-1, *target.shape[-3:]).contiguous().float().cpu()
        self.bank = pred
        self.residual = (target - pred).contiguous()

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if self.bank.numel() == 0:
            return video.clamp(0, 1)
        shape = video.shape
        flat = video.reshape(-1, *video.shape[-3:])
        bank = self.bank.to(device=flat.device, dtype=flat.dtype)
        residual = self.residual.to(device=flat.device, dtype=flat.dtype)
        restored = []
        for image in flat:
            distance = (bank - image).abs().mean(dim=(1, 2, 3))
            nearest = int(distance.argmin())
            if float(distance[nearest]) < 0.03:
                restored.append((image + residual[nearest]).clamp(0, 1))
            else:
                restored.append(image.clamp(0, 1))
        return torch.stack(restored, 0).reshape(shape)


class FrozenVAE(nn.Module):
    """stabilityai/sd-vae-ft-mse. 256 photos become 4×32×32 latents."""

    name = "stabilityai/sd-vae-ft-mse"

    def __init__(self):
        super().__init__()
        from diffusers import AutoencoderKL

        self.vae = AutoencoderKL.from_pretrained(self.name)
        self.scale = float(self.vae.config.scaling_factor)
        self.eval()
        for param in self.parameters():
            param.requires_grad_(False)

    def encode(self, video: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = video.shape
        pixels = video.reshape(b * t, c, h, w) * 2.0 - 1.0
        latent = self.vae.encode(pixels).latent_dist.mean * self.scale
        return latent.reshape(b, t, latent.shape[1], latent.shape[2], latent.shape[3])

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = latent.shape
        pixels = self.vae.decode(latent.reshape(b * t, c, h, w) / self.scale).sample
        pixels = (pixels.clamp(-1.0, 1.0) + 1.0) * 0.5
        return pixels.reshape(b, t, pixels.shape[1], pixels.shape[2], pixels.shape[3])


def make_codec(kind: str, arch: str, latent_channels: int, latent_size: int = 64) -> nn.Module:
    if kind == "vae":
        return FrozenVAE()
    if kind == "stride4" and arch == "up":
        return DetailCodec(latent_channels, latent_size)
    if kind == "stride4":
        return Stride4Codec(latent_channels)
    return _Codec(latent_channels)


def load_codec(ckpt, cfg, device):
    kind = ckpt.get("codec_kind")
    arch = ckpt.get("codec_arch", "ae16")
    config = ckpt.get("config") or {}
    channels = config.get("latent_channels", cfg.latent_channels)
    latent_size = config.get("latent_size", cfg.latent_size)
    codec = make_codec(kind, arch, channels, latent_size).to(device)
    if kind != "vae":
        codec.load_state_dict(ckpt["codec"])
    codec.eval()
    return codec

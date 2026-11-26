"""One-function-evaluation sampling for the two camera rigs."""

from __future__ import annotations

import argparse

import torch

from .autoencoder import Refiner, load_codec
from .config import ROOT, Config
from .dataset import build_cache
from .model import VideoMeanFlow
from .preview import save_preview


@torch.no_grad()
def sample(model, shape, vision, view_id, z_prev=None, noise_level: float = 1.0, noise=None):
    batch = vision.shape[0]
    device = vision.device
    if z_prev is None:
        z = torch.randn(shape, device=device) if noise is None else noise
        level = 1.0
    else:
        level = noise_level
        epsilon = torch.randn_like(z_prev) if noise is None else noise
        z = (1.0 - level) * z_prev + level * epsilon
    t = torch.full((batch,), level, device=device)
    r = torch.zeros(batch, device=device)
    u = model(z, t, r, vision, view_id)
    return z - level * u


@torch.no_grad()
def render_preview(codec, model, data, mean, std, device, source: str = "waymo", refiner=None):
    model.eval()
    codec.eval()
    vision = data["vision"][:1].to(device)
    target_drive = data["drive"][:1].to(device)
    target_reverse = data["reverse"][:1].to(device)
    surround = data["surround"][:1]
    shape = (1, target_drive.shape[1], model.latent_channels, model.latent_size, model.latent_size)
    drive_id = torch.zeros(1, dtype=torch.long, device=device)
    reverse_id = torch.ones(1, dtype=torch.long, device=device)
    z_drive = sample(model, shape, vision, drive_id)
    z_reverse = sample(model, shape, vision, reverse_id)
    z_switch = sample(model, shape, vision, reverse_id, z_prev=z_drive, noise_level=0.8)
    decode = lambda z: codec.decode(z * std + mean).clamp(0, 1)
    pred_drive, pred_reverse, pred_switch = decode(z_drive), decode(z_reverse), decode(z_switch)
    if refiner is not None:
        refiner.eval()
        pred_drive, pred_reverse, pred_switch = refiner(pred_drive), refiner(pred_reverse), refiner(pred_switch)
    pred_drive, pred_reverse, pred_switch = pred_drive.cpu(), pred_reverse.cpu(), pred_switch.cpu()
    rows = [
        ("环视", [surround[0, 0, i] for i in range(surround.shape[2])]),
        ("前进 目标", [target_drive[0, i].cpu() for i in range(4)]),
        ("前进 生成", [pred_drive[0, i] for i in range(4)]),
        ("倒车 目标", [target_reverse[0, i].cpu() for i in range(4)]),
        ("倒车 生成", [pred_reverse[0, i] for i in range(4)]),
        ("倒车切换", [pred_switch[0, i] for i in range(4)]),
    ]
    path = ROOT / "outputs" / "preview.png"
    save_preview(path, rows, source=source)
    print(f"预览 {path}", flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description="Sample SanBengZi Reality View")
    parser.add_argument("--ckpt", type=str, default=str(ROOT / "outputs" / "sbzrv.pt"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = Config(**ckpt["config"])
    codec = load_codec(ckpt, cfg, device)
    model = VideoMeanFlow(
        hidden=cfg.hidden,
        depth=cfg.depth,
        heads=cfg.heads,
        frames=cfg.frames,
        latent_channels=cfg.latent_channels,
        latent_size=cfg.latent_size,
        tubelet=cfg.tubelet,
        vision_dim=cfg.vision_dim,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    refiner = None
    if ckpt.get("refiner"):
        refiner = Refiner().eval()
        refiner.bank = ckpt["refiner"]["bank"]
        refiner.residual = ckpt["refiner"]["residual"]
        refiner.to(device)
    data = build_cache(cfg, device, rebuild=False)
    render_preview(codec, model, data, ckpt["mean"].to(device), ckpt["std"].to(device), device, cfg.source, refiner)


if __name__ == "__main__":
    main()

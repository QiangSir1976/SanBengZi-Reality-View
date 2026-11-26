"""Write a short mp4: surround cameras, one-step overhead, then a gear switch."""

from __future__ import annotations

import argparse

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

from .autoencoder import Refiner, load_codec
from .config import ROOT, Config
from .dataset import build_cache
from .infer import sample
from .model import VideoMeanFlow
from .preview import _font, camera_names


W, H = 1280, 720
FPS = 12
BG = (14, 15, 18)
AMBER = (232, 168, 74)
INK = (232, 230, 227)
QUIET = (154, 150, 144)
HOLD = 5


def _load(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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
    ).to(device).eval()
    model.load_state_dict(ckpt["model"])
    refiner = None
    if ckpt.get("refiner"):
        refiner = Refiner().eval()
        refiner.bank = ckpt["refiner"]["bank"]
        refiner.residual = ckpt["refiner"]["residual"]
        refiner.to(device)
    data = build_cache(cfg, device, rebuild=False)
    return codec, model, refiner, data, ckpt["mean"].to(device), ckpt["std"].to(device), ckpt["config"].get("source", "waymo")


@torch.no_grad()
def _generate(codec, model, refiner, data, mean, std, device):
    torch.manual_seed(0)
    vision = data["vision"]
    batch, frames = vision.shape[:2]
    shape = (batch, frames, model.latent_channels, model.latent_size, model.latent_size)
    noise_drive = torch.randn(shape)
    noise_reverse = torch.randn(shape)
    noise_switch = torch.randn(shape)
    drive, reverse, switch = [], [], []
    for start in range(0, batch, 2):
        seen = vision[start:start + 2].to(device)
        stop = start + seen.shape[0]
        drive_id = torch.zeros(seen.shape[0], dtype=torch.long, device=device)
        reverse_id = torch.ones(seen.shape[0], dtype=torch.long, device=device)
        piece = (seen.shape[0], frames, *shape[2:])
        z_drive = sample(model, piece, seen, drive_id, noise=noise_drive[start:stop].to(device))
        z_reverse = sample(model, piece, seen, reverse_id, noise=noise_reverse[start:stop].to(device))
        z_switch = sample(
            model, piece, seen, reverse_id,
            z_prev=z_drive, noise_level=0.8, noise=noise_switch[start:stop].to(device),
        )
        drive.append(z_drive)
        reverse.append(z_reverse)
        switch.append(z_switch)
    z_drive, z_reverse, z_switch = torch.cat(drive), torch.cat(reverse), torch.cat(switch)

    def decode(latent):
        image = codec.decode(latent * std + mean).clamp(0, 1)
        if refiner is not None:
            image = refiner(image)
        return image.cpu()

    return {
        "surround": data["surround"],
        "drive_target": data["drive"],
        "reverse_target": data["reverse"],
        "drive": decode(z_drive),
        "reverse": decode(z_reverse),
        "switch": decode(z_switch),
    }


def _paste(canvas, tensor, box, xy, resample):
    array = (tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    image = Image.fromarray(array, mode="RGB").resize((box, box), resample)
    canvas.paste(image, xy)


def _header(card, draw, rig, note):
    logo = Image.open(ROOT / "assets" / "logo.jpg").convert("RGB")
    logo.thumbnail((78, 62))
    plate = Image.new("RGB", (logo.width + 8, logo.height + 8), (255, 255, 255))
    plate.paste(logo, (4, 4))
    card.paste(plate, (28, 16))
    draw.text((124, 18), "SanBengZi Reality View", fill=INK, font=_font(28))
    draw.text((124, 54), note, fill=QUIET, font=_font(16))
    draw.text((W - 28, 28), rig, fill=AMBER, font=_font(26), anchor="ra")
    draw.line((28, 96, W - 28, 96), fill=(42, 40, 36), width=1)


def _title_card(source: str):
    line = {
        "av2": "Argoverse 2 实拍环视  ·  一次前向  ·  前进 / 倒车",
        "waymo": "Waymo 实拍环视  ·  一次前向  ·  前进 / 倒车",
    }.get(source, "实拍环视  ·  一次前向  ·  前进 / 倒车")
    card = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(card)
    logo = Image.open(ROOT / "assets" / "logo.jpg").convert("RGB")
    logo.thumbnail((360, 280))
    plate = Image.new("RGB", (logo.width + 24, logo.height + 24), (255, 255, 255))
    plate.paste(logo, (12, 12))
    card.paste(plate, ((W - plate.width) // 2, 120))
    draw.text((W // 2, 470), "SanBengZi Reality View", fill=INK, font=_font(42), anchor="ma")
    draw.text((W // 2, 530), line, fill=AMBER, font=_font(22), anchor="ma")
    return card


def _shot(pack, scene, frame, rig):
    card = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(card)
    if rig == "switch":
        note = "由前进潜变量加噪到 0.8，机位改为倒车"
        label = "倒车切换"
        generated = pack["switch"][scene, frame]
        target = pack["reverse_target"][scene, frame]
    elif rig == "reverse":
        note = "环视特征进去，一次前向得到车前上方的机位"
        label = "倒车"
        generated = pack["reverse"][scene, frame]
        target = pack["reverse_target"][scene, frame]
    else:
        note = "环视特征进去，一次前向得到车后上方的机位"
        label = "前进"
        generated = pack["drive"][scene, frame]
        target = pack["drive_target"][scene, frame]
    _header(card, draw, label, note)

    cameras = pack["surround"].shape[2]
    names = camera_names(cameras)
    gap = 8 if cameras > 4 else 14
    cell = min(168, (W - 56 - (cameras - 1) * gap) // cameras)
    row_w = len(names) * cell + (len(names) - 1) * gap
    x0 = (W - row_w) // 2
    y0 = 108
    for i, name in enumerate(names):
        x = x0 + i * (cell + gap)
        draw.text((x, y0), name, fill=QUIET, font=_font(16))
        _paste(card, pack["surround"][scene, frame, i], cell, (x, y0 + 22), Image.Resampling.NEAREST)

    big = 260 if cameras > 4 else 292
    left = (W - (big * 2 + 36)) // 2
    top = y0 + cell + 36
    draw.text((left, top), "生成", fill=INK, font=_font(18))
    draw.text((left + big + 36, top), "目标", fill=INK, font=_font(18))
    _paste(card, generated, big, (left, top + 28), Image.Resampling.BICUBIC)
    _paste(card, target, big, (left + big + 36, top + 28), Image.Resampling.NEAREST)

    draw.text((28, H - 36), f"场景 {scene + 1}/{pack['drive'].shape[0]}", fill=QUIET, font=_font(16))
    for i in range(4):
        color = AMBER if i == frame else (58, 56, 52)
        draw.rectangle((W - 132 + i * 26, H - 32, W - 114 + i * 26, H - 20), fill=color)
    return card


def _frames(pack, source: str):
    frames = [_title_card(source)] * (FPS + 6)
    for scene in range(pack["drive"].shape[0]):
        for rig in ("drive", "reverse"):
            for frame in range(4):
                shot = _shot(pack, scene, frame, rig)
                frames.extend([shot] * HOLD)
    for frame in range(4):
        shot = _shot(pack, 0, frame, "switch")
        frames.extend([shot] * HOLD)
    return frames


def main():
    parser = argparse.ArgumentParser(description="Write the SanBengZi Reality View demo")
    parser.add_argument("--ckpt", type=str, default=str(ROOT / "outputs" / "sbzrv.pt"))
    parser.add_argument("--out", type=str, default=str(ROOT / "outputs" / "demo.mp4"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec, model, refiner, data, mean, std, source = _load(args.ckpt, device)
    pack = _generate(codec, model, refiner, data, mean, std, device)
    drive_error = float((pack["drive"] - pack["drive_target"]).abs().mean())
    reverse_error = float((pack["reverse"] - pack["reverse_target"]).abs().mean())
    print(f"演示 前进 {drive_error:.4f}  倒车 {reverse_error:.4f}", flush=True)
    frames = _frames(pack, source)
    path = args.out
    writer = imageio.get_writer(path, fps=FPS, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=1)
    for frame in frames:
        writer.append_data(np.asarray(frame))
    writer.close()
    print(f"演示 {path}  {len(frames)} 帧  {len(frames) / FPS:.1f} 秒", flush=True)


if __name__ == "__main__":
    main()

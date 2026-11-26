"""Cached clips: solid render, photograph pass, frozen vision tokens."""

from __future__ import annotations

import os
import subprocess

import torch

from .config import ROOT, Config
from .encoder import DinoV2Encoder
from .render import render_clip


def cache_path(cfg: Config):
    if cfg.source == "av2":
        return ROOT / "outputs" / f"cache_av2_s{cfg.scenes}_f{cfg.frames}_{cfg.image_size}_street.pt"
    if cfg.source == "waymo":
        return ROOT / "outputs" / f"cache_waymo_s{cfg.scenes}_f{cfg.frames}_{cfg.image_size}.pt"
    return ROOT / "outputs" / f"cache_photo_s{cfg.scenes}_f{cfg.frames}_{cfg.image_size}.pt"


def _photograph(geometric: dict) -> dict:
    python = ROOT / ".venv" / "Scripts" / "python.exe"
    if not python.is_file():
        raise FileNotFoundError("缺少 .venv，无法把几何画面收成实景照片。")
    folder = ROOT / "outputs"
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / "_geom.pt"
    dest = folder / "_photo.pt"
    torch.save(geometric, source)
    env = os.environ.copy()
    env["USE_TF"] = "0"
    env["USE_FLAX"] = "0"
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    subprocess.check_call(
        [str(python), "-m", "sbzrv.photoreal", "--src", str(source), "--dst", str(dest)],
        cwd=ROOT,
        env=env,
    )
    photo = torch.load(dest, map_location="cpu", weights_only=False)
    source.unlink(missing_ok=True)
    dest.unlink(missing_ok=True)
    return photo


@torch.no_grad()
def build_cache(cfg: Config, device: torch.device, rebuild: bool = False):
    path = cache_path(cfg)
    if path.is_file() and not rebuild:
        return torch.load(path, map_location="cpu", weights_only=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    if cfg.source == "waymo":
        from .waymo import load_clips

        print("读取 Waymo 实拍", flush=True)
        data = load_clips(cfg)
    elif cfg.source == "av2":
        from .av2 import load_clips

        print("读取 Argoverse 2 实拍", flush=True)
        data = load_clips(cfg)
    else:
        clips = []
        for index in range(cfg.scenes):
            print(f"渲染场景 {index + 1}/{cfg.scenes}", flush=True)
            clips.append(render_clip(1000 + index, cfg.image_size, cfg.frames, device="cpu"))
        geometric = {
            "surround": torch.stack([clip["surround"] for clip in clips], 0),
            "drive": torch.stack([clip["drive"] for clip in clips], 0),
            "reverse": torch.stack([clip["reverse"] for clip in clips], 0),
        }
        print("收成实景照片", flush=True)
        data = _photograph(geometric)
    print("编码环视特征", flush=True)
    encoder = DinoV2Encoder().to(device)
    vision = []
    for index in range(data["surround"].shape[0]):
        tokens = encoder.encode_surround(data["surround"][index:index + 1].to(device), batch=8)
        vision.append(tokens.cpu())
    data["vision"] = torch.cat(vision, 0)
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()
    torch.save(data, path)
    return data


def batch_from(data, indices, view_id):
    both = torch.stack((data["drive"], data["reverse"]), dim=1)
    overhead = both[indices, view_id]
    vision = data["vision"][indices]
    return overhead, vision

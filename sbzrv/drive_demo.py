"""Forward-only drive. The model samples the chase view, never the reverse view."""

from __future__ import annotations

import argparse

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from .arrows import paint_route_tensor
from .config import ROOT
from .demo import _load
from .infer import sample


FPS = 12
HOLD = 4
SIZE = 960
# How far the car advances along the lane during one four-frame clip.
CLIP_TRAVEL = 12.0


@torch.no_grad()
def _drive(codec, model, refiner, data, mean, std, device):
    """One forward sample per scene. The reverse view is not sampled."""
    torch.manual_seed(0)
    vision = data["vision"]
    scenes, frames = vision.shape[:2]
    shape = (scenes, frames, model.latent_channels, model.latent_size, model.latent_size)
    noise = torch.randn(shape)
    pieces = []
    for start in range(0, scenes, 2):
        seen = vision[start:start + 2].to(device)
        view = torch.zeros(seen.shape[0], dtype=torch.long, device=device)
        piece = (seen.shape[0], frames, *shape[2:])
        latent = sample(model, piece, seen, view, noise=noise[start:start + seen.shape[0]].to(device))
        image = codec.decode(latent * std + mean).clamp(0, 1)
        if refiner is not None:
            image = refiner(image)
        pieces.append(image.cpu())
        print(f"  前行 {start + seen.shape[0]}/{scenes}", flush=True)
    return torch.cat(pieces, 0)


def main():
    parser = argparse.ArgumentParser(description="Write a forward-only route video from the model")
    parser.add_argument("--ckpt", type=str, default=str(ROOT / "outputs" / "sbzrv.pt"))
    parser.add_argument("--out", type=str, default=str(ROOT / "outputs" / "drive_route.mp4"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec, model, refiner, data, mean, std, _source = _load(args.ckpt, device)
    drive = _drive(codec, model, refiner, data, mean, std, device)

    scenes, frames = drive.shape[:2]
    writer = imageio.get_writer(
        args.out, fps=FPS, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=1,
    )
    for scene in range(scenes):
        for frame in range(frames):
            travel = CLIP_TRAVEL * frame / max(frames - 1, 1)
            painted = paint_route_tensor(drive[scene, frame], travel)
            image = Image.fromarray((painted.permute(1, 2, 0).numpy() * 255).astype("uint8"))
            panel = np.asarray(image.resize((SIZE, SIZE), Image.Resampling.LANCZOS))
            for _hold in range(HOLD):
                writer.append_data(panel)
        print(f"  写入场景 {scene + 1}/{scenes}", flush=True)
    writer.close()
    count = scenes * frames * HOLD
    print(f"前行视频 {args.out}  {count} 帧  {count / FPS:.1f} 秒", flush=True)


if __name__ == "__main__":
    main()

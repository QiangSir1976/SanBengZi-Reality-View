"""Fifteen-second demo: seven inputs, a forward view, and a rear view.

The car keeps driving forward. The left picture looks ahead and the right
picture looks back. Eight scenes fill exactly fifteen seconds.
"""

from __future__ import annotations

import argparse

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw

from .config import ROOT
from .demo import _load
from .encoder import DinoV2Encoder
from .live import LiveDrive
from .preview import _font


W, H = 1920, 1080
FPS = 24
SECONDS = 15
BG = (14, 15, 18)
AMBER = (232, 168, 74)
INK = (232, 230, 227)
QUIET = (154, 150, 144)
LINE = (42, 40, 36)
EDGE = (58, 56, 52)

# Left to right, the way the ring sits around a car that is driving forward.
CAMERAS = (
    (5, "左后"),
    (3, "左侧"),
    (1, "左前"),
    (0, "前"),
    (2, "右前"),
    (4, "右侧"),
    (6, "右后"),
)
CELL = 172
GAP = 16
VIEW = 620
VIEW_GAP = 36


def _rgb(tensor: torch.Tensor, size: int) -> Image.Image:
    array = (tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(array, mode="RGB").resize((size, size), Image.Resampling.LANCZOS)


def _cursor(step: int, frames: int):
    """Hold each forward frame, then a short dissolve into the next one."""
    hold, blend = 9, 3
    if step >= (frames - 1) * (hold + blend):
        return frames - 1, frames - 1, 0.0
    local = step % (hold + blend)
    index = step // (hold + blend)
    if local < hold:
        return index, index, 0.0
    return index, index + 1, (local - hold + 1) / blend


def _blend(left: Image.Image, right: Image.Image, weight: float) -> Image.Image:
    if weight <= 0.001:
        return left
    if weight >= 0.999:
        return right
    return Image.blend(left, right, weight)


def _chrome() -> Image.Image:
    card = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(card)
    draw.text((40, 22), "SanBengZi Reality View", fill=INK, font=_font(28))
    draw.text((40, 58), "七路输入  ·  前向与后向  ·  一直前行", fill=QUIET, font=_font(16))
    draw.text((W - 40, 32), "前进", fill=AMBER, font=_font(28), anchor="ra")
    draw.line((40, 96, W - 40, 96), fill=LINE, width=1)
    row = len(CAMERAS) * CELL + (len(CAMERAS) - 1) * GAP
    x0 = (W - row) // 2
    for index, (_, name) in enumerate(CAMERAS):
        x = x0 + index * (CELL + GAP)
        color = AMBER if name == "前" else QUIET
        draw.text((x, 108), name, fill=color, font=_font(16))
    left, right = _view_origins()
    draw.text((left[0], left[1] - 32), "前向", fill=AMBER, font=_font(18))
    draw.text((right[0], right[1] - 32), "后向", fill=INK, font=_font(18))
    draw.rectangle((40, H - 18, W - 40, H - 12), fill=LINE)
    return card


def _place(card: Image.Image, image: Image.Image, box, xy, accent: bool):
    card.paste(image, xy)
    draw = ImageDraw.Draw(card)
    x, y = xy
    draw.rectangle((x, y, x + box - 1, y + box - 1), outline=AMBER if accent else EDGE, width=2)


def _view_origins():
    total = VIEW * 2 + VIEW_GAP
    left = (W - total) // 2
    return (left, 352), (left + VIEW + VIEW_GAP, 352)


def _shot(base, cameras, front, rear, scene, scenes, clock: float) -> Image.Image:
    card = base.copy()
    row = len(CAMERAS) * CELL + (len(CAMERAS) - 1) * GAP
    x0 = (W - row) // 2
    y = 132
    for index, (camera, name) in enumerate(CAMERAS):
        x = x0 + index * (CELL + GAP)
        _place(card, cameras[camera], CELL, (x, y), name == "前")
    left, right = _view_origins()
    _place(card, front, VIEW, left, True)
    _place(card, rear, VIEW, right, False)
    draw = ImageDraw.Draw(card)
    draw.text((40, H - 52), f"片段 {scene + 1}/{scenes}", fill=QUIET, font=_font(16))
    draw.text((W - 40, H - 52), f"{clock:04.1f} / {SECONDS:.0f}.0 s", fill=QUIET, font=_font(16), anchor="ra")
    span = W - 80
    done = int(span * min(clock / SECONDS, 1.0))
    if done > 0:
        draw.rectangle((40, H - 18, 40 + done, H - 12), fill=AMBER)
    return card


def _tiles(live: LiveDrive, surround: torch.Tensor):
    scenes, frames, cameras = surround.shape[:3]
    if cameras != 7:
        raise RuntimeError(f"需要 7 路相机，缓存里是 {cameras} 路")
    live.step(surround[0, 0], 0.0, slot=0)
    camera_tiles = []
    front_tiles = []
    rear_tiles = []
    for scene in range(scenes):
        views = []
        fronts = []
        rears = []
        for frame in range(frames):
            for camera in range(cameras):
                views.append(_rgb(surround[scene, frame, camera], CELL))
            torch.manual_seed(10_000 + scene)
            front = live.step(surround[scene, frame], travel=float(frame), slot=frame, view=0, paint=False)
            torch.manual_seed(10_000 + scene)
            rear = live.step(surround[scene, frame], travel=float(frame), slot=frame, view=1, paint=False)
            fronts.append(front.resize((VIEW, VIEW), Image.Resampling.LANCZOS))
            rears.append(rear.resize((VIEW, VIEW), Image.Resampling.LANCZOS))
            print(f"  生成 片段 {scene + 1}/{scenes}  帧 {frame + 1}/{frames}", flush=True)
        camera_tiles.append(views)
        front_tiles.append(fronts)
        rear_tiles.append(rears)
    return camera_tiles, front_tiles, rear_tiles


def _grouped(flat, frames: int, cameras: int):
    grouped = []
    for frame in range(frames):
        start = frame * cameras
        grouped.append(flat[start:start + cameras])
    return grouped


def main():
    parser = argparse.ArgumentParser(description="15s demo with seven inputs, a forward view, and a rear view")
    parser.add_argument("--ckpt", type=str, default=str(ROOT / "outputs" / "sbzrv.pt"))
    parser.add_argument("--out", type=str, default=str(ROOT / "outputs" / "forward_demo.mp4"))
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec, model, _refiner, data, mean, std, _source = _load(args.ckpt, device)
    encoder = DinoV2Encoder().to(device).eval()
    live = LiveDrive(codec, model, encoder, mean, std, device)
    surround = data["surround"]
    scenes, frames = surround.shape[:2]
    camera_tiles, front_tiles, rear_tiles = _tiles(live, surround)
    per_scene = FPS * SECONDS // scenes
    if per_scene * scenes != FPS * SECONDS:
        raise RuntimeError("时长不能被片段数整除")
    base = _chrome()
    writer = imageio.get_writer(
        args.out, fps=FPS, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=1,
    )
    count = 0
    for scene in range(scenes):
        cameras = _grouped(camera_tiles[scene], frames, 7)
        fronts = front_tiles[scene]
        rears = rear_tiles[scene]
        for step in range(per_scene):
            left, right, weight = _cursor(step, frames)
            shown = [_blend(cameras[left][c], cameras[right][c], weight) for c in range(7)]
            front = _blend(fronts[left], fronts[right], weight)
            rear = _blend(rears[left], rears[right], weight)
            clock = count / FPS
            writer.append_data(np.asarray(_shot(base, shown, front, rear, scene, scenes, clock)))
            count += 1
        print(f"  写入片段 {scene + 1}/{scenes}", flush=True)
    writer.close()
    print(f"前行演示 {args.out}  {count} 帧  {count / FPS:.1f} 秒", flush=True)


if __name__ == "__main__":
    main()

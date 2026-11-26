"""Drive the chase view from seven live cameras.

Each tick encodes the seven new frames, generates one 256 image, and shows it.
A straight route is left unmarked. A bend still gets the floating chevrons.
"""

from __future__ import annotations

import argparse
import time

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from .arrows import paint_route_tensor
from .config import ROOT
from .demo import _load
from .encoder import DinoV2Encoder
from .infer import sample


def _cameras(frame: torch.Tensor, device: torch.device) -> torch.Tensor:
    """frame is (cameras, 3, H, W). Returns (1, 1, cameras, 3, H, W)."""
    return frame.to(device, non_blocking=True)[None, None]


class LiveDrive:
    """One new chase image per seven-camera tick."""

    def __init__(self, codec, model, encoder, mean, std, device):
        self.codec = codec
        self.model = model
        self.encoder = encoder
        self.mean = mean
        self.std = std
        self.device = device
        self.view = torch.zeros(1, dtype=torch.long, device=device)
        self.times = {"encode": [], "generate": [], "decode": []}

    def reset(self):
        return None

    @torch.inference_mode()
    def step(self, cameras: torch.Tensor, travel: float, slot: int = 0, view: int = 0, paint: bool = True) -> Image.Image:
        sync = torch.cuda.synchronize
        pack = _cameras(cameras, self.device)
        sync()
        started = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vision = self.encoder.encode_surround(pack, batch=7)
        sync()
        encoded = time.perf_counter()
        shape = (1, 1, self.model.latent_channels, self.model.latent_size, self.model.latent_size)
        self.model.time_slot = int(slot) % self.model.pos.shape[0]
        view_id = torch.full((1,), int(view), dtype=torch.long, device=self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            latent = sample(self.model, shape, vision, view_id)
        sync()
        generated = time.perf_counter()
        image = self.codec.decode(latent.float() * self.std + self.mean).clamp(0, 1)[0, 0]
        painted = paint_route_tensor(image, travel) if paint else image
        sync()
        self.times["encode"].append((encoded - started) * 1000)
        self.times["generate"].append((generated - encoded) * 1000)
        self.times["decode"].append((time.perf_counter() - generated) * 1000)
        array = (painted.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        return Image.fromarray(array, mode="RGB")


def _show(frames, fps: float):
    import tkinter as tk
    from PIL import ImageTk

    root = tk.Tk()
    root.title("七路环视 → 前进实景")
    view = frames[0].resize((768, 768), Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(view)
    label = tk.Label(root, image=photo, bg="#0e0f12")
    label.pack()
    state = {"index": 0, "photo": photo}

    def tick():
        state["index"] = (state["index"] + 1) % len(frames)
        view = frames[state["index"]].resize((768, 768), Image.Resampling.LANCZOS)
        state["photo"] = ImageTk.PhotoImage(view)
        label.configure(image=state["photo"])
        root.after(max(1, int(1000 / fps)), tick)

    root.after(max(1, int(1000 / fps)), tick)
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Live seven-camera chase view")
    parser.add_argument("--ckpt", type=str, default=str(ROOT / "outputs" / "sbzrv.pt"))
    parser.add_argument("--out", type=str, default=str(ROOT / "outputs" / "live_drive.mp4"))
    parser.add_argument("--window", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec, model, _refiner, data, mean, std, _source = _load(args.ckpt, device)
    encoder = DinoV2Encoder().to(device).eval()
    live = LiveDrive(codec, model, encoder, mean, std, device)
    surround = data["surround"]
    scenes, frames, cameras = surround.shape[:3]
    if cameras != 7:
        raise RuntimeError(f"需要 7 路相机，缓存里是 {cameras} 路")

    shown = []
    # One warmup tick, not counted.
    live.step(surround[0, 0], 0.0)
    live.reset()
    live.times = {"encode": [], "generate": [], "decode": []}
    for scene in range(scenes):
        live.reset()
        for frame in range(frames):
            image = live.step(surround[scene, frame], travel=8.0 + frame * 4.0, slot=frame)
            shown.append(image.resize((768, 768), Image.Resampling.LANCZOS))
        encode = float(np.mean(live.times["encode"][-frames:]))
        generate = float(np.mean(live.times["generate"][-frames:]))
        decode = float(np.mean(live.times["decode"][-frames:]))
        total = encode + generate + decode
        print(
            f"  场景 {scene + 1}/{scenes}  编码 {encode:.0f} ms  生成 {generate:.0f} ms  "
            f"解码 {decode:.0f} ms  合计 {total:.0f} ms  {1000 / total:.1f} fps",
            flush=True,
        )

    spent = [sum(item) for item in zip(live.times["encode"], live.times["generate"], live.times["decode"])]
    fps = 1000 / float(np.mean(spent))
    writer = imageio.get_writer(
        args.out, fps=max(1, round(fps)), codec="libx264", quality=8,
        pixelformat="yuv420p", macro_block_size=1,
    )
    for image in shown:
        writer.append_data(np.asarray(image))
    writer.close()
    print(f"实时前进 {args.out}  {len(shown)} 帧  {fps:.1f} fps", flush=True)
    if args.window:
        _show(shown, fps)


if __name__ == "__main__":
    main()

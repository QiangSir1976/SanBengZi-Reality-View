"""Relative depth for Waymo frames. Run this with the project .venv."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    args = parser.parse_args()
    pack = torch.load(args.src, map_location="cpu", weights_only=False)
    images = pack["images"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf")
    model = AutoModelForDepthEstimation.from_pretrained("depth-anything/Depth-Anything-V2-Small-hf").to(device).eval()
    depth = []
    with torch.inference_mode():
        for start in range(0, images.shape[0], 4):
            batch = images[start:start + 4]
            pil = [Image.fromarray((frame.permute(1, 2, 0).numpy() * 255).astype("uint8")) for frame in batch]
            inputs = processor(images=pil, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in inputs.items()}
            predicted = model(**inputs).predicted_depth
            predicted = F.interpolate(
                predicted[:, None],
                size=batch.shape[-2:],
                mode="bicubic",
                align_corners=False,
            )[:, 0]
            depth.append(predicted.float().cpu())
            print(f"深度 {min(start + 4, images.shape[0])}/{images.shape[0]}", flush=True)
    torch.save({"depth": torch.cat(depth, 0)}, args.dst)
    print(f"已写入 {args.dst}", flush=True)


if __name__ == "__main__":
    main()

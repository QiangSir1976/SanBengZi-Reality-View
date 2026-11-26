"""Turn solid shaded frames into daylight photographs with SDXL-Turbo.

Run this with the project virtual environment. The system Python cannot load
the diffusion stack.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import numpy as np
import torch
from PIL import Image

PROMPTS = {
    "front": "photorealistic dashcam photograph, front view of a city street, asphalt road with yellow lane markings, apartment buildings, daylight, natural colors, sharp, no text",
    "left": "photorealistic side camera photograph from a vehicle, city buildings and sidewalk, daylight, natural colors, sharp, no text",
    "right": "photorealistic side camera photograph from a vehicle, city buildings and sidewalk, daylight, natural colors, sharp, no text",
    "rear": "photorealistic rear camera photograph, city road behind the vehicle, daylight, natural colors, sharp, no text",
    "drive": "photorealistic drone photograph, high angle behind a small dark three-wheeled utility vehicle driving on a city road, yellow lane markings, apartment buildings, daylight, sharp, no text",
    "reverse": "photorealistic drone photograph, high angle in front of a small dark three-wheeled utility vehicle on a city road, looking back along the lane, daylight, sharp, no text",
}


def _pipe(device):
    from diffusers import StableDiffusionXLImg2ImgPipeline

    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe.set_progress_bar_config(disable=True)
    pipe.to(device)
    pipe.vae.enable_slicing()
    return pipe


def _pil(frame: torch.Tensor, side: int) -> Image.Image:
    array = (frame.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    image = Image.fromarray(array, mode="RGB")
    if image.size != (side, side):
        image = image.resize((side, side), Image.Resampling.BICUBIC)
    return image


def _tensor(image: Image.Image, size: int) -> torch.Tensor:
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    array = torch.from_numpy(np.array(image)).float() / 255.0
    return array.permute(2, 0, 1).contiguous()


def refine(pipe, frame, prompt, seed, size, device):
    generator = torch.Generator(device=device).manual_seed(seed)
    image = pipe(
        prompt=prompt,
        image=_pil(frame, 512),
        strength=0.55,
        guidance_scale=0.0,
        num_inference_steps=4,
        generator=generator,
    ).images[0]
    return _tensor(image, size)


def refine_pack(pipe, pack, device):
    size = pack["drive"].shape[-1]
    surround = torch.empty_like(pack["surround"])
    drive = torch.empty_like(pack["drive"])
    reverse = torch.empty_like(pack["reverse"])
    scenes, frames, cameras = pack["surround"].shape[:3]
    names = ("front", "left", "right", "rear")
    done = 0
    total = scenes * frames * (cameras + 2)
    for scene in range(scenes):
        for frame_i in range(frames):
            for camera, name in enumerate(names):
                surround[scene, frame_i, camera] = refine(
                    pipe,
                    pack["surround"][scene, frame_i, camera],
                    PROMPTS[name],
                    1000 + scene * 17 + camera,
                    size,
                    device,
                )
                done += 1
            drive[scene, frame_i] = refine(
                pipe, pack["drive"][scene, frame_i], PROMPTS["drive"], 5000 + scene, size, device
            )
            reverse[scene, frame_i] = refine(
                pipe, pack["reverse"][scene, frame_i], PROMPTS["reverse"], 8000 + scene, size, device
            )
            done += 2
            if done % 12 == 0:
                torch.cuda.empty_cache()
            print(f"实景 {done}/{total}", flush=True)
    return {"surround": surround, "drive": drive, "reverse": reverse}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pack = torch.load(args.src, map_location="cpu", weights_only=False)
    pipe = _pipe(device)
    photo = refine_pack(pipe, pack, device)
    torch.save(photo, args.dst)
    print(f"已写入 {args.dst}", flush=True)


if __name__ == "__main__":
    main()

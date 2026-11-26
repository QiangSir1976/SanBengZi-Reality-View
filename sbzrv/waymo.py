"""Waymo camera clips, reprojected into the two overhead rigs.

The shard is the Waymo end-to-end camera set: front, front-left and front-right.
There is no overhead camera, so the drive and reverse targets sample the real
front photo on the ground plane.
"""

from __future__ import annotations

import math
import os
import subprocess
import tarfile

import numpy as np
import torch
from PIL import Image

from .config import ROOT, Config
from .render import _basis


CAMERAS = ("front", "front_left", "front_right")
# Daylight segments from this shard, brightest first. The shard also contains night clips.
DAYLIGHT = (
    "0089ac6f4359b1b2f32a3b9b75afde95",
    "0327be0a97fed39f8bc13ce92539e36f",
    "00abf5543eeb9cf6c8bf7155e6f2028d",
    "011a7d7544adc71acd1608d6569e4112",
    "018c156bcbb819561be188ddc1bb6dff",
    "03195aefc01e511b9756a496c860efb6",
    "01263ac65b36925004b7751e6f2fe9dd",
    "01678cb89b1fe044b6011b6ea19b591a",
)
# The visible points sit in front of the car. These two rigs look along that street
# from behind and from ahead, instead of the synthetic street's longer baselines.
OVER_DRIVE = ((5.0, 0.0, 5.5), (16.0, 0.0, 0.3), 75.0)
OVER_REVERSE = ((16.0, 0.0, 5.5), (4.0, 0.0, 0.3), 75.0)
YAW = {"front": 0.0, "front_left": 52.0, "front_right": -52.0}
ORIGIN = {
    "front": (1.55, 0.0, 2.05),
    "front_left": (1.45, 0.50, 2.05),
    "front_right": (1.45, -0.50, 2.05),
}


def _tar_path() -> "os.PathLike":
    return ROOT / "data" / "waymo" / "test-000.tar"


def _scene_ids(count: int) -> list[str]:
    return list(DAYLIGHT[:count])


def _native_intrinsics(width: int, height: int):
    fx = 2056.0 * width / 1920.0
    fy = 2056.0 * height / 1280.0
    return fx, fy, width * 0.5, height * 0.5


def _square(image: Image.Image, size: int):
    width, height = image.size
    fx, fy, cx, cy = _native_intrinsics(width, height)
    side = min(width, height)
    x0 = (width - side) // 2
    y0 = (height - side) // 2
    cropped = image.crop((x0, y0, x0 + side, y0 + side)).resize((size, size), Image.Resampling.LANCZOS)
    scale = size / side
    return cropped, (fx * scale, fy * scale, (cx - x0) * scale, (cy - y0) * scale)


def _split_name(name: str):
    stem = name.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    for camera in CAMERAS:
        suffix = "_" + camera
        if stem.endswith(suffix):
            return stem[: -len(suffix)], camera
    return None, None


def _extract(cfg: Config) -> list[dict]:
    folder = ROOT / "data" / "waymo" / "frames"
    marker = folder / "ready.txt"
    scenes = _scene_ids(cfg.scenes)
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == ",".join(scenes):
        return _read_frames(folder, scenes, cfg)
    archive = _tar_path()
    if not archive.is_file() or archive.stat().st_size < 3_000_000_000:
        raise FileNotFoundError("Waymo 图像包还没下完：data/waymo/test-000.tar")
    wanted = set(scenes)
    grouped: dict[tuple[str, str], dict[str, str]] = {}
    with tarfile.open(archive, "r:") as tar:
        members = [item for item in tar.getmembers() if item.isfile()]
        chosen = []
        for item in members:
            scene = next((name for name in wanted if name in item.name), None)
            frame, camera = _split_name(item.name)
            if scene is None or frame is None:
                continue
            grouped.setdefault((scene, frame), {})[camera] = item.name
        for key, cameras in grouped.items():
            if all(name in cameras for name in CAMERAS):
                chosen.extend(cameras[name] for name in CAMERAS)
        picked = {item.name: item for item in members if item.name in set(chosen)}
        folder.mkdir(parents=True, exist_ok=True)
        for name, item in picked.items():
            scene = next(scene_name for scene_name in wanted if scene_name in name)
            frame, camera = _split_name(name)
            dest = folder / scene / f"{frame}_{camera}.jpg"
            dest.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(item)
            dest.write_bytes(source.read())
    marker.write_text(",".join(scenes), encoding="utf-8")
    return _read_frames(folder, scenes, cfg)


def _read_frames(folder, scenes: list[str], cfg: Config) -> list[dict]:
    clips = []
    for scene in scenes:
        scene_dir = folder / scene
        frames: dict[str, dict[str, str]] = {}
        for path in scene_dir.glob("*.jpg"):
            frame, camera = _split_name(path.name)
            if frame is None:
                continue
            frames.setdefault(frame, {})[camera] = str(path)
        complete = [frame for frame, cameras in frames.items() if all(name in cameras for name in CAMERAS)]
        complete.sort()
        if len(complete) < cfg.frames:
            raise RuntimeError(f"{scene} 只有 {len(complete)} 帧完整环视")
        start = len(complete) // 5
        stride = max(1, (len(complete) - start) // (cfg.frames * 3))
        chosen = [complete[start + i * stride] for i in range(cfg.frames)]
        views = []
        intrinsics = []
        for frame in chosen:
            images = []
            ks = []
            for camera in CAMERAS:
                image, k = _square(Image.open(frames[frame][camera]).convert("RGB"), cfg.image_size)
                images.append(torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0)
                ks.append(k)
            views.append(torch.stack(images, 0))
            intrinsics.append(ks)
        clips.append({"surround": torch.stack(views, 0), "intrinsics": intrinsics, "scene": scene})
        print(f"场景 {scene[:8]}  {len(complete)} 帧里取 {chosen}", flush=True)
    return clips


def _pose(camera: str):
    yaw = math.radians(YAW[camera])
    base = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float32)
    cosine, sine = math.cos(yaw), math.sin(yaw)
    turn = np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]], dtype=np.float32)
    return turn @ base, np.array(ORIGIN[camera], dtype=np.float32)


def _as_distance(depth: np.ndarray) -> np.ndarray:
    height = depth.shape[0]
    bottom = float(np.median(depth[int(height * 0.85):]))
    middle = float(np.median(depth[int(height * 0.45):int(height * 0.62)]))
    if bottom > middle:
        return depth.max() - depth + 1e-3
    return depth


def _metric_cloud(depth: np.ndarray, color: np.ndarray, k, rotation, origin):
    depth = _as_distance(depth)
    fx, fy, cx, cy = k
    height, width = depth.shape
    vs, us = np.mgrid[0:height, 0:width]
    ray = np.stack(((us - cx) / fx, (vs - cy) / fy, np.ones_like(depth)), -1).astype(np.float32)
    ray /= np.linalg.norm(ray, axis=-1, keepdims=True).clip(min=1e-6)
    direction = ray @ rotation.T
    scale = _ground_scale(depth, direction, origin)
    if scale < 0:
        depth = depth.max() - depth + 1e-3
        scale = _ground_scale(depth, direction, origin)
    distance = np.clip(scale, 1e-3, 80.0) * depth
    points = origin + distance[..., None] * direction
    flat = points.reshape(-1, 3)
    colors = color.reshape(-1, 3)
    span = np.linalg.norm(flat - origin, axis=1)
    keep = (span > 1.2) & (span < 40.0) & (flat[:, 2] > -0.4) & (flat[:, 2] < 6.0)
    return flat[keep], colors[keep]


def _ground_scale(depth, direction, origin):
    height, width = depth.shape
    vs = np.arange(height)[:, None]
    us = np.arange(width)[None, :]
    road = (vs > height * 0.62) & (np.abs(us - width * 0.5) < width * 0.22) & (direction[..., 2] < -0.05)
    denom = depth * direction[..., 2]
    usable = road & (np.abs(denom) > 1e-4)
    if int(usable.sum()) < 40:
        return 15.0
    scale = -float(origin[2]) / denom[usable]
    scale = scale[np.isfinite(scale)]
    if scale.size == 0:
        return 15.0
    return float(np.median(scale))


def _paint_ground(image, k, rotation, origin, eye, target, fov, size):
    """Sample the real front photo on the ground plane from an overhead rig."""
    fx, fy, cx, cy = k
    rot_v, eye_v = _basis(eye, target)
    focal = (size * 0.5) / math.tan(math.radians(fov) * 0.5)
    axis = np.arange(size, dtype=np.float32)
    uu, vv = np.meshgrid(axis, axis, indexing="xy")
    dirs = np.stack(((uu - size * 0.5) / focal, (vv - size * 0.5) / focal, np.ones_like(uu)), -1)
    dirs = dirs @ rot_v
    denom = dirs[..., 2]
    travel = np.where(np.abs(denom) > 1e-4, -float(eye_v[2]) / denom, -1.0)
    hit = eye_v + travel[..., None] * dirs
    cam = (hit - origin) @ rotation
    depth = cam[..., 2]
    u = fx * cam[..., 0] / np.clip(depth, 1e-3, None) + cx
    v = fy * cam[..., 1] / np.clip(depth, 1e-3, None) + cy
    valid = (travel > 0.3) & (depth > 0.4) & (u >= 0) & (u < image.shape[1] - 1) & (v >= 0) & (v < image.shape[0] - 1)
    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    u0 = np.clip(u0, 0, image.shape[1] - 2)
    v0 = np.clip(v0, 0, image.shape[0] - 2)
    du = (u - u0)[..., None]
    dv = (v - v0)[..., None]
    sample = (
        image[v0, u0] * (1 - du) * (1 - dv)
        + image[v0, u0 + 1] * du * (1 - dv)
        + image[v0 + 1, u0] * (1 - du) * dv
        + image[v0 + 1, u0 + 1] * du * dv
    )
    sky = np.array((0.72, 0.76, 0.80), dtype=np.float32)
    return np.where(valid[..., None], sample, sky).astype(np.float32)


def _paint(points, colors, eye, target, fov, size):
    rotation, eye_v = _basis(eye, target)
    focal = (size * 0.5) / math.tan(math.radians(fov) * 0.5)
    cam = (points - eye_v) @ rotation.T
    depth = cam[:, 2]
    keep = depth > 0.35
    u = np.rint(focal * cam[:, 0] / np.clip(depth, 1e-3, None) + size * 0.5).astype(np.int32)
    v = np.rint(focal * cam[:, 1] / np.clip(depth, 1e-3, None) + size * 0.5).astype(np.int32)
    keep &= (u >= 2) & (u < size - 2) & (v >= 2) & (v < size - 2)
    order = np.argsort(-depth[keep])
    u, v = u[keep][order], v[keep][order]
    pigment = colors[keep][order]
    image = np.full((size, size, 3), (0.42, 0.48, 0.54), dtype=np.float32)
    filled = np.zeros((size, size), dtype=bool)
    image[v, u] = pigment
    filled[v, u] = True
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            image[v + dy, u + dx] = pigment
            filled[v + dy, u + dx] = True
    for _ in range(8):
        total = np.zeros_like(image)
        count = np.zeros((size, size), dtype=np.float32)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            moved = np.roll(np.roll(filled, dy, 0), dx, 1)
            paint = np.roll(np.roll(image, dy, 0), dx, 1)
            if dy > 0:
                moved[:dy] = False
            elif dy < 0:
                moved[dy:] = False
            if dx > 0:
                moved[:, :dx] = False
            elif dx < 0:
                moved[:, dx:] = False
            total += paint * moved[..., None]
            count += moved
        take = (~filled) & (count >= 2)
        image[take] = total[take] / count[take, None]
        filled |= take
    return np.clip(image, 0, 1)


def _depths(images: torch.Tensor) -> torch.Tensor:
    folder = ROOT / "outputs"
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / "_waymo_rgb.pt"
    dest = folder / "_waymo_depth.pt"
    torch.save({"images": images}, source)
    python = ROOT / ".venv" / "Scripts" / "python.exe"
    env = os.environ.copy()
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    env["USE_TF"] = "0"
    subprocess.check_call([str(python), "-m", "sbzrv.waymo_depth", "--src", str(source), "--dst", str(dest)], cwd=ROOT, env=env)
    depth = torch.load(dest, map_location="cpu", weights_only=False)["depth"]
    source.unlink(missing_ok=True)
    dest.unlink(missing_ok=True)
    return depth


def load_clips(cfg: Config) -> dict:
    clips = _extract(cfg)
    surround, drive, reverse = [], [], []
    size = cfg.image_size
    front_pose = _pose("front")
    for clip in clips:
        frames, drive_frames, reverse_frames = [], [], []
        for frame in range(cfg.frames):
            image = clip["surround"][frame, 0].permute(1, 2, 0).numpy()
            k = clip["intrinsics"][frame][0]
            frames.append(clip["surround"][frame])
            drive_frames.append(_paint_ground(image, k, *front_pose, *OVER_DRIVE[:2], OVER_DRIVE[2], size))
            reverse_frames.append(_paint_ground(image, k, *front_pose, *OVER_REVERSE[:2], OVER_REVERSE[2], size))
        surround.append(torch.stack(frames, 0))
        drive.append(torch.from_numpy(np.stack(drive_frames, 0)).permute(0, 3, 1, 2).contiguous())
        reverse.append(torch.from_numpy(np.stack(reverse_frames, 0)).permute(0, 3, 1, 2).contiguous())
        print(f"俯视 {len(surround)}/{len(clips)}", flush=True)
    return {
        "surround": torch.stack(surround, 0),
        "drive": torch.stack(drive, 0),
        "reverse": torch.stack(reverse, 0),
    }

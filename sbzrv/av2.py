"""Argoverse 2 ring-camera clips and vision-only chase-camera targets.

The public sensor bucket has seven calibrated ring cameras. There is no roof
camera. Drive and reverse targets use the design chase rigs. Each chase ray
takes the ring photo in that direction. The drive frame looks up enough to
clear the hood. The reverse frame keeps the square above the vehicle body.
"""

from __future__ import annotations

import math
import os
import sys
import warnings
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import torch
from PIL import Image

from .config import ROOT, Config
from .render import DRIVE, REVERSE, _basis


CAMERAS = (
    "ring_front_center",
    "ring_front_left",
    "ring_front_right",
    "ring_side_left",
    "ring_side_right",
    "ring_rear_left",
    "ring_rear_right",
)
BUCKET = "https://argoverse.s3.amazonaws.com/"
PREFIX = "datasets/av2/sensor/val/"
# Rear-axle origin. The body is a stand-in so the chase view keeps the car
# in frame. It is not a measured mesh.
EGO_MIN = np.array((-1.15, -0.98, -0.40), dtype=np.float64)
EGO_MAX = np.array((3.85, 0.98, 1.35), dtype=np.float64)
_NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}


def _quiet_table(path: Path):
    sink = open(os.devnull, "w")
    stderr = sys.stderr
    sys.stderr = sink
    try:
        return feather.read_table(path)
    finally:
        sys.stderr = stderr
        sink.close()


def _column(table, name) -> np.ndarray:
    return np.array(table.column(name).to_pylist())


def _rotation(qw, qx, qy, qz) -> np.ndarray:
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def _get(url: str, dest: Path, tries: int = 4):
    if dest.is_file() and dest.stat().st_size > 1000:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for _ in range(tries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = response.read()
            if len(payload) < 1000:
                raise IOError(f"short response {len(payload)}")
            dest.write_bytes(payload)
            return
        except Exception as exc:
            last = exc
            dest.unlink(missing_ok=True)
    raise RuntimeError(f"下载失败 {url}: {last}")


def _list_keys(prefix: str) -> list[str]:
    token = None
    keys = []
    while True:
        query = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            query["continuation-token"] = token
        url = BUCKET + "?" + urllib.parse.urlencode(query)
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=180) as response:
            root = ET.fromstring(response.read())
        keys.extend(item.text for item in root.findall("s:Contents/s:Key", _NS))
        truncated = root.find("s:IsTruncated", _NS)
        if truncated is None or truncated.text != "true":
            break
        token = root.find("s:NextContinuationToken", _NS).text
    return keys


def _list_logs() -> list[str]:
    token = None
    logs = []
    while True:
        query = {"list-type": "2", "prefix": PREFIX, "delimiter": "/", "max-keys": "1000"}
        if token:
            query["continuation-token"] = token
        url = BUCKET + "?" + urllib.parse.urlencode(query)
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=180) as response:
            root = ET.fromstring(response.read())
        for item in root.findall("s:CommonPrefixes/s:Prefix", _NS):
            logs.append(item.text.rstrip("/").rsplit("/", 1)[-1])
        truncated = root.find("s:IsTruncated", _NS)
        if truncated is None or truncated.text != "true":
            break
        token = root.find("s:NextContinuationToken", _NS).text
    return logs


def _stamps(log: str, kind: str) -> list[int]:
    folder = ROOT / "data" / "av2" / log
    cache = folder / f"{kind}_stamps.txt"
    if cache.is_file():
        return [int(line) for line in cache.read_text(encoding="utf-8").split() if line]
    folder.mkdir(parents=True, exist_ok=True)
    if kind == "lidar":
        sub, suffix = "sensors/lidar/", ".feather"
    else:
        camera = "ring_front_center" if kind == "front" else kind
        sub, suffix = f"sensors/cameras/{camera}/", ".jpg"
    found = []
    for key in _list_keys(PREFIX + log + "/" + sub):
        name = key.rsplit("/", 1)[-1]
        if name.endswith(suffix):
            found.append(int(name[: -len(suffix)]))
    found.sort()
    cache.write_text("\n".join(str(item) for item in found), encoding="utf-8")
    return found


def _times(stamps: list[int], count: int) -> list[int]:
    stride = 10
    start = max(0, (len(stamps) - stride * (count - 1)) // 3)
    return [stamps[min(start + i * stride, len(stamps) - 1)] for i in range(count)]


def _nearest(stamps: list[int], stamp: int) -> int:
    position = int(np.searchsorted(stamps, stamp))
    choices = [stamps[index] for index in (position - 1, position) if 0 <= index < len(stamps)]
    picked = min(choices, key=lambda item: abs(item - stamp))
    if abs(picked - stamp) > 50_000_000:
        raise RuntimeError(f"最近的传感器时间差了 {abs(picked - stamp) / 1e6:.0f} ms")
    return picked


def _brightness(log: str) -> float:
    stamps = _stamps(log, "front")
    if len(stamps) < 8:
        return 0.0
    stamp = stamps[len(stamps) // 2]
    dest = ROOT / "data" / "av2" / log / "cameras" / "ring_front_center" / f"{stamp}.jpg"
    _get(BUCKET + PREFIX + log + f"/sensors/cameras/ring_front_center/{stamp}.jpg", dest)
    image = np.array(Image.open(dest).convert("RGB").resize((64, 64)))
    return float(image.mean())


def scene_ids(count: int) -> list[str]:
    marker = ROOT / "data" / "av2" / "scenes.txt"
    if marker.is_file():
        saved = [line.strip() for line in marker.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(saved) >= count:
            return saved[:count]
    chosen = []
    print("挑选白天的 Argoverse 2 验证片段", flush=True)
    for log in _list_logs():
        level = _brightness(log)
        print(f"  {log[:8]}  亮度 {level:.0f}", flush=True)
        if level >= 95:
            chosen.append(log)
        if len(chosen) == count:
            break
    if len(chosen) < count:
        raise RuntimeError(f"白天片段只有 {len(chosen)} 段，需要 {count} 段")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("\n".join(chosen) + "\n", encoding="utf-8")
    return chosen


def _fetch(log: str, count: int):
    folder = ROOT / "data" / "av2" / log
    ready = folder / "ready.txt"
    times = _times(_stamps(log, "front"), count)
    matched = {camera: [_nearest(_stamps(log, camera), stamp) for stamp in times] for camera in CAMERAS}
    signature = ",".join(str(stamp) for stamp in times)
    if ready.is_file() and ready.read_text(encoding="utf-8").strip() == signature:
        return matched
    jobs = []
    for name in ("calibration/intrinsics.feather", "calibration/egovehicle_SE3_sensor.feather"):
        jobs.append((BUCKET + PREFIX + log + "/" + name, folder / name))
    for camera, stamps in matched.items():
        for stamp in stamps:
            rel = f"sensors/cameras/{camera}/{stamp}.jpg"
            jobs.append((BUCKET + PREFIX + log + "/" + rel, folder / "cameras" / camera / f"{stamp}.jpg"))
    print(f"下载 {log[:8]}  {len(jobs)} 个文件", flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda job: _get(*job), jobs))
    ready.write_text(signature, encoding="utf-8")
    return matched


def _sensors(log: str) -> list[dict]:
    folder = ROOT / "data" / "av2" / log / "calibration"
    intr = _quiet_table(folder / "intrinsics.feather")
    extr = _quiet_table(folder / "egovehicle_SE3_sensor.feather")
    names = _column(intr, "sensor_name").tolist()
    pose_names = _column(extr, "sensor_name").tolist()
    cameras = []
    for camera in CAMERAS:
        row = names.index(camera)
        pose = pose_names.index(camera)
        qw, qx, qy, qz = (float(_column(extr, key)[pose]) for key in ("qw", "qx", "qy", "qz"))
        cameras.append(
            {
                "name": camera,
                "fx": float(_column(intr, "fx_px")[row]),
                "fy": float(_column(intr, "fy_px")[row]),
                "cx": float(_column(intr, "cx_px")[row]),
                "cy": float(_column(intr, "cy_px")[row]),
                "width": int(_column(intr, "width_px")[row]),
                "height": int(_column(intr, "height_px")[row]),
                "R": _rotation(qw, qx, qy, qz),
                "t": np.array([float(_column(extr, key)[pose]) for key in ("tx_m", "ty_m", "tz_m")]),
            }
        )
    return cameras


def _square(image: Image.Image, size: int) -> torch.Tensor:
    width, height = image.size
    side = min(width, height)
    x0 = (width - side) // 2
    extra = height - side
    y0 = int(extra * 0.72) if extra > 0 else 0
    cropped = image.crop((x0, y0, x0 + side, y0 + side)).resize((size, size), Image.Resampling.LANCZOS)
    return torch.from_numpy(np.array(cropped)).permute(2, 0, 1).float() / 255.0


def _points(path: Path) -> np.ndarray:
    table = _quiet_table(path)
    xyz = np.stack([_column(table, name).astype(np.float64) for name in ("x", "y", "z")], 1)
    flat = np.linalg.norm(xyz[:, :2], axis=1)
    keep = (flat > 1.5) & (flat < 46.0) & (xyz[:, 2] > -2.0) & (xyz[:, 2] < 12.0)
    return xyz[keep]


def _project(points: np.ndarray, camera: dict):
    local = (points - camera["t"]) @ camera["R"]
    depth = local[:, 2]
    scale = np.clip(depth, 1e-3, None)
    u = camera["fx"] * local[:, 0] / scale + camera["cx"]
    v = camera["fy"] * local[:, 1] / scale + camera["cy"]
    return u, v, depth


def _front_surface(u, v, depth, height, width):
    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)
    ok = (depth > 0.8) & (ui >= 1) & (ui < width - 2) & (vi >= 1) & (vi < height - 2)
    index = np.flatnonzero(ok)
    pixels = vi[index] * width + ui[index]
    order = np.argsort(depth[index])
    _, first = np.unique(pixels[order], return_index=True)
    chosen = index[order[first]]
    buffer = np.full(height * width, np.inf, dtype=np.float32)
    buffer[pixels[order[first]]] = depth[chosen].astype(np.float32)
    visible = np.zeros(len(depth), dtype=bool)
    visible[index] = depth[index] <= buffer[pixels] + 0.8
    return visible, buffer.reshape(height, width)


def _sample(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    finite = np.isfinite(u) & np.isfinite(v)
    u = np.clip(np.where(finite, u, -1.0), -1.0, width - 1.0)
    v = np.clip(np.where(finite, v, -1.0), -1.0, height - 1.0)
    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    u0 = np.clip(u0, 0, width - 2)
    v0 = np.clip(v0, 0, height - 2)
    du = (u - u0)[..., None]
    dv = (v - v0)[..., None]
    return (
        image[v0, u0] * (1 - du) * (1 - dv)
        + image[v0, u0 + 1] * du * (1 - dv)
        + image[v0 + 1, u0] * (1 - du) * dv
        + image[v0 + 1, u0 + 1] * du * dv
    )


def _colorize(points: np.ndarray, photos: list[np.ndarray], cameras: list[dict]):
    color = np.zeros((len(points), 3), dtype=np.float32)
    best = np.full(len(points), np.inf, dtype=np.float32)
    seen = np.zeros(len(points), dtype=bool)
    for photo, camera in zip(photos, cameras):
        u, v, depth = _project(points, camera)
        visible, _ = _front_surface(u, v, depth, camera["height"], camera["width"])
        take = visible & (depth < best)
        if not np.any(take):
            continue
        color[take] = _sample(photo, u[take], v[take]).astype(np.float32)
        best[take] = depth[take]
        seen[take] = True
    return color, seen


def _road_plane(points: np.ndarray):
    flat = np.linalg.norm(points[:, :2], axis=1)
    band = points[(flat > 4.0) & (flat < 30.0) & (points[:, 2] < 1.0)]
    if len(band) < 300:
        return 0.0, 0.0, -0.33
    road = band[band[:, 2] <= np.percentile(band[:, 2], 35)]
    design = np.stack((road[:, 0], road[:, 1], np.ones(len(road))), 1)
    slope_x, slope_y, offset = np.linalg.lstsq(design, road[:, 2], rcond=None)[0]
    if abs(slope_x) > 0.2 or abs(slope_y) > 0.2:
        return 0.0, 0.0, float(np.median(road[:, 2]))
    return float(slope_x), float(slope_y), float(offset)


def _hood_color(photo: np.ndarray, camera: dict) -> np.ndarray:
    height, width = photo.shape[:2]
    cx = int(np.clip(camera["cx"], 0, width - 1))
    y0 = int(height * 0.84)
    y1 = int(height * 0.96)
    x0 = max(0, cx - 48)
    x1 = min(width, cx + 48)
    patch = photo[y0:y1, x0:x1].reshape(-1, 3)
    return np.median(patch, axis=0).astype(np.float32)


def _paint_body(image, eye, dirs, travel, hood):
    t_near = np.full(dirs.shape[:2], -np.inf, dtype=np.float64)
    t_far = np.full(dirs.shape[:2], np.inf, dtype=np.float64)
    for axis in range(3):
        direction = dirs[..., axis]
        origin = float(eye[axis])
        parallel = np.abs(direction) < 1e-8
        safe = np.where(parallel, 1.0, direction)
        t0 = (EGO_MIN[axis] - origin) / safe
        t1 = (EGO_MAX[axis] - origin) / safe
        lo = np.minimum(t0, t1)
        hi = np.maximum(t0, t1)
        outside = parallel & ((origin < EGO_MIN[axis]) | (origin > EGO_MAX[axis]))
        lo = np.where(parallel, np.where(outside, np.inf, -np.inf), lo)
        hi = np.where(parallel, np.where(outside, -np.inf, np.inf), hi)
        t_near = np.maximum(t_near, lo)
        t_far = np.minimum(t_far, hi)
    enter = np.maximum(t_near, 0.0)
    ground = np.where(np.isfinite(travel) & (travel > 0.05), travel, np.inf)
    body = (t_far >= enter) & (enter < ground)
    image[body] = hood
    return image


def _paint_road(photos, cameras, buffers, plane, eye, target, fov, size, sky):
    slope_x, slope_y, offset = plane
    rotation, eye_v = _basis(eye, target)
    focal = (size * 0.5) / math.tan(math.radians(fov) * 0.5)
    axis = np.arange(size, dtype=np.float32)
    uu, vv = np.meshgrid(axis, axis, indexing="xy")
    dirs = np.stack(((uu - size * 0.5) / focal, (vv - size * 0.5) / focal, np.ones_like(uu)), -1)
    dirs = dirs @ rotation
    denom = dirs[..., 2] - slope_x * dirs[..., 0] - slope_y * dirs[..., 1]
    travel = (slope_x * eye_v[0] + slope_y * eye_v[1] + offset - eye_v[2]) / np.where(np.abs(denom) > 1e-4, denom, np.nan)
    hit = eye_v + travel[..., None] * dirs
    image = np.broadcast_to(sky, (size, size, 3)).copy()
    score = np.full((size, size), np.inf, dtype=np.float32)
    flat = hit.reshape(-1, 3)
    for photo, camera, buffer in zip(photos, cameras, buffers):
        u, v, depth = _project(flat, camera)
        u = u.reshape(size, size)
        v = v.reshape(size, size)
        depth = depth.reshape(size, size)
        finite = np.isfinite(u) & np.isfinite(v) & np.isfinite(depth)
        u = np.clip(np.where(finite, u, -1.0), -1.0, camera["width"] - 1.0)
        v = np.clip(np.where(finite, v, -1.0), -1.0, camera["height"] - 1.0)
        ui = np.rint(u).astype(np.int32)
        vi = np.rint(v).astype(np.int32)
        inside = (
            finite
            & (travel > 0.8)
            & (travel < 42.0)
            & (depth > 0.8)
            & (ui >= 2)
            & (ui < camera["width"] - 3)
            & (vi >= 2)
            & (vi < camera["height"] - 3)
        )
        if camera["name"] == "ring_front_center":
            inside &= vi < int(camera["height"] * 0.78)
        stored = np.full((size, size), np.inf, dtype=np.float32)
        stored[inside] = buffer[vi[inside], ui[inside]]
        visible = inside & (depth <= stored + 1.2)
        take = visible & (depth < score)
        if not np.any(take):
            continue
        sampled = _sample(photo, u, v)
        image[take] = sampled[take]
        score[take] = depth[take]
    near = (
        np.isfinite(travel)
        & (travel > 0.4)
        & (travel < 42.0)
        & (np.abs(hit[..., 0] + 0.2) < 7.0)
        & (np.abs(hit[..., 1]) < 3.5)
        & (score == np.inf)
    )
    image[near] = (0.28, 0.29, 0.31)
    return image, eye_v, dirs, travel


def _splat(image, points, colors, eye, target, fov, radius=1):
    size = image.shape[0]
    rotation, eye_v = _basis(eye, target)
    focal = (size * 0.5) / math.tan(math.radians(fov) * 0.5)
    local = (points - eye_v) @ rotation.T
    depth = local[:, 2]
    u = focal * local[:, 0] / np.clip(depth, 1e-3, None) + size * 0.5
    v = focal * local[:, 1] / np.clip(depth, 1e-3, None) + size * 0.5
    zbuf = np.full((size, size), np.inf, dtype=np.float32)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx * dx + dy * dy > radius * radius + 0.1:
                continue
            uu = np.rint(u + dx).astype(np.int32)
            vv = np.rint(v + dy).astype(np.int32)
            ok = (depth > 0.5) & (uu >= 0) & (uu < size) & (vv >= int(size * 0.18)) & (vv < size)
            index = np.flatnonzero(ok)
            if index.size == 0:
                continue
            pixels = vv[index] * size + uu[index]
            order = np.argsort(depth[index])
            _, first = np.unique(pixels[order], return_index=True)
            chosen = index[order[first]]
            closer = depth[chosen] < zbuf[vv[chosen], uu[chosen]]
            chosen = chosen[closer]
            image[vv[chosen], uu[chosen]] = colors[chosen]
            zbuf[vv[chosen], uu[chosen]] = depth[chosen]
    return image


def _overhead(points, colors, seen, photos, cameras, plane, rig, sky, hood):
    eye, target, fov = rig
    size_image, eye_v, dirs, travel = _paint_road(photos, cameras, _road_buffers(points, cameras), plane, eye, target, fov, 256, sky)
    height = points[:, 2] - (plane[0] * points[:, 0] + plane[1] * points[:, 1] + plane[2])
    cell = np.floor(points / 0.4).astype(np.int64)
    code = cell[:, 0] * 73856093 + cell[:, 1] * 19349663 + cell[:, 2] * 83492791
    _, inverse, counts = np.unique(code, return_inverse=True, return_counts=True)
    above = seen & (height > 0.85) & (counts[inverse] >= 6)
    body_box = (points[:, 0] > -1.2) & (points[:, 0] < 4.0) & (np.abs(points[:, 1]) < 1.15) & (points[:, 2] < 1.8)
    above &= ~body_box
    if np.any(above):
        size_image = _splat(size_image, points[above], colors[above], eye, target, fov)
    return _paint_body(size_image, eye_v, dirs, travel, hood)


def _road_buffers(points, cameras):
    buffers = []
    for camera in cameras:
        u, v, depth = _project(points, camera)
        _, buffer = _front_surface(u, v, depth, camera["height"], camera["width"])
        buffers.append(buffer)
    return buffers


GRID_X0, GRID_X1 = -80.0, 80.0
GRID_Y0, GRID_Y1 = -20.0, 20.0
# Design values. Near cells hold a facade; far cells only place a surface.
NEAR_REACH = 25.0
NEAR_CELL = 0.3
FAR_CELL = 1.0


def _agreement(points: np.ndarray, photos, cameras, reject_top: bool = True):
    """Median color where at least two cameras see the same point."""
    count_cameras = len(cameras)
    colors = np.full((len(points), count_cameras, 3), np.nan, dtype=np.float32)
    for index, (photo, camera) in enumerate(zip(photos, cameras)):
        u, v, depth = _project(points, camera)
        finite = np.isfinite(u) & np.isfinite(v) & (depth > 0.8)
        u = np.clip(np.where(finite, u, -1.0), -1.0, camera["width"] - 1.0)
        v = np.clip(np.where(finite, v, -1.0), -1.0, camera["height"] - 1.0)
        ui = np.rint(u).astype(np.int32)
        vi = np.rint(v).astype(np.int32)
        inside = (
            finite
            & (ui >= 2)
            & (ui < camera["width"] - 3)
            & (vi >= 2)
            & (vi < camera["height"] - 3)
        )
        if reject_top:
            inside &= vi > int(camera["height"] * 0.20)
        if camera["name"] == "ring_front_center":
            inside &= vi < int(camera["height"] * 0.78)
        if np.any(inside):
            colors[inside, index] = _sample(photo, u[inside], v[inside]).astype(np.float32)
    present = np.isfinite(colors[..., 0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(np.where(present[..., None], colors, np.nan), axis=1)
        deviation = np.mean(np.abs(colors - median[:, None, :]), axis=2)
        deviation = np.where(present, deviation, np.inf)
        agree = deviation < 0.09
        agree_count = agree.sum(1)
        score = np.full(len(points), np.inf, dtype=np.float32)
        enough = agree_count >= 2
        mean_deviation = np.nanmean(np.where(agree, deviation, np.nan), axis=1)
        agreed = np.nanmedian(np.where(agree[..., None], colors, np.nan), axis=1)
    score[enough] = mean_deviation[enough].astype(np.float32)
    agreed = np.nan_to_num(agreed, nan=0.0).astype(np.float32)
    return agreed, agree_count.astype(np.int32), score


def _vision_plane(photos, cameras):
    """Axle height for this rig. Cross-camera color does not pick a unique plane."""
    del photos, cameras
    return 0.0, 0.0, -0.33


def _occupancy(photos, cameras, plane, sky, x0, x1, y0, y1, cell):
    """Solid cells where multi-view color has a sharp height minimum."""
    slope_x, _, offset = plane
    xs = np.arange(x0, x1, cell, dtype=np.float32)
    ys = np.arange(y0, y1, cell, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    ground = slope_x * xx + offset
    layers = np.arange(1.0, 14.0, 0.8, dtype=np.float32)
    shape = xx.shape + layers.shape
    points = np.stack((
        np.broadcast_to(xx[..., None], shape),
        np.broadcast_to(yy[..., None], shape),
        ground[..., None] + layers,
    ), -1)
    flat = points.reshape(-1, 3)
    color, count, score = _agreement(flat, photos, cameras)
    del count
    score = np.where(np.isfinite(score), score, 1e3).reshape(xx.shape + layers.shape)
    color = color.reshape(xx.shape + layers.shape + (3,))
    sky_like = np.linalg.norm(color - sky, axis=-1) < 0.08
    score = np.where(sky_like, 1e3, score)
    top_index = np.argmin(score, axis=-1)
    best = np.take_along_axis(score, top_index[..., None], -1)[..., 0]
    below = np.take_along_axis(score, np.clip(top_index - 2, 0, layers.size - 1)[..., None], -1)[..., 0]
    above = np.take_along_axis(score, np.clip(top_index + 2, 0, layers.size - 1)[..., None], -1)[..., 0]
    dipped = (best < 0.05) & (best + 0.04 < below) & (best + 0.04 < above)
    top = np.where(dipped, ground + layers[top_index], -1.0).astype(np.float32)
    ego = (xx > -1.2) & (xx < 4.0) & (np.abs(yy) < 1.15)
    top[ego] = -1.0
    solid = top > ground + 0.8
    neighbor = np.zeros(solid.shape, dtype=np.int32)
    neighbor[1:-1, 1:-1] = (
        solid[:-2, :-2].astype(np.int32) + solid[:-2, 1:-1] + solid[:-2, 2:]
        + solid[1:-1, :-2] + solid[1:-1, 2:]
        + solid[2:, :-2] + solid[2:, 1:-1] + solid[2:, 2:]
    )
    keep = solid & (neighbor >= 2)
    top = np.where(keep, top, -1.0).astype(np.float32)
    return {"origin": (float(x0), float(y0)), "cell": float(cell), "top": top}


def _height_grids(photos, cameras, plane, sky):
    far = _occupancy(photos, cameras, plane, sky, GRID_X0, GRID_X1, GRID_Y0, GRID_Y1, FAR_CELL)
    near = _occupancy(
        photos, cameras, plane, sky, -NEAR_REACH, NEAR_REACH, GRID_Y0, GRID_Y1, NEAR_CELL,
    )
    return far, near


def _closest_color(points, photos, cameras, allow_sky=False):
    """Color from the nearest camera, rejected when another camera disagrees."""
    best_depth = np.full(len(points), np.inf, dtype=np.float32)
    best_color = np.zeros((len(points), 3), dtype=np.float32)
    samples = []
    for photo, camera in zip(photos, cameras):
        u, v, depth = _project(points, camera)
        finite = np.isfinite(u) & np.isfinite(v) & (depth > 0.8)
        u = np.clip(np.where(finite, u, -1.0), -1.0, camera["width"] - 1.0)
        v = np.clip(np.where(finite, v, -1.0), -1.0, camera["height"] - 1.0)
        ui = np.rint(u).astype(np.int32)
        vi = np.rint(v).astype(np.int32)
        inside = (
            finite
            & (ui >= 2)
            & (ui < camera["width"] - 3)
            & (vi >= 2)
            & (vi < camera["height"] - 3)
        )
        if not allow_sky:
            inside &= vi > int(camera["height"] * 0.20)
        if camera["name"] == "ring_front_center":
            inside &= vi < int(camera["height"] * 0.78)
        sampled = np.zeros((len(points), 3), dtype=np.float32)
        if np.any(inside):
            sampled[inside] = _sample(photo, u[inside], v[inside]).astype(np.float32)
        closer = inside & (depth < best_depth)
        best_depth = np.where(closer, depth.astype(np.float32), best_depth)
        best_color = np.where(closer[:, None], sampled, best_color)
        samples.append((inside, sampled))
    seen = best_depth < 1e8
    veto = np.zeros(len(points), dtype=bool)
    for inside, sampled in samples:
        delta = np.mean(np.abs(sampled - best_color), axis=1)
        veto |= inside & seen & (delta > 0.18)
    return best_color, seen, veto


def _paint_ground(photos, cameras, plane, eye, target, fov, size, sky):
    slope_x, slope_y, offset = plane
    rotation, eye_v = _basis(eye, target)
    focal = (size * 0.5) / math.tan(math.radians(fov) * 0.5)
    axis = np.arange(size, dtype=np.float32)
    uu, vv = np.meshgrid(axis, axis, indexing="xy")
    dirs = np.stack(((uu - size * 0.5) / focal, (vv - size * 0.5) / focal, np.ones_like(uu)), -1)
    dirs = dirs @ rotation
    denom = dirs[..., 2] - slope_x * dirs[..., 0] - slope_y * dirs[..., 1]
    travel = (slope_x * eye_v[0] + slope_y * eye_v[1] + offset - eye_v[2]) / np.where(np.abs(denom) > 1e-4, denom, np.nan)
    hit = eye_v + travel[..., None] * dirs
    image = np.broadcast_to(sky, (size, size, 3)).copy()
    flat = np.nan_to_num(hit.reshape(-1, 3), nan=0.0)
    color, seen, veto = _closest_color(flat, photos, cameras)
    color = color.reshape(size, size, 3)
    seen = seen.reshape(size, size)
    veto = veto.reshape(size, size)
    distance = np.linalg.norm(flat[:, :2], axis=1).reshape(size, size)
    veto = veto & (distance < 25.0)
    take = np.isfinite(travel) & (travel > 0.8) & (travel < 40.0) & seen & ~veto
    image[take] = color[take]
    near = (
        np.isfinite(travel)
        & (travel > 0.4)
        & (travel < 40.0)
        & (np.abs(hit[..., 0] + 0.2) < 7.0)
        & (np.abs(hit[..., 1]) < 3.5)
        & ~take
    )
    image[near] = (0.28, 0.29, 0.31)
    return image, eye_v, dirs, travel, take | near


def _roofs(points, grids):
    """Far cells first. The near grid then owns its rectangle, including empty cells."""
    roof = np.full(points.shape[:2], -1.0, dtype=np.float32)
    for grid in grids:
        x0, y0 = grid["origin"]
        cell = grid["cell"]
        top = grid["top"]
        gx = np.floor((points[..., 0] - x0) / cell).astype(np.int32)
        gy = np.floor((points[..., 1] - y0) / cell).astype(np.int32)
        inside = (gx >= 0) & (gy >= 0) & (gx < top.shape[0]) & (gy < top.shape[1])
        roof[inside] = top[gx[inside], gy[inside]]
    return roof


def _paint_volume(image, eye, dirs, travel, plane, grids, photos, cameras):
    slope_x, slope_y, offset = plane
    far = np.where(np.isfinite(travel) & (travel > 1.0), np.minimum(travel, 100.0), 80.0)
    hits = np.full(image.shape, np.nan, dtype=np.float32)
    painted = np.zeros(image.shape[:2], dtype=bool)
    for scale in np.linspace(0.04, 1.0, 48):
        distance = np.maximum(far * scale, 0.5)
        points = eye + distance[..., None] * dirs
        road = slope_x * points[..., 0] + slope_y * points[..., 1] + offset
        roof = _roofs(points, grids)
        solid = ~painted & (roof > road + 0.8) & (points[..., 2] <= roof) & (points[..., 2] > road + 0.25)
        if not np.any(solid):
            continue
        hits[solid] = points[solid]
        painted |= solid
    need = painted.reshape(-1)
    if not np.any(need):
        return image
    color, seen, _ = _closest_color(hits.reshape(-1, 3)[need], photos, cameras, allow_sky=True)
    write = np.zeros(image.shape[:2], dtype=bool)
    write.reshape(-1)[need] = seen
    chosen = np.zeros_like(image)
    chosen.reshape(-1, 3)[need] = color
    image[write] = chosen[write]
    return image


def _paint_sky(image, eye, dirs, photos, cameras, covered):
    """Sample ring photos along rays that miss the road, including the real sky."""
    length = np.linalg.norm(dirs, axis=-1, keepdims=True)
    direction = dirs / np.clip(length, 1e-6, None)
    need = ~covered
    if not np.any(need):
        return image
    points = eye + 200.0 * direction[need]
    color, seen, _ = _closest_color(points, photos, cameras, allow_sky=True)
    painted = np.zeros(image.shape[:2], dtype=bool)
    painted[need] = seen
    chosen = np.zeros_like(image)
    chosen[need] = color
    image[painted] = chosen[painted]
    return image


def _ray_samples():
    """Design values. One meter along the chase ray, and only out to 40 m."""
    return np.arange(2.0, 40.0, 1.0, dtype=np.float32)


def _surface_score(points, photos, cameras, sky, reject_top: bool = True):
    color, _, score = _agreement(points, photos, cameras, reject_top=reject_top)
    sky_like = np.linalg.norm(color - sky, axis=-1) < 0.08
    score = np.where(sky_like, np.inf, score)
    return color, score


def _paint_surfaces(image, eye, dirs, travel, plane, photos, cameras, sky):
    """One depth per ray: the nearest place where multi-view color dips."""
    slope_x, slope_y, offset = plane
    distances = _ray_samples()
    size = image.shape[0]
    hit_dist = np.full((size, size), np.nan, dtype=np.float32)
    hit_color = np.zeros_like(image)
    near_ray = np.isfinite(travel) & (travel > 2.5) & (travel < 40.0)
    limit = np.where(near_ray, travel - 0.6, 0.0).astype(np.float32)
    for r0 in range(0, size, 32):
        r1 = min(size, r0 + 32)
        sub_dirs = dirs[r0:r1]
        points = eye + distances[None, None, :, None] * sub_dirs[:, :, None, :]
        usable = distances[None, None, :] < limit[r0:r1, :, None]
        flat = points.reshape(-1, 3)
        mask = usable.reshape(-1)
        if not np.any(mask):
            continue
        color, score = _surface_score(flat[mask], photos, cameras, sky, reject_top=False)
        full_score = np.full(mask.shape, np.inf, dtype=np.float32)
        full_color = np.zeros((mask.size, 3), dtype=np.float32)
        full_score[mask] = score
        full_color[mask] = color
        samples = distances.size
        full_score = full_score.reshape(r1 - r0, size, samples)
        full_color = full_color.reshape(r1 - r0, size, samples, 3)
        road = slope_x * points[..., 0] + slope_y * points[..., 1] + offset
        full_score = np.where(points[..., 2] < road + 0.35, np.inf, full_score)
        center = full_score[:, :, 1:-1]
        before = full_score[:, :, :-2]
        after = full_score[:, :, 2:]
        dipped = np.zeros(full_score.shape, dtype=bool)
        dipped[:, :, 1:-1] = (
            np.isfinite(center)
            & np.isfinite(before)
            & np.isfinite(after)
            & (center < 0.04)
            & (center + 0.05 < before)
            & (center + 0.05 < after)
        )
        found = dipped.any(axis=-1)
        index = np.argmax(dipped, axis=-1)
        rows = np.arange(r1 - r0)[:, None]
        cols = np.arange(size)[None, :]
        chosen = distances[index]
        picked = full_color[rows, cols, index]
        block_dist = hit_dist[r0:r1]
        block_color = hit_color[r0:r1]
        block_dist[found] = chosen[found]
        block_color[found] = picked[found]
    measured = hit_dist.copy()
    measured[:2, :] = np.nan
    measured[-2:, :] = np.nan
    measured[:, :2] = np.nan
    measured[:, -2:] = np.nan
    patches = [
        np.roll(np.roll(measured, dy, 0), dx, 1)
        for dy in range(-2, 3)
        for dx in range(-2, 3)
    ]
    stack = np.stack(patches, 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        depth = np.nanmedian(stack, axis=0).astype(np.float32)
    keep = np.isfinite(stack).sum(0) >= 8
    depth = np.where(keep, depth, np.nan)
    points = eye + depth[..., None] * dirs
    road = slope_x * points[..., 0] + slope_y * points[..., 1] + offset
    above = np.isfinite(depth) & (points[..., 2] > road + 0.35)
    ego = (
        (points[..., 0] > -1.2)
        & (points[..., 0] < 4.0)
        & (np.abs(points[..., 1]) < 1.15)
        & (points[..., 2] < 1.8)
    )
    take = above & ~ego
    if not np.any(take):
        return image
    color, seen, _ = _closest_color(points.reshape(-1, 3)[take.reshape(-1)], photos, cameras, allow_sky=True)
    write = np.zeros(take.shape, dtype=bool)
    write.reshape(-1)[take.reshape(-1)] = seen
    chosen = np.zeros_like(image)
    chosen.reshape(-1, 3)[take.reshape(-1)] = color
    image[write] = chosen[write]
    return image


def _outside_ego(point) -> bool:
    return not (
        float(point[0]) > -1.2 and float(point[0]) < 4.0
        and abs(float(point[1])) < 1.15 and float(point[2]) < 1.8
    )


def _vertical_hit(origin, direction, anchor):
    """Ray hit on a vertical plane through the ground contact. Side walls use y, the far wall uses x."""
    if abs(float(anchor[1])) > 2.5:
        denom = float(direction[1])
        plane = float(anchor[1]) - float(origin[1])
    else:
        denom = float(direction[0])
        plane = float(anchor[0]) - float(origin[0])
    if abs(denom) < 1e-3:
        return None
    travel = plane / denom
    if travel < 0.8 or travel > 90.0:
        return None
    return origin + np.float32(travel) * direction


def _ring_surface_points(photos, cameras, plane, sky):
    """Upright surfaces from each ring photo.

    A pixel that other cameras agree is on the ground stays on the ground.
    Pixels above that, in the same column, stand on a vertical plane through
    the last ground contact. Design values: one sample every 4 pixels,
    side walls where the contact is more than 2.5 m off the center line.
    """
    slope_x, slope_y, offset = plane
    stride = 4
    gathered_points = []
    gathered_colors = []
    for photo, camera in zip(photos, cameras):
        us = np.arange(stride // 2, camera["width"], stride, dtype=np.float32)
        vs = np.arange(stride // 2, camera["height"], stride, dtype=np.float32)
        uu, vv = np.meshgrid(us, vs, indexing="xy")
        keep = np.ones(uu.shape, dtype=bool)
        if camera["name"] == "ring_front_center":
            keep &= vv < camera["height"] * 0.78
        ui = np.clip(np.rint(uu).astype(np.int32), 0, photo.shape[1] - 1)
        vi = np.clip(np.rint(vv).astype(np.int32), 0, photo.shape[0] - 1)
        color = photo[vi, ui]
        keep &= np.linalg.norm(color - sky, axis=-1) >= 0.08
        if not np.any(keep):
            continue
        dir_cam = np.stack(
            ((uu - camera["cx"]) / camera["fx"], (vv - camera["cy"]) / camera["fy"], np.ones_like(uu)),
            -1,
        )
        direction = (dir_cam @ np.asarray(camera["R"], dtype=np.float64).T).astype(np.float32)
        origin = np.asarray(camera["t"], dtype=np.float32)
        denom = direction[..., 2] - slope_x * direction[..., 0] - slope_y * direction[..., 1]
        travel = (slope_x * origin[0] + slope_y * origin[1] + offset - origin[2]) / np.where(
            np.abs(denom) > 1e-4, denom, np.nan
        )
        ground = origin + travel[..., None] * direction
        usable = keep & np.isfinite(travel) & (travel > 1.0) & (travel < 90.0)
        on_ground_pixel = np.zeros(uu.shape, dtype=bool)
        score = np.full(uu.shape, np.inf, dtype=np.float32)
        flat_index = np.flatnonzero(usable)
        for start in range(0, len(flat_index), 4000):
            chosen_index = flat_index[start:start + 4000]
            _, _, part = _agreement(ground.reshape(-1, 3)[chosen_index], photos, cameras, reject_top=False)
            score.reshape(-1)[chosen_index] = part
        on_ground_pixel = usable & (vv > camera["cy"]) & (score < 0.06)
        rows, cols = uu.shape
        contact_x = np.full(cols, np.nan, dtype=np.float32)
        contact_y = np.full(cols, np.nan, dtype=np.float32)
        column_points = []
        column_colors = []
        for col in range(cols):
            order = np.argsort(-vv[:, col])
            for row in order:
                if not on_ground_pixel[row, col]:
                    continue
                base = ground[row, col]
                contact_x[col] = base[0]
                contact_y[col] = base[1]
                if _outside_ego(base):
                    column_points.append(np.array(base, dtype=np.float32))
                    column_colors.append(np.array(color[row, col], dtype=np.float32))
        smooth_x = contact_x.copy()
        smooth_y = contact_y.copy()
        for col in range(cols):
            lo = max(0, col - 4)
            hi = min(cols, col + 5)
            sx = contact_x[lo:hi]
            sy = contact_y[lo:hi]
            if np.isfinite(sx).any():
                smooth_x[col] = np.nanmedian(sx)
                smooth_y[col] = np.nanmedian(sy)
        finite = np.isfinite(smooth_x)
        if finite.any() and camera["name"] in {"ring_front_center", "ring_rear_left", "ring_rear_right", "ring_front_left", "ring_front_right"}:
            middle = np.zeros(cols, dtype=bool)
            middle[int(cols * 0.35):int(cols * 0.65)] = True
            left = np.arange(cols) < int(cols * 0.35)
            right = np.arange(cols) >= int(cols * 0.65)
            far_x = float(np.nanmedian(smooth_x[middle & finite])) if np.any(middle & finite) else float(np.nanmedian(smooth_x[finite]))
            left_y = float(np.nanmedian(smooth_y[left & finite])) if np.any(left & finite) else 6.0
            right_y = float(np.nanmedian(smooth_y[right & finite])) if np.any(right & finite) else -6.0
            for col in range(cols):
                for row in range(rows):
                    if not keep[row, col] or on_ground_pixel[row, col]:
                        continue
                    anchor = np.array((far_x, 0.0, offset), dtype=np.float32)
                    direction_row = direction[row, col]
                    if uu[row, col] < camera["cx"] - camera["width"] * 0.12:
                        anchor[1] = left_y
                    elif uu[row, col] > camera["cx"] + camera["width"] * 0.12:
                        anchor[1] = right_y
                    point = _vertical_hit(origin, direction_row, anchor)
                    if point is None or not _outside_ego(point):
                        continue
                    if float(point[2]) < offset + 0.4 or float(point[2]) > 30.0:
                        continue
                    column_points.append(np.asarray(point, dtype=np.float32))
                    column_colors.append(np.array(color[row, col], dtype=np.float32))
        if column_points:
            gathered_points.append(np.stack(column_points).astype(np.float32))
            gathered_colors.append(np.stack(column_colors).astype(np.float32))
    if not gathered_points:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(gathered_points, 0), np.concatenate(gathered_colors, 0)


def _splat_far(image, far, points, colors, eye, target, fov):
    """Project ring points into the chase view. Only the far pixels are written."""
    if len(points) == 0 or not np.any(far):
        return far & False
    size = image.shape[0]
    rotation, _ = _basis(eye, target)
    focal = (size * 0.5) / math.tan(math.radians(fov) * 0.5)
    camera = (points - eye) @ rotation.T
    depth = camera[:, 2]
    keep = depth > 0.6
    u = camera[:, 0] / np.clip(depth, 1e-3, None) * focal + size * 0.5
    v = camera[:, 1] / np.clip(depth, 1e-3, None) * focal + size * 0.5
    keep &= (u >= 1) & (u < size - 1) & (v >= 1) & (v < size - 1)
    if not np.any(keep):
        return np.zeros(far.shape, dtype=bool)
    depth = depth[keep]
    u = u[keep]
    v = v[keep]
    colors = colors[keep]
    order = np.argsort(depth)
    zbuf = np.full(far.shape, np.inf, dtype=np.float32)
    wrote = np.zeros(far.shape, dtype=bool)
    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)
    for index in order:
        y = int(vi[index])
        x = int(ui[index])
        if not far[y, x] or depth[index] >= zbuf[y, x]:
            continue
        zbuf[y, x] = depth[index]
        image[y, x] = colors[index]
        wrote[y, x] = True
    for _ in range(2):
        filled_z = zbuf.copy()
        filled = image.copy()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                shifted_z = np.roll(np.roll(zbuf, dy, 0), dx, 1)
                shifted = np.roll(np.roll(image, dx, 1), dy, 0)
                better = far & ~wrote & np.isfinite(shifted_z) & (shifted_z < filled_z)
                filled_z = np.where(better, shifted_z, filled_z)
                filled = np.where(better[..., None], shifted, filled)
        zbuf = filled_z
        image[:] = filled
        wrote |= far & np.isfinite(zbuf) & (zbuf < 1e8)
    return wrote


def _fill_real_sky(image, need, eye, dirs, photos, cameras, sky):
    """Uncovered far pixels take a ring pixel only when that pixel is sky."""
    if not np.any(need):
        return image
    photo, camera = photos[0], cameras[0]
    length = np.linalg.norm(dirs, axis=-1, keepdims=True)
    direction = dirs / np.clip(length, 1e-6, None)
    points = (eye + 80.0 * direction).reshape(-1, 3)
    u, v, depth = _project(points, camera)
    finite = np.isfinite(u) & np.isfinite(v) & (depth > 0.8)
    color = np.zeros((points.shape[0], 3), dtype=np.float32)
    if np.any(finite):
        color[finite] = _sample(photo, u[finite], v[finite]).astype(np.float32)
    sky_like = finite & (np.linalg.norm(color - sky, axis=1) < 0.12)
    mask = need.reshape(-1) & sky_like
    chosen = image.reshape(-1, 3)
    chosen[mask] = color[mask]
    image[:] = chosen.reshape(image.shape)
    return image


def _ground_hits(camera, plane):
    """Ground intersections of the downward pixels. Used to place the street walls."""
    slope_x, slope_y, offset = plane
    stride = 16
    us = np.arange(stride // 2, camera["width"], stride, dtype=np.float32)
    vs = np.arange(stride // 2, camera["height"], stride, dtype=np.float32)
    vs = vs[vs > camera["cy"]]
    if camera["name"] == "ring_front_center":
        vs = vs[vs < camera["height"] * 0.78]
    if len(vs) == 0 or len(us) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.float32)
    uu, vv = np.meshgrid(us, vs, indexing="xy")
    dir_cam = np.stack(
        ((uu - camera["cx"]) / camera["fx"], (vv - camera["cy"]) / camera["fy"], np.ones_like(uu)),
        -1,
    )
    direction = (dir_cam @ np.asarray(camera["R"], dtype=np.float64).T).astype(np.float32)
    origin = np.asarray(camera["t"], dtype=np.float32)
    denom = direction[..., 2] - slope_x * direction[..., 0] - slope_y * direction[..., 1]
    travel = (slope_x * origin[0] + slope_y * origin[1] + offset - origin[2]) / np.where(
        np.abs(denom) > 1e-4, denom, np.nan
    )
    ground = origin + travel[..., None] * direction
    ok = np.isfinite(travel) & (travel > 4.0) & (travel < 70.0) & (np.abs(ground[..., 1]) < 18.0)
    return ground[ok].astype(np.float32), uu[ok].astype(np.float32)


def _street_walls(cameras, plane):
    """Place the facades outside the lane. The far plane is only a backdrop."""
    named = {camera["name"]: camera for camera in cameras}
    left_hits, _ = _ground_hits(named["ring_front_left"], plane)
    right_hits, _ = _ground_hits(named["ring_front_right"], plane)
    left_y = float(np.percentile(left_hits[:, 1], 75)) if len(left_hits) >= 8 else 7.2
    right_y = float(np.percentile(right_hits[:, 1], 25)) if len(right_hits) >= 8 else -7.2
    if left_y < 5.8:
        left_y = 7.2
    if right_y > -5.8:
        right_y = -7.2
    front = named["ring_front_center"]
    hits, uu = _ground_hits(front, plane)
    far_x = 72.0
    if len(hits):
        middle = np.abs(uu - front["cx"]) < front["width"] * 0.12
        distant = hits[middle & (hits[:, 0] > 25.0), 0] if np.any(middle) else hits[:0, 0]
        if len(distant):
            far_x = float(np.median(distant))
    if not np.isfinite(far_x) or far_x < 40.0:
        far_x = 72.0
    back = []
    for name in ("ring_rear_left", "ring_rear_right"):
        rear_hits, _ = _ground_hits(named[name], plane)
        if len(rear_hits):
            back.append(rear_hits[:, 0])
    back_x = -56.0
    if back:
        xs = np.concatenate(back)
        far_back = xs[xs < -20.0]
        if len(far_back):
            back_x = float(np.median(far_back))
    if not np.isfinite(back_x) or back_x > -20.0:
        back_x = -56.0
    return far_x, back_x, left_y, right_y


def _nearest_color(points, photos, cameras, sky):
    """Color from the ring camera that faces the point, skipping sky pixels."""
    color = np.zeros((len(points), 3), dtype=np.float32)
    best = np.full(len(points), -1.0, dtype=np.float32)
    seen = np.zeros(len(points), dtype=bool)
    for photo, camera in zip(photos, cameras):
        u, v, depth = _project(points, camera)
        ui = np.rint(u).astype(np.int32)
        vi = np.rint(v).astype(np.int32)
        inside = (
            (depth > 0.8)
            & (ui >= 2) & (ui < camera["width"] - 3)
            & (vi >= 2) & (vi < camera["height"] - 3)
        )
        if camera["name"] == "ring_front_center":
            inside &= vi < int(camera["height"] * 0.78)
        axis = np.asarray(camera["R"], dtype=np.float64)[:, 2]
        toward = points - np.asarray(camera["t"], dtype=np.float64)
        cosine = (toward * axis).sum(axis=1) / np.clip(np.linalg.norm(toward, axis=1) * np.linalg.norm(axis), 1e-4, None)
        cosine = cosine.astype(np.float32)
        take = inside & (cosine > 0.45) & (cosine > best)
        if not np.any(take):
            continue
        sampled = _sample(photo, u[take], v[take]).astype(np.float32)
        sky_like = np.linalg.norm(sampled - sky, axis=1) < 0.10
        keep = take.copy()
        keep[np.flatnonzero(take)] = ~sky_like
        if np.any(keep):
            color[keep] = _sample(photo, u[keep], v[keep]).astype(np.float32)
        best = np.where(keep, cosine, best)
        seen |= keep
    return color, seen


def _camera_index(photos, cameras):
    return {camera["name"]: (photo, camera) for photo, camera in zip(photos, cameras)}


def _project_inside(points, camera, ground_only):
    """Pixels of one camera. The first camera that contains a point keeps it."""
    u, v, depth = _project(points, camera)
    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)
    inside = (
        np.isfinite(u) & np.isfinite(v) & (depth > 0.8)
        & (ui >= 2) & (ui < camera["width"] - 3)
        & (vi >= 2) & (vi < camera["height"] - 3)
    )
    if camera["name"] == "ring_front_center":
        inside &= vi < int(camera["height"] * 0.78)
    if ground_only:
        inside &= vi > int(camera["cy"])
    return u, v, inside


def _sample_order(points, order, indexed, ground_only):
    color = np.zeros((len(points), 3), dtype=np.float32)
    seen = np.zeros(len(points), dtype=bool)
    for name in order:
        if name not in indexed or not np.any(~seen):
            continue
        photo, camera = indexed[name]
        u, v, inside = _project_inside(points, camera, ground_only)
        take = ~seen & inside
        if not np.any(take):
            continue
        color[take] = _sample(photo, u[take], v[take]).astype(np.float32)
        seen[take] = True
    return color, seen


def _fill_facing(image, need, dirs, order, indexed):
    """Empty pixels take the ring photo in the same direction. One camera per pixel."""
    if not np.any(need):
        return image
    length = np.linalg.norm(dirs, axis=-1, keepdims=True)
    direction = dirs / np.clip(length, 1e-6, None)
    color = np.zeros_like(image)
    seen = np.zeros(need.shape, dtype=bool)
    for name in order:
        remain = need & ~seen
        if name not in indexed or not np.any(remain):
            continue
        photo, camera = indexed[name]
        local = direction @ np.asarray(camera["R"], dtype=np.float64)
        depth = local[..., 2]
        scale = np.clip(depth, 1e-3, None)
        u = camera["fx"] * local[..., 0] / scale + camera["cx"]
        v = camera["fy"] * local[..., 1] / scale + camera["cy"]
        ui = np.rint(u).astype(np.int32)
        vi = np.rint(v).astype(np.int32)
        inside = (
            remain & (depth > 0.25)
            & (ui >= 2) & (ui < camera["width"] - 3)
            & (vi >= 2) & (vi < camera["height"] - 3)
        )
        if camera["name"] == "ring_front_center":
            inside &= vi < int(camera["height"] * 0.78)
        if not np.any(inside):
            continue
        color[inside] = _sample(photo, u[inside], v[inside]).astype(np.float32)
        seen[inside] = True
    remain = need & ~seen
    if np.any(remain) and order and order[0] in indexed:
        photo, camera = indexed[order[0]]
        local = direction @ np.asarray(camera["R"], dtype=np.float64)
        depth = local[..., 2]
        scale = np.clip(depth, 1e-3, None)
        u = camera["fx"] * local[..., 0] / scale + camera["cx"]
        v = camera["fy"] * local[..., 1] / scale + camera["cy"]
        take = remain & (depth > 0.25) & (v < camera["height"]) & ((v < 2.0) | (u < 2.0) | (u > camera["width"] - 3.0))
        if camera["name"] == "ring_front_center":
            take &= v < camera["height"] * 0.74
        if np.any(take):
            u = np.clip(u, 2.0, camera["width"] - 3.0)
            v = np.clip(v, 2.0, camera["height"] - 3.0)
            color[take] = _sample(photo, u[take], v[take]).astype(np.float32)
            seen[take] = True
    image[seen] = color[seen]
    return image


def _fill_blend(image, need, dirs, names, indexed, body_frac):
    """Two rear cameras meet on a short ramp, so the street has one seam instead of a cut."""
    colors = []
    weights = []
    for name in names:
        photo, camera = indexed[name]
        length = np.linalg.norm(dirs, axis=-1, keepdims=True)
        direction = dirs / np.clip(length, 1e-6, None)
        local = direction @ np.asarray(camera["R"], dtype=np.float64)
        depth = local[..., 2]
        scale = np.clip(depth, 1e-3, None)
        u = camera["fx"] * local[..., 0] / scale + camera["cx"]
        v = camera["fy"] * local[..., 1] / scale + camera["cy"]
        ui = np.rint(u).astype(np.int32)
        vi = np.rint(v).astype(np.int32)
        inside = (
            need & (depth > 0.25)
            & (ui >= 2) & (ui < camera["width"] - 3)
            & (vi >= 2) & (vi < camera["height"] - 3)
            & (v < camera["height"] * body_frac)
        )
        color = np.zeros_like(image)
        if np.any(inside):
            color[inside] = _sample(photo, u[inside], v[inside]).astype(np.float32)
        center = 1.0 - np.clip(np.abs(u - camera["cx"]) / (camera["width"] * 0.5), 0.0, 1.0)
        weight = np.where(inside, depth.astype(np.float32) * (0.2 + center), 0.0).astype(np.float32)
        colors.append(color)
        weights.append(weight)
    left, right = weights
    total = left + right
    take = need & (total > 1e-4)
    delta = left - right
    share = 1.0 / (1.0 + np.exp(-delta * 8.0))
    mixed = colors[0] * share[..., None] + colors[1] * (1.0 - share[..., None])
    image[take] = mixed[take]
    return image


def _square_above_body(image, dirs, camera, body_frac):
    """Keep the square the rear cameras actually see, above the vehicle body."""
    column = dirs[:, image.shape[1] // 2]
    length = np.linalg.norm(column, axis=-1, keepdims=True)
    direction = column / np.clip(length, 1e-6, None)
    local = direction @ np.asarray(camera["R"], dtype=np.float64)
    depth = local[:, 2]
    row = camera["fy"] * local[:, 1] / np.clip(depth, 1e-3, None) + camera["cy"]
    valid = (depth > 0.25) & (row > 2.0) & (row < camera["height"] * body_frac)
    rows = np.flatnonzero(valid)
    if len(rows) < 32:
        return image
    last = int(rows[0])
    for item in rows[1:]:
        if int(item) > last + 2:
            break
        last = int(item)
    side = min(last + 1, image.shape[0], image.shape[1])
    x0 = (image.shape[1] - side) // 2
    crop = np.clip(image[:side, x0:x0 + side], 0.0, 1.0)
    picture = Image.fromarray((crop * 255.0).astype(np.uint8))
    picture = picture.resize((image.shape[1], image.shape[0]), Image.Resampling.LANCZOS)
    return np.asarray(picture).astype(np.float32) / 255.0


def _paint_walls(image, eye, dirs, travel, plane, photos, cameras, sky, walls):
    """Warp one ring photo. Pixels aimed at the hood take the ground point instead."""
    del plane, walls
    indexed = _camera_index(photos, cameras)
    looking_back = float(eye[0]) > 0.0
    if looking_back:
        road_order = ("ring_rear_left", "ring_rear_right")
        fill_order = road_order
    else:
        road_order = ("ring_front_center",)
        fill_order = road_order
    canvas = np.broadcast_to(sky, image.shape).copy()
    covered = np.ones(canvas.shape[:2], dtype=bool)
    if looking_back:
        canvas = _fill_blend(canvas, covered, dirs, fill_order, indexed, 0.76)
    else:
        canvas = _fill_facing(canvas, covered, dirs, fill_order, indexed)
    hole = np.max(np.abs(canvas - sky), axis=-1) < 1e-6
    road = hole & np.isfinite(travel) & (travel > 1.2) & (travel < 80.0)
    if (not looking_back) and np.any(road):
        point = eye + travel[..., None] * dirs
        flat_road = np.reshape(road, -1)
        color, seen = _sample_order(np.reshape(point, (-1, 3))[flat_road], road_order, indexed, ground_only=True)
        write = np.zeros(road.shape, dtype=bool)
        write[road] = seen
        chosen = np.zeros_like(canvas)
        chosen[road] = color
        canvas[write] = chosen[write]
    if looking_back:
        camera = indexed["ring_rear_left"][1]
        canvas = _square_above_body(canvas, dirs, camera, 0.76)
    return canvas


def _pitch_target(eye, target, degrees):
    """Look up so the frame stays on the street instead of the vehicle body."""
    eye = np.asarray(eye, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.array((0.0, 0.0, 1.0), dtype=np.float64))
    right = right / np.linalg.norm(right)
    cam_up = np.cross(right, forward)
    angle = math.radians(degrees)
    forward = forward * math.cos(angle) + cam_up * math.sin(angle)
    return eye + forward


def _bottom_photo_row(eye, target, fov, size, camera):
    rotation, _eye_v = _basis(eye, target)
    focal = (size * 0.5) / math.tan(math.radians(fov) * 0.5)
    camera_dir = np.array((0.0, (size - 1.0 - size * 0.5) / focal, 1.0), dtype=np.float64)
    world = camera_dir @ np.asarray(rotation, dtype=np.float64)
    world = world / np.linalg.norm(world)
    local = world @ np.asarray(camera["R"], dtype=np.float64)
    if local[2] <= 0.05:
        return 1.0e9
    return float(camera["fy"] * local[1] / local[2] + camera["cy"])


def _aim_clear(eye, target, fov, size, camera, limit):
    low, high = 0.0, 28.0
    for _ in range(18):
        mid = 0.5 * (low + high)
        row = _bottom_photo_row(eye, _pitch_target(eye, target, mid), fov, size, camera)
        if row > limit:
            low = mid
        else:
            high = mid
    return _pitch_target(eye, target, high)


def _vision_overhead(photos, cameras, plane, rig, sky, hood, size, walls):
    del hood
    eye, target, fov = rig
    named = {camera["name"]: camera for camera in cameras}
    if float(eye[0]) < 0.0:
        camera = named["ring_front_center"]
        target = _aim_clear(eye, target, fov, size, camera, camera["height"] * 0.74)
    image, eye_v, dirs, travel, _covered = _paint_ground(photos, cameras, plane, eye, target, fov, size, sky)
    return _paint_walls(image, eye_v, dirs, travel, plane, photos, cameras, sky, walls)


def _frame(log: str, cameras: list[dict], stamps: dict, size: int):
    folder = ROOT / "data" / "av2" / log
    photos = []
    surround = []
    for camera in cameras:
        image = Image.open(folder / "cameras" / camera["name"] / f"{stamps[camera['name']]}.jpg").convert("RGB")
        photos.append(np.array(image).astype(np.float32) / 255.0)
        surround.append(_square(image, size))
    sky = np.median(photos[0][:80].reshape(-1, 3), axis=0).astype(np.float32)
    hood = _hood_color(photos[0], cameras[0])
    plane = _vision_plane(photos, cameras)
    drive = _vision_overhead(photos, cameras, plane, DRIVE, sky, hood, size, None)
    reverse = _vision_overhead(photos, cameras, plane, REVERSE, sky, hood, size, None)
    return torch.stack(surround, 0), drive, reverse


def load_clips(cfg: Config) -> dict:
    logs = scene_ids(cfg.scenes)
    surround, drive, reverse = [], [], []
    for log in logs:
        matched = _fetch(log, cfg.frames)
        cameras = _sensors(log)
        views, drives, reverses = [], [], []
        for index in range(cfg.frames):
            print(f"  表面 {index + 1}/{cfg.frames}", flush=True)
            stamps = {camera: matched[camera][index] for camera in CAMERAS}
            view, drive_frame, reverse_frame = _frame(log, cameras, stamps, cfg.image_size)
            views.append(view)
            drives.append(drive_frame)
            reverses.append(reverse_frame)
        surround.append(torch.stack(views, 0))
        drive.append(torch.from_numpy(np.stack(drives, 0)).permute(0, 3, 1, 2).contiguous())
        reverse.append(torch.from_numpy(np.stack(reverses, 0)).permute(0, 3, 1, 2).contiguous())
        print(f"机位 {len(surround)}/{len(logs)}  {log[:8]}", flush=True)
    return {
        "surround": torch.stack(surround, 0).contiguous(),
        "drive": torch.stack(drive, 0).contiguous(),
        "reverse": torch.stack(reverse, 0).contiguous(),
    }

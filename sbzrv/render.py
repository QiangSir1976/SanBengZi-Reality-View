"""Solid shaded street renderer.

Boxes are filled polygons, not point splats. The image is the control picture
that the photo pass turns into a daylight street photograph.
"""

from __future__ import annotations

import math

import numpy as np
import torch


SURROUND = (
    ((2.3, 0.0, 1.45), (22.0, 0.0, 1.0), 90.0),
    ((0.5, 1.15, 1.5), (16.0, 7.0, 1.1), 100.0),
    ((0.5, -1.15, 1.5), (16.0, -7.0, 1.1), 100.0),
    ((-1.8, 0.0, 1.55), (-20.0, 0.0, 1.0), 100.0),
)
# Bottom ray clears the roof. The horizon sits in the upper third, and the
# center of the frame is about 60 m down the road. Design values, not Tesla extrinsics.
DRIVE = ((-10.0, 0.0, 12.0), (60.0, 0.0, -0.2), 48.0)
REVERSE = ((12.7, 0.0, 12.0), (-57.0, 0.0, -0.2), 48.0)
SUN = np.array((0.25, 0.35, 0.90), dtype=np.float32)
SUN /= np.linalg.norm(SUN)


def _shade(albedo, normal):
    light = 0.34 + 0.66 * max(0.0, float(np.dot(normal, SUN)))
    color = np.clip(np.array(albedo, dtype=np.float32) * light, 0, 1)
    return color


def _quad(origin, axis_u, axis_v):
    o = np.array(origin, dtype=np.float32)
    u = np.array(axis_u, dtype=np.float32)
    v = np.array(axis_v, dtype=np.float32)
    return [o, o + u, o + u + v, o + v]


def _add_box(quads, center, size, albedo, shadow=True):
    c = np.array(center, dtype=np.float32)
    h = np.array(size, dtype=np.float32) * 0.5
    faces = (
        (np.array((1, 0, 0), np.float32), np.array((0, h[1] * 2, 0), np.float32), np.array((0, 0, h[2] * 2), np.float32), c + (h[0], -h[1], -h[2])),
        (np.array((-1, 0, 0), np.float32), np.array((0, 0, h[2] * 2), np.float32), np.array((0, h[1] * 2, 0), np.float32), c + (-h[0], -h[1], -h[2])),
        (np.array((0, 1, 0), np.float32), np.array((0, 0, h[2] * 2), np.float32), np.array((h[0] * 2, 0, 0), np.float32), c + (-h[0], h[1], -h[2])),
        (np.array((0, -1, 0), np.float32), np.array((h[0] * 2, 0, 0), np.float32), np.array((0, 0, h[2] * 2), np.float32), c + (-h[0], -h[1], -h[2])),
        (np.array((0, 0, 1), np.float32), np.array((h[0] * 2, 0, 0), np.float32), np.array((0, h[1] * 2, 0), np.float32), c + (-h[0], -h[1], h[2])),
        (np.array((0, 0, -1), np.float32), np.array((0, h[1] * 2, 0), np.float32), np.array((h[0] * 2, 0, 0), np.float32), c + (-h[0], -h[1], -h[2])),
    )
    for normal, axis_u, axis_v, origin in faces:
        quads.append((_quad(origin, axis_u, axis_v), _shade(albedo, normal)))
    if shadow:
        quads.append((
            _quad((c[0] - h[0] - 0.25, c[1] - h[1] - 0.2, 0.02), (size[0] + 0.5, 0, 0), (0, size[1] + 0.4, 0)),
            np.array((0.07, 0.07, 0.07), dtype=np.float32),
        ))


def _add_facade(quads, center, size, wall):
    c = np.array(center, dtype=np.float32)
    h = np.array(size, dtype=np.float32) * 0.5
    glass = np.array((0.16, 0.22, 0.28), dtype=np.float32)
    wall = np.array(wall, dtype=np.float32)
    floors = max(2, int(size[2] / 1.1))
    bays = max(2, int(max(size[0], size[1]) / 1.3))
    faces = (
        (np.array((1.0, 0, 0), np.float32), np.array((0, 1, 0), np.float32), np.array((0, 0, 1), np.float32), 1, 0),
        (np.array((-1.0, 0, 0), np.float32), np.array((0, 0, 1), np.float32), np.array((0, 1, 0), np.float32), 0, 1),
        (np.array((0, 1.0, 0), np.float32), np.array((0, 0, 1), np.float32), np.array((1, 0, 0), np.float32), 0, 0),
        (np.array((0, -1.0, 0), np.float32), np.array((1, 0, 0), np.float32), np.array((0, 0, 1), np.float32), 0, 0),
    )
    # Roof and a plain base course.
    _add_box(quads, (c[0], c[1], c[2] + h[2] - 0.15), (size[0], size[1], 0.3), wall * 0.75, shadow=False)
    for normal, _, _, _, _ in faces:
        pass
    specs = (
        ((1, 0, 0), (0, size[1], 0), (0, 0, size[2]), c + (h[0], -h[1], -h[2]), bays, floors),
        ((-1, 0, 0), (0, 0, size[2]), (0, size[1], 0), c + (-h[0], -h[1], -h[2]), floors, bays),
        ((0, 1, 0), (0, 0, size[2]), (size[0], 0, 0), c + (-h[0], h[1], -h[2]), floors, bays),
        ((0, -1, 0), (size[0], 0, 0), (0, 0, size[2]), c + (-h[0], -h[1], -h[2]), bays, floors),
    )
    for normal, axis_u, axis_v, origin, nu, nv in specs:
        normal = np.array(normal, dtype=np.float32)
        axis_u = np.array(axis_u, dtype=np.float32)
        axis_v = np.array(axis_v, dtype=np.float32)
        origin = np.array(origin, dtype=np.float32)
        du, dv = axis_u / nu, axis_v / nv
        for iu in range(nu):
            for iv in range(nv):
                margin_u = du * 0.18
                margin_v = dv * 0.22
                cell = origin + du * iu + dv * iv
                quads.append((_quad(cell, du, dv), _shade(wall, normal)))
                glass_origin = cell + margin_u + margin_v + normal * 0.05
                quads.append((_quad(glass_origin, du - margin_u * 2, dv - margin_v * 2), _shade(glass, normal)))


def _add_car(quads, x, y, color):
    paint = np.array(color, dtype=np.float32)
    _add_box(quads, (x, y, 0.55), (4.3, 1.85, 0.85), paint * 0.92)
    _add_box(quads, (x - 0.15, y, 1.15), (2.3, 1.7, 0.7), paint * 0.55 + np.array((0.05, 0.08, 0.1)), shadow=False)
    _add_box(quads, (x + 0.85, y, 1.05), (1.15, 1.6, 0.55), (0.12, 0.16, 0.18), shadow=False)
    for wx, wy in ((1.35, 0.82), (1.35, -0.82), (-1.35, 0.82), (-1.35, -0.82)):
        _add_box(quads, (x + wx, y + wy, 0.32), (0.55, 0.28, 0.58), (0.04, 0.04, 0.04), shadow=False)


def _add_trike(quads):
    body = (0.17, 0.18, 0.19)
    _add_box(quads, (0.2, 0.0, 0.95), (2.1, 1.55, 0.9), body)
    _add_box(quads, (0.15, 0.0, 1.6), (1.5, 1.4, 0.7), (0.22, 0.30, 0.34), shadow=False)
    _add_box(quads, (1.15, 0.0, 0.72), (0.7, 1.2, 0.35), body, shadow=False)
    _add_box(quads, (1.45, 0.0, 0.95), (0.08, 0.7, 0.08), (0.92, 0.93, 0.88), shadow=False)
    _add_box(quads, (1.7, 0.0, 0.34), (0.38, 0.24, 0.58), (0.05, 0.05, 0.05), shadow=False)
    _add_box(quads, (-0.55, 0.72, 0.36), (0.62, 0.32, 0.64), (0.05, 0.05, 0.05), shadow=False)
    _add_box(quads, (-0.55, -0.72, 0.36), (0.62, 0.32, 0.64), (0.05, 0.05, 0.05), shadow=False)
    _add_box(quads, (1.55, 0.62, 0.62), (0.1, 0.12, 0.1), (1.0, 0.48, 0.08), shadow=False)
    _add_box(quads, (1.55, -0.62, 0.62), (0.1, 0.12, 0.1), (1.0, 0.48, 0.08), shadow=False)


def scene_quads(seed: int, frame: int):
    rng = np.random.default_rng(seed)
    shift = frame * 0.9
    quads = []
    walls = (
        (0.55, 0.52, 0.48),
        (0.42, 0.46, 0.50),
        (0.62, 0.58, 0.52),
        (0.38, 0.36, 0.34),
    )
    for i, x0 in enumerate((-18.0, -8.0, 7.0, 16.0, 26.0, 36.0)):
        for side in (-1.0, 1.0):
            height = 7.5 + (i % 3) * 2.4
            _add_facade(
                quads,
                (x0 - shift, side * 8.2, height * 0.5),
                (6.5, 4.0, height),
                walls[(i + int(side > 0)) % len(walls)],
            )
    paints = (
        (0.72, 0.12, 0.10),
        (0.12, 0.28, 0.62),
        (0.86, 0.86, 0.84),
        (0.10, 0.40, 0.24),
        (0.78, 0.48, 0.10),
    )
    for i in range(4):
        lane = 2.8 if (seed + i) % 2 == 0 else -2.8
        x0 = 8.0 + i * 7.0 + float(rng.uniform(0, 1.2))
        own = float(rng.uniform(-1.0, 5.0))
        _add_car(quads, x0 - shift + own * frame * 0.08, lane, paints[(seed + i) % len(paints)])
    _add_car(quads, -9.0 - shift, -2.8, (0.90, 0.90, 0.88))
    world = list(quads)
    _add_trike(quads)
    return world, quads, shift


def _basis(eye, target):
    eye = np.array(eye, dtype=np.float32)
    forward = np.array(target, dtype=np.float32) - eye
    forward /= np.linalg.norm(forward)
    up = np.array((0.0, 0.0, 1.0), dtype=np.float32)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-4:
        right = np.array((1.0, 0.0, 0.0), dtype=np.float32)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack((right, down, forward), 0)
    return rotation, eye


def _ground(size, rotation, eye, focal, phase):
    axis = np.arange(size, dtype=np.float32)
    uu, vv = np.meshgrid(axis, axis, indexing="xy")
    dirs = np.stack(((uu - size * 0.5) / focal, (vv - size * 0.5) / focal, np.ones_like(uu)), -1)
    dirs = dirs @ rotation
    denom = dirs[..., 2]
    t_hit = np.where(np.abs(denom) > 1e-4, -eye[2] / denom, -1.0)
    hit = eye + t_hit[..., None] * dirs
    sky_t = np.clip(dirs[..., 2] * 0.5 + 0.5, 0, 1)
    horizon = np.array((0.78, 0.82, 0.86), dtype=np.float32)
    zenith = np.array((0.42, 0.60, 0.82), dtype=np.float32)
    sky = horizon * (1 - sky_t[..., None]) + zenith * sky_t[..., None]
    x, y = hit[..., 0], hit[..., 1]
    ay = np.abs(y)
    noise = np.sin(x * 0.65 + phase) * np.sin(y * 0.8)
    noise = noise * 0.5 + 0.5
    asphalt = np.array((0.16, 0.16, 0.17), dtype=np.float32) + (noise[..., None] - 0.5) * 0.04
    walk = np.array((0.45, 0.43, 0.40), dtype=np.float32)
    color = np.where(ay[..., None] < 5.4, asphalt, walk)
    dash = (np.mod(x + phase, 4.0) < 1.5) & (((ay - 1.8) < 0.09) & ((ay - 1.8) > -0.09) | (ay < 0.07)) & (ay < 5.4)
    edge = (np.abs(ay - 5.15) < 0.08)
    paint = np.array((0.90, 0.78, 0.28), dtype=np.float32)
    white = np.array((0.82, 0.82, 0.80), dtype=np.float32)
    color = np.where(dash[..., None], paint, color)
    color = np.where(edge[..., None], white, color)
    image = np.where((t_hit > 0.05)[..., None], color, sky)
    depth = np.where(t_hit > 0.05, t_hit, 1e4).astype(np.float32)
    return np.clip(image, 0, 1), depth


def _fill_convex(canvas, screen, color):
    height, width = canvas.shape[:2]
    minx = max(int(np.floor(screen[:, 0].min())), 0)
    maxx = min(int(np.ceil(screen[:, 0].max())), width - 1)
    miny = max(int(np.floor(screen[:, 1].min())), 0)
    maxy = min(int(np.ceil(screen[:, 1].max())), height - 1)
    if maxx <= minx or maxy <= miny:
        return
    xs = np.arange(minx, maxx + 1)
    ys = np.arange(miny, maxy + 1)
    xx, yy = np.meshgrid(xs, ys)
    count = len(screen)
    area = 0.0
    for i in range(count):
        area += float(screen[i, 0] * screen[(i + 1) % count, 1] - screen[(i + 1) % count, 0] * screen[i, 1])
    sign = 1.0 if area >= 0 else -1.0
    inside = np.ones(xx.shape, dtype=bool)
    for i in range(count):
        ax, ay = screen[i]
        bx, by = screen[(i + 1) % count]
        cross = (bx - ax) * (yy - ay) - (by - ay) * (xx - ax)
        inside &= sign * cross >= -0.75
    view = canvas[miny:maxy + 1, minx:maxx + 1]
    view[inside] = color


def _draw(image, rotation, eye, focal, quads):
    canvas = np.clip(image, 0, 1).astype(np.float32)
    order = []
    for corners, color in quads:
        pts = np.stack(corners, 0)
        normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal /= norm
        if np.dot(normal, eye - pts[0]) <= 0:
            continue
        cam = (pts - eye) @ rotation.T
        if np.any(cam[:, 2] < 0.25):
            continue
        depth = float(cam[:, 2].mean())
        screen = np.stack((
            focal * cam[:, 0] / cam[:, 2] + image.shape[1] * 0.5,
            focal * cam[:, 1] / cam[:, 2] + image.shape[0] * 0.5,
        ), 1)
        fog = math.exp(-depth * 0.018)
        horizon = np.array((0.78, 0.82, 0.86), dtype=np.float32)
        shaded = np.clip(color * fog + horizon * (1 - fog), 0, 1)
        order.append((depth, screen, shaded))
    order.sort(key=lambda item: -item[0])
    for _, screen, shaded in order:
        _fill_convex(canvas, screen, shaded)
    return canvas


def render(quads, eye, target, fov, size, phase):
    rotation, eye_v = _basis(eye, target)
    focal = (size * 0.5) / math.tan(math.radians(fov) * 0.5)
    ground, _ = _ground(size, rotation, eye_v, focal, phase)
    return _draw(ground, rotation, eye_v, focal, quads)


def render_clip(seed: int, size: int = 256, frames: int = 4, device: str | torch.device = "cpu"):
    del device
    surround, drive, reverse = [], [], []
    for frame in range(frames):
        world, both, phase = scene_quads(seed, frame)
        views = [render(world, eye, look, fov, size, phase) for eye, look, fov in SURROUND]
        surround.append(np.stack(views, 0))
        drive.append(render(both, *DRIVE[:2], DRIVE[2], size, phase))
        reverse.append(render(both, *REVERSE[:2], REVERSE[2], size, phase))
    surround = torch.from_numpy(np.stack(surround, 0)).permute(0, 1, 4, 2, 3).contiguous()
    pack = lambda seq: torch.from_numpy(np.stack(seq, 0)).permute(0, 3, 1, 2).contiguous()
    return {"surround": surround, "drive": pack(drive), "reverse": pack(reverse)}

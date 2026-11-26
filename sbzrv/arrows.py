"""Paint a Tesla-style route on the drive view.

A smooth blue ribbon lies on the lane. Above it, a row of chevrons with a
blue fill and a white edge floats at roof height. Both use the chase camera.
This prototype has no route planner, so the path is a design value.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image, ImageDraw

from .render import DRIVE, _basis

LANE_Y = -0.15
GROUND_Z = -0.33
PITCH = 12.0
PATH = (0, 176, 255, 228)
CHEVRON = (58, 168, 255, 235)
EDGE = (255, 255, 255, 255)
SCALE = 4


def _pitch_target(eye, target, degrees):
    eye = np.asarray(eye, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, np.array((0.0, 0.0, 1.0)))
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    angle = math.radians(degrees)
    forward = forward * math.cos(angle) + up * math.sin(angle)
    return eye + forward


def _project(points, size):
    eye, target, fov = DRIVE
    target = _pitch_target(eye, target, PITCH)
    rotation, eye_v = _basis(np.asarray(eye, np.float32), np.asarray(target, np.float32))
    focal = (size * 0.5) / math.tan(math.radians(fov) * 0.5)
    local = (np.asarray(points, np.float32) - eye_v) @ rotation.T
    depth = local[:, 2]
    valid = depth > 0.5
    u = np.full(len(points), np.nan, dtype=np.float32)
    v = np.full(len(points), np.nan, dtype=np.float32)
    u[valid] = focal * local[valid, 0] / depth[valid] + size * 0.5
    v[valid] = focal * local[valid, 1] / depth[valid] + size * 0.5
    return np.stack((u, v), 1), valid


def _curve(controls, count):
    """Smooth Catmull-Rom curve through the control points."""
    points = np.asarray(controls, dtype=np.float64)
    extended = np.vstack((points[0], points, points[-1]))
    samples = []
    steps = max(count // (len(points) - 1), 8)
    for i in range(1, len(extended) - 2):
        p0, p1, p2, p3 = extended[i - 1:i + 3]
        t = np.linspace(0.0, 1.0, steps, endpoint=False)[:, None]
        samples.append(
            0.5 * (
                (2 * p1)
                + (-p0 + p2) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3
            )
        )
    samples.append(points[-1:])
    return np.vstack(samples)


def _centerline(count=160):
    """Ground path: in the lane near the car, then bending right under the chevrons."""
    return _curve(
        (
            (21.0, LANE_Y),
            (36.0, -0.8),
            (48.0, -5.0),
            (60.0, -9.5),
            (74.0, -13.5),
        ),
        count,
    )


_ROUTE = None


def _route_table(step=0.5):
    """The route the chase camera is driving: straight ahead along the lane.

    Heading 0 points along +x, which is the forward direction of the drive view.
    """
    distance = np.arange(0.0, 120.0 + step, step)
    xy = np.column_stack((distance, np.full(len(distance), LANE_Y)))
    yaw = np.zeros(len(distance), dtype=np.float64)
    return distance, xy, yaw


def _route():
    global _ROUTE
    if _ROUTE is None:
        _ROUTE = _route_table()
    return _ROUTE


def _pose(distance, xy, yaw, travel):
    travel = float(np.clip(travel, distance[0], distance[-1]))
    index = int(np.searchsorted(distance, travel))
    index = min(max(index, 1), len(distance) - 1)
    span = distance[index] - distance[index - 1]
    blend = 0.0 if span <= 1e-6 else (travel - distance[index - 1]) / span
    point = (1.0 - blend) * xy[index - 1] + blend * xy[index]
    heading = (1.0 - blend) * yaw[index - 1] + blend * yaw[index]
    return point, heading


def _to_ego(points, origin, heading):
    delta = np.asarray(points, dtype=np.float64) - origin
    cosine, sine = math.cos(heading), math.sin(heading)
    ex = cosine * delta[:, 0] + sine * delta[:, 1]
    ey = -sine * delta[:, 0] + cosine * delta[:, 1]
    return np.stack((ex, ey), 1)


def _camera(travel):
    """The chase view keeps looking down the street while the car rolls forward."""
    return np.array((float(travel), LANE_Y), dtype=np.float64), 0.0


def _ahead(travel, near=18.0, far=70.0):
    """The route ahead of the car, in the chase camera."""
    distance, xy, _yaw = _route()
    origin, heading = _camera(travel)
    window = (distance >= travel + near) & (distance <= travel + far)
    if int(window.sum()) < 8:
        return None
    ego = _to_ego(xy[window], origin, heading)
    samples = np.linspace(0.0, len(ego) - 1, 140)
    left = np.floor(samples).astype(int)
    right = np.minimum(left + 1, len(ego) - 1)
    blend = (samples - left)[:, None]
    return (1.0 - blend) * ego[left] + blend * ego[right]


def _ribbon(center, width):
    tangent = np.gradient(center, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-8
    normal = np.stack((-tangent[:, 1], tangent[:, 0]), 1)
    fade = np.linspace(0.0, 1.0, len(center))
    half = np.full(len(center), width * 0.5)
    tip = fade > 0.86
    half[tip] = width * 0.5 * (1.0 - fade[tip]) / 0.14
    left = center + normal * half[:, None]
    right = center - normal * half[:, None]
    return np.vstack((left, right[::-1]))


# Tip leads along the path. Wings sit behind it, so the mark reads as one chevron.
CHEVRON_SHAPE = (
    (0.05, -1.00),
    (0.42, -0.46),
    (1.05, 0.00),
    (0.42, 0.46),
    (0.05, 1.00),
    (0.34, 0.40),
    (0.58, 0.00),
    (0.34, -0.40),
)
HOVER = 6.4


def _screen_chevron(draw, anchor, heading, scale, fill, size):
    """A camera-facing chevron. Its point follows the path on the screen."""
    ahead = np.array((anchor[0] + heading[0], anchor[1] + heading[1], anchor[2]), np.float32)
    lifted = np.array((anchor[0], anchor[1], anchor[2] + 1.0), np.float32)
    pixels, valid = _project(np.stack((anchor, ahead, lifted)).astype(np.float32), size)
    if not bool(valid[0]):
        return
    origin = pixels[0]
    direction = pixels[1] - pixels[0] if valid[1] else np.array((0.0, -1.0), np.float32)
    length = float(np.linalg.norm(direction))
    direction = np.array((0.0, -1.0), np.float32) if length < 0.4 else direction / length
    ppm = float(np.linalg.norm(pixels[2] - pixels[0])) if valid[2] else 8.0
    # Perspective shrinks the far marks, but each one stays a readable chevron.
    span = max(13.0, ppm * 2.1) * scale
    side = np.array((-direction[1], direction[0]))
    glyph = []
    for along, across in CHEVRON_SHAPE:
        point = origin + direction * (along - 0.45) * span + side * across * span * 0.72
        glyph.append((float(point[0] * SCALE), float(point[1] * SCALE)))
    stroke = max(2, int(span * SCALE * 0.075))
    glow = (150, 226, 255, 100)
    draw.line(glyph + [glyph[0]], fill=glow, width=stroke + max(4, stroke), joint="curve")
    draw.polygon(glyph, fill=fill)
    draw.line(glyph + [glyph[0]], fill=EDGE, width=stroke, joint="curve")


def _pixels(xyz, size):
    pixels, valid = _project(np.asarray(xyz, dtype=np.float32), size)
    if not np.all(valid):
        return None
    return [(float(u * SCALE), float(v * SCALE)) for u, v in pixels]


def _fill(draw, xyz, fill, size, outline=None, width=1):
    points = _pixels(xyz, size)
    if points is None:
        return
    draw.polygon(points, fill=fill)
    if outline is not None:
        draw.line(points + [points[0]], fill=outline, width=width, joint="curve")


def _route_bends(travel, threshold=math.radians(8.0)):
    """True when the route ahead of the car changes heading."""
    distance, xy, yaw = _route()
    _point, near = _pose(distance, xy, yaw, travel + 18.0)
    _far_point, far = _pose(distance, xy, yaw, travel + 70.0)
    return abs(far - near) >= threshold


def _moving_marks(travel):
    """Chevrons ahead of the car. Near ones are larger and sit above the ribbon."""
    distance, xy, yaw = _route()
    origin, heading = _camera(travel)
    marks = []
    for place, (ahead, scale) in enumerate((
        (24.0, 1.12),
        (32.0, 0.90),
        (41.0, 0.72),
        (51.0, 0.58),
        (62.0, 0.46),
        (74.0, 0.36),
    )):
        point, mark_yaw = _pose(distance, xy, yaw, travel + ahead)
        ego = _to_ego(point.reshape(1, 2), origin, heading)[0]
        tangent = np.array((math.cos(mark_yaw), math.sin(mark_yaw)))
        marks.append((ego, tangent, scale, place))
    return marks


def paint_route(image: Image.Image, travel: float | None = None) -> Image.Image:
    """Draw the row of roof-height chevrons when the route ahead bends.

    travel is how far the car has driven along its own route, in meters.
    A straight route is left unmarked.
    """
    size = image.size[0]

    if travel is None:
        marks = None
    else:
        center = _ahead(travel)
        marks = _moving_marks(travel) if _route_bends(travel) else []
        if center is None or not marks:
            return image.convert("RGB")

    if marks is None:
        # Same order as the reference: a row rising to the right, large on the left.
        anchors = (
            (26.0, 2.4, 1.15),
            (34.0, -0.6, 0.92),
            (42.0, -4.2, 0.74),
            (51.0, -8.0, 0.60),
            (61.0, -11.5, 0.48),
            (72.0, -14.5, 0.38),
        )
        marks = []
        for place, (x, y, scale) in enumerate(anchors):
            nxt = anchors[min(place + 1, len(anchors) - 1)]
            heading = np.array((nxt[0] - x, nxt[1] - y), np.float64)
            if place == len(anchors) - 1:
                prev = anchors[place - 1]
                heading = np.array((x - prev[0], y - prev[1]), np.float64)
            marks.append((np.array((x, y)), heading, scale, place))

    overlay = Image.new("RGBA", (size * SCALE, size * SCALE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    for point, heading, scale, place in reversed(marks):
        anchor = np.array((point[0], point[1], GROUND_Z + HOVER), np.float32)
        fill = (CHEVRON[0], CHEVRON[1], CHEVRON[2], int(242 - place * 18))
        _screen_chevron(draw, anchor, heading, scale, fill, size)

    overlay = overlay.resize(image.size, Image.Resampling.LANCZOS)
    base = image.convert("RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")


def paint_route_tensor(frame: torch.Tensor, travel: float | None = None) -> torch.Tensor:
    """frame is 3xHxW in 0..1. Returns the same layout."""
    array = (frame.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    painted = paint_route(Image.fromarray(array, mode="RGB"), travel)
    out = torch.from_numpy(np.asarray(painted).astype(np.float32) / 255.0)
    return out.permute(2, 0, 1).to(frame.dtype)

"""Contact sheet and a local page for the latest sample."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from .config import ROOT


def _font(size: int):
    for path in (
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _image(tensor: torch.Tensor, box: int) -> Image.Image:
    array = (tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    image = Image.fromarray(array, mode="RGB")
    if image.size != (box, box):
        image = image.resize((box, box), Image.Resampling.NEAREST)
    return image


_SOURCE_COPY = {
    "av2": (
        "环视是 Argoverse 2 实拍，远景和天空由七路相机合成",
        "环视是 Argoverse 2 Sensor 的七路实拍。车顶没有实拍相机，前进在车后高处看向前方约 60 米，倒车在车前高处看向车后。路面、远处的楼和天空都取自这些照片。这是研究原型，不是特斯拉的车端模型。",
    ),
    "waymo": (
        "环视是 Waymo 实拍，俯视由这些照片重投影",
        "环视是 Waymo 端到端数据里的真实相机照片。俯视没有对应的实拍机位，由这些照片按地面重投影到固定的前进和倒车机位。这是研究原型，不是特斯拉的车端模型。",
    ),
}


def camera_names(count: int) -> tuple[str, ...]:
    table = {
        3: ("前", "左前", "右前"),
        4: ("前", "左", "右", "后"),
        7: ("前", "左前", "右前", "左侧", "右侧", "左后", "右后"),
    }
    return table.get(count, tuple(f"相机{i + 1}" for i in range(count)))


def save_scene_check(clip: dict, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    box = 160
    cameras = clip["surround"].shape[1]
    names = camera_names(cameras)
    labels = names + ("前进", "倒车")
    frames = [clip["surround"][0, i] for i in range(cameras)]
    frames.extend((clip["drive"][0], clip["reverse"][0]))
    sheet = Image.new("RGB", (box * len(frames) + 70, box + 48), (14, 15, 18))
    draw = ImageDraw.Draw(sheet)
    font = _font(18)
    for i, (frame, label) in enumerate(zip(frames, labels)):
        sheet.paste(_image(frame, box), (16 + i * (box + 8), 36))
        draw.text((16 + i * (box + 8), 8), label, fill=(232, 230, 227), font=font)
    temporary = path.with_suffix(".tmp.png")
    sheet.save(temporary)
    try:
        temporary.replace(path)
    except OSError:
        temporary.replace(path.with_name(path.stem + "_new.png"))


def save_preview(path: Path, rows: list[tuple[str, list[torch.Tensor]]], source: str = "waymo"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    box = 148 if max(len(frames) for _, frames in rows) > 5 else 180
    pad = 16
    label_w = 132
    width = label_w + max(len(frames) for _, frames in rows) * (box + 10) + pad
    height = 168 + len(rows) * (box + 18)
    sheet = Image.new("RGB", (width, height), (14, 15, 18))
    draw = ImageDraw.Draw(sheet)
    logo = Image.open(ROOT / "assets" / "logo.jpg").convert("RGB")
    logo.thumbnail((210, 130))
    plate = Image.new("RGB", (logo.width + 16, logo.height + 16), (255, 255, 255))
    plate.paste(logo, (8, 8))
    sheet.paste(plate, (pad, 18))
    title = _font(32)
    body = _font(18)
    draw.text((pad + plate.width + 24, 28), "SanBengZi Reality View", fill=(242, 240, 236), font=title)
    draw.text((pad + plate.width + 24, 78), "本机原型  ·  一次前向  ·  前进 / 倒车两个机位", fill=(232, 168, 74), font=body)
    caption, page_text = _SOURCE_COPY.get(source, _SOURCE_COPY["waymo"])
    draw.text((pad + plate.width + 24, 108), caption, fill=(160, 156, 150), font=body)
    y = 168
    for label, frames in rows:
        draw.text((pad, y + box // 2 - 10), label, fill=(232, 230, 227), font=body)
        for i, frame in enumerate(frames):
            sheet.paste(_image(frame, box), (label_w + i * (box + 10), y))
        y += box + 18
    temporary = path.with_suffix(".tmp.png")
    sheet.save(temporary)
    temporary.replace(path)
    _write_page(path, page_text)


def _write_page(preview: Path, quiet: str):
    page = preview.parent / "index.html"
    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>SanBengZi Reality View</title>
<style>
  body { margin: 0; background: #0e0f12; color: #e8e6e3; font-family: "Segoe UI", "Microsoft YaHei", sans-serif; }
  header { display: flex; gap: 28px; align-items: center; padding: 32px 40px 12px; }
  .plate { background: #fff; padding: 8px; }
  .plate img { display: block; height: 120px; width: auto; }
  h1 { margin: 0 0 8px; font-weight: 560; letter-spacing: 0.03em; font-size: 32px; }
  .amber { color: #e8a84a; margin: 0; }
  .quiet { color: #9a9690; max-width: 640px; line-height: 1.55; }
  main { padding: 12px 40px 48px; }
  main img { width: min(1280px, 100%); height: auto; background: #0e0f12; }
</style>
</head>
<body>
<header>
  <div class="plate"><img src="../assets/logo.jpg" alt="SanBengZi"></div>
  <div>
    <h1>SanBengZi Reality View</h1>
    <p class="amber">本机原型 · 冻结 DINOv2 · 视频 MeanFlow · 一次前向</p>
    <p class="quiet">QUIET_TEXT</p>
  </div>
</header>
<main>
  <img src="preview.png" alt="latest preview">
</main>
</body>
</html>
"""
    page.write_text(html.replace("QUIET_TEXT", quiet), encoding="utf-8")

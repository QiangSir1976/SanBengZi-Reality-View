"""Train the tiny codec, then the video MeanFlow, and write a preview."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from .autoencoder import FrozenVAE, Refiner, Stride4Codec, _Codec, load_codec, make_codec
from .config import ROOT, Config
from .dataset import batch_from, build_cache
from .meanflow import endpoint_loss, meanflow_loss
from .model import VideoMeanFlow
from .preview import save_preview, save_scene_check
from .infer import render_preview, sample


def _cfg_from_args(args) -> Config:
    cfg = Config()
    for key in ("scenes", "ae_steps", "steps", "batch_size", "seed", "device", "latent_channels", "latent_size"):
        value = getattr(args, key)
        if value is not None:
            setattr(cfg, key, value)
    return cfg


def _take(data, cfg, device):
    n = data["drive"].shape[0]
    indices = torch.randint(0, n, (cfg.batch_size,))
    view = torch.randint(0, 2, (cfg.batch_size,))
    overhead, vision = batch_from(data, indices, view)
    return overhead.to(device), vision.to(device), view.to(device)


def train_codec(codec, data, cfg, device):
    codec.train()
    for param in codec.parameters():
        param.requires_grad_(True)
    opt = torch.optim.AdamW(codec.parameters(), lr=cfg.ae_lr, weight_decay=0.0)
    best_error = float("inf")
    best_score = float("inf")
    best_state = None
    clips = torch.cat((data["drive"], data["reverse"]), 0)
    chunk = 2
    parts = max(clips.shape[0] // chunk, 1)
    for round_index in range(2):
        if round_index == 1:
            if best_error < 0.02:
                break
            for group in opt.param_groups:
                group["lr"] = cfg.ae_lr * 0.1
            print(f"重建 {best_error:.4f}，低学习率再训一轮", flush=True)
        for step in range(1, cfg.ae_steps + 1):
            if round_index == 0 and (step == cfg.ae_steps // 2 or step == 3 * cfg.ae_steps // 4):
                for group in opt.param_groups:
                    group["lr"] *= 0.3
            opt.zero_grad(set_to_none=True)
            color_sum = 0.0
            sharp_sum = 0.0
            for start in range(0, clips.shape[0], chunk):
                batch = clips[start:start + chunk].to(device)
                recon = codec.decode(codec.encode(batch))
                color = (recon - batch).abs().mean()
                residual = (_highpass(recon) - _highpass(batch)).abs()
                sharp = residual.mean()
                strength = _highpass(batch).detach().abs().mean(dim=-3, keepdim=True)
                weight = strength / strength.mean().clamp_min(1e-4)
                detail = (residual * weight).mean()
                line = ((recon - batch).abs() * weight).mean()
                loss = color + 4.0 * sharp + 4.0 * detail + 2.0 * line
                (loss / parts).backward()
                color_sum += float(color.detach())
                sharp_sum += float(sharp.detach())
            opt.step()
            if step == 1 or step % 100 == 0 or step == cfg.ae_steps:
                error, sharp_error = codec_report(codec, data, device)
                codec.train()
                score = error + 2.0 * sharp_error
                if score < best_score:
                    best_score = score
                    best_error = error
                    best_state = {key: value.detach().cpu().clone() for key, value in codec.state_dict().items()}
                print(
                    f"解码器 {step}/{cfg.ae_steps}  批 {color_sum / parts:.4f}  "
                    f"高频 {sharp_sum / parts:.4f}  全量 {error:.4f}  全量高频 {sharp_error:.4f}",
                    flush=True,
                )
    codec.load_state_dict(best_state)
    codec.eval()
    for param in codec.parameters():
        param.requires_grad_(False)
    return best_error


@torch.no_grad()
def latent_stats(codec, data, device):
    pieces = []
    for view_name in ("drive", "reverse"):
        video = data[view_name].to(device)
        for start in range(0, video.shape[0], 2):
            pieces.append(codec.encode(video[start:start + 2]))
    latent = torch.cat(pieces, 0)
    mean = latent.mean(dim=(0, 1, 3, 4), keepdim=True)
    std = latent.std(dim=(0, 1, 3, 4), keepdim=True).clamp_min(1e-3)
    return mean.cpu(), std.cpu()


def _grad_norm(loss, params) -> float:
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    square = 0.0
    for grad in grads:
        if grad is not None:
            square += float(grad.detach().pow(2).sum())
    return square ** 0.5


def _highpass(image: torch.Tensor) -> torch.Tensor:
    flat = image.reshape(-1, image.shape[-3], image.shape[-2], image.shape[-1])
    blur = F.avg_pool2d(flat, kernel_size=5, stride=1, padding=2)
    return image - blur.reshape_as(image)


def _photo_loss(codec, pred, overhead, std, mean):
    """Color, frame difference, and the high-frequency residual.

    The residual is weighted with the frame difference so a smooth image
    cannot satisfy the pixel term. The light codec sees every frame. The
    frozen VAE still decodes two frames so the step fits in 16 GB.
    """
    if isinstance(codec, FrozenVAE):
        index = int(torch.randint(0, pred.shape[1] - 1, ()).item())
        decoded = codec.decode(pred[:, index:index + 2] * std + mean)
        target = overhead[:, index:index + 2]
    else:
        decoded = codec.decode(pred * std + mean)
        target = overhead
    color = (decoded - target).abs().mean()
    motion = ((decoded[:, 1:] - decoded[:, :-1]) - (target[:, 1:] - target[:, :-1])).abs().mean()
    residual = (_highpass(decoded) - _highpass(target)).abs()
    sharp = residual.mean()
    strength = _highpass(target).detach().abs().mean(dim=-3, keepdim=True)
    weight = strength / strength.mean().clamp_min(1e-4)
    detail = (residual * weight).mean()
    return color + 2.0 * motion + 2.0 * sharp + 4.0 * detail, color.detach(), motion.detach(), sharp.detach()


@torch.no_grad()
def codec_report(codec, data, device) -> tuple[float, float]:
    color_total = 0.0
    sharp_total = 0.0
    count = 0
    was_training = codec.training
    codec.eval()
    for name in ("drive", "reverse"):
        video = data[name]
        for index in range(video.shape[0]):
            batch = video[index:index + 1].to(device)
            recon = codec.decode(codec.encode(batch))
            color_total += float((recon - batch).abs().mean())
            sharp_total += float((_highpass(recon) - _highpass(batch)).abs().mean())
            count += 1
    if was_training:
        codec.train()
    scale = max(count, 1)
    return color_total / scale, sharp_total / scale


@torch.no_grad()
def codec_error(codec, data, device) -> float:
    color, _sharp = codec_report(codec, data, device)
    return color


@torch.no_grad()
def _frame(tensor: torch.Tensor) -> Image.Image:
    array = (tensor.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
    return Image.fromarray(array, mode="RGB")


def _paste_nearest(sheet: Image.Image, image: torch.Tensor, x: int, y: int, scale: int) -> None:
    pic = _frame(image)
    if scale != 1:
        pic = pic.resize((pic.width * scale, pic.height * scale), Image.Resampling.NEAREST)
    sheet.paste(pic, (x, y))


@torch.no_grad()
def _nearest_baseline(data, device):
    path = ROOT / "outputs" / "codec_stride4.pt"
    if not path.is_file():
        return None
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if blob.get("arch") not in (None, "ae16"):
        return None
    codec = Stride4Codec(int(blob.get("channels", 16))).to(device).eval()
    codec.load_state_dict(blob["codec"])
    video = data["drive"][:1].to(device)
    return codec.decode(codec.encode(video)).clamp(0, 1)[0, 0].cpu()


@torch.no_grad()
def _save_line_sheet(codec, data, device, path: Path):
    """Scene-0 drive: target, nearest-upsample recon, learned-phase recon."""
    video = data["drive"][:1].to(device)
    fresh = codec.decode(codec.encode(video)).clamp(0, 1)[0, 0].cpu()
    target = video[0, 0].detach().cpu()
    previous = _nearest_baseline(data, device)
    columns = [target, fresh] if previous is None else [target, previous, fresh]
    scale = 2
    crops = (
        (slice(0, 96), slice(0, 256)),
        (slice(24, 168), slice(156, 256)),
        (slice(148, 236), slice(48, 208)),
    )
    column_w = 256 * scale
    crop_h = sum((ys.stop - ys.start) * scale for ys, _ in crops)
    sheet = Image.new("RGB", (column_w * len(columns), column_w + crop_h), (14, 15, 18))
    for index, image in enumerate(columns):
        _paste_nearest(sheet, image, column_w * index, 0, scale)
    top = column_w
    for ys, xs in crops:
        for index, image in enumerate(columns):
            _paste_nearest(sheet, image[:, ys, xs], column_w * index, top, scale)
        top += (ys.stop - ys.start) * scale
    temporary = path.with_suffix(".tmp.png")
    sheet.save(temporary)
    try:
        temporary.replace(path)
    except OSError:
        temporary.replace(path.with_name("codec_lines_new.png"))


@torch.no_grad()
def _save_codec_check(codec, data, device, path: Path):
    video = data["drive"][:1].to(device)
    recon = codec.decode(codec.encode(video)).clamp(0, 1).cpu()
    target = video.cpu()
    frames = target.shape[1]
    sheet = Image.new("RGB", (256 * frames, 256 * 2), (14, 15, 18))
    for row, clip in enumerate((target, recon)):
        for index in range(frames):
            array = (clip[0, index].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
            sheet.paste(Image.fromarray(array, mode="RGB"), (256 * index, 256 * row))
    temporary = path.with_suffix(".tmp.png")
    sheet.save(temporary)
    try:
        temporary.replace(path)
    except OSError:
        temporary.replace(path.with_name("codec_check_new.png"))


@torch.no_grad()
def _seeded_views(codec, model, data, mean, std, device):
    """Same noise order as the demo: drive, reverse, then the gear switch."""
    torch.manual_seed(0)
    vision = data["vision"]
    count, frames = vision.shape[:2]
    shape = (count, frames, model.latent_channels, model.latent_size, model.latent_size)
    noise_drive = torch.randn(shape)
    noise_reverse = torch.randn(shape)
    noise_switch = torch.randn(shape)
    drive, reverse, switch = [], [], []
    for start in range(0, count, 2):
        seen = vision[start:start + 2].to(device)
        stop = start + seen.shape[0]
        drive_id = torch.zeros(seen.shape[0], dtype=torch.long, device=device)
        reverse_id = torch.ones(seen.shape[0], dtype=torch.long, device=device)
        z_drive = sample(model, (seen.shape[0], frames, *shape[2:]), seen, drive_id, noise=noise_drive[start:stop].to(device))
        z_reverse = sample(model, (seen.shape[0], frames, *shape[2:]), seen, reverse_id, noise=noise_reverse[start:stop].to(device))
        z_switch = sample(
            model, (seen.shape[0], frames, *shape[2:]), seen, reverse_id,
            z_prev=z_drive, noise_level=0.8, noise=noise_switch[start:stop].to(device),
        )
        drive.append(codec.decode(z_drive * std + mean).clamp(0, 1))
        reverse.append(codec.decode(z_reverse * std + mean).clamp(0, 1))
        switch.append(codec.decode(z_switch * std + mean).clamp(0, 1))
    return torch.cat(drive, 0), torch.cat(reverse, 0), torch.cat(switch, 0)


def train_refiner(refiner, pred, target, steps, device):
    refiner.train()
    opt = torch.optim.AdamW(refiner.parameters(), lr=1e-3, weight_decay=0.0)
    count = pred.shape[0]
    last = None
    for step in range(1, steps + 1):
        if step == steps // 2 or step == 3 * steps // 4:
            for group in opt.param_groups:
                group["lr"] *= 0.3
        opt.zero_grad(set_to_none=True)
        order = torch.randperm(count)
        color_sum = 0.0
        parts = 0
        for start in range(0, count, 4):
            index = order[start:start + 4]
            guess = refiner(pred[index].to(device))
            truth = target[index].to(device)
            color = (guess - truth).abs().mean()
            sharp = (_highpass(guess) - _highpass(truth)).abs().mean()
            loss = (color + 2.0 * sharp) / max((count + 3) // 4, 1)
            loss.backward()
            color_sum += float(color.detach())
            parts += 1
        torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
        opt.step()
        last = color_sum / max(parts, 1)
        if step == 1 or step % 20 == 0 or step == steps:
            with torch.no_grad():
                total = 0.0
                for start in range(0, count, 4):
                    chunk = pred[start:start + 4].to(device)
                    truth = target[start:start + 4].to(device)
                    total += float((refiner(chunk) - truth).abs().mean()) * chunk.shape[0]
                full = total / count
            refiner.train()
            print(f"锐化 {step}/{steps}  批 {last:.4f}  全量 {full:.4f}", flush=True)
    refiner.eval()
    for param in refiner.parameters():
        param.requires_grad_(False)
    return last


def refine_saved(steps, device):
    ckpt_path = ROOT / "outputs" / "sbzrv.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = Config(**ckpt["config"])
    codec = load_codec(ckpt, cfg, device)
    model = VideoMeanFlow(
        hidden=cfg.hidden, depth=cfg.depth, heads=cfg.heads, frames=cfg.frames,
        latent_channels=cfg.latent_channels, latent_size=cfg.latent_size,
        tubelet=cfg.tubelet, vision_dim=cfg.vision_dim,
    ).to(device).eval()
    model.load_state_dict(ckpt["model"])
    data = build_cache(cfg, device, rebuild=False)
    mean, std = ckpt["mean"].to(device), ckpt["std"].to(device)
    drive, reverse, switch = _seeded_views(codec, model, data, mean, std, device)
    target = torch.cat((data["drive"], data["reverse"]), 0)
    refiner = Refiner().to(device)
    refiner.set_pairs(
        torch.cat((drive, reverse, switch), 0),
        torch.cat((data["drive"].to(device), data["reverse"].to(device), data["reverse"].to(device)), 0),
    )
    with torch.no_grad():
        replay_drive, replay_reverse, replay_switch = _seeded_views(codec, model, data, mean, std, device)
        replay = torch.cat((replay_drive, replay_reverse), 0)
        gap = float((replay - torch.cat((drive, reverse), 0)).abs().mean())
        error = float((refiner(replay) - target.to(replay.device)).abs().mean())
        sharp_drive = refiner(replay_drive).cpu()
        sharp_reverse = refiner(replay_reverse).cpu()
        sharp_switch = refiner(replay_switch).cpu()
    print(f"重放差 {gap:.6f}  锐化后全量 {error:.4f}", flush=True)
    ckpt["refiner"] = refiner.state_dict()
    torch.save(ckpt, ckpt_path)
    print(f"已保存 {ckpt_path}", flush=True)
    surround = data["surround"][:1]
    rows = [
        ("环视", [surround[0, 0, i] for i in range(surround.shape[2])]),
        ("前进 目标", [data["drive"][0, i] for i in range(4)]),
        ("前进 生成", [sharp_drive[0, i] for i in range(4)]),
        ("倒车 目标", [data["reverse"][0, i] for i in range(4)]),
        ("倒车 生成", [sharp_reverse[0, i] for i in range(4)]),
        ("倒车切换", [sharp_switch[0, i] for i in range(4)]),
    ]
    path = ROOT / "outputs" / "preview.png"
    save_preview(path, rows, source=cfg.source)
    print(f"预览 {path}", flush=True)
    return error


def train_flow(model, codec, data, mean, std, cfg, device):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95))
    params = [p for p in model.parameters() if p.requires_grad]
    weight = (1.0, 4.0, 4.0)
    clip = 24.0
    raw_last = None
    for step in range(1, cfg.steps + 1):
        overhead, vision, view = _take(data, cfg, device)
        with torch.no_grad():
            latent = (codec.encode(overhead) - mean) / std
        flow, raw = meanflow_loss(model, latent, vision, view, cfg.data_proportion)
        jump, raw_jump, pred = endpoint_loss(model, latent, vision, view)
        photo, color, motion, sharp = _photo_loss(codec, pred, overhead, std, mean)
        if step == 1 or step % 500 == 0:
            flow_norm = _grad_norm(flow, params)
            jump_norm = _grad_norm(jump, params)
            photo_norm = _grad_norm(photo, params)
            weight = tuple(4.0 / max(norm, 1e-4) for norm in (flow_norm, jump_norm, photo_norm))
            clip = 24.0
            print(
                f"梯度 均值流 {flow_norm:.2f}  一步 {jump_norm:.2f}  像素 {photo_norm:.2f}  "
                f"权重 {weight[0]:.3f} {weight[1]:.3f} {weight[2]:.3f}  裁剪 {clip:.0f}",
                flush=True,
            )
        loss = weight[0] * flow + weight[1] * jump + weight[2] * photo
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(params, clip)
        if step == 1 and float(grad) == 0.0:
            raise RuntimeError("MeanFlow 第一步梯度为 0，JVP 没有接到参数上。")
        opt.step()
        raw_last = float(raw)
        if step == 1 or step % 100 == 0 or step == cfg.steps:
            print(
                f"均值流 {step}/{cfg.steps}  raw {raw_last:.4f}  一步 {float(raw_jump):.4f}  "
                f"像素 {float(color):.4f}  帧差 {float(motion):.4f}  高频 {float(sharp):.4f}  grad {float(grad):.3f}",
                flush=True,
            )
    return raw_last


def main():
    parser = argparse.ArgumentParser(description="Train SanBengZi Reality View")
    parser.add_argument("--scenes", type=int)
    parser.add_argument("--ae-steps", type=int, dest="ae_steps")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--latent-channels", type=int, dest="latent_channels")
    parser.add_argument("--latent-size", type=int, dest="latent_size")
    parser.add_argument("--batch-size", type=int, dest="batch_size")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--codec-only", action="store_true")
    parser.add_argument("--retrain-codec", action="store_true")
    parser.add_argument("--codec-arch", default=None)
    parser.add_argument("--refine-only", action="store_true")
    parser.add_argument("--refine-steps", type=int, default=800)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备 {device}", flush=True)
    if args.refine_only:
        refine_saved(args.refine_steps, device)
        return
    ckpt_path = ROOT / "outputs" / "sbzrv.pt"
    saved = torch.load(ckpt_path, map_location="cpu", weights_only=False) if args.resume else None
    cfg = Config(**saved["config"]) if saved else Config()
    cfg = _cfg_from_args(args) if not saved else cfg
    if saved and args.steps is not None:
        cfg.steps = args.steps
    if saved and args.batch_size is not None:
        cfg.batch_size = args.batch_size
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    data = build_cache(cfg, device, rebuild=args.rebuild_cache)
    save_scene_check(
        {"surround": data["surround"][0], "drive": data["drive"][0], "reverse": data["reverse"][0]},
        ROOT / "outputs" / "scene_check.png",
    )
    arch = saved.get("codec_arch", "ae16") if saved else (args.codec_arch or "ae16")
    if arch == "up":
        codec_path = ROOT / "outputs" / f"codec_up{cfg.latent_channels}_s{cfg.latent_size}.pt"
    else:
        codec_path = ROOT / "outputs" / "codec_stride4.pt"
    if cfg.latent_codec == "vae":
        codec = FrozenVAE().to(device).eval()
        mean, std = latent_stats(codec, data, device)
        mean, std = mean.to(device), std.to(device)
        with torch.no_grad():
            sample = data["drive"][:1, :1].to(device)
            recon = (codec.decode(codec.encode(sample)) - sample).abs().mean()
        print(f"冻结 VAE 重建 {float(recon):.4f}", flush=True)
    elif cfg.latent_codec == "stride4":
        codec = make_codec("stride4", arch, cfg.latent_channels, cfg.latent_size).to(device)
        blob = torch.load(codec_path, map_location="cpu", weights_only=False) if codec_path.is_file() and not args.retrain_codec else None
        if blob is not None and (
            blob.get("channels") != cfg.latent_channels
            or blob.get("latent_size") != cfg.latent_size
            or blob.get("arch") != arch
        ):
            blob = None
        if blob is not None:
            codec.load_state_dict(blob["codec"])
            codec.eval()
            for param in codec.parameters():
                param.requires_grad_(False)
            print(f"沿用 4 倍解码器  重建 {codec_error(codec, data, device):.4f}", flush=True)
            if saved is None:
                mean, std = latent_stats(codec, data, device)
                blob["mean"] = mean
                blob["std"] = std
                torch.save(blob, codec_path)
                print("按当前画面重算潜变量均值和方差", flush=True)
            else:
                mean, std = blob["mean"], blob["std"]
            mean, std = mean.to(device), std.to(device)
        else:
            if arch == "up":
                old_path = ROOT / "outputs" / "codec_stride4.pt"
                if old_path.is_file():
                    old_blob = torch.load(old_path, map_location="cpu", weights_only=False)
                    old = Stride4Codec(int(old_blob.get("channels", 16))).to(device).eval()
                    old.load_state_dict(old_blob["codec"])
                    old_color, old_sharp = codec_report(old, data, device)
                    print(f"最近邻解码器  全量 {old_color:.4f}  全量高频 {old_sharp:.4f}", flush=True)
                    del old, old_blob
            error = codec_error(codec, data, device) if getattr(codec, "exact", False) else train_codec(codec, data, cfg, device)
            mean, std = latent_stats(codec, data, device)
            mean, std = mean.to(device), std.to(device)
            torch.save(
                {
                    "codec": codec.state_dict(),
                    "channels": cfg.latent_channels,
                    "latent_size": cfg.latent_size,
                    "arch": arch,
                    "mean": mean.cpu(),
                    "std": std.cpu(),
                },
                codec_path,
            )
            check = ROOT / "outputs" / (f"codec_up{cfg.latent_channels}_check.png" if arch == "up" else "codec_check.png")
            _save_codec_check(codec, data, device, check)
            if arch == "up":
                _save_line_sheet(codec, data, device, ROOT / "outputs" / f"codec_lines_{cfg.latent_channels}.png")
            print(f"4 倍解码器重建 {error:.4f}", flush=True)
            if args.codec_only:
                print(f"解码器已保存 {codec_path}", flush=True)
                return
            if error >= 0.02:
                raise RuntimeError(f"解码器重建 {error:.4f}，细线过不了这一关，均值流先不开始。")
        if args.codec_only:
            _save_codec_check(codec, data, device, ROOT / "outputs" / "codec_check.png")
            if arch == "up":
                _save_line_sheet(codec, data, device, ROOT / "outputs" / f"codec_lines_{cfg.latent_channels}.png")
            print(f"解码器已保存 {codec_path}", flush=True)
            return
    else:
        codec = _Codec(cfg.latent_channels).to(device)
        if saved:
            codec.load_state_dict(saved["codec"])
            codec.eval()
            for param in codec.parameters():
                param.requires_grad_(False)
            mean, std = saved["mean"].to(device), saved["std"].to(device)
            print("沿用已训练的解码器", flush=True)
        else:
            train_codec(codec, data, cfg, device)
            mean, std = latent_stats(codec, data, device)
            mean, std = mean.to(device), std.to(device)
    model = VideoMeanFlow(
        hidden=cfg.hidden,
        depth=cfg.depth,
        heads=cfg.heads,
        frames=cfg.frames,
        latent_channels=cfg.latent_channels,
        latent_size=cfg.latent_size,
        tubelet=cfg.tubelet,
        vision_dim=cfg.vision_dim,
    ).to(device)
    if saved:
        model.load_state_dict(saved["model"])
    count = sum(p.numel() for p in model.parameters())
    print(f"视频 MeanFlow 参数 {count / 1e6:.2f} M  序列 {cfg.spatial_tokens}", flush=True)
    train_flow(model, codec, data, mean, std, cfg, device)
    ckpt = ROOT / "outputs" / "sbzrv.pt"
    torch.save(
        {
            "codec_kind": cfg.latent_codec,
            "codec_arch": arch,
            "codec": None if cfg.latent_codec == "vae" else codec.state_dict(),
            "model": model.state_dict(),
            "mean": mean.cpu(),
            "std": std.cpu(),
            "config": cfg.__dict__,
        },
        ckpt,
    )
    print(f"已保存 {ckpt}", flush=True)
    render_preview(codec, model, data, mean, std, device, cfg.source)


if __name__ == "__main__":
    main()

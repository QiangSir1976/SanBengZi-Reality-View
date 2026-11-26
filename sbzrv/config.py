"""Laptop design values for SanBengZi Reality View.

These numbers are the scaled prototype that fits a 16 GB laptop GPU.
They are not a claim about Tesla's model.
"""

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Config:
    image_size: int = 256
    frames: int = 4
    cameras: int = 4
    latent_channels: int = 16
    latent_codec: str = "stride4"
    latent_size: int = 64
    tubelet: int = 2
    hidden: int = 384
    depth: int = 8
    heads: int = 6
    vision_dim: int = 768
    vision_tokens: int = 64
    source: str = "av2"
    scenes: int = 8
    batch_size: int = 1
    ae_steps: int = 600
    ae_lr: float = 1e-3
    steps: int = 1200
    lr: float = 1e-4
    data_proportion: float = 0.75
    seed: int = 0
    device: str = "cuda"

    @property
    def spatial_tokens(self) -> int:
        side = self.latent_size // self.tubelet
        return side * side

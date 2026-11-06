"""The four custom model architectures, defined from scratch (not fine-tuned
off-the-shelf backbones) so they're genuinely our own designs. Small enough
to train on CPU in a reasonable time and to run per-frame in real time.

This module imports torch, so it is imported ONLY by the offline training
scripts. The runtime providers never import it: trainers save each model as
a self-contained TorchScript file, and providers load it with
`torch.jit.load`, needing torch but not these class definitions.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BlinkCNN(nn.Module):
    """Eye open/closed classifier — input 1x32x32 grayscale eye crop.

    `width`/`fc`/`dropout` default to the original 16/32, 64, 0.3 design, and
    are constructor args only so the tuning search can vary them without
    editing this file per trial."""

    def __init__(self, width: int = 16, fc: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(width * 2 * 8 * 8, fc), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(fc, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))   # logits (N, 2)


class GazeCNN(nn.Module):
    """Appearance-based gaze estimator — input 1x36x60 grayscale eye patch,
    output (pitch, yaw) gaze direction in radians (MPIIGaze convention)."""

    def __init__(self, width: int = 16, fc: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 4, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(), nn.Linear(width * 4 * 4 * 7, fc), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(fc, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.features(x))     # (N, 2) = (pitch, yaw) rad


class HeadPoseCNN(nn.Module):
    """Head-pose regressor — input 3x64x64 RGB face crop, output
    (yaw, pitch, roll) in degrees."""

    def __init__(self, width: int = 16, fc: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 4, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(), nn.Linear(width * 4 * 8 * 8, fc), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(fc, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.features(x))     # (N, 3) = (yaw, pitch, roll) deg


class VoiceKWSCNN(nn.Module):
    """Keyword-spotting classifier — input 1 x n_mels x T log-mel spectrogram.
    AdaptiveAvgPool makes it length-agnostic, so variable-length utterances
    need no padding/truncation."""

    def __init__(self, n_classes: int, n_mels: int = 40, width: int = 16,
                 fc: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 3, padding=1), nn.BatchNorm2d(width * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 2, width * 4, 3, padding=1), nn.BatchNorm2d(width * 4), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(width * 4, fc), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(fc, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))    # logits (N, n_classes)

"""Custom CNN+MLP feature extractor for the lens polishing environment.

Processes 2-channel 2D maps (error + roughness) through a CNN, concatenates
with scalar features (tool_pos, time_ratio), and feeds into MLP layers.
Compatible with Stable-Baselines3's feature extractor API.
"""

import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class LensPolishingExtractor(BaseFeaturesExtractor):
    """CNN processes (error_map, roughness_map), MLP merges with scalars."""

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        cnn_channels: list[int] = None,
        features_dim: int = 256,
    ):
        super().__init__(observation_space, features_dim=features_dim)

        if cnn_channels is None:
            cnn_channels = [32, 64, 128]

        maps_shape = observation_space["maps"].shape
        in_channels = maps_shape[0]
        h, w = maps_shape[1], maps_shape[2]

        cnn_layers = []
        prev_ch = in_channels
        for ch in cnn_channels:
            cnn_layers.extend([
                nn.Conv2d(prev_ch, ch, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            ])
            prev_ch = ch

        self.cnn = nn.Sequential(*cnn_layers)

        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, h, w)
            cnn_out = self.cnn(dummy)
            cnn_flat_dim = int(cnn_out.numel())

        scalar_dim = observation_space["scalar"].shape[0]

        self.fc = nn.Sequential(
            nn.Linear(cnn_flat_dim + scalar_dim, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        maps = observations["maps"]
        scalar = observations["scalar"]

        cnn_out = self.cnn(maps)
        cnn_flat = cnn_out.reshape(cnn_out.size(0), -1)

        combined = torch.cat([cnn_flat, scalar], dim=1)
        return self.fc(combined)

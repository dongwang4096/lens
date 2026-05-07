"""Tool Influence Function (TIF) model.

Predicts the U-shaped removal function characteristics (width, depth)
from process parameters (pressure, speed, concentration).
Uses a small MLP trained on experimental data.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


class TIFNet(nn.Module):
    """MLP that maps (pressure, speed, concentration) -> (width, depth)."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        self.input_mean = nn.Parameter(torch.zeros(3), requires_grad=False)
        self.input_std = nn.Parameter(torch.ones(3), requires_grad=False)
        self.output_mean = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.output_std = nn.Parameter(torch.ones(2), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - self.input_mean) / (self.input_std + 1e-8)
        y_norm = self.net(x_norm)
        return y_norm * self.output_std + self.output_mean

    def set_normalization(self, X: np.ndarray, Y: np.ndarray):
        self.input_mean.data = torch.tensor(X.mean(axis=0), dtype=torch.float32)
        self.input_std.data = torch.tensor(X.std(axis=0), dtype=torch.float32)
        self.output_mean.data = torch.tensor(Y.mean(axis=0), dtype=torch.float32)
        self.output_std.data = torch.tensor(Y.std(axis=0), dtype=torch.float32)


class TIFModel:
    """High-level TIF predictor wrapping the neural network."""

    def __init__(self, checkpoint_path: str | Path | None = None):
        self.net = TIFNet()
        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.net.load_state_dict(state)
        self.net.eval()

    def predict(self, pressure: float, speed: float, concentration: float) -> tuple[float, float]:
        """Return (removal_width_mm, removal_depth_um_per_s)."""
        x = torch.tensor([[pressure, speed, concentration]], dtype=torch.float32)
        with torch.no_grad():
            y = self.net(x)
        return float(y[0, 0]), float(y[0, 1])

    def predict_batch(self, params: np.ndarray) -> np.ndarray:
        """params: (N, 3), returns (N, 2)."""
        x = torch.tensor(params, dtype=torch.float32)
        with torch.no_grad():
            y = self.net(x)
        return y.numpy()

    def generate_tif_2d(
        self,
        pressure: float,
        speed: float,
        concentration: float,
        grid_coords: np.ndarray,
        center: tuple[float, float],
    ) -> np.ndarray:
        """Generate a 2D Gaussian TIF on the given grid.

        Args:
            pressure, speed, concentration: process parameters
            grid_coords: (H, W) meshgrid of distances not needed — we use
                         explicit x, y meshgrids
            center: (cx, cy) tool center in mm

        Returns:
            tif_2d: (H, W) removal rate in micron/s at each grid point
        """
        width, depth = self.predict(pressure, speed, concentration)
        sigma = width / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        cx, cy = center
        r2 = (grid_coords[0] - cx) ** 2 + (grid_coords[1] - cy) ** 2
        tif = depth * np.exp(-r2 / (2.0 * sigma**2))
        return tif

    def save(self, path: str | Path):
        torch.save(self.net.state_dict(), path)

    def train_from_data(self, X: np.ndarray, Y: np.ndarray, epochs: int = 2000, lr: float = 1e-3):
        """Train the TIF network on experimental data.

        X: (N, 3) pressure, speed, concentration
        Y: (N, 2) removal_width, removal_depth
        """
        self.net.train()
        self.net.set_normalization(X, Y)

        X_t = torch.tensor(X, dtype=torch.float32)
        Y_t = torch.tensor(Y, dtype=torch.float32)

        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        loss_fn = nn.MSELoss()

        best_loss = float("inf")
        for epoch in range(epochs):
            pred = self.net(X_t)
            loss = loss_fn(pred, Y_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            cur_loss = loss.detach().item()
            if cur_loss < best_loss:
                best_loss = cur_loss

            if (epoch + 1) % 500 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

        self.net.eval()
        print(f"  Training complete. Best loss: {best_loss:.6f}")

        with torch.no_grad():
            pred = self.net(X_t).numpy()
            for i, name in enumerate(["width", "depth"]):
                mae = np.mean(np.abs(pred[:, i] - Y[:, i]))
                rel = mae / np.mean(np.abs(Y[:, i])) * 100
                print(f"  {name}: MAE={mae:.4f}, RelError={rel:.1f}%")

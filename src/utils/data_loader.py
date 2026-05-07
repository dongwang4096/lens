"""Load and clean experimental polishing data from xlsx."""

from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path


COLUMN_MAP = {
    "抛光压力(N)": "pressure",
    "转速(r/min)": "speed",
    "抛光液浓度(wt%)": "concentration",
    "球头直径(mm)": "ball_diameter",
    "去除函数范围（mm）": "removal_width",
    "去除函数深度（micron/second）": "removal_depth",
    "去除体积（micron³/second）": "removal_volume",
    "驻留时间（s）": "dwell_time",
}


def load_polishing_data(xlsx_path: str | Path) -> pd.DataFrame:
    """Load all sheets from the xlsx file, clean NaN rows, and return a
    unified DataFrame with standardized English column names.

    Only keeps the 4 input features (pressure, speed, concentration,
    ball_diameter) and 3 output targets (removal_width, removal_depth,
    removal_volume).
    """
    xlsx_path = Path(xlsx_path)
    all_frames = []

    for sheet in ["P", "V", "C", "B", "new"]:
        df = pd.read_excel(xlsx_path, sheet_name=sheet)
        rename = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
        df = df.rename(columns=rename)

        keep_cols = [c for c in COLUMN_MAP.values() if c in df.columns]
        df = df[keep_cols]
        df = df.dropna(subset=["pressure", "speed", "concentration"])
        df["source_sheet"] = sheet
        all_frames.append(df)

    combined = pd.concat(all_frames, ignore_index=True)

    required = ["pressure", "speed", "concentration", "removal_width", "removal_depth"]
    combined = combined.dropna(subset=required)

    return combined


def get_training_arrays(df: pd.DataFrame):
    """Return (X, Y) numpy arrays for TIF model fitting.

    X: (N, 3) — pressure, speed, concentration
    Y: (N, 2) — removal_width (mm), removal_depth (micron/s)
    """
    X = df[["pressure", "speed", "concentration"]].values.astype(np.float32)
    Y = df[["removal_width", "removal_depth"]].values.astype(np.float32)
    return X, Y


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parents[2] / "副本数据汇总正式版.xlsx"
    df = load_polishing_data(data_path)
    print(f"Total samples: {len(df)}")
    print(df.describe())
    X, Y = get_training_arrays(df)
    print(f"\nX shape: {X.shape}, Y shape: {Y.shape}")
    print(f"Pressure range: [{X[:,0].min():.1f}, {X[:,0].max():.1f}] N")
    print(f"Speed range:    [{X[:,1].min():.1f}, {X[:,1].max():.1f}] rpm")
    print(f"Concentration:  [{X[:,2].min():.1f}, {X[:,2].max():.1f}] wt%")
    print(f"Width range:    [{Y[:,0].min():.2f}, {Y[:,0].max():.2f}] mm")
    print(f"Depth range:    [{Y[:,1].min():.4f}, {Y[:,1].max():.4f}] micron/s")

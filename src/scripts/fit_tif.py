"""Fit the TIF model from experimental data and save checkpoint."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.data_loader import load_polishing_data, get_training_arrays
from models.tif_model import TIFModel


def main():
    data_path = Path(__file__).resolve().parents[2] / "副本数据汇总正式版.xlsx"
    print(f"Loading data from {data_path}")
    df = load_polishing_data(data_path)
    X, Y = get_training_arrays(df)
    print(f"Training samples: {X.shape[0]}")
    print(f"  Pressure: [{X[:,0].min():.0f}, {X[:,0].max():.0f}] N")
    print(f"  Speed:    [{X[:,1].min():.0f}, {X[:,1].max():.0f}] rpm")
    print(f"  Conc:     [{X[:,2].min():.0f}, {X[:,2].max():.0f}] wt%")
    print(f"  Width:    [{Y[:,0].min():.2f}, {Y[:,0].max():.2f}] mm")
    print(f"  Depth:    [{Y[:,1].min():.4f}, {Y[:,1].max():.4f}] um/s")

    print("\nTraining TIF model...")
    model = TIFModel()
    model.train_from_data(X, Y, epochs=3000, lr=1e-3)

    ckpt_dir = Path(__file__).resolve().parents[1] / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt_path = ckpt_dir / "tif_model.pt"
    model.save(ckpt_path)
    print(f"\nModel saved to {ckpt_path}")

    print("\nSample predictions:")
    test_cases = [
        (10, 300, 5),
        (30, 800, 5),
        (50, 1000, 2),
        (25, 600, 7),
    ]
    for p, v, c in test_cases:
        w, d = model.predict(p, v, c)
        print(f"  P={p:2d}N, V={v:4d}rpm, C={c}wt% -> W={w:.2f}mm, D={d:.4f}um/s")


if __name__ == "__main__":
    main()

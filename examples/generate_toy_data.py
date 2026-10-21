"""Generate a small smoke-test dataset, not a manuscript simulation replicate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args(argv)
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        parser.error(f"Refusing to overwrite {output}")
    rng = np.random.default_rng(args.seed)
    y = np.repeat(np.arange(3), 30)
    centers = np.array([[2.5, 0.0], [-1.25, 2.165], [-1.25, -2.165]])
    signal = centers[y] + rng.normal(0, 0.7, size=(len(y), 2))
    x = np.column_stack([signal, rng.normal(size=(len(y), 10))])
    x = ((x - x.mean(0)) / x.std(0, ddof=1)).astype(np.float32)
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "X.npy", x, allow_pickle=False)
    np.save(output / "y.npy", y, allow_pickle=False)
    (output / "description.json").write_text(json.dumps({
        "purpose": "smoke test only; not a manuscript simulation replicate",
        "seed": args.seed, "n": len(y), "p": x.shape[1], "K": 3,
        "standardization": "sample feature mean and sample SD, ddof=1",
    }, indent=2), encoding="utf-8")
    print(f"Toy data saved to {output}")


if __name__ == "__main__":
    main()

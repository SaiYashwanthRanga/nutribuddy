#!/usr/bin/env python
"""Command-line entry point for training the RadioML CNN classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:  # pragma: no cover - import guard for optional dependency
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover - executed when torch missing
    raise ModuleNotFoundError(
        "PyTorch is required for training. Install it via `pip install 'rml-cnn[torch]'`."
    ) from exc

from rml_cnn.data import (
    download_rml2016_dataset,
    load_rml2016_dataset,
    prepare_rml2016_datasets,
)
from rml_cnn.model import RMLCNN
from rml_cnn.train import train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download the dataset if it is missing.",
    )
    parser.add_argument(
        "--dataset-url",
        type=str,
        default=None,
        help="Optional custom URL for downloading the dataset.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit the number of samples used for quick experiments.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("artifacts/rml_cnn.pt"),
        help="Path where the trained model weights will be stored.",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=Path("artifacts/metrics.json"),
        help="Where to save the training history and test metrics as JSON.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars to keep logs clean.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_path = args.data_dir / "RML2016.10a_dict.pkl"
    if not dataset_path.exists():
        if args.download or args.dataset_url:
            mirrors = [args.dataset_url] if args.dataset_url else None
            download_rml2016_dataset(args.data_dir, mirrors=mirrors)
        else:
            raise FileNotFoundError(
                "Dataset not found. Provide --download or --dataset-url to fetch it."
            )

    raw_dataset = load_rml2016_dataset(dataset_path)
    splits, encoder = prepare_rml2016_datasets(
        raw_dataset, max_samples=args.max_samples
    )

    num_classes = len(encoder.classes_)
    model = RMLCNN(input_channels=splits["train"][0].shape[1], num_classes=num_classes)

    model, history, test_metrics = train_model(
        model,
        splits,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        progress=not args.no_progress,
    )

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": encoder.classes_.tolist(),
            "normalization": {
                "mean": float(np.mean(splits["train"][0])),
                "std": float(np.std(splits["train"][0])),
            },
        },
        args.model_out,
    )

    history_payload: dict[str, Any] = {
        "history": [metric.__dict__ for metric in history],
        "test_loss": test_metrics[0],
        "test_accuracy": test_metrics[1],
    }
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.write_text(json.dumps(history_payload, indent=2))

    print(f"Test loss: {test_metrics[0]:.4f} | Test accuracy: {test_metrics[1]:.4f}")


if __name__ == "__main__":
    main()

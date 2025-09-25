"""Training utilities for the RadioML CNN classifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class NumpyDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """A thin wrapper that turns NumPy arrays into a :class:`Dataset`."""

    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        if features.shape[0] != labels.shape[0]:
            raise ValueError("Features and labels must have matching lengths")
        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.features.shape[0]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float


@torch.no_grad()
def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_features, batch_labels in data_loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)
        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)
        total_loss += loss.item() * batch_features.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == batch_labels).sum().item()
        total += batch_features.size(0)

    return total_loss / total, correct / total


def train_model(
    model: nn.Module,
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
    *,
    epochs: int = 20,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device | None = None,
    progress: bool = True,
) -> Tuple[nn.Module, Iterable[EpochMetrics], Tuple[float, float]]:
    """Train ``model`` using the provided dataset splits."""

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = NumpyDataset(*datasets["train"])
    val_ds = NumpyDataset(*datasets["val"])
    test_ds = NumpyDataset(*datasets["test"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    history: list[EpochMetrics] = []
    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        data_iter = train_loader
        if progress:
            data_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for batch_features, batch_labels in data_iter:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_features.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_labels).sum().item()
            total += batch_features.size(0)

        train_loss = epoch_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_loader, device)

        history.append(
            EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_acc,
                val_loss=val_loss,
                val_accuracy=val_acc,
            )
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, device)
    return model, history, test_metrics

"""Utilities for downloading and preparing the RadioML 2016.10a dataset."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Tuple

import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

RML2016_DATA_URLS: Tuple[str, ...] = (
    # Public mirrors that host the pre-processed dictionary version of the dataset.
    # These links may occasionally change; the downloader will iterate through
    # them until one succeeds.
    "https://huggingface.co/datasets/RadioML/RadioML2016.10a/resolve/main/RML2016.10a_dict.pkl",
    "https://zenodo.org/record/2680980/files/RML2016.10a_dict.pkl?download=1",
    "https://storage.googleapis.com/radio_ml_data_public/RML2016.10a_dict.pkl",
)


class DownloadError(RuntimeError):
    """Raised when the dataset cannot be downloaded from any known mirror."""


def download_file(url: str, target_path: Path, chunk_size: int = 8 * 1024 * 1024) -> None:
    """Download a file from ``url`` to ``target_path`` with a progress bar."""

    target_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        progress = tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=f"Downloading {target_path.name}",
        )
        with target_path.open("wb") as file_obj:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                file_obj.write(chunk)
                progress.update(len(chunk))
        progress.close()


def download_rml2016_dataset(
    target_dir: Path | str,
    *,
    mirrors: Iterable[str] | None = None,
    filename: str = "RML2016.10a_dict.pkl",
    overwrite: bool = False,
) -> Path:
    """Download the RadioML 2016.10a dataset.

    Parameters
    ----------
    target_dir:
        Directory where the dataset file should be stored.
    mirrors:
        Optional iterable of candidate download URLs. When ``None`` the default
        :data:`RML2016_DATA_URLS` is used.
    filename:
        Output filename for the dataset file.
    overwrite:
        Whether to re-download the file if it already exists.

    Returns
    -------
    Path
        The path to the downloaded dataset.
    """

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = target_dir / filename

    if dataset_path.exists() and not overwrite:
        return dataset_path

    mirrors = tuple(mirrors or RML2016_DATA_URLS)
    errors = {}
    for url in mirrors:
        try:
            download_file(url, dataset_path)
            return dataset_path
        except Exception as exc:  # pragma: no cover - error paths are situational
            errors[url] = exc
            if dataset_path.exists():
                dataset_path.unlink()

    error_messages = "\n".join(f"{url}: {err}" for url, err in errors.items())
    raise DownloadError(
        "Unable to download RML2016.10a dataset from known mirrors.\n" + error_messages
    )


def load_rml2016_dataset(dataset_path: Path | str) -> MutableMapping[Tuple[str, int], np.ndarray]:
    """Load the RadioML 2016.10a dataset from the provided pickle file."""

    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    with dataset_path.open("rb") as file_obj:
        data = pickle.load(file_obj, encoding="latin1")

    if not isinstance(data, MutableMapping):
        raise TypeError("Unexpected dataset format")

    return data


def prepare_rml2016_datasets(
    data: Mapping[Tuple[str, int], np.ndarray],
    *,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    stratify: bool = True,
    max_samples: int | None = None,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], LabelEncoder]:
    """Convert the RadioML dictionary structure into train/validation/test sets."""

    features: list[np.ndarray] = []
    labels: list[str] = []
    collected = 0

    for (modulation, _snr), samples in data.items():
        samples = np.asarray(samples)
        if samples.ndim != 3:
            raise ValueError("Expected samples with shape (n, 2, 128)")

        if max_samples is not None:
            remaining = max_samples - collected
            if remaining <= 0:
                break
            samples = samples[:remaining]

        features.append(samples)
        labels.extend([modulation] * len(samples))
        collected += len(samples)

    if not features:
        raise ValueError("No samples were collected from the dataset")

    X = np.concatenate(features, axis=0).astype(np.float32)
    y = np.asarray(labels)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    stratify_labels = y_encoded if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_labels,
    )

    val_ratio = val_size / (1 - test_size)
    stratify_train = y_train if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_ratio,
        random_state=random_state,
        stratify=stratify_train,
    )

    splits = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }

    return splits, encoder

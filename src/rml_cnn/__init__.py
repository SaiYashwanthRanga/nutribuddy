"""RML2016.10a classification helpers."""

from .data import download_rml2016_dataset, load_rml2016_dataset, prepare_rml2016_datasets

__all__ = [
    "download_rml2016_dataset",
    "load_rml2016_dataset",
    "prepare_rml2016_datasets",
]

try:  # pragma: no cover - exercised when PyTorch is installed
    from .model import RMLCNN
    from .train import train_model

    __all__.extend(["RMLCNN", "train_model"])
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    if exc.name != "torch":
        raise

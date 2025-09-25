import numpy as np

from rml_cnn.data import prepare_rml2016_datasets


def make_fake_dataset(num_classes: int = 4, samples_per_class: int = 64):
    data = {}
    for idx in range(num_classes):
        modulation = f"MOD_{idx}"
        snr = 0
        # Create deterministic but unique waveforms per class.
        base = np.linspace(0, 1, 128, dtype=np.float32)
        iq = np.stack([
            np.sin(base * (idx + 1)),
            np.cos(base * (idx + 1)),
        ])
        samples = np.stack([
            iq + 0.01 * np.random.randn(2, 128).astype(np.float32)
            for _ in range(samples_per_class)
        ])
        data[(modulation, snr)] = samples
    return data


def test_prepare_rml2016_datasets_splits_data_correctly():
    raw = make_fake_dataset()
    splits, encoder = prepare_rml2016_datasets(raw, test_size=0.25, val_size=0.1)

    total_samples = sum(split[0].shape[0] for split in splits.values())
    assert total_samples == 4 * 64
    assert splits["train"][0].shape[1:] == (2, 128)
    assert len(encoder.classes_) == 4


def test_prepare_rml2016_datasets_respects_max_samples():
    raw = make_fake_dataset(samples_per_class=10)
    splits, _ = prepare_rml2016_datasets(raw, max_samples=20)
    total_samples = sum(split[0].shape[0] for split in splits.values())
    assert total_samples == 20

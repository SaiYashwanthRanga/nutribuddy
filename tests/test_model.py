import numpy as np
import pytest

torch = pytest.importorskip("torch")
from rml_cnn.model import RMLCNN


def test_rmlcnn_forward_pass():
    batch_size = 8
    channels = 2
    length = 128
    num_classes = 5
    model = RMLCNN(input_channels=channels, num_classes=num_classes)
    dummy_input = torch.from_numpy(np.random.randn(batch_size, channels, length).astype(np.float32))
    logits = model(dummy_input)
    assert logits.shape == (batch_size, num_classes)

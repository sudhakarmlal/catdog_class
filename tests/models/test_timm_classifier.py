import pytest
import torch

import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.models.timm_classifier import TimmClassifier


def test_timm_classifier_forward():
    model = TimmClassifier(base_model="resnet18", num_classes=2)
    batch_size, channels, height, width = 4, 3, 224, 224
    x = torch.randn(batch_size, channels, height, width)
    output = model(x)
    assert output.shape == (batch_size, 2)

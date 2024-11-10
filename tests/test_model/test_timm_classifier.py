import pytest
import torch
import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.models.timm_classifier import TimmClassifier

@pytest.fixture
def model():
    return TimmClassifier(num_classes=10)

@pytest.fixture
def sample_batch():
    # Create a sample batch of 4 images with 3 channels and 224x224 size
    images = torch.randn(4, 3, 224, 224)
    labels = torch.randint(0, 10, (4,))
    return images, labels

def test_model_forward(model, sample_batch):
    images, _ = sample_batch
    output = model(images)
    assert output.shape == (4, 10)

def test_model_training_step(model, sample_batch):
    loss = model.training_step(sample_batch, 0)
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == ()

def test_model_validation_step(model, sample_batch):
    model.validation_step(sample_batch, 0)
    assert model.val_loss.compute() > 0
    assert 0 <= model.val_acc.compute() <= 1

def test_model_test_step(model, sample_batch):
    model.test_step(sample_batch, 0)
    assert model.test_loss.compute() > 0
    assert 0 <= model.test_acc.compute() <= 1

import pytest
from pathlib import Path
import sys
import torch
from omegaconf import OmegaConf
import hydra
import matplotlib.pyplot as plt

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.infer import inference, save_prediction, denormalize, main

# Mock the rootutils setup
sys.modules['rootutils'] = type('MockRootutils', (), {'setup_root': lambda *args, **kwargs: None})()

@pytest.fixture
def config():
    with hydra.initialize(version_base="1.3", config_path="../../configs"):
        cfg = hydra.compose(config_name="infer")
    return cfg

@pytest.fixture
def mock_model(mocker):
    model = mocker.Mock()
    model.eval.return_value = None
    model.device = torch.device("cpu")
    model.return_value = torch.randn(1, 10)
    return model

@pytest.fixture
def mock_datamodule(mocker):
    datamodule = mocker.Mock()
    datamodule.setup.return_value = None
    datamodule.test_dataset = [(torch.randn(3, 224, 224), 0) for _ in range(10)]
    return datamodule

def test_denormalize():
    tensor = torch.randn(3, 224, 224)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    denormalized = denormalize(tensor, mean, std)
    assert denormalized.shape == (3, 224, 224)
    assert denormalized.min() >= 0 and denormalized.max() <= 1

def test_inference(mock_model):
    img = torch.randn(1, 3, 224, 224)
    mock_model.return_value = torch.randn(1, 10)  # Ensure the output matches the number of classes
    label, confidence = inference(mock_model, img)
    assert isinstance(label, str)
    assert isinstance(confidence, float)
    assert 0 <= confidence <= 1

def test_save_prediction(tmp_path):
    img = torch.randn(3, 224, 224)
    actual_label = "Beagle"
    predicted_label = "Boxer"
    confidence = 0.8
    output_path = str(tmp_path / "test_prediction.png")
    
    save_prediction(img, actual_label, predicted_label, confidence, output_path)
    assert Path(output_path).exists()

@pytest.mark.parametrize("num_samples", [1, 5, 10])
def test_main(config, mock_model, mock_datamodule, mocker, tmp_path, num_samples):
    # Mock hydra.utils.instantiate
    mocker.patch("hydra.utils.instantiate", side_effect=[mock_model, mock_datamodule])
    
    # Mock model loading
    mocker.patch.object(type(mock_model), "load_from_checkpoint", return_value=mock_model)
    
    # Mock matplotlib.pyplot.savefig to avoid actually saving files
    mocker.patch.object(plt, 'savefig')
    
    # Mock the logger
    mocker.patch("logging.getLogger")
    
    # Update config for testing
    config.num_samples = num_samples
    config.paths.root_dir = str(tmp_path)
    config.ckpt_path = str(tmp_path / "mock_checkpoint.ckpt")
    
    # Ensure all necessary config keys exist
    if 'model' not in config:
        config.model = OmegaConf.create({})
    if 'data' not in config:
        config.data = OmegaConf.create({})
    if 'trainer' not in config:
        config.trainer = OmegaConf.create({})
    
    # Run main function
    main(config)
    
    # Check if the correct number of predictions were "saved"
    assert plt.savefig.call_count == num_samples

if __name__ == "__main__":
    pytest.main([__file__])

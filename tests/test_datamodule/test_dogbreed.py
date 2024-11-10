import pytest
from pathlib import Path
import os

# Setup root directory
root = Path("/app")  # This matches the WORKDIR in the Dockerfile

from src.datamodules.dogbreed import DogImageDataModule

@pytest.fixture
def datamodule():
    # Use the data_dir from catdog.yaml, but with the Docker path
    data_dir = Path("/app/data/dogbreed")
    return DogImageDataModule(
        data_dir=str(data_dir),
        batch_size=32,
        num_workers=0,  # Set to 0 for debugging
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        pin_memory=False  # Set to False for debugging
    )

def test_dogbreed_datamodule_init(datamodule):
    assert isinstance(datamodule, DogImageDataModule)
    assert str(datamodule.data_dir) == "/app/data/dogbreed"
    assert datamodule.batch_size == 32
    assert datamodule.num_workers == 0
    assert datamodule.train_split == 0.7
    assert datamodule.val_split == 0.15
    assert datamodule.test_split == 0.15
    assert datamodule.pin_memory == False

def test_dogbreed_datamodule_setup(datamodule):
    # Check if the data directory exists
    assert os.path.exists(datamodule.data_dir), f"Data directory {datamodule.data_dir} does not exist"
    
    datamodule.setup(stage="fit")
    assert datamodule.train_dataset is not None
    assert datamodule.val_dataset is not None
    
    datamodule.setup(stage="test")
    assert datamodule.test_dataset is not None

def test_dogbreed_datamodule_dataloaders(datamodule):
    datamodule.setup(stage="fit")
    datamodule.setup(stage="test")

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None

    try:
        batch = next(iter(train_loader))
        assert len(batch) == 2  # (images, labels)
        assert batch[0].shape[1:] == (3, 224, 224)  # (batch_size, channels, height, width)
    except Exception as e:
        pytest.fail(f"Failed to get batch from train_loader: {str(e)}")

if __name__ == "__main__":
    pytest.main([__file__])

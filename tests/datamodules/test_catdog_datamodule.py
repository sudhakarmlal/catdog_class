import pytest

import rootutils

# Setup root directory
root = rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.datamodules.catdog_datamodule import CatDogImageDataModule


@pytest.fixture
def datamodule():
    return CatDogImageDataModule(data_dir="data", batch_size=8)


def test_catdog_datamodule_setup(datamodule):
    datamodule.prepare_data()
    datamodule.setup()

    assert datamodule.train_dataset is not None
    assert datamodule.val_dataset is not None
    assert datamodule.test_dataset is not None

    total_size = (
        len(datamodule.train_dataset)
        + len(datamodule.val_dataset)
        + len(datamodule.test_dataset)
    )
    assert total_size == len(datamodule._dataset)


def test_catdog_datamodule_train_val_test_splits(datamodule):
    datamodule.prepare_data()
    datamodule.setup()

    assert len(datamodule.train_dataset) > len(datamodule.val_dataset)
    assert len(datamodule.train_dataset) > len(datamodule.test_dataset)


def test_catdog_datamodule_dataloaders(datamodule):
    datamodule.prepare_data()
    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None

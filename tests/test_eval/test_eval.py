import pytest
import hydra
from pathlib import Path
import sys
from omegaconf import OmegaConf
import torch

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import main function from eval.py
from src.eval import main

@pytest.fixture
def config():
    with hydra.initialize(version_base=None, config_path="../../configs"):
        cfg = hydra.compose(config_name="eval")
    return cfg

def create_dummy_checkpoint(path):
    dummy_state = {
        "state_dict": {"dummy_key": torch.tensor([1.0, 2.0, 3.0])},
        "epoch": 1,
        "global_step": 100
    }
    torch.save(dummy_state, path)

def test_eval_script(config, tmp_path):
    try:
        # Update paths for testing
        config.paths.output_dir = str(tmp_path / "output")
        config.paths.log_dir = str(tmp_path / "logs")
        config.ckpt_path = str(tmp_path / "best_model.ckpt")

        # Create a dummy checkpoint file
        create_dummy_checkpoint(config.ckpt_path)

        # Instead, just call main once
        main(config)

        # Verify output directory exists and contains files
        output_dir = Path(config.paths.output_dir)
        assert output_dir.exists()
        assert len(list(output_dir.glob('*'))) > 0

        # Verify log directory exists
        assert Path(config.paths.log_dir).exists()

    except Exception as e:
        pytest.fail(f"An error occurred during evaluation: {str(e)}\nConfig: {OmegaConf.to_yaml(config)}")

if __name__ == "__main__":
    pytest.main([__file__])

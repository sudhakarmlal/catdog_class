import logging
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Imports that require root directory setup
from src.utils.logging_utils import setup_logger, task_wrapper

log = logging.getLogger(__name__)

@task_wrapper
@hydra.main(version_base="1.3", config_path="../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    """Evaluation function using Hydra configuration."""
    
    # Instantiate datamodule
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.data)
    
    # Instantiate model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model)
    
    # Load model from checkpoint
    log.info(f"Loading model from checkpoint: {cfg.ckpt_path}")
    model = type(model).load_from_checkpoint(cfg.ckpt_path)
    
    # Instantiate trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer)
    
    # Set up the data module for test data
    datamodule.setup(stage="test")
    
    # Evaluate the model
    log.info("Starting evaluation!")
    results = trainer.test(model=model, datamodule=datamodule)
    
    log.info("Evaluation completed!")
    log.info(f"Evaluation results: {results}")

if __name__ == "__main__":
    main()

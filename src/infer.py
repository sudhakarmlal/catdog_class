import random
from typing import Tuple
from pathlib import Path
import logging
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Imports that require root directory setup
from src.utils.logging_utils import setup_logger, task_wrapper

log = logging.getLogger(__name__)


# Define class labels
CLASS_LABELS = [
    "Cat",
    "Dog",
]


def denormalize(tensor, mean, std):
    # Ensure tensor is on CPU and in the correct shape (C, H, W)
    tensor = tensor.cpu()
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() == 3 and tensor.shape[0] != 3:
        tensor = tensor.permute(2, 0, 1)

    # Reshape mean and std to (C, 1, 1) for broadcasting
    mean = torch.tensor(mean, dtype=tensor.dtype).view(-1, 1, 1)
    std = torch.tensor(std, dtype=tensor.dtype).view(-1, 1, 1)

    # Apply denormalization
    return (tensor * std + mean).clamp(0, 1)


def inference(model: pl.LightningModule, img: torch.Tensor) -> Tuple[str, float]:
    """
    Perform inference on a given image using a trained model.

    Args:
        model (pl.LightningModule): Trained PyTorch Lightning model.
        img (torch.Tensor): Input image tensor.

    Returns:
        Tuple[str, float]: predicted label, and confidence.
    """

    # Set the model in evaluation mode
    model.eval()

    # Perform inference
    with torch.no_grad():
        output = model(img)
        probability = F.softmax(output, dim=1)
        predicted = torch.argmax(probability, dim=1).item()

    predicted_label = CLASS_LABELS[predicted]
    confidence = probability[0][predicted].item()

    return predicted_label, confidence


def save_prediction(
    img: torch.Tensor,
    actual_label: str,
    predicted_label: str,
    confidence: float,
    output_path: str,
):
    """
    Save an image with actual and predicted labels, along with confidence.

    Args:
        img (torch.Tensor): The image tensor to be displayed and saved.
        actual_label (str): The ground truth label of the image.
        predicted_label (str): The label predicted by the model.
        confidence (float): The confidence score of the prediction.
        output_path (str): The path where the image with annotations will be saved.
    """

    # Denormalize the image
    img = denormalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # Convert the tensor to numpy array
    # From (C, H, W) to (H, W, C)
    img = img.permute(1, 2, 0).numpy()

    plt.figure(figsize=(9, 9))
    plt.imshow(img)
    plt.axis("off")
    plt.title(
        f"Actual: {actual_label} | Predicted: {predicted_label} | (Confidence: {confidence:.2f})"
    )
    plt.savefig(output_path)
    plt.close()


@hydra.main(version_base="1.3", config_path="../configs", config_name="infer")
def main(cfg: DictConfig) -> None:
    """Main function for inference using Hydra configuration."""
    
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)
    
    log.info(f"Loading model from checkpoint: {cfg.ckpt_path}")
    # Change this line
    model = type(model).load_from_checkpoint(cfg.ckpt_path)
    # model.to(cfg.trainer.accelerator)
    
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.data)
    
    # Set up the data module for validation data
    datamodule.setup(stage="test")
    test_dataset = datamodule.test_dataset
    
    # Create output directory
    output_folder = Path(cfg.paths.root_dir) / "predictions"
    output_folder.mkdir(exist_ok=True)
    
    # Get the indices for sampling
    num_samples = min(cfg.num_samples, len(test_dataset))
    sampled_indices = random.sample(range(len(test_dataset)), num_samples)
    
    for idx in sampled_indices:
        img, label_index = test_dataset[idx]
        img_tensor = img.unsqueeze(0).to(model.device)
        
        actual_label = CLASS_LABELS[label_index]

        predicted_label, confidence = inference(model, img_tensor)
        print(actual_label, predicted_label)
        
        output_image_path = output_folder / f"sample_{idx}_prediction.png"
        
        save_prediction(img, actual_label, predicted_label, confidence, str(output_image_path))
        
    log.info(f"Predictions saved in {output_folder}")

if __name__ == "__main__":
    main()

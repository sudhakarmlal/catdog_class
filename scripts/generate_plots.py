import os
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import torch
import hydra
from omegaconf import DictConfig
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Get the project root directory
PROJECT_ROOT = rootutils.find_root(__file__, indicator=".project-root")

def generate_confusion_matrix(model, dataloader, class_names, dataset_name):
    model.eval()
    all_preds = []
    all_labels = []
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch
            images = images.to(device).float() 
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {dataset_name} Dataset')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(PROJECT_ROOT, f'{dataset_name.lower()}_confusion_matrix.png'))
    plt.close()

def create_plot(df, x_col, y_cols, title, ylabel, filename):
    plt.figure(figsize=(10, 6))
    for col in y_cols:
        if col in df.columns:
            plt.plot(df[x_col], df[col], label=col)
    plt.xlabel(x_col)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(PROJECT_ROOT, filename))
    plt.close()

@hydra.main(version_base="1.3", config_path="../configs", config_name="eval")
def main(cfg: DictConfig):
    # Find the most recent metrics.csv file
    csv_files = glob(os.path.join(cfg.paths.log_dir, "train/runs/*/csv/version_*/metrics.csv"))
    if not csv_files:
        raise FileNotFoundError("No metrics.csv file found")
    latest_csv = max(csv_files, key=os.path.getctime)

    print(f"Using metrics file: {latest_csv}")

    # Read the CSV file
    df = pd.read_csv(latest_csv)

    # Print column names for debugging
    print("Columns in the CSV file:", df.columns.tolist())

    # Remove rows with all NaN values
    df = df.dropna(how='all')

    # Group by step and take the first non-NaN value for each column
    df = df.groupby('step').first().reset_index()

    # Sort by step
    df = df.sort_values('step')

    # Fill NaN values with the previous non-NaN value
    df = df.ffill()

    # Create loss plot
    create_plot(df, "step", ["train/loss", "val/loss"], "Training and Validation Loss", "Loss", "loss_plot.png")

    # Create accuracy plot
    create_plot(df, "step", ["train/acc", "val/acc"], "Training and Validation Accuracy", "Accuracy", "accuracy_plot.png")

    # Generate test metrics table
    test_metrics = df[df['test/acc'].notna()].iloc[-1]
    test_table = "| Metric | Value |\n|--------|-------|\n"
    for metric in ['test/acc', 'test/loss']:
        if metric in test_metrics:
            test_table += f"| {metric} | {test_metrics[metric]:.4f} |\n"
        else:
            print(f"Warning: {metric} not found in the CSV file")

    print("\nTest metrics:")
    print(test_table)

    # Write the test metrics table to a file in the project root directory
    with open(os.path.join(PROJECT_ROOT, "test_metrics.md"), "w") as f:
        f.write(test_table)

    # Instantiate the data module
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("test")

    # Load the trained model
    model_class = hydra.utils.get_class(cfg.model._target_)
    model = model_class.load_from_checkpoint(cfg.ckpt_path)
    model.eval()

    # Move the model to the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Generate confusion matrices
    generate_confusion_matrix(model, datamodule.train_dataloader(), datamodule.class_names, "Train")
    generate_confusion_matrix(model, datamodule.test_dataloader(), datamodule.class_names, "Test")

    print("Plots and test metrics table generated successfully.")


if __name__ == "__main__":
    main()

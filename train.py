#!/usr/bin/env python
import yaml
import shutil
import argparse
from datetime import datetime
from pytz import timezone
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision.transforms import RandomCrop
from torcheval.metrics import MulticlassAccuracy
from omegaconf import DictConfig, OmegaConf
from hydra import initialize, compose
from hydra.utils import instantiate
import mlflow
from dl4longcbc.dataset import MyDataset, load_dataset
from dl4longcbc.net import instantiate_neuralnetwork


# ----------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------
def train(config: DictConfig):
    now_datetime = datetime.now(timezone('Asia/Tokyo')).strftime('%Y%m%d_%H%M%S')
    mlflow.start_run(run_name=now_datetime)
    run = mlflow.get_run(mlflow.active_run().info.run_id)
    artifact_directory = run.info.artifact_uri.replace("file:", "")
    # Save the configuration as a JSON file
    with open(f"{artifact_directory}/config_train.yaml", "w") as f:
        yaml.dump(OmegaConf.to_container(config), f)
    # Log the configuration file as an artifact
    mlflow.log_artifact(f"{artifact_directory}/config_train.yaml", "config")

    # Set device
    print(f'Is gpu available? {torch.cuda.is_available()}')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Make dataloader
    transforms = nn.Sequential(
        RandomCrop((128, 128))
    )
    inputs, labels = load_dataset(f'{config.dataset.datadir}/train/', ['noise', 'cbc'], 10000, (128, 192))
    tensor_dataset_tr = MyDataset(inputs, labels, transforms)
    dataloader_tr = DataLoader(tensor_dataset_tr, batch_size=config.train.batchsize, shuffle=True, drop_last=True, num_workers=4)
    inputs, labels = load_dataset(f'{config.dataset.datadir}/validate/', ['noise', 'cbc'], 1000, (128, 192))
    tensor_dataset_val = MyDataset(inputs, labels, transforms)
    dataloader_val = DataLoader(tensor_dataset_val, batch_size=config.train.batchsize, shuffle=True, drop_last=True, num_workers=4)

    # Create model
    net = instantiate_neuralnetwork(config)
    mlflow.log_text(str(summary(net)), "models/summary.txt")
    net = net.to(device)
    # Define loss function and optimizer
    criterion = instantiate(config.train.loss)
    optimizer = instantiate(config.train.optimizer, net.parameters())
    metric = MulticlassAccuracy(num_classes=2)
    # Train model
    for epoch in range(config.train.num_epochs):
        net.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader_tr):
            inputs, labels = inputs.to(device), labels.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        running_loss /= (i + 1)
        mlflow.log_metric("train_loss", running_loss, step=epoch + 1)

        # Evaluate model
        valloss = 0
        net.eval()
        with torch.no_grad():
            for j, data in enumerate(dataloader_val):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                valloss += criterion(outputs, labels).item()
                metric.update(outputs, labels)
        valloss /= (j + 1)
        accuracy = metric.compute().item()
        print(f"[Epoch {epoch+1}] {running_loss:.5f} {valloss:.5f} {accuracy:.2f}")

        # Log metrics to mlflow
        mlflow.log_metric("validation_loss", valloss, step=epoch + 1)
        mlflow.log_metric('accuracy', accuracy, step=epoch + 1)
        # Reset metric
        metric.reset()
    # Save the trained model
    mlflow.pytorch.log_model(net, "models")
    # End the mlflow run
    mlflow.end_run()


# ----------------------------------------------------------------
# Main function
# ----------------------------------------------------------------
def main(experiment_name):
    # Initialize Hydra
    initialize(version_base=None, config_path="./config", job_name=experiment_name)
    # Load config
    cfg = compose(config_name="config_train.yaml")
    # Set experiment
    mlflow.set_tracking_uri("./data/mlruns")
    mlflow.set_experiment(experiment_name)
    # Copy the config file to mlruns directory
    shutil.copy("./config/config_train.yaml", f"./data/mlruns/{mlflow.get_experiment_by_name(experiment_name).experiment_id}")
    train(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parse args")
    args = parser.parse_args()

    with open("config/config_train.yaml", "r") as file:
        config = yaml.safe_load(file)
    main(config["train"]["experiment_name"])

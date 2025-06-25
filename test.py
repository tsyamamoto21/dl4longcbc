#!/usr/bin/env python
import argparse
import datetime
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
from torchvision.transforms import RandomCrop
from hydra import initialize
import mlflow
from dl4longcbc.dataset import load_dataset, MyDataset


class TestResult(torch.nn.Module):
    def __init__(self, result_dict):
        super(TestResult, self).__init__()
        self.label = nn.Parameter(result_dict["label"], requires_grad=False)
        self.output = nn.Parameter(result_dict["output"], requires_grad=False)


# >>> test loop >>>
def main(args):
    # Initialize Hydra
    initialize(version_base=None, config_path="./config", job_name=args.experiment_name)
    # Load config
    # cfg = compose(config_name="config_test.yaml")
    # Set experiment
    mlflow.set_tracking_uri("./data/mlruns")
    mlflow.set_experiment(args.experiment_name)

    # Output directory under `artifacts`
    outdir = args.outdir
    datadir = args.datadir
    ndata = args.ndata
    run_id = args.run_id

    # time zone
    jst = datetime.timezone(datetime.timedelta(hours=9), 'JST')
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # >>> Start mlflow run >>>
    mlflow.start_run(run_id=run_id)
    run = mlflow.get_run(mlflow.active_run().info.run_id)
    artifact_directory = run.info.artifact_uri.replace("file:", "")
    # <<< Start mlflow run <<<

    # >>> Load model >>>
    model = mlflow.pytorch.load_model(f"{artifact_directory}/models")
    model = model.to(device)
    model.eval()
    # <<< Load model <<<

    # Make output and label tensors
    outputtensor = torch.zeros((ndata, 2))
    labeltensor = torch.zeros((ndata,), dtype=torch.long)

    # Prepare test dataset
    nb = args.batchsize
    transforms = nn.Sequential(
        RandomCrop((128, 128))
    )

    nsplit = 1
    ndata_per_split = ndata // nsplit
    for idx_split in range(nsplit):
        idx_offset = idx_split * ndata_per_split
        inputs, labels = load_dataset(f'{datadir}/', ['cbc'], ndata_per_split, (128, 192), labellist=[1], ninit=idx_split * ndata_per_split)
        tensor_dataset = MyDataset(inputs, labels, transforms)
        dataloader = DataLoader(tensor_dataset, shuffle=False, drop_last=False, batch_size=nb, num_workers=4)
    
        # Test model
        with torch.no_grad():
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data
                inputs = inputs.to(device)
                outputs = softmax(model(inputs).cpu(), dim=1)
                kini = i * nb + idx_offset
                kend = (i + 1) * nb + idx_offset
                outputtensor[kini: kend] = outputs
                labeltensor[kini: kend] = labels
    labeltensor = nn.functional.one_hot(labeltensor, num_classes=2)
    print(f"[{datetime.datetime.now(jst)}] Test: Test data processed.")
    result_dict = {
        "label": labeltensor,
        "output": outputtensor
    }
    result_model = TestResult(result_dict)
    mlflow.pytorch.log_model(result_model, outdir)
    print(f"[{datetime.datetime.now(jst)}] Test: Result saved.")
    mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('--outdir', type=str, help='output directory name')
    parser.add_argument('--datadir', type=str, help='dataset directory (including ***/test/ or noise)')
    parser.add_argument('--ndata', type=int, help='The number of test data')
    parser.add_argument('--experiment_name', type=str, help='Experiment name')
    parser.add_argument('--run_id', type=str, help='Run ID')
    parser.add_argument('--batchsize', type=int, default=200, help='Batch size')
    args = parser.parse_args()

    main(args)

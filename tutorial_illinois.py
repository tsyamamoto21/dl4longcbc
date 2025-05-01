"""
tutorial_illinois.py
"""
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms


class CropSegment:
    def __init__(self, nseg, nseg_org, start_range) -> None:
        self.nseg = nseg
        self.nseg_org = nseg_org
        self.start_range = start_range
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Crop (nseg,) from (nseg_org,) data
        kstart = np.random.randint(self.start_range[0], self.start_range[1])
        x = x[:, :, kstart: kstart + self.nseg]
        return x


class InjectGaussianNoise:
    def __init__(self, nseg, std=1.0) -> None:
        self.nseg = nseg
        self.std = std
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Inject Noise
        noise = torch.empty_like(x).normal_(0.0, self.std)
        return x + noise


# pytorch
class DataNormalize:
    """Normalize input data.
    The normalized data has zero mean and variance unity.

    Args:
        data (torch.tensor): Input data.
        mode (str): Specify the normalization method. "mean" or "max".
        If "mean", the data is normalized so that its average and variance become 1 and 0, respectively.
        If "max", the data is normalized so that the maximum value of the absolute values of data becomes unity.

    Returns:
        torch.tensor: Normalized input data.
    """
    def __init__(self, mode="max", dim=None) -> None:
        self.mode = mode
        self.dim = dim
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "mean":
            mean = torch.mean(x, dim=self.dim, keepdim=True)
            std = torch.std(x, dim=self.dim, keepdim=True)
            x = (x - mean) / std
        elif self.mode == "max":
            xmax = self.TensorMax(torch.abs(x), dim=self.dim, keepdim=True)
            x = x / xmax
        else:
            pass
        return x

    def TensorMax(self, tensor, dim=None, keepdim=False):
        if dim is None:
            dim = range(tensor.ndim)
        elif isinstance(dim, int):
            dim = [dim]
        else:
            dim = sorted(dim)

        for d in dim[::-1]:
            tensor = tensor.max(dim=d, keepdim=keepdim)[0]

        return tensor


class deepernet(nn.Module):
    def __init__(self, out_features=2):
        super(deepernet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=16)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=16, dilation=2)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=16, dilation=2)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=32, dilation=2)
        self.pool4 = nn.MaxPool1d(4)
        self.dense1 = nn.Linear(in_features=7168, out_features=128)
        self.dense2 = nn.Linear(in_features=128, out_features=64)
        self.dense3 = nn.Linear(in_features=64, out_features=out_features)

    def forward(self, inputs):
        x = F.relu(self.pool1(self.conv1(inputs)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool3(self.conv3(x)))
        x = F.relu(self.pool4(self.conv4(x)))
        x = x.view(-1, 7168)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x


def main():
    learning_rate = 1.0e-3
    duration = 1.0
    duration_org = 1.2
    fs = 8192
    dt = 1.0 / fs
    nseg = int(duration * fs)
    nseg_org = int(duration_org * fs)
    ndata_tr = 6
    ndata_val = 6
    nbatch = 3
    nepoch = 10

    tstart_range = [0.0, 0.2]
    nstart_range = [int(tstart_range[0] * fs), int(tstart_range[1] * fs)]

    # Define neural network
    net = deepernet()
    # net.cuda(device=gpudevice)

    # Define a loss function and an optimizer
    criterion = nn.CrossEntropyLoss()  # loss function
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # Transform
    trans = transforms.Compose([
        CropSegment(nseg, nseg_org, start_range=nstart_range),
        InjectGaussianNoise(nseg, std=1.0),
        DataNormalize(mode="max", dim=2)
    ])

    # Load dataset
    # To be implemented

    # gwwaveform_tr = torch.empty((ndata_tr, 1, nseg_org), dtype=torch.float32).uniform_(-1.0, 1.0)
    gwwaveform_tr = torch.zeros((ndata_tr * 2, 1, nseg_org), dtype=torch.float32)
    gwsignal = np.tile(np.cos(2.0 * np.pi * 10 * np.arange(nseg_org) * dt), (ndata_tr, 1, 1)) * 10
    gwsignal = torch.tensor(gwsignal, dtype=torch.float32)
    gwwaveform_tr[:ndata_tr] = gwsignal
    labels_tr = torch.zeros((ndata_tr * 2, 2), dtype=torch.float32)
    labels_tr[:ndata_tr, 1] = 1
    labels_tr[ndata_tr:, 0] = 1
    traindata_tensors = TensorDataset(gwwaveform_tr, labels_tr)
    dataloader_tr = DataLoader(traindata_tensors, batch_size=nbatch, shuffle=True, drop_last=True, num_workers=2)

    # gwwaveform_val = torch.empty((ndata_val, 1, nseg_org), dtype=torch.float32).uniform_(-1.0, 1.0)
    gwwaveform_val = torch.zeros((ndata_val * 2, 1, nseg_org), dtype=torch.float32)
    gwwaveform_val[:ndata_val] = gwsignal
    labels_val = torch.zeros((ndata_val * 2, 2), dtype=torch.float32)
    labels_val[:ndata_val, 1] = 1
    labels_val[ndata_val:, 0] = 1
    valdata_tensors = TensorDataset(gwwaveform_val, labels_val)
    dataloader_val = DataLoader(valdata_tensors, batch_size=nbatch, shuffle=False, drop_last=False, num_workers=2)

    # Training
    for epoch in range(nepoch):
        for i, data in enumerate(dataloader_tr):
            inputs, labels = data
            inputs = trans(inputs)
            # inputs = inputs.cuda(device=gpudevice)
            preds = net(inputs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            print(f"{epoch+1:d} {loss.data:.5e}\n")

        # with open("./train_log.dat", "a+") as f:
        #     f.write(f"{epoch+1:d} {loss.data:.5e}\n")

        # Validate
        accuracy = 0
        with torch.no_grad():
            for i, data in enumerate(dataloader_val):
                inputs, labels = data
                inputs = trans(inputs)
                # inputs = inputs.cuda(device=gpudevice)
                preds = net(inputs)
                _, predclass = torch.max(preds, 1)
                accuracy = (predclass == labels).sum().item() * 1.0 / nbatch
                loss = criterion(preds, labels)
            print(f"Accuracy: {epoch+1:d} {accuracy:.2f}\n")
        #         with open("./validate_log.dat", "a+") as f:
        #             f.write(f"{epoch+1:d} {loss.data:.5e} {accuracy:.2f}\n")

    # # save the trained model
    # torch.save(net.state_dict(), MODELFILE)
    # print(f"Model saved: {MODELFILE}")


if __name__ == "__main__":
    import time
    starttime = time.time()
    main()      # Training and validation
    endtime = time.time()
    print(f"elapsed time: {endtime - starttime}[sec]")

import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = target
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        if self.transform:
            x = self.transform(x)
        y = self.target[index]
        return x, y


def normalize_tensor(x):
    # x.size = (1, H, W) is assumed.
    xmin = torch.min(x)
    xmax = torch.max(x)
    return (x - xmin) / (xmax - xmin)


def load_dataset(datadir, labelnamelist, ndata, imgsize, labellist=None, ninit=0):
    # The number of classes
    nclass = len(labelnamelist)
    if labellist is None:
        labellist = [i for i in range(nclass)]
    # Prepare input tensors and label tensors
    input_tensors = torch.zeros((ndata * nclass, 1, imgsize[0], imgsize[1]), dtype=torch.float32)
    label_tensors = torch.zeros((ndata * nclass,), dtype=torch.long)
    for i, (label, labelname) in enumerate(zip(labellist, labelnamelist)):
        offset = i * ndata
        for idx in range(ndata):
            filename = f'{datadir}/{labelname}/{labelname}_{ninit + idx:d}.pth'
            input_tensors[idx + offset] = normalize_tensor(torch.load(filename))
            label_tensors[idx + offset] = label
    return input_tensors, label_tensors

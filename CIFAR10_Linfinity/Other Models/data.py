import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
from typing import Callable, Dict, Optional, Sequence, Set, Tuple, Union
import torch.utils.data as data


PREPROCESSINGS = {
    'Res256Crop224':
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]),
    'Crop288':
    transforms.Compose([transforms.CenterCrop(288),
                        transforms.ToTensor()]),
    None:
    transforms.Compose([transforms.ToTensor()]),
    'Res224':
    transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ]),
    'BicubicRes256Crop224':
    transforms.Compose([
        transforms.Resize(
            256,
            interpolation=transforms.InterpolationMode("bicubic")),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
}

def _load_dataset(dataset: Dataset, n_examples: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = 100
    test_loader = data.DataLoader(dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=0)

    x_test, y_test = [], []
    for i, (x, y) in enumerate(test_loader):
        x_test.append(x)
        y_test.append(y)
        if n_examples is not None and batch_size * i >= n_examples:
            break
    x_test_tensor = torch.cat(x_test)
    y_test_tensor = torch.cat(y_test)

    if n_examples is not None:
        x_test_tensor = x_test_tensor[:n_examples]
        y_test_tensor = y_test_tensor[:n_examples]

    return x_test_tensor, y_test_tensor

def load_cifar10(
    n_examples: Optional[int] = None,
    flag_train: bool = False,
    data_dir: str = './data',
    transforms_test: Callable = PREPROCESSINGS[None]
) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset = datasets.CIFAR10(root=data_dir,
                               train=flag_train,
                               transform=transforms_test,
                               download=True)
    return _load_dataset(dataset, n_examples)



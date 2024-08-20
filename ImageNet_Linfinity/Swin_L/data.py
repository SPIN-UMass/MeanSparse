import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import Dataset
from typing import Callable, Dict, Optional, Sequence, Set, Tuple, Union
import torch.utils.data as data
from robustbench.data import load_cifar100
import os
import scipy
import scipy.io
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


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

def _load_dataset(
        dataset: Dataset,
        n_examples: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
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

def load_cifar100(
    n_examples: Optional[int] = None,
    flag_train: bool = False,
    data_dir: str = './data',
    transforms_test: Callable = PREPROCESSINGS[None]
) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset = datasets.CIFAR100(root=data_dir,
                                train=flag_train,
                                transform=transforms_test,
                                download=True)
    return _load_dataset(dataset, n_examples)
class ImageNetDataset_val(Dataset):
    def __init__(self, data_dir, annotation_file, transform=None):
        self.data_dir = data_dir
        self.annotations = pd.read_csv(annotation_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.annotations.iloc[idx, 0] + '.JPEG')
        image = Image.open(img_name).convert('RGB')
        label = self.annotations.iloc[idx, 1][0:9]

        if self.transform:
            image = self.transform(image)

        return image, label

class ImageNetDataset_train(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        counter = 0
        for label_dir in os.listdir(root_dir):
            counter += 1
            print("Loading images of class:", counter)
            class_dir = os.path.join(root_dir, label_dir)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(label_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def ImageNet_train_loader(batch_size=64, shuffle=False):
    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode("bicubic")),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    trainset = ImageNetDataset_train(root_dir='/datasets/ai/imagenet-1k/ILSVRC/Data/CLS-LOC/train',
                                transform=transform)
    #trainset = datasets.ImageFolder('/datasets/ai/imagenet-1k/ILSVRC/Data/CLS-LOC/train', transform=transform)
    return DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
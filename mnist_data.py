import numpy as np
import cv2
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torchvision import datasets, transforms

path = Path('data/')
mnist = datasets.MNIST(path, download=True)

batch_size = 64
test_batch_size = 64

# https://github.com/pytorch/examples/blob/master/mnist/main.py

train_dl = torch.utils.data.DataLoader(
    datasets.MNIST(path, train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)
test_dl = torch.utils.data.DataLoader(
    datasets.MNIST(path, train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=test_batch_size, shuffle=False)
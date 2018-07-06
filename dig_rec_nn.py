import os
import cv2
import copy
import torch
import random
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.models as models

from collections import deque
from pathlib import Path
from scipy import ndimage
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from mnist_data import train_dl, test_dl

def get_model(M = 300):
	net = nn.Sequential(nn.Linear(28*28, M),
						nn.ReLU(),
						nn.Linear(M, M - 120),
						nn.ReLU(),
						nn.Linear(M - 120, M - 240),
						nn.ReLU(),
						nn.Linear(M - 240, 10))
	return net


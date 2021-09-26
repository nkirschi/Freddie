import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import *


class BaseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten(-2, -1)
        self.linear1 = nn.Linear(3 * len(COLS_3D), 9)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(9, 5)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class SahibCNN(nn.Module):
    def __init__(self, input_dim = 3, bands = 1):
        """
        Description:
        3 layer simple CNN followed by 2 FC layers.
        Args:
        input_dim: dimensions in the data, default = 3 for X, Y,Z
        bands : number of features/channels. eg: Aircraft position --> 1
        """
        super(SahibCNN, self).__init__()
        self.bands = bands
        self.conv1 = nn.Conv1d(input_dim, 32, 1)
        self.conv2 = nn.Conv1d(32, 64, 1)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.fc1 = nn.Linear(128 * self.bands , 64)
        self.fc2 = nn.Linear(64, 5)
        self.activation = nn.LogSoftmax(dim = 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(-1, 128 * self.bands)
        x = F.relu(self.fc1(x))
        x = self.activation(self.fc2(x))

        return x


class SahibWindowedCNN(nn.Module):
    def __init__(self, input_dim = 3, bands = 1):

        super(SahibWindowedCNN, self).__init__()
        self.bands = bands
        self.conv1 = nn.Conv1d(input_dim, 32, 1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc0 = nn.Linear(128,64)
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(64 , 32)
        self.fc3 = nn.Linear(32, 5)
        #self.fc4 = nn.Linear(64, 5)
        self.activation = nn.LogSoftmax(dim = -1)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.transpose(1,2)
        x = F.relu(self.fc0(x))
        activations = F.relu(self.fc2(x))
        output = self.activation(self.fc3(activations))
        return output
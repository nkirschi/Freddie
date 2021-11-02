import torch.nn as nn
import torch.nn.functional as F


class BaseNet(nn.Module):
    """
    Baseline model with two linear layers and one activation.
    """

    def __init__(self, window_size, future_size, num_channels, num_classes=5):
        super().__init__()

        self.flatten = nn.Flatten(-2, -1)
        self.linear1 = nn.Linear(num_channels * window_size, 32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32, num_classes * (window_size + future_size))
        self.unflatten = nn.Unflatten(-1, (num_classes, window_size + future_size))

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.unflatten(x)

        return x


class BaseCNN(nn.Module):
    """
    Baseline CNN for window classification.
    """

    def __init__(self, *, window_size, num_channels, num_classes=5, input_dim=3):
        super().__init__()

        self.conv1 = nn.Conv1d(num_channels * input_dim, 32, 1)
        self.conv2 = nn.Conv1d(32, 64, 1)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.fc1 = nn.Linear(window_size * 128, 256)
        self.fc2 = nn.Linear(256, window_size * num_classes)

    def forward(self, x):
        x = x.flatten(-2, -1)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = x.flatten(-2, -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x.view(-1, 5, 10)


class SimpleCNN(nn.Module):
    """
    3 layer simple CNN followed by 2 FC layers.
    """

    def __init__(self, num_bands, input_dim=3):
        super().__init__()
        self.num_bands = num_bands
        self.conv1 = nn.Conv1d(input_dim, 32, 1)
        self.conv2 = nn.Conv1d(32, 64, 1)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.fc1 = nn.Linear(128 * self.num_bands, 64)
        self.fc2 = nn.Linear(64, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(-1, 128 * self.num_bands)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


class SimpleBatchNormedCNN(nn.Module):
    """
    3 layer simple batch-normed CNN followed by 3 FC layers.
    """

    def __init__(self, num_bands, input_dim=3):
        super(SimpleBatchNormedCNN, self).__init__()
        self.num_bands = num_bands
        self.conv1 = nn.Conv1d(input_dim, 32, 1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.transpose(1, 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

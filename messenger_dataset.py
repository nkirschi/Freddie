import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd


# TODO document
class MessengerDataset(Dataset):
    NUM_CLASSES = 5

    def __init__(self, orbits_path, window_size=3):
        self.data = pd.read_csv(orbits_path)
        self.window_size = window_size
        self.orbit_ids = self.data["ORBIT"].unique()
        self.max_num_windows = self.data.groupby("ORBIT").size().min()

    def __len__(self):
        return len(self.orbit_ids) * self.max_num_windows

    def __getitem__(self, idx):
        o_idx = idx // self.max_num_windows
        w_idx = idx % self.max_num_windows

        orbit = self.data[self.data["ORBIT"] == self.orbit_ids[o_idx]]
        window = orbit.iloc[w_idx : (w_idx + self.window_size)]

        flux = torch.tensor(window[["BX_MSO", "BY_MSO", "BZ_MSO"]].values)
        position = torch.tensor(window[["X_MSO", "Y_MSO", "Z_MSO"]].values)
        velocity = torch.tensor(window[["VX", "VY", "VZ"]].values)

        sample = torch.stack([flux, position, velocity], dim=1)  # window_size x 3 x 3
        label = F.one_hot(torch.tensor(list(window["LABEL"])),
                          num_classes=self.NUM_CLASSES)  # window_size x 5

        return sample, label
